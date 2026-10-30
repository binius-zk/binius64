// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP prover channel.

use std::ops::Deref;

use binius_compute::Allocator;
use binius_field::{BinaryField, Field, PackedField, util::powers};
use binius_iop::{
	basefold::{layout::RoundLayout, libra},
	channel::{OracleSchedule, OracleSpec},
	fri::FRIParams,
};
use binius_ip::sumcheck::{RoundCoeffs, infinity::to_mle_point};
use binius_ip_prover::{
	channel::{IPProverChannel, WordIPProverChannel},
	sumcheck::{self, infinity::ProductProver},
};
use binius_math::{
	FieldBuffer, FieldSlice, FieldSliceMut, FieldVec,
	inner_product::inner_product_par,
	line::extrapolate_line,
	multilinear::{
		eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars, eq_ind_zero},
		hypercube::{self, InfCube},
	},
	ntt::AdditiveNTT,
};
use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};
use itertools::izip;
use rand::{CryptoRng, SeedableRng, rngs::StdRng};

use crate::{
	basefold::{libra::LibraMask, prove_mlecheck_basefold},
	channel::IOPProverChannel,
	fri::{self, FRIFoldProver, MaskedCodeword},
	merkle_channel::MerkleIPProverChannel,
};

/// Oracle handle returned by [`BaseFoldProverChannel::send_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldOracle {
	/// The round whose oracle this one is committed in.
	round: usize,
	/// The oracle's block within its round oracle, in units of its own length.
	block_offset: usize,
	spec: OracleSpec,
}

/// Committed round oracle data stored internally.
struct CommittedOracleData<P: PackedField, C, Data: Deref<Target = [P]>> {
	/// The mask buffer generated during [`fri::encode_masked`] for a ZK oracle, held by the
	/// channel because it is the only party that knows it. `None` for a non-ZK (unmasked) oracle.
	mask: Option<FieldBuffer<P, Data>>,
	/// RS-encoded codeword, drawn from the channel's allocator.
	codeword: FieldBuffer<P, Data>,
	/// The Merkle commitment handle for query proofs, owning the committed tree.
	commitment: C,
	/// The committed round oracle `pi_i`, holding each constituent sent by
	/// [`IOPProverChannel::send_oracle`] in its block, in the caller's allocator.
	message: FieldBuffer<P, Data>,
}

/// One oracle's operand in a queued relation.
struct QueuedOperand<P: PackedField, Data: Deref<Target = [P]>> {
	/// Index of the relation, in queue order.
	relation: usize,
	/// The oracle's block within its round oracle, in units of the transparent's length.
	block_offset: usize,
	/// The transparent multilinear `t` the oracle is opened against, backed by the caller's
	/// allocator. Only the oracle's own block of the round oracle is ever written.
	transparent: FieldBuffer<P, Data>,
}

/// A prover channel that uses ZK BaseFold for all oracle commitments and openings.
///
/// Each round of the schedule is committed as one round oracle, laid out by
/// [`layout_rounds`](binius_iop::basefold::layout::layout_rounds): every oracle is copied into its
/// block as it arrives, and the round oracle is encoded and committed with the round's last one.
///
/// This channel owns an [`StdRng`] and generates random masks internally during
/// [`send_oracle`](IOPProverChannel::send_oracle). The caller provides only the raw witness
/// buffer (not doubled). The channel handles:
/// - Generating a random mask of equal length
/// - Interleaving witness and mask for FRI commitment
/// - Running ZK BaseFold proofs in [`Self::finish`]
///
/// # Type Parameters
///
/// - `F`: The binary field type
/// - `P`: The packed field type with `Scalar = F`
/// - `NTT`: The additive NTT for Reed-Solomon encoding
/// - `Channel`: The Merkle channel carrying all prover interaction
/// - `A`: The allocator the queued messages are drawn from, and the one [`Self::finish`] runs the
///   opening with
pub struct BaseFoldProverChannel<'a, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
	A: Allocator,
{
	/// The Merkle channel carrying all prover interaction: field elements, challenges,
	/// commitments, and openings.
	channel: Channel,
	ntt: &'a NTT,
	schedule: &'a OracleSchedule,
	layouts: &'a [RoundLayout],
	/// The combined FRI parameters over all round oracles.
	fri_params: FRIParams<F>,
	/// One entry per round committed so far, so its length is also the open round's index.
	committed_oracles: Vec<CommittedOracleData<P, Channel::Commitment, A::Vec<P>>>,
	/// The open round's oracle, allocated on its first constituent and committed on its last.
	open: Option<FieldBuffer<P, A::Vec<P>>>,
	/// How many oracles have been sent so far.
	n_sent: usize,
	/// How many of the open round's oracles have been sent so far.
	n_in_round: usize,
	/// Operands of the relations queued by [`IOPProverChannel::prove_multi_oracle_relation`],
	/// indexed by round and opened together in [`Self::finish`]. One entry per committed round.
	queue: Vec<Vec<QueuedOperand<P, A::Vec<P>>>>,
	/// The number of queued relations.
	n_relations: usize,
	rng: StdRng,
	/// The allocator every codeword, mask and encode temporary is drawn from.
	alloc: A,
}

impl<'a, F, P, NTT, Channel, A> BaseFoldProverChannel<'a, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
	A: Allocator,
{
	/// Creates a new BaseFold ZK prover channel over a Merkle channel from precomputed FRI
	/// parameters.
	///
	/// The FRI parameters should already account for ZK (log_batch_size = 1, doubled message
	/// length).
	///
	/// The RNG seeds the channel's own generator, whose only output is the ZK masks.
	/// A mask is what hides a committed witness at the positions the verifier opens.
	/// Hiding is therefore only as strong as this RNG, so it must be a cryptographic one.
	pub fn new(
		channel: Channel,
		ntt: &'a NTT,
		schedule: &'a OracleSchedule,
		layouts: &'a [RoundLayout],
		fri_params: FRIParams<F>,
		mut rng: impl CryptoRng,
		alloc: A,
	) -> Self {
		Self {
			channel,
			ntt,
			schedule,
			layouts,
			fri_params,
			committed_oracles: Vec::new(),
			open: None,
			n_sent: 0,
			n_in_round: 0,
			queue: Vec::new(),
			n_relations: 0,
			rng: StdRng::from_rng(&mut rng),
			alloc,
		}
	}

	/// Encodes and commits the next round oracle, the one at index `committed_oracles.len()`.
	fn commit_round_oracle(&mut self, message: FieldBuffer<P, A::Vec<P>>) {
		let index = self.committed_oracles.len();
		let layout = &self.layouts[index];

		// Encode round oracle `index` of the combined FRI parameters. ZK oracles interleave a fresh
		// mask (`encode_masked`); non-ZK oracles encode the message alone (`encode_interleaved`).
		let (codeword, mask) = if layout.is_zk {
			let MaskedCodeword { codeword, mask } = fri::encode_masked(
				&self.fri_params,
				index,
				self.ntt,
				message.as_view(),
				&mut self.rng,
				&self.alloc,
			);
			(codeword, Some(mask))
		} else {
			(
				fri::encode_interleaved(
					&self.fri_params,
					index,
					self.ntt,
					message.as_view(),
					&self.alloc,
				),
				None,
			)
		};

		// Commit the codeword over the Merkle channel, with one interleaved coset per leaf.
		let merkle_scope = tracing::debug_span!("Merkle commit").entered();
		let leaf_size = 1 << self.fri_params.input_oracles()[index].log_batch_size();
		let commitment = self
			.channel
			.send_merkle_commitment(codeword.as_view(), leaf_size);
		drop(merkle_scope);

		self.committed_oracles.push(CommittedOracleData {
			mask,
			codeword,
			commitment,
			message,
		});
	}

	/// Consumes the channel and proves the single combined opening over **all** committed oracles.
	///
	/// All oracle relations queued by
	/// [`prove_multi_oracle_relation`](IOPProverChannel::prove_multi_oracle_relation) across every
	/// call are processed here in one batch: batching into one relation, the Libra-masked sumcheck
	/// and its rerandomization reducing it to evaluations at prefixes of one point, then one
	/// combined FRI opening over every committed oracle (in oracle-index order). Mirrors
	/// [`BaseFoldVerifierChannel::finish`].
	///
	/// [`BaseFoldVerifierChannel::finish`]: binius_iop::basefold::channel::BaseFoldVerifierChannel::finish
	pub fn finish(mut self) {
		let n_remaining = self.schedule.specs().len() - self.n_sent;
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining",);

		if self.n_relations == 0 {
			return;
		}

		// A masked opening commits the Libra oracle, in the round after the schedule's.
		let n_rounds = self.committed_oracles.len();
		let libra_mask = self.layouts.get(n_rounds).map(|libra_layout| {
			let n_vars = max_log_len(&self.layouts[..n_rounds]);
			let mask = LibraMask::random(&mut self.rng, n_vars, libra_layout.log_len);
			self.commit_round_oracle(FieldBuffer::from_view_in(
				&self.alloc,
				mask.table().as_view(),
			));
			mask
		});

		let Self {
			mut channel,
			ntt,
			schedule: _,
			layouts,
			fri_params,
			committed_oracles,
			open: _,
			n_sent: _,
			n_in_round: _,
			queue,
			n_relations,
			rng: _,
			alloc,
		} = self;

		prove_batch_zk_basefold(
			&mut channel,
			ntt,
			layouts,
			&fri_params,
			committed_oracles,
			queue,
			n_relations,
			libra_mask.as_ref(),
			&alloc,
		);
	}
}

/// The variable count of the largest round oracle.
fn max_log_len(layouts: &[RoundLayout]) -> usize {
	layouts
		.iter()
		.map(|layout| layout.log_len)
		.max()
		.expect("at least one oracle")
}

/// Proves the combined ZK BaseFold opening over all committed round oracles.
///
/// This drives `channel` — the Merkle channel taken from the destructured
/// [`BaseFoldProverChannel`] — through its [`MerkleIPProverChannel`] interface, in the three phases
/// of [`binius_iop::basefold::channel::BaseFoldVerifierChannel::finish`], which it mirrors.
///
/// Everything per oracle runs in round order: `operands` arrives indexed by schedule round, and the
/// per-round data (`layouts`, `fri_params`, `committed_oracles`) carry the Libra round last when
/// `libra_mask` is present.
#[allow(clippy::too_many_arguments)]
fn prove_batch_zk_basefold<A, F, P, NTT, Channel>(
	channel: &mut Channel,
	ntt: &NTT,
	layouts: &[RoundLayout],
	fri_params: &FRIParams<F>,
	mut committed_oracles: Vec<CommittedOracleData<P, Channel::Commitment, A::Vec<P>>>,
	operands: Vec<Vec<QueuedOperand<P, A::Vec<P>>>>,
	n_relations: usize,
	libra_mask: Option<&LibraMask<P>>,
	alloc: &A,
) where
	A: Allocator,
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
{
	let n_committed = committed_oracles.len();
	assert_eq!(layouts.len(), n_committed);
	let n_rounds = operands.len();
	let round_layouts = &layouts[..n_rounds];

	// `ℓ = max_r ℓ_r` over the schedule's rounds, the variable count of Phase A.
	let n_vars = max_log_len(round_layouts);

	// === Phase A: the Libra-masked batch sumcheck over {0, ∞}^ℓ ===
	let transparents = batch_relations(channel, round_layouts, operands, n_relations, alloc);

	let alpha = libra_mask.map(|mask| {
		channel.send_one(mask.sum());
		channel.sample()
	});

	// One prover of `Σ_x M_r^∞(x) · T_r^∞(x) = <M_r, T_r>` per round. The provers fold their
	// tables, so each takes a copy of its round oracle.
	// ponytail: one copy per round oracle; fold the first round out of place if it shows in
	// profiles.
	let mut provers = izip!(transparents, &committed_oracles)
		.map(|(transparent, committed)| {
			let message = FieldBuffer::from_view_in(alloc, committed.message.as_view());
			ProductProver::new([message, transparent])
		})
		.collect::<Vec<_>>();

	let mut challenges = Vec::with_capacity(n_vars);
	{
		let _scope =
			tracing::debug_span!("Reduce linear relations to committed openings").entered();
		for round in 0..n_vars {
			let n_vars_remaining = n_vars - round;
			let round_coeffs = provers
				.iter()
				.map(|prover| {
					if prover.n_vars() == n_vars_remaining {
						prover.round_coeffs()
					} else {
						// The summand does not depend on this variable, so R(X) is the claim.
						RoundCoeffs(vec![prover.claim()])
					}
				})
				.collect::<Vec<_>>();

			let mut batched = round_coeffs.iter().cloned().sum::<RoundCoeffs<F>>();
			if let (Some(mask), Some(alpha)) = (libra_mask, alpha) {
				batched += &(mask.round_coeffs(&challenges) * alpha);
			}
			channel.send_many(batched.truncate().coeffs());

			let challenge = channel.sample();
			challenges.push(challenge);
			for (prover, coeffs) in izip!(&mut provers, &round_coeffs) {
				if prover.n_vars() == n_vars_remaining {
					prover.fold(coeffs, challenge);
				}
			}
		}
	}

	// The sumcheck yields binding-order challenges; reverse to variable-indexed (low-to-high).
	challenges.reverse();
	let point = challenges;

	let beta = libra_mask.map(|mask| {
		let beta = mask.evaluate(&point);
		channel.send_one(beta);
		beta
	});

	// `α_r = M_r^∞(ρ_{<ℓ_r})`, one per round.
	let alphas = provers
		.into_iter()
		.map(|prover| prover.finish()[0])
		.collect::<Vec<_>>();
	channel.send_many(&alphas);

	// === Phase A′: rerandomize the claims onto prefixes of one point ===
	let (point, evals) = match (libra_mask, beta) {
		(Some(mask), Some(_)) => {
			let m = layouts[n_rounds].log_len;

			// Round `r`'s claim is `Σ_y M_r^∞(y, ρ_{m ≤ · < ℓ_r}) · e_ρ(y)` over its own
			// `min(m, ℓ_r)` low variables, and the Libra query is `Σ_y h'^∞(y) ·
			// libra-eval_ρ^∞(y)`.
			let mut pairs = izip!(&committed_oracles, round_layouts)
				.map(|(committed, layout)| {
					let n = m.min(layout.log_len);
					[
						fold_high_monomial(
							alloc,
							committed.message.as_view(),
							&point[n..layout.log_len],
						),
						hypercube::eq_ind_partial_eval_in::<InfCube, _, P>(alloc, &point[..n]),
					]
				})
				.collect::<Vec<_>>();
			let libra_eval = FieldBuffer::<P>::from_values(&libra::libra_eval_table(&point, m));
			pairs.push([
				FieldBuffer::from_view_in(alloc, mask.table().as_view()),
				FieldBuffer::from_view_in(alloc, libra_eval.as_view()),
			]);

			let output = sumcheck::infinity::batch_prove(pairs, channel);
			let evals = output
				.multilinear_evals
				.iter()
				.map(|evals| evals[0])
				.collect::<Vec<_>>();
			channel.send_many(&evals);

			// `ρ'' = (ρ', ρ_{≥m})`.
			let mut point_prime = output.challenges;
			point_prime.reverse();
			point_prime.extend(point.into_iter().skip(m));
			(point_prime, evals)
		}
		_ => (point, alphas),
	};

	// === Phase B: single combined-FRI MLE-check over the piecewise-concatenated oracle ===
	// Translate each claim `M_r^∞(ρ''_{<ℓ_r}) = α'_r` into `M̃_r(σ_{<ℓ_r}) = a_r`.
	let (point, scales) = to_mle_point(&point);
	let point = &point;
	let log_n_oracles = log2_ceil_usize(n_committed);
	let outer_challenges = channel.sample_many(log_n_oracles);
	let eq_tensor = eq_ind_partial_eval_scalars(&outer_challenges);

	// Each oracle occupies the low 2^{n_i} of every 2^{log_lift}·2^{n_i}-sized lift block, and that
	// block is *repeated* across the 2^{log_repeat} high dims so the small oracle is constant along
	// them (matching the FRI lift/repeat structure). Repeat dims contribute 1; only the lift dims
	// contribute an eq-to-zero factor.
	let weights = izip!(fri_params.input_oracles(), layouts, &eq_tensor)
		.map(|(fri_oracle, layout, &eq_i)| {
			eq_i * eq_ind_zero(&point[layout.log_len..][..fri_oracle.log_lift])
		})
		.collect::<Vec<_>>();
	let s_prime = izip!(&weights, layouts, &evals)
		.map(|(&weight, layout, &eval)| weight * eval * scales[layout.log_len].invert_or_zero())
		.sum::<F>();

	// The interleave round: the claim is at `Y = 0` of `(1 - Y)·M_r + Y·ω_r`. Send the round
	// polynomial's value at `Y = 1`, where a masked oracle contributes `ω̃_r(σ)` and an unmasked
	// one, constant in `Y`, its own claim. Then blind each masked message in place to
	// `(1 - γ)·M_r + γ·ω_r`, the fold the FRI opens.
	let (s_prime, gamma) = match libra_mask {
		Some(_) => {
			let _scope = tracing::debug_span!("Interleave round").entered();
			let s_one = izip!(&weights, layouts, &evals, &committed_oracles)
				.map(|(&weight, layout, &eval, committed)| {
					let n_i = layout.log_len;
					let value = committed.mask.as_ref().map_or_else(
						|| eval * scales[n_i].invert_or_zero(),
						|mask| inner_product_par(mask, &eq_ind_partial_eval::<P>(&point[..n_i])),
					);
					weight * value
				})
				.sum::<F>();
			channel.send_one(s_one);
			let gamma = channel.sample();

			let gamma_broadcast = P::broadcast(gamma);
			for committed in &mut committed_oracles {
				let CommittedOracleData { message, mask, .. } = committed;
				if let Some(mask) = mask {
					(message.as_mut(), mask.as_ref())
						.into_par_iter()
						.with_min_task(WorkPerItem::FieldMuls)
						.for_each(|(message_i, &mask_i)| {
							*message_i = extrapolate_line(*message_i, mask_i, gamma_broadcast);
						});
				}
			}
			(extrapolate_line(s_prime, s_one, gamma), Some(gamma))
		}
		None => (s_prime, None),
	};

	let combined = {
		let _scope = tracing::debug_span!("Compute batched witness").entered();
		let max_n = point.len();
		let mut combined = FieldBuffer::zeros_in(alloc, max_n);
		for (fri_oracle, committed, &eq_i) in
			izip!(fri_params.input_oracles(), &committed_oracles, &eq_tensor)
		{
			let n_i = committed.message.log_len();
			// Repeat placement: add scalar · π_i' into the first 2^{n_i} entries of each of the
			// 2^{log_repeat} chunks of size 2^{n_i + log_lift}.
			place_repeated(
				combined.as_mut_view(),
				committed.message.as_view(),
				eq_i,
				n_i + fri_oracle.log_lift,
			);
		}
		combined
	};

	// Codeword commitments in oracle-index order, matching `open_fri_params.input_oracles()`.
	let committed_codewords = committed_oracles
		.into_iter()
		.map(|committed| (committed.codeword, committed.commitment))
		.collect();

	let fri_folder = FRIFoldProver::new_batch(fri_params, ntt, committed_codewords);
	prove_mlecheck_basefold(
		combined,
		point,
		s_prime,
		gamma,
		&outer_challenges,
		fri_folder,
		channel,
		alloc,
	);
}

/// The coefficient table of `u^∞(·, coords)`: `u`'s monomial coefficients with its top
/// variables bound to `coords`, low-to-high, so `v ↦ Σ_w u(v, w) · coords^w`.
fn fold_high_monomial<F, P, A>(alloc: &A, u: FieldSlice<'_, P>, coords: &[F]) -> FieldVec<P, A>
where
	F: Field,
	P: PackedField<Scalar = F>,
	A: Allocator,
{
	let Some((&top, rest)) = coords.split_last() else {
		return FieldBuffer::from_view_in(alloc, u);
	};
	let mut folded = hypercube::fold_highest_var::<InfCube, _, _, _>(alloc, &u, top);
	for &coord in rest.iter().rev() {
		hypercube::fold_highest_var_inplace::<InfCube, _, _>(&mut folded, coord);
	}
	folded
}

/// Batches the queued relations into the single relation `Σ_i <π_i, T_i> = S`.
///
/// Relation `j`, `Σ_i <π_i, t_ji> = s_j`, is weighted by `μ^j` for one sampled challenge μ:
///
/// ```text
/// T_i = Σ_j μ^j · t_ji     the combined transparent of oracle i
/// S   = Σ_j μ^j · s_j      the combined claim
/// ```
///
/// The inner product is linear in the transparent, so the combined relation holds exactly when
/// every relation does, except with probability at most `(k - 1) / |F|` over μ for `k` relations.
/// μ is drawn after every claim it combines is already bound to the transcript, so no claim can be
/// chosen as a function of it. The prover never needs `S` itself.
///
/// Returns each round oracle's `T_i`, in round order. A round in no relation gets `T_i = 0`.
///
/// Each `t_ji` is accumulated into only its oracle's block of `T_i`, so it costs only that block's
/// size. A round whose first operand is in relation 0, with weight 1, and spans the whole round
/// oracle takes that operand as its accumulator with no copy.
///
/// Mirrors [`binius_iop::basefold::channel`]'s verifier-side batching.
fn batch_relations<A, F, P, Channel>(
	channel: &mut Channel,
	layouts: &[RoundLayout],
	operands: Vec<Vec<QueuedOperand<P, A::Vec<P>>>>,
	n_relations: usize,
	alloc: &A,
) -> Vec<FieldBuffer<P, A::Vec<P>>>
where
	A: Allocator,
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: MerkleIPProverChannel<F>,
{
	let mu = channel.sample();
	let mu_powers = powers(mu).take(n_relations).collect::<Vec<_>>();

	izip!(operands, layouts)
		.map(|(operands, layout)| {
			let mut operands = operands.into_iter().peekable();
			let mut transparent = match operands.next_if(|operand| {
				operand.relation == 0 && operand.transparent.log_len() == layout.log_len
			}) {
				Some(first) => first.transparent,
				None => FieldBuffer::zeros_in(alloc, layout.log_len),
			};
			for operand in operands {
				let mut block =
					transparent.chunk_mut(operand.transparent.log_len(), operand.block_offset);
				accumulate_scaled_buffer(
					block.chunk(),
					operand.transparent.as_view(),
					P::broadcast(mu_powers[operand.relation]),
				);
			}
			transparent
		})
		.collect()
}

/// Copies the first `len` scalars of `src` into `dst`, leaving the rest of `dst` untouched.
///
/// Whole packed words are copied as words, and a partial last word scalar by scalar.
fn copy_prefix<P: PackedField>(mut dst: FieldSliceMut<'_, P>, src: FieldSlice<'_, P>, len: usize) {
	let n_words = len >> P::LOG_WIDTH;
	dst.as_mut()[..n_words].copy_from_slice(&src.as_ref()[..n_words]);
	for i in n_words << P::LOG_WIDTH..len {
		dst.set(i, src.get(i));
	}
}

/// Adds `scalar · src` into the low `2^src.log_len()` scalars of every `2^log_block`-sized block
/// of `dst`.
///
/// This is the lift/repeat placement of one oracle into the combined buffer: the oracle occupies
/// the low part of a lift block, and that block repeats across the high dims so the oracle is
/// constant along them.
///
/// A block spanning at least one whole packed element gets a chunk of the buffer to itself. A
/// narrower block does not: several of them then share one element, and no chunking can express
/// the placement. The scalars of one element are the same repeating pattern for every element, so
/// that pattern is built once and added to all of them.
///
/// ## Preconditions
///
/// * `src.log_len() <= log_block <= dst.log_len()`
fn place_repeated<P: PackedField>(
	mut dst: FieldSliceMut<'_, P>,
	src: FieldSlice<'_, P>,
	scalar: P::Scalar,
	log_block: usize,
) {
	assert!(src.log_len() <= log_block); // precondition
	assert!(log_block <= dst.log_len()); // precondition

	let scalar_broadcast = P::broadcast(scalar);
	if log_block >= P::LOG_WIDTH {
		let chunk_packed = 1usize << (log_block - P::LOG_WIDTH);
		dst.as_mut().par_chunks_mut(chunk_packed).for_each(|chunk| {
			let chunk_buf = FieldSliceMut::from_slice(log_block, chunk);
			accumulate_scaled_buffer(chunk_buf, src.as_view(), scalar_broadcast);
		});
	} else {
		// Lane `k` of every element sits at position `k % 2^log_block` of its block, and carries
		// the oracle only over the low `2^src.log_len()` of them. A buffer shorter than one
		// element leaves its high lanes out of the pattern, so they stay zero.
		let block_mask = (1usize << log_block) - 1;
		let src_len = 1usize << src.log_len();
		let lanes = P::WIDTH.min(1usize << dst.log_len());
		let pattern = P::from_scalars((0..lanes).map(|lane| {
			let position = lane & block_mask;
			if position < src_len {
				src.get(position)
			} else {
				P::Scalar::ZERO
			}
		}));
		dst.as_mut()
			.par_iter_mut()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|dst_i| *dst_i += scalar_broadcast * pattern);
	}
}

fn accumulate_scaled_buffer<P: PackedField>(
	mut dst: FieldSliceMut<'_, P>,
	src: FieldSlice<'_, P>,
	scalar_broadcast: P,
) {
	if src.log_len() >= P::LOG_WIDTH {
		let src = src.as_ref();
		// This accumulation already runs inside a parallel loop over chunks.
		// One chunk is small, so a second split here would only add handoff cost.
		dst.as_mut()
			.par_iter_mut()
			.zip(src.as_ref())
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|(dst_i, src_i)| {
				*dst_i += scalar_broadcast * *src_i;
			});
	} else {
		let src = P::from_scalars(src.iter_scalars());
		dst.as_mut()[0] += scalar_broadcast * src;
	}
}

impl<'a, F, P, NTT, Channel, A> IPProverChannel<F>
	for BaseFoldProverChannel<'a, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
	A: Allocator,
{
	fn send_one(&mut self, elem: F) {
		self.channel.send_one(elem);
	}

	fn send_many(&mut self, elems: &[F]) {
		self.channel.send_many(elems);
	}

	fn send_public_claim(&mut self, elem: F) {
		self.channel.send_public_claim(elem);
	}

	fn observe_one(&mut self, val: F) {
		self.channel.observe_one(val);
	}

	fn observe_many(&mut self, vals: &[F]) {
		self.channel.observe_many(vals);
	}

	fn sample(&mut self) -> F {
		self.channel.sample()
	}
}

impl<F, P, NTT, Channel, A> WordIPProverChannel<F>
	for BaseFoldProverChannel<'_, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
	A: Allocator,
{
	type Word = Channel::Word;

	fn observe_words(&mut self, words: &[Self::Word]) {
		self.channel.observe_words(words);
	}

	fn sample_bits(&mut self, bits: usize) -> Self::Word {
		self.channel.sample_bits(bits)
	}
}

impl<'a, F, P, NTT, Channel, A> IOPProverChannel<P, A>
	for BaseFoldProverChannel<'a, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
	A: Allocator,
{
	type Oracle = BaseFoldOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.schedule.specs()[self.n_sent..]
	}

	fn send_oracle(&mut self, buffer: FieldSlice<'_, P>) -> Self::Oracle {
		let spec = *self
			.remaining_oracle_specs()
			.first()
			.expect("send_oracle called but no remaining oracle specs");

		// ZK channel expects raw witness buffer (NOT doubled).
		assert_eq!(
			buffer.log_len(),
			spec.log_msg_len(),
			"oracle buffer log_len mismatch: expected {}, got {}",
			spec.log_msg_len(),
			buffer.log_len()
		);

		let index = self.committed_oracles.len();
		let layout = &self.layouts[index];
		let oracle = BaseFoldOracle {
			round: index,
			block_offset: layout.block_offsets[self.n_in_round],
			spec,
		};
		self.n_sent += 1;
		self.n_in_round += 1;

		// A lone oracle is its round oracle, padding included. Any other is copied into its block,
		// content only, since a round-mate may nest in the slack above it.
		let message = if layout.block_offsets.len() == 1 {
			FieldBuffer::from_view_in(&self.alloc, buffer)
		} else {
			let message = self
				.open
				.get_or_insert_with(|| FieldBuffer::zeros_in(&self.alloc, layout.log_len));
			let mut block = message.chunk_mut(spec.log_msg_len(), oracle.block_offset);
			copy_prefix(block.chunk(), buffer, spec.len);
			drop(block);

			// The round's last oracle commits the whole round oracle.
			if self.n_in_round < layout.block_offsets.len() {
				return oracle;
			}
			self.open
				.take()
				.expect("the open round oracle was just written")
		};
		self.n_in_round = 0;
		self.commit_round_oracle(message);
		self.queue.push(Vec::new());

		oracle
	}

	fn prove_multi_oracle_relation(
		&mut self,
		operands: impl IntoIterator<Item = (Self::Oracle, FieldBuffer<P, A::Vec<P>>)>,
		_claim: P::Scalar,
	) {
		// Queue each operand under its round; the actual opening (sumchecks + combined
		// FRI) happens once, over all committed round oracles, in [`Self::finish`].
		let relation = self.n_relations;
		let n_committed = self.queue.len();
		for (oracle, transparent) in operands {
			let BaseFoldOracle {
				round,
				block_offset,
				spec,
			} = oracle;
			assert!(round < n_committed, "oracle's round {round} is not committed yet");
			assert_eq!(
				transparent.log_len(),
				spec.log_msg_len(),
				"transparent log_len must match the oracle's"
			);
			debug_assert!(
				transparent
					.iter_scalars()
					.skip(spec.len)
					.all(|value| value == F::ZERO),
				"an operand's transparent must vanish above its oracle's length"
			);

			self.queue[round].push(QueuedOperand {
				relation,
				block_offset,
				transparent,
			});
		}
		self.n_relations += 1;
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_field::{
		BinaryField, Field, Ghash128b, Ghash128b as B128, PackedField, PackedGhash1x128b,
		PackedGhash2x128b, PackedGhash4x128b, Random,
	};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		basefold::compiler::BaseFoldVerifierCompiler,
		channel::{IOPVerifierChannel, OracleSchedule, OracleSpec, TransparentEvalFn},
		fri::MinProofSizeStrategy,
		merkle_tree::BinaryMerkleTreeScheme,
	};
	use binius_ip::channel::IPVerifierChannel;
	use binius_ip_prover::channel::IPProverChannel;
	use binius_math::{
		FieldBuffer,
		inner_product::inner_product_buffers,
		multilinear::eq::eq_ind_partial_eval,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
	use proptest::prelude::*;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::{IOPProverChannel, place_repeated};
	use crate::basefold::compiler::BaseFoldProverCompiler;

	type StdChallenger = HasherChallenger<StdDigest>;

	const LOG_INV_RATE: usize = 1;
	const SECURITY_BITS: usize = 32;

	fn calculate_n_test_queries(security_bits: usize, log_inv_rate: usize) -> usize {
		security_bits.div_ceil(log_inv_rate)
	}

	fn make_ntt(log_domain_size: usize) -> NeighborsLastSingleThread<GaoMateerOnTheFly<Ghash128b>> {
		let domain_context = GaoMateerOnTheFly::generate(log_domain_size);
		NeighborsLastSingleThread::new(domain_context)
	}

	fn make_merkle_scheme() -> BinaryMerkleTreeScheme<Ghash128b, StdHashSuite> {
		BinaryMerkleTreeScheme::new()
	}

	/// The verifier's evaluator of a transparent the test holds in full.
	fn eq_eval_fn<P: PackedField<Scalar = Ghash128b>>(
		transparent: FieldBuffer<P>,
	) -> TransparentEvalFn<Ghash128b> {
		Box::new(move |point| {
			let eq = eq_ind_partial_eval::<P>(point);
			inner_product_buffers(&transparent, &eq)
		})
	}

	fn generate_zk_oracle_data<F, P, R: Rng>(
		rng: &mut R,
		n_vars: usize,
	) -> (FieldBuffer<P>, FieldBuffer<P>, F)
	where
		F: BinaryField,
		P: PackedField<Scalar = F>,
	{
		let buffer = random_field_buffer::<P>(&mut *rng, n_vars);
		let evaluation_point = random_scalars::<F>(&mut *rng, n_vars);
		let transparent_poly = eq_ind_partial_eval::<P>(&evaluation_point);
		let evaluation_claim = inner_product_buffers(&buffer, &transparent_poly);
		(buffer, transparent_poly, evaluation_claim)
	}

	#[test]
	fn test_basefold_channel_single_oracle() {
		type F = Ghash128b;
		type P = PackedGhash1x128b;

		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 8;

		let (buffer, transparent_poly, eval_claim) =
			generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

		let oracle_schedule = OracleSchedule::from_rounds([[OracleSpec::new_zk(1 << n_vars)]]);

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			oracle_schedule,
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// === PROVER SIDE ===
		let ntt = make_ntt(verifier_compiler.max_log_domain_size());
		let prover_compiler =
			BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_rng = StdRng::seed_from_u64(1);
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
				prover_rng,
				GlobalAllocator,
			);

		let oracle = prover_channel.send_oracle(buffer.as_view());
		assert_eq!(oracle.round, 0);

		prover_channel.prove_oracle_relation(oracle, transparent_poly.clone(), eval_claim);
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let v_oracle = verifier_channel.recv_oracle(1 << n_vars, true).unwrap();

		verifier_channel
			.verify_oracle_relation(v_oracle, eq_eval_fn(transparent_poly), eval_claim)
			.unwrap();
		verifier_channel.finish().unwrap();
	}

	#[test]
	fn test_basefold_channel_two_oracles() {
		type F = Ghash128b;
		type P = PackedGhash1x128b;

		let mut rng = StdRng::seed_from_u64(0);
		let n_vars_1 = 6;
		let n_vars_2 = 8;

		let (buffer_1, transparent_poly_1, eval_claim_1) =
			generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars_1);
		let (buffer_2, transparent_poly_2, eval_claim_2) =
			generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars_2);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

		let oracle_schedule = OracleSchedule::from_rounds([
			[OracleSpec::new_zk(1 << n_vars_1)],
			[OracleSpec::new_zk(1 << n_vars_2)],
		]);

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			oracle_schedule,
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// === PROVER SIDE ===
		let ntt = make_ntt(verifier_compiler.max_log_domain_size());
		let prover_compiler =
			BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_rng = StdRng::seed_from_u64(1);
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
				prover_rng,
				GlobalAllocator,
			);

		let oracle_1 = prover_channel.send_oracle(buffer_1.as_view());
		let oracle_2 = prover_channel.send_oracle(buffer_2.as_view());

		prover_channel.prove_oracle_relation(oracle_1, transparent_poly_1.clone(), eval_claim_1);
		prover_channel.prove_oracle_relation(oracle_2, transparent_poly_2.clone(), eval_claim_2);
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let v_oracle_1 = verifier_channel.recv_oracle(1 << n_vars_1, true).unwrap();
		let v_oracle_2 = verifier_channel.recv_oracle(1 << n_vars_2, true).unwrap();

		let tp1 = transparent_poly_1;
		let tp2 = transparent_poly_2;

		verifier_channel
			.verify_oracle_relation(v_oracle_1, eq_eval_fn(tp1), eval_claim_1)
			.unwrap();
		verifier_channel
			.verify_oracle_relation(v_oracle_2, eq_eval_fn(tp2), eval_claim_2)
			.unwrap();
		verifier_channel.finish().unwrap();
	}

	/// Runs a full prove/verify cycle of the Batched ZK BaseFold channel over oracles of the given
	/// sizes. If `tamper`, the verifier's claim on the first oracle is corrupted; verification must
	/// then fail. Returns whether verification accepted.
	fn run_zk_channel<P: PackedField<Scalar = Ghash128b>>(
		n_vars_list: &[usize],
		tamper: bool,
	) -> bool {
		type F = Ghash128b;

		let mut rng = StdRng::seed_from_u64(0);
		let data: Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> = n_vars_list
			.iter()
			.map(|&n| generate_zk_oracle_data::<F, P, _>(&mut rng, n))
			.collect();

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
		let oracle_specs = n_vars_list.iter().map(|&n| OracleSpec::new_zk(1 << n));

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			OracleSchedule::from_rounds(oracle_specs.map(|spec| [spec])),
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// === PROVER SIDE ===
		let ntt = make_ntt(verifier_compiler.max_log_domain_size());
		let prover_compiler =
			BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_rng = StdRng::seed_from_u64(1);
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
				prover_rng,
				GlobalAllocator,
			);

		let oracles: Vec<_> = data
			.iter()
			.map(|(buffer, _, _)| prover_channel.send_oracle(buffer.as_view()))
			.collect();
		for (oracle, (_, transparent, claim)) in iter::zip(oracles, &data) {
			prover_channel.prove_oracle_relation(oracle, transparent.clone(), *claim);
		}
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let v_oracles: Vec<_> = n_vars_list
			.iter()
			.map(|&n| verifier_channel.recv_oracle(1 << n, true).unwrap())
			.collect();
		for (i, (oracle, (_, transparent, claim))) in iter::zip(v_oracles, &data).enumerate() {
			let transparent = transparent.clone();
			let claim = if tamper && i == 0 {
				*claim + F::ONE
			} else {
				*claim
			};
			verifier_channel
				.verify_oracle_relation(oracle, eq_eval_fn(transparent), claim)
				.expect("verify_oracle_relation only queues");
		}
		verifier_channel.finish().is_ok()
	}

	/// Like `run_zk_channel` but with per-oracle `(n_vars, is_zk)` flags, exercising the mixed
	/// ZK/non-ZK opening. If `tamper`, the verifier's claim on the first oracle is corrupted. If
	/// `corrupt_byte` is set, that byte of the proof is flipped.
	fn run_mixed_channel<P: PackedField<Scalar = Ghash128b>>(
		specs: &[(usize, bool)],
		tamper: bool,
		corrupt_byte: Option<usize>,
	) -> bool {
		type F = Ghash128b;

		let mut rng = StdRng::seed_from_u64(0);
		let data: Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> = specs
			.iter()
			.map(|&(n, _)| generate_zk_oracle_data::<F, P, _>(&mut rng, n))
			.collect();

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
		let oracle_specs = specs.iter().map(|&(n, is_zk)| {
			if is_zk {
				OracleSpec::new_zk(1 << n)
			} else {
				OracleSpec::new(1 << n)
			}
		});

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			OracleSchedule::from_rounds(oracle_specs.map(|spec| [spec])),
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		let ntt = make_ntt(verifier_compiler.max_log_domain_size());
		let prover_compiler =
			BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_rng = StdRng::seed_from_u64(1);
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
				prover_rng,
				GlobalAllocator,
			);

		let oracles: Vec<_> = data
			.iter()
			.map(|(buffer, _, _)| prover_channel.send_oracle(buffer.as_view()))
			.collect();
		for (oracle, (_, transparent, claim)) in iter::zip(oracles, &data) {
			prover_channel.prove_oracle_relation(oracle, transparent.clone(), *claim);
		}
		prover_channel.finish();

		let mut proof = prover_transcript.finalize();
		if let Some(index) = corrupt_byte {
			proof[index] ^= 1;
		}
		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let v_oracles: Vec<_> = specs
			.iter()
			.map(|&(n, _)| verifier_channel.recv_oracle(1 << n, true).unwrap())
			.collect();
		for (i, (oracle, (_, transparent, claim))) in iter::zip(v_oracles, &data).enumerate() {
			let transparent = transparent.clone();
			let claim = if tamper && i == 0 {
				*claim + F::ONE
			} else {
				*claim
			};
			verifier_channel
				.verify_oracle_relation(oracle, eq_eval_fn(transparent), claim)
				.expect("verify_oracle_relation only queues");
		}
		verifier_channel.finish().is_ok()
	}

	#[test]
	fn test_basefold_channel_three_oracles_non_power_of_two() {
		// 3 oracles (not a power of two) of unequal sizes: exercises oracle padding (Lifted FRI)
		// and the `⌈log 3⌉ = 2` outer oracle-combine rounds.
		assert!(run_zk_channel::<PackedGhash1x128b>(&[5, 6, 8], false));
	}

	/// A batch whose lift blocks are narrower than one packed field element must still prove.
	///
	/// Placing an oracle into the combined buffer chunks that buffer by the lift block. Every
	/// block is lifted to the combined dimension, so a block narrower than a packed element means
	/// the whole buffer is — there is no whole-packed chunk to place into, and the placement has
	/// to write into part of a single element instead.
	///
	/// Whether that happens is a function of the packed width alone, so this pins the width rather
	/// than leaving it to `-Ctarget-cpu=native` and the host: under `PackedGhash4x128b` the
	/// `[0, 1]` batch reaches the case on every machine, while the 128-bit type the other tests use
	/// never does. That batch also lifts its first oracle (`log_lift = 1`), so the sub-packed
	/// placement is exercised on a lifted block rather than only on an unlifted one.
	///
	/// The `[1, 2]` batch sits just the other side of the boundary — its blocks are exactly one
	/// packed element — and covers the chunked path that the clamp must leave alone.
	/// [`place_repeated`] must match the definition it implements, at every shape.
	///
	/// The two regimes it splits on — a lift block spanning whole packed elements, and several
	/// blocks sharing one element — are selected by `log_block` against `P::LOG_WIDTH`, so the grid
	/// runs three packed widths against every `(log_src, log_block, log_dst)` they admit. That
	/// reaches shapes the FRI parameters do not currently produce, which is the point: the
	/// placement should not depend on which of them the optimizer happens to choose.
	#[test]
	fn place_repeated_matches_the_naive_placement() {
		fn check<P: PackedField<Scalar = B128>>(log_src: usize, log_block: usize, log_dst: usize) {
			let mut rng = StdRng::seed_from_u64(0);
			let src = random_field_buffer::<P>(&mut rng, log_src);
			let initial = random_field_buffer::<P>(&mut rng, log_dst);
			let scalar = B128::random(&mut rng);

			// The definition: `scalar * src` lands in the low `2^log_src` scalars of each
			// `2^log_block`-sized block, and nowhere else.
			let mut expected = initial.clone();
			for index in 0..1usize << log_dst {
				let position = index % (1usize << log_block);
				if position < 1usize << log_src {
					expected.set(index, expected.get(index) + scalar * src.get(position));
				}
			}

			let mut actual = initial;
			place_repeated(actual.as_mut_view(), src.as_view(), scalar, log_block);

			for index in 0..1usize << log_dst {
				assert_eq!(
					actual.get(index),
					expected.get(index),
					"P::LOG_WIDTH={}, log_src={log_src}, log_block={log_block}, log_dst={log_dst}, \
					 index={index}",
					P::LOG_WIDTH,
				);
			}
		}

		fn check_all_shapes<P: PackedField<Scalar = B128>>() {
			for log_dst in 0..=4 {
				for log_block in 0..=log_dst {
					for log_src in 0..=log_block {
						check::<P>(log_src, log_block, log_dst);
					}
				}
			}
		}

		check_all_shapes::<PackedGhash1x128b>();
		check_all_shapes::<PackedGhash2x128b>();
		check_all_shapes::<PackedGhash4x128b>();
	}

	#[test]
	fn batch_narrower_than_a_packed_element_proves() {
		const {
			assert!(
				PackedGhash4x128b::LOG_WIDTH > 1,
				"the fixture needs a packed element wider than the `[0, 1]` batch's lift block"
			);
		};
		for sizes in [[0, 1], [1, 2]] {
			assert!(
				run_zk_channel::<PackedGhash4x128b>(&sizes, false),
				"batch of {sizes:?}-variable oracles"
			);
		}
	}

	// Heterogeneous mixed/zero-ZK openings: each non-ZK oracle's batch fold is routed to the
	// *later* window of the first-fold challenge slice `[early ++ outer ++ later]`, so the
	// non-ZK oracles' batch-fold challenges come from the leading MLE rounds (which follow the
	// outer challenges in feed order) and land correctly in the FirstFold's later window.
	#[test]
	fn test_basefold_channel_mixed_zk_non_zk() {
		// One non-ZK oracle (8 vars) and one ZK oracle (6 vars): exercises conditional masking,
		// the heterogeneous combined-buffer lift/repeat placement, and the non-ZK unmasked commit.
		assert!(run_mixed_channel::<PackedGhash1x128b>(&[(8, false), (6, true)], false, None));
	}

	#[test]
	fn test_basefold_channel_zero_zk() {
		// All non-ZK oracles: γ must never be sampled and the proof must still verify.
		assert!(run_mixed_channel::<PackedGhash1x128b>(&[(6, false), (8, false)], false, None));
	}

	#[test]
	fn test_basefold_channel_mixed_invalid_proof() {
		// Tampering the claim on a mixed batch must be rejected.
		assert!(!run_mixed_channel::<PackedGhash1x128b>(&[(8, false), (6, true)], true, None));
	}

	/// A masked opening over `[(8, false), (6, true)]` writes, in order: the two round
	/// commitments and the Libra oracle's, then `s_h`, the `ℓ = 8` rounds of two coefficients, β,
	/// and the `α_r`. A digest is 32 bytes and a field element 16.
	const S_H_OFFSET: usize = 3 * 32;
	const ALPHA_0_OFFSET: usize = S_H_OFFSET + 16 + 8 * 2 * 16 + 16;

	#[test]
	fn test_basefold_channel_tampered_s_h_is_rejected() {
		let specs = [(8, false), (6, true)];
		assert!(!run_mixed_channel::<PackedGhash1x128b>(&specs, false, Some(S_H_OFFSET)));
	}

	#[test]
	fn test_basefold_channel_tampered_alpha_is_rejected() {
		let specs = [(8, false), (6, true)];
		assert!(!run_mixed_channel::<PackedGhash1x128b>(&specs, false, Some(ALPHA_0_OFFSET)));
	}

	#[test]
	fn test_basefold_channel_invalid_proof() {
		assert!(!run_zk_channel::<PackedGhash1x128b>(&[6, 8], true));
	}

	/// Generates a committed buffer of `n_vars` variables together with `n_relations` independent
	/// `(transparent, claim)` pairs, each opening the buffer at a different point.
	fn generate_oracle_relations<F, P, R: Rng>(
		rng: &mut R,
		n_vars: usize,
		n_relations: usize,
	) -> (FieldBuffer<P>, Vec<(FieldBuffer<P>, F)>)
	where
		F: BinaryField,
		P: PackedField<Scalar = F>,
	{
		let buffer = random_field_buffer::<P>(&mut *rng, n_vars);
		let relations = (0..n_relations)
			.map(|_| {
				let point = random_scalars::<F>(&mut *rng, n_vars);
				let transparent = eq_ind_partial_eval::<P>(&point);
				let claim = inner_product_buffers(&buffer, &transparent);
				(transparent, claim)
			})
			.collect();
		(buffer, relations)
	}

	/// Runs a full prove/verify cycle over oracles described as `(n_vars, is_zk, n_relations)`.
	///
	/// The relations are queued round-robin over the oracles, so they arrive interleaved and the
	/// channel's grouping by oracle is exercised. If `tamper` is set, the verifier's claim at that
	/// arrival position is corrupted; verification must then fail. Returns whether verification
	/// accepted.
	fn run_multi_relation_channel(specs: &[(usize, bool, usize)], tamper: Option<usize>) -> bool {
		type F = Ghash128b;
		type P = PackedGhash1x128b;

		let mut rng = StdRng::seed_from_u64(0);
		let data = specs
			.iter()
			.map(|&(n_vars, _, n_relations)| {
				generate_oracle_relations::<F, P, _>(&mut rng, n_vars, n_relations)
			})
			.collect::<Vec<_>>();

		// Arrival order of the relations, as `(oracle position, relation position)`.
		let max_relations = specs
			.iter()
			.map(|&(_, _, k)| k)
			.max()
			.expect("at least one oracle");
		let arrivals = (0..max_relations)
			.flat_map(|round| {
				specs
					.iter()
					.enumerate()
					.filter(move |&(_, &(_, _, k))| round < k)
					.map(move |(index, _)| (index, round))
			})
			.collect::<Vec<_>>();

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
		let oracle_specs = specs.iter().map(|&(n_vars, is_zk, _)| {
			if is_zk {
				OracleSpec::new_zk(1 << n_vars)
			} else {
				OracleSpec::new(1 << n_vars)
			}
		});

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			OracleSchedule::from_rounds(oracle_specs.map(|spec| [spec])),
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// === PROVER SIDE ===
		let ntt = make_ntt(verifier_compiler.max_log_domain_size());
		let prover_compiler =
			BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_rng = StdRng::seed_from_u64(1);
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
				prover_rng,
				GlobalAllocator,
			);

		let oracles = data
			.iter()
			.map(|(buffer, _)| prover_channel.send_oracle(buffer.as_view()))
			.collect::<Vec<_>>();
		for &(index, round) in &arrivals {
			let (transparent, claim) = &data[index].1[round];
			prover_channel.prove_oracle_relation(oracles[index], transparent.clone(), *claim);
		}
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let v_oracles = specs
			.iter()
			.map(|&(n_vars, _, _)| verifier_channel.recv_oracle(1 << n_vars, true).unwrap())
			.collect::<Vec<_>>();
		for (position, &(index, round)) in arrivals.iter().enumerate() {
			let (transparent, claim) = &data[index].1[round];
			let transparent = transparent.clone();
			let claim = if tamper == Some(position) {
				*claim + F::ONE
			} else {
				*claim
			};
			verifier_channel
				.verify_oracle_relation(v_oracles[index], eq_eval_fn(transparent), claim)
				.expect("verify_oracle_relation only queues");
		}
		verifier_channel.finish().is_ok()
	}

	#[test]
	fn test_basefold_channel_two_relations_one_oracle() {
		// Two claims on the same committed oracle: the channel batches them behind one μ.
		assert!(run_multi_relation_channel(&[(6, true, 2)], None));
	}

	#[test]
	fn test_basefold_channel_two_relations_one_oracle_invalid() {
		// Tampering the second of the two claims must be rejected.
		assert!(!run_multi_relation_channel(&[(6, true, 2)], Some(1)));
	}

	#[test]
	fn test_basefold_channel_mixed_relation_counts() {
		// A non-ZK oracle with one claim and a ZK oracle with three, arriving interleaved: the σ
		// accounting must follow the batched relations.
		assert!(run_multi_relation_channel(&[(8, false, 1), (6, true, 3)], None));
	}

	#[test]
	fn test_basefold_channel_mixed_relation_counts_invalid() {
		// Tampering a claim on the non-ZK oracle in a mixed batch must be rejected.
		assert!(!run_multi_relation_channel(&[(8, false, 2), (6, true, 2)], Some(2)));
	}

	/// Runs a full prove/verify cycle over oracles described as `(n_vars, is_zk)`, with one
	/// relation spanning the first two oracles and a single-oracle relation on the second.
	///
	/// Every further oracle appears in no relation. If `tamper`, the verifier's claim on the
	/// spanning relation is corrupted; verification must then fail. Returns whether verification
	/// accepted.
	fn run_spanning_relation_channel(specs: &[(usize, bool)], tamper: bool) -> bool {
		type F = Ghash128b;
		type P = PackedGhash1x128b;

		let mut rng = StdRng::seed_from_u64(0);
		let data = specs
			.iter()
			.map(|&(n_vars, _)| generate_oracle_relations::<F, P, _>(&mut rng, n_vars, 1))
			.collect::<Vec<_>>();
		let (_, relations_0) = &data[0];
		let (_, relations_1) = &data[1];
		let (transparent_0, claim_0) = relations_0[0].clone();
		let (transparent_1, claim_1) = relations_1[0].clone();
		let spanning_claim = claim_0 + claim_1;

		// A second relation on oracle 1 alone, so the spanning one is batched with another.
		let point = random_scalars::<F>(&mut rng, specs[1].0);
		let transparent_2 = eq_ind_partial_eval::<P>(&point);
		let claim_2 = inner_product_buffers(&data[1].0, &transparent_2);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
		let oracle_specs = specs.iter().map(|&(n_vars, is_zk)| {
			if is_zk {
				OracleSpec::new_zk(1 << n_vars)
			} else {
				OracleSpec::new(1 << n_vars)
			}
		});

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			OracleSchedule::from_rounds(oracle_specs.map(|spec| [spec])),
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// === PROVER SIDE ===
		let ntt = make_ntt(verifier_compiler.max_log_domain_size());
		let prover_compiler =
			BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
				StdRng::seed_from_u64(1),
				GlobalAllocator,
			);

		let oracles = data
			.iter()
			.map(|(buffer, _)| prover_channel.send_oracle(buffer.as_view()))
			.collect::<Vec<_>>();
		prover_channel.prove_multi_oracle_relation(
			[
				(oracles[0], transparent_0.clone()),
				(oracles[1], transparent_1.clone()),
			],
			spanning_claim,
		);
		prover_channel.prove_oracle_relation(oracles[1], transparent_2.clone(), claim_2);
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let v_oracles = specs
			.iter()
			.map(|&(n_vars, _)| verifier_channel.recv_oracle(1 << n_vars, true).unwrap())
			.collect::<Vec<_>>();
		let spanning_claim = if tamper {
			spanning_claim + F::ONE
		} else {
			spanning_claim
		};
		verifier_channel
			.verify_multi_oracle_relation(
				[
					(v_oracles[0], eq_eval_fn(transparent_0)),
					(v_oracles[1], eq_eval_fn(transparent_1)),
				],
				spanning_claim,
			)
			.expect("verify_oracle_relation only queues");
		verifier_channel
			.verify_oracle_relation(v_oracles[1], eq_eval_fn(transparent_2), claim_2)
			.expect("verify_oracle_relation only queues");
		verifier_channel.finish().is_ok()
	}

	#[test]
	fn test_basefold_channel_relation_spanning_two_oracles() {
		// Oracles of different sizes, so the spanning relation's operands differ in length.
		assert!(run_spanning_relation_channel(&[(6, true), (8, true)], false));
	}

	#[test]
	fn test_basefold_channel_relation_spanning_two_oracles_invalid() {
		assert!(!run_spanning_relation_channel(&[(6, true), (8, true)], true));
	}

	#[test]
	fn test_basefold_channel_relation_spanning_zk_and_non_zk() {
		// A non-ZK operand carries the weight 1-γ in the masked sumcheck, and the third oracle is
		// in no relation at all.
		assert!(run_spanning_relation_channel(&[(8, false), (6, true), (5, true)], false));
	}

	#[test]
	fn test_basefold_channel_relation_spanning_zk_and_non_zk_invalid() {
		assert!(!run_spanning_relation_channel(&[(8, false), (6, true), (5, true)], true));
	}

	/// Runs a prove/verify cycle over oracles grouped into rounds, with a challenge between rounds.
	///
	/// Each oracle is `(len, is_zk)`. Its buffer is random above `len` too, which no relation may
	/// see, and it carries two relations whose transparents vanish above `len`. If `tamper`, the
	/// verifier's first claim is corrupted. Returns whether verification accepted.
	fn run_round_trip<P: PackedField<Scalar = Ghash128b>>(
		rounds: &[Vec<(usize, bool)>],
		tamper: bool,
	) -> bool {
		type F = Ghash128b;

		let mut rng = StdRng::seed_from_u64(0);
		let schedule = OracleSchedule::from_rounds(
			rounds
				.iter()
				.map(|round| round.iter().map(|&(len, is_zk)| OracleSpec { len, is_zk })),
		);
		let data = schedule
			.specs()
			.iter()
			.map(|spec| {
				let n_vars = spec.log_msg_len();
				let buffer = random_field_buffer::<P>(&mut rng, n_vars);
				let relations = (0..2)
					.map(|_| {
						let point = random_scalars::<F>(&mut rng, n_vars);
						let mut transparent = eq_ind_partial_eval::<P>(&point);
						for i in spec.len..1 << n_vars {
							transparent.set(i, F::ZERO);
						}
						let claim = inner_product_buffers(&buffer, &transparent);
						(transparent, claim)
					})
					.collect::<Vec<_>>();
				(buffer, relations)
			})
			.collect::<Vec<_>>();

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			schedule,
			LOG_INV_RATE,
			calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE),
			&MinProofSizeStrategy,
		);
		let ntt = make_ntt(verifier_compiler.max_log_domain_size());
		let prover_compiler =
			BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

		// === PROVER SIDE ===
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
				StdRng::seed_from_u64(1),
				GlobalAllocator,
			);
		let mut data_iter = data.iter();
		let mut oracles = Vec::new();
		for round in rounds {
			for (buffer, _) in data_iter.by_ref().take(round.len()) {
				oracles.push(prover_channel.send_oracle(buffer.as_view()));
			}
			IPProverChannel::sample(&mut prover_channel);
		}
		for (&oracle, (_, relations)) in iter::zip(&oracles, &data) {
			for (transparent, claim) in relations {
				prover_channel.prove_oracle_relation(oracle, transparent.clone(), *claim);
			}
		}
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);
		let mut v_oracles = Vec::new();
		for round in rounds {
			for &(len, _) in round {
				v_oracles.push(verifier_channel.recv_oracle(len, true).unwrap());
			}
			IPVerifierChannel::sample(&mut verifier_channel);
		}
		for (position, (&oracle, (_, relations))) in iter::zip(&v_oracles, data).enumerate() {
			for (index, (transparent, claim)) in relations.into_iter().enumerate() {
				let claim = if tamper && position == 0 && index == 0 {
					claim + F::ONE
				} else {
					claim
				};
				verifier_channel
					.verify_oracle_relation(oracle, eq_eval_fn(transparent), claim)
					.expect("verify_oracle_relation only queues");
			}
		}
		verifier_channel.finish().is_ok()
	}

	#[test]
	fn test_basefold_channel_round_oracle_before_a_larger_round_mate() {
		// The 2^2 oracle arrives first but is placed after the 2^4 one, so its block is written
		// before the block below it.
		let rounds = [
			vec![(1 << 2, true), (1 << 4, true), (3, false)],
			vec![(1 << 5, true)],
		];
		assert!(run_round_trip::<PackedGhash1x128b>(&rounds, false));
		assert!(run_round_trip::<PackedGhash4x128b>(&rounds, false));
		assert!(!run_round_trip::<PackedGhash1x128b>(&rounds, true));
	}

	#[test]
	fn test_basefold_channel_tenants_nest_in_slack() {
		// The 9-entry oracle nests in the 15-entry one's slack, and the 2^2 one in the 9-entry
		// one's, all inside one 2^5 round oracle.
		let rounds = [vec![(15, false), (9, true), (1 << 2, true)]];
		assert!(run_round_trip::<PackedGhash1x128b>(&rounds, false));
		assert!(run_round_trip::<PackedGhash4x128b>(&rounds, false));
		assert!(!run_round_trip::<PackedGhash4x128b>(&rounds, true));
	}

	/// One oracle of `2^ℓ` slots, for `ℓ < 5`, holding more than half of them.
	fn arb_oracle() -> impl Strategy<Value = (usize, bool)> {
		(0usize..5).prop_flat_map(|log_len| {
			let min_len = (1usize << log_len >> 1) + 1;
			(min_len..=1usize << log_len, any::<bool>())
		})
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(32))]

		#[test]
		fn round_trip_proptest(
			rounds in prop::collection::vec(prop::collection::vec(arb_oracle(), 1..5), 1..5),
		) {
			// The wider packing mixes word-aligned blocks with sub-word ones.
			prop_assert!(run_round_trip::<PackedGhash1x128b>(&rounds, false));
			prop_assert!(run_round_trip::<PackedGhash4x128b>(&rounds, false));
		}
	}
}
