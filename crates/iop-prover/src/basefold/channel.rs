// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP prover channel.

use std::ops::Deref;

use binius_compute::Allocator;
use binius_field::{BinaryField, Field, PackedField};
use binius_iop::{channel::OracleSpec, fri::FRIParams};
use binius_ip_prover::{
	channel::{IPProverChannel, WordIPProverChannel},
	sumcheck::{
		self, PaddedSumcheckDecorator, batch::BatchSumcheckOutput,
		bivariate_product_evaluator::BivariateProductEvaluator, mle_store::MleStore,
		round_evaluator::SharedSumcheckProver,
	},
};
use binius_math::{
	FieldBuffer, FieldSlice, FieldSliceMut, FieldVec,
	inner_product::inner_product_par,
	line::{extrapolate_line, extrapolate_line_packed},
	multilinear::eq::{eq_ind_partial_eval_scalars, eq_ind_zero},
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
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
	basefold::prove_mlecheck_basefold,
	channel::IOPProverChannel,
	fri::{self, FRIFoldProver, MaskedCodeword},
	merkle_channel::MerkleIPProverChannel,
};

/// Oracle handle returned by [`BaseFoldProverChannel::send_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldOracle {
	index: usize,
}

/// Committed oracle data stored internally.
struct CommittedOracleData<P: PackedField, C, Data: Deref<Target = [P]>> {
	/// The mask buffer generated during [`fri::encode_masked`] for a ZK oracle, held by the
	/// channel because it is the only party that knows it. `None` for a non-ZK (unmasked) oracle.
	mask: Option<FieldBuffer<P, Data>>,
	/// RS-encoded codeword, drawn from the channel's allocator.
	codeword: FieldBuffer<P, Data>,
	/// The Merkle commitment handle for query proofs, owning the committed tree.
	commitment: C,
	/// The committed multilinear message `pi_i`, backed by the caller's allocator. Handed over by
	/// [`IOPProverChannel::finalize_oracle`], and `None` until then.
	message: Option<FieldBuffer<P, Data>>,
}

/// A committed-oracle relation queued for the single batched opening.
struct QueuedRelation<P: PackedField, Data: Deref<Target = [P]>> {
	/// The transparent multilinear `t` the message is opened against, backed by the caller's
	/// allocator.
	transparent: FieldBuffer<P, Data>,
	/// The claimed inner product `s = <pi, t>`.
	claim: P::Scalar,
}

/// A prover channel that uses ZK BaseFold for all oracle commitments and openings.
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
	oracle_specs: Vec<OracleSpec>,
	/// The combined FRI parameters over all committed oracles.
	fri_params: FRIParams<F>,
	committed_oracles: Vec<CommittedOracleData<P, Channel::Commitment, A::Vec<P>>>,
	/// Oracle relations queued by [`IOPProverChannel::prove_oracle_relation`], indexed by oracle
	/// index and opened together in [`Self::finish`]. One entry per committed oracle, so its
	/// length is also the number of oracles committed so far.
	queue: Vec<Vec<QueuedRelation<P, A::Vec<P>>>>,
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
	/// length). The `rng` is used to seed an internal `StdRng` for mask generation.
	pub fn new(
		channel: Channel,
		ntt: &'a NTT,
		oracle_specs: Vec<OracleSpec>,
		fri_params: FRIParams<F>,
		mut rng: impl Rng,
		alloc: A,
	) -> Self {
		Self {
			channel,
			ntt,
			oracle_specs,
			fri_params,
			committed_oracles: Vec::new(),
			queue: Vec::new(),
			rng: StdRng::from_rng(&mut rng),
			alloc,
		}
	}

	/// Consumes the channel and proves the single combined opening over **all** committed oracles.
	///
	/// All oracle relations queued by
	/// [`prove_oracle_relation`](IOPProverChannel::prove_oracle_relation) across every call are
	/// processed here in one batch: masking, one batched sumcheck reducing the masked claims to a
	/// shared point `r`, then one combined FRI opening over every committed oracle
	/// (in oracle-index order). Mirrors [`BaseFoldVerifierChannel::finish`].
	///
	/// [`BaseFoldVerifierChannel::finish`]: binius_iop::basefold::channel::BaseFoldVerifierChannel::finish
	pub fn finish(self) {
		let Self {
			mut channel,
			ntt,
			oracle_specs,
			fri_params,
			committed_oracles,
			queue,
			rng: _,
			alloc,
		} = self;

		let n_remaining = oracle_specs.len() - queue.len();
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining",);

		if queue.iter().all(Vec::is_empty) {
			return;
		}

		prove_batch_zk_basefold(
			&mut channel,
			ntt,
			&oracle_specs,
			&fri_params,
			committed_oracles,
			queue,
			&alloc,
		);
	}
}

/// Proves the combined ZK BaseFold opening over all committed oracles.
///
/// This drives `channel` — the Merkle channel taken from the destructured
/// [`BaseFoldProverChannel`] — through its [`MerkleIPProverChannel`] interface: it sends the
/// masked inner products σ_i, runs one batched sumcheck reducing the masked claims to a shared
/// point `r`, then opens all committed oracles together with a single combined FRI. Mirrors
/// [`binius_iop::basefold::channel::BaseFoldVerifierChannel::finish`].
///
/// Everything runs in oracle-index order: `relations` arrives indexed by oracle, as do the
/// per-oracle data (`oracle_specs`, `fri_params`, `committed_oracles`), the masking inner products
/// σ_i, the sumcheck provers, the reduced evaluations α_i, and the FRI openings.
fn prove_batch_zk_basefold<A, F, P, NTT, Channel>(
	channel: &mut Channel,
	ntt: &NTT,
	oracle_specs: &[OracleSpec],
	fri_params: &FRIParams<F>,
	mut committed_oracles: Vec<CommittedOracleData<P, Channel::Commitment, A::Vec<P>>>,
	relations: Vec<Vec<QueuedRelation<P, A::Vec<P>>>>,
	alloc: &A,
) where
	A: Allocator,
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
{
	let n_committed = committed_oracles.len();
	assert_eq!(oracle_specs.len(), n_committed);
	assert_eq!(relations.len(), n_committed);

	// TODO: Remove this limitation, it shouldn't be necessary. It is currently because of how the
	// sumcheck reduces to the multilinear evaluations (alphas): an oracle with no relation gets no
	// sumcheck prover, so its α would have to come from a plain multilinear evaluation.
	assert!(
		relations.iter().all(|relations| !relations.is_empty()),
		"expects at least one relation per committed oracle",
	);

	// Take ownership of the committed messages π_i, leaving the masks and codewords in place. Every
	// committed oracle must have been handed back with `finalize_oracle`.
	let mut messages = committed_oracles
		.iter_mut()
		.enumerate()
		.map(|(index, oracle)| {
			oracle
				.message
				.take()
				.unwrap_or_else(|| panic!("oracle {index} was committed but never finalized"))
		})
		.collect::<Vec<_>>();

	// Batch each oracle's claims into one, so everything below runs exactly one relation per
	// committed oracle.
	let relations = batch_relations_per_oracle::<A, _, _, _>(channel, relations);

	// `𝐧 = max_i log_msg_len_i`, the variable count of the combined opening / materialized buffer.
	let max_n = oracle_specs
		.iter()
		.map(|spec| spec.log_msg_len)
		.max()
		.expect("at least one oracle");

	// === Masking step (whitepaper 7.2) ===
	// Only ZK oracles are masked. Send their σ_i = ⟨ω_i, T_i⟩ against the batched transparent (one
	// per ZK oracle), then sample the single shared masking challenge γ — skipped entirely when no
	// ZK oracle is present.
	let any_zk_openings = oracle_specs.iter().any(|spec| spec.is_zk);
	let (sigmas, gamma) = if any_zk_openings {
		let _scope = tracing::debug_span!("Compute ZK mask opening values").entered();
		let sigmas = izip!(&relations, oracle_specs, &committed_oracles)
			.filter(|(_, spec, _)| spec.is_zk)
			.map(|(relation, _, committed)| {
				let mask = committed.mask.as_ref().expect("ZK oracle carries a mask");
				inner_product_par(mask, &relation.transparent)
			})
			.collect::<Vec<_>>();
		channel.send_many(&sigmas);

		let gamma = channel.sample();

		(sigmas, Some(gamma))
	} else {
		(Vec::new(), None)
	};

	// Blind each ZK oracle's message in place: π_i' = (1-γ)π_i + γω_i. A non-ZK oracle's message
	// passes through unmasked.
	for (message, spec, committed) in izip!(&mut messages, oracle_specs, &committed_oracles) {
		let n_i = spec.log_msg_len;
		assert_eq!(message.log_len(), n_i); // pre-condition

		if spec.is_zk {
			let mask = committed.mask.as_ref().expect("ZK oracle carries a mask");
			let gamma_broadcast = P::broadcast(gamma.expect("γ sampled when ZK oracles present"));

			let _scope = tracing::debug_span!("Fold message and ZK mask", log_len = n_i).entered();
			(message.as_mut(), mask.as_ref())
				.into_par_iter()
				.with_min_task(WorkPerItem::FieldMuls)
				.for_each(|(message_i, &mask_i)| {
					*message_i = extrapolate_line_packed(*message_i, mask_i, gamma_broadcast);
				});
		}
	}

	// === Phase A: batched sumcheck on the masked claims ⟨π_i', T_i⟩ = s_i' ===
	// One prover per committed oracle, in oracle-index order, each padded to `max_n`.
	let mut sigma_iter = sigmas.into_iter();
	let provers = izip!(relations, &messages, oracle_specs)
		.map(|(QueuedRelation { transparent, claim }, message, spec)| {
			let n_i = spec.log_msg_len;
			assert_eq!(transparent.log_len(), n_i); // pre-condition

			// ZK oracle: mask the claim with σ_i.
			// Non-ZK oracle: the claim passes through unmasked.
			let sum_prime = if spec.is_zk {
				let sigma = sigma_iter.next().expect("one σ per ZK oracle");
				let gamma = gamma.expect("γ sampled when ZK oracles present");
				extrapolate_line(claim, sigma, gamma)
			} else {
				claim
			};

			let mut store = MleStore::new(n_i, alloc);
			let message_col = store.push(message.to_ref());
			let transparent_col = store.push_owned(transparent);
			let inner = SharedSumcheckProver::new(
				store,
				[(sum_prime, BivariateProductEvaluator::new([message_col, transparent_col]))],
			);
			PaddedSumcheckDecorator::new(inner, max_n - n_i, vec![sum_prime])
		})
		.collect::<Vec<_>>();

	let BatchSumcheckOutput {
		challenges,
		multilinear_evals,
	} = {
		let _scope =
			tracing::debug_span!("Reduce linear relations to committed openings").entered();
		sumcheck::batch_prove(provers, channel)
	};

	// Reduced oracle evaluations α_i = π_i'(ρ_i), one per committed oracle in oracle-index order.
	let alphas = multilinear_evals
		.iter()
		.map(|evals| evals[0])
		.collect::<Vec<_>>();
	channel.send_many(&alphas);

	// === Phase B: single combined-FRI MLE-check over the piecewise-concatenated oracle ===
	// Collapse the oracle-index variables up front at sampled batching challenges `r'`: build the
	// combined multilinear 𝛑(X) = Σ_i e[i]·π_i^↑(X) with e = eq(·, r') into one 2^𝐧 buffer, and the
	// combined target s' = 𝛑(r) = Σ_i e[i]·α_i·∏_{j≥n_i}(1 - r_j).
	// `batch_prove` returns binding-order challenges; reverse to variable-indexed (low-to-high),
	// so that ρ_i is the first n_i coords.
	let mut challenges = challenges;
	challenges.reverse();
	let point = &challenges;
	let log_n_oracles = log2_ceil_usize(n_committed);
	let outer_challenges = channel.sample_many(log_n_oracles);

	let (combined, s_prime) = {
		let _scope = tracing::debug_span!("Compute batched witness").entered();

		let eq_tensor = eq_ind_partial_eval_scalars(&outer_challenges);

		let mut combined = FieldBuffer::zeros_in(alloc, max_n);
		let mut s_prime = F::ZERO;
		for (fri_oracle, witness_prime, eq_i, alpha_i) in
			izip!(fri_params.input_oracles(), messages, eq_tensor, alphas)
		{
			let n_i = witness_prime.log_len();
			// Each oracle occupies the low 2^{n_i} of every 2^{log_lift}·2^{n_i}-sized lift block,
			// and that block is *repeated* across the 2^{log_repeat} high dims so the small
			// oracle is constant along them (matching the FRI lift/repeat structure).
			let log_lift = fri_oracle.log_lift;

			// Repeat placement: add scalar · π_i' into the first 2^{n_i} entries of each of the
			// 2^{log_repeat} chunks of size 2^{n_i + log_lift}.
			// Borrow as a slice before the closure: the allocator's buffer type is only `Send`, so
			// a closure capturing the owned buffer would not be `Sync` as `for_each` requires.
			place_repeated(combined.to_mut(), witness_prime.to_ref(), eq_i, n_i + log_lift);

			// Repeat dims contribute 1; only the lift dims contribute an eq-to-zero factor.
			s_prime += eq_i * alpha_i * eq_ind_zero(&point[n_i..][..log_lift]);
		}

		(combined, s_prime)
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

/// Batches each oracle's queued relations down to one, in oracle-index order.
///
/// An oracle carrying `k > 1` claims has them folded into a single claim against a single
/// transparent, using a batching challenge λ shared by every oracle:
///
/// ```text
/// T_i = Σ_j λ^j · t_ij     the combined transparent
/// S_i = Σ_j λ^j · s_ij     the combined claim
/// ```
///
/// The inner product is linear in the transparent, so `⟨π_i, T_i⟩ = S_i` holds exactly when every
/// `⟨π_i, t_ij⟩ = s_ij` does, except with probability at most `Σ_i (k_i - 1) / |F|` over λ. λ is
/// drawn after every claim it combines is already bound to the transcript, so no claim can be
/// chosen as a function of it. The batched per-oracle claims are then combined again by the
/// sumcheck's own outer batching coefficient.
///
/// Mirrors [`binius_iop::basefold::channel`]'s verifier-side batching.
fn batch_relations_per_oracle<A, F, P, Channel>(
	channel: &mut Channel,
	relations: Vec<Vec<QueuedRelation<P, A::Vec<P>>>>,
) -> Vec<QueuedRelation<P, A::Vec<P>>>
where
	A: Allocator,
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: MerkleIPProverChannel<F>,
{
	let lambda = channel.sample();

	relations
		.into_iter()
		.map(|relations| {
			let mut relations = relations.into_iter();
			let mut batched = relations
				.next()
				.expect("pre-condition: every committed oracle carries at least one relation");

			// Powers λ, λ², … scale the oracle's remaining relations into the first. An oracle
			// carrying a single relation folds nothing.
			let mut coeff = lambda;
			for relation in relations {
				// pre-condition: all of an oracle's transparents match its message length
				assert_eq!(relation.transparent.log_len(), batched.transparent.log_len());

				accumulate_scaled_buffer(
					batched.transparent.to_mut(),
					relation.transparent.to_ref(),
					P::broadcast(coeff),
				);
				batched.claim += coeff * relation.claim;
				coeff *= lambda;
			}
			batched
		})
		.collect()
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
	mut dst: FieldSliceMut<P>,
	src: FieldSlice<P>,
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
			accumulate_scaled_buffer(chunk_buf, src.to_ref(), scalar_broadcast);
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
	mut dst: FieldSliceMut<P>,
	src: FieldSlice<P>,
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
		&self.oracle_specs[self.queue.len()..]
	}

	fn send_oracle(&mut self, buffer: FieldSlice<P>) -> Self::Oracle {
		let remaining = self.remaining_oracle_specs();
		assert!(!remaining.is_empty(), "send_oracle called but no remaining oracle specs");

		let index = self.queue.len();
		let spec = &remaining[0];

		// ZK channel expects raw witness buffer (NOT doubled).
		assert_eq!(
			buffer.log_len(),
			spec.log_msg_len,
			"oracle buffer log_len mismatch: expected {}, got {}",
			spec.log_msg_len,
			buffer.log_len()
		);

		// Encode oracle `index` of the combined FRI parameters. ZK oracles interleave a fresh mask
		// (`encode_masked`); non-ZK oracles encode the message alone (`encode_interleaved`).
		let (codeword, mask) = if spec.is_zk {
			let MaskedCodeword { codeword, mask } = fri::encode_masked(
				&self.fri_params,
				index,
				self.ntt,
				buffer.to_ref(),
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
					buffer.to_ref(),
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
			.send_merkle_commitment(codeword.to_ref(), leaf_size);
		drop(merkle_scope);

		self.committed_oracles.push(CommittedOracleData {
			mask,
			codeword,
			commitment,
			message: None,
		});
		self.queue.push(Vec::new());

		BaseFoldOracle { index }
	}

	fn prove_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: FieldVec<P, A>,
		claim: P::Scalar,
	) {
		// Queue the relation under its oracle; the actual opening (masking + sumcheck + combined
		// FRI) happens once, over all committed oracles, in [`Self::finish`].
		let n_committed = self.queue.len();
		self.queue
			.get_mut(oracle.index)
			.unwrap_or_else(|| {
				panic!("oracle index {} out of bounds, expected < {n_committed}", oracle.index)
			})
			.push(QueuedRelation { transparent, claim });
	}

	fn finalize_oracle(&mut self, oracle: Self::Oracle, buffer: FieldVec<P, A>) {
		let committed = self
			.committed_oracles
			.get_mut(oracle.index)
			.unwrap_or_else(|| panic!("oracle index {} out of bounds", oracle.index));
		assert!(
			committed.message.replace(buffer).is_none(),
			"oracle {} finalized twice",
			oracle.index
		);
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_field::{
		BinaryField, BinaryField128bGhash, BinaryField128bGhash as B128, Field,
		PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b, PackedField,
		Random,
	};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		basefold::compiler::BaseFoldVerifierCompiler,
		channel::{IOPVerifierChannel, OracleSpec},
		fri::MinProofSizeStrategy,
		merkle_tree::BinaryMerkleTreeScheme,
	};
	use binius_math::{
		FieldBuffer,
		inner_product::inner_product_buffers,
		multilinear::eq::eq_ind_partial_eval,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::{IOPProverChannel, place_repeated};
	use crate::basefold::compiler::BaseFoldProverCompiler;

	type StdChallenger = HasherChallenger<StdDigest>;

	const LOG_INV_RATE: usize = 1;
	const SECURITY_BITS: usize = 32;

	fn calculate_n_test_queries(security_bits: usize, log_inv_rate: usize) -> usize {
		security_bits.div_ceil(log_inv_rate)
	}

	fn make_ntt(
		log_domain_size: usize,
	) -> NeighborsLastSingleThread<GaoMateerOnTheFly<BinaryField128bGhash>> {
		let domain_context = GaoMateerOnTheFly::generate(log_domain_size);
		NeighborsLastSingleThread::new(domain_context)
	}

	fn make_merkle_scheme() -> BinaryMerkleTreeScheme<BinaryField128bGhash, StdHashSuite> {
		BinaryMerkleTreeScheme::new()
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
		type F = BinaryField128bGhash;
		type P = PackedBinaryGhash1x128b;

		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 8;

		let (buffer, transparent_poly, eval_claim) =
			generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

		let oracle_specs = vec![OracleSpec::new_zk(n_vars)];

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			oracle_specs,
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

		let oracle = prover_channel.send_oracle(buffer.to_ref());
		assert_eq!(oracle.index, 0);

		prover_channel.prove_oracle_relation(oracle, transparent_poly.clone(), eval_claim);
		prover_channel.finalize_oracle(oracle, buffer);
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let v_oracle = verifier_channel.recv_oracle(n_vars, true).unwrap();

		verifier_channel
			.verify_oracle_relation(
				v_oracle,
				Box::new(move |point: &[F]| {
					let eq = eq_ind_partial_eval::<P>(point);
					inner_product_buffers(&transparent_poly, &eq)
				}),
				eval_claim,
			)
			.unwrap();
		verifier_channel.finish().unwrap();
	}

	#[test]
	fn test_basefold_channel_two_oracles() {
		type F = BinaryField128bGhash;
		type P = PackedBinaryGhash1x128b;

		let mut rng = StdRng::seed_from_u64(0);
		let n_vars_1 = 6;
		let n_vars_2 = 8;

		let (buffer_1, transparent_poly_1, eval_claim_1) =
			generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars_1);
		let (buffer_2, transparent_poly_2, eval_claim_2) =
			generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars_2);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

		let oracle_specs = vec![OracleSpec::new_zk(n_vars_1), OracleSpec::new_zk(n_vars_2)];

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			oracle_specs,
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

		let oracle_1 = prover_channel.send_oracle(buffer_1.to_ref());
		let oracle_2 = prover_channel.send_oracle(buffer_2.to_ref());

		prover_channel.prove_oracle_relation(oracle_1, transparent_poly_1.clone(), eval_claim_1);
		prover_channel.prove_oracle_relation(oracle_2, transparent_poly_2.clone(), eval_claim_2);
		prover_channel.finalize_oracle(oracle_1, buffer_1);
		prover_channel.finalize_oracle(oracle_2, buffer_2);
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let v_oracle_1 = verifier_channel.recv_oracle(n_vars_1, true).unwrap();
		let v_oracle_2 = verifier_channel.recv_oracle(n_vars_2, true).unwrap();

		let tp1 = transparent_poly_1;
		let tp2 = transparent_poly_2;

		verifier_channel
			.verify_oracle_relation(
				v_oracle_1,
				Box::new(move |point: &[F]| {
					let eq = eq_ind_partial_eval::<P>(point);
					inner_product_buffers(&tp1, &eq)
				}),
				eval_claim_1,
			)
			.unwrap();
		verifier_channel
			.verify_oracle_relation(
				v_oracle_2,
				Box::new(move |point: &[F]| {
					let eq = eq_ind_partial_eval::<P>(point);
					inner_product_buffers(&tp2, &eq)
				}),
				eval_claim_2,
			)
			.unwrap();
		verifier_channel.finish().unwrap();
	}

	/// Runs a full prove/verify cycle of the Batched ZK BaseFold channel over oracles of the given
	/// sizes. If `tamper`, the verifier's claim on the first oracle is corrupted; verification must
	/// then fail. Returns whether verification accepted.
	fn run_zk_channel<P: PackedField<Scalar = BinaryField128bGhash>>(
		n_vars_list: &[usize],
		tamper: bool,
	) -> bool {
		type F = BinaryField128bGhash;

		let mut rng = StdRng::seed_from_u64(0);
		let data: Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> = n_vars_list
			.iter()
			.map(|&n| generate_zk_oracle_data::<F, P, _>(&mut rng, n))
			.collect();

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
		let oracle_specs: Vec<OracleSpec> =
			n_vars_list.iter().map(|&n| OracleSpec::new_zk(n)).collect();

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			oracle_specs,
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
			.map(|(buffer, _, _)| prover_channel.send_oracle(buffer.to_ref()))
			.collect();
		for (oracle, (buffer, transparent, claim)) in iter::zip(oracles, &data) {
			prover_channel.prove_oracle_relation(oracle, transparent.clone(), *claim);
			prover_channel.finalize_oracle(oracle, buffer.clone());
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
			.map(|&n| verifier_channel.recv_oracle(n, true).unwrap())
			.collect();
		for (i, (oracle, (_, transparent, claim))) in iter::zip(v_oracles, &data).enumerate() {
			let transparent = transparent.clone();
			let claim = if tamper && i == 0 {
				*claim + F::ONE
			} else {
				*claim
			};
			verifier_channel
				.verify_oracle_relation(
					oracle,
					Box::new(move |point: &[F]| {
						let eq = eq_ind_partial_eval::<P>(point);
						inner_product_buffers(&transparent, &eq)
					}),
					claim,
				)
				.expect("verify_oracle_relation only queues");
		}
		verifier_channel.finish().is_ok()
	}

	/// Like `run_zk_channel` but with per-oracle `(n_vars, is_zk)` flags, exercising the mixed
	/// ZK/non-ZK opening. If `tamper`, the verifier's claim on the first oracle is corrupted.
	fn run_mixed_channel<P: PackedField<Scalar = BinaryField128bGhash>>(
		specs: &[(usize, bool)],
		tamper: bool,
	) -> bool {
		type F = BinaryField128bGhash;

		let mut rng = StdRng::seed_from_u64(0);
		let data: Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> = specs
			.iter()
			.map(|&(n, _)| generate_zk_oracle_data::<F, P, _>(&mut rng, n))
			.collect();

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
		let oracle_specs: Vec<OracleSpec> = specs
			.iter()
			.map(|&(n, is_zk)| {
				if is_zk {
					OracleSpec::new_zk(n)
				} else {
					OracleSpec::new(n)
				}
			})
			.collect();

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			oracle_specs,
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
			.map(|(buffer, _, _)| prover_channel.send_oracle(buffer.to_ref()))
			.collect();
		for (oracle, (buffer, transparent, claim)) in iter::zip(oracles, &data) {
			prover_channel.prove_oracle_relation(oracle, transparent.clone(), *claim);
			prover_channel.finalize_oracle(oracle, buffer.clone());
		}
		prover_channel.finish();

		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let v_oracles: Vec<_> = specs
			.iter()
			.map(|&(n, _)| verifier_channel.recv_oracle(n, true).unwrap())
			.collect();
		for (i, (oracle, (_, transparent, claim))) in iter::zip(v_oracles, &data).enumerate() {
			let transparent = transparent.clone();
			let claim = if tamper && i == 0 {
				*claim + F::ONE
			} else {
				*claim
			};
			verifier_channel
				.verify_oracle_relation(
					oracle,
					Box::new(move |point: &[F]| {
						let eq = eq_ind_partial_eval::<P>(point);
						inner_product_buffers(&transparent, &eq)
					}),
					claim,
				)
				.expect("verify_oracle_relation only queues");
		}
		verifier_channel.finish().is_ok()
	}

	#[test]
	fn test_basefold_channel_three_oracles_non_power_of_two() {
		// 3 oracles (not a power of two) of unequal sizes: exercises oracle padding (Lifted FRI)
		// and the `⌈log 3⌉ = 2` outer oracle-combine rounds.
		assert!(run_zk_channel::<PackedBinaryGhash1x128b>(&[5, 6, 8], false));
	}

	/// A batch whose lift blocks are narrower than one packed field element must still prove.
	///
	/// Placing an oracle into the combined buffer chunks that buffer by the lift block. Every
	/// block is lifted to the combined dimension, so a block narrower than a packed element means
	/// the whole buffer is — there is no whole-packed chunk to place into, and the placement has
	/// to write into part of a single element instead.
	///
	/// Whether that happens is a function of the packed width alone, so this pins the width rather
	/// than leaving it to `-Ctarget-cpu=native` and the host: under `PackedBinaryGhash4x128b` the
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
			place_repeated(actual.to_mut(), src.to_ref(), scalar, log_block);

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

		check_all_shapes::<PackedBinaryGhash1x128b>();
		check_all_shapes::<PackedBinaryGhash2x128b>();
		check_all_shapes::<PackedBinaryGhash4x128b>();
	}

	#[test]
	fn batch_narrower_than_a_packed_element_proves() {
		const {
			assert!(
				PackedBinaryGhash4x128b::LOG_WIDTH > 1,
				"the fixture needs a packed element wider than the `[0, 1]` batch's lift block"
			)
		};
		for sizes in [[0, 1], [1, 2]] {
			assert!(
				run_zk_channel::<PackedBinaryGhash4x128b>(&sizes, false),
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
		assert!(run_mixed_channel::<PackedBinaryGhash1x128b>(&[(8, false), (6, true)], false));
	}

	#[test]
	fn test_basefold_channel_zero_zk() {
		// All non-ZK oracles: γ must never be sampled and the proof must still verify.
		assert!(run_mixed_channel::<PackedBinaryGhash1x128b>(&[(6, false), (8, false)], false));
	}

	#[test]
	fn test_basefold_channel_mixed_invalid_proof() {
		// Tampering the claim on a mixed batch must be rejected.
		assert!(!run_mixed_channel::<PackedBinaryGhash1x128b>(&[(8, false), (6, true)], true));
	}

	#[test]
	fn test_basefold_channel_invalid_proof() {
		assert!(!run_zk_channel::<PackedBinaryGhash1x128b>(&[6, 8], true));
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
		type F = BinaryField128bGhash;
		type P = PackedBinaryGhash1x128b;

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
		let oracle_specs = specs
			.iter()
			.map(|&(n_vars, is_zk, _)| {
				if is_zk {
					OracleSpec::new_zk(n_vars)
				} else {
					OracleSpec::new(n_vars)
				}
			})
			.collect::<Vec<_>>();

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&make_merkle_scheme(),
			oracle_specs,
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
			.map(|(buffer, _)| prover_channel.send_oracle(buffer.to_ref()))
			.collect::<Vec<_>>();
		for &(index, round) in &arrivals {
			let (transparent, claim) = &data[index].1[round];
			prover_channel.prove_oracle_relation(oracles[index], transparent.clone(), *claim);
		}
		for (oracle, (buffer, _)) in iter::zip(&oracles, &data) {
			prover_channel.finalize_oracle(*oracle, buffer.clone());
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
			.map(|&(n_vars, _, _)| verifier_channel.recv_oracle(n_vars, true).unwrap())
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
				.verify_oracle_relation(
					v_oracles[index],
					Box::new(move |point: &[F]| {
						let eq = eq_ind_partial_eval::<P>(point);
						inner_product_buffers(&transparent, &eq)
					}),
					claim,
				)
				.expect("verify_oracle_relation only queues");
		}
		verifier_channel.finish().is_ok()
	}

	#[test]
	fn test_basefold_channel_two_relations_one_oracle() {
		// Two claims on the same committed oracle: the channel batches them behind one λ.
		assert!(run_multi_relation_channel(&[(6, true, 2)], None));
	}

	#[test]
	fn test_basefold_channel_two_relations_one_oracle_invalid() {
		// Tampering the second of the two claims must be rejected.
		assert!(!run_multi_relation_channel(&[(6, true, 2)], Some(1)));
	}

	#[test]
	fn test_basefold_channel_mixed_relation_counts() {
		// A non-ZK oracle with one claim and a ZK oracle with three, arriving interleaved: only the
		// multi-claim oracle draws a λ, and the σ accounting must follow the batched relations.
		assert!(run_multi_relation_channel(&[(8, false, 1), (6, true, 3)], None));
	}

	#[test]
	fn test_basefold_channel_mixed_relation_counts_invalid() {
		// Tampering a claim on the non-ZK oracle in a mixed batch must be rejected.
		assert!(!run_multi_relation_channel(&[(8, false, 2), (6, true, 2)], Some(2)));
	}
}
