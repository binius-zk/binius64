// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP prover channel.

use std::ops::Deref;

use binius_alloc::Allocator;
use binius_field::{BinaryField, PackedField, util::powers};
use binius_iop::{
	basefold::layout::RoundLayout,
	channel::{OracleSchedule, OracleSpec},
	fri::FRIParams,
};
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_math::{
	FieldBuffer, FieldSlice, FieldSliceMut, inner_product::inner_product_par,
	line::extrapolate_line, ntt::AdditiveNTT,
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
	basefold::{index_rounds::IndexNode, prove_sumcheck_basefold},
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
	///
	/// A ZK round oracle also holds its ZK pad. It is stored in natural order; the encoder commits
	/// its reversal.
	message: FieldBuffer<P, Data>,
}

/// A relation queued by [`IOPProverChannel::prove_multi_oracle_relation`]: its operands, each an
/// oracle with the transparent multilinear it is opened against, backed by the caller's allocator.
type Relation<P, Data> = Vec<(BaseFoldOracle, FieldBuffer<P, Data>)>;

/// A prover channel that uses ZK BaseFold for all oracle commitments and openings.
///
/// Each round of the schedule is committed as one round oracle, laid out by
/// [`layout_rounds`](binius_iop::basefold::layout::layout_rounds): every oracle is copied into its
/// block as it arrives, and the round oracle is encoded and committed with the round's last one.
///
/// This channel owns an [`StdRng`] and generates random masks internally during
/// [`send_oracle`](IOPProverChannel::send_oracle). The caller provides only the raw witness
/// buffer (not doubled). The channel handles:
/// - Filling each ZK round's pad with uniform elements and reversing the round as it is committed
///   (see [`binius_iop::basefold::layout`])
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
	/// The relations queued by [`IOPProverChannel::prove_multi_oracle_relation`], opened together
	/// in [`Self::finish`].
	relations: Vec<Relation<P, A::Vec<P>>>,
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
	/// The RNG seeds the channel's own generator, whose only output is the ZK masks and pads.
	/// A mask hides a round oracle through the folding phase, and a pad hides the codeword symbols
	/// the verifier opens. Hiding is therefore only as strong as this RNG, so it must be a
	/// cryptographic one.
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
			relations: Vec::new(),
			rng: StdRng::from_rng(&mut rng),
			alloc,
		}
	}

	/// Encodes and commits the next round oracle, the one at index `committed_oracles.len()`.
	fn commit_round_oracle(&mut self, mut message: FieldBuffer<P, A::Vec<P>>) {
		let index = self.committed_oracles.len();
		let layout = &self.layouts[index];

		// A ZK round fills its pad with uniform elements. Every round is committed reversed, which
		// puts the pad on the lowest coefficients of its codeword. No operand reaches the pad,
		// since every transparent vanishes above its oracle's content.
		if layout.is_zk {
			for i in layout.pad() {
				message.set(i, F::random(&mut self.rng));
			}
		}

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
	/// call are processed here in one batch: batching into one relation, then one BaseFold sumcheck
	/// opening over every committed oracle (in oracle-index order). Mirrors
	/// [`BaseFoldVerifierChannel::finish`].
	///
	/// [`BaseFoldVerifierChannel::finish`]: binius_iop::basefold::channel::BaseFoldVerifierChannel::finish
	pub fn finish(self) {
		let n_remaining = self.schedule.specs().len() - self.n_sent;
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining",);

		if self.relations.is_empty() {
			return;
		}

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
			relations,
			rng: _,
			alloc,
		} = self;

		prove_batch_zk_basefold(
			&mut channel,
			ntt,
			layouts,
			&fri_params,
			committed_oracles,
			relations,
			&alloc,
		);
	}
}

/// Proves the combined ZK BaseFold opening over all committed round oracles.
///
/// This drives `channel` — the Merkle channel taken from the destructured
/// [`BaseFoldProverChannel`] — through its [`MerkleIPProverChannel`] interface, in the steps of
/// [`binius_iop::basefold::channel::BaseFoldVerifierChannel::finish`], which it mirrors:
/// batch the relations, send the mask claim `s' = Σ_r <ω_r, T_r>` and fold with γ if some round is
/// masked, then run one BaseFold sumcheck on `<G, T>`.
///
/// Everything per oracle runs in round order: the per-round data (`layouts`, `fri_params`,
/// `committed_oracles`) is indexed by round.
#[allow(clippy::too_many_arguments)]
fn prove_batch_zk_basefold<A, F, P, NTT, Channel>(
	channel: &mut Channel,
	ntt: &NTT,
	layouts: &[RoundLayout],
	fri_params: &FRIParams<F>,
	mut committed_oracles: Vec<CommittedOracleData<P, Channel::Commitment, A::Vec<P>>>,
	relations: Vec<Relation<P, A::Vec<P>>>,
	alloc: &A,
) where
	A: Allocator,
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
{
	assert_eq!(layouts.len(), committed_oracles.len());

	let log_n_oracles = log2_ceil_usize(layouts.len());

	let transparents = batch_relations(channel, layouts, relations, alloc);

	// The claim `s = Σ_r <M_r, T_r>`, which the verifier holds as the batched claim.
	let s = izip!(&committed_oracles, &transparents)
		.map(|(committed, transparent)| inner_product_par(&committed.message, transparent))
		.sum::<F>();

	// A masked opening sends the mask claim `s' = Σ_r <ω_r, T_r>`, then folds each masked message
	// to `(1 - γ)·M_r + γ·ω_r`, the fold the FRI opens, and targets `lerp(γ, s, s')`. An unmasked
	// round has `ω_r = M_r`, so its fold is a no-op and it contributes its own claim to `s'`.
	let (target, gamma) = if layouts.iter().any(|layout| layout.is_zk) {
		let _scope = tracing::debug_span!("Mask fold").entered();
		let s_prime = izip!(&committed_oracles, &transparents)
			.map(|(committed, transparent)| {
				let omega = committed.mask.as_ref().unwrap_or(&committed.message);
				inner_product_par(omega, transparent)
			})
			.sum::<F>();
		channel.send_one(s_prime);
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
		(extrapolate_line(s, s_prime, gamma), Some(gamma))
	} else {
		(s, None)
	};

	// Codeword commitments in oracle-index order, matching `fri_params.input_oracles()`.
	// The sumcheck binds the oracle-index variables `Z` first, and the FRI's outer tensor takes its
	// first challenge as the low bit of the oracle index, so it pairs oracle `2k` with `2k + 1`.
	// The oracle count is padded to a power of two with absent oracles.
	let mut nodes = Vec::with_capacity(1 << log_n_oracles);
	let mut committed_codewords = Vec::with_capacity(committed_oracles.len());
	for (fri_oracle, committed, transparent) in
		izip!(fri_params.input_oracles(), committed_oracles, transparents)
	{
		nodes.push(Some(IndexNode {
			log_w: committed.message.log_len() + fri_oracle.log_lift,
			message: committed.message,
			transparent,
		}));
		committed_codewords.push((committed.codeword, committed.commitment));
	}
	nodes.resize_with(1 << log_n_oracles, || None);

	let fri_folder = FRIFoldProver::new_batch(fri_params, ntt, committed_codewords);
	prove_sumcheck_basefold(nodes, target, gamma, fri_folder, channel, alloc);
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
	relations: Vec<Relation<P, A::Vec<P>>>,
	alloc: &A,
) -> Vec<FieldBuffer<P, A::Vec<P>>>
where
	A: Allocator,
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: MerkleIPProverChannel<F>,
{
	let mu = channel.sample();

	let mut accumulators = layouts.iter().map(|_| None).collect::<Vec<_>>();
	for (j, (mu_j, operands)) in izip!(powers(mu), relations).enumerate() {
		for (oracle, transparent) in operands {
			let log_len = layouts[oracle.round].log_len;
			let slot = &mut accumulators[oracle.round];
			if slot.is_none() && j == 0 && transparent.log_len() == log_len {
				// Weight 1 and spans the whole round oracle: take it as the accumulator.
				*slot = Some(transparent);
				continue;
			}
			let accumulator = slot.get_or_insert_with(|| FieldBuffer::zeros_in(alloc, log_len));
			let mut block = accumulator.chunk_mut(transparent.log_len(), oracle.block_offset);
			accumulate_scaled_buffer(block.chunk(), transparent.as_view(), P::broadcast(mu_j));
		}
	}

	izip!(accumulators, layouts)
		.map(|(accumulator, layout)| {
			accumulator.unwrap_or_else(|| FieldBuffer::zeros_in(alloc, layout.log_len))
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

		// A lone oracle whose block is the whole round is its round oracle, padding included. Any
		// other is copied into its block, content only, since a round-mate may nest in the slack
		// above it, or the round may have grown past it to hold the ZK pad.
		let message = if layout.block_offsets.len() == 1 && layout.log_len == spec.log_msg_len() {
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

		oracle
	}

	fn prove_multi_oracle_relation(
		&mut self,
		operands: impl IntoIterator<Item = (Self::Oracle, FieldBuffer<P, A::Vec<P>>)>,
		_claim: P::Scalar,
	) {
		// Queue the relation; the actual opening (sumchecks + combined
		// FRI) happens once, over all committed round oracles, in [`Self::finish`].
		let n_committed = self.committed_oracles.len();
		let operands = operands
			.into_iter()
			.map(|(oracle, transparent)| {
				assert!(
					oracle.round < n_committed,
					"oracle's round {} is not committed yet",
					oracle.round
				);
				assert_eq!(
					transparent.log_len(),
					oracle.spec.log_msg_len(),
					"transparent log_len must match the oracle's"
				);
				debug_assert!(
					transparent
						.iter_scalars()
						.skip(oracle.spec.len)
						.all(|value| value == F::ZERO),
					"an operand's transparent must vanish above its oracle's length"
				);
				(oracle, transparent)
			})
			.collect();
		self.relations.push(operands);
	}
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeSet, iter};

	use binius_alloc::GlobalAllocator;
	use binius_field::{
		BinaryField, Field, Ghash128b, Ghash128b as B128, PackedField, PackedGhash1x128b,
		PackedGhash4x128b,
	};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		basefold::{compiler::BaseFoldVerifierCompiler, layout::layout_rounds},
		channel::{IOPVerifierChannel, OracleSchedule, OracleSpec, TransparentEvalFn},
		fri::FRIParams,
		merkle_tree::BinaryMerkleTreeScheme,
	};
	use binius_ip::channel::IPVerifierChannel;
	use binius_ip_prover::channel::IPProverChannel;
	use binius_math::{
		FieldBuffer,
		inner_product::inner_product_buffers,
		multilinear::eq::eq_ind_partial_eval,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		reed_solomon::ReedSolomonCode,
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
	use proptest::prelude::*;
	use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

	use super::IOPProverChannel;
	use crate::{basefold::compiler::BaseFoldProverCompiler, fri};

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
	/// commitments, then the mask claim `s'`, then the sumcheck round polynomials of two
	/// coefficients each, the first of which is over the oracle-index variable `Z`. A digest is 32
	/// bytes and a field element 16.
	const S_PRIME_OFFSET: usize = 2 * 32;
	const Z_ROUND_OFFSET: usize = S_PRIME_OFFSET + 16;

	#[test]
	fn test_basefold_channel_tampered_s_prime_is_rejected() {
		let specs = [(8, false), (6, true)];
		assert!(!run_mixed_channel::<PackedGhash1x128b>(&specs, false, Some(S_PRIME_OFFSET)));
	}

	#[test]
	fn test_basefold_channel_tampered_z_round_is_rejected() {
		let specs = [(8, false), (6, true)];
		assert!(!run_mixed_channel::<PackedGhash1x128b>(&specs, false, Some(Z_ROUND_OFFSET)));
	}

	#[test]
	fn test_basefold_channel_untampered_mixed_is_accepted() {
		// The offsets above flip a byte; with none flipped the same opening verifies.
		let specs = [(8, false), (6, true)];
		assert!(run_mixed_channel::<PackedGhash1x128b>(&specs, false, None));
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
		// one's, all inside the first 2^5 entries. The round is masked, so its ZK pad grows the
		// round oracle to 2^6 above them.
		let rounds = [vec![(15, false), (9, true), (1 << 2, true)]];
		assert!(run_round_trip::<PackedGhash1x128b>(&rounds, false));
		assert!(run_round_trip::<PackedGhash4x128b>(&rounds, false));
		assert!(!run_round_trip::<PackedGhash4x128b>(&rounds, true));
	}

	/// The rank of a matrix over a field, by Gaussian elimination.
	fn rank<F: Field>(mut rows: Vec<Vec<F>>) -> usize {
		let n_cols = rows.first().map_or(0, Vec::len);
		let mut rank = 0;
		for col in 0..n_cols {
			let Some(pivot) = (rank..rows.len()).find(|&row| rows[row][col] != F::ZERO) else {
				continue;
			};
			rows.swap(rank, pivot);
			let pivot_inverse = rows[rank][col].invert_or_zero();
			let pivot_row = rows[rank].clone();
			for (index, row) in rows.iter_mut().enumerate() {
				if index != rank && row[col] != F::ZERO {
					let factor = row[col] * pivot_inverse;
					for (entry, &pivot_entry) in iter::zip(&mut row[col..], &pivot_row[col..]) {
						*entry -= factor * pivot_entry;
					}
				}
			}
			rank += 1;
		}
		rank
	}

	/// Any `q` opened symbols of a ZK round, restricted to its pad, are linearly independent.
	///
	/// This is what makes the opened symbols uniform given the rest of the round oracle. Encoding
	/// is linear, so the test encodes the unit vector of each pad coordinate with the encoder the
	/// round is committed with, and checks that their symbols at `q` positions form a matrix of
	/// full rank.
	///
	/// The positions `0..q` are the hard case. They lie in the low subspaces `U_k`, where a symbol
	/// reads only the entries the encoder takes as low coefficients. Without the reversal, those
	/// entries are the round oracle's first, and the pad reaches none of the symbols there. The
	/// encoder commits every round reversed, which puts the pad on those entries.
	#[test]
	fn any_q_opened_symbols_of_a_zk_round_are_independent_on_its_pad() {
		type F = B128;
		type P = PackedGhash1x128b;
		let n_test_queries = 20;

		let schedule = OracleSchedule::from_rounds([[OracleSpec::new_zk(100)]]);
		let layout = layout_rounds(&schedule, n_test_queries).swap_remove(0);
		let rs_code = ReedSolomonCode::<F>::new(layout.log_len, LOG_INV_RATE);
		let ntt = make_ntt(rs_code.log_len());

		let fri_params = FRIParams::new(rs_code.clone(), 0, vec![], n_test_queries);

		let encode_pad_unit_vectors = |reversed: bool| {
			layout
				.pad()
				.map(|coordinate| {
					let mut message = FieldBuffer::<P>::zeros(layout.log_len);
					message.set(coordinate, F::ONE);
					if reversed {
						fri::encode_interleaved(
							&fri_params,
							0,
							&ntt,
							message.as_view(),
							&GlobalAllocator,
						)
					} else {
						rs_code.encode_batch(&ntt, message.as_view(), 0, &GlobalAllocator)
					}
				})
				.collect::<Vec<_>>()
		};
		let codewords = encode_pad_unit_vectors(true);
		let rank_at = |positions: &[usize]| {
			let rows = positions
				.iter()
				.map(|&position| {
					codewords
						.iter()
						.map(|codeword| codeword.get(position))
						.collect()
				})
				.collect();
			rank::<F>(rows)
		};

		let low_positions = (0..n_test_queries).collect::<Vec<_>>();
		assert_eq!(rank_at(&low_positions), n_test_queries);

		let mut rng = StdRng::seed_from_u64(0);
		for _ in 0..16 {
			let mut positions = BTreeSet::new();
			while positions.len() < n_test_queries {
				positions.insert(rng.random_range(0..rs_code.len()));
			}
			let positions = positions.into_iter().collect::<Vec<_>>();
			assert_eq!(rank_at(&positions), n_test_queries, "positions {positions:?}");
		}

		// Without the reversal, the symbol at position 0 is the round oracle's coordinate 0.
		let unreversed = encode_pad_unit_vectors(false);
		assert!(unreversed.iter().all(|codeword| codeword.get(0) == F::ZERO));
	}

	/// Every symbol of a ZK round's committed codeword depends on the prover's coins.
	///
	/// This commits the same witness under two prover seeds and compares the `Enc(M_r)` half of
	/// the codewords. A symbol that agrees between them is a fixed function of the witness, which
	/// the query phase reveals whenever a query lands on its position. The witness starts at
	/// coordinate 0, which an unreversed encoder reads as the constant coefficient, so without
	/// the reversal the symbol at position 0 would be the witness's first entry.
	///
	/// The two specs cover a round that holds its pad in its slack, and one that grows for it.
	#[test]
	fn every_symbol_of_a_zk_round_depends_on_the_prover_coins() {
		type F = B128;
		type P = PackedGhash1x128b;
		let n_vars = 7;
		let witness = random_field_buffer::<P>(StdRng::seed_from_u64(0), n_vars);

		let commit = |spec: OracleSpec, seed: u64| -> Vec<F> {
			let verifier_compiler = BaseFoldVerifierCompiler::new(
				&make_merkle_scheme(),
				OracleSchedule::from_rounds([[spec]]),
				LOG_INV_RATE,
				calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE),
			);
			let ntt = make_ntt(verifier_compiler.max_log_domain_size());
			let prover_compiler =
				BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);
			let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
			let mut prover_channel = prover_compiler
				.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
					&mut prover_transcript,
					StdRng::seed_from_u64(seed),
					GlobalAllocator,
				);
			prover_channel.send_oracle(witness.as_view());

			// The codeword interleaves `Enc(M_r)` and `Enc(ω_r)` symbol by symbol.
			prover_channel.committed_oracles[0]
				.codeword
				.iter_scalars()
				.step_by(2)
				.collect()
		};

		for spec in [OracleSpec::new_zk(90), OracleSpec::new_zk(1 << n_vars)] {
			let symbols = commit(spec, 1);
			let other_symbols = commit(spec, 2);
			for (position, (symbol, other_symbol)) in
				iter::zip(&symbols, &other_symbols).enumerate()
			{
				assert_ne!(symbol, other_symbol, "{spec:?}, position {position}");
			}
		}
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
