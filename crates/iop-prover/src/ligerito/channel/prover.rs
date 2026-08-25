// Copyright 2026 The Binius Developers

//! The prover channel: a queue of relations in, one ladder opening out.

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_iop::{channel::OracleSpec, ligerito::LigeritoParams};
use binius_ip_prover::{
	channel::{IPProverChannel, WordIPProverChannel},
	sumcheck::{
		self, PaddedSumcheckDecorator, batch::BatchSumcheckOutput,
		bivariate_product_evaluator::BivariateProductEvaluator, mle_store::MleStore,
		round_evaluator::SharedSumcheckProver,
	},
};
use binius_math::{
	FieldBuffer, FieldSlice, FieldVec, line::extrapolate_line, multilinear::hypercube::Hypercube,
	ntt::AdditiveNTT,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use itertools::izip;
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{
	LigeritoOracle, combined_message::CombinedMessage, committed_oracle::CommittedOracle,
	mask::Mask, relation::QueuedRelation,
};
use crate::{
	channel::{IOPProverChannel, grinding::GrindingProverChannel},
	fri::BatchBrakedownOracleProver,
	ligerito::{LigeritoProver, opening::commit_level},
	merkle_channel::MerkleIPProverChannel,
};

/// A prover channel that opens every committed oracle with one Ligerito ladder.
///
/// The counterpart of the Ligerito verifier channel, where the reductions are described.
///
/// Each oracle is masked or not according to its own specification, so a batch may mix the two.
/// A masked oracle draws a fresh mask of its message's own length from the channel's generator.
/// That mask is committed in a lane beside the message and never leaves the channel.
///
/// # Type Parameters
///
/// - `'a`: the lifetime of the parameters and the transform the compiler owns
/// - `F`: the binary field the ladder is committed over
/// - `P`: the packed field the buffers are held in
/// - `NTT`: the additive transform every level encodes over
/// - `Channel`: the Merkle channel carrying all prover interaction
/// - `A`: the allocator the queued transparents and the opening's working buffers are drawn from
pub struct LigeritoProverChannel<'a, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
	A: Allocator,
{
	/// The Merkle channel carrying all prover interaction.
	/// Field elements, challenges, commitments and openings all pass through it.
	channel: Channel,
	/// The transform every level encodes over, sized for the longest of them.
	ntt: &'a NTT,
	/// The ladder the opening runs down.
	params: &'a LigeritoParams,
	/// The oracles this channel expects, in the order it will commit them.
	oracle_specs: Vec<OracleSpec>,
	/// The committed oracles, in the order their commitments were sent.
	oracles: Vec<CommittedOracle<P, Channel::Commitment, A>>,
	/// Relations queued against each oracle, all opened together once the caller finishes.
	/// One entry per committed oracle, so its length is also the number committed so far.
	queue: Vec<Vec<QueuedRelation<P, A>>>,
	/// The generator every mask is drawn from, seeded once when the channel is built.
	rng: StdRng,
	/// The allocator the opening's working buffers are drawn from.
	alloc: A,
}

impl<'a, F, P, NTT, Channel, A> LigeritoProverChannel<'a, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F, Word = Word>,
	A: Allocator,
{
	/// Creates a channel that opens the given oracles with the given ladder.
	///
	/// `rng` seeds the generator every mask is drawn from.
	/// A channel with no zero-knowledge oracle never reads it, so its seed cannot reach a proof.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` is non-empty.
	/// * The longest message is the ladder's message length.
	/// * `ntt`'s domain covers every level's codeword domain.
	pub fn new(
		channel: Channel,
		ntt: &'a NTT,
		oracle_specs: Vec<OracleSpec>,
		params: &'a LigeritoParams,
		mut rng: impl Rng,
		alloc: A,
	) -> Self {
		assert!(
			!oracle_specs.is_empty(),
			"precondition: a Ligerito channel opens at least one oracle"
		);
		assert_eq!(
			oracle_specs
				.iter()
				.map(|spec| spec.log_msg_len)
				.max()
				.expect("oracle_specs is non-empty"),
			params.log_msg_len(),
			"precondition: the longest oracle's message length must be the ladder's message length"
		);

		Self {
			channel,
			ntt,
			params,
			oracle_specs,
			oracles: Vec::new(),
			queue: Vec::new(),
			rng: StdRng::from_rng(&mut rng),
			alloc,
		}
	}

	/// Consumes the channel and proves every queued relation in one opening.
	///
	/// Mirrors the verifier's own finishing step, message for message.
	/// Nothing happens when no relation was queued, since commitments alone assert nothing.
	///
	/// The channel has to be able to pay a proof of work, because the ladder may ask for one.
	/// A configuration that grinds nothing still asks for the capability, and never uses it.
	///
	/// ## Preconditions
	///
	/// * Either every oracle carries a relation, or none of them does.
	/// * Every oracle carrying a relation was finalized.
	pub fn finish(self)
	where
		Channel: GrindingProverChannel,
	{
		let Self {
			mut channel,
			ntt,
			params,
			oracle_specs,
			oracles,
			queue,
			rng: _,
			alloc,
		} = self;

		// Every oracle is committed at most once, so what remains is whatever was not.
		let n_remaining = oracle_specs.len() - oracles.len();
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining");

		if queue.iter().all(Vec::is_empty) {
			return;
		}
		assert!(
			queue.iter().all(|relations| !relations.is_empty()),
			"precondition: every committed oracle must carry at least one relation"
		);

		// The codewords answer the query phase; the masks and messages drive the reductions.
		let mut codewords = Vec::with_capacity(oracles.len());
		let mut masks = Vec::with_capacity(oracles.len());
		let mut messages = Vec::with_capacity(oracles.len());
		for oracle in oracles {
			let parts = oracle.split();
			codewords.push(parts.codeword);
			masks.push(parts.mask);
			messages.push(parts.message);
		}
		let prover = LigeritoProver::new(params, ntt, BatchBrakedownOracleProver::new(codewords));

		Self::prove(&mut channel, &prover, &oracle_specs, &masks, messages, queue, &alloc);
	}

	/// Proves the queued relations against the committed ladder.
	///
	/// The reductions the verifier runs, in the same order.
	///
	/// ```text
	///     <pi_i, t_ij> = s_ij      the relations, batched per oracle into <pi_i, T_i> = S_i
	///       -> blend               at gamma, against sigma_i = <omega_i, T_i>
	///     <pi_i', T_i> = S_i'      one claim per oracle, about the blended message
	///       -> sumcheck            binds every variable at a sampled point r
	///     pi_i'(r) = alpha_i       one evaluation claim per oracle
	///       -> eq-combine          at coefficients e_i drawn once every alpha is on the wire
	///     PI(r) = sum_i e_i alpha_i
	///       -> ladder
	/// ```
	///
	/// An oracle committed in the clear skips the blend, and its message rides on unchanged.
	///
	/// The sumcheck leaves both multilinears of every oracle evaluated at `r`.
	/// Only the committed ones have to be sent, since the verifier builds the transparents itself.
	fn prove(
		channel: &mut Channel,
		prover: &LigeritoProver<'_, P, Channel::Commitment, NTT>,
		oracle_specs: &[OracleSpec],
		masks: &[Option<Mask<P, A>>],
		mut messages: Vec<FieldVec<P, A>>,
		queue: Vec<Vec<QueuedRelation<P, A>>>,
		alloc: &A,
	) where
		Channel: GrindingProverChannel,
	{
		let n_oracles = messages.len();
		let max_n_vars = prover.params().log_msg_len();

		// Every claim in the queue is already bound to the transcript, so a coefficient drawn
		// here cannot be anticipated by any of them.
		let lambda = channel.sample();
		let relations = queue
			.into_iter()
			.map(|relations| QueuedRelation::batch(relations, lambda))
			.collect::<Vec<_>>();

		// What each mask pairs to against the transparent its oracle was just batched against.
		// Sending these before the blending challenge is what stops a mask being chosen for it.
		let sigmas = izip!(masks, &relations)
			.filter_map(|(mask, relation)| {
				mask.as_ref()
					.map(|mask| mask.pair(relation.transparent.as_view()))
			})
			.collect::<Vec<_>>();
		channel.send_many(&sigmas);
		let gamma = (!sigmas.is_empty()).then(|| channel.sample());

		// Every masked message becomes the blend the whole ladder is then about.
		for (message, mask) in std::iter::zip(&mut messages, masks) {
			if let Some(mask) = mask {
				mask.blend(message, gamma.expect("the challenge is drawn when a mask is present"));
			}
		}

		// One padded sumcheck prover per oracle, in the order the oracles were committed. A
		// message shorter than the longest one rides the same rounds, its claim carried through
		// the extra ones by the equality indicator at zero.
		let mut sigmas = sigmas.into_iter();
		let provers = izip!(relations, &messages, oracle_specs)
			.map(|(relation, message, spec)| {
				let QueuedRelation { transparent, claim } = relation;
				let n_vars = spec.log_msg_len;

				// A masked oracle's claim moves onto the blended message, along the same line.
				let claim = if spec.is_zk {
					extrapolate_line(
						claim,
						sigmas.next().expect("one mask pairing per masked oracle"),
						gamma.expect("the challenge is drawn when a mask is present"),
					)
				} else {
					claim
				};

				let mut store = MleStore::new(n_vars, alloc);
				let message_col = store.push(message.as_view());
				let transparent_col = store.push_owned(transparent);
				let inner = SharedSumcheckProver::new(
					store,
					[(claim, BivariateProductEvaluator::new([message_col, transparent_col]))],
				);
				PaddedSumcheckDecorator::new(inner, max_n_vars - n_vars, vec![claim])
			})
			.collect::<Vec<_>>();

		let BatchSumcheckOutput {
			mut challenges,
			multilinear_evals,
		} = sumcheck::batch_prove(provers, channel);

		// The store holds each message first, so its evaluation is the one the ladder opens.
		let alphas = multilinear_evals
			.iter()
			.map(|evals| evals[0])
			.collect::<Vec<_>>();
		channel.send_many(&alphas);

		// Sumcheck rounds bind the highest variable first, so reversing gives variable order.
		challenges.reverse();
		let point = challenges;

		// Every stated evaluation is now bound to the transcript, so the coefficients that combine
		// the messages into one cannot be anticipated either.
		let outer_challenges = channel.sample_many(log2_ceil_usize(n_oracles));
		let coefficients = Hypercube::One.expand(&outer_challenges).build_scalars();

		let mut combined = CombinedMessage::zeros_in(alloc, max_n_vars);
		let mut combined_claim = F::ZERO;
		for (message, spec, coefficient, alpha) in
			izip!(messages, oracle_specs, &coefficients, &alphas)
		{
			combined.add_scaled(message.as_view(), *coefficient);
			combined_claim +=
				*coefficient * *alpha * Hypercube::One.eq_ind_zero(&point[spec.log_msg_len..]);
		}

		let combined = combined.into_buffer();
		prover.prove(combined.as_view(), &point, combined_claim, alloc, channel);
	}
}

impl<F, P, NTT, Channel, A> IPProverChannel<F> for LigeritoProverChannel<'_, F, P, NTT, Channel, A>
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
	for LigeritoProverChannel<'_, F, P, NTT, Channel, A>
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
	for LigeritoProverChannel<'a, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
	A: Allocator,
{
	type Oracle = LigeritoOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.oracles.len()..]
	}

	fn send_oracle(&mut self, buffer: FieldSlice<'_, P>) -> Self::Oracle {
		let remaining = self.remaining_oracle_specs();
		assert!(!remaining.is_empty(), "send_oracle called but no remaining oracle specs");
		assert_eq!(
			buffer.log_len(),
			remaining[0].log_msg_len,
			"oracle buffer log_len mismatch: expected {}, got {}",
			remaining[0].log_msg_len,
			buffer.log_len()
		);
		let spec = remaining[0];

		// Every oracle shares level 0's column count, so only the lane count follows the message
		// length. A message below one column block commits a single zero-padded lane, and a
		// zero-knowledge oracle commits one lane more, holding its mask.
		let level = self.params.level_zero_shape(&spec);
		// The length each half of the committed buffer is padded out to, mask half included.
		let log_padded_len = level.log_msg_len() - usize::from(spec.is_zk);

		// A masked oracle draws its mask here, before any challenge in the transcript exists.
		let mask = spec
			.is_zk
			.then(|| Mask::draw(&self.alloc, buffer.log_len(), &mut self.rng));

		// Only a message below one column block needs padding, so a longer one is never copied.
		let padded = match &mask {
			Some(mask) => Some(mask.interleaved_with(&self.alloc, buffer, log_padded_len)),
			None => (buffer.log_len() < log_padded_len).then(|| {
				FieldBuffer::from_view_in(&self.alloc, buffer)
					.zero_extend_in(&self.alloc, log_padded_len)
			}),
		};
		let message = padded
			.as_ref()
			.map_or_else(|| buffer.as_view(), FieldBuffer::as_view);

		// Encoding and committing level 0 is the whole of the commit phase.
		// The deeper levels only exist once the folds above them have run.
		let index = self.oracles.len();
		let codeword = commit_level(&level, self.ntt, message, &mut self.channel);
		self.oracles.push(CommittedOracle::new(codeword, mask));
		self.queue.push(Vec::new());

		LigeritoOracle { index }
	}

	fn prove_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: FieldVec<P, A>,
		claim: P::Scalar,
	) {
		// A handle can only exist once its oracle was committed, so the slot is already there.
		let n_committed = self.queue.len();
		self.queue
			.get_mut(oracle.index)
			.unwrap_or_else(|| {
				panic!("oracle index {} out of bounds, expected < {n_committed}", oracle.index)
			})
			.push(QueuedRelation { transparent, claim });
	}

	fn finalize_oracle(&mut self, oracle: Self::Oracle, buffer: FieldVec<P, A>) {
		// The ladder folds the message down to the residual.
		// So it needs the buffer itself and not just the codeword it was encoded into.
		self.oracles[oracle.index].finalize(buffer);
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, Ghash128b as B128, Random, arithmetic_traits::InvertOrZero};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		channel::{Error, IOPVerifierChannel, grinding::GrindingVerifierChannel},
		ligerito::{LigeritoLevel, compiler::LigeritoVerifierCompiler},
		merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
		soundness::{Grinding, SoundnessRegime},
	};
	use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
	use binius_math::{
		multilinear::Multilinear,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::random_field_buffer,
	};
	use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::ligerito::compiler::LigeritoProverCompiler;

	type StdChallenger = HasherChallenger<StdDigest>;

	/// One oracle a run commits, opens, and states relations about.
	///
	/// Splitting the committed message from the opened one is how a dishonest prover is expressed.
	/// Every value it sends in the clear is consistent with the opened buffer.
	/// The codeword level 0's queries land in encodes the committed one.
	/// An honest oracle holds the same buffer twice.
	#[derive(Clone)]
	struct Oracle {
		/// The buffer level 0's codeword encodes.
		committed: FieldBuffer<B128>,
		/// The buffer every value sent in the clear is consistent with.
		opened: FieldBuffer<B128>,
		/// The transparent multilinears, each with the inner product claimed against it.
		relations: Vec<(FieldBuffer<B128>, B128)>,
		/// Whether this oracle is committed with a mask interleaved beside its message.
		is_zk: bool,
	}

	impl Oracle {
		/// An honest oracle over `log_msg_len` variables, carrying `n_relations` true claims.
		///
		/// A dense random weight rather than an equality indicator.
		/// So each relation is a general inner product, not the evaluation claim the ladder takes.
		fn honest(rng: &mut StdRng, log_msg_len: usize, n_relations: usize) -> Self {
			let message = random_field_buffer::<B128>(&mut *rng, log_msg_len);
			let relations = (0..n_relations)
				.map(|_| {
					let transparent = random_field_buffer::<B128>(&mut *rng, log_msg_len);
					let claim = message.inner_product(&transparent);
					(transparent, claim)
				})
				.collect();
			Self {
				committed: message.clone(),
				opened: message,
				relations,
				is_zk: false,
			}
		}

		/// The same oracle, asking to be committed with a mask.
		fn masked(mut self) -> Self {
			self.is_zk = true;
			self
		}

		/// A different message satisfying this oracle's single relation, claim for claim.
		///
		/// The two messages differ by a vector the transparent pairs to zero with.
		/// Two positions carry that difference:
		///
		///     delta[0] = d,   delta[h] = d * t[0] / t[h]
		///     <t, delta> = t[0] * d + t[h] * d * t[0] / t[h] = 0   over a binary field
		///
		/// So the claim on the wire is unchanged and only the witness behind it moves.
		/// That is the pair a hiding argument has to keep apart.
		///
		/// ## Preconditions
		///
		/// * The oracle carries exactly one relation, over at least one variable.
		fn sibling(&self, rng: &mut StdRng) -> Self {
			assert_eq!(self.relations.len(), 1, "a sibling is built against one transparent");
			let log_msg_len = self.opened.log_len();
			let (transparent, _) = &self.relations[0];

			let (left, right) = (0, 1 << (log_msg_len - 1));
			let head = B128::random(&mut *rng);
			let tail = head * transparent.get(left) * transparent.get(right).invert_or_zero();

			let mut message = self.opened.clone();
			message.set(left, message.get(left) + head);
			message.set(right, message.get(right) + tail);

			Self {
				committed: message.clone(),
				opened: message,
				relations: self.relations.clone(),
				is_zk: self.is_zk,
			}
		}

		/// The oracle's shape, as the channel is told about it up front.
		fn spec(&self) -> OracleSpec {
			if self.is_zk {
				OracleSpec::new_zk(self.opened.log_len())
			} else {
				OracleSpec::new(self.opened.log_len())
			}
		}
	}

	/// A ladder whose level `i` commits at inverse rate `2^(i + 1)` and opens `n_queries` rows.
	///
	/// `lanes[i]` is level `i`'s fold amount, and `log_msg_cols` is level 0's column count.
	/// Level `i + 1` takes what level `i` folds to, so its columns are `cols_i - lanes_{i+1}`.
	fn ladder(log_msg_cols: usize, lanes: &[usize]) -> LigeritoParams {
		let mut log_msg_cols = log_msg_cols;
		let levels = lanes
			.iter()
			.enumerate()
			.map(|(i, &log_lanes)| {
				// Level 0 keeps the column count it was given; every deeper one loses its lanes.
				if i > 0 {
					log_msg_cols -= log_lanes;
				}
				LigeritoLevel {
					log_msg_cols,
					log_lanes,
					log_inv_rate: i + 1,
					n_queries: 5,
				}
			})
			.collect();
		LigeritoParams::new(levels, SoundnessRegime::default(), 32)
	}

	/// One honest oracle filling the ladder, carrying one relation.
	fn one_oracle(params: &LigeritoParams, seed: u64) -> Vec<Oracle> {
		let mut rng = StdRng::seed_from_u64(seed);
		vec![Oracle::honest(&mut rng, params.log_msg_len(), 1)]
	}

	/// The Merkle channel one verification runs over.
	type InnerChannel<'a> = VerifierMerkleTranscriptChannel<
		&'a mut VerifierTranscript<StdChallenger>,
		StdChallenger,
		B128,
		StdHashSuite,
	>;

	/// The Merkle channel a verification runs over, with the cleartext residual kept aside.
	///
	/// A ladder that opens anything receives exactly one committed vector, at its last level.
	/// That vector is the residual.
	/// Everything else is delegated untouched, so a recorded run is the ordinary run.
	struct ResidualRecorder<'a> {
		/// The channel every call is forwarded to.
		inner: InnerChannel<'a>,
		/// The committed vectors received so far, in the order the ladder read them.
		received: Vec<Vec<B128>>,
	}

	impl<'a> ResidualRecorder<'a> {
		fn new(transcript: &'a mut VerifierTranscript<StdChallenger>) -> Self {
			Self {
				inner: VerifierMerkleTranscriptChannel::new(transcript),
				received: Vec::new(),
			}
		}

		/// The committed vector a ladder sends in the clear, absent when nothing was opened.
		fn into_residual(self) -> Option<Vec<B128>> {
			let mut received = self.received;
			assert!(received.len() <= 1, "a ladder sends at most one cleartext vector");
			received.pop()
		}
	}

	impl IPVerifierChannel<B128> for ResidualRecorder<'_> {
		type Elem = B128;

		fn recv_one(&mut self) -> Result<B128, binius_ip::channel::Error> {
			self.inner.recv_one()
		}

		fn recv_many(&mut self, n: usize) -> Result<Vec<B128>, binius_ip::channel::Error> {
			self.inner.recv_many(n)
		}

		fn recv_array<const N: usize>(&mut self) -> Result<[B128; N], binius_ip::channel::Error> {
			self.inner.recv_array()
		}

		fn recv_public_claim(&mut self) -> Result<B128, binius_ip::channel::Error> {
			self.inner.recv_public_claim()
		}

		fn sample(&mut self) -> B128 {
			self.inner.sample()
		}

		fn observe_one(&mut self, val: B128) -> B128 {
			self.inner.observe_one(val)
		}

		fn observe_many(&mut self, vals: &[B128]) -> Vec<B128> {
			self.inner.observe_many(vals)
		}

		fn assert_zero(&mut self, val: B128) -> Result<(), binius_ip::channel::Error> {
			self.inner.assert_zero(val)
		}
	}

	impl WordIPVerifierChannel<B128> for ResidualRecorder<'_> {
		type Word = Word;

		fn observe_words(&mut self, words: &[Word]) -> Vec<Word> {
			self.inner.observe_words(words)
		}

		fn subset_sum(&mut self, elems: &[B128], word: &Word) -> B128 {
			self.inner.subset_sum(elems, word)
		}

		fn select(&mut self, elems: &[B128], word: &Word) -> B128 {
			self.inner.select(elems, word)
		}

		fn sample_bits(&mut self, bits: usize) -> Word {
			self.inner.sample_bits(bits)
		}

		fn pack_words(&mut self, words: &[Word]) -> Vec<B128> {
			self.inner.pack_words(words)
		}
	}

	impl MerkleIPVerifierChannel<B128> for ResidualRecorder<'_> {
		type Commitment = <InnerChannel<'static> as MerkleIPVerifierChannel<B128>>::Commitment;

		fn recv_merkle_commitment(
			&mut self,
			leaf_size: usize,
			depth: usize,
		) -> Result<Self::Commitment, binius_iop::merkle_channel::Error> {
			self.inner.recv_merkle_commitment(leaf_size, depth)
		}

		fn recv_openings(
			&mut self,
			commitment: &Self::Commitment,
			indices: &[Word],
		) -> Result<Vec<B128>, binius_iop::merkle_channel::Error> {
			self.inner.recv_openings(commitment, indices)
		}

		fn recv_committed_vector(
			&mut self,
			commitment: &Self::Commitment,
		) -> Result<Vec<B128>, binius_iop::merkle_channel::Error> {
			let values = self.inner.recv_committed_vector(commitment)?;
			self.received.push(values.clone());
			Ok(values)
		}
	}

	impl GrindingVerifierChannel for ResidualRecorder<'_> {
		fn verify_grind(&mut self, bits: usize) -> Result<(), binius_transcript::Error> {
			self.inner.verify_grind(bits)
		}
	}

	/// What one honest run leaves behind, once the verifier has accepted it.
	#[derive(Debug)]
	struct Opening {
		/// The finished proof bytes.
		proof: Vec<u8>,
		/// The residual the last level sent in the clear, absent when nothing was opened.
		residual: Option<Vec<B128>>,
	}

	/// Commits every oracle, proves its relations, and verifies the whole batch in one opening.
	///
	/// `mask_seed` seeds the generator every masked oracle draws its mask from.
	/// It has no effect on a run whose oracles are all committed in the clear.
	///
	/// Nothing is returned unless the verifier accepted.
	/// So a byte count or a residual is only ever read off a transcript that convinced it.
	fn run(params: &LigeritoParams, oracles: &[Oracle], mask_seed: u64) -> Result<Opening, Error> {
		let specs = oracles.iter().map(Oracle::spec).collect::<Vec<_>>();
		let verifier_compiler = LigeritoVerifierCompiler::<B128>::new(specs, params.clone());

		// One transform for the whole ladder, sized by the compiler rather than by hand.
		let domain = GaoMateerOnTheFly::generate(verifier_compiler.max_log_domain_size());
		let prover_compiler = LigeritoProverCompiler::<B128, _>::from_verifier_compiler(
			&verifier_compiler,
			NeighborsLastSingleThread::new(domain),
		);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
				StdRng::seed_from_u64(mask_seed),
				GlobalAllocator,
			);

		// The commit phase: one level-0 codeword per oracle, and nothing else.
		let handles = oracles
			.iter()
			.map(|oracle| prover_channel.send_oracle(oracle.committed.as_view()))
			.collect::<Vec<_>>();
		for (handle, oracle) in std::iter::zip(&handles, oracles) {
			for (transparent, claim) in &oracle.relations {
				prover_channel.prove_oracle_relation(*handle, transparent.clone(), *claim);
			}
			// The ladder folds the message itself, so it needs the buffer and not just its
			// codeword.
			prover_channel.finalize_oracle(*handle, oracle.opened.clone());
		}
		prover_channel.finish();

		let proof = prover_transcript.finalize();

		let mut verifier_transcript =
			VerifierTranscript::new(StdChallenger::default(), proof.clone());
		let mut verifier_channel =
			verifier_compiler.create_channel(ResidualRecorder::new(&mut verifier_transcript));

		let handles = oracles
			.iter()
			.map(|oracle| verifier_channel.recv_oracle(oracle.opened.log_len(), true))
			.collect::<Result<Vec<_>, _>>()?;
		for (handle, oracle) in std::iter::zip(&handles, oracles) {
			for (transparent, claim) in &oracle.relations {
				let transparent = transparent.clone();
				verifier_channel.verify_oracle_relation(
					*handle,
					Box::new(move |point: &[B128]| transparent.evaluate(point)),
					*claim,
				)?;
			}
		}
		let recorder = verifier_channel.finish()?;

		Ok(Opening {
			proof,
			residual: recorder.into_residual(),
		})
	}

	/// Every ladder shape an honest prover can produce must verify through the channel.
	///
	/// The shapes vary the depth, the fold amounts and the column count independently.
	/// One level is the terminal case, where the ladder is a single query round.
	/// A level folding no lanes is the degenerate case.
	/// It recommits at a lower rate without reducing anything.
	#[test]
	fn an_honest_opening_verifies() {
		let shapes: &[(usize, &[usize])] = &[
			(3, &[1]),
			(5, &[2, 1]),
			(6, &[1, 1, 1]),
			(6, &[2, 2, 2]),
			(4, &[2, 0, 1]),
		];
		for &(log_msg_cols, lanes) in shapes {
			// Both grinding profiles run the whole channel, so the compiler carries the ladder's
			// proof of work to both sides and the two stay in step through it.
			for grinding in [Grinding::NONE, Grinding::new(3, 4)] {
				let params = ladder(log_msg_cols, lanes).with_grinding(grinding);
				let oracles = one_oracle(&params, 0);
				run(&params, &oracles, 0)
					.unwrap_or_else(|err| panic!("{log_msg_cols} {lanes:?} {grinding:?}: {err}"));
			}
		}
	}

	/// Every ladder shape must also verify with a mask interleaved at level 0.
	///
	/// A masked oracle commits one lane more than it folds, over the same codeword.
	/// So the query positions, the fold rounds and the residual are all unchanged in shape.
	/// What moves is the leaf width and the one extra element the mask pairing costs.
	#[test]
	fn an_honest_masked_opening_verifies() {
		let shapes: &[(usize, &[usize])] = &[
			(3, &[1]),
			(5, &[2, 1]),
			(6, &[1, 1, 1]),
			(6, &[2, 2, 2]),
			(4, &[2, 0, 1]),
		];
		for &(log_msg_cols, lanes) in shapes {
			// Both grinding profiles run the whole channel, so masking has to leave the two proof
			// of work call sites exactly where the ladder puts them.
			for grinding in [Grinding::NONE, Grinding::new(3, 4)] {
				let params = ladder(log_msg_cols, lanes).with_grinding(grinding);
				let oracles = one_oracle(&params, 0)
					.into_iter()
					.map(Oracle::masked)
					.collect::<Vec<_>>();
				run(&params, &oracles, 0)
					.unwrap_or_else(|err| panic!("{log_msg_cols} {lanes:?} {grinding:?}: {err}"));
			}
		}
	}

	/// A masked opening and a transparent one over the same message both verify, and differ.
	///
	/// The masked run pays for one extra interleaved lane at level 0 and one mask pairing.
	///
	///     transparent: 2^log_lanes elements per opened row
	///     masked     : 2^(log_lanes + 1) elements per opened row, plus one sigma
	///
	/// So a mask is bought with proof size, and nothing deeper than level 0 changes shape.
	#[test]
	fn a_masked_opening_costs_one_lane_and_still_verifies() {
		let params = ladder(5, &[2, 1]);
		let clear = one_oracle(&params, 11);
		let masked = clear
			.iter()
			.cloned()
			.map(Oracle::masked)
			.collect::<Vec<_>>();

		let clear = run(&params, &clear, 0).expect("a transparent opening");
		let masked = run(&params, &masked, 0).expect("a masked opening");

		assert!(
			masked.proof.len() > clear.proof.len(),
			"a masked proof wrote {} bytes against a transparent {}",
			masked.proof.len(),
			clear.proof.len()
		);
		assert_ne!(masked.residual, clear.residual);
	}

	/// A batch may mix masked and transparent oracles, and each gets what it asked for.
	///
	/// Masking is declared per oracle, so one ladder can carry both kinds at once.
	/// Level 0 opens each committed codeword on its own, at the shared positions.
	/// Everything below is one combined message, so a single mask anywhere blinds the residual.
	#[test]
	fn a_batch_may_mix_masked_and_transparent_oracles() {
		let params = ladder(5, &[2, 1]);
		let mut rng = StdRng::seed_from_u64(12);
		let batch = [
			Oracle::honest(&mut rng, params.log_msg_len(), 1).masked(),
			Oracle::honest(&mut rng, 6, 1),
			Oracle::honest(&mut rng, 6, 1).masked(),
		];

		run(&params, &batch, 0).expect("a mixed batch of three oracles");
	}

	/// A masked commitment that does not encode the opened message must still be caught.
	///
	/// The blend is a linear map, so the fold identity level 0 checks survives it unchanged.
	/// A mask must therefore buy hiding without buying the prover any freedom.
	///
	///     relation sumcheck: about `opened`      -> consistent
	///     level 0 codeword : encodes `committed` -> caught
	#[test]
	fn a_masked_commitment_that_does_not_encode_the_opened_message_is_rejected() {
		let params = ladder(5, &[2, 1]);
		let mut rng = StdRng::seed_from_u64(13);
		let mut oracle = Oracle::honest(&mut rng, params.log_msg_len(), 1).masked();

		// Mutation: the oracle commits a different buffer from the one it opens.
		oracle.committed = random_field_buffer::<B128>(&mut rng, params.log_msg_len());

		let err = run(&params, std::slice::from_ref(&oracle), 0)
			.expect_err("the committed codeword encodes a different message");

		// The ladder's own assertion, so the error arrives wrapped rather than raised here.
		match err {
			Error::Ligerito(binius_iop::ligerito::Error::IPChannel(
				binius_ip::channel::Error::InvalidAssert,
			)) => {}
			other => panic!("wrong error variant: {other:?}"),
		}
	}

	/// The residual is a function of the witness only through the mask.
	///
	/// Without a mask it is fixed by the witness: two runs over one oracle write the same bytes.
	/// With a mask it moves with the generator's seed alone, the witness held still.
	/// That is the property the cleartext residual needs.
	/// It is the one value a ladder never hides behind a commitment.
	#[test]
	fn the_residual_moves_with_the_mask_and_not_without_one() {
		let params = ladder(5, &[2, 1]);
		let clear = one_oracle(&params, 14);
		let masked = clear
			.iter()
			.cloned()
			.map(Oracle::masked)
			.collect::<Vec<_>>();

		// Fixture state: one oracle, two mask seeds, the same message throughout.
		let residual = |oracles: &[Oracle], seed| {
			run(&params, oracles, seed)
				.expect("an honest opening")
				.residual
				.expect("an opened ladder sends a residual")
		};

		// A transparent ladder is deterministic, so the seed reaches nothing.
		assert_eq!(residual(&clear, 0), residual(&clear, 1));

		// A masked one is not, and every coordinate of the residual moves.
		let first = residual(&masked, 0);
		let second = residual(&masked, 1);
		assert_eq!(first.len(), second.len());
		for (index, (left, right)) in std::iter::zip(&first, &second).enumerate() {
			assert_ne!(left, right, "residual coordinate {index} did not move");
		}

		// And it is not the transparent residual either, so the mask is really in the value.
		assert_ne!(first, residual(&clear, 0));
	}

	/// Two witnesses satisfying one claim leave residuals no bucket count tells apart.
	///
	/// This is the distribution test the cleartext residual deserves.
	/// The two oracles state the same claim against the same transparent and differ in the witness.
	/// Each is opened under many mask seeds.
	/// The residual coordinates are then bucketed by their leading bits.
	///
	///     buckets  : 8, by the top three bits of each coordinate
	///     samples  : 32 seeds * 2^4 coordinates = 512 per witness
	///     statistic: sum over buckets of (observed - expected)^2 / expected
	///
	/// A residual carrying the witness in the clear would be one fixed vector per witness.
	/// Its 512 samples would collapse onto 16 values and the statistic would explode.
	/// Both collections instead sit inside the bound a uniform draw clears with probability 0.999.
	#[test]
	fn two_witnesses_with_one_claim_leave_residuals_of_one_distribution() {
		const SEEDS: u64 = 32;
		const BUCKETS: usize = 8;
		// The 0.999 quantile of a chi-squared with 7 degrees of freedom.
		const BOUND: f64 = 24.32;

		let params = ladder(5, &[2, 1]);
		let mut rng = StdRng::seed_from_u64(15);
		let first = Oracle::honest(&mut rng, params.log_msg_len(), 1).masked();
		let second = first.sibling(&mut rng);

		// Fixture state: the two witnesses differ, and their claims do not.
		assert_ne!(first.opened.as_ref(), second.opened.as_ref());
		assert_eq!(first.relations[0].1, second.relations[0].1);

		let statistic = |oracle: &Oracle| {
			let mut counts = [0usize; BUCKETS];
			for seed in 0..SEEDS {
				let residual = run(&params, std::slice::from_ref(oracle), seed)
					.expect("an honest masked opening")
					.residual
					.expect("an opened ladder sends a residual");
				for value in residual {
					// The top three bits of a 128-bit value, which are uniform when it is.
					counts[(u128::from(value) >> 125) as usize] += 1;
				}
			}
			let total = counts.iter().sum::<usize>() as f64;
			let expected = total / BUCKETS as f64;
			counts
				.iter()
				.map(|&count| (count as f64 - expected).powi(2) / expected)
				.sum::<f64>()
		};

		for (name, oracle) in [("first", &first), ("second", &second)] {
			let statistic = statistic(oracle);
			assert!(statistic < BOUND, "{name} witness scored {statistic:.2} against {BOUND}");
		}
	}

	/// Several relations on one oracle collapse into a single opening.
	///
	/// Each relation weighs the same message with a different transparent.
	/// The batching coefficient folds them into one claim before the sumcheck ever runs.
	///
	///     relations: <pi, t_0> = s_0, <pi, t_1> = s_1, <pi, t_2> = s_2
	///     batched:   <pi, t_0 + lambda t_1 + lambda^2 t_2> = s_0 + lambda s_1 + lambda^2 s_2
	///
	/// Three of them cost the same ladder as one, so the proof must be the same size.
	#[test]
	fn several_relations_on_one_oracle_share_a_single_opening() {
		let params = ladder(5, &[2, 1]);
		let mut rng = StdRng::seed_from_u64(1);
		let one = [Oracle::honest(&mut rng, params.log_msg_len(), 1)];
		let three = [Oracle::honest(&mut rng, params.log_msg_len(), 3)];

		let one_size = run(&params, &one, 0)
			.expect("one honest relation")
			.proof
			.len();
		let three_size = run(&params, &three, 0)
			.expect("three honest relations")
			.proof
			.len();

		// The extra relations are folded before anything is committed or opened.
		// So they add nothing to the transcript.
		assert_eq!(one_size, three_size);
	}

	/// Several oracles of the same length collapse into a single ladder.
	///
	/// Their level-0 codewords have one length over one domain, so one query set serves all three.
	/// Their folded rows are combined at the batching coefficients into one claim per position.
	/// That is a claim about the single message level 1 commits.
	///
	///     level 0: three Merkle trees, one query set, three openings per query
	///     level 1: one Merkle tree over `sum_i e_i * X_1^(i)`
	///
	/// So a batch pays three level-0 openings but only one of everything below.
	#[test]
	fn several_oracles_of_one_length_share_a_single_ladder() {
		let params = ladder(5, &[2, 1]);
		let mut rng = StdRng::seed_from_u64(2);
		let batch = [
			Oracle::honest(&mut rng, params.log_msg_len(), 1),
			Oracle::honest(&mut rng, params.log_msg_len(), 1),
			Oracle::honest(&mut rng, params.log_msg_len(), 1),
		];

		let batch_size = run(&params, &batch, 0)
			.expect("three honest oracles")
			.proof
			.len();
		let alone = batch
			.iter()
			.map(|oracle| {
				run(&params, std::slice::from_ref(oracle), 0)
					.expect("one honest oracle")
					.proof
					.len()
			})
			.sum::<usize>();

		assert!(
			batch_size < alone,
			"one ladder over three oracles wrote {batch_size} bytes, three ladders {alone}"
		);
	}

	/// Oracles of different lengths share the ladder of the longest.
	///
	/// Level 0's column count is shared, so a shorter message carries fewer interleaved lanes.
	/// One shorter than a single column block carries one lane, zero-padded out to the count.
	///
	/// Fixture state: 2^5 columns and 2^2 lanes at level 0, so the ladder commits 2^7 elements.
	///
	///     2^7 elements -> 2^2 lanes
	///     2^6 elements -> 2^1 lanes
	///     2^5 elements -> 2^0 lanes, exactly one column block
	///     2^3 elements -> 2^0 lanes, zero-padded out to 2^5 columns
	#[test]
	fn oracles_of_different_lengths_share_a_single_ladder() {
		let params = ladder(5, &[2, 1]);
		let mut rng = StdRng::seed_from_u64(3);
		let batch = [7, 6, 5, 3]
			.map(|log_msg_len| Oracle::honest(&mut rng, log_msg_len, 1))
			.into_iter()
			.collect::<Vec<_>>();

		run(&params, &batch, 0).expect("four honest oracles of four lengths");
	}

	/// A claim the message does not satisfy must be caught by the relation sumcheck.
	///
	/// The ladder itself is honest here.
	/// Every commitment encodes the message that was opened.
	/// Every value sent in the clear is internally consistent.
	/// Only one claim, on one oracle of several, is wrong.
	/// So the reduced sumcheck value no longer matches the stated evaluations multiplied.
	#[test]
	fn a_claim_the_message_does_not_satisfy_is_rejected() {
		let params = ladder(5, &[2, 1]);
		let mut rng = StdRng::seed_from_u64(4);
		let mut batch = [
			Oracle::honest(&mut rng, params.log_msg_len(), 1),
			Oracle::honest(&mut rng, 6, 1),
		];

		// Mutation: the second oracle's claim is off by one, and nothing else changes.
		batch[1].relations[0].1 += B128::ONE;

		let err = run(&params, &batch, 0).expect_err("the inner-product claim is off by one");

		match err {
			Error::IPChannel(binius_ip::channel::Error::InvalidAssert) => {}
			other => panic!("wrong error variant: {other:?}"),
		}
	}

	/// A commitment that does not encode the opened message must be caught by the ladder.
	///
	/// The relation is honest about the message the prover reduced.
	/// So the sumcheck passes and every stated evaluation matches.
	/// The rows the query phase opens come from a different codeword, and only the ladder sees it.
	///
	///     relation sumcheck: about `opened`      -> consistent
	///     level 0 codeword : encodes `committed` -> caught
	#[test]
	fn a_commitment_that_does_not_encode_the_opened_message_is_rejected() {
		let params = ladder(5, &[2, 1]);
		let mut rng = StdRng::seed_from_u64(5);
		let mut batch = [
			Oracle::honest(&mut rng, params.log_msg_len(), 1),
			Oracle::honest(&mut rng, 6, 1),
		];

		// Mutation: the second oracle commits a different buffer from the one it opens.
		batch[1].committed = random_field_buffer::<B128>(&mut rng, 6);

		let err = run(&params, &batch, 0)
			.expect_err("the committed codeword encodes a different message");

		// The ladder's own assertion, so the error arrives wrapped rather than raised here.
		match err {
			Error::Ligerito(binius_iop::ligerito::Error::IPChannel(
				binius_ip::channel::Error::InvalidAssert,
			)) => {}
			other => panic!("wrong error variant: {other:?}"),
		}
	}

	/// Commitments carrying no relation open nothing.
	///
	/// A commitment on its own asserts nothing about the message.
	/// So neither side runs a sumcheck, a query round, or a residual.
	/// The transcript then holds the Merkle roots and nothing more.
	#[test]
	fn commitments_with_no_relation_open_nothing() {
		let params = ladder(5, &[2, 1]);
		let mut oracles = one_oracle(&params, 6);
		let opened_size = run(&params, &oracles, 0)
			.expect("one honest relation")
			.proof
			.len();

		oracles[0].relations.clear();
		let empty_size = run(&params, &oracles, 0)
			.expect("a commitment alone always verifies")
			.proof
			.len();

		assert!(
			empty_size < opened_size,
			"an unopened commitment wrote {empty_size} bytes, an opened one {opened_size}"
		);
	}

	/// Opening some oracles but not others is refused rather than silently skipped.
	///
	/// One ladder opens every committed message together, so leaving one out has no meaning.
	/// The prover would have to commit a codeword whose rows nothing ever checks.
	#[test]
	#[should_panic(expected = "every committed oracle must carry at least one relation")]
	fn an_oracle_left_unopened_beside_an_opened_one_is_refused() {
		let params = ladder(5, &[2, 1]);
		let mut rng = StdRng::seed_from_u64(7);
		let mut batch = [
			Oracle::honest(&mut rng, params.log_msg_len(), 1),
			Oracle::honest(&mut rng, 6, 1),
		];

		// Mutation: the second oracle is committed and finalized, but never opened.
		batch[1].relations.clear();
		let _ = run(&params, &batch, 0);
	}

	/// The buffer handed to the commit phase must have the length its spec announced.
	///
	/// A shorter one would encode into a codeword of the wrong shape.
	/// The verifier would then read its Merkle tree at a depth the prover never committed.
	#[test]
	#[should_panic(expected = "oracle buffer log_len mismatch")]
	fn a_message_of_the_wrong_length_is_refused() {
		let params = ladder(5, &[2, 1]);
		let mut oracles = one_oracle(&params, 8);

		// Mutation: the buffer loses a variable while its spec keeps the ladder's length.
		let short =
			random_field_buffer::<B128>(&mut StdRng::seed_from_u64(9), params.log_msg_len() - 1);
		oracles[0].committed = short;
		let _ = run(&params, &oracles, 0);
	}

	/// The committed buffer must be handed back before the opening runs.
	///
	/// The ladder folds the message down to the residual, so the codeword alone is not enough.
	#[test]
	#[should_panic(expected = "the oracle was committed but never finalized")]
	fn an_unfinalized_oracle_cannot_be_opened() {
		let params = ladder(5, &[2, 1]);
		let oracles = one_oracle(&params, 10);
		let (transparent, claim) = oracles[0].relations[0].clone();

		let verifier_compiler =
			LigeritoVerifierCompiler::<B128>::new(vec![oracles[0].spec()], params);
		let domain = GaoMateerOnTheFly::generate(verifier_compiler.max_log_domain_size());
		let prover_compiler = LigeritoProverCompiler::<B128, _>::from_verifier_compiler(
			&verifier_compiler,
			NeighborsLastSingleThread::new(domain),
		);

		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let mut channel = prover_compiler
			.create_channel_without_zk_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut transcript,
				GlobalAllocator,
			);

		// Mutation: the relation is queued, but the message is never handed over.
		let handle = channel.send_oracle(oracles[0].committed.as_view());
		channel.prove_oracle_relation(handle, transparent, claim);
		channel.finish();
	}
}
