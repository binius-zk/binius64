// Copyright 2026 The Binius Developers

//! The prover channel: a queue of relations in, one ladder opening out.

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_iop::{channel::OracleSpec, whir::WHIRParams};
use binius_ip_prover::{
	channel::{IPProverChannel, WordIPProverChannel},
	sumcheck::{
		self, PaddedSumcheckDecorator, batch::BatchSumcheckOutput,
		bivariate_product_evaluator::BivariateProductEvaluator, mle_store::MleStore,
		round_evaluator::SharedSumcheckProver,
	},
};
use binius_math::{
	FieldBuffer, FieldSlice, FieldVec,
	multilinear::eq::{eq_ind_partial_eval_scalars, eq_ind_zero},
	ntt::AdditiveNTT,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use itertools::izip;

use super::{WHIROracle, combined_message::CombinedMessage, relation::QueuedRelation};
use crate::{
	channel::{IOPProverChannel, grinding::GrindingProverChannel},
	fri::{BatchBrakedownOracleProver, BrakedownOracleProver},
	merkle_channel::MerkleIPProverChannel,
	whir::{WHIRProver, opening::commit_level},
};

/// A prover channel that opens every committed oracle with one WHIR ladder.
///
/// The counterpart of the WHIR verifier channel, where the three reductions are described.
///
/// The channel is transparent rather than zero-knowledge.
/// No mask is drawn, so it needs no randomness of its own.
///
/// # Type Parameters
///
/// - `'a`: the lifetime of the parameters and the transform the compiler owns
/// - `F`: the binary field the ladder is committed over
/// - `P`: the packed field the buffers are held in
/// - `NTT`: the additive transform every level encodes over
/// - `Channel`: the Merkle channel carrying all prover interaction
/// - `A`: the allocator the queued transparents and the opening's working buffers are drawn from
pub struct WHIRProverChannel<'a, F, P, NTT, Channel, A>
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
	params: &'a WHIRParams,
	/// The oracles this channel expects, in the order it will commit them.
	oracle_specs: Vec<OracleSpec>,
	/// The committed level-0 codewords, in the order their commitments were sent.
	oracles: Vec<BrakedownOracleProver<P, Channel::Commitment, A::Vec<P>>>,
	/// The committed messages, each handed over when the caller finalizes its oracle.
	/// One entry per committed oracle, absent until that oracle is finalized.
	messages: Vec<Option<FieldVec<P, A>>>,
	/// Relations queued against each oracle, all opened together once the caller finishes.
	/// One entry per committed oracle, so its length is also the number committed so far.
	queue: Vec<Vec<QueuedRelation<P, A>>>,
	/// The allocator the opening's working buffers are drawn from.
	alloc: A,
}

impl<'a, F, P, NTT, Channel, A> WHIRProverChannel<'a, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F, Word = Word>,
	A: Allocator,
{
	/// Creates a channel that opens the given oracles with the given ladder.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` is non-empty.
	/// * No spec is zero-knowledge.
	/// * The longest message is the ladder's message length.
	/// * `ntt`'s domain covers every level's codeword domain.
	pub fn new(
		channel: Channel,
		ntt: &'a NTT,
		oracle_specs: Vec<OracleSpec>,
		params: &'a WHIRParams,
		alloc: A,
	) -> Self {
		assert!(!oracle_specs.is_empty(), "precondition: a WHIR channel opens at least one oracle");
		assert!(
			oracle_specs.iter().all(|spec| !spec.is_zk),
			"precondition: WHIR commits no mask, so a zero-knowledge oracle cannot be opened"
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
			messages: Vec::new(),
			queue: Vec::new(),
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
			messages,
			queue,
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

		let messages = messages
			.into_iter()
			.map(|message| message.expect("the oracle was committed but never finalized"))
			.collect::<Vec<_>>();
		let prover = WHIRProver::new(params, ntt, BatchBrakedownOracleProver::new(oracles));

		Self::prove(&mut channel, &prover, &oracle_specs, &messages, queue, &alloc);
	}

	/// Proves the queued relations against the committed ladder.
	///
	/// The three reductions the verifier runs, in the same order.
	///
	/// ```text
	///     <pi_i, t_ij> = s_ij      the relations, batched per oracle into <pi_i, T_i> = S_i
	///       -> sumcheck            binds every variable at a sampled point r
	///     pi_i(r) = alpha_i        one evaluation claim per oracle
	///       -> eq-combine          at coefficients e_i drawn once every alpha is on the wire
	///     PI(r) = sum_i e_i alpha_i
	///       -> ladder
	/// ```
	///
	/// The sumcheck leaves both multilinears of every oracle evaluated at `r`.
	/// Only the committed ones have to be sent, since the verifier builds the transparents itself.
	fn prove(
		channel: &mut Channel,
		prover: &WHIRProver<'_, P, Channel::Commitment, NTT, A::Vec<P>>,
		oracle_specs: &[OracleSpec],
		messages: &[FieldVec<P, A>],
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

		// One padded sumcheck prover per oracle, in the order the oracles were committed. A
		// message shorter than the longest one rides the same rounds, its claim carried through
		// the extra ones by the equality indicator at zero.
		let provers = izip!(queue, messages, oracle_specs)
			.map(|(relations, message, spec)| {
				let QueuedRelation { transparent, claim } =
					QueuedRelation::batch(relations, lambda);
				let n_vars = spec.log_msg_len;

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
		let coefficients = eq_ind_partial_eval_scalars(&outer_challenges);

		let mut combined = CombinedMessage::zeros_in(alloc, max_n_vars);
		let mut combined_claim = F::ZERO;
		for (message, spec, coefficient, alpha) in
			izip!(messages, oracle_specs, &coefficients, &alphas)
		{
			combined.add_scaled(message.as_view(), *coefficient);
			combined_claim += *coefficient * *alpha * eq_ind_zero(&point[spec.log_msg_len..]);
		}

		let combined = combined.into_buffer();
		prover.prove(combined.as_view(), &point, combined_claim, alloc, channel);
	}
}

impl<F, P, NTT, Channel, A> IPProverChannel<F> for WHIRProverChannel<'_, F, P, NTT, Channel, A>
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

impl<F, P, NTT, Channel, A> WordIPProverChannel<F> for WHIRProverChannel<'_, F, P, NTT, Channel, A>
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
	for WHIRProverChannel<'a, F, P, NTT, Channel, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
	A: Allocator,
{
	type Oracle = WHIROracle;

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

		// Every oracle shares level 0's column count, so only the lane count follows the message
		// length. A message below one column block commits a single zero-padded lane.
		let level = self.params.level_zero_shape(buffer.log_len());
		// Only a message below one column block needs padding, so a longer one is never copied.
		let padded = (buffer.log_len() < level.log_msg_len()).then(|| {
			FieldBuffer::from_view_in(&self.alloc, buffer)
				.zero_extend_in(&self.alloc, level.log_msg_len())
		});
		let message = padded
			.as_ref()
			.map_or_else(|| buffer.as_view(), FieldBuffer::as_view);

		// Encoding and committing level 0 is the whole of the commit phase.
		// The deeper levels only exist once the folds above them have run.
		let index = self.oracles.len();
		self.oracles
			.push(commit_level(&level, self.ntt, message, &self.alloc, &mut self.channel));
		self.messages.push(None);
		self.queue.push(Vec::new());

		WHIROracle { index }
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
		assert!(
			self.messages[oracle.index].replace(buffer).is_none(),
			"the oracle was finalized twice"
		);
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, Ghash128b as B128};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		channel::{Error, IOPVerifierChannel},
		soundness::{Grinding, SoundnessRegime},
		whir::{WHIRLevel, compiler::WHIRVerifierCompiler},
	};
	use binius_math::{
		inner_product::inner_product_buffers,
		multilinear::evaluate::evaluate,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::random_field_buffer,
	};
	use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::whir::compiler::WHIRProverCompiler;

	type StdChallenger = HasherChallenger<StdDigest>;

	/// One oracle a run commits, opens, and states relations about.
	///
	/// Splitting the committed message from the opened one is how a dishonest prover is expressed.
	/// Every value it sends in the clear is consistent with the opened buffer.
	/// The codeword level 0's queries land in encodes the committed one.
	/// An honest oracle holds the same buffer twice.
	struct Oracle {
		/// The buffer level 0's codeword encodes.
		committed: FieldBuffer<B128>,
		/// The buffer every value sent in the clear is consistent with.
		opened: FieldBuffer<B128>,
		/// The transparent multilinears, each with the inner product claimed against it.
		relations: Vec<(FieldBuffer<B128>, B128)>,
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
					let claim = inner_product_buffers(&message, &transparent);
					(transparent, claim)
				})
				.collect();
			Self {
				committed: message.clone(),
				opened: message,
				relations,
			}
		}

		/// The oracle's shape, as the channel is told about it up front.
		fn spec(&self) -> OracleSpec {
			OracleSpec::new(self.opened.log_len())
		}
	}

	/// A ladder whose level `i` commits at inverse rate `2^(i + 1)` and opens `n_queries` rows.
	///
	/// `lanes[i]` is level `i`'s fold amount, and `log_msg_cols` is level 0's column count.
	/// Level `i + 1` takes what level `i` folds to, so its columns are `cols_i - lanes_{i+1}`.
	fn ladder(log_msg_cols: usize, lanes: &[usize]) -> WHIRParams {
		let mut log_msg_cols = log_msg_cols;
		let levels = lanes
			.iter()
			.enumerate()
			.map(|(i, &log_lanes)| {
				// Level 0 keeps the column count it was given; every deeper one loses its lanes.
				if i > 0 {
					log_msg_cols -= log_lanes;
				}
				WHIRLevel {
					log_msg_cols,
					log_lanes,
					log_inv_rate: i + 1,
					n_queries: 5,
				}
			})
			.collect();
		WHIRParams::new(levels, SoundnessRegime::default(), 32)
	}

	/// One honest oracle filling the ladder, carrying one relation.
	fn one_oracle(params: &WHIRParams, seed: u64) -> Vec<Oracle> {
		let mut rng = StdRng::seed_from_u64(seed);
		vec![Oracle::honest(&mut rng, params.log_msg_len(), 1)]
	}

	/// Commits every oracle, proves its relations, and verifies the whole batch in one opening.
	///
	/// Returns the finished proof's length in bytes.
	/// So a byte count is only ever taken from a transcript that convinced the verifier.
	fn run(params: &WHIRParams, oracles: &[Oracle]) -> Result<usize, Error> {
		let specs = oracles.iter().map(Oracle::spec).collect::<Vec<_>>();
		let verifier_compiler = WHIRVerifierCompiler::<B128>::new(specs, params.clone());

		// One transform for the whole ladder, sized by the compiler rather than by hand.
		let domain = GaoMateerOnTheFly::generate(verifier_compiler.max_log_domain_size());
		let prover_compiler = WHIRProverCompiler::<B128, _>::from_verifier_compiler(
			&verifier_compiler,
			NeighborsLastSingleThread::new(domain),
		);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
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
		let proof_size = proof.len();

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let handles = oracles
			.iter()
			.map(|oracle| verifier_channel.recv_oracle(oracle.opened.log_len(), true))
			.collect::<Result<Vec<_>, _>>()?;
		for (handle, oracle) in std::iter::zip(&handles, oracles) {
			for (transparent, claim) in &oracle.relations {
				let transparent = transparent.clone();
				verifier_channel.verify_oracle_relation(
					*handle,
					Box::new(move |point: &[B128]| evaluate(&transparent, point)),
					*claim,
				)?;
			}
		}
		verifier_channel.finish()?;

		Ok(proof_size)
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
				run(&params, &oracles)
					.unwrap_or_else(|err| panic!("{log_msg_cols} {lanes:?} {grinding:?}: {err}"));
			}
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

		let one_size = run(&params, &one).expect("one honest relation");
		let three_size = run(&params, &three).expect("three honest relations");

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

		let batch_size = run(&params, &batch).expect("three honest oracles");
		let alone = batch
			.iter()
			.map(|oracle| run(&params, std::slice::from_ref(oracle)).expect("one honest oracle"))
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

		run(&params, &batch).expect("four honest oracles of four lengths");
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

		let err = run(&params, &batch).expect_err("the inner-product claim is off by one");

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

		let err =
			run(&params, &batch).expect_err("the committed codeword encodes a different message");

		// The ladder's own assertion, so the error arrives wrapped rather than raised here.
		match err {
			Error::WHIR(binius_iop::whir::Error::IPChannel(
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
		let opened_size = run(&params, &oracles).expect("one honest relation");

		oracles[0].relations.clear();
		let empty_size = run(&params, &oracles).expect("a commitment alone always verifies");

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
		let _ = run(&params, &batch);
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
		let _ = run(&params, &oracles);
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

		let verifier_compiler = WHIRVerifierCompiler::<B128>::new(vec![oracles[0].spec()], params);
		let domain = GaoMateerOnTheFly::generate(verifier_compiler.max_log_domain_size());
		let prover_compiler = WHIRProverCompiler::<B128, _>::from_verifier_compiler(
			&verifier_compiler,
			NeighborsLastSingleThread::new(domain),
		);

		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let mut channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut transcript,
				GlobalAllocator,
			);

		// Mutation: the relation is queued, but the message is never handed over.
		let handle = channel.send_oracle(oracles[0].committed.as_view());
		channel.prove_oracle_relation(handle, transparent, claim);
		channel.finish();
	}
}
