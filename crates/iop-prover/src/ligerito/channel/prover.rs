// Copyright 2026 The Binius Developers

//! The prover channel: a queue of relations in, one ladder opening out.

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_iop::{channel::OracleSpec, ligerito::LigeritoParams};
use binius_ip_prover::{
	channel::{IPProverChannel, WordIPProverChannel},
	sumcheck::{ProveSingleOutput, bivariate_product_prover, prove_single},
};
use binius_math::{FieldBuffer, FieldSlice, FieldVec, ntt::AdditiveNTT};

use super::{LigeritoOracle, relation::QueuedRelation};
use crate::{
	channel::{IOPProverChannel, grinding::GrindingProverChannel},
	ligerito::LigeritoProver,
	merkle_channel::MerkleIPProverChannel,
};

/// A prover channel that opens its one committed oracle with a Ligerito ladder.
///
/// The counterpart of the Ligerito verifier channel, where the two reductions are described.
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
	/// The one oracle this channel expects, held as a vector so the trait can hand it back.
	oracle_specs: Vec<OracleSpec>,
	/// Level 0's committed codeword and the handle that opens it, absent until the oracle is sent.
	prover: Option<LigeritoProver<'a, P, Channel::Commitment, NTT>>,
	/// The committed message, handed over when the caller finalizes the oracle.
	message: Option<FieldVec<P, A>>,
	/// Relations queued against the oracle, all opened together once the caller finishes.
	queue: Vec<QueuedRelation<P, A>>,
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
	/// Creates a channel that opens one oracle with the given ladder.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` holds exactly one spec.
	/// * That spec is not zero-knowledge.
	/// * Its message length is the ladder's message length.
	/// * `ntt`'s domain covers every level's codeword domain.
	pub fn new(
		channel: Channel,
		ntt: &'a NTT,
		oracle_specs: Vec<OracleSpec>,
		params: &'a LigeritoParams,
		alloc: A,
	) -> Self {
		assert_eq!(
			oracle_specs.len(),
			1,
			"precondition: a Ligerito channel opens exactly one oracle, got {}",
			oracle_specs.len()
		);
		assert!(
			!oracle_specs[0].is_zk,
			"precondition: Ligerito commits no mask, so a zero-knowledge oracle cannot be opened"
		);
		assert_eq!(
			oracle_specs[0].log_msg_len,
			params.log_msg_len(),
			"precondition: the oracle's message length must be the ladder's message length"
		);

		Self {
			channel,
			ntt,
			params,
			oracle_specs,
			prover: None,
			message: None,
			queue: Vec::new(),
			alloc,
		}
	}

	/// Consumes the channel and proves every queued relation in one opening.
	///
	/// Mirrors the verifier's own finishing step, message for message.
	/// Nothing happens when no relation was queued, since the commitment alone asserts nothing.
	///
	/// The channel has to be able to pay a proof of work, because the ladder may ask for one.
	/// A configuration that grinds nothing still asks for the capability, and never uses it.
	pub fn finish(self)
	where
		Channel: GrindingProverChannel,
	{
		let Self {
			mut channel,
			ntt: _,
			params: _,
			oracle_specs,
			prover,
			message,
			queue,
			alloc,
		} = self;

		// The oracle is committed at most once, so what remains is one spec or none.
		let n_remaining = oracle_specs.len() - usize::from(prover.is_some());
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining");

		if queue.is_empty() {
			return;
		}

		let prover = prover.expect("a relation can only be queued against a committed oracle");
		let message = message.expect("the oracle was committed but never finalized");
		Self::prove(&mut channel, &prover, message.as_view(), queue, &alloc);
	}

	/// Proves the queued relations against the committed ladder.
	///
	/// The two reductions the verifier runs, in the same order.
	///
	/// ```text
	///     <pi, t> = s          the relations, batched into one
	///       -> sumcheck        binds every variable at a sampled point r
	///     pi(r) = alpha        the evaluation claim the ladder takes
	///       -> ladder
	/// ```
	///
	/// The sumcheck leaves both multilinears evaluated at `r`.
	/// Only the committed one has to be sent, since the verifier builds the transparent itself.
	fn prove(
		channel: &mut Channel,
		prover: &LigeritoProver<'_, P, Channel::Commitment, NTT>,
		message: FieldSlice<'_, P>,
		queue: Vec<QueuedRelation<P, A>>,
		alloc: &A,
	) where
		Channel: GrindingProverChannel,
	{
		// Every claim in the queue is already bound to the transcript, so a coefficient drawn
		// here cannot be anticipated by any of them.
		let lambda = channel.sample();
		let QueuedRelation { transparent, claim } = QueuedRelation::batch(queue, lambda);

		// The sumcheck consumes both of its multilinears, and the ladder still needs the message.
		// So it runs over a copy.
		let sumcheck_prover = bivariate_product_prover(
			alloc,
			[
				FieldBuffer::from_view_in(alloc, message.as_view()),
				transparent,
			],
			claim,
		);
		let ProveSingleOutput {
			multilinear_evals,
			mut challenges,
		} = prove_single(sumcheck_prover, channel);

		// The store holds the message first, so its evaluation is the one the ladder opens.
		let alpha = multilinear_evals[0];
		channel.send_one(alpha);

		// Sumcheck rounds bind the highest variable first, so reversing gives variable order.
		challenges.reverse();

		prover.prove(message, &challenges, alpha, alloc, channel);
	}
}

/// Proves the queued relations against the committed ladder.
///
/// The two reductions the verifier runs, in the same order.
///
/// ```text
///     <pi, t> = s          the relations, batched into one
///       -> sumcheck        binds every variable at a sampled point r
///     pi(r) = alpha        the evaluation claim the ladder takes
///       -> ladder
/// ```
///
/// The sumcheck leaves both multilinears evaluated at `r`.
/// Only the committed one has to be sent, since the verifier builds the transparent itself.
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
		&self.oracle_specs[usize::from(self.prover.is_some())..]
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

		// Encoding and committing level 0 is the whole of the commit phase.
		// The deeper levels only exist once the folds above them have run.
		self.prover =
			Some(LigeritoProver::commit(self.params, self.ntt, buffer, &mut self.channel));

		LigeritoOracle(())
	}

	fn prove_oracle_relation(
		&mut self,
		_oracle: Self::Oracle,
		transparent: FieldVec<P, A>,
		claim: P::Scalar,
	) {
		// A handle can only exist once the oracle was committed, and there is one oracle.
		// So the handle names no index to look up.
		self.queue.push(QueuedRelation { transparent, claim });
	}

	fn finalize_oracle(&mut self, _oracle: Self::Oracle, buffer: FieldVec<P, A>) {
		// The ladder folds the message down to the residual.
		// So it needs the buffer itself and not just the codeword it was encoded into.
		assert!(self.message.replace(buffer).is_none(), "the oracle was finalized twice");
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, Ghash128b as B128};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		channel::{Error, IOPVerifierChannel},
		ligerito::{LigeritoLevel, compiler::LigeritoVerifierCompiler},
		soundness::{Grinding, SoundnessRegime},
	};
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

	/// A random multilinear of the ladder's message shape.
	fn message(params: &LigeritoParams, seed: u64) -> FieldBuffer<B128> {
		random_field_buffer(&mut StdRng::seed_from_u64(seed), params.log_msg_len())
	}

	/// A random dense transparent over the message, paired with the claim it truthfully makes.
	///
	/// A dense random weight rather than an equality indicator.
	/// So the relation is a general inner product, not the evaluation claim the ladder takes.
	fn relation(rng: &mut StdRng, message: &FieldBuffer<B128>) -> (FieldBuffer<B128>, B128) {
		let transparent = random_field_buffer::<B128>(rng, message.log_len());
		let claim = message.inner_product(&transparent);
		(transparent, claim)
	}

	/// Commits `committed`, proves `relations` about `opened`, and verifies the result.
	///
	/// Splitting the committed message from the opened one is how a dishonest prover is expressed.
	/// Every value it sends in the clear is consistent with the opened buffer.
	/// The codeword level 0's queries land in encodes the committed one.
	/// An honest run passes the same buffer twice.
	///
	/// Returns the finished proof's length in bytes.
	/// So a byte count is only ever taken from a transcript that convinced the verifier.
	fn run(
		params: &LigeritoParams,
		committed: &FieldBuffer<B128>,
		opened: &FieldBuffer<B128>,
		relations: &[(FieldBuffer<B128>, B128)],
	) -> Result<usize, Error> {
		let n_vars = params.log_msg_len();
		let verifier_compiler =
			LigeritoVerifierCompiler::<B128>::new(vec![OracleSpec::new(n_vars)], params.clone());

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
				GlobalAllocator,
			);

		// The commit phase: level 0's codeword, and nothing else.
		let oracle = prover_channel.send_oracle(committed.as_view());
		for (transparent, claim) in relations {
			prover_channel.prove_oracle_relation(oracle, transparent.clone(), *claim);
		}
		// The ladder folds the message itself, so it needs the buffer and not just its codeword.
		prover_channel.finalize_oracle(oracle, opened.clone());
		prover_channel.finish();

		let proof = prover_transcript.finalize();
		let proof_size = proof.len();

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);

		let oracle = verifier_channel.recv_oracle(n_vars, true)?;
		for (transparent, claim) in relations {
			let transparent = transparent.clone();
			verifier_channel.verify_oracle_relation(
				oracle,
				Box::new(move |point: &[B128]| transparent.evaluate(point)),
				*claim,
			)?;
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
				let msg = message(&params, 0);
				let mut rng = StdRng::seed_from_u64(1);
				let relations = [relation(&mut rng, &msg)];
				run(&params, &msg, &msg, &relations)
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
		let msg = message(&params, 0);
		let mut rng = StdRng::seed_from_u64(1);

		let one = [relation(&mut rng, &msg)];
		let three = [
			relation(&mut rng, &msg),
			relation(&mut rng, &msg),
			relation(&mut rng, &msg),
		];

		let one_size = run(&params, &msg, &msg, &one).expect("one honest relation");
		let three_size = run(&params, &msg, &msg, &three).expect("three honest relations");

		// The extra relations are folded before anything is committed or opened.
		// So they add nothing to the transcript.
		assert_eq!(one_size, three_size);
	}

	/// A claim the message does not satisfy must be caught by the relation sumcheck.
	///
	/// The ladder itself is honest here.
	/// The commitment encodes the message that was opened.
	/// Every value sent in the clear is internally consistent.
	/// Only the claim is wrong.
	/// So the reduced sumcheck value no longer matches the two stated evaluations multiplied.
	#[test]
	fn a_claim_the_message_does_not_satisfy_is_rejected() {
		let params = ladder(5, &[2, 1]);
		let msg = message(&params, 0);
		let mut rng = StdRng::seed_from_u64(1);
		let (transparent, claim) = relation(&mut rng, &msg);

		// Mutation: the claim is off by one, and nothing else changes.
		let relations = [(transparent, claim + B128::ONE)];
		let err = run(&params, &msg, &msg, &relations)
			.expect_err("the inner-product claim is off by one");

		match err {
			Error::IPChannel(binius_ip::channel::Error::InvalidAssert) => {}
			other => panic!("wrong error variant: {other:?}"),
		}
	}

	/// A commitment that does not encode the opened message must be caught by the ladder.
	///
	/// The relation is honest about the message the prover reduced.
	/// So the sumcheck passes and the stated evaluation matches.
	/// The rows the query phase opens come from a different codeword, and only the ladder sees it.
	///
	///     relation sumcheck: about `opened`      -> consistent
	///     level 0 codeword : encodes `committed` -> caught
	#[test]
	fn a_commitment_that_does_not_encode_the_opened_message_is_rejected() {
		let params = ladder(5, &[2, 1]);
		let committed = message(&params, 0);
		let opened = message(&params, 1);
		let mut rng = StdRng::seed_from_u64(1);
		let relations = [relation(&mut rng, &opened)];

		let err = run(&params, &committed, &opened, &relations)
			.expect_err("the committed codeword encodes a different message");

		// The ladder's own assertion, so the error arrives wrapped rather than raised here.
		match err {
			Error::Ligerito(binius_iop::ligerito::Error::IPChannel(
				binius_ip::channel::Error::InvalidAssert,
			)) => {}
			other => panic!("wrong error variant: {other:?}"),
		}
	}

	/// A commitment carrying no relation opens nothing.
	///
	/// The commitment on its own asserts nothing about the message.
	/// So neither side runs a sumcheck, a query round, or a residual.
	/// The transcript then holds the Merkle root and nothing more.
	#[test]
	fn a_commitment_with_no_relation_opens_nothing() {
		let params = ladder(5, &[2, 1]);
		let msg = message(&params, 0);
		let mut rng = StdRng::seed_from_u64(1);
		let relations = [relation(&mut rng, &msg)];

		let empty_size = run(&params, &msg, &msg, &[]).expect("a commitment alone always verifies");
		let opened_size = run(&params, &msg, &msg, &relations).expect("one honest relation");

		assert!(
			empty_size < opened_size,
			"an unopened commitment wrote {empty_size} bytes, an opened one {opened_size}"
		);
	}

	/// The buffer handed to the commit phase must have the ladder's message length.
	///
	/// A shorter one would encode into a codeword of the wrong shape.
	/// The verifier would then read its Merkle tree at a depth the prover never committed.
	#[test]
	#[should_panic(expected = "oracle buffer log_len mismatch")]
	fn a_message_of_the_wrong_length_is_refused() {
		let params = ladder(5, &[2, 1]);
		let short =
			random_field_buffer::<B128>(&mut StdRng::seed_from_u64(0), params.log_msg_len() - 1);
		run(&params, &short, &short, &[]).expect("the commit phase panics before this");
	}

	/// The committed buffer must be handed back before the opening runs.
	///
	/// The ladder folds the message down to the residual, so the codeword alone is not enough.
	#[test]
	#[should_panic(expected = "the oracle was committed but never finalized")]
	fn an_unfinalized_oracle_cannot_be_opened() {
		let params = ladder(5, &[2, 1]);
		let msg = message(&params, 0);
		let mut rng = StdRng::seed_from_u64(1);
		let (transparent, claim) = relation(&mut rng, &msg);

		let verifier_compiler = LigeritoVerifierCompiler::<B128>::new(
			vec![OracleSpec::new(params.log_msg_len())],
			params,
		);
		let domain = GaoMateerOnTheFly::generate(verifier_compiler.max_log_domain_size());
		let prover_compiler = LigeritoProverCompiler::<B128, _>::from_verifier_compiler(
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
		let oracle = channel.send_oracle(msg.as_view());
		channel.prove_oracle_relation(oracle, transparent, claim);
		channel.finish();
	}
}
