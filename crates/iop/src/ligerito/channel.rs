// Copyright 2026 The Binius Developers

//! Ligerito implementation of the IOP verifier channel.

use binius_core::word::Word;
use binius_field::BinaryField;
use binius_ip::{
	channel::{IPVerifierChannel, WordIPVerifierChannel},
	sumcheck::{self, SumcheckOutput},
};
use binius_math::univariate::evaluate_univariate;

use super::{LigeritoParams, LigeritoVerifier};
use crate::{
	channel::{Error, IOPVerifierChannel, OracleSpec, TransparentEvalFn},
	merkle_channel::MerkleIPVerifierChannel,
};

/// The relation sumcheck sends two coefficients per round.
///
/// Its summand is a product of two multilinears, so a round polynomial has degree 2.
/// That is three coefficients.
/// The constant one is recovered from the running claim, leaving two on the wire.
const RELATION_DEGREE: usize = 2;

/// A handle to the oracle a Ligerito channel opens.
///
/// The inner field is private, so the only way to hold one is to have received the commitment.
/// A ladder opens exactly one committed message, so the handle carries nothing else.
#[derive(Debug, Clone, Copy)]
pub struct LigeritoOracle(());

/// A committed-oracle relation queued until the opening runs.
struct QueuedRelation<Elem> {
	/// Evaluates the transparent multilinear at the point the relation sumcheck reduces to.
	transparent: TransparentEvalFn<Elem>,
	/// The claimed inner product of the committed multilinear with the transparent one.
	claim: Elem,
}

/// A verifier channel that opens its one committed oracle with a Ligerito ladder.
///
/// The channel is transparent rather than zero-knowledge.
/// The committed message is never masked.
/// The ladder's residual reaches the verifier in the clear.
///
/// # Type Parameters
///
/// - `'a`: the lifetime of the parameters the compiler owns
/// - `F`: the binary field the ladder is committed over
/// - `Channel`: the Merkle channel carrying all prover interaction
pub struct LigeritoVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	/// The Merkle channel carrying all prover interaction.
	/// Field elements, challenges, commitments and openings all pass through it.
	channel: Channel,
	/// The one oracle this channel expects, held as a slice so the trait can hand it back.
	oracle_specs: &'a [OracleSpec],
	/// The ladder the opening runs down.
	params: &'a LigeritoParams,
	/// The commitment to level 0's interleaved codeword, absent until the oracle is received.
	commitment: Option<Channel::Commitment>,
	/// Relations queued against the oracle, all opened together once the caller finishes.
	queue: Vec<QueuedRelation<Channel::Elem>>,
}

impl<'a, F, Channel> LigeritoVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	/// Creates a channel that opens one oracle with the given ladder.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` holds exactly one spec.
	/// * That spec is not zero-knowledge.
	/// * Its message length is the ladder's message length.
	pub fn new(
		channel: Channel,
		oracle_specs: &'a [OracleSpec],
		params: &'a LigeritoParams,
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
			oracle_specs,
			params,
			commitment: None,
			queue: Vec::new(),
		}
	}

	/// Consumes the channel and verifies every queued relation in one opening.
	///
	/// The relations are folded into one, then reduced to a single evaluation claim.
	/// That claim is what the ladder opens.
	/// Nothing happens when no relation was queued, since a commitment alone asserts nothing.
	///
	/// Returns the Merkle channel, so a caller can still reach what it accumulated.
	pub fn finish(self) -> Result<Channel, Error> {
		let Self {
			mut channel,
			oracle_specs,
			params,
			commitment,
			queue,
		} = self;

		// The oracle is received at most once, so what remains is one spec or none.
		let n_remaining = oracle_specs.len() - usize::from(commitment.is_some());
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining");

		if queue.is_empty() {
			return Ok(channel);
		}

		let commitment = commitment.expect("a relation can only be queued against a commitment");
		verify_ligerito(&mut channel, params, commitment, queue)?;

		Ok(channel)
	}
}

/// Verifies the queued relations against the committed ladder.
///
/// Two reductions run back to back.
/// A sumcheck turns the relation into an evaluation claim at a point neither party chose.
/// That is the only shape the ladder opens.
///
/// ```text
///     <pi, t> = s          the relations, batched into one
///       -> sumcheck        binds every variable at a sampled point r
///     pi(r) = alpha        the evaluation claim the ladder takes
///       -> ladder
/// ```
///
/// The sumcheck's reduced value is the product of both multilinears at `r`.
/// The prover states `pi(r)` and the verifier computes `t(r)` itself.
/// So the product pins the stated evaluation.
fn verify_ligerito<F, Channel>(
	channel: &mut Channel,
	params: &LigeritoParams,
	commitment: Channel::Commitment,
	relations: Vec<QueuedRelation<Channel::Elem>>,
) -> Result<(), Error>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	let QueuedRelation { transparent, claim } = batch_relations(channel, relations);

	// Bind every variable of the committed multilinear at a point the prover cannot predict.
	let SumcheckOutput { eval, challenges } =
		sumcheck::verify::<F, _>(params.log_msg_len(), RELATION_DEGREE, claim, channel)?;

	// The evaluation of the committed multilinear at that point, which the prover has to state.
	let alpha = channel.recv_one()?;

	// Sumcheck rounds bind the highest variable first, so reversing gives variable order.
	let mut point = challenges;
	point.reverse();

	// The reduced value is the product of the two multilinears, and only one factor is committed.
	channel.assert_zero(eval - alpha.clone() * transparent(&point))?;

	LigeritoVerifier::new(params, commitment).verify(&point, alpha, channel)?;

	Ok(())
}

/// Folds every queued relation into one, against one transparent and one claim.
///
/// Relations `j = 0, 1, ...` are combined with the powers of a sampled coefficient `lambda`:
///
/// ```text
///     T = sum_j lambda^j * t_j     the combined transparent
///     S = sum_j lambda^j * s_j     the combined claim
/// ```
///
/// An inner product is linear in the transparent.
/// So `<pi, T> = S` holds exactly when every `<pi, t_j> = s_j` does.
/// The exception has probability at most `(k - 1) / |F|` over `lambda`.
/// The coefficient is drawn once every claim it combines is already bound to the transcript.
/// So no claim can be chosen as a function of it.
///
/// Mirrors the prover-side batching in `binius_iop_prover::ligerito::channel`.
fn batch_relations<F, Channel>(
	channel: &mut Channel,
	mut relations: Vec<QueuedRelation<Channel::Elem>>,
) -> QueuedRelation<Channel::Elem>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	let lambda = channel.sample();

	// A single relation folds nothing.
	// Evaluating one transparent directly is cheaper than a closure that wraps it.
	if relations.len() <= 1 {
		return relations
			.pop()
			.expect("precondition: the queue is non-empty");
	}

	// Split the queue so the combined claim can be a univariate evaluation at `lambda`.
	let (transparents, claims): (Vec<_>, Vec<_>) = relations
		.into_iter()
		.map(|relation| (relation.transparent, relation.claim))
		.unzip();
	let claim = evaluate_univariate(&claims, &lambda);

	QueuedRelation {
		transparent: Box::new(move |point: &[Channel::Elem]| {
			// The combined transparent is only ever read at one point.
			// So it stays a closure rather than a materialized multilinear.
			let evals = transparents
				.iter()
				.map(|transparent| transparent(point))
				.collect::<Vec<_>>();
			evaluate_univariate(&evals, &lambda)
		}),
		claim,
	}
}

impl<F, Channel> IPVerifierChannel<F> for LigeritoVerifierChannel<'_, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	type Elem = Channel::Elem;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		self.channel.recv_one()
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<Self::Elem>, binius_ip::channel::Error> {
		self.channel.recv_many(n)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[Self::Elem; N], binius_ip::channel::Error> {
		self.channel.recv_array()
	}

	fn recv_public_claim(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		self.channel.recv_public_claim()
	}

	fn sample(&mut self) -> Self::Elem {
		self.channel.sample()
	}

	fn observe_one(&mut self, val: F) -> Self::Elem {
		self.channel.observe_one(val)
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<Self::Elem> {
		self.channel.observe_many(vals)
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		self.channel.assert_zero(val)
	}
}

impl<F, Channel> WordIPVerifierChannel<F> for LigeritoVerifierChannel<'_, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	type Word = Channel::Word;

	fn observe_words(&mut self, words: &[Word]) -> Vec<Self::Word> {
		self.channel.observe_words(words)
	}

	fn subset_sum(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem {
		self.channel.subset_sum(elems, word)
	}

	fn select(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem {
		self.channel.select(elems, word)
	}

	fn sample_bits(&mut self, bits: usize) -> Self::Word {
		self.channel.sample_bits(bits)
	}

	fn pack_words(&mut self, words: &[Self::Word]) -> Vec<Self::Elem> {
		self.channel.pack_words(words)
	}
}

impl<F, Channel> IOPVerifierChannel<F> for LigeritoVerifierChannel<'_, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	type Oracle = LigeritoOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[usize::from(self.commitment.is_some())..]
	}

	fn recv_oracle(
		&mut self,
		_log_msg_len: usize,
		_is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error> {
		// The shape was pinned against the ladder when the channel was built.
		// So the arguments carry nothing this call still needs.
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining oracle specs"
		);

		// Level 0 holds one codeword position per leaf, across every interleaved lane.
		let level = &self.params.levels()[0];
		let commitment = self
			.channel
			.recv_merkle_commitment(1 << level.log_lanes, level.log_codeword_len())?;

		self.commitment = Some(commitment);

		Ok(LigeritoOracle(()))
	}

	fn verify_oracle_relation(
		&mut self,
		_oracle: Self::Oracle,
		transparent: TransparentEvalFn<Self::Elem>,
		claim: Self::Elem,
	) -> Result<(), Error> {
		// A handle can only exist once the commitment arrived, and there is one oracle.
		// So the handle names no index to look up.
		self.queue.push(QueuedRelation { transparent, claim });
		Ok(())
	}
}
