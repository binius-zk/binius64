// Copyright 2026 The Binius Developers

//! The verifier channel: a queue of relations in, one ladder opening out.

use binius_core::word::Word;
use binius_field::BinaryField;
use binius_ip::{
	channel::{IPVerifierChannel, WordIPVerifierChannel},
	sumcheck::{self, BatchSumcheckOutput},
};
use binius_math::{
	multilinear::eq::{eq_ind_partial_eval_scalars, eq_ind_zero},
	univariate::evaluate_univariate,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use itertools::izip;

use super::{LigeritoOracle, relation::QueuedRelation};
use crate::{
	channel::{
		Error, IOPVerifierChannel, OracleSpec, TransparentEvalFn, grinding::GrindingVerifierChannel,
	},
	ligerito::{CommittedOracle, LigeritoParams, LigeritoVerifier},
	merkle_channel::MerkleIPVerifierChannel,
};

/// The relation sumcheck sends two coefficients per round.
///
/// Its summand is a product of two multilinears, so a round polynomial has degree 2.
/// That is three coefficients.
/// The constant one is recovered from the running claim, leaving two on the wire.
const RELATION_DEGREE: usize = 2;

/// A verifier channel that opens every committed oracle with one Ligerito ladder.
///
/// The channel is transparent rather than zero-knowledge.
/// No committed message is ever masked.
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
	/// The oracles this channel expects, in the order it will receive them.
	oracle_specs: &'a [OracleSpec],
	/// The ladder the opening runs down.
	params: &'a LigeritoParams,
	/// The commitments to the level-0 codewords, in the order they were received.
	commitments: Vec<Channel::Commitment>,
	/// Relations queued against each oracle, all opened together once the caller finishes.
	/// One entry per received oracle, so its length is also the number received so far.
	queue: Vec<Vec<QueuedRelation<Channel::Elem>>>,
}

impl<'a, F, Channel> LigeritoVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	/// Creates a channel that opens the given oracles with the given ladder.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` is non-empty.
	/// * No spec is zero-knowledge.
	/// * The longest message is the ladder's message length.
	pub fn new(
		channel: Channel,
		oracle_specs: &'a [OracleSpec],
		params: &'a LigeritoParams,
	) -> Self {
		assert!(
			!oracle_specs.is_empty(),
			"precondition: a Ligerito channel opens at least one oracle"
		);
		assert!(
			oracle_specs.iter().all(|spec| !spec.is_zk),
			"precondition: Ligerito commits no mask, so a zero-knowledge oracle cannot be opened"
		);
		// Every oracle shares level 0's column count, so the ladder must fit the longest message.
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
			oracle_specs,
			params,
			commitments: Vec::new(),
			queue: Vec::new(),
		}
	}

	/// Consumes the channel and verifies every queued relation in one opening.
	///
	/// The relations on each oracle are folded into one, then reduced to one evaluation claim.
	/// Those claims are then combined into the single claim the ladder opens.
	/// Nothing happens when no relation was queued, since commitments alone assert nothing.
	///
	/// Returns the Merkle channel, so a caller can still reach what it accumulated.
	///
	/// The channel has to be able to check a proof of work, because the ladder may pay one.
	/// A configuration that grinds nothing still asks for the capability, and never uses it.
	///
	/// ## Preconditions
	///
	/// * Either every oracle carries a relation, or none of them does.
	pub fn finish(self) -> Result<Channel, Error>
	where
		Channel: GrindingVerifierChannel,
	{
		let Self {
			mut channel,
			oracle_specs,
			params,
			commitments,
			queue,
		} = self;

		// Every oracle is received at most once, so what remains is whatever was not.
		let n_remaining = oracle_specs.len() - commitments.len();
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining");

		if queue.iter().all(Vec::is_empty) {
			return Ok(channel);
		}
		assert!(
			queue.iter().all(|relations| !relations.is_empty()),
			"precondition: every committed oracle must carry at least one relation"
		);

		Self::verify(&mut channel, oracle_specs, params, commitments, queue)?;

		Ok(channel)
	}

	/// Verifies the queued relations against the committed ladder.
	///
	/// Three reductions run back to back.
	/// Each oracle's relations fold into one at a shared coefficient.
	/// A batched sumcheck turns those into one evaluation claim per oracle, at a shared point.
	/// Those claims are combined into a claim about the one message the ladder folds.
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
	/// A message shorter than the longest one is that multilinear zero-extended.
	/// Its claim then carries the factor the extra coordinates contribute at zero.
	///
	/// # Soundness
	///
	/// Every coefficient is drawn only after the claims it combines are bound to the transcript.
	/// The three batching steps then cost, in order:
	///
	/// - `sum_i (k_i - 1) / |F|`, for folding oracle `i`'s `k_i` relations by powers of one
	///   coefficient rather than by `k_i` independent ones;
	/// - `(k - 1) / |F|`, for folding the `k` oracle claims into one sumcheck the same way;
	/// - a wider row union at level 0, for combining the `k` messages into the one the ladder
	///   folds.
	///
	/// The third is a proximity-gap statement rather than a linear-independence one.
	/// So no number of queries buys it back.
	/// Level 0 folds a tensor over its own lane index *and* the message index.
	/// Its row union therefore grows from `2^log_lanes` rows to `2^log_lanes * k` of them.
	/// The ladder prices exactly that.
	/// Nothing below level 0 is affected, since the batch is one combined message from level 1 on.
	///
	/// Two limits on where that third charge comes from, since neither is obvious from the code.
	///
	/// The coefficients `e_i` are a tensor rather than a power curve.
	/// So the step they batch is covered by [DP24], and only in the unique-decoding regime.
	/// Read a Johnson-regime figure as a bound on the fold, never on the batch.
	///
	/// [NA25] section 6.5 sketches this very construction, level 0 per proof and combined after.
	/// It states no error bound for it, and suspects only that communication is reduced.
	/// So the row-union charge is this repository's conservative reading, not a citation.
	///
	/// [DP24]: <https://eprint.iacr.org/2024/504>
	/// [NA25]: <https://eprint.iacr.org/2025/1187>
	fn verify(
		channel: &mut Channel,
		oracle_specs: &[OracleSpec],
		params: &LigeritoParams,
		commitments: Vec<Channel::Commitment>,
		queue: Vec<Vec<QueuedRelation<Channel::Elem>>>,
	) -> Result<(), Error>
	where
		Channel: GrindingVerifierChannel,
	{
		let n_oracles = commitments.len();
		let max_n_vars = params.log_msg_len();

		// Every claim in the queue is already bound to the transcript, so a coefficient drawn
		// here cannot be anticipated by any of them.
		let lambda = channel.sample();
		let relations = queue
			.into_iter()
			.map(|relations| QueuedRelation::batch(relations, lambda.clone()))
			.collect::<Vec<_>>();
		let claims = relations
			.iter()
			.map(|relation| relation.claim.clone())
			.collect::<Vec<_>>();

		// Bind every variable of every committed multilinear at a point the prover cannot predict.
		// A shorter message rides the same rounds, zero-extended over the coordinates it lacks.
		let BatchSumcheckOutput {
			batch_coeff,
			eval,
			challenges,
		} = sumcheck::batch_verify::<F, _>(max_n_vars, RELATION_DEGREE, &claims, channel)?;

		// The evaluation of each committed multilinear at that point, which the prover has to
		// state, in the order the oracles were received.
		let alphas = channel.recv_many(n_oracles)?;

		// Sumcheck rounds bind the highest variable first, so reversing gives variable order.
		let mut point = challenges;
		point.reverse();

		// The reduced value is the batched product of each message with its transparent, and only
		// the messages are committed.
		let contributions = izip!(relations, oracle_specs, &alphas)
			.map(|(relation, spec, alpha)| {
				let (eval_point, padding) = point.split_at(spec.log_msg_len);
				alpha.clone() * (relation.transparent)(eval_point) * eq_ind_zero(padding)
			})
			.collect::<Vec<_>>();
		channel.assert_zero(eval - evaluate_univariate(&contributions, &batch_coeff))?;

		// Every stated evaluation is now bound to the transcript, so the coefficients that combine
		// them into one message cannot be anticipated either.
		let outer_challenges = channel.sample_many(log2_ceil_usize(n_oracles));
		let coefficients = eq_ind_partial_eval_scalars(&outer_challenges);

		// The combined message is the zero-extended messages summed at those coefficients, so its
		// evaluation is the same combination of the stated ones.
		let combined_claim = izip!(oracle_specs, &coefficients, &alphas)
			.map(|(spec, coefficient, alpha)| {
				coefficient.clone() * alpha.clone() * eq_ind_zero(&point[spec.log_msg_len..])
			})
			.sum::<Channel::Elem>();

		let oracles = izip!(commitments, oracle_specs, coefficients)
			.map(|(commitment, spec, coefficient)| {
				let log_lanes = params.level_zero_shape(spec.log_msg_len).log_lanes;
				CommittedOracle::new(commitment, log_lanes, coefficient)
			})
			.collect();

		LigeritoVerifier::batched(params, oracles).verify(&point, combined_claim, channel)?;

		Ok(())
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
		&self.oracle_specs[self.commitments.len()..]
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

		// Level 0 holds one codeword position per leaf, across every interleaved lane. Every
		// oracle shares the column count, so only the lane count follows the message length.
		let index = self.commitments.len();
		let level = self
			.params
			.level_zero_shape(self.oracle_specs[index].log_msg_len);
		let commitment = self
			.channel
			.recv_merkle_commitment(1 << level.log_lanes, level.log_codeword_len())?;

		self.commitments.push(commitment);
		self.queue.push(Vec::new());

		Ok(LigeritoOracle { index })
	}

	fn verify_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: TransparentEvalFn<Self::Elem>,
		claim: Self::Elem,
	) -> Result<(), Error> {
		// A handle can only exist once its commitment arrived, so the slot is already there.
		let n_received = self.queue.len();
		self.queue
			.get_mut(oracle.index)
			.unwrap_or_else(|| {
				panic!("oracle index {} out of bounds, expected < {n_received}", oracle.index)
			})
			.push(QueuedRelation { transparent, claim });
		Ok(())
	}
}
