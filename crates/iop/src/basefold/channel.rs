// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP verifier channel.

use std::rc::Rc;

use binius_core::word::Word;
use binius_field::{BinaryField, FieldOps, util::powers};
use binius_ip::{
	channel::{IPVerifierChannel, WordIPVerifierChannel},
	sumcheck::{self, SumcheckOutput},
};
use binius_math::{
	line::extrapolate_line,
	multilinear::eq::{eq_ind, eq_ind_partial_eval_scalars, eq_ind_zero},
	univariate::evaluate_univariate,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use itertools::izip;

use crate::{
	basefold::{self, layout::RoundLayout},
	channel::{Error, IOPVerifierChannel, OracleSchedule, OracleSpec, TransparentEvalFn},
	fri::FRIParams,
	merkle_channel::MerkleIPVerifierChannel,
};

/// Oracle handle returned by [`BaseFoldVerifierChannel::recv_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldOracle {
	/// The round whose oracle this one is committed in.
	round: usize,
	/// The oracle's block within its round oracle, in units of its own length.
	block_offset: usize,
	/// The oracle's own `log_msg_len`.
	log_len: usize,
}

/// One round oracle's operand in a queued relation.
struct QueuedOperand<Elem> {
	/// Index of the relation, into [`BaseFoldVerifierChannel::claims`].
	relation: usize,
	/// Evaluates the transparent multilinear, over the round oracle's variables, at the point the
	/// opening reduces to.
	transparent: TransparentEvalFn<Elem>,
}

/// A verifier channel that uses ZK BaseFold for all oracle commitments and openings.
///
/// Each round of the schedule is committed as one round oracle, laid out by
/// [`layout_rounds`](basefold::layout::layout_rounds), whose commitment arrives with the round's
/// last oracle. The opening runs over the round oracles.
///
/// # Type Parameters
///
/// - `'a`: Lifetime for borrowed references
/// - `F`: The binary field type
/// - `Channel`: The Merkle channel carrying all prover interaction
pub struct BaseFoldVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	/// The Merkle channel carrying all prover interaction: field elements, challenges,
	/// commitments, and openings.
	channel: Channel,
	schedule: &'a OracleSchedule,
	layouts: &'a [RoundLayout],
	fri_params: &'a FRIParams<F>,
	/// One commitment per round committed so far, so its length is also the open round's index.
	oracle_commitments: Vec<Channel::Commitment>,
	/// How many oracles have been received so far.
	n_received: usize,
	/// How many of the open round's oracles have been received so far.
	n_in_round: usize,
	/// Operands of the relations queued by [`IOPVerifierChannel::verify_multi_oracle_relation`],
	/// indexed by round and opened together in [`Self::finish`]. One entry per committed round.
	queue: Vec<Vec<QueuedOperand<Channel::Elem>>>,
	/// The claim of each queued relation, in queue order.
	claims: Vec<Channel::Elem>,
}

impl<'a, F, Channel> BaseFoldVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	/// Creates a new BaseFold ZK verifier channel over a Merkle channel from precomputed FRI
	/// parameters.
	///
	/// The FRI parameters should already account for ZK (log_batch_size = 1, doubled message
	/// length).
	pub const fn new(
		channel: Channel,
		schedule: &'a OracleSchedule,
		layouts: &'a [RoundLayout],
		fri_params: &'a FRIParams<F>,
	) -> Self {
		Self {
			channel,
			schedule,
			layouts,
			fri_params,
			oracle_commitments: Vec::new(),
			n_received: 0,
			n_in_round: 0,
			queue: Vec::new(),
			claims: Vec::new(),
		}
	}

	/// Consumes the channel and verifies the single combined opening over **all** committed
	/// oracles.
	///
	/// All oracle relations queued by
	/// [`verify_multi_oracle_relation`](IOPVerifierChannel::verify_multi_oracle_relation) across
	/// every call are processed here in one batch: batching into one relation, masking, one
	/// sumcheck reducing the masked claim to a shared point `r`, then one combined FRI opening
	/// over every committed oracle (in oracle-index order). Because the whole opening is deferred
	/// to this point, every oracle is committed and there is a single sumcheck point, so the
	/// precomputed combined `FRIParams` (`optimal_for_batch` over all oracle specs) serves the
	/// opening.
	///
	/// Returns the Merkle channel, so a caller can still reach what it accumulated.
	pub fn finish(self) -> Result<Channel, Error> {
		let Self {
			mut channel,
			schedule,
			layouts,
			fri_params,
			oracle_commitments,
			n_received,
			n_in_round: _,
			queue,
			claims,
		} = self;

		let n_remaining = schedule.specs().len() - n_received;
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining",);

		if !claims.is_empty() {
			verify_batch_zk_basefold(
				&mut channel,
				layouts,
				fri_params,
				&oracle_commitments,
				queue,
				&claims,
			)?;
		}

		Ok(channel)
	}
}

/// Verifies the combined ZK BaseFold opening over all committed round oracles.
///
/// This drives `channel` — the Merkle channel taken from the destructured
/// [`BaseFoldVerifierChannel`] — through its [`MerkleIPVerifierChannel`] interface: it batches the
/// queued relations into one, reads the masked inner products σ_i, runs one sumcheck reducing the
/// masked claim to a point `r`, then opens all committed oracles together with a single combined
/// FRI over the piecewise-concatenated oracle.
///
/// Everything per oracle runs in round order: `operands` arrives indexed by round, as do
/// `layouts`, `oracle_commitments`, the masking inner products σ_i and the reduced evaluations
/// α_i.
///
/// Phase B collapses the oracle-index variables up front at sampled batching challenges `r'`: the
/// combined target is `s' = Σ_i e[i]·α_i·∏_{j≥n_i}(1 - r_j)` with `e` the indicator expanded at
/// `r'`, and the single combined FRI (`fri_params`) opens all `k` committed `[π_i ‖ ω_i]`
/// codewords.
fn verify_batch_zk_basefold<F, Channel>(
	channel: &mut Channel,
	layouts: &[RoundLayout],
	fri_params: &FRIParams<F>,
	oracle_commitments: &[Channel::Commitment],
	operands: Vec<Vec<QueuedOperand<Channel::Elem>>>,
	claims: &[Channel::Elem],
) -> Result<(), Error>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	let n_committed = oracle_commitments.len();
	assert_eq!(operands.len(), n_committed);

	// `𝐧 = max_i log_msg_len_i`, the variable count of the combined opening / materialized buffer.
	let max_n = layouts
		.iter()
		.map(|layout| layout.log_len)
		.max()
		.expect("at least one oracle");

	// Batch the relations into the single relation `Σ_i <π_i, T_i> = S`.
	let (transparents, claim) = batch_relations(channel, operands, claims);

	// === Masking step ===
	// Only ZK oracles are masked: read their σ_i (one per ZK oracle) and sample the single shared
	// γ. With no ZK oracle, γ is never sampled.
	let n_zk = layouts.iter().filter(|layout| layout.is_zk).count();
	let sigmas = channel.recv_many(n_zk)?;
	let gamma = (!sigmas.is_empty()).then(|| channel.sample());

	// The sumcheck runs over `Σ_i w_i·<π_i', T_i>`. A ZK oracle's message is masked to
	// π_i' = (1-γ)·π_i + γ·ω_i and has weight 1; a non-ZK oracle's is unmasked and has weight 1-γ.
	// Every term then carries the same (1-γ) on its share of S, so the masked claim is
	// S' = (1-γ)·S + γ·Σ_i σ_i.
	let (sum_prime, weights) = match &gamma {
		Some(gamma) => {
			let sigma_sum = sigmas.into_iter().sum::<Channel::Elem>();
			let non_zk_weight = Channel::Elem::one() - gamma.clone();
			let weights = layouts
				.iter()
				.map(|layout| {
					if layout.is_zk {
						Channel::Elem::one()
					} else {
						non_zk_weight.clone()
					}
				})
				.collect::<Vec<_>>();
			(extrapolate_line(claim, sigma_sum, gamma.clone()), weights)
		}
		None => (claim, vec![Channel::Elem::one(); n_committed]),
	};

	// === Phase A: one sumcheck on the masked claim (degree 2, sum of bivariate products) ===
	let SumcheckOutput {
		eval: sumcheck_reduced_eval,
		challenges: sumcheck_challenges,
	} = sumcheck::verify::<F, _>(max_n, 2, sum_prime, channel)?;

	// Receive the evaluation of each oracle at the challenge point.
	let alphas = channel.recv_many(n_committed)?;

	// `verify` returns binding-order challenges; reverse to variable-indexed (low-to-high).
	let mut point = sumcheck_challenges;
	point.reverse();

	// Reduce the claim: each oracle contributes w_i · α_i · T_i(ρ_i) · eq(0^extra, padding).
	let expected = izip!(transparents, layouts, &alphas, weights)
		.map(|(transparent, layout, alpha_i, weight)| {
			let (eval_coords, padding_coords) = point.split_at(layout.log_len);
			let pad_eq = eq_ind_zero(padding_coords);
			weight * alpha_i.clone() * transparent(eval_coords) * pad_eq
		})
		.sum::<Channel::Elem>();
	channel.assert_zero(sumcheck_reduced_eval - expected)?;

	// === Phase B: single combined-FRI MLE-check over the piecewise-concatenated oracle ===
	// Collapse the oracle-index variables up front at sampled batching challenges `r'`: the
	// combined multilinear is 𝛑(X) = Σ_i e[i]·π_i^↑(X) with e = eq(·, r'), and the combined target
	// is s' = 𝛑(r) = Σ_i e[i]·α_i·∏_{j≥n_i}(1 - r_j).
	let log_n_oracles = log2_ceil_usize(n_committed);
	let outer_challenges = channel.sample_many(log_n_oracles);
	let eq_tensor = eq_ind_partial_eval_scalars(&outer_challenges);
	// In the combined buffer each oracle is zero-padded over its `log_lift` dims and *repeated*
	// over the remaining `log_repeat = max_n - n_i - log_lift` high dims, so its evaluation at
	// `point` picks up the eq-to-zero factor over the lift dims only (the repeat dims contribute
	// 1).
	let s_prime = izip!(fri_params.input_oracles(), layouts, eq_tensor, alphas)
		.map(|(fri_oracle, layout, eq_i, alpha_i)| {
			let n_i = layout.log_len;
			let log_lift = fri_oracle.log_lift;
			eq_i * alpha_i * eq_ind_zero(&point[n_i..][..log_lift])
		})
		.sum::<Channel::Elem>();

	// The opening routine asserts the final FRI/MLE-check consistency internally.
	basefold::verify_mlecheck_basefold(
		fri_params,
		oracle_commitments,
		s_prime,
		&point,
		gamma,
		&outer_challenges,
		channel,
	)?;

	Ok(())
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
/// chosen as a function of it. An oracle in no relation gets `T_i = 0`.
///
/// Returns the evaluator of each `T_i`, in oracle-index order, and `S`.
///
/// Mirrors the prover-side batching in `binius_iop_prover::basefold::channel`.
fn batch_relations<F, Channel>(
	channel: &mut Channel,
	operands: Vec<Vec<QueuedOperand<Channel::Elem>>>,
	claims: &[Channel::Elem],
) -> (Vec<TransparentEvalFn<Channel::Elem>>, Channel::Elem)
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	let mu = channel.sample();
	let claim = evaluate_univariate(claims, &mu);
	let mu_powers = Rc::new(powers(mu).take(claims.len()).collect::<Vec<_>>());

	let transparents = operands
		.into_iter()
		.map(|operands| {
			let mu_powers = Rc::clone(&mu_powers);
			Box::new(move |point: &[Channel::Elem]| {
				operands
					.iter()
					.map(|operand| {
						mu_powers[operand.relation].clone() * (operand.transparent)(point)
					})
					.sum()
			}) as TransparentEvalFn<_>
		})
		.collect();

	(transparents, claim)
}

impl<F, Channel> IPVerifierChannel<F> for BaseFoldVerifierChannel<'_, F, Channel>
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

impl<F, Channel> WordIPVerifierChannel<F> for BaseFoldVerifierChannel<'_, F, Channel>
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

impl<'a, F, Channel> IOPVerifierChannel<F> for BaseFoldVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	type Oracle = BaseFoldOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.schedule.specs()[self.n_received..]
	}

	fn recv_oracle(
		&mut self,
		len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error> {
		let spec = *self
			.remaining_oracle_specs()
			.first()
			.expect("recv_oracle called but no remaining oracle specs");
		assert_eq!(len, spec.len, "oracle content length must match its spec");
		// A spec is zero-knowledge iff the protocol is and the oracle is witness-dependent. The
		// protocol-level flag is not known here, so only one direction can be checked.
		assert!(
			!spec.is_zk || is_witness_dependent,
			"a zero-knowledge oracle spec must be received as witness-dependent"
		);

		let index = self.oracle_commitments.len();
		let layout = &self.layouts[index];
		let oracle = BaseFoldOracle {
			round: index,
			block_offset: layout.block_offsets[self.n_in_round],
			log_len: spec.log_msg_len(),
		};
		self.n_received += 1;
		self.n_in_round += 1;

		// The round's last oracle brings the commitment to the whole round oracle.
		if self.n_in_round < layout.block_offsets.len() {
			return Ok(oracle);
		}
		self.n_in_round = 0;

		// Receive the commitment with its Merkle tree shape, matching the prover-side commit: the
		// oracle's codeword has dimension `log_dim - log_lift` and one interleaved coset of
		// `2^log_batch_size` scalars per leaf.
		let fri_oracle = &self.fri_params.input_oracles()[index];
		let depth = (self.fri_params.rs_code().log_dim() - fri_oracle.log_lift)
			+ self.fri_params.rs_code().log_inv_rate();

		// The committed message length implied by this shape is `log_batch_size + depth -
		// log_inv_rate`; it must cover the round oracle plus, for a ZK one, the equal-length
		// interleaved mask.
		assert_eq!(
			fri_oracle.log_batch_size() + depth - self.fri_params.rs_code().log_inv_rate(),
			layout.log_len + usize::from(layout.is_zk),
			"invariant: the FRI commitment shape must be consistent with the round oracle's length"
		);

		let commitment = self
			.channel
			.recv_merkle_commitment(1 << fri_oracle.log_batch_size(), depth)?;

		self.oracle_commitments.push(commitment);
		self.queue.push(Vec::new());

		Ok(oracle)
	}

	fn verify_multi_oracle_relation(
		&mut self,
		operands: impl IntoIterator<Item = (Self::Oracle, TransparentEvalFn<Self::Elem>)>,
		claim: Self::Elem,
	) -> Result<(), Error> {
		// Queue each operand under its round; the actual opening (masking + sumcheck + combined
		// FRI) happens once, over all committed round oracles, in [`Self::finish`].
		let relation = self.claims.len();
		let n_committed = self.queue.len();
		for (oracle, transparent) in operands {
			let BaseFoldOracle {
				round,
				block_offset,
				log_len,
			} = oracle;
			assert!(round < n_committed, "oracle's round {round} is not committed yet");

			// On the round oracle, the operand is `t(v) · eq(b, u)`: the oracle's own
			// transparent over its low variables, and zero outside its block `b` over the high.
			let block_bits = (0..self.layouts[round].log_len - log_len)
				.map(|bit| {
					if (block_offset >> bit) & 1 == 1 {
						Channel::Elem::one()
					} else {
						Channel::Elem::zero()
					}
				})
				.collect::<Vec<_>>();
			let transparent = Box::new(move |point: &[Channel::Elem]| {
				let (low, high) = point.split_at(log_len);
				eq_ind(high, &block_bits) * transparent(low)
			});

			self.queue[round].push(QueuedOperand {
				relation,
				transparent,
			});
		}
		self.claims.push(claim);
		Ok(())
	}
}
