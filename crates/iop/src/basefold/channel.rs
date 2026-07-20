// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP verifier channel.

use binius_field::{BinaryField, util::FieldFn};
use binius_ip::{
	channel::IPVerifierChannel,
	sumcheck::{self, BatchSumcheckOutput},
};
use binius_math::{
	line::extrapolate_line_packed,
	multilinear::eq::{eq_ind_partial_eval_scalars, eq_ind_zero},
	univariate::evaluate_univariate,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use itertools::izip;

use crate::{
	basefold,
	channel::{Error, IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	fri::FRIParams,
	merkle_channel::MerkleIPVerifierChannel,
};

/// Oracle handle returned by [`BaseFoldVerifierChannel::recv_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldOracle {
	index: usize,
}

/// A verifier channel that uses ZK BaseFold for all oracle commitments and openings.
///
/// This channel always applies zero-knowledge blinding. The FRI parameters must be set up
/// with `log_batch_size = 1` and `log_msg_len = witness_log_len + 1` to account for the mask.
///
/// # Type Parameters
///
/// - `'a`: Lifetime for borrowed references
/// - `F`: The binary field type
/// - `Channel`: The Merkle channel carrying all prover interaction
pub struct BaseFoldVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem = F>,
{
	/// The Merkle channel carrying all prover interaction: field elements, challenges,
	/// commitments, and openings.
	channel: Channel,
	oracle_specs: &'a [OracleSpec],
	fri_params: &'a FRIParams<F>,
	oracle_commitments: Vec<Channel::Commitment>,
	/// Oracle relations queued by [`Self::verify_oracle_relations`], opened together in
	/// [`Self::finish`].
	queue: Vec<OracleLinearRelation<BaseFoldOracle, F>>,
	next_oracle_index: usize,
}

impl<'a, F, Channel> BaseFoldVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem = F>,
{
	/// Creates a new BaseFold ZK verifier channel over a Merkle channel from precomputed FRI
	/// parameters.
	///
	/// The FRI parameters should already account for ZK (log_batch_size = 1, doubled message
	/// length).
	pub const fn new(
		channel: Channel,
		oracle_specs: &'a [OracleSpec],
		fri_params: &'a FRIParams<F>,
	) -> Self {
		Self {
			channel,
			oracle_specs,
			fri_params,
			oracle_commitments: Vec::new(),
			queue: Vec::new(),
			next_oracle_index: 0,
		}
	}

	/// Consumes the channel and verifies the single combined opening over **all** committed
	/// oracles.
	///
	/// All oracle relations queued by
	/// [`verify_oracle_relations`](IOPVerifierChannel::verify_oracle_relations) across every call
	/// are processed here in one batch: masking, one batched sumcheck reducing the masked claims
	/// to a shared point `r`, then one combined FRI opening over every committed oracle
	/// (in oracle-index order). Because the whole opening is deferred to this point, every oracle
	/// is committed and there is a single sumcheck point, so the precomputed combined `FRIParams`
	/// (`optimal_for_batch` over all oracle specs) serves the opening.
	pub fn finish(self) -> Result<(), Error> {
		let Self {
			mut channel,
			oracle_specs,
			fri_params,
			oracle_commitments,
			queue,
			next_oracle_index,
		} = self;

		let n_remaining = oracle_specs.len() - next_oracle_index;
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining",);

		if queue.is_empty() {
			return Ok(());
		}

		verify_batch_zk_basefold(&mut channel, oracle_specs, fri_params, oracle_commitments, queue)
	}
}

/// Verifies the combined ZK BaseFold opening over all committed oracles.
///
/// This drives `channel` — the Merkle channel taken from the destructured
/// [`BaseFoldVerifierChannel`] — through its [`MerkleIPVerifierChannel`] interface: it reads the
/// masked inner products σ_i, runs one batched sumcheck reducing the masked claims to a shared
/// point `r`, then opens all committed oracles together with a single combined FRI over the
/// piecewise-concatenated oracle.
///
/// The masking inner products and the batched sumcheck process the `relations` in arrival order (so
/// each reduced eval lines up with its batched-claim coefficient), while the per-oracle evaluations
/// α_i are indexed by oracle index. Each relation carries its oracle's index, so the two orders are
/// reconciled by indexing rather than by sorting the relations; `oracle_specs` and
/// `oracle_commitments` are indexed by oracle index.
///
/// Phase B collapses the oracle-index variables up front at sampled batching challenges `r'`: the
/// combined target is `s' = Σ_i e[i]·α_i·∏_{j≥n_i}(1 - r_j)` with `e = eq_ind_partial_eval(r')`,
/// and the single combined FRI (`fri_params`) opens all `k` committed `[π_i ‖ ω_i]` codewords.
fn verify_batch_zk_basefold<F, Channel>(
	channel: &mut Channel,
	oracle_specs: &[OracleSpec],
	fri_params: &FRIParams<F>,
	oracle_commitments: Vec<Channel::Commitment>,
	relations: Vec<OracleLinearRelation<BaseFoldOracle, F>>,
) -> Result<(), Error>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem = F>,
{
	let n_committed = oracle_commitments.len();

	// `𝐧 = max_i log_msg_len_i`, the variable count of the combined opening / materialized buffer.
	let max_n = oracle_specs
		.iter()
		.map(|spec| spec.log_msg_len)
		.max()
		.expect("at least one oracle");

	// === Masking step ===
	// Only ZK oracles are masked: read their σ_i (one per ZK oracle, in relation order) and sample
	// the single shared γ. With no ZK oracle, γ is never sampled.
	let n_zk = oracle_specs.iter().filter(|s| s.is_zk).count();
	let sigmas = channel.recv_many(n_zk)?;
	let gamma = (!sigmas.is_empty()).then(|| channel.sample());

	// Masked claim per relation: ZK → s_i' = extrapolate_line(claim, σ_i, γ); non-ZK → s_i' =
	// claim.
	let mut sigma_iter = sigmas.into_iter();
	let sum_primes = relations
		.iter()
		.map(|relation| {
			if oracle_specs[relation.oracle.index].is_zk {
				let sigma = sigma_iter.next().expect("one σ per ZK oracle");
				extrapolate_line_packed(
					relation.claim,
					sigma,
					gamma.expect("γ sampled when ZK oracles present"),
				)
			} else {
				relation.claim
			}
		})
		.collect::<Vec<_>>();

	// === Phase A: batched sumcheck on the masked claims (degree 2, bivariate product) ===
	let BatchSumcheckOutput {
		batch_coeff: sumcheck_batch_coeff,
		eval: sumcheck_reduced_eval,
		challenges: sumcheck_challenges,
	} = sumcheck::batch_verify::<F, _>(max_n, 2, &sum_primes, channel)?;

	// Receive the evaluation of each oracle at the challenge point.
	let alphas: Vec<F> = channel.recv_many(n_committed)?;

	// `batch_verify` returns binding-order challenges; reverse to variable-indexed (low-to-high).
	let mut point = sumcheck_challenges;
	point.reverse();

	// Reduce the batched claim: each oracle contributes α_i · t_i(ρ_i) · eq(0^extra, padding).
	let contributions = relations
		.into_iter()
		.map(|relation| {
			let alpha_i = alphas[relation.oracle.index];
			let n_i = oracle_specs[relation.oracle.index].log_msg_len;
			let (eval_coords, padding_coords) = point.split_at(n_i);
			let pad_eq = eq_ind_zero(padding_coords);
			let transparent_eval = (relation.transparent)(eval_coords);
			alpha_i * transparent_eval * pad_eq
		})
		.collect::<Vec<_>>();
	let expected = evaluate_univariate(&contributions, sumcheck_batch_coeff);
	channel.assert_zero(sumcheck_reduced_eval - expected)?;

	// === Phase B: single combined-FRI MLE-check over the piecewise-concatenated oracle ===
	// Collapse the oracle-index variables up front at sampled batching challenges `r'`: the
	// combined multilinear is 𝛑(X) = Σ_i e[i]·π_i^↑(X) with e = eq(·, r'), and the combined target
	// is s' = 𝛑(r) = Σ_i e[i]·α_i·∏_{j≥n_i}(1 - r_j).
	let log_n_oracles = log2_ceil_usize(n_committed);
	let outer_challenges = channel.sample_many(log_n_oracles);
	let eq_tensor = eq_ind_partial_eval_scalars::<F>(&outer_challenges);
	// In the combined buffer each oracle is zero-padded over its `log_lift` dims and *repeated*
	// over the remaining `log_repeat = max_n - n_i - log_lift` high dims, so its evaluation at
	// `point` picks up the eq-to-zero factor over the lift dims only (the repeat dims contribute
	// 1).
	let s_prime = izip!(fri_params.input_oracles(), oracle_specs, eq_tensor, alphas)
		.map(|(fri_oracle, spec, eq_i, alpha_i)| {
			let n_i = spec.log_msg_len;
			let log_lift = fri_oracle.log_lift;
			eq_i * alpha_i * eq_ind_zero(&point[n_i..][..log_lift])
		})
		.sum::<F>();

	let basefold::ReducedOutput {
		final_fri_value,
		final_sumcheck_value,
		..
	} = basefold::verify_mlecheck_basefold(
		fri_params,
		&oracle_commitments,
		s_prime,
		&point,
		gamma,
		&outer_challenges,
		channel,
	)?;

	// The MLE-check internalizes the eq factor, so consistency is plain equality.
	channel.assert_zero(final_sumcheck_value - final_fri_value)?;

	Ok(())
}

impl<F, Channel> IPVerifierChannel<F> for BaseFoldVerifierChannel<'_, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem = F>,
{
	type Elem = F;

	fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
		self.channel.recv_one()
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
		self.channel.recv_many(n)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
		self.channel.recv_array()
	}

	fn sample(&mut self) -> F {
		self.channel.sample()
	}

	fn observe_one(&mut self, val: F) -> F {
		self.channel.observe_one(val)
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
		self.channel.observe_many(vals)
	}

	fn assert_zero(&mut self, val: F) -> Result<(), binius_ip::channel::Error> {
		self.channel.assert_zero(val)
	}

	fn compute_public_value(&mut self, inputs: &[F], f: impl FieldFn<F>) -> F {
		self.channel.compute_public_value(inputs, f)
	}
}

impl<'a, F, Channel> IOPVerifierChannel<F> for BaseFoldVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem = F>,
{
	type Oracle = BaseFoldOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn recv_oracle(
		&mut self,
		_log_msg_len: usize,
		_is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error> {
		// A BaseFold commitment is a fixed-size Merkle digest, so `log_msg_len` is not needed here;
		// the per-oracle specs (used for the FRI opening) are supplied at channel construction.
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining oracle specs"
		);

		let index = self.next_oracle_index;

		// Receive the commitment with its Merkle tree shape, matching the prover-side commit: the
		// oracle's codeword has dimension `log_dim - log_lift` and one interleaved coset of
		// `2^log_batch_size` scalars per leaf.
		let fri_oracle = &self.fri_params.input_oracles()[index];
		let depth = (self.fri_params.rs_code().log_dim() - fri_oracle.log_lift)
			+ self.fri_params.rs_code().log_inv_rate();

		// The committed message length implied by this shape is `log_batch_size + depth -
		// log_inv_rate`; it must cover the spec's message plus, for a ZK oracle, the equal-length
		// interleaved mask.
		let spec = &self.oracle_specs[index];
		assert_eq!(
			fri_oracle.log_batch_size() + depth - self.fri_params.rs_code().log_inv_rate(),
			spec.log_msg_len + usize::from(spec.is_zk),
			"invariant: the FRI commitment shape must be consistent with the oracle spec's \
			 log_msg_len"
		);

		let commitment = self
			.channel
			.recv_merkle_commitment(1 << fri_oracle.log_batch_size(), depth)
			.map_err(|_| Error::ProofEmpty)?;

		self.oracle_commitments.push(commitment);
		self.next_oracle_index += 1;

		Ok(BaseFoldOracle { index })
	}

	fn verify_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<Self::Oracle, Self::Elem>>,
	) -> Result<(), Error> {
		// Queue the relations; the actual opening (masking + sumcheck + combined FRI) happens once,
		// over all committed oracles, in [`Self::finish`].
		for relation in oracle_relations {
			assert!(
				relation.oracle.index < self.oracle_commitments.len(),
				"oracle index {} out of bounds, expected < {}",
				relation.oracle.index,
				self.oracle_commitments.len()
			);
			self.queue.push(relation);
		}
		Ok(())
	}
}
