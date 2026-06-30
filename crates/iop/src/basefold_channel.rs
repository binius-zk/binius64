// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP verifier channel.
//!
//! This module provides [`BaseFoldVerifierChannel`], which implements [`IOPVerifierChannel`]
//! using FRI commitment and the batched ZK BaseFold opening protocol. ZK oracles are blinded with a
//! mask; non-ZK oracles are committed without a mask. All committed oracles are opened with a
//! single combined FRI.

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
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::{DeserializeBytes, checked_arithmetics::log2_ceil_usize};
use itertools::izip;

use crate::{
	basefold,
	channel::{Error, IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	fri::FRIParams,
	merkle_tree::MerkleTreeScheme,
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
/// - `MerkleScheme_`: The Merkle tree scheme for commitments
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct BaseFoldVerifierChannel<'a, F, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F>,
	Challenger_: Challenger,
{
	transcript: &'a mut VerifierTranscript<Challenger_>,
	merkle_scheme: &'a MerkleScheme_,
	oracle_specs: &'a [OracleSpec],
	fri_params: &'a FRIParams<F>,
	oracle_commitments: Vec<MerkleScheme_::Digest>,
	/// Oracle relations queued by [`Self::verify_oracle_relations`], opened together in
	/// [`Self::finish`].
	queue: Vec<OracleLinearRelation<'a, BaseFoldOracle, F>>,
	next_oracle_index: usize,
}

impl<'a, F, MerkleScheme_, Challenger_> BaseFoldVerifierChannel<'a, F, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	/// Creates a new BaseFold ZK verifier channel from precomputed FRI parameters.
	///
	/// The FRI parameters should already account for ZK (log_batch_size = 1, doubled message
	/// length).
	pub const fn from_precomputed(
		transcript: &'a mut VerifierTranscript<Challenger_>,
		merkle_scheme: &'a MerkleScheme_,
		oracle_specs: &'a [OracleSpec],
		fri_params: &'a FRIParams<F>,
	) -> Self {
		Self {
			transcript,
			merkle_scheme,
			oracle_specs,
			fri_params,
			oracle_commitments: Vec::new(),
			queue: Vec::new(),
			next_oracle_index: 0,
		}
	}

	/// Returns a reference to the underlying transcript.
	pub const fn transcript(&self) -> &VerifierTranscript<Challenger_> {
		self.transcript
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
			transcript,
			merkle_scheme,
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

		verify_batch_zk_basefold(
			transcript,
			merkle_scheme,
			oracle_specs,
			fri_params,
			oracle_commitments,
			queue,
		)
	}
}

/// Verifies the combined ZK BaseFold opening over all committed oracles.
///
/// This drives `channel` — the [`VerifierTranscript`] taken from the destructured
/// [`BaseFoldVerifierChannel`] — through its [`IPVerifierChannel`] interface: it reads the masked
/// inner products σ_i, runs one batched sumcheck reducing the masked claims to a shared point `r`,
/// then opens all committed oracles together with a single combined FRI over the
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
///
/// `channel` is the concrete [`VerifierTranscript`] rather than an arbitrary [`IPVerifierChannel`]
/// because the Phase-B FRI openings read Merkle query proofs, which fall outside that interface.
#[allow(clippy::too_many_arguments)]
fn verify_batch_zk_basefold<F, MerkleScheme_, Challenger_>(
	channel: &mut VerifierTranscript<Challenger_>,
	merkle_scheme: &MerkleScheme_,
	oracle_specs: &[OracleSpec],
	fri_params: &FRIParams<F>,
	oracle_commitments: Vec<MerkleScheme_::Digest>,
	relations: Vec<OracleLinearRelation<'_, BaseFoldOracle, F>>,
) -> Result<(), Error>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
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
	let gamma = (!sigmas.is_empty()).then(|| IPVerifierChannel::<F>::sample(channel));

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
		merkle_scheme,
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

impl<F, MerkleScheme_, Challenger_> IPVerifierChannel<F>
	for BaseFoldVerifierChannel<'_, F, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	type Elem = F;

	fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
		self.transcript
			.message()
			.read_scalar()
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
		self.transcript
			.message()
			.read_scalar_slice(n)
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
		self.transcript
			.message()
			.read()
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn sample(&mut self) -> F {
		CanSample::sample(&mut self.transcript)
	}

	fn observe_one(&mut self, val: F) -> F {
		self.transcript.observe().write_scalar(val);
		val
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
		self.transcript.observe().write_scalar_slice(vals);
		vals.to_vec()
	}

	fn assert_zero(&mut self, val: F) -> Result<(), binius_ip::channel::Error> {
		if val == F::ZERO {
			Ok(())
		} else {
			Err(binius_ip::channel::Error::InvalidAssert)
		}
	}

	fn compute_public_value(&mut self, inputs: &[F], f: impl FieldFn<F>) -> F {
		f.call::<F>(inputs)
	}
}

impl<'a, F, MerkleScheme_, Challenger_> IOPVerifierChannel<'a, F>
	for BaseFoldVerifierChannel<'a, F, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	type Oracle = BaseFoldOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, Error> {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining oracle specs"
		);

		let index = self.next_oracle_index;

		let commitment = self
			.transcript
			.message()
			.read::<MerkleScheme_::Digest>()
			.map_err(|_| Error::ProofEmpty)?;

		self.oracle_commitments.push(commitment);
		self.next_oracle_index += 1;

		Ok(BaseFoldOracle { index })
	}

	fn verify_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'a, Self::Oracle, Self::Elem>>,
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
