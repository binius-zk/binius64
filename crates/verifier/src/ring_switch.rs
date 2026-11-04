// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField, FieldOps, PackedField};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{
	multilinear::{
		eq::eq_ind_partial_eval_scalars, eq_prefix::eq_ind_prefix_eval_tensor_algebra,
		evaluate::evaluate_inplace_scalars,
	},
	tensor_algebra::TensorAlgebra,
};

use crate::config::B1;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("channel: {0}")]
	Channel(#[from] binius_ip::channel::Error),
}

/// Output of ring-switching verification.
pub struct RingSwitchVerifyOutput<F: FieldOps> {
	/// The row-batching challenges (expanded via eq_ind).
	pub eq_r_double_prime: Vec<F>,
	/// The verified sumcheck claim for BaseFold.
	pub sumcheck_claim: F,
}

/// Verify the ring-switching reduction.
///
/// Takes the expected evaluation claim and eval point, and:
/// 1. Receives s_hat_v from prover via channel
/// 2. Verifies the partial evaluation matches the expected claim
/// 3. Samples row-batching challenges
/// 4. Computes the sumcheck claim for BaseFold
///
/// Returns the eq_r_double_prime (for final check) and sumcheck claim.
///
/// ## Arguments
///
/// * `evaluation_claim` - the expected evaluation of the B1 polynomial
/// * `eval_point` - the evaluation point from shift reduction
/// * `channel` - the verifier channel for receiving/sampling
///
/// ## Preconditions
///
/// * `eval_point.len()` must equal `log_witness_elems + log_packing` where log_packing is the
///   base-2 log of the extension degree of F over B1
pub fn verify<F, C>(
	evaluation_claim: C::Elem,
	eval_point: &[C::Elem],
	channel: &mut C,
) -> Result<RingSwitchVerifyOutput<C::Elem>, Error>
where
	F: BinaryField + PackedField<Scalar = F>,
	C: IPVerifierChannel<F>,
	C::Elem: FieldOps<Scalar = F> + From<F>,
{
	let log_packing = F::LOG_DEGREE;
	let (eval_point_low, _eval_point_high) = eval_point.split_at(log_packing);

	// Receive s_hat_v
	let s_hat_v = channel.recv_many(1 << log_packing)?;

	// Verify partial eval matches expected claim
	let computed_claim = evaluate_inplace_scalars(s_hat_v.clone(), eval_point_low);
	channel.assert_zero(evaluation_claim - computed_claim)?;

	// Basis transpose
	let s_hat_u = TensorAlgebra::<B1, C::Elem>::new(s_hat_v).transpose();

	// Sample r_double_prime
	let r_double_prime = channel.sample_many(log_packing);
	let eq_r_double_prime = eq_ind_partial_eval_scalars(&r_double_prime);

	// Compute sumcheck claim
	let sumcheck_claim = evaluate_inplace_scalars(s_hat_u.elems, &r_double_prime);

	Ok(RingSwitchVerifyOutput {
		eq_r_double_prime,
		sumcheck_claim,
	})
}

/// Evaluate the ring switching equality indicator at a given point.
///
/// The ring switching equality indicator is the multilinear function $A$ from [DP24],
/// Construction 3.1, generalized to an oracle of `n_pack` entries rather than a power of two
/// (binius.xyz#128, §Ring-Switching): its cell factor is the prefix indicator
/// $\mathrm{eq}_{<n}(z, y)$, so $A$ vanishes above the oracle. It is evaluated succinctly by
/// computing that indicator over the tensor algebra, where the first components live in the
/// vertical subring and the later components live in the horizontal subring. Then we apply
/// row-batching to the tensor algebra element.
///
/// `z_vals` may have more coordinates than `query`, when the claim spans a wider word-index space
/// than the oracle. The oracle is zero there, so those surplus coordinates contribute
/// $c = \prod_j (1 - z_j)$. The row-batching map is only $\mathbb{F}_2$-linear, so $c$ enters
/// through the vertical subring rather than as a scalar on the result.
///
/// ## Arguments
///
/// * `n_pack` - the oracle's length
/// * `z_vals` - the vertical evaluation point
/// * `query` - the horizontal evaluation point, over the oracle's variables
/// * `expanded_row_batch_query` - the scaling elements for row-batching
///
/// ## Pre-conditions
///
/// * `z_vals` has at least as many components as `query`
/// * `n_pack` is at most `2^query.len()`
/// * the length of `expanded_row_batch_query` must equal `FE::DEGREE`
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub fn eval_rs_eq<F>(n_pack: usize, z_vals: &[F], query: &[F], expanded_row_batch_query: &[F]) -> F
where
	F: FieldOps<Scalar: BinaryField>,
{
	assert!(z_vals.len() >= query.len()); // pre-condition
	assert_eq!(expanded_row_batch_query.len(), F::Scalar::DEGREE,); // pre-condition

	let (z_vals, surplus) = z_vals.split_at(query.len());
	let surplus_scale = surplus.iter().map(|z_j| F::one() - z_j).product::<F>();
	eq_ind_prefix_eval_tensor_algebra::<B1, F>(n_pack, z_vals, query)
		.scale_vertical(&surplus_scale)
		.fold_vertical(expanded_row_batch_query)
}
