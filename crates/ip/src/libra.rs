// Copyright 2026 The Binius Developers

//! The Libra mask polynomial and its committed coefficient table.
//!
//! A Libra mask on $n$ variables is the separable polynomial $g(X) = \sum_i g_i(X_i)$, each $g_i$ a
//! univariate of degree $d$. Its coefficients are committed as a table $g'$ of
//! $2^{m_n} \times 2^{m_d}$ entries, row $j$ holding the coefficients of $g_j$, so that
//! $g(r) = \langle g', \textsf{libra-eval}_r \rangle$ is a linear query on the committed table.

use binius_field::{field::FieldOps, util::powers};
use binius_math::multilinear::eq::eq_ind_partial_eval_scalars;

/// Returns `(m_n, m_d)` dimensions for a mask polynomial buffer.
///
/// The ZK MLE-check protocol uses a separable mask polynomial with `n_vars` univariate polynomials,
/// each of degree `d`. The mask coefficients are stored in a `2^m_n × 2^m_d` matrix where:
/// - `m_n`: log of number of rows (one per variable, padded to power of two)
/// - `m_d`: log of row size (degree + 1 coefficients, padded to power of two)
///
/// The protocol imposes `n * d + 1` linear constraints on the coefficients. The `n_extra_dof`
/// parameter specifies additional degrees of freedom needed (e.g., for FRI openings), which
/// may increase `m_n` to ensure `2^(m_n + m_d) >= n * d + 1 + n_extra_dof`.
///
/// # Arguments
///
/// * `n_vars` - Number of variables (n).
/// * `degree` - Degree of each univariate polynomial (d).
/// * `n_extra_dof` - Number of additional degrees of freedom.
///
/// # Returns
///
/// A tuple `(m_n, m_d)` where the total mask buffer size is `2^(m_n + m_d)`.
pub fn mask_buffer_dimensions(n_vars: usize, degree: usize, n_extra_dof: usize) -> (usize, usize) {
	let min_buffer_size = n_vars * degree + 1 + n_extra_dof;
	let m_d = (degree + 1).next_power_of_two().ilog2() as usize;
	// m_n must be large enough to hold n_vars rows AND satisfy the DOF constraint
	let m_n_for_vars = n_vars.next_power_of_two().ilog2() as usize;
	let m_n_for_size = (min_buffer_size.next_power_of_two().ilog2() as usize).saturating_sub(m_d);
	let m_n = m_n_for_vars.max(m_n_for_size);
	(m_n, m_d)
}

/// Evaluates the MLE of the libra_eval polynomial at a query point.
///
/// Computes the multilinear extension of `libra_eval_r` at the query point `(query_j, query_k)`:
///
/// ```text
/// Σⱼ Σₖ eq(j, query_j) · eq(k, query_k) · r[j]^k
/// ```
///
/// for `j < n_vars` and `k ≤ degree`, where `eq` is the equality indicator polynomial.
///
/// # Arguments
///
/// * `challenge_point` - The challenge point `r` from sumcheck (length `n_vars`)
/// * `query_j` - Query point for the variable index (length `m_n`)
/// * `query_k` - Query point for the power index (length `m_d`)
/// * `n_vars` - Number of variables in the mask polynomial
/// * `degree` - Degree of each univariate in the mask polynomial
pub fn libra_eval<F: FieldOps>(
	challenge_point: &[F],
	query_j: &[F],
	query_k: &[F],
	n_vars: usize,
	degree: usize,
) -> F {
	let eq_j = eq_ind_partial_eval_scalars(query_j);
	let eq_k = eq_ind_partial_eval_scalars(query_k);

	eq_j.iter()
		.take(n_vars)
		.zip(challenge_point)
		.map(|(eq_j_val, r_j)| {
			eq_k.iter()
				.take(degree + 1)
				.zip(powers(r_j.clone()))
				.map(|(eq_k_val, r_j_power)| eq_j_val.clone() * eq_k_val.clone() * r_j_power)
				.sum::<F>()
		})
		.sum()
}
