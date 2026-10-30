// Copyright 2026 The Binius Developers

//! The Libra mask of the batch sumcheck, and the oracle the compiler commits it in.
//!
//! The mask on `ℓ` variables is the polynomial of individual degree 2
//!
//! ```text
//! h(X) = Σ_{j < ℓ} (h'(j, 0) + h'(j, 1) X_j + h'(j, 2) X_j²)
//! ```
//!
//! read off a coefficient table `h'` over `B_ν × B_2`, `ν = ⌈log2 ℓ⌉`. Entry `(j, k)` sits at
//! index `j + k · 2^ν`, and the entries with `j ≥ ℓ` or `k = 3` are slack. Then
//! `h(r) = <h', libra-eval_r>` with `libra-eval_r(j, k) = r_j^k` on the used entries and zero on
//! the slack, so the evaluation is a linear query on the committed table.

use binius_field::field::FieldOps;
use binius_math::multilinear::hypercube::{InfCube, eq_ind_partial_eval_scalars};
use binius_utils::checked_arithmetics::log2_ceil_usize;

use crate::basefold::layout::RoundLayout;

/// `δ = ⌈log2 (d + 1)⌉` for the mask's degree `d = 2`.
pub const LOG_DEGREE: usize = 2;

/// The `ν` of a mask on `n_vars` variables: the log of the table's row count.
pub const fn log_n_rows(n_vars: usize) -> usize {
	log2_ceil_usize(n_vars)
}

/// The layout of the round the compiler commits the Libra oracle in, if the opening is masked.
///
/// The mask covers the batch sumcheck over `ℓ = max_r ℓ_r` of the schedule's `layouts`, and is only
/// needed when some round is zero-knowledge. Its table of `2^{ν + δ}` entries is a round of its
/// own.
pub fn libra_layout(layouts: &[RoundLayout]) -> Option<RoundLayout> {
	if !layouts.iter().any(|layout| layout.is_zk) {
		return None;
	}
	let n_vars = layouts
		.iter()
		.map(|layout| layout.log_len)
		.max()
		.expect("a ZK layout exists");
	Some(RoundLayout {
		log_len: log_n_rows(n_vars) + LOG_DEGREE,
		is_zk: true,
		block_offsets: vec![0],
	})
}

/// The table `libra-eval_r` over `log_len` variables, so that `h(r) = <h', libra-eval_r>`.
pub fn libra_eval_table<F: FieldOps>(point: &[F], log_len: usize) -> Vec<F> {
	let log_n_rows = log_n_rows(point.len());
	assert!(log_n_rows + LOG_DEGREE <= log_len); // precondition

	let mut table = vec![F::zero(); 1 << log_len];
	for (j, r_j) in point.iter().enumerate() {
		table[j] = F::one();
		table[j | 1 << log_n_rows] = r_j.clone();
		table[j | 2 << log_n_rows] = r_j.clone().square();
	}
	table
}

/// Evaluates the multilinear with monomial coefficients `libra-eval_r` at `point_prime`.
///
/// The table is nonzero only where the coordinates above `ν + δ` are zero, and there it factors
/// into the row `j` and the degree `k`, so the evaluation is
/// `Σ_{j < ℓ} point_prime^j · (1 + r_j y_1 + r_j² y_2)` with `y` the monomials of the two degree
/// coordinates. That costs `O(ℓ)` rather than a pass over the table.
pub fn libra_eval_monomial<F: FieldOps>(point: &[F], point_prime: &[F]) -> F {
	let log_n_rows = log_n_rows(point.len());
	let rows = eq_ind_partial_eval_scalars::<InfCube, F>(&point_prime[..log_n_rows]);
	let degrees =
		eq_ind_partial_eval_scalars::<InfCube, F>(&point_prime[log_n_rows..][..LOG_DEGREE]);
	let (y_1, y_2) = (&degrees[1], &degrees[2]);
	point
		.iter()
		.zip(rows)
		.map(|(r_j, row)| row * (F::one() + r_j.clone() * y_1 + r_j.clone().square() * y_2))
		.sum()
}
