// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{BinaryField, Field, field::FieldOps};
use itertools::iterate;

#[derive(Debug, Clone, PartialEq)]
pub struct IntMulOutput<F> {
	pub eval_point: Vec<F>,
	pub a_evals: Vec<F>,
	pub b_evals: Vec<F>,
	pub c_lo_evals: Vec<F>,
	pub c_hi_evals: Vec<F>,
}

/// Output of Phase 1: GKR reduction of the exponentiation product tree.
///
/// Contains the evaluation point after prodcheck and the $2^k$ leaf evaluations of
/// $\widetilde{Q_i}$.
pub struct Phase1Output<F> {
	pub eval_point: Vec<F>,
	pub b_leaves_evals: Vec<F>,
}

pub struct Phase2Output<F> {
	pub twisted_eval_points: Vec<Vec<F>>,
	pub twisted_evals: Vec<F>,
}

/// Output of Phase 3: batched Frobenius selector sumcheck and LO * HI product sumcheck.
///
/// Contains the new evaluation point, $\widetilde{b}$ exponent evaluations, the selector
/// evaluation on $\widetilde{P}$, and root evaluations of $\widetilde{\textsf{LO}}$ and
/// $\widetilde{\textsf{HI}}$.
#[derive(Debug, Clone)]
pub struct Phase3Output<F> {
	pub eval_point: Vec<F>,
	pub b_exponent_evals: Vec<F>,
	pub selector_eval: F,
	pub c_lo_root_eval: F,
	pub c_hi_root_eval: F,
}

/// Output of Phase 4: all but last GKR layer for $\widetilde{a}$, $\widetilde{c}_{\textsf{lo}}$,
/// $\widetilde{c}_{\textsf{hi}}$.
///
/// Contains the evaluation point and leaf evaluations for each of the three product trees at
/// depth `log_bits - 1`.
pub struct Phase4Output<F> {
	pub eval_point: Vec<F>,
	pub a_evals: Vec<F>,
	pub c_lo_evals: Vec<F>,
	pub c_hi_evals: Vec<F>,
}

/// Compute the inverse Frobenius endomorphism $\varphi^{-i}(x)$.
///
/// The Frobenius endomorphism on $\mathbb{F}_{2^d}$ is $\varphi(x) = x^2$, so $\varphi^i(x) =
/// x^{2^i}$. Its order is $d$ (the extension degree), meaning $\varphi^d = \textsf{id}$.
/// Therefore $\varphi^{-i} = \varphi^{d - i}$, and we compute $\varphi^{-i}(x) = x^{2^{d-i}}$
/// by repeated squaring $d - i$ times.
fn inv_frobenius<F>(x: F, i: usize) -> F
where
	F: FieldOps,
	F::Scalar: BinaryField,
{
	let degree = F::Scalar::N_BITS;
	iterate(x, |g| g.clone().square())
		.nth(degree - i)
		.expect("infinite iterator")
}

/// Compute the inverse Frobenius sequence $[\varphi^{0}(x), \varphi^{-1}(x), \ldots,
/// \varphi^{-(n-1)}(x)]$ where $d$ is the extension degree of $\mathbb{F}_{2^d}$.
fn inv_frobenius_sequence<F>(x: F, n: usize) -> Vec<F>
where
	F: FieldOps,
	F::Scalar: BinaryField,
{
	let degree = F::Scalar::N_BITS;
	assert!(n <= degree + 1);
	let mut seq: Vec<F> = iterate(x, |g| g.clone().square())
		.take(degree + 1)
		.collect();
	seq.reverse();
	seq.truncate(n);
	seq
}

/// Apply inverse Frobenius twists to the leaf evaluation claims from Phase 1.
///
/// This reduces $2^k$ evaluation claims on $2^k$ separate multilinears $\widetilde{Q_i}$ at a
/// shared point $r$ to $2^k$ claims on a single multilinear $\widetilde{P}$ at $2^k$ different
/// points. Concretely, given claims $(r, s_i)$ where $s_i = \widetilde{Q_i}(r)$ and
/// $\widetilde{Q_i}(x) = \widetilde{P}(x)^{2^i}$, this applies $\varphi^{-i}$ (the inverse
/// Frobenius endomorphism) to both the evaluation point and the evaluation value. This linearizes
/// the degree-$2^i$ relation into a degree-1 claim: $\varphi^{-i}(s_i) =
/// \widetilde{P}(\varphi^{-i}(r))$, since $\varphi^{-i}(x^{2^i}) = x$ in $\mathbb{F}_{2^d}$.
///
/// # Arguments
///
/// * `k` - The log of the bit-width; there are $2^k$ leaf claims.
/// * `eval_point` - The shared evaluation point $r$.
/// * `evals` - The $2^k$ evaluations $s_0, \ldots, s_{2^k - 1}$.
pub fn frobenius_twist<F>(k: usize, eval_point: &[F], evals: &[F]) -> Phase2Output<F>
where
	F: FieldOps,
	F::Scalar: BinaryField,
{
	let n = 1 << k;
	assert_eq!(evals.len(), n);

	// Precompute inv_frobenius_sequence for each coordinate in eval_point.
	let coord_seqs: Vec<Vec<F>> = eval_point
		.iter()
		.map(|coord| inv_frobenius_sequence(coord.clone(), n))
		.collect();

	let twisted_eval_points = (0..n)
		.map(|i| coord_seqs.iter().map(|seq| seq[i].clone()).collect())
		.collect();

	let twisted_evals = evals
		.iter()
		.enumerate()
		.map(|(i, eval)| inv_frobenius(eval.clone(), i))
		.collect();

	Phase2Output {
		twisted_eval_points,
		twisted_evals,
	}
}

/// Recovers the multilinear evaluations of the $a, c_{\textsf{lo}}, c_{\textsf{hi}}$ polynomials.
/// 
/// The product checks for the exponentiations reduce to multilinear evaluations of affine translations of 
/// the $a, c_{\textsf{lo}}, c_{\textsf{hi}}$ polynomials. Specifically, the sumcheck reduces to
/// evaluations of
/// 
/// * $\textsf{select}(a(i, r), g^{2^i})$,
/// * $\textsf{select}(c_{\textsf{lo}}(i, r), g^{2^i})$,
/// * $\textsf{select}(c_{\textsf{hi}}(i, r), g^{2^(i + k)}$,
/// 
/// for all $i$ in $\{0, \ldots, 2^k - 1\}$, where
/// 
/// $$
/// \textsf{select}(S, V) = S * (V - 1) + 1.
/// $$
/// 
/// $g$ is a constant multiplicative generator of the field $F$.
/// 
/// Given, these evaluations, this function computes and returns $a(i, r), c_{\textsf{lo}}(i, r), c_{\textsf{hi}}(i, r)$.
pub fn normalize_a_c_exponent_evals<F, E>(
	k: usize,
	selected_a_evals: Vec<E>,
	selected_c_lo_evals: Vec<E>,
	selected_c_hi_evals: Vec<E>,
) -> [Vec<E>; 3]
where
	F: Field,
	E: FieldOps<Scalar = F> + From<F>,
{
	assert_eq!(selected_a_evals.len(), 1 << k);
	assert_eq!(selected_c_lo_evals.len(), 1 << k);
	assert_eq!(selected_c_hi_evals.len(), 1 << k);

	// Compute the normalization factors (conjugate - 1)^{-1} in F, then convert to E.
	let inv_factors: Vec<E> = iterate(F::MULTIPLICATIVE_GENERATOR, |g| g.square())
		.take(2 << k)
		.map(|conjugate| E::from((conjugate - F::ONE).invert().expect("non-zero")))
		.collect();

	let (lo_inv_factors, hi_inv_factors) = inv_factors.split_at(1 << k);

	let a_evals = recover_selectors(selected_a_evals, lo_inv_factors);
	let c_lo_evals = recover_selectors(selected_c_lo_evals, lo_inv_factors);
	let c_hi_evals = recover_selectors(selected_c_hi_evals, hi_inv_factors);

	[a_evals, c_lo_evals, c_hi_evals]
}

fn recover_selectors<F: FieldOps>(selecteds: Vec<F>, inv_factors: &[F]) -> Vec<F> {
	assert_eq!(selecteds.len(), inv_factors.len());

	let one = F::one();
	iter::zip(selecteds, inv_factors)
		.map(|(selected, inv_factor)| {
			// z_i = s_i * (v_i - 1) + 1
			// Recover s_i = (z_i - 1) * (v_i - 1)^{-1}
			(selected - one.clone()) * inv_factor
		})
		.collect()
}
