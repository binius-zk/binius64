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
/// Contains the new evaluation point $r$, the recombined $\widetilde{b}$ exponent claim, $A(r)$,
/// $C_{\textsf{lo}}(r)$, and $C_{\textsf{hi}}(r)$.
#[derive(Debug, Clone)]
pub struct Phase3Output<F> {
	pub eval_point: Vec<F>,
	/// The recombination point $r_I^b \in K^k$ sampled to collapse the $2^k$ per-bit
	/// $\widetilde{b}$ claims into one.
	pub r_ib: Vec<F>,
	/// The recombined exponent claim $\widetilde{b}(r_I^b, r)$, where $r$ is `eval_point`.
	pub b_recomb: F,
	/// $A(r)$, where $r$ is `eval_point`.
	pub gpow_a_eval: F,
	/// $C_{\textsf{lo}}(r)$.
	pub gpow_c_lo_eval: F,
	/// The base-$g$ high-half claim $\widetilde{C}_{\textsf{hi}}(\varphi^{-64}(r)) =
	/// \varphi^{-64}(\widetilde{D}_{\textsf{hi}}(r))$, obtained by twisting the base-$g^{2^{64}}$
	/// high-half claim so the $c_{\textsf{hi}}$ product tree runs at base $g$ uniformly with $a$
	/// and $c_{\textsf{lo}}$ (spec §IntMul high-half twist). Evaluated at `c_hi_eval_point`.
	pub gpow_c_hi_eval: F,
	/// The twisted constraint point $\varphi^{-64}(r)$ at which `gpow_c_hi_eval` is taken. Only
	/// the $c_{\textsf{hi}}$ product tree uses it (at Phase-4 layer 0); $a$ and $c_{\textsf{lo}}$
	/// use `eval_point`.
	pub c_hi_eval_point: Vec<F>,
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
pub fn inv_frobenius<F>(x: F, i: usize) -> F
where
	F: FieldOps,
	F::Scalar: BinaryField,
{
	let degree = F::Scalar::N_BITS;
	iterate(x, |g| g.clone().square())
		.nth(degree - i)
		.expect("infinite iterator")
}

/// Apply the inverse Frobenius endomorphism $\varphi^{-i}$ coordinate-wise to an evaluation point.
///
/// Used by the high-half twist (spec §IntMul): a base-$g^{2^i}$ root claim $\widetilde{D}(r)$ is
/// converted to the base-$g$ claim $\widetilde{C}(\varphi^{-i}(r)) =
/// \varphi^{-i}(\widetilde{D}(r))$, which evaluates the same multilinear at the twisted point
/// $\varphi^{-i}(r)$.
pub fn inv_frobenius_point<F>(point: &[F], i: usize) -> Vec<F>
where
	F: FieldOps,
	F::Scalar: BinaryField,
{
	point
		.iter()
		.map(|coord| inv_frobenius(coord.clone(), i))
		.collect()
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
/// The product checks for the exponentiations reduce to multilinear evaluations of affine
/// translations of the $a, c_{\textsf{lo}}, c_{\textsf{hi}}$ polynomials. Specifically, the
/// sumcheck reduces to evaluations of
///
/// * $\textsf{select}(a(i, r), g^{2^i})$,
/// * $\textsf{select}(c_{\textsf{lo}}(i, r), g^{2^i})$,
/// * $\textsf{select}(c_{\textsf{hi}}(i, r), g^{2^i})$,
///
/// for all $i$ in $\{0, \ldots, 2^k - 1\}$, where
///
/// $$
/// \textsf{select}(S, V) = S * (V - 1) + 1.
/// $$
///
/// $g$ is a constant multiplicative generator of the field $F$.
///
/// Given, these evaluations, this function computes and returns $a(i, r), c_{\textsf{lo}}(i, r),
/// c_{\textsf{hi}}(i, r)$.
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

	// Compute the normalization factors (conjugate - 1)^{-1} in F, then convert to E. All three
	// trees run at base g (the c_hi high-half twist is applied upstream), so they share the same
	// low generator powers g^{2^i} for i in 0..2^k.
	let inv_factors: Vec<E> = iterate(F::MULTIPLICATIVE_GENERATOR, |g| g.square())
		.take(1 << k)
		// Safety: `conjugate` ranges over powers of the multiplicative generator, which has odd
		// order, so `conjugate - 1` is never zero.
		.map(|conjugate| E::from(unsafe { (conjugate - F::ONE).invert() }))
		.collect();

	let a_evals = recover_selectors(selected_a_evals, &inv_factors);
	let c_lo_evals = recover_selectors(selected_c_lo_evals, &inv_factors);
	let c_hi_evals = recover_selectors(selected_c_hi_evals, &inv_factors);

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

/// Reconstruct the "selected" leaf evaluations from the raw per-bit evaluations.
///
/// This is the forward direction of [`normalize_a_c_exponent_evals`]: given the raw bit
/// evaluations $a(i, r), c_{\textsf{lo}}(i, r), c_{\textsf{hi}}(i, r)$, it returns the selected
/// leaf values
///
/// * $\textsf{select}(a(i, r), g^{2^i})$,
/// * $\textsf{select}(c_{\textsf{lo}}(i, r), g^{2^i})$,
/// * $\textsf{select}(c_{\textsf{hi}}(i, r), g^{2^i})$,
///
/// for $i \in \{0, \ldots, 2^k - 1\}$, where $\textsf{select}(S, V) = S \cdot (V - 1) + 1$.
///
/// The verifier reconstructs these forward from the prover's raw evaluations and binds them to the
/// GKR-verified leaf-product claims, rather than receiving them and inverting. $g$ is a constant
/// multiplicative generator of the field $F$.
pub fn reconstruct_selecteds<F, E>(
	k: usize,
	a_evals: &[E],
	c_lo_evals: &[E],
	c_hi_evals: &[E],
) -> [Vec<E>; 3]
where
	F: Field,
	E: FieldOps<Scalar = F> + From<F>,
{
	assert_eq!(a_evals.len(), 1 << k);
	assert_eq!(c_lo_evals.len(), 1 << k);
	assert_eq!(c_hi_evals.len(), 1 << k);

	// powers[j] = g^{2^j}, for j in 0..2^k. All three trees run at base g (the c_hi high-half twist
	// is applied upstream), so c_hi reconstructs from the same low powers as a and c_lo.
	let powers: Vec<E> = iterate(F::MULTIPLICATIVE_GENERATOR, |g| g.square())
		.take(1 << k)
		.map(E::from)
		.collect();

	[
		apply_selectors(a_evals, &powers),
		apply_selectors(c_lo_evals, &powers),
		apply_selectors(c_hi_evals, &powers),
	]
}

/// Apply the affine selector `z * (V - 1) + 1` pointwise, given the generator powers `V_i`.
fn apply_selectors<E: FieldOps>(raw_evals: &[E], powers: &[E]) -> Vec<E> {
	assert_eq!(raw_evals.len(), powers.len());

	let one = E::one();
	iter::zip(raw_evals, powers)
		.map(|(raw, power)| raw.clone() * (power.clone() - one.clone()) + one.clone())
		.collect()
}
