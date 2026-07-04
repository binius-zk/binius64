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
	/// $C_{\textsf{hi}}(r)$.
	pub gpow_c_hi_eval: F,
}

/// Output of Phase 4: all but last GKR layer for $\widetilde{a}$, $\widetilde{c}_{\textsf{lo}}$,
/// $\widetilde{c}_{\textsf{hi}}$.
///
/// Rather than binding the all-but-last-layer evaluations here, Phase 4 hands the reduced prodcheck
/// claim straight to Phase 5: the content-and-node point at which the three trees' all-but-last
/// layers are claimed, the reduced selector coordinates that batch them, and the combined claimed
/// evaluation. Phase 5 receives one reduced eval per tree and checks they recombine, weighted by
/// `eq(selector)`, to `combined_eval`.
pub struct Phase4Output<F> {
	/// The point `[suffix, bit_index]` (content coordinates followed by all-but-last-layer node
	/// coordinates) at which each tree's all-but-last-layer multilinear is claimed.
	pub eval_point: Vec<F>,
	/// The reduced selector coordinates that batch the three trees (padded to four).
	pub selector: Vec<F>,
	/// The batched prodcheck output evaluation: `Σ_t eq(selector, t) · eval_t`.
	pub combined_eval: F,
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

/// Reconstruct the "selected" leaf evaluations from the raw per-bit evaluations.
///
/// The product checks for the exponentiations reduce to multilinear evaluations of affine
/// translations of the $a, c_{\textsf{lo}}, c_{\textsf{hi}}$ polynomials. Given the raw bit
/// evaluations $a(i, r), c_{\textsf{lo}}(i, r), c_{\textsf{hi}}(i, r)$, this returns the selected
/// leaf values
///
/// * $\textsf{select}(a(i, r), g^{2^i})$,
/// * $\textsf{select}(c_{\textsf{lo}}(i, r), g^{2^i})$,
/// * $\textsf{select}(c_{\textsf{hi}}(i, r), g^{2^{i + k}})$,
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

	// powers[j] = g^{2^j}, for j in 0..2^{k+1}.
	let powers: Vec<E> = iterate(F::MULTIPLICATIVE_GENERATOR, |g| g.square())
		.take(2 << k)
		.map(E::from)
		.collect();
	let (lo_powers, hi_powers) = powers.split_at(1 << k);

	[
		apply_selectors(a_evals, lo_powers),
		apply_selectors(c_lo_evals, lo_powers),
		apply_selectors(c_hi_evals, hi_powers),
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
