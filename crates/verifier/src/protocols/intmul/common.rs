// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_field::{BinaryField, field::FieldOps};
use itertools::iterate;

use crate::config::LOG_WORD_SIZE_BITS;

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

/// The log2 of the number of limbs each exponent word is split into for the Phase 4/5 fixed-base
/// lookup.
pub const LOG_N_LIMBS: usize = 2;

/// The number of exponent limbs per word.
pub const N_LIMBS: usize = 1 << LOG_N_LIMBS;

/// The bit width of one exponent limb; the lookup table has `2^LIMB_BITS` rows.
pub const LIMB_BITS: usize = 1 << (LOG_WORD_SIZE_BITS - LOG_N_LIMBS);

/// The number of looked-up limb columns: one per limb of $\widetilde{a}$,
/// $\widetilde{c}_{\textsf{lo}}$, and $\widetilde{c}_{\textsf{hi}}$.
pub const N_LIMB_COLUMNS: usize = 3 * N_LIMBS;

/// Output of Phase 4: the three constant-base GKR product trees reduced to per-limb evaluation
/// claims.
///
/// Each tree's leaf layer is the concatenation of its `N_LIMBS` limb columns; Phase 4 reduces the
/// three roots to one evaluation claim per limb column at the shared content point `eval_point`.
pub struct Phase4Output<F> {
	/// The shared content point at which the limb columns are claimed.
	pub eval_point: Vec<F>,
	/// The `N_LIMBS` per-limb evaluations of the $\widetilde{a}$ tree leaf columns.
	pub a_limb_evals: Vec<F>,
	/// The `N_LIMBS` per-limb evaluations of the $\widetilde{c}_{\textsf{lo}}$ tree leaf columns.
	pub c_lo_limb_evals: Vec<F>,
	/// The `N_LIMBS` per-limb evaluations of the $\widetilde{c}_{\textsf{hi}}$ tree leaf columns.
	pub c_hi_limb_evals: Vec<F>,
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

/// The Frobenius twist amounts (numbers of squarings) mapping each looked-up limb column onto the
/// shared generator table, ordered `[a limbs, c_lo limbs, c_hi limbs]`.
///
/// Limb $l$ of $\widetilde{a}$ and $\widetilde{c}_{\textsf{lo}}$ has base $g^{2^{wl}}$ (where $w$
/// is the limb bit width); limb $l$ of $\widetilde{c}_{\textsf{hi}}$ has base
/// $g^{2^{w(\textsf{N\_LIMBS} + l)}}$ because the $\widetilde{c}_{\textsf{hi}}$ tree exponentiates
/// $g^{2^{2^k}}$. So column $(t, l)$ is the Frobenius power $\varphi^{ws}$ of the shared table
/// column $g^{(\cdot)}$, with $s = l$ for $t \in \{a, c_{\textsf{lo}}\}$ and $s = \textsf{N\_LIMBS}
/// + l$ for $t = c_{\textsf{hi}}$.
pub fn limb_column_twists() -> [usize; N_LIMB_COLUMNS] {
	std::array::from_fn(|j| {
		let (tree, limb) = (j / N_LIMBS, j % N_LIMBS);
		let s = if tree == 2 { N_LIMBS + limb } else { limb };
		LIMB_BITS * s
	})
}

/// Twist a limb-column evaluation claim onto the shared generator table.
///
/// The limb column satisfies $L = \varphi^{a} \circ U$ pointwise on the cube, where $U$ is the
/// looked-up column of the shared table and $a$ is the twist amount. Since $\varphi$ is
/// $\mathbb{F}_2$-linear and fixes the cube, the MLE claim twists as $\widetilde{U}
/// (\varphi^{-a}(r)) = \varphi^{-a}(\widetilde{L}(r))$, applying $\varphi^{-a}$ to the value and
/// every point coordinate.
pub fn twist_limb_claim<F>(twist: usize, eval_point: &[F], eval: F) -> (Vec<F>, F)
where
	F: FieldOps,
	F::Scalar: BinaryField,
{
	let twisted_point = eval_point
		.iter()
		.map(|coord| inv_frobenius(coord.clone(), twist))
		.collect();
	(twisted_point, inv_frobenius(eval, twist))
}

/// Evaluate the MLE of the fixed-base power table `i ↦ g^i` at a point.
///
/// Row $i$ is the product over the set bits $j$ of $i$ of $g^{2^j}$, so the MLE factors as
/// $\prod_j \textsf{select}(y_j, g^{2^j})$ with $\textsf{select}(S, V) = S \cdot (V - 1) + 1$ —
/// the same select formulation as the exponentiation tree leaves. The verifier evaluates this
/// directly in `point.len()` multiplications, so the table needs no commitment.
pub fn eval_power_table_mle<F, E>(point: &[E]) -> E
where
	F: BinaryField,
	E: FieldOps<Scalar = F> + From<F>,
{
	let one = E::one();
	iter::zip(iterate(F::MULTIPLICATIVE_GENERATOR, |g| g.square()), point)
		.map(|(power, coord)| coord.clone() * (E::from(power) - one.clone()) + one.clone())
		.fold(E::one(), |acc, factor| acc * factor)
}
