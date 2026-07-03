// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{BinaryField, Field, field::FieldOps};
use binius_ip::{
	channel::IPVerifierChannel,
	prodcheck::{self, MultilinearEvalClaim},
	sumcheck::{BatchSumcheckOutput, batch_verify},
};
use binius_math::{
	multilinear::{
		eq::{eq_ind, eq_ind_zero},
		evaluate::evaluate_inplace_scalars,
	},
	univariate::evaluate_univariate,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;

use super::{
	common::{
		IntMulOutput, Phase1Output, Phase2Output, Phase3Output, Phase4Output, frobenius_twist,
		reconstruct_selecteds,
	},
	error::Error,
};

/// Verify Phase 1: GKR step on the exponentiation product tree.
///
/// Runs prodcheck verification to reduce the root claim on $\widetilde{Q}$ to $2^k$ leaf
/// evaluation claims, then verifies the leaf evaluations against the prover's claimed values.
fn verify_phase_1<F, C>(
	log_bits: usize,
	initial_eval_point: &[C::Elem],
	initial_b_eval: C::Elem,
	channel: &mut C,
) -> Result<Phase1Output<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	let n_vars = initial_eval_point.len();

	// Run prodcheck verification
	let claim = MultilinearEvalClaim {
		eval: initial_b_eval,
		point: initial_eval_point.to_vec(),
	};
	let output_claim = prodcheck::verify(log_bits, claim, channel)?;

	// Split output point: first n are x-point, last k are z-challenges
	let (eval_point, z_suffix) = output_claim.point.split_at(n_vars);

	// Read 2^k leaf evaluations from channel
	let b_leaves_evals = channel.recv_many(1 << log_bits)?;

	// Verify: output_claim.eval = multilinear_eval(b_leaves_evals, z_suffix)
	// The leaf evals form a multilinear over log_bits variables; evaluate at z_suffix
	let expected_eval = evaluate_inplace_scalars(b_leaves_evals.clone(), z_suffix);

	channel.assert_zero(expected_eval - output_claim.eval)?;

	Ok(Phase1Output {
		eval_point: eval_point.to_vec(),
		b_leaves_evals,
	})
}

/// Verify Phase 3: batched Frobenius selector sumcheck and LO * HI product sumcheck.
///
/// Batches two sumchecks: (a) the Frobenius-twisted selector sumcheck reducing the Phase 2
/// claims to exponent evaluations on $\widetilde{b}$ and a selector on $\widetilde{P}$, and
/// (b) the product claim $\widetilde{\textsf{LO}} \cdot \widetilde{\textsf{HI}}$.
fn verify_phase_3<F, C>(
	log_bits: usize,
	twisted_eval_points: Vec<Vec<C::Elem>>,
	twisted_evals: Vec<C::Elem>,
	c_eval_point: &[C::Elem],
	c_eval: C::Elem,
	channel: &mut C,
) -> Result<Phase3Output<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	let n_vars = c_eval_point.len();

	assert_eq!(twisted_eval_points.len(), 1 << log_bits);

	for twisted_eval_point in &twisted_eval_points {
		assert_eq!(twisted_eval_point.len(), c_eval_point.len());
	}

	// Batch the 2^k Frobenius-twisted leaf claims with eq_k(γ, i): sample γ in K^k and take the
	// multilinear evaluation of the 2^k claims at γ, matching the prover's CombineClaimsDecorator.
	// The aggregate and the LO·HI claim are then combined into the same sumcheck by the univariate
	// batch coefficient. γ is sampled before the sumcheck so its round polynomials are fixed
	// against it.
	//
	// The two batched terms (each degree 3) are:
	// - the 2^k aggregate: Σ_i eq_k(γ, i) * (b(i, X) * (A(X) - 1) + 1) * eq(φ⁻ⁱ(x) ; X)
	// - LO(X) * HI(X) * eq(c_eval_point ; X)
	let gamma = channel.sample_many(log_bits);
	let selector_agg_eval = evaluate_inplace_scalars(twisted_evals, &gamma);
	let evals = [selector_agg_eval, c_eval];

	let BatchSumcheckOutput {
		batch_coeff,
		mut challenges,
		eval,
	} = batch_verify(n_vars, 3, &evals, channel)?;
	challenges.reverse();

	// b(i, r) for i in 0..2^k
	let b_evals = channel.recv_many(1 << log_bits)?;

	// A(r)
	let gpow_a_eval = channel.recv_one()?;

	// C_lo(r), C_hi(r)
	let [gpow_c_lo_eval, gpow_c_hi_eval] = channel.recv_array::<2>()?;

	// Recombine the 2^k per-bit exponent claims b(i, r) into a single claim b(r_I^b, r) by
	// sampling a recombination point r_I^b in K^k. This carries one exponent claim (rather than
	// 2^k) into Phases 4 and 5.
	let r_ib = channel.sample_many(log_bits);
	let b_recomb = evaluate_inplace_scalars(b_evals.clone(), &r_ib);

	let eval_point = challenges;

	let expected_selected_terms = iter::zip(twisted_eval_points, &b_evals)
		.map(|(twisted_eval_point, b_eval)| {
			let one = C::Elem::one();
			(b_eval.clone() * (gpow_a_eval.clone() - one.clone()) + one)
				* eq_ind(&twisted_eval_point, &eval_point)
		})
		.collect::<Vec<_>>();
	// Combine the 2^k selector terms with eq_k(γ, i) — the multilinear evaluation at γ — mirroring
	// the prover's CombineClaimsDecorator.
	let expected_selected_agg = evaluate_inplace_scalars(expected_selected_terms, &gamma);

	// - c_lo(r) * c_hi(r) * eq(c_eval_point ; r)
	let expected_c_prod_eval =
		gpow_c_lo_eval.clone() * gpow_c_hi_eval.clone() * eq_ind(c_eval_point, &eval_point);

	let expected_terms = [expected_selected_agg, expected_c_prod_eval];
	let expected_batched_eval = evaluate_univariate(&expected_terms, batch_coeff);

	channel.assert_zero(expected_batched_eval - eval)?;

	Ok(Phase3Output {
		eval_point,
		r_ib,
		b_recomb,
		gpow_a_eval,
		gpow_c_lo_eval,
		gpow_c_hi_eval,
	})
}

/// Verify Phase 4: all but last layer of the GKR product trees for $\widetilde{a}$,
/// $\widetilde{c}_{\textsf{lo}}$, and $\widetilde{c}_{\textsf{hi}}$.
///
/// Reduces the three tree roots (claimed at the Phase-3 point) to the all-but-last (depth
/// `log_bits - 1`) layer claim via a single batched prodcheck over the three trees, batched with
/// $\lceil \log_2 3 \rceil = 2$ selector variables. Rather than binding the all-but-last-layer
/// evaluations here, the reduced claim (content-and-node point, reduced selector coordinates, and
/// combined evaluation) is handed straight to Phase 5.
fn verify_phase_4<F, C>(
	log_bits: usize,
	eval_point: &[C::Elem],
	a_root_eval: C::Elem,
	gpow_c_lo_eval: C::Elem,
	gpow_c_hi_eval: C::Elem,
	channel: &mut C,
) -> Result<Phase4Output<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	assert!(log_bits >= 1);

	let n_vars = eval_point.len();
	let n_layers = log_bits - 1;

	let log_n_trees = log2_ceil_usize(3); // = 2 selector variables

	// Sample the selector challenges that batch the three trees (padded to 4).
	let selector = channel.sample_many(log_n_trees);

	// Combined initial claim: eq(selector)-weighted sum of the three root evals (padded to 4 with a
	// zero), at the point selector ++ eval_point (the Phase-3 content point).
	let root_evals = vec![a_root_eval, gpow_c_lo_eval, gpow_c_hi_eval, C::Elem::zero()];
	let combined_root_eval = evaluate_inplace_scalars(root_evals, &selector);

	let claim = MultilinearEvalClaim {
		eval: combined_root_eval,
		point: [selector, eval_point.to_vec()].concat(),
	};

	// Run the batched prodcheck verification (n_layers = log_bits - 1 reduction layers).
	let output_claim = prodcheck::verify(n_layers, claim, channel)?;

	// The reduced point is [selector (log_n_trees), suffix (n_vars), bit_index (n_layers)]:
	//  - selector: the batching coordinates for the three trees, reduced through the selector
	//    rounds,
	//  - suffix ++ bit_index: the content-and-node point at which each tree's all-but-last-layer
	//    multilinear is now claimed.
	assert_eq!(output_claim.point.len(), log_n_trees + n_layers + n_vars);
	let (selector_point, a_c_eval_point) = output_claim.point.split_at(log_n_trees);

	Ok(Phase4Output {
		eval_point: a_c_eval_point.to_vec(),
		selector: selector_point.to_vec(),
		combined_eval: output_claim.eval,
	})
}

/// Verify Phase 5: final GKR layer, $\widetilde{b}$ rerandomization, and parity zerocheck.
///
/// First receives one reduced eval per tree ($\widetilde{a}$, $\widetilde{c}_{\textsf{lo}}$,
/// $\widetilde{c}_{\textsf{hi}}$) and checks they recombine, weighted by
/// $\textsf{eq}(\text{selector})$, to the Phase-4 prodcheck output claim. Then batches five
/// sumchecks: the final (widest) GKR layer of each of the three trees (each a regular
/// prodcheck-layer bivariate product MLE-check seeded by its reduced eval), the parity zerocheck
/// $a_0 \cdot b_0 = c_{\textsf{lo},0}$, and a single-claim rerandomization of the recombined
/// $\widetilde{b}(r_I^b, \cdot)$ exponent claim. The latter two span only the content variables and
/// are padded up to the trees' variable count.
#[allow(clippy::too_many_arguments)]
fn verify_phase_5<F, C>(
	log_bits: usize,
	a_c_eval_point: &[C::Elem],
	selector: &[C::Elem],
	combined_eval: C::Elem,
	b_eval_point: &[C::Elem],
	r_ib: &[C::Elem],
	b_recomb: C::Elem,
	channel: &mut C,
) -> Result<IntMulOutput<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
	C::Elem: FieldOps<Scalar = F> + From<F>,
{
	assert!(log_bits >= 1);
	let n_vars = b_eval_point.len();
	let n_extra = log_bits - 1;
	assert_eq!(a_c_eval_point.len(), n_vars + n_extra);

	// Receive the three per-tree reduced claims and check they recombine, weighted by eq(selector)
	// (padded to four), to the batched prodcheck output claim.
	let [a_eval, c_lo_eval, c_hi_eval] = channel.recv_array::<3>()?;
	let per_tree = vec![
		a_eval.clone(),
		c_lo_eval.clone(),
		c_hi_eval.clone(),
		C::Elem::zero(),
	];
	let combined = evaluate_inplace_scalars(per_tree, selector);
	channel.assert_zero(combined - combined_eval)?;

	// The batched sumcheck's five per-prover sum claims: the three tree final-layer sumchecks
	// (seeded by the reduced evals), the overflow parity zerocheck (0), and the single recombined
	// b exponent eval for the rerandomization.
	let evals = [a_eval, c_lo_eval, c_hi_eval, C::Elem::zero(), b_recomb];

	let BatchSumcheckOutput {
		batch_coeff,
		mut challenges,
		eval,
	} = batch_verify(n_vars + n_extra, 3, &evals, channel)?;
	challenges.reverse();

	// challenges = [r_content (n_vars), r_bit (n_extra)]: r_content is the shared output point;
	// r_bit collapses the node dimension (and is the padding point for the overflow / b checks).
	let (r_content, r_bit) = challenges.split_at(n_vars);

	// The prover sends the raw per-bit evaluations at r_content; the verifier reconstructs the leaf
	// selectors forward (rather than receiving selectors and inverting them).
	let a_evals = channel.recv_many(1 << log_bits)?;
	let c_lo_evals = channel.recv_many(1 << log_bits)?;
	let c_hi_evals = channel.recv_many(1 << log_bits)?;
	let b_evals = channel.recv_many(1 << log_bits)?;

	let [a_selected_evals, c_lo_selected_evals, c_hi_selected_evals] =
		reconstruct_selecteds(log_bits, &a_evals, &c_lo_evals, &c_hi_evals);

	// eq(a_c_eval_point ; challenges) is the MLE-check equality factor shared by the three
	// final-layer sumchecks.
	let a_c_eq_eval = eq_ind(a_c_eval_point, &challenges);

	// Each tree's final-layer product reduces the all-but-last claim to the two halves of its leaf
	// layer (split on the highest node bit). The verifier recombines the selected leaf evals into
	// each half via eq(r_bit): half_lo = Σ_{z<half} eq(r_bit, z) · selected[z] and half_hi over
	// selected[z + half]. The bivariate product is eq_ac · half_lo · half_hi.
	let tree_product = |selecteds: &[C::Elem]| {
		let half = selecteds.len() / 2;
		let lo = evaluate_inplace_scalars(selecteds[..half].to_vec(), r_bit);
		let hi = evaluate_inplace_scalars(selecteds[half..].to_vec(), r_bit);
		a_c_eq_eval.clone() * lo * hi
	};
	let expected_a = tree_product(&a_selected_evals);
	let expected_c_lo = tree_product(&c_lo_selected_evals);
	let expected_c_hi = tree_product(&c_hi_selected_evals);

	// The overflow and b checks span only the content variables, so their padded contribution
	// carries the factor eq(0^{n_extra} ; r_bit).
	let eq_pad = eq_ind_zero(r_bit);

	let b_eq_eval = eq_ind(b_eval_point, r_content);

	// We must check that `a_0 * b_0 = c_lo_0` across all rows, where these represent the least
	// significant bits of `a_exponents`, `b_exponents`, and `c_lo_exponents` respectively.
	// This check is performed in GF(2) (interpreting bits as field elements 0 and 1).
	//
	// Purpose: This prevents an attack when `a*b = 0` (due to `a=0` or `b=0`). A malicious
	// prover could set `c = 2^128 - 1`, which satisfies `a*b ≡ c (mod 2^128-1)` since
	// `0 ≡ 2^128-1 (mod 2^128-1)`. However, we need `a*b = c (mod 2^128)` where `0 ≠ 2^128-1`.
	// This check catches the attack: if `c = 2^128-1` then `c_lo_0 = 1` (since 2^128-1 is odd),
	// but `a_0 * b_0 = 0` when `a=0` or `b=0`, so the check `a_0 * b_0 = c_lo_0` fails.
	//
	// Implementation: A zerocheck on `a_0 * b_0 - c_lo_0 = 0`, reusing the Phase-2 constraint point
	// `b_eval_point` (r_2) as the zerocheck challenge — available for free because the `b`
	// re-randomization already evaluates at r_2.
	let expected_overflow_eval =
		eq_pad.clone() * &b_eq_eval * (a_evals[0].clone() * &b_evals[0] - &c_lo_evals[0]);

	// Bind the prover's raw per-bit evals to the single recombined rerandomization claim:
	// b(r_I^b, r_content) = sum_i eq(r_I^b, i) * b(i, r_content).
	let b_at_rx = evaluate_inplace_scalars(b_evals.clone(), r_ib);
	let expected_b_rerand_eval = eq_pad * &b_eq_eval * &b_at_rx;

	let expected_unbatched_evals = [
		expected_a,
		expected_c_lo,
		expected_c_hi,
		expected_overflow_eval,
		expected_b_rerand_eval,
	];
	let expected_batched_eval = evaluate_univariate(&expected_unbatched_evals, batch_coeff);

	// Compare expected evaluation against given evaluation `eval`.
	channel.assert_zero(expected_batched_eval - eval)?;

	Ok(IntMulOutput {
		eval_point: r_content.to_vec(),
		a_evals,
		b_evals,
		c_lo_evals,
		c_hi_evals,
	})
}

/// Verify the integer multiplication check (IntMul) protocol.
///
/// The IntMul protocol is a reduction that checks a relation on four virtual multilinear
/// polynomials: $\widetilde{a}, \widetilde{b}, \widetilde{c}_{\textsf{lo}},
/// \widetilde{c}_{\textsf{hi}}$. These multilinear polynomials are over $\mathbb{F}_2$ and have
/// $k + n$ variables. We write $a, b, c_{\textsf{lo}}, c_{\textsf{hi}} \in \mathbb{F}_2^{n \times
/// k}$ for their boolean hypercube evaluations. Let $\textsf{int}(M) \in \mathbb{N}^n$ map one of
/// the four matrices, $M$, to a vector of their interpretations as a $k$-bit unsigned integer. That
/// is, it embeds the $\mathbb{F}_2$ elements into $\mathbb{N}$ and multiplies by $(2^0, 2^1,
/// \ldots, 2^{k-1})$.
///
/// ## Protocol
///
/// The IntMul protocol reduces this relation to claims on the partial multilinear evaluations of
/// $\widetilde{a}, \widetilde{b}, \widetilde{c}_{\textsf{lo}}, \widetilde{c}_{\textsf{hi}}$ at a
/// common $n$-coordinate random evaluation point.
///
/// ### Exponentiation identity
///
/// The core technique reduces integer multiplication to field arithmetic via exponentiation. Let
/// $g$ be a generator of the multiplicative group of $\mathbb{F}_{2^{2k}}$, which has order
/// $2^{2k} - 1$. Then $\textsf{int}(a) \cdot \textsf{int}(b) = \textsf{int}(c_{\textsf{hi}})
/// \cdot 2^k + \textsf{int}(c_{\textsf{lo}})$ over the integers is equivalent to
///
/// $$\widetilde{Q}(x) = \widetilde{\textsf{LO}}(x) \cdot \widetilde{\textsf{HI}}(x) \quad
/// \forall x \in \{0, 1\}^n$$
///
/// where $\widetilde{Q}$ is obtained by exponentiating $g^{\widetilde{a}}$ by $\widetilde{b}$,
/// $\widetilde{\textsf{LO}} = g^{\widetilde{c}_{\textsf{lo}}}$, and $\widetilde{\textsf{HI}} =
/// (g^{2^k})^{\widetilde{c}_{\textsf{hi}}}$.
///
/// There is a wraparound edge case: when $a \cdot b = 0$, a malicious prover could set
/// $c_{\textsf{hi}} \| c_{\textsf{lo}} = 2^{2k} - 1$, which satisfies the exponentiation
/// identity modulo $2^{2k} - 1$ but not over the integers. A parity check on the least
/// significant bits ($a_0 \cdot b_0 = c_{\textsf{lo},0}$) rules this out.
///
/// ### Phases
///
/// - **Phase 1 — GKR step on $\widetilde{Q}$:** The verifier samples a random evaluation point $r$
///   and the prover sends the claimed evaluation $s = \widetilde{Q}(r)$. The parties run a GKR step
///   ($k$-layer balanced binary tree of bivariate products) reducing $s$ to $2^k$ leaf claims
///   $s'_{Q,i} = \widetilde{Q_i}(r')$.
///
/// - **Phase 2 — Frobenius step:** The verifier applies $\varphi^{-i}$ (inverse Frobenius) to each
///   leaf claim, reducing degree-$2^i$ expressions to degree-1. This is a local verifier
///   computation with no interaction.
///
/// - **Phase 3 — Batched Frobenius sumcheck + $\widetilde{\textsf{LO}} \cdot
///   \widetilde{\textsf{HI}}$ product sumcheck:** Two sumchecks batched into one: (a) The
///   Frobenius-twisted selector sumcheck on the $\widetilde{Q_i}$ claims, reducing to claims on
///   $\widetilde{b}$ exponent evaluations and the base $\widetilde{P}$ (i.e. $g^{\widetilde{a}}$).
///   (b) The deferred product claim $s = \sum \textsf{eq}(r, x) \cdot \widetilde{\textsf{LO}}(x)
///   \cdot \widetilde{\textsf{HI}}(x)$. This yields root claims on $\widetilde{P}$ (the
///   $\widetilde{a}$ selector), $\widetilde{\textsf{LO}}$, $\widetilde{\textsf{HI}}$, plus $2^k$
///   exponent claims on $\widetilde{b}$. The verifier then samples a recombination point $r_I^b \in
///   K^k$ and collapses the $2^k$ exponent claims into a single claim $\widetilde{b}(r_I^b, r) =
///   \sum_i \textsf{eq}(r_I^b, i) \cdot \widetilde{b}(i, r)$, carried into Phases 4 and 5.
///
/// - **Phase 4 — GKR on $\widetilde{a}$, $\widetilde{c}_{\textsf{lo}}$,
///   $\widetilde{c}_{\textsf{hi}}$ (all but last layer):** Batched GKR layers for the three
///   remaining exponentiation product trees. Each layer is a batched bivariate product sumcheck.
///   Since the bases ($g$ and $g^{2^k}$) are fixed, the Frobenius steps can be skipped — the
///   verifier locally reduces "scaled" evaluations to plain exponent evaluations.
///
/// - **Phase 5 — Final GKR layer + $\widetilde{b}$ rerandomization + parity check:** The final
///   (widest) GKR layer for $\widetilde{a}$, $\widetilde{c}_{\textsf{lo}}$,
///   $\widetilde{c}_{\textsf{hi}}$ is batched with: (a) A single-claim rerandomization sumcheck on
///   the recombined $\widetilde{b}(r_I^b, \cdot)$ exponent claim from Phase 3, bringing it to the
///   same evaluation point as $\widetilde{a}$ and $\widetilde{c}$. The prover sends the $2^k$ raw
///   per-bit evals $\widetilde{b}(i, r_x)$, which the verifier binds via $\sum_i \textsf{eq}(r_I^b,
///   i) \cdot \widetilde{b}(i, r_x) = \widetilde{b}(r_I^b, r_x)$. (b) A zerocheck verifying $a_0
///   \cdot b_0 = c_{\textsf{lo},0}$ (least significant bits), ruling out the wraparound edge case.
///
/// ### Output
///
/// The protocol outputs evaluation claims on $\widetilde{a}_i$, $\widetilde{b}_i$,
/// $\widetilde{c}_{\textsf{lo},i}$, $\widetilde{c}_{\textsf{hi},i}$ (for $i \in \{0, \ldots,
/// 2^k - 1\}$) at a common $n$-dimensional evaluation point. These are passed to the shift
/// reduction.
///
/// ### Parameters
///
/// - `log_bits`: $k$, where $2^k$ is the bit-width of the integer operands.
/// - `n_vars`: Number of variables in the row dimension (i.e., $\log_2$ of the number of
///   multiplication constraints).
pub fn verify<F, C>(
	log_bits: usize,
	n_vars: usize,
	channel: &mut C,
) -> Result<IntMulOutput<C::Elem>, Error>
where
	F: BinaryField,
	C: IPVerifierChannel<F>,
	C::Elem: FieldOps<Scalar = F> + From<F>,
{
	assert!(log_bits >= 1);
	assert!((1 << (log_bits + 1)) <= F::N_BITS);

	let initial_eval_point = channel.sample_many(n_vars);

	// Read the evaluation of the multilinear extension of the powers of the generator.
	let exp_eval = channel.recv_one()?;

	// Phase 1
	let Phase1Output {
		eval_point: phase_1_eval_point,
		b_leaves_evals,
	} = verify_phase_1(log_bits, &initial_eval_point, exp_eval.clone(), channel)?;

	assert_eq!(phase_1_eval_point.len(), n_vars);
	assert_eq!(b_leaves_evals.len(), 1 << log_bits);

	// Phase 2
	let Phase2Output {
		twisted_eval_points,
		twisted_evals,
	} = frobenius_twist(log_bits, &phase_1_eval_point, &b_leaves_evals);

	// Phase 3
	let Phase3Output {
		eval_point: phase_3_eval_point,
		r_ib,
		b_recomb,
		gpow_a_eval,
		gpow_c_lo_eval,
		gpow_c_hi_eval,
	} = verify_phase_3(
		log_bits,
		twisted_eval_points,
		twisted_evals,
		&initial_eval_point,
		exp_eval,
		channel,
	)?;

	// Phase 4
	let Phase4Output {
		eval_point: phase_4_eval_point,
		selector,
		combined_eval,
	} = verify_phase_4(
		log_bits,
		&phase_3_eval_point,
		gpow_a_eval,
		gpow_c_lo_eval,
		gpow_c_hi_eval,
		channel,
	)?;

	// Phase 5
	verify_phase_5(
		log_bits,
		&phase_4_eval_point,
		&selector,
		combined_eval,
		&phase_3_eval_point,
		&r_ib,
		b_recomb,
		channel,
	)
}
