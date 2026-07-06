// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_field::{BinaryField, BinaryField1b, ExtensionField, Field, field::FieldOps};
use binius_iop::{channel::IOPVerifierChannel, logup_star};
use binius_ip::{
	channel::IPVerifierChannel,
	logup_star::LookerClaim,
	prodcheck::{self, MultilinearEvalClaim},
	sumcheck::{BatchSumcheckOutput, batch_verify},
};
use binius_math::{
	multilinear::{eq::eq_ind, evaluate::evaluate_inplace_scalars},
	univariate::evaluate_univariate,
};
use binius_utils::checked_arithmetics::log2_ceil_usize;

use super::{
	common::{
		IntMulOutput, LIMB_BITS, LOG_N_LIMBS, N_LIMB_COLUMNS, N_LIMBS, Phase1Output, Phase2Output,
		Phase3Output, Phase4Output, eval_power_table_mle, frobenius_twist, limb_column_twists,
		twist_limb_claim,
	},
	error::Error,
};
use crate::config::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS};

/// Verify Phase 1: GKR step on the exponentiation product tree.
///
/// Runs prodcheck verification to reduce the root claim on $\widetilde{Q}$ to $2^k$ leaf
/// evaluation claims, then verifies the leaf evaluations against the prover's claimed values.
fn verify_phase_1<F, C>(
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
	let output_claim = prodcheck::verify(LOG_WORD_SIZE_BITS, claim, channel)?;

	// Split output point: first n are x-point, last k are z-challenges
	let (eval_point, z_suffix) = output_claim.point.split_at(n_vars);

	// Read 2^k leaf evaluations from channel
	let b_leaves_evals = channel.recv_many(WORD_SIZE_BITS)?;

	// Verify: output_claim.eval = multilinear_eval(b_leaves_evals, z_suffix)
	// The leaf evals form a multilinear over LOG_WORD_SIZE_BITS variables; evaluate at z_suffix
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

	assert_eq!(twisted_eval_points.len(), WORD_SIZE_BITS);

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
	let gamma = channel.sample_many(LOG_WORD_SIZE_BITS);
	let selector_agg_eval = evaluate_inplace_scalars(twisted_evals, &gamma);
	let evals = [selector_agg_eval, c_eval];

	let BatchSumcheckOutput {
		batch_coeff,
		mut challenges,
		eval,
	} = batch_verify(n_vars, 3, &evals, channel)?;
	challenges.reverse();

	// b(i, r) for i in 0..2^k
	let b_evals = channel.recv_many(WORD_SIZE_BITS)?;

	// A(r)
	let gpow_a_eval = channel.recv_one()?;

	// C_lo(r), C_hi(r)
	let [gpow_c_lo_eval, gpow_c_hi_eval] = channel.recv_array::<2>()?;

	// Recombine the 2^k per-bit exponent claims b(i, r) into a single claim b(r_I^b, r) by
	// sampling a recombination point r_I^b in K^k. This carries one exponent claim (rather than
	// 2^k) into Phases 4 and 5.
	let r_ib = channel.sample_many(LOG_WORD_SIZE_BITS);
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

/// Verify Phase 4: the GKR product trees for $\widetilde{a}$, $\widetilde{c}_{\textsf{lo}}$, and
/// $\widetilde{c}_{\textsf{hi}}$, reduced to per-limb evaluation claims.
///
/// Each tree's leaf layer is the concatenation of its `N_LIMBS` limb columns. A single batched
/// prodcheck over the three trees (batched with $\lceil \log_2 3 \rceil = 2$ selector variables)
/// runs the all-but-last reduction layer; the final (widest) layer of each tree is then proven by
/// a batched bivariate-product sumcheck seeded by the three per-tree reduced claims. The prover
/// sends the per-limb evaluations at the shared content point, which the verifier binds to the
/// final-layer sumcheck by recombining each tree's two leaf halves via `eq(r_limb)`.
fn verify_phase_4<F, C>(
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
	let n_vars = eval_point.len();

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

	// Run the batched prodcheck verification over all LOG_N_LIMBS layers.
	let output_claim = prodcheck::verify(LOG_N_LIMBS, claim, channel)?;

	// The reduced point is [selector (log_n_trees), r_content (n_vars), r_limb (LOG_N_LIMBS)]:
	//  - selector: the batching coordinates for the three trees, reduced through the selector
	//    rounds,
	//  - r_content: the shared point at which the limb columns are claimed,
	//  - r_limb: the limb-index coordinates of each tree's leaf layer.
	assert_eq!(output_claim.point.len(), log_n_trees + n_vars + LOG_N_LIMBS);
	let (selector_point, rest) = output_claim.point.split_at(log_n_trees);
	let (r_content, r_limb) = rest.split_at(n_vars);

	// The prover sends the per-limb evaluations at r_content for each tree. Each tree's leaf
	// multilinear stacks its limb columns over the limb coordinates, so its evaluation is the
	// eq(r_limb)-fold of the per-limb evals; the batched output claim is the eq(selector)-weighted
	// combination of the tree evaluations (the padding tree slot is zero).
	let a_limb_evals = channel.recv_many(N_LIMBS)?;
	let c_lo_limb_evals = channel.recv_many(N_LIMBS)?;
	let c_hi_limb_evals = channel.recv_many(N_LIMBS)?;

	let tree_eval = |limb_evals: &[C::Elem]| evaluate_inplace_scalars(limb_evals.to_vec(), r_limb);
	let per_tree = vec![
		tree_eval(&a_limb_evals),
		tree_eval(&c_lo_limb_evals),
		tree_eval(&c_hi_limb_evals),
		C::Elem::zero(),
	];
	let combined = evaluate_inplace_scalars(per_tree, selector_point);
	channel.assert_zero(combined - output_claim.eval)?;

	Ok(Phase4Output {
		eval_point: r_content.to_vec(),
		a_limb_evals,
		c_lo_limb_evals,
		c_hi_limb_evals,
	})
}

/// Verify Phase 5: the logup* fixed-base exponentiation lookup, $\widetilde{b}$ rerandomization,
/// and parity zerocheck.
///
/// The per-limb claims from Phase 4 are Frobenius-twisted onto the shared table of generator
/// powers `i ↦ g^i`, batched to a single stacked lookup claim, and read from the table via a
/// committed logup* reduction. The reduced table claim is checked against the table's succinct
/// MLE; the pushforward claim is opened through the channel inside the reduction;
/// the reduced index claims are carried into a final batched sumcheck — together with the
/// $\widetilde{b}(r_I^b, \cdot)$ rerandomization and the parity zerocheck $a_0 \cdot b_0 =
/// c_{\textsf{lo},0}$ — that brings every output claim to one shared point.
fn verify_phase_5<F, C>(
	phase_4_output: &Phase4Output<C::Elem>,
	b_eval_point: &[C::Elem],
	r_ib: &[C::Elem],
	b_recomb: C::Elem,
	channel: &mut C,
) -> Result<IntMulOutput<C::Elem>, Error>
where
	F: BinaryField,
	C: IOPVerifierChannel<F>,
	C::Elem: FieldOps<Scalar = F> + From<F>,
{
	let n_vars = b_eval_point.len();
	assert_eq!(phase_4_output.eval_point.len(), n_vars);

	// Twist the per-limb claims onto the shared table: column (t, l) is the Frobenius power
	// φ^{twist} of the looked-up column U_{t,l}(x) = T[e_{t,l}(x)], so its claim becomes a claim on
	// U_{t,l} at the twisted point.
	let twists = limb_column_twists();
	let limb_evals = [
		&phase_4_output.a_limb_evals,
		&phase_4_output.c_lo_limb_evals,
		&phase_4_output.c_hi_limb_evals,
	];
	let twisted_claims = iter::zip(&twists, limb_evals.into_iter().flatten())
		.map(|(&twist, eval)| twist_limb_claim(twist, &phase_4_output.eval_point, eval.clone()))
		.collect::<Vec<_>>();

	// Read the N_LIMB_COLUMNS looked-up columns from the shared table via the committed multi-
	// looker logup* reduction. The pushforward oracle is received inside; its opening relation is
	// returned to the caller. The reduction returns one index claim per column, all at the shared
	// content point.
	let looker_claims = twisted_claims
		.iter()
		.map(|(twisted_point, twisted_eval)| LookerClaim {
			eval_point: twisted_point,
			eval_claim: twisted_eval.clone(),
		})
		.collect::<Vec<_>>();
	let log_cols = log2_ceil_usize(N_LIMB_COLUMNS);
	let logup_proof = logup_star::verify::<F, C>(LIMB_BITS, &looker_claims, channel)?;

	// The table is succinct: the verifier evaluates its MLE directly.
	let expected_table_eval = eval_power_table_mle::<F, C::Elem>(&logup_proof.table_eval_point);
	channel.assert_zero(expected_table_eval - logup_proof.table_eval_claim)?;

	// The reduction hands back the per-column embedded-index claims directly, all at the shared
	// content point (padding columns read row 0, whose embedding is zero).
	let index_content_point = logup_proof.index_eval_point.as_slice();
	let mut padded_column_evals = logup_proof.index_eval_claims.clone();
	padded_column_evals.resize(1 << log_cols, C::Elem::zero());

	// Collapse the per-column claims into a single claim on the eq(ρ)-folded column V by sampling
	// ρ, so the final unification runs over the content variables only.
	let rho = channel.sample_many(log_cols);
	let folded_index_claim = evaluate_inplace_scalars(padded_column_evals, &rho);

	// Final unification: one batched sumcheck brings the folded index claim, the recombined b
	// exponent claim, and the parity zerocheck to a shared output point.
	let evals = [folded_index_claim, C::Elem::zero(), b_recomb];
	let BatchSumcheckOutput {
		batch_coeff,
		mut challenges,
		eval,
	} = batch_verify(n_vars, 3, &evals, channel)?;
	challenges.reverse();
	let r_out = challenges.as_slice();

	// The prover sends the raw per-bit evaluations at r_out.
	let a_evals = channel.recv_many(WORD_SIZE_BITS)?;
	let c_lo_evals = channel.recv_many(WORD_SIZE_BITS)?;
	let c_hi_evals = channel.recv_many(WORD_SIZE_BITS)?;
	let b_evals = channel.recv_many(WORD_SIZE_BITS)?;

	// Bind the per-bit evals to the folded index claim. The index entries are the GF(2)-linear
	// embeddings iota(e_{t,l}) = Σ_u basis(u) · bit_u(e_{t,l}), and bit u of limb l is bit
	// (LIMB_BITS·l + u) of the word, so each column's MLE is a fixed basis-weighted combination of
	// the word's per-bit column MLEs.
	let per_word_evals = [&a_evals, &c_lo_evals, &c_hi_evals];
	let mut column_evals = (0..N_LIMB_COLUMNS)
		.map(|j| {
			let (tree, limb) = (j / N_LIMBS, j % N_LIMBS);
			(0..LIMB_BITS)
				.map(|u| {
					let basis = <F as ExtensionField<BinaryField1b>>::basis(u);
					per_word_evals[tree][limb * LIMB_BITS + u].clone() * C::Elem::from(basis)
				})
				.fold(C::Elem::zero(), |acc, term| acc + term)
		})
		.collect::<Vec<_>>();
	column_evals.resize(1 << log_cols, C::Elem::zero());
	let folded_index_eval = evaluate_inplace_scalars(column_evals, &rho);
	let expected_index_eval = eq_ind(index_content_point, r_out) * folded_index_eval;

	let b_eq_eval = eq_ind(b_eval_point, r_out);

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
		b_eq_eval.clone() * (a_evals[0].clone() * &b_evals[0] - &c_lo_evals[0]);

	// Bind the prover's raw per-bit evals to the single recombined rerandomization claim:
	// b(r_I^b, r_out) = sum_i eq(r_I^b, i) * b(i, r_out).
	let b_at_rx = evaluate_inplace_scalars(b_evals.clone(), r_ib);
	let expected_b_rerand_eval = b_eq_eval * &b_at_rx;

	let expected_unbatched_evals = [
		expected_index_eval,
		expected_overflow_eval,
		expected_b_rerand_eval,
	];
	let expected_batched_eval = evaluate_univariate(&expected_unbatched_evals, batch_coeff);

	// Compare expected evaluation against given evaluation `eval`.
	channel.assert_zero(expected_batched_eval - eval)?;

	Ok(IntMulOutput {
		eval_point: r_out.to_vec(),
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
///   $\widetilde{c}_{\textsf{hi}}$, down to per-limb claims:** Each exponent word splits into
///   `N_LIMBS` limbs, so each of the three constant-base exponentiations factors as a product of
///   `N_LIMBS` limb columns. A batched GKR product check over the three (depth-`LOG_N_LIMBS`) trees
///   reduces the root claims to one evaluation claim per limb column at a shared content point.
///
/// - **Phase 5 — logup* lookup + $\widetilde{b}$ rerandomization + parity check:** Limb column $(t,
///   l)$ is the Frobenius power $\varphi^{ws}$ of the looked-up column $T[e_{t,l}(\cdot)]$ over the
///   shared table $T\colon i \mapsto g^i$ of $2^w$ generator powers (where $w$ is the limb bit
///   width). The per-limb claims are Frobenius-twisted onto $T$, batched to one stacked lookup
///   claim, and reduced via committed logup* ([Soukhanov25]): the pushforward oracle is committed
///   mid-protocol and its opening relation returned to the caller; the table claim is checked
///   against the table's succinct product-of-selects MLE; the index claim is bound to the per-bit
///   output evals in a final batched sumcheck, together with (a) a single-claim rerandomization of
///   the recombined $\widetilde{b}(r_I^b, \cdot)$ exponent claim from Phase 3 and (b) a zerocheck
///   verifying $a_0 \cdot b_0 = c_{\textsf{lo},0}$ (least significant bits), ruling out the
///   wraparound edge case.
///
/// [Soukhanov25]: <https://eprint.iacr.org/2025/946>
///
/// ### Output
///
/// The protocol outputs evaluation claims on $\widetilde{a}_i$, $\widetilde{b}_i$,
/// $\widetilde{c}_{\textsf{lo},i}$, $\widetilde{c}_{\textsf{hi},i}$ (for $i \in \{0, \ldots,
/// 2^k - 1\}$) at a common $n$-dimensional evaluation point. The claims are passed to the shift
/// reduction; the logup* pushforward commitment is opened through the channel inside phase 5.
///
/// ### Parameters
///
/// - `n_vars`: Number of variables in the row dimension (i.e., $\log_2$ of the number of
///   multiplication constraints).
///
/// The integer operands are fixed at the `WORD_SIZE_BITS` bit width.
pub fn verify<F, C>(n_vars: usize, channel: &mut C) -> Result<IntMulOutput<C::Elem>, Error>
where
	F: BinaryField,
	C: IOPVerifierChannel<F>,
	C::Elem: FieldOps<Scalar = F> + From<F>,
{
	assert!(2 * WORD_SIZE_BITS <= F::N_BITS);

	let initial_eval_point = channel.sample_many(n_vars);

	// Read the evaluation of the multilinear extension of the powers of the generator.
	let exp_eval = channel.recv_one()?;

	// Phase 1
	let Phase1Output {
		eval_point: phase_1_eval_point,
		b_leaves_evals,
	} = verify_phase_1(&initial_eval_point, exp_eval.clone(), channel)?;

	assert_eq!(phase_1_eval_point.len(), n_vars);
	assert_eq!(b_leaves_evals.len(), WORD_SIZE_BITS);

	// Phase 2
	let Phase2Output {
		twisted_eval_points,
		twisted_evals,
	} = frobenius_twist(LOG_WORD_SIZE_BITS, &phase_1_eval_point, &b_leaves_evals);

	// Phase 3
	let Phase3Output {
		eval_point: phase_3_eval_point,
		r_ib,
		b_recomb,
		gpow_a_eval,
		gpow_c_lo_eval,
		gpow_c_hi_eval,
	} = verify_phase_3(twisted_eval_points, twisted_evals, &initial_eval_point, exp_eval, channel)?;

	// Phase 4
	let phase_4_output =
		verify_phase_4(&phase_3_eval_point, gpow_a_eval, gpow_c_lo_eval, gpow_c_hi_eval, channel)?;

	// Phase 5
	verify_phase_5(&phase_4_output, &phase_3_eval_point, &r_ib, b_recomb, channel)
}
