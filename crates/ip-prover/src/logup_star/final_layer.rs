// Copyright 2026 The Binius Developers

//! The batched final layer of the table side.
//!
//! It fuses two reductions that both end in an evaluation of the pushforward `Y`:
//!
//! - the last fractional-addition GKR layer of the table circuit,
//! - the product sumcheck `<T, Y> = e`.
//!
//! Both split their leaf multilinears on the highest variable.
//! So both can share one `(m-1)`-variable sumcheck and one line-fold over that variable.
//! That collapses the two `Y` evaluations into a single evaluation point.
//!
//! This is the prover mirror of the verifier's final layer in [`binius_ip::logup_star`].

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, inner_product::inner_product_par, line::extrapolate_line};

use crate::{
	channel::IPProverChannel,
	fracaddcheck::{FracAddCheckProver, FracEvalClaim},
	sumcheck::{
		batch::batch_prove, bivariate_product_evaluator::BivariateProductEvaluator,
		round_evaluator::RoundEvaluator,
	},
};

/// The evaluation claims that the batched final layer reduces to.
pub struct FinalLayerOutput<F> {
	/// The `m`-coordinate point shared by the table and pushforward evaluation claims.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluation of the table multilinear `T` at the point.
	pub table_eval_claim: F,
	/// The claimed evaluation of the pushforward multilinear `Y` at the point.
	pub pushforward_eval_claim: F,
}

/// Prove the batched final layer of the table side.
///
/// Four sum claims are batched over the `m-1`-variable cube, then the highest variable is folded:
///
/// ```text
///     S_1  = sum_{x'} eq(x'; Z) * (Y_0 * D_1 + Y_1 * D_0)(x') = num_1(Z)
///     S_2  = sum_{x'} eq(x'; Z) * (D_0 * D_1)(x')             = den_1(Z)
///     S_3a = sum_{x'} (Y_0 * T_0)(x')                         = e_0
///     S_3b = sum_{x'} (Y_1 * T_1)(x')                         = e_1
/// ```
///
/// The claims play two different roles, but all four read one shared column store:
///
/// - `S_1` and `S_2` are the layer-1 numerator and denominator of the fractional-addition circuit.
///   They carry the equality factor `eq(x'; Z)`, so they run as eq-weighted MLE-check evaluators.
/// - `S_3a` and `S_3b` split the product claim `<T, Y> = e` on the highest variable. They carry no
///   `eq` factor; each is a plain [`BivariateProductEvaluator`].
///
/// The fractional prover popped for the leaf layer already owns the four columns `[Y_0, Y_1, D_0,
/// D_1]`, and its numerator halves `Y_0, Y_1` are the pushforward halves the product claims read.
/// So only the table halves `T_0, T_1` are pushed as new columns, and converting the fractional
/// prover to a plain sumcheck (folding the `eq` factor into its round polynomials) lets all four
/// claims batch as one shared-store sumcheck over `[Y_0, Y_1, D_0, D_1, T_0, T_1]`.
///
/// The split obeys `e_0 + e_1 = e`, so only `e_0` is sent.
/// The verifier recovers `e_1 = e - e_0`.
/// One batching coefficient combines the four round polynomials.
/// The verifier reconstructs the same batch and recomputes the public denominator halves itself.
/// So this routine writes only `e_0` and the leaf-half evaluations `[Y_0, Y_1, T_0, T_1]`.
///
/// # Arguments
///
/// * `eval_claim` - The product claim `e = <T, Y>`.
/// * `table_prover` - The table-side fractional-addition prover, holding only its leaf layer.
/// * `layer1` - The layer-1 numerator and denominator claims, sharing the point `Z`.
/// * `pushforward` - The pushforward `Y` over the `m`-variable cube.
/// * `table` - The table `T` over the `m`-variable cube.
/// * `channel` - The prover channel.
#[tracing::instrument(skip_all, level = "debug", name = "logup* final layer")]
pub fn prove_final_layer<F, P>(
	eval_claim: F,
	table_prover: FracAddCheckProver<P>,
	layer1: FracEvalClaim<F>,
	pushforward: &FieldBuffer<P>,
	table: &FieldBuffer<P>,
	channel: &mut impl IPProverChannel<F>,
) -> FinalLayerOutput<F>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	// Both layer-1 claims share the point Z.
	debug_assert_eq!(layer1.0.point, layer1.1.point, "layer-1 claims must share the point");

	// S_1, S_2: the fractional-addition numerator/denominator, weighted by eq(.; Z).
	//
	//     leaf layer holds numerator Y and denominator D
	//     splitting both on the highest variable gives the mle-check over [Y_0, Y_1, D_0, D_1]
	//
	// `layer_prover` owns those four columns in the returned prover's store and hands back their
	// column ids; `y_0_col`/`y_1_col` are the pushforward halves the product check reuses. Each
	// prover retains exactly its final layer, so consume it popping that layer.
	debug_assert_eq!(table_prover.n_layers(), 1, "the final layer is the last table-side layer");
	let (remaining_prover, frac_prover, [y_0_col, y_1_col, _d_0_col, _d_1_col]) =
		table_prover.layer_prover(layer1);
	assert!(remaining_prover.is_none());

	// The eq factor is folded into the fractional prover's round polynomials, turning it into a
	// plain sumcheck over the shared store so the product claims — which carry no eq factor — join
	// it in one evaluator group.
	let mut prover = frac_prover.into_shared_sumcheck();

	// The product check <T, Y> = e is split on the highest variable into two leaf products.
	//
	//     half 0: highest variable fixed to 0
	//     half 1: highest variable fixed to 1
	//
	// Y_0, Y_1 are already store columns (the fractional numerator halves), so only the table
	// halves T_0, T_1 are pushed as new columns. Only Y_0 is needed here, to form e_0; e_1 follows
	// as e - e_0.
	let [y_0, _y_1] = split_halves(pushforward);
	let [t_0, t_1] = split_halves(table);

	// e_0 is the first half's product sum.
	// Only e_0 is sent, since the verifier recovers e_1 = e - e_0.
	//
	//     e_0 = sum_{x'} (Y_0 * T_0)(x'),   e_1 = e - e_0 = sum_{x'} (Y_1 * T_1)(x')
	let e_0 = inner_product_par(&y_0, &t_0);
	channel.send_one(e_0);
	let e_1 = eval_claim - e_0;

	// S_3a, S_3b: each half is a bivariate product over the shared Y column and the pushed T
	// column.
	let t_0_col = prover.push_owned_column(t_0);
	let t_1_col = prover.push_owned_column(t_1);
	let product_0 = BivariateProductEvaluator::new([y_0_col, t_0_col]);
	prover.add_evaluator(e_0, Box::new(product_0) as Box<dyn RoundEvaluator<F, P>>);
	let product_1 = BivariateProductEvaluator::new([y_1_col, t_1_col]);
	prover.add_evaluator(e_1, Box::new(product_1) as Box<dyn RoundEvaluator<F, P>>);

	// Drive the one shared-store sumcheck.
	//
	// The flattened round-polynomial order is [num_1, den_1, e_0, e_1].
	// That matches the verifier's batched order [layer1_num, layer1_den, e_0, e_1].
	let output = batch_prove(vec![prover], channel);

	// The shared prover emits each store column's evaluation once, in push order
	// [Y_0, Y_1, D_0, D_1, T_0, T_1]; the public denominator halves are not sent.
	let [y_0_eval, y_1_eval, _d_0_eval, _d_1_eval, t_0_eval, t_1_eval]: [F; 6] = output
		.multilinear_evals[0]
		.as_slice()
		.try_into()
		.expect("the final-layer shared store has six columns");
	channel.send_many(&[y_0_eval, y_1_eval, t_0_eval, t_1_eval]);

	// Fold the highest variable once to collapse each pair of halves into one evaluation.
	let r = channel.sample();
	let pushforward_eval_claim = extrapolate_line(y_0_eval, y_1_eval, r);
	let table_eval_claim = extrapolate_line(t_0_eval, t_1_eval, r);

	// batch_prove returns challenges low-to-high; the folded variable is the highest coordinate.
	let mut table_eval_point = output.challenges;
	table_eval_point.push(r);

	FinalLayerOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
	}
}

/// Split a multilinear on its highest variable into owned low and high halves.
///
/// The low half fixes the highest variable to 0, the high half to 1.
fn split_halves<P: PackedField>(buffer: &FieldBuffer<P>) -> [FieldBuffer<P>; 2] {
	// split_half_ref borrows the two halves; copy each into an owned buffer for the sub-provers.
	let (low, high) = buffer.split_half_ref();
	[
		FieldBuffer::new(low.log_len(), low.as_ref().into()),
		FieldBuffer::new(high.log_len(), high.as_ref().into()),
	]
}
