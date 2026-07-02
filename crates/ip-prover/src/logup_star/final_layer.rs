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
use either::Either;

use super::error::LogupStarError;
use crate::{
	channel::IPProverChannel,
	fracaddcheck::{FracAddCheckProver, FracEvalClaim},
	sumcheck::{
		MleToSumCheckDecorator, batch::batch_prove,
		bivariate_product::BivariateProductSumcheckProver,
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
/// The claims play two different roles:
///
/// - `S_1` and `S_2` are the layer-1 numerator and denominator of the fractional-addition circuit.
/// - They carry the equality factor `eq(x'; Z)`, so they run as a decorated MLE-check prover.
/// - `S_3a` and `S_3b` split the product claim `<T, Y> = e` on the highest variable.
/// - They carry no `eq` factor; each is a plain [`BivariateProductSumcheckProver`].
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
pub fn prove_final_layer<F, P>(
	eval_claim: F,
	table_prover: FracAddCheckProver<P>,
	layer1: FracEvalClaim<F>,
	pushforward: &FieldBuffer<P>,
	table: &FieldBuffer<P>,
	channel: &mut impl IPProverChannel<F>,
) -> Result<FinalLayerOutput<F>, LogupStarError>
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
	// The eq factor stays implicit in the MLE-check prover.
	// The decorator reinstates it each round as an (X - alpha) multiplier, a regular sumcheck.
	let (frac_prover, remaining) = table_prover.layer_prover(layer1)?;
	debug_assert!(remaining.is_none(), "the final layer consumes the last table-side layer");
	let frac_prover = MleToSumCheckDecorator::new(frac_prover);

	// The product check <T, Y> = e is split on the highest variable into two leaf products.
	//
	//     half 0: highest variable fixed to 0
	//     half 1: highest variable fixed to 1
	let [y_0, y_1] = split_halves(pushforward);
	let [t_0, t_1] = split_halves(table);

	// e_0 is the first half's product sum.
	// Only e_0 is sent, since the verifier recovers e_1 = e - e_0.
	//
	//     e_0 = sum_{x'} (Y_0 * T_0)(x'),   e_1 = e - e_0 = sum_{x'} (Y_1 * T_1)(x')
	let e_0 = inner_product_par(&y_0, &t_0);
	channel.send_one(e_0);
	let e_1 = eval_claim - e_0;

	// S_3a, S_3b: each half is a bivariate product, held as [Y_half, T_half].
	// So each prover's two final evaluations are exactly (Y_half, T_half) at the challenge point.
	let product_0 = BivariateProductSumcheckProver::new([y_0, t_0], e_0)?;
	let product_1 = BivariateProductSumcheckProver::new([y_1, t_1], e_1)?;

	// Batch the three provers into one sumcheck.
	//
	// The flattened round-polynomial order is [num_1, den_1, e_0, e_1].
	// That matches the verifier's batched order [layer1_num, layer1_den, e_0, e_1].
	let output = batch_prove(
		vec![
			Either::Left(frac_prover),
			Either::Right(product_0),
			Either::Right(product_1),
		],
		channel,
	)?;

	// Each product prover holds [Y_half, T_half]; read both halves' evaluations, in verifier order.
	// The fractional prover's Y evaluations agree at the same point, so they need not be sent.
	let [y_0_eval, t_0_eval] = output.multilinear_evals[1]
		.as_slice()
		.try_into()
		.expect("a bivariate product prover evaluates two multilinears");
	let [y_1_eval, t_1_eval] = output.multilinear_evals[2]
		.as_slice()
		.try_into()
		.expect("a bivariate product prover evaluates two multilinears");
	channel.send_many(&[y_0_eval, y_1_eval, t_0_eval, t_1_eval]);

	// Fold the highest variable once to collapse each pair of halves into one evaluation.
	let r = channel.sample();
	let pushforward_eval_claim = extrapolate_line(y_0_eval, y_1_eval, r);
	let table_eval_claim = extrapolate_line(t_0_eval, t_1_eval, r);

	// batch_prove returns challenges low-to-high; the folded variable is the highest coordinate.
	let mut table_eval_point = output.challenges;
	table_eval_point.push(r);

	Ok(FinalLayerOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
	})
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
