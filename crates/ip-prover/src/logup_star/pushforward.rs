// Copyright 2026 The Binius Developers

//! The pushforward reduction that closes the table side.
//!
//! The table-side fractional-addition GKR runs all the way to its leaf, which claims the
//! pushforward `Y` at a point `z`. Its denominator half is the public `D = J - c`, so the verifier
//! checks that one itself. That leaves two claims that both read `Y`:
//!
//! - the GKR leaf claim `Y(z)`,
//! - the product claim `<T, Y> = e`.
//!
//! One batched `m`-variable sumcheck over the shared column pair `[Y, T]` reduces both to a single
//! evaluation point, giving one evaluation of `Y` and one of `T`.
//!
//! This is the prover mirror of the verifier's pushforward reduction in [`binius_ip::logup_star`].

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_math::FieldSlice;

use crate::{
	channel::IPProverChannel,
	sumcheck::{
		batch::batch_prove,
		bivariate_product_evaluator::BivariateProductEvaluator,
		mle_store::MleStore,
		multilinear_eval::MultilinearEvalEvaluator,
		round_evaluator::{SharedMleCheckProver, SumcheckRoundEvaluator},
	},
};

/// The evaluation claims that the pushforward reduction ends in.
pub struct PushforwardOutput<F> {
	/// The `m`-coordinate point shared by the table and pushforward evaluation claims.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluation of the table multilinear `T` at the point.
	pub table_eval_claim: F,
	/// The claimed evaluation of the pushforward multilinear `Y` at the point.
	pub pushforward_eval_claim: F,
}

/// Reduce the pushforward's two claims and the table's product claim to one evaluation point.
///
/// Two sum claims are batched over the `m`-variable cube:
///
/// ```text
///     S_1 = sum_x eq(x; z) * Y(x) = Y(z)      the GKR leaf claim, re-randomized
///     S_2 = sum_x (Y * T)(x)      = e         the product claim
/// ```
///
/// `S_1` carries the equality factor `eq(x; z)`, so it starts as an eq-weighted MLE-check
/// evaluator; folding that factor into its round polynomials turns it into a plain sumcheck that
/// batches with `S_2`, which carries no `eq` factor. Both read the same `Y` column, so the shared
/// store holds just `[Y, T]` and the reduction emits one evaluation of each.
///
/// Both summands are degree 2 per variable — `eq` times a multilinear, and a product of two
/// multilinears — so the batched degree is 2.
///
/// # Arguments
///
/// * `alloc` - The allocator the sumcheck store draws its folded columns from.
/// * `table` - The table `T` over the `m`-variable cube.
/// * `pushforward` - The pushforward `Y` over the same cube.
/// * `eval_claim` - The product claim `e = <T, Y>`.
/// * `pushforward_eval_claim` - The GKR leaf claim `Y(z)`.
/// * `pushforward_eval_point` - The leaf point `z`, of length `m`.
/// * `channel` - The prover channel.
///
/// Both multilinears are borrowed: the store's first fold contracts them into half-size buffers of
/// its own, so neither is copied at full width.
///
/// # Preconditions
///
/// * `table.log_len() == pushforward.log_len() == pushforward_eval_point.len()`
#[tracing::instrument(skip_all, level = "debug", name = "logup* pushforward reduction")]
pub fn prove_pushforward<'a, A, F, P>(
	alloc: &'a A,
	table: FieldSlice<'a, P>,
	pushforward: FieldSlice<'a, P>,
	eval_claim: F,
	pushforward_eval_claim: F,
	pushforward_eval_point: &[F],
	channel: &mut impl IPProverChannel<F>,
) -> PushforwardOutput<F>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	let m = pushforward_eval_point.len();
	assert_eq!(table.log_len(), m); // precondition
	assert_eq!(pushforward.log_len(), m); // precondition

	// S_1: the leaf claim as an eq-weighted evaluation of Y at z. The store owns both columns as
	// borrows, so its first fold is the only place either is written, at half width.
	let mut store = MleStore::new(m, alloc);
	let y_col = store.push(pushforward);
	let evaluator = MultilinearEvalEvaluator::new(y_col);
	let mle_prover = SharedMleCheckProver::new(
		store,
		[(pushforward_eval_claim, evaluator)],
		pushforward_eval_point.to_vec(),
	);

	// Folding the eq factor into S_1's round polynomials turns it into a plain sumcheck, which is
	// what lets the eq-free product claim join it in one evaluator group over the shared store.
	let mut prover = mle_prover.into_shared_sumcheck();

	// S_2: the product claim over the shared Y column and the pushed table column.
	let t_col = prover.store_mut().push(table);
	let product = BivariateProductEvaluator::new([y_col, t_col]);
	prover
		.add_evaluator(eval_claim, Box::new(product) as Box<dyn SumcheckRoundEvaluator<F, P> + 'a>);

	// Drive the one shared-store sumcheck. The flattened round-polynomial order is
	// [pushforward_eval, product], matching the verifier's batched claim order.
	let output = batch_prove(vec![prover], channel);

	// The shared prover emits each store column's evaluation once, in push order [Y, T].
	let [pushforward_eval_claim, table_eval_claim]: [F; 2] = output.multilinear_evals[0]
		.as_slice()
		.try_into()
		.expect("the pushforward store has two columns");
	channel.send_many(&[pushforward_eval_claim, table_eval_claim]);

	// `batch_prove` returns binding-order challenges; reverse to variable-indexed (low-to-high).
	let mut table_eval_point = output.challenges;
	table_eval_point.reverse();

	PushforwardOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
	}
}
