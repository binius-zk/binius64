// Copyright 2026 The Binius Developers

//! The pushforward reduction that closes the table side.
//!
//! Each table's fractional-addition GKR runs all the way to its leaf, which claims that table's
//! pushforward `Y_t` at a point `z_t`. Its denominator half is the public `D = J - c_t`, so the
//! verifier checks that one itself. That leaves two claims per table that both read `Y_t`:
//!
//! - the GKR leaf claim `Y_t(z_t)`,
//! - the product claim `<T_t, Y_t> = e_t`.
//!
//! One batched sumcheck reduces every one of them to a single evaluation point, giving one
//! evaluation of `Y_t` and one of `T_t` per table.
//!
//! This is the prover mirror of the verifier's pushforward reduction in [`binius_ip::logup_star`].

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_math::FieldSlice;

use crate::{
	channel::IPProverChannel,
	sumcheck::{
		PaddedSumcheckDecorator,
		batch::batch_prove,
		bivariate_product_evaluator::BivariateProductEvaluator,
		mle_store::MleStore,
		multilinear_eval::MultilinearEvalEvaluator,
		round_evaluator::{SharedMleCheckProver, SumcheckRoundEvaluator},
	},
};

/// One table's witnesses and claims entering the pushforward reduction.
pub struct TableWitness<'a, P: PackedField> {
	/// The table `T_t` over its `m_t`-variable cube.
	pub table: FieldSlice<'a, P>,
	/// The pushforward `Y_t` over the same cube.
	pub pushforward: FieldSlice<'a, P>,
	/// The product claim `e_t = <T_t, Y_t>`.
	pub eval_claim: P::Scalar,
	/// The fractional-addition leaf claim `Y_t(z_t)`.
	pub pushforward_eval_claim: P::Scalar,
	/// The leaf point `z_t`, of length `m_t`.
	pub pushforward_eval_point: &'a [P::Scalar],
}

/// The evaluation claims that the pushforward reduction ends in.
pub struct PushforwardOutput<F> {
	/// The point spanning the widest table, at which every table's claims are taken.
	///
	/// Table `t`, over `m_t` variables, is claimed at the **first `m_t`** coordinates: the batch
	/// pads each table at its high coordinates.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluations of the tables `T_t`, each at its own prefix of the point.
	pub table_eval_claims: Vec<F>,
	/// The claimed evaluations of the pushforwards `Y_t`, each at the same prefix.
	pub pushforward_eval_claims: Vec<F>,
}

/// Reduce every table's pushforward and product claims to one evaluation point.
///
/// Two sum claims are batched per table, over that table's `m_t`-variable cube:
///
/// ```text
///     S_1 = sum_x eq(x; z_t) * Y_t(x) = Y_t(z_t)      the GKR leaf claim, re-randomized
///     S_2 = sum_x (Y_t * T_t)(x)      = e_t           the product claim
/// ```
///
/// `S_1` carries the equality factor `eq(x; z_t)`, so it starts as an eq-weighted MLE-check
/// evaluator; folding that factor into its round polynomials turns it into a plain sumcheck that
/// batches with `S_2`, which carries no `eq` factor. Both read the same `Y_t` column, so a table's
/// store holds just `[Y_t, T_t]` and the reduction emits one evaluation of each.
///
/// Both summands are degree 2 per variable — `eq` times a multilinear, and a product of two
/// multilinears — so the batched degree is 2.
///
/// Tables need not agree on a size: a table shallower than the deepest one is wrapped in a
/// [`PaddedSumcheckDecorator`], which raises it to the shared round count by multiplying its claims
/// by an equality factor over the extra variables. Those are the highest-indexed ones and so are
/// bound first, which puts a table's own challenges at the low coordinates of the shared point.
///
/// # Arguments
///
/// * `alloc` - The allocator the sumcheck stores draw their folded columns from.
/// * `tables` - The per-table witnesses and claims; their cubes may differ in size.
/// * `channel` - The prover channel.
///
/// Every multilinear is borrowed: a store's first fold contracts its columns into half-size buffers
/// of its own, so nothing is copied at full width.
///
/// # Preconditions
///
/// * `tables` is non-empty.
/// * For each table, `table.log_len() == pushforward.log_len() == pushforward_eval_point.len()`.
#[tracing::instrument(skip_all, level = "debug", name = "logup* pushforward reduction")]
pub fn prove_pushforward<'a, A, F, P>(
	alloc: &'a A,
	tables: impl IntoIterator<Item = TableWitness<'a, P>>,
	channel: &mut impl IPProverChannel<F>,
) -> PushforwardOutput<F>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	// The witnesses are walked twice — once for the deepest table, once to build the provers — so
	// the iterator is materialized once here.
	let tables = tables.into_iter().collect::<Vec<_>>();
	assert!(!tables.is_empty(), "at least one table is required"); // precondition

	let max_m = tables
		.iter()
		.map(|table| table.pushforward_eval_point.len())
		.max()
		.expect("tables is non-empty");

	let provers = tables
		.into_iter()
		.map(|table| {
			let m = table.pushforward_eval_point.len();
			assert_eq!(table.table.log_len(), m); // precondition
			assert_eq!(table.pushforward.log_len(), m); // precondition

			// S_1: the leaf claim as an eq-weighted evaluation of Y_t at z_t. The store owns both
			// columns as borrows, so its first fold is the only place either is written, at half
			// width.
			let mut store = MleStore::new(m, alloc);
			let y_col = store.push(table.pushforward);
			let evaluator = MultilinearEvalEvaluator::new(y_col);
			let mle_prover = SharedMleCheckProver::new(
				store,
				[(table.pushforward_eval_claim, evaluator)],
				table.pushforward_eval_point.to_vec(),
			);

			// Folding the eq factor into S_1's round polynomials turns it into a plain sumcheck,
			// which is what lets the eq-free product claim join it in one evaluator group over the
			// shared store.
			let mut prover = mle_prover.into_shared_sumcheck();

			// S_2: the product claim over the shared Y_t column and the pushed table column.
			let t_col = prover.store_mut().push(table.table);
			let product = BivariateProductEvaluator::new([y_col, t_col]);
			prover.add_evaluator(
				table.eval_claim,
				Box::new(product) as Box<dyn SumcheckRoundEvaluator<F, P> + 'a>,
			);

			// Raise this table to the batch's round count. The claims are in the same order as the
			// evaluators, which is the order the verifier batches them in.
			PaddedSumcheckDecorator::new(
				prover,
				max_m - m,
				vec![table.pushforward_eval_claim, table.eval_claim],
			)
		})
		.collect::<Vec<_>>();

	// Drive the one batched sumcheck. The flattened round-polynomial order is
	// [pushforward_eval, product] per table, in table order, matching the verifier's claim order.
	let output = batch_prove(provers, channel);

	// Each table's shared prover emits its store columns' evaluations once, in push order
	// [Y_t, T_t]. Send them in the same flat order the verifier reads.
	let (pushforward_eval_claims, table_eval_claims): (Vec<_>, Vec<_>) = output
		.multilinear_evals
		.iter()
		.map(|evals| {
			let [pushforward_eval_claim, table_eval_claim]: [F; 2] = evals
				.as_slice()
				.try_into()
				.expect("a pushforward store has two columns");
			(pushforward_eval_claim, table_eval_claim)
		})
		.unzip();
	output.send_evals(channel);

	// `batch_prove` returns binding-order challenges; reverse to variable-indexed (low-to-high).
	let mut table_eval_point = output.challenges;
	table_eval_point.reverse();

	PushforwardOutput {
		table_eval_point,
		table_eval_claims,
		pushforward_eval_claims,
	}
}
