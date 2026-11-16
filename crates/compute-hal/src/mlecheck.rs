// Copyright 2026 The Binius Developers

//! A drop-in [`MleCheckProver`] on [`Compute`]: the port of `SharedMleCheckProver`.
//!
//! The real prover owns an `MleStore` of columns and one evaluator per claim. Here the columns are
//! [`Column`]s on a backend and the evaluators are [`PairKernel`]s that share one pass:
//! [`quadratic`] ports `QuadraticMleEvaluator` and [`multilinear_eval`] ports
//! `MultilinearEvalEvaluator`. Compose them statically, e.g. `(quadratic(..), vec_of_evals)`.
//!
//! Each round is one [`Compute::fold_pair_map_reduce`] and one [`Compute::exec`]. Like the real
//! prover, [`MleCheckProver::fold`] only records the challenge; the next `execute` (or `finish`)
//! applies it inside that fused op. The eq weights are the backend's business (`eq_point`).

use std::iter;

use binius_field::BinaryField;
use binius_ip::sumcheck::RoundCoeffs;
use binius_ip_prover::sumcheck::{common::MleCheckProver, round_state::RoundState};

use crate::{
	Column, Compute, Elem, PairKernel,
	sumcheck::{RoundKernel, read_folded, round_coeffs},
};

/// Feeds columns `cols` of the pass (which has `n_in` columns) to `inner`, so claims over different
/// columns of one store can share a pass.
pub struct Pick<K> {
	n_in: usize,
	cols: Vec<usize>,
	inner: K,
}

impl<F, K: PairKernel<F>> PairKernel<F> for Pick<K> {
	fn n_in(&self) -> usize {
		self.n_in
	}
	fn n_out(&self) -> usize {
		self.inner.n_out()
	}
	fn scratch(&self) -> usize {
		2 * self.cols.len() + self.inner.scratch()
	}

	fn map<T: Elem<F>>(&self, lo: &[T], hi: &[T], scratch: &mut [T], out: &mut [T]) {
		let (l, rest) = scratch.split_at_mut(self.cols.len());
		let (h, rest) = rest.split_at_mut(self.cols.len());
		for (i, &c) in self.cols.iter().enumerate() {
			(l[i], h[i]) = (lo[c].clone(), hi[c].clone());
		}
		self.inner.map(l, h, rest, out);
	}
}

/// `R(1)` of the identity composition: the high half itself.
pub struct HiKernel;

impl<F> PairKernel<F> for HiKernel {
	fn n_in(&self) -> usize {
		1
	}
	fn n_out(&self) -> usize {
		1
	}
	fn scratch(&self) -> usize {
		0
	}
	fn map<T: Elem<F>>(&self, _: &[T], hi: &[T], _: &mut [T], out: &mut [T]) {
		out[0] = hi[0].clone();
	}
}

/// `QuadraticMleEvaluator`: `comp` over columns `cols` of an `n_in`-column pass. It emits
/// `C::DEGREE` slots (`R(1), R(inf), ..`), so the claim's degree is `C::DEGREE`.
pub const fn quadratic<'c, C>(
	comp: &'c C,
	n_in: usize,
	cols: Vec<usize>,
) -> Pick<RoundKernel<'c, C>> {
	Pick {
		n_in,
		cols,
		inner: RoundKernel(comp),
	}
}

/// `MultilinearEvalEvaluator`: the evaluation claim of column `col` of an `n_in`-column pass. It
/// emits one slot, so the claim's degree is 1.
pub fn multilinear_eval(n_in: usize, col: usize) -> Pick<HiKernel> {
	Pick {
		n_in,
		cols: vec![col],
		inner: HiKernel,
	}
}

/// The prime round polynomial from the claim and the claim's slots: `RoundEvals::interpolate_eq`
/// for degrees 1 and 2, and the general interpolation above.
fn interpolate_eq<F: BinaryField>(claim: F, alpha: F, slots: &[F]) -> RoundCoeffs<F> {
	let y0 = (claim - slots[0] * alpha) * (F::ONE - alpha).invert_or_zero();
	match slots {
		[y1] => RoundCoeffs(vec![y0, *y1 - y0]),
		_ => round_coeffs(y0, slots),
	}
}

/// An MLE-check prover for several claims at one point, over the columns of one backend.
///
/// `kernel` produces every claim's slots, concatenated; claim `i` owns the next `degrees[i]` of
/// them. Columns bind the highest variable first, like the verifier expects.
pub struct ComputeMleCheckProver<'a, F: BinaryField, C: Compute<F>, K> {
	compute: &'a C,
	cols: Vec<Column<C::FieldVec, C::FieldSlice<'a>>>,
	kernel: K,
	/// Per claim: its claim, or the round polynomial awaiting `fold`.
	states: Vec<RoundState<RoundCoeffs<F>, F>>,
	degrees: Vec<usize>,
	point: Vec<F>,
	n_vars: usize,
	/// A fold the columns have not seen: the next round's fused op applies it.
	challenge: Option<F>,
}

impl<'a, F: BinaryField, C: Compute<F>, K: PairKernel<F>> ComputeMleCheckProver<'a, F, C, K> {
	/// `claims` is each claim's value and degree (slot count), in kernel output order. Every
	/// column must have `point.len()` variables.
	pub fn new(
		compute: &'a C,
		cols: Vec<Column<C::FieldVec, C::FieldSlice<'a>>>,
		kernel: K,
		claims: Vec<(F, usize)>,
		point: Vec<F>,
	) -> Self {
		assert_eq!(claims.iter().map(|c| c.1).sum::<usize>(), kernel.n_out());
		let (claims, degrees) = claims.into_iter().unzip::<_, _, Vec<_>, _>();
		Self {
			compute,
			cols,
			kernel,
			states: claims.into_iter().map(RoundState::Claim).collect(),
			degrees,
			n_vars: point.len(),
			point,
			challenge: None,
		}
	}
}

impl<F: BinaryField, C: Compute<F>, K: PairKernel<F>> MleCheckProver<F>
	for ComputeMleCheckProver<'_, F, C, K>
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let n = self.n_vars;
		assert!(n > 0);
		// The highest remaining coordinate is this round's `alpha`: left out of the eq weights.
		let alpha = self.point[n - 1];
		let round = self.compute.fold_pair_map_reduce(
			&mut self.cols,
			self.challenge.take(),
			Some(&self.point[..n - 1]),
			&self.kernel,
		);
		let mut slots = self.compute.exec(round).into_iter();
		iter::zip(&mut self.states, &self.degrees)
			.map(|(state, &degree)| {
				let slots: Vec<F> = slots.by_ref().take(degree).collect();
				let coeffs = interpolate_eq(*state.claim(), alpha, &slots);
				*state = RoundState::Coeffs(coeffs.clone());
				coeffs
			})
			.collect()
	}

	fn fold(&mut self, challenge: F) {
		assert!(self.challenge.is_none(), "fold called twice without an intervening execute");
		for state in &mut self.states {
			*state = RoundState::Claim(state.coeffs().evaluate(&challenge));
		}
		self.challenge = Some(challenge);
		self.n_vars -= 1;
	}

	fn finish(mut self) -> Vec<F> {
		let r = self.challenge.expect("finish called before any fold");
		read_folded(self.compute, &mut self.cols, r)
	}

	fn eval_point(&self) -> &[F] {
		&self.point[..self.n_vars]
	}
}
