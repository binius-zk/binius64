// Copyright 2026 The Binius Developers

//! A lazy compute backend for the Binius64 prover (design B: future combinators).
//!
//! Every [`Compute`] op returns a description of work (a [`Compute::Future`]) instead of doing it.
//! The host composes descriptions with [`Compute::join`] and [`Compute::then`], and a single
//! [`Compute::exec`] runs the whole pipeline: the one synchronization point.
//!
//! Use this crate to write prover control flow (sumcheck, Fiat-Shamir) once, generic over a
//! backend that may keep its data in a separate memory space.
//!
//! Key types:
//! - [`Compute`]: the backend trait.
//! - [`Composition`] / [`PairKernel`]: the per-element math handed to a backend.
//! - [`ReferenceCompute`], [`SimdCompute`], [`FakeDeviceCompute`]: the backends.
//! - [`sumcheck::prove`]: the shared mini-prover.
//!
//! Related crates: `binius-math` (multilinear helpers), `binius-ip` (the sumcheck verifier).

mod fake;
mod reference;
mod simd;
pub mod sumcheck;
#[cfg(test)]
mod tests;

use std::ops::Mul;

use binius_core::word::Word;
use binius_field::{BinaryField, FieldOps};
pub use fake::{FakeDeviceCompute, Stats};
pub use reference::ReferenceCompute;
pub use simd::SimdCompute;

/// A multivariate polynomial applied to the columns of a sumcheck, element by element.
pub trait Composition<F>: Sync {
	/// Number of input columns.
	const N_IN: usize;
	/// Total degree of the composition.
	const DEGREE: usize;

	/// Evaluates on one element per column. `T` is a scalar, a packed field, or a traced
	/// expression node.
	fn eval<T: FieldOps<Scalar = F> + Mul<F, Output = T>>(&self, x: &[T]) -> T;
}

/// The per-pair work of [`Compute::pair_map_reduce`].
pub trait PairKernel<F>: Sync {
	/// Number of input columns.
	const N_IN: usize;
	/// Number of outputs, each summed over all pairs.
	const N_OUT: usize;

	/// Maps the low and high half entries of one pair of every column to `N_OUT` values.
	fn map<T: FieldOps<Scalar = F> + Mul<F, Output = T>>(&self, lo: &[T], hi: &[T], out: &mut [T]);
}

/// The future type every backend in this crate uses: a boxed closure run by `exec`.
pub type Lazy<'a, T> = Box<dyn FnOnce() -> T + Send + 'a>;

/// A compute backend. Ops build futures; nothing runs until [`Compute::exec`].
///
/// The trait is generic over the field (not an associated type) so a backend can implement it for
/// several fields, e.g. a SIMD backend per packed type.
///
/// Column data is owned by value and threaded through futures (`fold_highest_var` and
/// `pair_map_reduce` take and return the columns). A `&mut` slice cannot be held by an unexecuted
/// future while a later future reads it, so ownership threading is the only way to express
/// fold-then-evaluate in one pipeline.
pub trait Compute<F: BinaryField>: Sync {
	/// An owned buffer of `2^log_len` field elements.
	type FieldVec: Send;
	/// A borrowed view of a field buffer.
	type FieldSlice<'a>: Copy + Send + 'a;
	/// An owned buffer of words.
	type WordVec: Send;
	/// A borrowed view of a word buffer.
	type WordSlice<'a>: Copy + Send + 'a;
	/// A description of work yielding `T`. Runs only in [`Compute::exec`].
	type Future<'a, T: Send + 'a>: Send + 'a;

	/// Uploads `values`, whose length is a power of two. Eager: uploads are not pipelined.
	fn copy_field_from_host(&self, values: &[F]) -> Self::FieldVec;
	/// Uploads `words`. Eager.
	fn copy_words_from_host(&self, words: &[Word]) -> Self::WordVec;
	/// Borrows a field buffer.
	fn as_slice<'a>(&self, v: &'a Self::FieldVec) -> Self::FieldSlice<'a>;
	/// Borrows a word buffer.
	fn as_word_slice<'a>(&self, v: &'a Self::WordVec) -> Self::WordSlice<'a>;

	/// Downloads columns to the host. The only way host code can observe field data.
	fn to_host<'a>(&self, cols: Vec<Self::FieldVec>) -> Self::Future<'a, Vec<Vec<F>>>;
	/// The equality-indicator tensor of `point` (`eq_ind_partial_eval`).
	fn tensor_expand<'a>(&self, point: &'a [F]) -> Self::Future<'a, Self::FieldVec>;
	/// Sum of pointwise products.
	fn inner_product<'a>(
		&self,
		a: Self::FieldSlice<'a>,
		b: Self::FieldSlice<'a>,
	) -> Self::Future<'a, F>
	where
		Self: 'a;
	/// Evaluates the multilinear `mle` at `point`.
	///
	/// A required op rather than `tensor_expand` + `inner_product`: the temporary tensor cannot be
	/// borrowed by a future that outlives the closure that built it.
	fn evaluate<'a>(&self, mle: Self::FieldSlice<'a>, point: &'a [F]) -> Self::Future<'a, F>;
	/// `out[i] = sum_b bit_b(words[i]) * weights[b]`, zero-padded to a power of two.
	fn fold_words<'a>(
		&self,
		words: Self::WordSlice<'a>,
		weights: &[F; 64],
	) -> Self::Future<'a, Self::FieldVec>
	where
		Self: 'a;
	/// Binds the highest variable of every column to `r`; each column loses one variable.
	fn fold_highest_var<'a>(
		&self,
		cols: Vec<Self::FieldVec>,
		r: F,
	) -> Self::Future<'a, Vec<Self::FieldVec>>;
	/// For every `i < 2^(n-1)` maps the pair `(col[i], col[i + 2^(n-1)])` of all columns through
	/// `kernel` and sums the outputs. Returns the columns and the `K::N_OUT` sums.
	fn pair_map_reduce<'a, K: PairKernel<F> + 'a>(
		&self,
		cols: Vec<Self::FieldVec>,
		kernel: &'a K,
	) -> Self::Future<'a, (Vec<Self::FieldVec>, Vec<F>)>;

	/// Runs both futures; a device may overlap them.
	fn join<'a, A: Send + 'a, B: Send + 'a>(
		&self,
		a: Self::Future<'a, A>,
		b: Self::Future<'a, B>,
	) -> Self::Future<'a, (A, B)>;
	/// Sequences `f` after `a`. `f` only builds the next future from the (opaque) result of `a`;
	/// it must not read device data, so on a device the whole chain stays in one submission.
	fn then<'a, A: Send + 'a, B: Send + 'a>(
		&self,
		a: Self::Future<'a, A>,
		f: impl FnOnce(A) -> Self::Future<'a, B> + Send + 'a,
	) -> Self::Future<'a, B>;
	/// Runs the pipeline and returns its result: the synchronization point.
	fn exec<'a, T: Send + 'a>(&self, f: Self::Future<'a, T>) -> T;
}

/// Scalar reference implementations shared by the backends.
pub(crate) mod scalar {
	use binius_field::Field;

	use super::*;

	/// Sums `kernel` over the pairs of the given columns.
	pub fn pair_map_reduce<F: Field, K: PairKernel<F>>(cols: &[&[F]], kernel: &K) -> Vec<F> {
		assert_eq!(cols.len(), K::N_IN);
		let half = cols[0].len() / 2;
		assert!(half > 0);
		(0..half).fold(vec![F::ZERO; K::N_OUT], |mut acc, i| {
			let lo: Vec<F> = cols.iter().map(|c| c[i]).collect();
			let hi: Vec<F> = cols.iter().map(|c| c[i + half]).collect();
			let mut out = vec![F::ZERO; K::N_OUT];
			kernel.map(&lo, &hi, &mut out);
			acc.iter_mut().zip(out).for_each(|(a, o)| *a += o);
			acc
		})
	}

	/// Binds the highest variable of `col` to `r`.
	pub fn fold_highest_var<F: Field>(col: &mut Vec<F>, r: F) {
		let half = col.len() / 2;
		for i in 0..half {
			let (lo, hi) = (col[i], col[i + half]);
			col[i] = lo + r * (hi - lo);
		}
		col.truncate(half);
	}

	/// See [`Compute::fold_words`].
	pub fn fold_words<F: Field>(words: &[Word], weights: &[F; 64]) -> Vec<F> {
		let mut out: Vec<F> = words
			.iter()
			.map(|w| {
				(0..64)
					.filter(|&b| w.extract_bit(b))
					.map(|b| weights[b])
					.sum()
			})
			.collect();
		out.resize(out.len().next_power_of_two(), F::ZERO);
		out
	}
}
