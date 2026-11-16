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
//! - [`Composition`] / [`PairKernel`] / [`Elem`]: the per-element math handed to a backend.
//! - [`ReferenceCompute`], [`SimdCompute`], [`FakeDeviceCompute`]: the backends.
//! - [`sumcheck::prove`]: the shared mini-prover.
//!
//! Related crates: `binius-math` (multilinear helpers), `binius-ip` (the sumcheck verifier).

mod fake;
pub mod mlecheck;
mod reference;
mod simd;
pub mod sumcheck;
#[cfg(test)]
mod tests;

use std::ops::{Add, Mul, Sub};

use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
pub use fake::{ExecStats, FakeDeviceCompute, Stats};
pub use reference::ReferenceCompute;
pub use simd::SimdCompute;

/// What a kernel needs of an element: a scalar, a packed field, or a traced expression node.
pub trait Elem<F>:
	Clone
	+ Default
	+ Send
	+ Sync
	+ Add<Output = Self>
	+ Sub<Output = Self>
	+ Mul<Output = Self>
	+ Mul<F, Output = Self>
{
}

// Every `Field` is a `PackedField` of width 1, so this covers scalars too.
impl<P: PackedField> Elem<P::Scalar> for P {}

/// A multivariate polynomial applied to the columns of a sumcheck, element by element.
pub trait Composition<F>: Sync {
	/// Number of input columns.
	const N_IN: usize;
	/// Degree of the composition along a line, i.e. in one folded variable.
	const DEGREE: usize;

	/// Evaluates on one element per column.
	fn eval<T: Elem<F>>(&self, x: &[T]) -> T;
	/// The homogeneous part of degree `DEGREE` (e.g. `a * b` for `a * b - c`): the leading
	/// coefficient of the composition along a line with direction `x`.
	fn eval_infinity<T: Elem<F>>(&self, x: &[T]) -> T;
}

/// The per-pair work of [`Compute::fold_pair_map_reduce`]. Every kernel sees all columns; kernels
/// that share a pass pick their columns by index.
pub trait PairKernel<F>: Sync {
	/// Number of input columns.
	fn n_in(&self) -> usize;
	/// Number of outputs, each summed over all pairs.
	fn n_out(&self) -> usize;
	/// Length of the `scratch` slice `map` needs. The backend allocates it once per worker.
	fn scratch(&self) -> usize;

	/// Maps the low and high half entries of one pair of every column to `n_out` values.
	fn map<T: Elem<F>>(&self, lo: &[T], hi: &[T], scratch: &mut [T], out: &mut [T]);
}

/// Two kernels in one pass; outputs are concatenated.
impl<F, A: PairKernel<F>, B: PairKernel<F>> PairKernel<F> for (A, B) {
	fn n_in(&self) -> usize {
		self.0.n_in()
	}
	fn n_out(&self) -> usize {
		self.0.n_out() + self.1.n_out()
	}
	fn scratch(&self) -> usize {
		self.0.scratch() + self.1.scratch()
	}
	fn map<T: Elem<F>>(&self, lo: &[T], hi: &[T], scratch: &mut [T], out: &mut [T]) {
		let (s0, s1) = scratch.split_at_mut(self.0.scratch());
		let (o0, o1) = out.split_at_mut(self.0.n_out());
		self.0.map(lo, hi, s0, o0);
		self.1.map(lo, hi, s1, o1);
	}
}

/// Any number of kernels of one type in one pass; outputs are concatenated.
impl<F, K: PairKernel<F>> PairKernel<F> for Vec<K> {
	fn n_in(&self) -> usize {
		// An empty `Vec` adds no kernels, so it constrains nothing.
		self.first().map_or(0, K::n_in)
	}
	fn n_out(&self) -> usize {
		self.iter().map(|k| k.n_out()).sum()
	}
	fn scratch(&self) -> usize {
		self.iter().map(|k| k.scratch()).sum()
	}
	fn map<T: Elem<F>>(&self, lo: &[T], hi: &[T], mut scratch: &mut [T], mut out: &mut [T]) {
		for k in self {
			assert_eq!(k.n_in(), self[0].n_in());
			let (s, rest_s) = scratch.split_at_mut(k.scratch());
			let (o, rest_o) = out.split_at_mut(k.n_out());
			k.map(lo, hi, s, o);
			(scratch, out) = (rest_s, rest_o);
		}
	}
}

/// A column of a sumcheck: owned by the host, or lent for the whole proof.
///
/// A lent column is read in place; its first fold writes a fresh half-size [`Column::Owned`] buffer
/// instead of copying the column up front (the real prover's `Column::Borrowed`).
pub enum Column<V, S> {
	/// A buffer the backend may overwrite.
	Owned(V),
	/// A read-only view of the caller's buffer.
	Borrowed(S),
}

impl<V, S> Column<V, S> {
	/// The owned buffer. Panics on a lent column, which no fold has replaced yet.
	pub fn owned(&self) -> &V {
		match self {
			Self::Owned(v) => v,
			Self::Borrowed(_) => panic!("column is still borrowed; fold it first"),
		}
	}
}

/// The future type every backend in this crate uses: a boxed closure run by `exec`.
pub type Lazy<'a, T> = Box<dyn FnOnce() -> T + Send + 'a>;

/// A compute backend. Ops build futures; nothing runs until [`Compute::exec`].
///
/// The trait is generic over the field (not an associated type) so a backend can implement it for
/// several fields, e.g. a SIMD backend per packed type.
///
/// Columns are owned by the host and borrowed by futures: read-only ops take a `FieldSlice<'a>`,
/// and ops that change columns take `&'a mut [Column<..>]`, so they can fold a lent column into a
/// fresh buffer.
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

	/// Downloads a column to the host. The only way host code can observe field data.
	fn to_host<'a>(&self, v: Self::FieldSlice<'a>) -> Self::Future<'a, Vec<F>>
	where
		Self: 'a;
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
	/// A required op rather than `tensor_expand` + `inner_product` via `then`: `inner_product`
	/// takes slices, and the tensor is owned by the `then` closure, so no slice of it outlives the
	/// closure.
	fn evaluate<'a>(&self, mle: Self::FieldSlice<'a>, point: &'a [F]) -> Self::Future<'a, F>;
	/// `out[i] = sum_b bit_b(words[i]) * weights[b]`, zero-padded to a power of two.
	fn fold_words<'a>(
		&self,
		words: Self::WordSlice<'a>,
		weights: &[F; 64],
	) -> Self::Future<'a, Self::FieldVec>
	where
		Self: 'a;
	/// Binds the highest variable of every column to `r`; each column loses one variable and is
	/// [`Column::Owned`] afterwards. Returns the columns, so a following `then` can read them.
	#[allow(clippy::type_complexity)] // the column slice type is spelled out in full
	fn fold_highest_var<'a, 's: 'a>(
		&self,
		cols: &'a mut [Column<Self::FieldVec, Self::FieldSlice<'s>>],
		r: F,
	) -> Self::Future<'a, &'a mut [Column<Self::FieldVec, Self::FieldSlice<'s>>]>
	where
		Self: 'a;
	/// One sumcheck round in one op: folds the columns by `r` (if `Some`), then, for every
	/// `i < 2^(n-1)`, maps the pair `(col[i], col[i + 2^(n-1)])` of all columns through `kernel`
	/// and sums the outputs. Returns the `kernel.n_out()` sums. A [`Column::Borrowed`] is read in
	/// place and replaced by its folded copy.
	///
	/// All columns must have the same `log_len` (else this panics); `n` is that length after the
	/// fold. With `eq_point = Some(p)` (`p.len() == n - 1`) each pair's outputs are weighted by
	/// `eq(p, i)`, where bit `k` of `i` pairs with `p[k]`. How eq is realized is up to the backend:
	/// SIMD never materializes the full tensor.
	fn fold_pair_map_reduce<'a, 's: 'a, K: PairKernel<F> + 'a>(
		&self,
		cols: &'a mut [Column<Self::FieldVec, Self::FieldSlice<'s>>],
		r: Option<F>,
		eq_point: Option<&'a [F]>,
		kernel: &'a K,
	) -> Self::Future<'a, Vec<F>>
	where
		Self: 'a;

	/// Runs both futures; a device may overlap them.
	fn join<'a, A: Send + 'a, B: Send + 'a>(
		&self,
		a: Self::Future<'a, A>,
		b: Self::Future<'a, B>,
	) -> Self::Future<'a, (A, B)>;
	/// Runs all futures; a device may overlap them.
	fn join_all<'a, T: Send + 'a>(
		&self,
		futures: Vec<Self::Future<'a, T>>,
	) -> Self::Future<'a, Vec<T>>;
	/// Monadic bind: runs `f` on the result of `a` and continues with the future it returns.
	///
	/// Over handle-valued futures (`FieldVec`s) this is free on a device: the backend enqueues `a`,
	/// calls `f` with the handles, and keeps enqueuing, so the chain is one submission. A bind
	/// whose closure consumes a device-computed scalar (e.g. the `F` of `inner_product`) forces a
	/// hidden device-to-host sync inside `exec`. `FakeDeviceCompute` counts those.
	///
	/// `f` is opaque until its input exists: the backend can enqueue back-to-back but cannot fuse
	/// across it. Fusion needs an explicit op, such as `fold_pair_map_reduce`.
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
	use binius_math::multilinear::eq::eq_ind_partial_eval_scalars;

	use super::*;

	/// Sums `kernel` over the pairs `range`, with `get(col, i)` giving the `(lo, hi)` entries. Each
	/// output is weighted by `eq[i]` if given. Buffers are allocated once per call.
	pub fn reduce_pairs<F: Field, T: Elem<F>, K: PairKernel<F>>(
		kernel: &K,
		range: std::ops::Range<usize>,
		get: impl Fn(usize, usize) -> (T, T),
		eq: Option<&[F]>,
	) -> Vec<T> {
		let (mut lo, mut hi) =
			(vec![T::default(); kernel.n_in()], vec![T::default(); kernel.n_in()]);
		let mut scratch = vec![T::default(); kernel.scratch()];
		let mut out = vec![T::default(); kernel.n_out()];
		let mut acc = vec![T::default(); kernel.n_out()];
		for i in range {
			for c in 0..kernel.n_in() {
				(lo[c], hi[c]) = get(c, i);
			}
			kernel.map(&lo, &hi, &mut scratch, &mut out);
			for (a, o) in acc.iter_mut().zip(&out) {
				*a = a.clone() + eq.map_or_else(|| o.clone(), |eq| o.clone() * eq[i]);
			}
		}
		acc
	}

	/// The `log_len` `n` of the pairs a round reduces after folding. Panics if the columns differ
	/// in length (see [`Compute::fold_pair_map_reduce`]).
	pub fn round_log_len(log_lens: &[usize], fold: bool) -> usize {
		let max = log_lens.iter().copied().max().expect("no columns");
		assert!(max > usize::from(fold), "columns are too short to reduce");
		assert!(log_lens.iter().all(|&l| l == max), "columns have different log_len: {log_lens:?}");
		max - usize::from(fold)
	}

	/// Sums `kernel` over the pairs of the given (already folded) columns, eq-weighted if given.
	pub fn pair_map_reduce<F: Field, K: PairKernel<F>>(
		cols: &[&[F]],
		eq_point: Option<&[F]>,
		kernel: &K,
	) -> Vec<F> {
		assert_eq!(cols.len(), kernel.n_in());
		let log_lens: Vec<usize> = cols.iter().map(|c| c.len().ilog2() as usize).collect();
		let log_n = round_log_len(&log_lens, false);
		let n = 1 << log_n;
		let eq = eq_point.map(|p| {
			assert_eq!(p.len() + 1, log_n, "eq_point must have n - 1 coordinates");
			eq_ind_partial_eval_scalars(p)
		});
		reduce_pairs(kernel, 0..n / 2, |c, i| (cols[c][i], cols[c][i + n / 2]), eq.as_deref())
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
