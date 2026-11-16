// Copyright 2026 The Binius Developers

//! The packed, multi-threaded backend over `FieldBuffer<P>`.
//!
//! `fold_words` calls the prover's `BitAxisFolder`.

use std::marker::PhantomData;

use binius_compute::GlobalAllocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, WideMul};
use binius_math::{
	FieldBuffer, FieldSlice,
	inner_product::inner_product_par,
	multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate, fold::fold_highest_var_inplace},
};
use binius_prover::fold_word::BitAxisFolder;
use rayon::prelude::*;

use crate::{Compute, Lazy, PairKernel, scalar};

/// Packed backend using rayon for parallelism.
pub struct SimdCompute<P>(PhantomData<fn() -> P>);

impl<P> SimdCompute<P> {
	/// Creates the backend.
	pub fn new() -> Self {
		Self(PhantomData)
	}
}

impl<P> Default for SimdCompute<P> {
	fn default() -> Self {
		Self::new()
	}
}

/// Log2 of the scalars per chunk of the eq-weighted pass (64 KiB of columns at 128 bits).
const MAX_CHUNK_VARS: usize = 12;

/// Sums `kernel` over the pairs of `cols`, packed when a half spans at least one packed element.
///
/// Without eq the sums are plain `P` sums. With `eq_point`, like `SharedMleCheckProver::execute`:
/// only the eq tensor over the low `chunk_vars` coordinates is built. Each chunk accumulates
/// `wide_mul(kernel_out, eq)` unreduced and reduces once; the remaining (suffix) coordinates are
/// folded in while combining chunk results, `lo + z * (hi - lo)`.
fn pair_map_reduce<P, K>(
	cols: &[FieldBuffer<P>],
	eq_point: Option<&[P::Scalar]>,
	kernel: &K,
) -> Vec<P::Scalar>
where
	P: PackedField<Scalar: BinaryField>,
	K: PairKernel<P::Scalar>,
{
	assert_eq!(cols.len(), kernel.n_in());
	let log_lens: Vec<usize> = cols.iter().map(|c| c.log_len()).collect();
	let log_n = scalar::round_log_len(&log_lens, false);
	if let Some(p) = eq_point {
		assert_eq!(p.len() + 1, log_n, "eq_point must have n - 1 coordinates");
	}
	if log_n - 1 < P::LOG_WIDTH {
		let scalars: Vec<Vec<P::Scalar>> =
			cols.iter().map(|c| c.iter_scalars().collect()).collect();
		let views: Vec<&[P::Scalar]> = scalars.iter().map(Vec::as_slice).collect();
		return scalar::pair_map_reduce(&views, eq_point, kernel);
	}
	// Lengths in packed elements. Bind the halves once, outside the element loop.
	let n = 1 << (log_n - P::LOG_WIDTH);
	let los: Vec<&[P]> = cols.iter().map(|c| &c.as_ref()[..n / 2]).collect();
	let his: Vec<&[P]> = cols.iter().map(|c| &c.as_ref()[n / 2..]).collect();
	let get = |c: usize, i: usize| (los[c][i], his[c][i]);

	let sums = eq_point.map_or_else(
		|| {
			// shortcut: plain P sums, no wide accumulation; the weighted path is the hot one.
			const CHUNK: usize = 256;
			(0..(n / 2).div_ceil(CHUNK))
				.into_par_iter()
				.map(|k| {
					scalar::reduce_pairs::<_, P, _>(
						kernel,
						k * CHUNK..((k + 1) * CHUNK).min(n / 2),
						get,
						None,
					)
				})
				.reduce(
					|| vec![P::zero(); kernel.n_out()],
					|mut a, b| {
						a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
						a
					},
				)
		},
		|p| {
			let chunk_vars = (log_n - 1).min(MAX_CHUNK_VARS);
			let eq_chunk = eq_ind_partial_eval::<P>(&p[..chunk_vars]);
			let chunk_len = 1 << (chunk_vars - P::LOG_WIDTH);
			let map = |k: usize| {
				let eq = eq_chunk.as_ref();
				let n_in = kernel.n_in();
				let (mut lo, mut hi) = (vec![P::zero(); n_in], vec![P::zero(); n_in]);
				let mut scratch = vec![P::zero(); kernel.scratch()];
				let mut out = vec![P::zero(); kernel.n_out()];
				let mut wide = vec![<P as WideMul>::Output::default(); kernel.n_out()];
				for (j, &eq_i) in eq.iter().enumerate() {
					for c in 0..n_in {
						(lo[c], hi[c]) = get(c, k * chunk_len + j);
					}
					kernel.map(&lo, &hi, &mut scratch, &mut out);
					for (w, &o) in wide.iter_mut().zip(&out) {
						*w += P::wide_mul(o, eq_i);
					}
				}
				wide.into_iter().map(P::reduce).collect::<Vec<P>>()
			};
			reduce_tree(0, log_n - 1 - chunk_vars, chunk_vars, p, &map)
		},
	);
	sums.iter().map(|p| p.iter().sum()).collect()
}

/// Maps chunks `first..first + 2^depth` and combines them, folding eq coordinate
/// `p[chunk_vars + depth - 1]` at each level of the bisection.
fn reduce_tree<P: PackedField>(
	first: usize,
	depth: usize,
	chunk_vars: usize,
	p: &[P::Scalar],
	map: &(impl Fn(usize) -> Vec<P> + Sync),
) -> Vec<P> {
	if depth == 0 {
		return map(first);
	}
	let (mut lhs, rhs) = rayon::join(
		|| reduce_tree(first, depth - 1, chunk_vars, p, map),
		|| reduce_tree(first + (1 << (depth - 1)), depth - 1, chunk_vars, p, map),
	);
	let z = P::broadcast(p[chunk_vars + depth - 1]);
	for (lo, hi) in lhs.iter_mut().zip(rhs) {
		*lo += z * (hi - *lo);
	}
	lhs
}

impl<P: PackedField<Scalar: BinaryField>> Compute<P::Scalar> for SimdCompute<P> {
	type FieldVec = FieldBuffer<P>;
	type FieldSlice<'a> = FieldSlice<'a, P>;
	type WordVec = Vec<Word>;
	type WordSlice<'a> = &'a [Word];
	type Future<'a, T: Send + 'a> = Lazy<'a, T>;

	fn copy_field_from_host(&self, values: &[P::Scalar]) -> FieldBuffer<P> {
		FieldBuffer::from_values(values)
	}

	fn copy_words_from_host(&self, words: &[Word]) -> Vec<Word> {
		words.to_vec()
	}

	fn as_slice<'a>(&self, v: &'a FieldBuffer<P>) -> FieldSlice<'a, P> {
		v.as_view()
	}

	fn as_word_slice<'a>(&self, v: &'a Vec<Word>) -> &'a [Word] {
		v
	}

	fn to_host<'a>(&self, v: FieldSlice<'a, P>) -> Lazy<'a, Vec<P::Scalar>>
	where
		Self: 'a,
	{
		Box::new(move || v.iter_scalars().collect())
	}

	fn tensor_expand<'a>(&self, point: &'a [P::Scalar]) -> Lazy<'a, FieldBuffer<P>> {
		Box::new(move || eq_ind_partial_eval(point))
	}

	fn inner_product<'a>(&self, a: FieldSlice<'a, P>, b: FieldSlice<'a, P>) -> Lazy<'a, P::Scalar>
	where
		Self: 'a,
	{
		Box::new(move || inner_product_par(&a, &b))
	}

	fn evaluate<'a>(&self, mle: FieldSlice<'a, P>, point: &'a [P::Scalar]) -> Lazy<'a, P::Scalar> {
		Box::new(move || evaluate(&mle, point))
	}

	fn fold_words<'a>(
		&self,
		words: &'a [Word],
		weights: &[P::Scalar; 64],
	) -> Lazy<'a, FieldBuffer<P>>
	where
		Self: 'a,
	{
		let weights = *weights;
		Box::new(move || {
			let folded = BitAxisFolder::new(&weights).fold::<P, _>(&GlobalAllocator, words);
			let mut padded: Vec<P::Scalar> = folded.iter_scalars().take(words.len()).collect();
			padded.resize(words.len().next_power_of_two(), P::Scalar::ZERO);
			FieldBuffer::from_values(&padded)
		})
	}

	fn fold_highest_var<'a>(
		&self,
		cols: &'a mut [FieldBuffer<P>],
		r: P::Scalar,
	) -> Lazy<'a, &'a mut [FieldBuffer<P>]>
	where
		Self: 'a,
	{
		Box::new(move || {
			cols.iter_mut()
				.for_each(|col| fold_highest_var_inplace(col, r));
			cols
		})
	}

	fn fold_pair_map_reduce<'a, K: PairKernel<P::Scalar> + 'a>(
		&self,
		cols: &'a mut [FieldBuffer<P>],
		r: Option<P::Scalar>,
		eq_point: Option<&'a [P::Scalar]>,
		kernel: &'a K,
	) -> Lazy<'a, Vec<P::Scalar>>
	where
		Self: 'a,
	{
		Box::new(move || {
			let log_lens: Vec<usize> = cols.iter().map(|c| c.log_len()).collect();
			let n = scalar::round_log_len(&log_lens, r.is_some());
			if let Some(r) = r {
				cols.iter_mut()
					.filter(|c| c.log_len() == n + 1)
					.for_each(|col| fold_highest_var_inplace(col, r));
			}
			pair_map_reduce(cols, eq_point, kernel)
		})
	}

	fn join<'a, A: Send + 'a, B: Send + 'a>(
		&self,
		a: Lazy<'a, A>,
		b: Lazy<'a, B>,
	) -> Lazy<'a, (A, B)> {
		Box::new(move || rayon::join(a, b))
	}

	fn join_all<'a, T: Send + 'a>(&self, futures: Vec<Lazy<'a, T>>) -> Lazy<'a, Vec<T>> {
		Box::new(move || futures.into_par_iter().map(|f| f()).collect())
	}

	fn then<'a, A: Send + 'a, B: Send + 'a>(
		&self,
		a: Lazy<'a, A>,
		f: impl FnOnce(A) -> Lazy<'a, B> + Send + 'a,
	) -> Lazy<'a, B> {
		Box::new(move || f(a())())
	}

	fn exec<'a, T: Send + 'a>(&self, f: Lazy<'a, T>) -> T {
		f()
	}
}
