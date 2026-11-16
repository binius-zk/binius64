// Copyright 2026 The Binius Developers

//! The packed, multi-threaded backend over `FieldBuffer<P>`.
//!
//! Gap: compositions are evaluated over `P`, so they cannot use `WideMul` unreduced accumulation.
//! `fold_words` calls the prover's `BitAxisFolder`.

use std::marker::PhantomData;

use binius_compute::GlobalAllocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField};
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

/// Sums `kernel` over the pairs of `cols`, packed when a half spans at least one packed element.
fn pair_map_reduce<P, K>(cols: &[FieldBuffer<P>], kernel: &K) -> Vec<P::Scalar>
where
	P: PackedField<Scalar: BinaryField>,
	K: PairKernel<P::Scalar>,
{
	assert_eq!(cols.len(), K::N_IN);
	let log_lens: Vec<usize> = cols.iter().map(|c| c.log_len()).collect();
	let log_n = scalar::round_log_len(&log_lens, false);
	if log_n - 1 < P::LOG_WIDTH {
		let scalars: Vec<Vec<P::Scalar>> =
			cols.iter().map(|c| c.iter_scalars().collect()).collect();
		let views: Vec<&[P::Scalar]> = scalars.iter().map(Vec::as_slice).collect();
		return scalar::pair_map_reduce(&views, kernel);
	}
	// Lengths in packed elements; a broadcast column is half as long.
	let n = 1 << (log_n - P::LOG_WIDTH);
	let get = |c: usize, i: usize| {
		let col = cols[c].as_ref();
		if col.len() == n {
			(col[i], col[i + n / 2])
		} else {
			(col[i], col[i])
		}
	};
	// One set of kernel buffers per chunk of pairs.
	const CHUNK: usize = 256;
	let sums = (0..(n / 2).div_ceil(CHUNK))
		.into_par_iter()
		.map(|k| {
			scalar::reduce_pairs::<_, P, _>(kernel, k * CHUNK..((k + 1) * CHUNK).min(n / 2), get)
		})
		.reduce(
			|| vec![P::zero(); K::N_OUT],
			|mut a, b| {
				a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
				a
			},
		);
	sums.iter().map(|p| p.iter().sum()).collect()
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
			pair_map_reduce(cols, kernel)
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
