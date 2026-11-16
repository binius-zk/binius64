// Copyright 2026 The Binius Developers

//! The obviously-correct single-threaded backend over `Vec<F>`.

use binius_core::word::Word;
use binius_field::BinaryField;
use binius_math::{inner_product::inner_product, multilinear::eq::eq_ind_partial_eval_scalars};

use crate::{Column, Compute, Lazy, PairKernel, scalar};

fn view<'x, F>(col: &'x Column<Vec<F>, &[F]>) -> &'x [F] {
	match col {
		Column::Owned(v) => v,
		Column::Borrowed(s) => s,
	}
}

/// Binds the highest variable of `col`; a borrowed column becomes a fresh owned one.
fn fold_col<F: BinaryField>(col: &mut Column<Vec<F>, &[F]>, r: F) {
	match col {
		Column::Owned(v) => scalar::fold_highest_var(v, r),
		Column::Borrowed(s) => {
			let (s, half) = (*s, s.len() / 2);
			*col = Column::Owned((0..half).map(|i| s[i] + r * (s[i + half] - s[i])).collect());
		}
	}
}

/// Scalar, single-threaded backend. Futures are closures run on `exec`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceCompute;

impl<F: BinaryField> Compute<F> for ReferenceCompute {
	type FieldVec = Vec<F>;
	type FieldSlice<'a> = &'a [F];
	type WordVec = Vec<Word>;
	type WordSlice<'a> = &'a [Word];
	type Future<'a, T: Send + 'a> = Lazy<'a, T>;

	fn copy_field_from_host(&self, values: &[F]) -> Vec<F> {
		assert!(values.len().is_power_of_two());
		values.to_vec()
	}

	fn copy_words_from_host(&self, words: &[Word]) -> Vec<Word> {
		words.to_vec()
	}

	fn as_slice<'a>(&self, v: &'a Vec<F>) -> &'a [F] {
		v
	}

	fn as_word_slice<'a>(&self, v: &'a Vec<Word>) -> &'a [Word] {
		v
	}

	fn to_host<'a>(&self, v: &'a [F]) -> Lazy<'a, Vec<F>>
	where
		Self: 'a,
	{
		Box::new(move || v.to_vec())
	}

	fn tensor_expand<'a>(&self, point: &'a [F]) -> Lazy<'a, Vec<F>> {
		Box::new(move || eq_ind_partial_eval_scalars(point))
	}

	fn inner_product<'a>(&self, a: &'a [F], b: &'a [F]) -> Lazy<'a, F>
	where
		Self: 'a,
	{
		Box::new(move || inner_product(a.iter().copied(), b.iter().copied()))
	}

	fn evaluate<'a>(&self, mle: &'a [F], point: &'a [F]) -> Lazy<'a, F> {
		Box::new(move || inner_product(mle.iter().copied(), eq_ind_partial_eval_scalars(point)))
	}

	fn fold_words<'a>(&self, words: &'a [Word], weights: &[F; 64]) -> Lazy<'a, Vec<F>>
	where
		Self: 'a,
	{
		let weights = *weights;
		Box::new(move || scalar::fold_words(words, &weights))
	}

	fn fold_highest_var<'a, 's: 'a>(
		&self,
		cols: &'a mut [Column<Vec<F>, &'s [F]>],
		r: F,
	) -> Lazy<'a, &'a mut [Column<Vec<F>, &'s [F]>]>
	where
		Self: 'a,
	{
		Box::new(move || {
			cols.iter_mut().for_each(|col| fold_col(col, r));
			cols
		})
	}

	fn fold_pair_map_reduce<'a, 's: 'a, K: PairKernel<F> + 'a>(
		&self,
		cols: &'a mut [Column<Vec<F>, &'s [F]>],
		r: Option<F>,
		eq_point: Option<&'a [F]>,
		kernel: &'a K,
	) -> Lazy<'a, Vec<F>>
	where
		Self: 'a,
	{
		Box::new(move || {
			let log_lens: Vec<usize> = cols
				.iter()
				.map(|c| view(c).len().ilog2() as usize)
				.collect();
			scalar::round_log_len(&log_lens, r.is_some());
			if let Some(r) = r {
				cols.iter_mut().for_each(|col| fold_col(col, r));
			}
			let views: Vec<&[F]> = cols.iter().map(|c| view(c)).collect();
			scalar::pair_map_reduce(&views, eq_point, kernel)
		})
	}

	fn join<'a, A: Send + 'a, B: Send + 'a>(
		&self,
		a: Lazy<'a, A>,
		b: Lazy<'a, B>,
	) -> Lazy<'a, (A, B)> {
		Box::new(move || (a(), b()))
	}

	fn join_all<'a, T: Send + 'a>(&self, futures: Vec<Lazy<'a, T>>) -> Lazy<'a, Vec<T>> {
		Box::new(move || futures.into_iter().map(|f| f()).collect())
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
