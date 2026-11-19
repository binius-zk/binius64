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
	line::extrapolate_line,
	multilinear::{
		eq::eq_ind_partial_eval,
		evaluate::evaluate,
		fold::{fold_highest_var, fold_highest_var_inplace},
	},
};
use binius_prover::fold_word::BitAxisFolder;
use rayon::prelude::*;

use crate::{Column, Compute, Lazy, PairKernel, scalar};

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

/// Binds the highest variable of `col`; a borrowed column becomes a fresh owned one.
fn fold_col<P: PackedField>(col: &mut Column<FieldBuffer<P>, FieldSlice<'_, P>>, r: P::Scalar) {
	match col {
		Column::Owned(v) => fold_highest_var_inplace(v, r),
		Column::Borrowed(s) => *col = Column::Owned(fold_highest_var(&GlobalAllocator, s, r)),
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
	cols: &[FieldSlice<'_, P>],
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
						|c, i| (los[c][i], his[c][i]),
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
			let chunk_vars = (log_n - 1).min(MAX_CHUNK_VARS.max(P::LOG_WIDTH));
			let eq_chunk = eq_ind_partial_eval::<P>(&p[..chunk_vars]);
			let chunk_len = 1 << (chunk_vars - P::LOG_WIDTH);
			let map = |k: usize| {
				let range = k * chunk_len..(k + 1) * chunk_len;
				let lo: Vec<&[P]> = los.iter().map(|c| &c[range.clone()]).collect();
				let hi: Vec<&[P]> = his.iter().map(|c| &c[range.clone()]).collect();
				chunk_map(kernel, eq_chunk.as_ref(), &lo, &hi)
			};
			reduce_tree(0, log_n - 1 - chunk_vars, chunk_vars, p, &split_chunks, &map)
		},
	);
	sums.iter().map(|p| p.iter().sum()).collect()
}

/// The eq-weighted sum of `kernel` over one chunk: the packed elements of `lo[c]`/`hi[c]` pair with
/// `eq`. Accumulates `wide_mul` unreduced and reduces once.
fn chunk_map<P, K>(kernel: &K, eq: &[P], lo: &[&[P]], hi: &[&[P]]) -> Vec<P>
where
	P: PackedField<Scalar: BinaryField>,
	K: PairKernel<P::Scalar>,
{
	let n_in = kernel.n_in();
	let (mut l, mut h) = (vec![P::zero(); n_in], vec![P::zero(); n_in]);
	let mut scratch = vec![P::zero(); kernel.scratch()];
	let mut out = vec![P::zero(); kernel.n_out()];
	let mut wide = vec![<P as WideMul>::Output::default(); kernel.n_out()];
	for (j, &eq_i) in eq.iter().enumerate() {
		for c in 0..n_in {
			(l[c], h[c]) = (lo[c][j], hi[c][j]);
		}
		kernel.map(&l, &h, &mut scratch, &mut out);
		for (w, &o) in wide.iter_mut().zip(&out) {
			*w += P::wide_mul(o, eq_i);
		}
	}
	wide.into_iter().map(P::reduce).collect()
}

/// Splits chunks `first..first + 2^depth` in two.
const fn split_chunks(first: usize, depth: usize) -> [usize; 2] {
	[first, first + (1 << (depth - 1))]
}

/// Maps the `2^depth` chunks of `state` and combines them, folding eq coordinate
/// `p[chunk_vars + depth - 1]` at each level of the bisection.
fn reduce_tree<P: PackedField, S: Send>(
	state: S,
	depth: usize,
	chunk_vars: usize,
	p: &[P::Scalar],
	split: &(impl Fn(S, usize) -> [S; 2] + Sync),
	map: &(impl Fn(S) -> Vec<P> + Sync),
) -> Vec<P> {
	if depth == 0 {
		return map(state);
	}
	let [a, b] = split(state, depth);
	let (mut lhs, rhs) = rayon::join(
		|| reduce_tree(a, depth - 1, chunk_vars, p, split, map),
		|| reduce_tree(b, depth - 1, chunk_vars, p, split, map),
	);
	let z = P::broadcast(p[chunk_vars + depth - 1]);
	for (lo, hi) in lhs.iter_mut().zip(rhs) {
		*lo += z * (hi - *lo);
	}
	lhs
}

/// One half (`lo` or `hi`) of a column's folded output, deferred: `a` and `b` are the old column's
/// matching segments on the highest variable. In place overwrites `a`; out of place fills `dst`.
enum Seg<'a, P> {
	InPlace {
		a: &'a mut [P],
		b: &'a [P],
	},
	OutOfPlace {
		dst: &'a mut [P],
		a: &'a [P],
		b: &'a [P],
	},
}

impl<'a, P: PackedField> Seg<'a, P> {
	const fn split_half(self) -> [Self; 2] {
		match self {
			Self::InPlace { a, b } => {
				let (a0, a1) = a.split_at_mut(a.len() / 2);
				let (b0, b1) = b.split_at(b.len() / 2);
				[
					Self::InPlace { a: a0, b: b0 },
					Self::InPlace { a: a1, b: b1 },
				]
			}
			Self::OutOfPlace { dst, a, b } => {
				let (d0, d1) = dst.split_at_mut(dst.len() / 2);
				let (a0, a1) = a.split_at(a.len() / 2);
				let (b0, b1) = b.split_at(b.len() / 2);
				[
					Self::OutOfPlace {
						dst: d0,
						a: a0,
						b: b0,
					},
					Self::OutOfPlace {
						dst: d1,
						a: a1,
						b: b1,
					},
				]
			}
		}
	}

	/// Writes `a + r * (b - a)` and returns it.
	fn fold(self, r: P) -> &'a [P] {
		match self {
			Self::InPlace { a, b } => {
				for (x, &y) in a.iter_mut().zip(b) {
					*x = extrapolate_line(*x, y, r);
				}
				a
			}
			Self::OutOfPlace { dst, a, b } => {
				for (d, (&x, &y)) in dst.iter_mut().zip(a.iter().zip(b)) {
					*d = extrapolate_line(x, y, r);
				}
				dst
			}
		}
	}
}

/// Bisects every column's `[lo, hi]` pair on its highest remaining variable.
fn split_cols<P: PackedField>(
	cols: Vec<[Seg<'_, P>; 2]>,
	_depth: usize,
) -> [Vec<[Seg<'_, P>; 2]>; 2] {
	let (c0, c1) = cols
		.into_iter()
		.map(|[lo, hi]| {
			let ([lo0, lo1], [hi0, hi1]) = (lo.split_half(), hi.split_half());
			([lo0, hi0], [lo1, hi1])
		})
		.unzip();
	[c0, c1]
}

/// `pair_map_reduce` of the columns folded by `r`, with the fold fused into the map's read, like
/// `MleStore::map_reduce_with_fold`. Owned columns fold in place and truncate; borrowed ones fold
/// into a fresh buffer and become owned. Needs `chunk_vars >= P::LOG_WIDTH` (the caller checks).
fn fold_pair_map_reduce_fused<P, K>(
	cols: &mut [Column<FieldBuffer<P>, FieldSlice<'_, P>>],
	r: P::Scalar,
	p: &[P::Scalar],
	kernel: &K,
) -> Vec<P::Scalar>
where
	P: PackedField<Scalar: BinaryField>,
	K: PairKernel<P::Scalar>,
{
	let n = p.len(); // folded log_len minus one: the variable the chunks and tree split
	let half = 1 << (n + 1 - P::LOG_WIDTH); // old column = [a, b], `half` packed elements each
	let mut dsts: Vec<Option<FieldBuffer<P>>> = cols
		.iter()
		.map(|c| matches!(c, Column::Borrowed(_)).then(|| FieldBuffer::zeros(n + 1)))
		.collect();
	let segs: Vec<[Seg<'_, P>; 2]> = cols
		.iter_mut()
		.zip(&mut dsts)
		.map(|(col, dst)| match (col, dst) {
			(Column::Owned(v), _) => {
				let (a, b) = v.as_mut().split_at_mut(half);
				let (a0, a1) = a.split_at_mut(half / 2);
				let (b0, b1) = b.split_at(half / 2);
				[Seg::InPlace { a: a0, b: b0 }, Seg::InPlace { a: a1, b: b1 }]
			}
			(Column::Borrowed(s), Some(dst)) => {
				let s: &FieldSlice<'_, P> = s;
				let (a, b) = s.as_ref().split_at(half);
				let (d0, d1) = dst.as_mut().split_at_mut(half / 2);
				let (a0, a1) = a.split_at(half / 2);
				let (b0, b1) = b.split_at(half / 2);
				[
					Seg::OutOfPlace {
						dst: d0,
						a: a0,
						b: b0,
					},
					Seg::OutOfPlace {
						dst: d1,
						a: a1,
						b: b1,
					},
				]
			}
			(Column::Borrowed(_), None) => unreachable!(),
		})
		.collect();

	let chunk_vars = n.min(MAX_CHUNK_VARS.max(P::LOG_WIDTH));
	let eq_chunk = eq_ind_partial_eval::<P>(&p[..chunk_vars]);
	let r = P::broadcast(r);
	let map = |cols: Vec<[Seg<'_, P>; 2]>| {
		let (lo, hi): (Vec<_>, Vec<_>) = cols
			.into_iter()
			.map(|[l, h]| (l.fold(r), h.fold(r)))
			.unzip();
		chunk_map(kernel, eq_chunk.as_ref(), &lo, &hi)
	};
	let sums = reduce_tree(segs, n - chunk_vars, chunk_vars, p, &split_cols, &map);

	for (col, dst) in cols.iter_mut().zip(dsts) {
		match (col, dst) {
			(Column::Owned(v), _) => v.truncate(n + 1),
			(col, dst) => *col = Column::Owned(dst.expect("borrowed columns get a destination")),
		}
	}
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

	fn fold_highest_var<'a, 's: 'a>(
		&self,
		cols: &'a mut [Column<FieldBuffer<P>, FieldSlice<'s, P>>],
		r: P::Scalar,
	) -> Lazy<'a, &'a mut [Column<FieldBuffer<P>, FieldSlice<'s, P>>]>
	where
		Self: 'a,
	{
		Box::new(move || {
			cols.iter_mut().for_each(|col| fold_col(col, r));
			cols
		})
	}

	fn fold_pair_map_reduce<'a, 's: 'a, K: PairKernel<P::Scalar> + 'a>(
		&self,
		cols: &'a mut [Column<FieldBuffer<P>, FieldSlice<'s, P>>],
		r: Option<P::Scalar>,
		eq_point: Option<&'a [P::Scalar]>,
		kernel: &'a K,
	) -> Lazy<'a, Vec<P::Scalar>>
	where
		Self: 'a,
	{
		Box::new(move || {
			if let (Some(r), Some(p)) = (r, eq_point) {
				let log_lens: Vec<usize> = cols
					.iter()
					.map(|c| match c {
						Column::Owned(v) => v.log_len(),
						Column::Borrowed(s) => s.log_len(),
					})
					.collect();
				let n = scalar::round_log_len(&log_lens, true);
				// Below packing width the columns are cache-resident: fold, then map.
				if n > P::LOG_WIDTH {
					return fold_pair_map_reduce_fused(cols, r, p, kernel);
				}
			}
			if let Some(r) = r {
				cols.iter_mut().for_each(|col| fold_col(col, r));
			}
			let views: Vec<_> = cols
				.iter()
				.map(|col| match col {
					Column::Owned(v) => v.as_view(),
					Column::Borrowed(s) => s.as_view(),
				})
				.collect();
			pair_map_reduce(&views, eq_point, kernel)
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
