// Copyright 2024-2025 Irreducible Inc.

use std::{
	marker::PhantomData,
	ops::{Index, IndexMut, Range},
	slice,
};

use crate::rayon::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("dimensions do not match data size")]
	DimensionMismatch,
}

/// A mutable view of an 2D array in row-major order that allows for parallel processing of
/// vertical slices.
#[derive(Debug)]
pub struct StridedArray2DViewMut<'a, T> {
	data: *mut T,
	data_width: usize,
	height: usize,
	cols: Range<usize>,
	_marker: PhantomData<&'a mut [T]>,
}

// SAFETY: the view has exclusive access to its columns, matching `&mut [T]`.
unsafe impl<T: Send> Send for StridedArray2DViewMut<'_, T> {}
// SAFETY: a shared reference to the view yields only `&T`, matching `&mut [T]`.
unsafe impl<T: Sync> Sync for StridedArray2DViewMut<'_, T> {}

impl<'a, T> StridedArray2DViewMut<'a, T> {
	/// Create a single-piece view of the data.
	pub const fn without_stride(
		data: &'a mut [T],
		height: usize,
		width: usize,
	) -> Result<Self, Error> {
		if width * height != data.len() {
			return Err(Error::DimensionMismatch);
		}
		Ok(Self {
			data: data.as_mut_ptr(),
			data_width: width,
			height,
			cols: 0..width,
			_marker: PhantomData,
		})
	}

	/// Returns a reference to the data at the given indices without bounds checking.
	/// # Safety
	/// The caller must ensure that `i < self.height` and `j < self.width()`.
	pub unsafe fn get_unchecked_ref(&self, i: usize, j: usize) -> &T {
		debug_assert!(i < self.height);
		debug_assert!(j < self.width());
		// SAFETY:
		// - Provenance: reborrowed from the exclusive borrow the view was built from.
		// - Bounds: construction pairs the buffer length with the dimensions, and the caller
		//   guarantees the indices are in range.
		// - Non-overlap: sibling views hold disjoint column ranges.
		// - Only the addressed element is ever referenced, never a span of the buffer.
		unsafe { &*self.data.add(i * self.data_width + j + self.cols.start) }
	}

	/// Returns a mutable reference to the data at the given indices without bounds checking.
	/// # Safety
	/// The caller must ensure that `i < self.height` and `j < self.width()`.
	pub unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T {
		debug_assert!(i < self.height);
		debug_assert!(j < self.width());
		// SAFETY:
		// - Provenance: reborrowed from the exclusive borrow the view was built from.
		// - Bounds: construction pairs the buffer length with the dimensions, and the caller
		//   guarantees the indices are in range.
		// - Non-overlap: sibling views hold disjoint column ranges.
		// - Only the addressed element is ever referenced, never a span of the buffer.
		unsafe { &mut *self.data.add(i * self.data_width + j + self.cols.start) }
	}

	/// Returns this view's columns of row `i`, which are contiguous in memory.
	///
	/// # Panics
	///
	/// Panics if `i` is out of bounds.
	pub fn row(&self, i: usize) -> &[T] {
		assert!(i < self.height);
		let start = i * self.data_width + self.cols.start;
		// SAFETY:
		// - Provenance: reborrowed from the exclusive borrow the view was built from.
		// - Bounds: construction pairs the buffer length with the dimensions, `i` is checked above,
		//   and the column range is a subrange of the row.
		// - Non-overlap: the slice spans only this view's columns, which no sibling view holds.
		unsafe { slice::from_raw_parts(self.data.add(start), self.width()) }
	}

	/// Returns the `mut_rows` rows mutably alongside the `shared_rows` rows.
	///
	/// # Panics
	///
	/// Panics if any index is out of bounds.
	/// Panics if a mutable row repeats, or appears among the shared rows.
	pub fn rows_mut<const M: usize, const S: usize>(
		&mut self,
		mut_rows: [usize; M],
		shared_rows: [usize; S],
	) -> ([&mut [T]; M], [&[T]; S]) {
		for &shared in &shared_rows {
			assert!(shared < self.height);
		}
		for (k, &row) in mut_rows.iter().enumerate() {
			assert!(row < self.height);
			assert!(!mut_rows[..k].contains(&row), "a mutable row repeats");
			assert!(!shared_rows.contains(&row), "a mutable row is also a shared row");
		}

		let (data_width, start, width) = (self.data_width, self.cols.start, self.width());
		let base = self.data;
		// SAFETY:
		// Row `i` occupies `i * data_width + start .. + width`, inside `data` because
		// `i < height` and the column range is a subrange of the row.
		// The asserts above make the mutable rows pairwise distinct and disjoint from the
		// shared rows, so no two of the returned slices ever overlap.
		// Dropping either assert would hand out two references to one element.
		unsafe {
			(
				mut_rows
					.map(|i| slice::from_raw_parts_mut(base.add(i * data_width + start), width)),
				shared_rows.map(|i| slice::from_raw_parts(base.add(i * data_width + start), width)),
			)
		}
	}

	pub const fn height(&self) -> usize {
		self.height
	}

	pub const fn width(&self) -> usize {
		self.cols.end - self.cols.start
	}

	/// Iterate over the mutable references to the elements in the specified column.
	pub fn iter_column_mut(&mut self, col: usize) -> impl Iterator<Item = &mut T> + '_ {
		assert!(col < self.width());
		let start = col + self.cols.start;
		let data = self.data;
		(0..self.height).map(move |i|
				// SAFETY:
				// - Provenance: reborrowed from the exclusive borrow the view was built from.
				// - Bounds: the row is below the height and the column is checked above.
				// - Non-overlap: one row per step, so no two yielded references coincide.
				unsafe { &mut *data.add(i * self.data_width + start) })
	}

	/// Returns iterator over vertical slices of the data for the given stride.
	pub fn into_strides(self, stride: usize) -> impl Iterator<Item = Self> + 'a {
		let Self {
			data,
			data_width,
			height,
			cols,
			..
		} = self;

		cols.clone().step_by(stride).map(move |start| {
			let end = (start + stride).min(cols.end);
			Self {
				data,
				data_width,
				height,
				cols: start..end,
				_marker: PhantomData,
			}
		})
	}

	/// Returns parallel iterator over vertical slices of the data for the given stride.
	pub fn into_par_strides(self, stride: usize) -> impl IndexedParallelIterator<Item = Self> + 'a
	where
		T: Send + Sync,
	{
		let Self {
			data,
			data_width,
			height,
			cols,
			..
		} = self;
		let data = SendPtr(data);

		cols.clone()
			.into_par_iter()
			.step_by(stride)
			.map(move |start| {
				let end = (start + stride).min(cols.end);
				Self {
					data: data.as_ptr(),
					data_width,
					height,
					cols: start..end,
					_marker: PhantomData,
				}
			})
	}

	/// Returns iterator over single-column mutable views of the data.
	pub fn iter_cols(&mut self) -> impl Iterator<Item = StridedArray2DColMut<'_, T>> + '_ {
		let data = self.data;
		self.cols.clone().map(move |col| StridedArray2DColMut {
			data,
			data_width: self.data_width,
			height: self.height,
			col,
			_marker: PhantomData,
		})
	}

	/// Returns parallel iterator over single-column mutable views of the data.
	pub fn par_iter_cols(
		&mut self,
	) -> impl IndexedParallelIterator<Item = StridedArray2DColMut<'_, T>> + '_
	where
		T: Send + Sync,
	{
		let data = SendPtr(self.data);
		let data_width = self.data_width;
		let height = self.height;
		self.cols
			.clone()
			.into_par_iter()
			.map(move |col| StridedArray2DColMut {
				data: data.as_ptr(),
				data_width,
				height,
				col,
				_marker: PhantomData,
			})
	}
}

/// A mutable view of a single column (vertical slice) of a 2D array in row-major order.
#[derive(Debug)]
pub struct StridedArray2DColMut<'a, T> {
	data: *mut T,
	data_width: usize,
	height: usize,
	col: usize,
	_marker: PhantomData<&'a mut [T]>,
}

// SAFETY: the view has exclusive access to its column, matching `&mut [T]`.
unsafe impl<T: Send> Send for StridedArray2DColMut<'_, T> {}
// SAFETY: a shared reference to the view yields only `&T`, matching `&mut [T]`.
unsafe impl<T: Sync> Sync for StridedArray2DColMut<'_, T> {}

impl<'a, T> StridedArray2DColMut<'a, T> {
	pub const fn height(&self) -> usize {
		self.height
	}

	/// Returns a reference to the data at the given row index without bounds checking.
	/// # Safety
	/// The caller must ensure that `i < self.height`.
	pub unsafe fn get_unchecked_ref(&self, i: usize) -> &T {
		debug_assert!(i < self.height);
		// SAFETY:
		// - Provenance: reborrowed from the exclusive borrow the parent view was built from.
		// - Bounds: the parent's dimensions cover the buffer, and the caller guarantees the row is
		//   below the height.
		// - Non-overlap: the constructors give each view its own column.
		// - Only the addressed element is ever referenced, never a span of the buffer.
		unsafe { &*self.data.add(i * self.data_width + self.col) }
	}

	/// Returns a mutable reference to the data at the given row index without bounds checking.
	/// # Safety
	/// The caller must ensure that `i < self.height`.
	pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut T {
		debug_assert!(i < self.height);
		// SAFETY:
		// - Provenance: reborrowed from the exclusive borrow the parent view was built from.
		// - Bounds: the parent's dimensions cover the buffer, and the caller guarantees the row is
		//   below the height.
		// - Non-overlap: the constructors give each view its own column.
		// - Only the addressed element is ever referenced, never a span of the buffer.
		unsafe { &mut *self.data.add(i * self.data_width + self.col) }
	}
}

impl<T> Index<usize> for StridedArray2DColMut<'_, T> {
	type Output = T;

	fn index(&self, i: usize) -> &T {
		assert!(i < self.height());
		unsafe { self.get_unchecked_ref(i) }
	}
}

impl<T> IndexMut<usize> for StridedArray2DColMut<'_, T> {
	fn index_mut(&mut self, i: usize) -> &mut Self::Output {
		assert!(i < self.height());
		unsafe { self.get_unchecked_mut(i) }
	}
}

impl<T> Index<(usize, usize)> for StridedArray2DViewMut<'_, T> {
	type Output = T;

	fn index(&self, (i, j): (usize, usize)) -> &T {
		assert!(i < self.height());
		assert!(j < self.width());
		unsafe { self.get_unchecked_ref(i, j) }
	}
}

impl<T> IndexMut<(usize, usize)> for StridedArray2DViewMut<'_, T> {
	fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
		assert!(i < self.height());
		assert!(j < self.width());
		unsafe { self.get_unchecked_mut(i, j) }
	}
}

/// A wrapper around a raw pointer that implements Send and Sync.
///
/// # Safety
/// The caller must ensure that the pointer is valid and that concurrent access
/// through multiple `SendPtr` instances does not cause data races.
struct SendPtr<T>(*mut T);

impl<T> SendPtr<T> {
	const fn as_ptr(self) -> *mut T {
		self.0
	}
}

impl<T> Clone for SendPtr<T> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<T> Copy for SendPtr<T> {}

// Safety: SendPtr is only used internally where we ensure non-overlapping access
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Sync> Sync for SendPtr<T> {}

#[cfg(test)]
mod tests {
	use std::array;

	use super::*;

	#[test]
	fn test_indexing() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let mut arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();
		assert_eq!(arr[(3, 1)], 10);
		arr[(2, 2)] = 88;
		assert_eq!(data[8], 88);
	}

	#[test]
	fn test_strides() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();

		{
			let mut strides = arr.into_strides(2);
			let mut stride0 = strides.next().unwrap();
			let mut stride1 = strides.next().unwrap();
			assert!(strides.next().is_none());

			assert_eq!(stride0.width(), 2);
			assert_eq!(stride1.width(), 1);

			stride0[(0, 0)] = 88;
			stride1[(1, 0)] = 99;
		}

		assert_eq!(data[0], 88);
		assert_eq!(data[5], 99);
	}

	#[test]
	fn test_parallel_strides() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();

		{
			let mut strides: Vec<_> = arr.into_par_strides(2).collect();
			assert_eq!(strides.len(), 2);
			assert_eq!(strides[0].width(), 2);
			assert_eq!(strides[1].width(), 1);

			strides[0][(0, 0)] = 88;
			strides[1][(1, 0)] = 99;
		}

		assert_eq!(data[0], 88);
		assert_eq!(data[5], 99);
	}

	#[test]
	fn a_row_covers_only_the_columns_the_view_holds() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();

		// Row 1 of the array is 3, 4, 5, and a full-width view sees all of it.
		assert_eq!(arr.row(1), &[3, 4, 5]);

		// The second stride of width 2 holds column 2 alone.
		let stride = arr.into_strides(2).nth(1).unwrap();
		assert_eq!(stride.row(1), &[5]);
	}

	#[test]
	fn split_rows_alias_nothing_and_write_through() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let mut arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();

		let ([sum, diff], [x, y]) = arr.rows_mut([0, 1], [2, 3]);
		assert_eq!(x, &[6, 7, 8]);
		assert_eq!(y, &[9, 10, 11]);
		for i in 0..3 {
			sum[i] = x[i] + y[i];
			diff[i] = y[i] - x[i];
		}

		assert_eq!(&data[0..3], &[15, 17, 19]);
		assert_eq!(&data[3..6], &[3, 3, 3]);
	}

	#[test]
	#[should_panic(expected = "a mutable row is also a shared row")]
	fn a_mutable_row_may_not_be_a_shared_row() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let mut arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();
		arr.rows_mut([1], [0, 1]);
	}

	#[test]
	#[should_panic(expected = "a mutable row repeats")]
	fn two_mutable_rows_may_not_name_one_row() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let mut arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();
		arr.rows_mut([2, 2], [0]);
	}

	#[test]
	fn test_iter_column_mut() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let data_clone = data;
		let mut arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();

		let mut col_iter = arr.iter_column_mut(1);
		assert_eq!(col_iter.next().copied(), Some(data_clone[1]));
		assert_eq!(col_iter.next().copied(), Some(data_clone[4]));
		assert_eq!(col_iter.next().copied(), Some(data_clone[7]));
		assert_eq!(col_iter.next().copied(), Some(data_clone[10]));
		assert_eq!(col_iter.next(), None);
	}

	#[test]
	fn test_col_mut_indexing() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let mut arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();

		let mut cols: Vec<_> = arr.iter_cols().collect();
		assert_eq!(cols.len(), 3);

		// Test reading - column 1 contains elements at indices 1, 4, 7, 10
		assert_eq!(cols[1][0], 1);
		assert_eq!(cols[1][1], 4);
		assert_eq!(cols[1][2], 7);
		assert_eq!(cols[1][3], 10);

		// Test writing
		cols[0][2] = 88;
		cols[2][1] = 99;

		assert_eq!(data[6], 88); // row 2, col 0
		assert_eq!(data[5], 99); // row 1, col 2
	}

	#[test]
	fn test_iter_cols() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let mut arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();

		{
			let mut cols = arr.iter_cols();
			let mut col0 = cols.next().unwrap();
			let mut col1 = cols.next().unwrap();
			let mut col2 = cols.next().unwrap();
			assert!(cols.next().is_none());

			assert_eq!(col0.height(), 4);
			assert_eq!(col1.height(), 4);
			assert_eq!(col2.height(), 4);

			col0[0] = 88;
			col1[1] = 99;
			col2[3] = 77;
		}

		assert_eq!(data[0], 88); // row 0, col 0
		assert_eq!(data[4], 99); // row 1, col 1
		assert_eq!(data[11], 77); // row 3, col 2
	}

	#[test]
	fn test_views_are_send_and_sync() {
		fn assert_send_sync<T: Send + Sync>() {}
		assert_send_sync::<StridedArray2DViewMut<'_, usize>>();
		assert_send_sync::<StridedArray2DColMut<'_, usize>>();
	}

	#[test]
	fn test_par_iter_cols() {
		let mut data = array::from_fn::<_, 12, _>(|i| i);
		let mut arr = StridedArray2DViewMut::without_stride(&mut data, 4, 3).unwrap();

		{
			let mut cols: Vec<_> = arr.par_iter_cols().collect();
			assert_eq!(cols.len(), 3);

			cols[0][0] = 88;
			cols[1][1] = 99;
			cols[2][3] = 77;
		}

		assert_eq!(data[0], 88); // row 0, col 0
		assert_eq!(data[4], 99); // row 1, col 1
		assert_eq!(data[11], 77); // row 3, col 2
	}
}
