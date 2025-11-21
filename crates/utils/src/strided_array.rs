// Copyright 2024-2025 Irreducible Inc.

use core::slice;
use std::ops::{Index, IndexMut, Range};

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
	data: &'a mut [T],
	data_width: usize,
	height: usize,
	cols: Range<usize>,
}

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
			data,
			data_width: width,
			height,
			cols: 0..width,
		})
	}

	/// Returns a reference to the data at the given indices without bounds checking.
	/// # Safety
	/// The caller must ensure that `i < self.height` and `j < self.width()`.
	pub unsafe fn get_unchecked_ref(&self, i: usize, j: usize) -> &T {
		debug_assert!(i < self.height);
		debug_assert!(j < self.width());
		unsafe {
			self.data
				.get_unchecked(i * self.data_width + j + self.cols.start)
		}
	}

	/// Returns a mutable reference to the data at the given indices without bounds checking.
	/// # Safety
	/// The caller must ensure that `i < self.height` and `j < self.width()`.
	pub unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T {
		debug_assert!(i < self.height);
		debug_assert!(j < self.width());
		unsafe {
			self.data
				.get_unchecked_mut(i * self.data_width + j + self.cols.start)
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
		let data_ptr = self.data.as_mut_ptr();
		(0..self.height).map(move |i|
				// Safety: 
				// - `data_ptr` points to the start of the data slice.
				// - `col` is within bounds of the width.
				// - different iterator values do not overlap.
				unsafe { &mut *data_ptr.add(i * self.data_width + start) })
	}

	/// Returns iterator over vertical slices of the data for the given stride.
	#[allow(dead_code)]
	pub fn into_strides(self, stride: usize) -> impl Iterator<Item = Self> + 'a {
		let Self {
			data,
			data_width,
			height,
			cols,
		} = self;

		cols.clone().step_by(stride).map(move |start| {
			let end = (start + stride).min(cols.end);
			Self {
				// Safety: different instances of StridedArray2DViewMut created with the same data
				// slice do not access overlapping indices.
				data: unsafe { slice::from_raw_parts_mut(data.as_mut_ptr(), data.len()) },
				data_width,
				height,
				cols: start..end,
			}
		})
	}

	/// Returns parallel iterator over vertical slices of the data for the given stride.
	pub fn into_par_strides(self, stride: usize) -> impl ParallelIterator<Item = Self> + 'a
	where
		T: Send + Sync,
	{
		self.cols
			.clone()
			.into_par_iter()
			.step_by(stride)
			.map(move |start| {
				let end = (start + stride).min(self.cols.end);
				// We are setting the same lifetime as `self` captures.
				Self {
					// Safety: different instances of StridedArray2DViewMut created with the same
					// data slice do not access overlapping indices.
					data: unsafe {
						slice::from_raw_parts_mut(self.data.as_ptr() as *mut T, self.data.len())
					},
					data_width: self.data_width,
					height: self.height,
					cols: start..end,
				}
			})
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
}
