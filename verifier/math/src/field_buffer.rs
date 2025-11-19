// Copyright 2025 Irreducible Inc.

use std::{
	ops::{Deref, DerefMut, Index, IndexMut},
	slice,
};

use binius_field::{
	Field, PackedField,
	packed::{get_packed_slice_unchecked, set_packed_slice_unchecked},
};
use binius_utils::{
	checked_arithmetics::{checked_log_2, strict_log_2},
	rayon::{prelude::*, slice::ParallelSlice},
};
use bytemuck::zeroed_vec;

use crate::Error;

/// A power-of-two-sized buffer containing field elements, stored in packed fields.
///
/// This struct maintains a set of invariants:
///  1) `values.len()` is a power of two
///  2) `values.len() >= 1 << log_len.saturating_sub(P::LOG_WIDTH)`.
#[derive(Debug, Clone, Eq)]
pub struct FieldBuffer<P: PackedField, Data: Deref<Target = [P]> = Box<[P]>> {
	/// log2 the number over elements in the buffer.
	log_len: usize,
	/// The packed values.
	values: Data,
}

impl<P: PackedField, Data: Deref<Target = [P]>> PartialEq for FieldBuffer<P, Data> {
	fn eq(&self, other: &Self) -> bool {
		// Custom equality impl is needed because values beyond length until capacity can be
		// arbitrary.
		if self.log_len < P::LOG_WIDTH {
			let iter_1 = self
				.values
				.first()
				.expect("len >= 1")
				.iter()
				.take(1 << self.log_len);
			let iter_2 = other
				.values
				.first()
				.expect("len >= 1")
				.iter()
				.take(1 << self.log_len);
			iter_1.eq(iter_2)
		} else {
			let prefix = 1 << (self.log_len - P::LOG_WIDTH);
			self.log_len == other.log_len && self.values[..prefix] == other.values[..prefix]
		}
	}
}

impl<P: PackedField> FieldBuffer<P> {
	/// Create a new FieldBuffer from a vector of values.
	///
	/// # Throws
	///
	/// * `PowerOfTwoLengthRequired` if the number of values is not a power of two.
	pub fn from_values(values: &[P::Scalar]) -> Result<Self, Error> {
		let Some(log_len) = strict_log_2(values.len()) else {
			return Err(Error::PowerOfTwoLengthRequired);
		};

		Self::from_values_truncated(values, log_len)
	}

	/// Create a new FieldBuffer from a vector of values.
	///
	/// Capacity `log_cap` is bumped to at least `P::LOG_WIDTH`.
	///
	/// # Throws
	///
	/// * `PowerOfTwoLengthRequired` if the number of values is not a power of two.
	/// * `IncorrectArgumentLength` if the number of values exceeds `1 << log_cap`.
	pub fn from_values_truncated(values: &[P::Scalar], log_cap: usize) -> Result<Self, Error> {
		if !values.len().is_power_of_two() {
			return Err(Error::PowerOfTwoLengthRequired);
		}

		let log_len = values.len().ilog2() as usize;
		if log_len > log_cap {
			return Err(Error::IncorrectArgumentLength {
				arg: "values".to_string(),
				expected: 1 << log_cap,
			});
		}

		let packed_cap = 1 << log_cap.saturating_sub(P::LOG_WIDTH);
		let mut packed_values = Vec::with_capacity(packed_cap);
		packed_values.extend(
			values
				.chunks(P::WIDTH)
				.map(|chunk| P::from_scalars(chunk.iter().copied())),
		);
		packed_values.resize(packed_cap, P::zero());

		Ok(Self {
			log_len,
			values: packed_values.into_boxed_slice(),
		})
	}

	/// Create a new [`FieldBuffer`] of zeros with the given log_len.
	pub fn zeros(log_len: usize) -> Self {
		Self::zeros_truncated(log_len, log_len).expect("log_len == log_cap")
	}

	/// Create a new [`FieldBuffer`] of zeros with the given log_len and capacity log_cap.
	///
	/// Capacity `log_cap` is bumped to at least `P::LOG_WIDTH`.
	pub fn zeros_truncated(log_len: usize, log_cap: usize) -> Result<Self, Error> {
		if log_len > log_cap {
			return Err(Error::IncorrectArgumentLength {
				arg: "log_len".to_string(),
				expected: log_cap,
			});
		}
		let packed_len = 1 << log_cap.saturating_sub(P::LOG_WIDTH);
		let values = zeroed_vec(packed_len).into_boxed_slice();
		Ok(Self { log_len, values })
	}
}

#[allow(clippy::len_without_is_empty)]
impl<P: PackedField, Data: Deref<Target = [P]>> FieldBuffer<P, Data> {
	/// Create a new FieldBuffer from a slice of packed values.
	///
	/// # Throws
	///
	/// * `IncorrectArgumentLength` if the number of field elements does not fit the `values.len()`
	///   exactly.
	pub fn new(log_len: usize, values: Data) -> Result<Self, Error> {
		let expected_packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		if values.len() != expected_packed_len {
			return Err(Error::IncorrectArgumentLength {
				arg: "values".to_string(),
				expected: expected_packed_len,
			});
		}
		Self::new_truncated(log_len, values)
	}

	/// Create a new FieldBuffer from a slice of packed values.
	///
	/// # Throws
	///
	/// * `IncorrectArgumentLength` if the number of field elements does not fit into the `values`.
	/// * `PowerOfTwoLengthRequired` if the `values.len()` is not a power of two.
	pub fn new_truncated(log_len: usize, values: Data) -> Result<Self, Error> {
		let min_packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
		if values.len() < min_packed_len {
			return Err(Error::IncorrectArgumentLength {
				arg: "values".to_string(),
				expected: min_packed_len,
			});
		}

		if !values.len().is_power_of_two() {
			return Err(Error::PowerOfTwoLengthRequired);
		}

		Ok(Self { log_len, values })
	}

	/// Returns log2 the number of field elements that the underlying collection may take.
	pub fn log_cap(&self) -> usize {
		checked_log_2(self.values.len()) + P::LOG_WIDTH
	}

	/// Returns the number of field elements that the underlying collection may take.
	pub fn cap(&self) -> usize {
		1 << self.log_cap()
	}

	/// Returns log2 the number of field elements.
	pub const fn log_len(&self) -> usize {
		self.log_len
	}

	/// Returns the number of field elements.
	pub fn len(&self) -> usize {
		1 << self.log_len
	}

	/// Borrows the buffer as a [`FieldSlice`].
	pub fn to_ref(&self) -> FieldSlice<'_, P> {
		FieldSlice::from_slice(self.log_len, self.as_ref())
			.expect("log_len matches values.len() by struct invariant")
	}

	/// Get a field element at the given index.
	///
	/// # Preconditions
	///
	/// * the index is in the range `0..self.len()`
	pub fn get(&self, index: usize) -> P::Scalar {
		self.get_checked(index)
			.expect("precondition: index is in range")
	}

	/// Get a field element at the given index.
	///
	/// # Throws
	///
	/// * `Error::ArgumentRangeError` if the index is out of bounds.
	pub fn get_checked(&self, index: usize) -> Result<P::Scalar, Error> {
		if index >= self.len() {
			return Err(Error::ArgumentRangeError {
				arg: "index".to_string(),
				range: 0..self.len(),
			});
		}

		// Safety: bound check on index performed above. The buffer length is at least
		// `self.len() >> P::LOG_WIDTH` by struct invariant.
		let val = unsafe { get_packed_slice_unchecked(&self.values, index) };
		Ok(val)
	}

	/// Returns an iterator over the scalar elements in the buffer.
	pub fn iter_scalars(&self) -> impl Iterator<Item = P::Scalar> + Send + Clone + '_ {
		P::iter_slice(self.as_ref()).take(self.len())
	}

	/// Get an aligned chunk of size `2^log_chunk_size`.
	///
	/// Chunk start offset divides chunk size; the result is essentially
	/// `chunks(log_chunk_size).nth(chunk_index)` but unlike `chunks` it does
	/// support sizes smaller than packing width.
	pub fn chunk(
		&self,
		log_chunk_size: usize,
		chunk_index: usize,
	) -> Result<FieldSlice<'_, P>, Error> {
		if log_chunk_size > self.log_len {
			return Err(Error::ArgumentRangeError {
				arg: "log_chunk_size".to_string(),
				range: 0..self.log_len + 1,
			});
		}

		let chunk_count = 1 << (self.log_len - log_chunk_size);
		if chunk_index >= chunk_count {
			return Err(Error::ArgumentRangeError {
				arg: "chunk_index".to_string(),
				range: 0..chunk_count,
			});
		}

		let values = if log_chunk_size >= P::LOG_WIDTH {
			let packed_log_chunk_size = log_chunk_size - P::LOG_WIDTH;
			let chunk =
				&self.values[chunk_index << packed_log_chunk_size..][..1 << packed_log_chunk_size];
			FieldSliceData::Slice(chunk)
		} else {
			let packed_log_chunks = P::LOG_WIDTH - log_chunk_size;
			let packed = self.values[chunk_index >> packed_log_chunks];
			let chunk_subindex = chunk_index & ((1 << packed_log_chunks) - 1);
			let chunk = P::from_scalars(
				(0..1 << log_chunk_size).map(|i| packed.get(chunk_subindex << log_chunk_size | i)),
			);
			FieldSliceData::Single(chunk)
		};

		Ok(FieldBuffer {
			log_len: log_chunk_size,
			values,
		})
	}

	/// Split the buffer into chunks of size `2^log_chunk_size`.
	///
	/// # Errors
	///
	/// * [`Error::ArgumentRangeError`] if `log_chunk_size < P::LOG_WIDTH` or `log_chunk_size >
	///   log_len`.
	pub fn chunks(
		&self,
		log_chunk_size: usize,
	) -> Result<impl Iterator<Item = FieldSlice<'_, P>> + Clone, Error> {
		if log_chunk_size < P::LOG_WIDTH || log_chunk_size > self.log_len {
			return Err(Error::ArgumentRangeError {
				arg: "log_chunk_size".to_string(),
				range: P::LOG_WIDTH..self.log_len + 1,
			});
		}

		let chunk_count = 1 << (self.log_len - log_chunk_size);
		let packed_chunk_size = 1 << (log_chunk_size - P::LOG_WIDTH);
		let chunks = self
			.values
			.chunks(packed_chunk_size)
			.take(chunk_count)
			.map(move |chunk| FieldBuffer {
				log_len: log_chunk_size,
				values: FieldSliceData::Slice(chunk),
			});

		Ok(chunks)
	}

	/// Creates an iterator over chunks of size `2^log_chunk_size` in parallel.
	///
	/// # Throws
	///
	/// * [`Error::ArgumentRangeError`] if `log_chunk_size < P::LOG_WIDTH` or `log_chunk_size >
	///   log_len`.
	pub fn chunks_par(
		&self,
		log_chunk_size: usize,
	) -> Result<impl IndexedParallelIterator<Item = FieldSlice<'_, P>>, Error> {
		if log_chunk_size < P::LOG_WIDTH || log_chunk_size > self.log_len {
			return Err(Error::ArgumentRangeError {
				arg: "log_chunk_size".to_string(),
				range: P::LOG_WIDTH..self.log_len + 1,
			});
		}

		let log_len = log_chunk_size.min(self.log_len);
		let packed_chunk_size = 1 << (log_chunk_size - P::LOG_WIDTH);
		let chunks = self
			.values
			.par_chunks(packed_chunk_size)
			.map(move |chunk| FieldBuffer {
				log_len,
				values: FieldSliceData::Slice(chunk),
			});

		Ok(chunks)
	}

	/// Splits the buffer in half and returns a pair of borrowed slices.
	///
	/// # Throws
	///
	/// * [`Error::CannotSplit`] if `self.log_len() == 0`
	pub fn split_half(&self) -> Result<(FieldSlice<'_, P>, FieldSlice<'_, P>), Error> {
		if self.log_len == 0 {
			return Err(Error::CannotSplit);
		}

		let new_log_len = self.log_len - 1;
		let (first, second) = if new_log_len < P::LOG_WIDTH {
			// The result will be two Single variants
			// We have exactly one packed element that needs to be split
			let packed = self.values[0];
			let zeros = P::default();

			let (first_half, second_half) = packed.interleave(zeros, new_log_len);

			let first = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceData::Single(first_half),
			};
			let second = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceData::Single(second_half),
			};

			(first, second)
		} else {
			// Split the packed values slice in half
			let half_len = 1 << (new_log_len - P::LOG_WIDTH);
			let (first_half, second_half) = self.values.split_at(half_len);
			let second_half = &second_half[..half_len];

			let first = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceData::Slice(first_half),
			};
			let second = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceData::Slice(second_half),
			};

			(first, second)
		};

		Ok((first, second))
	}
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> FieldBuffer<P, Data> {
	/// Borrows the buffer mutably as a [`FieldSliceMut`].
	pub fn to_mut(&mut self) -> FieldSliceMut<'_, P> {
		FieldSliceMut::from_slice(self.log_len, self.as_mut())
			.expect("log_len matches values.len() by struct invariant")
	}

	/// Set a field element at the given index.
	///
	/// # Preconditions
	///
	/// * the index is in the range `0..self.len()`
	pub fn set(&mut self, index: usize, value: P::Scalar) {
		self.set_checked(index, value)
			.expect("precondition: index is in range");
	}

	/// Set a field element at the given index.
	///
	/// # Throws
	///
	/// * `Error::ArgumentRangeError` if the index is out of bounds.
	pub fn set_checked(&mut self, index: usize, value: P::Scalar) -> Result<(), Error> {
		if index >= self.len() {
			return Err(Error::ArgumentRangeError {
				arg: "index".to_string(),
				range: 0..self.len(),
			});
		}

		// Safety: bound check on index performed above. The buffer length is at least
		// `self.len() >> P::LOG_WIDTH` by struct invariant.
		unsafe { set_packed_slice_unchecked(&mut self.values, index, value) };
		Ok(())
	}

	/// Truncates a field buffer to a shorter length.
	///
	/// If `new_log_len` is not less than current `log_len()`, this has no effect.
	pub fn truncate(&mut self, new_log_len: usize) {
		self.log_len = self.log_len.min(new_log_len);
	}

	/// Zero extends a field buffer to a longer length.
	///
	/// If `new_log_len` is not greater than current `log_len()`, this has no effect.
	///
	/// # Throws
	/// * `Error::IncorrectArgumentLength` if the zero extended size exceeds underlying capacity.
	pub fn zero_extend(&mut self, new_log_len: usize) -> Result<(), Error> {
		if new_log_len <= self.log_len {
			return Ok(());
		}

		if new_log_len > self.log_cap() {
			return Err(Error::IncorrectArgumentLength {
				arg: "new_log_len".to_string(),
				expected: self.log_cap(),
			});
		}

		if self.log_len < P::LOG_WIDTH {
			let first_elem = self.values.first_mut().expect("values.len() >= 1");
			for i in 1 << self.log_len..(1 << new_log_len).min(P::WIDTH) {
				first_elem.set(i, P::Scalar::ZERO);
			}
		}

		let packed_start = 1 << self.log_len.saturating_sub(P::LOG_WIDTH);
		let packed_end = 1 << new_log_len.saturating_sub(P::LOG_WIDTH);
		self.values[packed_start..packed_end].fill(P::zero());

		self.log_len = new_log_len;
		Ok(())
	}

	/// Sets the new log length. If the new log length is bigger than the current log length,
	/// the new values (in case when `self.log_len < new_log_len`) will be filled with
	/// the values from the existing buffer.
	///
	/// # Throws
	///
	/// * `Error::IncorrectArgumentLength` if the new log length exceeds the buffer's capacity.
	pub fn resize(&mut self, new_log_len: usize) -> Result<(), Error> {
		if new_log_len > self.log_cap() {
			return Err(Error::IncorrectArgumentLength {
				arg: "new_log_len".to_string(),
				expected: self.log_cap(),
			});
		}

		self.log_len = new_log_len;
		Ok(())
	}

	/// Split the buffer into mutable chunks of size `2^log_chunk_size`.
	///
	/// # Throws
	///
	/// * [`Error::ArgumentRangeError`] if `log_chunk_size < P::LOG_WIDTH` or `log_chunk_size >
	///   log_len`.
	pub fn chunks_mut(
		&mut self,
		log_chunk_size: usize,
	) -> Result<impl Iterator<Item = FieldSliceMut<'_, P>>, Error> {
		if log_chunk_size < P::LOG_WIDTH || log_chunk_size > self.log_len {
			return Err(Error::ArgumentRangeError {
				arg: "log_chunk_size".to_string(),
				range: P::LOG_WIDTH..self.log_len + 1,
			});
		}

		let chunk_count = 1 << (self.log_len - log_chunk_size);
		let packed_chunk_size = 1 << log_chunk_size.saturating_sub(P::LOG_WIDTH);
		let chunks = self
			.values
			.chunks_mut(packed_chunk_size)
			.take(chunk_count)
			.map(move |chunk| FieldBuffer {
				log_len: log_chunk_size,
				values: FieldSliceDataMut::Slice(chunk),
			});

		Ok(chunks)
	}

	/// Get a mutable aligned chunk of size `2^log_chunk_size`.
	///
	/// This method behaves like [`FieldBuffer::chunk`] but returns a mutable reference.
	/// For small chunks (log_chunk_size < P::LOG_WIDTH), this returns a wrapper that
	/// implements deferred writes - modifications are written back when the wrapper is dropped.
	///
	/// # Throws
	///
	/// * [`Error::ArgumentRangeError`] if `log_chunk_size > log_len` or `chunk_index` is out of
	///   range.
	pub fn chunk_mut(
		&mut self,
		log_chunk_size: usize,
		chunk_index: usize,
	) -> Result<FieldBufferChunkMut<'_, P>, Error> {
		if log_chunk_size > self.log_len {
			return Err(Error::ArgumentRangeError {
				arg: "log_chunk_size".to_string(),
				range: 0..self.log_len + 1,
			});
		}

		let chunk_count = 1 << (self.log_len - log_chunk_size);
		if chunk_index >= chunk_count {
			return Err(Error::ArgumentRangeError {
				arg: "chunk_index".to_string(),
				range: 0..chunk_count,
			});
		}

		let inner = if log_chunk_size >= P::LOG_WIDTH {
			// Large chunk: return a mutable slice directly
			let packed_log_chunk_size = log_chunk_size - P::LOG_WIDTH;
			let chunk = &mut self.values[chunk_index << packed_log_chunk_size..]
				[..1 << packed_log_chunk_size];
			FieldBufferChunkMutInner::Slice {
				log_len: log_chunk_size,
				chunk,
			}
		} else {
			// Small chunk: extract from a single packed element and defer writes
			let packed_log_chunks = P::LOG_WIDTH - log_chunk_size;
			let packed_index = chunk_index >> packed_log_chunks;
			let chunk_subindex = chunk_index & ((1 << packed_log_chunks) - 1);
			let chunk_offset = chunk_subindex << log_chunk_size;

			let packed = self.values[packed_index];
			let chunk =
				P::from_scalars((0..1 << log_chunk_size).map(|i| packed.get(chunk_offset | i)));

			FieldBufferChunkMutInner::Single {
				log_len: log_chunk_size,
				chunk,
				parent: &mut self.values[packed_index],
				chunk_offset,
			}
		};

		Ok(FieldBufferChunkMut(inner))
	}

	/// Splits the buffer in half and calls a closure with the two halves.
	///
	/// If the buffer contains a single packed element that needs to be split,
	/// this method will create temporary copies, call the closure, and then
	/// write the results back to the original buffer.
	///
	/// # Throws
	///
	/// * [`Error::CannotSplit`] if `self.log_len() == 0`
	pub fn split_half_mut<F, R>(&mut self, f: F) -> Result<R, Error>
	where
		F: FnOnce(&mut FieldSliceMut<'_, P>, &mut FieldSliceMut<'_, P>) -> R,
	{
		if self.log_len == 0 {
			return Err(Error::CannotSplit);
		}

		let new_log_len = self.log_len - 1;

		if new_log_len < P::LOG_WIDTH {
			// Extract the values using interleave
			let packed = self.values[0];
			let zeros = P::default();
			let (mut first_half, mut second_half) = packed.interleave(zeros, new_log_len);

			// Create temporary buffers
			//
			// We use Slice instead of Single variants because:
			// 1. The closure takes ownership of the FieldBuffers
			// 2. We need to retrieve the potentially modified values after the closure runs
			// 3. With arrays, we maintain access to the modified values through the array
			let mut first = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceDataMut::Slice(slice::from_mut(&mut first_half)),
			};
			let mut second = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceDataMut::Slice(slice::from_mut(&mut second_half)),
			};

			// Call the closure with the temporary buffers
			let result = f(&mut first, &mut second);

			// Write back the results by interleaving them back together
			// The arrays may have been modified by the closure
			(self.values[0], _) = first_half.interleave(second_half, new_log_len);

			Ok(result)
		} else {
			// Normal case: split the packed values slice in half
			let half_len = 1 << (new_log_len - P::LOG_WIDTH);
			let (first_half, second_half) = self.values.split_at_mut(half_len);
			let second_half = &mut second_half[..half_len];

			let mut first = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceDataMut::Slice(first_half),
			};
			let mut second = FieldBuffer {
				log_len: new_log_len,
				values: FieldSliceDataMut::Slice(second_half),
			};

			Ok(f(&mut first, &mut second))
		}
	}

	/// Splits the buffer in half and returns a [`FieldBufferSplitMut`] for accessing the halves.
	///
	/// This returns an object that can be used to access mutable references to the two halves.
	/// This method unfortunately can't simply return a tuple of slices because the buffer may have
	/// only one packed element. If the buffer contains a single packed element that needs to be
	/// split, this method will create temporary copies, call the closure, and then write the
	/// results back to the original buffer when the returned [`FieldBufferSplitMut`] is dropped.
	///
	/// # Throws
	///
	/// * [`Error::CannotSplit`] if `self.log_len() == 0`
	pub fn split_half_mut_no_closure(&mut self) -> Result<FieldBufferSplitMut<'_, P>, Error> {
		if self.log_len == 0 {
			return Err(Error::CannotSplit);
		}

		let new_log_len = self.log_len - 1;
		if new_log_len < P::LOG_WIDTH {
			// Extract the values using interleave
			let packed = self.values[0];
			let zeros = P::default();
			let (lo_half, hi_half) = packed.interleave(zeros, new_log_len);

			Ok(FieldBufferSplitMut(FieldBufferSplitMutInner::Singles {
				log_len: new_log_len,
				lo_half,
				hi_half,
				parent: &mut self.values[0],
			}))
		} else {
			// Normal case: split the packed values slice in half
			let half_len = 1 << (new_log_len - P::LOG_WIDTH);
			let (lo_half, hi_half) = self.values.split_at_mut(half_len);
			let hi_half = &mut hi_half[..half_len];

			Ok(FieldBufferSplitMut(FieldBufferSplitMutInner::Slices {
				log_len: new_log_len,
				lo_half,
				hi_half,
			}))
		}
	}
}

impl<P: PackedField, Data: Deref<Target = [P]>> AsRef<[P]> for FieldBuffer<P, Data> {
	#[inline]
	fn as_ref(&self) -> &[P] {
		&self.values[..1 << self.log_len.saturating_sub(P::LOG_WIDTH)]
	}
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> AsMut<[P]> for FieldBuffer<P, Data> {
	#[inline]
	fn as_mut(&mut self) -> &mut [P] {
		&mut self.values[..1 << self.log_len.saturating_sub(P::LOG_WIDTH)]
	}
}

impl<F: Field, Data: Deref<Target = [F]>> Index<usize> for FieldBuffer<F, Data> {
	type Output = F;

	fn index(&self, index: usize) -> &Self::Output {
		&self.values[index]
	}
}

impl<F: Field, Data: DerefMut<Target = [F]>> IndexMut<usize> for FieldBuffer<F, Data> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.values[index]
	}
}

/// Alias for a field buffer over a borrowed slice.
pub type FieldSlice<'a, P> = FieldBuffer<P, FieldSliceData<'a, P>>;

/// Alias for a field buffer over a mutably borrowed slice.
pub type FieldSliceMut<'a, P> = FieldBuffer<P, FieldSliceDataMut<'a, P>>;

impl<'a, P: PackedField> FieldSlice<'a, P> {
	/// Create a new FieldSlice from a slice of packed values.
	///
	/// # Throws
	///
	/// * `IncorrectArgumentLength` if the number of field elements does not fit the `slice.len()`
	///   exactly.
	pub fn from_slice(log_len: usize, slice: &'a [P]) -> Result<Self, Error> {
		FieldBuffer::new(log_len, FieldSliceData::Slice(slice))
	}
}

impl<'a, P: PackedField, Data: Deref<Target = [P]>> From<&'a FieldBuffer<P, Data>>
	for FieldSlice<'a, P>
{
	fn from(buffer: &'a FieldBuffer<P, Data>) -> Self {
		buffer.to_ref()
	}
}

impl<'a, P: PackedField> FieldSliceMut<'a, P> {
	/// Create a new FieldSliceMut from a mutable slice of packed values.
	///
	/// # Throws
	///
	/// * `IncorrectArgumentLength` if the number of field elements does not fit the `slice.len()`
	///   exactly.
	pub fn from_slice(log_len: usize, slice: &'a mut [P]) -> Result<Self, Error> {
		FieldBuffer::new(log_len, FieldSliceDataMut::Slice(slice))
	}
}

impl<'a, P: PackedField, Data: DerefMut<Target = [P]>> From<&'a mut FieldBuffer<P, Data>>
	for FieldSliceMut<'a, P>
{
	fn from(buffer: &'a mut FieldBuffer<P, Data>) -> Self {
		buffer.to_mut()
	}
}

#[derive(Debug)]
pub enum FieldSliceData<'a, P> {
	Single(P),
	Slice(&'a [P]),
}

impl<'a, P> Deref for FieldSliceData<'a, P> {
	type Target = [P];

	fn deref(&self) -> &Self::Target {
		match self {
			FieldSliceData::Single(val) => slice::from_ref(val),
			FieldSliceData::Slice(slice) => slice,
		}
	}
}

#[derive(Debug)]
pub enum FieldSliceDataMut<'a, P> {
	Single(P),
	Slice(&'a mut [P]),
}

impl<'a, P> Deref for FieldSliceDataMut<'a, P> {
	type Target = [P];

	fn deref(&self) -> &Self::Target {
		match self {
			FieldSliceDataMut::Single(val) => slice::from_ref(val),
			FieldSliceDataMut::Slice(slice) => slice,
		}
	}
}

impl<'a, P> DerefMut for FieldSliceDataMut<'a, P> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		match self {
			FieldSliceDataMut::Single(val) => slice::from_mut(val),
			FieldSliceDataMut::Slice(slice) => slice,
		}
	}
}

/// Return type of [`FieldBuffer::split_half_mut_no_closure`].
#[derive(Debug)]
pub struct FieldBufferSplitMut<'a, P: PackedField>(FieldBufferSplitMutInner<'a, P>);

impl<'a, P: PackedField> FieldBufferSplitMut<'a, P> {
	pub fn halves(&mut self) -> (FieldSliceMut<'_, P>, FieldSliceMut<'_, P>) {
		match &mut self.0 {
			FieldBufferSplitMutInner::Singles {
				log_len,
				lo_half,
				hi_half,
				parent: _,
			} => (
				FieldBuffer {
					log_len: *log_len,
					values: FieldSliceDataMut::Slice(slice::from_mut(lo_half)),
				},
				FieldBuffer {
					log_len: *log_len,
					values: FieldSliceDataMut::Slice(slice::from_mut(hi_half)),
				},
			),
			FieldBufferSplitMutInner::Slices {
				log_len,
				lo_half,
				hi_half,
			} => (
				FieldBuffer {
					log_len: *log_len,
					values: FieldSliceDataMut::Slice(lo_half),
				},
				FieldBuffer {
					log_len: *log_len,
					values: FieldSliceDataMut::Slice(hi_half),
				},
			),
		}
	}
}

#[derive(Debug)]
enum FieldBufferSplitMutInner<'a, P: PackedField> {
	Singles {
		log_len: usize,
		lo_half: P,
		hi_half: P,
		parent: &'a mut P,
	},
	Slices {
		log_len: usize,
		lo_half: &'a mut [P],
		hi_half: &'a mut [P],
	},
}

impl<'a, P: PackedField> Drop for FieldBufferSplitMutInner<'a, P> {
	fn drop(&mut self) {
		match self {
			Self::Singles {
				log_len,
				lo_half,
				hi_half,
				parent,
			} => {
				// Write back the results by interleaving them back together
				// The arrays may have been modified by the closure
				(**parent, _) = (*lo_half).interleave(*hi_half, *log_len);
			}
			Self::Slices { .. } => {}
		}
	}
}

/// Return type of [`FieldBuffer::chunk_mut`] for small chunks.
#[derive(Debug)]
pub struct FieldBufferChunkMut<'a, P: PackedField>(FieldBufferChunkMutInner<'a, P>);

impl<'a, P: PackedField> FieldBufferChunkMut<'a, P> {
	pub fn get(&mut self) -> FieldSliceMut<'_, P> {
		match &mut self.0 {
			FieldBufferChunkMutInner::Single {
				log_len,
				chunk,
				parent: _,
				chunk_offset: _,
			} => FieldBuffer {
				log_len: *log_len,
				values: FieldSliceDataMut::Slice(slice::from_mut(chunk)),
			},
			FieldBufferChunkMutInner::Slice { log_len, chunk } => FieldBuffer {
				log_len: *log_len,
				values: FieldSliceDataMut::Slice(chunk),
			},
		}
	}
}

#[derive(Debug)]
enum FieldBufferChunkMutInner<'a, P: PackedField> {
	Single {
		log_len: usize,
		chunk: P,
		parent: &'a mut P,
		chunk_offset: usize,
	},
	Slice {
		log_len: usize,
		chunk: &'a mut [P],
	},
}

impl<'a, P: PackedField> Drop for FieldBufferChunkMutInner<'a, P> {
	fn drop(&mut self) {
		match self {
			Self::Single {
				log_len,
				chunk,
				parent,
				chunk_offset,
			} => {
				// Write back the modified chunk values to the parent packed element
				for i in 0..1 << *log_len {
					parent.set(*chunk_offset | i, chunk.get(i));
				}
			}
			Self::Slice { .. } => {}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::test_utils::{B128, Packed128b};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn test_zeros() {
		// Make a buffer with `zeros()` and check that all elements are zero.
		// Test with log_len >= LOG_WIDTH
		let buffer = FieldBuffer::<P>::zeros(6); // 64 elements
		assert_eq!(buffer.log_len(), 6);
		assert_eq!(buffer.len(), 64);

		// Check all elements are zero
		for i in 0..64 {
			assert_eq!(buffer.get_checked(i).unwrap(), F::ZERO);
		}

		// Test with log_len < LOG_WIDTH
		let buffer = FieldBuffer::<P>::zeros(1); // 2 elements
		assert_eq!(buffer.log_len(), 1);
		assert_eq!(buffer.len(), 2);

		// Check all elements are zero
		for i in 0..2 {
			assert_eq!(buffer.get_checked(i).unwrap(), F::ZERO);
		}
	}

	#[test]
	fn test_from_values_below_packing_width() {
		// Make a buffer using `from_values()`, where the number of scalars is below the packing
		// width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		let values = vec![F::new(1), F::new(2)]; // 2 elements < 4
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		assert_eq!(buffer.log_len(), 1); // log2(2) = 1
		assert_eq!(buffer.len(), 2);

		// Verify the values
		assert_eq!(buffer.get_checked(0).unwrap(), F::new(1));
		assert_eq!(buffer.get_checked(1).unwrap(), F::new(2));
	}

	#[test]
	fn test_from_values_above_packing_width() {
		// Make a buffer using `from_values()`, where the number of scalars is above the packing
		// width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		let values: Vec<F> = (0..16).map(F::new).collect(); // 16 elements > 4
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		assert_eq!(buffer.log_len(), 4); // log2(16) = 4
		assert_eq!(buffer.len(), 16);

		// Verify all values
		for i in 0..16 {
			assert_eq!(buffer.get_checked(i).unwrap(), F::new(i as u128));
		}
	}

	#[test]
	fn test_from_values_non_power_of_two() {
		// Fail to make a buffer using `from_values()`, where the number of scalars is not a power
		// of two
		let values: Vec<F> = (0..7).map(F::new).collect(); // 7 is not a power of two
		let result = FieldBuffer::<P>::from_values(&values);

		assert!(matches!(result, Err(Error::PowerOfTwoLengthRequired)));

		// Also test with 0 elements
		let values: Vec<F> = vec![];
		let result = FieldBuffer::<P>::from_values(&values);
		assert!(matches!(result, Err(Error::PowerOfTwoLengthRequired)));
	}

	#[test]
	fn test_new_below_packing_width() {
		// Make a buffer using `new()`, where the number of scalars is below the packing
		// width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		// For log_len = 1 (2 elements), we need 1 packed value
		let mut packed_values = vec![P::default()];
		let mut buffer = FieldBuffer::new(1, packed_values.as_mut_slice()).unwrap();

		assert_eq!(buffer.log_len(), 1);
		assert_eq!(buffer.len(), 2);

		// Set and verify values
		buffer.set_checked(0, F::new(10)).unwrap();
		buffer.set_checked(1, F::new(20)).unwrap();
		assert_eq!(buffer.get_checked(0).unwrap(), F::new(10));
		assert_eq!(buffer.get_checked(1).unwrap(), F::new(20));
	}

	#[test]
	fn test_new_above_packing_width() {
		// Make a buffer using `new()`, where the number of scalars is above the packing
		// width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		// For log_len = 4 (16 elements), we need 4 packed values
		let mut packed_values = vec![P::default(); 4];
		let mut buffer = FieldBuffer::new(4, packed_values.as_mut_slice()).unwrap();

		assert_eq!(buffer.log_len(), 4);
		assert_eq!(buffer.len(), 16);

		// Set and verify values
		for i in 0..16 {
			buffer.set_checked(i, F::new(i as u128 * 10)).unwrap();
		}
		for i in 0..16 {
			assert_eq!(buffer.get_checked(i).unwrap(), F::new(i as u128 * 10));
		}
	}

	#[test]
	fn test_new_non_power_of_two() {
		// Fail to make a buffer using `new()`, where the number of scalars is not a power of two
		// For log_len = 4 (16 elements), we need 4 packed values
		// Provide wrong number of packed values
		let packed_values = vec![P::default(); 3]; // Wrong: should be 4
		let result = FieldBuffer::new(4, packed_values.as_slice());

		assert!(matches!(result, Err(Error::IncorrectArgumentLength { .. })));

		// Another test with too many packed values
		let packed_values = vec![P::default(); 5]; // Wrong: should be 4
		let result = FieldBuffer::new(4, packed_values.as_slice());

		assert!(matches!(result, Err(Error::IncorrectArgumentLength { .. })));
	}

	#[test]
	fn test_get_set() {
		let mut buffer = FieldBuffer::<P>::zeros(3); // 8 elements

		// Set some values
		for i in 0..8 {
			buffer.set_checked(i, F::new(i as u128)).unwrap();
		}

		// Get them back
		for i in 0..8 {
			assert_eq!(buffer.get_checked(i).unwrap(), F::new(i as u128));
		}

		// Test out of bounds
		assert!(buffer.get_checked(8).is_err());
		assert!(buffer.set_checked(8, F::new(0)).is_err());
	}

	#[test]
	fn test_chunk() {
		let log_len = 8;
		let values: Vec<F> = (0..1 << log_len).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		// Test invalid chunk size (too large)
		assert!(buffer.chunk(log_len + 1, 0).is_err());

		for log_chunk_size in 0..=log_len {
			let chunk_count = 1 << (log_len - log_chunk_size);

			// Test invalid chunk index
			assert!(buffer.chunk(log_chunk_size, chunk_count).is_err());

			for chunk_index in 0..chunk_count {
				let chunk = buffer.chunk(log_chunk_size, chunk_index).unwrap();
				for i in 0..1 << log_chunk_size {
					assert_eq!(
						chunk.get_checked(i).unwrap(),
						buffer
							.get_checked(chunk_index << log_chunk_size | i)
							.unwrap()
					);
				}
			}
		}
	}

	#[test]
	fn test_chunk_mut() {
		let log_len = 8;
		let mut buffer = FieldBuffer::<P>::zeros(log_len);

		// Initialize with test data
		for i in 0..1 << log_len {
			buffer.set_checked(i, F::new(i as u128)).unwrap();
		}

		// Test invalid chunk size (too large)
		assert!(buffer.chunk_mut(log_len + 1, 0).is_err());

		// Test mutations for different chunk sizes
		for log_chunk_size in 0..=log_len {
			let chunk_count = 1 << (log_len - log_chunk_size);

			// Test invalid chunk index
			assert!(buffer.chunk_mut(log_chunk_size, chunk_count).is_err());

			// Modify each chunk by multiplying by 10
			for chunk_index in 0..chunk_count {
				let mut chunk_wrapper = buffer.chunk_mut(log_chunk_size, chunk_index).unwrap();
				let mut chunk = chunk_wrapper.get();
				for i in 0..1 << log_chunk_size {
					let old_val = chunk.get_checked(i).unwrap();
					chunk.set_checked(i, F::new(old_val.val() * 10)).unwrap();
				}
				// chunk_wrapper drops here and writes back changes
			}

			// Verify modifications
			for chunk_index in 0..chunk_count {
				for i in 0..1 << log_chunk_size {
					let index = chunk_index << log_chunk_size | i;
					let expected = F::new((index as u128) * 10);
					assert_eq!(
						buffer.get_checked(index).unwrap(),
						expected,
						"Failed at log_chunk_size={}, chunk_index={}, i={}",
						log_chunk_size,
						chunk_index,
						i
					);
				}
			}

			// Reset buffer for next iteration
			for i in 0..1 << log_len {
				buffer.set_checked(i, F::new(i as u128)).unwrap();
			}
		}

		// Test large chunks (log_chunk_size >= P::LOG_WIDTH)
		let mut buffer = FieldBuffer::<P>::zeros(6);
		for i in 0..64 {
			buffer.set_checked(i, F::new(i as u128)).unwrap();
		}

		// Modify first chunk of size 16 (log_chunk_size = 4 >= P::LOG_WIDTH = 2)
		{
			let mut chunk_wrapper = buffer.chunk_mut(4, 0).unwrap();
			let mut chunk = chunk_wrapper.get();
			for i in 0..16 {
				chunk.set_checked(i, F::new(100 + i as u128)).unwrap();
			}
		}

		// Verify large chunk modifications
		for i in 0..16 {
			assert_eq!(buffer.get_checked(i).unwrap(), F::new(100 + i as u128));
		}
		for i in 16..64 {
			assert_eq!(buffer.get_checked(i).unwrap(), F::new(i as u128));
		}

		// Test small chunks (log_chunk_size < P::LOG_WIDTH)
		let mut buffer = FieldBuffer::<P>::zeros(3);
		for i in 0..8 {
			buffer.set_checked(i, F::new(i as u128)).unwrap();
		}

		// Modify third chunk of size 1 (log_chunk_size = 0 < P::LOG_WIDTH = 2)
		{
			let mut chunk_wrapper = buffer.chunk_mut(0, 3).unwrap();
			let mut chunk = chunk_wrapper.get();
			chunk.set_checked(0, F::new(42)).unwrap();
		}

		// Verify small chunk modifications
		for i in 0..8 {
			let expected = if i == 3 {
				F::new(42)
			} else {
				F::new(i as u128)
			};
			assert_eq!(buffer.get_checked(i).unwrap(), expected);
		}
	}

	#[test]
	fn test_chunks() {
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		// Split into 4 chunks of size 4
		let chunks: Vec<_> = buffer.chunks(2).unwrap().collect();
		assert_eq!(chunks.len(), 4);

		for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
			assert_eq!(chunk.len(), 4);
			for i in 0..4 {
				let expected = F::new((chunk_idx * 4 + i) as u128);
				assert_eq!(chunk.get_checked(i).unwrap(), expected);
			}
		}

		// Test invalid chunk size (too large)
		assert!(buffer.chunks(5).is_err());

		// Test invalid chunk size (too small - below P::LOG_WIDTH)
		// P::LOG_WIDTH = 2, so chunks(0) and chunks(1) should fail
		assert!(buffer.chunks(0).is_err());
		assert!(buffer.chunks(1).is_err());
	}

	#[test]
	fn test_chunks_par() {
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		// Split into 4 chunks of size 4
		let chunks: Vec<_> = buffer.chunks_par(2).unwrap().collect();
		assert_eq!(chunks.len(), 4);

		for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
			assert_eq!(chunk.len(), 4);
			for i in 0..4 {
				let expected = F::new((chunk_idx * 4 + i) as u128);
				assert_eq!(chunk.get_checked(i).unwrap(), expected);
			}
		}

		// Test invalid chunk size (too large)
		assert!(buffer.chunks_par(5).is_err());

		// Test invalid chunk size (too small - below P::LOG_WIDTH)
		// P::LOG_WIDTH = 2, so chunks_par(0) and chunks_par(1) should fail
		assert!(buffer.chunks_par(0).is_err());
		assert!(buffer.chunks_par(1).is_err());
	}

	#[test]
	fn test_chunks_mut() {
		let mut buffer = FieldBuffer::<P>::zeros(4); // 16 elements

		// Modify via chunks
		let mut chunks: Vec<_> = buffer.chunks_mut(2).unwrap().collect();
		assert_eq!(chunks.len(), 4);

		for (chunk_idx, chunk) in chunks.iter_mut().enumerate() {
			for i in 0..chunk.len() {
				chunk
					.set_checked(i, F::new((chunk_idx * 10 + i) as u128))
					.unwrap();
			}
		}

		// Verify modifications
		for chunk_idx in 0..4 {
			for i in 0..4 {
				let expected = F::new((chunk_idx * 10 + i) as u128);
				assert_eq!(buffer.get_checked(chunk_idx * 4 + i).unwrap(), expected);
			}
		}

		// Test invalid chunk size (too small - below P::LOG_WIDTH)
		assert!(buffer.chunks_mut(0).is_err());
		assert!(buffer.chunks_mut(1).is_err());
	}

	#[test]
	fn test_to_ref_to_mut() {
		let mut buffer = FieldBuffer::<P>::zeros_truncated(3, 5).unwrap();

		// Test to_ref
		let slice_ref = buffer.to_ref();
		assert_eq!(slice_ref.len(), buffer.len());
		assert_eq!(slice_ref.log_len(), buffer.log_len());
		assert_eq!(slice_ref.as_ref().len(), 1 << slice_ref.log_len().saturating_sub(P::LOG_WIDTH));

		// Test to_mut
		let mut slice_mut = buffer.to_mut();
		slice_mut.set_checked(0, F::new(123)).unwrap();
		assert_eq!(slice_mut.as_mut().len(), 1 << slice_mut.log_len().saturating_sub(P::LOG_WIDTH));
		assert_eq!(buffer.get_checked(0).unwrap(), F::new(123));
	}

	#[test]
	fn test_split_half() {
		// Test with buffer size > P::WIDTH (multiple packed elements)
		let values: Vec<F> = (0..16).map(F::new).collect();
		// Leave spare capacity for 32 elements
		let buffer = FieldBuffer::<P>::from_values_truncated(&values, 5).unwrap();

		let (first, second) = buffer.split_half().unwrap();
		assert_eq!(first.len(), 8);
		assert_eq!(second.len(), 8);

		// Verify values
		for i in 0..8 {
			assert_eq!(first.get_checked(i).unwrap(), F::new(i as u128));
			assert_eq!(second.get_checked(i).unwrap(), F::new((i + 8) as u128));
		}

		// Test with buffer size = P::WIDTH (single packed element)
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		// Note that underlying collection has two packed fields.
		let values: Vec<F> = (0..4).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values_truncated(&values, 3).unwrap();

		let (first, second) = buffer.split_half().unwrap();
		assert_eq!(first.len(), 2);
		assert_eq!(second.len(), 2);

		// Verify we got Single variants
		match &first.values {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for first half"),
		}
		match &second.values {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for second half"),
		}

		// Verify values
		assert_eq!(first.get_checked(0).unwrap(), F::new(0));
		assert_eq!(first.get_checked(1).unwrap(), F::new(1));
		assert_eq!(second.get_checked(0).unwrap(), F::new(2));
		assert_eq!(second.get_checked(1).unwrap(), F::new(3));

		// Test with buffer size = 2 (less than P::WIDTH)
		let values: Vec<F> = vec![F::new(10), F::new(20)];
		let buffer = FieldBuffer::<P>::from_values_truncated(&values, 3).unwrap();

		let (first, second) = buffer.split_half().unwrap();
		assert_eq!(first.len(), 1);
		assert_eq!(second.len(), 1);

		// Verify we got Single variants
		match &first.values {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for first half"),
		}
		match &second.values {
			FieldSliceData::Single(_) => {}
			_ => panic!("Expected Single variant for second half"),
		}

		assert_eq!(first.get_checked(0).unwrap(), F::new(10));
		assert_eq!(second.get_checked(0).unwrap(), F::new(20));

		// Test error case: buffer of size 1
		let values = vec![F::new(42)];
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		let result = buffer.split_half();
		assert!(matches!(result, Err(Error::CannotSplit)));
	}

	#[test]
	fn test_split_half_mut() {
		// Test with buffer size > P::WIDTH (multiple packed elements)
		let mut buffer = FieldBuffer::<P>::zeros_truncated(4, 5).unwrap(); // 16 elements, 32 element capacity

		// Fill with test data
		for i in 0..16 {
			buffer.set_checked(i, F::new(i as u128)).unwrap();
		}

		buffer
			.split_half_mut(|first, second| {
				assert_eq!(first.len(), 8);
				assert_eq!(second.len(), 8);

				// Modify through the split halves
				for i in 0..8 {
					first.set_checked(i, F::new((i * 10) as u128)).unwrap();
					second.set_checked(i, F::new((i * 20) as u128)).unwrap();
				}
			})
			.unwrap();

		// Verify changes were made to original buffer
		for i in 0..8 {
			assert_eq!(buffer.get_checked(i).unwrap(), F::new((i * 10) as u128));
			assert_eq!(buffer.get_checked(i + 8).unwrap(), F::new((i * 20) as u128));
		}

		// Test with buffer size = P::WIDTH (single packed element)
		// P::LOG_WIDTH = 2, so a buffer with log_len = 2 (4 elements) can now be split
		let mut buffer = FieldBuffer::<P>::zeros_truncated(2, 4).unwrap(); // 4 elements, 16 element capacity

		// Fill with test data
		for i in 0..4 {
			buffer.set_checked(i, F::new(i as u128)).unwrap();
		}

		buffer
			.split_half_mut(|first, second| {
				assert_eq!(first.len(), 2);
				assert_eq!(second.len(), 2);

				// Modify values
				first.set_checked(0, F::new(100)).unwrap();
				first.set_checked(1, F::new(101)).unwrap();
				second.set_checked(0, F::new(200)).unwrap();
				second.set_checked(1, F::new(201)).unwrap();
			})
			.unwrap();

		// Verify changes were written back
		assert_eq!(buffer.get_checked(0).unwrap(), F::new(100));
		assert_eq!(buffer.get_checked(1).unwrap(), F::new(101));
		assert_eq!(buffer.get_checked(2).unwrap(), F::new(200));
		assert_eq!(buffer.get_checked(3).unwrap(), F::new(201));

		// Test with buffer size = 2
		let mut buffer = FieldBuffer::<P>::zeros_truncated(1, 4).unwrap(); // 2 elements, 16 element capacity

		buffer.set_checked(0, F::new(10)).unwrap();
		buffer.set_checked(1, F::new(20)).unwrap();

		buffer
			.split_half_mut(|first, second| {
				assert_eq!(first.len(), 1);
				assert_eq!(second.len(), 1);

				// Modify values
				first.set_checked(0, F::new(30)).unwrap();
				second.set_checked(0, F::new(40)).unwrap();
			})
			.unwrap();

		// Verify changes
		assert_eq!(buffer.get_checked(0).unwrap(), F::new(30));
		assert_eq!(buffer.get_checked(1).unwrap(), F::new(40));

		// Test error case: buffer of size 1
		let mut buffer = FieldBuffer::<P>::zeros(0); // 1 element

		let result = buffer.split_half_mut(|_, _| {});
		assert!(matches!(result, Err(Error::CannotSplit)));
	}

	#[test]
	fn test_zero_extend() {
		let log_len = 10;
		let nonzero_scalars = (0..1 << log_len).map(|i| F::new(i + 1)).collect::<Vec<_>>();
		let mut buffer = FieldBuffer::<P>::from_values(&nonzero_scalars).unwrap();
		buffer.truncate(0);

		for i in 0..log_len {
			buffer.zero_extend(i + 1).unwrap();

			for j in 1 << i..1 << (i + 1) {
				assert!(buffer.get_checked(j).unwrap().is_zero());
			}
		}
	}

	#[test]
	fn test_resize() {
		let mut buffer = FieldBuffer::<P>::zeros(4); // 16 elements

		// Fill with test data
		for i in 0..16 {
			buffer.set_checked(i, F::new(i as u128)).unwrap();
		}

		buffer.resize(3).unwrap();
		assert_eq!(buffer.log_len(), 3);
		assert_eq!(buffer.get_checked(7).unwrap(), F::new(7));

		buffer.resize(4).unwrap();
		assert_eq!(buffer.log_len(), 4);
		assert_eq!(buffer.get_checked(15).unwrap(), F::new(15));

		assert!(
			matches!(buffer.resize(5), Err(Error::IncorrectArgumentLength { arg, expected }) if arg == "new_log_len" && expected == 4)
		);

		buffer.resize(2).unwrap();
		assert_eq!(buffer.log_len(), 2);
	}

	#[test]
	fn test_iter_scalars() {
		// Test with buffer size below packing width
		// P::LOG_WIDTH = 2, so P::WIDTH = 4
		let values = vec![F::new(10), F::new(20)]; // 2 elements < 4
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Verify it matches individual get calls
		for (i, &val) in collected.iter().enumerate() {
			assert_eq!(val, buffer.get(i));
		}

		// Test with buffer size equal to packing width
		let values = vec![F::new(1), F::new(2), F::new(3), F::new(4)]; // 4 elements = P::WIDTH
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Test with buffer size above packing width
		let values: Vec<F> = (0..16).map(F::new).collect(); // 16 elements > 4
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Verify it matches individual get calls
		for (i, &val) in collected.iter().enumerate() {
			assert_eq!(val, buffer.get(i));
		}

		// Test with single element buffer
		let values = vec![F::new(42)];
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Test with large buffer
		let values: Vec<F> = (0..256).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);

		// Test that iterator is cloneable and can be used multiple times
		let values: Vec<F> = (0..8).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values).unwrap();

		let iter1 = buffer.iter_scalars();
		let iter2 = iter1.clone();

		let collected1: Vec<F> = iter1.collect();
		let collected2: Vec<F> = iter2.collect();
		assert_eq!(collected1, collected2);
		assert_eq!(collected1, values);

		// Test with buffer that has extra capacity
		let values: Vec<F> = (0..8).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values_truncated(&values, 5).unwrap(); // 8 elements, capacity for 32

		let collected: Vec<F> = buffer.iter_scalars().collect();
		assert_eq!(collected, values);
		assert_eq!(collected.len(), 8); // Should only iterate over actual elements, not capacity
	}

	#[test]
	fn test_split_half_mut_no_closure() {
		// Test with buffer size > P::WIDTH (multiple packed elements)
		let mut buffer = FieldBuffer::<P>::zeros(4); // 16 elements

		// Fill with test data
		for i in 0..16 {
			buffer.set_checked(i, F::new(i as u128)).unwrap();
		}

		{
			let mut split = buffer.split_half_mut_no_closure().unwrap();
			let (mut first, mut second) = split.halves();

			assert_eq!(first.len(), 8);
			assert_eq!(second.len(), 8);

			// Modify through the split halves
			for i in 0..8 {
				first.set_checked(i, F::new((i * 10) as u128)).unwrap();
				second.set_checked(i, F::new((i * 20) as u128)).unwrap();
			}
			// split drops here and writes back the changes
		}

		// Verify changes were made to original buffer
		for i in 0..8 {
			assert_eq!(buffer.get_checked(i).unwrap(), F::new((i * 10) as u128));
			assert_eq!(buffer.get_checked(i + 8).unwrap(), F::new((i * 20) as u128));
		}

		// Test with buffer size = P::WIDTH (single packed element)
		// P::LOG_WIDTH = 2, so a buffer with log_len = 2 (4 elements) can now be split
		let mut buffer = FieldBuffer::<P>::zeros(2); // 4 elements

		// Fill with test data
		for i in 0..4 {
			buffer.set_checked(i, F::new(i as u128)).unwrap();
		}

		{
			let mut split = buffer.split_half_mut_no_closure().unwrap();
			let (mut first, mut second) = split.halves();

			assert_eq!(first.len(), 2);
			assert_eq!(second.len(), 2);

			// Modify values
			first.set_checked(0, F::new(100)).unwrap();
			first.set_checked(1, F::new(101)).unwrap();
			second.set_checked(0, F::new(200)).unwrap();
			second.set_checked(1, F::new(201)).unwrap();
			// split drops here and writes back the changes using interleave
		}

		// Verify changes were written back
		assert_eq!(buffer.get_checked(0).unwrap(), F::new(100));
		assert_eq!(buffer.get_checked(1).unwrap(), F::new(101));
		assert_eq!(buffer.get_checked(2).unwrap(), F::new(200));
		assert_eq!(buffer.get_checked(3).unwrap(), F::new(201));

		// Test with buffer size = 2
		let mut buffer = FieldBuffer::<P>::zeros(1); // 2 elements

		buffer.set_checked(0, F::new(10)).unwrap();
		buffer.set_checked(1, F::new(20)).unwrap();

		{
			let mut split = buffer.split_half_mut_no_closure().unwrap();
			let (mut first, mut second) = split.halves();

			assert_eq!(first.len(), 1);
			assert_eq!(second.len(), 1);

			// Modify values
			first.set_checked(0, F::new(30)).unwrap();
			second.set_checked(0, F::new(40)).unwrap();
			// split drops here and writes back the changes using interleave
		}

		// Verify changes
		assert_eq!(buffer.get_checked(0).unwrap(), F::new(30));
		assert_eq!(buffer.get_checked(1).unwrap(), F::new(40));

		// Test error case: buffer of size 1
		let mut buffer = FieldBuffer::<P>::zeros(0); // 1 element

		let result = buffer.split_half_mut_no_closure();
		assert!(matches!(result, Err(Error::CannotSplit)));
	}
}
