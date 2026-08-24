// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Borrowed views over a field buffer.
//!
//! A view is the same buffer type parameterized by a borrowed store.
//! So every method on an owned buffer is available on a view of one.

use std::{
	ops::{Deref, DerefMut},
	slice,
};

use binius_compute::Allocator;
use binius_field::PackedField;

use super::FieldBuffer;

/// A field buffer whose backing store is drawn from an allocator.
///
/// Over a buffer pool this is a recyclable pooled buffer.
/// Over the global allocator it is an ordinary vector-backed buffer.
pub type FieldVec<P, A> = FieldBuffer<P, <A as Allocator>::Vec<P>>;

/// Alias for a field buffer over a borrowed slice.
pub type FieldSlice<'a, P> = FieldBuffer<P, FieldSliceData<'a, P>>;

/// Alias for a field buffer over a mutably borrowed slice.
pub type FieldSliceMut<'a, P> = FieldBuffer<P, &'a mut [P]>;

impl<'a, P: PackedField> FieldSlice<'a, P> {
	/// Create a new FieldSlice from a slice of packed words.
	///
	/// # Preconditions
	///
	/// * `slice.len()` must equal the expected packed length for `log_len`.
	#[track_caller]
	pub fn from_slice(log_len: usize, slice: &'a [P]) -> Self {
		FieldBuffer::new(log_len, FieldSliceData::Slice(slice))
	}
}

impl<'a, P: PackedField, Data: Deref<Target = [P]>> From<&'a FieldBuffer<P, Data>>
	for FieldSlice<'a, P>
{
	fn from(buffer: &'a FieldBuffer<P, Data>) -> Self {
		buffer.as_view()
	}
}

impl<'a, P: PackedField> FieldSliceMut<'a, P> {
	/// Create a new FieldSliceMut from a mutable slice of packed words.
	///
	/// # Preconditions
	///
	/// * `slice.len()` must equal the expected packed length for `log_len`.
	#[track_caller]
	pub fn from_slice(log_len: usize, slice: &'a mut [P]) -> Self {
		FieldBuffer::new(log_len, slice)
	}
}

impl<'a, P: PackedField, Data: DerefMut<Target = [P]>> From<&'a mut FieldBuffer<P, Data>>
	for FieldSliceMut<'a, P>
{
	fn from(buffer: &'a mut FieldBuffer<P, Data>) -> Self {
		buffer.as_mut_view()
	}
}

#[derive(Debug, Clone, Copy)]
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
