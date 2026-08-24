// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Guard for writing through halves of a field buffer narrower than one packed word.
//!
//! Two such halves share a single word, so neither can be lent out as a mutable slice of words.
//! The guard therefore works in three steps:
//!
//! ```text
//! detach   copy each half out into a word of its own
//! lend     hand out mutable views over those owned words
//! merge    interleave the words back into the parent word when the guard drops
//! ```

use std::{ops::DerefMut, slice};

use binius_field::PackedField;

use super::{FieldBuffer, FieldSliceMut};

/// Guards the two halves of a buffer that was split along its highest variable.
///
/// Holds the parent store, and lends each half out as a mutable view.
#[derive(Debug)]
pub struct SplitMut<P: PackedField, Data: DerefMut<Target = [P]>> {
	pub(super) log_len: usize,
	pub(super) singles: Option<[P; 2]>,
	pub(super) data: Data,
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> SplitMut<P, Data> {
	/// Lends the two halves out as mutable views, the low half first.
	///
	/// A half of whole words is a view straight onto the store, so edits land at once.
	/// A narrower half is a view onto a detached word, so edits land when this guard drops.
	pub fn halves(&mut self) -> (FieldSliceMut<'_, P>, FieldSliceMut<'_, P>) {
		match &mut self.singles {
			Some([lo_half, hi_half]) => (
				FieldBuffer {
					log_len: self.log_len,
					values: slice::from_mut(lo_half),
				},
				FieldBuffer {
					log_len: self.log_len,
					values: slice::from_mut(hi_half),
				},
			),
			None => {
				let half_len = 1 << (self.log_len - P::LOG_WIDTH);
				let (lo_half, hi_half) = self.data.split_at_mut(half_len);
				(
					FieldBuffer {
						log_len: self.log_len,
						values: lo_half,
					},
					FieldBuffer {
						log_len: self.log_len,
						values: hi_half,
					},
				)
			}
		}
	}
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> Drop for SplitMut<P, Data> {
	fn drop(&mut self) {
		// Detached halves are the only shape with anything to write back.
		if let Some([lo_half, hi_half]) = self.singles {
			(self.data[0], _) = lo_half.interleave(hi_half, self.log_len);
		}
	}
}
