// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Guards for writing through a region of a field buffer narrower than one packed word.
//!
//! Such a region cannot be lent out as a mutable slice of packed words.
//! The two halves of a split share a word, and a small chunk occupies only some of a word's lanes.
//!
//! Each guard therefore works in three steps:
//!
//! ```text
//! detach   copy the region out into owned words
//! lend     hand out mutable views over those owned words
//! merge    fold the words back into the parent word when the guard drops
//! ```

use std::{ops::DerefMut, slice};

use binius_field::PackedField;

use super::{FieldBuffer, FieldSliceMut, chunks::SubWordChunk};

/// Guards the two halves of a buffer that was split along its highest variable.
///
/// Holds the parent store, and lends each half out as a mutable view.
#[derive(Debug)]
pub struct FieldBufferSplitMut<P: PackedField, Data: DerefMut<Target = [P]>> {
	pub(super) log_len: usize,
	pub(super) singles: Option<[P; 2]>,
	pub(super) data: Data,
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> FieldBufferSplitMut<P, Data> {
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

impl<P: PackedField, Data: DerefMut<Target = [P]>> Drop for FieldBufferSplitMut<P, Data> {
	fn drop(&mut self) {
		// Detached halves are the only shape with anything to write back.
		if let Some([lo_half, hi_half]) = self.singles {
			(self.data[0], _) = lo_half.interleave(hi_half, self.log_len);
		}
	}
}

/// Guards one mutably borrowed chunk of a buffer.
///
/// A chunk of at least one packed word is lent straight from the store.
/// A smaller one is detached into an owned word and merged back on drop.
#[derive(Debug)]
pub struct FieldBufferChunkMut<'a, P: PackedField>(FieldBufferChunkMutInner<'a, P>);

impl<'a, P: PackedField> FieldBufferChunkMut<'a, P> {
	/// Guards a chunk narrower than a packed word, already detached out of `parent`.
	pub(super) const fn detached(location: SubWordChunk<P>, chunk: P, parent: &'a mut P) -> Self {
		Self(FieldBufferChunkMutInner::Single {
			location,
			chunk,
			parent,
		})
	}

	/// Guards a chunk of whole words, lent straight from the store.
	pub(super) const fn borrowed(log_len: usize, chunk: &'a mut [P]) -> Self {
		Self(FieldBufferChunkMutInner::Slice { log_len, chunk })
	}

	/// Lends the chunk out as a mutable view.
	///
	/// A chunk of whole words is a view straight onto the store, so edits land at once.
	/// A narrower chunk is a view onto a detached word, so edits land when this guard drops.
	pub const fn get(&mut self) -> FieldSliceMut<'_, P> {
		match &mut self.0 {
			FieldBufferChunkMutInner::Single {
				location,
				chunk,
				parent: _,
			} => FieldBuffer {
				log_len: location.log_len(),
				values: slice::from_mut(chunk),
			},
			FieldBufferChunkMutInner::Slice { log_len, chunk } => FieldBuffer {
				log_len: *log_len,
				values: chunk,
			},
		}
	}
}

impl<P: PackedField> Drop for FieldBufferChunkMut<'_, P> {
	fn drop(&mut self) {
		match &mut self.0 {
			FieldBufferChunkMutInner::Single {
				location,
				chunk,
				parent,
			} => location.merge_into(parent, chunk),
			// A chunk lent from the store was edited in place, so there is nothing to merge.
			FieldBufferChunkMutInner::Slice { .. } => {}
		}
	}
}

/// The two shapes a mutably borrowed chunk takes, decided by its size against the packing width.
#[derive(Debug)]
enum FieldBufferChunkMutInner<'a, P: PackedField> {
	Single {
		/// Which lanes of the parent word the chunk occupies.
		location: SubWordChunk<P>,
		/// The detached copy the caller edits.
		chunk: P,
		/// The word the copy is merged back into.
		parent: &'a mut P,
	},
	Slice {
		log_len: usize,
		chunk: &'a mut [P],
	},
}
