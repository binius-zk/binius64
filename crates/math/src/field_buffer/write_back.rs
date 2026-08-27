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
//!
//! A region that already spans whole words needs none of that.
//! Both guards then lend the store out directly, and their merge step does nothing.

use std::{ops::DerefMut, slice};

use binius_field::PackedField;

use super::{FieldBuffer, FieldSliceMut, chunks::SubWordChunk};

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
					words: slice::from_mut(lo_half),
				},
				FieldBuffer {
					log_len: self.log_len,
					words: slice::from_mut(hi_half),
				},
			),
			None => {
				let half_len = 1 << (self.log_len - P::LOG_WIDTH);
				let (lo_half, hi_half) = self.data.split_at_mut(half_len);
				(
					FieldBuffer {
						log_len: self.log_len,
						words: lo_half,
					},
					FieldBuffer {
						log_len: self.log_len,
						words: hi_half,
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

/// Guards one mutably borrowed chunk of a buffer.
///
/// The chunk size against the packing width decides which of two shapes the guard takes:
///
/// ```text
/// chunk >= one packed word  ->  a run of whole words, lent straight from the store
/// chunk <  one packed word  ->  the chunk's lanes copied into a word of their own
/// ```
///
/// The first shape edits the store itself.
/// Every edit is therefore already in place, and dropping the guard does nothing.
///
/// The second shape edits a copy taken when the guard is built.
/// That copy holds the chunk's lanes shifted down to start at lane 0.
/// Dropping the guard writes them back to the lanes they came from, and nothing else.
///
/// ```text
/// WIDTH = 4, chunks of 2 elements, chunk 1 of word 0
///
/// word before   [ a b c d ]
/// detached      [ c d . . ]   lanes 2..4 copied down to lanes 0..2
/// after edits   [ y z . . ]
/// word on drop  [ a b y z ]   lanes 0..2 written back to lanes 2..4
/// ```
///
/// So an edit to a sub-word chunk reaches the buffer when the guard drops, and not before.
/// Neighbouring chunks sharing the word keep their elements, since the merge skips their lanes.
///
/// Only one such guard can exist at a time, since it borrows the buffer mutably.
/// Two live guards over chunks of one word would each merge a stale copy.
/// The later merge would then undo the earlier one.
#[derive(Debug)]
pub struct ChunkMut<'a, P: PackedField>(ChunkMutInner<'a, P>);

impl<'a, P: PackedField> ChunkMut<'a, P> {
	/// Guards a chunk narrower than a packed word, already copied out of the word holding it.
	pub(super) const fn detached(location: SubWordChunk<P>, chunk: P, parent: &'a mut P) -> Self {
		Self(ChunkMutInner::Detached {
			location,
			chunk,
			parent,
		})
	}

	/// Guards a chunk of whole words, lent straight from the store.
	pub(super) const fn borrowed(log_len: usize, chunk: &'a mut [P]) -> Self {
		Self(ChunkMutInner::Borrowed { log_len, chunk })
	}

	/// Lends the chunk out as a mutable view, its first element at index 0.
	///
	/// A chunk of whole words is a view straight onto the store, so edits land at once.
	/// A narrower chunk is a view onto the detached copy, so edits land when this guard drops.
	pub const fn chunk(&mut self) -> FieldSliceMut<'_, P> {
		match &mut self.0 {
			// The copy is one word wide.
			// So the view is that single word, cut down to the chunk's element count.
			ChunkMutInner::Detached {
				location,
				chunk,
				parent: _,
			} => FieldBuffer {
				log_len: location.log_len(),
				words: slice::from_mut(chunk),
			},
			// The run of words is already the chunk, so the view spans all of it.
			ChunkMutInner::Borrowed { log_len, chunk } => FieldBuffer {
				log_len: *log_len,
				words: chunk,
			},
		}
	}
}

impl<P: PackedField> Drop for ChunkMut<'_, P> {
	fn drop(&mut self) {
		match &mut self.0 {
			// A detached copy is the only shape with anything to write back.
			ChunkMutInner::Detached {
				location,
				chunk,
				parent,
			} => location.merge_into(parent, chunk),
			// A chunk lent from the store was edited in place, so there is nothing to merge.
			ChunkMutInner::Borrowed { .. } => {}
		}
	}
}

/// The two shapes a mutably borrowed chunk takes, decided by its size against the packing width.
#[derive(Debug)]
enum ChunkMutInner<'a, P: PackedField> {
	/// A chunk below one packed word, lifted out of the lanes it shares with its neighbours.
	Detached {
		/// Which lanes of the parent word the chunk occupies.
		location: SubWordChunk<P>,
		/// The detached copy the caller edits.
		chunk: P,
		/// The word the copy is merged back into.
		parent: &'a mut P,
	},
	/// A chunk of one or more whole words, borrowed from the store and edited there.
	Borrowed {
		/// Element count of the chunk, as a base-2 logarithm.
		log_len: usize,
		/// The run of words the chunk occupies.
		chunk: &'a mut [P],
	},
}
