// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Building a [`KeyCollection`] from a constraint system.
//!
//! A collection indexes, per word, every shifted reference the constraints make.
//! The order the constraints are walked in fixes its layout.
//! So the collection is built as a stable counting sort by word:
//!
//! ```text
//!   count    the references each word receives, and the keys each segment names
//!   scatter  each reference into its word's slot, in walk order
//!   group    each word's references into its keys, chunk by chunk, in parallel
//! ```
//!
//! The first two passes leave each word's references contiguous and in walk order.
//! The third only has to group them, and its chunks are independent.
//!
//! [`build_key_collection`] is the entry point.

use std::{iter, ops::Range};

use binius_core::constraint_system::{ConstraintSystem, InoutSegment, Operand, Shift};
use binius_utils::rayon::prelude::*;
use binius_verifier::protocols::shift::{LOG_SHIFT_COUNT, SHIFT_COUNT};
use tracing::instrument;

use super::{
	collection::KeyCollection,
	dense_shift_encoding::DenseShiftEncoding,
	key::{ConstraintIndex, Key},
	key_segment::KeySegment,
	operation::Operation,
};

/// The bits a shift sequence's code occupies: one slot's index above the other's.
const SEQ_BITS: usize = 2 * LOG_SHIFT_COUNT;

/// The bits a key code occupies: the operation above the shift sequence.
const KEY_CODE_BITS: usize = Operation::PACKED_CODE_BITS + SEQ_BITS;

/// The number of key codes the packing spans.
///
/// The tables indexed by code are sized by this.
const KEY_CODE_SPACE: usize = 1 << KEY_CODE_BITS;

/// The bits a [`ConstraintIndex`] occupies: the operand position above the index.
const CONSTRAINT_INDEX_BITS: usize = u8::BITS as usize + u32::BITS as usize;

/// A packed reference carries a key code and a constraint index in one word.
const _: () = assert!(
	KEY_CODE_BITS + CONSTRAINT_INDEX_BITS <= u64::BITS as usize,
	"a packed reference does not fit in a u64"
);

/// The most words one parallel work item covers.
const MAX_WORDS_PER_CHUNK: usize = 1 << 12;

/// The references one parallel work item aims for.
///
/// A word is never split, so a word carrying more than this becomes an item of its own.
const MAX_REFS_PER_CHUNK: usize = 1 << 16;

/// The identity of the key a reference belongs to.
///
/// ```text
/// key code = operation << 2*LOG_SHIFT_COUNT | outer index << LOG_SHIFT_COUNT | inner index
/// ```
///
/// [`Shift::index`] enumerates the `(variant, amount)` spellings injectively.
/// So two references share a code exactly when they belong to the same key.
///
/// The sequence is addressed outer-major, as [`DenseShiftEncoding::shift_indices`] is.
#[inline]
const fn key_code(operation: Operation, shift_seq: [Shift; 2]) -> u32 {
	let [inner, outer] = shift_seq;
	((operation.packed_code() as usize) << SEQ_BITS
		| outer.index() << LOG_SHIFT_COUNT
		| inner.index()) as u32
}

/// The shift sequence a key code names, inverting the lower half of [`key_code`].
#[inline]
fn decode_shift_seq(key_code: u32) -> [Shift; 2] {
	let slot = |index: usize| Shift::from_index(index % SHIFT_COUNT);
	[
		slot(key_code as usize),
		slot(key_code as usize >> LOG_SHIFT_COUNT),
	]
}

/// The operation a key code names, inverting the upper half of [`key_code`].
#[inline]
const fn decode_operation(key_code: u32) -> Operation {
	Operation::from_packed_code((key_code >> SEQ_BITS) as u8)
}

/// One shifted-word reference a constraint makes.
#[derive(Clone, Copy)]
struct Reference {
	/// The word of the value vector the reference reads.
	word: usize,
	/// The key the reference belongs to, as [`key_code`] packs it.
	key_code: u32,
	/// The constraint and operand position the reference comes from.
	constraint_index: ConstraintIndex,
}

/// Visits every shifted reference of a constraint system, in the collection's layout order.
///
/// ```text
/// operation, in the order Zero, BitwiseAnd, IntegerMul, BinMul
///   -> operand position
///     -> constraint index
///       -> the terms of that operand
/// ```
///
/// A key segment stores each word's references in exactly this order.
/// So a stable sort by word is all that separates the walk from the layout.
#[inline]
fn for_each_reference(cs: &ConstraintSystem, mut visit: impl FnMut(Reference)) {
	fn walk<C, const ARITY: usize>(
		cs: &ConstraintSystem,
		operation: Operation,
		constraints: &[C],
		visit: &mut impl FnMut(Reference),
	) where
		C: AsRef<[Operand; ARITY]>,
	{
		// The operand position is outermost because a key groups its references by it.
		for operand_index in 0..ARITY {
			for (constraint_index, constraint) in constraints.iter().enumerate() {
				for term in &constraint.as_ref()[operand_index] {
					visit(Reference {
						word: cs.word_offset(term.value_index),
						key_code: key_code(operation, term.shift_seq),
						constraint_index: ConstraintIndex {
							operand_index: operand_index as u8,
							constraint_index: constraint_index as u32,
						},
					});
				}
			}
		}
	}

	walk(cs, Operation::Zero, &cs.zero_constraints, &mut visit);
	walk(cs, Operation::BitwiseAnd, &cs.and_constraints, &mut visit);
	walk(cs, Operation::IntegerMul, &cs.imul_constraints, &mut visit);
	walk(cs, Operation::BinMul, &cs.bmul_constraints, &mut visit);
}

/// One shifted reference, as the scattered array holds it.
///
/// ```text
///  63     60 59          40 39          32 31                0
/// [ unused ][ key code    ][ operand pos ][ constraint index ]
/// ```
///
/// The word is not stored.
/// The array is indexed by word, so a reference's position already names its word.
#[derive(Clone, Copy)]
struct PackedRef(u64);

impl PackedRef {
	/// The value the scatter pass allocates with.
	///
	/// The counts that size the array are the counts the scatter fills.
	/// So every slot is written before it is read.
	const PLACEHOLDER: Self = Self(0);

	/// Packs a reference: its key code above its constraint index.
	#[inline]
	const fn new(key_code: u32, constraint_index: ConstraintIndex) -> Self {
		Self(
			(key_code as u64) << CONSTRAINT_INDEX_BITS
				| (constraint_index.operand_index as u64) << u32::BITS
				| constraint_index.constraint_index as u64,
		)
	}

	/// The key this reference belongs to.
	#[inline]
	const fn key_code(self) -> u32 {
		(self.0 >> CONSTRAINT_INDEX_BITS) as u32
	}

	/// The constraint and operand position this reference comes from.
	#[inline]
	const fn constraint_index(self) -> ConstraintIndex {
		ConstraintIndex {
			operand_index: (self.0 >> u32::BITS) as u8,
			constraint_index: self.0 as u32,
		}
	}
}

/// The fixed part of every key one dense key id names.
///
/// A key's operation and shift index follow from its code alone.
/// So they are resolved once per id, not once per key.
#[derive(Clone, Copy)]
struct KeyTemplate {
	/// The constraint kind the key's constraints belong to.
	operation: Operation,
	/// Where the key's shift sequence sits in its segment's dense encoding.
	dense_shift_idx: u16,
}

/// The keys one segment can name, densely numbered.
///
/// A reference names its key by a code drawn from [`KEY_CODE_SPACE`] of them.
/// A real system uses a few dozen.
/// Numbering those few turns "which key is this?" into an array index.
struct SegmentKeys {
	/// One entry per key code: its dense id, or [`Self::UNUSED`].
	///
	/// Only the codes the segment names are ever looked up.
	id_of: Box<[u32]>,
	/// The template each dense id resolves to, in ascending code order.
	templates: Box<[KeyTemplate]>,
}

impl SegmentKeys {
	/// Marks a code the segment does not name.
	const UNUSED: u32 = u32::MAX;

	/// Numbers a segment's key codes and resolves each against its shift encoding.
	///
	/// # Arguments
	///
	/// - `codes`: the segment's key codes, ascending and distinct.
	/// - `dense_shift_enc`: the segment's encoding, covering every sequence those codes name.
	fn new(codes: &[u32], dense_shift_enc: &DenseShiftEncoding) -> Self {
		debug_assert!(
			codes.windows(2).all(|pair| pair[0] < pair[1]),
			"a segment's key codes are ascending and distinct"
		);

		let mut id_of = vec![Self::UNUSED; KEY_CODE_SPACE].into_boxed_slice();
		for (id, &code) in codes.iter().enumerate() {
			id_of[code as usize] = id as u32;
		}

		let templates = codes
			.iter()
			.map(|&code| KeyTemplate {
				operation: decode_operation(code),
				dense_shift_idx: dense_shift_enc.dense_idx(decode_shift_seq(code)),
			})
			.collect();

		Self { id_of, templates }
	}

	/// The number of distinct keys the segment can name.
	fn len(&self) -> usize {
		self.templates.len()
	}

	/// The dense id of one key code.
	///
	/// # Panics
	///
	/// In debug builds, panics if the segment does not name the code.
	#[inline]
	fn id(&self, code: u32) -> usize {
		let id = self.id_of[code as usize];
		debug_assert_ne!(id, Self::UNUSED, "a reference names a key of its own segment");
		id as usize
	}

	/// The key one dense id names, over the references at `range`.
	#[inline]
	fn key(&self, id: u32, range: Range<u32>) -> Key {
		let KeyTemplate {
			operation,
			dense_shift_idx,
		} = self.templates[id as usize];
		Key {
			operation,
			dense_shift_idx,
			range,
		}
	}
}

/// A partition of a segment's words into parallel work items.
///
/// Reference counts per word span orders of magnitude.
/// A handful of constants take a large share of a real system's references.
/// So an item is bounded by both the words and the references it covers.
struct WordChunks {
	/// The first word of each chunk, then a sentinel at the segment's word count.
	bounds: Vec<usize>,
}

impl WordChunks {
	/// Partitions the words `offsets` indexes.
	fn new(offsets: &[usize]) -> Self {
		let n_words = offsets.len() - 1;
		let mut bounds = vec![0];
		let mut first = 0;
		while first < n_words {
			let limit = (first + MAX_WORDS_PER_CHUNK).min(n_words);
			// A chunk always takes at least one word, however many references it carries.
			// Each further word has to keep the chunk inside its reference budget.
			let mut last = first + 1;
			while last < limit && offsets[last + 1] - offsets[first] <= MAX_REFS_PER_CHUNK {
				last += 1;
			}
			bounds.push(last);
			first = last;
		}
		Self { bounds }
	}

	/// The number of chunks the words are partitioned into.
	const fn len(&self) -> usize {
		self.bounds.len() - 1
	}

	/// The words one chunk covers, as segment-relative indices.
	fn word_range(&self, chunk: usize) -> Range<usize> {
		self.bounds[chunk]..self.bounds[chunk + 1]
	}

	/// Splits a buffer holding one entry per reference into one slice per chunk.
	///
	/// A chunk's references are contiguous, since its words are.
	///
	/// # Arguments
	///
	/// - `offsets`: the word offsets the partition was built from.
	/// - `buf`: one entry per reference of the segment.
	fn split_references<'a, T>(&self, offsets: &[usize], buf: &'a mut [T]) -> Vec<&'a mut [T]> {
		let mut rest = buf;
		let mut chunks = Vec::with_capacity(self.len());
		for chunk in 0..self.len() {
			let words = self.word_range(chunk);
			let (head, tail) = rest.split_at_mut(offsets[words.end] - offsets[words.start]);
			chunks.push(head);
			rest = tail;
		}
		debug_assert!(rest.is_empty(), "the chunks tile the segment's references");
		chunks
	}
}

/// Per-worker workspace for grouping words into keys.
///
/// The tables are indexed by dense key id, so a segment's key count sizes them.
/// One workspace serves every word of a chunk.
struct WordGrouping {
	/// Where each key id sits among the keys of the word being grouped, or [`Self::UNSEEN`].
	slot_of: Box<[u32]>,
	/// The key ids the word names, in first-appearance order.
	ids: Vec<u32>,
	/// Per slot, the key's reference count, then its running write cursor.
	cursors: Vec<u32>,
}

impl WordGrouping {
	/// Marks a key id the word being grouped has not reached yet.
	const UNSEEN: u32 = u32::MAX;

	/// A workspace for a segment naming `n_keys` distinct keys.
	fn new(n_keys: usize) -> Self {
		Self {
			slot_of: vec![Self::UNSEEN; n_keys].into_boxed_slice(),
			ids: Vec::new(),
			cursors: Vec::new(),
		}
	}

	/// Groups one word's references into its keys.
	///
	/// The references arrive in walk order.
	/// The keys come out in first-appearance order.
	/// Each key keeps its own references in walk order.
	/// Together those are the layout a key segment stores.
	///
	/// # Arguments
	///
	/// - `refs`: the word's references, in walk order.
	/// - `keys`: the word's keys, appended in first-appearance order.
	/// - `out`: the word's slice of the segment's flattened reference list.
	/// - `segment_keys`: the segment's dense key numbering.
	/// - `first_ref`: where `out` starts in that flattened list.
	///
	/// # Returns
	///
	/// The number of keys the word names.
	fn group(
		&mut self,
		refs: &[PackedRef],
		keys: &mut Vec<Key>,
		out: &mut [ConstraintIndex],
		segment_keys: &SegmentKeys,
		first_ref: u32,
	) -> u32 {
		// Count each key's references, meeting the keys in first-appearance order.
		self.ids.clear();
		self.cursors.clear();
		for reference in refs {
			let id = segment_keys.id(reference.key_code());
			match self.slot_of[id] {
				Self::UNSEEN => {
					self.slot_of[id] = self.ids.len() as u32;
					self.ids.push(id as u32);
					self.cursors.push(1);
				}
				slot => self.cursors[slot as usize] += 1,
			}
		}

		// The counts give each key its range, and become its write cursor in place.
		let mut start = 0;
		for (&id, cursor) in iter::zip(&self.ids, &mut self.cursors) {
			let n_refs = *cursor;
			keys.push(segment_keys.key(id, first_ref + start..first_ref + start + n_refs));
			*cursor = start;
			start += n_refs;
		}

		// Place each reference at its key's cursor.
		// Walk order survives because a cursor only ever advances.
		for reference in refs {
			let slot = self.slot_of[segment_keys.id(reference.key_code())] as usize;
			let cursor = &mut self.cursors[slot];
			out[*cursor as usize] = reference.constraint_index();
			*cursor += 1;
		}

		// Leave the workspace clean, touching only the slots this word used.
		for &id in &self.ids {
			self.slot_of[id as usize] = Self::UNSEEN;
		}

		self.ids.len() as u32
	}
}

/// The keys of one chunk's words, ready to be appended to the segment.
struct ChunkKeys {
	/// Every key of the chunk, word by word, each word's in first-appearance order.
	///
	/// Reference ranges are already segment-relative.
	/// The counting pass fixed where each word's references land before any grouping ran.
	keys: Vec<Key>,
	/// One range per word of the chunk, naming that word's keys inside `keys`.
	///
	/// Chunk-relative: a chunk cannot know how many keys the chunks before it produced.
	key_ranges: Vec<Range<u32>>,
}

/// Counts the references each word receives, and the key codes each segment names.
///
/// # Arguments
///
/// - `cs`: the constraint system to walk.
/// - `n_public_words`: the word the hidden segment starts at.
///
/// # Returns
///
/// The word offsets, then the public segment's key codes, then the hidden segment's.
///
/// Offsets carry one entry per word plus a sentinel.
/// So word `w` owns `offsets[w]..offsets[w + 1]` once the references are scattered.
/// Both code lists come out ascending and distinct.
#[instrument(skip_all, name = "count_references")]
fn count_references(
	cs: &ConstraintSystem,
	n_public_words: usize,
) -> (Vec<usize>, Vec<u32>, Vec<u32>) {
	/// Marks a code the public segment names.
	const PUBLIC: u8 = 1;
	/// Marks a code the hidden segment names.
	const HIDDEN: u8 = 2;

	let n_words = cs.value_vec_len();
	let mut offsets = vec![0usize; n_words + 1];
	let mut segments_of_code = vec![0u8; KEY_CODE_SPACE];

	for_each_reference(cs, |reference| {
		offsets[reference.word + 1] += 1;
		segments_of_code[reference.key_code as usize] |= if reference.word < n_public_words {
			PUBLIC
		} else {
			HIDDEN
		};
	});

	// A running sum turns each count into that word's start in the scattered array.
	for word in 0..n_words {
		offsets[word + 1] += offsets[word];
	}

	let mut public_codes = Vec::new();
	let mut hidden_codes = Vec::new();
	for (code, &segments) in segments_of_code.iter().enumerate() {
		if segments & PUBLIC != 0 {
			public_codes.push(code as u32);
		}
		if segments & HIDDEN != 0 {
			hidden_codes.push(code as u32);
		}
	}

	(offsets, public_codes, hidden_codes)
}

/// Scatters every reference into its word's slot, keeping walk order within each word.
///
/// # Arguments
///
/// - `cs`: the constraint system, walked in the order [`count_references`] walked it.
/// - `offsets`: the word offsets that walk produced.
#[instrument(skip_all, name = "scatter_references")]
fn scatter_references(cs: &ConstraintSystem, offsets: &[usize]) -> Vec<PackedRef> {
	let n_refs = *offsets.last().expect("word offsets end in a sentinel");
	let mut refs = vec![PackedRef::PLACEHOLDER; n_refs];

	// One cursor per word, walking that word's slot from the start.
	// Advancing on every write is what makes the sort stable.
	let mut cursors = offsets.to_owned();
	for_each_reference(cs, |reference| {
		refs[cursors[reference.word]] =
			PackedRef::new(reference.key_code, reference.constraint_index);
		cursors[reference.word] += 1;
	});

	refs
}

/// Builds one key segment from the words of the scattered array it covers.
///
/// # Arguments
///
/// - `offsets`: one entry per word of the segment plus a sentinel, indexing `refs`.
/// - `refs`: every reference of the constraint system, scattered by word.
/// - `codes`: the key codes this segment names, ascending and distinct.
#[instrument(skip_all, name = "build_key_segment")]
fn build_segment(offsets: &[usize], refs: &[PackedRef], codes: &[u32]) -> KeySegment {
	let n_words = offsets.len() - 1;
	let first_ref = offsets[0];
	// A key's range and a word's range both address the segment with a `u32`.
	let n_refs = u32::try_from(offsets[n_words] - first_ref)
		.expect("a key segment addresses its references with a u32");

	let dense_shift_enc = DenseShiftEncoding::new(codes.iter().copied().map(decode_shift_seq));
	let segment_keys = SegmentKeys::new(codes, &dense_shift_enc);

	// Each chunk's references occupy a known slice of the flattened list.
	// So the chunks fill it in place, rather than buffering copies to concatenate afterwards.
	let chunks = WordChunks::new(offsets);
	// Every slot is overwritten: the counts that size this are the counts the chunks write.
	let placeholder = ConstraintIndex {
		operand_index: 0,
		constraint_index: 0,
	};
	let mut constraint_indices = vec![placeholder; n_refs as usize];

	let chunk_keys = chunks
		.split_references(offsets, &mut constraint_indices)
		.into_par_iter()
		.enumerate()
		.map_init(
			|| WordGrouping::new(segment_keys.len()),
			|grouping, (chunk, mut chunk_refs)| {
				let words = chunks.word_range(chunk);
				let mut out = ChunkKeys {
					keys: Vec::new(),
					key_ranges: Vec::with_capacity(words.len()),
				};
				let mut n_keys = 0;
				for word in words {
					let (word_refs, rest) =
						chunk_refs.split_at_mut(offsets[word + 1] - offsets[word]);
					chunk_refs = rest;

					let start = n_keys;
					n_keys += grouping.group(
						&refs[offsets[word]..offsets[word + 1]],
						&mut out.keys,
						word_refs,
						&segment_keys,
						(offsets[word] - first_ref) as u32,
					);
					out.key_ranges.push(start..n_keys);
				}
				debug_assert!(chunk_refs.is_empty(), "the words tile the chunk's references");
				out
			},
		)
		.collect::<Vec<_>>();

	// Where each chunk's keys start in the flattened key list.
	// That offset is the one thing a chunk could not resolve on its own.
	let mut key_bases = Vec::with_capacity(chunks.len());
	let mut n_keys = 0u32;
	for chunk in &chunk_keys {
		key_bases.push(n_keys);
		n_keys = u32::try_from(n_keys as usize + chunk.keys.len())
			.expect("a key segment addresses its keys with a u32");
	}

	let mut keys = Vec::with_capacity(n_keys as usize);
	let mut key_ranges = Vec::with_capacity(n_words);
	for (chunk, base) in iter::zip(chunk_keys, key_bases) {
		keys.extend(chunk.keys);
		key_ranges.extend(
			chunk
				.key_ranges
				.into_iter()
				.map(|range| range.start + base..range.end + base),
		);
	}

	KeySegment {
		keys,
		key_ranges,
		constraint_indices,
		dense_shift_enc,
	}
}

/// Builds the prover's key collection from a constraint system.
///
/// # Arguments
///
/// - `cs`: the constraint system to walk.
/// - `inout`: the split point between the public and hidden segments.
#[instrument(skip_all, name = "build_key_collection")]
pub(super) fn build_key_collection(cs: &ConstraintSystem, inout: InoutSegment) -> KeyCollection {
	let n_public_words = cs.n_public_words(inout);
	let (offsets, public_codes, hidden_codes) = count_references(cs, n_public_words);
	let refs = scatter_references(cs, &offsets);

	KeyCollection {
		public: build_segment(&offsets[..=n_public_words], &refs, &public_codes),
		hidden: build_segment(&offsets[n_public_words..], &refs, &hidden_codes),
	}
}

#[cfg(test)]
mod tests {
	use binius_core::{
		constraint_system::{
			AndConstraint, BmulConstraint, ImulConstraint, ShiftVariant, ShiftedValueIndex,
			ValueIndex, ZeroConstraint,
		},
		word::Word,
	};
	use binius_utils::serialization::SerializeBytes;
	use proptest::prelude::*;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;

	/// The one-pass builder the counting sort replaces, kept as its equivalence oracle.
	///
	/// It collects each word's keys into their own vectors and flattens them at the end.
	/// That is the definition of the layout the counting sort has to reproduce.
	mod one_pass {
		use super::*;

		/// One key still being assembled while its segment is built.
		struct BuilderKey {
			/// The shift sequence this key's word is referenced under, inner shift first.
			shift_seq: [Shift; 2],
			/// The constraint kind this key's constraints belong to.
			operation: Operation,
			/// The constraint indices collected so far for this key.
			constraint_indices: Vec<ConstraintIndex>,
		}

		/// One builder key list per word of the constraint system, indexed by word position.
		struct BuilderKeyLists(Vec<Vec<BuilderKey>>);

		impl BuilderKeyLists {
			/// An empty list for every one of `word_count` words.
			fn new(word_count: usize) -> Self {
				Self((0..word_count).map(|_| Vec::new()).collect())
			}

			/// Records one operand's references into the keys of the words they touch.
			fn update_with_operand(
				&mut self,
				operation: Operation,
				operand_index: usize,
				operand_values: impl Iterator<Item = impl AsRef<Operand>>,
				cs: &ConstraintSystem,
			) {
				for (constraint_idx, operand_value) in operand_values.enumerate() {
					for term in operand_value.as_ref() {
						let builder_keys = &mut self.0[cs.word_offset(term.value_index)];
						let shift_seq = term.shift_seq;
						let constraint_index = ConstraintIndex {
							operand_index: operand_index as u8,
							constraint_index: constraint_idx as u32,
						};
						match builder_keys
							.iter_mut()
							.find(|key| key.shift_seq == shift_seq && key.operation == operation)
						{
							Some(key) => key.constraint_indices.push(constraint_index),
							None => builder_keys.push(BuilderKey {
								shift_seq,
								operation,
								constraint_indices: vec![constraint_index],
							}),
						}
					}
				}
			}

			/// Records every operand of every constraint of one operation.
			fn update_with_constraints<C, const ARITY: usize>(
				&mut self,
				operation: Operation,
				constraints: &[C],
				cs: &ConstraintSystem,
			) where
				C: AsRef<[Operand; ARITY]>,
			{
				for operand_index in 0..ARITY {
					self.update_with_operand(
						operation,
						operand_index,
						constraints
							.iter()
							.map(|constraint| &constraint.as_ref()[operand_index]),
						cs,
					);
				}
			}
		}

		/// Flattens one segment's builder keys into the layout a [`KeySegment`] stores.
		fn build_segment(builder_key_lists: Vec<Vec<BuilderKey>>) -> KeySegment {
			let dense_shift_enc = DenseShiftEncoding::new(
				builder_key_lists
					.iter()
					.flatten()
					.map(|builder_key| builder_key.shift_seq),
			);

			let key_ranges = builder_key_lists
				.iter()
				.scan(0u32, |offset, builder_keys| {
					let start = *offset;
					*offset += builder_keys.len() as u32;
					Some(start..*offset)
				})
				.collect();

			let mut keys = Vec::new();
			let mut constraint_indices = Vec::new();
			for builder_key in builder_key_lists.into_iter().flatten() {
				let BuilderKey {
					shift_seq,
					operation,
					constraint_indices: mut key_constraint_indices,
				} = builder_key;

				// Sorting by operand index groups each operand's entries together.
				key_constraint_indices.sort_by_key(|index| index.operand_index);

				let start = constraint_indices.len() as u32;
				constraint_indices.extend(key_constraint_indices);
				let end = constraint_indices.len() as u32;
				keys.push(Key {
					dense_shift_idx: dense_shift_enc.dense_idx(shift_seq),
					operation,
					range: start..end,
				});
			}

			KeySegment {
				keys,
				key_ranges,
				constraint_indices,
				dense_shift_enc,
			}
		}

		/// Builds a key collection by the one-pass route.
		pub(super) fn build_key_collection(
			cs: &ConstraintSystem,
			inout: InoutSegment,
		) -> KeyCollection {
			let mut lists = BuilderKeyLists::new(cs.value_vec_len());
			lists.update_with_constraints(Operation::Zero, &cs.zero_constraints, cs);
			lists.update_with_constraints(Operation::BitwiseAnd, &cs.and_constraints, cs);
			lists.update_with_constraints(Operation::IntegerMul, &cs.imul_constraints, cs);
			lists.update_with_constraints(Operation::BinMul, &cs.bmul_constraints, cs);

			let hidden = lists.0.split_off(cs.n_public_words(inout));
			KeyCollection {
				public: build_segment(lists.0),
				hidden: build_segment(hidden),
			}
		}
	}

	/// A collection's bytes, which pin every field of both its segments at once.
	fn serialized(collection: &KeyCollection) -> Vec<u8> {
		let mut bytes = Vec::new();
		collection.serialize(&mut bytes).unwrap();
		bytes
	}

	/// Asserts the counting sort reproduces the one-pass layout, on both splits.
	fn assert_matches_one_pass(cs: &ConstraintSystem) {
		for inout in [InoutSegment::Public, InoutSegment::Hidden] {
			let fast = build_key_collection(cs, inout);
			let slow = one_pass::build_key_collection(cs, inout);

			// The counts only make a failure readable; the bytes are the assertion.
			assert_eq!(fast.hidden.keys.len(), slow.hidden.keys.len(), "{inout:?}");
			assert_eq!(
				fast.hidden.constraint_indices.len(),
				slow.hidden.constraint_indices.len(),
				"{inout:?}"
			);
			assert!(serialized(&fast) == serialized(&slow), "the layouts differ at {inout:?}");
		}
	}

	/// A shift alphabet of `n_shifts` spellings, starting with the identity.
	///
	/// A bounded alphabet is what makes a word collect several keys.
	/// Drawing every shift afresh would give it one key per reference instead.
	fn shift_alphabet(n_shifts: usize) -> Vec<Shift> {
		(0..n_shifts)
			.map(|i| {
				if i == 0 {
					return Shift::IDENTITY;
				}
				let variant = ShiftVariant::ALL[i % ShiftVariant::ALL.len()];
				Shift::new(variant, 1 + i % (variant.max_amount() - 1))
			})
			.collect()
	}

	/// A random system with every operation, multi-term operands, and two-slot shifts.
	///
	/// A quarter of the references land on the first constant, and an eighth on a few other words.
	/// So a few words carry long key lists while most carry one or two.
	/// That skew is what a real circuit looks like, and what the partition has to cope with.
	fn random_constraint_system(
		seed: u64,
		n_and: usize,
		n_private: usize,
		n_shifts: usize,
	) -> ConstraintSystem {
		const N_CONST: usize = 8;
		const N_INOUT: usize = 6;

		let mut rng = StdRng::seed_from_u64(seed);
		let alphabet = shift_alphabet(n_shifts);

		let shift = |rng: &mut StdRng| alphabet[rng.random_range(0..alphabet.len())];
		let term = |rng: &mut StdRng| {
			let value_index = match rng.random_range(0..8u8) {
				0..=1 => ValueIndex::constant(0),
				2 => ValueIndex::constant(rng.random_range(0..N_CONST as u32)),
				3 => ValueIndex::inout(rng.random_range(0..N_INOUT as u32)),
				_ => ValueIndex::private(rng.random_range(0..n_private as u32)),
			};
			ShiftedValueIndex::new(value_index, [shift(rng), shift(rng)])
		};
		let operand = |rng: &mut StdRng| (0..rng.random_range(1..=4)).map(|_| term(rng)).collect();
		let operands = |rng: &mut StdRng, arity: usize| {
			(0..arity).map(|_| operand(rng)).collect::<Vec<Operand>>()
		};

		ConstraintSystem {
			constants: vec![Word::ZERO; N_CONST],
			n_inout: N_INOUT,
			n_private,
			zero_constraints: (0..n_and / 8)
				.map(|_| ZeroConstraint(operands(&mut rng, 1).try_into().ok().unwrap()))
				.collect(),
			and_constraints: (0..n_and)
				.map(|_| AndConstraint(operands(&mut rng, 3).try_into().ok().unwrap()))
				.collect(),
			imul_constraints: (0..n_and / 16)
				.map(|_| ImulConstraint(operands(&mut rng, 4).try_into().ok().unwrap()))
				.collect(),
			bmul_constraints: (0..n_and / 4)
				.map(|_| BmulConstraint(operands(&mut rng, 6).try_into().ok().unwrap()))
				.collect(),
		}
	}

	/// A system whose only constraint puts `terms` on the words they name.
	fn one_constraint_system(terms: Vec<ShiftedValueIndex>) -> ConstraintSystem {
		ConstraintSystem {
			constants: vec![Word::ZERO; 4],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: vec![AndConstraint([terms, Vec::new(), Vec::new()])],
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		}
	}

	proptest! {
		// Invariant: the counting sort produces the one-pass layout exactly.
		//
		// The two build the same thing by different routes.
		// So byte equality of the serialized collections pins every field of both segments:
		// the keys, their word ranges, their reference ranges, the flattened references, and
		// the shift encoding.
		//
		// The parameters span the axes the grouping depends on: how many references there are,
		// how many words they spread over, and how many keys one word can end up naming.
		#[test]
		fn counting_sort_matches_the_one_pass_builder(
			seed: u64,
			n_and in 0usize..2000,
			n_private in 1usize..3 * MAX_WORDS_PER_CHUNK,
			n_shifts in 1usize..24,
		) {
			assert_matches_one_pass(&random_constraint_system(seed, n_and, n_private, n_shifts));
		}
	}

	#[test]
	fn a_system_with_no_constraints_yields_empty_segments() {
		let cs = ConstraintSystem {
			constants: vec![Word::ZERO; 2],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		};

		let collection = build_key_collection(&cs, InoutSegment::Public);

		// Every word still gets a range, so an untouched word looks up empty rather than panics.
		assert_eq!(collection.public.key_ranges.len(), 2);
		assert_eq!(collection.hidden.key_ranges.len(), 4);
		assert!(collection.hidden.keys.is_empty());
		assert_matches_one_pass(&cs);
	}

	#[test]
	fn a_word_can_name_far_more_keys_than_a_real_one() {
		// Every variant at nine amounts: seventy-two sequences, all landing on one word.
		// A real word names a handful, so this runs the grouping well past its usual shape.
		let hidden = ValueIndex::private(1);
		let cs = one_constraint_system(
			ShiftVariant::ALL
				.into_iter()
				.flat_map(|variant| (1..10).map(move |amount| Shift::new(variant, amount)))
				.map(|shift| ShiftedValueIndex::new(hidden, [shift, Shift::IDENTITY]))
				.collect(),
		);

		assert_eq!(
			build_key_collection(&cs, InoutSegment::Public)
				.hidden
				.word_keys(1)
				.len(),
			72
		);
		assert_matches_one_pass(&cs);
	}

	#[test]
	fn repeated_references_share_one_key() {
		// One word under one sequence three times is one key holding three entries.
		let hidden = ValueIndex::private(1);
		let cs = one_constraint_system(vec![ShiftedValueIndex::srl(hidden, 3); 3]);

		let collection = build_key_collection(&cs, InoutSegment::Public);
		let keys = collection.hidden.word_keys(1);
		assert_eq!(keys.len(), 1);
		assert_eq!(keys[0].range, 0..3);
		assert_matches_one_pass(&cs);
	}

	#[test]
	fn non_canonical_identity_spellings_name_distinct_keys() {
		// `Slr(0)` and `Sll(0)` leave a word untouched alike, but they are distinct spellings.
		// `Shift::index` separates them, so they must not collapse into one key.
		let non_canonical = Shift {
			variant: ShiftVariant::Slr,
			amount: 0,
		};
		assert!(!non_canonical.is_canonical());

		let hidden = ValueIndex::private(1);
		let cs = one_constraint_system(vec![
			ShiftedValueIndex::plain(hidden),
			ShiftedValueIndex::new(hidden, [non_canonical, Shift::IDENTITY]),
		]);

		let collection = build_key_collection(&cs, InoutSegment::Public);
		assert_eq!(collection.hidden.word_keys(1).len(), 2);
		assert_matches_one_pass(&cs);
	}

	#[test]
	fn a_word_past_the_reference_budget_is_grouped_on_its_own() {
		// A word carrying more references than a chunk's budget cannot be split.
		// So it becomes a chunk of its own, and its neighbours land in chunks either side.
		let hidden = ValueIndex::private(1);
		let mut terms = vec![ShiftedValueIndex::plain(ValueIndex::private(0))];
		terms.extend((0..=MAX_REFS_PER_CHUNK).map(|i| ShiftedValueIndex::srl(hidden, 1 + i % 63)));
		terms.push(ShiftedValueIndex::plain(ValueIndex::private(2)));
		let cs = one_constraint_system(terms);

		let collection = build_key_collection(&cs, InoutSegment::Public);
		assert_eq!(collection.hidden.word_keys(1).len(), 63);
		assert_matches_one_pass(&cs);
	}

	#[test]
	fn word_chunks_bound_both_words_and_references() {
		// Sparse words fill a chunk right up to the word bound.
		let sparse = vec![0; 2 * MAX_WORDS_PER_CHUNK + 1];
		let chunks = WordChunks::new(&sparse);
		assert_eq!(chunks.len(), 2);
		assert_eq!(chunks.word_range(0), 0..MAX_WORDS_PER_CHUNK);
		assert_eq!(chunks.word_range(1), MAX_WORDS_PER_CHUNK..2 * MAX_WORDS_PER_CHUNK);

		// Dense ones stop at the reference bound instead, long before the word bound.
		let dense = (0..=8)
			.map(|w| w * (MAX_REFS_PER_CHUNK / 2))
			.collect::<Vec<_>>();
		let chunks = WordChunks::new(&dense);
		assert_eq!(chunks.len(), 4);
		assert_eq!(chunks.word_range(0), 0..2);

		// A word past the budget on its own cannot be split, so it takes a chunk to itself.
		let chunks = WordChunks::new(&[0, 1, MAX_REFS_PER_CHUNK + 2, MAX_REFS_PER_CHUNK + 3]);
		assert_eq!(chunks.len(), 3);
		assert_eq!(chunks.word_range(1), 1..2);
	}

	#[test]
	fn a_key_code_round_trips_through_its_operation_and_sequence() {
		// The grouping identifies a key by its code alone.
		// So the code has to carry both halves back out intact.
		for operation in [
			Operation::Zero,
			Operation::BitwiseAnd,
			Operation::IntegerMul,
			Operation::BinMul,
		] {
			for inner in shift_alphabet(17) {
				for outer in shift_alphabet(5) {
					let code = key_code(operation, [inner, outer]);
					assert_eq!(decode_operation(code), operation);
					assert_eq!(decode_shift_seq(code), [inner, outer]);
				}
			}
		}
	}

	#[test]
	fn a_packed_reference_round_trips_its_key_code_and_constraint_index() {
		// Every field at its maximum at once.
		// A field overlapping its neighbour would show up here as a changed value.
		let constraint_index = ConstraintIndex {
			operand_index: u8::MAX,
			constraint_index: u32::MAX,
		};
		let code = (KEY_CODE_SPACE - 1) as u32;
		let packed = PackedRef::new(code, constraint_index);

		assert_eq!(packed.key_code(), code);
		assert_eq!(packed.constraint_index().operand_index, u8::MAX);
		assert_eq!(packed.constraint_index().constraint_index, u32::MAX);
	}
}
