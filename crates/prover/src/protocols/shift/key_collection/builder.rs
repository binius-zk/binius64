// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Building a [`KeyCollection`] from a constraint system.
//!
//! The collection is a per-word index over every shifted reference the constraints make.
//! Its layout is fixed by the walk order — operations `Zero, BitwiseAnd, IntegerMul, BinMul`,
//! then operand position, then constraint index — so it is built as a stable counting sort by
//! word: one pass counts the references each word receives, one pass scatters them in walk order,
//! and the per-word grouping into keys then runs over disjoint word ranges in parallel. Nothing
//! is allocated per key.
//!
//! [`KeyCollection::build`] is the entry point; everything else here serves it.

use std::ops::Range;

use binius_core::constraint_system::{ConstraintSystem, InoutSegment, Operand, Shift};
use binius_utils::rayon::prelude::*;
use binius_verifier::protocols::shift::LOG_SHIFT_COUNT;

use super::{
	collection::KeyCollection,
	dense_shift_encoding::DenseShiftEncoding,
	key::{ConstraintIndex, Key},
	key_segment::KeySegment,
	operation::Operation,
};

/// One key still being assembled while its segment is built.
///
/// Constraint indices are collected directly into a vector here, rather than as a range into a
/// shared flattened vector like the final form uses.
/// That flattened layout is not known until every word's keys have been collected.
///
/// Only the reference builder below (kept as the oracle for the equivalence tests) still
/// produces these.
#[cfg(test)]
pub(super) struct BuilderKey {
	/// The shift sequence this key's word is referenced under, inner shift first.
	pub shift_seq: [Shift; 2],
	/// The constraint kind this key's constraints belong to.
	pub operation: Operation,
	/// The constraint indices collected so far for this key.
	pub constraint_indices: Vec<ConstraintIndex>,
}

/// One builder key list per word of the constraint system, indexed by word position.
///
/// Kept under `cfg(test)`: [`KeyCollection::build`] runs the counting sort below, and this
/// one-pass path remains as the byte-identity oracle its tests compare against.
#[cfg(test)]
pub(super) struct BuilderKeyLists(Vec<Vec<BuilderKey>>);

#[cfg(test)]
impl BuilderKeyLists {
	/// An empty list for every one of `word_count` words.
	pub(super) fn new(word_count: usize) -> Self {
		Self((0..word_count).map(|_| Vec::new()).collect())
	}

	/// Splits the words from `public_word_count` onward into a second set of lists.
	///
	/// This is what separates the public segment, kept here, from the hidden one, returned.
	pub(super) fn split_off(&mut self, public_word_count: usize) -> Self {
		Self(self.0.split_off(public_word_count))
	}

	/// The underlying per-word lists, ready for a segment to be built from them.
	pub(super) fn into_inner(self) -> Vec<Vec<BuilderKey>> {
		self.0
	}

	/// Records one operand's shifted-word references into the builder keys of the words they
	/// touch.
	///
	/// # Arguments
	///
	/// - `operation`: the constraint kind these constraints belong to.
	/// - `operand_index`: this operand's position in the constraint.
	/// - `operand_values`: this operand's shifted-word references, one list per constraint.
	/// - `cs`: resolves a value index to its segment-relative word position.
	fn update_with_operand(
		&mut self,
		operation: Operation,
		operand_index: usize,
		operand_values: impl Iterator<Item = impl AsRef<Operand>>,
		cs: &ConstraintSystem,
	) {
		for (constraint_idx, operand_value) in operand_values.enumerate() {
			// Each operand value is a Vec<ShiftedValueIndex> - multiple shifted word references
			for term in operand_value.as_ref() {
				// The lists are indexed by word position, so resolve the term's segment-relative
				// index against the segment starts.
				let builder_keys = &mut self.0[cs.word_offset(term.value_index)];
				let shift_seq = term.shift_seq;

				// Find existing builder key or create a new one for this (operation, sequence) pair
				let constraint_index = ConstraintIndex {
					operand_index: operand_index as u8,
					constraint_index: constraint_idx as u32,
				};
				if let Some(builder_key) = builder_keys
					.iter_mut()
					.find(|key| key.shift_seq == shift_seq && key.operation == operation)
				{
					builder_key.constraint_indices.push(constraint_index);
				} else {
					builder_keys.push(BuilderKey {
						shift_seq,
						operation,
						constraint_indices: vec![constraint_index],
					});
				}
			}
		}
	}

	/// Records every operand of every constraint of one operation into the builder keys of the
	/// words they touch.
	///
	/// Operands are indexed by their position in the constraint's operand array.
	/// That is also the order the shift reduction batches them in.
	pub(super) fn update_with_constraints<C, const ARITY: usize>(
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

/// One shifted reference, packed most-significant first: operation (2 bits) ‖ shift sequence
/// (`2 * LOG_SHIFT_COUNT`) ‖ operand position (8) ‖ constraint index (32).
///
/// The top bits are the key identity, so two references share a key exactly when they agree
/// there. The low 40 bits ascend in walk order, so a key's references keep that order without
/// being sorted — [`group_word`] groups by first appearance, which is the order the keys
/// themselves must come out in.
#[derive(Clone, Copy)]
struct PackedRef(u64);

/// The operation occupies bits 58-59, so the shift sequence below it must stop short of bit 58.
/// A ninth shift variant or a wider word would silently overlap the two fields.
const _: () = assert!(
	2 * LOG_SHIFT_COUNT + 8 + 32 <= 58,
	"the packed shift sequence would overlap the operation bits"
);

impl PackedRef {
	#[inline]
	const fn new(
		operation: Operation,
		shift_seq: [Shift; 2],
		operand_index: u8,
		constraint_index: u32,
	) -> Self {
		let seq = (shift_seq[1].index() << LOG_SHIFT_COUNT | shift_seq[0].index()) as u64;
		let op = operation_code(operation) as u64;
		Self(op << 58 | seq << 40 | (operand_index as u64) << 32 | constraint_index as u64)
	}
	/// `(operation, shift sequence)` — the identity of the key this reference belongs to.
	#[inline]
	const fn key_code(self) -> u32 {
		(self.0 >> 40) as u32
	}
	#[inline]
	const fn constraint_index(self) -> ConstraintIndex {
		ConstraintIndex {
			operand_index: (self.0 >> 32) as u8,
			constraint_index: self.0 as u32,
		}
	}
}

#[inline]
const fn operation_code(operation: Operation) -> u8 {
	match operation {
		Operation::Zero => 0,
		Operation::BitwiseAnd => 1,
		Operation::IntegerMul => 2,
		Operation::BinMul => 3,
	}
}

#[inline]
const fn operation_from_code(code: u8) -> Operation {
	match code {
		0 => Operation::Zero,
		1 => Operation::BitwiseAnd,
		2 => Operation::IntegerMul,
		_ => Operation::BinMul,
	}
}

/// The shift sequences seen so far, by their packed code, so a key code decodes exactly.
struct SeqTable(Vec<Option<[Shift; 2]>>);

impl SeqTable {
	fn new() -> Self {
		Self(vec![None; 1 << (2 * LOG_SHIFT_COUNT)])
	}
	#[inline]
	fn note(&mut self, shift_seq: [Shift; 2]) {
		let code = shift_seq[1].index() << LOG_SHIFT_COUNT | shift_seq[0].index();
		if self.0[code].is_none() {
			self.0[code] = Some(shift_seq);
		}
	}
	#[inline]
	fn seq(&self, key_code: u32) -> [Shift; 2] {
		let code = (key_code & ((1 << (2 * LOG_SHIFT_COUNT)) - 1)) as usize;
		self.0[code].expect("every key code was noted in the counting pass")
	}
}

/// Visits every shifted reference of the constraint system in key-collection walk order:
/// operations `Zero, BitwiseAnd, IntegerMul, BinMul`; within one, operand position first, then
/// constraint index, then the operand's terms.
#[inline]
fn for_each_reference(cs: &ConstraintSystem, mut visit: impl FnMut(usize, [Shift; 2], PackedRef)) {
	fn walk<C, const ARITY: usize>(
		cs: &ConstraintSystem,
		operation: Operation,
		constraints: &[C],
		visit: &mut impl FnMut(usize, [Shift; 2], PackedRef),
	) where
		C: AsRef<[Operand; ARITY]>,
	{
		for operand_index in 0..ARITY {
			for (constraint_index, constraint) in constraints.iter().enumerate() {
				for term in &constraint.as_ref()[operand_index] {
					visit(
						cs.word_offset(term.value_index),
						term.shift_seq,
						PackedRef::new(
							operation,
							term.shift_seq,
							operand_index as u8,
							constraint_index as u32,
						),
					);
				}
			}
		}
	}
	walk(cs, Operation::Zero, &cs.zero_constraints, &mut visit);
	walk(cs, Operation::BitwiseAnd, &cs.and_constraints, &mut visit);
	walk(cs, Operation::IntegerMul, &cs.imul_constraints, &mut visit);
	walk(cs, Operation::BinMul, &cs.bmul_constraints, &mut visit);
}

/// Words per parallel work item in the grouping pass.
pub(super) const WORDS_PER_CHUNK: usize = 1 << 12;

/// The keys of one contiguous word range, before their ranges are rebased onto the segment.
struct ChunkKeys {
	/// `(key code, reference count)` per key, word-major in first-appearance order.
	keys: Vec<(u32, u32)>,
	/// Keys per word.
	keys_per_word: Vec<u32>,
	/// The references, laid out key by key.
	constraint_indices: Vec<ConstraintIndex>,
	/// The distinct key codes the chunk uses, sorted.
	distinct_codes: Vec<u32>,
}

/// Groups one word's references — already in walk order — into keys in first-appearance order,
/// keeping walk order inside each key. Appends to the chunk buffers; `scratch` is reused across
/// words.
fn group_word(refs: &[PackedRef], out: &mut ChunkKeys, scratch: &mut Vec<(u32, u32)>) {
	scratch.clear();
	// First-appearance order of the keys, with their reference counts.
	for r in refs {
		let code = r.key_code();
		match scratch.iter_mut().find(|(c, _)| *c == code) {
			Some((_, n)) => *n += 1,
			None => scratch.push((code, 1)),
		}
	}
	let n_keys = scratch.len();
	out.keys_per_word.push(n_keys as u32);
	if n_keys == 1 {
		out.keys.push(scratch[0]);
		out.constraint_indices
			.extend(refs.iter().map(|r| r.constraint_index()));
		return;
	}
	// Place each reference in its key's slot; walk order is preserved because refs is walked
	// front to back and each key's cursor only advances.
	let base = out.constraint_indices.len();
	let mut cursors: [u32; 64] = [0; 64];
	let mut cursor_vec;
	let cursors: &mut [u32] = if n_keys <= 64 {
		&mut cursors[..n_keys]
	} else {
		cursor_vec = vec![0u32; n_keys];
		&mut cursor_vec[..]
	};
	let mut start = 0u32;
	for (k, (_, n)) in scratch.iter().enumerate() {
		cursors[k] = start;
		start += n;
	}
	out.constraint_indices.resize(
		base + refs.len(),
		ConstraintIndex {
			operand_index: 0,
			constraint_index: 0,
		},
	);
	for r in refs {
		let code = r.key_code();
		let k = scratch
			.iter()
			.position(|(c, _)| *c == code)
			.expect("counted above");
		out.constraint_indices[base + cursors[k] as usize] = r.constraint_index();
		cursors[k] += 1;
	}
	out.keys.extend_from_slice(scratch);
}

/// Builds one segment from its word range of the scattered references.
fn build_segment(offsets: &[usize], refs: &[PackedRef], seqs: &SeqTable) -> KeySegment {
	// `offsets` has one entry per word plus a sentinel; the segment's references are
	// refs[offsets[0]..offsets[n_words]].
	let n_words = offsets.len() - 1;
	let n_chunks = n_words.div_ceil(WORDS_PER_CHUNK);

	// Group each chunk's words; chunks are independent.
	let chunks: Vec<ChunkKeys> = (0..n_chunks)
		.into_par_iter()
		.map(|chunk| {
			let first = chunk * WORDS_PER_CHUNK;
			let last = (first + WORDS_PER_CHUNK).min(n_words);
			let mut out = ChunkKeys {
				keys: Vec::new(),
				keys_per_word: Vec::with_capacity(last - first),
				constraint_indices: Vec::with_capacity(offsets[last] - offsets[first]),
				distinct_codes: Vec::new(),
			};
			let mut scratch = Vec::with_capacity(32);
			for w in first..last {
				group_word(&refs[offsets[w]..offsets[w + 1]], &mut out, &mut scratch);
			}
			let mut codes: Vec<u32> = out.keys.iter().map(|&(code, _)| code).collect();
			codes.sort_unstable();
			codes.dedup();
			out.distinct_codes = codes;
			out
		})
		.collect();

	// The segment's shift sequences: the union of the chunks' distinct codes.
	let mut all_codes: Vec<u32> = chunks
		.iter()
		.flat_map(|c| c.distinct_codes.iter().copied())
		.collect();
	all_codes.sort_unstable();
	all_codes.dedup();
	let seq_mask = (1u32 << (2 * LOG_SHIFT_COUNT)) - 1;
	let dense_shift_enc = DenseShiftEncoding::new(
		all_codes
			.iter()
			.map(|&code| code & seq_mask)
			.collect::<std::collections::BTreeSet<_>>()
			.into_iter()
			.map(|seq_code| seqs.seq(seq_code)),
	);

	// Where each chunk's keys and references land in the segment.
	let mut key_base = Vec::with_capacity(n_chunks + 1);
	let mut ref_base = Vec::with_capacity(n_chunks + 1);
	key_base.push(0usize);
	ref_base.push(0usize);
	for c in &chunks {
		key_base.push(key_base.last().unwrap() + c.keys.len());
		ref_base.push(ref_base.last().unwrap() + c.constraint_indices.len());
	}
	let n_keys = *key_base.last().unwrap();
	let n_refs = *ref_base.last().unwrap();

	// Rebase each chunk's keys and word ranges onto the segment, in parallel.
	let rebased: Vec<(Vec<Key>, Vec<Range<u32>>)> = chunks
		.par_iter()
		.enumerate()
		.map(|(i, chunk)| {
			let mut keys = Vec::with_capacity(chunk.keys.len());
			let mut key_ranges = Vec::with_capacity(chunk.keys_per_word.len());
			let mut ref_offset = ref_base[i] as u32;
			let mut key_offset = key_base[i] as u32;
			let mut key_iter = chunk.keys.iter();
			for &n in &chunk.keys_per_word {
				let start = key_offset;
				for _ in 0..n {
					let &(code, count) = key_iter.next().expect("keys_per_word sums to keys.len()");
					keys.push(Key {
						operation: operation_from_code((code >> (2 * LOG_SHIFT_COUNT)) as u8),
						dense_shift_idx: dense_shift_enc.dense_idx(seqs.seq(code)),
						range: ref_offset..ref_offset + count,
					});
					ref_offset += count;
					key_offset += 1;
				}
				key_ranges.push(start..key_offset);
			}
			(keys, key_ranges)
		})
		.collect();

	let mut keys = Vec::with_capacity(n_keys);
	let mut key_ranges = Vec::with_capacity(n_words);
	for (k, r) in rebased {
		keys.extend(k);
		key_ranges.extend(r);
	}
	// Consuming the chunks frees each one's buffers as its references are appended, rather than
	// holding every chunk alive until the whole segment copy is built.
	let mut constraint_indices = Vec::with_capacity(n_refs);
	for mut chunk in chunks {
		constraint_indices.append(&mut chunk.constraint_indices);
	}

	KeySegment {
		keys,
		key_ranges,
		constraint_indices,
		dense_shift_enc,
	}
}

/// Builds the prover's dense key collection from a constraint system.
///
/// Three passes: count each word's references, scatter them into a flat array in walk order,
/// then group each word's slice into keys.
///
/// # Arguments
///
/// - `cs`: the constraint system to walk.
/// - `inout`: the split point between the public and hidden key segments.
pub(super) fn build_key_collection(cs: &ConstraintSystem, inout: InoutSegment) -> KeyCollection {
	let n_words = cs.value_vec_len();
	let n_public = cs.n_public_words(inout);

	// Pass 1: references per word, and every shift sequence in use.
	let mut offsets = vec![0usize; n_words + 1];
	let mut seqs = SeqTable::new();
	for_each_reference(cs, |word, shift_seq, _| {
		offsets[word + 1] += 1;
		seqs.note(shift_seq);
	});
	for w in 0..n_words {
		offsets[w + 1] += offsets[w];
	}
	let n_refs = offsets[n_words];

	// Pass 2: scatter every reference to its word's slot, in walk order.
	let mut refs = vec![PackedRef(0); n_refs];
	let mut cursors = offsets.clone();
	for_each_reference(cs, |word, _, r| {
		refs[cursors[word]] = r;
		cursors[word] += 1;
	});
	drop(cursors);

	// Pass 3: group per word into keys, one segment per half.
	KeyCollection {
		public: build_segment(&offsets[..=n_public], &refs, &seqs),
		hidden: build_segment(&offsets[n_public..], &refs, &seqs),
	}
}

/// The original one-pass builder, kept as the equivalence oracle for the counting-sort one.
#[cfg(test)]
pub(super) fn build_key_collection_reference(
	cs: &ConstraintSystem,
	inout: InoutSegment,
) -> KeyCollection {
	let mut builder_key_lists = BuilderKeyLists::new(cs.value_vec_len());

	// Update the builder keys lists with respect to each operand of each operation.
	builder_key_lists.update_with_constraints(Operation::Zero, &cs.zero_constraints, cs);
	builder_key_lists.update_with_constraints(Operation::BitwiseAnd, &cs.and_constraints, cs);
	builder_key_lists.update_with_constraints(Operation::IntegerMul, &cs.imul_constraints, cs);
	builder_key_lists.update_with_constraints(Operation::BinMul, &cs.bmul_constraints, cs);

	// Split the builder key lists at the public segment boundary, one half per segment.
	let hidden_lists = builder_key_lists.split_off(cs.n_public_words(inout));
	KeyCollection {
		public: KeySegment::build(builder_key_lists.into_inner()),
		hidden: KeySegment::build(hidden_lists.into_inner()),
	}
}
