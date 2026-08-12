// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{collections::BTreeSet, iter, mem, ops::Range};

use binius_core::{
	ShiftVariant,
	constraint_system::{ConstraintSystem, InoutSegment, Operand, Shift},
};
use binius_field::{Field, WideMul};
use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	serialization::{DeserializeBytes, SerializationError, SerializeBytes},
};
use bytes::{Buf, BufMut};

use super::{DOUBLE_SHIFT_UNSUPPORTED, PreparedOperatorData};

/// Represents the type of operations handled by the shift protocol.
///
/// The shift protocol supports four fundamental operation types that correspond
/// to the constraint types in Binius64:
///
/// # Operation Types
///
/// - **Zero**: Corresponds to ZERO constraints of the form `VAL = 0`
/// - **BitwiseAnd**: Corresponds to AND constraints of the form `A & B ^ C = 0`
/// - **IntegerMul**: Corresponds to IMUL constraints of the form `A * B = (HI << 64) | LO`
/// - **BinMul**: Corresponds to BMUL constraints of the form `A * B = C` in the GHASH field
///
/// These operations work with shifted value indices to efficiently encode
/// computations on 64-bit words without requiring separate shift constraints.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u16)]
pub enum Operation {
	Zero,
	BitwiseAnd,
	IntegerMul,
	BinMul,
}

/// A dense re-encoding of the shift sequences occurring in a key segment.
///
/// A key names a sequence of two shifts, each slot drawn from
/// [`SHIFT_COUNT`](binius_verifier::protocols::shift::SHIFT_COUNT) spellings.
///
/// - The sequence space is therefore `SHIFT_COUNT^2` = 262,144 entries.
/// - A constraint system uses a few dozen of them.
/// - So the sequences a segment does use are re-encoded as a contiguous range.
///
/// A per-sequence table is then sized by the sequences present, not by the space they come from.
/// At most one sequence exists per shifted value index, which is what it has always been.
/// Only the space grew, which is why a sequence is located by search rather than by a lookup table.
#[derive(Debug, Clone, Default)]
pub struct DenseShiftEncoding {
	/// The shift sequence each dense index encodes, in ascending sequence order.
	///
	/// Invariant: sorted and distinct, which is what makes [`Self::dense_idx`] a binary search and
	/// what a deserialized encoding is checked for.
	shifts: Vec<[Shift; 2]>,
}

impl DenseShiftEncoding {
	/// Builds the encoding of the shift sequences in an iterator, neither sorted nor distinct.
	///
	/// # Panics
	///
	/// Panics if the sequences do not fit the `u16` a [`Key`] addresses them with.
	/// That needs more than 65,536 distinct sequences in one segment, which no real system reaches.
	fn new(shifts: impl IntoIterator<Item = [Shift; 2]>) -> Self {
		// A sorted set dedupes and orders in one pass, over the sequences present.
		let shifts = shifts.into_iter().collect::<BTreeSet<_>>();
		assert!(
			shifts.len() <= u16::MAX as usize + 1,
			"a key segment uses {} distinct shift sequences, more than the u16 dense index addresses",
			shifts.len()
		);
		Self {
			shifts: shifts.into_iter().collect(),
		}
	}

	/// The number of distinct shift sequences the segment uses.
	pub const fn len(&self) -> usize {
		self.shifts.len()
	}

	/// Whether the segment uses no shifted values at all.
	pub const fn is_empty(&self) -> bool {
		self.shifts.is_empty()
	}

	/// The shift sequence a dense index encodes.
	///
	/// # Panics
	///
	/// Panics if the dense index is not below [`Self::len`].
	#[inline]
	pub fn decode(&self, dense_idx: usize) -> [Shift; 2] {
		self.shifts[dense_idx]
	}

	/// The shift sequences the segment uses, in dense index order.
	pub fn iter(&self) -> impl Iterator<Item = [Shift; 2]> + '_ {
		self.shifts.iter().copied()
	}

	/// Where the inner shift of every sequence the segment uses sits in the space one slot spans,
	/// in dense index order.
	///
	/// The reduction's row axis is one shift slot wide, so a sequence is placed by its inner shift.
	/// The outer slot holds the identity on every sequence, so distinct sequences have distinct
	/// inner shifts and the indices come out strictly increasing.
	/// That is what lets two segments' encodings merge.
	///
	/// # Panics
	///
	/// Panics in debug builds if a sequence carries a shift outside its inner slot.
	pub fn shift_indices(&self) -> impl Iterator<Item = usize> + '_ {
		self.shifts.iter().map(|&[inner, outer]| {
			debug_assert!(outer.is_identity(), "{DOUBLE_SHIFT_UNSUPPORTED}");
			inner.index()
		})
	}

	/// The dense index of one shift sequence, for lookup while the keys are built.
	///
	/// The sequences are sorted, so this is a binary search over the ones the segment uses.
	/// A lookup table over the whole sequence space would instead be 262,144 entries wide.
	///
	/// # Panics
	///
	/// Panics if the sequence is not one this encoding covers.
	fn dense_idx(&self, shift_seq: [Shift; 2]) -> u16 {
		let index = self
			.shifts
			.binary_search(&shift_seq)
			.expect("the encoding covers every shift sequence its segment's keys name");
		// `new` bounds the length, so every index it yields fits.
		index as u16
	}
}

/// A `Key` specifies an operation, a shift sequence, and a range of constraint indices.
///
/// The key identifies a 2D matrix of constraint information: the constraints of one operation in
/// which one witness word participates, shifted by one sequence of two shifts.
/// Every `Key` corresponds to a unique word (not referenced in the `Key`).
/// The `range` specifies a range within a list of constraint indices, those constraint indices in
/// which the word participates with respect to the key.
/// If constraint index `i` is among the values within the range, that means the word participates
/// in constraint `i` of operation `operation`, as the operand that constraint index names, shifted
/// by the sequence the key's shift index encodes.
///
/// # Relationship to Formal Specification
///
/// The paper defines one `M` multilinear polynomial for each (operation, operand, shift variant)
/// tuple. Each `M` multilinear forms a 3D matrix that decomposes into `Word::BITS`
/// 2D matrices. Each `Key` corresponds to one such 2D matrix. We operate at 2D granularity
/// because the prover performs field operations on 2D matrices during both protocol phases.
///
/// # Structure
///
/// - **Operation**: Constraint type (ZERO, AND, IMUL or BMUL)
/// - **Dense shift index**: The shift sequence, as encoded by the segment
/// - **Range**: Constraint indices where this shifted word appears
///
/// # Shift index encoding
///
/// The shift index is an index into the [`DenseShiftEncoding`] of the [`KeySegment`] holding the
/// key, which decodes it back to the sequence's two shifts.
/// Keys index the sequences their segment actually uses rather than the whole sequence space.
/// A table the prover builds per sequence is therefore proportional to the former.
///
/// # Performance Considerations
///
/// The operation remains separate from the shift index for cleaner code organization with no
/// performance cost. During proving, only the operation needs extraction; the shift index is used
/// as it stands.
#[derive(Debug, Clone)]
pub struct Key {
	pub operation: Operation,
	pub dense_shift_idx: u16,
	pub range: Range<u32>,
}

impl Key {
	/// Accumulates the partial evaluations of an operation matrix for the key, partitioned by
	/// operand index.
	///
	/// A [`Key`] references the operation constraints where one witness word is an operand. This
	/// accumulates the partial evaluation of the operation matrix for this key.
	///
	/// ## Returns
	/// An iterator of tuples, where the first is the operand ID in the operation and the second is
	/// the accumulated value of the partial evaluation tensor.
	#[inline]
	pub fn accumulate_by_operand<'a, F: Field>(
		&'a self,
		constraint_indices: &'a [ConstraintIndex],
		operator_data: &'a PreparedOperatorData<F>,
	) -> impl Iterator<Item = (usize, F)> + 'a {
		let Range { start, end } = self.range;

		let mut iter = constraint_indices[start as usize..end as usize].iter();
		let mut acc = F::ZERO;
		let mut maybe_current = iter.next();
		iter::from_fn(move || {
			let current = maybe_current?;

			acc += operator_data.r_x_prime_tensor.as_ref()[current.constraint_index as usize];
			for next in &mut iter {
				maybe_current = Some(next);
				if next.operand_index != current.operand_index {
					let ret = mem::take(&mut acc);
					return Some((current.operand_index as usize, ret));
				}
				acc += operator_data.r_x_prime_tensor.as_ref()[next.constraint_index as usize];
			}

			maybe_current = None;
			Some((current.operand_index as usize, mem::take(&mut acc)))
		})
	}

	/// Accumulates the partial evaluation of an operation matrix for the key, in unreduced (wide)
	/// form.
	///
	/// A [`Key`] references the operation constraints where one witness word is an operand. This
	/// accumulates the partial evaluation of the operation matrix for this key, weighting each
	/// operand's contribution by `scalars[operand_index]` and fusing that weighting into the
	/// consecutive-operand scan. The caller reduces the result via [`WideMul::reduce`], and may
	/// sum several wide accumulations before that single reduction.
	#[inline]
	pub fn accumulate_wide<F: Field>(
		&self,
		constraint_indices: &[ConstraintIndex],
		r_x_prime_tensor: &[F],
		scalars: &[F],
	) -> <F as WideMul>::Output {
		let Range { start, end } = self.range;
		let mut constraint_indices = constraint_indices[start as usize..end as usize].iter();

		let mut result = <F as WideMul>::Output::default();
		let Some(first) = constraint_indices.next() else {
			return result;
		};

		let mut operand_index = first.operand_index as usize;
		let mut acc = F::ZERO;
		acc += r_x_prime_tensor[first.constraint_index as usize];

		for current in constraint_indices {
			let current_operand_index = current.operand_index as usize;
			if current_operand_index != operand_index {
				result += F::wide_mul(acc, scalars[operand_index]);
				operand_index = current_operand_index;
				acc = F::ZERO;
			}
			acc += r_x_prime_tensor[current.constraint_index as usize];
		}

		result + F::wide_mul(acc, scalars[operand_index])
	}

	/// Accumulates the partial evaluation of an operation matrix for the key.
	///
	/// This is [`Self::accumulate_wide`] followed by a single reduction.
	#[inline]
	pub fn accumulate<F: Field>(
		&self,
		constraint_indices: &[ConstraintIndex],
		r_x_prime_tensor: &[F],
		scalars: &[F],
	) -> F {
		F::reduce(self.accumulate_wide(constraint_indices, r_x_prime_tensor, scalars))
	}
}

/// The keys for the words of one segment of the value vector.
///
/// The prover operates in both phases by iterating through `key_ranges` (one range per word of
/// the segment), then accessing the corresponding keys in the `keys` vector. Each key contains a
/// range that indexes into `constraint_indices` to identify which constraints involve that
/// particular shifted operand.
///
/// # Structure
///
/// - **keys**: All keys of the segment flattened into a single vector
/// - **key_ranges**: For every word of the segment there is a range of keys within the `keys`
///   vector
/// - **constraint_indices**: Flattened list of constraint indices referenced by the keys
/// - **dense_shift_enc**: The shift sequences the segment's keys name
///
/// # Organization
///
/// Keys are organized by word index for efficient batch processing. For the word at index `w`
/// *within the segment*, `key_ranges[w]` gives the range of keys in the `keys` vector that
/// correspond to that word. Each key's range field then points into `constraint_indices` to
/// specify which constraints involve that particular shifted operand.
#[derive(Debug, Clone)]
pub struct KeySegment {
	pub keys: Vec<Key>,
	pub key_ranges: Vec<Range<u32>>,
	pub constraint_indices: Vec<ConstraintIndex>,
	pub dense_shift_enc: DenseShiftEncoding,
}

impl KeySegment {
	/// The number of words the segment covers.
	pub const fn n_words(&self) -> usize {
		self.key_ranges.len()
	}

	/// The keys for the word at the given segment-relative index.
	pub fn word_keys(&self, index: usize) -> &[Key] {
		let Range { start, end } = self.key_ranges[index];
		&self.keys[start as usize..end as usize]
	}
}

/// A collection of keys that organizes the prover's view of the constraint system.
///
/// The keys are split by value-vector segment: one [`KeySegment`] for the public words
/// (value-vector indices `[0, n_public_words)`) and one for the hidden words (indices
/// `[n_public_words, combined_len)`). Word indices within each segment are
/// segment-relative. The phases iterate both segments in absolute value-vector order.
#[derive(Debug, Clone)]
pub struct KeyCollection {
	pub public: KeySegment,
	pub hidden: KeySegment,
}

impl KeyCollection {
	/// The total number of words covered by both segments.
	pub const fn n_words(&self) -> usize {
		self.public.n_words() + self.hidden.n_words()
	}

	/// The base-2 logarithm of the hidden segment length in words, rounded up to a power of
	/// two.
	///
	/// Matches [`ConstraintSystem::log_witness_words`] for the system the collection was built
	/// from; that system guarantees this is at least the public segment's logarithm.
	///
	/// [`ConstraintSystem::log_witness_words`]: binius_core::constraint_system::ConstraintSystem::log_witness_words
	pub const fn log_witness_words(&self) -> usize {
		log2_ceil_usize(self.hidden.n_words())
	}
}

/// A `BuilderKey` is a key that is being built up during `KeyCollection`
/// construction. It is a temporary structure that is later transformed
/// into a `Key`.
///
/// It differs from a `Key` by storing a vector of constraint indices directly,
/// rather than a range that indexes into the flattened `constraint_indices` vector.
/// During construction, these indices are later flattened to create the final `Key`.
struct BuilderKey {
	/// The shift sequence of the shifted value the key references, inner shift first.
	pub shift_seq: [Shift; 2],
	pub operation: Operation,
	pub constraint_indices: Vec<ConstraintIndex>,
}

/// Indexes a reference to a shifted value index, appearing in a constraint operand.
#[derive(Debug, Clone)]
pub struct ConstraintIndex {
	operand_index: u8,
	constraint_index: u32,
}

/// Updates the list of `BuilderKey` objects with respect to an operand of an operation during
/// `KeyCollection` construction.
fn update_with_operand(
	operation: Operation,
	operand_index: usize,
	operand_values: impl Iterator<Item = impl AsRef<Operand>>,
	cs: &ConstraintSystem,
	builder_key_lists: &mut [Vec<BuilderKey>],
) {
	for (constraint_idx, operand_value) in operand_values.enumerate() {
		// Each operand value is a Vec<ShiftedValueIndex> - multiple shifted word references
		for term in operand_value.as_ref() {
			// The lists are indexed by word position, so resolve the term's segment-relative
			// index against the segment starts.
			let builder_keys = &mut builder_key_lists[cs.word_offset(term.value_index)];
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

/// Updates the list of `BuilderKey` objects with respect to every operand of every constraint of
/// one operation.
///
/// Operands are indexed by their position in the constraint's operand array, which is the order
/// the shift reduction batches them in.
fn update_with_constraints<C, const ARITY: usize>(
	operation: Operation,
	constraints: &[C],
	cs: &ConstraintSystem,
	builder_key_lists: &mut [Vec<BuilderKey>],
) where
	C: AsRef<[Operand; ARITY]>,
{
	for operand_index in 0..ARITY {
		update_with_operand(
			operation,
			operand_index,
			constraints
				.iter()
				.map(|constraint| &constraint.as_ref()[operand_index]),
			cs,
			builder_key_lists,
		);
	}
}

/// Constructs a `KeyCollection` from a constraint system.
///
/// `inout` is where the proving protocol places the inout values, which is what the two key
/// segments split along.
pub fn build_key_collection(cs: &ConstraintSystem, inout: InoutSegment) -> KeyCollection {
	// Initialize a temporary list of builder keys lists, one for each committed word.
	let mut builder_key_lists: Vec<Vec<BuilderKey>> =
		(0..cs.value_vec_len()).map(|_| Vec::new()).collect();

	// Update the builder keys lists with respect to each operand of each operation.
	update_with_constraints(Operation::Zero, &cs.zero_constraints, cs, &mut builder_key_lists);
	update_with_constraints(Operation::BitwiseAnd, &cs.and_constraints, cs, &mut builder_key_lists);
	update_with_constraints(
		Operation::IntegerMul,
		&cs.imul_constraints,
		cs,
		&mut builder_key_lists,
	);
	update_with_constraints(Operation::BinMul, &cs.bmul_constraints, cs, &mut builder_key_lists);

	// Split the builder keys lists at the public segment boundary and build one `KeySegment`
	// per half.
	let hidden_lists = builder_key_lists.split_off(cs.n_public_words(inout));
	KeyCollection {
		public: build_key_segment(builder_key_lists),
		hidden: build_key_segment(hidden_lists),
	}
}

/// Computes all fields of a [`KeySegment`] from the builder keys lists of its words.
fn build_key_segment(builder_key_lists: Vec<Vec<BuilderKey>>) -> KeySegment {
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
			constraint_indices: mut builder_constraint_indices,
		} = builder_key;

		// Sort constraint indices by operand index so we can save work in [`Key::accumulate`].
		builder_constraint_indices.sort_by_key(|constraint_index| constraint_index.operand_index);

		let start = constraint_indices.len() as u32;
		constraint_indices.extend(builder_constraint_indices);
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

// Serialization implementations

impl SerializeBytes for Operation {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		let val = match self {
			Operation::BitwiseAnd => 0u8,
			Operation::IntegerMul => 1u8,
			Operation::BinMul => 2u8,
			Operation::Zero => 3u8,
		};
		val.serialize(write_buf)
	}
}

impl DeserializeBytes for Operation {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let val = u8::deserialize(&mut read_buf)?;
		match val {
			0 => Ok(Operation::BitwiseAnd),
			1 => Ok(Operation::IntegerMul),
			2 => Ok(Operation::BinMul),
			3 => Ok(Operation::Zero),
			_ => Err(SerializationError::UnknownEnumVariant {
				name: "Operation",
				index: val,
			}),
		}
	}
}

impl SerializeBytes for Key {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.operation.serialize(&mut write_buf)?;
		self.dense_shift_idx.serialize(&mut write_buf)?;
		self.range.start.serialize(&mut write_buf)?;
		self.range.end.serialize(write_buf)
	}
}

impl DeserializeBytes for Key {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let operation = Operation::deserialize(&mut read_buf)?;
		let dense_shift_idx = u16::deserialize(&mut read_buf)?;
		let start = u32::deserialize(&mut read_buf)?;
		let end = u32::deserialize(&mut read_buf)?;
		Ok(Key {
			operation,
			dense_shift_idx,
			range: start..end,
		})
	}
}

impl SerializeBytes for ConstraintIndex {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.operand_index.serialize(&mut write_buf)?;
		self.constraint_index.serialize(write_buf)
	}
}

impl DeserializeBytes for ConstraintIndex {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let operand_index = u8::deserialize(&mut read_buf)?;
		let constraint_index = u32::deserialize(&mut read_buf)?;
		Ok(ConstraintIndex {
			operand_index,
			constraint_index,
		})
	}
}

impl SerializeBytes for DenseShiftEncoding {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		(self.shifts.len() as u32).serialize(&mut write_buf)?;
		for shift_seq in &self.shifts {
			for shift in shift_seq {
				shift.variant.serialize(&mut write_buf)?;
				shift.amount.serialize(&mut write_buf)?;
			}
		}
		Ok(())
	}
}

impl DeserializeBytes for DenseShiftEncoding {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let len = u32::deserialize(&mut read_buf)? as usize;
		let mut shifts = Vec::with_capacity(len);
		for _ in 0..len {
			let mut shift_seq = [Shift::IDENTITY; 2];
			for shift in &mut shift_seq {
				let variant = ShiftVariant::deserialize(&mut read_buf)?;
				let amount = u8::deserialize(&mut read_buf)?;
				*shift = Shift { variant, amount };
			}
			shifts.push(shift_seq);
		}

		// Half-word (*32) variants cap at 32, full-width ones at 64.
		// An amount past its variant's bound denotes no shift at all.
		let amounts_in_range = shifts
			.iter()
			.flatten()
			.all(|shift| (shift.amount as usize) < shift.variant.max_amount());
		// A dense index is only meaningful against a list of distinct sequences, which the strictly
		// ascending order this writes them in also gives.
		let strictly_ascending = shifts.windows(2).all(|window| window[0] < window[1]);
		if !amounts_in_range || !strictly_ascending {
			return Err(SerializationError::InvalidConstruction {
				name: "DenseShiftEncoding::shifts",
			});
		}
		// A key addresses a sequence with a `u16`, so a longer list could not be indexed.
		if len > u16::MAX as usize + 1 {
			return Err(SerializationError::InvalidConstruction {
				name: "DenseShiftEncoding::shifts",
			});
		}

		Ok(DenseShiftEncoding { shifts })
	}
}

impl SerializeBytes for KeySegment {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.keys.serialize(&mut write_buf)?;

		// Serialize key_ranges as pairs of start/end
		(self.key_ranges.len() as u32).serialize(&mut write_buf)?;
		for range in &self.key_ranges {
			range.start.serialize(&mut write_buf)?;
			range.end.serialize(&mut write_buf)?;
		}

		self.constraint_indices.serialize(&mut write_buf)?;
		self.dense_shift_enc.serialize(write_buf)
	}
}

impl DeserializeBytes for KeySegment {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let keys = Vec::<Key>::deserialize(&mut read_buf)?;

		// Deserialize key_ranges
		let len = u32::deserialize(&mut read_buf)? as usize;
		let mut key_ranges = Vec::with_capacity(len);
		for _ in 0..len {
			let start = u32::deserialize(&mut read_buf)?;
			let end = u32::deserialize(&mut read_buf)?;
			key_ranges.push(start..end);
		}

		let constraint_indices = Vec::<ConstraintIndex>::deserialize(&mut read_buf)?;
		let dense_shift_enc = DenseShiftEncoding::deserialize(&mut read_buf)?;

		Ok(KeySegment {
			keys,
			key_ranges,
			constraint_indices,
			dense_shift_enc,
		})
	}
}

impl SerializeBytes for KeyCollection {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		// Version for forward compatibility; version 3 introduced the dense shift encoding.
		const VERSION: u32 = 3;
		VERSION.serialize(&mut write_buf)?;

		self.public.serialize(&mut write_buf)?;
		self.hidden.serialize(write_buf)
	}
}

impl DeserializeBytes for KeyCollection {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		const VERSION: u32 = 3;
		let version = u32::deserialize(&mut read_buf)?;
		if version != VERSION {
			return Err(SerializationError::InvalidConstruction {
				name: "KeyCollection::version",
			});
		}

		let public = KeySegment::deserialize(&mut read_buf)?;
		let hidden = KeySegment::deserialize(read_buf)?;

		Ok(KeyCollection { public, hidden })
	}
}

#[cfg(test)]
mod tests {
	use binius_core::{
		constraint_system::{AndConstraint, ShiftedValueIndex, ValueIndex},
		word::Word,
	};
	use binius_field::{BinaryField128bGhash, Field};
	use binius_math::FieldBuffer;
	use binius_verifier::protocols::shift::SHIFT_COUNT;

	use super::*;

	type F = BinaryField128bGhash;

	fn f(value: u128) -> F {
		F::new(value)
	}

	/// A shift sequence carrying one shift, which the canonical form places in the inner slot.
	fn single(shift: Shift) -> [Shift; 2] {
		[shift, Shift::IDENTITY]
	}

	/// A constraint system with a handful of distinct shifts, differing between the two segments.
	///
	/// The public segment references `Sll(0)` and `Slr(3)`.
	/// The hidden one references `Sll(0)`, `Sar(7)` and `Rotr(1)`.
	/// Every outer slot is the identity, so the sequences sort by their inner shift alone.
	fn shifted_constraint_system() -> ConstraintSystem {
		// The system has four constants and no inout values, so the public segment is the
		// constants and the hidden one is the private values.
		let public = ValueIndex::constant(1);
		let hidden = ValueIndex::private(1);
		ConstraintSystem {
			constants: vec![Word::ZERO; 4],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: vec![AndConstraint([
				vec![
					ShiftedValueIndex::plain(public),
					ShiftedValueIndex::srl(public, 3),
				],
				vec![ShiftedValueIndex::sar(hidden, 7)],
				vec![
					ShiftedValueIndex::rotr(hidden, 1),
					ShiftedValueIndex::plain(hidden),
				],
			])],
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		}
	}

	#[test]
	fn dense_shift_encoding_covers_the_sequences_its_segment_uses() {
		let key_collection =
			build_key_collection(&shifted_constraint_system(), InoutSegment::Public);

		let public_sequences = key_collection
			.public
			.dense_shift_enc
			.iter()
			.collect::<Vec<_>>();
		assert_eq!(public_sequences, [single(Shift::IDENTITY), single(Shift::srl(3))]);

		let hidden_sequences = key_collection
			.hidden
			.dense_shift_enc
			.iter()
			.collect::<Vec<_>>();
		assert_eq!(
			hidden_sequences,
			[
				single(Shift::IDENTITY),
				single(Shift::sar(7)),
				single(Shift::rotr(1)),
			]
		);

		// The point of the encoding: a segment names far fewer sequences than the space holds, and
		// the space is now the square of one slot's alphabet.
		assert!(key_collection.hidden.dense_shift_enc.len() < SHIFT_COUNT * SHIFT_COUNT);
	}

	#[test]
	fn dense_shift_encoding_distinguishes_sequences_sharing_an_inner_shift() {
		// Two terms sharing an inner shift but differing outside must land on distinct indices.
		// Keyed on the inner shift alone they would collide and accumulate into one row.
		let hidden = ValueIndex::private(1);
		let cs = ConstraintSystem {
			constants: vec![Word::ZERO; 4],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: vec![AndConstraint([
				vec![
					ShiftedValueIndex::new(hidden, [Shift::srl(3), Shift::sll(3)]),
					ShiftedValueIndex::new(hidden, [Shift::srl(3), Shift::sll(5)]),
					ShiftedValueIndex::srl(hidden, 3),
				],
				Vec::new(),
				Vec::new(),
			])],
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		};

		let key_collection = build_key_collection(&cs, InoutSegment::Public);
		let sequences = key_collection
			.hidden
			.dense_shift_enc
			.iter()
			.collect::<Vec<_>>();
		assert_eq!(
			sequences,
			[
				single(Shift::srl(3)),
				[Shift::srl(3), Shift::sll(3)],
				[Shift::srl(3), Shift::sll(5)],
			]
		);

		// The word's three keys therefore hold three distinct dense indices.
		let mut indices = key_collection
			.hidden
			.word_keys(1)
			.iter()
			.map(|key| key.dense_shift_idx)
			.collect::<Vec<_>>();
		indices.sort_unstable();
		assert_eq!(indices, [0, 1, 2]);
	}

	#[test]
	fn keys_index_their_segments_dense_encoding() {
		let key_collection =
			build_key_collection(&shifted_constraint_system(), InoutSegment::Public);

		// The shift sequences a word's keys name, as its own segment's encoding decodes them.
		let word_sequences = |segment: &KeySegment, word: usize| {
			let mut sequences = segment
				.word_keys(word)
				.iter()
				.map(|key| segment.dense_shift_enc.decode(key.dense_shift_idx as usize))
				.collect::<Vec<_>>();
			sequences.sort();
			sequences
		};

		// Value index 1 is the second public word; value index 5 the second hidden one.
		assert_eq!(
			word_sequences(&key_collection.public, 1),
			[single(Shift::IDENTITY), single(Shift::srl(3))]
		);
		assert_eq!(
			word_sequences(&key_collection.hidden, 1),
			[
				single(Shift::IDENTITY),
				single(Shift::sar(7)),
				single(Shift::rotr(1)),
			]
		);
	}

	#[test]
	fn dense_shift_encoding_survives_serialization() {
		let key_collection =
			build_key_collection(&shifted_constraint_system(), InoutSegment::Public);

		let mut buf = Vec::new();
		key_collection.serialize(&mut buf).unwrap();
		let deserialized = KeyCollection::deserialize(buf.as_slice()).unwrap();

		for (segment, deserialized) in [
			(&key_collection.public, &deserialized.public),
			(&key_collection.hidden, &deserialized.hidden),
		] {
			assert_eq!(
				segment.dense_shift_enc.iter().collect::<Vec<_>>(),
				deserialized.dense_shift_enc.iter().collect::<Vec<_>>()
			);
		}
	}

	// Serializes an encoding built raw, bypassing `new`'s sorting and deduplication, so that a
	// malformed list reaches the deserializer.
	fn deserialize_raw(shifts: Vec<[Shift; 2]>) -> Result<DenseShiftEncoding, SerializationError> {
		let mut buf = Vec::new();
		DenseShiftEncoding { shifts }.serialize(&mut buf).unwrap();
		DenseShiftEncoding::deserialize(buf.as_slice())
	}

	#[test]
	fn dense_shift_encoding_rejects_an_unordered_serialization() {
		// A dense index means nothing against an unsorted list: `dense_idx` binary-searches it.
		match deserialize_raw(vec![single(Shift::srl(3)), single(Shift::IDENTITY)]).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "DenseShiftEncoding::shifts");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn dense_shift_encoding_rejects_a_repeated_sequence() {
		// Ascending order is checked strictly, so a repeat is rejected along with a swap: two equal
		// sequences would give one shift sequence two dense indices.
		match deserialize_raw(vec![single(Shift::srl(3)), single(Shift::srl(3))]).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "DenseShiftEncoding::shifts");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn dense_shift_encoding_rejects_an_out_of_range_shift_amount() {
		// The bound is the variant's own: a half-word (*32) variant caps at 32, not at 64.
		let out_of_range = Shift {
			variant: ShiftVariant::Sll32,
			amount: 32,
		};
		match deserialize_raw(vec![single(out_of_range)]).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "DenseShiftEncoding::shifts");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn dense_shift_encoding_rejects_an_out_of_range_outer_shift_amount() {
		// Both slots are checked, so an outer amount its variant cannot represent is rejected too.
		let out_of_range = Shift {
			variant: ShiftVariant::Sll,
			amount: Word::BITS as u8,
		};
		match deserialize_raw(vec![[Shift::srl(3), out_of_range]]).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "DenseShiftEncoding::shifts");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
	}

	#[test]
	fn dense_shift_encoding_indexes_every_sequence_it_covers() {
		// Indexing inverts decoding, so a key's index names the sequence it is weighted by.
		// The input is unsorted and repeats one sequence, which also pins the sort and the dedupe.
		let sequences = [
			[Shift::srl(3), Shift::sll(3)],
			single(Shift::rotr(1)),
			single(Shift::IDENTITY),
			single(Shift::rotr(1)),
		];
		let encoding = DenseShiftEncoding::new(sequences);

		assert_eq!(encoding.len(), 3);
		for dense_idx in 0..encoding.len() {
			let sequence = encoding.decode(dense_idx);
			assert_eq!(encoding.dense_idx(sequence) as usize, dense_idx);
		}
	}

	#[test]
	fn accumulate_matches_grouped_operand_accumulation() {
		let constraint_indices = vec![
			ConstraintIndex {
				operand_index: 0,
				constraint_index: 1,
			},
			ConstraintIndex {
				operand_index: 0,
				constraint_index: 3,
			},
			ConstraintIndex {
				operand_index: 1,
				constraint_index: 0,
			},
			ConstraintIndex {
				operand_index: 2,
				constraint_index: 2,
			},
			ConstraintIndex {
				operand_index: 2,
				constraint_index: 4,
			},
		];
		let key = Key {
			operation: Operation::BitwiseAnd,
			dense_shift_idx: 0,
			range: 0..constraint_indices.len() as u32,
		};
		let operator_data = PreparedOperatorData {
			batched_eval: F::ZERO,
			r_zhat_prime: F::ZERO,
			r_x_prime_tensor: FieldBuffer::from_values(&[
				f(2),
				f(3),
				f(5),
				f(7),
				f(11),
				f(13),
				f(17),
				f(19),
			]),
			lambda_powers: vec![f(23), f(29), f(31)],
		};

		let expected = key
			.accumulate_by_operand(&constraint_indices, &operator_data)
			.map(|(operand_index, acc)| acc * operator_data.lambda_powers[operand_index])
			.sum::<F>();

		assert_eq!(
			key.accumulate(
				&constraint_indices,
				operator_data.r_x_prime_tensor.as_ref(),
				&operator_data.lambda_powers
			),
			expected
		);

		let non_contiguous_constraint_indices = vec![
			ConstraintIndex {
				operand_index: 0,
				constraint_index: 1,
			},
			ConstraintIndex {
				operand_index: 1,
				constraint_index: 3,
			},
			ConstraintIndex {
				operand_index: 0,
				constraint_index: 0,
			},
			ConstraintIndex {
				operand_index: 2,
				constraint_index: 2,
			},
			ConstraintIndex {
				operand_index: 1,
				constraint_index: 4,
			},
		];
		let non_contiguous_key = Key {
			operation: Operation::BitwiseAnd,
			dense_shift_idx: 0,
			range: 0..non_contiguous_constraint_indices.len() as u32,
		};
		let non_contiguous_expected = non_contiguous_key
			.accumulate_by_operand(&non_contiguous_constraint_indices, &operator_data)
			.map(|(operand_index, acc)| acc * operator_data.lambda_powers[operand_index])
			.sum::<F>();

		assert_eq!(
			non_contiguous_key.accumulate(
				&non_contiguous_constraint_indices,
				operator_data.r_x_prime_tensor.as_ref(),
				&operator_data.lambda_powers
			),
			non_contiguous_expected
		);

		let empty_key = Key {
			operation: Operation::BitwiseAnd,
			dense_shift_idx: 0,
			range: 0..0,
		};
		assert_eq!(
			empty_key.accumulate(
				&constraint_indices,
				operator_data.r_x_prime_tensor.as_ref(),
				&operator_data.lambda_powers
			),
			F::ZERO
		);
	}
}
