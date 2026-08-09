// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, mem, ops::Range};

use binius_core::{
	ShiftVariant,
	constraint_system::{ConstraintSystem, Operand, ShiftedValueIndex},
	word::Word,
};
use binius_field::{Field, WideMul};
use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	serialization::{DeserializeBytes, SerializationError, SerializeBytes},
};
use binius_verifier::protocols::shift::SHIFT_VARIANT_COUNT;
use bytes::{Buf, BufMut};

use super::PreparedOperatorData;

/// The number of shifts a key could name, over which [`shift_index`] is dense.
const SHIFT_PAIR_COUNT: usize = SHIFT_VARIANT_COUNT * Word::BITS;

/// Places a shift in the full space of `SHIFT_PAIR_COUNT` shifts, for the tables that build and
/// invert a [`DenseShiftEncoding`].
///
/// # Panics
///
/// Panics if the shift amount is not below `Word::BITS`.
fn shift_index((variant, amount): (ShiftVariant, u8)) -> usize {
	assert!((amount as usize) < Word::BITS, "shift amount {amount} out of range");
	variant as usize * Word::BITS + amount as usize
}

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

/// A dense re-encoding of the shifts occurring in a key segment.
///
/// A key names one of `SHIFT_PAIR_COUNT` shifts, of which a constraint system typically uses a few
/// dozen. Encoding the shifts a segment does use as a contiguous range lets the prover size its
/// per-shift tables by the shifts actually used rather than by the whole space.
#[derive(Debug, Clone, Default)]
pub struct DenseShiftEncoding {
	/// The shift each dense index encodes, ordered by variant and then by amount.
	shifts: Vec<(ShiftVariant, u8)>,
}

impl DenseShiftEncoding {
	/// Builds the encoding of the shifts in an iterator, which need be neither sorted nor distinct.
	///
	/// # Panics
	///
	/// Panics if any shift amount is not below `Word::BITS`.
	fn new(shifts: impl IntoIterator<Item = (ShiftVariant, u8)>) -> Self {
		let mut used = [None; SHIFT_PAIR_COUNT];
		for shift in shifts {
			used[shift_index(shift)] = Some(shift);
		}
		Self {
			shifts: used.into_iter().flatten().collect(),
		}
	}

	/// The number of distinct shifts the segment uses.
	pub const fn len(&self) -> usize {
		self.shifts.len()
	}

	/// Whether the segment uses no shifted values at all.
	pub const fn is_empty(&self) -> bool {
		self.shifts.is_empty()
	}

	/// The shift a dense index encodes.
	///
	/// # Panics
	///
	/// Panics if the dense index is not below [`Self::len`].
	#[inline]
	pub fn decode(&self, dense_idx: usize) -> (ShiftVariant, u8) {
		self.shifts[dense_idx]
	}

	/// The shifts the segment uses, in dense index order.
	pub fn iter(&self) -> impl Iterator<Item = (ShiftVariant, u8)> + '_ {
		self.shifts.iter().copied()
	}

	/// The dense index of every shift, for lookup while the keys are built.
	///
	/// The entries of the shifts the segment does not use are meaningless.
	fn inverse(&self) -> [u16; SHIFT_PAIR_COUNT] {
		let mut inverse = [0u16; SHIFT_PAIR_COUNT];
		for (dense_idx, &shift) in self.shifts.iter().enumerate() {
			inverse[shift_index(shift)] = dense_idx as u16;
		}
		inverse
	}
}

/// A `Key` specifies an operation, a shift, and a range of constraint indices.
///
/// The key identifies a 2D matrix of constraint information: the constraints of one operation in
/// which one witness word participates, shifted by one shift variant and one shift amount. Every
/// `Key` corresponds to a unique word (not referenced in the `Key`). The `range` specifies a range
/// within a list of constraint indices, those constraint indices in which the word participates
/// with respect to the key. If constraint index `i` is among the values within the range, that
/// means the word participates in constraint `i` of operation `operation`, as the operand that
/// constraint index names, shifted by the variant and amount the key's shift index encodes.
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
/// - **Dense shift index**: The shift variant and shift amount, as encoded by the segment
/// - **Range**: Constraint indices where this shifted word appears
///
/// # Shift index encoding
///
/// The shift index is an index into the [`DenseShiftEncoding`] of the [`KeySegment`] holding the
/// key, which decodes it back to the shift variant and shift amount. Keys index the shifts their
/// segment actually uses rather than the whole shift space, so a table the prover builds per shift
/// is proportional to the former.
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
/// - **dense_shift_enc**: The `(shift variant, shift amount)` pairs the segment's keys name
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
	/// The `(shift variant, shift amount)` of the shifted value the key references.
	pub shift: (ShiftVariant, u8),
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
		for ShiftedValueIndex {
			value_index,
			shift_variant,
			amount,
		} in operand_value.as_ref()
		{
			// The lists are indexed by word position, so resolve the term's segment-relative
			// index against the segment starts.
			let builder_keys = &mut builder_key_lists[cs.word_offset(*value_index)];
			let shift = (*shift_variant, *amount);

			// Find existing builder key or create a new one for this (operation, shift) pair
			let constraint_index = ConstraintIndex {
				operand_index: operand_index as u8,
				constraint_index: constraint_idx as u32,
			};
			if let Some(builder_key) = builder_keys
				.iter_mut()
				.find(|key| key.shift == shift && key.operation == operation)
			{
				builder_key.constraint_indices.push(constraint_index);
			} else {
				builder_keys.push(BuilderKey {
					shift,
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
pub fn build_key_collection(cs: &ConstraintSystem) -> KeyCollection {
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
	let hidden_lists = builder_key_lists.split_off(cs.n_public_words());
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
			.map(|builder_key| builder_key.shift),
	);
	let dense_shift_idx = dense_shift_enc.inverse();

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
			shift,
			operation,
			constraint_indices: mut builder_constraint_indices,
		} = builder_key;

		// Sort constraint indices by operand index so we can save work in [`Key::accumulate`].
		builder_constraint_indices.sort_by_key(|constraint_index| constraint_index.operand_index);

		let start = constraint_indices.len() as u32;
		constraint_indices.extend(builder_constraint_indices);
		let end = constraint_indices.len() as u32;
		keys.push(Key {
			dense_shift_idx: dense_shift_idx[shift_index(shift)],
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
		for (variant, amount) in &self.shifts {
			variant.serialize(&mut write_buf)?;
			amount.serialize(&mut write_buf)?;
		}
		Ok(())
	}
}

impl DeserializeBytes for DenseShiftEncoding {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
		let len = u32::deserialize(&mut read_buf)? as usize;
		let mut shifts = Vec::with_capacity(len);
		for _ in 0..len {
			let variant = ShiftVariant::deserialize(&mut read_buf)?;
			let amount = u8::deserialize(&mut read_buf)?;
			shifts.push((variant, amount));
		}

		// A dense index is only meaningful against a list of distinct shifts, which the ordering
		// this writes them in also gives.
		let amounts_in_range = shifts
			.iter()
			.all(|&(_, amount)| (amount as usize) < Word::BITS);
		if !amounts_in_range
			|| !shifts
				.windows(2)
				.all(|shifts| shift_index(shifts[0]) < shift_index(shifts[1]))
		{
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
	use binius_core::constraint_system::{AndConstraint, ValueIndex};
	use binius_field::{BinaryField128bGhash, Field};
	use binius_math::FieldBuffer;

	use super::*;

	type F = BinaryField128bGhash;

	fn f(value: u128) -> F {
		F::new(value)
	}

	/// A constraint system with a handful of distinct shifts, differing between the two segments.
	///
	/// The public segment references `(Sll, 0)` and `(Slr, 3)`; the hidden one references
	/// `(Sll, 0)`, `(Sar, 7)` and `(Rotr, 1)`.
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
	fn dense_shift_encoding_covers_the_pairs_its_segment_uses() {
		let key_collection = build_key_collection(&shifted_constraint_system());

		let public_pairs = key_collection
			.public
			.dense_shift_enc
			.iter()
			.collect::<Vec<_>>();
		assert_eq!(public_pairs, [(ShiftVariant::Sll, 0), (ShiftVariant::Slr, 3)]);

		let hidden_pairs = key_collection
			.hidden
			.dense_shift_enc
			.iter()
			.collect::<Vec<_>>();
		assert_eq!(
			hidden_pairs,
			[
				(ShiftVariant::Sll, 0),
				(ShiftVariant::Sar, 7),
				(ShiftVariant::Rotr, 1),
			]
		);

		// The point of the encoding: a segment names far fewer pairs than the space holds.
		assert!(key_collection.hidden.dense_shift_enc.len() < SHIFT_PAIR_COUNT);
	}

	#[test]
	fn keys_index_their_segments_dense_encoding() {
		let key_collection = build_key_collection(&shifted_constraint_system());

		// The shifts a word's keys name, as its own segment's encoding decodes them.
		let word_pairs = |segment: &KeySegment, word: usize| {
			let mut pairs = segment
				.word_keys(word)
				.iter()
				.map(|key| segment.dense_shift_enc.decode(key.dense_shift_idx as usize))
				.collect::<Vec<_>>();
			pairs.sort();
			pairs
		};

		// Value index 1 is the second public word; value index 5 the second hidden one.
		assert_eq!(
			word_pairs(&key_collection.public, 1),
			[(ShiftVariant::Sll, 0), (ShiftVariant::Slr, 3)]
		);
		assert_eq!(
			word_pairs(&key_collection.hidden, 1),
			[
				(ShiftVariant::Sll, 0),
				(ShiftVariant::Sar, 7),
				(ShiftVariant::Rotr, 1),
			]
		);
	}

	#[test]
	fn dense_shift_encoding_survives_serialization() {
		let key_collection = build_key_collection(&shifted_constraint_system());

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

	#[test]
	fn dense_shift_encoding_rejects_an_unordered_serialization() {
		let encoding = DenseShiftEncoding {
			shifts: vec![(ShiftVariant::Slr, 3), (ShiftVariant::Sll, 0)],
		};

		let mut buf = Vec::new();
		encoding.serialize(&mut buf).unwrap();

		assert!(matches!(
			DenseShiftEncoding::deserialize(buf.as_slice()),
			Err(SerializationError::InvalidConstruction { .. })
		));
	}

	#[test]
	fn dense_shift_encoding_rejects_an_out_of_range_shift_amount() {
		let encoding = DenseShiftEncoding {
			shifts: vec![(ShiftVariant::Sll, Word::BITS as u8)],
		};

		let mut buf = Vec::new();
		encoding.serialize(&mut buf).unwrap();

		assert!(matches!(
			DenseShiftEncoding::deserialize(buf.as_slice()),
			Err(SerializationError::InvalidConstruction { .. })
		));
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
