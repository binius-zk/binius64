// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_compute::Allocator;
use binius_core::constraint_system::{ConstraintSystem, InoutSegment};
use binius_field::{BinaryField, PackedField, WideMul};
use binius_math::{FieldBuffer, FieldVec, multilinear::eq::eq_ind_partial_eval};
use binius_utils::{
	buffer::VecLike,
	checked_arithmetics::log2_ceil_usize,
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
	serialization::{DeserializeBytes, SerializationError, SerializeBytes},
};
use binius_verifier::protocols::shift::{BINMUL_ARITY, BITAND_ARITY, INTMUL_ARITY, ZERO_ARITY};
use bytes::{Buf, BufMut};
use tracing::instrument;

use super::{
	builder, dense_shift_encoding::DenseShiftEncoding, key_segment::KeySegment,
	operation::Operation,
};
use crate::protocols::shift::{
	claims::PreparedOperatorClaims,
	monster::{OuterSlotWeights, ScalarTables},
	shift_ind::ShiftChallenge,
};

/// The prover's complete view of a constraint system's shift keys, split by value-vector segment.
///
/// - The public segment covers value-vector indices from 0 up to the public word count.
/// - The hidden segment covers the rest, up to the combined length.
/// - Word indices inside each segment are relative to that segment's own start.
/// - Both phases of the shift reduction iterate the two segments in absolute value-vector order.
#[derive(Debug, Clone)]
pub struct KeyCollection {
	/// The keys of the public segment: constants and inout words.
	pub public: KeySegment,
	/// The keys of the hidden segment: private words.
	pub hidden: KeySegment,
}

impl KeyCollection {
	/// Walks a constraint system, collecting every shift key into its segment.
	///
	/// Runs as a stable counting sort by word: one pass counts the references each word
	/// receives, one pass scatters them into a flat array in walk order, and the per-word
	/// grouping into keys runs over disjoint word ranges in parallel.
	///
	/// # Arguments
	///
	/// - `cs`: the constraint system to walk.
	/// - `inout`: the split point between the public and hidden segments.
	pub fn build(cs: &ConstraintSystem, inout: InoutSegment) -> Self {
		builder::build_key_collection(cs, inout)
	}

	/// The base-2 logarithm of the hidden segment length in words, rounded up to a power of two.
	///
	/// Matches the corresponding quantity for the constraint system the collection was built from.
	/// That system guarantees this is at least the public segment's logarithm.
	///
	/// ```text
	/// log_witness_words = ceil_log2( hidden segment length in words )
	/// ```
	pub const fn log_witness_words(&self) -> usize {
		log2_ceil_usize(self.hidden.n_words())
	}

	/// Builds the constraint-matrix multilinear's two segments.
	///
	/// For each witness word, this sums every one of its keys' contributions.
	/// A key's contribution is its constraint-index accumulation, scaled by a scalar that
	/// folds together the batching power, the bit-index evaluation, and the two
	/// equality-indicator weights of the shift sequence the key names.
	///
	/// # Returns
	///
	/// The public segment, spanning the rounded-up public word count, and the hidden segment,
	/// spanning the full witness word count.
	/// The phase-2 sumcheck's sparse first round consumes both directly, without
	/// materializing their combined buffer.
	#[instrument(skip_all, name = "build_monster_segments")]
	pub fn build_monster_segments<F, P: PackedField<Scalar = F>, A: Allocator>(
		&self,
		alloc: &A,
		prepared: &PreparedOperatorClaims<F>,
		h_eval: F,
		inner: &ShiftChallenge<F>,
		outer: &ShiftChallenge<F>,
	) -> (FieldVec<P, A>, FieldVec<P, A>)
	where
		F: BinaryField,
	{
		let r_v_tensor = eq_ind_partial_eval::<F>(&inner.variant);
		let r_s_tensor = eq_ind_partial_eval::<F>(&inner.amount);

		// Invariant: a key's sequence weight factorizes across its two slots.
		//
		//     eq(r_v1, v_1) * eq(r_s1, s_1)  *  eq(r_v2, v_2) * eq(r_s2, s_2)
		//     \______ inner slot _________/     \______ outer slot ________/
		//
		// So one table per slot suffices, at `2 * SHIFT_COUNT` entries instead of
		// `SHIFT_COUNT^2`.
		let outer_weights = OuterSlotWeights::<F>::new(outer);

		// The scalars of one operation, laid out with the operand index innermost, so a
		// key's weights form one contiguous chunk its wide accumulation can index by
		// operand.
		//
		// A key's sequence selects itself through an equality indicator over both slots.
		// The h evaluation is one factor shared by every key of the operation.
		let build_scalars = |arity: usize,
		                     lambda_powers: &[F],
		                     dense_shift_enc: &DenseShiftEncoding| {
			let mut scalars = vec![F::ZERO; arity * dense_shift_enc.len()];
			for (dense_shift_idx, [inner_shift, outer_shift]) in dense_shift_enc.iter().enumerate()
			{
				let shift_scalar = h_eval
					* r_v_tensor.as_ref()[inner_shift.variant as usize]
					* r_s_tensor.as_ref()[inner_shift.amount as usize]
					* outer_weights.weight(outer_shift);
				for operand_idx in 0..arity {
					scalars[dense_shift_idx * arity + operand_idx] =
						lambda_powers[operand_idx] * shift_scalar;
				}
			}
			scalars
		};

		// Each segment has its own dense shift encoding, so it has its own scalar tables.
		let build_scalar_tables = |dense_shift_enc: &DenseShiftEncoding| ScalarTables {
			zero: build_scalars(ZERO_ARITY, &prepared.zero.lambda_powers, dense_shift_enc),
			bitand: build_scalars(BITAND_ARITY, &prepared.bitand.lambda_powers, dense_shift_enc),
			intmul: build_scalars(INTMUL_ARITY, &prepared.intmul.lambda_powers, dense_shift_enc),
			binmul: build_scalars(BINMUL_ARITY, &prepared.binmul.lambda_powers, dense_shift_enc),
		};

		// The scalar for one word of a segment: the accumulated contribution of all its
		// keys, summed unreduced and reduced once at the end.
		let word_scalar = |segment: &KeySegment, tables: &ScalarTables<F>, index: usize| {
			let wide = segment
				.word_keys(index)
				.iter()
				.map(|key| {
					// The scalar table is per operation, and its stride is that
					// operation's arity.
					let (scalars, arity) = match key.operation {
						Operation::Zero => (&tables.zero, ZERO_ARITY),
						Operation::BitwiseAnd => (&tables.bitand, BITAND_ARITY),
						Operation::IntegerMul => (&tables.intmul, INTMUL_ARITY),
						Operation::BinMul => (&tables.binmul, BINMUL_ARITY),
					};
					let base = key.dense_shift_idx as usize * arity;
					key.accumulate_wide(
						&segment.constraint_indices,
						prepared[key.operation].r_x_prime_tensor.as_ref(),
						&scalars[base..base + arity],
					)
				})
				.sum::<<F as WideMul>::Output>();
			F::reduce(wide)
		};

		// Each segment sits at the base of its buffer: the public piece fills its
		// power-of-two length exactly, the hidden piece is zero-padded up to the hidden
		// segment length.
		let build_segment = |segment: &KeySegment, log_len: usize| {
			let tables = build_scalar_tables(&segment.dense_shift_enc);
			let capacity = 1 << log_len.saturating_sub(P::LOG_WIDTH);
			let n_words = segment.n_words();
			// Full packed elements: each maps exactly `P::WIDTH` words, so `from_scalars`
			// sees a statically-sized iterator.
			// The trailing partial element is filled separately below.
			let n_full = n_words / P::WIDTH;
			// Allocate the buffer once, then fill its aligned packed elements in parallel
			// through its spare capacity.
			let mut values = alloc.alloc::<P>(capacity);
			values.spare_capacity_mut()[..n_full]
				.par_iter_mut()
				.enumerate()
				.with_min_task(WorkPerItem::FieldMuls)
				.for_each(|(chunk_index, slot)| {
					let start = chunk_index * P::WIDTH;
					slot.write(P::from_scalars(
						(0..P::WIDTH).map(|i| word_scalar(segment, &tables, start + i)),
					));
				});
			// Safety: the parallel loop above initialized every one of the `n_full`
			// slots.
			unsafe { values.set_len(n_full) };
			if !n_words.is_multiple_of(P::WIDTH) {
				let start = n_full * P::WIDTH;
				values.push(P::from_scalars(
					(start..n_words).map(|word_index| word_scalar(segment, &tables, word_index)),
				));
			}
			values.resize(capacity, P::default());
			FieldBuffer::new(log_len, values)
		};

		// The segment word count need not be a power of two; the constraint-matrix
		// multilinear spans the rounded-up count.
		let log_public_words = log2_ceil_usize(self.public.n_words());
		let public_monster = build_segment(&self.public, log_public_words);
		let hidden_monster = build_segment(&self.hidden, self.log_witness_words());

		(public_monster, hidden_monster)
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
		constraint_system::{AndConstraint, Operand, Shift, ShiftedValueIndex, ValueIndex},
		word::Word,
	};
	use binius_verifier::protocols::shift::SHIFT_COUNT;

	use super::{
		super::key::{ConstraintIndex, Key},
		*,
	};

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
			KeyCollection::build(&shifted_constraint_system(), InoutSegment::Public);

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

		let key_collection = KeyCollection::build(&cs, InoutSegment::Public);
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
			KeyCollection::build(&shifted_constraint_system(), InoutSegment::Public);

		// The shift sequences a word's keys name, as its own segment's encoding recovers them.
		let word_sequences = |segment: &KeySegment, word: usize| {
			let mut sequences = segment
				.word_keys(word)
				.iter()
				.map(|key| {
					segment
						.dense_shift_enc
						.iter()
						.nth(key.dense_shift_idx as usize)
						.unwrap()
				})
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
			KeyCollection::build(&shifted_constraint_system(), InoutSegment::Public);

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

	/// One key still being assembled while its segment is built, collecting constraint indices
	/// into its own vector rather than as a range into the segment's flattened one.
	struct BuilderKey {
		shift_seq: [Shift; 2],
		operation: Operation,
		constraint_indices: Vec<ConstraintIndex>,
	}

	/// One builder key list per word of the constraint system, indexed by word position.
	///
	/// This is the one-pass builder `KeyCollection::build` ran before the counting sort, kept
	/// verbatim as the byte-identity oracle the equivalence tests below compare against.
	struct BuilderKeyLists(Vec<Vec<BuilderKey>>);

	impl BuilderKeyLists {
		fn new(word_count: usize) -> Self {
			Self((0..word_count).map(|_| Vec::new()).collect())
		}

		fn split_off(&mut self, public_word_count: usize) -> Self {
			Self(self.0.split_off(public_word_count))
		}

		fn into_inner(self) -> Vec<Vec<BuilderKey>> {
			self.0
		}

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

	/// The former `KeySegment::build`: one segment from the builder keys lists of its words.
	fn build_key_segment_reference(builder_key_lists: Vec<Vec<BuilderKey>>) -> KeySegment {
		// Every distinct shift sequence across every word, before any per-key index is assigned.
		let dense_shift_enc = DenseShiftEncoding::new(
			builder_key_lists
				.iter()
				.flatten()
				.map(|builder_key| builder_key.shift_seq),
		);

		// Word w's keys occupy a contiguous run in the flattened keys vector.
		// A running offset gives each word's run its start and end.
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

			// Sort constraint indices by operand index, so a later linear scan can detect each
			// operand's boundary with no extra bookkeeping.
			builder_constraint_indices
				.sort_by_key(|constraint_index| constraint_index.operand_index);

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

	/// The original one-pass builder, the equivalence oracle for the counting-sort one.
	fn build_key_collection_reference(cs: &ConstraintSystem, inout: InoutSegment) -> KeyCollection {
		let mut builder_key_lists = BuilderKeyLists::new(cs.value_vec_len());

		builder_key_lists.update_with_constraints(Operation::Zero, &cs.zero_constraints, cs);
		builder_key_lists.update_with_constraints(Operation::BitwiseAnd, &cs.and_constraints, cs);
		builder_key_lists.update_with_constraints(Operation::IntegerMul, &cs.imul_constraints, cs);
		builder_key_lists.update_with_constraints(Operation::BinMul, &cs.bmul_constraints, cs);

		let hidden_lists = builder_key_lists.split_off(cs.n_public_words(inout));
		KeyCollection {
			public: build_key_segment_reference(builder_key_lists.into_inner()),
			hidden: build_key_segment_reference(hidden_lists.into_inner()),
		}
	}

	fn serialized(kc: &KeyCollection) -> Vec<u8> {
		let mut bytes = Vec::new();
		kc.serialize(&mut bytes).unwrap();
		bytes
	}

	#[test]
	fn counting_sort_builder_matches_the_reference_on_the_shifted_system() {
		let cs = shifted_constraint_system();
		for inout in [InoutSegment::Public, InoutSegment::Hidden] {
			assert_eq!(
				serialized(&KeyCollection::build(&cs, inout)),
				serialized(&build_key_collection_reference(&cs, inout)),
			);
		}
	}

	/// A random system with every operation, multi-term operands, two-slot shifts, and a few
	/// words (constants) referenced far more often than the rest — the shape that makes the
	/// per-word key lists long and interleaved.
	fn random_constraint_system(seed: u64, n_and: usize) -> ConstraintSystem {
		random_constraint_system_over(seed, n_and, 4096)
	}

	/// As [`random_constraint_system`], with the private word count chosen by the caller.
	///
	/// The count is what decides how many chunks a segment spans, so a caller that wants the
	/// cross-chunk paths exercised picks a multiple of `WORDS_PER_CHUNK`.
	fn random_constraint_system_over(
		seed: u64,
		n_and: usize,
		n_private: usize,
	) -> ConstraintSystem {
		use binius_core::constraint_system::{
			BmulConstraint, ImulConstraint, Operand, ShiftVariant, ZeroConstraint,
		};
		use rand::{RngExt, SeedableRng, rngs::StdRng};
		let mut rng = StdRng::seed_from_u64(seed);
		let n_const = 8usize;
		let n_inout = 6usize;
		let variants = [
			ShiftVariant::Sll,
			ShiftVariant::Slr,
			ShiftVariant::Sar,
			ShiftVariant::Rotr,
			ShiftVariant::Sll32,
			ShiftVariant::Srl32,
			ShiftVariant::Sra32,
			ShiftVariant::Rotr32,
		];
		let shift = |rng: &mut StdRng| -> Shift {
			if rng.random_bool(0.5) {
				return Shift::IDENTITY;
			}
			let variant = variants[rng.random_range(0..variants.len())];
			// 32-bit lanewise variants shift by less than 32.
			let bound = if variant.is_half_word() { 32 } else { 64 };
			Shift::new(variant, rng.random_range(1..bound))
		};
		let term = |rng: &mut StdRng| -> ShiftedValueIndex {
			// A quarter of all references land on the first constant, another quarter on a
			// handful of hot words; the rest spread over the whole vector.
			let value_index = match rng.random_range(0..4u8) {
				0 => ValueIndex::constant(0),
				1 => match rng.random_range(0..3u8) {
					0 => ValueIndex::constant(rng.random_range(0..n_const as u32)),
					1 => ValueIndex::inout(rng.random_range(0..n_inout as u32)),
					_ => ValueIndex::private(rng.random_range(0..8)),
				},
				_ => ValueIndex::private(rng.random_range(0..n_private as u32)),
			};
			ShiftedValueIndex::new(value_index, [shift(rng), shift(rng)])
		};
		let operand = |rng: &mut StdRng| -> Operand {
			(0..rng.random_range(1..=4)).map(|_| term(rng)).collect()
		};
		let operands = |rng: &mut StdRng, arity: usize| -> Vec<Operand> {
			(0..arity).map(|_| operand(rng)).collect()
		};
		let zero_constraints = (0..n_and / 8)
			.map(|_| ZeroConstraint(operands(&mut rng, 1).try_into().ok().unwrap()))
			.collect();
		let and_constraints = (0..n_and)
			.map(|_| AndConstraint(operands(&mut rng, 3).try_into().ok().unwrap()))
			.collect();
		let imul_constraints = (0..n_and / 16)
			.map(|_| ImulConstraint(operands(&mut rng, 4).try_into().ok().unwrap()))
			.collect();
		let bmul_constraints = (0..n_and / 4)
			.map(|_| BmulConstraint(operands(&mut rng, 6).try_into().ok().unwrap()))
			.collect();
		ConstraintSystem {
			constants: vec![Word::ZERO; n_const],
			n_inout,
			n_private,
			zero_constraints,
			and_constraints,
			imul_constraints,
			bmul_constraints,
		}
	}

	#[test]
	fn counting_sort_builder_matches_the_reference_on_random_systems() {
		for (seed, n_and) in [(1u64, 64usize), (2, 1000), (3, 20_000)] {
			let cs = random_constraint_system(seed, n_and);
			for inout in [InoutSegment::Public, InoutSegment::Hidden] {
				let fast = KeyCollection::build(&cs, inout);
				let slow = build_key_collection_reference(&cs, inout);
				assert_eq!(fast.public.keys.len(), slow.public.keys.len(), "seed {seed}");
				assert_eq!(fast.hidden.keys.len(), slow.hidden.keys.len(), "seed {seed}");
				assert_eq!(serialized(&fast), serialized(&slow), "seed {seed} inout {inout:?}");
			}
		}
	}

	// The cases above all land at two chunks, the second holding a handful of words: the default
	// private word count is `WORDS_PER_CHUNK`. Nothing there exercises rebasing a chunk whose bases
	// are neither zero nor the last, or unioning shift sequences that appear in some chunks only.
	// This spans five chunks and seventy-four, where both are ordinary rather than edge cases.
	#[test]
	fn counting_sort_builder_matches_the_reference_across_many_chunks() {
		for n_private in [5 * builder::WORDS_PER_CHUNK, 74 * builder::WORDS_PER_CHUNK] {
			let cs = random_constraint_system_over(4, 2_000, n_private);
			for inout in [InoutSegment::Public, InoutSegment::Hidden] {
				let fast = KeyCollection::build(&cs, inout);
				let slow = build_key_collection_reference(&cs, inout);
				assert_eq!(
					serialized(&fast),
					serialized(&slow),
					"n_private {n_private} inout {inout:?}"
				);
			}
		}
	}

	#[test]
	fn counting_sort_builder_handles_an_empty_system() {
		let cs = ConstraintSystem {
			constants: vec![Word::ZERO; 2],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		};
		let kc = KeyCollection::build(&cs, InoutSegment::Public);
		assert_eq!(kc.public.keys.len(), 0);
		assert_eq!(kc.hidden.keys.len(), 0);
		assert_eq!(kc.public.key_ranges.len(), 2);
		assert_eq!(kc.hidden.key_ranges.len(), 4);
		assert_eq!(
			serialized(&kc),
			serialized(&build_key_collection_reference(&cs, InoutSegment::Public))
		);
	}
}
