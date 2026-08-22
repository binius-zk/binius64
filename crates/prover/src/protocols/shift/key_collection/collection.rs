// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_compute::{Allocator, VecLike};
use binius_field::{BinaryField, PackedField, WideMul};
use binius_math::{FieldBuffer, FieldVec, multilinear::eq::eq_ind_partial_eval};
use binius_utils::{
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
	dense_shift_encoding::DenseShiftEncoding, key_segment::KeySegment, operation::Operation,
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
