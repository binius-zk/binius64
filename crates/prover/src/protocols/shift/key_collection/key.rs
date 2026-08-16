// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::Range;

use binius_field::{Field, WideMul};
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use super::operation::Operation;

/// One operation's constraints on one witness word, under one fixed sequence of two shifts.
///
/// # Overview
///
/// - A binary-field IOP paper defines one multilinear polynomial per operation, operand, and shift
///   variant.
/// - That polynomial decomposes into one matrix per bit of a word, and a key identifies one such
///   matrix.
/// - The word itself is not stored here: a key is only ever reached by looking up its word first.
///
/// # Performance
///
/// - The operation and the shift index are stored as separate fields, not merged into one key.
/// - This costs nothing during proving, since only the operation is extracted per round while the
///   shift index is used as it stands.
#[derive(Debug, Clone)]
pub struct Key {
	/// The constraint kind this key's constraints belong to.
	pub operation: Operation,
	/// Index into the owning segment's dense shift encoding.
	///
	/// The encoding decodes this back to the sequence of two shifts applied to the word.
	pub dense_shift_idx: u16,
	/// The constraint indices this key covers, as a range into the segment's flattened list.
	///
	/// A constraint index in this range names one constraint of this key's operation.
	/// The word participates in that constraint as the operand the index names, under this key's
	/// shift sequence.
	pub range: Range<u32>,
}

impl Key {
	/// Accumulates one key's weighted partial evaluation, in unreduced (wide) form.
	///
	/// - Scans the key's constraint indices once, in index order.
	/// - Weights each operand's contribution by its scalar before summing.
	/// - Leaves the sum unreduced, so several keys can be combined before one reduction.
	///
	/// ```text
	/// result = sum_operand  scalars[operand] * ( sum_{i : operand_index(i) = operand} tensor[i] )
	/// ```
	///
	/// # Arguments
	///
	/// - `constraint_indices`: the segment's full flattened list; this key reads only its range.
	/// - `r_x_prime_tensor`: the tensor value for each constraint index.
	/// - `scalars`: one weight per operand index.
	///
	/// # Returns
	///
	/// The unreduced accumulation.
	/// Reduce it to get a field element.
	#[inline]
	pub fn accumulate_wide<F: Field>(
		&self,
		constraint_indices: &[ConstraintIndex],
		r_x_prime_tensor: &[F],
		scalars: &[F],
	) -> <F as WideMul>::Output {
		// Only this key's own slice of the segment's constraint list.
		let Range { start, end } = self.range;
		let mut constraint_indices = constraint_indices[start as usize..end as usize].iter();

		let mut result = <F as WideMul>::Output::default();
		let Some(first) = constraint_indices.next() else {
			return result; // No constraints, no contribution.
		};

		// The slice is sorted by operand index, so same-operand entries sit together.
		// `acc` sums one operand's tensor values; `operand_index` says which operand that is.
		let mut operand_index = first.operand_index as usize;
		let mut acc = F::ZERO;
		acc += r_x_prime_tensor[first.constraint_index as usize];

		for current in constraint_indices {
			let current_operand_index = current.operand_index as usize;
			if current_operand_index != operand_index {
				// A new operand started: weight and bank the finished one, then start over.
				result += F::wide_mul(acc, scalars[operand_index]);
				operand_index = current_operand_index;
				acc = F::ZERO;
			}
			acc += r_x_prime_tensor[current.constraint_index as usize];
		}

		// The last operand never triggered a flush inside the loop, so weight it here.
		result + F::wide_mul(acc, scalars[operand_index])
	}

	/// Accumulates one key's weighted partial evaluation, reduced to a field element.
	///
	/// - Scans the key's constraint indices once, in index order.
	/// - Weights each operand's contribution by its scalar before summing.
	/// - Reduces the wide sum to a single field element.
	///
	/// ```text
	/// result = reduce( sum_operand  scalars[operand] * ( sum_{i : operand_index(i) = operand} tensor[i] ) )
	/// ```
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

/// One constraint referencing a shifted word, as one operand of one operation.
#[derive(Debug, Clone)]
pub struct ConstraintIndex {
	/// Which operand position of the constraint the word fills.
	pub(super) operand_index: u8,
	/// Which constraint, among the operation's constraints, this is.
	pub(super) constraint_index: u32,
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

#[cfg(test)]
mod tests {
	use std::{iter, mem};

	use binius_field::BinaryField128bGhash;
	use binius_math::FieldBuffer;

	use super::*;
	use crate::protocols::shift::PreparedOperatorData;

	type F = BinaryField128bGhash;

	fn f(value: u128) -> F {
		F::new(value)
	}

	/// Reference oracle for the weighted accumulation above.
	///
	/// Groups the key's constraint indices by operand index, instead of scanning them
	/// consecutively like the optimized version does.
	fn accumulate_by_operand<'a>(
		key: &'a Key,
		constraint_indices: &'a [ConstraintIndex],
		operator_data: &'a PreparedOperatorData<F>,
	) -> impl Iterator<Item = (usize, F)> + 'a {
		let Range { start, end } = key.range;

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

		let expected = accumulate_by_operand(&key, &constraint_indices, &operator_data)
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
		let non_contiguous_expected = accumulate_by_operand(
			&non_contiguous_key,
			&non_contiguous_constraint_indices,
			&operator_data,
		)
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
