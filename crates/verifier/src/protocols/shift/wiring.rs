// Copyright 2026 The Binius Developers

use binius_core::constraint_system::{ConstraintSystem, InoutSegment, Operand};
use binius_math::multilinear::sparse::SparseBitVector;
use getset::{CopyGetters, Getters};

use super::{LOG_SHIFT_COUNT, log_constraint_point};
use crate::reduction::LOG_OPERANDS;

/// The address bits below the word index: the operand column, then the inner and outer shift.
pub(super) const LOG_TERM_BITS: usize = LOG_OPERANDS + 2 * LOG_SHIFT_COUNT;

/// The wiring matrix as two bit vectors, one per committed segment, with the widths of the point
/// they are read at.
///
/// Each set bit is one operand term, at the address
///
/// ```text
/// bits, low to high:  operand (LOG_OPERANDS) | inner shift (9) | outer shift (9) | word | constraint
/// ```
///
/// A shift slot is [`Shift::index`](binius_core::constraint_system::Shift::index). The word index
/// is relative to its segment, `log_public_words` wide for the public vector and
/// `log_segment_words` wide for the hidden one. The constraint index is `log_constraint_point`
/// wide. The operand column runs over each operation's operands in `[zero, bitand, intmul,
/// binmul]` order. A term listed twice cancels.
///
/// The bit order is the order of the point the shift reduction evaluates the vectors at, so this
/// is the one place that fixes the layout.
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct WiringInfo {
	/// The terms whose word lies in the public segment.
	#[getset(get = "pub")]
	public_segment: SparseBitVector,
	/// The terms whose word lies in the hidden segment.
	#[getset(get = "pub")]
	hidden_segment: SparseBitVector,
	/// The word-index variables the public segment spans.
	#[getset(get_copy = "pub")]
	log_public_words: usize,
	/// The word-index variables the shift reduction runs over, the wider of the two segments'
	/// spans: the length of `r_y`.
	#[getset(get_copy = "pub")]
	log_segment_words: usize,
	/// The constraint-index variables: the length of `r_x`.
	#[getset(get_copy = "pub")]
	log_constraint_point: usize,
}

impl WiringInfo {
	/// Lays out the wiring matrix of `cs`, with its inout values in `inout`.
	///
	/// # Panics
	///
	/// Panics if the hidden address does not fit a `u64` index below `2^63`: that is,
	/// `LOG_OPERANDS + 18 + log_constraint_point + log_segment_words < 64`, or
	/// `log_constraint_point + log_segment_words ≤ 41`.
	pub fn new(cs: &ConstraintSystem, inout: InoutSegment) -> Self {
		let log_public_words = cs.log_public_words(inout);
		let log_segment_words = cs.log_segment_words(inout);
		let log_constraint_point = log_constraint_point(cs);
		let log_words = [log_public_words, log_segment_words];
		let log_lens = log_words.map(|log_words| LOG_TERM_BITS + log_words + log_constraint_point);
		assert!(
			log_lens[1] < u64::BITS as usize,
			"the wiring address needs {} bits; at most 63 fit",
			log_lens[1]
		);

		// Each operand term's segment and address, in the column order the reduction batches.
		let n_public_words = cs.n_public_words(inout);
		let mut indices = [Vec::new(), Vec::new()];
		let mut push = |column: usize, constraint_index: usize, terms: &Operand| {
			for term in terms {
				let word = cs.word_offset(term.value_index);
				let (segment, word) = if word < n_public_words {
					(0, word)
				} else {
					(1, word - n_public_words)
				};
				let index = column as u64
					| (term.inner().index() as u64) << LOG_OPERANDS
					| (term.outer().index() as u64) << (LOG_OPERANDS + LOG_SHIFT_COUNT)
					| (word as u64) << LOG_TERM_BITS
					| (constraint_index as u64) << (LOG_TERM_BITS + log_words[segment]);
				indices[segment].push(index);
			}
		};

		/// Walks one operation, its operands at the columns from `first_column` on, and returns the
		/// column after its last.
		fn walk<C: AsRef<[Operand; ARITY]>, const ARITY: usize>(
			first_column: usize,
			constraints: &[C],
			push: &mut impl FnMut(usize, usize, &Operand),
		) -> usize {
			for (constraint_index, constraint) in constraints.iter().enumerate() {
				for (operand_index, terms) in constraint.as_ref().iter().enumerate() {
					push(first_column + operand_index, constraint_index, terms);
				}
			}
			first_column + ARITY
		}

		let column = walk(0, &cs.zero_constraints, &mut push);
		let column = walk(column, &cs.and_constraints, &mut push);
		let column = walk(column, &cs.imul_constraints, &mut push);
		walk(column, &cs.bmul_constraints, &mut push);

		let [public, hidden] = indices;
		let [public_log_len, hidden_log_len] = log_lens;
		Self {
			public_segment: SparseBitVector::new(public_log_len, public),
			hidden_segment: SparseBitVector::new(hidden_log_len, hidden),
			log_public_words,
			log_segment_words,
			log_constraint_point,
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::ZeroConstraint;

	use super::*;

	/// A system of `2^log_constraints` empty ZERO constraints over `2^log_words` private words.
	fn empty_system(log_constraints: usize, log_words: usize) -> ConstraintSystem {
		ConstraintSystem {
			constants: Vec::new(),
			n_inout: 0,
			n_private: 1 << log_words,
			zero_constraints: vec![ZeroConstraint::default(); 1 << log_constraints],
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		}
	}

	#[test]
	fn the_widest_address_that_fits_is_63_bits() {
		let wiring = WiringInfo::new(&empty_system(15, 26), InoutSegment::Public);
		assert_eq!(wiring.hidden_segment().log_len(), 63);
	}

	#[test]
	#[should_panic(expected = "the wiring address needs 64 bits")]
	fn a_system_whose_address_does_not_fit_is_rejected() {
		let _ = WiringInfo::new(&empty_system(16, 26), InoutSegment::Public);
	}
}
