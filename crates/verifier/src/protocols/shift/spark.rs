// Copyright 2026 The Binius Developers

//! Dense encoding of the combined monster multilinear, as Spark consumes it.
//!
//! The monster multilinear is the sparse wiring matrix of the shift reduction. Evaluating it
//! transparently ([`OperationEvalFn`](super::OperationEvalFn)) walks every operand term of every
//! constraint, which is what makes the verifier's work linear in the constraint system. Spark
//! replaces that walk with a sub-IOP over a *dense* list of the nonzero terms, so this module fixes
//! what that list is.
//!
//! ## The identity
//!
//! Written over the dense terms, the transparent evaluation is
//!
//! $$
//! \sum_k \text{mem}_x[\text{row}(k)] \cdot \text{mem}_y[\text{col}(k)] \cdot
//!        \text{shift}[\text{shift\\_id}(k)] \cdot \text{operand}[\text{operand\\_slot}(k)],
//! $$
//!
//! which is what [`DenseMonster::evaluate`] computes. The two memories are the large dimensions
//! Spark treats as memories; the two scalar tables are small and verifier-computable, so they stay
//! transparent per-term factors.
//!
//! ## Term order
//!
//! Terms are enumerated by operation (BitAnd, IntMul, BinMul), then constraint, then operand, then
//! the shifted values within the operand — the order
//! [`OperationEvalFn`](super::OperationEvalFn) already iterates. The order fixes the memory access
//! pattern the read-correctness argument is proved against, so prover and verifier must agree on
//! it; both derive it from the same [`ConstraintSystem`], so they agree by construction.
//!
//! Zero constraints are not encoded: the shift reduction does not batch them.
//!
//! ## Address spaces
//!
//! * **Rows.** One memory holding every operation's constraint-index indicator. Each operation gets
//!   a block of `2^{log constraints}` cells holding $\widetilde{\text{eq}}(\cdot, r_x')$ for its
//!   own challenge. Blocks are laid out longest first, which lands each at an offset that is a
//!   multiple of its own length, so a block is still `eq(high bits = offset) · eq̃(low bits, r_x')`
//!   and stays evaluable in logarithmic time.
//! * **Columns.** One memory over the word-index address space `(r_y, r_segment)`, the segment
//!   challenge being the highest variable: public word `v` sits at address `v`, hidden word `v` at
//!   `2^{log_witness_words} + v - n_public_words`. This is the address space
//!   [`MonsterEvalFn`](super::check_eval) already tensors over, so the memory is exactly
//!   $\widetilde{\text{eq}}(\cdot, (r_y, r_\text{segment}))$ with nothing zero-extended.
//! * **Terms.** Padded to a power of two with terms carrying [`PADDING_OPERAND_SLOT`], whose scalar
//!   is zero. Padding terms read address 0 of both memories, so the read-correctness argument
//!   treats them as ordinary reads and needs no special case.

use std::{cmp::Reverse, iter, ops::Range};

use binius_core::{
	constraint_system::{ConstraintSystem, Operand, ValueIndex},
	word::Word,
};
use binius_field::{Field, util::powers};
use binius_math::multilinear::eq::eq_ind_partial_eval_scalars;
use binius_utils::checked_arithmetics::log2_ceil_usize;

use super::{BINMUL_ARITY, BITAND_ARITY, INTMUL_ARITY, SHIFT_VARIANT_COUNT};

/// The number of operations the shift reduction batches.
pub const N_OPERATIONS: usize = 3;

/// The arity of each batched operation, in term-enumeration order.
pub const OPERATION_ARITIES: [usize; N_OPERATIONS] = [BITAND_ARITY, INTMUL_ARITY, BINMUL_ARITY];

/// The number of `(operation, operand)` slots, one per operand of one operation.
pub const N_OPERAND_SLOTS: usize = BITAND_ARITY + INTMUL_ARITY + BINMUL_ARITY;

/// The operand slot of the padding terms, whose scalar is zero.
pub const PADDING_OPERAND_SLOT: u8 = N_OPERAND_SLOTS as u8;

/// The number of `(shift variant, shift amount)` pairs a term can select.
pub const N_SHIFT_IDS: usize = SHIFT_VARIANT_COUNT * Word::BITS;

/// One nonzero term of the combined monster multilinear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonsterTerm {
	/// Address of the term's constraint in the row memory.
	pub row: u32,
	/// Address of the term's value-vector word in the column memory.
	pub col: u32,
	/// The term's `shift_variant * Word::BITS + amount`, selecting its shift scalar.
	pub shift_id: u16,
	/// The term's `(operation, operand)` slot, selecting its operand scalar.
	pub operand_slot: u8,
}

/// The padding term, which contributes nothing to the evaluation.
const PADDING_TERM: MonsterTerm = MonsterTerm {
	row: 0,
	col: 0,
	shift_id: 0,
	operand_slot: PADDING_OPERAND_SLOT,
};

/// The dense encoding of a constraint system's combined monster multilinear.
///
/// See the module documentation for the term order and the two address spaces.
#[derive(Debug, Clone)]
pub struct DenseMonster {
	terms: Vec<MonsterTerm>,
	row_blocks: [Range<u32>; N_OPERATIONS],
	log_row_memory_len: usize,
	log_col_memory_len: usize,
}

impl DenseMonster {
	/// Encodes the monster multilinear of a constraint system.
	pub fn new(constraint_system: &ConstraintSystem) -> Self {
		let row_blocks = place_row_blocks([
			1 << constraint_system.log_and_constraints().unwrap_or(0),
			1 << constraint_system.log_imul_constraints().unwrap_or(0),
			1 << constraint_system.log_bmul_constraints().unwrap_or(0),
		]);

		// The public words occupy the low addresses of the segment-0 half; the hidden words start
		// at the segment-1 half.
		let n_public_words = constraint_system.n_public_words() as u32;
		let hidden_offset = 1u32 << constraint_system.log_witness_words();
		let col_of = |value_index: ValueIndex| match value_index.0 {
			index if index < n_public_words => index,
			index => hidden_offset + index - n_public_words,
		};

		let mut terms = Vec::new();
		push_operation_terms(
			&constraint_system.and_constraints,
			row_blocks[0].start,
			0,
			&col_of,
			&mut terms,
		);
		push_operation_terms(
			&constraint_system.imul_constraints,
			row_blocks[1].start,
			BITAND_ARITY as u8,
			&col_of,
			&mut terms,
		);
		push_operation_terms(
			&constraint_system.bmul_constraints,
			row_blocks[2].start,
			(BITAND_ARITY + INTMUL_ARITY) as u8,
			&col_of,
			&mut terms,
		);
		terms.resize(terms.len().next_power_of_two(), PADDING_TERM);

		let m_x = row_blocks
			.iter()
			.map(|block| block.end)
			.max()
			.expect("there is at least one operation");
		Self {
			terms,
			log_row_memory_len: log2_ceil_usize(m_x as usize),
			log_col_memory_len: constraint_system.log_witness_words() + 1,
			row_blocks,
		}
	}

	/// The dense terms, padded to a power-of-two count.
	pub fn terms(&self) -> &[MonsterTerm] {
		&self.terms
	}

	/// The base-2 logarithm of the term count.
	pub const fn log_n_terms(&self) -> usize {
		self.terms.len().ilog2() as usize
	}

	/// The base-2 logarithm of the row memory size.
	pub const fn log_row_memory_len(&self) -> usize {
		self.log_row_memory_len
	}

	/// The base-2 logarithm of the column memory size.
	pub const fn log_col_memory_len(&self) -> usize {
		self.log_col_memory_len
	}

	/// The row memory block of each operation, in term-enumeration order.
	pub const fn row_blocks(&self) -> &[Range<u32>; N_OPERATIONS] {
		&self.row_blocks
	}

	/// The row memory access pattern, one address per term.
	pub fn row_addrs(&self) -> Vec<usize> {
		self.terms.iter().map(|term| term.row as usize).collect()
	}

	/// The column memory access pattern, one address per term.
	pub fn col_addrs(&self) -> Vec<usize> {
		self.terms.iter().map(|term| term.col as usize).collect()
	}

	/// The row memory: each operation's block holds the equality indicator of its own constraint
	/// challenge, and the cells outside every block are zero.
	///
	/// # Panics
	///
	/// Panics unless each challenge point has as many coordinates as its block has address bits.
	pub fn row_memory<F: Field>(&self, r_x_primes: [&[F]; N_OPERATIONS]) -> Vec<F> {
		let mut memory = vec![F::ZERO; 1 << self.log_row_memory_len];
		for (block, r_x_prime) in iter::zip(&self.row_blocks, r_x_primes) {
			let tensor = eq_ind_partial_eval_scalars::<F>(r_x_prime);
			assert_eq!(tensor.len(), block.len());
			memory[block.start as usize..block.end as usize].copy_from_slice(&tensor);
		}
		memory
	}

	/// The column memory: the equality indicator over the word-index challenges, with the segment
	/// challenge as the highest variable.
	///
	/// # Panics
	///
	/// Panics unless `r_y` has one coordinate less than the column memory has address bits.
	pub fn col_memory<F: Field>(&self, r_y: &[F], r_segment: F) -> Vec<F> {
		assert_eq!(r_y.len() + 1, self.log_col_memory_len);
		let point = [r_y, &[r_segment]].concat();
		eq_ind_partial_eval_scalars(&point)
	}

	/// Evaluates the monster multilinear from the dense encoding.
	///
	/// This is the identity Spark's evaluation sumcheck proves, spelled out directly: it walks
	/// every term, so it costs what the transparent evaluation costs and exists to define the
	/// encoding's meaning, not to run in the protocol. It agrees with the transparent evaluation of
	/// [`OperationEvalFn`](super::OperationEvalFn) summed over the operations.
	pub fn evaluate<F: Field>(
		&self,
		row_memory: &[F],
		col_memory: &[F],
		shift_scalars: &[F; N_SHIFT_IDS],
		operand_scalars: &[F; N_OPERAND_SLOTS + 1],
	) -> F {
		self.terms
			.iter()
			.map(|term| {
				row_memory[term.row as usize]
					* col_memory[term.col as usize]
					* shift_scalars[term.shift_id as usize]
					* operand_scalars[term.operand_slot as usize]
			})
			.sum()
	}
}

/// The operand scalars, indexed by operand slot: the batching coefficient
/// `lambda^{operand_index + 1}` of the slot's operation.
///
/// The trailing [`PADDING_OPERAND_SLOT`] entry is zero, which is what neutralizes the padding
/// terms.
pub fn operand_scalars<F: Field>(lambdas: [F; N_OPERATIONS]) -> [F; N_OPERAND_SLOTS + 1] {
	let mut scalars = [F::ZERO; N_OPERAND_SLOTS + 1];
	let lambda_powers = iter::zip(lambdas, OPERATION_ARITIES)
		.flat_map(|(lambda, arity)| powers(lambda).skip(1).take(arity));
	// The zip stops one short of the padding slot, leaving it zero.
	for (scalar, lambda_power) in iter::zip(&mut scalars, lambda_powers) {
		*scalar = lambda_power;
	}
	scalars
}

/// Lays out the operations' row memory blocks, longest first.
///
/// Every block length is a power of two, so laying them out in descending order places each at an
/// offset that is a multiple of its own length. Alignment is what keeps a block's indicator
/// `eq(high bits = offset) · eq̃(low bits, r_x')`, and hence evaluable in logarithmic time.
fn place_row_blocks(block_lens: [u32; N_OPERATIONS]) -> [Range<u32>; N_OPERATIONS] {
	let mut operations = [0, 1, 2];
	operations.sort_by_key(|&operation| Reverse(block_lens[operation]));

	let mut blocks = [const { 0..0 }; N_OPERATIONS];
	let mut offset = 0;
	for operation in operations {
		blocks[operation] = offset..offset + block_lens[operation];
		offset += block_lens[operation];
	}
	blocks
}

/// Appends the terms of one operation's constraints, in constraint-then-operand order.
fn push_operation_terms<C, const ARITY: usize>(
	constraints: &[C],
	row_offset: u32,
	operand_slot_offset: u8,
	col_of: &impl Fn(ValueIndex) -> u32,
	terms: &mut Vec<MonsterTerm>,
) where
	C: AsRef<[Operand; ARITY]>,
{
	terms.extend(
		constraints
			.iter()
			.enumerate()
			.flat_map(|(constraint_index, constraint)| {
				constraint
					.as_ref()
					.iter()
					.enumerate()
					.flat_map(move |(operand_index, operand)| {
						operand.iter().map(move |shifted_value| MonsterTerm {
							row: row_offset + constraint_index as u32,
							col: col_of(shifted_value.value_index),
							shift_id: shifted_value.shift_variant as u16 * Word::BITS as u16
								+ shifted_value.amount as u16,
							operand_slot: operand_slot_offset + operand_index as u8,
						})
					})
			}),
	);
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::{
		AndConstraint, BmulConstraint, ImulConstraint, ShiftVariant, ShiftedValueIndex,
		ValueVecLayout,
	};
	use binius_math::test_utils::random_scalars;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::config::B128;

	/// A constraint system over a value vector of 8 public and 16 hidden words, holding the given
	/// numbers of empty constraints.
	fn constraint_system(n_and: usize, n_imul: usize, n_bmul: usize) -> ConstraintSystem {
		let layout = ValueVecLayout {
			n_const: 2,
			n_inout: 2,
			n_witness: 8,
			n_internal: 8,
			offset_inout: 4,
			offset_witness: 8,
			n_hidden_words: 16,
			n_scratch: 0,
		};
		let mut constraint_system = layout.constraint_system_shape(vec![Word::ZERO; 2]);
		constraint_system.and_constraints = vec![AndConstraint::default(); n_and];
		constraint_system.imul_constraints = vec![ImulConstraint::default(); n_imul];
		constraint_system.bmul_constraints = vec![BmulConstraint::default(); n_bmul];
		constraint_system.validate_shape().unwrap();
		constraint_system
	}

	fn shifted_value(index: u32, shift_variant: ShiftVariant, amount: u8) -> ShiftedValueIndex {
		ShiftedValueIndex {
			value_index: ValueIndex(index),
			shift_variant,
			amount,
		}
	}

	/// A two-constraint system whose terms are spread over operands and both value-vector
	/// segments: constraint 0 puts one term in operand 0 and one in operand 2, constraint 1 puts
	/// two terms in operand 1, and the last of those references a hidden word.
	fn constraint_system_with_terms() -> ConstraintSystem {
		let mut constraint_system = constraint_system(2, 0, 0);
		constraint_system.and_constraints[0].0[0] = vec![shifted_value(3, ShiftVariant::Slr, 5)];
		constraint_system.and_constraints[0].0[2] = vec![shifted_value(7, ShiftVariant::Sll, 0)];
		constraint_system.and_constraints[1].0[1] = vec![
			shifted_value(0, ShiftVariant::Rotr32, 63),
			shifted_value(8, ShiftVariant::Sar, 1),
		];
		constraint_system
	}

	#[test]
	fn terms_follow_the_transparent_iteration_order() {
		let dense = DenseMonster::new(&constraint_system_with_terms());

		assert_eq!(
			dense.terms(),
			[
				MonsterTerm {
					row: 0,
					col: 3,
					shift_id: Word::BITS as u16 + 5,
					operand_slot: 0,
				},
				MonsterTerm {
					row: 0,
					col: 7,
					shift_id: 0,
					operand_slot: 2,
				},
				MonsterTerm {
					row: 1,
					col: 0,
					shift_id: 7 * Word::BITS as u16 + 63,
					operand_slot: 1,
				},
				// The first hidden word sits at the start of the segment-1 half.
				MonsterTerm {
					row: 1,
					col: 16,
					shift_id: 2 * Word::BITS as u16 + 1,
					operand_slot: 1,
				},
			]
		);
	}

	#[test]
	fn terms_are_padded_to_a_power_of_two_with_neutral_terms() {
		let mut constraint_system = constraint_system(1, 0, 0);
		constraint_system.and_constraints[0].0[0] = vec![
			shifted_value(1, ShiftVariant::Sll, 2),
			shifted_value(2, ShiftVariant::Sll, 3),
			shifted_value(3, ShiftVariant::Sll, 4),
		];

		let dense = DenseMonster::new(&constraint_system);

		assert_eq!(dense.terms().len(), 4);
		assert_eq!(dense.log_n_terms(), 2);
		assert_eq!(dense.terms()[3], PADDING_TERM);
		// The padding term's operand scalar is zero, so it drops out of the evaluation.
		let operand_scalars = operand_scalars::<B128>([B128::ONE; N_OPERATIONS]);
		assert_eq!(operand_scalars[PADDING_OPERAND_SLOT as usize], B128::ZERO);
	}

	#[test]
	fn row_blocks_are_disjoint_and_self_aligned() {
		for (n_and, n_imul, n_bmul) in [(21, 5, 3), (64, 0, 0), (1, 100, 7), (0, 0, 0)] {
			let dense = DenseMonster::new(&constraint_system(n_and, n_imul, n_bmul));
			let blocks = dense.row_blocks();

			let mut sorted = blocks.clone();
			sorted.sort_by_key(|block| block.start);
			let mut end = 0;
			for block in &sorted {
				assert!(block.start >= end, "blocks overlap: {blocks:?}");
				assert_eq!(
					block.start % block.len() as u32,
					0,
					"block is not self-aligned: {blocks:?}"
				);
				end = block.end;
			}
			assert!(end <= 1 << dense.log_row_memory_len());
		}
	}

	/// The two address sequences index the memories they are read from, so every address must be
	/// in range.
	#[test]
	fn address_sequences_index_within_the_two_memories() {
		let dense = DenseMonster::new(&constraint_system_with_terms());

		assert_eq!(dense.row_addrs().len(), dense.terms().len());
		assert_eq!(dense.col_addrs().len(), dense.terms().len());
		assert!(
			dense
				.row_addrs()
				.iter()
				.all(|&addr| addr < 1 << dense.log_row_memory_len())
		);
		assert!(
			dense
				.col_addrs()
				.iter()
				.all(|&addr| addr < 1 << dense.log_col_memory_len())
		);
	}

	#[test]
	fn each_operation_indexes_its_own_block_of_the_row_memory() {
		let dense = DenseMonster::new(&constraint_system(21, 5, 3));

		let mut rng = StdRng::seed_from_u64(0);
		let r_x_primes = [5, 3, 2].map(|n_vars| random_scalars::<B128>(&mut rng, n_vars));
		let row_memory = dense.row_memory([&r_x_primes[0], &r_x_primes[1], &r_x_primes[2]]);

		assert_eq!(row_memory.len(), 1 << dense.log_row_memory_len());
		for (block, r_x_prime) in iter::zip(dense.row_blocks(), &r_x_primes) {
			assert_eq!(
				row_memory[block.start as usize..block.end as usize],
				eq_ind_partial_eval_scalars::<B128>(r_x_prime)
			);
		}
	}
}
