// Copyright 2025 Irreducible Inc.

//! The batched BitAnd-check witness built from a populated batch value table.

use binius_core::{
	constraint_system::{AndConstraint, ShiftedValueIndex},
	verify::eval_shifted_word,
	word::Word,
};
use binius_utils::rayon::prelude::*;

use crate::ValueTable;

/// The operand columns of the BitAnd check for a whole batch of instances.
///
/// The BitAnd check works on three columns of words, one row per AND constraint.
/// It enforces the bitwise relation `A & B == C` on every row.
///
/// This holds those three columns for a batch of `K = 2^log_instances` instances at once.
/// The rows are stacked in instance-major order, the same layout the batch witness uses:
///
/// ```text
///         instance 0         instance 1            instance K-1
///   A: [ a_0 .. a_{n-1} ][ a_0 .. a_{n-1} ] ... [ a_0 .. a_{n-1} ]
///   B: [ b_0 .. b_{n-1} ][ b_0 .. b_{n-1} ] ... [ b_0 .. b_{n-1} ]
///   C: [ c_0 .. c_{n-1} ][ c_0 .. c_{n-1} ] ... [ c_0 .. c_{n-1} ]
///       \___ n_and ___/
/// ```
///
/// The row index splits cleanly into the instance and the per-instance constraint:
///
/// ```text
/// row = instance * n_and + local_constraint
/// ```
///
/// - The high `log_instances` bits select the instance.
/// - The low bits select the constraint within that instance.
/// - The reduction reads each column as a multilinear over `log_instances + log(n_and)` bits.
///
/// `n_and` is a power of two when the constraints come from a prepared constraint system.
/// So the batch forms a clean hypercube whose high coordinates are the instance index.
#[derive(Clone, Debug)]
pub struct BatchAndCheckWitness {
	/// Operand `A` of every constraint of every instance, instance-major.
	a: Vec<Word>,
	/// Operand `B` of every constraint of every instance, instance-major.
	b: Vec<Word>,
	/// Operand `C` of every constraint of every instance, instance-major.
	c: Vec<Word>,
}

impl BatchAndCheckWitness {
	/// Builds the batched BitAnd witness from a populated batch value table.
	///
	/// Every AND constraint is evaluated against every instance's committed words.
	/// This produces `K * n_and` rows, laid out instance-major.
	///
	/// An operand is a XOR of shifted committed values.
	/// Its evaluation is the same fold the single-instance check uses, over a different word slice.
	///
	/// # Arguments
	///
	/// - `table`: the populated batch witness, one block of committed words per instance.
	/// - `and_constraints`: the per-instance AND constraints, shared by every instance.
	///
	/// Pass constraints from a prepared constraint system, so their count is a power of two.
	///
	/// The constraints reference only committed values, never the dropped scratch tail.
	/// So each instance's committed-word block holds everything an operand can read.
	///
	/// # Panics
	///
	/// Panics if the constraint count or the instance count is not a power of two.
	pub fn build(table: &ValueTable, and_constraints: &[AndConstraint]) -> Self {
		// Rows per instance, and total rows across the batch.
		let n_and = and_constraints.len();
		let n_instances = table.n_instances();
		let total = n_instances * n_and;

		// Both dimensions are powers of two, so both are at least 1.
		// - The row count `K * n_and` is then at least 1, so the witness is never empty.
		// - The chunk size used below is then never zero, so the parallel split is well-defined.
		assert!(n_and.is_power_of_two(), "constraint count must be a power of two");
		assert!(n_instances.is_power_of_two(), "instance count must be a power of two");

		// One column each for the three operands, laid out instance-major.
		let mut a = vec![Word::ZERO; total];
		let mut b = vec![Word::ZERO; total];
		let mut c = vec![Word::ZERO; total];

		// Fill one instance's contiguous block of rows per parallel task.
		// Each chunk is exactly one instance.
		// The blocks are disjoint, so the instances never contend for the same rows.
		a.par_chunks_mut(n_and)
			.zip(b.par_chunks_mut(n_and))
			.zip(c.par_chunks_mut(n_and))
			.enumerate()
			.for_each(|(instance, ((a_block, b_block), c_block))| {
				// This instance's committed words; every operand index lands inside this slice.
				let words = table.instance(instance);

				// Evaluate the three operands of each constraint against this instance's words.
				for (j, constraint) in and_constraints.iter().enumerate() {
					a_block[j] = eval_operand_words(words, &constraint.a);
					b_block[j] = eval_operand_words(words, &constraint.b);
					c_block[j] = eval_operand_words(words, &constraint.c);
				}
			});

		Self { a, b, c }
	}

	/// Operand `A` column, `K * n_and` rows in instance-major order.
	pub fn a(&self) -> &[Word] {
		&self.a
	}

	/// Operand `B` column, `K * n_and` rows in instance-major order.
	pub fn b(&self) -> &[Word] {
		&self.b
	}

	/// Operand `C` column, `K * n_and` rows in instance-major order.
	pub fn c(&self) -> &[Word] {
		&self.c
	}

	/// Consumes the witness into its three operand columns `(A, B, C)`.
	///
	/// This is the shape the AND reduction destructures to drive its sumcheck.
	pub fn into_columns(self) -> (Vec<Word>, Vec<Word>, Vec<Word>) {
		(self.a, self.b, self.c)
	}
}

/// Evaluates one operand against a single instance's committed words.
///
/// An operand is a XOR of shifted values:
///
/// ```text
/// operand(words) = XOR_t  shift_t( words[index_t] )
/// ```
///
/// The words are read straight from the instance's slice.
/// So the batch never rebuilds one value vector per instance.
/// Every operand index is below the committed length, so indexing the slice stays in range.
#[inline]
fn eval_operand_words(words: &[Word], operand: &[ShiftedValueIndex]) -> Word {
	operand.iter().fold(Word::ZERO, |acc, sv| {
		let word = words[sv.value_index.0 as usize];
		acc ^ eval_shifted_word(word, sv.shift_variant, sv.amount)
	})
}

#[cfg(test)]
mod tests {
	use binius_core::{constraint_system::ValueVec, verify::eval_operand};
	use binius_frontend::{Circuit, CircuitBuilder, Wire};
	use proptest::prelude::*;

	use super::*;

	// The prepared per-instance AND constraints, padded to a power of two.
	// This mirrors how the prover feeds constraints downstream.
	fn table_constraints(c: &AndCircuit) -> Vec<AndConstraint> {
		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		cs.and_constraints
	}

	// A circuit asserting `z == (x & y) ^ w`, over four public words.
	//
	//     inputs : x, y, w, z   (all inout)
	//     gate   : and = x & y
	//     assert : and ^ w == z
	struct AndCircuit {
		circuit: Circuit,
		x: Wire,
		y: Wire,
		w: Wire,
		z: Wire,
	}

	fn and_circuit() -> AndCircuit {
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let y = builder.add_inout();
		let w = builder.add_inout();
		let z = builder.add_inout();
		let and = builder.band(x, y);
		let lhs = builder.bxor(and, w);
		builder.assert_eq("z_eq_x_and_y_xor_w", lhs, z);
		AndCircuit {
			circuit: builder.build(),
			x,
			y,
			w,
			z,
		}
	}

	// Populate one instance per input tuple; the instance count is the tuple count.
	//
	// Each tuple is `(x, y, w)`, the three free inputs.
	// The output is derived as `z = (x & y) ^ w`, so every tuple satisfies the circuit.
	//
	// `w` is an arbitrary mask that only feeds the XOR, never the AND.
	// So a tuple like `(1, 3, 7)` means `x=1, y=3, w=7`, not `1 & 3 = 7`.
	fn populate_table(c: &AndCircuit, inputs: &[(u64, u64, u64)]) -> ValueTable {
		let log_instances = inputs.len().ilog2() as usize;
		ValueTable::populate(&c.circuit, log_instances, |i, filler| {
			let (x, y, w) = inputs[i];
			filler[c.x] = Word(x);
			filler[c.y] = Word(y);
			filler[c.w] = Word(w);
			filler[c.z] = Word((x & y) ^ w);
		})
		.unwrap()
	}

	// The reference for one instance: the core operand evaluator on its reconstructed value vec.
	// This is exactly what the single-instance BitAnd witness builder computes.
	fn reference_rows(
		and_constraints: &[AndConstraint],
		vv: &ValueVec,
	) -> (Vec<Word>, Vec<Word>, Vec<Word>) {
		let mut a = Vec::new();
		let mut b = Vec::new();
		let mut c = Vec::new();
		for constraint in and_constraints {
			a.push(eval_operand(vv, &constraint.a));
			b.push(eval_operand(vv, &constraint.b));
			c.push(eval_operand(vv, &constraint.c));
		}
		(a, b, c)
	}

	#[test]
	fn columns_are_instance_major_with_the_expected_shape() {
		let c = and_circuit();

		// Fixture state: 2^2 = 4 instances with distinct, satisfying inputs.
		let inputs = [
			(1, 3, 7),
			(5, 6, 0),
			(9, 12, 0xFF),
			(0xABCD, 0x0F0F, 0x1234),
		];
		let table = populate_table(&c, &inputs);

		let and_constraints = &table_constraints(&c);
		let witness = BatchAndCheckWitness::build(&table, and_constraints);

		// Shape: K * n_and rows, with K = 4.
		let n_and = and_constraints.len();
		assert_eq!(witness.a().len(), 4 * n_and);
		assert_eq!(witness.b().len(), 4 * n_and);
		assert_eq!(witness.c().len(), 4 * n_and);

		// Invariant: row `instance * n_and + j` is constraint `j` of that instance.
		// Each instance's block equals the single-instance reference for its inputs.
		for instance in 0..table.n_instances() {
			let vv = table.instance_value_vec(instance);
			let (a_ref, b_ref, c_ref) = reference_rows(and_constraints, &vv);

			let start = instance * n_and;
			assert_eq!(&witness.a()[start..start + n_and], a_ref.as_slice());
			assert_eq!(&witness.b()[start..start + n_and], b_ref.as_slice());
			assert_eq!(&witness.c()[start..start + n_and], c_ref.as_slice());
		}
	}

	#[test]
	fn and_relation_holds_on_every_row() {
		let c = and_circuit();

		// Fixture state: 4 satisfying instances, each tuple `(x, y, w)`.
		let table = populate_table(&c, &[(1, 3, 7), (5, 6, 0), (9, 12, 0xFF), (0xF0, 0x0F, 1)]);
		let witness = BatchAndCheckWitness::build(&table, &table_constraints(&c));

		// The single AND constraint is `and = x & y`, so each row is `A=x`, `B=y`, `C=x&y`.
		// A satisfying witness therefore makes `A & B == C` hold on every row.
		//
		// Padded rows have empty operands, so `0 & 0 == 0` holds there too.
		for ((a, b), c) in witness.a().iter().zip(witness.b()).zip(witness.c()) {
			assert_eq!(a.0 & b.0, c.0);
		}
	}

	#[test]
	fn single_instance_batch_matches_the_reference() {
		let c = and_circuit();

		// Fixture state: log_instances = 0 → exactly one instance (K = 1).
		let table = populate_table(&c, &[(0xABCD, 0x0F0F, 0x55)]);
		let and_constraints = table_constraints(&c);
		let witness = BatchAndCheckWitness::build(&table, &and_constraints);

		// The degenerate batch reproduces the single-instance BitAnd columns exactly.
		let vv = table.instance_value_vec(0);
		let (a_ref, b_ref, c_ref) = reference_rows(&and_constraints, &vv);
		assert_eq!(witness.a(), a_ref.as_slice());
		assert_eq!(witness.b(), b_ref.as_slice());
		assert_eq!(witness.c(), c_ref.as_slice());
	}

	#[test]
	#[should_panic(expected = "constraint count must be a power of two")]
	fn build_rejects_non_power_of_two_constraint_count() {
		let c = and_circuit();

		// Fixture state: a valid batch with one instance (K = 1).
		let table = populate_table(&c, &[(1, 3, 7)]);

		// Invariant: the per-instance constraint count must be a power of two.
		//
		// Mutation: hand the builder 3 constraints.
		//
		//     3 is not a power of two → build asserts on the count before reading any row.
		//
		// The operands are empty, so the panic is the count check, never an out-of-range index.
		let three = vec![AndConstraint::default(); 3];
		let _ = BatchAndCheckWitness::build(&table, &three);
	}

	proptest! {
		// Invariant: every batch row equals the single-instance reference for that instance.
		//
		//     witness[instance * n_and + j]  ==  eval_operand(instance value vec, constraint j)
		//
		// This pins the batched, slice-based evaluator to the core value-vec evaluator.
		// And since each instance is satisfying, the AND relation `A & B == C` holds on every row.
		#[test]
		fn batch_rows_match_single_instance_reference(
			inputs in prop::collection::vec((any::<u64>(), any::<u64>(), any::<u64>()), 4),
		) {
			let c = and_circuit();
			let table = populate_table(&c, &inputs);

			let and_constraints = table_constraints(&c);
			let n_and = and_constraints.len();
			let witness = BatchAndCheckWitness::build(&table, &and_constraints);

			for instance in 0..table.n_instances() {
				let vv = table.instance_value_vec(instance);
				let (a_ref, b_ref, c_ref) = reference_rows(&and_constraints, &vv);

				let start = instance * n_and;
				prop_assert_eq!(&witness.a()[start..start + n_and], a_ref.as_slice());
				prop_assert_eq!(&witness.b()[start..start + n_and], b_ref.as_slice());
				prop_assert_eq!(&witness.c()[start..start + n_and], c_ref.as_slice());
			}

			// The built columns satisfy the AND constraint on every row, padding included.
			for ((a, b), c) in witness.a().iter().zip(witness.b()).zip(witness.c()) {
				prop_assert_eq!(a.0 & b.0, c.0);
			}
		}
	}
}
