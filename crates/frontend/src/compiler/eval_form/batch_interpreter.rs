// Copyright 2025-2026 The Binius Developers
//! Batched bytecode interpreter for circuit evaluation.
//!
//! This is the structure-of-arrays counterpart to the single-instance interpreter.
//! It evaluates the same bytecode over many independent instances of one circuit at once.
//! The opcode dispatch is shared through the executor and the execution-context trait.
//!
//! The value vector is transposed into a 2D array:
//! - rows are value-vector indices (wires).
//! - columns are instances.
//!
//! An instruction applies its scalar operation across a whole row: every instance in one pass.
//! This is the memory order the batch prover wants downstream.
//!
//! ```text
//!                  instance 0   instance 1   ...   instance n-1
//!   value index 0 [   w        |   w        | ... |   w        ]   <- one row
//!   value index 1 [   w        |   w        | ... |   w        ]
//!         ...
//! ```

use binius_core::Word;
use binius_utils::strided_array::StridedArray2DViewMut;

use super::exec::{EvalContext, Executor};
use crate::compiler::{
	circuit::PopulateError,
	hints::HintRegistry,
	pathspec::{PathSpec, PathSpecTree},
};

/// The cap on how many assertion failures are retained across the whole batch.
///
/// This mirrors the single-instance interpreter's cap. Failures past it are counted but not stored.
const MAX_ASSERTION_FAILURES: usize = 100;

/// A single assertion failure, tagged with the instance whose values violated it.
struct InstanceAssertionFailure {
	instance: usize,
	path_spec: PathSpec,
	message: String,
}

/// The failure of batch witness population, attributed to a single instance.
///
/// Serial batched evaluation reports the lowest-indexed failing instance. Parallel batched
/// evaluation runs over independent stripes, and may report the first failing stripe observed.
#[derive(Debug)]
pub struct BatchPopulateError {
	/// The index of the reported failing instance.
	pub instance: usize,
	/// The assertion failures recorded for that instance.
	pub source: PopulateError,
}

/// Execution context holding the transposed value array during batch evaluation.
struct BatchExecutionContext<'a, 'v> {
	/// Rows are value-vector indices; columns are instances.
	values: &'a mut StridedArray2DViewMut<'v, Word>,
	/// The global instance index represented by local column 0.
	instance_offset: usize,
	/// Assertion failures recorded during evaluation, capped by [`MAX_ASSERTION_FAILURES`].
	failures: Vec<InstanceAssertionFailure>,
	/// The total number of assertion violations recorded, across all instances.
	total_count: usize,
	/// The lowest-indexed instance that has failed an assertion, tracked even past the cap.
	min_failing_instance: Option<usize>,
}

impl<'a, 'v> BatchExecutionContext<'a, 'v> {
	const fn new(values: &'a mut StridedArray2DViewMut<'v, Word>, instance_offset: usize) -> Self {
		Self {
			values,
			instance_offset,
			failures: Vec::new(),
			total_count: 0,
			min_failing_instance: None,
		}
	}

	/// Turn recorded failures into an error attributed to the lowest-failing instance.
	fn check_assertions(
		self,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		let Some(instance) = self.min_failing_instance else {
			return Ok(());
		};

		// Collect and symbolicate just the reported instance's messages.
		let mut messages = Vec::new();
		let mut total_count = 0;
		for failure in self.failures.into_iter().filter(|f| f.instance == instance) {
			total_count += 1;
			let message = if let Some(tree) = path_spec_tree {
				let mut path = String::new();
				tree.stringify(failure.path_spec, &mut path);
				if path.is_empty() {
					failure.message
				} else {
					format!("{}: {}", path, failure.message)
				}
			} else {
				failure.message
			};
			messages.push(message);
		}

		Err(BatchPopulateError {
			instance,
			source: PopulateError {
				messages,
				total_count,
			},
		})
	}
}

impl EvalContext for BatchExecutionContext<'_, '_> {
	fn n_instances(&self) -> usize {
		self.values.width()
	}

	#[inline]
	fn load(&self, reg: u32, instance: usize) -> Word {
		self.values[(reg as usize, instance)]
	}

	#[inline]
	fn store(&mut self, reg: u32, instance: usize, value: Word) {
		self.values[(reg as usize, instance)] = value;
	}

	/// Record an assertion failure for one local instance.
	///
	/// The failure may be dropped from the stored list once the cap is reached.
	/// It always updates the count and the lowest-failing-instance tracker.
	/// The stripe offset remaps the local index to a global instance index.
	#[cold]
	fn note_assertion_failure(&mut self, instance: usize, path_spec: PathSpec, message: String) {
		let instance = self.instance_offset + instance;
		self.total_count += 1;
		self.min_failing_instance = Some(
			self.min_failing_instance
				.map_or(instance, |m| m.min(instance)),
		);
		if self.failures.len() < MAX_ASSERTION_FAILURES {
			self.failures.push(InstanceAssertionFailure {
				instance,
				path_spec,
				message,
			});
		}
	}
}

/// Bytecode interpreter that evaluates one circuit over many instances at once.
pub struct BatchInterpreter<'a> {
	executor: Executor<'a>,
}

impl<'a> BatchInterpreter<'a> {
	pub const fn new(bytecode: &'a [u8], hints: &'a HintRegistry) -> Self {
		Self {
			executor: Executor::new(bytecode, hints),
		}
	}

	/// Evaluate the bytecode over the transposed value array, filling every instance's wires.
	///
	/// The constant and input rows must already be populated for every instance. Returns an error
	/// naming the lowest-indexed instance whose assignment fails an assertion.
	pub fn run(
		&mut self,
		values: &mut StridedArray2DViewMut<'_, Word>,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		self.run_with_instance_offset(values, 0, path_spec_tree)
	}

	/// Evaluate the bytecode over a view whose local column 0 corresponds to `instance_offset`.
	pub(crate) fn run_with_instance_offset(
		&mut self,
		values: &mut StridedArray2DViewMut<'_, Word>,
		instance_offset: usize,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		let mut ctx = BatchExecutionContext::new(values, instance_offset);
		self.executor.run(&mut ctx);
		ctx.check_assertions(path_spec_tree)
	}
}

#[cfg(test)]
mod tests {
	use binius_core::Word;
	use binius_utils::strided_array::StridedArray2DViewMut;

	use crate::compiler::CircuitBuilder;

	// The batched interpreter must reproduce, for every instance, exactly what the single-instance
	// interpreter produces for the same inputs. This is the core equivalence guarantee.
	#[test]
	fn batched_matches_scalar_per_instance() {
		// A circuit that exercises a spread of opcodes plus a constant, with only witness inputs
		// and force-committed outputs (no inout wires — the M4 setting).
		let builder = CircuitBuilder::new();
		let a = builder.add_witness();
		let b = builder.add_witness();
		let k = builder.add_constant_64(0x0123_4567_89ab_cdef);
		let c = builder.band(a, b);
		let d = builder.bxor(a, k);
		let (sum, _cout) = builder.iadd(a, b);
		let e = builder.rotr(b, 7);
		let f = builder.bor(c, e);
		builder.force_commit(c);
		builder.force_commit(d);
		builder.force_commit(sum);
		builder.force_commit(f);
		let circuit = builder.build();

		let layout = circuit.constraint_system().value_vec_layout.clone();
		assert_eq!(layout.n_inout, 0, "fixture should have no inout wires");
		let combined = layout.combined_len();
		let full_len = combined + layout.n_scratch;
		let n = 8usize;

		// Distinct inputs per instance.
		let inputs: Vec<(u64, u64)> = (0..n)
			.map(|i| {
				let i = i as u64;
				(i.wrapping_mul(0x9e37_79b9_7f4a_7c15), i ^ 0x0000_0000_dead_beef)
			})
			.collect();

		// Single-instance reference: populate each instance on its own.
		let scalar: Vec<Vec<Word>> = inputs
			.iter()
			.map(|&(x, y)| {
				let mut filler = circuit.new_witness_filler();
				filler[a] = Word(x);
				filler[b] = Word(y);
				circuit.populate_wire_witness(&mut filler).unwrap();
				filler.value_vec().combined_witness().to_vec()
			})
			.collect();

		// Batched: fill the input rows for every instance, then evaluate all at once.
		let a_row = circuit.witness_index(a).0 as usize;
		let b_row = circuit.witness_index(b).0 as usize;
		let mut data = vec![Word::ZERO; full_len * n];
		let mut view = StridedArray2DViewMut::without_stride(&mut data, full_len, n).unwrap();
		for (instance, &(x, y)) in inputs.iter().enumerate() {
			view[(a_row, instance)] = Word(x);
			view[(b_row, instance)] = Word(y);
		}
		circuit.populate_wire_witness_batched(&mut view).unwrap();

		// Every instance's committed prefix must equal the single-instance witness.
		for instance in 0..n {
			for row in 0..combined {
				assert_eq!(
					view[(row, instance)],
					scalar[instance][row],
					"mismatch at row {row}, instance {instance}"
				);
			}
		}
	}

	// A batched run must flag the lowest-indexed instance whose inputs violate an assertion.
	#[test]
	fn batched_reports_lowest_failing_instance() {
		// Assert a == b; instances where a != b fail.
		let builder = CircuitBuilder::new();
		let a = builder.add_witness();
		let b = builder.add_witness();
		builder.assert_eq("a_eq_b", a, b);
		let circuit = builder.build();

		let layout = circuit.constraint_system().value_vec_layout.clone();
		let full_len = layout.combined_len() + layout.n_scratch;
		let n = 4usize;

		// Instances 2 and 3 violate a == b; instance 2 is the lowest.
		let inputs = [(1u64, 1u64), (7, 7), (4, 5), (9, 8)];
		let a_row = circuit.witness_index(a).0 as usize;
		let b_row = circuit.witness_index(b).0 as usize;
		let mut data = vec![Word::ZERO; full_len * n];
		let mut view = StridedArray2DViewMut::without_stride(&mut data, full_len, n).unwrap();
		for (instance, &(x, y)) in inputs.iter().enumerate() {
			view[(a_row, instance)] = Word(x);
			view[(b_row, instance)] = Word(y);
		}

		let err = circuit
			.populate_wire_witness_batched(&mut view)
			.expect_err("instances 2 and 3 violate a == b");
		assert_eq!(err.instance, 2);
		assert_eq!(err.source.total_count, 1);
		assert_eq!(
			err.source.messages,
			vec![".a_eq_b: Word(0x0000000000000004) != Word(0x0000000000000005)".to_string()]
		);
	}
}
