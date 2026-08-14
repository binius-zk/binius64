// Copyright 2025-2026 The Binius Developers
//! Batched execution context for circuit evaluation.
//!
//! This is the structure-of-arrays counterpart to the single-instance context.
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

use super::{
	assertion::{MAX_ASSERTION_FAILURES, render_path},
	exec::EvalContext,
};
use crate::{
	artifact::witness::{AssertionFailure, PopulateError},
	ir::path::{PathSpec, PathSpecTree},
};

/// A single assertion failure recorded for the current lowest-failing instance.
struct InstanceAssertionFailure {
	path_spec: PathSpec,
	message: String,
}

/// The failure of batch witness population, attributed to a single instance.
///
/// Serial batched evaluation reports the lowest-indexed failing instance. Parallel batched
/// evaluation runs over independent stripes, and may report the first failing stripe observed.
///
/// The inner [`PopulateError`] is the error's source, so the pair renders as one chain.
#[derive(Debug, thiserror::Error)]
#[error("instance {instance} is not satisfied: {source}")]
#[non_exhaustive]
pub struct BatchPopulateError {
	/// The index of the reported failing instance.
	pub instance: usize,
	/// The assertion failures recorded for that instance.
	#[source]
	pub source: PopulateError,
}

/// Execution context holding the transposed value array during batch evaluation.
pub struct BatchExecutionContext<'a, 'v> {
	/// Rows are value-vector indices; columns are instances.
	values: &'a mut StridedArray2DViewMut<'v, Word>,
	/// The global instance index represented by local column 0.
	instance_offset: usize,
	/// Failures recorded for [`Self::min_failing_instance`], capped by [`MAX_ASSERTION_FAILURES`].
	///
	/// Cleared whenever a strictly lower failing instance is found.
	/// So a higher-numbered instance's failures never survive to be reported.
	failures: Vec<InstanceAssertionFailure>,
	/// Every violation recorded for [`Self::min_failing_instance`], capped or not.
	///
	/// Reset alongside `failures` whenever a strictly lower failing instance is discovered.
	min_failure_count: usize,
	/// The lowest-indexed instance that has failed an assertion so far.
	min_failing_instance: Option<usize>,
}

impl<'a, 'v> BatchExecutionContext<'a, 'v> {
	pub const fn new(
		values: &'a mut StridedArray2DViewMut<'v, Word>,
		instance_offset: usize,
	) -> Self {
		Self {
			values,
			instance_offset,
			failures: Vec::new(),
			min_failure_count: 0,
			min_failing_instance: None,
		}
	}

	/// Turn recorded failures into an error attributed to the lowest-failing instance.
	pub fn check_assertions(
		self,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		let Some(instance) = self.min_failing_instance else {
			return Ok(());
		};

		// `failures` already holds only records for the reported instance.
		// Resolve each one's path against the tree.
		let failures: Vec<AssertionFailure> = self
			.failures
			.into_iter()
			.map(|f| AssertionFailure {
				path: render_path(path_spec_tree, f.path_spec),
				detail: f.message,
			})
			.collect();

		Err(BatchPopulateError {
			instance,
			source: PopulateError {
				total: self.min_failure_count,
				failures,
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
	/// A failure for a higher instance than the current lowest-failing one is dropped.
	/// It can never become the reported instance.
	/// A failure for a new, strictly lower instance clears every record kept so far.
	/// Those records belonged to an instance that turned out not to be the one reported.
	/// The stripe offset remaps the local index to a global instance index.
	#[cold]
	fn note_assertion_failure(&mut self, instance: usize, path_spec: PathSpec, message: String) {
		let instance = self.instance_offset + instance;
		match self.min_failing_instance {
			Some(min) if instance > min => return,
			Some(min) if instance < min => {
				self.min_failing_instance = Some(instance);
				self.failures.clear();
				self.min_failure_count = 0;
			}
			// Either the first failure ever seen, or another failure of the current minimum.
			_ => self.min_failing_instance = Some(instance),
		}

		self.min_failure_count += 1;
		if self.failures.len() < MAX_ASSERTION_FAILURES {
			self.failures
				.push(InstanceAssertionFailure { path_spec, message });
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_core::Word;
	use binius_utils::strided_array::StridedArray2DViewMut;

	use crate::{
		MAX_ASSERTION_FAILURES, Wire, artifact::witness::AssertionFailure, builder::CircuitBuilder,
	};

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

		let layout = circuit.value_vec_layout().clone();
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
		let a_row = circuit.witness_row(a);
		let b_row = circuit.witness_row(b);
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

		let layout = circuit.value_vec_layout().clone();
		let full_len = layout.combined_len() + layout.n_scratch;
		let n = 4usize;

		// Instances 2 and 3 violate a == b; instance 2 is the lowest.
		let inputs = [(1u64, 1u64), (7, 7), (4, 5), (9, 8)];
		let a_row = circuit.witness_row(a);
		let b_row = circuit.witness_row(b);
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
		assert_eq!(err.source.total, 1);
		assert_eq!(
			err.source.failures,
			vec![AssertionFailure {
				path: ".a_eq_b".to_string(),
				detail: "Word(0x0000000000000004) != Word(0x0000000000000005)".to_string(),
			}]
		);
		// thiserror supplies the instance prefix and chains the inner error as the source.
		let rendered = err.to_string();
		assert!(rendered.starts_with("instance 2 is not satisfied: "), "{rendered}");
		assert!(rendered.contains(".a_eq_b: Word(0x0000000000000004)"), "{rendered}");
		// An error message must not carry its own trailing newline.
		assert!(!rendered.ends_with('\n'), "{rendered}");
		assert!(std::error::Error::source(&err).is_some(), "the inner error must be the source");
	}

	// Invariant: the lowest-failing instance is reported, however full the cap got beforehand.
	//
	// Fixture state, two assertions in program order:
	//   assertion one fails for every instance from 50 up, 150 instances in total.
	//   assertion two fails only for instance 10.
	//
	// The cap on retained failures is 100.
	// Assertion one alone fills it, with records for instances 50..149, before assertion two runs.
	// Instance 10 is the true minimum, so it must still be the reported instance.
	// Its own record must survive, unevicted by the unrelated higher-numbered records.
	#[test]
	fn batch_min_failing_instance_survives_a_full_cap_of_higher_instances() {
		let builder = CircuitBuilder::new();
		let a = builder.add_witness();
		let b = builder.add_witness();
		let c = builder.add_witness();
		let d = builder.add_witness();
		// Assertion 1: fails for instance >= 50 when the inputs are driven that way below.
		builder.assert_eq("assertion_one", a, b);
		// Assertion 2, recorded after assertion 1 for every instance: fails only for instance 10.
		builder.assert_eq("assertion_two", c, d);
		let circuit = builder.build();

		let layout = circuit.value_vec_layout().clone();
		let full_len = layout.combined_len() + layout.n_scratch;
		// 150 instances fail assertion 1 (50..199).
		// That overflows the cap of 100 well before assertion 2 ever runs.
		let n = 200usize;

		let a_row = circuit.witness_row(a);
		let b_row = circuit.witness_row(b);
		let c_row = circuit.witness_row(c);
		let d_row = circuit.witness_row(d);
		let mut data = vec![Word::ZERO; full_len * n];
		let mut view = StridedArray2DViewMut::without_stride(&mut data, full_len, n).unwrap();
		for instance in 0..n {
			// a == b everywhere except instance >= 50, where assertion 1 fails.
			let a_val = Word(1);
			let b_val = Word(if instance >= 50 { 2 } else { 1 });
			// c == d everywhere except instance 10, where assertion 2 fails.
			let c_val = Word(3);
			let d_val = Word(if instance == 10 { 4 } else { 3 });
			view[(a_row, instance)] = a_val;
			view[(b_row, instance)] = b_val;
			view[(c_row, instance)] = c_val;
			view[(d_row, instance)] = d_val;
		}

		let err = circuit
			.populate_wire_witness_batched(&mut view)
			.expect_err("instance 10 and instances 50..199 all violate an assertion");

		// Instance 10 is the true minimum: 10 < 50.
		assert_eq!(err.instance, 10);
		// Exactly one assertion fails for instance 10: assertion 2.
		assert_eq!(err.source.total, 1);
		assert_eq!(
			err.source.failures,
			vec![AssertionFailure {
				path: ".assertion_two".to_string(),
				detail: "Word(0x0000000000000003) != Word(0x0000000000000004)".to_string(),
			}]
		);
	}

	// Invariant: `total` counts every violation the reported instance racked up.
	// So it can exceed `failures.len()` once the cap kicks in.
	#[test]
	fn batch_total_counts_every_violation_past_the_cap_for_the_reported_instance() {
		// Half again the cap, so the cap alone cannot retain them all.
		const N: usize = MAX_ASSERTION_FAILURES + 50;

		let builder = CircuitBuilder::new();
		let x: [Wire; N] = core::array::from_fn(|_| builder.add_witness());
		let y: [Wire; N] = core::array::from_fn(|_| builder.add_witness());
		builder.assert_eq_v("pairs", x, y);
		let circuit = builder.build();

		let layout = circuit.value_vec_layout().clone();
		let full_len = layout.combined_len() + layout.n_scratch;
		let n_instances = 4usize;

		let x_rows: Vec<usize> = x.iter().map(|&w| circuit.witness_row(w)).collect();
		let y_rows: Vec<usize> = y.iter().map(|&w| circuit.witness_row(w)).collect();
		let mut data = vec![Word::ZERO; full_len * n_instances];
		let mut view =
			StridedArray2DViewMut::without_stride(&mut data, full_len, n_instances).unwrap();
		for instance in 0..n_instances {
			for i in 0..N {
				// x == y everywhere except instance 0, where every one of the N pairs disagrees.
				let x_val = Word(i as u64);
				let y_val = if instance == 0 {
					Word(i as u64 + 1)
				} else {
					x_val
				};
				view[(x_rows[i], instance)] = x_val;
				view[(y_rows[i], instance)] = y_val;
			}
		}

		let err = circuit
			.populate_wire_witness_batched(&mut view)
			.expect_err("instance 0 violates every one of the N pairwise assertions");

		assert_eq!(err.instance, 0);
		// Every one of the N assertions failed for instance 0.
		assert_eq!(err.source.total, N);
		// Only the cap's worth of records survives, so `total` exceeds `failures.len()`.
		assert_eq!(err.source.failures.len(), MAX_ASSERTION_FAILURES);
		assert!(err.source.total > err.source.failures.len());
	}
}
