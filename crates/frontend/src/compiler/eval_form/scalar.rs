// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Single-instance execution context for circuit evaluation.
//!
//! This is the one-instance case of the shared executor.
//! It evaluates the bytecode against a single value vector.
//! The batched counterpart evaluates many instances at once.

use binius_core::{ValueIndex, ValueVec, Word};

use super::{
	assertion::{MAX_ASSERTION_FAILURES, symbolicate},
	exec::EvalContext,
};
use crate::compiler::{
	circuit::PopulateError,
	pathspec::{PathSpec, PathSpecTree},
};

/// Assertion failure information
struct AssertionFailure {
	path_spec: PathSpec,
	message: String,
}

/// Execution context holds a reference to ValueVec during execution
pub struct ExecutionContext<'a> {
	value_vec: &'a mut ValueVec,
	/// Assertion failures recorded during the evaluation of the circuit.
	///
	/// This list is capped by [`MAX_ASSERTION_FAILURES`].
	assertion_failures: Vec<AssertionFailure>,
	/// The total number of assert violations recorded.
	assertion_count: usize,
}

impl<'a> ExecutionContext<'a> {
	pub const fn new(value_vec: &'a mut ValueVec) -> Self {
		Self {
			value_vec,
			assertion_failures: Vec::new(),
			assertion_count: 0,
		}
	}

	/// Check assertions and return error if any failed
	pub fn check_assertions(
		self,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), PopulateError> {
		if self.assertion_failures.is_empty() {
			return Ok(());
		}

		Err(PopulateError {
			messages: self
				.assertion_failures
				.into_iter()
				.map(|f| symbolicate(path_spec_tree, f.path_spec, f.message))
				.collect(),
			total_count: self.assertion_count,
		})
	}
}

impl EvalContext for ExecutionContext<'_> {
	// One value vector: a single instance.
	fn n_instances(&self) -> usize {
		1
	}

	fn load(&self, reg: u32, _instance: usize) -> Word {
		self.value_vec[ValueIndex(reg)]
	}

	fn store(&mut self, reg: u32, _instance: usize, value: Word) {
		self.value_vec[ValueIndex(reg)] = value;
	}

	#[cold]
	fn note_assertion_failure(&mut self, _instance: usize, path_spec: PathSpec, message: String) {
		self.assertion_count += 1;
		if self.assertion_failures.len() < MAX_ASSERTION_FAILURES {
			self.assertion_failures
				.push(AssertionFailure { path_spec, message });
		}
	}
}
