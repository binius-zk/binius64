// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Single-instance bytecode interpreter for circuit evaluation.
//!
//! This is the one-instance case of the shared executor.
//! It evaluates the bytecode against a single value vector.
//! The batched counterpart evaluates many instances at once.

use binius_core::{ValueIndex, ValueVec, Word};

use super::exec::{EvalContext, Executor};
use crate::compiler::{
	circuit::PopulateError,
	hints::HintRegistry,
	pathspec::{PathSpec, PathSpecTree},
};

const MAX_ASSERTION_FAILURES: usize = 100;

/// Assertion failure information
pub struct AssertionFailure {
	pub path_spec: PathSpec,
	pub message: String,
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
		if !self.assertion_failures.is_empty() {
			let messages = if let Some(tree) = path_spec_tree {
				// Symbolicate the path specs
				self.assertion_failures
					.into_iter()
					.map(|f| {
						let mut path = String::new();
						tree.stringify(f.path_spec, &mut path);
						if path.is_empty() {
							f.message
						} else {
							format!("{}: {}", path, f.message)
						}
					})
					.collect()
			} else {
				// No tree provided, just use messages as-is
				self.assertion_failures
					.into_iter()
					.map(|f| f.message)
					.collect()
			};

			Err(PopulateError {
				messages,
				total_count: self.assertion_count,
			})
		} else {
			Ok(())
		}
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

pub struct Interpreter<'a> {
	executor: Executor<'a>,
}

impl<'a> Interpreter<'a> {
	pub const fn new(bytecode: &'a [u8], hints: &'a HintRegistry) -> Self {
		Self {
			executor: Executor::new(bytecode, hints),
		}
	}

	/// Evaluate the bytecode against `value_vec`, returning any assertion failures as an error.
	pub fn run_with_value_vec(
		&mut self,
		value_vec: &mut ValueVec,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), PopulateError> {
		let mut ctx = ExecutionContext::new(value_vec);
		self.run(&mut ctx)?;
		ctx.check_assertions(path_spec_tree)
	}

	/// Evaluate the bytecode against a caller-owned context.
	///
	/// Assertion failures are recorded on the context.
	/// Call the context's assertion check to turn them into an error.
	pub fn run(&mut self, ctx: &mut ExecutionContext<'_>) -> Result<(), PopulateError> {
		self.executor.run(ctx);
		Ok(())
	}
}
