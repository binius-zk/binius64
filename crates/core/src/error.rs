// Copyright 2025 Irreducible Inc.
//! Hosts error definitions for the core crate.

use crate::{ConstraintSystem, constraint_system::ConstraintKind};

/// Constraint system related error.
#[allow(missing_docs)] // errors are self-documenting
#[derive(Debug, thiserror::Error)]
pub enum ConstraintSystemError {
	#[error("the public input segment must have power of two length")]
	PublicInputPowerOfTwo,
	#[error(
		"the public input segment must be at least {} words, got: {pub_input_size}",
		ConstraintSystem::MIN_WORDS_PER_SEGMENT
	)]
	PublicInputTooShort { pub_input_size: usize },
	#[error(
		"the hidden segment must be at least as long as the public segment (public: {public_len}, hidden: {hidden_len})"
	)]
	HiddenSegmentTooShort {
		public_len: usize,
		hidden_len: usize,
	},
	#[error("the inout values must be {expected} words, got: {actual}")]
	IncorrectInoutLength { expected: usize, actual: usize },
	#[error(
		"{constraint_kind} #{constraint_index} uses non canonical shift in its {operand_name} operand"
	)]
	NonCanonicalShift {
		constraint_kind: ConstraintKind,
		constraint_index: usize,
		operand_name: &'static str,
	},
	#[error(
		"{constraint_kind} #{constraint_index} refers to padding in its {operand_name} operand"
	)]
	PaddingValueIndex {
		constraint_kind: ConstraintKind,
		operand_name: &'static str,
		constraint_index: usize,
	},
	#[error(
		"{constraint_kind} #{constraint_index} uses shift amount n={shift_amount}>={max_amount} in {operand_name} operand"
	)]
	ShiftAmountTooLarge {
		constraint_kind: ConstraintKind,
		constraint_index: usize,
		operand_name: &'static str,
		shift_amount: usize,
		max_amount: usize,
	},
	#[error(
		"{constraint_kind} #{constraint_index} refers to out-of-range value index in {operand_name} operand (index {value_index} >= total length {total_len})"
	)]
	OutOfRangeValueIndex {
		constraint_kind: ConstraintKind,
		constraint_index: usize,
		operand_name: &'static str,
		value_index: u32,
		total_len: usize,
	},
}
