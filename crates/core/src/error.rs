// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Hosts error definitions for the core crate.

use crate::constraint_system::{Composition, ConstraintKind, ValueSegment};

/// Constraint system related error.
#[allow(missing_docs)] // errors are self-documenting
#[derive(Debug, thiserror::Error)]
pub enum ConstraintSystemError {
	#[error(
		"{constraint_kind} #{constraint_index} uses non canonical shift in its {operand_name} operand"
	)]
	NonCanonicalShift {
		constraint_kind: ConstraintKind,
		constraint_index: usize,
		operand_name: &'static str,
	},
	#[error(
		"{constraint_kind} #{constraint_index} puts a lone shift in the outer slot of its {operand_name} operand; the canonical form places it inner"
	)]
	NonCanonicalShiftSequence {
		constraint_kind: ConstraintKind,
		constraint_index: usize,
		operand_name: &'static str,
	},
	#[error(
		"{constraint_kind} #{constraint_index} uses a shift pair in its {operand_name} operand that composes to {composition:?} rather than staying a pair"
	)]
	CollapsibleShiftSequence {
		constraint_kind: ConstraintKind,
		constraint_index: usize,
		operand_name: &'static str,
		composition: Composition,
	},
	#[error(
		"{constraint_kind} #{constraint_index} refers to a scratch value in its {operand_name} operand"
	)]
	ScratchValueIndex {
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
		"{constraint_kind} #{constraint_index} refers to out-of-range value index in {operand_name} operand ({segment:?} index {value_index} >= segment length {segment_len})"
	)]
	OutOfRangeValueIndex {
		constraint_kind: ConstraintKind,
		constraint_index: usize,
		operand_name: &'static str,
		segment: ValueSegment,
		value_index: u32,
		segment_len: usize,
	},
	#[error("chip call #{call_index} has a malformed operand #{operand_index}: {source}")]
	ChipCallOperand {
		call_index: usize,
		operand_index: usize,
		#[source]
		source: OperandFault,
	},
	#[error("{} calls chip {chip_id}, but the system has {n_chips} chips", chip_name(*chip_index))]
	OutOfRangeChipId {
		chip_index: Option<usize>,
		chip_id: usize,
		n_chips: usize,
	},
	#[error(
		"{}'s call #{call_index} passes {arity} operands to chip {chip_id}, which has {n_inout} inout values",
		chip_name(*chip_index)
	)]
	WrongCallArity {
		chip_index: Option<usize>,
		call_index: usize,
		chip_id: usize,
		arity: usize,
		n_inout: usize,
	},
	#[error("the chip call graph has a cycle")]
	CyclicChipCalls,
}

/// Names the chip of an M4 constraint system that a diagnostic is about.
///
/// The main chip is not one of the numbered chips, so it has no index.
fn chip_name(chip_index: Option<usize>) -> String {
	match chip_index {
		Some(chip_index) => format!("chip #{chip_index}"),
		None => "the main chip".to_string(),
	}
}

/// The way one term of an operand is malformed, said without naming where the operand sits.
///
/// A diagnostic pairs this with the position of the operand it checked, which differs between a
/// constraint and a chip call.
#[allow(missing_docs)] // errors are self-documenting
#[derive(Debug, thiserror::Error)]
pub enum OperandFault {
	#[error("the shift is not canonical")]
	NonCanonicalShift,
	#[error("the shift amount n={shift_amount}>={max_amount}")]
	ShiftAmountTooLarge {
		shift_amount: usize,
		max_amount: usize,
	},
	#[error("a lone shift sits in the outer slot; the canonical form places it inner")]
	NonCanonicalShiftSequence,
	#[error("a shift pair composes to {composition:?} rather than staying a pair")]
	CollapsibleShiftSequence { composition: Composition },
	#[error("it refers to a scratch value")]
	ScratchValueIndex,
	#[error("it refers to {segment:?} index {value_index} >= segment length {segment_len}")]
	OutOfRangeValueIndex {
		segment: ValueSegment,
		value_index: u32,
		segment_len: usize,
	},
}

/// The arithmetic by which a single constraint fails on a value vector.
///
/// Every variant carries the operand words as the value vector evaluates them.
/// The failing relation reads straight off the message, with no need to recompute it.
#[derive(Debug, thiserror::Error)]
pub enum ConstraintViolation {
	/// An operand required to vanish holds a nonzero word.
	#[error("{val:016x} != 0")]
	Zero {
		/// The word the operand evaluates to.
		val: u64,
	},
	/// A conjunction of two operands disagrees with the operand claiming it.
	#[error("({a:016x} & {b:016x}) ^ {c:016x} = {residue:016x} != 0")]
	And {
		/// The first operand of the conjunction.
		a: u64,
		/// The second operand of the conjunction.
		b: u64,
		/// The claimed conjunction.
		c: u64,
		/// The bits on which the claim and the conjunction differ.
		residue: u64,
	},
	/// An integer product disagrees with the word pair claiming it.
	#[error("{a:016x} * {b:016x} = {expected_hi:016x}{expected_lo:016x}, got {hi:016x}{lo:016x}")]
	Imul {
		/// The first factor.
		a: u64,
		/// The second factor.
		b: u64,
		/// The claimed low 64 bits of the product.
		lo: u64,
		/// The claimed high 64 bits of the product.
		hi: u64,
		/// The low 64 bits the product actually has.
		expected_lo: u64,
		/// The high 64 bits the product actually has.
		expected_hi: u64,
	},
	/// A binary-field product disagrees with the element claiming it.
	#[error("{a:032x} * {b:032x} = {expected:032x}, got {c:032x}")]
	Bmul {
		/// The first factor, with bit `i` holding the coefficient of `X^i`.
		a: u128,
		/// The second factor, with bit `i` holding the coefficient of `X^i`.
		b: u128,
		/// The claimed product.
		c: u128,
		/// The product the two factors actually have.
		expected: u128,
	},
}

impl ConstraintViolation {
	/// Returns the kind of constraint that failed.
	///
	/// The kind follows from which relation was violated.
	/// Storing it as a separate field would let the two disagree.
	pub const fn kind(&self) -> ConstraintKind {
		match self {
			Self::Zero { .. } => ConstraintKind::Zero,
			Self::And { .. } => ConstraintKind::And,
			Self::Imul { .. } => ConstraintKind::Imul,
			Self::Bmul { .. } => ConstraintKind::Bmul,
		}
	}
}

/// Reason a value vector fails to satisfy a constraint system.
#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	/// A word declared as a constant opens to something else in the value vector.
	///
	/// Constraints read constants through the value vector.
	/// A vector that opens one to the wrong word therefore satisfies a different system.
	#[error(
		"value {value_index} is {actual:016x}, but the system declares the constant {expected:016x}"
	)]
	ConstantMismatch {
		/// Position of the disagreeing word in the value vector.
		value_index: u32,
		/// The word the system declares at that position.
		expected: u64,
		/// The word the value vector opens there.
		actual: u64,
	},
	/// A constraint does not hold on the value vector.
	#[error("{} #{constraint_index} is unsatisfied: {source}", source.kind())]
	Unsatisfied {
		/// Position of the constraint among those of its own kind, in storage order.
		constraint_index: usize,
		/// The relation that failed, carrying the words that failed it.
		source: ConstraintViolation,
	},
}

/// Names the chip instance an M4 diagnostic blames a call on.
///
/// The main chip runs once, so only a numbered chip's instance is worth naming.
fn caller_name(chip_index: Option<usize>, instance: usize) -> String {
	match chip_index {
		Some(chip_index) => format!("chip #{chip_index} instance #{instance}"),
		None => "the main chip".to_string(),
	}
}

/// Reason a witness fails to satisfy an M4 constraint system.
///
/// A witness is the main chip's value vector and one list of instance value vectors per chip.
/// It must satisfy every chip's local constraints on every instance, and serve every chip call
/// with the instance at the call's position.
#[allow(missing_docs)] // errors are self-documenting
#[derive(Debug, thiserror::Error)]
pub enum VerificationM4Error {
	#[error("the witness covers {n_witness_chips} chips, but the system has {n_chips}")]
	WrongChipCount {
		n_witness_chips: usize,
		n_chips: usize,
	},
	#[error("the main chip is not satisfied: {0}")]
	Main(#[from] VerificationError),
	#[error("chip #{chip_id} instance #{instance} is not satisfied: {source}")]
	ChipInstance {
		chip_id: usize,
		instance: usize,
		#[source]
		source: VerificationError,
	},
	#[error("chip #{chip_id} has {n_instances} instances, fewer than its {n_active} active ones")]
	MissingInstances {
		chip_id: usize,
		n_instances: usize,
		n_active: usize,
	},
	#[error(
		"{n_invocations} invocations reach chip #{chip_id}, which has {n_active} active instances"
	)]
	WrongInvocationCount {
		chip_id: usize,
		n_invocations: usize,
		n_active: usize,
	},
	#[error(
		"call #{call_index} of {} reaches chip #{chip_id} as invocation #{row}, passing {passed:016x} \
		 as inout value {word}, but the instance holds {served:016x}",
		caller_name(*caller_chip, *caller_instance)
	)]
	CallMismatch {
		chip_id: usize,
		row: usize,
		caller_chip: Option<usize>,
		caller_instance: usize,
		call_index: usize,
		word: usize,
		passed: u64,
		served: u64,
	},
}
