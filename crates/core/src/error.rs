// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Hosts error definitions for the core crate.

use std::fmt;

use crate::constraint_system::{Composition, ConstraintKind, ValueSegment};

/// Constraint system related error.
#[allow(missing_docs)] // errors are self-documenting
#[derive(Debug, thiserror::Error)]
pub enum ConstraintSystemError {
	#[error("{constraint_kind} #{constraint_index} operand {operand_name} is malformed: {source}")]
	ConstraintOperand {
		constraint_kind: ConstraintKind,
		constraint_index: usize,
		operand_name: &'static str,
		#[source]
		source: OperandFault,
	},
	#[error("chip call #{call_index} has a malformed operand #{operand_index}: {source}")]
	ChipCallOperand {
		call_index: usize,
		operand_index: usize,
		#[source]
		source: OperandFault,
	},
	#[error("{} calls chip {chip_id}, but the system has {n_chips} chips", ChipName(*chip_index))]
	OutOfRangeChipId {
		chip_index: Option<usize>,
		chip_id: usize,
		n_chips: usize,
	},
	#[error(
		"{}'s call #{call_index} passes {arity} operands to chip {chip_id}, which has {n_inout} inout values",
		ChipName(*chip_index)
	)]
	WrongCallArity {
		chip_index: Option<usize>,
		call_index: usize,
		chip_id: usize,
		arity: usize,
		n_inout: usize,
	},
	#[error("chip #{chip_index} calls chip {callee}, which is not a later chip")]
	CallOutOfOrder { chip_index: usize, callee: usize },
}

/// Names the chip of an M4 system that a diagnostic is about: `chip #3`, or `the main chip`.
///
/// The main chip is not one of the numbered chips, so it has no index. The frontend's circuit form
/// numbers its chips the same way, so its diagnostics name them through this too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChipName(pub Option<usize>);

impl fmt::Display for ChipName {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self.0 {
			Some(chip_index) => write!(f, "chip #{chip_index}"),
			None => f.write_str("the main chip"),
		}
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

/// Names the chip instance an M4 diagnostic blames a call on, by chip index and instance.
///
/// The main chip runs once, so only a numbered chip's instance is worth naming. Unlike
/// [`ChipName`], nothing outside this module names a caller, so this stays private to it.
struct CallerName(Option<(usize, usize)>);

impl fmt::Display for CallerName {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self.0 {
			Some((chip_index, instance)) => write!(f, "chip #{chip_index} instance #{instance}"),
			None => f.write_str("the main chip"),
		}
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
		CallerName(*caller)
	)]
	CallMismatch {
		chip_id: usize,
		row: usize,
		/// The calling chip instance, or `None` for the main chip.
		caller: Option<(usize, usize)>,
		call_index: usize,
		word: usize,
		passed: u64,
		served: u64,
	},
}
