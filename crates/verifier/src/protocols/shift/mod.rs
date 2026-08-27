// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::word::Word;

/// The base-2 logarithm of [`SHIFT_VARIANT_COUNT`].
///
/// Phase 1 folds the shift variant with this many sumcheck variables, in the high index
/// positions of its two multilinears.
pub const LOG_SHIFT_VARIANT_COUNT: usize = 3;
pub const SHIFT_VARIANT_COUNT: usize = 1 << LOG_SHIFT_VARIANT_COUNT;

/// The number of variables the shift reduction's sumcheck binds outside the word index.
///
/// A shifted value index names two shifts applied in sequence, so the claim sums over two shift
/// slots — a variant and an amount each — and three bit indices: the witness bit the inner shift
/// reads, the intermediate-word bit the two indicators meet at, and the output bit the oblong
/// weights attach to. This counts all of those; the word index is sized by the constraint system.
pub const SHIFT_LOG_VARS: usize = Word::LOG_BITS * 3 + 2 * LOG_SHIFT_COUNT;

/// The number of index bits one shift slot occupies: the variant above the amount.
///
/// A reduction axis over one slot is this many variables wide, which is what
/// [`Shift::index`](binius_core::constraint_system::Shift::index) addresses.
pub const LOG_SHIFT_COUNT: usize = LOG_SHIFT_VARIANT_COUNT + Word::LOG_BITS;

/// The number of `(variant, amount)` spellings one shift slot can take.
///
/// A shift is a variant paired with an amount below the word width.
/// One weight table per shift slot has this many entries.
///
/// A term carries two slots, so the sequences it could name number the square of this.
/// The weight factorizes across the slots, so two tables of this size replace one of that square.
pub const SHIFT_COUNT: usize = 1 << LOG_SHIFT_COUNT;

pub const ZERO_ARITY: usize = 1;
pub const BITAND_ARITY: usize = 3;
pub const INTMUL_ARITY: usize = 4;
pub const BINMUL_ARITY: usize = 6;

/// The number of operations the shift reduction batches: ZERO, AND, IMUL and BMUL.
pub const OPERATION_COUNT: usize = 4;

/// The base-2 logarithm of the operation axis the reduction batches over.
///
/// An operation's claims are weighted by the equality indicator of this many challenges, evaluated
/// at the operation's own index in the order above.
pub const LOG_OPERATION_COUNT: usize = 2;

/// The base-2 logarithm of the operand axis the reduction batches over.
///
/// An operand's claim is weighted by the equality indicator of this many challenges, evaluated at
/// the operand's position. The four operations share one such axis, padded to a cube: an operation
/// of lower arity reads a prefix of the same weights, and the slots above its arity name no claim
/// and contribute nothing.
pub const LOG_MAX_ARITY: usize = 3;

// The two axes above are cubes, so each has to cover what it indexes.
const _: () = assert!(OPERATION_COUNT <= 1 << LOG_OPERATION_COUNT);
const _: () = assert!(ZERO_ARITY <= 1 << LOG_MAX_ARITY);
const _: () = assert!(BITAND_ARITY <= 1 << LOG_MAX_ARITY);
const _: () = assert!(INTMUL_ARITY <= 1 << LOG_MAX_ARITY);
const _: () = assert!(BINMUL_ARITY <= 1 << LOG_MAX_ARITY);

mod monster;
mod shift_ind;

pub use monster::*;
mod error;
mod verify;

pub use error::Error;
pub use shift_ind::evaluate_shift_inds;
pub use verify::{
	DeferredWiringClaim, OperatorData, VerifyOutput, WiringEvalClaim, WiringEvalFn,
	WiringEvalShape, check_eval, evaluate_words_mle, verify,
};
