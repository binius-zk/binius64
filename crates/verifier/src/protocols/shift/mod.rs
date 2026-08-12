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
/// The claim sums over five axes: the shift variant, the shift amount, the bit position within a
/// word, the bit index the shift indicators read, and the word index. This counts the first four;
/// the word index is sized by the constraint system.
pub const SHIFT_LOG_VARS: usize = Word::LOG_BITS * 3 + LOG_SHIFT_VARIANT_COUNT;

pub const ZERO_ARITY: usize = 1;
pub const BITAND_ARITY: usize = 3;
pub const INTMUL_ARITY: usize = 4;
pub const BINMUL_ARITY: usize = 6;

mod monster;
mod shift_ind;

pub use monster::*;
mod error;
mod verify;

pub use error::Error;
pub use shift_ind::evaluate_shift_inds;
pub use verify::{OperatorData, VerifyOutput, check_eval, evaluate_words_mle, verify};
