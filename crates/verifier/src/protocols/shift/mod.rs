// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

/// The base-2 logarithm of [`SHIFT_VARIANT_COUNT`].
///
/// Phase 1 folds the shift variant with this many sumcheck variables, in the high index
/// positions of its two multilinears.
pub const LOG_SHIFT_VARIANT_COUNT: usize = 3;
pub const SHIFT_VARIANT_COUNT: usize = 1 << LOG_SHIFT_VARIANT_COUNT;
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
