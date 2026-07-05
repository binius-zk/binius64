// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub const SHIFT_VARIANT_COUNT: usize = 8;
pub const BITAND_ARITY: usize = 3;
pub const INTMUL_ARITY: usize = 4;

mod monster;
mod shift_ind;

pub use monster::*;
mod error;
mod verify;

pub use error::Error;
pub use verify::{OperatorData, VerifyOutput, check_eval, evaluate_words_mle, verify};
