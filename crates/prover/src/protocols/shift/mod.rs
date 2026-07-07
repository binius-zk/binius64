// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_verifier::protocols::shift::{BITAND_ARITY, INTMUL_ARITY, SHIFT_VARIANT_COUNT};

mod key_collection;
// `monster`, `phase_1`, and `phase_2` are internal implementation, exposed (via `#[doc(hidden)]`
// `pub mod`) only so the `shift_reduction` benchmark can time individual phase functions (see
// `benches/shift_reduction.rs`). Not a stable API.
#[doc(hidden)]
pub mod monster;
#[doc(hidden)]
pub mod phase_1;
#[doc(hidden)]
pub mod phase_2;
mod prove;

pub use key_collection::{KeyCollection, KeySegment, Operation, build_key_collection};
pub use prove::{OperatorData, PreparedOperatorData, prove};
