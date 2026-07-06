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
// `wiring` holds the alternate `WiringInfo` phase-1 layout, benchmarked head-to-head against the
// `KeyCollection` path (BINIUS-228). Exposed only so the `shift_reduction` benchmark can build it.
#[doc(hidden)]
pub mod wiring;

pub use key_collection::{KeyCollection, KeySegment, build_key_collection};
pub use prove::{OperatorData, PreparedOperatorData, prove};
pub use wiring::{WiringCollection, WiringEntry, WiringInfo, WiringMatrix, build_wiring_info};
