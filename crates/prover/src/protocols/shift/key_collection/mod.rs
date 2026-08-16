// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The prover's dense re-encoding of a constraint system's shift keys.

mod builder;
mod collection;
mod dense_shift_encoding;
mod key;
mod key_segment;
mod operation;

pub use builder::build_key_collection;
pub use collection::KeyCollection;
pub use dense_shift_encoding::DenseShiftEncoding;
pub use key_segment::KeySegment;
pub use operation::Operation;
