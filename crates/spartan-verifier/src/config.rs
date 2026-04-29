// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField1b, BinaryField128bGhash};
pub use binius_hash::{StdCompression, StdDigest};

/// The default [`binius_transcript::fiat_shamir::Challenger`] implementation.
pub type StdChallenger = binius_transcript::fiat_shamir::HasherChallenger<StdDigest>;

pub type B1 = BinaryField1b;
pub type B128 = BinaryField128bGhash;
