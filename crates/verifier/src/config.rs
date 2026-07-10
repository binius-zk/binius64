// Copyright 2025 Irreducible Inc.
//! Specifies standard trait implementations and parameters.

use binius_core::word::Word;
use binius_field::{AESTowerField8b, BinaryField, BinaryField1b, BinaryField128bGhash};
use binius_hash::StdDigest;
use binius_transcript::fiat_shamir::{Challenger, HasherChallenger};
use binius_utils::checked_arithmetics::checked_log_2;

// Exports the binary fields that this system uses
pub type B1 = BinaryField1b;
pub type B128 = BinaryField128bGhash;

/// The intention of this trait is to capture the moment when a StandardChallenger type is changed.
pub trait ChallengerWithName: Challenger {
	const NAME: &'static str;
}

impl ChallengerWithName for HasherChallenger<StdDigest> {
	const NAME: &'static str = "HasherChallenger<Sha256>";
}

/// The default [`binius_transcript::fiat_shamir::Challenger`] implementation.
pub type StdChallenger = HasherChallenger<StdDigest>;

/// log2 of the number of [`Word`]s packed into one field element.
pub const LOG_WORDS_PER_ELEM: usize = checked_log_2(B128::N_BITS) - Word::LOG_BITS;

pub const PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES: [AESTowerField8b; 3] = [
	AESTowerField8b::new(0x2),
	AESTowerField8b::new(0x4),
	AESTowerField8b::new(0x10),
];
