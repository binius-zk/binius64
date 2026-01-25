// Copyright 2026 The Binius Developers

//! Channel abstraction for public-coin interactive protocol verifiers.
//!
//! In a public-coin interactive protocol, the verifier's messages consist entirely of random
//! challenges, while the prover sends deterministic messages based on the protocol state.
//! This module provides the [`IPVerifierChannel`] trait that models the verifier's view of such
//! an interaction.
//!
//! The trait abstracts over:
//! - Receiving prover messages (field elements)
//! - Sampling random challenges (which, in the Fiat-Shamir transform, are derived deterministically
//!   from the transcript)
//!
//! This abstraction allows protocol implementations to be generic over the underlying
//! communication mechanism, whether it's an actual interactive channel or a non-interactive
//! transcript using the Fiat-Shamir heuristic.

use std::iter::repeat_with;

use binius_field::Field;
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

/// Channel for receiving prover messages and sampling challenges in a public-coin interactive
/// protocol.
///
/// In a public-coin protocol, the verifier only sends random challenges (no secret information),
/// so the verifier's role is to:
/// 1. Receive field elements from the prover via `recv_*` methods
/// 2. Sample random challenges via `sample`
///
/// When used with a Fiat-Shamir transcript, the challenges are derived deterministically from
/// the transcript state, making the protocol non-interactive.
pub trait IPVerifierChannel<F> {
	/// Receives a single field element from the prover.
	fn recv_one(&mut self) -> Result<F, Error>;

	/// Receives `n` field elements from the prover.
	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, Error> {
		repeat_with(|| self.recv_one()).take(n).collect()
	}

	/// Receives a fixed-size array of field elements from the prover.
	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], Error>;

	/// Samples a random challenge.
	///
	/// In a Fiat-Shamir transcript, this derives the challenge deterministically from
	/// the current transcript state.
	fn sample(&mut self) -> F;

	/// Samples `n` random challenges.
	fn sample_many(&mut self, n: usize) -> Vec<F> {
		repeat_with(|| self.sample()).take(n).collect()
	}

	/// Samples a fixed-size array of random challenges.
	fn sample_array<const N: usize>(&mut self) -> [F; N] {
		std::array::from_fn(|_| self.sample())
	}
}

impl<F, Challenger_> IPVerifierChannel<F> for VerifierTranscript<Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	fn recv_one(&mut self) -> Result<F, Error> {
		self.message().read_scalar().map_err(|_| Error::ProofEmpty)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, Error> {
		self.message()
			.read_scalar_slice(n)
			.map_err(|_| Error::ProofEmpty)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], Error> {
		self.message().read().map_err(|_| Error::ProofEmpty)
	}

	fn sample(&mut self) -> F {
		CanSample::sample(self)
	}
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("proof is empty")]
	ProofEmpty,
}
