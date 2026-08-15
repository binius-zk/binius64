// Copyright 2026 The Binius Developers

//! Channel abstraction for public-coin interactive protocol provers.
//!
//! In a public-coin interactive protocol, the prover sends deterministic messages while the
//! verifier's messages consist entirely of random challenges. This module provides the
//! [`IPProverChannel`] trait that models the prover's view of such an interaction.
//!
//! The trait abstracts over:
//! - Sending prover messages (field elements)
//! - Sampling random challenges (which must match what the verifier samples)
//!
//! This abstraction allows protocol implementations to be generic over the underlying
//! communication mechanism, whether it's an actual interactive channel or a non-interactive
//! transcript using the Fiat-Shamir heuristic.

use std::ops::Shr;

use binius_core::word::Word;
use binius_field::Field;
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, CanSampleBits, Challenger},
};

/// Channel for sending prover messages and sampling challenges in a public-coin interactive
/// protocol.
///
/// In a public-coin protocol, the prover's role is to:
/// 1. Send field elements to the verifier via `send_*` methods
/// 2. Sample the same random challenges as the verifier via `sample`
///
/// When used with a Fiat-Shamir transcript, the challenges are derived deterministically from
/// the transcript state, ensuring prover and verifier derive identical challenges.
pub trait IPProverChannel<F: Field> {
	/// Sends a single field element to the verifier.
	fn send_one(&mut self, elem: F);

	/// Sends multiple field elements to the verifier.
	fn send_many(&mut self, elems: &[F]) {
		for &elem in elems {
			self.send_one(elem);
		}
	}

	/// Sends a value the verifier could compute for itself, as advice.
	///
	/// The verifier's counterpart is
	/// [`recv_public_claim`](binius_ip::channel::IPVerifierChannel::recv_public_claim), which
	/// documents what a claim is. A claim depends on public-channel-derived values alone, so a
	/// channel that masks the prover's messages sends this one in the clear. The default is the
	/// plain send, for a channel that draws no such distinction.
	fn send_public_claim(&mut self, elem: F) {
		self.send_one(elem);
	}

	/// Observes a single field element, feeding it into the Fiat-Shamir state.
	fn observe_one(&mut self, val: F);

	/// Observes multiple field elements, feeding them into the Fiat-Shamir state.
	fn observe_many(&mut self, vals: &[F]) {
		for &val in vals {
			self.observe_one(val);
		}
	}

	/// Samples a random challenge.
	///
	/// In a Fiat-Shamir transcript, this derives the challenge deterministically from
	/// the current transcript state, matching what the verifier will sample.
	fn sample(&mut self) -> F;

	/// Samples `n` random challenges.
	fn sample_many(&mut self, n: usize) -> Vec<F> {
		std::iter::repeat_with(|| self.sample()).take(n).collect()
	}

	/// Samples a fixed-size array of random challenges.
	fn sample_array<const N: usize>(&mut self) -> [F; N] {
		std::array::from_fn(|_| self.sample())
	}
}

/// A prover channel whose protocol carries 64-bit words alongside field elements.
///
/// The prover-side counterpart of
/// [`WordIPVerifierChannel`](binius_ip::channel::WordIPVerifierChannel). It carries only the
/// operations both parties perform — lifting constants, observing, shifting and sampling — since
/// the arithmetic over a word's bits is the verifier's alone.
pub trait WordIPProverChannel<F: Field>: IPProverChannel<F> {
	/// The word type this channel carries.
	///
	/// Mirrors [`WordIPVerifierChannel::Word`](binius_ip::channel::WordIPVerifierChannel::Word),
	/// including the [`From<Word>`](From) and [`Shr`] bounds that keep lifting and index
	/// arithmetic plain operations rather than channel methods.
	type Word: Clone + From<Word> + Shr<u32, Output = Self::Word>;

	/// Feeds words into the Fiat-Shamir state, each as eight little-endian bytes.
	fn observe_words(&mut self, words: &[Self::Word]);

	/// Samples a uniform word of the given bit width, matching what the verifier samples.
	///
	/// The result is masked to `bits` bits.
	fn sample_bits(&mut self, bits: usize) -> Self::Word;
}

impl<F, Challenger_> IPProverChannel<F> for ProverTranscript<Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	fn send_one(&mut self, elem: F) {
		self.message().write_scalar(elem);
	}

	fn send_many(&mut self, elems: &[F]) {
		self.message().write_scalar_slice(elems);
	}

	fn observe_one(&mut self, val: F) {
		self.observe().write_scalar(val);
	}

	fn observe_many(&mut self, vals: &[F]) {
		self.observe().write_scalar_slice(vals);
	}

	fn sample(&mut self) -> F {
		CanSample::sample(self)
	}
}

impl<F, Challenger_> WordIPProverChannel<F> for ProverTranscript<Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	type Word = Word;

	fn observe_words(&mut self, words: &[Word]) {
		self.observe().write_slice(words);
	}

	fn sample_bits(&mut self, bits: usize) -> Word {
		Word::from_u64(CanSampleBits::<u32>::sample_bits(self, bits) as u64)
	}
}
