// Copyright 2026 The Binius Developers

//! Filling a recursive circuit's witness by replaying the verifier.

use std::borrow::BorrowMut;

use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, util::FieldFn};
use binius_frontend::WitnessFiller;
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop::merkle_channel::{
	self, MerkleIPVerifierChannel, TranscriptMerkleCommitment, VerifierMerkleTranscriptChannel,
};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, FixedSizeSerializeBytes};
use digest::Output;

use crate::{channel::DIGEST_WORDS, shared::Input};

/// A channel that runs the verifier for real while writing what it sees into a witness.
///
/// The builder channel records the wires it could not derive, in the order it reached them. This
/// runs the same verifier over the real transcript and writes each value into the next of those
/// wires. Both runs visit the same operations in the same order, because no shape in the protocol
/// depends on a received value, so a single cursor keeps them aligned.
///
/// That alignment is the whole witness story, so each recorded wire carries the operation that
/// allocated it and every fill checks the two agree. A count alone would not: two operations
/// diverging in opposite directions still add up, and every later value would land in the wrong
/// wire silently.
///
/// It exists because the skeleton leaves the Fiat-Shamir state and the Merkle openings
/// unconstrained. Every one of those wires is a value the circuit ought to derive, so as gadgets
/// land the recorded list shrinks, and with all of them in place only the proof itself remains.
pub struct WitnessFillerChannel<'a, 'c, T, Challenger_, H: HashSuite> {
	inner: VerifierMerkleTranscriptChannel<T, Challenger_, B128, H>,
	filler: &'a mut WitnessFiller<'c>,
	wires: std::vec::IntoIter<Input>,
}

impl<'a, 'c, T, Challenger_, H> WitnessFillerChannel<'a, 'c, T, Challenger_, H>
where
	H: HashSuite,
{
	/// Replays over a transcript, filling `wires` in the order the build recorded them.
	pub fn new(transcript: T, filler: &'a mut WitnessFiller<'c>, wires: Vec<Input>) -> Self {
		Self {
			inner: VerifierMerkleTranscriptChannel::new(transcript),
			filler,
			wires: wires.into_iter(),
		}
	}

	/// Checks that the replay consumed exactly the wires the build recorded.
	pub fn finish(self) {
		let remaining = self.wires.len();
		assert_eq!(remaining, 0, "the replay left {remaining} recorded wires unfilled");
	}

	/// Writes one word, checking it belongs to the operation the build recorded here.
	fn fill_word(&mut self, kind: &'static str, value: Word) {
		let input = self
			.wires
			.next()
			.expect("the replay asked for more wires than the build recorded");
		assert_eq!(
			input.kind, kind,
			"the replay diverged from the build: it filled a {kind} where the build recorded a {}",
			input.kind,
		);
		self.filler[input.wire] = value;
	}

	fn fill_elem(&mut self, kind: &'static str, value: B128) {
		let value = u128::from(value);
		self.fill_word(kind, Word::from_u64(value as u64));
		self.fill_word(kind, Word::from_u64((value >> 64) as u64));
	}
}

impl<T, Challenger_, H> IPVerifierChannel<B128> for WitnessFillerChannel<'_, '_, T, Challenger_, H>
where
	T: BorrowMut<VerifierTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
{
	type Elem = B128;

	fn recv_one(&mut self) -> Result<B128, binius_ip::channel::Error> {
		let value = self.inner.recv_one()?;
		self.fill_elem("recv_one", value);
		Ok(value)
	}

	fn sample(&mut self) -> B128 {
		let value = self.inner.sample();
		self.fill_elem("sample", value);
		value
	}

	fn observe_one(&mut self, val: B128) -> B128 {
		let value = self.inner.observe_one(val);
		self.fill_elem("observe_one", value);
		value
	}

	fn assert_zero(&mut self, val: B128) -> Result<(), binius_ip::channel::Error> {
		// The real check still runs, so a bad proof is caught here rather than surfacing as an
		// unsatisfiable circuit.
		self.inner.assert_zero(val)
	}

	fn compute_public_value(&mut self, inputs: &[B128], f: impl FieldFn<B128>) -> B128 {
		// The builder evaluates this symbolically, so it records no wires and none are filled.
		self.inner.compute_public_value(inputs, f)
	}
}

impl<T, Challenger_, H> WordIPVerifierChannel<B128>
	for WitnessFillerChannel<'_, '_, T, Challenger_, H>
where
	T: BorrowMut<VerifierTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
{
	type Word = Word;

	fn observe_words(&mut self, words: &[Word]) -> Vec<Word> {
		// The build allocated one input wire per statement word here, so the replay fills them in
		// the same order before forwarding to the real Fiat-Shamir state.
		for &word in words {
			self.fill_word("observe_words", word);
		}
		self.inner.observe_words(words)
	}

	fn subset_sum(&mut self, elems: &[B128], word: &Word) -> B128 {
		self.inner.subset_sum(elems, word)
	}

	fn select(&mut self, elems: &[B128], word: &Word) -> B128 {
		self.inner.select(elems, word)
	}

	fn sample_bits(&mut self, bits: usize) -> Word {
		let value = self.inner.sample_bits(bits);
		self.fill_word("sample_bits", value);
		value
	}

	// The build pairs up wires it already has, allocating none, so there is nothing to fill.
	fn pack_words(&mut self, words: &[Word]) -> Vec<B128> {
		self.inner.pack_words(words)
	}
}

impl<T, Challenger_, H> MerkleIPVerifierChannel<B128>
	for WitnessFillerChannel<'_, '_, T, Challenger_, H>
where
	T: BorrowMut<VerifierTranscript<Challenger_>>,
	Challenger_: Challenger,
	H: HashSuite,
	Output<H::LeafHash>: DeserializeBytes,
	B128: FixedSizeSerializeBytes,
{
	type Commitment = TranscriptMerkleCommitment<Output<H::LeafHash>>;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Self::Commitment, merkle_channel::Error> {
		let commitment = self.inner.recv_merkle_commitment(leaf_size, depth)?;

		// The root goes in as words so the wires hold the digest the prover sent, rather than a
		// placeholder that would look right until the Merkle gadget started reading it.
		let bytes = commitment.commitment.root.as_slice();
		assert_eq!(bytes.len(), DIGEST_WORDS * Word::BYTES);
		for chunk in bytes.chunks(Word::BYTES) {
			let word = u64::from_le_bytes(chunk.try_into().expect("chunks of eight bytes"));
			self.fill_word("merkle_root", Word::from_u64(word));
		}
		Ok(commitment)
	}

	fn recv_openings(
		&mut self,
		commitment: &Self::Commitment,
		indices: &[Word],
	) -> Result<Vec<B128>, merkle_channel::Error> {
		let values = self.inner.recv_openings(commitment, indices)?;
		for &value in &values {
			self.fill_elem("opening", value);
		}
		Ok(values)
	}

	fn recv_committed_vector(
		&mut self,
		commitment: &Self::Commitment,
	) -> Result<Vec<B128>, merkle_channel::Error> {
		let values = self.inner.recv_committed_vector(commitment)?;
		for &value in &values {
			self.fill_elem("committed_vector", value);
		}
		Ok(values)
	}
}
