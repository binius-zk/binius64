// Copyright 2026 The Binius Developers

//! Filling a recursive circuit's witness by replaying the verifier.

use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, util::FieldFn};
use binius_frontend::{Wire, WitnessFiller};
use binius_iop::merkle_channel::{self, MerkleIPVerifierChannel};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};

/// A channel that runs the verifier for real while writing what it sees into a witness.
///
/// The builder channel records the wires it could not derive, in the order it reached them. This
/// runs the same verifier over the real transcript and writes each value into the next of those
/// wires. Both runs visit the same operations in the same order, because no shape in the protocol
/// depends on a received value, so a single cursor keeps them aligned.
///
/// It exists because the skeleton leaves the Fiat-Shamir state and the Merkle openings
/// unconstrained. Every one of those wires is a value the circuit ought to derive, so as gadgets
/// land the recorded list shrinks, and with all of them in place only the proof itself remains.
pub struct WitnessFillerChannel<'a, 'c, Inner> {
	inner: Inner,
	filler: &'a mut WitnessFiller<'c>,
	wires: std::vec::IntoIter<Wire>,
}

impl<'a, 'c, Inner> WitnessFillerChannel<'a, 'c, Inner> {
	/// Replays over `inner`, filling `wires` in order.
	pub fn new(inner: Inner, filler: &'a mut WitnessFiller<'c>, wires: Vec<Wire>) -> Self {
		Self {
			inner,
			filler,
			wires: wires.into_iter(),
		}
	}

	/// Checks that the replay consumed exactly the wires the build recorded.
	///
	/// A mismatch means the two runs diverged, which would leave the witness silently wrong rather
	/// than merely incomplete.
	pub fn finish(self) {
		let remaining = self.wires.len();
		assert_eq!(remaining, 0, "the replay left {remaining} recorded wires unfilled");
	}

	fn fill_word(&mut self, value: Word) {
		let wire = self
			.wires
			.next()
			.expect("the replay asked for more wires than the build recorded");
		self.filler[wire] = value;
	}

	fn fill_elem(&mut self, value: B128) {
		let value = u128::from(value);
		self.fill_word(Word::from_u64(value as u64));
		self.fill_word(Word::from_u64((value >> 64) as u64));
	}
}

impl<Inner: IPVerifierChannel<B128, Elem = B128>> IPVerifierChannel<B128>
	for WitnessFillerChannel<'_, '_, Inner>
{
	type Elem = B128;

	fn recv_one(&mut self) -> Result<B128, binius_ip::channel::Error> {
		let value = self.inner.recv_one()?;
		self.fill_elem(value);
		Ok(value)
	}

	fn sample(&mut self) -> B128 {
		let value = self.inner.sample();
		self.fill_elem(value);
		value
	}

	fn observe_one(&mut self, val: B128) -> B128 {
		let value = self.inner.observe_one(val);
		self.fill_elem(value);
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

impl<Inner> WordIPVerifierChannel<B128> for WitnessFillerChannel<'_, '_, Inner>
where
	Inner: WordIPVerifierChannel<B128, Elem = B128, Word = Word>,
{
	type Word = Word;

	fn observe_words(&mut self, words: &[Word]) {
		self.inner.observe_words(words);
	}

	fn subset_sum(&mut self, elems: &[B128], word: &Word) -> B128 {
		self.inner.subset_sum(elems, word)
	}

	fn select(&mut self, elems: &[B128], word: &Word) -> B128 {
		self.inner.select(elems, word)
	}

	fn sample_bits(&mut self, bits: usize) -> Word {
		let value = self.inner.sample_bits(bits);
		self.fill_word(value);
		value
	}
}

impl<Inner> MerkleIPVerifierChannel<B128> for WitnessFillerChannel<'_, '_, Inner>
where
	Inner: MerkleIPVerifierChannel<B128, Elem = B128, Word = Word>,
{
	type Commitment = Inner::Commitment;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Self::Commitment, merkle_channel::Error> {
		let commitment = self.inner.recv_merkle_commitment(leaf_size, depth)?;
		// The builder reads a root as four wires. The concrete channel keeps it as a digest rather
		// than words, so the replay has nothing to copy and leaves them zero — one more thing the
		// Merkle gadget will settle.
		for _ in 0..super::channel::DIGEST_WORDS {
			self.fill_word(Word::ZERO);
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
			self.fill_elem(value);
		}
		Ok(values)
	}

	fn recv_committed_vector(
		&mut self,
		commitment: &Self::Commitment,
	) -> Result<Vec<B128>, merkle_channel::Error> {
		let values = self.inner.recv_committed_vector(commitment)?;
		for &value in &values {
			self.fill_elem(value);
		}
		Ok(values)
	}
}
