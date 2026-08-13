// Copyright 2026 The Binius Developers

//! Filling a recursive circuit's witness by replaying the verifier.

use std::{borrow::BorrowMut, marker::PhantomData};

use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, util::FieldFn};
use binius_frontend::WitnessFiller;
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop::{
	merkle_channel::{self, MerkleIPVerifierChannel, TranscriptMerkleCommitment},
	merkle_tree::{BinaryMerkleTreeScheme, MerkleTreeScheme},
};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, FixedSizeSerializeBytes};
use digest::Output;

use crate::{
	merkle::{DIGEST_WORDS, digest_words},
	shared::Input,
};

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
/// What it fills is proof data the circuit cannot derive, plus the statement being verified.
/// Every value the circuit ought to derive, it now does.
///
/// The transcript is read directly rather than through a `VerifierMerkleTranscriptChannel`.
/// That channel consumes the layer and branch digests internally, never handing them back.
/// A circuit needs them on wires, so they are read here instead.
pub struct WitnessFillerChannel<'a, 'c, T, Challenger_, H: HashSuite> {
	transcript: T,
	/// Reproduces the layer depth and the native checks the builder's scheme also picks.
	scheme: BinaryMerkleTreeScheme<B128, H>,
	filler: &'a mut WitnessFiller<'c>,
	wires: std::vec::IntoIter<Input>,
	_challenger_marker: PhantomData<Challenger_>,
}

impl<'a, 'c, T, Challenger_, H> WitnessFillerChannel<'a, 'c, T, Challenger_, H>
where
	H: HashSuite,
{
	/// Replays over a transcript, filling `wires` in the order the build recorded them.
	pub fn new(transcript: T, filler: &'a mut WitnessFiller<'c>, wires: Vec<Input>) -> Self {
		Self {
			transcript,
			scheme: BinaryMerkleTreeScheme::new(),
			filler,
			wires: wires.into_iter(),
			_challenger_marker: PhantomData,
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

	/// Writes one digest, in the packed byte order the hashing gadgets read.
	fn fill_digest(&mut self, kind: &'static str, digest: &Output<H::LeafHash>) {
		let bytes = digest.as_slice();
		assert_eq!(bytes.len(), DIGEST_WORDS * Word::BYTES);
		let bytes = bytes.try_into().expect("the length is checked above");
		for word in digest_words(bytes) {
			self.fill_word(kind, Word(word));
		}
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
		let value = self.transcript.borrow_mut().recv_one()?;
		self.fill_elem("recv_one", value);
		Ok(value)
	}

	fn sample(&mut self) -> B128 {
		// The circuit derives its own challenges now, so the build records no wire to fill here.
		// The draw still has to happen, since it advances the native Fiat-Shamir state.
		IPVerifierChannel::<B128>::sample(self.transcript.borrow_mut())
	}

	fn observe_one(&mut self, val: B128) -> B128 {
		let value = self.transcript.borrow_mut().observe_one(val);
		self.fill_elem("observe_one", value);
		value
	}

	fn assert_zero(&mut self, val: B128) -> Result<(), binius_ip::channel::Error> {
		// The real check still runs, so a bad proof is caught here rather than surfacing as an
		// unsatisfiable circuit.
		self.transcript.borrow_mut().assert_zero(val)
	}

	fn compute_public_value(&mut self, inputs: &[B128], f: impl FieldFn<B128>) -> B128 {
		// The builder evaluates this symbolically, so it records no wires and none are filled.
		self.transcript.borrow_mut().compute_public_value(inputs, f)
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
		WordIPVerifierChannel::<B128>::observe_words(self.transcript.borrow_mut(), words)
	}

	fn subset_sum(&mut self, elems: &[B128], word: &Word) -> B128 {
		self.transcript.borrow_mut().subset_sum(elems, word)
	}

	fn select(&mut self, elems: &[B128], word: &Word) -> B128 {
		self.transcript.borrow_mut().select(elems, word)
	}

	fn sample_bits(&mut self, bits: usize) -> Word {
		// Derived in-circuit now, so the build records no wire to fill here.
		// The draw still has to happen, since it advances the native Fiat-Shamir state.
		WordIPVerifierChannel::<B128>::sample_bits(self.transcript.borrow_mut(), bits)
	}

	// The build pairs up wires it already has, allocating none, so there is nothing to fill.
	fn pack_words(&mut self, words: &[Word]) -> Vec<B128> {
		self.transcript.borrow_mut().pack_words(words)
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
		let root: Output<H::LeafHash> = self.transcript.borrow_mut().message().read()?;
		self.fill_digest("merkle_root", &root);
		Ok(TranscriptMerkleCommitment {
			commitment: binius_iop::merkle_tree::Commitment { root, depth },
			leaf_size,
		})
	}

	/// Reads the same advice the native channel reads, and fills the wires the gadget checks.
	///
	/// The climb is not repeated natively: the circuit hashes each leaf and matches the layer.
	/// A forged opening therefore surfaces as an unsatisfied constraint, not an error at this call.
	fn recv_openings(
		&mut self,
		commitment: &Self::Commitment,
		indices: &[Word],
	) -> Result<Vec<B128>, merkle_channel::Error> {
		let tree_depth = commitment.commitment.depth;
		let indices = indices
			.iter()
			.map(|index| index.as_u64() as usize)
			.collect::<Vec<_>>();
		assert!(indices.iter().all(|&index| index < 1 << tree_depth)); // precondition

		let layer_depth = self.scheme.optimal_verify_layer(indices.len(), tree_depth);

		// The reader borrows the transcript for as long as it lives, so the whole batch is read
		// before any of it is written into wires.
		let (layer, openings) = {
			let mut advice = self.transcript.borrow_mut().decommitment();
			let layer = advice.read_vec::<Output<H::LeafHash>>(1 << layer_depth)?;
			let mut openings = Vec::with_capacity(indices.len());
			for _ in &indices {
				let leaf = advice.read_scalar_slice::<B128>(commitment.leaf_size)?;
				let branch = advice.read_vec::<Output<H::LeafHash>>(tree_depth - layer_depth)?;
				openings.push((leaf, branch));
			}
			(layer, openings)
		};

		// The layer folds to the root over concrete digests, so that half stays a native check.
		self.scheme
			.verify_layer(&commitment.commitment.root, layer_depth, &layer)?;

		// Filled in the order the build allocated: the shared layer, then a leaf and a branch per
		// query.
		for digest in &layer {
			self.fill_digest("merkle_layer", digest);
		}
		let mut values = Vec::with_capacity(indices.len() * commitment.leaf_size);
		for (leaf, branch) in openings {
			for &value in &leaf {
				self.fill_elem("opening", value);
			}
			for digest in &branch {
				self.fill_digest("merkle_branch", digest);
			}
			values.extend_from_slice(&leaf);
		}
		Ok(values)
	}

	fn recv_committed_vector(
		&mut self,
		commitment: &Self::Commitment,
	) -> Result<Vec<B128>, merkle_channel::Error> {
		let len = commitment.leaf_size << commitment.commitment.depth;
		let data = self
			.transcript
			.borrow_mut()
			.decommitment()
			.read_scalar_slice::<B128>(len)?;
		// The rebuild runs over concrete values too, so this stays a native check.
		self.scheme
			.verify_vector(&commitment.commitment.root, &data, commitment.leaf_size)?;

		for &value in &data {
			self.fill_elem("committed_vector", value);
		}
		Ok(data)
	}
}
