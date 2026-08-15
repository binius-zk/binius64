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
//!
//! [`WordIPVerifierChannel`] extends it for protocols that also carry 64-bit words, such as a
//! constraint system's public inputs or a code proximity test's query indices.

use std::{iter::repeat_with, ops::Shr};

use binius_core::word::Word;
use binius_field::{BinaryField, BinaryField1b, ExtensionField, Field, field::FieldOps};
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, CanSampleBits, Challenger},
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
pub trait IPVerifierChannel<F: Field> {
	/// The element type returned by receive and sample methods.
	type Elem: FieldOps<Scalar = F>;

	/// Receives a single field element from the prover.
	fn recv_one(&mut self) -> Result<Self::Elem, Error>;

	/// Receives `n` field elements from the prover.
	fn recv_many(&mut self, n: usize) -> Result<Vec<Self::Elem>, Error> {
		repeat_with(|| self.recv_one()).take(n).collect()
	}

	/// Receives a fixed-size array of field elements from the prover.
	fn recv_array<const N: usize>(&mut self) -> Result<[Self::Elem; N], Error> {
		array_util::try_from_fn(|_| self.recv_one())
	}

	/// Receives a value the verifier could compute for itself, taken as advice.
	///
	/// The prover states it and the verifier checks it — by recomputing it, or by any argument
	/// that establishes the same thing — which is worth doing when the check is cheaper than the
	/// computation, or when several such claims are better checked at once.
	///
	/// The caller MUST check what it receives here. Nothing else does.
	///
	/// A claim is a function of public-channel-derived values alone, so it carries nothing about
	/// the witness. Channels that mask the prover's messages must therefore leave this one in the
	/// clear, and channels that carry elements as wires allocate it as a public wire rather than a
	/// masked private one. The default is the plain read, for a channel that draws no such
	/// distinction.
	fn recv_public_claim(&mut self) -> Result<Self::Elem, Error> {
		self.recv_one()
	}

	/// Samples a random challenge.
	///
	/// In a Fiat-Shamir transcript, this derives the challenge deterministically from
	/// the current transcript state.
	fn sample(&mut self) -> Self::Elem;

	/// Samples `n` random challenges.
	fn sample_many(&mut self, n: usize) -> Vec<Self::Elem> {
		repeat_with(|| self.sample()).take(n).collect()
	}

	/// Samples a fixed-size array of random challenges.
	fn sample_array<const N: usize>(&mut self) -> [Self::Elem; N] {
		std::array::from_fn(|_| self.sample())
	}

	/// Observes a single field element, feeding it into the Fiat-Shamir state.
	///
	/// Returns the element converted to `Self::Elem`.
	fn observe_one(&mut self, val: F) -> Self::Elem;

	/// Observes multiple field elements, feeding them into the Fiat-Shamir state.
	///
	/// Returns the elements converted to `Vec<Self::Elem>`.
	fn observe_many(&mut self, vals: &[F]) -> Vec<Self::Elem> {
		vals.iter().map(|&val| self.observe_one(val)).collect()
	}

	/// Asserts that a value is zero.
	///
	/// Returns [`Error::InvalidAssert`] if the value is not zero.
	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), Error>;
}

/// A verifier channel whose protocol carries 64-bit words alongside field elements.
///
/// Some values a verifier handles are words rather than field elements: the public inputs of a
/// Binius64 constraint system, and the query indices a code proximity test samples. A channel that
/// symbolically executes a verifier to build a circuit carries those as wires, so protocol code
/// cannot name a concrete word type and goes through [`Self::Word`] instead.
///
/// [`Self::subset_sum`] and [`Self::select`] are how protocol code reaches the bits of a word. A
/// channel over concrete values reads them directly; a circuit-building one emits a sub-circuit.
pub trait WordIPVerifierChannel<F: Field>: IPVerifierChannel<F> {
	/// The word type this channel carries.
	///
	/// A channel over concrete values uses [`Word`]. A channel that builds a circuit uses a wire
	/// type that folds the operations below over a builder.
	///
	/// [`From<Word>`](From) lifts a word the protocol description fixes, such as a constraint
	/// system constant, and [`Shr`] is the index arithmetic a code proximity test performs between
	/// fold rounds. Both are plain operations on the type rather than channel methods, so protocol
	/// code writes `word.into()` and `word >> n` whatever the channel is. The shift amount is a
	/// `u32` to match [`Word`]'s own [`Shl`](std::ops::Shl) and [`Shr`] impls.
	type Word: Clone + From<Word> + Shr<u32, Output = Self::Word>;

	/// Feeds words into the Fiat-Shamir state, each as eight little-endian bytes, and returns them
	/// as this channel's word type.
	///
	/// The words go in concrete, because the statement is fixed data the verifier is handed rather
	/// than something the protocol derives. They come back as [`Self::Word`], which is where a
	/// channel that carries words as wires introduces them: it allocates the wires here, and the
	/// protocol sees the statement symbolically from this point on. A channel over concrete values
	/// hands the same words straight back.
	fn observe_words(&mut self, words: &[Word]) -> Vec<Self::Word>;

	/// Returns the sum of the `elems` selected by the low bits of `word`, low bit first.
	///
	/// This is the inner product of `elems` with the bit decomposition of `word`. Bits of `word`
	/// at or above `elems.len()` do not contribute.
	///
	/// ## Preconditions
	///
	/// * `elems.len()` must be at most 64.
	fn subset_sum(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem;

	/// Returns the element of `elems` at the index in the low bits of `word`.
	///
	/// Bits of `word` at or above `log2(elems.len())` are ignored.
	///
	/// ## Preconditions
	///
	/// * `elems` must be non-empty and its length must be a power of two.
	fn select(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem;

	/// Samples a uniform word of the given bit width.
	///
	/// The result is masked to `bits` bits. Protocols rely on that bound, so an implementation
	/// must enforce it rather than assume the sampled value already fits.
	fn sample_bits(&mut self, bits: usize) -> Self::Word;

	/// Packs words into field elements, as many words to an element as one holds.
	///
	/// Word `i` occupies bits `[Word::BITS * i, Word::BITS * (i + 1))` of element
	/// `i / words_per_elem`, so the packed form reads the same way the committed trace does: the
	/// low bit-index coordinates address the bit within a word, the next the word within its
	/// element. A word count that does not fill the last element leaves its high words zero.
	///
	/// This is a channel method rather than a free function because the words may be wires: a
	/// channel over concrete values computes the elements, and a circuit-building one emits the
	/// gates that assemble them.
	fn pack_words(&mut self, words: &[Self::Word]) -> Vec<Self::Elem>;
}

/// The number of elements [`WordIPVerifierChannel::pack_words`] returns for `n_words` words.
///
/// A channel that packs the words into wires rather than computing them needs the count on its
/// own, so the layout stays in one place.
pub const fn n_packed_elems<F: BinaryField>(n_words: usize) -> usize {
	n_words.div_ceil(F::N_BITS / Word::BITS)
}

/// [`WordIPVerifierChannel::pack_words`] over concrete words, for channels carrying [`Word`].
///
/// A set bit contributes the basis element at its position in the packed layout, which is what
/// makes this agree with reading the element's bits back out.
pub fn pack_words_concrete<F, E>(words: &[Word]) -> Vec<E>
where
	F: BinaryField,
	E: FieldOps<Scalar = F> + From<F>,
{
	let words_per_elem = F::N_BITS / Word::BITS;
	words
		.chunks(words_per_elem)
		.map(|chunk| {
			let packed = chunk
				.iter()
				.enumerate()
				.flat_map(|(i, word)| {
					(0..Word::BITS)
						.filter(|&bit| word.extract_bit(bit))
						.map(move |bit| {
							<F as ExtensionField<BinaryField1b>>::basis(i * Word::BITS + bit)
						})
				})
				.sum::<F>();
			E::from(packed)
		})
		.collect()
}

/// [`WordIPVerifierChannel::subset_sum`] over a concrete word, for channels carrying [`Word`].
///
/// ## Preconditions
///
/// * `elems.len()` must be at most 64.
pub fn subset_sum_word<E: FieldOps>(elems: &[E], word: Word) -> E {
	assert!(elems.len() <= Word::BITS); // precondition

	elems
		.iter()
		.enumerate()
		.filter(|&(bit, _)| word.extract_bit(bit))
		.map(|(_, elem)| elem.clone())
		.sum()
}

/// [`WordIPVerifierChannel::select`] over a concrete word, for channels carrying [`Word`].
///
/// ## Preconditions
///
/// * `elems` must be non-empty and its length must be a power of two.
pub fn select_word<E: FieldOps>(elems: &[E], word: Word) -> E {
	assert!(!elems.is_empty() && elems.len().is_power_of_two()); // precondition

	elems[word.as_u64() as usize & (elems.len() - 1)].clone()
}

impl<F, Challenger_> IPVerifierChannel<F> for VerifierTranscript<Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	type Elem = F;

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

	fn observe_one(&mut self, val: F) -> F {
		self.observe().write_scalar(val);
		val
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
		self.observe().write_scalar_slice(vals);
		vals.to_vec()
	}

	fn assert_zero(&mut self, val: F) -> Result<(), Error> {
		if val == F::ZERO {
			Ok(())
		} else {
			Err(Error::InvalidAssert)
		}
	}
}

impl<F, Challenger_> WordIPVerifierChannel<F> for VerifierTranscript<Challenger_>
where
	F: BinaryField,
	Challenger_: Challenger,
{
	type Word = Word;

	fn observe_words(&mut self, words: &[Word]) -> Vec<Word> {
		self.observe().write_slice(words);
		words.to_vec()
	}

	fn subset_sum(&mut self, elems: &[F], word: &Word) -> F {
		subset_sum_word(elems, *word)
	}

	fn select(&mut self, elems: &[F], word: &Word) -> F {
		select_word(elems, *word)
	}

	fn sample_bits(&mut self, bits: usize) -> Word {
		Word::from_u64(CanSampleBits::<u32>::sample_bits(self, bits) as u64)
	}

	fn pack_words(&mut self, words: &[Word]) -> Vec<F> {
		pack_words_concrete::<F, F>(words)
	}
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("proof is empty")]
	ProofEmpty,
	#[error("invalid assertion: value is not zero")]
	InvalidAssert,
}
