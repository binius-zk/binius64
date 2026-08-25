// Copyright 2026 The Binius Developers

//! A Merkle channel that counts proof bytes instead of verifying them.

use std::{marker::PhantomData, mem::size_of};

use binius_core::word::Word;
use binius_field::BinaryField;
use binius_ip::channel::{
	IPVerifierChannel, WordIPVerifierChannel, pack_words_concrete, select_word, subset_sum_word,
};
use binius_utils::serialization::FixedSizeSerializeBytes;

use crate::{
	channel::grinding::GrindingVerifierChannel,
	merkle_channel::{Error, MerkleIPVerifierChannel},
	merkle_tree::MerkleTreeScheme,
};

/// What a commitment fixes about the tree behind it.
///
/// A commitment carries no data here, only the two numbers that decide what opening it costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommittedShape {
	/// Field elements in each leaf.
	leaf_size: usize,
	/// Base-2 logarithm of the number of leaves.
	depth: usize,
}

/// A Merkle channel that counts the bytes a proof would occupy, without checking any of them.
///
/// Every receive returns zeros and adds what the real bytes would weigh.
/// Sampling returns zeros too, so nothing here needs a prover.
///
/// Counting at this layer measures a protocol by running it, rather than by a formula kept in
/// step with it by hand.
/// Anything speaking this interface can be measured, oracle reduction or not.
///
/// Zero survives every fold and equality check a verifier performs.
/// So the verifier takes the path it would on a real proof, reaching the same receives in order.
pub struct SizeTrackingChannel<'a, F, MerkleScheme_> {
	scheme: &'a MerkleScheme_,
	proof_size: usize,
	_field_marker: PhantomData<F>,
}

impl<'a, F, MerkleScheme_> SizeTrackingChannel<'a, F, MerkleScheme_> {
	/// Creates a channel that sizes openings for the given scheme.
	pub const fn new(scheme: &'a MerkleScheme_) -> Self {
		Self {
			scheme,
			proof_size: 0,
			_field_marker: PhantomData,
		}
	}

	/// The bytes counted so far.
	pub const fn proof_size(&self) -> usize {
		self.proof_size
	}
}

impl<F, MerkleScheme_> IPVerifierChannel<F> for SizeTrackingChannel<'_, F, MerkleScheme_>
where
	F: BinaryField + FixedSizeSerializeBytes,
	MerkleScheme_: MerkleTreeScheme<F>,
{
	// Nothing here reads a value, so a zero-sized stand-in would seem to fit.
	// It does not: openings arrive as field elements, and folding mixes them with sampled
	// challenges, so the two must be one type.
	type Elem = F;

	fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
		self.proof_size += F::BYTE_SIZE;
		Ok(F::ZERO)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
		self.proof_size += n * F::BYTE_SIZE;
		Ok(vec![F::ZERO; n])
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
		self.proof_size += N * F::BYTE_SIZE;
		Ok([F::ZERO; N])
	}

	fn sample(&mut self) -> F {
		F::ZERO
	}

	fn observe_one(&mut self, _val: F) -> F {
		F::ZERO
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
		vec![F::ZERO; vals.len()]
	}

	fn assert_zero(&mut self, _val: F) -> Result<(), binius_ip::channel::Error> {
		Ok(())
	}
}

impl<F, MerkleScheme_> WordIPVerifierChannel<F> for SizeTrackingChannel<'_, F, MerkleScheme_>
where
	F: BinaryField + FixedSizeSerializeBytes,
	MerkleScheme_: MerkleTreeScheme<F>,
{
	type Word = Word;

	// Observing feeds the Fiat-Shamir state rather than the proof tape, so it costs no bytes.
	fn observe_words(&mut self, words: &[Word]) -> Vec<Word> {
		words.to_vec()
	}

	fn subset_sum(&mut self, elems: &[F], word: &Word) -> F {
		subset_sum_word(elems, *word)
	}

	fn select(&mut self, elems: &[F], word: &Word) -> F {
		select_word(elems, *word)
	}

	// Which leaves are opened does not change what an opening costs, only how many.
	fn sample_bits(&mut self, _bits: usize) -> Word {
		Word::ZERO
	}

	fn pack_words(&mut self, words: &[Word]) -> Vec<F> {
		pack_words_concrete::<F, F>(words)
	}
}

impl<F, MerkleScheme_> GrindingVerifierChannel for SizeTrackingChannel<'_, F, MerkleScheme_> {
	fn verify_grind(&mut self, bits: usize) -> Result<(), binius_transcript::Error> {
		// Zero difficulty is not a grind, so nothing reaches the tape and nothing is charged.
		if bits == 0 {
			return Ok(());
		}
		// A grind puts one `u64` nonce on the tape, whatever its difficulty.
		self.proof_size += size_of::<u64>();
		Ok(())
	}
}

impl<F, MerkleScheme_> MerkleIPVerifierChannel<F> for SizeTrackingChannel<'_, F, MerkleScheme_>
where
	F: BinaryField + FixedSizeSerializeBytes,
	MerkleScheme_: MerkleTreeScheme<F>,
{
	type Commitment = CommittedShape;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Self::Commitment, Error> {
		// A commitment is its root, one digest wide.
		self.proof_size += size_of::<MerkleScheme_::Digest>();
		Ok(CommittedShape { leaf_size, depth })
	}

	fn recv_openings(
		&mut self,
		commitment: &Self::Commitment,
		indices: &[Word],
	) -> Result<Vec<F>, Error> {
		// A multi-opening sends one internal layer, then one branch per index up to that layer.
		// The scheme prices both, since it is the scheme that decides where the layer sits.
		let layer_depth = self
			.scheme
			.optimal_verify_layer(indices.len(), commitment.depth);
		self.proof_size +=
			self.scheme
				.proof_size(1 << commitment.depth, indices.len(), layer_depth);

		// Each opened leaf also sends its own values.
		let n_values = indices.len() * commitment.leaf_size;
		self.proof_size += n_values * F::BYTE_SIZE;
		Ok(vec![F::ZERO; n_values])
	}

	fn recv_committed_vector(&mut self, commitment: &Self::Commitment) -> Result<Vec<F>, Error> {
		// The whole vector goes on the wire, so the receiver rebuilds the tree and needs no branch.
		let len = commitment.leaf_size << commitment.depth;
		self.proof_size += len * F::BYTE_SIZE;
		Ok(vec![F::ZERO; len])
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b as B128;
	use binius_hash::StdHashSuite;

	use super::*;
	use crate::merkle_tree::BinaryMerkleTreeScheme;

	#[test]
	fn a_grind_is_charged_one_nonce_and_a_zero_bit_one_is_charged_nothing() {
		// Invariant: this channel prices a protocol by running it, so what it charges for a grind
		// has to be what the prover really writes. That is one `u64` nonce, whatever the
		// difficulty, and nothing at all when the difficulty is zero.
		//
		// Fixture state: a channel over the shipped Merkle scheme, with nothing else received.
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		let mut channel = SizeTrackingChannel::<B128, _>::new(&scheme);
		assert_eq!(channel.proof_size(), 0);

		channel
			.verify_grind(0)
			.expect("a zero-bit grind cannot fail");
		assert_eq!(channel.proof_size(), 0);

		for bits in [1, 8, 32] {
			channel.verify_grind(bits).expect("nothing here can fail");
		}
		assert_eq!(channel.proof_size(), 3 * size_of::<u64>());
	}
}
