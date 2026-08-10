// Copyright 2026 The Binius Developers

//! A Merkle channel that counts proof bytes instead of verifying them.

use std::{marker::PhantomData, mem::size_of};

use binius_field::{Field, util::FieldFn};
use binius_ip::channel::IPVerifierChannel;
use binius_utils::serialization::FixedSizeSerializeBytes;

use crate::{
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
	F: Field + FixedSizeSerializeBytes,
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

	fn compute_public_value(&mut self, inputs: &[F], f: impl FieldFn<F>) -> F {
		f.call_native(inputs)
	}
}

impl<F, MerkleScheme_> MerkleIPVerifierChannel<F> for SizeTrackingChannel<'_, F, MerkleScheme_>
where
	F: Field + FixedSizeSerializeBytes,
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
		indices: &[usize],
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

	fn sample_bits(&mut self, _bits: usize) -> usize {
		// Which leaves are opened does not change what the opening costs, only how many.
		0
	}
}
