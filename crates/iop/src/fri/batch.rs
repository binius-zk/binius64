// Copyright 2026 The Binius Developers

use binius_field::BinaryField;
use binius_math::multilinear;
use binius_transcript::TranscriptReader;
use binius_utils::DeserializeBytes;
use bytes::Buf;

use crate::merkle_tree::{Commitment, MerkleTreeScheme};
/// A verification interface for the query phase of a code proximity test.
///
/// The interactive code proximity tests used in this project (eg. FRI) follow the structure of:
///
/// 1. the verifier and prover interactively and randomly fold the codeword
/// 2. the verifier randomly samples codeword symbol indices
/// 3. the verifier opens Merkle commitments at those indices and performs consistency checks on the
///    committed values
pub trait ProxQueryVerifier<F: BinaryField> {
	/// Verify decommitted prover advice for satisfying values at the committed indices.
	///
	/// This has a batch interface for verifying multiple queries because opening multiple Merkle
	/// tree locations at once amortizes the proof size.
	///
	/// ## Returns
	/// The values of the virtual oracle at the queried indices. The virtual oracle is defined by
	/// the committed oracle and the folding challenges.
	fn verify_queries<B: Buf>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptReader<B>,
	) -> Result<Vec<F>, Error>;
}

/// A [ProxQueryVerifier] implementation for a [Brakedown]-style interleaved code proximity check.
/// 
/// [Brakedown]: <https://dl.acm.org/doi/10.1007/978-3-031-38545-2_7>
pub struct BrakedownQueryVerifier<F, MTScheme>
where
	MTScheme: MerkleTreeScheme<F>,
{
	challenges: Vec<F>,
	commitment: Commitment<MTScheme::Digest>,
	merkle_scheme: MTScheme,
}

impl<F: BinaryField, MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>> ProxQueryVerifier<F>
	for BrakedownQueryVerifier<F, MTScheme>
{
	fn verify_queries<B: Buf>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptReader<B>,
	) -> Result<Vec<F>, Error> {
		let tree_depth = self.commitment.depth;
		let layer_depth = self
			.merkle_scheme
			.optimal_verify_layer(indices.len(), tree_depth);
		let layer_digests = advice.read_vec(1 << layer_depth)?;
		self.merkle_scheme
			.verify_layer(&self.commitment.root, layer_depth, &layer_digests)?;

		indices
			.iter()
			.map(|&index| {
				// Receive the interleaved codeword symbol at the index and verify their consistency
				// with the commitment.
				let values = advice.read_scalar_slice::<F>(1 << self.challenges.len())?;
				self.merkle_scheme.verify_opening(
					index,
					&values,
					layer_depth,
					tree_depth,
					&layer_digests,
					advice,
				)?;

				let folded_value =
					multilinear::evaluate::evaluate_inplace_scalars(values, &self.challenges);
				Ok(folded_value)
			})
			.collect()
	}
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Merkle tree error: {0}")]
	MerkleError(#[from] crate::merkle_tree::Error),
	#[error("transcript error: {0}")]
	TranscriptError(#[from] binius_transcript::Error),
}
