// Copyright 2026 The Binius Developers

use binius_field::BinaryField;
use binius_math::{line::extrapolate_line, multilinear, ntt::DomainContext};
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
		verify_query_openings(
			&self.merkle_scheme,
			&self.commitment,
			self.challenges.len(),
			indices,
			advice,
		)?
		.map(|opening| {
			let (_index, values) = opening?;
			// Fold the coset using a multilinear tensor fold over the challenges.
			Ok(multilinear::evaluate::evaluate_inplace_scalars(values, &self.challenges))
		})
		.collect()
	}
}

/// A [ProxQueryVerifier] implementation for a FRI-style code proximity check.
///
/// Note that this is distinct from the `FRIQueryVerifier` in the `verify` module, which implements
/// the full FRI query phase. This one only verifies the openings of a single committed oracle and
/// folds each opened coset into a single value using FRI folding.
pub struct FRIQueryVerifier<F, MTScheme, DC>
where
	MTScheme: MerkleTreeScheme<F>,
	DC: DomainContext<Field = F>,
{
	challenges: Vec<F>,
	commitment: Commitment<MTScheme::Digest>,
	merkle_scheme: MTScheme,
	domain_context: DC,
}

impl<F, MTScheme, DC> FRIQueryVerifier<F, MTScheme, DC>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F>,
	DC: DomainContext<Field = F>,
{
	/// Folds an opened coset into a single value.
	///
	/// This implements the fold operation from Definition 4.6 of [DP24], reading twiddle factors
	/// directly from the held [`DomainContext`]. The twiddle layer is absolute within the full NTT
	/// domain; for the committed oracle the codeword length in log terms is the Merkle tree depth
	/// plus the number of folding challenges (one coset per leaf).
	///
	/// [DP24]: <https://eprint.iacr.org/2024/504>
	fn fold_coset(&self, chunk_index: usize, mut values: Vec<F>) -> F {
		let n_challenges = self.challenges.len();
		let mut log_len = self.commitment.depth + n_challenges;
		let mut log_size = n_challenges;
		for &challenge in &self.challenges {
			for index_offset in 0..1 << (log_size - 1) {
				// Perform the inverse additive NTT butterfly, then extrapolate the resulting line at
				// the folding challenge.
				let mut u = values[index_offset << 1];
				let mut v = values[(index_offset << 1) | 1];
				let twiddle = self
					.domain_context
					.twiddle(log_len - 1, (chunk_index << (log_size - 1)) | index_offset);
				v += u;
				u += v * twiddle;
				values[index_offset] = extrapolate_line(u, v, challenge);
			}

			log_len -= 1;
			log_size -= 1;
		}

		values[0]
	}
}

impl<F, MTScheme, DC> ProxQueryVerifier<F> for FRIQueryVerifier<F, MTScheme, DC>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	DC: DomainContext<Field = F>,
{
	fn verify_queries<B: Buf>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptReader<B>,
	) -> Result<Vec<F>, Error> {
		verify_query_openings(
			&self.merkle_scheme,
			&self.commitment,
			self.challenges.len(),
			indices,
			advice,
		)?
		.map(|opening| {
			let (index, values) = opening?;
			Ok(self.fold_coset(index, values))
		})
		.collect()
	}
}

/// Verifies the Merkle openings shared by the [ProxQueryVerifier] implementations.
///
/// First decommits and verifies the optimal internal layer of the Merkle tree, then returns a lazy
/// iterator that, for each queried index, reads the opened coset of `1 << coset_log_size` values
/// from the advice and verifies its opening against that layer. Each item is the queried index
/// paired with the verified coset values.
fn verify_query_openings<'a, F, MTScheme, B>(
	merkle_scheme: &'a MTScheme,
	commitment: &'a Commitment<MTScheme::Digest>,
	coset_log_size: usize,
	indices: &'a [usize],
	advice: &'a mut TranscriptReader<B>,
) -> Result<impl Iterator<Item = Result<(usize, Vec<F>), Error>> + 'a, Error>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	B: Buf,
{
	let tree_depth = commitment.depth;
	let layer_depth = merkle_scheme.optimal_verify_layer(indices.len(), tree_depth);
	let layer_digests = advice.read_vec(1 << layer_depth)?;
	merkle_scheme.verify_layer(&commitment.root, layer_depth, &layer_digests)?;

	let openings = indices.iter().map(move |&index| {
		// Receive the codeword symbols of the coset at the index and verify their consistency with
		// the commitment.
		let values = advice.read_scalar_slice::<F>(1 << coset_log_size)?;
		merkle_scheme.verify_opening(
			index,
			&values,
			layer_depth,
			tree_depth,
			&layer_digests,
			advice,
		)?;
		Ok((index, values))
	});
	Ok(openings)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Merkle tree error: {0}")]
	MerkleError(#[from] crate::merkle_tree::Error),
	#[error("transcript error: {0}")]
	TranscriptError(#[from] binius_transcript::Error),
}
