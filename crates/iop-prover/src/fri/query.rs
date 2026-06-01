// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, PackedField};
use binius_iop::{fri::FRIParams, merkle_tree::MerkleTreeScheme};
use binius_math::{FieldBuffer, FieldSlice};
use binius_transcript::TranscriptWriter;
use binius_utils::SerializeBytes;
use bytes::BufMut;
use tracing::instrument;

use crate::{fri::Error, merkle_tree::MerkleTreeProver};

/// A prover for the FRI query phase.
#[derive(Debug)]
pub struct FRIQueryProver<'a, F, P, MerkleProver, VCS>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
{
	pub(super) params: &'a FRIParams<F>,
	pub(super) codeword: FieldBuffer<P>,
	pub(super) codeword_committed: &'a MerkleProver::Committed,
	pub(super) round_committed: Vec<(FieldBuffer<F>, MerkleProver::Committed)>,
	pub(super) merkle_prover: &'a MerkleProver,
}

impl<F, P, MerkleProver, VCS> FRIQueryProver<'_, F, P, MerkleProver, VCS>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
{
	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		self.params.n_oracles()
	}

	/// Proves the FRI challenge queries, batched per oracle.
	///
	/// For each committed oracle (the codeword first, then each fold-round oracle excluding the
	/// terminal codeword) this writes the oracle's optimal Merkle layer once, followed by the coset
	/// opening for each query index. This per-oracle batched layout matches the verifier, which
	/// reads each oracle's layer and then all of its query openings together.
	///
	/// ## Arguments
	///
	/// * `indices` - the sampled query indices into the original codeword domain
	#[instrument(skip_all, name = "fri::FRIQueryProver::prove_queries", level = "debug")]
	pub fn prove_queries<B>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptWriter<B>,
	) -> Result<(), Error>
	where
		B: BufMut,
		VCS::Digest: SerializeBytes,
	{
		let scheme = self.merkle_prover.scheme();
		let n_queries = indices.len();

		// The codeword oracle, opened as interleaved cosets. Its Merkle tree has one coset per
		// leaf, so its depth is the number of index bits.
		let mut tree_depth = self.params.index_bits();
		let layer_depth = scheme.optimal_verify_layer(n_queries, tree_depth);
		let layer = self
			.merkle_prover
			.layer(self.codeword_committed, layer_depth)?;
		advice.write_slice(layer);
		for &index in indices {
			prove_coset_opening(
				self.merkle_prover,
				self.codeword.to_ref(),
				self.codeword_committed,
				index,
				self.params.log_batch_size(),
				layer_depth,
				advice,
			)?;
		}

		// The fold-round oracles, excluding the terminal codeword which is sent in full.
		let round_committed_excluding_terminal =
			&self.round_committed[..self.round_committed.len() - 1];
		let mut shift = 0;
		for ((codeword, committed), &arity) in round_committed_excluding_terminal
			.iter()
			.zip(self.params.fold_arities())
		{
			shift += arity;
			tree_depth -= arity;
			let layer_depth = scheme.optimal_verify_layer(n_queries, tree_depth);
			let layer = self.merkle_prover.layer(committed, layer_depth)?;
			advice.write_slice(layer);
			for &index in indices {
				prove_coset_opening(
					self.merkle_prover,
					codeword.to_ref(),
					committed,
					index >> shift,
					arity,
					layer_depth,
					advice,
				)?;
			}
		}

		Ok(())
	}
}

fn prove_coset_opening<F, P, MTProver, B>(
	merkle_prover: &MTProver,
	codeword: FieldSlice<P>,
	committed: &MTProver::Committed,
	coset_index: usize,
	log_coset_size: usize,
	optimal_layer_depth: usize,
	advice: &mut TranscriptWriter<B>,
) -> Result<(), Error>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MTProver: MerkleTreeProver<F>,
	B: BufMut,
{
	assert!(coset_index < (1 << (codeword.log_len() - log_coset_size))); // precondition

	let values = codeword.chunk(log_coset_size, coset_index);
	advice.write_scalar_iter(values.iter_scalars());

	merkle_prover.prove_opening(committed, optimal_layer_depth, coset_index, advice)?;

	Ok(())
}
