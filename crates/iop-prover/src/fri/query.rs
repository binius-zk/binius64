// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, PackedField};
use binius_iop::{
	fri::{FRIParams, vcs_optimal_layers_depths_iter},
	merkle_tree::MerkleTreeScheme,
};
use binius_math::{FieldBuffer, FieldSlice};
use binius_transcript::TranscriptWriter;
use bytes::BufMut;
use itertools::izip;
use tracing::instrument;

use crate::{fri::Error, merkle_tree::MerkleTreeProver};

/// A prover for the FRI query phase.
///
/// Uses separate Merkle tree provers for the initial codeword (which may be hiding/salted)
/// and for FRI round commitments (which are non-hiding).
#[derive(Debug)]
pub struct FRIQueryProver<
	'a,
	F,
	P,
	MerkleProver,
	VCS,
	RoundMerkleProver = MerkleProver,
	RoundVCS = VCS,
> where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
	RoundMerkleProver: MerkleTreeProver<F, Scheme = RoundVCS>,
	RoundVCS: MerkleTreeScheme<F>,
{
	pub(super) params: &'a FRIParams<F>,
	pub(super) codeword: FieldBuffer<P>,
	pub(super) codeword_committed: &'a MerkleProver::Committed,
	pub(super) round_committed: Vec<(FieldBuffer<F>, RoundMerkleProver::Committed)>,
	pub(super) merkle_prover: &'a MerkleProver,
	pub(super) round_merkle_prover: &'a RoundMerkleProver,
}

impl<F, P, MerkleProver, VCS, RoundMerkleProver, RoundVCS>
	FRIQueryProver<'_, F, P, MerkleProver, VCS, RoundMerkleProver, RoundVCS>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
	RoundMerkleProver: MerkleTreeProver<F, Scheme = RoundVCS>,
	RoundVCS: MerkleTreeScheme<F>,
{
	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		self.params.n_oracles()
	}

	/// Writes the proof data for `verify_vector` on the terminal codeword commitment.
	///
	/// For hiding trees, this writes the salt for each leaf of the terminal Merkle tree.
	pub fn prove_terminate_vector<B: BufMut>(
		&self,
		proof: &mut TranscriptWriter<B>,
	) -> Result<(), Error> {
		let (_, committed) = self
			.round_committed
			.last()
			.expect("round_committed is non-empty");
		self.round_merkle_prover.prove_vector(committed, proof)?;
		Ok(())
	}

	/// Proves a FRI challenge query.
	///
	/// ## Arguments
	///
	/// * `index` - an index into the original codeword domain
	#[instrument(skip_all, name = "fri::FRIQueryProver::prove_query", level = "debug")]
	pub fn prove_query<B>(
		&self,
		mut index: usize,
		advice: &mut TranscriptWriter<B>,
	) -> Result<(), Error>
	where
		B: BufMut,
	{
		let mut layer_depths_iter =
			vcs_optimal_layers_depths_iter(self.params, self.merkle_prover.scheme());
		let first_layer_depth = layer_depths_iter
			.next()
			.expect("not empty by post-condition");

		prove_coset_opening(
			self.merkle_prover,
			self.codeword.to_ref(),
			self.codeword_committed,
			index,
			self.params.log_batch_size(),
			first_layer_depth,
			advice,
		)?;

		for ((codeword, committed), &arity, optimal_layer_depth) in
			izip!(&self.round_committed, self.params.fold_arities(), layer_depths_iter)
		{
			index >>= arity;
			prove_coset_opening(
				self.round_merkle_prover,
				codeword.to_ref(),
				committed,
				index,
				arity,
				optimal_layer_depth,
				advice,
			)?;
		}

		Ok(())
	}

	pub fn vcs_optimal_layers(&self) -> Result<Vec<Vec<VCS::Digest>>, Error>
	where
		VCS::Digest: From<RoundVCS::Digest>,
	{
		let mut layers = Vec::new();

		// First layer: codeword commitment (uses the hiding prover)
		let mut layer_depths_iter =
			vcs_optimal_layers_depths_iter(self.params, self.merkle_prover.scheme());
		let first_depth = layer_depths_iter.next().expect("at least one commitment");
		let first_layer = self.merkle_prover.layer(self.codeword_committed, first_depth)?;
		layers.push(first_layer.to_vec());

		// Round layers (excluding terminal): use the round prover
		let round_committed_excluding_terminal =
			&self.round_committed[..self.round_committed.len() - 1];
		for ((_, committed), optimal_layer_depth) in round_committed_excluding_terminal
			.iter()
			.zip(layer_depths_iter)
		{
			let layer = self
				.round_merkle_prover
				.layer(committed, optimal_layer_depth)?;
			layers.push(layer.iter().cloned().map(Into::into).collect());
		}

		Ok(layers)
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
