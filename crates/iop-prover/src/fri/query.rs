// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, Field, PackedField};
use binius_iop::merkle_tree::MerkleTreeScheme;
use binius_math::{FieldBuffer, FieldSlice};
use binius_transcript::TranscriptWriter;
use binius_utils::SerializeBytes;
use bytes::BufMut;
use tracing::instrument;

use crate::{fri::Error, merkle_tree::MerkleTreeProver};

/// The prover counterpart of a `ProxTestOracle` (verifier side), producing the per-oracle query
/// openings.
pub trait ProxTestOracleProver<F> {
	/// Writes the per-oracle batched query openings: the oracle's optimal Merkle layer once,
	/// followed by each queried coset's values and Merkle opening proof.
	fn open_queries<B: BufMut>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptWriter<B>,
	) -> Result<(), Error>;
}

/// The [`ProxTestOracleProver`] for a [Brakedown]-style interleaved code proximity check.
///
/// [Brakedown]: <https://dl.acm.org/doi/10.1007/978-3-031-38545-2_7>
pub struct BrakedownOracleProver<'a, P, MerkleProver>
where
	P: PackedField,
	MerkleProver: MerkleTreeProver<P::Scalar>,
{
	codeword: FieldBuffer<P>,
	committed: &'a MerkleProver::Committed,
	merkle_prover: &'a MerkleProver,
	log_batch_size: usize,
	/// log2 the lift factor (oracle padding). The committed codeword is virtually duplicated
	/// `2^log_lift` times to reach the common first-round length; a query at global index `k`
	/// opens the committed codeword at `k >> log_lift`. Zero when no lifting is needed.
	log_lift: usize,
}

impl<'a, P, MerkleProver> BrakedownOracleProver<'a, P, MerkleProver>
where
	P: PackedField,
	MerkleProver: MerkleTreeProver<P::Scalar>,
{
	/// Constructs a new oracle prover wrapping a committed interleaved codeword.
	///
	/// `log_lift` is the oracle-padding lift factor (the committed codeword is virtually duplicated
	/// `2^log_lift` times to reach the common first-round length); pass `0` when no lifting is
	/// needed.
	pub fn new(
		codeword: FieldBuffer<P>,
		committed: &'a MerkleProver::Committed,
		merkle_prover: &'a MerkleProver,
		log_batch_size: usize,
		log_lift: usize,
	) -> Self {
		Self {
			codeword,
			committed,
			merkle_prover,
			log_batch_size,
			log_lift,
		}
	}
}

impl<F, P, MerkleProver, VCS> ProxTestOracleProver<F> for BrakedownOracleProver<'_, P, MerkleProver>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F, Digest: SerializeBytes>,
{
	fn open_queries<B: BufMut>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptWriter<B>,
	) -> Result<(), Error> {
		open_oracle_queries(
			self.merkle_prover,
			&self.codeword,
			self.committed,
			self.log_batch_size,
			self.log_lift,
			indices,
			advice,
		)
	}
}

/// A [`ProxTestOracleProver`] bundling several separately committed [`BrakedownOracleProver`]s.
///
/// The bundled oracles all wrap interleaved codewords of the same length that are batched into a
/// single folded codeword during the first FRI fold. Their query openings are written sequentially,
/// one oracle's full decommitment after another, so the verifier reads each committed oracle's
/// advice in turn.
pub struct BatchBrakedownOracleProver<'a, P, MerkleProver>
where
	P: PackedField,
	MerkleProver: MerkleTreeProver<P::Scalar>,
{
	oracles: Vec<BrakedownOracleProver<'a, P, MerkleProver>>,
}

impl<'a, P, MerkleProver> BatchBrakedownOracleProver<'a, P, MerkleProver>
where
	P: PackedField,
	MerkleProver: MerkleTreeProver<P::Scalar>,
{
	/// Constructs a batch oracle prover from the per-commitment oracle provers.
	pub fn new(oracles: Vec<BrakedownOracleProver<'a, P, MerkleProver>>) -> Self {
		Self { oracles }
	}
}

impl<F, P, MerkleProver, VCS> ProxTestOracleProver<F>
	for BatchBrakedownOracleProver<'_, P, MerkleProver>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F, Digest: SerializeBytes>,
{
	fn open_queries<B: BufMut>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptWriter<B>,
	) -> Result<(), Error> {
		for oracle in &self.oracles {
			oracle.open_queries(indices, advice)?;
		}
		Ok(())
	}
}

/// The [`ProxTestOracleProver`] for a FRI-style code proximity check.
pub struct FRIOracleProver<'a, F, MerkleProver>
where
	F: Field,
	MerkleProver: MerkleTreeProver<F>,
{
	codeword: FieldBuffer<F>,
	committed: MerkleProver::Committed,
	merkle_prover: &'a MerkleProver,
	coset_log_size: usize,
}

impl<'a, F, MerkleProver> FRIOracleProver<'a, F, MerkleProver>
where
	F: BinaryField,
	MerkleProver: MerkleTreeProver<F>,
{
	/// Constructs a new oracle prover wrapping a committed fold-round codeword.
	pub fn new(
		codeword: FieldBuffer<F>,
		committed: MerkleProver::Committed,
		merkle_prover: &'a MerkleProver,
		coset_log_size: usize,
	) -> Self {
		Self {
			codeword,
			committed,
			merkle_prover,
			coset_log_size,
		}
	}
}

impl<F, MerkleProver, VCS> ProxTestOracleProver<F> for FRIOracleProver<'_, F, MerkleProver>
where
	F: BinaryField,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F, Digest: SerializeBytes>,
{
	fn open_queries<B: BufMut>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptWriter<B>,
	) -> Result<(), Error> {
		open_oracle_queries(
			self.merkle_prover,
			&self.codeword,
			&self.committed,
			self.coset_log_size,
			0,
			indices,
			advice,
		)
	}
}

/// Writes the optimal Merkle layer once, then a coset opening for each queried index.
///
/// The coset is opened at the index directly, mirroring the verifier's `open_queries`. When the
/// oracle is lifted (`log_lift > 0`), each global query index is translated to the committed
/// codeword by dropping its low `log_lift` bits (the duplicated copies), mirroring the verifier.
fn open_oracle_queries<F, P, MerkleProver, B>(
	merkle_prover: &MerkleProver,
	codeword: &FieldBuffer<P>,
	committed: &MerkleProver::Committed,
	coset_log_size: usize,
	log_lift: usize,
	indices: &[usize],
	advice: &mut TranscriptWriter<B>,
) -> Result<(), Error>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F>,
	<MerkleProver::Scheme as MerkleTreeScheme<F>>::Digest: SerializeBytes,
	B: BufMut,
{
	let scheme = merkle_prover.scheme();
	// The Merkle tree has one coset per leaf, so its depth is the codeword length minus the coset.
	let tree_depth = codeword.log_len() - coset_log_size;
	let layer_depth = scheme.optimal_verify_layer(indices.len(), tree_depth);
	advice.write_slice(merkle_prover.layer(committed, layer_depth)?);
	for &index in indices {
		prove_coset_opening(
			merkle_prover,
			codeword.to_ref(),
			committed,
			index >> log_lift,
			coset_log_size,
			layer_depth,
			advice,
		)?;
	}

	Ok(())
}

/// A prover for the FRI query phase.
///
/// This is a composition of [`ProxTestOracleProver`]s mirroring the verifier's `FRIQueryVerifier`:
/// a [`BatchBrakedownOracleProver`] for the codeword's interleaved reduction, then one
/// [`FRIOracleProver`] per fold arity for the subsequent reductions.
pub struct FRIQueryProver<'a, F, P, MerkleProver, VCS>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
{
	codeword_oracle: BatchBrakedownOracleProver<'a, P, MerkleProver>,
	fri_oracles: Vec<FRIOracleProver<'a, F, MerkleProver>>,
}

impl<'a, F, P, MerkleProver, VCS> FRIQueryProver<'a, F, P, MerkleProver, VCS>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
{
	/// Constructs a query prover from the per-oracle provers built during the fold phase.
	///
	/// `codeword_oracle` is the [`BatchBrakedownOracleProver`] for the originally committed
	/// interleaved codeword(s), and `fri_oracles` holds one [`FRIOracleProver`] per fold arity. The
	/// terminal codeword is sent in full and therefore is not represented here.
	pub fn new(
		codeword_oracle: BatchBrakedownOracleProver<'a, P, MerkleProver>,
		fri_oracles: Vec<FRIOracleProver<'a, F, MerkleProver>>,
	) -> Self {
		Self {
			codeword_oracle,
			fri_oracles,
		}
	}

	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		1 + self.fri_oracles.len()
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
		self.codeword_oracle.open_queries(indices, advice)?;

		// Each subsequent oracle indexes the previous virtual oracle, so shift the query indices
		// right by the round's arity before opening it.
		let mut indices = indices.to_vec();
		for fri_oracle in &self.fri_oracles {
			for index in &mut indices {
				*index >>= fri_oracle.coset_log_size;
			}
			fri_oracle.open_queries(&indices, advice)?;
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
