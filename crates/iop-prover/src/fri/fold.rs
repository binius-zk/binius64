// Copyright 2024-2025 Irreducible Inc.

use std::{iter, mem};

use binius_field::{BinaryField, Field, PackedField};
use binius_iop::{
	fri::{FRIParams, fold::fold_chunk},
	merkle_tree::MerkleTreeScheme,
};
use binius_math::{
	FieldBuffer, FieldSlice, inner_product::inner_product_buffers,
	multilinear::eq::eq_ind_partial_eval, ntt::AdditiveNTT,
};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSampleBits, Challenger},
};
use binius_utils::{SerializeBytes, checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use tracing::instrument;

use super::{error::Error, query::FRIQueryProver};
use crate::{
	fri::{BatchBrakedownOracleProver, BrakedownOracleProver, FRIOracleProver},
	merkle_tree::MerkleTreeProver,
};

/// The type of the termination round codeword in the FRI protocol.
pub type TerminateCodeword<F> = FieldBuffer<F>;

pub enum FoldRoundOutput<VCSCommitment> {
	NoCommitment,
	Commitment(VCSCommitment),
}

enum FRIFolderState<'a, P, MTProver>
where
	P: PackedField,
	MTProver: MerkleTreeProver<P::Scalar>,
{
	FirstFold(BatchBrakedownFolder<'a, P, MTProver>),
	LaterFolds {
		first_oracle: BatchBrakedownOracleProver<'a, P, MTProver>,
		last_codeword: FieldBuffer<P::Scalar>,
		last_committed: MTProver::Committed,
		round_oracles: Vec<FRIOracleProver<'a, P::Scalar, MTProver>>,
	},
}
/// A stateful prover for the FRI fold phase.
pub struct FRIFoldProver<'a, F, P, NTT, MerkleProver>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	MerkleProver: MerkleTreeProver<F>,
{
	params: &'a FRIParams<F>,
	ntt: &'a NTT,
	merkle_prover: &'a MerkleProver,
	state: Option<FRIFolderState<'a, P, MerkleProver>>,
	curr_round: usize,
	next_commit_round: Option<usize>,
	unprocessed_challenges: Vec<F>,
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver> FRIFoldProver<'a, F, P, NTT, MerkleProver>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MerkleProver: MerkleTreeProver<F, Scheme = MerkleScheme>,
{
	/// Constructs a new folder for a single committed input oracle.
	pub fn new(
		params: &'a FRIParams<F>,
		ntt: &'a NTT,
		merkle_prover: &'a MerkleProver,
		committed_codeword: FieldBuffer<P>,
		committed: &'a MerkleProver::Committed,
	) -> Result<Self, Error> {
		Self::new_batch(params, ntt, merkle_prover, vec![(committed_codeword, committed)])
	}

	/// Constructs a new folder for a batch of committed input oracles.
	///
	/// The input oracles share the Reed-Solomon code but may have differing batch sizes; they are
	/// folded and combined into a single first-round codeword. The codewords must be supplied in
	/// the same order as [`FRIParams::input_oracles`], and each must match its oracle's batch
	/// size.
	pub fn new_batch(
		params: &'a FRIParams<F>,
		ntt: &'a NTT,
		merkle_prover: &'a MerkleProver,
		committed_codewords: Vec<(FieldBuffer<P>, &'a MerkleProver::Committed)>,
	) -> Result<Self, Error> {
		let input_oracles = params.input_oracles();
		if committed_codewords.len() != input_oracles.len() {
			return Err(Error::InvalidArgs(format!(
				"got {} committed codewords, expected {}",
				committed_codewords.len(),
				input_oracles.len(),
			)));
		}

		// Each input oracle's Reed-Solomon dimension must not exceed the first-round (reduced) code
		// dimension; smaller oracles are lifted (padded) to it. FRIParams only guarantees the
		// inequality `log_msg_len - log_batch_size <= rs_code.log_dim()`, so validate it here
		// rather than trusting the caller.
		let log_dim = params.rs_code().log_dim();
		let log_inv_rate = params.rs_code().log_inv_rate();
		for spec in input_oracles {
			if spec.log_msg_len - spec.log_batch_size > log_dim {
				return Err(Error::InvalidArgs(format!(
					"input oracle dimension {} exceeds the reduced code dimension {log_dim}",
					spec.log_msg_len - spec.log_batch_size,
				)));
			}
		}

		let folders = iter::zip(committed_codewords, input_oracles)
			.map(|((codeword, committed), spec)| {
				// The oracle's own codeword has dimension `log_msg_len - log_batch_size`, so its
				// interleaved length is `log_msg_len + log_inv_rate`. It is lifted to the common
				// first-round length by duplicating each entry `2^log_lift` times.
				let oracle_log_dim = spec.log_msg_len - spec.log_batch_size;
				let expected_log_len = spec.log_msg_len + log_inv_rate;
				if codeword.log_len() != expected_log_len {
					return Err(Error::InvalidArgs(
						"interleaved codeword length must match the oracle's Reed-Solomon code length \
						 plus its batch size"
							.to_string(),
					));
				}
				Ok(ProxTestFolder {
					log_batch_size: spec.log_batch_size,
					log_lift: log_dim - oracle_log_dim,
					codeword,
					merkle_committed: committed,
				})
			})
			.collect::<Result<Vec<_>, Error>>()?;
		let batch_folder = BatchBrakedownFolder::new(folders, params.rs_code().log_len());

		let next_commit_round = Some(params.log_batch_size());
		Ok(Self {
			params,
			ntt,
			merkle_prover,
			state: Some(FRIFolderState::FirstFold(batch_folder)),
			curr_round: 0,
			next_commit_round,
			unprocessed_challenges: Vec::with_capacity(params.rs_code().log_dim()),
		})
	}

	/// Number of fold rounds, including the final fold.
	pub fn n_rounds(&self) -> usize {
		self.params.n_fold_rounds()
	}

	/// Number of times `execute_fold_round` has been called.
	pub fn n_rounds_remaining(&self) -> usize {
		self.n_rounds() - self.curr_round
	}

	fn is_commitment_round(&self) -> bool {
		self.next_commit_round
			.is_some_and(|round| round == self.curr_round)
	}

	pub fn receive_challenge(&mut self, challenge: F) {
		self.unprocessed_challenges.push(challenge);
		self.curr_round += 1;
	}

	/// Executes the next fold round and returns the folded codeword commitment.
	///
	/// As a memory efficient optimization, this method may not actually do the folding, but instead
	/// accumulate the folding challenge for processing at a later time. This saves us from storing
	/// intermediate folded codewords.
	pub fn execute_fold_round(&mut self) -> FoldRoundOutput<MerkleScheme::Digest> {
		if !self.is_commitment_round() {
			return FoldRoundOutput::NoCommitment;
		}

		let state = self
			.state
			.take()
			.expect("state is always Some by struct invariant");

		let (root, new_state) = match state {
			FRIFolderState::FirstFold(folder) => {
				let _scope = tracing::debug_span!(
					"FRI Initial Fold",
					log_len = folder.log_len(),
					arity = self.unprocessed_challenges.len()
				)
				.entered();

				// Fold the batch of interleaved codewords that were originally committed into a
				// single codeword with the same block length, and turn them into a batched
				// Brakedown query oracle.
				let challenges = mem::take(&mut self.unprocessed_challenges);
				let (folded_codeword, first_oracle) = folder.fold(self.merkle_prover, &challenges);

				let next_arity = self.params.fold_arities().first().copied();
				let (root, last_committed) = self.commit_round(&folded_codeword, next_arity);

				let new_state = FRIFolderState::LaterFolds {
					first_oracle,
					last_codeword: folded_codeword,
					last_committed,
					round_oracles: Vec::with_capacity(self.params.fold_arities().len()),
				};
				(root, new_state)
			}
			FRIFolderState::LaterFolds {
				first_oracle,
				last_codeword,
				last_committed,
				mut round_oracles,
			} => {
				let _fri_round_scope = tracing::debug_span!(
					"FRI Round Fold",
					log_len = last_codeword.log_len(),
					arity = self.unprocessed_challenges.len()
				)
				.entered();

				// Fold a full codeword committed in the previous FRI round into a codeword with
				// reduced dimension and rate.
				let challenges = mem::take(&mut self.unprocessed_challenges);
				let fri_fold_span = tracing::debug_span!("FRI Fold").entered();
				let folded_codeword = fold_codeword(self.ntt, last_codeword.to_ref(), &challenges);
				drop(fri_fold_span);
				let oracle = FRIOracleProver::new(
					last_codeword,
					last_committed,
					self.merkle_prover,
					challenges.len(),
				);

				let next_arity = self
					.params
					.fold_arities()
					.get(round_oracles.len() + 1)
					.copied();
				let (root, last_committed) = self.commit_round(&folded_codeword, next_arity);

				round_oracles.push(oracle);
				let new_state = FRIFolderState::LaterFolds {
					first_oracle,
					last_codeword: folded_codeword,
					last_committed,
					round_oracles,
				};
				(root, new_state)
			}
		};

		self.state = Some(new_state);
		FoldRoundOutput::Commitment(root)
	}

	/// Commits to a folded codeword, advancing `next_commit_round` for the next fold.
	///
	/// The coset size is determined by the arity of the *next* fold round, or by the number of
	/// final challenges once there are no more committed rounds (the terminal codeword).
	fn commit_round(
		&mut self,
		folded_codeword: &FieldBuffer<F>,
		next_arity: Option<usize>,
	) -> (MerkleScheme::Digest, MerkleProver::Committed) {
		let log_coset_size = next_arity.unwrap_or_else(|| self.params.n_final_challenges());
		let coset_size = 1 << log_coset_size;

		let _merkle_tree_span = tracing::debug_span!("Merkle Tree").entered();
		let (commitment, committed) = self
			.merkle_prover
			.commit(folded_codeword.as_ref(), coset_size)
			.expect("merkle commitment cannot fail for a valid codeword");

		// The next commitment lands `next_arity` rounds after the current one. Once there is no
		// next arity, this is the terminal codeword and no further commitments are made.
		self.next_commit_round = next_arity.map(|arity| self.curr_round + arity);

		(commitment.root, committed)
	}

	/// Finalizes the FRI folding process.
	///
	/// This step will process any unprocessed folding challenges to produce the
	/// final folded codeword. Then it will decode this final folded codeword
	/// to get the final message. The result is the final message and a query prover instance.
	///
	/// This returns the final message and a query prover instance.
	#[instrument(skip_all, name = "fri::FRIFolder::finalize", level = "debug")]
	#[allow(clippy::type_complexity)]
	pub fn finalize(
		mut self,
	) -> Result<(TerminateCodeword<F>, FRIQueryProver<'a, F, P, MerkleProver, MerkleScheme>), Error>
	{
		if self.curr_round != self.n_rounds() {
			return Err(Error::EarlyProverFinish);
		}

		self.unprocessed_challenges.clear();

		match self
			.state
			.take()
			.expect("state is always Some by struct invariant")
		{
			// The final fold round produced the terminal codeword and committed the prior one, so
			// the state is always `LaterFolds` once `curr_round` reaches `n_rounds`. The
			// terminal codeword is sent in full and therefore is not wrapped in a query oracle.
			FRIFolderState::LaterFolds {
				first_oracle,
				last_codeword,
				round_oracles,
				..
			} => {
				let query_prover = FRIQueryProver::new(first_oracle, round_oracles);
				Ok((last_codeword, query_prover))
			}
			// The first fold fires at `curr_round == log_batch_size <= n_rounds` and
			// `execute_fold_round` runs every round, so the first fold always precedes `finalize`.
			FRIFolderState::FirstFold(_) => {
				unreachable!("the first fold always runs before curr_round reaches n_rounds")
			}
		}
	}

	pub fn finish_proof<Challenger_>(
		self,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error>
	where
		Challenger_: Challenger,
	{
		let n_test_queries = self.params.n_test_queries();
		let index_bits = self.params.index_bits();
		let (terminate_codeword, query_prover) = self.finalize()?;

		// Sample all query indices before writing the (per-oracle batched) query openings. The
		// decommitment advice is not absorbed by the challenger, so this matches the verifier
		// sampling all indices up front.
		let indices = (0..n_test_queries)
			.map(|_| transcript.sample_bits(index_bits) as usize)
			.collect::<Vec<_>>();

		// Write the per-oracle batched query openings, then the terminal codeword in full.
		query_prover.prove_queries(&indices, &mut transcript.decommitment())?;
		transcript
			.decommitment()
			.write_scalar_slice(terminate_codeword.as_ref());

		Ok(())
	}
}

/// Fold the interleaved codeword into a single codeword using the given challenges.
///
/// ## Arguments
///
/// * `codeword` - an interleaved codeword.
/// * `challenges` - the folding challenges. The length must be at least `log_batch_size`.
/// * `log_len` - the binary logarithm of the code length.
/// * `log_batch_size` - the binary logarithm of the interleaved code batch size.
///
/// See [DP24], Def. 3.6 and Lemma 3.9 for more details.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[instrument(skip_all, level = "debug")]
fn fold_interleaved<F, P>(
	codeword: FieldSlice<P>,
	challenges: &[F],
	log_len: usize,
	log_batch_size: usize,
) -> FieldBuffer<F>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert_eq!(codeword.log_len(), log_len + log_batch_size);
	assert_eq!(challenges.len(), log_batch_size);

	let tensor = eq_ind_partial_eval(challenges);

	// For each chunk of size `2^chunk_size` in the interleaved codeword, fold it with the folding
	// challenges.
	let values = codeword
		.chunks_par(log_batch_size)
		.map(|chunk| inner_product_buffers(&chunk, &tensor))
		.collect::<Vec<_>>();
	FieldBuffer::new(log_len, values.into_boxed_slice())
}

/// FRI-fold the codeword using the given challenges.
///
/// ## Arguments
///
/// * `ntt` - the NTT instance, used to look up the twiddle values.
/// * `codeword` - an interleaved codeword.
/// * `challenges` - the folding challenges. The length must be at least `log_batch_size`.
/// * `log_len` - the binary logarithm of the code length.
///
/// See [DP24], Def. 3.6 and Lemma 3.9 for more details.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[instrument(skip_all, level = "debug")]
fn fold_codeword<F, NTT>(ntt: &NTT, codeword: FieldSlice<F>, challenges: &[F]) -> FieldBuffer<F>
where
	F: BinaryField,
	NTT: AdditiveNTT<Field = F> + Sync,
{
	let log_len = codeword.log_len();
	assert!(challenges.len() <= log_len);

	let folded_log_len = log_len - challenges.len();

	// For each coset of size `2^chunk_size` in the codeword, fold it with the folding challenges.
	let chunk_size = 1 << challenges.len();
	let values: Vec<F> = codeword
		.chunks_par(challenges.len())
		.enumerate()
		.map_init(
			|| vec![F::default(); chunk_size],
			|scratch_buffer, (i, chunk)| {
				scratch_buffer.copy_from_slice(chunk.as_ref());
				fold_chunk(ntt, log_len, i, scratch_buffer, challenges)
			},
		)
		.collect();
	FieldBuffer::new(folded_log_len, values.into_boxed_slice())
}

pub struct ProxTestFolder<'a, P: PackedField, MTProver: MerkleTreeProver<P::Scalar>> {
	log_batch_size: usize,
	/// log2 the lift factor (oracle padding): how many times each folded codeword entry is
	/// duplicated to reach the common first-round length. Zero when no lifting is needed.
	log_lift: usize,
	codeword: FieldBuffer<P>,
	merkle_committed: &'a MTProver::Committed,
}

impl<'a, F: Field, P: PackedField<Scalar = F>, MTProver> ProxTestFolder<'a, P, MTProver>
where
	MTProver: MerkleTreeProver<F>,
{
	pub fn log_folded_len(&self) -> usize {
		self.codeword.log_len() - self.log_batch_size
	}

	pub fn fold(
		self,
		merkle_prover: &'a MTProver,
		challenges: &[F],
	) -> (FieldBuffer<F>, BrakedownOracleProver<'a, P, MTProver>) {
		assert_eq!(challenges.len(), self.log_batch_size); // precondition

		let ProxTestFolder {
			log_batch_size,
			log_lift,
			codeword,
			merkle_committed,
		} = self;

		let log_len = codeword.log_len() - log_batch_size;
		let folded_codeword =
			fold_interleaved(codeword.to_ref(), challenges, log_len, log_batch_size);
		let oracle = BrakedownOracleProver::new(
			codeword,
			merkle_committed,
			merkle_prover,
			log_batch_size,
			log_lift,
		);
		(folded_codeword, oracle)
	}
}

/// Folds and commits a batch of interleaved codewords that share a folded length.
///
/// Each [`ProxTestFolder`] is committed separately and folds (by the same challenges) to a codeword
/// of the common length `codeword.log_len() - log_batch_size`. The folded codewords are summed into
/// a single codeword that continues through the FRI rounds, and the per-commitment
/// [`BrakedownOracleProver`]s are bundled into a [`BatchBrakedownOracleProver`].
pub struct BatchBrakedownFolder<'a, P: PackedField, MTProver: MerkleTreeProver<P::Scalar>> {
	log_code_len: usize,
	folders: Vec<ProxTestFolder<'a, P, MTProver>>,
}

impl<'a, F: Field, P: PackedField<Scalar = F>, MTProver> BatchBrakedownFolder<'a, P, MTProver>
where
	MTProver: MerkleTreeProver<F>,
{
	/// Constructs a batch folder from one or more interleaved-codeword folders.
	///
	/// `log_code_len` is the common (first-round) codeword length the folders combine into. Each
	/// folder's own folded length must not exceed it; folders that fall short are lifted (their
	/// folded codewords duplicated) up to `log_code_len` during [`Self::fold`].
	pub fn new(folders: Vec<ProxTestFolder<'a, P, MTProver>>, log_code_len: usize) -> Self {
		assert!(!folders.is_empty()); // precondition
		for folder in &folders {
			assert!(folder.log_folded_len() <= log_code_len);
		}
		Self {
			log_code_len,
			folders,
		}
	}

	/// Log2 length of the (interleaved) codewords being folded.
	fn log_len(&self) -> usize {
		let max_folder_log_len = self
			.folders
			.iter()
			.map(|folder| folder.codeword.log_len())
			.max()
			.expect("folders is not empty by struct invariant");
		let log_folders = log2_ceil_usize(self.folders.len());
		max_folder_log_len + log_folders
	}

	pub fn fold(
		self,
		merkle_prover: &'a MTProver,
		challenges: &[F],
	) -> (FieldBuffer<F>, BatchBrakedownOracleProver<'a, P, MTProver>) {
		let max_log_batch_size = self
			.folders
			.iter()
			.map(|folder| folder.log_batch_size)
			.max()
			.expect("folders is not empty by struct invariant");

		let (inner_challenges, outer_challenges) = challenges.split_at(max_log_batch_size);

		let n_folders = self.folders.len();
		let folds = self.folders.into_iter().map(|folder| {
			let start_idx = max_log_batch_size - folder.log_batch_size;
			folder.fold(merkle_prover, &inner_challenges[start_idx..])
		});

		let outer_tensor = eq_ind_partial_eval::<F>(outer_challenges);
		let mut combined_codeword = FieldBuffer::zeros(self.log_code_len);
		let mut oracles = Vec::with_capacity(n_folders);
		// TODO: Special cases when outer_challenges.len() = 0 or 1 for computational efficiency (to
		// reduce # of scaling muls)
		for ((folded_codeword, oracle), &scalar) in iter::zip(folds, outer_tensor.as_ref()) {
			// Lift (pad) the folded codeword up to the common length by duplicating each entry
			// `2^log_lift` times: the lifted entry at index `j` is the folded entry at `j >>
			// log_lift` (a contiguous duplication, matching the Reed-Solomon codeword
			// duplication identity).
			let log_lift = self.log_code_len - folded_codeword.log_len();
			let folded = folded_codeword.as_ref();
			for (j, acc) in combined_codeword.as_mut().iter_mut().enumerate() {
				*acc += folded[j >> log_lift] * scalar;
			}
			oracles.push(oracle);
		}

		(combined_codeword, BatchBrakedownOracleProver::new(oracles))
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField128bGhash as B128;
	use binius_math::{
		BinarySubspace,
		fold::fold_cols,
		ntt::{NeighborsLastReference, domain_context::GenericOnTheFly},
		test_utils::{random_field_buffer, random_scalars},
	};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	proptest! {
		#[test]
		fn test_fri_compatible_ntt_domains(log_dim in 0..8usize, arity in 0..4usize) {
			test_help_fri_compatible_ntt_domains(log_dim, arity);
		}
	}

	fn test_help_fri_compatible_ntt_domains(log_dim: usize, arity: usize) {
		let subspace = BinarySubspace::with_dim(32);
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastReference { domain_context };

		let mut rng = StdRng::seed_from_u64(0);
		let msg = random_field_buffer(&mut rng, log_dim + arity);
		let challenges = random_scalars(&mut rng, arity);

		let query = eq_ind_partial_eval::<B128>(&challenges);

		// Fold the message using regular folding.
		let mut folded_msg = fold_cols(&msg, &query);
		assert_eq!(folded_msg.log_len(), log_dim);

		// Encode the message over the large domain.
		let mut codeword = msg;
		ntt.forward_transform(codeword.to_mut(), 0, 0);

		// Fold the encoded message using FRI folding.
		let folded_codeword = fold_codeword(&ntt, codeword.to_ref(), &challenges);

		// Encode the folded message.
		ntt.forward_transform(folded_msg.to_mut(), 0, 0);

		// Check that folding and encoding commute.
		assert_eq!(folded_codeword, folded_msg);
	}
}
