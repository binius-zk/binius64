// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter::{self, repeat_with};

use binius_field::BinaryField;
use binius_math::ntt::domain_context::GenericOnTheFly;
use binius_transcript::{
	TranscriptReader, VerifierTranscript,
	fiat_shamir::{CanSampleBits, Challenger},
};
use binius_utils::{DeserializeBytes, checked_arithmetics::log2_ceil_usize};
use bytes::Buf;

use super::{
	batch::{BatchBrakedownOracle, BrakedownOracle, FRIOracle, ProxTestOracle, fold_coset},
	common::FRIParams,
	error::{Error, VerificationError},
};
use crate::merkle_tree::{Commitment, MerkleTreeScheme};

/// A verifier for the FRI query phase.
///
/// The verifier is instantiated after the folding rounds and is used to test consistency of the
/// round messages and the original purported codeword.
///
/// Internally, this is a composition of `ProxTestOracle`s: a `BatchBrakedownOracle` performs
/// the first, interleaved reduction of the committed codeword(s), then one `FRIOracle` per fold
/// arity performs each subsequent FRI reduction. The verifier orchestrates the consistency checks
/// between these oracles and the final, fully-folded terminal codeword.
pub struct FRIQueryVerifier<'a, F, VCS>
where
	F: BinaryField,
	VCS: MerkleTreeScheme<F>,
{
	params: &'a FRIParams<F>,
	vcs: &'a VCS,
	/// Commitment to the fully-folded terminal codeword, sent in full by the prover.
	terminal_commitment: &'a VCS::Digest,
	/// The folding challenges applied after the last committed oracle.
	final_challenges: &'a [F],
	/// Performs the first, interleaved reduction of the committed codeword(s).
	codeword_oracle: BatchBrakedownOracle<F, &'a VCS>,
	/// Performs each subsequent FRI reduction, one per fold arity.
	fri_oracles: Vec<FRIOracle<F, &'a VCS, GenericOnTheFly<F>>>,
}

impl<'a, F, VCS> FRIQueryVerifier<'a, F, VCS>
where
	F: BinaryField,
	VCS: MerkleTreeScheme<F, Digest: DeserializeBytes>,
{
	pub fn new(
		params: &'a FRIParams<F>,
		vcs: &'a VCS,
		codeword_commitment: &'a VCS::Digest,
		round_commitments: &'a [VCS::Digest],
		challenges: &'a [F],
	) -> Self {
		Self::new_batch(
			params,
			vcs,
			std::slice::from_ref(codeword_commitment),
			round_commitments,
			challenges,
		)
	}

	/// Constructs a query verifier for a batch of committed input oracles.
	///
	/// The input oracles share the Reed-Solomon code but may have differing batch sizes; they are
	/// reduced into a single first-round FRI oracle. The commitments must be supplied in the same
	/// order as [`FRIParams::input_oracles`].
	///
	/// ## Preconditions
	///
	/// * `codeword_commitments.len()` must equal `params.input_oracles().len()`.
	/// * `round_commitments.len()` must equal `params.n_oracles()`.
	/// * `challenges.len()` must equal `params.n_fold_rounds()`.
	/// * Each input oracle's dimension (`log_msg_len - log_batch_size`) must be at most
	///   `params.rs_code().log_dim()`.
	pub fn new_batch(
		params: &'a FRIParams<F>,
		vcs: &'a VCS,
		codeword_commitments: &'a [VCS::Digest],
		round_commitments: &'a [VCS::Digest],
		challenges: &'a [F],
	) -> Self {
		assert_eq!(
			codeword_commitments.len(),
			params.input_oracles().len(),
			"precondition: codeword_commitments.len() must equal params.input_oracles().len()"
		);
		assert_eq!(
			round_commitments.len(),
			params.n_oracles(),
			"precondition: round_commitments.len() must equal params.n_oracles()"
		);
		assert_eq!(
			challenges.len(),
			params.n_fold_rounds(),
			"precondition: challenges.len() must equal params.n_fold_rounds()"
		);

		// Each input oracle's Reed-Solomon dimension (`log_dim - log_lift`) must not exceed the
		// first-round (reduced) code dimension; smaller oracles are lifted (padded) to it. This
		// holds whenever `log_lift <= log_dim`, so assert it here rather than trusting the
		// caller.
		let log_dim = params.rs_code().log_dim();
		let log_inv_rate = params.rs_code().log_inv_rate();
		for spec in params.input_oracles() {
			assert!(
				spec.log_lift <= log_dim,
				"precondition: input oracle dimension must not exceed the reduced code dimension"
			);
		}

		// The committed codeword's Merkle tree has one coset per leaf, so its depth is the number
		// of index bits.
		let index_bits = params.index_bits();
		// The first fold consumes `log_batch_size()` challenges, ordered `[early ++ outer ++
		// later]`: `max_early` early within-oracle batch challenges, then `log_n_oracles` outer
		// challenges (batching the oracles together), then `max_later` later within-oracle batch
		// challenges. Oracle `i` folds its interleaving with `early_window ++ later_window`, the
		// suffixes of the early and later groups of lengths `log_early_batch_size_i` and
		// `log_later_batch_size_i`.
		let max_early = params
			.input_oracles()
			.iter()
			.map(|spec| spec.log_early_batch_size)
			.max()
			.expect("input_oracles is non-empty as an invariant");
		let max_later = params
			.input_oracles()
			.iter()
			.map(|spec| spec.log_later_batch_size)
			.max()
			.expect("input_oracles is non-empty as an invariant");
		let log_n_oracles = log2_ceil_usize(params.input_oracles().len());
		let early_challenges = &challenges[..max_early];
		let outer_challenges = challenges[max_early..max_early + log_n_oracles].to_vec();
		let later_challenges = &challenges[max_early + log_n_oracles..params.log_batch_size()];
		let codeword_sub_oracles = iter::zip(codeword_commitments, params.input_oracles())
			.map(|(commitment, spec)| {
				// The oracle's own codeword has dimension `log_dim - log_lift`, so its Merkle tree
				// depth is that plus the inverse rate. It is lifted to the common first-round
				// length (`index_bits`) by duplicating each entry `2^log_lift` times.
				let oracle_log_dim = log_dim - spec.log_lift;
				let depth = oracle_log_dim + log_inv_rate;
				let log_lift = spec.log_lift;
				let early_window = &early_challenges[max_early - spec.log_early_batch_size..];
				let later_window = &later_challenges[max_later - spec.log_later_batch_size..];
				let fold_challenges: Vec<F> =
					early_window.iter().chain(later_window).copied().collect();
				BrakedownOracle::new(
					fold_challenges,
					Commitment {
						root: commitment.clone(),
						depth,
					},
					vcs,
					log_lift,
				)
			})
			.collect();
		let codeword_oracle = BatchBrakedownOracle::new(codeword_sub_oracles, outer_challenges);

		// All FRI reductions fold cosets of the same Reed–Solomon codeword domain, so they share a
		// single domain context.
		let domain_context = GenericOnTheFly::generate_from_subspace(params.rs_code().subspace());
		let mut fri_oracles = Vec::with_capacity(params.fold_arities().len());
		let mut depth = index_bits;
		let mut fold_round = params.log_batch_size();
		for (round_commitment, &arity) in iter::zip(round_commitments, params.fold_arities()) {
			depth -= arity;
			fri_oracles.push(FRIOracle::new(
				challenges[fold_round..fold_round + arity].to_vec(),
				Commitment {
					root: round_commitment.clone(),
					depth,
				},
				vcs,
				domain_context.clone(),
			));
			fold_round += arity;
		}

		let final_challenges = &challenges[fold_round..];
		let terminal_commitment = round_commitments
			.last()
			.expect("round_commitments is non-empty as an invariant");

		Self {
			params,
			vcs,
			terminal_commitment,
			final_challenges,
			codeword_oracle,
			fri_oracles,
		}
	}

	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		self.params.n_oracles()
	}

	pub fn verify<Challenger_>(
		&self,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<F, Error>
	where
		Challenger_: Challenger,
	{
		// Sample all query indices up front to facilitate batched Merkle openings.
		let mut indices = repeat_with(|| transcript.sample_bits(self.params.index_bits()) as usize)
			.take(self.params.n_test_queries())
			.collect::<Vec<_>>();

		// Open and reduce the queries through each oracle in turn, reading the per-oracle batched
		// openings from a single continuous decommitment stream.
		let mut advice = transcript.decommitment();
		let mut claims = self.codeword_oracle.open_queries(&indices, &mut advice)?;
		for (query_round, (oracle, &arity)) in self
			.fri_oracles
			.iter()
			.zip(self.params.fold_arities())
			.enumerate()
		{
			claims = oracle
				.reduce_queries(&indices, &claims, &mut advice)
				.map_err(|err| match err {
					super::batch::Error::ClaimMismatch { index } => {
						VerificationError::IncorrectFold { query_round, index }.into()
					}
					err => Error::from(err),
				})?;
			for index in &mut indices {
				*index >>= arity;
			}
		}

		// Check the fully-reduced queries against the terminal codeword sent in full.
		self.verify_terminal_queries(&claims, &indices, &mut advice)
	}

	/// Verifies the terminal codeword the prover sends in full at the end of the query phase.
	///
	/// Reads the terminal codeword from the transcript and checks it against its commitment, then
	/// checks that the fully-reduced query `claims` match it at the queried `indices`. Finally it
	/// folds each coset of the terminal codeword and checks they are equal, i.e. that it is a
	/// repetition codeword of the claimed low degree, and returns the fully-folded message value.
	fn verify_terminal_queries<B: Buf>(
		&self,
		claims: &[F],
		indices: &[usize],
		advice: &mut TranscriptReader<B>,
	) -> Result<F, Error> {
		let n_final_challenges = self.params.n_final_challenges();
		let log_inv_rate = self.params.rs_code().log_inv_rate();
		let terminate_codeword_len = 1 << (n_final_challenges + log_inv_rate);

		let terminate_codeword = advice
			.read_scalar_slice::<F>(terminate_codeword_len)
			.map_err(Error::TranscriptError)?;
		self.vcs.verify_vector(
			self.terminal_commitment,
			&terminate_codeword,
			1 << n_final_challenges,
			advice,
		)?;

		// Check the fully-reduced claims against the terminal codeword the verifier holds in full.
		for (&claim, &index) in iter::zip(claims, indices) {
			if claim != terminate_codeword[index] {
				return Err(VerificationError::IncorrectFold {
					query_round: self.n_oracles() - 1,
					index,
				}
				.into());
			}
		}

		// Fold each coset of the terminal codeword and check that the folds are all equal, i.e.
		// that the codeword has the claimed low degree.
		let domain_context =
			GenericOnTheFly::generate_from_subspace(self.params.rs_code().subspace());
		let log_len = n_final_challenges + log_inv_rate;
		let repetition_codeword = terminate_codeword
			.chunks(1 << n_final_challenges)
			.enumerate()
			.map(|(coset_index, coset)| {
				fold_coset(
					&domain_context,
					log_len,
					coset_index,
					self.final_challenges,
					coset.to_vec(),
				)
			})
			.collect::<Vec<_>>();

		let final_value = repetition_codeword[0];

		// Check that the fully-folded purported codeword is a repetition codeword.
		if repetition_codeword[1..]
			.iter()
			.any(|&entry| entry != final_value)
		{
			return Err(VerificationError::IncorrectDegree.into());
		}

		Ok(final_value)
	}
}
