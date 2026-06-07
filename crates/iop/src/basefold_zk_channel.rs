// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP verifier channel.
//!
//! This module provides [`BaseFoldZKVerifierChannel`], which implements [`IOPVerifierChannel`]
//! using FRI commitment and ZK BaseFold opening protocols. Unlike [`super::basefold_channel`],
//! this channel always applies zero-knowledge blinding to all oracles.

use std::iter;

use binius_field::BinaryField;
use binius_ip::{
	channel::IPVerifierChannel,
	sumcheck::{self, BatchSumcheckOutput},
};
use binius_math::{
	line::extrapolate_line_packed, multilinear::eq::eq_ind_zero, univariate::evaluate_univariate,
};
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::DeserializeBytes;
use itertools::izip;

use crate::{
	basefold,
	channel::{Error, IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	fri::FRIParams,
	merkle_tree::MerkleTreeScheme,
};

/// Oracle handle returned by [`BaseFoldZKVerifierChannel::recv_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldZKOracle {
	index: usize,
}

/// A verifier channel that uses ZK BaseFold for all oracle commitments and openings.
///
/// This channel always applies zero-knowledge blinding. The FRI parameters must be set up
/// with `log_batch_size = 1` and `log_msg_len = witness_log_len + 1` to account for the mask.
///
/// # Type Parameters
///
/// - `'a`: Lifetime for borrowed references
/// - `F`: The binary field type
/// - `MerkleScheme_`: The Merkle tree scheme for commitments
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct BaseFoldZKVerifierChannel<'a, F, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F>,
	Challenger_: Challenger,
{
	transcript: &'a mut VerifierTranscript<Challenger_>,
	merkle_scheme: &'a MerkleScheme_,
	oracle_specs: &'a [OracleSpec],
	fri_params: &'a [FRIParams<F>],
	oracle_commitments: Vec<MerkleScheme_::Digest>,
	next_oracle_index: usize,
}

impl<'a, F, MerkleScheme_, Challenger_> BaseFoldZKVerifierChannel<'a, F, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	/// Creates a new BaseFold ZK verifier channel from precomputed FRI parameters.
	///
	/// The FRI parameters should already account for ZK (log_batch_size = 1, doubled message
	/// length).
	pub fn from_precomputed(
		transcript: &'a mut VerifierTranscript<Challenger_>,
		merkle_scheme: &'a MerkleScheme_,
		oracle_specs: &'a [OracleSpec],
		fri_params: &'a [FRIParams<F>],
	) -> Self {
		Self {
			transcript,
			merkle_scheme,
			oracle_specs,
			fri_params,
			oracle_commitments: Vec::new(),
			next_oracle_index: 0,
		}
	}

	/// Returns a reference to the underlying transcript.
	pub fn transcript(&self) -> &VerifierTranscript<Challenger_> {
		self.transcript
	}

	/// Consumes the channel, asserting all oracle specs have been consumed.
	pub fn finish(self) {
		let n_remaining = self.oracle_specs.len() - self.next_oracle_index;
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining",);
	}
}

impl<F, MerkleScheme_, Challenger_> IPVerifierChannel<F>
	for BaseFoldZKVerifierChannel<'_, F, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	type Elem = F;

	fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
		self.transcript
			.message()
			.read_scalar()
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
		self.transcript
			.message()
			.read_scalar_slice(n)
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
		self.transcript
			.message()
			.read()
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn sample(&mut self) -> F {
		CanSample::sample(&mut self.transcript)
	}

	fn observe_one(&mut self, val: F) -> F {
		self.transcript.observe().write_scalar(val);
		val
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
		self.transcript.observe().write_scalar_slice(vals);
		vals.to_vec()
	}

	fn assert_zero(&mut self, val: F) -> Result<(), binius_ip::channel::Error> {
		if val == F::ZERO {
			Ok(())
		} else {
			Err(binius_ip::channel::Error::InvalidAssert)
		}
	}

	fn compute_public_value(&mut self, inputs: &[F], f: impl FnOnce(&[F]) -> F) -> F {
		f(inputs)
	}
}

impl<F, MerkleScheme_, Challenger_> IOPVerifierChannel<F>
	for BaseFoldZKVerifierChannel<'_, F, MerkleScheme_, Challenger_>
where
	F: BinaryField,
	MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	type Oracle = BaseFoldZKOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, Error> {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining oracle specs"
		);

		let index = self.next_oracle_index;

		let commitment = self
			.transcript
			.message()
			.read::<MerkleScheme_::Digest>()
			.map_err(|_| Error::ProofEmpty)?;

		self.oracle_commitments.push(commitment);
		self.next_oracle_index += 1;

		Ok(BaseFoldZKOracle { index })
	}

	fn verify_oracle_relations<'a>(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'a, Self::Oracle, Self::Elem>>,
	) -> Result<(), Error> {
		let relations = oracle_relations.into_iter().collect::<Vec<_>>();
		if relations.is_empty() {
			return Ok(());
		}

		let indices: Vec<usize> = relations.iter().map(|r| r.oracle.index).collect();
		for &index in &indices {
			assert!(
				index < self.oracle_commitments.len(),
				"oracle index {index} out of bounds, expected < {}",
				self.oracle_commitments.len()
			);
		}
		let n_vars: Vec<usize> = indices
			.iter()
			.map(|&i| self.oracle_specs[i].log_msg_len)
			.collect();
		let max_n = *n_vars.iter().max().expect("relations is non-empty");

		// === Masking step ===
		// Read the masked inner products σ_i, sample γ, and form the masked claims s_i'.
		let sigmas = self.recv_many(relations.len())?;
		let gamma = self.sample();
		let sum_primes = iter::zip(&relations, sigmas)
			.map(|(relation, sigma)| extrapolate_line_packed(relation.claim, sigma, gamma))
			.collect::<Vec<_>>();

		// === Phase A: batched sumcheck on the masked claims (degree 2, bivariate product) ===
		let BatchSumcheckOutput {
			batch_coeff: sumcheck_batch_coeff,
			eval: sumcheck_reduced_eval,
			challenges: sumcheck_challenges,
		} = sumcheck::batch_verify::<F, _>(max_n, 2, &sum_primes, self.transcript)?;
		let alphas = self.recv_many(relations.len())?;

		// `batch_verify` returns binding-order challenges; reverse to variable-indexed
		// (low-to-high).
		let mut point = sumcheck_challenges;
		point.reverse();

		// Reduce the batched claim: each oracle contributes α_i · t_i(ρ_i) · eq(0^extra, padding),
		// combined with the batching coefficient.
		let contributions: Vec<F> = izip!(relations, &n_vars, &alphas)
			.map(|(relation, &n_i, &alpha_i)| {
				let (eval_coords, padding_coords) = point.split_at(n_i);
				let pad_eq = eq_ind_zero(padding_coords);
				let transparent_eval = (relation.transparent)(eval_coords);
				alpha_i * transparent_eval * pad_eq
			})
			.collect();
		let expected = evaluate_univariate(&contributions, sumcheck_batch_coeff);
		self.assert_zero(sumcheck_reduced_eval - expected)?;

		// === Phase B: per-oracle MLE-check BaseFold verification ===
		for (index, alpha, n_i) in izip!(indices, alphas, n_vars) {
			let commitment = self.oracle_commitments[index].clone();
			let basefold::ReducedOutput {
				final_fri_value,
				final_sumcheck_value,
				..
			} = basefold::verify_mlecheck_basefold_zk(
				&self.fri_params[index],
				self.merkle_scheme,
				commitment,
				alpha,
				&point[..n_i],
				gamma,
				self.transcript,
			)?;

			// The MLE-check internalizes the eq factor, so consistency is plain equality.
			self.assert_zero(final_sumcheck_value - final_fri_value)?;
		}

		Ok(())
	}
}
