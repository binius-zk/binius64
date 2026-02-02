// Copyright 2025-2026 The Binius Developers
use std::iter::chain;

use binius_field::{BinaryField, Field, PackedField};
use binius_iop_prover::{
	fri::{self, CommitOutput, Error, commit_interleaved},
	merkle_tree::MerkleTreeProver,
};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, ntt::AdditiveNTT};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{fri::FRIParams, merkle_tree::MerkleTreeScheme};
use itertools::Itertools;

use crate::protocols::{
	logup::LogUp,
	sumcheck::{
		Error as SumcheckError,
		batch::{BatchSumcheckOutput, batch_prove_and_write_evals},
		batch_quadratic::BatchQuadraticSumcheckProver,
	},
};

/// Output of the pushforward sumcheck, grouped by lookup and table claims.
#[derive(Debug, Clone)]
pub struct PushforwardEvalClaims<F: Field> {
	/// Sumcheck challenge point (low-to-high order).
	pub challenges: Vec<F>,
	/// Pushforward evaluations at the sumcheck point.
	pub pushforward_evals: Vec<F>,
	/// Table evaluations at the sumcheck point.
	pub table_evals: Vec<F>,
}

impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
	/// Proves the outer instance, reducing lookup value claims to pushforward claims.
	pub fn prove_pushforward<
		// N_MLES is the total number of MLEs involved, this is precisely N_LOOKUPS + N_TABLES.
		const N_MLES: usize,
	>(
		&self,
		channel: &mut impl IPProverChannel<F>,
	) -> Result<PushforwardEvalClaims<F>, SumcheckError> {
		// TODO: Remove implicit assumption of equal table size.
		assert_eq!(2 * N_TABLES, N_MLES);

		let mles: [FieldBuffer<P>; N_MLES] =
			chain(self.push_forwards.iter().cloned(), self.tables.iter().cloned())
				.collect_array()
				.expect("N_TABLES + N_LOOKUPS == N_MLES");

		let pushforward_composition = |mle_evals: [P; N_MLES], comp_evals: &mut [P; N_TABLES]| {
			// Enforce pushforward[i] * table[i] at each lookup slot.
			let (pushforwards, tables) = mle_evals.split_at(N_TABLES);
			for i in 0..N_TABLES {
				comp_evals[i] = pushforwards[i] * tables[i]
			}
		};
		// The composition is purely quadratic, so the infinity composition matches the regular one.

		// Build a single quadratic sumcheck that ties each lookup batch to its table.
		let prover = BatchQuadraticSumcheckProver::new(
			mles,
			pushforward_composition,
			pushforward_composition,
			self.batched_evals,
		)?;

		let BatchSumcheckOutput {
			challenges,
			multilinear_evals,
		} = batch_prove_and_write_evals(vec![prover], channel)?;

		let (pushforward_evals, table_evals) = multilinear_evals[0].split_at(N_TABLES);
		// The batch MLE order is [pushforwards..., tables...], so split accordingly.

		Ok(PushforwardEvalClaims {
			challenges,
			pushforward_evals: pushforward_evals.to_vec(),
			table_evals: table_evals.to_vec(),
		})
	}

	pub fn commit_pushforward<NTT, VCS, MerkleProver>(
		&self,
		params: &FRIParams<F>,
		ntt: &NTT,
		merkle_prover: &MerkleProver,
	) -> Result<CommitOutput<P, VCS::Digest, MerkleProver::Committed>, Error>
	where
		F: BinaryField,
		NTT: AdditiveNTT<Field = F> + Sync,
		MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
		VCS: MerkleTreeScheme<F>,
	{
		// We assume that all pushforwards are of the same length.
		let pushforward_log_len = self.push_forwards[0].log_len();
		let batch_next_pow_2 = 1 << (log2_ceil_usize(N_TABLES) + pushforward_log_len);
		let mut batch_pushforward: Vec<F> = self
			.push_forwards
			.iter()
			.flat_map(|push_forward| push_forward.iter_scalars())
			.collect();

		batch_pushforward.resize_with(batch_next_pow_2, || F::ZERO);

		let batch_pushforward = FieldBuffer::from_values(&batch_pushforward);

		commit_interleaved(params, ntt, merkle_prover, batch_pushforward.to_ref())
	}
}
