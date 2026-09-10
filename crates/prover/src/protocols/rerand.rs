// Copyright 2026 The Binius Developers

//! Prover for the BitAnd sumcheck batched with the operand-column MLE-checks.
//!
//! See [`binius_verifier::protocols::rerand`] for the protocol.

use std::iter;

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		MleToSumCheckDecorator, PaddedSumcheckDecorator, batch::batch_prove_and_write_evals,
		common::MleCheckProver, mle_store::MleStore, multilinear_eval::MultilinearEvalEvaluator,
		round_evaluator::SharedMleCheckProver,
	},
};
use binius_math::inner_product::inner_product;
use binius_verifier::protocols::rerand::{OperandClaims, RerandOutput, SUMCHECK_DEGREE};
use either::Either;

use crate::fold_word::BitAxisFolder;

/// One reduction's operand word columns with their per-bit claims.
pub struct OperandWitness<'a, F> {
	/// One column per operand, in the same order as `claims.columns`.
	pub words: Vec<&'a [Word]>,
	/// The per-bit claims on the columns, at the reduction's point.
	pub claims: OperandClaims<'a, F>,
}

/// Proves the BitAnd sumcheck batched with the operand-column MLE-checks.
///
/// Every summand runs as a sumcheck in plain form, zero-padded on its high variables to the
/// longest summand's variable count. The evaluations go out flat in summand order:
/// `[a, b, c, sigma_1, ..., sigma_n]`.
///
/// # Arguments
///
/// * `bitand` - the BitAnd MLE-check prover, from the univariate skip's fold.
/// * `bitand_claim` - its claim, the univariate-skip polynomial at `z`.
/// * `lagrange` - the Lagrange weights at `z` on the 64-point domain.
/// * `operands` - the operand columns of each multiplication reduction, with their claims.
///
/// The per-bit claims of `operands` must be in the transcript before `z` was drawn, matching
/// [`binius_verifier::protocols::rerand::verify`].
///
/// # Preconditions
///
/// * each operand's column count equals its claim count
/// * each column's length rounds up to `2^point.len()` for its reduction's point
pub fn prove<F, P, Channel, A>(
	bitand: impl MleCheckProver<F>,
	bitand_claim: F,
	lagrange: &[F],
	operands: &[OperandWitness<'_, F>],
	channel: &mut Channel,
	alloc: &A,
) -> RerandOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	let _scope = tracing::debug_span!("BitAnd batched sumcheck").entered();

	// ponytail: ten columns in the sumcheck where two theta-combined tables would do (spec §4.7.1).
	// Combine them only if this sumcheck shows in a profile.
	let folder = BitAxisFolder::new(lagrange);
	let operand_provers = operands.iter().map(|operand| {
		let OperandClaims { point, columns } = &operand.claims;
		assert_eq!(operand.words.len(), columns.len());

		let mut store = MleStore::<A, P>::new(point.len(), alloc);
		let evaluators = iter::zip(&operand.words, columns)
			.map(|(words, per_bit_claims)| {
				let col = store.push_owned(folder.fold::<P, _>(alloc, words));
				let claim = inner_product(per_bit_claims.iter().copied(), lagrange.iter().copied());
				(claim, MultilinearEvalEvaluator::new(col))
			})
			.collect::<Vec<_>>();
		let claims = evaluators.iter().map(|&(claim, _)| claim).collect();
		(SharedMleCheckProver::new(store, evaluators, point.to_vec()), claims)
	});
	let summands = iter::once((Either::Left(bitand), vec![bitand_claim]))
		.chain(operand_provers.map(|(prover, claims)| (Either::Right(prover), claims)))
		.collect::<Vec<_>>();

	let n_vars = summands
		.iter()
		.map(|(prover, _)| prover.n_vars())
		.max()
		.expect("the BitAnd summand is always present");
	let provers = summands
		.into_iter()
		.map(|(prover, claims)| {
			let n_extra_vars = n_vars - prover.n_vars();
			let prover = MleToSumCheckDecorator::new(prover);
			PaddedSumcheckDecorator::new(prover, n_extra_vars, claims, SUMCHECK_DEGREE)
		})
		.collect();

	let output = batch_prove_and_write_evals(provers, channel);

	let mut evals = output.multilinear_evals.into_iter();
	let bitand_evals = evals
		.next()
		.and_then(|evals| evals.try_into().ok())
		.expect("the BitAnd summand reduces to its three columns");
	let operand_evals = evals.flatten().collect();
	let mut eval_point = output.challenges;
	eval_point.reverse();
	RerandOutput {
		eval_point,
		bitand_evals,
		operand_evals,
	}
}

#[cfg(test)]
mod tests {
	use std::{array, iter::repeat_with};

	use binius_compute::GlobalAllocator;
	use binius_field::{Field, Rijndael8b as B8, arch::OptimalPackedB128};
	use binius_ip::channel::Error as ChannelError;
	use binius_math::{
		BinarySubspace, FieldBuffer,
		multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
		test_utils::random_scalars,
		univariate::EvaluationDomain,
	};
	use binius_transcript::{ProverTranscript, VerifierTranscript};
	use binius_verifier::{
		Error as VerifierError,
		config::{B128, StdChallenger},
		protocols::bitand::AndCheckOutput,
		verify_bitand_reduction,
	};
	use rand::prelude::*;

	use super::*;
	use crate::protocols::bitand;

	fn random_words(rng: &mut StdRng, n: usize) -> Vec<Word> {
		repeat_with(|| Word(rng.random())).take(n).collect()
	}

	/// A synthetic multiplication reduction: random word columns, and their honest per-bit claims
	/// at a random point.
	struct Reduction {
		columns: Vec<Vec<Word>>,
		point: Vec<B128>,
		per_bit_claims: Vec<[B128; Word::BITS]>,
	}

	impl Reduction {
		fn random(rng: &mut StdRng, n_vars: usize, n_columns: usize) -> Self {
			let point = random_scalars::<B128>(&mut *rng, n_vars);
			let eq = eq_ind_partial_eval_scalars(&point);
			let columns = repeat_with(|| random_words(rng, 1 << n_vars))
				.take(n_columns)
				.collect::<Vec<_>>();
			// Bit `i` of every word, as a multilinear, evaluated at the point.
			let per_bit_claims = columns
				.iter()
				.map(|words| {
					array::from_fn(|bit| {
						iter::zip(words, &eq)
							.filter(|(word, _)| (word.0 >> bit) & 1 == 1)
							.map(|(_, &eq)| eq)
							.sum()
					})
				})
				.collect();
			Self {
				columns,
				point,
				per_bit_claims,
			}
		}

		fn claims(&self) -> OperandClaims<'_, B128> {
			OperandClaims {
				point: &self.point,
				columns: self.per_bit_claims.iter().collect(),
			}
		}

		fn witness(&self) -> OperandWitness<'_, B128> {
			OperandWitness {
				words: self.columns.iter().map(Vec::as_slice).collect(),
				claims: self.claims(),
			}
		}
	}

	fn andcheck_domain() -> BinarySubspace<B128> {
		BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1).isomorphic()
	}

	/// Proves BitAnd over random columns of `2^log_and` rows, batched with `reductions`.
	fn prove_random(
		rng: &mut StdRng,
		log_and: usize,
		reductions: &[Reduction],
	) -> (AndCheckOutput<B128>, Vec<u8>) {
		let columns = [
			random_words(rng, 1 << log_and),
			random_words(rng, 1 << log_and),
		];
		let operands = reductions
			.iter()
			.map(Reduction::witness)
			.collect::<Vec<_>>();
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let output = bitand::prove::<_, B128, OptimalPackedB128, _, _>(
			columns,
			&operands,
			&mut transcript,
			&GlobalAllocator,
		);
		(output, transcript.finalize())
	}

	fn verify_proof(
		log_and: usize,
		reductions: &[Reduction],
		proof: Vec<u8>,
	) -> Result<AndCheckOutput<B128>, VerifierError> {
		let claims = reductions.iter().map(Reduction::claims).collect::<Vec<_>>();
		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let output =
			verify_bitand_reduction(log_and, &andcheck_domain(), &claims, &mut transcript)?;
		transcript.finalize().expect("no trailing proof data");
		Ok(output)
	}

	#[test]
	fn prover_and_verifier_agree() {
		let mut rng = StdRng::seed_from_u64(0);
		// `(log_and, [log_z per reduction])`: shorter, equal and longer than BitAnd, two reductions
		// of different lengths, no reductions, and an empty BitAnd axis.
		let grid: [(usize, &[usize]); 6] = [
			(5, &[3]),
			(4, &[4]),
			(3, &[5]),
			(4, &[2, 6]),
			(4, &[]),
			(0, &[2]),
		];
		for (log_and, log_zs) in grid {
			// IntMul's and BinMul's column counts, alternating.
			let reductions = iter::zip(log_zs, [4, 6])
				.map(|(&n_vars, n_columns)| Reduction::random(&mut rng, n_vars, n_columns))
				.collect::<Vec<_>>();
			let (output, proof) = prove_random(&mut rng, log_and, &reductions);
			let verified = verify_proof(log_and, &reductions, proof).unwrap();
			assert_eq!(output, verified, "at log_and = {log_and}, log_zs = {log_zs:?}");

			// Each operand eval is its folded column at its point's prefix of the unified point.
			let lagrange = andcheck_domain()
				.reduce_dim(Word::LOG_BITS)
				.lagrange_evals(&output.z_challenge);
			let folder = &BitAxisFolder::new(&lagrange);
			let rerand = output.rerand;
			let expected = reductions
				.iter()
				.flat_map(|reduction| {
					let point = &rerand.eval_point[..reduction.point.len()];
					reduction.columns.iter().map(move |words| {
						let folded: FieldBuffer<B128> = folder.fold(&GlobalAllocator, words);
						evaluate(&folded, point)
					})
				})
				.collect::<Vec<_>>();
			assert_eq!(rerand.operand_evals, expected);
		}
	}

	#[test]
	fn mutated_operand_eval_is_rejected() {
		let mut rng = StdRng::seed_from_u64(1);
		let reductions = [Reduction::random(&mut rng, 5, 4)];
		let (_, mut proof) = prove_random(&mut rng, 4, &reductions);

		// The last operand eval is the proof's final element.
		*proof.last_mut().unwrap() ^= 1;

		let err = verify_proof(4, &reductions, proof).unwrap_err();
		assert!(matches!(err, VerifierError::Channel(ChannelError::InvalidAssert)));
	}

	#[test]
	fn mutated_per_bit_claim_is_rejected() {
		let mut rng = StdRng::seed_from_u64(2);
		let mut reductions = [Reduction::random(&mut rng, 3, 6)];
		let (_, proof) = prove_random(&mut rng, 4, &reductions);

		reductions[0].per_bit_claims[2][17] += B128::ONE;

		let err = verify_proof(4, &reductions, proof).unwrap_err();
		assert!(matches!(err, VerifierError::Channel(ChannelError::InvalidAssert)));
	}
}
