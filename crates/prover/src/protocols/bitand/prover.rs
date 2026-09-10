// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::Deref;

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField, Rijndael8b as B8};
use binius_ip_prover::sumcheck::{common::MleCheckProver, quadratic_mlecheck_prover};
use binius_math::{BinarySubspace, univariate::EvaluationDomain};
use binius_verifier::{
	config::PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES, protocols::bitand::ROWS_PER_HYPERCUBE_VERTEX,
};

use super::sumcheck_round_messages;
use crate::fold_word::BitAxisFolder;

/// Prover for the AND constraint reduction protocol via oblong univariate zerocheck.
///
/// See [`binius_verifier::protocols::bitand`] for the protocol specification.
///
/// The columns are generic over their backing store `Data` (anything that dereferences to
/// `[Word]`), so callers can supply pooled buffers ([`PoolVec`](binius_compute::PoolVec)) or plain
/// `Vec<Word>` interchangeably.
pub struct OblongZerocheckProver<FChallenge, Data>
where
	FChallenge: BinaryField,
{
	log_words: usize,
	first_col: Data,
	second_col: Data,
	big_field_zerocheck_challenges: Vec<FChallenge>,
	univariate_round_message: [FChallenge; ROWS_PER_HYPERCUBE_VERTEX],
	univariate_round_message_domain: BinarySubspace<FChallenge>,
}

impl<F, Data> OblongZerocheckProver<F, Data>
where
	F: BinaryField + From<B8>,
	Data: Deref<Target = [Word]>,
{
	/// Creates a new oblong zerocheck prover for AND constraint reduction.
	///
	/// This constructor sets up the prover by precomputing the univariate polynomial evaluations
	/// that will be sent in the first round. The polynomial encodes the AND constraint verification
	/// across all values in the oblong dimension.
	///
	/// The C operand of the AND constraint `A & B ^ C = 0` is not an input.
	/// The prover derives it word-by-word as `A & B`.
	///
	/// # Why deriving C is sound
	///
	/// - A satisfying witness makes `C = A & B` hold on every row.
	/// - Folding is F2-linear on word bits.
	/// - Equal words therefore fold to equal field elements.
	/// - So an honest prover emits the exact same transcript as with an explicit C column.
	/// - A cheating witness is still rejected.
	/// - The shift reduction later checks the claimed C evaluation against the committed witness.
	///
	/// # Arguments
	///
	/// * `log_words` - Base-2 logarithm of the constraint axis's length
	/// * `first_col` - The oblong multilinear polynomial A in the AND constraint A & B ^ C = 0
	/// * `second_col` - The oblong multilinear polynomial B in the AND constraint
	/// * `big_field_zerocheck_challenges` - Challenges Z_{k+1},...,Zₙ in the large field
	///   `FChallenge`
	/// * `prover_message_domain` - The domain for evaluating the univariate polynomial
	///
	/// The two columns must have equal length, at most `1 << log_words`. A column shorter than the
	/// axis has its remaining rows read as zero, in both the round-1 message and the fold; the
	/// reduction skips them rather than working over them.
	///
	/// # Implementation Details
	///
	/// The constructor:
	/// 1. Computes the equality indicator polynomial from the big field challenges
	/// 2. Uses the NTT lookup to efficiently compute the univariate polynomial evaluations
	/// 3. Caches these evaluations for later use in the [`round_message`](Self::round_message)
	///    method
	pub fn new(
		log_words: usize,
		first_col: Data,
		second_col: Data,
		big_field_zerocheck_challenges: Vec<F>,
		prover_message_domain: &BinarySubspace<B8>,
	) -> Self {
		let univariate_round_message = tracing::debug_span!("Compute univariate round message")
			.in_scope(|| {
				sumcheck_round_messages::univariate_round_message_extension_domain::<F>(
					log_words,
					&first_col,
					&second_col,
					&big_field_zerocheck_challenges,
					prover_message_domain,
				)
			});

		Self {
			log_words,
			first_col,
			second_col,
			univariate_round_message,
			big_field_zerocheck_challenges,
			univariate_round_message_domain: prover_message_domain.isomorphic(),
		}
	}

	/// Executes the first phase of the AND reduction protocol by computing the univariate
	/// polynomial.
	///
	/// This method computes the univariate polynomial R₀(Z) that encodes the AND constraint
	/// verification. The polynomial is evaluated on the extension domain (upper half) and these
	/// evaluations are sent to the verifier as the first round message.
	///
	/// # Returns
	///
	/// Returns a reference to the precomputed univariate polynomial evaluations on the extension
	/// domain. These are exactly `ROWS_PER_HYPERCUBE_VERTEX` field elements that represent
	/// R₀(Z) for Z in the upper half of the univariate domain.
	///
	/// # Note
	///
	/// The polynomial evaluations are precomputed in the constructor using the NTT lookup table
	/// for efficiency. This method simply returns the cached result.
	pub const fn round_message(&self) -> &[F; ROWS_PER_HYPERCUBE_VERTEX] {
		&self.univariate_round_message
	}

	/// The univariate round polynomial at `challenge`: the claim the multilinear rounds prove.
	///
	/// The polynomial is zero on the base half of the domain, and the round message holds its
	/// evaluations on the upper half.
	pub fn univariate_claim(&self, challenge: F) -> F {
		let mut coeffs = vec![F::ZERO; 2 * ROWS_PER_HYPERCUBE_VERTEX];
		coeffs[ROWS_PER_HYPERCUBE_VERTEX..].copy_from_slice(&self.univariate_round_message);
		self.univariate_round_message_domain
			.extrapolate(&coeffs, &challenge)
	}

	/// Folds the oblong multilinears at the univariate challenge and creates the sumcheck prover.
	///
	/// This method performs the transition between Phase 1 (univariate polynomial) and Phase 2
	/// (multilinear sumcheck) of the AND reduction protocol. It folds the oblong multilinear
	/// polynomials by fixing X₀ to the challenge value, effectively reducing them to standard
	/// multilinear polynomials over the remaining variables.
	///
	/// # Arguments
	///
	/// * `round_message_domain` - The domain for the univariate polynomial (same as used in
	///   execute)
	/// * `challenge` - The random challenge z for Z received from the verifier
	///
	/// # Returns
	///
	/// Returns an MLE-check prover configured to prove the sumcheck claim:
	/// R₀(z) = ∑_{X₀,...,Xₙ₋₁ ∈ {0,1}} (A(z,X₀,...,Xₙ₋₁)·B(z,X₀,...,Xₙ₋₁) -
	/// C(z,X₀,...,Xₙ₋₁))·eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁)
	///
	/// # Process
	///
	/// 1. Creates a fold lookup table for efficiently folding at the challenge point
	/// 2. Folds A, B, and the derived C = A & B at Z = challenge, in one fused pass
	/// 3. Combines the zerocheck challenges (small field + big field)
	/// 4. Evaluates the univariate polynomial at the challenge to get the sumcheck claim
	/// 5. Constructs the AND reduction sumcheck prover with the folded multilinears
	pub fn fold_and_send_reduced_prover<
		'alloc,
		PChallenge: PackedField<Scalar = F>,
		A: Allocator,
	>(
		self,
		round_message_domain: &BinarySubspace<F>,
		challenge: F,
		alloc: &'alloc A,
	) -> impl MleCheckProver<F> + 'alloc {
		let claim = self.univariate_claim(challenge);
		let univariate_domain = round_message_domain.reduce_dim(round_message_domain.dim() - 1);
		let lagrange_evals = univariate_domain.lagrange_evals(&challenge);
		let folder = BitAxisFolder::new(&lagrange_evals);

		let proving_polys =
			folder.fold_bitand_operands::<PChallenge, _>(alloc, &self.first_col, &self.second_col);

		let upcasted_small_field_challenges = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES
			.iter()
			.copied()
			.take(self.log_words)
			.map(F::from);

		let verifier_field_zerocheck_challenges = upcasted_small_field_challenges
			.chain(self.big_field_zerocheck_challenges)
			.collect::<Vec<_>>();

		quadratic_mlecheck_prover(
			alloc,
			proving_polys,
			|[a, b, c]| a * b - c,
			|[a, b, _]| a * b,
			verifier_field_zerocheck_challenges,
			claim,
		)
	}
}

#[cfg(test)]
mod test {
	use std::{iter, iter::repeat_with};

	use binius_compute::GlobalAllocator;
	use binius_core::word::Word;
	use binius_field::{Rijndael8b, arch::OptimalPackedB128};
	use binius_math::{
		BinarySubspace, FieldBuffer, multilinear::evaluate::evaluate, univariate::EvaluationDomain,
	};
	use binius_transcript::ProverTranscript;
	use binius_verifier::{
		config::{B128, StdChallenger},
		protocols::bitand::{AndCheckOutput, SKIPPED_VARS},
		verify_bitand_reduction,
	};
	use rand::prelude::*;

	use crate::{fold_word::BitAxisFolder, protocols::bitand::prove};

	fn random_words(log_num_words: usize, mut rng: impl Rng) -> Vec<Word> {
		repeat_with(|| Word(rng.random()))
			.take(1 << log_num_words)
			.collect()
	}

	#[test]
	fn test_transcript_prover_verifies() {
		let mut prover_challenger = ProverTranscript::new(StdChallenger::default());
		let log_num_rows = 6;
		let mut rng = StdRng::seed_from_u64(0);

		let first_mlv = random_words(log_num_rows, &mut rng);
		let second_mlv = random_words(log_num_rows, &mut rng);
		// The prover receives only the A and B columns.
		// This materialized C = A & B feeds only the verifier-side fold check at the end.
		let third_mlv: Vec<Word> = iter::zip(&first_mlv, &second_mlv)
			.map(|(&a, &b)| a & b)
			.collect();

		// Agreed-upon proof parameter
		let prover_message_domain = BinarySubspace::<Rijndael8b>::with_dim(SKIPPED_VARS + 1);
		let verifier_message_domain = prover_message_domain.isomorphic();

		let prove_output = prove::<_, B128, OptimalPackedB128, _, _>(
			[first_mlv.clone(), second_mlv.clone()],
			&[],
			&mut prover_challenger,
			&GlobalAllocator,
		);

		let mut verifier_challenger = prover_challenger.into_verifier();
		let verify_output = verify_bitand_reduction(
			log_num_rows,
			&verifier_message_domain,
			&[],
			&mut verifier_challenger,
		)
		.unwrap();

		assert_eq!(prove_output, verify_output);

		let AndCheckOutput {
			z_challenge,
			rerand,
		} = verify_output;
		let [a_eval, b_eval, c_eval] = rerand.bitand_evals;
		let eval_point = rerand.eval_point;

		let verifier_univariate_domain = verifier_message_domain.reduce_dim(SKIPPED_VARS);

		let one_bit_mlvs = [first_mlv, second_mlv, third_mlv];

		let verifier_lagrange_evals = verifier_univariate_domain.lagrange_evals(&z_challenge);
		let folder = BitAxisFolder::new(&verifier_lagrange_evals);
		for (i, eval) in [a_eval, b_eval, c_eval].iter().enumerate() {
			let folded: FieldBuffer<B128> = folder.fold(&GlobalAllocator, &one_bit_mlvs[i]);
			assert_eq!(evaluate(&folded, &eval_point), *eval);
		}
	}
}
