// Copyright 2025 Irreducible Inc.

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::{AESTowerField8b as B8, BinaryField};
use binius_math::{
	BinarySubspace,
	inner_product::inner_product,
	ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
	univariate::lagrange_evals,
};
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::{
	DeserializeBytes,
	checked_arithmetics::{checked_log_2, log2_ceil_usize},
};
use digest::{Digest, Output, core_api::BlockSizeUser};
use itertools::{Itertools, chain};

use super::error::Error;
use crate::{
	and_reduction::verifier::{AndCheckOutput, verify_with_transcript},
	config::{
		B128, LOG_WORD_SIZE_BITS, LOG_WORDS_PER_ELEM, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES,
	},
	fri::{FRIParams, estimate_optimal_arity},
	hash::PseudoCompressionFunction,
	merkle_tree::BinaryMerkleTreeScheme,
	pcs,
	protocols::{
		intmul::{IntMulOutput, verify as verify_intmul_reduction},
		shift::{self, OperatorData},
	},
};

pub const SECURITY_BITS: usize = 96;

/// Struct for verifying instances of a particular constraint system.
///
/// The [`Self::setup`] constructor determines public parameters for proving instances of the given
/// constraint system. Then [`Self::verify`] is called one or more times with individual instances.
#[derive(Debug, Clone)]
pub struct Verifier<MerkleHash, MerkleCompress> {
	constraint_system: ConstraintSystem,
	fri_params: FRIParams<B128>,
	merkle_scheme: BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress>,
	log_public_words: usize,
}

impl<MerkleHash, MerkleCompress> Verifier<MerkleHash, MerkleCompress>
where
	MerkleHash: Digest + BlockSizeUser,
	MerkleCompress: PseudoCompressionFunction<Output<MerkleHash>, 2>,
	Output<MerkleHash>: DeserializeBytes,
{
	/// Constructs a verifier for a constraint system.
	///
	/// See [`Verifier`] struct documentation for details.
	pub fn setup(
		mut constraint_system: ConstraintSystem,
		log_inv_rate: usize,
		compression: MerkleCompress,
	) -> Result<Self, Error> {
		constraint_system.validate_and_prepare()?;

		// Use offset_witness which is guaranteed to be power of two and be at least one full
		// element.
		let n_public = constraint_system.value_vec_layout.offset_witness;
		let log_public_words = log2_ceil_usize(n_public);
		assert!(n_public.is_power_of_two());
		assert!(log_public_words >= LOG_WORDS_PER_ELEM);

		// The number of field elements that constitute the packed witness.
		let log_witness_words =
			log2_ceil_usize(constraint_system.value_vec_len()).max(LOG_WORDS_PER_ELEM);
		let log_witness_elems = log_witness_words - LOG_WORDS_PER_ELEM;

		let log_code_len = log_witness_elems + log_inv_rate;
		let fri_arity = estimate_optimal_arity(
			log_code_len,
			size_of::<Output<MerkleHash>>(),
			size_of::<B128>(),
		);

		let subspace = BinarySubspace::with_dim(log_code_len)?;
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);
		let fri_params = FRIParams::choose_with_constant_fold_arity(
			&ntt,
			log_witness_elems,
			SECURITY_BITS,
			log_inv_rate,
			fri_arity,
		)?;

		let merkle_scheme = BinaryMerkleTreeScheme::new(compression);

		Ok(Self {
			constraint_system,
			fri_params,
			merkle_scheme,
			log_public_words,
		})
	}

	/// Returns log2 of the number of words in the witness.
	pub fn log_witness_words(&self) -> usize {
		self.log_witness_elems() + LOG_WORDS_PER_ELEM
	}

	/// Returns log2 of the number of field elements in the packed trace.
	pub fn log_witness_elems(&self) -> usize {
		let rs_code = self.fri_params.rs_code();
		rs_code.log_dim() + self.fri_params.log_batch_size()
	}

	/// Returns the constraint system.
	pub fn constraint_system(&self) -> &ConstraintSystem {
		&self.constraint_system
	}

	/// Returns the chosen FRI parameters.
	pub fn fri_params(&self) -> &FRIParams<B128> {
		&self.fri_params
	}

	/// Returns the [`crate::merkle_tree::MerkleTreeScheme`] instance used.
	pub fn merkle_scheme(&self) -> &BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress> {
		&self.merkle_scheme
	}

	/// Returns log2 of the number of public constants and input/output words.
	pub fn log_public_words(&self) -> usize {
		self.log_public_words
	}

	pub fn verify<Challenger_: Challenger>(
		&self,
		public: &[Word],
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		let _verify_guard =
			tracing::info_span!("Verify", operation = "verify", perfetto_category = "operation")
				.entered();

		// Check that the public input length is correct
		if public.len() != 1 << self.log_public_words() {
			return Err(Error::IncorrectPublicInputLength {
				expected: 1 << self.log_public_words(),
				actual: public.len(),
			});
		}

		let subfield_subspace = BinarySubspace::<B8>::default().isomorphic();
		let extended_subspace = subfield_subspace
			.reduce_dim(LOG_WORD_SIZE_BITS + 1)
			.expect("extended_subspace has dimension 8; LOG_WORD_SIZE_BITS + 1 < 8");
		let domain_subspace = extended_subspace
			.reduce_dim(LOG_WORD_SIZE_BITS)
			.expect("extended_subspace has dimension 8; LOG_WORD_SIZE_BITS < 8");

		// Receive the trace commitment.
		let trace_commitment = transcript.message().read::<Output<MerkleHash>>()?;

		// [phase] Verify IntMul Reduction - multiplication constraint verification
		let intmul_guard = tracing::info_span!(
			"[phase] Verify IntMul Reduction",
			phase = "verify_intmul_reduction",
			perfetto_category = "phase",
			n_constraints = self.constraint_system.n_mul_constraints()
		)
		.entered();
		let log_n_constraints = checked_log_2(self.constraint_system.n_mul_constraints());
		let intmul_output =
			verify_intmul_reduction(LOG_WORD_SIZE_BITS, log_n_constraints, transcript)?;
		drop(intmul_guard);

		// [phase] Verify BitAnd Reduction - AND constraint verification
		let bitand_guard = tracing::info_span!(
			"[phase] Verify BitAnd Reduction",
			phase = "verify_bitand_reduction",
			perfetto_category = "phase",
			n_constraints = self.constraint_system.n_and_constraints()
		)
		.entered();
		let bitand_claim = {
			let log_n_constraints = checked_log_2(self.constraint_system.n_and_constraints());
			let AndCheckOutput {
				a_eval,
				b_eval,
				c_eval,
				z_challenge,
				eval_point,
			}: AndCheckOutput<B128> =
				verify_bitand_reduction(log_n_constraints, &extended_subspace, transcript)?;
			OperatorData::new(z_challenge, eval_point, [a_eval, b_eval, c_eval])
		};
		drop(bitand_guard);

		// Build `OperatorData` for IntMul using the same `r_zhat_prime`
		// challenge as in BitAnd. Sharing this univariate challenge
		// improves prover ShiftReduction perf.
		let intmul_claim = {
			let IntMulOutput {
				a_evals,
				b_evals,
				c_lo_evals,
				c_hi_evals,
				eval_point,
			} = intmul_output;

			let r_zhat_prime = bitand_claim.r_zhat_prime;
			let l_tilde = lagrange_evals(&domain_subspace, r_zhat_prime);
			let make_final_claim = |evals| inner_product(evals, l_tilde.iter_scalars());
			OperatorData::new(
				r_zhat_prime,
				eval_point,
				[
					make_final_claim(a_evals),
					make_final_claim(b_evals),
					make_final_claim(c_lo_evals),
					make_final_claim(c_hi_evals),
				],
			)
		};

		// [phase] Verify Shift Reduction - shift operations and constraint validation
		let constraint_guard = tracing::info_span!(
			"[phase] Verify Shift Reduction",
			phase = "verify_shift_reduction",
			perfetto_category = "phase"
		)
		.entered();
		let shift_output = shift::verify(
			self.constraint_system(),
			public,
			&bitand_claim,
			&intmul_claim,
			transcript,
		)?;
		drop(constraint_guard);

		// [phase] Verify Public Input - public input verification
		let public_guard = tracing::info_span!(
			"[phase] Verify Public Input",
			phase = "verify_public_input",
			perfetto_category = "phase"
		)
		.entered();
		shift::check_eval(
			self.constraint_system(),
			&bitand_claim,
			&intmul_claim,
			&domain_subspace,
			&shift_output,
		)?;
		drop(public_guard);

		// [phase] Verify PCS Opening - polynomial commitment verification
		let pcs_guard = tracing::info_span!(
			"[phase] Verify PCS Opening",
			phase = "verify_pcs_opening",
			perfetto_category = "phase"
		)
		.entered();

		let eval_point = [shift_output.r_j(), shift_output.r_y()].concat();
		pcs::verify(
			transcript,
			shift_output.witness_eval(),
			&eval_point,
			trace_commitment,
			&self.fri_params,
			&self.merkle_scheme,
		)?;

		drop(pcs_guard);

		Ok(())
	}
}

fn verify_bitand_reduction<F: BinaryField + From<B8>, Challenger_: Challenger>(
	log_constraint_count: usize,
	eval_domain: &BinarySubspace<F>,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<AndCheckOutput<F>, Error> {
	// The structure of the AND reduction requires that it verifies at least 2^3 word-level
	// constraints, you can zero-pad if necessary to reach this minimum
	assert!(log_constraint_count >= checked_log_2(binius_core::consts::MIN_AND_CONSTRAINTS));

	let big_field_zerocheck_challenges = transcript.sample_vec(log_constraint_count - 3);

	let small_field_zerocheck_challenges = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES
		.into_iter()
		.map(F::from)
		.collect_vec();

	let zerocheck_challenges =
		chain!(small_field_zerocheck_challenges, big_field_zerocheck_challenges)
			.collect::<Vec<_>>();
	verify_with_transcript(&zerocheck_challenges, transcript, eval_domain)
}
