// Copyright 2025 Irreducible Inc.

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::{AESTowerField8b as B8, BinaryField, ExtensionField, FieldOps};
use binius_iop::{
	basefold_compiler::BaseFoldVerifierCompiler,
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{
	BinarySubspace,
	inner_product::inner_product_scalars,
	multilinear::{eq::eq_ind, evaluate::evaluate_inplace_scalars},
	univariate::lagrange_evals_scalars,
};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{
	DeserializeBytes,
	checked_arithmetics::{checked_log_2, log2_ceil_usize},
};
use digest::{Digest, Output, core_api::BlockSizeUser};
use itertools::{Itertools, chain};

use super::error::Error;
use crate::{
	config::{
		B1, B128, LOG_WORD_SIZE_BITS, LOG_WORDS_PER_ELEM, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES,
	},
	fri::{ConstantArityStrategy, FRIParams, calculate_n_test_queries},
	hash::PseudoCompressionFunction,
	merkle_tree::BinaryMerkleTreeScheme,
	protocols::{
		bitand::{AndCheckOutput, verify_with_channel},
		intmul::{IntMulOutput, verify as verify_intmul_reduction},
		shift::{self, OperatorData},
	},
	ring_switch,
};

pub const SECURITY_BITS: usize = 96;

/// IOP verifier for a particular constraint system.
///
/// This struct encapsulates the constraint system, providing the core verification logic
/// independent of the specific IOP compilation strategy. Most users should use [`Verifier`]
/// instead, which wraps this with a BaseFold compiler.
#[derive(Debug, Clone)]
pub struct IOPVerifier {
	constraint_system: ConstraintSystem,
	log_public_words: usize,
}

impl IOPVerifier {
	/// Constructs an IOP verifier for a constraint system.
	///
	/// The constraint system must already be validated via
	/// [`ConstraintSystem::validate_and_prepare`].
	pub fn new(constraint_system: ConstraintSystem, log_public_words: usize) -> Self {
		Self {
			constraint_system,
			log_public_words,
		}
	}

	/// Returns the constraint system.
	pub fn constraint_system(&self) -> &ConstraintSystem {
		&self.constraint_system
	}

	/// Consumes the IOP verifier and returns the inner constraint system.
	pub fn into_constraint_system(self) -> ConstraintSystem {
		self.constraint_system
	}

	/// Returns log2 of the number of public constants and input/output words.
	pub fn log_public_words(&self) -> usize {
		self.log_public_words
	}

	/// Returns log2 of the number of field elements in the packed trace.
	pub fn log_witness_elems(&self) -> usize {
		let log_witness_words =
			log2_ceil_usize(self.constraint_system.value_vec_len()).max(LOG_WORDS_PER_ELEM);
		log_witness_words - LOG_WORDS_PER_ELEM
	}

	/// Returns log2 of the number of words in the witness.
	pub fn log_witness_words(&self) -> usize {
		self.log_witness_elems() + LOG_WORDS_PER_ELEM
	}

	/// Returns the oracle specs for the IOP channel.
	///
	/// These describe the oracles (the witness) that the prover commits to.
	pub fn oracle_specs(&self) -> Vec<OracleSpec> {
		vec![OracleSpec {
			log_msg_len: self.log_witness_elems(),
		}]
	}

	/// Verifies a proof using an IOP channel.
	///
	/// This is the core verification logic, independent of the specific IOP compilation strategy.
	/// For most users, [`Verifier::verify`] is the simpler interface.
	pub fn verify<Channel>(&self, public: &[Word], channel: &mut Channel) -> Result<(), Error>
	where
		Channel: IOPVerifierChannel<B128>,
		Channel::Elem: FieldOps<Scalar = B128> + From<B128>,
	{
		// Check that the public input length is correct
		if public.len() != 1 << self.log_public_words() {
			return Err(Error::IncorrectPublicInputLength {
				expected: 1 << self.log_public_words(),
				actual: public.len(),
			});
		}

		// Verifier observes the public input (includes it in Fiat-Shamir).
		let public_elems = channel.observe_many(&encode_public(public));

		let _verify_guard =
			tracing::info_span!("Verify", operation = "verify", perfetto_category = "operation")
				.entered();

		let subfield_subspace = BinarySubspace::<B8>::default().isomorphic();
		let extended_subspace = subfield_subspace.reduce_dim(LOG_WORD_SIZE_BITS + 1);
		let domain_subspace = extended_subspace.reduce_dim(LOG_WORD_SIZE_BITS);

		// Receive the trace oracle commitment via channel.
		let trace_oracle = channel.recv_oracle()?;

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
			verify_intmul_reduction::<B128, _>(LOG_WORD_SIZE_BITS, log_n_constraints, channel)?;
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
			} = verify_bitand_reduction(log_n_constraints, &extended_subspace, channel)?;
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

			let r_zhat_prime = bitand_claim.r_zhat_prime.clone();
			let l_tilde = lagrange_evals_scalars(&domain_subspace, r_zhat_prime.clone());
			let make_final_claim = |evals| inner_product_scalars(evals, l_tilde.iter().cloned());
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
		let shift_output =
			shift::verify(self.constraint_system(), &bitand_claim, &intmul_claim, channel)?;
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
			channel,
		)?;
		drop(public_guard);

		// [phase] Ring-Switching + Verify PCS Opening
		let pcs_guard = tracing::info_span!(
			"[phase] Verify PCS Opening",
			phase = "verify_pcs_opening",
			perfetto_category = "phase"
		)
		.entered();

		// Ring-switching verification
		let eval_point = [shift_output.r_j(), shift_output.r_y()].concat();
		let ring_switch::RingSwitchVerifyOutput {
			eq_r_double_prime,
			sumcheck_claim,
		} = ring_switch::verify(shift_output.witness_eval().clone(), &eval_point, channel)?;

		// Public input check batched with ring-switch
		let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
		let eval_point_high = eval_point[log_packing..].to_vec();

		let log_public_elems = self.log_public_words() - LOG_WORDS_PER_ELEM;
		let pubcheck_point = eval_point_high[..log_public_elems].to_vec();
		let pubcheck_claim = evaluate_inplace_scalars(public_elems, &pubcheck_point);

		let batch_coeff = channel.sample();
		let batched_claim = sumcheck_claim + batch_coeff.clone() * pubcheck_claim;

		// Build the transparent closure combining ring-switch and public input check
		let transparent = Box::new(move |point: &[Channel::Elem]| {
			let rs_eq_eval =
				ring_switch::eval_rs_eq(&eval_point_high, point, eq_r_double_prime.as_ref());
			let pubcheck_eq_eval = eval_pubcheck_eq(&pubcheck_point, point);
			rs_eq_eval + batch_coeff.clone() * pubcheck_eq_eval
		});

		// Verify oracle relations (runs BaseFold internally and verifies the product check)
		channel.verify_oracle_relations([OracleLinearRelation {
			oracle: trace_oracle,
			transparent,
			claim: batched_claim,
		}])?;

		drop(pcs_guard);

		Ok(())
	}
}

/// Struct for verifying instances of a particular constraint system.
///
/// The [`Self::setup`] constructor determines public parameters for proving instances of the given
/// constraint system. Then [`Self::verify`] is called one or more times with individual instances.
#[derive(Debug, Clone)]
pub struct Verifier<MerkleHash, MerkleCompress>
where
	MerkleHash: Digest + BlockSizeUser,
	MerkleCompress: PseudoCompressionFunction<Output<MerkleHash>, 2>,
{
	iop_verifier: IOPVerifier,
	iop_compiler:
		BaseFoldVerifierCompiler<B128, BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress>>,
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

		let iop_verifier = IOPVerifier::new(constraint_system, log_public_words);

		let log_witness_elems = iop_verifier.log_witness_elems();
		let oracle_specs = iop_verifier.oracle_specs();

		let log_code_len = log_witness_elems + log_inv_rate;
		let merkle_scheme = BinaryMerkleTreeScheme::new(compression);
		let fri_arity =
			ConstantArityStrategy::with_optimal_arity::<B128, _>(&merkle_scheme, log_code_len)
				.arity;

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, log_inv_rate);

		let iop_compiler = BaseFoldVerifierCompiler::new(
			merkle_scheme,
			oracle_specs,
			log_inv_rate,
			n_test_queries,
			&ConstantArityStrategy::new(fri_arity),
		);

		Ok(Self {
			iop_verifier,
			iop_compiler,
		})
	}

	/// Returns a reference to the IOP verifier.
	pub fn iop_verifier(&self) -> &IOPVerifier {
		&self.iop_verifier
	}

	/// Consumes the verifier and returns the inner IOP verifier.
	pub fn into_iop_verifier(self) -> IOPVerifier {
		self.iop_verifier
	}

	/// Returns log2 of the number of words in the witness.
	pub fn log_witness_words(&self) -> usize {
		self.iop_verifier.log_witness_words()
	}

	/// Returns log2 of the number of field elements in the packed trace.
	pub fn log_witness_elems(&self) -> usize {
		self.iop_verifier.log_witness_elems()
	}

	/// Returns the constraint system.
	pub fn constraint_system(&self) -> &ConstraintSystem {
		self.iop_verifier.constraint_system()
	}

	/// Returns the chosen FRI parameters.
	pub fn fri_params(&self) -> &FRIParams<B128> {
		// There is exactly one oracle spec (the witness)
		&self.iop_compiler.fri_params()[0]
	}

	/// Returns the [`crate::merkle_tree::MerkleTreeScheme`] instance used.
	pub fn merkle_scheme(&self) -> &BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress> {
		self.iop_compiler.merkle_scheme()
	}

	/// Returns log2 of the number of public constants and input/output words.
	pub fn log_public_words(&self) -> usize {
		self.iop_verifier.log_public_words()
	}

	/// Returns the IOP compiler for creating verifier channels.
	pub fn iop_compiler(
		&self,
	) -> &BaseFoldVerifierCompiler<B128, BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress>>
	{
		&self.iop_compiler
	}

	pub fn verify<Challenger_: Challenger>(
		&self,
		public: &[Word],
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		// Create channel and delegate to IOPVerifier::verify
		let mut channel = self.iop_compiler.create_channel(transcript);
		self.iop_verifier.verify(public, &mut channel)
	}
}

fn verify_bitand_reduction<F, C>(
	log_constraint_count: usize,
	eval_domain: &BinarySubspace<F>,
	channel: &mut C,
) -> Result<AndCheckOutput<C::Elem>, Error>
where
	F: BinaryField + From<B8>,
	C: IPVerifierChannel<F>,
	// Used to make deterministic basis challenges symbolic
	C::Elem: From<F>,
{
	// The structure of the AND reduction requires that it verifies at least 2^3 word-level
	// constraints, you can zero-pad if necessary to reach this minimum
	assert!(log_constraint_count >= checked_log_2(binius_core::consts::MIN_AND_CONSTRAINTS));

	let big_field_zerocheck_challenges = channel.sample_many(log_constraint_count - 3);

	let small_field_zerocheck_challenges = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES
		.into_iter()
		.map(|b8_val| C::Elem::from(F::from(b8_val)))
		.collect_vec();

	let zerocheck_challenges =
		chain!(small_field_zerocheck_challenges, big_field_zerocheck_challenges)
			.collect::<Vec<_>>();
	verify_with_channel(&zerocheck_challenges, channel, eval_domain)
}

/// Evaluate the public input equality indicator at a query point.
///
/// Computes `eq(pubcheck_point || 0, query)`, which selects the first `2^k` entries of the
/// committed polynomial (the public inputs). In characteristic 2:
/// `eq(a || 0, x) = eq(a, x[..k]) * prod_{i >= k} (1 - x_i)`
fn eval_pubcheck_eq<F: FieldOps>(pubcheck_point: &[F], query: &[F]) -> F {
	let one = F::one();
	let (query_prefix, query_suffix) = query.split_at(pubcheck_point.len());
	let prefix_eq = eq_ind(pubcheck_point, query_prefix);
	let suffix_prod = query_suffix
		.iter()
		.fold(one.clone(), |acc, x| acc * (one.clone() - x));
	prefix_eq * suffix_prod
}

/// Encode public input words as B128 elements, for compliance with the IOP interface.
fn encode_public(public: &[Word]) -> Vec<B128> {
	let (word_pairs, remaining) = public.as_chunks::<2>();
	assert!(
		remaining.is_empty(),
		"ValueVecLayout ensures the public section has a multiple of two number of words"
	);
	word_pairs
		.iter()
		.map(|[w0, w1]| B128::new(((w1.as_u64() as u128) << 64) | w0.as_u64() as u128))
		.collect()
}
