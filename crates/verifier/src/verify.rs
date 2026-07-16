// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::{AESTowerField8b as B8, BinaryField, ExtensionField, FieldOps};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop::{
	basefold_compiler::BaseFoldVerifierCompiler,
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	oracle_setup_channel::OracleSetupChannel,
};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{
	BinarySubspace, inner_product::inner_product_scalars, univariate::lagrange_evals_scalars,
};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, checked_arithmetics::checked_log_2};
use digest::Output;
use itertools::chain;

use super::error::Error;
use crate::{
	config::{B1, B128, LOG_WORDS_PER_ELEM, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES},
	fri::{ConstantArityStrategy, FRIParams, calculate_n_test_queries},
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
	pub const fn new(constraint_system: ConstraintSystem, log_public_words: usize) -> Self {
		Self {
			constraint_system,
			log_public_words,
		}
	}

	/// Returns the constraint system.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		&self.constraint_system
	}

	/// Consumes the IOP verifier and returns the inner constraint system.
	pub fn into_constraint_system(self) -> ConstraintSystem {
		self.constraint_system
	}

	/// Returns log2 of the number of public constants and input/output words.
	pub const fn log_public_words(&self) -> usize {
		self.log_public_words
	}

	/// Returns log2 of the number of field elements in the packed trace.
	///
	/// The trace oracle commits only the witness's hidden segment, padded to the segment
	/// length; the public segment is a verifier-known polynomial.
	pub const fn log_witness_elems(&self) -> usize {
		let log_witness_words = self.constraint_system.value_vec_layout.log_witness_words();
		log_witness_words - LOG_WORDS_PER_ELEM
	}

	/// Returns log2 of the number of words in the committed trace.
	pub const fn log_witness_words(&self) -> usize {
		self.log_witness_elems() + LOG_WORDS_PER_ELEM
	}

	/// Returns the oracle specs for the IOP channel.
	///
	/// These describe the oracles (the witness) that the prover commits to.
	///
	/// `is_zk` is the protocol-level zero-knowledge flag: in a ZK proof the witness oracle is
	/// masked, in a transparent proof it is not. The flag is taken per call so that a non-ZK
	/// oracle can still participate in a ZK protocol (e.g. indexed relation openings).
	///
	/// The specs are derived by running [`Self::verify`] against an [`OracleSetupChannel`], which
	/// records each oracle received (via `recv_oracle`) without performing any real verification.
	/// This keeps the spec sequence automatically in lockstep with the `recv_oracle` calls in
	/// `verify`, rather than duplicating it here.
	pub fn oracle_specs(&self, is_zk: bool) -> Vec<OracleSpec> {
		let mut channel = OracleSetupChannel::new(is_zk);
		let public = vec![Word::ZERO; 1 << self.log_public_words()];
		// The result is discarded: the setup channel performs no real verification (all `recv_*`
		// return zero, `assert_zero` is a no-op), so we only read back the recorded oracle specs.
		let _ = self.verify(&public, &mut channel);
		channel.into_oracle_specs()
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
		channel.observe_many(&encode_public(public));

		let _verify_guard =
			tracing::info_span!("Verify", operation = "verify", perfetto_category = "operation")
				.entered();

		let subfield_subspace = BinarySubspace::<B8>::default().isomorphic();
		let extended_subspace = subfield_subspace.reduce_dim(Word::LOG_BITS + 1);
		let domain_subspace = extended_subspace.reduce_dim(Word::LOG_BITS);

		// Receive the trace oracle commitment via channel. The trace is the witness, so it is
		// witness-dependent (masked in a ZK proof).
		let trace_oracle = channel.recv_oracle(self.log_witness_elems(), true)?;

		// SOUNDNESS: the IntMul reduction must run *before* the BitAnd reduction. The BitAnd
		// reduction samples the univariate challenge `r_zhat_prime` (the `channel.sample()` in
		// `bitand::verify_with_channel`), and the IntMul per-bit `a`/`b`/`c_lo`/`c_hi` evaluations
		// are collapsed at that point via the Lagrange weights `l_tilde(r_zhat_prime)` below. Those
		// evaluations are bound to the transcript while the IntMul reduction runs, so they must be
		// committed *before* `r_zhat_prime` is drawn; otherwise a malicious prover could choose
		// them adaptively as a function of `r_zhat_prime` and satisfy the collapsed claim without
		// a valid witness. Do not reorder these two reductions, and keep the same order in
		// `prover::prove`.
		//
		// [phase] Verify IntMul Reduction - multiplication constraint verification
		//
		// Skipped (no transcript reads) when the constraint system has no IMUL constraints,
		// mirroring the prover's identical guard so the transcript stays in sync.
		let intmul_output = if self.constraint_system.n_imul_constraints() > 0 {
			let intmul_guard = tracing::info_span!(
				"[phase] Verify IntMul Reduction",
				phase = "verify_intmul_reduction",
				perfetto_category = "phase",
				n_constraints = self.constraint_system.n_imul_constraints()
			)
			.entered();
			let log_n_constraints = checked_log_2(self.constraint_system.n_imul_constraints());
			let intmul_output = verify_intmul_reduction::<B128, _>(log_n_constraints, channel)?;
			drop(intmul_guard);
			Some(intmul_output)
		} else {
			None
		};

		// [phase] Verify BitAnd Reduction - AND constraint verification
		let bitand_guard = tracing::info_span!(
			"[phase] Verify BitAnd Reduction",
			phase = "verify_bitand_reduction",
			perfetto_category = "phase",
			n_constraints = self.constraint_system.n_and_constraints()
		)
		.entered();
		let (r_zhat_prime, bitand_claim) = {
			let log_n_constraints = checked_log_2(self.constraint_system.n_and_constraints());
			let AndCheckOutput {
				a_eval,
				b_eval,
				c_eval,
				z_challenge,
				eval_point,
			} = verify_bitand_reduction(log_n_constraints, &extended_subspace, channel)?;
			(z_challenge, OperatorData::new(eval_point, [a_eval, b_eval, c_eval]))
		};
		drop(bitand_guard);

		// Build `OperatorData` for IntMul. The univariate challenge `r_zhat_prime` is
		// shared with BitAnd (computed above) — sharing it improves prover
		// ShiftReduction perf and lets the verifier compute `h_op_evals` once for both
		// operations in `shift::check_eval`. When IntMul was skipped, synthesize a zero claim
		// (four zero evals at an empty point); it contributes zero to the shift reduction, whose
		// monster evaluation iterates the (empty) IMUL constraints.
		let intmul_claim = match intmul_output {
			Some(IntMulOutput {
				a_evals,
				b_evals,
				c_lo_evals,
				c_hi_evals,
				eval_point,
			}) => {
				let l_tilde = lagrange_evals_scalars(&domain_subspace, r_zhat_prime.clone());
				let make_final_claim =
					|evals| inner_product_scalars(evals, l_tilde.iter().cloned());
				OperatorData::new(
					eval_point,
					[
						make_final_claim(a_evals),
						make_final_claim(b_evals),
						make_final_claim(c_lo_evals),
						make_final_claim(c_hi_evals),
					],
				)
			}
			None => OperatorData::new(Vec::new(), std::array::from_fn(|_| Channel::Elem::zero())),
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
			public,
			&bitand_claim,
			&intmul_claim,
			&domain_subspace,
			r_zhat_prime,
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

		// Ring-switching verification of the witness claim.
		let eval_point = [shift_output.r_j(), shift_output.r_y()].concat();
		let ring_switch::RingSwitchVerifyOutput {
			eq_r_double_prime,
			sumcheck_claim,
		} = ring_switch::verify(shift_output.witness_eval().clone(), &eval_point, channel)?;

		let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
		let eval_point_high = eval_point[log_packing..].to_vec();

		let transparent = Box::new(move |point: &[Channel::Elem]| {
			ring_switch::eval_rs_eq(&eval_point_high, point, eq_r_double_prime.as_ref())
		});

		// Verify oracle relations (runs BaseFold internally and verifies the product check). The
		// intmul pushforward relation, when the IntMul reduction ran, was already queued inside
		// phase 5.
		channel.verify_oracle_relations([OracleLinearRelation {
			oracle: trace_oracle,
			transparent,
			claim: sumcheck_claim,
		}])?;

		drop(pcs_guard);

		Ok(())
	}
}

/// Struct for verifying instances of a particular constraint system.
///
/// The [`Self::setup`] constructor determines public parameters for proving instances of the given
/// constraint system. Then [`Self::verify`] is called one or more times with individual instances.
#[derive(Clone)]
pub struct Verifier<H: HashSuite> {
	iop_verifier: IOPVerifier,
	iop_compiler: BaseFoldVerifierCompiler<B128>,
	/// The verifier creates its Merkle transcript channels with the hash suite `H`.
	_hash_marker: PhantomData<H>,
}

impl<H> Verifier<H>
where
	H: HashSuite,
	Output<H::LeafHash>: DeserializeBytes,
{
	/// Constructs a verifier for a constraint system.
	///
	/// See [`Verifier`] struct documentation for details.
	pub fn setup(
		mut constraint_system: ConstraintSystem,
		log_inv_rate: usize,
	) -> Result<Self, Error> {
		constraint_system.validate_and_prepare()?;

		// The validated layout guarantees a power-of-two public segment of at least one full
		// element.
		let log_public_words = constraint_system.value_vec_layout.log_public_words();
		assert!(log_public_words >= LOG_WORDS_PER_ELEM);

		let iop_verifier = IOPVerifier::new(constraint_system, log_public_words);

		let log_witness_elems = iop_verifier.log_witness_elems();
		// A plain `Verifier` produces a transparent (non-ZK) proof, so the witness oracle is not
		// masked.
		let oracle_specs = iop_verifier.oracle_specs(false);

		let log_code_len = log_witness_elems + log_inv_rate;
		let merkle_scheme = BinaryMerkleTreeScheme::<B128, H>::new();
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
			_hash_marker: PhantomData,
		})
	}

	/// Returns a reference to the IOP verifier.
	pub const fn iop_verifier(&self) -> &IOPVerifier {
		&self.iop_verifier
	}

	/// Consumes the verifier and returns the inner IOP verifier.
	pub fn into_iop_verifier(self) -> IOPVerifier {
		self.iop_verifier
	}

	/// Returns log2 of the number of words in the witness.
	pub const fn log_witness_words(&self) -> usize {
		self.iop_verifier.log_witness_words()
	}

	/// Returns log2 of the number of field elements in the packed trace.
	pub const fn log_witness_elems(&self) -> usize {
		self.iop_verifier.log_witness_elems()
	}

	/// Returns the constraint system.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		self.iop_verifier.constraint_system()
	}

	/// Returns the chosen FRI parameters.
	pub const fn fri_params(&self) -> &FRIParams<B128> {
		self.iop_compiler.fri_params()
	}

	/// Returns log2 of the number of public constants and input/output words.
	pub const fn log_public_words(&self) -> usize {
		self.iop_verifier.log_public_words()
	}

	/// Returns the IOP compiler for creating verifier channels.
	pub const fn iop_compiler(&self) -> &BaseFoldVerifierCompiler<B128> {
		&self.iop_compiler
	}

	pub fn verify<Challenger_: Challenger>(
		&self,
		public: &[Word],
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		let cs = self.iop_verifier.constraint_system();

		let _verify_scope = tracing::info_span!(
			"Verify",
			n_hidden_words = cs.value_vec_layout.n_hidden_words,
			n_bitand = cs.and_constraints.len(),
			n_intmul = cs.imul_constraints.len(),
		)
		.entered();

		// Create channel, delegate to IOPVerifier::verify, then finish it.
		let mut channel = self
			.iop_compiler
			.create_channel_from_transcript::<H, Challenger_, _>(transcript);
		self.iop_verifier.verify(public, &mut channel)?;
		channel.finish()?;
		Ok(())
	}
}

/// Verifies the batched BitAnd check: `A & B == C` on every row.
///
/// This is the univariate-skip zerocheck of `A(Z, X) * B(Z, X) - C(Z, X) == 0` for all rows
/// `(Z, X)`, where `Z` is the bit index within a 64-bit word and `X` is the row index.
///
/// # Arguments
///
/// - `log_constraint_count`: base-2 logarithm of the row count.
/// - `eval_domain`: the univariate-skip domain, one dimension above the 64-bit word, already lifted
///   to `F`. The caller passes it so it matches the shift reduction's domain by construction.
/// - `channel`: the verifier channel that reads messages and redraws Fiat-Shamir challenges.
///
/// # Errors
///
/// Returns an error if any sumcheck round message or the final consistency check fails.
pub fn verify_bitand_reduction<F, C>(
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
	let small_field_zerocheck_challenges = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES
		.into_iter()
		.take(log_constraint_count)
		.map(|b8_val| C::Elem::from(F::from(b8_val)))
		.collect::<Vec<_>>();

	let big_field_zerocheck_challenges =
		channel.sample_many(log_constraint_count - small_field_zerocheck_challenges.len());

	let zerocheck_challenges =
		chain!(small_field_zerocheck_challenges, big_field_zerocheck_challenges)
			.collect::<Vec<_>>();
	verify_with_channel(&zerocheck_challenges, channel, eval_domain)
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
