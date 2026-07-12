// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::constraint_system::ConstraintSystem;
use binius_field::ExtensionField;
use binius_hash::StdHashSuite;
use binius_iop::{
	basefold_compiler::BaseFoldVerifierCompiler,
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	fri::{ConstantArityStrategy, calculate_n_test_queries},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_verifier::{
	Error,
	config::{B1, B128},
	ring_switch::{self, RingSwitchVerifyOutput},
};

use crate::{
	commit::BatchCommitLayout,
	reduction::{ReductionVerifierOutput, verify_reduction},
};

/// The target soundness, in bits.
///
/// This matches the Binius64 verifier's target.
/// It only sets the FRI query count.
const SECURITY_BITS: usize = 96;

/// The Merkle commitment scheme over the committed field.
type Scheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

/// Verifies the data-parallel M4 proof for a batch of `2^log_instances` circuit instances.
///
/// The proof reduces the whole batch to one claim about the committed trace, then opens the trace.
/// One-time setup fixes the constraint system, the committed-oracle shape, and the FRI parameters.
/// A later verification checks one proof against that fixed setup.
///
/// The prover is built from this verifier, so both sides share one set of FRI parameters.
pub struct Verifier {
	/// The prepared single-instance constraint system shared by every instance.
	cs: ConstraintSystem,
	/// The committed-multilinear shape of the batch.
	layout: BatchCommitLayout,
	/// The precomputed BaseFold verifier, holding the FRI parameters.
	iop_compiler: BaseFoldVerifierCompiler<B128>,
}

impl Verifier {
	/// Builds the verifier for `2^log_instances` instances of one circuit at the given code rate.
	///
	/// # Arguments
	///
	/// - `cs`: the prepared single-instance constraint system shared by every instance.
	/// - `log_instances`: base-2 logarithm of the instance count.
	/// - `log_inv_rate`: base-2 logarithm of the inverse Reed-Solomon rate.
	pub fn setup(cs: &ConstraintSystem, log_instances: usize, log_inv_rate: usize) -> Self {
		// The committed shape follows from one instance's length and the instance count.
		let layout = BatchCommitLayout::for_constraint_system(cs, log_instances);

		// One oracle: the packed batch witness, committed without zero-knowledge.
		// ZK-ness is a higher-level choice that M4 does not make here.
		let oracle_specs = vec![OracleSpec::new(layout.log_witness_elems)];

		// Pick the proof-size-optimal FRI fold arity for this codeword length.
		let log_code_len = layout.log_witness_elems + log_inv_rate;
		let merkle_scheme = Scheme::new();
		let fri_arity =
			ConstantArityStrategy::with_optimal_arity::<B128, _>(&merkle_scheme, log_code_len)
				.arity;

		// The query count is fixed by the rate and the soundness target.
		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, log_inv_rate);

		let iop_compiler = BaseFoldVerifierCompiler::new(
			merkle_scheme,
			oracle_specs,
			log_inv_rate,
			n_test_queries,
			&ConstantArityStrategy::new(fri_arity),
		);

		Self {
			cs: cs.clone(),
			layout,
			iop_compiler,
		}
	}

	/// The prepared constraint system this verifier checks against.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		&self.cs
	}

	/// The committed-multilinear shape this verifier expects.
	pub const fn layout(&self) -> &BatchCommitLayout {
		&self.layout
	}

	/// The precomputed BaseFold verifier compiler.
	///
	/// The prover reuses it so both sides share one set of FRI parameters.
	pub const fn iop_compiler(&self) -> &BaseFoldVerifierCompiler<B128> {
		&self.iop_compiler
	}

	/// Verifies one M4 proof.
	///
	/// The steps mirror the prover on the same transcript:
	/// - Receive the trace commitment.
	/// - Replay the AND-check and shift reduction to one folded-witness claim.
	/// - Ring-switch that claim onto the committed trace.
	/// - Check the trace opening.
	///
	/// The reduction ends with a claim about the witness folded over instances at `r_rho`.
	/// The trace's bit index is `[bit | instance | wire]`.
	/// So evaluating its instance coordinates at `r_rho` performs that fold.
	/// The ring-switch therefore opens the trace at `r_j || r_rho || r_y`.
	/// That evaluation equals the folded-witness claim the reduction produced.
	///
	/// # Errors
	///
	/// Returns an error if the reduction, the ring-switch, or the trace opening fails.
	pub fn verify<Challenger_>(
		&self,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error>
	where
		Challenger_: Challenger,
	{
		let mut channel = self
			.iop_compiler
			.create_channel_from_transcript::<StdHashSuite, Challenger_, _>(transcript);

		// Receive the trace commitment.
		// The witness is committed without zero-knowledge.
		let trace_oracle = channel.recv_oracle(self.layout.log_witness_elems, true)?;

		// Replay the AND-check and shift reduction to a single folded-witness claim.
		let ReductionVerifierOutput { r_rho, shift } =
			verify_reduction(&self.cs, self.layout.log_instances, &mut channel)?;

		// Ring-switch the reduced claim onto the committed trace.
		// The point is `r_j || r_rho || r_y`.
		// Its instance coordinates fold the trace at `r_rho`.
		let trace_point = [shift.r_j(), r_rho.as_slice(), shift.r_y()].concat();
		let RingSwitchVerifyOutput {
			eq_r_double_prime,
			sumcheck_claim,
		} = ring_switch::verify(shift.witness_eval, &trace_point, &mut channel)?;

		// Open the trace oracle against the ring-switch's transparent multilinear.
		// BaseFold reduces to a challenge point where the transparent evaluates as below.
		let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
		let eval_point_high = trace_point[log_packing..].to_vec();
		channel.verify_oracle_relations([OracleLinearRelation {
			oracle: trace_oracle,
			transparent: Box::new(move |pt: &[B128]| {
				ring_switch::eval_rs_eq(&eval_point_high, pt, &eq_r_double_prime)
			}),
			claim: sumcheck_claim,
		}])?;
		channel.finish()?;

		Ok(())
	}
}
