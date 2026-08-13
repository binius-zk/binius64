// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

use binius_core::constraint_system::{ConstraintSystem, InoutSegment};
use binius_field::{ExtensionField, FieldOps};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop::{
	basefold::compiler::BaseFoldVerifierCompiler,
	channel::{IOPVerifierChannel, OracleSpec, oracle_setup::OracleSetupChannel},
	fri::{ConstantArityStrategy, calculate_n_test_queries},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_ip::channel::WordIPVerifierChannel;
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::DeserializeBytes;
use binius_verifier::{
	Error, SECURITY_BITS,
	config::{B1, B128},
	reduction::{Instances, reduce_constraints},
	ring_switch::{self, RingSwitchVerifyOutput},
};
use digest::Output;

use crate::commit::BatchCommitLayout;

/// The Merkle commitment scheme over the committed field, for a given hash suite.
pub(crate) type Scheme<H> = BinaryMerkleTreeScheme<B128, H>;

/// IOP verifier for the M4 constraint reduction of a particular constraint system.
///
/// This struct encapsulates the constraint system and the committed-multilinear shape, providing
/// the core verification logic independent of the specific IOP compilation strategy. Most users
/// should use [`Verifier`] instead, which wraps this with a BaseFold compiler.
///
/// Verification composes the AND-check, the shift reduction, and the ring-switching opening on
/// one transcript, mirroring the prover crate's `IOPProver::prove_chip`:
///
/// 1. The AND-check verifies `A & B == C` over all rows, yielding operand claims at a row point.
/// 2. That point's low coordinates are the instance index `r_rho`, its high coordinates `r_x`.
/// 3. The shift reduction reduces the operand claims to one evaluation of the folded witness.
/// 4. The public-input consistency check ties in the shared constants.
/// 5. The ring-switch opens the committed trace at `r_j || r_rho || r_y`, matching that claim.
///
/// When the circuit has IMUL constraints the IntMul check verifies too.
/// It yields per-bit operand claims at its own instance point, distinct from the AND-check's.
/// A batched multilinear-evaluation sumcheck unifies both onto one shared `r_rho`.
/// Both operand claims at that point then feed the shift.
#[derive(Debug, Clone)]
pub struct IOPVerifier {
	/// The validated single-instance constraint system shared by every instance.
	cs: ConstraintSystem,
	/// The committed-multilinear shape of the batch.
	layout: BatchCommitLayout,
}

impl IOPVerifier {
	/// Constructs an IOP verifier for `2^log_instances` instances of one circuit.
	pub const fn new(cs: ConstraintSystem, layout: BatchCommitLayout) -> Self {
		Self { cs, layout }
	}

	/// The validated constraint system this verifier checks against.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		&self.cs
	}

	/// The committed-multilinear shape this verifier expects.
	pub const fn layout(&self) -> &BatchCommitLayout {
		&self.layout
	}

	/// Consumes the IOP verifier and returns the inner constraint system.
	pub fn into_constraint_system(self) -> ConstraintSystem {
		self.cs
	}

	/// Returns the oracle specs the prover commits to: the trace, plus the IntMul logup*
	/// pushforward when the circuit has IMUL constraints.
	///
	/// The specs are derived by replaying the oracle-receiving sequence against an
	/// [`OracleSetupChannel`] — which records each `recv_oracle` without doing real verification —
	/// rather than hand-maintaining the list. M4 commits its oracles without zero-knowledge, so the
	/// setup channel is constructed with `is_zk = false`.
	pub fn oracle_specs(&self) -> Vec<OracleSpec> {
		let mut channel = OracleSetupChannel::new(false);
		// The setup channel performs no real verification — every `recv_*` / `sample` /
		// `assert_zero` is a no-op — so `verify` cannot fail here; it only records the
		// `recv_oracle` calls read back below. An error would mean that invariant broke, so
		// surface it rather than swallowing it.
		self.verify_chip(&mut channel)
			.expect("verifying against the no-op OracleSetupChannel cannot fail");
		channel.into_oracle_specs()
	}

	/// Verifies one M4 proof using an IOP channel.
	///
	/// This is the core verification logic, independent of the specific IOP compilation strategy.
	/// For most users, [`Verifier::verify_chip`] is the simpler interface.
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
	pub fn verify_chip<Channel>(&self, channel: &mut Channel) -> Result<(), Error>
	where
		Channel: IOPVerifierChannel<B128> + WordIPVerifierChannel<B128>,
		Channel::Elem: FieldOps<Scalar = B128> + From<B128>,
	{
		// Receive the trace commitment.
		// The witness is committed without zero-knowledge.
		let trace_oracle = channel.recv_oracle(self.layout.log_witness_elems, true)?;

		// Reduce every instance's constraints to one claim on the committed trace.
		// A batch hides its inout words, so the public data is the shared constants alone. They are
		// fixed by the constraint system, so they lift into the channel's word type as themselves.
		let constants = self
			.cs
			.constants
			.iter()
			.map(|&word| Channel::Word::from(word))
			.collect::<Vec<_>>();
		let reduction = reduce_constraints(
			&self.cs,
			Instances::Batch {
				log_instances: self.layout.log_instances,
			},
			InoutSegment::Hidden,
			&constants,
			channel,
		)?;

		// Ring-switch the reduced claim onto the committed trace.
		let trace_point = reduction.trace_point();
		let RingSwitchVerifyOutput {
			eq_r_double_prime,
			sumcheck_claim,
		} = ring_switch::verify(reduction.shift.witness_eval, &trace_point, channel)?;

		// Open the trace oracle against the ring-switch's transparent multilinear.
		// BaseFold reduces to a challenge point where the transparent evaluates as below.
		let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
		let eval_point_high = trace_point[log_packing..].to_vec();
		channel.verify_oracle_relation(
			trace_oracle,
			Box::new(move |pt: &[Channel::Elem]| {
				ring_switch::eval_rs_eq(&eval_point_high, pt, &eq_r_double_prime)
			}),
			sumcheck_claim,
		)?;

		Ok(())
	}
}

/// Verifies the data-parallel M4 proof for a batch of `2^log_instances` circuit instances.
///
/// The proof reduces the whole batch to one claim about the committed trace, then opens the trace.
/// One-time setup fixes the constraint system, the committed-oracle shape, and the FRI parameters.
/// A later verification checks one proof against that fixed setup.
///
/// The prover is built from this verifier, so both sides share one set of FRI parameters.
///
/// `H` is the hash suite the Merkle commitments and the transcript channel use, as it is for
/// [`binius_verifier::Verifier`].
pub struct Verifier<H: HashSuite> {
	/// The IOP verifier, holding the constraint system and the committed shape.
	iop_verifier: IOPVerifier,
	/// The precomputed BaseFold verifier, holding the FRI parameters.
	iop_compiler: BaseFoldVerifierCompiler<B128>,
	/// The verifier creates its Merkle transcript channels with the hash suite `H`.
	_hash_marker: PhantomData<H>,
}

impl<H> Verifier<H>
where
	H: HashSuite,
	Output<H::LeafHash>: DeserializeBytes,
{
	/// Builds the verifier for `2^log_instances` instances of one circuit at the given code rate.
	///
	/// # Arguments
	///
	/// - `cs`: the validated single-instance constraint system shared by every instance.
	/// - `log_instances`: base-2 logarithm of the instance count.
	/// - `log_inv_rate`: base-2 logarithm of the inverse Reed-Solomon rate.
	pub fn setup(cs: &ConstraintSystem, log_instances: usize, log_inv_rate: usize) -> Self {
		// The committed shape follows from one instance's length and the instance count.
		let layout = BatchCommitLayout::for_constraint_system(cs, log_instances);
		let iop_verifier = IOPVerifier::new(cs.clone(), layout);

		// The oracle specs the prover commits to — the trace, plus the IntMul logup* pushforward
		// when the circuit has IMUL constraints. Derived by replaying the verifier's
		// oracle-receiving sequence against an `OracleSetupChannel`, so the list can never drift
		// out of sync with the oracles the checks actually commit.
		let oracle_specs = iop_verifier.oracle_specs();

		// Pick the proof-size-optimal FRI fold arity for this codeword length.
		let log_code_len = layout.log_witness_elems + log_inv_rate;
		let merkle_scheme = Scheme::<H>::new();
		let fri_arity =
			ConstantArityStrategy::with_optimal_arity::<B128, _>(&merkle_scheme, log_code_len)
				.arity;

		// The query count is fixed by the rate and the soundness target.
		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, log_inv_rate);

		let iop_compiler = BaseFoldVerifierCompiler::new(
			&merkle_scheme,
			oracle_specs,
			log_inv_rate,
			n_test_queries,
			&ConstantArityStrategy::new(fri_arity),
		);

		Self {
			iop_verifier,
			iop_compiler,
			_hash_marker: PhantomData,
		}
	}

	/// The validated constraint system this verifier checks against.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		self.iop_verifier.constraint_system()
	}

	/// The committed-multilinear shape this verifier expects.
	pub const fn layout(&self) -> &BatchCommitLayout {
		self.iop_verifier.layout()
	}

	/// Returns a reference to the IOP verifier.
	///
	/// The prover clones this to build its matching `IOPProver`.
	pub const fn iop_verifier(&self) -> &IOPVerifier {
		&self.iop_verifier
	}

	/// The precomputed BaseFold verifier compiler.
	///
	/// The prover reuses it so both sides share one set of FRI parameters.
	pub const fn iop_compiler(&self) -> &BaseFoldVerifierCompiler<B128> {
		&self.iop_compiler
	}

	/// Verifies one M4 proof.
	///
	/// Creates the IOP channel from the transcript, delegates to [`IOPVerifier::verify_chip`], then
	/// finishes the channel.
	///
	/// # Errors
	///
	/// Returns an error if the reduction, the ring-switch, or the trace opening fails.
	pub fn verify_chip<Challenger_>(
		&self,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error>
	where
		Challenger_: Challenger,
	{
		let mut channel = self
			.iop_compiler
			.create_channel_from_transcript::<H, Challenger_, _>(transcript);
		self.iop_verifier.verify_chip(&mut channel)?;
		channel.finish()?;

		Ok(())
	}
}
