// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::{AESTowerField8b as B8, ExtensionField, FieldOps};
use binius_hash::StdHashSuite;
use binius_iop::{
	basefold_compiler::BaseFoldVerifierCompiler,
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	fri::{ConstantArityStrategy, calculate_n_test_queries},
	merkle_tree::BinaryMerkleTreeScheme,
	oracle_setup_channel::OracleSetupChannel,
};
use binius_ip::{
	channel::IPVerifierChannel,
	sumcheck::{BatchSumcheckOutput, batch_verify},
};
use binius_math::{
	BinarySubspace,
	inner_product::inner_product_scalars,
	multilinear::eq::eq_ind,
	univariate::{evaluate_univariate, lagrange_evals_scalars},
};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::checked_arithmetics::checked_log_2;
use binius_verifier::{
	Error,
	config::{B1, B128},
	protocols::{
		bitand::AndCheckOutput,
		intmul::{IntMulOutput, verify as verify_intmul_reduction},
		shift::{self, BITAND_ARITY, INTMUL_ARITY, OperatorData},
	},
	ring_switch::{self, RingSwitchVerifyOutput},
	verify_bitand_reduction,
};

use crate::commit::BatchCommitLayout;

/// The target soundness, in bits.
///
/// This matches the Binius64 verifier's target.
/// It only sets the FRI query count.
const SECURITY_BITS: usize = 96;

/// The Merkle commitment scheme over the committed field.
type Scheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

/// IOP verifier for the M4 constraint reduction of a particular constraint system.
///
/// This struct encapsulates the constraint system and the committed-multilinear shape, providing
/// the core verification logic independent of the specific IOP compilation strategy. Most users
/// should use [`Verifier`] instead, which wraps this with a BaseFold compiler.
///
/// Verification composes the AND-check, the shift reduction, and the ring-switching opening on
/// one transcript, mirroring the prover crate's `IOPProver::prove`:
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
	/// The prepared single-instance constraint system shared by every instance.
	cs: ConstraintSystem,
	/// The committed-multilinear shape of the batch.
	layout: BatchCommitLayout,
}

impl IOPVerifier {
	/// Constructs an IOP verifier for `2^log_instances` instances of one circuit.
	pub const fn new(cs: ConstraintSystem, layout: BatchCommitLayout) -> Self {
		Self { cs, layout }
	}

	/// The prepared constraint system this verifier checks against.
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
		self.verify(&mut channel)
			.expect("verifying against the no-op OracleSetupChannel cannot fail");
		channel.into_oracle_specs()
	}

	/// Verifies one M4 proof using an IOP channel.
	///
	/// This is the core verification logic, independent of the specific IOP compilation strategy.
	/// For most users, [`Verifier::verify`] is the simpler interface.
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
	pub fn verify<Channel>(&self, channel: &mut Channel) -> Result<(), Error>
	where
		Channel: IOPVerifierChannel<B128>,
		Channel::Elem: FieldOps<Scalar = B128> + From<B128>,
	{
		let cs = &self.cs;
		let log_instances = self.layout.log_instances;

		// Receive the trace commitment.
		// The witness is committed without zero-knowledge.
		let trace_oracle = channel.recv_oracle(self.layout.log_witness_elems, true)?;

		// One base domain shared by the AND-check, the shift, and the IntMul operand collapse.
		// The AND-check's univariate-skip domain spans one dimension above the 64-bit word.
		// `verify_bitand_reduction` expects the domain already lifted to the channel's field.
		let subfield_subspace = BinarySubspace::<B8>::default().isomorphic::<B128>();
		let andcheck_domain = subfield_subspace.reduce_dim(Word::LOG_BITS + 1);
		// The shift domain drops that extra dimension.
		let shift_domain = andcheck_domain.reduce_dim(Word::LOG_BITS);

		// SOUNDNESS: the IntMul check verifies before the BitAnd check, mirroring the prover.
		// Its per-bit operand evaluations are read from the transcript here.
		// BitAnd then draws the univariate challenge that collapses them.
		// Reading them first stops a malicious prover choosing them as a function of that
		// challenge. Do not reorder these, and keep the same order in `IOPProver::prove`.
		//
		// The IntMul columns span every instance's constraints.
		// So the check runs over `log_instances + log_n_imul` row variables.
		let intmul_output = if cs.n_imul_constraints() > 0 {
			let log_n_imul = checked_log_2(cs.n_imul_constraints());
			Some(verify_intmul_reduction::<B128, _>(log_instances + log_n_imul, channel)?)
		} else {
			None
		};

		// AND-check over all `K * n_and` rows.
		let log_n_and = checked_log_2(cs.and_constraints.len());
		let AndCheckOutput {
			a_eval,
			b_eval,
			c_eval,
			z_challenge,
			eval_point,
		} = verify_bitand_reduction(log_instances + log_n_and, &andcheck_domain, channel)?;

		// The AND-check row point is `r_rho_and || r_x_and`: the instance index low, the constraint
		// index high.
		let (r_rho_and, r_x_and) = eval_point.split_at(log_instances);

		// Reduce to one shared instance point and both operand claims at it.
		let (r_rho, bitand, intmul) = match intmul_output {
			Some(intmul_output) => {
				// Both operations enter the re-randomization as operand claims at their own
				// instance point. BitAnd is already oblong; IntMul is collapsed from its
				// per-bit form.
				let lagrange = lagrange_evals_scalars::<B128, Channel::Elem>(
					&shift_domain,
					z_challenge.clone(),
				);
				RerandomizedOperations {
					bitand: OperationClaim::new([a_eval, b_eval, c_eval], r_x_and, r_rho_and),
					intmul: OperationClaim::from_intmul(intmul_output, &lagrange, log_instances),
				}
				.verify(channel)?
			}
			// No IMUL constraints: the AND-check instance point is used directly.
			// The IntMul claim is a zero claim at an empty point.
			None => (
				r_rho_and.to_vec(),
				OperatorData::new(r_x_and.to_vec(), [a_eval, b_eval, c_eval]),
				OperatorData::new(Vec::new(), std::array::from_fn(|_| Channel::Elem::zero())),
			),
		};

		// Reduce the operand claims to one witness evaluation.
		let shift = shift::verify::<B128, _>(cs, &bitand, &intmul, channel)?;

		// Tie in the shared constants through the public-input consistency check.
		// The shift evaluates them over the layout's power-of-two word count.
		// Their count need not be a power of two, so they are passed unpadded.
		shift::check_eval::<B128, _>(
			cs,
			&cs.constants,
			&bitand,
			&intmul,
			&shift_domain,
			z_challenge,
			&shift,
			channel,
		)?;

		// Ring-switch the reduced claim onto the committed trace.
		// The point is `r_j || r_rho || r_y`.
		// Its instance coordinates fold the trace at `r_rho`.
		let trace_point = [shift.r_j(), r_rho.as_slice(), shift.r_y()].concat();
		let RingSwitchVerifyOutput {
			eq_r_double_prime,
			sumcheck_claim,
		} = ring_switch::verify(shift.witness_eval, &trace_point, channel)?;

		// Open the trace oracle against the ring-switch's transparent multilinear.
		// BaseFold reduces to a challenge point where the transparent evaluates as below.
		let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
		let eval_point_high = trace_point[log_packing..].to_vec();
		channel.verify_oracle_relations([OracleLinearRelation {
			oracle: trace_oracle,
			transparent: Box::new(move |pt: &[Channel::Elem]| {
				ring_switch::eval_rs_eq(&eval_point_high, pt, &eq_r_double_prime)
			}),
			claim: sumcheck_claim,
		}])?;

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
pub struct Verifier {
	/// The IOP verifier, holding the constraint system and the committed shape.
	iop_verifier: IOPVerifier,
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
		let iop_verifier = IOPVerifier::new(cs.clone(), layout);

		// The oracle specs the prover commits to — the trace, plus the IntMul logup* pushforward
		// when the circuit has IMUL constraints. Derived by replaying the verifier's
		// oracle-receiving sequence against an `OracleSetupChannel`, so the list can never drift
		// out of sync with the oracles the checks actually commit.
		let oracle_specs = iop_verifier.oracle_specs();

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
			iop_verifier,
			iop_compiler,
		}
	}

	/// The prepared constraint system this verifier checks against.
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
	/// Creates the IOP channel from the transcript, delegates to [`IOPVerifier::verify`], then
	/// finishes the channel.
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
		self.iop_verifier.verify(&mut channel)?;
		channel.finish()?;

		Ok(())
	}
}

/// The degree of the re-randomization's round polynomials.
///
/// Each operand is a multilinear evaluation, expressed with the quadratic evaluator.
/// Its degree-2 prime polynomial gains one degree from the equality factor, giving 3.
// TODO: a degree-1 multilinear-eval store evaluator would drop this to 2; none exists yet.
const RERAND_DEGREE: usize = 3;

/// The shared instance point together with both operations' operand data at that point.
type RerandOutput<F> = (Vec<F>, OperatorData<F, BITAND_ARITY>, OperatorData<F, INTMUL_ARITY>);

/// One operation's oblong operand claims and the points they are claimed at.
///
/// The AND-check and the IntMul check both reduce to this shape.
/// The re-randomization transports the claims to the instance point shared by both operations.
///
/// Generic over the channel's element type `F`, so this composes with a channel whose challenges
/// live in an extension of the base field (e.g. a symbolic verifier channel), not only `B128`.
struct OperationClaim<F, const ARITY: usize> {
	/// The oblong operand claim per operand, in operand order.
	operand_claims: [F; ARITY],
	/// The constraint-index point the operands are claimed at.
	r_x: Vec<F>,
	/// The instance-index point the operands are claimed at.
	r_rho: Vec<F>,
}

impl<F: FieldOps, const ARITY: usize> OperationClaim<F, ARITY> {
	/// The operand claims at the constraint point `r_x` and instance point `r_rho`.
	fn new(operand_claims: [F; ARITY], r_x: &[F], r_rho: &[F]) -> Self {
		Self {
			operand_claims,
			r_x: r_x.to_vec(),
			r_rho: r_rho.to_vec(),
		}
	}
}

impl<F: FieldOps> OperationClaim<F, INTMUL_ARITY> {
	/// Builds the IntMul claim by collapsing its per-bit operand claims to oblong claims.
	///
	/// The Lagrange weights fold the per-bit claims at the univariate challenge.
	/// This gives the oblong form the BitAnd claims already have.
	/// The IntMul row point splits into an instance part (low) and a constraint part (high).
	fn from_intmul(intmul_output: IntMulOutput<F>, lagrange: &[F], log_instances: usize) -> Self {
		let IntMulOutput {
			eval_point: r_out_mul,
			a_evals,
			b_evals,
			c_lo_evals,
			c_hi_evals,
		} = intmul_output;
		let oblong = |evals: Vec<F>| inner_product_scalars(evals, lagrange.iter().cloned());
		let (r_rho, r_x) = r_out_mul.split_at(log_instances);
		Self::new(
			[
				oblong(a_evals),
				oblong(b_evals),
				oblong(c_lo_evals),
				oblong(c_hi_evals),
			],
			r_x,
			r_rho,
		)
	}
}

/// The two operations' claims entering the batched instance re-randomization.
struct RerandomizedOperations<F> {
	/// The BitAnd operand claims at the AND-check instance point.
	bitand: OperationClaim<F, BITAND_ARITY>,
	/// The IntMul operand claims at the IntMul instance point.
	intmul: OperationClaim<F, INTMUL_ARITY>,
}

impl<F: FieldOps> RerandomizedOperations<F> {
	/// Verifies the batched sumcheck that unifies the two operations' instance points.
	///
	/// - Check the sumcheck transporting every operand claim onto one shared instance point.
	/// - Read the reduced operand evaluations at that point.
	/// - Bind them to the sumcheck.
	///
	/// # Returns
	///
	/// The shared instance point, the BitAnd operand data, and the IntMul operand data.
	fn verify<Channel>(self, channel: &mut Channel) -> Result<RerandOutput<F>, Error>
	where
		Channel: IPVerifierChannel<B128, Elem = F>,
	{
		// Both operations reduce over the same instance axis; recover its width from either point.
		let log_instances = self.bitand.r_rho.len();

		// Verify the batched sumcheck: one multilinear-eval claim per operand, ordered
		// [BitAnd a, b, c | IntMul a, b, lo, hi].
		let sums: Vec<F> = self
			.bitand
			.operand_claims
			.into_iter()
			.chain(self.intmul.operand_claims)
			.collect();
		let BatchSumcheckOutput {
			batch_coeff,
			eval,
			mut challenges,
		} = batch_verify(log_instances, RERAND_DEGREE, &sums, channel)?;
		challenges.reverse();
		let r_rho = challenges;

		// The prover wrote the reduced operand evaluations at `r_rho`, grouped by operation.
		// These are the operand claims the shift consumes.
		let bitand_evals = channel.recv_array::<BITAND_ARITY>()?;
		let intmul_evals = channel.recv_array::<INTMUL_ARITY>()?;

		// Bind the reduced evals to the sumcheck: each claim's contribution is its
		// eq(instance_point, r_rho) weight times its reduced eval, batched by `batch_coeff`.
		let eq_and = eq_ind(&self.bitand.r_rho, &r_rho);
		let eq_mul = eq_ind(&self.intmul.r_rho, &r_rho);
		let expected: Vec<F> = bitand_evals
			.clone()
			.map(|eval| eval * &eq_and)
			.into_iter()
			.chain(intmul_evals.clone().map(|eval| eval * &eq_mul))
			.collect();
		channel.assert_zero(evaluate_univariate(&expected, batch_coeff) - eval)?;

		let bitand_data = OperatorData::new(self.bitand.r_x, bitand_evals);
		let intmul_data = OperatorData::new(self.intmul.r_x, intmul_evals);
		Ok((r_rho, bitand_data, intmul_data))
	}
}
