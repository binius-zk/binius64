// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::{AESTowerField8b as B8, Field, PackedField};
use binius_hash::StdHashSuite;
use binius_iop_prover::{basefold_compiler::BaseFoldProverCompiler, channel::IOPProverChannel};
use binius_ip_prover::sumcheck::{
	MleToSumCheckEvaluator,
	batch::batch_prove_and_write_evals,
	mle_store::MleStore,
	quadratic_mle_evaluator::QuadraticMleEvaluator,
	round_evaluator::{RoundEvaluator, SharedSumcheckProver},
};
use binius_m4_verifier::{IOPVerifier, Verifier};
use binius_math::{
	BinarySubspace, FieldBuffer,
	inner_product::inner_product,
	multilinear::eq::eq_ind_partial_eval_scalars,
	ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
	univariate::lagrange_evals_scalars,
};
use binius_prover::{
	fold_word::fold_words,
	protocols::{
		intmul::{prove::IntMulProver, witness::Witness as IntMulWitness},
		shift::{KeyCollection, OperatorData, build_key_collection},
	},
	ring_switch::{self, RingSwitchOutput},
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::{checked_arithmetics::checked_log_2, rayon::prelude::*};
use binius_verifier::{
	config::B128,
	protocols::{
		bitand::AndCheckOutput,
		intmul::IntMulOutput,
		shift::{BITAND_ARITY, INTMUL_ARITY},
	},
};

use crate::{
	BatchAndCheckWitness, ValueTable,
	operand_witness::build_intmul_witness,
	shift::{fold_instances, prove as prove_shift},
};

/// The multithreaded additive NTT used to encode the committed codeword.
type ProverNtt = NeighborsLastMultiThread<GenericPreExpanded<B128>>;

/// IOP prover for the M4 constraint reduction of a particular constraint system.
///
/// This struct encapsulates the constraint system and the pre-computed shift keys, providing the
/// core proving logic independent of the specific IOP compilation strategy. Most users should use
/// [`Prover`] instead, which wraps this with a BaseFold compiler.
///
/// Proving composes the commitment, the reduction, and the ring-switching opening on one
/// transcript, mirroring [`IOPVerifier::verify`](binius_m4_verifier::IOPVerifier::verify):
/// - Pack the table into one B128 multilinear and commit it as the trace oracle.
/// - Run the AND-check and shift reduction to a claim about the instance-folded witness.
/// - Ring-switch that claim onto the committed trace and open it.
///
/// The trace commits before the reduction draws its challenges.
/// So Fiat-Shamir binds every challenge to the committed data.
///
/// The reduction ends with a claim about the witness folded over instances at `r_rho`.
/// The trace's bit index is `[bit | instance | wire]`.
/// Evaluating its instance coordinates at `r_rho` performs that fold.
/// So the ring-switch opens the trace at `r_j || r_rho || r_y`, matching the reduced claim.
///
/// The trace oracle is not ZK, so the channel masks nothing and needs no randomness.
///
/// With IMUL constraints the reduction commits one further oracle: the IntMul logup* pushforward.
/// The IntMul check queues that oracle's opening itself.
/// The final combined FRI opening covers it alongside the trace, so it needs no handling here.
pub struct IOPProver {
	/// The prepared single-instance constraint system shared by every instance.
	cs: ConstraintSystem,
	/// The shift keys for the constraint system, built once and reused across proofs.
	key_collection: KeyCollection,
}

impl IOPProver {
	/// Constructs an IOP prover from an IOP verifier and pre-computed shift keys.
	pub fn new(iop_verifier: IOPVerifier, key_collection: KeyCollection) -> Self {
		Self {
			cs: iop_verifier.into_constraint_system(),
			key_collection,
		}
	}

	/// Returns the constraint system.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		&self.cs
	}

	/// Returns a reference to the KeyCollection.
	pub const fn key_collection(&self) -> &KeyCollection {
		&self.key_collection
	}

	/// Proves that every instance in the batch satisfies the constraint system, using an IOP
	/// channel.
	///
	/// This is the core proving logic, independent of the specific IOP compilation strategy. For
	/// most users, [`Prover::prove`] is the simpler interface.
	pub fn prove<P, Channel>(&self, table: &ValueTable, channel: &mut Channel)
	where
		P: PackedField<Scalar = B128>,
		Channel: IOPProverChannel<P>,
	{
		let cs = &self.cs;

		// Pack the 2-D table into one multilinear and commit it as the trace oracle.
		let trace_packed = {
			let _scope = tracing::debug_span!("Prepare trace").entered();
			table.pack::<P>()
		};
		let trace_oracle = {
			let _scope = tracing::debug_span!("Commit trace").entered();
			channel.send_oracle(trace_packed.to_ref())
		};

		// One base domain shared by the AND-check and the shift, consistent by construction.
		// The AND-check's univariate-skip domain spans one dimension above the 64-bit word.
		let andcheck_domain = BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1);
		// The shift domain drops that extra dimension.
		// This is exactly the domain the AND-check folds its bit axis over.
		// So the operand claims rebuilt below at the shared univariate challenge match the
		// AND-check's.
		let shift_domain = andcheck_domain
			.reduce_dim(Word::LOG_BITS)
			.isomorphic::<B128>();

		// Build the IntMul operand columns and run the IntMul check, only when the circuit has IMUL
		// constraints.
		//
		// SOUNDNESS: the IntMul check runs before the BitAnd check below.
		// Its per-bit operand evaluations are bound to the transcript here.
		// BitAnd then draws the univariate challenge that collapses them.
		// Committing them first stops a malicious prover choosing them as a function of that
		// challenge. Do not reorder these, and keep the same order in `IOPVerifier::verify`.
		//
		// The columns are the four operands of every constraint over every instance, laid out
		// constraint-major.
		// They are kept alongside the check output.
		// The re-randomization re-reads them to build the instance-axis multilinears it transports.
		let mul = (!cs.imul_constraints.is_empty()).then(|| {
			let columns = {
				let _scope = tracing::debug_span!("Assemble IntMul witness").entered();
				// `build_intmul_witness` yields `[A, B, HI, LO]`; reorder to the `[A, B, LO, HI]`
				// operand order the IntMul witness and the shift both expect.
				let [a, b, hi, lo] =
					build_intmul_witness(table, &cs.constants, &cs.imul_constraints);
				[a, b, lo, hi]
			};
			let output = prove_intmul::<P, _>(&columns, channel);
			(columns, output)
		});

		// AND-check the `A & B == C` relation over all `K * n_and` rows.
		// Retain the operand columns when IntMul ran, since the re-randomization re-reads them.
		let (
			and_columns,
			AndCheckOutput {
				a_eval,
				b_eval,
				c_eval,
				z_challenge,
				eval_point,
			},
		) = {
			let _scope = tracing::debug_span!("BitAnd check").entered();

			let and_witness = {
				let _scope = tracing::debug_span!("Assemble BitAnd witness").entered();
				BatchAndCheckWitness::build(table, &cs.constants, &cs.and_constraints)
			};
			let and_columns = mul.is_some().then(|| {
				[
					and_witness.a().to_vec(),
					and_witness.b().to_vec(),
					and_witness.c().to_vec(),
				]
			});
			(and_columns, and_witness.prove::<P, _>(&andcheck_domain, channel))
		};

		// The AND-check row point is `r_rho_and || r_x_and`: the instance index on the low
		// coordinates, the constraint index on the high coordinates.
		let (r_rho_and, r_x_and) = eval_point.split_at(table.log_instances());

		// Reduce to one shared instance point `r_rho` and the operand claims at that point.
		let (r_rho, bitand_data, intmul_data) = match mul {
			Some((mul_columns, intmul_output)) => {
				// Both operations enter the re-randomization as operand columns with their oblong
				// claims at their own instance point.
				// BitAnd is already oblong.
				// IntMul is collapsed from its per-bit form.
				let lagrange = lagrange_evals_scalars::<B128, B128>(&shift_domain, z_challenge);
				let and_columns = and_columns
					.expect("AND columns are retained whenever there are IMUL constraints");
				RerandomizedOperations {
					bitand: Operation::new(
						&and_columns,
						[a_eval, b_eval, c_eval],
						r_x_and,
						r_rho_and,
					),
					intmul: Operation::from_intmul(
						&mul_columns,
						intmul_output,
						&lagrange,
						table.log_instances(),
					),
				}
				.prove::<P, _>(&lagrange, z_challenge, channel)
			}
			// No IMUL constraints: the AND-check instance point is used directly.
			// The IntMul claim is a zero claim at an empty point, contributing nothing to the
			// shift.
			None => (
				r_rho_and.to_vec(),
				OperatorData {
					evals: vec![a_eval, b_eval, c_eval],
					r_zhat_prime: z_challenge,
					r_x_prime: r_x_and.to_vec(),
				},
				OperatorData {
					evals: vec![B128::ZERO; 4],
					r_zhat_prime: z_challenge,
					r_x_prime: Vec::new(),
				},
			),
		};

		// Fold the committed witness over the instance axis at the shared point.
		let folded_witness = {
			let _scope = tracing::debug_span!("Fold instances").entered();
			fold_instances::<B128>(table, &r_rho)
		};

		// The public segment is the shared constants, padded with zeros to the layout's
		// power-of-two word count.
		// The shift folds it against the monster's public part, which is sized to that padded
		// count. The padding makes the two lengths agree, and matches the zeros the verifier
		// assumes.
		let mut public_words = cs.constants.clone();
		public_words.resize(cs.value_vec_layout.n_public_words(), Word::ZERO);

		// Reduce the operand claims to one witness evaluation.
		let witness_claim = {
			let _scope = tracing::debug_span!("Prove shift reduction").entered();
			prove_shift::<B128, P, _>(
				&self.key_collection,
				&public_words,
				&folded_witness,
				bitand_data,
				intmul_data,
				&shift_domain,
				channel,
			)
		};

		let RingSwitchOutput {
			rs_eq_ind,
			sumcheck_claim,
		} = {
			let _scope = tracing::debug_span!("Ring-switching reduction").entered();

			// Split the shift's final point `r_j || r_y || r_segment` into its three parts.
			// The bit index `r_j` is the low coordinates addressing a bit within a 64-bit word.
			// The segment selector `r_segment` is the last coordinate, choosing public or hidden
			// words. The hidden-only trace drops it.
			// The word index `r_y` is everything in between.
			let challenges = &witness_claim.challenges;
			let r_j = &challenges[..Word::LOG_BITS];
			let r_y = &challenges[Word::LOG_BITS..challenges.len() - 1];

			// Ring-switch the reduced claim onto the committed trace.
			// The point is `r_j || r_rho || r_y`.
			// Its instance coordinates fold the trace at `r_rho`.
			let trace_point = [r_j, r_rho.as_slice(), r_y].concat();
			ring_switch::prove(&trace_packed, &trace_point, channel)
		};

		// Queue the trace opening against the ring-switch's transparent multilinear.
		// The final call runs the single combined FRI opening and writes it to the transcript.
		channel.prove_oracle_relations([(trace_oracle, trace_packed, rs_eq_ind, sumcheck_claim)]);
	}
}

/// Proves the data-parallel M4 statement for a batch of `2^log_instances` circuit instances.
///
/// One-time setup builds the shift keys and the BaseFold prover, reusing the verifier's parameters.
/// A later proving call commits a witness table and proves it satisfies every AND constraint.
pub struct Prover<P>
where
	P: PackedField<Scalar = B128>,
{
	iop_prover: IOPProver,
	/// The precomputed BaseFold prover, holding the NTT and the FRI parameters.
	basefold_compiler: BaseFoldProverCompiler<P, ProverNtt>,
}

impl<P> Prover<P>
where
	P: PackedField<Scalar = B128>,
{
	/// Builds the prover from a verifier, inheriting its constraint system and FRI parameters.
	///
	/// The prover encodes the codeword with the multithreaded NTT, spread across the cores.
	/// Reusing the verifier's compiler keeps both sides on one set of FRI parameters.
	pub fn setup(verifier: &Verifier) -> Self {
		// Reuse the verifier's evaluation domain so both sides agree on the code.
		let domain_context =
			GenericPreExpanded::generate_from_subspace(verifier.iop_compiler().max_subspace());

		// Spread the NTT across the available cores.
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		// Inherit the verifier's oracle specs and FRI parameters verbatim.
		let basefold_compiler =
			BaseFoldProverCompiler::from_verifier_compiler(verifier.iop_compiler(), ntt);

		// Build the shift keys once from the shared constraint system.
		let key_collection = build_key_collection(verifier.constraint_system());

		let iop_prover = IOPProver::new(verifier.iop_verifier().clone(), key_collection);

		Self {
			iop_prover,
			basefold_compiler,
		}
	}

	/// Returns a reference to the IOP prover.
	pub const fn iop_prover(&self) -> &IOPProver {
		&self.iop_prover
	}

	/// Proves that every instance in the batch satisfies the constraint system.
	///
	/// Creates the IOP channel from the transcript, delegates to [`IOPProver::prove`], then
	/// finishes the channel with the combined FRI opening.
	pub fn prove<Challenger_>(
		&self,
		table: &ValueTable,
		transcript: &mut ProverTranscript<Challenger_>,
	) where
		Challenger_: Challenger,
	{
		let mut channel = self
			.basefold_compiler
			.create_channel_without_zk_from_transcript::<StdHashSuite, Challenger_, _>(transcript);

		self.iop_prover.prove::<P, _>(table, &mut channel);

		let _scope = tracing::debug_span!("PCS opening").entered();
		channel.finish();
	}
}

/// Runs the IntMul check over the batched operand columns.
///
/// The four columns are the multiplicands and the low and high product words, in the order
/// `[A, B, LO, HI]`.
/// Each is `K * n_imul` rows, laid out constraint-major.
/// The check reduces the multiplication relation to per-bit evaluation claims on the four columns.
/// Those claims share a common row point.
fn prove_intmul<P, Channel>(columns: &[Vec<Word>; 4], channel: &mut Channel) -> IntMulOutput<B128>
where
	P: PackedField<Scalar = B128>,
	Channel: IOPProverChannel<P>,
{
	let _scope = tracing::debug_span!("IntMul check").entered();

	// The columns are the multiplicand `a`, the multiplicand `b`, and the product's low and high
	// words, in the order the IntMul witness expects.
	let [a, b, lo, hi] = columns;

	// A prepared constraint system pads its constraint and instance counts to powers of two, so the
	// columns always have a power-of-two length as the witness requires.
	let witness = IntMulWitness::<P>::new(a, b, lo, hi)
		.expect("a prepared constraint system yields power-of-two operand columns");

	// Switchover 0 keeps the exponentiation trees in the small field for every round.
	let mut prover = IntMulProver::<P, _>::new(0, channel);
	prover.prove(witness)
}

/// One operation's operand columns, oblong claims, and the points they are claimed at.
///
/// The AND-check and the IntMul check both reduce to this shape.
/// The re-randomization folds each column into its instance-axis multilinear.
/// It then transports the claims to the instance point shared by both operations.
struct Operation<'a, const ARITY: usize> {
	/// The operand columns, constraint-major, one per operand.
	columns: &'a [Vec<Word>; ARITY],
	/// The oblong operand claim per operand: its multilinear-eval claim at the instance point.
	operand_claims: [B128; ARITY],
	/// The constraint-index point the operands are claimed at.
	r_x: Vec<B128>,
	/// The instance-index point the operands are claimed at.
	r_rho: Vec<B128>,
}

impl<'a, const ARITY: usize> Operation<'a, ARITY> {
	/// The operand columns with their claims at the constraint point `r_x` and instance point
	/// `r_rho`.
	fn new(
		columns: &'a [Vec<Word>; ARITY],
		operand_claims: [B128; ARITY],
		r_x: &[B128],
		r_rho: &[B128],
	) -> Self {
		Self {
			columns,
			operand_claims,
			r_x: r_x.to_vec(),
			r_rho: r_rho.to_vec(),
		}
	}

	/// Adds this operation's operand evaluators to a shared store.
	///
	/// The operation's instance-point indicator is registered once, and every operand reads it.
	/// So the store expands that indicator once, not once per operand.
	/// Each operand's instance-axis multilinear becomes a store column.
	/// Its evaluator is an identity-composition quadratic MLE-check: a multilinear evaluation.
	fn push_to<P>(
		&self,
		lagrange: &[B128],
		store: &mut MleStore<P>,
		evaluators: &mut Vec<Box<dyn RoundEvaluator<B128, P>>>,
	) where
		P: PackedField<Scalar = B128>,
	{
		let eq_tracker = store.register_eq_tracker(&self.r_rho);
		// The constraint tensor is the same for every operand of this operation, so expand it once.
		let r_x_tensor = eq_ind_partial_eval_scalars::<B128>(&self.r_x);
		for (column, &claim) in self.columns.iter().zip(&self.operand_claims) {
			let col = store.push_owned(operand_rho_multilinear::<P>(column, lagrange, &r_x_tensor));
			let evaluator = QuadraticMleEvaluator::new(
				[col],
				eq_tracker,
				|[operand]: [P; 1]| operand,
				|[_operand]: [P; 1]| P::zero(),
				claim,
			);
			evaluators.push(Box::new(MleToSumCheckEvaluator::new(evaluator, eq_tracker)));
		}
	}
}

impl<'a> Operation<'a, INTMUL_ARITY> {
	/// Builds the IntMul operation by collapsing its per-bit operand claims to oblong claims.
	///
	/// The Lagrange weights fold the per-bit claims at the univariate challenge.
	/// This gives the oblong form the BitAnd claims already have.
	/// The IntMul row point splits into an instance part (low) and a constraint part (high).
	fn from_intmul(
		columns: &'a [Vec<Word>; INTMUL_ARITY],
		intmul_output: IntMulOutput<B128>,
		lagrange: &[B128],
		log_instances: usize,
	) -> Self {
		let IntMulOutput {
			eval_point: r_out_mul,
			a_evals,
			b_evals,
			c_lo_evals,
			c_hi_evals,
		} = intmul_output;
		let oblong =
			|evals: &[B128]| inner_product(evals.iter().copied(), lagrange.iter().copied());
		let (r_rho, r_x) = r_out_mul.split_at(log_instances);
		Self::new(
			columns,
			[
				oblong(&a_evals),
				oblong(&b_evals),
				oblong(&c_lo_evals),
				oblong(&c_hi_evals),
			],
			r_x,
			r_rho,
		)
	}
}

/// The two operations entering the batched instance re-randomization.
struct RerandomizedOperations<'a> {
	/// The BitAnd operation, at the AND-check instance point.
	bitand: Operation<'a, BITAND_ARITY>,
	/// The IntMul operation, at the IntMul instance point.
	intmul: Operation<'a, INTMUL_ARITY>,
}

impl RerandomizedOperations<'_> {
	/// Re-randomizes both operations' instance points to one shared point.
	///
	/// Each operation reduces to operand claims at its own instance point.
	/// The witness folds over the instance axis only once, so the points must be unified first.
	///
	/// - Push both operations' operand multilinears onto one store.
	/// - Register one equality tracker per operation, so each indicator is expanded once.
	/// - A batched sumcheck transports every claim to one shared instance point.
	/// - The reduced evaluations there are the operand claims the shift consumes.
	///
	/// # Returns
	///
	/// The shared instance point, the BitAnd operand data, and the IntMul operand data.
	fn prove<P, Channel>(
		self,
		lagrange: &[B128],
		z_challenge: B128,
		channel: &mut Channel,
	) -> (Vec<B128>, OperatorData<B128>, OperatorData<B128>)
	where
		P: PackedField<Scalar = B128>,
		Channel: IOPProverChannel<P>,
	{
		let _scope = tracing::debug_span!("Re-randomize instances").entered();

		// Both operations reduce over the same instance axis.
		// Recover its width from either point.
		let log_instances = self.bitand.r_rho.len();

		// One shared store over the instance axis holds every operation's operand multilinears.
		// The evaluators list the operands in order [BitAnd a, b, c | IntMul a, b, lo, hi].
		// The verifier reads the reduced evaluations back in the same order.
		let mut store = MleStore::<P>::new(log_instances);
		let mut evaluators: Vec<Box<dyn RoundEvaluator<B128, P>>> =
			Vec::with_capacity(BITAND_ARITY + INTMUL_ARITY);
		self.bitand.push_to(lagrange, &mut store, &mut evaluators);
		self.intmul.push_to(lagrange, &mut store, &mut evaluators);

		// One shared prover drives all claims over the store in a single round pass.
		// Its evaluations are the store's per-column values at the shared instance point, in push
		// order.
		let shared = SharedSumcheckProver::new(store, evaluators);
		let output = batch_prove_and_write_evals(vec![shared], channel);
		let reduced = &output.multilinear_evals[0];

		// The reduced evaluations split back into the two operations at the operand-count boundary.
		let split = self.bitand.operand_claims.len();
		let bitand_data = OperatorData {
			evals: reduced[..split].to_vec(),
			r_zhat_prime: z_challenge,
			r_x_prime: self.bitand.r_x,
		};
		let intmul_data = OperatorData {
			evals: reduced[split..].to_vec(),
			r_zhat_prime: z_challenge,
			r_x_prime: self.intmul.r_x,
		};
		(output.challenges, bitand_data, intmul_data)
	}
}

/// Builds the instance-axis multilinear of one operand column, folded over its bit and constraint
/// axes.
///
/// The operand column is constraint-major: `row = local_constraint * n_instances + instance`.
/// Folding collapses the two other axes and leaves one field element per instance:
///
/// ```text
/// M[rho] = sum_{local, j} lagrange[j] * r_x_tensor[local] * bit_j(column[local * K + rho])
/// ```
///
/// - The Lagrange weights fold each 64-bit word over its bit axis at the shared univariate
///   challenge.
/// - The constraint tensor `r_x_tensor` folds the constraint axis; the caller expands it once per
///   operation and shares it across the operands.
///
/// Its evaluation at the operation's instance point equals that operation's oblong operand claim.
/// So the re-randomization sumcheck can transport that claim to a shared instance point.
fn operand_rho_multilinear<P>(
	column: &[Word],
	lagrange: &[B128],
	r_x_tensor: &[B128],
) -> FieldBuffer<P>
where
	P: PackedField<Scalar = B128>,
{
	// Fold each word's bits at the univariate challenge: one scalar per row, laid out
	// constraint-major.
	// Folding into scalars keeps the row indexing flat for the constraint fold.
	let folded_rows = fold_words::<B128, B128>(column, lagrange);
	let folded_rows = folded_rows.as_ref();

	// Produce the packed instance-axis multilinear directly, one packed element per parallel task.
	// Each element's lanes are the constraint folds of consecutive instances.
	// Lanes past the instance count are the multilinear's zero padding.
	//
	// The constraint axis is the high, strided axis: constraint `local` of instance `rho` sits at
	// row `local * n_instances + rho`.
	let n_constraints = r_x_tensor.len();
	let n_instances = folded_rows.len() / n_constraints;
	let log_instances = checked_log_2(n_instances);
	let log_packed = log_instances.saturating_sub(P::LOG_WIDTH);
	let packed = (0..1usize << log_packed)
		.into_par_iter()
		.map(|packed_index| {
			P::from_scalars((0..P::WIDTH).map(|lane| {
				let instance = (packed_index << P::LOG_WIDTH) | lane;
				if instance < n_instances {
					r_x_tensor
						.iter()
						.enumerate()
						.map(|(local, &weight)| {
							weight * folded_rows[local * n_instances + instance]
						})
						.sum()
				} else {
					B128::ZERO
				}
			}))
		})
		.collect::<Vec<P>>();

	FieldBuffer::new(log_instances, packed.into_boxed_slice())
}

#[cfg(test)]
mod tests {
	use std::array;

	use assert_matches::assert_matches;
	use binius_field::PackedBinaryGhash1x128b;
	use binius_frontend::CircuitBuilder;
	use binius_iop::{
		basefold::{Error as BaseFoldError, VerificationError as BaseFoldVerificationError},
		channel::Error as IOPChannelError,
		fri::VerificationError as FriVerificationError,
		merkle_tree::VerificationError as MerkleVerificationError,
	};
	use binius_transcript::VerifierTranscript;
	use binius_verifier::{Error, config::StdChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{N_INPUT_WORDS, crc64_circuit, populate_crc64_witness};

	type P = PackedBinaryGhash1x128b;

	// Builds a batch of `2^log_instances` CRC-64 instances with random input words.
	fn setup_batch(log_instances: usize, seed: u64) -> (ConstraintSystem, ValueTable) {
		let c = crc64_circuit();
		let n_instances = 1usize << log_instances;
		let mut rng = StdRng::seed_from_u64(seed);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		(cs, table)
	}

	// The prover and verifier run the whole protocol on one transcript.
	// A faithful proof over 64 instances verifies and leaves no trailing data.
	#[test]
	fn protocol_round_trips() {
		let log_instances = 6;
		let (cs, table) = setup_batch(log_instances, 0);

		// Setup once: the verifier fixes the shape and FRI parameters.
		// The prover inherits them.
		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		// Prover: commit, reduce, and open on a fresh transcript.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);

		// Verifier: replay the same transcript end to end.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// A circuit carrying IMUL constraints round-trips through the whole protocol.
	//
	// With IMUL constraints the proof commits two oracles rather than one:
	//
	//     trace oracle    : the packed batch witness
	//     logup* oracle   : the IntMul check's pushforward
	//
	// The IntMul and AND checks reduce to different instance points, which the re-randomization
	// unifies before the witness is folded.
	//
	// Fixture: one unsigned 64x64 -> 128 product per instance over 2^6 instances, both product
	// words force-committed. The `imul` gate emits one IMUL constraint and one AND security check.
	//
	// A faithful proof verifies, both oracles open, and no trailing data is left.
	#[test]
	fn protocol_round_trips_with_mul() {
		// One product per instance, with both result words committed as hidden words.
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let y = builder.add_witness();
		let (hi, lo) = builder.imul(x, y);
		builder.force_commit(hi);
		builder.force_commit(lo);
		let circuit = builder.build();

		let mut cs = circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		// Confirm the fixture genuinely exercises the IntMul path.
		assert!(!cs.imul_constraints.is_empty(), "the fixture must emit an IMUL constraint");

		// Fill each instance's two multiplicands from a per-instance seed; the circuit derives the
		// two product words.
		let log_instances = 6;
		let table = ValueTable::populate(&circuit, log_instances, |i, w| {
			let mut rng = StdRng::seed_from_u64(i as u64);
			w[x] = Word(rng.next_u64());
			w[y] = Word(rng.next_u64());
		})
		.unwrap();

		// Setup once: the verifier fixes the shape and FRI parameters, the prover inherits them.
		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		// Prover: commit both oracles, reduce, and open on a fresh transcript.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);

		// Verifier: replay the same transcript end to end.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// A circuit whose constant count is not a power of two proves and verifies: the shift evaluates
	// the constants over the layout's power-of-two word count, treating the words past the constant
	// count as zero, so no caller padding is needed.
	//
	// A single BLAKE3 compression per instance is a real circuit with a non-power-of-two constant
	// count, so it exercises exactly that padding path.
	#[test]
	fn protocol_round_trips_with_non_power_of_two_constants() {
		use binius_circuits::blake3::blake3_compress;
		use binius_frontend::Wire;

		let builder = CircuitBuilder::new();
		let cv: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let block: [Wire; 16] = array::from_fn(|_| builder.add_witness());
		let counter = builder.add_witness();
		let block_len = builder.add_witness();
		let flags = builder.add_witness();
		// Force-commit the output so the circuit has no inout wires.
		let out = blake3_compress(&builder, cv, block, counter, block_len, flags);
		for wire in out {
			builder.force_commit(wire);
		}
		let circuit = builder.build();

		let mut cs = circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		// Confirm the fixture is genuine: the constant count is not a power of two.
		assert!(!cs.constants.len().is_power_of_two());

		// Fill each instance's inputs from a per-instance seed; the compression derives the rest.
		let log_instances = 6;
		let table = ValueTable::populate(&circuit, log_instances, |i, w| {
			let mut rng = StdRng::seed_from_u64(i as u64);
			// A 32-bit value per chaining-value word.
			for wire in cv {
				w[wire] = Word(rng.next_u32() as u64);
			}
			// A 32-bit value per message word.
			for wire in block {
				w[wire] = Word(rng.next_u32() as u64);
			}
			// A full 64-bit block counter.
			w[counter] = Word(rng.next_u64());
			// A byte length in 0..=64.
			w[block_len] = Word((rng.next_u32() % 65) as u64);
			// Arbitrary domain-separation flags.
			w[flags] = Word(rng.next_u32() as u64);
		})
		.unwrap();

		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// Independent AND constraints alongside IMUL constraints, so the two operations reduce to
	// constraint points of different lengths (`log_n_and != log_n_imul`) and to genuinely different
	// instance points that the re-randomization must unify.
	//
	// Proving with a width-2 packing exercises the packed lane layout and zero-padding of the
	// instance-axis multilinears, which the width-1 fixtures never reach.
	#[test]
	fn protocol_round_trips_with_mixed_constraints_and_wide_packing() {
		use binius_field::PackedBinaryGhash2x128b;
		use binius_frontend::Wire;

		type WideP = PackedBinaryGhash2x128b;

		let builder = CircuitBuilder::new();
		let inputs: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		// Four standalone AND gates on distinct wires — the AND work is not tied to the products.
		for pair in inputs.chunks_exact(2) {
			let and = builder.band(pair[0], pair[1]);
			builder.force_commit(and);
		}
		// Two products — fewer IMUL constraints than AND constraints.
		for pair in inputs.chunks_exact(2).take(2) {
			let (hi, lo) = builder.imul(pair[0], pair[1]);
			builder.force_commit(hi);
			builder.force_commit(lo);
		}
		let circuit = builder.build();

		let mut cs = circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		// Confirm the fixture genuinely exercises the asymmetric case: the two operations have
		// different constraint-point lengths.
		let log_n_and = checked_log_2(cs.and_constraints.len());
		let log_n_imul = checked_log_2(cs.imul_constraints.len());
		assert_ne!(
			log_n_and, log_n_imul,
			"the fixture must give the operations different r_x lengths"
		);

		let log_instances = 6;
		let table = ValueTable::populate(&circuit, log_instances, |i, w| {
			let mut rng = StdRng::seed_from_u64(i as u64);
			for &wire in &inputs {
				w[wire] = Word(rng.next_u64());
			}
		})
		.unwrap();

		// Prove with the wide packing; the verifier is packing-agnostic.
		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<WideP>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// Tampering with the trace opening breaks the final FRI check.
	#[test]
	fn tampered_opening_is_rejected() {
		let log_instances = 6;
		let (cs, table) = setup_batch(log_instances, 1);

		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		// Produce a faithful proof, then collect its bytes.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Flip one bit in the last byte, which lands in a FRI query's Merkle opening.
		// The opening no longer matches the committed root, so BaseFold verification rejects it.
		let last = proof.len() - 1;
		proof[last] ^= 1;

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let err = verifier.verify(&mut verifier_transcript).unwrap_err();
		assert_matches!(
			err,
			Error::IOPChannel(IOPChannelError::BaseFold(BaseFoldError::Verification(
				BaseFoldVerificationError::FRI(FriVerificationError::MerkleError(
					MerkleVerificationError::InvalidProof
				))
			)))
		);
	}

	// One product per instance, both result words committed as hidden words. A single flipped bit
	// in the IntMul check's first message must be rejected somewhere in the composed protocol.
	#[test]
	fn tampered_mul_opening_is_rejected() {
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let y = builder.add_witness();
		let (hi, lo) = builder.imul(x, y);
		builder.force_commit(hi);
		builder.force_commit(lo);
		let circuit = builder.build();

		let mut cs = circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();

		let log_instances = 6;
		let table = ValueTable::populate(&circuit, log_instances, |i, w| {
			let mut rng = StdRng::seed_from_u64(i as u64 + 1);
			w[x] = Word(rng.next_u64());
			w[y] = Word(rng.next_u64());
		})
		.unwrap();

		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Flip a bit early in the proof, in the IntMul check's first message. The verifier then
		// redraws a diverging challenge, so the composed protocol rejects the proof somewhere
		// downstream of that check.
		proof[0] ^= 1;

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		assert!(
			verifier.verify(&mut verifier_transcript).is_err(),
			"a proof tampered in the IntMul check's first message must not verify"
		);
	}
}
