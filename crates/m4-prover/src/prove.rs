// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{marker::PhantomData, ops::Deref};

use binius_compute::{Allocator, BufferPool, VecLike};
use binius_core::{
	constraint_system::{ConstraintSystem, InoutSegment, ValueTable},
	word::Word,
};
use binius_field::{PackedField, Rijndael8b as B8};
use binius_hash_prover::ParallelHashSuite;
use binius_iop_prover::{basefold::compiler::BaseFoldProverCompiler, channel::IOPProverChannel};
use binius_m4_verifier::{IOPVerifier, Verifier};
use binius_math::{
	BinarySubspace,
	ntt::{NeighborsLastMultiThread, domain_context::GaoMateerPreExpanded},
};
use binius_prover::{
	protocols::{
		binmul, bitand, intmul,
		rerand::OperandWitness,
		shift::{KeyCollection, OperatorClaims},
	},
	ring_switch::{self, RingSwitchOutput},
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::SerializeBytes;
use binius_verifier::{config::B128, protocols::bitand::AndCheckOutput};
use digest::Output;

use crate::{
	shift::prove as prove_shift,
	value_table::pack_table,
	witness::{FoldedWitness, OperandColumns},
};

/// The multithreaded additive NTT used to encode the committed codeword.
pub(crate) type ProverNtt = NeighborsLastMultiThread<GaoMateerPreExpanded<B128>>;

/// IOP prover for the M4 constraint reduction of a particular constraint system.
///
/// This struct encapsulates the constraint system and the pre-computed shift keys, providing the
/// core proving logic independent of the specific IOP compilation strategy. Most users should use
/// [`Prover`] instead, which wraps this with a BaseFold compiler.
///
/// Proving composes the commitment, the reduction, and the ring-switching opening on one
/// transcript, mirroring
/// [`IOPVerifier::verify_chip`](binius_m4_verifier::IOPVerifier::verify_chip):
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
	/// The validated single-instance constraint system shared by every instance.
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
	/// most users, [`Prover::prove_chip`] is the simpler interface.
	pub fn prove_chip<P, Channel, A, Data>(
		&self,
		table: &ValueTable<Data>,
		channel: &mut Channel,
		alloc: &A,
	) where
		P: PackedField<Scalar = B128>,
		Channel: IOPProverChannel<P, A>,
		A: Allocator,
		Data: Deref<Target = [Word]>,
	{
		let cs = &self.cs;

		// Pack the 2-D table into one multilinear and commit it as the trace oracle.
		let trace_packed = {
			let _scope = tracing::debug_span!("Prepare trace").entered();
			pack_table::<P, _, _>(table, alloc)
		};
		let trace_oracle = {
			let _scope = tracing::debug_span!("Commit trace").entered();
			channel.send_oracle(trace_packed.as_view())
		};

		// One base domain shared by the AND-check and the shift, consistent by construction.
		// The AND-check's univariate-skip domain spans one dimension above the 64-bit word.
		let andcheck_domain = BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1);
		// The shift domain drops that extra dimension.
		// This is exactly the domain the AND-check folds its bit axis over.
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
		// challenge. Do not reorder these, and keep the same order in `IOPVerifier::verify_chip`.
		//
		// The columns are the four operands of every constraint over every instance, laid out
		// constraint-major.
		// They are kept alongside the check output, since the BitAnd sumcheck re-reads them.
		let mul = (!cs.imul_constraints.is_empty()).then(|| {
			let columns = {
				let _scope = tracing::debug_span!("Assemble IntMul witness").entered();
				OperandColumns::build(table, &cs.constants, &cs.imul_constraints, alloc)
			};
			// The columns are built together from one constraint slice, so they are equal-length —
			// the only shape IntMul requires. It rounds the constraint axis up itself.
			let output = intmul::prove::<_, _, P, _>(columns.as_slices(), channel, alloc)
				.expect("the operand columns are equal-length");
			(columns, output)
		});

		// Build the BinMul operand columns and run the BinMul check, only when the circuit has BMUL
		// constraints.
		//
		// SOUNDNESS: the BinMul check runs after the IntMul check and before the BitAnd check
		// below. Its per-bit operand evaluations are bound to the transcript here, before BitAnd
		// draws the univariate challenge that collapses them. Do not reorder these, and keep the
		// same order in `IOPVerifier::verify_chip`.
		//
		// The six columns are the `(lo, hi)` word pairs of the two multiplicands and the product of
		// every constraint over every instance, laid out constraint-major. They are kept alongside
		// the check output, since the BitAnd sumcheck re-reads them. BinMul commits no oracle, so
		// nothing is added to `oracle_specs`.
		let bmul = (!cs.bmul_constraints.is_empty()).then(|| {
			let columns = {
				let _scope = tracing::debug_span!("Assemble BinMul witness").entered();
				OperandColumns::build(table, &cs.constants, &cs.bmul_constraints, alloc)
			};
			let output = binmul::prove::<_, B128, P, _>(columns.as_slices(), channel, alloc);
			(columns, output)
		});

		// The BitAnd sumcheck carries the multiplications' per-bit operand claims, IntMul first.
		let operands = [
			mul.as_ref().map(|(columns, output)| OperandWitness {
				words: columns.as_slices().into(),
				claims: output.operand_claims(),
			}),
			bmul.as_ref().map(|(columns, output)| OperandWitness {
				words: columns.as_slices().into(),
				claims: output.operand_claims(),
			}),
		]
		.into_iter()
		.flatten()
		.collect::<Vec<_>>();

		// AND-check the `A & B == C` relation over all `K * n_and` rows.
		let AndCheckOutput {
			z_challenge,
			rerand,
		} = {
			let _scope = tracing::debug_span!("BitAnd check").entered();

			let columns = {
				let _scope = tracing::debug_span!("Assemble BitAnd witness").entered();
				OperandColumns::build(table, &cs.constants, &cs.and_constraints, alloc)
			};
			bitand::prove::<_, B128, P, _, _>(columns.as_slices(), &operands, channel, alloc)
		};

		// Every operation is claimed at a prefix of the sumcheck's point `r_rho || r_x_star`, so
		// all of them share the instance point `r_rho`. The Zero point draws its extension here,
		// where `IOPVerifier::verify_chip` does.
		let log_instances = table.log_instances();
		let r_rho = rerand.eval_point[..log_instances].to_vec();
		let claims = OperatorClaims::from_rerand(cs, log_instances, z_challenge, &rerand, || {
			channel.sample()
		});

		// Fold the committed witness over the instance axis at the shared point.
		let folded_witness = {
			let _scope = tracing::debug_span!("Fold instances").entered();
			FoldedWitness::<B128, _>::fold_instances(table, &r_rho, alloc)
		};

		// The public segment is the shared constants alone: the inout values are committed, so
		// nothing else is public. The shift folds it against the monster's public part, which is
		// sized to the same count.
		let mut public_words = alloc.alloc::<Word>(cs.constants.len());
		public_words.extend_from_slice(&cs.constants);

		// Reduce the operand claims to one witness evaluation.
		let witness_claim = {
			let _scope = tracing::debug_span!("Prove shift reduction").entered();
			prove_shift::<B128, P, _, _>(
				&self.key_collection,
				&public_words,
				&folded_witness,
				claims,
				&shift_domain,
				channel,
				alloc,
			)
		};

		// Split the shift's final point `r_j || r_y || r_segment` into its three parts.
		// The bit index `r_j` is the low coordinates addressing a bit within a 64-bit word.
		// The segment selector `r_segment` is the last coordinate, choosing public or hidden
		// words. The hidden-only trace drops it.
		// The word index `r_y` is everything in between.
		let challenges = &witness_claim.sumcheck.challenges;
		let r_j = &challenges[..Word::LOG_BITS];
		let r_y = &challenges[Word::LOG_BITS..challenges.len() - 1];

		// Prove the public segment's evaluation claim, which the verifier's public-input check
		// consumes.
		ring_switch::prove_public_eval::<_, P, _>(alloc, &public_words, r_j, r_y, channel);

		// The wiring evaluation the verifier closes the shift check with, sent where it reads it:
		// after the public segment's claim.
		channel.send_public_claim(witness_claim.wiring_eval);

		let RingSwitchOutput {
			rs_eq_ind,
			sumcheck_claim,
		} = {
			let _scope = tracing::debug_span!("Ring-switching reduction").entered();

			// Ring-switch the reduced claim onto the committed trace.
			// The point is `r_j || r_rho || r_y`.
			// Its instance coordinates fold the trace at `r_rho`.
			let trace_point = [r_j, r_rho.as_slice(), r_y].concat();
			ring_switch::prove(alloc, trace_packed.as_view(), &trace_point, channel)
		};

		// Queue the trace opening against the ring-switch's transparent multilinear.
		// The final call runs the single combined FRI opening and writes it to the transcript.
		channel.prove_oracle_relation(trace_oracle.clone(), rs_eq_ind, sumcheck_claim);
		channel.finalize_oracle(trace_oracle, trace_packed);
	}
}

/// Proves the data-parallel M4 statement for a batch of `2^log_instances` circuit instances.
///
/// One-time setup builds the shift keys and the BaseFold prover, reusing the verifier's parameters.
/// A later proving call commits a witness table and proves it satisfies every AND constraint.
pub struct Prover<P, H>
where
	P: PackedField<Scalar = B128>,
	H: ParallelHashSuite,
{
	iop_prover: IOPProver,
	/// The precomputed BaseFold prover, holding the NTT and the FRI parameters.
	basefold_compiler: BaseFoldProverCompiler<P, ProverNtt>,
	/// The pool that recycles this prover's working buffers. It lives for the prover's lifetime,
	/// so blocks freed by one `prove` call are reused by the next.
	pool: BufferPool,
	/// The prover creates its Merkle transcript channels with the hash suite `H`, matching the
	/// verifier it was built from.
	_hash_marker: PhantomData<H>,
}

impl<P, H> Prover<P, H>
where
	P: PackedField<Scalar = B128>,
	H: ParallelHashSuite,
	Output<H::LeafHash>: SerializeBytes,
{
	/// Builds the prover from a verifier, inheriting its constraint system and FRI parameters.
	///
	/// The prover encodes the codeword with the multithreaded NTT, spread across the cores.
	/// Reusing the verifier's compiler keeps both sides on one set of FRI parameters.
	pub fn setup(verifier: &Verifier<H>) -> Self {
		// Reuse the verifier's evaluation domain so both sides agree on the code: its compiler
		// fixed that domain as the Gao-Mateer basis of this dimension.
		let domain_context =
			GaoMateerPreExpanded::generate(verifier.iop_compiler().max_log_domain_size());

		// Spread the NTT across the available cores.
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		// Inherit the verifier's oracle specs and FRI parameters verbatim.
		let basefold_compiler =
			BaseFoldProverCompiler::from_verifier_compiler(verifier.iop_compiler(), ntt);

		// Build the shift keys once from the shared constraint system.
		let key_collection =
			KeyCollection::build(verifier.constraint_system(), InoutSegment::Hidden);

		let iop_prover = IOPProver::new(verifier.iop_verifier().clone(), key_collection);

		Self {
			iop_prover,
			basefold_compiler,
			pool: BufferPool::new(),
			_hash_marker: PhantomData,
		}
	}

	/// Returns a reference to the IOP prover.
	pub const fn iop_prover(&self) -> &IOPProver {
		&self.iop_prover
	}

	/// Proves that every instance in the batch satisfies the constraint system.
	///
	/// Creates the IOP channel from the transcript, delegates to [`IOPProver::prove_chip`], then
	/// finishes the channel with the combined FRI opening.
	pub fn prove_chip<Challenger_, Data>(
		&self,
		table: &ValueTable<Data>,
		transcript: &mut ProverTranscript<Challenger_>,
	) where
		Challenger_: Challenger,
		Data: Deref<Target = [Word]>,
	{
		// Working buffers for this proof are drawn from the prover's pool, recycling blocks freed
		// by earlier proofs. The channel commits its Merkle trees out of the same pool.
		let alloc = &self.pool;
		let mut channel = self
			.basefold_compiler
			.create_channel_without_zk_from_transcript::<H, Challenger_, _, _>(transcript, alloc);
		self.iop_prover
			.prove_chip::<P, _, _, _>(table, &mut channel, &alloc);

		let _scope = tracing::debug_span!("PCS opening").entered();
		channel.finish();
	}
}

#[cfg(test)]
mod tests {
	use std::array;

	use assert_matches::assert_matches;
	use binius_compute::GlobalAllocator;
	use binius_field::PackedGhash1x128b;
	use binius_frontend::CircuitBuilder;
	use binius_hash::StdHashSuite;
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

	type P = PackedGhash1x128b;

	// Builds a batch of `2^log_instances` CRC-64 instances with random input words.
	fn setup_batch(log_instances: usize, seed: u64) -> (ConstraintSystem, ValueTable) {
		let c = crc64_circuit();
		let n_instances = 1usize << log_instances;
		let mut rng = StdRng::seed_from_u64(seed);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		let cs = c.circuit.constraint_system().clone();
		cs.validate().unwrap();
		(cs, table)
	}

	// Two proofs from one `Prover` are byte-identical, and both verify.
	//
	// A `Prover` owns its `BufferPool` for its whole lifetime, so the second proof draws blocks the
	// first one freed. Within a single proof the pool already recycles — each reduction returns its
	// working buffers as it finishes, and later allocations land on them — so a round trip does
	// exercise dirty memory. What only a second proof reaches is reuse of the blocks held for the
	// *whole* of the first: the operand columns and the instance-folded witness, which are live
	// until the proof ends and so are recycled nowhere else.
	//
	// The byte-for-byte equality is the part no other test asserts. It surfaces a slot a pooled
	// buffer leaves unwritten when the first proof reads it fresh and the second reads the first's
	// leftovers, without having to predict which buffer or which block that is. It is not a
	// superset of the round-trip tests: a slot that happens to read the same both times yields two
	// identically-wrong proofs, and the verification below is what catches those.
	//
	// The fixture carries AND, IMUL, and BMUL constraints so every operation's pooled buffers are
	// live, rather than only the AND path a mul-free circuit reaches.
	#[test]
	fn two_proofs_from_one_prover_are_byte_identical() {
		use binius_frontend::Wire;

		let builder = CircuitBuilder::new();
		let inputs: [Wire; 4] = array::from_fn(|_| builder.add_inout());
		let (hi, lo) = builder.imul(inputs[0], inputs[1]);
		let (c_lo, c_hi) = builder.bmul(inputs[0], inputs[1], inputs[2], inputs[3]);
		for wire in [builder.band(inputs[0], inputs[1]), hi, lo, c_lo, c_hi] {
			builder.mark_inout(wire);
		}
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		// Confirm the fixture reaches all three operations, so the recycled buffers really do span
		// every operand-column shape.
		assert!(!cs.imul_constraints.is_empty(), "the fixture must emit IMUL constraints");
		assert!(!cs.bmul_constraints.is_empty(), "the fixture must emit BMUL constraints");

		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64);
				for &wire in &inputs {
					w[wire] = Word(rng.next_u64());
				}
			})
			.unwrap();

		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		// One prover, two proofs of the same table: the second reuses the first's freed blocks.
		let prove_once = || {
			let mut transcript = ProverTranscript::new(StdChallenger::default());
			prover.prove_chip(&table, &mut transcript);
			transcript.finalize()
		};
		let first = prove_once();
		let second = prove_once();
		assert_eq!(first, second, "a second proof from the same prover must reproduce the first");

		// Both proofs stand on their own against the verifier.
		for proof in [first, second] {
			let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
			verifier
				.verify_chip(&mut verifier_transcript)
				.expect("a faithful proof verifies");
			verifier_transcript
				.finalize()
				.expect("no trailing proof data");
		}
	}

	// A batch carrying ZERO constraints alongside AND, IMUL and BMUL round-trips.
	//
	// The BitAnd sumcheck carries the IMUL and BMUL operand claims, and the Zero claim sits at a
	// prefix of its constraint point. A ZERO constraint array vanishes identically, so its claim is
	// zero at any point.
	#[test]
	fn protocol_round_trips_with_zero_constraints() {
		use binius_frontend::{Options, Wire};

		// The `bxor` chain lowers to ZERO constraints under the option, `band` keeps the AND set
		// non-empty, and `imul`/`bmul` bring the other two operations into the BitAnd sumcheck.
		//
		// Gate fusion is off: it inlines a linear definition into the gate that consumes it, which
		// would leave no linear constraint to lower.
		let mut opts = Options::default();
		opts.enable_gate_fusion = false;
		let builder = CircuitBuilder::with_opts(opts);
		let inputs: [Wire; 4] = array::from_fn(|_| builder.add_inout());
		let x = builder.bxor(inputs[0], inputs[1]);
		let y = builder.bxor(x, inputs[2]);
		let (hi, lo) = builder.imul(inputs[0], inputs[1]);
		let (c_lo, c_hi) = builder.bmul(inputs[0], inputs[1], inputs[2], inputs[3]);
		for wire in [x, y, builder.band(inputs[0], inputs[1]), hi, lo, c_lo, c_hi] {
			builder.mark_inout(wire);
		}
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		assert!(!cs.zero_constraints.is_empty(), "the fixture must emit ZERO constraints");
		assert!(!cs.and_constraints.is_empty(), "the fixture must emit AND constraints");
		assert!(!cs.imul_constraints.is_empty(), "the fixture must emit IMUL constraints");
		assert!(!cs.bmul_constraints.is_empty(), "the fixture must emit BMUL constraints");

		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64);
				for &wire in &inputs {
					w[wire] = Word(rng.next_u64());
				}
			})
			.unwrap();

		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// A batch violating a ZERO constraint is rejected.
	//
	// Dropping the last term of a satisfied ZERO constraint leaves one that the same table no
	// longer satisfies — `x ^ y ^ z = 0` becomes `x ^ y = 0`, false for all but a vanishing
	// fraction of the random inputs. The prover has nothing to send for the Zero reduction, so it
	// claims the constant zero regardless; the shift reduction, running against the committed
	// witness, is what catches the discrepancy.
	#[test]
	fn protocol_rejects_violated_zero_constraint() {
		use binius_core::constraint_system::ZeroConstraint;
		use binius_frontend::{Options, Wire};

		// Gate fusion off, so the `bxor` survives as a linear constraint to lower.
		let mut opts = Options::default();
		opts.enable_gate_fusion = false;
		let builder = CircuitBuilder::with_opts(opts);
		let inputs: [Wire; 3] = array::from_fn(|_| builder.add_inout());
		builder.mark_inout(builder.bxor(inputs[0], inputs[1]));
		let circuit = builder.build();

		let mut cs = circuit.constraint_system().clone();
		let victim = cs
			.zero_constraints
			.iter()
			.position(|c| c.val().len() > 2)
			.expect("the fixture must emit a ZERO constraint with a droppable term");
		let mut terms = cs.zero_constraints[victim].val().clone();
		terms.pop();
		cs.zero_constraints[victim] = ZeroConstraint::new(terms);
		cs.validate().unwrap();

		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64);
				for &wire in &inputs {
					w[wire] = Word(rng.next_u64());
				}
			})
			.unwrap();

		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		assert!(
			verifier.verify_chip(&mut verifier_transcript).is_err(),
			"a violated ZERO constraint must not verify"
		);
	}

	// The prover and verifier run the whole protocol on one transcript.
	// A faithful proof over 64 instances verifies and leaves no trailing data.
	#[test]
	fn protocol_round_trips() {
		let log_instances = 6;
		let (cs, table) = setup_batch(log_instances, 0);

		// Setup once: the verifier fixes the shape and FRI parameters.
		// The prover inherits them.
		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		// Prover: commit, reduce, and open on a fresh transcript.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		// Verifier: replay the same transcript end to end.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
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
	// The IntMul and AND checks reduce to different instance points, which the BitAnd sumcheck
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
		let x = builder.add_inout();
		let y = builder.add_inout();
		let (hi, lo) = builder.imul(x, y);
		builder.mark_inout(hi);
		builder.mark_inout(lo);
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		// Confirm the fixture genuinely exercises the IntMul path.
		assert!(!cs.imul_constraints.is_empty(), "the fixture must emit an IMUL constraint");

		// Fill each instance's two multiplicands from a per-instance seed; the circuit derives the
		// two product words.
		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64);
				w[x] = Word(rng.next_u64());
				w[y] = Word(rng.next_u64());
			})
			.unwrap();

		// Setup once: the verifier fixes the shape and FRI parameters, the prover inherits them.
		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		// Prover: commit both oracles, reduce, and open on a fresh transcript.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		// Verifier: replay the same transcript end to end.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// A circuit declaring inout wires round-trips through the whole protocol.
	//
	// The inout values are committed with the private ones, so they lead the hidden segment and the
	// public segment is the constants alone. That moves the boundary the shift reduction splits at,
	// which every stage below it must agree on: the key collection, the word-index tensor, the
	// committed shape, and the trace point the ring-switch opens at.
	//
	// Fixture: a per-instance public input and output either side of a private computation, over
	// 2^6 instances. A constant keeps the public segment non-empty, so both halves carry words.
	#[test]
	fn protocol_round_trips_with_inout_wires() {
		let builder = CircuitBuilder::new();
		let input = builder.add_inout();
		let output = builder.add_inout();
		let secret = builder.add_witness();
		let k = builder.add_constant_64(0x0123_4567_89ab_cdef);
		// output == (input & secret) ^ k, so both inout wires are read by real constraints.
		let masked = builder.band(input, secret);
		builder.assert_eq("output", output, builder.bxor(masked, k));
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		// Confirm the fixture genuinely exercises the inout path, on both sides of the boundary.
		assert!(cs.n_inout > 0, "the fixture must declare inout wires");
		assert!(!cs.constants.is_empty(), "the public segment must hold words of its own");

		// Every instance chooses its own inout words — the reason they cannot be shared public
		// data.
		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64);
				let input_word = rng.next_u64();
				let secret_word = rng.next_u64();
				w[input] = Word(input_word);
				w[secret] = Word(secret_word);
				w[output] = Word((input_word & secret_word) ^ 0x0123_4567_89ab_cdef);
			})
			.unwrap();

		// The committed segment covers the inout words as well as the private ones.
		assert_eq!(
			table.n_hidden_words(),
			cs.n_hidden_words(InoutSegment::Hidden),
			"the table commits the inout values with the private ones"
		);

		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
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
		let cv: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		let block: [Wire; 16] = array::from_fn(|_| builder.add_inout());
		let counter = builder.add_inout();
		let block_len = builder.add_inout();
		let flags = builder.add_inout();
		// Promoting the output chaining value keeps the compression alive under dead-code
		// elimination.
		for wire in blake3_compress(&builder, cv, block, counter, block_len, flags) {
			builder.mark_inout(wire);
		}
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		// Confirm the fixture is genuine: the constant count is not a power of two.
		assert!(!cs.constants.len().is_power_of_two());

		// Fill each instance's inputs from a per-instance seed; the compression derives the rest.
		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
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

		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// Independent AND constraints alongside IMUL constraints, so the two operations reduce to
	// constraint points of different lengths (`log_n_and != log_n_imul`) and to genuinely different
	// instance points that the BitAnd sumcheck must unify.
	//
	// Proving with a width-2 packing exercises the packed lane layout of the folded operand
	// columns, which the width-1 fixtures never reach.
	#[test]
	fn protocol_round_trips_with_mixed_constraints_and_wide_packing() {
		use binius_field::PackedGhash2x128b;
		use binius_frontend::Wire;

		type WideP = PackedGhash2x128b;

		let builder = CircuitBuilder::new();
		let inputs: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		// Four standalone AND gates on distinct wires — the AND work is not tied to the products.
		for pair in inputs.chunks_exact(2) {
			builder.mark_inout(builder.band(pair[0], pair[1]));
		}
		// Two products — fewer IMUL constraints than AND constraints.
		for pair in inputs.chunks_exact(2).take(2) {
			let (hi, lo) = builder.imul(pair[0], pair[1]);
			builder.mark_inout(hi);
			builder.mark_inout(lo);
		}
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		// Confirm the fixture genuinely exercises the asymmetric case: the two operations have
		// different constraint-point lengths.
		// An absent operation contributes an empty `r_x`, as does an AND set of one row.
		let log_n_and = cs.log_and_constraints().unwrap_or(0);
		let log_n_imul = cs.log_imul_constraints().unwrap_or(0);
		assert_ne!(
			log_n_and, log_n_imul,
			"the fixture must give the operations different r_x lengths"
		);

		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64);
				for &wire in &inputs {
					w[wire] = Word(rng.next_u64());
				}
			})
			.unwrap();

		// Prove with the wide packing; the verifier is packing-agnostic.
		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<WideP, StdHashSuite>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// A circuit carrying BMUL constraints round-trips through the whole protocol.
	//
	// BinMul commits no oracle, so the proof still commits only the trace oracle. The BinMul and
	// AND checks reduce to different instance points, which the BitAnd sumcheck unifies before the
	// witness is folded.
	//
	// Fixture: one GHASH-field product `x * x` per instance over 2^6 instances, both product words
	// force-committed. The `bmul` gate emits one BMUL constraint.
	//
	// A faithful proof verifies and no trailing data is left.
	#[test]
	fn protocol_round_trips_with_binmul() {
		// One GHASH-field squaring per instance: `(c_lo, c_hi) = (x_lo, x_hi)^2`, with both result
		// words committed as hidden words.
		let builder = CircuitBuilder::new();
		let x_lo = builder.add_inout();
		let x_hi = builder.add_inout();
		let (c_lo, c_hi) = builder.bmul(x_lo, x_hi, x_lo, x_hi);
		builder.mark_inout(c_lo);
		builder.mark_inout(c_hi);
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		// Confirm the fixture genuinely exercises the BinMul path.
		assert!(!cs.bmul_constraints.is_empty(), "the fixture must emit a BMUL constraint");

		// Fill each instance's multiplicand from a per-instance seed; the circuit derives the two
		// product words.
		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64);
				w[x_lo] = Word(rng.next_u64());
				w[x_hi] = Word(rng.next_u64());
			})
			.unwrap();

		// Setup once: the verifier fixes the shape and FRI parameters, the prover inherits them.
		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		// Prover: commit the trace, reduce, and open on a fresh transcript.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		// Verifier: replay the same transcript end to end.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// AND, IMUL, and BMUL constraints together, so the three operations reduce to constraint points
	// of differing lengths and to genuinely different instance points that the BitAnd sumcheck
	// must unify onto one shared point.
	//
	// Proving with a width-2 packing exercises the packed lane layout of the folded operand
	// columns, which the width-1 fixtures never reach.
	#[test]
	fn protocol_round_trips_with_and_intmul_binmul_and_wide_packing() {
		use binius_field::PackedGhash2x128b;
		use binius_frontend::Wire;
		type WideP = PackedGhash2x128b;

		let builder = CircuitBuilder::new();
		let inputs: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		// Four standalone AND gates on distinct wires.
		for pair in inputs.chunks_exact(2) {
			builder.mark_inout(builder.band(pair[0], pair[1]));
		}
		// Two integer products — fewer IMUL constraints than AND constraints.
		for pair in inputs.chunks_exact(2).take(2) {
			let (hi, lo) = builder.imul(pair[0], pair[1]);
			builder.mark_inout(hi);
			builder.mark_inout(lo);
		}
		// One GHASH-field product — the fewest of the three operations.
		let (c_lo, c_hi) = builder.bmul(inputs[0], inputs[1], inputs[2], inputs[3]);
		builder.mark_inout(c_lo);
		builder.mark_inout(c_hi);
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		// Confirm the fixture genuinely exercises the asymmetric case: the three operations do not
		// all reduce to constraint points of the same length.
		// An absent operation contributes an empty `r_x`, as does an AND set of one row.
		let log_n_and = cs.log_and_constraints().unwrap_or(0);
		let log_n_imul = cs.log_imul_constraints().unwrap_or(0);
		let log_n_binmul = cs.log_bmul_constraints().unwrap_or(0);
		assert!(!cs.imul_constraints.is_empty(), "the fixture must emit IMUL constraints");
		assert!(!cs.bmul_constraints.is_empty(), "the fixture must emit BMUL constraints");
		let lengths = [log_n_and, log_n_imul, log_n_binmul];
		assert!(
			lengths.iter().any(|&len| len != lengths[0]),
			"the fixture must give the operations differing r_x lengths"
		);

		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64);
				for &wire in &inputs {
					w[wire] = Word(rng.next_u64());
				}
			})
			.unwrap();

		// Prove with the wide packing; the verifier is packing-agnostic.
		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<WideP, StdHashSuite>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// None of the three operations has a power-of-two constraint count, so each one's operand
	// columns stop partway through the constraint axis its reduction runs over.
	//
	// Every other fixture in this module happens to land on a power of two, where the columns span
	// the axis exactly and the short-column path is never taken. This is the one that reaches it.
	#[test]
	fn protocol_round_trips_with_non_power_of_two_constraint_counts() {
		use binius_frontend::Wire;
		let builder = CircuitBuilder::new();
		let inputs: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		// Three standalone AND gates, three integer products, and three GHASH-field products.
		for pair in inputs.chunks_exact(2).take(3) {
			builder.mark_inout(builder.band(pair[0], pair[1]));
		}
		for pair in inputs.chunks_exact(2).take(3) {
			let (hi, lo) = builder.imul(pair[0], pair[1]);
			builder.mark_inout(hi);
			builder.mark_inout(lo);
		}
		for i in 0..3 {
			let (c_lo, c_hi) = builder.bmul(inputs[i], inputs[i + 1], inputs[i + 2], inputs[i + 3]);
			builder.mark_inout(c_lo);
			builder.mark_inout(c_hi);
		}
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		// Confirm the fixture reaches the case it exists for. A power-of-two count would leave
		// every column spanning its axis exactly, which the other fixtures already cover.
		for (name, count) in [
			("AND", cs.n_and_constraints()),
			("IMUL", cs.n_imul_constraints()),
			("BMUL", cs.n_bmul_constraints()),
		] {
			assert!(
				!count.is_power_of_two(),
				"the fixture must give {name} a non-power-of-two constraint count, got {count}"
			);
		}

		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64);
				for &wire in &inputs {
					w[wire] = Word(rng.next_u64());
				}
			})
			.unwrap();

		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify_chip(&mut verifier_transcript)
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

		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		// Produce a faithful proof, then collect its bytes.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Flip one bit in the last byte, which lands in a FRI query's Merkle opening.
		// The opening no longer matches the committed root, so BaseFold verification rejects it.
		let last = proof.len() - 1;
		proof[last] ^= 1;

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let err = verifier.verify_chip(&mut verifier_transcript).unwrap_err();
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
		let x = builder.add_inout();
		let y = builder.add_inout();
		let (hi, lo) = builder.imul(x, y);
		builder.mark_inout(hi);
		builder.mark_inout(lo);
		let circuit = builder.build();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();

		let log_instances = 6;
		let table = circuit
			.populate_batch(&GlobalAllocator, log_instances, |i, w| {
				let mut rng = StdRng::seed_from_u64(i as u64 + 1);
				w[x] = Word(rng.next_u64());
				w[y] = Word(rng.next_u64());
			})
			.unwrap();

		let verifier = Verifier::<StdHashSuite>::setup(&cs, log_instances, 1);
		let prover = Prover::<P, StdHashSuite>::setup(&verifier);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove_chip(&table, &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Flip a bit early in the proof, in the IntMul check's first message. The verifier then
		// redraws a diverging challenge, so the composed protocol rejects the proof somewhere
		// downstream of that check.
		proof[0] ^= 1;

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		assert!(
			verifier.verify_chip(&mut verifier_transcript).is_err(),
			"a proof tampered in the IntMul check's first message must not verify"
		);
	}
}
