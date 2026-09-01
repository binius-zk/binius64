// Copyright 2026 The Binius Developers

//! A WHIR opening proved natively, then verified inside a Binius64 circuit.
//!
//! ```text
//!   prover:    native, writing one real transcript
//!   verifier:  the same code over the builder channel, emitting gates instead of checking values
//! ```
//!
//! A satisfied constraint system is then the statement "this transcript opens this claim".
//! An outer proof can check that statement, which is what makes this one step of recursion.
//!
//! The sibling test over the FRI-based scheme sets out the shared machinery in full.
//! What follows is what a ladder adds to it.
//!
//! # Why the ladder is expressible at all
//!
//! Every level does three things a channel already speaks.
//!
//! ```text
//!   fold rounds     a received element, then a challenge drawn after it
//!   query rows      a masked bit sample per position, and the leaves under them
//!   induced basis   one bit-decomposed sum over fixed constants per position
//! ```
//!
//! Nothing asks the channel to invert a value it received.
//! Nothing asks it to materialize the weight vector a level's rows induce.
//!
//! The second is settled by the types rather than by discipline.
//! Materializing that vector needs a packed field, and the element a circuit carries is not one.
//! So the dense route is unreachable from the verifier's body, and this file compiling says so.
//!
//! # What the residual costs
//!
//! The last level sends its remaining matrix in the clear, against its own commitment.
//! A circuit rebuilds that whole tree, one leaf per element.
//! So the cleartext residual is the only part of a ladder growing as `2^r`.
//! Everything else grows as a logarithm.
//!
//! The sweep at the end of this file prices that difference.
//!
//! # Proof of work
//!
//! A ground nonce enters as a wire, is absorbed, and the draw it decides is asserted zero.
//! That is the same check the native verifier makes, phrased as a constraint.

use std::iter;

use binius_compute::GlobalAllocator;
use binius_core::word::Word;
use binius_field::{Ghash128b as B128, PackedGhash1x128b};
use binius_frontend::{CircuitStat, MAX_ASSERTION_FAILURES, PopulateError, Wire};
use binius_hash::{StdDigest, StdHashSuite};
use binius_iop::{
	merkle_channel::MerkleIPVerifierChannel,
	merkle_tree::{BinaryMerkleTreeScheme, MerkleTreeScheme},
	soundness::{Grinding, SoundnessRegime},
	whir::{WHIRLevel, WHIRParams, WHIRVerifier},
};
use binius_iop_prover::{merkle_channel::ProverMerkleTranscriptChannel, whir::WHIRProver};
use binius_ip::channel::WordIPVerifierChannel;
use binius_ip_prover::channel::WordIPProverChannel;
use binius_math::{
	multilinear::evaluate::evaluate,
	ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
	test_utils::{random_field_buffer, random_scalars},
};
use binius_recursion::{Binius64BuilderChannel, Recorded, WitnessFillerChannel};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
use rand::{SeedableRng, rngs::StdRng};

/// The Fiat-Shamir challenger the in-circuit challenger reproduces.
type StdChallenger = HasherChallenger<StdDigest>;

/// The packed field the native prover runs over.
type P = PackedGhash1x128b;

/// Words one field element occupies on the transcript, low half first.
const ELEMENT_WORDS: usize = 2;

/// Bytes one SHA-256 digest occupies on the tape, which is what a commitment root is.
const DIGEST_BYTES: usize = 32;

/// Bytes one field element occupies on the tape.
const ELEMENT_BYTES: usize = ELEMENT_WORDS * Word::BYTES;

/// The security target the ladders here are labelled with.
///
/// Nothing in this file reads it.
/// The shapes here are small enough to build a circuit from, far below any real target.
const SECURITY_BITS: usize = 32;

/// The transcript words one field element serializes to.
fn element_words(value: B128) -> [Word; ELEMENT_WORDS] {
	let value = u128::from(value);
	[
		Word::from_u64(value as u64),
		Word::from_u64((value >> 64) as u64),
	]
}

/// The shape of one WHIR ladder.
///
/// Every field is fixed before a proof exists, so a shape is exactly what a circuit is built for.
#[derive(Clone, Copy, Debug)]
struct Shape {
	/// Base-2 logarithm of the columns level 0 commits.
	log_msg_cols: usize,
	/// The lanes each level folds away, outermost level first.
	lanes: &'static [usize],
	/// Rows every level opens against its own codeword.
	n_queries: usize,
	/// The proof of work each level pays, at the two points a level pays one.
	grinding: Grinding,
}

/// A two-level ladder over a `2^8` message, leaving a `2^4` residual.
///
/// Level 0 commits `2^6` columns and 4 lanes at rate `1/2`.
/// Level 1 takes what that folds to, `2^4` columns and 4 lanes, at rate `1/4`.
const NATIVE_SHAPE: Shape = Shape {
	log_msg_cols: 6,
	lanes: &[2, 2],
	n_queries: 8,
	grinding: Grinding::NONE,
};

/// What a shape fixes ahead of any proof: the ladder, and the transform its levels encode over.
struct Setup {
	/// The ladder, one entry per committed level.
	params: WHIRParams,
	/// The additive NTT the prover encodes every level with.
	ntt: NeighborsLastSingleThread<GaoMateerOnTheFly<B128>>,
}

impl Shape {
	/// Base-2 logarithm of the elements the committed message holds.
	///
	/// Level 0 spans its columns and its lanes, and that is the whole message.
	const fn log_msg_len(&self) -> usize {
		self.log_msg_cols + self.lanes[0]
	}

	/// Transcript words the statement occupies: the point, then the claim.
	const fn statement_words(&self) -> usize {
		(self.log_msg_len() + 1) * ELEMENT_WORDS
	}

	/// Transcript words the evaluation point alone occupies.
	const fn point_words(&self) -> usize {
		self.log_msg_len() * ELEMENT_WORDS
	}

	/// Derives the ladder and the encoding transform, neither of which needs a witness.
	///
	/// Level `i` commits at inverse rate `2^(i + 1)`.
	/// That is the strictly falling rate a ladder requires.
	///
	/// Level `i + 1` takes what level `i` folded to.
	/// So its columns are level `i`'s less its own lanes.
	fn setup(&self) -> Setup {
		let mut log_msg_cols = self.log_msg_cols;
		let levels = self
			.lanes
			.iter()
			.enumerate()
			.map(|(i, &log_lanes)| {
				// Level 0 keeps the column count it was given; every deeper one loses its lanes.
				if i > 0 {
					log_msg_cols -= log_lanes;
				}
				WHIRLevel {
					log_msg_cols,
					log_lanes,
					log_inv_rate: i + 1,
					n_queries: self.n_queries,
				}
			})
			.collect();
		let params = WHIRParams::new(levels, SoundnessRegime::UniqueDecoding, SECURITY_BITS)
			.with_grinding(self.grinding);

		// One transform serves the whole ladder, sized for its longest codeword.
		let ntt = NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(
			params.max_log_codeword_len(),
		));
		Setup { params, ntt }
	}

	/// Proves one opening of this ladder, drawing the witness and the point from `seed`.
	///
	/// The statement is written to the transcript rather than handed over out of band.
	/// That is what lets it be wires rather than build-time constants.
	/// It also binds the proof to the one statement it was written for.
	///
	/// The point is observed first, since it is known before anything is committed.
	/// The claim follows level 0's root, since it is a claim about what that root commits.
	fn prove(&self, setup: &Setup, seed: u64) -> Opening {
		// One seed fixes the polynomial and the point, so a transcript is reproducible.
		let mut rng = StdRng::seed_from_u64(seed);
		let witness = random_field_buffer::<P>(&mut rng, self.log_msg_len());
		let eval_point: Vec<B128> = random_scalars(&mut rng, self.log_msg_len());
		let eval_claim = evaluate(&witness, &eval_point);

		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let mut channel =
			ProverMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::new(
				&mut transcript,
			);

		let point_words = eval_point
			.iter()
			.flat_map(|value| element_words(*value))
			.collect::<Vec<_>>();
		WordIPProverChannel::<B128>::observe_words(&mut channel, &point_words);

		// Encodes level 0 lane by lane and sends its Merkle root.
		let prover = WHIRProver::commit(&setup.params, &setup.ntt, witness.as_view(), &mut channel);
		WordIPProverChannel::<B128>::observe_words(&mut channel, &element_words(eval_claim));

		prover.prove(witness.as_view(), &eval_point, eval_claim, &GlobalAllocator, &mut channel);
		// Hand the transcript back, releasing the channel's borrow of it.
		channel.into_transcript();

		Opening {
			eval_claim,
			eval_point,
			proof: transcript.finalize(),
		}
	}

	/// Records the verifier for this ladder as a circuit, by running it over the builder channel.
	///
	/// No proof reaches this, and no statement either.
	/// Observing reads only the *length* of what it is handed.
	/// Placeholder words therefore allocate the wires a real statement later fills.
	fn record(&self, setup: &Setup) -> VerifierCircuit {
		let mut channel = Binius64BuilderChannel::new();

		// Placeholders, for their count alone.
		// The values never reach a gate.
		let placeholder = vec![Word::ZERO; self.statement_words()];
		let statement = channel.observe_words(&placeholder[..self.point_words()]);
		let eval_point = channel.pack_words(&statement);

		// Level 0's root is the caller's to receive, so the ladder is handed a commitment.
		// One leaf holds a codeword position across every lane.
		let level = &setup.params.levels()[0];
		let commitment = channel
			.recv_merkle_commitment(1 << level.log_lanes, level.log_codeword_len())
			.expect("reading a commitment cannot fail");

		let claim_words = channel.observe_words(&placeholder[self.point_words()..]);
		let eval_claim = channel.pack_words(&claim_words)[0].clone();

		// Every check the verifier makes becomes a constraint rather than a comparison.
		// So the call cannot fail while the circuit is being built.
		WHIRVerifier::new(&setup.params, commitment)
			.verify::<B128, _>(&eval_point, eval_claim, &mut channel)
			.expect("the builder channel records rather than checks, so it cannot fail");

		// The statement becomes the circuit's public interface.
		// An outer proof can then pin what was verified rather than trust whoever filled it.
		let public = channel.bind_public(statement.into_iter().chain(claim_words).collect());
		let recorded = channel.build();
		let stat = CircuitStat::collect(&recorded.circuit);
		VerifierCircuit {
			recorded,
			public,
			stat,
		}
	}
}

impl Setup {
	/// The tape offset of the last residual element the ladder sends.
	///
	/// The residual is sent before the last level's rows are opened, so the tape ends like this:
	///
	/// ```text
	///     .. -> residual root -> residual elements -> decommitted layer -> leaves and branches
	/// ```
	///
	/// Everything after the residual belongs to that one opening, whose size the ladder fixes.
	/// So the residual's last byte is the whole tape less that tail.
	fn last_residual_byte(&self, proof_len: usize) -> usize {
		let level = self
			.params
			.levels()
			.last()
			.expect("a ladder has at least one level");
		let tree_depth = level.log_codeword_len();

		// The same layer depth both channels pick, so the same digest count is on the tape.
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		let layer_depth = scheme.optimal_verify_layer(level.n_queries, tree_depth);

		// One leaf holds a codeword position across every lane, and one branch climbs to the layer.
		let leaf_bytes = (1 << level.log_lanes) * ELEMENT_BYTES;
		let branch_bytes = (tree_depth - layer_depth) * DIGEST_BYTES;
		let tail =
			(1 << layer_depth) * DIGEST_BYTES + level.n_queries * (leaf_bytes + branch_bytes);

		proof_len - tail - 1
	}
}

/// One opening proved natively: the statement it proves, and the transcript proving it.
struct Opening {
	/// The claimed evaluation of the committed multilinear at the point below.
	eval_claim: B128,
	/// The point the claim is made at, low-to-high variable order.
	eval_point: Vec<B128>,
	/// The proof byte tape.
	proof: Vec<u8>,
}

impl Opening {
	/// The statement as transcript words, in the order both halves observe it.
	fn statement(&self) -> Vec<Word> {
		self.eval_point
			.iter()
			.chain(iter::once(&self.eval_claim))
			.flat_map(|value| element_words(*value))
			.collect()
	}
}

/// A circuit that verifies any opening of one fixed ladder.
struct VerifierCircuit {
	/// The compiled circuit and the wires a replay fills.
	recorded: Recorded,
	/// One public wire per statement word, each equal to what the verifier observed.
	public: Vec<Wire>,
	/// Constraint counts and trace size.
	stat: CircuitStat,
}

impl VerifierCircuit {
	/// Populates the circuit from a statement and a proof, and reports whether it is satisfied.
	///
	/// The statement is written twice over.
	/// Onto the public wires here, and onto the wires the replay fills as it observes it.
	/// The binding between them is what ties the two.
	fn check(
		&self,
		shape: &Shape,
		setup: &Setup,
		statement: &[Word],
		proof: &[u8],
	) -> Result<(), PopulateError> {
		let mut w = self.recorded.circuit.new_witness_filler();
		for (&wire, &word) in iter::zip(&self.public, statement) {
			w[wire] = word;
		}

		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof.to_vec());
		let mut channel = WitnessFillerChannel::<_, StdChallenger, StdHashSuite>::new(
			&mut transcript,
			&mut w,
			self.recorded.inputs.clone(),
		);
		let point = channel.observe_words(&statement[..shape.point_words()]);
		let eval_point = channel.pack_words(&point);
		let level = &setup.params.levels()[0];
		let commitment = channel
			.recv_merkle_commitment(1 << level.log_lanes, level.log_codeword_len())
			.expect("the tape carries a commitment");
		let claim = channel.observe_words(&statement[shape.point_words()..]);
		let eval_claim = channel.pack_words(&claim)[0];

		WHIRVerifier::new(&setup.params, commitment)
			.verify::<B128, _>(&eval_point, eval_claim, &mut channel)
			.expect("the replay generates a witness rather than judging one");
		channel.finish();

		self.recorded.circuit.populate_wire_witness(&mut w)
	}

	/// Populates and checks one natively proved opening.
	fn check_opening(
		&self,
		shape: &Shape,
		setup: &Setup,
		opening: &Opening,
	) -> Result<(), PopulateError> {
		self.check(shape, setup, &opening.statement(), &opening.proof)
	}

	/// Asserts the circuit rejects `proof`, and returns the subcircuits whose assertions failed.
	fn rejected(
		&self,
		shape: &Shape,
		setup: &Setup,
		opening: &Opening,
		proof: &[u8],
	) -> Vec<String> {
		let error = self
			.check(shape, setup, &opening.statement(), proof)
			.expect_err("a corrupted proof must leave the circuit unsatisfied");

		// `..` is forced:
		// `PopulateError` is non-exhaustive.
		// Both of its fields are checked here.
		let PopulateError {
			failures, total, ..
		} = error;
		assert!(total > 0, "an unsatisfied circuit must report a failing assertion");
		assert_eq!(failures.len(), total.min(MAX_ASSERTION_FAILURES));
		for failure in &failures {
			assert!(!failure.detail.is_empty(), "a failure must carry a diagnostic");
		}

		// The leading path component names the check, which is what each test below asserts on.
		let mut named = failures
			.into_iter()
			.map(|failure| {
				failure
					.path
					.trim_start_matches('.')
					.split('.')
					.next()
					.unwrap_or_default()
					.trim_end_matches(|c: char| c.is_ascii_digit() || c == '[' || c == ']')
					.to_string()
			})
			.collect::<Vec<_>>();
		named.sort();
		named.dedup();
		named
	}
}

#[test]
fn a_native_opening_verifies_in_circuit() {
	// Invariant: the circuit built from the verifier accepts a transcript the prover wrote.
	//
	// Fixture state: the native ladder of two levels over a 2^8 message, and one proof.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = shape.prove(&setup, 0);
	// Nothing about the proof reaches the builder, so the order of these two lines is immaterial.
	let verifier = shape.record(&setup);

	println!(
		"whir verifier: {} gates, {} AND, {} BMUL, {} ZERO, {} committed words, {} recorded inputs",
		verifier.stat.n_gates,
		verifier.stat.n_and_constraints,
		verifier.stat.n_bmul_constraints,
		verifier.stat.n_zero_constraints,
		verifier.stat.committed_allocated,
		verifier.recorded.inputs.len(),
	);

	// The statement is the circuit's public interface, one wire per transcript word.
	assert_eq!(verifier.public.len(), shape.statement_words());

	// Nothing else is an input.
	// Every recorded wire is a value read off the tape, or the statement itself.
	// No challenge, no query index, no digest the circuit could have recomputed.
	//
	// The residual arrives in the clear.
	// The FRI-based sibling calls that entry a terminal codeword.
	let mut kinds = verifier
		.recorded
		.inputs
		.iter()
		.map(|input| input.kind)
		.collect::<Vec<_>>();
	kinds.sort_unstable();
	kinds.dedup();
	assert_eq!(
		kinds,
		[
			"committed_vector",
			"merkle_branch",
			"merkle_layer",
			"merkle_root",
			"observe_words",
			"opening",
			"recv_one",
		]
	);

	verifier
		.check_opening(&shape, &setup, &opening)
		.expect("the circuit must accept the opening the native prover proved");
}

#[test]
fn one_circuit_verifies_two_different_openings() {
	// Invariant: one compiled circuit verifies every opening of its ladder.
	//
	// Fixture state: two proofs of the native ladder, from two different seeds.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();

	// Different witness and different evaluation point, so a different transcript throughout.
	let first = shape.prove(&setup, 1);
	let second = shape.prove(&setup, 2);
	// Two openings that happened to coincide would make the rest of the test vacuous.
	assert_ne!(first.eval_point, second.eval_point, "the two openings must differ");
	assert_ne!(first.proof, second.proof, "the two transcripts must differ");
	// Equal lengths are the observable half of "the layout depends only on the ladder".
	assert_eq!(
		first.proof.len(),
		second.proof.len(),
		"one ladder means one tape length, whatever the witness"
	);

	// Built once, before either proof is looked at.
	let verifier = shape.record(&setup);
	for (i, opening) in [&first, &second].into_iter().enumerate() {
		verifier
			.check_opening(&shape, &setup, opening)
			.unwrap_or_else(|error| panic!("opening {i} must satisfy the shared circuit: {error}"));
	}
}

#[test]
fn a_one_level_ladder_verifies_in_circuit() {
	// Invariant: a one-entry ladder is the protocol with no glue at all.
	// Level 0 folds, and its rows meet the residual directly with no sumcheck to join.
	//
	// Fixture state: one level of 2^6 columns and 4 lanes, leaving a 2^6 residual.
	let shape = Shape {
		lanes: &[2],
		..NATIVE_SHAPE
	};
	let setup = shape.setup();
	assert_eq!(setup.params.n_levels(), 1);
	assert_eq!(setup.params.log_residual_dim(), 6);

	let opening = shape.prove(&setup, 9);
	let verifier = shape.record(&setup);
	verifier
		.check_opening(&shape, &setup, &opening)
		.expect("a single level must verify with no glued claim to carry");
}

#[test]
fn a_three_level_ladder_verifies_in_circuit() {
	// Invariant: every level below level 0 runs a plain degree-2 sumcheck and glues its rows in.
	// Two glued bases then have to follow the fold down to the residual.
	//
	// Fixture state: three levels over a 2^8 message, leaving a 2^2 residual.
	//
	//     level 0:  2^6 cols, 4 lanes, rate 1/2
	//     level 1:  2^4 cols, 4 lanes, rate 1/4
	//     level 2:  2^2 cols, 4 lanes, rate 1/8
	let shape = Shape {
		lanes: &[2, 2, 2],
		..NATIVE_SHAPE
	};
	let setup = shape.setup();
	assert_eq!(setup.params.log_residual_dim(), 2);

	let opening = shape.prove(&setup, 10);
	let verifier = shape.record(&setup);
	verifier
		.check_opening(&shape, &setup, &opening)
		.expect("two glued bases must reach the residual folded the same way both sides fold them");
}

#[test]
fn an_odd_query_count_verifies_in_circuit() {
	// Invariant: the openings of one commitment climb in pairs.
	// An odd query count leaves the last one climbing alone.
	// Both routes must reach the same layer.
	//
	// Fixture state: the native ladder at 7 queries, so every opened level has a leftover.
	let shape = Shape {
		n_queries: 7,
		..NATIVE_SHAPE
	};
	let setup = shape.setup();
	let opening = shape.prove(&setup, 11);
	let verifier = shape.record(&setup);

	verifier
		.check_opening(&shape, &setup, &opening)
		.expect("the leftover query must verify beside the paired ones");
}

#[test]
fn a_ground_ladder_verifies_in_circuit() {
	// Invariant: a nonce the prover ground is checked in-circuit at the point it was paid.
	// The nonce is a wire.
	// Absorbing it moves the Fiat-Shamir state.
	// The draw it decides is asserted zero.
	//
	// Fixture state: the native ladder, 4 bits before each fold challenge and 3 before each query.
	//
	//     level i:  round message -> nonce -> 4 bits asserted zero -> fold challenge
	//               next commitment -> nonce -> 3 bits asserted zero -> query positions
	let shape = Shape {
		grinding: Grinding::new(4, 3),
		..NATIVE_SHAPE
	};
	let setup = shape.setup();
	let opening = shape.prove(&setup, 12);
	let verifier = shape.record(&setup);

	// Two levels folding 2 lanes each is 4 challenge grinds, plus one query grind per level.
	let nonces = verifier
		.recorded
		.inputs
		.iter()
		.filter(|input| input.kind == "grind_nonce")
		.count();
	assert_eq!(nonces, 6, "one nonce per fold round, and one per level's query draw");

	verifier
		.check_opening(&shape, &setup, &opening)
		.expect("a ground transcript must satisfy the circuit built for that difficulty");
}

#[test]
fn a_corrupted_nonce_fails_the_proof_of_work() {
	// Invariant: the nonce is not advice the circuit takes on trust.
	// It is absorbed, and the draw it decides is asserted zero, so a nonce that did no work fails.
	//
	// Fixture state: the native ladder paying 8 bits before every fold challenge.
	// The first nonce byte is flipped.
	//
	// The tape opens with level 0's root, then its first round polynomial, then that nonce:
	//
	//     [0, 32)   level 0's Merkle root
	//     [32, 48)  the first round polynomial, one element because an MLE-check recovers the rest
	//     [48, 56)  the nonce, little-endian
	//
	//     before:  nonce  -> 8 sampled bits, all zero
	//     after:   nonce' -> 8 sampled bits that land on zero once in 256
	let shape = Shape {
		grinding: Grinding::new(8, 0),
		..NATIVE_SHAPE
	};
	let setup = shape.setup();
	let opening = shape.prove(&setup, 13);
	let verifier = shape.record(&setup);

	let nonce = DIGEST_BYTES + ELEMENT_BYTES;
	let paths = verifier.rejected(&shape, &setup, &opening, &corrupt(&opening.proof, nonce));
	assert!(
		paths.iter().any(|path| path == "grind"),
		"a nonce that met no difficulty must fail the proof-of-work assertion: {paths:?}"
	);
}

/// Returns `proof` with the low bit of one byte flipped.
fn corrupt(proof: &[u8], offset: usize) -> Vec<u8> {
	// One bit is the smallest change a sound protocol must still reject.
	let mut proof = proof.to_vec();
	proof[offset] ^= 1;
	proof
}

#[test]
fn a_corrupted_level_zero_root_is_rejected() {
	// Invariant: the root is what every opening below it is bound to, so it cannot be moved.
	//
	// Fixture state: the native ladder, with the first tape byte flipped.
	// Level 0's root is written before anything else the ladder sends.
	//
	//     before:  layer  -> fold -> the root the tape carries
	//     after:   layer  -> fold -> a root it no longer matches
	//
	// The evaluation point is observed rather than written, so it never reaches the tape.
	// Byte 0 is therefore inside the root.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = shape.prove(&setup, 3);
	let verifier = shape.record(&setup);

	let paths = verifier.rejected(&shape, &setup, &opening, &corrupt(&opening.proof, 0));
	assert!(
		paths.iter().any(|path| path == "layer"),
		"a corrupted root must fail the fold that binds the decommitted layer to it: {paths:?}"
	);
}

#[test]
fn a_corrupted_round_polynomial_moves_the_query_indices() {
	// Invariant: a round polynomial is a *received* message, so the challenger absorbs it.
	// Corrupting it moves the Fiat-Shamir state, and every challenge and index drawn after.
	//
	// Fixture state: the native ladder, with the first byte after the 32-byte root flipped.
	//
	//     before:  round value  -> challenge -> ... -> query indices -> the committed positions
	//     after:   round value' -> a different challenge from that round on, indices included
	//
	// The opened values and siblings on the tape are untouched.
	// The indices addressing them are not.
	// So the openings climb to the wrong entries of the decommitted layer.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = shape.prove(&setup, 4);
	let verifier = shape.record(&setup);

	let paths = verifier.rejected(&shape, &setup, &opening, &corrupt(&opening.proof, DIGEST_BYTES));
	assert!(
		paths.iter().any(|path| path == "opening"),
		"a moved query index must fail a Merkle opening: {paths:?}"
	);
}

#[test]
fn a_corrupted_residual_is_rejected() {
	// Invariant: the residual is bound two ways at once.
	// By the tree rebuilt over it, and by the last level's rows paired against it.
	//
	// Fixture state: the native ladder, with the last byte of the residual flipped.
	// The residual is sent before the last level's rows are opened.
	// So it is not the last byte of the tape.
	//
	//     before:  entry  -> rebuilt tree == root, and row pairing == batched rows
	//     after:   entry' -> a different root, and a pairing that no longer matches
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = shape.prove(&setup, 5);
	let verifier = shape.record(&setup);

	let offset = setup.last_residual_byte(opening.proof.len());
	let paths = verifier.rejected(&shape, &setup, &opening, &corrupt(&opening.proof, offset));
	assert!(
		paths.iter().any(|path| path == "vector"),
		"a corrupted residual must fail the rebuilt tree: {paths:?}"
	);
	assert!(
		paths.iter().any(|path| path == "assert_zero"),
		"and one of the two closing checks that read it: {paths:?}"
	);
}

#[test]
fn a_tampered_statement_is_rejected() {
	// Invariant: the statement is observed, so one opening's proof cannot be replayed on another.
	// Nothing about the proof bytes changes here.
	//
	// Fixture state: the native ladder, one honest proof, and one altered claim.
	//
	//     before:  the claim the prover observed  -> its Fiat-Shamir state
	//     after:   a claim one away               -> a different state from that point on
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = shape.prove(&setup, 6);
	let verifier = shape.record(&setup);

	// The claim is the last element of the statement, so its low word is the second from the end.
	let mut statement = opening.statement();
	let low = statement.len() - ELEMENT_WORDS;
	statement[low] = Word(statement[low].0 ^ 1);

	verifier
		.check(&shape, &setup, &statement, &opening.proof)
		.expect_err("a tampered claim must leave the circuit unsatisfied");
}

#[test]
fn every_corrupted_byte_across_the_tape_is_rejected() {
	// Invariant: no region of the tape is unchecked.
	// The tests above name the mechanism for three bytes, this one asserts there is no gap.
	//
	// Fixture state: the native ladder, one byte flipped at each of 32 evenly spread offsets.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = shape.prove(&setup, 7);
	let verifier = shape.record(&setup);

	let n_probes = 32;
	for probe in 0..n_probes {
		// Evenly spread, so every phase of the tape is covered.
		// The roots, the round polynomials, the layers, the openings, and the residual.
		let offset = probe * opening.proof.len() / n_probes;
		let paths = verifier.rejected(&shape, &setup, &opening, &corrupt(&opening.proof, offset));
		assert!(!paths.is_empty(), "byte {offset} must be checked by something");
	}
}

/// One ladder shape priced: what its proof costs in bytes, and what verifying it costs in gates.
struct Priced {
	/// Base-2 logarithm of the elements the cleartext residual holds.
	log_residual_dim: usize,
	/// Bytes of a real transcript for this ladder.
	proof_bytes: usize,
	/// AND constraints, which is where the circuit's hashing lands.
	and: usize,
	/// BMUL constraints, one per field multiplication the verifier performs.
	bmul: usize,
	/// Wires a replay has to fill, which is the proof data the circuit cannot derive.
	inputs: usize,
}

#[test]
fn the_cost_surface_over_residual_widths() {
	// Invariant: the residual is what an in-circuit verifier pays for twice.
	// Once in the tree it rebuilds over it, once in the rows it pairs against it.
	// So shrinking it is the lever, and this measures what the lever costs.
	//
	// Fixture state: four ladders over one 2^12 message, each one level deeper than the last.
	//
	//     lanes [2]        residual 2^10   one level,   nothing glued
	//     lanes [2,2]      residual 2^8    two levels,  one glued basis
	//     lanes [2,2,2]    residual 2^6    three,       two glued
	//     lanes [2,2,2,2]  residual 2^4    four,        three glued
	//
	// Every ladder commits the same message and opens the same rows per level.
	// So the residual is the only thing moving.
	let ladders: [&'static [usize]; 4] = [&[2], &[2, 2], &[2, 2, 2], &[2, 2, 2, 2]];

	println!(
		"\n{:>8} {:>9} {:>10} {:>10} {:>9} {:>9}",
		"levels", "residual", "bytes", "AND", "BMUL", "inputs"
	);
	let mut rows = Vec::new();
	for lanes in ladders {
		let shape = Shape {
			log_msg_cols: 10,
			lanes,
			..NATIVE_SHAPE
		};
		let setup = shape.setup();
		// A real transcript, so the byte column is measured rather than modelled.
		let opening = shape.prove(&setup, 20);
		let verifier = shape.record(&setup);
		// The circuit is only worth pricing if it accepts the proof it is priced against.
		verifier
			.check_opening(&shape, &setup, &opening)
			.expect("every ladder in the sweep must verify its own proof");

		let priced = Priced {
			log_residual_dim: setup.params.log_residual_dim(),
			proof_bytes: opening.proof.len(),
			and: verifier.stat.n_and_constraints,
			bmul: verifier.stat.n_bmul_constraints,
			inputs: verifier.recorded.inputs.len(),
		};
		println!(
			"{:>8} {:>9} {:>10} {:>10} {:>9} {:>9}",
			lanes.len(),
			format!("2^{}", priced.log_residual_dim),
			priced.proof_bytes,
			priced.and,
			priced.bmul,
			priced.inputs,
		);
		rows.push(priced);
	}

	// Each extra level folds two more variables away before the clear text starts.
	for pair in rows.windows(2) {
		assert!(
			pair[1].log_residual_dim < pair[0].log_residual_dim,
			"each ladder must leave a shallower residual: 2^{} then 2^{}",
			pair[0].log_residual_dim,
			pair[1].log_residual_dim
		);
	}

	// A residual is hashed one element to a leaf.
	// So quartering its width more than halves the hashing the whole verification does.
	assert!(
		2 * rows[1].and < rows[0].and,
		"quartering a wide residual must more than halve the AND constraints: {} against {}",
		rows[1].and,
		rows[0].and
	);

	// It stops paying once the residual is narrow.
	// The per-level work an extra level adds then outweighs the tree it removes.
	assert!(
		2 * rows[3].and > rows[2].and,
		"a narrow residual must stop halving: {} against {}",
		rows[3].and,
		rows[2].and
	);

	// And the tape turns around at the same place, so the deepest ladder is not the smallest proof.
	assert!(
		rows[3].proof_bytes > rows[2].proof_bytes,
		"the last level must cost bytes rather than save them: {} against {}",
		rows[3].proof_bytes,
		rows[2].proof_bytes
	);

	// The widest residual is the most expensive on every axis at once.
	// So recursing at all pays for itself in-circuit, whatever the byte count says.
	assert!(
		rows[0].bmul > rows[3].bmul && rows[0].proof_bytes > rows[3].proof_bytes,
		"a one-level ladder must cost both more multiplications and more bytes than a four-level \
		 one"
	);

	// The wires a replay fills track the same term, since the residual arrives in the clear.
	assert!(
		rows[0].inputs > rows[2].inputs,
		"a wide residual must put more proof data on the tape: {} against {}",
		rows[0].inputs,
		rows[2].inputs
	);
}
