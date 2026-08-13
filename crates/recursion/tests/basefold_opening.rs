// Copyright 2026 The Binius Developers

//! A polynomial-commitment opening proved natively, then verified inside a Binius64 circuit.
//!
//! ```text
//!   prover:    native, writing one real transcript
//!   verifier:  the same code over the builder channel, emitting gates instead of checking values
//! ```
//!
//! A satisfied constraint system is then the statement "this transcript opens this claim".
//! An outer proof can check that statement, which is what makes this one step of recursion.
//!
//! # What enters the circuit
//!
//! Two things, and nothing else:
//!
//! ```text
//!   the proof      one recorded wire per value the verifier reads off the tape
//!   the statement  the evaluation point and the claim, on public wires
//! ```
//!
//! Challenges, query indices, folded values and recomputed digests are all gate outputs.
//!
//! # Build once, verify many
//!
//! Every shape the circuit depends on is settled before a proof exists.
//! The code rate, the fold arities, the oracle layout, the tree depths, the query count.
//! So one circuit serves every opening of that shape, and a test below verifies two with one.
//!
//! # The statement is observed
//!
//! The native round trip in `binius-iop-prover` hands the point and the claim out of band.
//! It can, being a unit test rather than a protocol.
//! Here both are written to the transcript, by the prover and by each verifier alike.
//!
//! That is what lets them be wires rather than build-time constants, as "build once" needs.
//! It also binds them: a proof of one statement cannot be replayed against another.
//!
//! The point is observed before the commitment, since it is known up front.
//! The claim is observed after the masking challenge.
//! It is a claim about the masked polynomial, so it does not exist until that challenge does.

use std::iter;

use binius_compute::GlobalAllocator;
use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, PackedBinaryGhash1x128b};
use binius_frontend::{CircuitStat, MAX_ASSERTION_FAILURES, PopulateError, Wire};
use binius_hash::{StdDigest, StdHashSuite};
use binius_iop::{
	basefold::verify_mlecheck_basefold, channel::OracleSpec, fri::FRIParams,
	merkle_channel::MerkleIPVerifierChannel, merkle_tree::BinaryMerkleTreeScheme,
};
use binius_iop_prover::{
	basefold::prove_mlecheck_basefold,
	fri::{self, FRIFoldProver, MaskedCodeword},
	merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel},
	merkle_tree::prover::BinaryMerkleTreeProver,
};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_math::{
	inner_product::inner_product_buffers,
	line::extrapolate_line_packed,
	multilinear::eq::eq_ind_partial_eval,
	ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
	test_utils::{random_field_buffer, random_scalars},
};
use binius_recursion::{Binius64BuilderChannel, Recorded, WitnessFillerChannel};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
use binius_utils::rayon::prelude::*;
use rand::{SeedableRng, rngs::StdRng};

/// The Fiat-Shamir challenger the in-circuit challenger reproduces.
type StdChallenger = HasherChallenger<StdDigest>;

/// The packed field the native prover runs over.
type P = PackedBinaryGhash1x128b;

/// Words one field element occupies on the transcript, low half first.
const ELEMENT_WORDS: usize = 2;

/// Bytes one SHA-256 digest occupies on the tape, which is what a commitment root is.
const DIGEST_BYTES: usize = 32;

/// Field elements one committed leaf holds.
///
/// A zero-knowledge oracle commits the polynomial interleaved with its mask.
/// One leaf covers one `(pi || omega)` coset, so the tree has one leaf per codeword position.
const LEAF_ELEMENTS: usize = 2;

/// The transcript words one field element serializes to.
fn element_words(value: B128) -> [Word; ELEMENT_WORDS] {
	let value = u128::from(value);
	[
		Word::from_u64(value as u64),
		Word::from_u64((value >> 64) as u64),
	]
}

/// The shape of a single-oracle zero-knowledge opening.
///
/// Every field is fixed before a proof exists, so a shape is exactly what a circuit is built for.
#[derive(Clone, Copy, Debug)]
struct Shape {
	/// Variables of the committed multilinear.
	n_vars: usize,
	/// Base-2 logarithm of the inverse Reed-Solomon rate.
	log_inv_rate: usize,
	/// FRI consistency queries the verifier makes.
	n_test_queries: usize,
}

/// The shape the native round trip in the prover crate uses.
const NATIVE_SHAPE: Shape = Shape {
	n_vars: 8,
	log_inv_rate: 1,
	n_test_queries: 32,
};

/// What a shape fixes ahead of any proof: the fold parameters and the transform that encodes.
struct Setup {
	/// The fold layout: code dimension, per-oracle shapes, fold arities and query count.
	fri_params: FRIParams<B128>,
	/// The additive NTT the prover encodes with.
	ntt: NeighborsLastSingleThread<GaoMateerOnTheFly<B128>>,
}

impl Shape {
	/// Depth of the Merkle tree over the committed interleaved codeword, one coset per leaf.
	const fn codeword_depth(&self) -> usize {
		self.n_vars + self.log_inv_rate
	}

	/// Transcript words the statement occupies: the point, then the claim.
	const fn statement_words(&self) -> usize {
		(self.n_vars + 1) * ELEMENT_WORDS
	}

	/// Derives the fold parameters and the encoding transform, neither of which needs a witness.
	fn setup(&self) -> Setup {
		// Only the scheme's digest and node-cost model are consulted, so no tree is built.
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();

		// The domain must span the interleaved codeword.
		// One extra variable for the mask that buys zero-knowledge, plus the rate.
		let domain_context = GaoMateerOnTheFly::generate(self.n_vars + 1 + self.log_inv_rate);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		// One zero-knowledge oracle makes the batch parameters valid for the masked encoder too.
		// That means one interleaved batch dimension, and a code dimension of `n_vars`.
		let (fri_params, _) = FRIParams::optimal_for_batch(
			&scheme,
			&[OracleSpec::new_zk(self.n_vars)],
			self.log_inv_rate,
			self.n_test_queries,
		);
		Setup { fri_params, ntt }
	}
}

/// One opening proved natively: the statement it proves, and the transcript proving it.
struct Opening {
	/// The claimed evaluation `pi'(r)`.
	eval_claim: B128,
	/// The point `r`, low-to-high variable order.
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

/// Proves one opening of `shape`, drawing the witness and the point from `seed`.
///
/// The masking challenge is sampled off the transcript.
/// A verifier therefore derives the same value rather than being handed it.
fn prove(shape: &Shape, setup: &Setup, seed: u64) -> Opening {
	// One seed fixes both the committed polynomial and the point, so a transcript is reproducible.
	let mut rng = StdRng::seed_from_u64(seed);
	let witness = random_field_buffer::<P>(&mut rng, shape.n_vars);
	let eval_point: Vec<B128> = random_scalars(&mut rng, shape.n_vars);

	// A masked encoding interleaves the polynomial with a random mask of the same size.
	// That is what stops the opened cosets leaking the polynomial.
	let merkle_prover = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();
	let MaskedCodeword { codeword, mask } =
		fri::encode_masked(&setup.fri_params, 0, &setup.ntt, witness.to_ref(), &mut rng);

	let mut transcript = ProverTranscript::new(StdChallenger::default());
	let mut channel =
		ProverMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::with_merkle_prover(
			&mut transcript,
			merkle_prover,
		);

	// The point is known before anything is committed, so it is bound first.
	let point_words = eval_point
		.iter()
		.flat_map(|value| element_words(*value))
		.collect::<Vec<_>>();
	WordIPProverChannel::<B128>::observe_words(&mut channel, &point_words);

	let commitment = channel.send_merkle_commitment(codeword.to_ref(), LEAF_ELEMENTS);

	// Fold the interleaved (pi || omega) codeword to pi' = (1 - gamma) pi + gamma omega.
	let gamma: B128 = IPProverChannel::sample(&mut channel);
	let mut witness_prime = witness.clone();
	let broadcast = P::broadcast(gamma);
	(witness_prime.as_mut(), mask.as_ref())
		.into_par_iter()
		.for_each(|(w, &m)| *w = extrapolate_line_packed(*w, m, broadcast));

	// The claim is the folded polynomial against the equality indicator, which is pi'(r).
	// It exists only once gamma does, so it is bound here rather than up front.
	let eval_claim = inner_product_buffers(&witness_prime, &eq_ind_partial_eval::<P>(&eval_point));
	WordIPProverChannel::<B128>::observe_words(&mut channel, &element_words(eval_claim));

	// The sum-check rounds and the codeword folding advance together, so the transcript carries
	// both.
	let fri_folder =
		FRIFoldProver::new_batch(&setup.fri_params, &setup.ntt, vec![(codeword, commitment)]);
	prove_mlecheck_basefold(
		witness_prime,
		&eval_point,
		eval_claim,
		Some(gamma),
		&[],
		fri_folder,
		&mut channel,
		&GlobalAllocator,
	);
	// Hand the transcript back, releasing the channel's borrow of it.
	channel.into_transcript();

	Opening {
		eval_claim,
		eval_point,
		proof: transcript.finalize(),
	}
}

/// A circuit that verifies any opening of one fixed shape.
struct VerifierCircuit {
	/// The compiled circuit and the wires a replay fills.
	recorded: Recorded,
	/// One public wire per statement word, each equal to what the verifier observed.
	public: Vec<Wire>,
	/// Constraint counts and trace size.
	stat: CircuitStat,
}

/// Records the verifier for `shape` as a circuit, by running it over the builder channel.
///
/// No proof reaches this, and no statement either:
/// `observe_words` reads only the *length* of what it is handed.
/// Placeholder words therefore allocate the wires a real statement later fills.
fn record(shape: &Shape, setup: &Setup) -> VerifierCircuit {
	let mut channel = Binius64BuilderChannel::new();

	// Placeholders, for their count alone.
	// The values never reach a gate.
	let placeholder = vec![Word::ZERO; shape.statement_words()];
	let statement = channel.observe_words(&placeholder[..shape.n_vars * ELEMENT_WORDS]);
	let eval_point = channel.pack_words(&statement);

	// The root is read next, so everything the query phase later opens is already observed.
	let commitment = channel
		.recv_merkle_commitment(LEAF_ELEMENTS, shape.codeword_depth())
		.expect("reading a commitment cannot fail");
	let gamma = IPVerifierChannel::<B128>::sample(&mut channel);

	let claim_words = channel.observe_words(&placeholder[shape.n_vars * ELEMENT_WORDS..]);
	let eval_claim = channel.pack_words(&claim_words)[0].clone();

	// Every check the verifier makes becomes a constraint rather than a comparison.
	// So the call cannot fail while the circuit is being built.
	verify_mlecheck_basefold(
		&setup.fri_params,
		&[commitment],
		eval_claim,
		&eval_point,
		Some(gamma),
		&[],
		&mut channel,
	)
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
		let point = channel.observe_words(&statement[..shape.n_vars * ELEMENT_WORDS]);
		let eval_point = channel.pack_words(&point);
		let commitment = channel
			.recv_merkle_commitment(LEAF_ELEMENTS, shape.codeword_depth())
			.expect("the tape carries a commitment");
		let gamma: B128 = IPVerifierChannel::sample(&mut channel);
		let claim = channel.observe_words(&statement[shape.n_vars * ELEMENT_WORDS..]);
		let eval_claim = channel.pack_words(&claim)[0];

		verify_mlecheck_basefold(
			&setup.fri_params,
			&[commitment],
			eval_claim,
			&eval_point,
			Some(gamma),
			&[],
			&mut channel,
		)
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
}

#[test]
fn a_native_opening_verifies_in_circuit() {
	// Invariant: a transcript the native prover wrote satisfies the circuit built from the
	// verifier.
	//
	// Fixture state: the native shape of 8 variables, rate 1 and 32 queries, and one proof.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 0);
	// Nothing about the proof reaches the builder, so the order of these two lines is immaterial.
	let verifier = record(&shape, &setup);

	println!(
		"basefold verifier: {} gates, {} AND, {} BMUL, {} ZERO, {} committed words, {} recorded inputs",
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
	// Invariant: one compiled circuit verifies every opening of its shape.
	//
	// Fixture state: two proofs of the native shape, from two different seeds.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();

	// Different witness and different evaluation point, so a different transcript throughout.
	let first = prove(&shape, &setup, 1);
	let second = prove(&shape, &setup, 2);
	// Two openings that happened to coincide would make the rest of the test vacuous.
	assert_ne!(first.eval_point, second.eval_point, "the two openings must differ");
	assert_ne!(first.proof, second.proof, "the two transcripts must differ");
	// Equal lengths are the observable half of "the layout depends only on the shape".
	assert_eq!(
		first.proof.len(),
		second.proof.len(),
		"one shape means one tape length, whatever the witness"
	);

	// Built once, before either proof is looked at.
	let verifier = record(&shape, &setup);
	for (i, opening) in [&first, &second].into_iter().enumerate() {
		verifier
			.check_opening(&shape, &setup, opening)
			.unwrap_or_else(|error| panic!("opening {i} must satisfy the shared circuit: {error}"));
	}
}

/// Returns `proof` with the low bit of one byte flipped.
fn corrupt(proof: &[u8], offset: usize) -> Vec<u8> {
	// One bit is the smallest change a sound protocol must still reject.
	let mut proof = proof.to_vec();
	proof[offset] ^= 1;
	proof
}

/// Asserts the circuit rejects `proof`, and returns the subcircuits whose assertions failed.
fn rejected(
	verifier: &VerifierCircuit,
	shape: &Shape,
	setup: &Setup,
	opening: &Opening,
	proof: &[u8],
) -> Vec<String> {
	let error = verifier
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

#[test]
fn a_corrupted_commitment_root_is_rejected() {
	// Invariant: the root is what every opening below it is bound to, so it cannot be moved.
	//
	// Fixture state: the native shape, with the first tape byte flipped.
	// The root is written first, so byte 0 is inside it.
	//
	//     before:  layer  -> fold -> the root the tape carries
	//     after:   layer  -> fold -> a root it no longer matches
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 3);
	let verifier = record(&shape, &setup);

	let paths = rejected(&verifier, &shape, &setup, &opening, &corrupt(&opening.proof, 0));
	assert!(
		paths.iter().any(|path| path == "layer"),
		"a corrupted root must fail the fold that binds the decommitted layer to it: {paths:?}"
	);
}

#[test]
fn a_corrupted_round_polynomial_moves_the_query_indices() {
	// Invariant: a round polynomial is an *observed* message.
	// Corrupting it moves the Fiat-Shamir state, and every challenge and index drawn after.
	//
	// Fixture state: the native shape, with the first byte after the 32-byte root flipped.
	//
	//     before:  round value  -> challenge -> ... -> query indices -> the committed positions
	//     after:   round value' -> a different challenge from that round on, indices included
	//
	// The opened values and siblings on the tape are untouched, the indices addressing them are
	// not. So the openings climb to the wrong entries of the decommitted layer.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 4);
	let verifier = record(&shape, &setup);

	let paths =
		rejected(&verifier, &shape, &setup, &opening, &corrupt(&opening.proof, DIGEST_BYTES));
	assert!(
		paths.iter().any(|path| path == "opening"),
		"a moved query index must fail a Merkle opening: {paths:?}"
	);
}

#[test]
fn a_corrupted_terminal_codeword_is_rejected() {
	// Invariant: the terminal codeword is bound two ways at once.
	// By the tree rebuilt over it, and by the queries folded down to it.
	//
	// Fixture state: the native shape, with the last tape byte flipped.
	// It is the last thing read.
	//
	//     before:  entry  -> rebuilt tree == root, and folded query == entry
	//     after:   entry' -> a different root, and a folded query that no longer matches
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 5);
	let verifier = record(&shape, &setup);

	let last = opening.proof.len() - 1;
	let paths = rejected(&verifier, &shape, &setup, &opening, &corrupt(&opening.proof, last));
	assert!(
		paths.iter().any(|path| path == "vector"),
		"a corrupted terminal entry must fail the rebuilt tree: {paths:?}"
	);
	assert!(
		paths.iter().any(|path| path == "assert_zero"),
		"and the fold equality that reads it: {paths:?}"
	);
}

#[test]
fn a_tampered_statement_is_rejected() {
	// Invariant: the statement is observed, so one opening's proof cannot be replayed on another.
	// Nothing about the proof bytes changes here.
	//
	// Fixture state: the native shape, one honest proof, and one altered claim.
	//
	//     before:  the claim the prover observed  -> its Fiat-Shamir state
	//     after:   a claim one away               -> a different state from that point on
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 6);
	let verifier = record(&shape, &setup);

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
	// Fixture state: the native shape, one byte flipped at each of 32 evenly spread offsets.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 7);
	let verifier = record(&shape, &setup);

	let n_probes = 32;
	for probe in 0..n_probes {
		// Evenly spread, so every phase of the tape is covered: the root, the round polynomials,
		// the decommitted layers, the opened leaves and branches, and the terminal codeword.
		let offset = probe * opening.proof.len() / n_probes;
		let paths = rejected(&verifier, &shape, &setup, &opening, &corrupt(&opening.proof, offset));
		assert!(!paths.is_empty(), "byte {offset} must be checked by something");
	}
}

#[test]
fn the_cost_surface_over_shapes() {
	// Invariant: cost rises with the query count and the committed size, neither one naively.
	//
	// Fixture state: five shapes, each one lever off the native shape.
	// Building a circuit needs no proof, so this measures the whole surface without proving
	// anything.
	let shapes = [
		NATIVE_SHAPE,
		Shape {
			n_vars: 10,
			..NATIVE_SHAPE
		},
		Shape {
			log_inv_rate: 2,
			..NATIVE_SHAPE
		},
		Shape {
			n_test_queries: 16,
			..NATIVE_SHAPE
		},
		Shape {
			n_test_queries: 64,
			..NATIVE_SHAPE
		},
	];

	println!(
		"\n{:<10} {:>6} {:>8} {:>10} {:>9} {:>9}",
		"n_vars", "rate", "queries", "AND", "BMUL", "inputs"
	);
	let mut totals = Vec::new();
	for shape in shapes {
		let verifier = record(&shape, &shape.setup());
		println!(
			"{:<10} {:>6} {:>8} {:>10} {:>9} {:>9}",
			shape.n_vars,
			shape.log_inv_rate,
			shape.n_test_queries,
			verifier.stat.n_and_constraints,
			verifier.stat.n_bmul_constraints,
			verifier.recorded.inputs.len(),
		);
		totals.push((shape, verifier.stat.n_and_constraints));
	}

	// Pinning the variable count too, so the query rows are not confused with the deeper shape.
	let and_of = |queries: usize| {
		totals
			.iter()
			.find(|(shape, _)| {
				shape.n_test_queries == queries && shape.n_vars == NATIVE_SHAPE.n_vars
			})
			.map(|&(_, and)| and)
			.expect("the sweep covers this query count")
	};

	// Each query buys its own climb and its own leaf hash, so more of them must cost more.
	// The rise is sublinear: the scheme deepens the shared layer as the query count grows,
	// which shortens every climb.
	assert!(and_of(64) > and_of(32), "more queries must cost more");
	assert!(and_of(32) > and_of(16), "more queries must cost more");

	// Two more variables deepen every tree by two levels, paid once per query.
	let deeper = totals
		.iter()
		.find(|(shape, _)| shape.n_vars == 10)
		.map(|&(_, and)| and)
		.expect("the sweep covers n_vars = 10");
	// Doubling would mean the cost tracks the committed size rather than its logarithm.
	assert!(
		deeper < 2 * and_of(32),
		"four times the committed size must not double the verifier: {deeper} against {}",
		and_of(32)
	);
}
