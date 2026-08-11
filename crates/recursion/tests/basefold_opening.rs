// Copyright 2026 The Binius Developers

//! A polynomial-commitment opening proved natively, then verified inside a Binius64 circuit.
//!
//! # The two halves
//!
//! The proof system already has a native commit-prove-verify round trip, and both ends are reused
//! here:
//!
//! ```text
//!     prover:   native and unchanged, writing one real transcript
//!     verifier: the same code, over a channel that emits gates instead of checking values
//! ```
//!
//! A satisfied constraint system is then the statement "this transcript opens this claim".
//! An outer proof can check that statement, which is what makes this one step of recursion.
//!
//! # What is an input
//!
//! Two things enter the circuit, and nothing else:
//!
//! ```text
//!     the proof byte stream   one witness wire per eight-byte proof word
//!     the inner statement     the claimed evaluation and the point, on inout wires
//! ```
//!
//! Challenges, folded values and recomputed digests are all gate outputs.
//!
//! The statement deliberately does not enter as a build-time constant.
//! A constant would fold one opening's numbers into the gates, destroying the property below.
//!
//! # Build once, verify many
//!
//! Every shape the circuit depends on is settled before any proof exists:
//!
//! - the code rate and the fold arities
//! - the oracle layout and the Merkle tree depths
//! - the number of consistency queries
//!
//! So one circuit serves every opening of that shape.
//! That is what makes recursion practical, since the expensive compilation is paid once per
//! protocol rather than once per proof.
//!
//! # Measurement
//!
//! Two of the tests print a constraint table and a phase breakdown.
//! Run them with `--nocapture` to see the output.
//!
//! At `n_vars = 8`, `log_inv_rate = 1` and 32 queries, over a 13,536-byte proof, the measured cost
//! is 229,675 AND, 14,887 BMUL, 71,362 ZERO and 524,288 committed words, split as:
//!
//! ```text
//!     phase                          AND     BMUL    AND %
//!     Merkle openings             212315     9216    92.4%
//!     Fiat-Shamir challenger        9643        0     4.2%
//!     terminal codeword             7773        0     3.4%
//!     FRI folding and equalities       0     5655     0.0%
//!     MLE-check arithmetic             0       16     0.0%
//! ```
//!
//! So the verifier is SHA-256 and almost nothing else:
//!
//! - the three hashing phases are the whole AND column
//! - Merkle openings alone are 92% of it
//! - the field arithmetic that the fold and the sum-check actually reduce spends no AND at all,
//!   only 5,671 BMUL
//!
//! The AND column is what binds.
//!
//! # The cost surface
//!
//! One sweep moves a single lever at a time off that shape:
//!
//! ```text
//!     shape                            AND    vs base
//!     n_vars 8,  rate 1, 32 queries  229675         --
//!     n_vars 10, rate 1, 32 queries  346681      +51%
//!     n_vars 8,  rate 2, 32 queries  285585      +24%
//!     n_vars 8,  rate 1, 16 queries  146249      -36%
//!     n_vars 8,  rate 1, 64 queries  280193      +22%
//! ```
//!
//! Two readings are worth keeping.
//!
//! 1. **Queries are sublinear** — 16 to 32 queries costs +57% but 32 to 64 only +22%, because the
//!    shared decommitment layer deepens as the query count grows and every climb gets shorter.
//! 2. **Committed size is not flat** — four times the committed size costs 51% more rather than a
//!    few percent, so growth is logarithmic with a steep constant.
//!
//! Doubling the query count is therefore cheaper than it looks, and the layer depth is the reason.
//!
//! Every extra Merkle level is paid once per query across every tree in the fold schedule.
//! Here the terminal codeword grew with it, from 7,773 to 26,781 AND.
//!
//! # Where this sits
//!
//! Against the 2.5M-5M AND estimate for a full recursive verifier, one opening of this shape is
//! around a tenth of the lower bound.
//!
//! A production shape costs more: more queries, deeper trees, several oracles, and the
//! constraint-system checks that sit above the opening.
//! The levers are visibly the same three, in this order:
//!
//! 1. fewer SHA-256 compressions per Merkle level
//! 2. fewer Merkle levels
//! 3. the layer-depth trade-off that sets how many levels each query climbs
//!
//! The sweep above is the surface that last trade-off would be tuned against.

use std::iter;

use binius_compute::GlobalAllocator;
use binius_core::{VerificationError, word::Word};
use binius_field::{BinaryField128bGhash as B128, Field, PackedBinaryGhash1x128b};
use binius_frontend::{
	Circuit, CircuitBuilder, CircuitStat, MAX_ASSERTION_FAILURES, PopulateError,
};
use binius_hash::{StdDigest, StdHashSuite, sha256::Sha256HashSuite};
use binius_iop::{
	basefold::verify_mlecheck_basefold,
	channel::OracleSpec,
	fri::FRIParams,
	merkle_channel::MerkleIPVerifierChannel,
	merkle_tree::{BinaryMerkleTreeScheme, MerkleTreeScheme},
};
use binius_iop_prover::{
	basefold::prove_mlecheck_basefold,
	fri::{self, FRIFoldProver, MaskedCodeword},
	merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel},
	merkle_tree::prover::BinaryMerkleTreeProver,
};
use binius_ip::{
	channel::{IPVerifierChannel, WordIPVerifierChannel},
	mlecheck,
	sumcheck::RoundCoeffs,
};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace,
	inner_product::inner_product_buffers,
	line::extrapolate_line_packed,
	multilinear::eq::eq_ind_partial_eval,
	ntt::{AdditiveNTT, NeighborsLastSingleThread, domain_context::GenericOnTheFly},
	test_utils::{random_field_buffer, random_scalars},
};
use binius_recursion::{
	challenger::Sha256Challenger,
	channel::{
		ChannelWord, MerkleCommitment, MerkleVerifierChannel, ProofLayout, ProofRead, ReadKind,
	},
	elem::Elem,
	merkle::{self, DIGEST_WORDS, Digest, ELEMENT_WORDS, Element, populate_element},
};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use binius_utils::rayon::prelude::*;
use rand::{SeedableRng, rngs::StdRng};

/// The Fiat-Shamir challenger the in-circuit challenger reproduces.
type StdChallenger = HasherChallenger<StdDigest>;

/// The packed field the native prover runs over.
type P = PackedBinaryGhash1x128b;

/// The prover channel driving the native half of a round trip.
type NativeProverChannel<'a> = ProverMerkleTranscriptChannel<
	&'a mut ProverTranscript<StdChallenger>,
	StdChallenger,
	B128,
	StdHashSuite,
>;

/// Field elements one committed leaf holds.
///
/// A zero-knowledge oracle commits the polynomial interleaved with its mask, so one leaf covers one
/// `(pi || omega)` coset and the tree has one leaf per codeword position.
const LEAF_ELEMENTS: usize = 2;

/// The shape of a single-oracle zero-knowledge opening.
///
/// Every field is fixed before a proof exists, so a shape is exactly what a circuit is built for.
#[derive(Clone, Copy, Debug)]
struct Shape {
	/// Variables of the committed multilinear.
	n_vars: usize,
	/// log2 the inverse Reed-Solomon rate.
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
	ntt: NeighborsLastSingleThread<GenericOnTheFly<B128>>,
}

impl Shape {
	/// Depth of the Merkle tree over the committed interleaved codeword: one coset per leaf.
	const fn codeword_depth(&self) -> usize {
		// One leaf per codeword position, and the codeword is 2^log_inv_rate times the message.
		self.n_vars + self.log_inv_rate
	}

	/// Derives the fold parameters and the encoding transform, neither of which needs a witness.
	fn setup(&self) -> Setup {
		// Only the scheme's digest and node-cost model are consulted here, so no tree is built.
		let scheme = BinaryMerkleTreeScheme::<B128, Sha256HashSuite>::new();

		// The evaluation domain must span the interleaved codeword: one extra variable for the
		// mask that buys zero-knowledge, plus the rate.
		let subspace = BinarySubspace::with_dim(self.n_vars + 1 + self.log_inv_rate);
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		// A single zero-knowledge oracle makes the combined batch parameters valid for the masked
		// encoder too: one interleaved batch dimension, and a code dimension of `n_vars`.
		let (fri_params, _) = FRIParams::optimal_for_batch(
			ntt.domain_context(),
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

/// Proves one opening of `shape`, drawing the witness and the point from `seed`.
///
/// This is the native round trip's prover half, unchanged.
/// The masking challenge is sampled off the transcript, so the circuit derives the same value
/// rather than being handed it.
fn prove(shape: &Shape, setup: &Setup, seed: u64) -> Opening {
	// One seed fixes both the committed polynomial and the point, so a transcript is reproducible.
	let mut rng = StdRng::seed_from_u64(seed);
	let witness = random_field_buffer::<P>(&mut rng, shape.n_vars);
	let eval_point: Vec<B128> = random_scalars(&mut rng, shape.n_vars);

	// Phase 1: encode.
	//
	// A masked encoding interleaves the polynomial with a random mask of the same size, which is
	// what stops the opened cosets leaking the polynomial.
	let merkle_prover = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();
	let MaskedCodeword { codeword, mask } =
		fri::encode_masked(&setup.fri_params, 0, &setup.ntt, witness.to_ref(), &mut rng);

	// Phase 2: commit.
	//
	// The transcript is the Fiat-Shamir tape, and the commitment is the first thing written to it.
	let mut transcript = ProverTranscript::new(StdChallenger::default());
	let mut channel = NativeProverChannel::with_merkle_prover(&mut transcript, merkle_prover);
	let commitment = channel.send_merkle_commitment(codeword.to_ref(), LEAF_ELEMENTS);

	// Fold the interleaved (pi || omega) codeword to pi' = (1 - gamma) pi + gamma omega.
	let gamma: B128 = IPProverChannel::sample(&mut channel);
	let mut witness_prime = witness.clone();
	let broadcast = P::broadcast(gamma);
	(witness_prime.as_mut(), mask.as_ref())
		.into_par_iter()
		.for_each(|(w, &m)| *w = extrapolate_line_packed(*w, m, broadcast));

	// The claim is the folded polynomial against the equality indicator at the point, which is
	// exactly pi'(r).
	let eval_claim = inner_product_buffers(&witness_prime, &eq_ind_partial_eval::<P>(&eval_point));

	// Phase 3: prove.
	//
	// The sum-check rounds and the codeword folding advance together, so the transcript ends up
	// carrying both.
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

	// Finalizing seals the tape into the byte stream the circuit reads.
	Opening {
		eval_claim,
		eval_point,
		proof: transcript.finalize(),
	}
}

/// Why a populated witness fails to satisfy the verifier circuit.
///
/// Population evaluates the gates and checks the assertions, so an unsatisfiable proof normally
/// surfaces there.
/// Constraint verification is the independent second opinion.
#[derive(Debug)]
enum Unsatisfied {
	/// Assertions the witness filler found failing, with the circuit paths they sit under.
	Assertions(PopulateError),
	/// A constraint the value vector does not satisfy.
	Constraints(VerificationError),
}

/// A circuit that verifies any opening of one fixed shape.
struct VerifierCircuit {
	/// The compiled circuit.
	circuit: Circuit,
	/// Where the proof byte stream lands.
	layout: ProofLayout,
	/// The evaluation claim, on inout wires.
	eval_claim: Element,
	/// The evaluation point, one element per variable, on inout wires.
	eval_point: Vec<Element>,
	/// The Fiat-Shamir schedule the verifier drove, for the phase breakdown.
	challenger_ops: Vec<ChallengerOp>,
	/// The Merkle operations the verifier asked for, for the phase breakdown.
	merkle_ops: Vec<MerkleOp>,
	/// Constraint counts and trace size.
	stat: CircuitStat,
}

impl VerifierCircuit {
	/// Builds the verifier for `shape` by running the real verifier against the circuit channel.
	///
	/// The channel operations below are the native round trip's verifier half, in the same order.
	/// The Fiat-Shamir state is a function of that order, so it is not free to change.
	fn build(shape: &Shape, setup: &Setup) -> Self {
		// Two layers of channel: the inner one emits gates, the outer one only takes notes.
		let builder = CircuitBuilder::new();
		let mut inner = MerkleVerifierChannel::new(&builder);
		let mut channel = Recording::new(&mut inner);

		// The statement enters on inout wires, one pair per element for the low and high halves.
		// As build-time constants these numbers would fold into the gates, tying the circuit to one
		// opening.
		let claim_wires: Element = [builder.add_inout(), builder.add_inout()];
		let point_wires: Vec<Element> = (0..shape.n_vars)
			.map(|_| [builder.add_inout(), builder.add_inout()])
			.collect();
		let eval_claim = Elem::new(&builder, claim_wires[0], claim_wires[1]);
		let eval_point: Vec<Elem> = point_wires
			.iter()
			.map(|&[lo, hi]| Elem::new(&builder, lo, hi))
			.collect();

		// The root is read first, so everything the query phase later opens is already observed.
		let commitment = channel
			.recv_merkle_commitment(LEAF_ELEMENTS, shape.codeword_depth())
			.expect("reading a commitment cannot fail");

		// The masking challenge, drawn from the same Fiat-Shamir state the prover drew it from.
		let gamma = IPVerifierChannel::<B128>::sample(&mut channel);

		// Every check the verifier makes becomes an assertion rather than a comparison, so the call
		// cannot fail while the circuit is being built.
		verify_mlecheck_basefold(
			&setup.fri_params,
			&[commitment],
			eval_claim,
			&eval_point,
			Some(gamma),
			&[],
			&mut channel,
		)
		.expect("the circuit channel rejects nothing: every check it makes becomes a constraint");

		// Both schedules are complete once the verifier has run, and so is the proof layout.
		let (challenger_ops, merkle_ops) = channel.finish();
		let layout = inner.finish();
		let circuit = builder.build();
		let stat = CircuitStat::collect(&circuit);
		Self {
			circuit,
			layout,
			eval_claim: claim_wires,
			eval_point: point_wires,
			challenger_ops,
			merkle_ops,
			stat,
		}
	}

	/// Populates the circuit from a statement and a proof, and reports whether it is satisfied.
	fn check(
		&self,
		eval_claim: B128,
		eval_point: &[B128],
		proof: &[u8],
	) -> Result<(), Unsatisfied> {
		let mut w = self.circuit.new_witness_filler();

		// Input 1: the proof tape, one word per eight bytes, in the order the verifier read it.
		self.layout
			.populate(&mut w, proof)
			.expect("the layout reads exactly the proof the prover wrote");

		// Input 2: the statement, as sixteen little-endian bytes per element.
		populate_element(&mut w, &self.eval_claim, u128::from(eval_claim));
		for (element, value) in iter::zip(&self.eval_point, eval_point) {
			populate_element(&mut w, element, u128::from(*value));
		}

		// Population derives every other wire and reports the assertions that came out false.
		self.circuit
			.populate_wire_witness(&mut w)
			.map_err(Unsatisfied::Assertions)?;

		// Re-checking the finished witness against the constraint system is the second opinion: it
		// reads the compiled constraints rather than the gate graph.
		self.circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.map_err(Unsatisfied::Constraints)
	}

	/// Populates and checks one natively proved opening.
	fn check_opening(&self, opening: &Opening) -> Result<(), Unsatisfied> {
		// An honest opening supplies its own statement, so the two inputs come from one place.
		self.check(opening.eval_claim, &opening.eval_point, &opening.proof)
	}
}

// Noting the schedule, so the cost can be attributed to a phase

/// One Fiat-Shamir event the verifier caused, in the order it caused it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChallengerOp {
	/// `n` proof words fed to the hasher.
	Observe(usize),
	/// One 128-bit challenge drawn.
	SampleB128,
	/// A `bits`-wide index word drawn.
	SampleBits(usize),
}

/// One Merkle operation the verifier asked of the channel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MerkleOp {
	/// Openings of a `leaf_size`-element, depth-`depth` tree at `n_queries` sampled indices.
	Openings {
		/// Field elements one leaf holds.
		leaf_size: usize,
		/// Depth of the committed tree.
		depth: usize,
		/// Indices opened together, which fixes the layer the branches stop at.
		n_queries: usize,
	},
	/// A whole committed vector, read in the clear and checked by rebuilding its tree.
	Vector {
		/// Field elements one leaf holds.
		leaf_size: usize,
		/// Depth of the committed tree.
		depth: usize,
	},
}

/// A channel that records what it was asked while delegating to the circuit channel.
///
/// The proof layout already records every read, but not where the samples fall between them.
/// The challenger's cost is a function of exactly that interleaving, so recording it here is what
/// lets the phase breakdown price the challenger on its own.
struct Recording<'a, 'b> {
	/// The channel every call is forwarded to.
	inner: &'b mut MerkleVerifierChannel<'a>,
	/// The Fiat-Shamir schedule so far.
	challenger: Vec<ChallengerOp>,
	/// The Merkle operations so far.
	merkle: Vec<MerkleOp>,
}

impl<'a, 'b> Recording<'a, 'b> {
	/// Wraps `inner`, having recorded nothing.
	const fn new(inner: &'b mut MerkleVerifierChannel<'a>) -> Self {
		// Both schedules start empty, since recording begins with the first forwarded call.
		Self {
			inner,
			challenger: Vec::new(),
			merkle: Vec::new(),
		}
	}

	/// Consumes the recorder and returns the two schedules it recorded.
	fn finish(self) -> (Vec<ChallengerOp>, Vec<MerkleOp>) {
		// The borrow of the inner channel is dropped here, which is what lets the caller finish it.
		(self.challenger, self.merkle)
	}

	/// Records an observation of `n` proof words.
	///
	/// A zero-word read is not an observation: the channel only turns once it advances the tape.
	fn observed(&mut self, n: usize) {
		// Recording an empty read would insert a spurious channel turn into the replayed schedule.
		if n > 0 {
			self.challenger.push(ChallengerOp::Observe(n));
		}
	}
}

impl IPVerifierChannel<B128> for Recording<'_, '_> {
	type Elem = Elem;

	fn recv_one(&mut self) -> Result<Elem, binius_ip::channel::Error> {
		// A message read is hashed as it is consumed, and one element is two proof words.
		self.observed(ELEMENT_WORDS);
		self.inner.recv_one()
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<Elem>, binius_ip::channel::Error> {
		// One read of n elements, so the words are hashed as a single contiguous run.
		self.observed(n * ELEMENT_WORDS);
		self.inner.recv_many(n)
	}

	fn sample(&mut self) -> Elem {
		// A 128-bit draw consumes sixteen sampler bytes, plus any refill they force.
		self.challenger.push(ChallengerOp::SampleB128);
		IPVerifierChannel::sample(self.inner)
	}

	fn observe_one(&mut self, val: B128) -> Elem {
		// Observing always turns the channel, even with nothing to write, so it is always
		// recorded.
		self.challenger.push(ChallengerOp::Observe(ELEMENT_WORDS));
		self.inner.observe_one(val)
	}

	fn observe_many(&mut self, vals: &[B128]) -> Vec<Elem> {
		// Values the verifier computed rather than read, hashed together as one run.
		self.challenger
			.push(ChallengerOp::Observe(vals.len() * ELEMENT_WORDS));
		self.inner.observe_many(vals)
	}

	fn assert_zero(&mut self, val: Elem) -> Result<(), binius_ip::channel::Error> {
		// An equality constraint touches neither the tape nor the Fiat-Shamir state.
		self.inner.assert_zero(val)
	}

	fn compute_public_value(
		&mut self,
		inputs: &[Elem],
		f: impl binius_field::util::FieldFn<B128>,
	) -> Elem {
		// Arithmetic over wires the verifier already holds, so there is nothing to record.
		self.inner.compute_public_value(inputs, f)
	}
}

impl WordIPVerifierChannel<B128> for Recording<'_, '_> {
	type Word = ChannelWord;

	fn observe_words(&mut self, words: &[ChannelWord]) {
		// Words already in circuit form, hashed exactly as they are.
		self.challenger.push(ChallengerOp::Observe(words.len()));
		self.inner.observe_words(words);
	}

	fn subset_sum(&mut self, elems: &[Elem], word: &ChannelWord) -> Elem {
		// A table lookup over existing elements, so it is pure gate work.
		self.inner.subset_sum(elems, word)
	}

	fn select(&mut self, elems: &[Elem], word: &ChannelWord) -> Elem {
		// Also a lookup, and also invisible to both schedules.
		self.inner.select(elems, word)
	}

	fn sample_bits(&mut self, bits: usize) -> ChannelWord {
		// A query index draw, which reads four sampler bytes and masks them down to `bits`.
		self.challenger.push(ChallengerOp::SampleBits(bits));
		self.inner.sample_bits(bits)
	}
}

impl MerkleIPVerifierChannel<B128> for Recording<'_, '_> {
	type Commitment = MerkleCommitment;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<MerkleCommitment, binius_iop::merkle_channel::Error> {
		// A root is a message, so its four words are hashed before anything is opened against it.
		self.observed(DIGEST_WORDS);
		self.inner.recv_merkle_commitment(leaf_size, depth)
	}

	fn recv_openings(
		&mut self,
		commitment: &MerkleCommitment,
		indices: &[ChannelWord],
	) -> Result<Vec<Elem>, binius_iop::merkle_channel::Error> {
		// Pricing a climb needs all three numbers: leaf size, tree depth, and how many indices
		// share one decommitted layer.
		self.merkle.push(MerkleOp::Openings {
			leaf_size: commitment.leaf_size,
			depth: commitment.depth,
			n_queries: indices.len(),
		});
		self.inner.recv_openings(commitment, indices)
	}

	fn recv_committed_vector(
		&mut self,
		commitment: &MerkleCommitment,
	) -> Result<Vec<Elem>, binius_iop::merkle_channel::Error> {
		// A vector read in the clear is priced by the tree rebuilt over it, which the shape fixes.
		self.merkle.push(MerkleOp::Vector {
			leaf_size: commitment.leaf_size,
			depth: commitment.depth,
		});
		self.inner.recv_committed_vector(commitment)
	}
}

// The tests

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
	let verifier = VerifierCircuit::build(&shape, &setup);

	// The circuit reads the whole tape and nothing beyond it, which is what a native verifier
	// enforces by refusing to leave bytes unread.
	//
	//     circuit: as many wires as the reads it made
	//     prover:  the finalized byte tape
	//     -> equal, or the two halves disagree about the protocol
	assert_eq!(
		verifier.layout.n_bytes(),
		opening.proof.len(),
		"the circuit must read exactly the transcript the prover wrote"
	);

	// Both the assertions and the compiled constraints must pass, since either alone could miss.
	verifier
		.check_opening(&opening)
		.expect("the circuit must accept the opening the native prover proved");
}

#[test]
fn one_circuit_verifies_two_different_openings() {
	// Invariant: one compiled circuit verifies every opening of its shape, not just the one it was
	// measured against.
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
	let verifier = VerifierCircuit::build(&shape, &setup);
	// The same wires are refilled for the second opening, so nothing carries over between them.
	for (i, opening) in [&first, &second].into_iter().enumerate() {
		verifier.check_opening(opening).unwrap_or_else(|error| {
			panic!("opening {i} must satisfy the shared circuit: {error:?}")
		});
	}
}

/// The proof reads a corruption test aims at, located through the layout rather than by offset.
struct Targets {
	/// The first MLE-check round polynomial: an observed message.
	round_poly: ProofRead,
	/// The layer the first batch of openings decommits to: unobserved advice.
	layer: ProofRead,
	/// The first opened leaf: unobserved advice.
	leaf: ProofRead,
	/// The first authentication branch: unobserved advice.
	branch: ProofRead,
	/// The terminal codeword, sent in full: unobserved advice.
	terminal: ProofRead,
}

/// Locates the reads a corruption test aims at, from the reads the verifier recorded.
///
/// Nothing here is a byte offset: a parameter change moves every offset but leaves the *kinds* and
/// the order of the reads alone.
fn targets(shape: &Shape, layout: &ProofLayout) -> Targets {
	let reads = layout.reads();

	// A commitment root is four words and a degree-1 round polynomial is one element, so the round
	// polynomials are the only two-word messages on the tape.
	let round_poly = *reads
		.iter()
		.find(|read| read.kind == ReadKind::Message && read.n_words == ELEMENT_WORDS)
		.expect("every MLE-check round sends a round polynomial");

	// An opening batch reads the shared layer first, then a leaf and a branch per query, and the
	// codeword oracle is the first commitment opened.
	let decommitments = reads
		.iter()
		.filter(|read| read.kind == ReadKind::Decommitment)
		.collect::<Vec<_>>();
	// So the first three advice reads are, in order, the layer, one leaf and its branch.
	let [layer, leaf, branch, ..] = decommitments[..] else {
		panic!("the query phase reads a layer, a leaf and a branch at the very least");
	};

	// Recomputing the three widths from the shape is what pins the identification: a read of the
	// wrong width would mean the reads were matched to the wrong thing.
	let depth = shape.codeword_depth();
	let layer_depth = BinaryMerkleTreeScheme::<B128, Sha256HashSuite>::new()
		.optimal_verify_layer(shape.n_test_queries, depth);
	assert_eq!(layer.n_words, (1 << layer_depth) * DIGEST_WORDS, "the decommitted layer");
	assert_eq!(leaf.n_words, LEAF_ELEMENTS * ELEMENT_WORDS, "the opened coset");
	assert_eq!(branch.n_words, (depth - layer_depth) * DIGEST_WORDS, "the authentication branch");
	// A shape whose layer reaches the leaves would leave the branch test with nothing to corrupt.
	assert!(branch.n_words > 0, "the shape must leave a branch to climb");

	// The terminal codeword is the last thing the query phase reads.
	let terminal = *reads.last().expect("the verifier reads something");
	assert_eq!(terminal.kind, ReadKind::Decommitment, "the terminal codeword is advice");

	Targets {
		round_poly,
		layer: *layer,
		leaf: *leaf,
		branch: *branch,
		terminal,
	}
}

/// Returns `proof` with the low bit of the first byte of proof word `word_offset` flipped.
fn corrupt(proof: &[u8], word_offset: usize) -> Vec<u8> {
	let mut proof = proof.to_vec();
	// One bit is the smallest change a sound protocol must still reject, and the first byte of a
	// word is the one the little-endian read puts in the low bits.
	proof[word_offset * (Word::BITS / 8)] ^= 1;
	proof
}

/// Asserts the circuit rejects `proof`, and returns the paths of the assertions that failed.
fn rejected(verifier: &VerifierCircuit, opening: &Opening, proof: &[u8]) -> Vec<String> {
	match verifier.check(opening.eval_claim, &opening.eval_point, proof) {
		// Acceptance means the corruption slipped through a check that should have caught it.
		Ok(()) => panic!("a corrupted proof must leave the circuit unsatisfied"),
		// Reaching constraint verification means population missed a failure it could see, which
		// would make the assertion paths below useless as diagnostics.
		Err(Unsatisfied::Constraints(error)) => {
			panic!("population must find the failure before constraint verification: {error}")
		}
		Err(Unsatisfied::Assertions(PopulateError {
			failures, total, ..
		})) => {
			// A reported failure count of zero would contradict the error itself.
			assert!(total > 0, "an unsatisfied circuit must report a failing assertion");
			// The list is truncated only by its own cap, so a short list means a lost failure.
			assert_eq!(
				failures.len(),
				total.min(MAX_ASSERTION_FAILURES),
				"the failure list must be complete up to its cap"
			);
			// A path with no detail cannot be debugged, so an empty one is a defect in itself.
			for failure in &failures {
				assert!(!failure.detail.is_empty(), "a failure must carry a diagnostic");
			}
			// Only the paths are returned: which check fired is what each test asserts on.
			failures.into_iter().map(|failure| failure.path).collect()
		}
	}
}

#[test]
fn a_corrupted_opened_value_is_rejected() {
	// Invariant: an opened coset is decommitment advice, so its own Merkle path is the only thing
	// binding it.
	//
	// Fixture state: the native shape, and the first opened leaf found through the read log.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 3);
	let verifier = VerifierCircuit::build(&shape, &setup);
	let targets = targets(&shape, &verifier.layout);

	// Mutation: flip one bit of the first opened coset.
	//
	//     before:  leaf  -> hash -> climb -> the layer entry the index addresses
	//     after:   leaf' -> hash' -> climb -> a digest that entry does not hold
	//
	// The challenger never saw those bytes, so the Fiat-Shamir state is untouched and the Merkle
	// assertions alone catch it.
	let paths = rejected(&verifier, &opening, &corrupt(&opening.proof, targets.leaf.word_offset));
	assert!(
		paths.iter().any(|path| path.contains("opening")),
		"a corrupted coset must fail a Merkle opening check: {paths:?}"
	);
}

#[test]
fn a_corrupted_merkle_sibling_is_rejected() {
	// Invariant: an authentication sibling is advice too, so the same mechanism must catch it from
	// the other side of the compression.
	//
	// Fixture state: the native shape, and the first branch found through the read log.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 4);
	let verifier = VerifierCircuit::build(&shape, &setup);
	let targets = targets(&shape, &verifier.layout);

	// Mutation: flip one bit of the first sibling digest.
	//
	//     before:  (running, sibling)  -> parent -> ... -> the layer entry
	//     after:   (running, sibling') -> parent' -> ... -> a digest the layer does not hold
	let paths = rejected(&verifier, &opening, &corrupt(&opening.proof, targets.branch.word_offset));
	assert!(
		paths.iter().any(|path| path.contains("opening")),
		"a corrupted sibling must fail a Merkle opening check: {paths:?}"
	);
}

#[test]
fn a_corrupted_round_polynomial_is_rejected() {
	// Invariant: a round polynomial is an observed message, so corrupting it moves the Fiat-Shamir
	// state as well as the sum-check value.
	//
	// Fixture state: the native shape, and the first two-word message on the tape.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 5);
	let verifier = VerifierCircuit::build(&shape, &setup);
	let targets = targets(&shape, &verifier.layout);

	// Mutation: flip one bit of the first round polynomial.
	//
	//     before:  round value  -> challenge -> ... -> query indices -> the committed positions
	//     after:   round value' -> a different challenge from that round on, indices included
	//
	// The opened values and siblings on the tape are unchanged, but the indices addressing them are
	// not, so every opening now climbs to the wrong entry of the decommitted layer.
	let paths =
		rejected(&verifier, &opening, &corrupt(&opening.proof, targets.round_poly.word_offset));
	assert!(
		paths.iter().any(|path| path.contains("verify_opening")),
		"a corrupted round polynomial must move the query indices and fail an opening: {paths:?}"
	);
}

#[test]
fn a_corrupted_terminal_codeword_entry_is_rejected() {
	// Invariant: the terminal codeword is advice bound two ways at once, by its own commitment and
	// by the queries that were folded down to it.
	//
	// Fixture state: the native shape, and the last read on the tape.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 6);
	let verifier = VerifierCircuit::build(&shape, &setup);
	let targets = targets(&shape, &verifier.layout);

	// Mutation: flip one bit of the first terminal entry.
	//
	//     before:  entry  -> rebuilt tree == root, and folded query == entry
	//     after:   entry' -> a different root, and a folded query that no longer matches
	//
	// Either failure is enough, so the assertion below accepts whichever fires.
	let paths =
		rejected(&verifier, &opening, &corrupt(&opening.proof, targets.terminal.word_offset));
	assert!(
		paths
			.iter()
			.any(|path| path.contains("vector") || path.contains("assert_zero")),
		"a corrupted terminal entry must fail the vector check or a FRI equality: {paths:?}"
	);
}

#[test]
fn a_corrupted_layer_digest_is_rejected() {
	// Invariant: the decommitted layer is folded to the root once and then shared by every query,
	// so one bad digest breaks the fold whether or not a query climbs through it.
	//
	// Fixture state: the native shape, and the first advice read on the tape.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 7);
	let verifier = VerifierCircuit::build(&shape, &setup);
	let targets = targets(&shape, &verifier.layout);

	// Mutation: flip one bit of the first layer digest.
	//
	//     before:  layer  -> fold -> root, which the challenger already observed
	//     after:   layer' -> fold -> a root the tape never carried
	let paths = rejected(&verifier, &opening, &corrupt(&opening.proof, targets.layer.word_offset));
	assert!(
		paths.iter().any(|path| path.contains("layer")),
		"a corrupted layer digest must fail the layer fold: {paths:?}"
	);
}

#[test]
fn a_tampered_eval_claim_is_rejected() {
	// Invariant: the statement is bound by the protocol's own equalities, not by any hash, so a
	// wrong claim must still be caught.
	//
	// Fixture state: the native shape, one honest proof, and one altered claim.
	let shape = NATIVE_SHAPE;
	let setup = shape.setup();
	let opening = prove(&shape, &setup, 8);
	let verifier = VerifierCircuit::build(&shape, &setup);

	// Mutation: add one to the claimed evaluation, leaving the proof bytes alone.
	//
	//     before:  first round polynomial recovers to the claimed sum
	//     after:   it recovers to a value one away, and the error rides the folding to the end
	//
	// The claim is a statement wire and not a proof byte, so the Fiat-Shamir state is identical and
	// every hash still agrees.
	let tampered = opening.eval_claim + B128::ONE;
	match verifier.check(tampered, &opening.eval_point, &opening.proof) {
		// A tampered claim that satisfies the circuit would make the statement meaningless.
		Ok(()) => panic!("a tampered claim must leave the circuit unsatisfied"),
		// Population sees every assertion, so it must be what catches this.
		Err(Unsatisfied::Constraints(error)) => {
			panic!("population must find the failure before constraint verification: {error}")
		}
		Err(Unsatisfied::Assertions(PopulateError {
			failures, total, ..
		})) => {
			// The error must name at least one failing assertion to be worth anything.
			assert!(total > 0, "an unsatisfied circuit must report a failing assertion");
			assert_eq!(failures.len(), total.min(MAX_ASSERTION_FAILURES));
			// No hash can fail here, so a Merkle path in this list would mean the wires that carry
			// the statement leak into the hashing.
			assert!(
				failures
					.iter()
					.all(|failure| failure.path.contains("assert_zero")),
				"only the protocol's own equalities may fail: {:?}",
				failures.iter().map(|f| &f.path).collect::<Vec<_>>()
			);
		}
	}
}

// The measurement

/// AND and BMUL constraints one phase spends.
#[derive(Clone, Copy, Default, Debug)]
struct Cost {
	/// AND constraints.
	and: usize,
	/// BMUL constraints.
	bmul: usize,
}

impl Cost {
	/// The AND and BMUL columns of a built circuit.
	///
	/// Pinning an output so the dead-code pass keeps it costs only ZERO constraints, which neither
	/// column counts, so scaffolding needs no charging back out.
	fn of(circuit: &Circuit) -> Self {
		let stat = CircuitStat::collect(circuit);
		Self {
			and: stat.n_and_constraints,
			bmul: stat.n_bmul_constraints,
		}
	}
}

/// Allocates `n` digests on a proof stream, byte-reversed as the channel reverses them.
///
/// The tape carries a digest little-endian while a compression consumes big-endian halves.
fn read_digests(builder: &CircuitBuilder, n: usize) -> Vec<Digest> {
	(0..n)
		.map(|_| {
			// One reversal per 32-bit half, which is where the real channel pays it too.
			std::array::from_fn(|_| {
				binius_circuits::bytes::swap_bytes_32(builder, builder.add_witness())
			})
		})
		.collect()
}

/// Allocates `n` field elements on a proof stream, which need no reordering.
fn read_elements(builder: &CircuitBuilder, n: usize) -> Vec<Element> {
	(0..n)
		// An element is two little-endian words on the tape and two wires in the circuit.
		.map(|_| std::array::from_fn(|_| builder.add_witness()))
		.collect()
}

/// Prices the Fiat-Shamir challenger alone, by replaying the schedule the verifier drove.
///
/// Only the schedule matters: the native challenger's block boundaries, padding and sampled-byte
/// counter follow from the sequence of calls and never from the bytes.
fn challenger_cost(schedule: &[ChallengerOp]) -> Cost {
	// A fresh builder, so nothing but the challenger contributes to the count.
	let builder = CircuitBuilder::new();
	let mut challenger = Sha256Challenger::new(&builder);
	let mut pins = 0;
	for op in schedule {
		match *op {
			// Observed words: the values are irrelevant, so uninitialized witnesses will do.
			ChallengerOp::Observe(n) => {
				let words = (0..n).map(|_| builder.add_witness()).collect::<Vec<_>>();
				challenger.observe_words(&words);
			}
			// A sampled challenge has to be consumed by something, or the dead-code pass removes
			// the compressions that produced it.
			ChallengerOp::SampleB128 => {
				let (lo, hi) = challenger.sample_b128();
				let claimed = [builder.add_inout(), builder.add_inout()];
				builder.assert_eq_v(format!("sample[{pins}]"), [lo, hi], claimed);
				pins += ELEMENT_WORDS;
			}
			// Same reason, and one word means one pinned equality.
			ChallengerOp::SampleBits(bits) => {
				let word = challenger.sample_bits(bits);
				builder.assert_eq(format!("bits[{pins}]"), word, builder.add_inout());
				pins += 1;
			}
		}
	}
	Cost::of(&builder.build())
}

/// Prices the Merkle work of the recorded operations `keep` selects, on a fresh builder.
///
/// A channel call cannot be priced in place, so this repeats what the channel emits:
///
/// - the digest and element reads off the tape
/// - the fold of the decommitted layer up to the root
/// - the per-query climb from a leaf to that layer
/// - the tree rebuilt over a vector read in the clear
///
/// Observing the roots is deliberately not repeated, since that cost belongs to the challenger.
fn merkle_cost(ops: &[MerkleOp], keep: impl Fn(&MerkleOp) -> bool) -> Cost {
	let builder = CircuitBuilder::new();
	// The scheme is consulted only for the layer depth it would have chosen.
	let scheme = BinaryMerkleTreeScheme::<B128, Sha256HashSuite>::new();
	for op in ops.iter().filter(|op| keep(op)) {
		match *op {
			MerkleOp::Openings {
				leaf_size,
				depth,
				n_queries,
			} => {
				// The layer depth is what trades a wider shared layer against shorter climbs, so
				// it must be the same choice the real channel made.
				let layer_depth = scheme.optimal_verify_layer(n_queries, depth);
				let root = read_digests(&builder, 1)[0];
				let layer = read_digests(&builder, 1 << layer_depth);
				// Paid once, whatever the query count.
				merkle::verify_layer(&builder, root, &layer);
				// Paid once per query: one leaf hash plus one compression per level climbed.
				for _ in 0..n_queries {
					let leaf = read_elements(&builder, leaf_size);
					let branch = read_digests(&builder, depth - layer_depth);
					merkle::verify_opening(
						&builder,
						builder.add_witness(),
						&leaf,
						layer_depth,
						depth,
						&layer,
						&branch,
					);
				}
			}
			MerkleOp::Vector { leaf_size, depth } => {
				// A vector arrives whole, so the check is the entire tree over 2^depth leaves.
				let root = read_digests(&builder, 1)[0];
				let data = read_elements(&builder, leaf_size << depth);
				merkle::verify_vector(&builder, root, &data, leaf_size);
			}
		}
	}
	// Nothing here is pinned, since every read already feeds an assertion.
	Cost::of(&builder.build())
}

/// Prices the MLE-check's own arithmetic: recover the round polynomial, evaluate it, `n_vars`
/// times.
fn mlecheck_cost(n_vars: usize) -> Cost {
	let builder = CircuitBuilder::new();
	// Every value in this loop is a fresh witness, since only the arithmetic shape is being priced.
	let elem = || Elem::new(&builder, builder.add_witness(), builder.add_witness());

	let mut sum = elem();
	for _ in 0..n_vars {
		// A degree-1 round proof is one truncated coefficient.
		let round_proof = mlecheck::RoundProof(RoundCoeffs(vec![elem()]));
		// Recovery restores the dropped coefficient from the running sum, and evaluating at the
		// round challenge produces the sum the next round must match.
		let coeffs = round_proof.recover(sum, elem());
		sum = coeffs.evaluate(&elem());
	}

	// The final sum is pinned, or the whole chain is dead code.
	let (lo, hi) = sum.words(&builder);
	let claimed = [builder.add_inout(), builder.add_inout()];
	builder.assert_eq_v("sum", [lo, hi], claimed);
	Cost::of(&builder.build())
}

/// Prints the constraint table and the phase breakdown for `shape`, and returns the total.
fn report(shape: &Shape) -> CircuitStat {
	// The total comes from the real circuit, so the phases below are measured against something
	// that was actually built.
	let setup = shape.setup();
	let verifier = VerifierCircuit::build(shape, &setup);
	let stat = &verifier.stat;

	// Each phase is rebuilt on its own, from the schedules the verifier recorded.
	let challenger = challenger_cost(&verifier.challenger_ops);
	let openings = merkle_cost(&verifier.merkle_ops, |op| matches!(op, MerkleOp::Openings { .. }));
	let terminal = merkle_cost(&verifier.merkle_ops, |op| matches!(op, MerkleOp::Vector { .. }));
	let mlecheck = mlecheck_cost(shape.n_vars);

	// Whatever the four named phases do not account for is the folding and its equalities, so the
	// rows always sum to the total.
	let named = [
		("Merkle openings", openings),
		("Fiat-Shamir challenger", challenger),
		("terminal codeword", terminal),
		("MLE-check arithmetic", mlecheck),
	];
	let accounted: Cost = named.iter().fold(Cost::default(), |acc, (_, cost)| Cost {
		and: acc.and + cost.and,
		bmul: acc.bmul + cost.bmul,
	});
	// Saturating, because a phase priced in isolation can slightly overshoot what sharing achieves
	// inside the real circuit.
	let residual = Cost {
		and: stat.n_and_constraints.saturating_sub(accounted.and),
		bmul: stat.n_bmul_constraints.saturating_sub(accounted.bmul),
	};

	// The shape and the tape length, so a printed table can be traced back to what produced it.
	println!(
		"\n=== n_vars {} | log_inv_rate {} | {} queries | {} proof bytes ===",
		shape.n_vars,
		shape.log_inv_rate,
		shape.n_test_queries,
		verifier.layout.n_bytes(),
	);
	// All four columns, since which one is scarce depends on the circuit this is embedded in.
	println!(
		"total: {:>9} AND  {:>7} BMUL  {:>7} ZERO  {:>9} committed words",
		stat.n_and_constraints,
		stat.n_bmul_constraints,
		stat.n_zero_constraints,
		stat.committed_allocated,
	);
	println!("{:<26} {:>9} {:>8}   {:>6}", "phase", "AND", "BMUL", "AND %");
	// The residual is appended last, so the named phases read in descending cost order.
	for (name, cost) in named
		.into_iter()
		.chain([("FRI folding and equalities", residual)])
	{
		println!(
			"{:<26} {:>9} {:>8}   {:>5.1}%",
			name,
			cost.and,
			cost.bmul,
			100.0 * cost.and as f64 / stat.n_and_constraints as f64,
		);
	}

	verifier.stat
}

#[test]
fn the_verifier_cost_breaks_down_by_phase() {
	// Invariant: hashing dominates the verifier, by more than an order of magnitude.
	//
	// Fixture state: the native shape, printed as a total plus five phase rows.
	let stat = report(&NATIVE_SHAPE);

	// The AND column is what a Binius64 proof pays for most, and it is nearly all SHA-256 here.
	//
	//     AND  : Merkle climbs, the challenger, the rebuilt terminal tree
	//     BMUL : the field arithmetic the folding and the MLE-check reduce
	//     -> a factor of ten is a loose floor, so this catches a regression not a fluctuation
	assert!(
		stat.n_and_constraints > stat.n_bmul_constraints * 10,
		"SHA-256 must dominate the field arithmetic: {} AND against {} BMUL",
		stat.n_and_constraints,
		stat.n_bmul_constraints
	);
}

#[test]
fn the_cost_surface_over_shapes() {
	// Invariant: cost rises with the query count and with the committed size, and neither rise is
	// the naive one.
	//
	// Fixture state: five shapes, each one lever off the native shape.
	//
	//     row 1: the native shape, 8 variables, rate 1, 32 queries
	//     row 2: 10 variables, so four times the committed size
	//     row 3: rate 2, so twice the codeword
	//     row 4: 16 queries
	//     row 5: 64 queries
	//
	// One parameter moves at a time, so each row isolates one lever.
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

	// Each row is printed as it is built, so the table survives a later assertion failing.
	let mut totals = Vec::new();
	for shape in shapes {
		let stat = report(&shape);
		totals.push((shape, stat.n_and_constraints));
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
	//
	// The rise is sublinear because the scheme deepens the shared layer as the query count grows,
	// which shortens every climb.
	assert!(and_of(64) > and_of(32), "more queries must cost more");
	assert!(and_of(32) > and_of(16), "more queries must cost more");

	// Two more variables deepen every tree by two levels, paid once per query.
	//
	// Measured at +51%, so growth in the committed size is logarithmic but far from free.
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
