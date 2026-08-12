// Copyright 2026 The Binius Developers

use binius_field::{PackedBinaryGhash2x128b, field::FieldOps};
use binius_frontend::{CircuitStat, PopulateError};
use binius_iop::merkle_channel::VerifierMerkleTranscriptChannel;
use binius_iop_prover::merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel};
use binius_ip::channel::{select_word, subset_sum_word};
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_math::{FieldBuffer, test_utils::random_scalars};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use sha2::Sha256;

use super::*;

/// The Fiat-Shamir challenger all three sides run, and the one the circuit gadget mirrors.
type Challenger = HasherChallenger<Sha256>;

/// Packing the committed data is held in, two elements to a register.
type P = PackedBinaryGhash2x128b;

/// The native verifier channel, which checks a proof instead of recording one.
type NativeChannel = VerifierMerkleTranscriptChannel<
	binius_transcript::VerifierTranscript<Challenger>,
	Challenger,
	B128,
	Sha256HashSuite,
>;

/// The prover channel that writes the tape both verifier sides then read.
type NativeProverChannel =
	ProverMerkleTranscriptChannel<ProverTranscript<Challenger>, Challenger, B128, Sha256HashSuite>;

/// One step of a script driven identically against the native channel and the circuit channel.
///
/// Every operation a Merkle verifier channel offers appears here, so a script is a complete
/// exercise of the channel's Fiat-Shamir bookkeeping.
#[derive(Clone, Copy, Debug)]
enum Op {
	/// Receive `n` prover elements as an observed message.
	Recv(usize),
	/// Sample one field element.
	Sample,
	/// Observe `n` statement elements.
	Observe(usize),
	/// Observe `n` statement words.
	ObserveWords(usize),
	/// Sample `n` bits.
	Bits(usize),
	/// Sample `n` query indices and open the committed vector at them.
	Openings(usize),
	/// Receive the whole committed vector.
	Vector,
}

/// The committed data a Merkle script opens, and the shape it was committed in.
struct Committed {
	/// The committed vector, in the order the leaves hold it.
	scalars: Vec<B128>,
	/// Field elements one leaf holds.
	leaf_size: usize,
	/// Levels between a leaf and the root.
	depth: usize,
}

impl Committed {
	fn new(rng: &mut StdRng, log_len: usize, log_leaf_size: usize) -> Self {
		Self {
			// A power-of-two length, since a Merkle tree over the data must be balanced.
			scalars: random_scalars::<B128>(rng, 1 << log_len),
			leaf_size: 1 << log_leaf_size,
			// Every leaf holds the same number of elements, so the depth is what is left over.
			depth: log_len - log_leaf_size,
		}
	}
}

/// Runs `ops` on the prover side, returning the proof tape it wrote.
fn prove(committed: &Committed, ops: &[Op], statement: &Statement) -> Vec<u8> {
	// The prover commits to the same data both verifier sides will open.
	let data = FieldBuffer::<P, _>::from_values(&committed.scalars);
	let mut channel = NativeProverChannel::new(ProverTranscript::new(Challenger::default()));
	// The root opens the tape, which is what makes every later decommitment binding.
	let commitment = channel.send_merkle_commitment(data.to_ref(), committed.leaf_size);

	for op in ops {
		match *op {
			// Prover elements go onto the tape and into the Fiat-Shamir state.
			Op::Recv(n) => channel.send_many(&statement.sent[..n]),
			// A sample writes nothing, but it still advances the Fiat-Shamir state.
			Op::Sample => {
				IPProverChannel::<B128>::sample(&mut channel);
			}
			// Statement values are known to both sides, so they are absorbed and never sent.
			Op::Observe(n) => channel.observe_many(&statement.observed[..n]),
			Op::ObserveWords(n) => {
				WordIPProverChannel::<B128>::observe_words(&mut channel, &statement.words[..n]);
			}
			Op::Bits(n) => {
				WordIPProverChannel::<B128>::sample_bits(&mut channel, n);
			}
			Op::Openings(n) => {
				// Query indices are drawn from the Fiat-Shamir state, so the prover cannot choose
				// which leaves it opens.
				let indices = (0..n)
					.map(|_| {
						WordIPProverChannel::<B128>::sample_bits(&mut channel, committed.depth)
					})
					.collect::<Vec<_>>();
				channel.send_openings(&commitment, data.to_ref(), &indices);
			}
			// The whole vector, which needs no branches because the tree is rebuilt over it.
			Op::Vector => channel.send_committed_vector(&commitment, data.to_ref()),
		}
	}
	// The finalized tape is the only thing that crosses to the verifier sides.
	channel.into_transcript().finalize()
}

/// The statement values a script feeds in, fixed rather than read from the proof.
struct Statement {
	/// Elements the prover sends as observed messages.
	sent: Vec<B128>,
	/// Elements both sides observe without either sending them.
	observed: Vec<B128>,
	/// Words both sides observe, standing in for protocol constants.
	words: Vec<Word>,
}

impl Statement {
	fn new(rng: &mut StdRng, n: usize) -> Self {
		Self {
			// One pool per role, so an operation always has enough values to take a prefix of.
			sent: random_scalars::<B128>(&mut *rng, n),
			observed: random_scalars::<B128>(&mut *rng, n),
			words: (0..n).map(|_| Word(rng.random())).collect(),
		}
	}
}

/// Everything a script's verifier side hands back, for the two sides to be compared.
#[derive(Default, PartialEq, Eq, Debug)]
struct Trace {
	/// Field challenges, in order.
	samples: Vec<u128>,
	/// Word challenges, in order.
	bits: Vec<u64>,
	/// Elements received or opened, in order.
	values: Vec<u128>,
}

/// Runs `ops` against the native channel over `proof`, returning what it saw.
fn verify_natively(
	committed: &Committed,
	ops: &[Op],
	statement: &Statement,
	proof: &[u8],
) -> Trace {
	// A fresh challenger, so the native side derives its challenges from the tape alone.
	let transcript =
		binius_transcript::VerifierTranscript::new(Challenger::default(), proof.to_vec());
	let mut channel = NativeChannel::new(transcript);
	let commitment = channel
		.recv_merkle_commitment(committed.leaf_size, committed.depth)
		.expect("the tape opens with the commitment root");

	let mut trace = Trace::default();
	for op in ops {
		match *op {
			// Received elements are recorded so the circuit can be checked against them.
			Op::Recv(n) => trace.values.extend(
				channel
					.recv_many(n)
					.expect("the tape holds the sent elements")
					.into_iter()
					.map(u128::from),
			),
			Op::Sample => trace
				.samples
				.push(u128::from(IPVerifierChannel::<B128>::sample(&mut channel))),
			// An observe returns nothing to compare: it only moves the Fiat-Shamir state.
			Op::Observe(n) => {
				channel.observe_many(&statement.observed[..n]);
			}
			Op::ObserveWords(n) => {
				WordIPVerifierChannel::<B128>::observe_words(&mut channel, &statement.words[..n]);
			}
			Op::Bits(n) => {
				let word = WordIPVerifierChannel::<B128>::sample_bits(&mut channel, n);
				trace.bits.push(word.as_u64());
			}
			Op::Openings(n) => {
				// Indices are recorded too, so the circuit can be shown to query the same leaves.
				let indices = (0..n)
					.map(|_| {
						let word = WordIPVerifierChannel::<B128>::sample_bits(
							&mut channel,
							committed.depth,
						);
						trace.bits.push(word.as_u64());
						word
					})
					.collect::<Vec<_>>();
				trace.values.extend(
					channel
						.recv_openings(&commitment, &indices)
						.expect("the prover's own openings must verify")
						.into_iter()
						.map(u128::from),
				);
			}
			Op::Vector => trace.values.extend(
				channel
					.recv_committed_vector(&commitment)
					.expect("the prover's own vector must verify")
					.into_iter()
					.map(u128::from),
			),
		}
	}
	// Finalizing fails on leftover bytes, which is what pins the script to the exact tape.
	channel
		.into_transcript()
		.finalize()
		.expect("the tape must be read out in full");
	trace
}

/// Builds a circuit for `ops`, fills it from `proof`, and checks it reproduces `expected`.
///
/// Every sampled and received value is pinned to a public wire holding what the native channel
/// produced, so a disagreement fails witness population rather than a Rust comparison.
fn verify_in_circuit(
	committed: &Committed,
	ops: &[Op],
	statement: &Statement,
	proof: &[u8],
	expected: &Trace,
) -> (CircuitStat, ProofLayout) {
	let builder = CircuitBuilder::new();
	let mut channel = MerkleVerifierChannel::new(&builder);
	// The root is read first here too, since the circuit must read the tape in the same order.
	let commitment = channel
		.recv_merkle_commitment(committed.leaf_size, committed.depth)
		.expect("reading a commitment cannot fail");

	// Wires to pin, paired with the value the native channel produced.
	let mut pins: Vec<(Wire, u64)> = Vec::new();
	// Three cursors, walked in lockstep with the native run that filled them.
	let mut samples = expected.samples.iter();
	let mut bits = expected.bits.iter();
	let mut values = expected.values.iter();

	let pin_elem = |pins: &mut Vec<(Wire, u64)>, elem: &SymbolicElem, want: u128| {
		// A settled element becomes constant wires, so both forms are pinned the same way.
		let (lo, hi) = elem.to_wires(&builder);
		// The native value enters on public wires, which turns the comparison into a constraint.
		let claimed = [builder.add_inout(), builder.add_inout()];
		for (wire, word) in claimed.iter().zip(element_words(want)) {
			pins.push((*wire, word));
		}
		// A distinct name per claim keeps a failure traceable to the value that broke.
		builder.assert_eq_v(format!("pin[{}]", pins.len()), [lo, hi], claimed);
	};

	for op in ops {
		match *op {
			Op::Recv(n) => {
				for elem in channel.recv_many(n).expect("reading a message cannot fail") {
					// One native value per element, taken in the order both sides read them.
					let want = *values
						.next()
						.expect("one native value per received element");
					pin_elem(&mut pins, &elem, want);
				}
			}
			Op::Sample => {
				let elem = IPVerifierChannel::<B128>::sample(&mut channel);
				let want = *samples.next().expect("one native challenge per sample");
				pin_elem(&mut pins, &elem, want);
			}
			Op::Observe(n) => {
				channel.observe_many(&statement.observed[..n]);
			}
			Op::ObserveWords(n) => {
				// The statement enters on inout wires, so the native words are pinned onto them.
				let words = &statement.words[..n];
				let first = channel.statement().len();
				WordIPVerifierChannel::<B128>::observe_words(&mut channel, words);
				for (&wire, word) in channel.statement()[first..].iter().zip(words) {
					pins.push((wire, word.as_u64()));
				}
			}
			Op::Bits(n) => {
				let word = WordIPVerifierChannel::<B128>::sample_bits(&mut channel, n);
				let want = *bits.next().expect("one native word per bit sample");
				// A word is one wire, so one public wire pins it.
				let claimed = builder.add_inout();
				pins.push((claimed, want));
				builder.assert_eq(format!("bits[{}]", pins.len()), word.to_wire(&builder), claimed);
			}
			Op::Openings(n) => {
				let indices = (0..n)
					.map(|_| {
						let word = WordIPVerifierChannel::<B128>::sample_bits(
							&mut channel,
							committed.depth,
						);
						// Pinning the index too proves the circuit queries the same leaves.
						let want = *bits.next().expect("one native index per query");
						let claimed = builder.add_inout();
						pins.push((claimed, want));
						builder.assert_eq(
							format!("index[{}]", pins.len()),
							word.to_wire(&builder),
							claimed,
						);
						word
					})
					.collect::<Vec<_>>();
				for elem in channel
					.recv_openings(&commitment, &indices)
					.expect("reading openings cannot fail")
				{
					let want = *values.next().expect("one native value per opened element");
					pin_elem(&mut pins, &elem, want);
				}
			}
			Op::Vector => {
				for elem in channel
					.recv_committed_vector(&commitment)
					.expect("reading the vector cannot fail")
				{
					let want = *values
						.next()
						.expect("one native value per committed element");
					pin_elem(&mut pins, &elem, want);
				}
			}
		}
	}

	// An exhausted cursor is what rules out a circuit that quietly produced fewer values.
	assert_eq!(samples.len(), 0, "every native challenge must be pinned");
	assert_eq!(bits.len(), 0, "every native word must be pinned");
	assert_eq!(values.len(), 0, "every native value must be pinned");

	let layout = channel.finish();
	let circuit = builder.build();
	let stat = CircuitStat::collect(&circuit);

	let mut w = circuit.new_witness_filler();
	// The tape is the only proof-side input, and its length must match the reads exactly.
	layout
		.populate(&mut w, proof)
		.expect("the layout must read exactly the proof the prover wrote");
	// The native values are the only other input; everything else is derived by evaluation.
	for (wire, value) in pins {
		w[wire] = Word(value);
	}

	// Population evaluates every gate and every assertion, so a disagreement fails here.
	circuit
		.populate_wire_witness(&mut w)
		.expect("the circuit must agree with the native channel");
	// Verification then re-checks the same witness against the constraint system itself.
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("every constraint must hold");

	(stat, layout)
}

/// Drives one script through prover, native verifier and circuit, and returns the circuit's report.
fn run_script(
	seed: u64,
	log_len: usize,
	log_leaf_size: usize,
	ops: &[Op],
) -> (CircuitStat, ProofLayout) {
	let mut rng = StdRng::seed_from_u64(seed);
	let committed = Committed::new(&mut rng, log_len, log_leaf_size);
	// Eight values per role is more than any script below takes a prefix of.
	let statement = Statement::new(&mut rng, 8);

	// Three runs over one script:
	//
	//     prover  : writes the tape
	//     native  : reads it and reports the values it derived
	//     circuit : reads it on wires and is pinned to those same values
	let proof = prove(&committed, ops, &statement);
	let expected = verify_natively(&committed, ops, &statement, &proof);
	verify_in_circuit(&committed, ops, &statement, &proof, &expected)
}

#[test]
fn a_full_script_matches_the_native_channel() {
	// Invariant: the circuit channel derives the same challenges, query indices and values as the
	// native channel, over the same tape.
	//
	// Fixture state: 32 committed scalars in leaves of 2, so a tree of depth 4.
	//
	// The script below walks every operation, in interleavings chosen so that the Fiat-Shamir
	// channel turns in both directions and the sampler refills mid-value.
	let ops = [
		// A sample straight off the commitment root, before any other read.
		Op::Sample,
		// Widths that mask nothing, mask everything, and land mid-byte.
		Op::Bits(0),
		Op::Bits(32),
		Op::Bits(5),
		// A message read turns the Fiat-Shamir channel back to observing.
		Op::Recv(3),
		Op::Sample,
		// Statement values, which are constants in the circuit.
		Op::Observe(2),
		Op::ObserveWords(4),
		Op::Sample,
		// An empty observe still turns the channel, so the two below are not no-ops.
		Op::Observe(0),
		Op::ObserveWords(0),
		Op::Sample,
		// Queries at several counts, so the layer depth the openings decommit to varies.
		Op::Openings(1),
		Op::Sample,
		Op::Openings(3),
		Op::Recv(1),
		Op::Openings(6),
		Op::Sample,
		// The whole vector, checked by rebuilding the tree over it.
		Op::Vector,
		Op::Sample,
		Op::Bits(17),
	];
	run_script(0, 5, 1, &ops);
}

#[test]
fn a_single_leaf_tree_matches_the_native_channel() {
	// Invariant: a tree whose root is its only node verifies, with nothing to climb.
	//
	// Fixture state: 4 committed scalars in one leaf of 4, so depth 0.
	//
	//     decommitted layer:  the root itself, one digest
	//     branch per query:   0 digests, since a leaf is already at the layer
	run_script(1, 2, 2, &[Op::Openings(1), Op::Sample, Op::Vector, Op::Bits(8)]);
}

#[test]
fn wide_leaves_match_the_native_channel() {
	// Invariant: a leaf too wide for one hash block verifies, so the leaf hash chains its blocks.
	//
	// Fixture state: 64 committed scalars in leaves of 8, so a tree of depth 3.
	//
	//     leaf = 8 elements = 128 bytes  ->  two SHA-256 blocks
	run_script(2, 6, 3, &[Op::Openings(4), Op::Sample, Op::Vector]);
}

#[test]
fn the_proof_layout_accounts_for_every_byte_in_order() {
	// Invariant: the recorded reads tile the proof stream, so no word is read twice, skipped, or
	// left over.
	//
	// Fixture state: 16 committed scalars in leaves of 2, opened by four operations.
	let ops = [Op::Recv(2), Op::Openings(3), Op::Vector, Op::Recv(1)];
	let (_, layout) = run_script(3, 4, 1, &ops);

	// Walk the reads in order and require each to begin exactly where the last one ended.
	//
	//     read 0: [0, n_0)   read 1: [n_0, n_0 + n_1)   ...
	let mut cursor = 0;
	for read in layout.reads() {
		assert_eq!(read.word_offset, cursor, "a read must start where the last one ended");
		// A zero-word read is never recorded, since it would not turn the Fiat-Shamir channel.
		assert!(read.n_words > 0, "an empty read must not be recorded");
		cursor += read.n_words;
	}
	// Ending on the last wire is what rules out a gap or an unread tail.
	assert_eq!(cursor, layout.words().len(), "the reads must cover every proof word");
	// The byte length follows from the wire count, eight bytes to a word.
	assert_eq!(layout.n_bytes(), layout.words().len() * WORD_BYTES);
}

#[test]
fn only_the_root_and_prover_messages_are_observed() {
	// Invariant: exactly the commitment root and the prover's messages are observed.
	// Everything an opening or a committed vector reads is advice, bound by that root.
	//
	// Fixture state: 16 scalars in leaves of 2, so depth 3, and 2 queries decommitting to a layer
	// of 2 digests.
	let (log_len, log_leaf_size, n_queries): (usize, usize, usize) = (4, 1, 2);
	let (leaf_size, depth) = (1 << log_leaf_size, log_len - log_leaf_size);
	let layer_depth = n_queries.ilog2() as usize;

	let ops = [Op::Recv(1), Op::Openings(n_queries), Op::Vector];
	let (_, layout) = run_script(4, log_len, log_leaf_size, &ops);

	// The expected stream, read by read:
	//
	//     0  root                        message        4 words
	//     1  one received element        message        2 words
	//     2  decommitted layer           decommitment   8 words
	//     3  query 0 leaf                decommitment   4 words
	//     4  query 0 branch              decommitment   8 words
	//     5  query 1 leaf                decommitment   4 words
	//     6  query 1 branch              decommitment   8 words
	//     7  committed vector            decommitment  32 words
	let kinds = layout
		.reads()
		.iter()
		.map(|read| (read.kind, read.n_words))
		.collect::<Vec<_>>();

	// The root and the received element are messages, so the Fiat-Shamir state saw them.
	assert_eq!(kinds[0], (ReadKind::Message, DIGEST_WORDS), "the commitment root");
	assert_eq!(kinds[1], (ReadKind::Message, ELEMENT_WORDS), "the received element");

	// Everything the openings and the vector read is advice, bound by that root.
	assert!(
		kinds[2..]
			.iter()
			.all(|&(kind, _)| kind == ReadKind::Decommitment),
		"every read after the message must be decommitment advice: {kinds:?}"
	);

	// The layer is read once, whatever the query count, since every query folds against it.
	assert_eq!(kinds[2].1, (1 << layer_depth) * DIGEST_WORDS, "the decommitted layer");
	for query in 0..n_queries {
		// Then a leaf and the branch climbing from it to the layer, per query.
		assert_eq!(kinds[3 + 2 * query].1, leaf_size * ELEMENT_WORDS, "the opened leaf");
		assert_eq!(
			kinds[4 + 2 * query].1,
			(depth - layer_depth) * DIGEST_WORDS,
			"the authentication branch"
		);
	}
	// The vector is every leaf's elements at once, which is why it carries no branch.
	assert_eq!(
		kinds[3 + 2 * n_queries].1,
		(leaf_size << depth) * ELEMENT_WORDS,
		"the committed vector"
	);
	// A read count of its own rules out an extra read hiding past the vector.
	assert_eq!(kinds.len(), 4 + 2 * n_queries, "no read beyond the vector");
}

#[test]
fn populate_rejects_a_proof_of_the_wrong_length() {
	// Invariant: a tape that is not exactly as long as the circuit reads is refused, since a short
	// one would leave wires unset and a long one means the two sides disagree on the protocol.
	//
	// Fixture state: one read of 2 elements.
	//
	//     2 elements = 4 proof words = 32 bytes
	let builder = CircuitBuilder::new();
	let mut channel = MerkleVerifierChannel::new(&builder);
	channel.recv_many(2).expect("reading a message cannot fail");
	let layout = channel.finish();

	let expected = 2 * ELEMENT_WORDS * WORD_BYTES;
	assert_eq!(layout.n_bytes(), expected);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	// Mutation: empty, one byte short, and one byte long.
	for supplied in [0, expected - 1, expected + 1] {
		let error = layout
			.populate(&mut w, &vec![0u8; supplied])
			.expect_err("a proof of the wrong length must be rejected");
		// The error reports both lengths, so a caller can see which side is wrong.
		assert_eq!(
			error,
			ProofLengthError {
				expected,
				actual: supplied
			}
		);
	}
	// The exact length is accepted, which pins the rejection to the length and nothing else.
	layout
		.populate(&mut w, &vec![0u8; expected])
		.expect("a proof of the right length must be accepted");
}

/// Builds a circuit over one bit-selected sum or table lookup and checks it against the native
/// helper.
///
/// The word carrying the index is either fixed at build time or on a witness wire, which is the
/// split that decides whether any select gate is spent.
fn run_word_op(
	elems: &[B128],
	word: Word,
	fixed_index: bool,
	native: fn(&[B128], Word) -> B128,
	apply: fn(&mut MerkleVerifierChannel<'_>, &[SymbolicElem], &SymbolicWord) -> SymbolicElem,
) -> CircuitStat {
	let builder = CircuitBuilder::new();
	let mut channel = MerkleVerifierChannel::new(&builder);

	// Half the elements folded and half on wires, so both paths are exercised at once.
	//
	//     index 0, 2, 4, ...  settled at build time
	//     index 1, 3, 5, ...  a witness wire pair
	let mut inputs = Vec::new();
	let operands = elems
		.iter()
		.enumerate()
		.map(|(i, &value)| {
			if i % 2 == 0 {
				return SymbolicElem::Constant(value);
			}
			let wires = [builder.add_witness(), builder.add_witness()];
			// Record what each half must hold, low half first.
			for (wire, w) in wires.iter().zip(element_words(u128::from(value))) {
				inputs.push((*wire, w));
			}
			channel.elem(wires[0], wires[1])
		})
		.collect::<Vec<_>>();

	let index = if fixed_index {
		// A word the protocol fixed carries its value, so the operation can settle in Rust.
		SymbolicWord::from(word)
	} else {
		// A sampled word carries no value, so the operation must spend select gates.
		let wire = builder.add_witness();
		inputs.push((wire, word.as_u64()));
		SymbolicWord::wire(&channel.shared, wire)
	};

	let got = apply(&mut channel, &operands, &index);
	// A settled result becomes constant wires, so either form is pinned the same way.
	let (lo, hi) = got.to_wires(&builder);
	// The native answer enters on public wires, making the comparison a constraint.
	let claimed = [builder.add_inout(), builder.add_inout()];
	for (wire, w) in claimed
		.iter()
		.zip(element_words(u128::from(native(elems, word))))
	{
		inputs.push((*wire, w));
	}
	builder.assert_eq_v("result", [lo, hi], claimed);

	let circuit = builder.build();
	let stat = CircuitStat::collect(&circuit);
	let mut w = circuit.new_witness_filler();
	// Only the operands and the native answer are written; evaluation derives the rest.
	for (wire, value) in inputs {
		w[wire] = Word(value);
	}
	// Population evaluates every gate and every assertion, so a wrong result fails here.
	circuit
		.populate_wire_witness(&mut w)
		.expect("the circuit must reproduce the native result");
	// Verification then re-checks the same witness against the constraint system itself.
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("every constraint must hold");
	stat
}

#[test]
fn subset_sum_matches_the_native_helper() {
	// Invariant: the circuit sums exactly the elements whose bit is set, over every table size and
	// bit pattern that matters.
	//
	// Fixture state: element counts 1, 2, 7 and 64, each against three words and both index forms.
	let mut rng = StdRng::seed_from_u64(5);
	// One element, a full 64 so the top bit is read, and sizes in between.
	for n in [1usize, 2, 7, 64] {
		let elems = random_scalars::<B128>(&mut rng, n);
		// All bits clear, all set, and a random pattern.
		for word in [Word::ZERO, Word::ALL_ONE, Word(rng.random())] {
			for fixed_index in [false, true] {
				run_word_op(&elems, word, fixed_index, subset_sum_word, |channel, elems, index| {
					channel.subset_sum(elems, index)
				});
			}
		}
	}
}

#[test]
fn select_matches_the_native_helper() {
	// Invariant: the circuit reads the entry the low bits of the word address, and ignores the high
	// bits exactly as the native helper does.
	//
	// Fixture state: table sizes 1, 2, 8 and 32, each against every in-range index plus one word
	// whose high bits must be discarded.
	let mut rng = StdRng::seed_from_u64(6);
	for n in [1usize, 2, 8, 32] {
		let elems = random_scalars::<B128>(&mut rng, n);
		// Every in-range index, plus a word whose high bits must be ignored.
		let mut words = (0..n as u64).map(Word::from_u64).collect::<Vec<_>>();
		words.push(Word(rng.random()));
		for word in words {
			for fixed_index in [false, true] {
				run_word_op(&elems, word, fixed_index, select_word, |channel, elems, index| {
					channel.select(elems, index)
				});
			}
		}
	}
}

#[test]
fn select_and_subset_sum_cost_two_bmul_per_multiplexer_node() {
	// Invariant: both operations spend two select gates per multiplexer node, one for each half of
	// an element, and nothing else.
	//
	// Fixture state: 8 elements behind a sampled index.
	//
	//     lookup     :  a tree of 8 - 1 = 7 nodes  ->  14 BMUL
	//     subset sum :  one node per element, 8     ->  16 BMUL
	let mut rng = StdRng::seed_from_u64(7);
	let elems = random_scalars::<B128>(&mut rng, 8);
	let word = Word(rng.random());

	// A multiplexer over n entries is n - 1 select gates, twice over for the two wires.
	let stat = run_word_op(&elems, word, false, select_word, |channel, elems, index| {
		channel.select(elems, index)
	});
	assert_eq!(stat.n_bmul_constraints, 2 * (elems.len() - 1));

	// A subset sum selects each element against zero, so two gates an element.
	let stat = run_word_op(&elems, word, false, subset_sum_word, |channel, elems, index| {
		channel.subset_sum(elems, index)
	});
	assert_eq!(stat.n_bmul_constraints, 2 * elems.len());
}

#[test]
fn a_fixed_index_needs_no_select_gate() {
	// Invariant: an index the protocol fixed settles the whole operation in Rust, which is what
	// makes FRI's terminal fold free.
	//
	// Fixture state: the same 8 elements as the cost test, behind a build-time index of 5.
	//
	//     sampled index  ->  14 or 16 BMUL
	//     fixed index    ->   0 BMUL
	let mut rng = StdRng::seed_from_u64(8);
	let elems = random_scalars::<B128>(&mut rng, 8);
	let word = Word::from_u64(5);

	// A word the protocol fixed settles the lookup while the circuit is built.
	let selected = run_word_op(&elems, word, true, select_word, |channel, elems, index| {
		channel.select(elems, index)
	});
	assert_eq!(selected.n_bmul_constraints, 0, "a fixed index must emit no select gate");

	// The same holds for a sum, where the fixed bits settle which elements take part.
	let summed = run_word_op(&elems, word, true, subset_sum_word, |channel, elems, index| {
		channel.subset_sum(elems, index)
	});
	assert_eq!(summed.n_bmul_constraints, 0, "a fixed index must emit no select gate");
}

#[test]
fn a_zero_element_drops_out_of_a_subset_sum() {
	// Invariant: an element the protocol fixed to zero contributes nothing whichever way its bit
	// falls, so it costs no gate even behind a sampled index.
	//
	// Fixture state: 8 folded zeros behind a witness index.
	//
	//     8 zero elements  ->  0 selects  ->  the sum stays a build-time zero
	let builder = CircuitBuilder::new();
	let mut channel = MerkleVerifierChannel::new(&builder);
	let word = SymbolicWord::wire(&channel.shared, builder.add_witness());

	// Every element is the folded zero, so nothing can contribute whatever the bits say.
	let elems = vec![SymbolicElem::Constant(B128::ZERO); 8];
	let sum = channel.subset_sum(&elems, &word);
	assert!(matches!(sum, SymbolicElem::Constant(c) if c == B128::ZERO));

	// A settled result means no gate was emitted at all, which the counts confirm.
	let stat = CircuitStat::collect(&builder.build());
	assert_eq!(stat.n_bmul_constraints, 0);
	assert_eq!(stat.n_and_constraints, 0);
}

#[test]
fn assert_zero_rejects_a_non_zero_folded_value() {
	// Invariant: a zero claim over a value the protocol already fixed is decided while the circuit
	// is built, so it emits no constraint either way.
	//
	// Fixture state: two claims, one over a folded zero and one over a folded one.
	let builder = CircuitBuilder::new();
	let mut channel = MerkleVerifierChannel::new(&builder);

	channel
		.assert_zero(SymbolicElem::Constant(B128::ZERO))
		.expect("a folded zero satisfies the assertion");

	// A folded non-zero value is a claim no witness could rescue, so it is reported here.
	match channel.assert_zero(SymbolicElem::Constant(B128::ONE)) {
		Err(binius_ip::channel::Error::InvalidAssert) => {}
		other => panic!("expected InvalidAssert, got {other:?}"),
	}

	// An error rather than an unsatisfiable circuit, which the empty counts confirm.
	let stat = CircuitStat::collect(&builder.build());
	assert_eq!(stat.n_and_constraints, 0, "a folded assertion emits no constraint");
	assert_eq!(stat.n_bmul_constraints, 0);
}

/// Asserts a wire pair holding `value` is zero, and reports what filling the circuit said.
fn run_assert_zero(value: B128) -> Result<CircuitStat, PopulateError> {
	let builder = CircuitBuilder::new();
	let mut channel = MerkleVerifierChannel::new(&builder);
	// Public wires, so the value under test is chosen by the filler rather than derived.
	let wires = [builder.add_inout(), builder.add_inout()];
	channel
		.assert_zero(channel.elem(wires[0], wires[1]))
		.expect("a wire-carried assertion is always recorded");

	let circuit = builder.build();
	let stat = CircuitStat::collect(&circuit);
	let mut w = circuit.new_witness_filler();
	// Low half first, matching the order the element reads its halves.
	for (wire, word) in wires.iter().zip(element_words(u128::from(value))) {
		w[*wire] = Word(word);
	}

	// Population evaluates the assertions, so a non-zero value is reported from here.
	circuit.populate_wire_witness(&mut w)?;
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("a satisfied circuit must verify");
	Ok(stat)
}

#[test]
fn assert_zero_constrains_a_wire_pair() {
	// Invariant: a wire-carried zero claim constrains both halves of the element, so a non-zero bit
	// anywhere in its 128 fails.
	//
	// Fixture state: one claim over a public wire pair, at 2 AND constraints.
	let stat = run_assert_zero(B128::ZERO).expect("a zero wire pair satisfies the assertion");
	assert_eq!(stat.n_zero_constraints, ELEMENT_WORDS, "one constraint per wire");
	assert_eq!(stat.n_and_constraints, 0);
	assert_eq!(stat.n_bmul_constraints, 0);

	// Mutation: a bit set in the low half, then a bit set in the high half.
	//
	//     1        ->  lo = 1, hi = 0  ->  the low assertion fails
	//     2^64     ->  lo = 0, hi = 1  ->  the high assertion fails
	for value in [B128::ONE, <B128 as From<u128>>::from(1u128 << 64)] {
		let Err(error) = run_assert_zero(value) else {
			panic!("a non-zero wire pair must not satisfy the assertion");
		};
		// A truncated failure list would let an unexamined path hide, so require the full one.
		assert_eq!(
			error.total,
			error.failures.len(),
			"the failure list must be complete, so every path can be checked"
		);
		assert!(error.total > 0, "an unsatisfied circuit must report a failure");
		for failure in &error.failures {
			// Every failure must come from the named subcircuit the claim was emitted into.
			assert!(
				failure.path.starts_with(".assert_zero[0]"),
				"unexpected failing assertion {:?}",
				failure.path
			);
			assert!(!failure.detail.is_empty(), "a failure must carry a diagnostic");
		}
	}
}

#[test]
fn compute_public_value_evaluates_symbolically() {
	use binius_field::util::FieldFn;
	use binius_transcript::fiat_shamir::CanSample;

	/// Squares its input and adds one, in whatever field it is run over.
	struct SquarePlusOne;

	impl FieldFn<B128> for SquarePlusOne {
		fn call<E: FieldOps<Scalar = B128> + From<B128>>(&self, inputs: &[E]) -> E {
			// One multiplication and one addition, so a circuit run must show one select-free
			// field multiplication and nothing more.
			inputs[0].clone().square() + E::from(B128::ONE)
		}
	}

	// Invariant: a public-value computation is emitted as gates, not hinted, so the circuit itself
	// carries the arithmetic.
	//
	// Fixture state: the function above over the channel's first challenge.
	//
	//     symbolic evaluation  ->  1 BMUL for the squaring
	//     a hint               ->  0 BMUL, an unconstrained hole
	//
	// The channel opens on a fresh challenger, so its first challenge is fixed and computable here.
	let mut probe = binius_transcript::VerifierTranscript::new(Challenger::default(), Vec::new());
	let challenge: B128 = CanSample::sample(&mut probe);
	probe.finalize().expect("the probe reads no tape");
	let expected = SquarePlusOne.call_native(&[challenge]);

	let builder = CircuitBuilder::new();
	let mut channel = MerkleVerifierChannel::new(&builder);
	let x = IPVerifierChannel::<B128>::sample(&mut channel);
	let y = channel.compute_public_value(&[x], SquarePlusOne);

	// The native answer enters on public wires, so the comparison becomes a constraint.
	let claimed = [builder.add_inout(), builder.add_inout()];
	let (lo, hi) = y.to_wires(&builder);
	builder.assert_eq_v("public_value", [lo, hi], claimed);

	let circuit = builder.build();
	let stat = CircuitStat::collect(&circuit);
	// A symbolic evaluation leaves the squaring in the circuit; a hint would leave nothing.
	assert_eq!(stat.n_bmul_constraints, 1, "the squaring must be constrained, not hinted");

	let mut w = circuit.new_witness_filler();
	// Only the expected value is written: the challenge and the squaring are both derived.
	for (wire, word) in claimed.iter().zip(element_words(u128::from(expected))) {
		w[*wire] = Word(word);
	}
	circuit
		.populate_wire_witness(&mut w)
		.expect("the circuit must reproduce the native evaluation");
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("every constraint must hold");
}

#[test]
fn shift_word_shifts_right() {
	// Invariant: a shift narrows a word on its wire, and narrows a build-time value in Rust, so the
	// two forms stay in agreement.
	//
	// Fixture state: one sampled word shifted by 5, and one fixed word shifted by 8.
	//
	//     0xdead_beef_1234_5678 >> 5  ->  checked in the circuit
	//     0xff00               >> 8   ->  0xff, settled in Rust
	let builder = CircuitBuilder::new();
	let wire = builder.add_inout();
	let channel = MerkleVerifierChannel::new(&builder);
	let word = SymbolicWord::wire(&channel.shared, wire);
	let shifted = word >> 5;
	assert_eq!(shifted.value(), None, "a sampled word stays sampled through a shift");
	// The shifted wire is pinned to a public wire the filler sets to the native shift.
	let claimed = builder.add_inout();
	builder.assert_eq("shifted", shifted.to_wire(&builder), claimed);

	// A fixed word shifts in Rust as well as on its wire.
	let fixed = SymbolicWord::from(Word::from_u64(0xff00));
	assert_eq!((fixed >> 8).value(), Some(Word::from_u64(0xff)));

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	let value = 0xdead_beef_1234_5678u64;
	w[wire] = Word(value);
	// The native shift, so the assertion holds only if the gate agrees with it.
	w[claimed] = Word(value >> 5);
	circuit
		.populate_wire_witness(&mut w)
		.expect("the shift must match the native word shift");
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("every constraint must hold");
}

#[test]
fn the_documented_per_operation_costs_hold() {
	// Invariant: the cost table in the module docs is the measured cost, not an estimate.
	//
	// Every measurement pins its result to public wires, otherwise dead-code elimination prunes the
	// whole circuit, so the claim's own constraints are subtracted back out.

	/// The AND and BMUL counts of a circuit built by `build`.
	fn cost(build: impl FnOnce(&CircuitBuilder, &mut MerkleVerifierChannel<'_>)) -> (usize, usize) {
		// A fresh channel per measurement, so no earlier operation's gates are counted.
		let builder = CircuitBuilder::new();
		let mut channel = MerkleVerifierChannel::new(&builder);
		build(&builder, &mut channel);
		let stat = CircuitStat::collect(&builder.build());
		(stat.n_and_constraints, stat.n_bmul_constraints)
	}

	// A received element is two witness wires and nothing else, so nothing lands in either column.
	let (and, bmul) = cost(|builder, channel| {
		let elem = channel.recv_one().expect("reading a message cannot fail");
		let (lo, hi) = elem.to_wires(builder);
		builder.assert_eq_v("recv", [lo, hi], [builder.add_inout(), builder.add_inout()]);
	});
	assert_eq!((and, bmul), (0, 0), "recv_one");

	// The seed digest serves the first sample with no compression, so this is the assembly alone.
	let (and, bmul) = cost(|builder, channel| {
		let elem = IPVerifierChannel::<B128>::sample(channel);
		let (lo, hi) = elem.to_wires(builder);
		builder.assert_eq_v("sample", [lo, hi], [builder.add_inout(), builder.add_inout()]);
	});
	assert_eq!((and, bmul), (12, 0), "sample");

	// A width of 17 needs the mask, so this is four bytes assembled and masked down.
	let (and, bmul) = cost(|builder, channel| {
		let word = WordIPVerifierChannel::<B128>::sample_bits(channel, 17);
		builder.assert_eq("bits", word.to_wire(builder), builder.add_inout());
	});
	assert_eq!((and, bmul), (4, 0), "sample_bits");

	// A digest read pays one byte reversal per word, four AND constraints apiece.
	let (and, bmul) = cost(|builder, channel| {
		let commitment = channel
			.recv_merkle_commitment(1, 0)
			.expect("reading a commitment cannot fail");
		builder.assert_eq_v("root", commitment.root, array::from_fn(|_| builder.add_inout()));
	});
	assert_eq!((and, bmul), (4 * DIGEST_WORDS, 0), "one digest read");
}
