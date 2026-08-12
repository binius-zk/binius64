// Copyright 2026 The Binius Developers

//! A verifier channel that builds a circuit instead of checking a proof.
//!
//! A verifier in this codebase never touches a transcript directly.
//! It talks to a channel, and the protocol code across the interactive-proof, IOP and verifier
//! crates is generic over which channel it is handed.
//!
//! Running that code against this channel verifies nothing.
//! It *records* the verification as a Binius64 circuit, satisfiable exactly when the proof it was
//! handed would have been accepted.
//!
//! Three gadgets are assembled here: field arithmetic over wires, the Fiat-Shamir state over wires,
//! and the Merkle commitment checks.
//!
//! # Against the skeleton channel
//!
//! [`Binius64BuilderChannel`](crate::Binius64BuilderChannel) records the same operations.
//! It leaves the hashing out, so a replay supplies what it could not derive:
//!
//! ```text
//!     skeleton:  proof -> replay -> wires the circuit could not derive
//!     this one:  proof -> wires, and every other value is a gate output
//! ```
//!
//! Deriving those values is what costs the SHA-256.
//! It is also what makes an unsatisfied circuit mean a rejected proof.
//!
//! # The proof stream
//!
//! A verifier transcript holds one byte tape, and two readers consume it strictly in order.
//! They differ in one respect only:
//!
//! ```text
//!     message reader:       reads the tape and feeds what it read to the Fiat-Shamir state
//!     decommitment reader:  reads the tape and feeds the Fiat-Shamir state nothing
//! ```
//!
//! Advice read through the second reader is sound because something already observed binds it: a
//! Merkle root read as a message.
//!
//! Getting that split wrong does not make the circuit unsatisfiable on its own.
//! It makes the circuit disagree with every real transcript, since the challenges it derives are
//! then those of a different byte sequence.
//!
//! That makes the split the load-bearing property of this module, so the classification is spelled
//! out:
//!
//! | read                                     | reader       | observed |
//! | ---------------------------------------- | ------------ | -------- |
//! | one received element                     | message      | yes      |
//! | a run of received elements               | message      | yes      |
//! | a commitment root                        | message      | yes      |
//! | an opening's layer, leaves and branches  | decommitment | no       |
//! | a whole committed vector                 | decommitment | no       |
//!
//! Every read is a whole number of eight-byte groups, since a field element is sixteen bytes and a
//! SHA-256 digest is thirty-two.
//! So the tape is modelled as a sequence of *proof words*, one witness wire each, in stream order:
//!
//! ```text
//!     word i  carries proof bytes 8i .. 8i + 8, read little-endian
//! ```
//!
//! # What is an input and what is a gate
//!
//! The proof bytes and the inner statement are the only circuit inputs.
//! Challenges, folded values and recomputed digests are gate outputs, so the compiled circuit
//! derives every one of them from those inputs on its own.
//!
//! - There is no replay of the verifier against a second channel.
//! - There is no build-versus-fill ordering to keep in step.
//! - Hints change none of that, since a hint also evaluates from wires the filler has settled.
//!
//! # Cost
//!
//! Counts are constraints as the frontend compiles them, with `n` the element count.
//! This module's tests pin every row.
//!
//! | operation                                  | AND | BMUL      | ZERO |
//! | ------------------------------------------ | --- | --------- | ---- |
//! | receiving one element                      | 0   | 0         | 0    |
//! | sampling one field element                 | 12  | 0         | 0    |
//! | sampling one index word                    | 4   | 0         | 0    |
//! | asserting a wire-carried element is zero   | 0   | 0         | 2    |
//! | a table lookup over `n` entries            | 0   | 2 (n - 1) | 0    |
//! | a bit-selected sum over `n` elements       | 0   | 2 n       | 0    |
//! | one digest read                            | 16  | 0         | 0    |
//!
//! An assertion spends the ZERO column, which carries no sumcheck of its own.
//!
//! The two sampling rows are the byte assembly alone.
//! A draw that empties the sampler forces a refill, and 32 bytes of sampler output cost about 700
//! AND — see the [challenger docs](crate::challenger).
//!
//! A digest read pays one byte reversal per word, turning the tape's little-endian words into the
//! big-endian halves a compression consumes.
//!
//! A table lookup and a bit-selected sum are the two operations that reach a word's bits, so their
//! cost is the select gates they spend.
//!
//! Both drop to zero when the index word is one the protocol fixed rather than sampled.
//! FRI's terminal fold walks its cosets at fixed indices, so every sum there settles while the
//! circuit is being built.

use std::{array, rc::Rc};

use binius_circuits::{bytes::swap_bytes_32, multiplexer::multi_wire_multiplex};
use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, Field, util::FieldFn};
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use binius_hash::sha256::Sha256HashSuite;
use binius_iop::{
	merkle_channel::MerkleIPVerifierChannel,
	merkle_tree::{BinaryMerkleTreeScheme, MerkleTreeScheme},
};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel, select_word, subset_sum_word};

use crate::{
	challenger::Sha256Challenger,
	merkle::{self, DIGEST_WORDS, Digest, ELEMENT_WORDS, Element, element_words},
	shared::Shared,
	symbolic::{SymbolicElem, SymbolicWord},
};

#[cfg(test)]
mod tests;

/// Proof bytes one proof word carries.
const WORD_BYTES: usize = Word::BITS / 8;

/// Which of a transcript's two readers a proof-stream read came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadKind {
	/// An observed read: the bytes are fed to the Fiat-Shamir state.
	Message,
	/// An unobserved read: the bytes are advice, bound by an already-observed commitment.
	Decommitment,
}

/// One read a verifier made from the proof stream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ProofRead {
	/// Which reader the bytes came from, and so whether the Fiat-Shamir state saw them.
	pub kind: ReadKind,
	/// Index of this read's first proof word.
	pub word_offset: usize,
	/// Proof words this read consumed.
	pub n_words: usize,
}

/// Where a circuit's proof byte stream lives, in the order the verifier read it.
///
/// Every read is a whole number of eight-byte groups, so the stream is a sequence of proof words:
///
/// ```text
///     word i  carries proof bytes 8i .. 8i + 8, read little-endian
/// ```
///
/// Filling a circuit from a raw byte tape needs nothing else, since the proof is the only
/// proof-side input.
/// The list of reads recorded alongside the wires is there for accounting and diagnostics.
#[derive(Clone, Debug, Default)]
pub struct ProofLayout {
	/// One witness wire per proof word, in stream order.
	words: Vec<Wire>,
	/// The reads that allocated those words, in order.
	reads: Vec<ProofRead>,
}

impl ProofLayout {
	/// The witness wires holding the proof stream, in order.
	pub fn words(&self) -> &[Wire] {
		// Stream order is the order the verifier read the tape in, which is what population relies
		// on.
		&self.words
	}

	/// The reads the verifier made, in order.
	pub fn reads(&self) -> &[ProofRead] {
		// Exposed for accounting: a caller can confirm which reads were observed and which were
		// not.
		&self.reads
	}

	/// Bytes of proof the circuit reads.
	pub const fn n_bytes(&self) -> usize {
		// Every proof word is eight tape bytes, so the wire count fixes the byte length exactly.
		self.words.len() * WORD_BYTES
	}

	/// Writes a proof byte tape into the wires that carry it.
	///
	/// This is the whole of witness population for the proof.
	/// Nothing else the channel produced is an input, so the compiled circuit derives the rest.
	///
	/// # Errors
	///
	/// Returns an error unless the tape is exactly as long as the circuit reads.
	///
	/// - A short tape would leave wires unset.
	/// - A long one means the circuit and the prover disagree about the protocol.
	pub fn populate(&self, w: &mut WitnessFiller, proof: &[u8]) -> Result<(), ProofLengthError> {
		// Neither a short nor a long tape is worth filling, so the length is checked up front.
		if proof.len() != self.n_bytes() {
			return Err(ProofLengthError {
				expected: self.n_bytes(),
				actual: proof.len(),
			});
		}
		// Wire i takes tape bytes 8i to 8i + 8, which is the convention every read shares.
		for (&wire, chunk) in self.words.iter().zip(proof.chunks_exact(WORD_BYTES)) {
			let bytes = chunk
				.try_into()
				.expect("chunks_exact yields chunks of exactly WORD_BYTES");
			// Little-endian, matching how the transcript serialized the bytes in the first place.
			w[wire] = Word(u64::from_le_bytes(bytes));
		}
		Ok(())
	}
}

/// The proof handed to a layout is not the length the circuit reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug, thiserror::Error)]
#[error("the circuit reads {expected} proof bytes, but {actual} were supplied")]
pub struct ProofLengthError {
	/// Bytes the circuit reads.
	pub expected: usize,
	/// Bytes supplied.
	pub actual: usize,
}

/// A Merkle commitment a verifier channel received.
#[derive(Clone, Debug)]
pub struct MerkleCommitment {
	/// The commitment root, on wires.
	pub root: Digest,
	/// Depth of the committed tree.
	pub depth: usize,
	/// Field elements each leaf holds.
	pub leaf_size: usize,
}

/// A verifier channel that records a verification as a circuit on a builder.
///
/// Drive it by running an ordinary verifier over it, take the proof layout it recorded, and build:
///
/// ```text
///     let builder = CircuitBuilder::new();
///     let mut channel = MerkleVerifierChannel::new(&builder);
///     verify_something(&mut channel, statement)?;      // emits the gates
///     let layout = channel.finish();
///     let circuit = builder.build();
///     layout.populate(&mut filler, proof_bytes)?;      // the only proof-side input
/// ```
///
/// Each call emits its gates immediately rather than deferring them.
/// That is why the builder is borrowed for this object's whole life.
///
/// See the [module docs](self) for the proof-stream convention and the cost of each operation.
pub struct MerkleVerifierChannel<'a> {
	/// Where this channel's gates are emitted.
	builder: &'a CircuitBuilder,
	/// The anchor a [`SymbolicElem`] holds so its arithmetic can reach the builder.
	///
	/// Its input list stays empty, since nothing here is left for a replay to supply.
	shared: Rc<Shared>,
	/// The in-circuit Fiat-Shamir state.
	challenger: Sha256Challenger<'a>,
	/// The proof stream read so far.
	layout: ProofLayout,
	/// The inout wires the inner statement enters on, in the order it was observed.
	statement: Vec<Wire>,
	/// Source of the layer depth an opening decommits to, shared with the native verifier.
	scheme: BinaryMerkleTreeScheme<B128, Sha256HashSuite>,
	/// Merkle verifications emitted so far, used to name subcircuits.
	n_merkle_checks: usize,
	/// Zero assertions emitted so far, used to name them.
	n_assertions: usize,
}

impl<'a> MerkleVerifierChannel<'a> {
	/// Creates a channel over a fresh Fiat-Shamir state, having read nothing.
	pub fn new(builder: &'a CircuitBuilder) -> Self {
		Self {
			builder,
			shared: Rc::new(Shared::with_builder(builder.clone())),
			// The Fiat-Shamir state opens on its protocol seed, which is a constant and so free.
			challenger: Sha256Challenger::new(builder),
			// No proof word has been read yet, so the stream starts empty.
			layout: ProofLayout::default(),
			// The statement arrives through `observe_words`, which has not been called yet.
			statement: Vec::new(),
			// Consulted only for the layer depth an opening decommits to, which must be the depth
			// the native verifier would have picked.
			scheme: BinaryMerkleTreeScheme::new(),
			n_merkle_checks: 0,
			n_assertions: 0,
		}
	}

	/// The proof stream read so far.
	pub const fn proof_layout(&self) -> &ProofLayout {
		// Readable mid-verification, unlike the consuming accessor, so a caller can measure as it
		// goes.
		&self.layout
	}

	/// The inout wires the inner statement entered on, in observation order.
	pub fn statement(&self) -> &[Wire] {
		// One wire per statement word, which is what a caller populates alongside the proof.
		&self.statement
	}

	/// Consumes the channel and returns the proof stream it read.
	pub fn finish(self) -> ProofLayout {
		// The layout outlives the channel because filling a witness happens long after the last
		// gate is emitted.
		self.layout
	}

	/// Allocates `n` proof words, recording the read and observing it when it is a message.
	fn read_words(&mut self, kind: ReadKind, n: usize) -> Vec<Wire> {
		// A read of nothing must not turn the Fiat-Shamir channel: a native reader only turns it
		// once it advances the tape.
		//
		// An explicit request to observe nothing is the opposite case, and does turn it.
		if n == 0 {
			return Vec::new();
		}

		// The stream is append-only, so this read starts where the previous one ended.
		let word_offset = self.layout.words.len();
		// One witness wire per eight tape bytes, which is the only proof-side input there is.
		let words = (0..n)
			.map(|_| self.builder.add_witness())
			.collect::<Vec<_>>();
		self.layout.words.extend_from_slice(&words);
		// Recorded so the stream can be audited read by read, and filled in this same order.
		self.layout.reads.push(ProofRead {
			kind,
			word_offset,
			n_words: n,
		});

		// Only a message read is observed, which is the whole correctness content of the split.
		if kind == ReadKind::Message {
			self.challenger.observe_words(&words);
		}
		words
	}

	/// Reads `n` field elements from the proof stream.
	fn read_elements(&mut self, kind: ReadKind, n: usize) -> Vec<Element> {
		// Two words per element, low half first, which is how the tape serializes one.
		self.read_words(kind, n * ELEMENT_WORDS)
			.chunks_exact(ELEMENT_WORDS)
			.map(|words| array::from_fn(|k| words[k]))
			.collect()
	}

	/// Reads `n` digests from the proof stream.
	///
	/// The tape carries a digest as bytes, so its proof words are little-endian.
	/// A compression instead consumes big-endian halves, and one byte reversal per word bridges the
	/// two.
	fn read_digests(&mut self, kind: ReadKind, n: usize) -> Vec<Digest> {
		// Four words per digest, each reversed into the form the hashing gadgets read.
		self.read_words(kind, n * DIGEST_WORDS)
			.chunks_exact(DIGEST_WORDS)
			.map(|words| array::from_fn(|j| swap_bytes_32(self.builder, words[j])))
			.collect()
	}

	/// A subcircuit named for the next Merkle verification.
	fn merkle_subcircuit(&mut self, what: &str) -> CircuitBuilder {
		// A distinct name per check keeps a failing assertion traceable to the check that broke.
		let name = format!("{what}[{}]", self.n_merkle_checks);
		self.n_merkle_checks += 1;
		self.builder.subcircuit(name)
	}

	/// Wraps a `(lo, hi)` wire pair as an element anchored to this channel's builder.
	///
	/// This is how a caller hands the verifier a statement it allocated its own wires for.
	pub fn elem(&self, lo: Wire, hi: Wire) -> SymbolicElem {
		SymbolicElem::wires(&self.shared, lo, hi)
	}

	/// Lifts every element onto wires and returns them as one group per element.
	fn lower(
		&self,
		builder: &CircuitBuilder,
		elems: &[SymbolicElem],
	) -> Vec<[Wire; ELEMENT_WORDS]> {
		elems
			.iter()
			.map(|elem| {
				// An element settled at build time becomes constant wires here, so whatever reads
				// the table sees one uniform shape.
				let (lo, hi) = elem.to_wires(builder);
				[lo, hi]
			})
			.collect()
	}
}

impl IPVerifierChannel<B128> for MerkleVerifierChannel<'_> {
	type Elem = SymbolicElem;

	fn recv_one(&mut self) -> Result<SymbolicElem, binius_ip::channel::Error> {
		// A prover message is observed: it is part of the byte sequence the challenges derive from.
		let words = self.read_words(ReadKind::Message, ELEMENT_WORDS);
		// Low half first, the order both the tape and a field multiplication use.
		Ok(self.elem(words[0], words[1]))
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<SymbolicElem>, binius_ip::channel::Error> {
		// One read rather than n, matching the native channel's single slice read, so the words
		// land contiguously and the reads stay comparable between the two sides.
		Ok(self
			.read_elements(ReadKind::Message, n)
			.into_iter()
			// An element off the tape is always wire-carried, never settled at build time.
			.map(|[lo, hi]| self.elem(lo, hi))
			.collect())
	}

	fn sample(&mut self) -> SymbolicElem {
		// Sixteen bytes of sampler output, packed little-endian into an element's two halves.
		let (lo, hi) = self.challenger.sample_b128();
		self.elem(lo, hi)
	}

	fn observe_one(&mut self, val: B128) -> SymbolicElem {
		// A value the verifier already holds concretely, so it is observed as a constant.
		self.observe_many(&[val]);
		// Handing it back settled is what lets the caller's arithmetic over it fold away.
		SymbolicElem::Constant(val)
	}

	fn observe_many(&mut self, vals: &[B128]) -> Vec<SymbolicElem> {
		// These are values the verifier computed rather than read, so they enter as constant wires,
		// which the frontend charges nothing for.
		let words = vals
			.iter()
			.flat_map(|val| element_words(u128::from(*val)))
			.map(|word| self.builder.add_constant_64(word))
			.collect::<Vec<_>>();
		// An empty slice is not a no-op: asking for nothing to be observed still turns the
		// Fiat-Shamir channel, exactly as a native observe does.
		self.challenger.observe_words(&words);
		// Nothing reached a wire, so every value stays settled for the caller.
		vals.iter().copied().map(SymbolicElem::Constant).collect()
	}

	fn assert_zero(&mut self, val: SymbolicElem) -> Result<(), binius_ip::channel::Error> {
		match val {
			SymbolicElem::Constant(c) => {
				// A value the protocol already fixed is decided right here, with no gate.
				if c == B128::ZERO {
					Ok(())
				} else {
					// A fixed non-zero value is a claim no witness could rescue, so it is reported
					// rather than turned into a constraint nothing can satisfy.
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			SymbolicElem::Wires { lo, hi, .. } => {
				// A distinct name per assertion keeps a failure traceable to the claim that broke.
				let builder = self
					.builder
					.subcircuit(format!("assert_zero[{}]", self.n_assertions));
				self.n_assertions += 1;
				// All 128 bits must vanish, so each half is asserted on its own.
				builder.assert_zero("lo", lo);
				builder.assert_zero("hi", hi);
				Ok(())
			}
		}
	}

	fn compute_public_value(
		&mut self,
		inputs: &[SymbolicElem],
		f: impl FieldFn<B128>,
	) -> SymbolicElem {
		// Evaluated symbolically rather than hinted: there is no outer verifier to rebind the
		// result, so a hint here would be an unconstrained hole the circuit could not close.
		f.call::<SymbolicElem>(inputs)
	}
}

impl WordIPVerifierChannel<B128> for MerkleVerifierChannel<'_> {
	type Word = SymbolicWord;

	fn observe_words(&mut self, words: &[Word]) -> Vec<SymbolicWord> {
		// One inout wire per statement word: as build-time constants these numbers would fold into
		// the gates, tying the circuit to one instance of the statement rather than to all of them.
		let wires = words
			.iter()
			.map(|_| self.builder.add_inout())
			.collect::<Vec<_>>();
		self.statement.extend_from_slice(&wires);
		// Absorbing them is what binds the derived challenges to the statement being verified.
		self.challenger.observe_words(&wires);
		wires
			.into_iter()
			.map(|wire| SymbolicWord::wire(&self.shared, wire))
			.collect()
	}

	/// Adds the elements whose bit is set in the given word, element `i` gated by bit `i`.
	fn subset_sum(&mut self, elems: &[SymbolicElem], word: &SymbolicWord) -> SymbolicElem {
		assert!(elems.len() <= Word::BITS, "precondition: at most one element per bit");

		// A fixed word settles which elements the sum runs over, and addition itself is free.
		if let Some(value) = word.value() {
			return subset_sum_word(elems, value);
		}

		// One subcircuit gathers this sum's select gates under a single path.
		let builder = self.builder.subcircuit("subset_sum");
		// The word is derived, so this lowering is the wire it already carries.
		let word_wire = word.to_wire(&builder);
		// What a cleared bit contributes to the sum.
		let zero = builder.add_constant(Word::ZERO);

		let mut terms = Vec::with_capacity(elems.len());
		for (bit, elem) in elems.iter().enumerate() {
			// A zero element contributes nothing whichever way its bit falls, so it needs no gate.
			if matches!(elem, SymbolicElem::Constant(c) if *c == B128::ZERO) {
				continue;
			}
			let (lo, hi) = elem.to_wires(&builder);
			// A select gate reads bit 63 alone, so the bit gating this element is lifted to it.
			let selected = builder.shl(word_wire, (Word::BITS - 1 - bit) as u32);
			// One select per half: the element when its bit is set, zero when it is clear.
			terms.push([
				builder.select(selected, lo, zero),
				builder.select(selected, hi, zero),
			]);
		}
		// Nothing could ever contribute, so the sum is the additive identity and stays settled.
		if terms.is_empty() {
			return SymbolicElem::Constant(B128::ZERO);
		}

		// Addition in characteristic 2 is a XOR, so accumulating the selected terms costs nothing.
		let fold = |k: usize| {
			// Column k is half k of every term, and the halves never mix.
			let column = terms.iter().map(|term| term[k]).collect::<Vec<_>>();
			builder.bxor_multi(&column)
		};
		self.elem(fold(0), fold(1))
	}

	/// Reads the table entry the low bits of the given word address.
	fn select(&mut self, elems: &[SymbolicElem], word: &SymbolicWord) -> SymbolicElem {
		assert!(
			!elems.is_empty() && elems.len().is_power_of_two(),
			"precondition: a power-of-two number of elements"
		);

		// A one-entry table needs no gate, since the index addresses no bit at all.
		if let [only] = elems {
			return only.clone();
		}
		// A fixed word settles which entry is read, so the lookup runs in Rust.
		if let Some(value) = word.value() {
			return select_word(elems, value);
		}

		// One subcircuit gathers the multiplexer's select gates under a single path.
		let builder = self.builder.subcircuit("select");
		// Settled entries become constant wires here, so every entry is two wires wide.
		let groups = self.lower(&builder, elems);
		let entries = groups.iter().map(|group| &group[..]).collect::<Vec<_>>();
		// A tree of selects over the entries, both wires of an entry moving together.
		let selected = multi_wire_multiplex(&builder, &entries, word.to_wire(&builder));
		self.elem(selected[0], selected[1])
	}

	fn sample_bits(&mut self, bits: usize) -> SymbolicWord {
		// The Fiat-Shamir gadget masks the draw in-circuit, so the result provably lies below
		// 2^bits, which is what lets a protocol drop its own index range check.
		let wire = self.challenger.sample_bits(bits);
		// A sampled word has no build-time value, so a lookup on it spends select gates.
		SymbolicWord::wire(&self.shared, wire)
	}

	fn pack_words(&mut self, words: &[SymbolicWord]) -> Vec<SymbolicElem> {
		// A `SymbolicElem` *is* the low and high wire of a 128-bit element, and a word fills half
		// of it, so packing is pairing the wires up. It costs no gates, and a trailing odd word
		// takes the low half against a zero high half.
		words
			.chunks(ELEMENT_WORDS)
			.map(|chunk| {
				let lo = chunk[0].to_wire(self.builder);
				let hi = chunk.get(1).map_or_else(
					|| self.builder.add_constant_64(0),
					|word| word.to_wire(self.builder),
				);
				self.elem(lo, hi)
			})
			.collect()
	}
}

impl MerkleIPVerifierChannel<B128> for MerkleVerifierChannel<'_> {
	type Commitment = MerkleCommitment;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<MerkleCommitment, binius_iop::merkle_channel::Error> {
		// The root is a message, and observing it is what binds every unobserved read below it.
		let root = self.read_digests(ReadKind::Message, 1)[0];
		// Depth and leaf width come from the statement rather than the tape, so they need no wire.
		Ok(MerkleCommitment {
			root,
			depth,
			leaf_size,
		})
	}

	/// Opens the commitment at every query index, returning the elements the opened leaves hold.
	///
	/// # Index range
	///
	/// - The native channel rejects an index at or above the tree's leaf count.
	/// - A wire holds no value while the circuit is built, so no such comparison is possible here.
	/// - An index is instead opened modulo the leaf count, since only its low bits are ever read.
	/// - A sampled index is masked in-circuit and a shift only narrows, so an index a protocol
	///   derived from a challenge is already in range.
	fn recv_openings(
		&mut self,
		commitment: &MerkleCommitment,
		indices: &[SymbolicWord],
	) -> Result<Vec<SymbolicElem>, binius_iop::merkle_channel::Error> {
		let tree_depth = commitment.depth;
		// The same rule the native verifier applies, so both sides stop climbing at the same level.
		let layer_depth = self.scheme.optimal_verify_layer(indices.len(), tree_depth);

		// Phase 1: one internal layer, read once and folded up to the root.
		//
		// The layer is advice, and the root already read as a message is what binds it.
		let layer_digests = self.read_digests(ReadKind::Decommitment, 1 << layer_depth);
		let builder = self.merkle_subcircuit("layer");
		merkle::verify_layer(&builder, commitment.root, &layer_digests);

		// Phase 2: every query climbs from its own leaf up to that shared layer.
		let mut values = Vec::with_capacity(indices.len() * commitment.leaf_size);
		for index in indices {
			// Leaf and branch are advice too, bound by the layer the root binds.
			let leaf = self.read_elements(ReadKind::Decommitment, commitment.leaf_size);
			let branch = self.read_digests(ReadKind::Decommitment, tree_depth - layer_depth);

			// Hashes the leaf, climbs the branch, then matches the layer entry the index addresses.
			let builder = self.merkle_subcircuit("opening");
			merkle::verify_opening(
				&builder,
				index.to_wire(&builder),
				&leaf,
				layer_depth,
				tree_depth,
				&layer_digests,
				&branch,
			);
			// The opened elements are wire-carried, and the check above is what makes them usable.
			values.extend(leaf.into_iter().map(|[lo, hi]| self.elem(lo, hi)));
		}
		Ok(values)
	}

	fn recv_committed_vector(
		&mut self,
		commitment: &MerkleCommitment,
	) -> Result<Vec<SymbolicElem>, binius_iop::merkle_channel::Error> {
		// One leaf's worth of elements per leaf, across every leaf of the tree.
		let len = commitment.leaf_size << commitment.depth;
		// The data is advice, bound by the root the commitment read as a message.
		let data = self.read_elements(ReadKind::Decommitment, len);

		// The data is the whole opening, so the tree is rebuilt over it rather than climbed.
		let builder = self.merkle_subcircuit("vector");
		merkle::verify_vector(&builder, commitment.root, &data, commitment.leaf_size);

		// Every element is wire-carried, and the rebuild above is what makes them usable.
		Ok(data.into_iter().map(|[lo, hi]| self.elem(lo, hi)).collect())
	}
}
