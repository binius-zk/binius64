// Copyright 2026 The Binius Developers

//! The proof system's Fiat-Shamir challenger, re-expressed as a circuit.
//!
//! A recursive verifier derives its own challenges instead of being handed them.
//! So it must recompute the native challenger's byte stream exactly, over wires.
//!
//! # Build-time shape, run-time bytes
//!
//! The native challenger's control flow never reads the data it hashes.
//! All of this follows from the *sequence* of operations alone:
//!
//! - buffer occupancy, and where block boundaries fall
//! - the running count of sampled bytes consumed
//! - the padding bytes and the length field
//!
//! Each of those therefore lives in an ordinary Rust field, settled while the circuit is built.
//! Only byte *values* ever reach a wire.
//!
//! Two things fall out of that split:
//!
//! - The circuit is a straight line, with no multiplexer, comparison or conditional anywhere.
//! - The bookkeeping is free, since counts and padding become constants and constants cost nothing.
//!
//! # The state machine being mirrored
//!
//! The native challenger is one SHA-256 hasher plus two halves that take turns:
//!
//! ```text
//!     observe:  bytes are appended to the hasher's input
//!     sample:   bytes are read out of a 32-byte buffer holding the latest digest
//! ```
//!
//! Four events matter, and each is reproduced here exactly.
//!
//! 1. **Construction** — the hasher is fed the digest of the empty string, which also seeds the
//!    sample buffer.
//! 2. **Buffer refill** — the hasher is finalized *and reset*, its digest fed back into the fresh
//!    hasher and also served as the next 32 sampled bytes.
//! 3. **Sample to observe** — the count of sampled bytes consumed so far is appended, little-endian
//!    in eight bytes.
//! 4. **Observe to sample** — the buffer is retired, so the next byte read forces a refill.
//!
//! An *epoch* is the span between two resets.
//! Its input always opens with the previous digest's 32 bytes, so an epoch is never empty.
//! One consequence: the first 32 bytes the sampler serves are constants, compressing nothing.
//!
//! The native observer also stages bytes in a 64-byte buffer before handing them over.
//! That staging is invisible here, for two reasons:
//!
//! - A digest depends on the concatenated input, not on how it was chunked.
//! - The staging buffer is always flushed before a finalize.
//!
//! # Wire conventions
//!
//! Everything is phrased in *stream bytes*: the sequence the native challenger absorbs, and the one
//! it hands out.
//! Two wire shapes carry them.
//!
//! ```text
//!     byte wire:  one stream byte in bits [0, 8), and bits [8, 64) must be zero
//!     word wire:  eight stream bytes little-endian, stream byte k in bits [8k, 8k + 8)
//! ```
//!
//! Little-endian is the transcript's own convention, not a choice made here.
//! A 128-bit challenge is 16 little-endian bytes and a bit sample reads four.
//! So a word wire lines up with a transcript word without any shuffling.
//!
//! SHA-256 instead consumes big-endian 32-bit words.
//! Observing a word wire therefore pays a byte reversal, while a byte wire is already in order and
//! costs nothing.
//!
//! # Cost
//!
//! Measured by the tests at the bottom of this module.
//!
//! | item | AND constraints |
//! |---|---|
//! | one compression core, covering two chained blocks | 720 |
//! | one observed 64-byte block, amortized | 400 |
//! | one observed word, for the byte reversal | 5 |
//! | one 128-bit challenge, refills excluded | 12 |
//! | one bit sample, refills excluded | 4 |
//! | a refill's digest masking, on top of its epoch's compressions | 8 |
//!
//! No multiplication constraints are emitted, at any size.
//!
//! Consecutive blocks of an epoch chain, so they compress in pairs.
//! One two-lane core runs both, one per 32-bit lane of a 64-bit datapath.
//! A one-lane compression costs about 690, so pairing nearly halves the dominant term.
//!
//! An epoch that observes nothing is a single block.
//! So 32 bytes of sampler output cost about 700 constraints, however they are cut into values.

use std::{array, mem};

use binius_circuits::{
	bytes::swap_bytes_32,
	sha256::{State, sha256_compress, sha256_compress_2x_seq},
	util::clear_high_bits,
};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

/// Bytes SHA-256 compresses at a time.
const BLOCK_BYTES: usize = 64;

/// Message-schedule words in one compression block.
const BLOCK_WORDS: usize = BLOCK_BYTES / 4;

/// Bytes in a SHA-256 digest, and so in one full sampler buffer.
const DIGEST_BYTES: usize = 32;

/// 32-bit words in a SHA-256 digest.
const DIGEST_WORDS: usize = DIGEST_BYTES / 4;

/// `SHA-256("")`, as eight big-endian 32-bit words.
///
/// The native challenger seeds itself with this digest twice over: it is both the hasher's first
/// 32 bytes of input and the first 32 bytes the sampler hands out.
const SEED: [u32; DIGEST_WORDS] = [
	0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924, 0x27ae41e4, 0x649b934c, 0xa495991b, 0x7852b855,
];

/// Which side of the channel is live.
///
/// The native challenger is an enum over its two halves, and switching halves is what emits the
/// transition bookkeeping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Channel {
	/// Bytes read out of the digest buffer.
	Sample,
	/// Bytes appended to the hasher.
	Observe,
}

/// The native Fiat-Shamir challenger, rebuilt over wires.
///
/// Drive it in exactly the order the native challenger is driven.
/// Every sampled wire then carries the value the native challenger would have produced.
///
/// Each call emits its gates immediately rather than deferring them.
///
/// The builder is held by value rather than borrowed.
/// A `CircuitBuilder` is a handle, so a clone emits into the same circuit.
/// That lets a channel own both the builder and a challenger over it.
///
/// See the [module docs](self) for the byte-packing conventions and the constraint cost.
pub struct Sha256Challenger {
	/// Where this gadget's gates are emitted.
	builder: CircuitBuilder,
	/// Midstate over the blocks of the current epoch already compressed.
	state: State,
	/// A completed block held back so the next one can share a two-lane compression core.
	pending: Option<[Wire; BLOCK_WORDS]>,
	/// The block being filled, as big-endian schedule words, each clean in its low 32 bits.
	block: [Wire; BLOCK_WORDS],
	/// Bytes absorbed into the current epoch, counting the ones still in `block`.
	absorbed: usize,
	/// The digest seeding the sampler, as big-endian 32-bit words.
	buffer: [Wire; DIGEST_WORDS],
	/// Bytes of `buffer` already handed out. `DIGEST_BYTES` means a refill is due.
	index: usize,
	/// The live channel.
	channel: Channel,
	/// Compression cores emitted so far, used to name subcircuits.
	n_cores: usize,
}

impl Sha256Challenger {
	/// Creates a challenger in the state a freshly defaulted native challenger starts in.
	///
	/// The seed digest is a constant, so nothing is committed and nothing is compressed yet.
	pub fn new(builder: &CircuitBuilder) -> Self {
		// The seed is fixed by the protocol, so it enters as constants and costs no constraints.
		let buffer = array::from_fn(|i| builder.add_constant_64(SEED[i] as u64));

		// Fresh state: an empty epoch, a full sample buffer, and the sampler live.
		let mut this = Self {
			builder: builder.clone(),
			state: State::iv(builder),
			pending: None,
			block: [builder.add_constant(Word::ZERO); BLOCK_WORDS],
			absorbed: 0,
			buffer,
			index: 0,
			channel: Channel::Sample,
			n_cores: 0,
		};
		// The hasher opens on the seed digest, which is also what the sampler serves first.
		for word in buffer {
			this.absorb_be32(word);
		}
		this
	}

	/// Observes one stream byte per wire.
	///
	/// An empty slice is not a no-op.
	/// It still turns the channel, which is what asking the native challenger to observe nothing
	/// does.
	///
	/// # Preconditions
	///
	/// - Each wire holds its byte in bits `[0, 8)`, with bits `[8, 64)` zero.
	/// - A wire violating that corrupts its neighbours in the message schedule, so the caller owns
	///   it — as with the underlying compression gadget.
	/// - Byte extraction on the builder produces wires that satisfy it.
	pub fn observe_bytes(&mut self, bytes: &[Wire]) {
		self.switch_to_observe();
		for &byte in bytes {
			self.absorb_byte(byte);
		}
	}

	/// Observes eight little-endian stream bytes per wire.
	///
	/// All 64 bits of every wire are observed, so there is no precondition to honour.
	///
	/// An empty slice still turns the channel, as when observing bytes.
	pub fn observe_words(&mut self, words: &[Wire]) {
		self.switch_to_observe();
		for &word in words {
			self.absorb_word(word);
		}
	}

	/// Samples 16 stream bytes as one 128-bit challenge, in two 64-bit halves.
	///
	/// A field element deserializes from 16 little-endian bytes, which fixes the split:
	///
	/// ```text
	///     returned.0:  stream bytes 0 to 7,   packed little-endian
	///     returned.1:  stream bytes 8 to 15,  packed little-endian
	/// ```
	///
	/// That is the operand order a 128-bit field multiplication constraint expects.
	pub fn sample_b128(&mut self) -> (Wire, Wire) {
		self.switch_to_sample();
		let low = self.sample_u64_le();
		let high = self.sample_u64_le();
		(low, high)
	}

	/// Samples a value of the requested bit width, zero-extended into a 64-bit wire.
	///
	/// Four little-endian stream bytes are read and masked down, with the width clamped to 32.
	///
	/// # Why the mask is a constraint, not a convention
	///
	/// The mask is a real gate on a derived wire, so the result *provably* lies below `2^bits`.
	/// That is what lets a query index be used without a separate range check.
	pub fn sample_bits(&mut self, bits: usize) -> Wire {
		self.switch_to_sample();
		// Reading past 32 bits is meaningless here: only four stream bytes are consumed.
		let bits = bits.min(u32::BITS as usize);

		// Little-endian assembly: stream byte k contributes at bit 8k.
		let terms: [Wire; 4] = array::from_fn(|k| {
			let byte = self.sample_byte();
			self.builder.shl(byte, 8 * k as u32)
		});
		// The four contributions occupy disjoint bits, so XOR is the same as addition here.
		let unmasked = self.builder.bxor_multi(&terms);

		// Every byte is zero-extended, so the assembled word is already below 2^32.
		if bits == u32::BITS as usize {
			return unmasked;
		}
		let mask = self.builder.add_constant_64((1u64 << bits) - 1);
		self.builder.band(unmasked, mask)
	}

	/// Switches to the observe channel, appending the transition bytes if the channel turned.
	fn switch_to_observe(&mut self) {
		if self.channel == Channel::Observe {
			return;
		}
		self.channel = Channel::Observe;
		// Binding how much sampler output was consumed is what stops a prover rewinding it.
		for byte in (self.index as u64).to_le_bytes() {
			self.absorb_const_byte(byte);
		}
	}

	/// Switches to the sample channel, retiring the buffer if the channel turned.
	fn switch_to_sample(&mut self) {
		if self.channel == Channel::Sample {
			return;
		}
		self.channel = Channel::Sample;
		// Marking the buffer spent is the whole transition: the next byte read refills it.
		self.buffer = [self.builder.add_constant(Word::ZERO); DIGEST_WORDS];
		self.index = DIGEST_BYTES;
	}

	/// Hands out the next stream byte, zero-extended into bits `[0, 8)`.
	fn sample_byte(&mut self) -> Wire {
		// A spent buffer is refilled lazily, so a caller never has to think about epoch boundaries.
		if self.index == DIGEST_BYTES {
			self.refill();
		}
		// Four bytes per digest word, so the cursor splits into word index and byte within it.
		let word = self.buffer[self.index / 4];
		// Big-endian: byte 0 of a word is its most significant one.
		let shift = 8 * (3 - self.index % 4) as u32;
		self.index += 1;

		let shifted = self.builder.shr(word, shift);
		// A digest word is zero above bit 32, so its top byte comes out alone.
		if shift == 24 {
			shifted
		} else {
			let mask = self.builder.add_constant_zx_8(0xff);
			self.builder.band(shifted, mask)
		}
	}

	/// Hands out the next eight stream bytes, packed little-endian into one wire.
	fn sample_u64_le(&mut self) -> Wire {
		// Stream byte k lands at bit 8k, and the eight bytes may straddle a refill.
		let terms: [Wire; 8] = array::from_fn(|k| {
			let byte = self.sample_byte();
			self.builder.shl(byte, 8 * k as u32)
		});
		// Disjoint bit ranges, so XOR assembles the word.
		self.builder.bxor_multi(&terms)
	}

	/// Closes the epoch and reopens a fresh one seeded by the digest it produced.
	///
	/// # Algorithm
	///
	/// ```text
	///     1. pad the epoch      0x80, zeros, then its own bit length
	///     2. compress the rest  at most one block is still unpaired
	///     3. take the digest    the midstate, with its high half cleaned
	///     4. reopen             reset, then absorb that digest as the new epoch's first 32 bytes
	///     5. serve              the same digest becomes the next 32 sampled bytes
	/// ```
	fn refill(&mut self) {
		// The length field records the epoch's own input, so read it before padding lengthens it.
		let bit_len = self.absorbed as u64 * 8;

		// SHA-256 padding: 0x80, zeros up to 56 bytes into a block, then the big-endian bit length.
		self.absorb_const_byte(0x80);
		while self.absorbed % BLOCK_BYTES != BLOCK_BYTES - 8 {
			self.absorb_const_byte(0);
		}
		for byte in bit_len.to_be_bytes() {
			self.absorb_const_byte(byte);
		}

		// Padding always ends on a block boundary, so at most one block is still unpaired.
		if let Some(block) = self.pending.take() {
			self.compress_single(block);
		}

		// A two-lane core leaves its partner's state in the high half, and sampling reads all
		// 64 bits, so the digest is the one place a clean high half is required.
		let digest: [Wire; DIGEST_WORDS] =
			array::from_fn(|i| clear_high_bits(&self.builder, self.state.0[i], 32));

		self.state = State::iv(&self.builder);
		self.block = [self.builder.add_constant(Word::ZERO); BLOCK_WORDS];
		self.absorbed = 0;
		self.buffer = digest;
		self.index = 0;

		// Feeding the digest back into the reset hasher opens the next epoch.
		for word in digest {
			self.absorb_be32(word);
		}
	}

	/// Absorbs eight stream bytes held little-endian in one wire.
	fn absorb_word(&mut self, word: Wire) {
		if !self.absorbed.is_multiple_of(4) {
			// An unaligned cursor splits the word across schedule words, so go byte by byte.
			for j in 0..8 {
				let byte = self.builder.extract_byte(word, j);
				self.absorb_byte(byte);
			}
			return;
		}
		// Reversing the bytes of each 32-bit half turns the little-endian pair into the two
		// big-endian schedule words, already in order.
		let swapped = swap_bytes_32(&self.builder, word);
		let first = clear_high_bits(&self.builder, swapped, 32);
		let second = self.builder.shr(swapped, 32);
		self.absorb_be32(first);
		self.absorb_be32(second);
	}

	/// Absorbs one big-endian 32-bit stream word, clean in its low 32 bits.
	///
	/// # Panics
	///
	/// Panics unless the byte cursor is 4-byte aligned.
	fn absorb_be32(&mut self, word: Wire) {
		assert!(
			self.absorbed.is_multiple_of(4),
			"absorb_be32 needs a 4-byte aligned cursor, got {}",
			self.absorbed
		);
		self.absorb_placed(word, 4);
	}

	/// Absorbs one stream byte, zero-extended into bits `[0, 8)` of `byte`.
	fn absorb_byte(&mut self, byte: Wire) {
		let placed = self.builder.shl(byte, self.byte_shift());
		self.absorb_placed(placed, 1);
	}

	/// Absorbs one stream byte whose value is fixed while the circuit is built.
	fn absorb_const_byte(&mut self, byte: u8) {
		let placed = self
			.builder
			.add_constant_64((byte as u64) << self.byte_shift());
		self.absorb_placed(placed, 1);
	}

	/// Bit offset within its schedule word at which the next stream byte lands.
	const fn byte_shift(&self) -> u32 {
		8 * (3 - self.absorbed % 4) as u32
	}

	/// Absorbs `n` bytes whose contribution is already shifted to its place in a schedule word.
	///
	/// Each schedule word is written left to right and only once per byte, so a plain XOR both
	/// starts the word and extends it.
	fn absorb_placed(&mut self, placed: Wire, n: usize) {
		// Which schedule word of the current block the cursor sits in.
		let slot = self.absorbed % BLOCK_BYTES / 4;
		self.block[slot] = self.builder.bxor(self.block[slot], placed);
		self.absorbed += n;

		// Landing on a boundary means the block is full and can be retired.
		if self.absorbed.is_multiple_of(BLOCK_BYTES) {
			self.close_block();
		}
	}

	/// Retires the completed block, pairing it with a held one when there is one.
	///
	/// Consecutive blocks of an epoch chain, so a pair costs one two-lane core rather than two
	/// one-lane cores:
	///
	/// ```text
	///     block 0 arrives   ->  held back, nothing emitted
	///     block 1 arrives   ->  one two-lane core runs both, in sequence
	///     block 2 arrives   ->  held back
	/// ```
	fn close_block(&mut self) {
		// Start a fresh accumulator and take ownership of the finished one.
		let fresh = [self.builder.add_constant(Word::ZERO); BLOCK_WORDS];
		let block = mem::replace(&mut self.block, fresh);

		match self.pending.take() {
			// No partner yet, so hold this block back and emit nothing.
			None => self.pending = Some(block),
			// A partner is waiting, so both chain through a single two-lane core.
			Some(first) => {
				let name = format!("compress[{}]", self.n_cores);
				self.state = sha256_compress_2x_seq(
					&self.builder.subcircuit(name),
					self.state,
					[first, block],
				);
				self.n_cores += 1;
			}
		}
	}

	/// Compresses one block on its own, for the odd block at the end of an epoch.
	///
	/// # Why a dirty incoming state is safe
	///
	/// A two-lane core leaves its partner's state in the high half of each wire.
	/// No compression step lets a carry or a rotate cross bit 32.
	/// So the low half of the result depends on the low halves of the inputs alone.
	fn compress_single(&mut self, block: [Wire; BLOCK_WORDS]) {
		let name = format!("compress[{}]", self.n_cores);
		self.state = sha256_compress(&self.builder.subcircuit(name), self.state, block);
		self.n_cores += 1;
	}
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;
	use binius_frontend::{CircuitBuilder, CircuitStat, Wire};
	use binius_transcript::{
		Buf, BufMut,
		fiat_shamir::{Challenger, HasherChallenger, sample_bits_reader},
	};
	use proptest::prelude::*;
	use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
	use sha2::{Digest, Sha256};

	use super::*;

	/// One step of a recorded challenger script.
	#[derive(Clone, Copy, Debug)]
	enum Op {
		/// Observe that many stream bytes, one wire per byte.
		Bytes(usize),
		/// Observe that many stream words, eight little-endian bytes per wire.
		Words(usize),
		/// Sample one 128-bit challenge.
		B128,
		/// Sample a value of that bit width.
		Bits(usize),
	}

	/// Runs one script through both challengers and checks that every sampled value matches.
	///
	/// ```text
	///     native : absorbs the same bytes, hands out the reference values
	///     circuit: absorbs the same bytes on wires, hands out wires
	/// ```
	///
	/// Each reference value is pinned to a public wire by an equality assertion.
	/// A mismatch therefore fails witness population, not a Rust comparison, which is what makes
	/// this a check on the *constraints* rather than on a side computation.
	fn check_script(ops: &[Op], seed: u64) {
		// Same seed for both halves, so the two runs absorb byte-identical data.
		let mut rng = StdRng::seed_from_u64(seed);
		let mut native = HasherChallenger::<Sha256>::default();

		let builder = CircuitBuilder::new();
		let mut gadget = Sha256Challenger::new(&builder);

		// Observed data, to load into the witness before evaluation.
		let mut inputs: Vec<(Wire, u64)> = Vec::new();
		// Sampled wires paired with the value the native challenger produced.
		let mut samples: Vec<(Wire, u64)> = Vec::new();

		for op in ops {
			match *op {
				Op::Bytes(n) => {
					let mut data = vec![0u8; n];
					rng.fill_bytes(&mut data);
					native.observer().put_slice(&data);

					let wires: Vec<Wire> = (0..n).map(|_| builder.add_witness()).collect();
					gadget.observe_bytes(&wires);
					inputs.extend(wires.iter().zip(&data).map(|(&w, &b)| (w, b as u64)));
				}
				Op::Words(n) => {
					let data: Vec<u64> = (0..n).map(|_| rng.random()).collect();
					// One call, so that a zero-length observe still turns the channel.
					let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
					native.observer().put_slice(&bytes);

					let wires: Vec<Wire> = (0..n).map(|_| builder.add_witness()).collect();
					gadget.observe_words(&wires);
					inputs.extend(wires.iter().zip(&data).map(|(&w, &v)| (w, v)));
				}
				Op::B128 => {
					// `CanSample<B128>` deserializes the field element from 16 little-endian bytes.
					let mut bytes = [0u8; 16];
					native.sampler().copy_to_slice(&mut bytes);

					let (low, high) = gadget.sample_b128();
					samples.push((low, u64::from_le_bytes(bytes[..8].try_into().unwrap())));
					samples.push((high, u64::from_le_bytes(bytes[8..].try_into().unwrap())));
				}
				Op::Bits(n) => {
					let expected = sample_bits_reader(native.sampler(), n);
					samples.push((gadget.sample_bits(n), expected as u64));
				}
			}
		}

		// Bind each sampled wire to a public wire carrying the reference value.
		//
		//     sampled wire  ==  public wire  <-  native value
		//
		// An equality assertion is the only thing tying them, so the check lives in the circuit.
		let pins: Vec<(Wire, u64)> = samples
			.iter()
			.enumerate()
			.map(|(i, &(wire, value))| {
				let pin = builder.add_inout();
				builder.assert_eq(format!("sample[{i}]"), wire, pin);
				(pin, value)
			})
			.collect();

		let circuit = builder.build();
		let cs = circuit.constraint_system();
		let mut w = circuit.new_witness_filler();

		// Only two kinds of wire are written: the observed data, and the reference values.
		// Everything else is derived by evaluating the gates.
		for (wire, value) in inputs.into_iter().chain(pins) {
			w[wire] = Word(value);
		}

		// Population evaluates every gate and every assertion, so a wrong sample fails here.
		circuit.populate_wire_witness(&mut w).unwrap();
		// Verification then re-checks the same witness against the constraint system itself.
		cs.verify(&w.into_value_vec()).unwrap();
	}

	#[test]
	fn seed_matches_sha256_of_empty_string() {
		// Invariant: the hardcoded seed is the digest of the empty string, the value the native
		// challenger opens with.
		//
		// Why it is pinned here: hardcoding the seed is what makes the first 32 sampled bytes free,
		// so it must be checked against the real hash rather than trusted.
		let digest = Sha256::digest([]);

		// The seed is stored as big-endian 32-bit words, matching the message-schedule convention.
		let words: [u32; DIGEST_WORDS] =
			array::from_fn(|i| u32::from_be_bytes(digest[4 * i..4 * i + 4].try_into().unwrap()));
		assert_eq!(SEED, words);
	}

	#[test]
	fn matches_native_on_recorded_script() {
		let script = [
			// Width 0 masks everything away, width 32 needs no mask, and 33 clamps down to 32.
			Op::Bits(0),
			Op::Bits(32),
			Op::Bits(33),
			Op::Bits(8),
			Op::Bits(31),
			// 20 of the seed digest's 32 bytes are gone, so this value spans the first refill.
			Op::B128,
			// A bare transition: the channel turns and only the sampled-byte count is absorbed.
			Op::Bytes(0),
			// Turning back retires the buffer, so this forces another refill.
			Op::B128,
			// A second value leaves the buffer exactly spent, so the count the next transition
			// absorbs is the full 32.
			Op::B128,
			Op::Bytes(3),
			// The cursor is 3 bytes past alignment, so these words take the byte-by-byte path.
			Op::Words(2),
			// Realigns the cursor, 4 bytes short of a block boundary.
			Op::Bytes(1),
			// So this word straddles two blocks, four of its bytes in each.
			Op::Words(1),
			// 72 bytes from an aligned cursor, closing further blocks on the fast path.
			Op::Words(9),
			// A zero-length observe while already observing changes nothing.
			Op::Words(0),
			Op::Bits(7),
			// One block's worth of bytes, then two blocks' worth and one over.
			Op::Bytes(64),
			Op::B128,
			Op::Bytes(129),
			Op::Bits(17),
			Op::Bits(1),
			// Three values in a row consume 48 bytes, so a refill lands mid-value.
			Op::B128,
			Op::B128,
			Op::B128,
		];
		check_script(&script, 0);
	}

	#[test]
	fn matches_native_at_every_sample_bits_width() {
		// Widths 0 to 32 inclusive, covering non-multiples of 8 and the full word.
		let ops: Vec<Op> = (0..=32).map(Op::Bits).collect();
		check_script(&ops, 1);
	}

	proptest! {
		// A dozen cases: each builds and evaluates a whole circuit, so they are not cheap.
		#![proptest_config(ProptestConfig::with_cases(12))]

		#[test]
		fn matches_native_on_random_scripts(
			// Random operation sequences reach interleavings no hand-written script would.
			// The ranges are chosen to keep block boundaries and refills landing in varied places:
			//
			//     Bytes(0..40)   spans a partial block, and 0 exercises a bare transition
			//     Words(0..6)    up to 48 bytes, so a word sometimes straddles a block
			//     B128           16 bytes, so refills land mid-value
			//     Bits(0..=33)   includes both mask edges and the clamped width
			ops in prop::collection::vec(
				prop_oneof![
					(0usize..40).prop_map(Op::Bytes),
					(0usize..6).prop_map(Op::Words),
					Just(Op::B128),
					(0usize..=33).prop_map(Op::Bits),
				],
				1..10,
			),
			// Varying the seed varies the observed bytes, not just the operation shape.
			seed: u64,
		) {
			check_script(&ops, seed);
		}
	}

	#[test]
	fn the_seed_digest_needs_no_compression() {
		let builder = CircuitBuilder::new();
		let mut gadget = Sha256Challenger::new(&builder);

		// Invariant: the first epoch's digest is a build-time constant, so serving its 32 bytes
		// compresses nothing, which is what makes short scripts cheap.
		//
		//     buffer: 32 constant bytes  ->  two 16-byte challenges  ->  spent
		gadget.sample_b128();
		gadget.sample_b128();
		assert_eq!(gadget.n_cores, 0);

		// A third challenge needs bytes the buffer no longer has, opening a second epoch.
		// That epoch is one block, so exactly one compression core appears.
		gadget.sample_b128();
		assert_eq!(gadget.n_cores, 1);
	}

	/// Builds a circuit that observes some words then samples one challenge, and reports its cost.
	///
	/// Varying only the word count isolates the marginal cost of observed data.
	fn cost(n_words: usize) -> CircuitStat {
		let builder = CircuitBuilder::new();
		let mut gadget = Sha256Challenger::new(&builder);

		let words: Vec<Wire> = (0..n_words).map(|_| builder.add_witness()).collect();
		gadget.observe_words(&words);

		// Pin the sample to public wires, otherwise dead-code elimination prunes the whole circuit.
		let (low, high) = gadget.sample_b128();
		builder.assert_eq("low", low, builder.add_inout());
		builder.assert_eq("high", high, builder.add_inout());

		CircuitStat::collect(&builder.build())
	}

	#[test]
	fn consecutive_blocks_share_a_compression_core() {
		// Invariant: two consecutive blocks of one epoch share a single two-lane core, so their
		// combined cost stays well under two separate one-lane compressions.
		//
		//     8 words  =  64 bytes  =  1 block
		//     24 words = 192 bytes  =  3 blocks
		//     difference            =  2 blocks, which must pair
		let marginal = cost(24).n_and_constraints - cost(8).n_and_constraints;

		// A one-block epoch: a single-lane core plus the fixed sampling overhead.
		let single = cost(0).n_and_constraints;

		// Comparing against 1.5 cores distinguishes paired from unpaired without hardcoding a
		// constraint count that the compiler is free to improve.
		assert!(
			marginal < single + single / 2,
			"two observed blocks cost {marginal} AND constraints against {single} for one core"
		);
	}

	#[test]
	fn no_multiplication_constraints_are_emitted() {
		// Invariant: this gadget is pure bit manipulation, so it never touches either
		// multiplication column.
		//
		// Why it matters: the surrounding verifier spends those columns on field arithmetic and
		// select gates, so a regression here would quietly compete with it for scarce space.
		//
		//     0 words   epoch of one block, no observed data
		//     8 words   exactly one block
		//     24 words  three blocks, so both the paired and the odd-block paths run
		for n_words in [0, 8, 24] {
			let stat = cost(n_words);
			assert_eq!(stat.n_bmul_constraints, 0, "n_words = {n_words}");
			assert_eq!(stat.n_imul_constraints, 0, "n_words = {n_words}");
		}
	}
}
