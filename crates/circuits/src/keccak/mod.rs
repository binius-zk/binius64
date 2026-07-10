// Copyright 2025 Irreducible Inc.

pub mod fixed_length;
pub mod permutation;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use permutation::Permutation;

use crate::multiplexer::{multi_wire_multiplex, single_wire_multiplex};

pub const N_WORDS_PER_DIGEST: usize = 4;
pub const N_WORDS_PER_STATE: usize = 25;
pub const RATE_BYTES: usize = 136;
pub const N_WORDS_PER_BLOCK: usize = RATE_BYTES / 8;

/// Keccak-256 circuit for variable-length inputs up to a fixed maximum.
///
/// Satisfiable iff `digest` is the Keccak-256 hash of `message[..len_bytes]`. The prover
/// chooses `len_bytes`; see [`Keccak256::new`] for the caller obligation that implies.
pub struct Keccak256 {
	/// Claimed message length. Only bounded by `len_bytes <= max_len_bytes`; callers must
	/// constrain the value (see [`Keccak256::new`]).
	pub len_bytes: Wire,
	pub digest: [Wire; N_WORDS_PER_DIGEST],
	pub message: Vec<Wire>,
	padded_message: Vec<Wire>,
	n_blocks: usize,
}

impl Keccak256 {
	/// Returns the padded-message witness wires (filled by [`Self::populate_message`]).
	pub fn padded_message(&self) -> &[Wire] {
		&self.padded_message
	}

	/// Create a new keccak circuit using the circuit builder.
	///
	/// The resulting circuit is satisfiable iff `digest` is the Keccak-256 hash of the first
	/// `len_bytes` bytes of `message`. The maximum supported length is fixed at construction
	/// time by `message.len()` (`max_len_bytes = message.len() * 8`).
	///
	/// # Arguments
	///
	/// * `b` - circuit builder object
	/// * `len_bytes` - wire holding the *claimed* input message length in bytes
	/// * `digest` - array of 4 wires holding the claimed 256-bit output digest
	/// * `message` - wires holding the claimed input message, packed 8 bytes per wire,
	///   little-endian; `message.len()` fixes `max_len_bytes` for this instance
	///
	/// # Preconditions
	/// * `message` is non-empty (`max_len_bytes > 0`).
	///
	/// # Soundness — callers must constrain `len_bytes`
	///
	/// This proves `digest == keccak256(message[..len_bytes])` for a *prover-chosen* `len_bytes`
	/// (the only in-circuit check is `len_bytes <= max_len_bytes`), **not**
	/// `digest == keccak256(message)`. So if `len_bytes` is a free witness and `digest` is
	/// prover-influenced, the system is satisfiable for any prefix: claim a shorter length and
	/// supply that prefix's digest with matching padding. A verifier-fixed public `digest`
	/// neutralizes this; a digest consumed internally does not. Callers must pin `len_bytes` — to
	/// a public input, a constant, or another constrained wire; for a compile-time length, prefer
	/// the cheaper [`fixed_length::keccak256`]. The in-tree `ethsign` circuit is the safe pattern.
	/// Message bytes past `len_bytes` are unconstrained.
	pub fn new(
		b: &CircuitBuilder,
		len_bytes: Wire,
		digest: [Wire; N_WORDS_PER_DIGEST],
		message: Vec<Wire>,
	) -> Self {
		let max_len_bytes = message.len() << 3;
		// number of blocks needed for the maximum sized message
		let n_blocks = (max_len_bytes + 1).div_ceil(RATE_BYTES);

		// constrain the message length claim to be explicitly within bounds
		let len_check = b.icmp_ugt(len_bytes, b.add_constant_64(max_len_bytes as u64)); // len_bytes > max_len_bytes
		b.assert_false("len_check", len_check);

		let padded_message: Vec<Wire> = (0..n_blocks * N_WORDS_PER_BLOCK)
			.map(|_| b.add_witness())
			.collect();

		// zero initialized keccak state
		let mut states: Vec<[Wire; N_WORDS_PER_STATE]> = Vec::with_capacity(n_blocks + 1);
		let zero = b.add_constant(Word::ZERO);
		states.push([zero; N_WORDS_PER_STATE]);

		// xor next message block into state and permute
		for block_no in 0..n_blocks {
			let state_in = states[block_no];
			let mut xored_state = state_in;
			for i in 0..N_WORDS_PER_BLOCK {
				xored_state[i] =
					b.bxor(state_in[i], padded_message[block_no * N_WORDS_PER_BLOCK + i]);
			}

			Permutation::keccak_f1600(b, &mut xored_state);

			states.push(xored_state);
		}

		// begin "constrain claimed digest".
		// want to do: `let block_index = (len_bytes + 1).divceil(136)`.
		// royal pain in the ass that 136 is not a power of 2, so we can't compute this in circuit
		// still though, i believe that there might be tricks better than what we're doing below.
		let mut end_block_index = b.add_constant(Word::ZERO);
		let mut is_not_last_column = b.add_constant(Word::ZERO);
		// `is_not_last_column` will be true if and only if `len_bytes >> 3` != 16 (mod 17).
		// true iff the WORD w/ the very first post-message byte is NOT the last word in its block.
		for block_no in 0..n_blocks {
			// start of this block
			let block_start = b.add_constant_64((block_no * RATE_BYTES) as u64);
			let block_end = b.add_constant_64(((block_no + 1) * RATE_BYTES) as u64);
			let last_word_start = b.add_constant_64(((block_no + 1) * RATE_BYTES - 8) as u64);

			let gte_start = b.icmp_ule(block_start, len_bytes);
			let lt_end = b.icmp_ult(len_bytes, block_end);
			let lt_last_word = b.icmp_ult(len_bytes, last_word_start);
			let is_final_block = b.band(gte_start, lt_end);

			// flag that this block is the final block per the claimed length
			end_block_index =
				b.select(is_final_block, b.add_constant_64(block_no as u64), end_block_index);
			is_not_last_column = b.select(is_final_block, lt_last_word, is_not_last_column);
		}

		let inputs: Vec<&[Wire]> = states[1..].iter().map(|arr| &arr[..]).collect();
		let computed_digest_vec = multi_wire_multiplex(b, &inputs, end_block_index);
		let computed_digest = computed_digest_vec[..N_WORDS_PER_DIGEST]
			.try_into()
			.unwrap();
		b.assert_eq_v("claimed digest is correct", digest, computed_digest);

		// begin treatment of boundary word.
		let word_boundary = b.shr(len_bytes, 3);
		let boundary_word = single_wire_multiplex(b, &message, word_boundary);
		let boundary_padded_word = single_wire_multiplex(b, &padded_message, word_boundary);
		// When the last word of the message is not full, we expect a padding byte to be
		// somewhere within the word. Since the top bit will also be in this word.
		let candidates: Vec<Wire> = (0..8)
			.map(|i| {
				let mask = b.add_constant_64(0x00FFFFFFFFFFFFFF >> ((7 - i) << 3));
				let padding_byte = b.add_constant_64(1 << (i << 3));
				let message_low = b.band(boundary_word, mask);
				b.bxor(message_low, padding_byte)
			})
			.collect();

		let zero = b.add_constant(Word::ZERO);
		let msb_one = b.add_constant(Word::MSB_ONE);
		let len_bytes_mod_8 = b.band(len_bytes, b.add_constant_64(7));
		let expected_partial = single_wire_multiplex(b, &candidates, len_bytes_mod_8);
		let with_possible_end =
			b.bxor(expected_partial, b.select(is_not_last_column, zero, msb_one));

		b.assert_eq("expected partial", with_possible_end, boundary_padded_word);

		// Within the final rate block, ensure that the pad byte and top bit are where they are
		// supposed to be
		for block_index in 0..n_blocks {
			let is_end_block = b.icmp_eq(end_block_index, b.add_constant_64(block_index as u64));
			for column_index in 0..N_WORDS_PER_BLOCK {
				let word_index = block_index * N_WORDS_PER_BLOCK + column_index;

				let padded_word = padded_message[word_index];

				// a potentially padded word is at this index
				let word_idx_wire = b.add_constant_64(word_index as u64);
				if word_index < message.len() {
					let message_word = message[word_index];
					let is_before_end = b.icmp_ult(word_idx_wire, word_boundary);
					b.assert_eq_cond("full", padded_word, message_word, is_before_end);
				}

				let is_past_message = b.icmp_ugt(word_idx_wire, word_boundary);

				if column_index == 16 {
					// last word in the block
					let must_check_delimiter = b.band(is_end_block, is_not_last_column);
					b.assert_eq_cond("delim", padded_word, msb_one, must_check_delimiter);
					// the case we need to deal with: we're in end block but `is_not_last_column`.
					// this means that the `boundary_message_word` is not the last word in its block
					// then the presence of the 0x80 delimiter is NOT treated with the boundary word
					// thus we must separately check that the ACTUAL last word in the block has it

					// if `is_end_block` is true but NOT `is_not_last_column`, then we're fine.
					// indeed: if `!is_not_last_column`, boundary message word IS in last column,
					// so we already handled the validity of that word, and there is nothing to do.

					// if NOT in end block, then again i claim there is nothing we need to check.
					// if we're in the last column but strictly before the end block, then we're
					// still in the message, by definition of `end_block`. indeed, the `0x80` byte
					// happens in the soonest possible block after the message ends, and no later.
					// thus we already checked the validity of this word above (a `message_word`).
					// the other case is that we're strictly after the end block. in this case,
					// we can just leave the `padded_word` completely unconstrained. after all,
					// said word will have no effect on `digest` whatsoever, so we just leave it.
				} else {
					b.assert_eq_cond("after-message padding", padded_word, zero, is_past_message);
					// we're strictly after the word w/ the 0x01 byte and not in the last column.
					// there are two cases: either we're within the end block or strictly after it.
					// if the former, we're after the boundary word but before the word w/ 0x80.
					// in that case, we must for the sake of correctness assert that this word is 0.
					// if strictly after the end block, this word will have no effect on `digest`;
					// thus we're free to assert that it's 0, but it's not necessary for soundness.
				}
			}
		}

		Self {
			len_bytes,
			digest,
			message,
			padded_message,
			n_blocks,
		}
	}

	pub const fn max_len_bytes(&self) -> usize {
		self.message.len() << 3
	}

	/// Populates the witness with the actual message length
	///
	/// ## Arguments
	///
	/// * w - The witness filler to populate
	/// * len_bytes - The actual byte length of the message
	pub fn populate_len_bytes(&self, w: &mut WitnessFiller<'_>, len_bytes: usize) {
		assert!(
			len_bytes <= self.max_len_bytes(),
			"Message length {} exceeds maximum {}",
			len_bytes,
			self.max_len_bytes()
		);
		w[self.len_bytes] = Word(len_bytes as u64);
	}

	/// Populates the witness with the expected digest value packed into 4 64-bit words
	///
	/// ## Arguments
	///
	/// * w - The witness filler to populate
	/// * digest - The expected 32-byte Keccak-256 digest
	pub fn populate_digest(&self, w: &mut WitnessFiller<'_>, digest: [u8; 32]) {
		for (i, bytes) in digest.chunks(8).enumerate() {
			let word = u64::from_le_bytes(bytes.try_into().unwrap());
			w[self.digest[i]] = Word(word);
		}
	}

	/// Populates the witness with padded byte message packed into 64-bit words
	///
	/// ## Arguments
	///
	/// * w - The witness filler to populate
	/// * message_bytes - The input message as a byte slice
	pub fn populate_message(&self, w: &mut WitnessFiller<'_>, message_bytes: &[u8]) {
		assert!(
			message_bytes.len() <= self.max_len_bytes(),
			"Message length {} exceeds maximum {}",
			message_bytes.len(),
			self.max_len_bytes()
		);

		// populate message words from input bytes
		let words = self.pack_bytes_into_words(message_bytes, self.max_len_bytes().div_ceil(8));
		for (i, word) in words.iter().enumerate() {
			if i < self.message.len() {
				w[self.message[i]] = Word(*word);
			}
		}

		let mut padded_bytes = vec![0u8; self.n_blocks * RATE_BYTES];

		padded_bytes[..message_bytes.len()].copy_from_slice(message_bytes);

		let msg_len = message_bytes.len();
		let num_full_blocks = msg_len / RATE_BYTES;
		let padding_block_start = num_full_blocks * RATE_BYTES;

		padded_bytes[msg_len] = 0x01;

		let padding_block_end = padding_block_start + RATE_BYTES - 1;
		padded_bytes[padding_block_end] |= 0x80;

		for block_idx in 0..self.n_blocks {
			for (i, chunk) in padded_bytes[block_idx * RATE_BYTES..(block_idx + 1) * RATE_BYTES]
				.chunks(8)
				.enumerate()
			{
				let word = u64::from_le_bytes(chunk.try_into().unwrap());
				w[self.padded_message[block_idx * N_WORDS_PER_BLOCK + i]] = Word(word);
			}
		}
	}

	fn pack_bytes_into_words(&self, bytes: &[u8], n_words: usize) -> Vec<u64> {
		let mut words = Vec::with_capacity(n_words);
		for i in 0..n_words {
			if i * 8 < bytes.len() {
				// to handle messages that are not multiples of 64, bytes are copied into
				// a little endian byte array and then converted to a u64
				let start = i * 8;
				let end = ((i + 1) * 8).min(bytes.len());
				let mut word_bytes = [0u8; 8];
				word_bytes[..end - start].copy_from_slice(&bytes[start..end]);
				let word = u64::from_le_bytes(word_bytes);
				words.push(word);
			}
		}

		words
	}
}

#[cfg(test)]
mod tests {
	use binius_core::{verify::verify_constraints, word::Word};
	use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
	use rand::prelude::*;
	use rstest::rstest;
	use sha3::Digest;

	use super::*;

	#[rstest]
	#[case(0, 100)] // Empty message
	#[case(1, 100)] // Single byte - minimal non-empty
	#[case(1, 144)] // Single byte - minimal non-empty
	#[case(135, 136)] // 135 bytes - one byte before block boundary
	#[case(136, 136)] // 136 bytes - exactly one block
	#[case(137, 272)] // 137 bytes - crosses block boundary
	#[case(271, 272)] // 271 bytes - one byte before two blocks
	#[case(272, 272)] // 272 bytes - exactly two blocks
	fn test_keccak_circuit(#[case] message_len_bytes: usize, #[case] max_message_len_bytes: usize) {
		// Create test message with deterministic random bytes seeded by the length inputs
		let seed = ((message_len_bytes as u64) << 32) | (max_message_len_bytes as u64);
		let mut rng = StdRng::seed_from_u64(seed);
		let mut message = vec![0u8; message_len_bytes];
		rng.fill_bytes(&mut message);

		// Compute expected digest using sha3 crate
		let mut hasher = sha3::Keccak256::new();
		hasher.update(&message);
		let expected_digest: [u8; 32] = hasher.finalize().into();

		// Build circuit
		assert!(
			message_len_bytes <= max_message_len_bytes,
			"Message length {} exceeds max capacity {} bytes",
			message_len_bytes,
			max_message_len_bytes
		);

		let b = CircuitBuilder::new();
		let len = b.add_witness();
		let digest: [Wire; N_WORDS_PER_DIGEST] = std::array::from_fn(|_| b.add_inout());
		let n_words = max_message_len_bytes.div_ceil(8);
		let message_wires = (0..n_words).map(|_| b.add_inout()).collect();

		let keccak = Keccak256::new(&b, len, digest, message_wires);
		let circuit = b.build();

		// Create and populate witness
		let mut witness = circuit.new_witness_filler();
		keccak.populate_len_bytes(&mut witness, message.len());
		keccak.populate_message(&mut witness, &message);
		keccak.populate_digest(&mut witness, expected_digest);

		// Verify circuit accepts the witness
		circuit
			.populate_wire_witness(&mut witness)
			.expect("Circuit should accept valid witness");

		// Verify all constraints are satisfied
		let cs = circuit.constraint_system();
		verify_constraints(cs, &witness.into_value_vec())
			.expect("All constraints should be satisfied");
	}

	// Characterization tests for the `len_bytes` caller obligation (see `Keccak256::new`).
	// They drive the constraint system directly with all wires prover-populated, modelling
	// the composed-circuit case where `digest` is prover-influenced. The gadget previously
	// had no negative tests.

	fn keccak256_native(msg: &[u8]) -> [u8; 32] {
		let mut h = sha3::Keccak256::new();
		h.update(msg);
		h.finalize().into()
	}

	/// Build a variable-length `Keccak256` with `len_bytes` a free witness, populate `msg`
	/// claiming `claimed_len`/`claimed_digest`, apply `tamper`, and return whether the
	/// system is satisfiable.
	fn run_free_len(
		cap_words: usize,
		msg: &[u8],
		claimed_len: usize,
		claimed_digest: [u8; 32],
		tamper: impl Fn(&mut WitnessFiller<'_>, &Keccak256),
	) -> Result<(), String> {
		let b = CircuitBuilder::new();
		let len = b.add_witness();
		let digest: [Wire; N_WORDS_PER_DIGEST] = std::array::from_fn(|_| b.add_inout());
		let message: Vec<Wire> = (0..cap_words).map(|_| b.add_inout()).collect();
		let keccak = Keccak256::new(&b, len, digest, message);
		let circuit = b.build();

		let mut w = circuit.new_witness_filler();
		keccak.populate_len_bytes(&mut w, claimed_len);
		keccak.populate_message(&mut w, msg);
		keccak.populate_digest(&mut w, claimed_digest);
		tamper(&mut w, &keccak);

		circuit
			.populate_wire_witness(&mut w)
			.map_err(|e| format!("populate: {e:?}"))?;
		verify_constraints(circuit.constraint_system(), &w.into_value_vec())
			.map_err(|e| format!("verify: {e}"))
	}

	/// Overwrite every `padded_message` wire with valid Keccak padding for `prefix`
	/// (assumed to fit in a single rate block), mounting the prefix-hash attack.
	fn install_prefix_padding(w: &mut WitnessFiller<'_>, k: &Keccak256, prefix: &[u8]) {
		assert!(prefix.len() < RATE_BYTES, "helper only builds single-block padding");
		let mut padded = vec![0u8; k.padded_message().len() * 8];
		padded[..prefix.len()].copy_from_slice(prefix);
		padded[prefix.len()] = 0x01; // start of keccak `pad10*1`
		padded[RATE_BYTES - 1] |= 0x80; // ... and its end, within the first rate block
		for (i, chunk) in padded.chunks_exact(8).enumerate() {
			w[k.padded_message()[i]] = Word(u64::from_le_bytes(chunk.try_into().unwrap()));
		}
	}

	/// With `len_bytes` a free witness, a prover can prove the hash of an arbitrary *prefix*:
	/// commit a 100-byte message, claim `len = 50`, supply `keccak256(message[..50])`, and the
	/// circuit is satisfiable. (Caller obligation, not a gadget bug — see `Keccak256::new`.)
	#[test]
	fn test_unconstrained_len_admits_prefix_hash() {
		const CAP_WORDS: usize = 51; // max_len_bytes = 408 => 4 rate blocks
		let msg: Vec<u8> = (0..100u32).map(|i| (i * 7 + 3) as u8).collect();

		// Honest control: claiming the true length verifies with the true digest.
		let full = keccak256_native(&msg);
		run_free_len(CAP_WORDS, &msg, msg.len(), full, |_, _| {})
			.expect("honest full-length witness must verify");

		// Attack: claim a shorter length and prove the prefix digest instead.
		let prefix = &msg[..50];
		let prefix_digest = keccak256_native(prefix);
		let res = run_free_len(CAP_WORDS, &msg, prefix.len(), prefix_digest, |w, k| {
			install_prefix_padding(w, k, prefix);
		});
		assert!(res.is_ok(), "an unconstrained len_bytes admits the prefix hash; got {res:?}");
	}

	/// The one length check the gadget *does* enforce is `len_bytes <= max_len_bytes`.
	/// Claiming a length past the message capacity is rejected in-circuit.
	#[test]
	fn test_len_greater_than_capacity_is_rejected() {
		const CAP_WORDS: usize = 51; // max_len_bytes = 408
		let msg: Vec<u8> = (0..100u32).map(|i| (i * 7 + 3) as u8).collect();
		let digest = keccak256_native(&msg);
		let res = run_free_len(CAP_WORDS, &msg, msg.len(), digest, |w, k| {
			w[k.len_bytes] = Word(500); // > max_len_bytes (408)
		});
		assert!(res.is_err(), "len_bytes > max_len_bytes must be rejected, got {res:?}");
	}

	/// Companion / mitigation for `test_unconstrained_len_admits_prefix_hash`: once the
	/// caller pins `len_bytes` to a constant at construction time, the prefix attack no
	/// longer verifies. (For a compile-time-constant length, `fixed_length::keccak256` is
	/// the cheaper choice; this test shows the variable gadget is also sound once its
	/// length wire is constrained.)
	#[test]
	fn test_constant_len_rejects_prefix_hash() {
		const CAP_WORDS: usize = 51;
		let msg: Vec<u8> = (0..100u32).map(|i| (i * 7 + 3) as u8).collect();
		let prefix = &msg[..50];
		let prefix_digest = keccak256_native(prefix);

		let b = CircuitBuilder::new();
		// Length pinned to the true message length at construction time.
		let len = b.add_constant_64(msg.len() as u64);
		let digest: [Wire; N_WORDS_PER_DIGEST] = std::array::from_fn(|_| b.add_inout());
		let message: Vec<Wire> = (0..CAP_WORDS).map(|_| b.add_inout()).collect();
		let keccak = Keccak256::new(&b, len, digest, message);
		let circuit = b.build();

		let mut w = circuit.new_witness_filler();
		// `len_bytes` is a constant wire, so it is not populated here.
		keccak.populate_message(&mut w, &msg);
		keccak.populate_digest(&mut w, prefix_digest);
		install_prefix_padding(&mut w, &keccak, prefix);

		let res = match circuit.populate_wire_witness(&mut w) {
			Err(e) => Err(format!("populate: {e:?}")),
			Ok(()) => verify_constraints(circuit.constraint_system(), &w.into_value_vec())
				.map_err(|e| format!("verify: {e}")),
		};
		assert!(res.is_err(), "a pinned len_bytes must reject the prefix hash, got {res:?}");
	}
}
