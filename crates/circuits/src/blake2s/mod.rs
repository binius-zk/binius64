// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! BLAKE2s hash function circuit.
//!
//! BLAKE2s is a cryptographic hash function optimized for 32-bit platforms.
//!
//! It produces digests from 1 to 32 bytes.
//!
//! This implementation follows RFC 7693.
//!
//! It supports fixed-length messages with unkeyed hashing.
//!
//! ## RFC 7693 compliance
//!
//! This implementation is fully compliant with RFC 7693 for the core BLAKE2s-256 hash function.
//!
//! The compression function, the G mixing function, and the message scheduling all match the
//! specification.
//!
//! ## Excluded features
//!
//! This circuit intentionally excludes the following optional features from RFC 7693:
//!
//! - Keyed hashing, also called MAC mode: only unkeyed hash verification is supported.
//! - The 8-byte salt field.
//! - The 8-byte personalization field.
//! - Tree hashing mode: only sequential mode is supported.
//! - Runtime-variable message length: the length is fixed at circuit construction time instead.
//! - Variable output length: the digest is fixed at 256 bits.
//! - Messages of 4 GiB or more.
//!
//! The high half of the byte counter is always the zero constant, which caps the supported
//! message length at just under 4 GiB.
//!
//! These exclusions suit a circuit whose job is hash verification, rather than
//! general-purpose hashing.
//!
//! # Algorithm overview
//!
//! BLAKE2s processes a message in 64-byte blocks.
//!
//! Each block goes through a compression function built from a modified ChaCha cipher core.
//!
//! A compression runs ten mixing rounds, and each round mixes the internal state with the
//! message block through eight calls to the G mixing function.
//!
//! # Circuit design
//!
//! This circuit verifies that a message of a fixed, compile-time-known length produces a
//! specific BLAKE2s digest.
//!
//! Blocks chain sequentially: each one's input state is the previous one's output.
//!
//! So consecutive blocks are compressed two at a time, packing both compressions into the two
//! 32-bit lanes of one parallel core.
//!
//! A trailing block with no partner runs through the same paired core with its second lane
//! left dead, except when the whole message is a single block, which stays fully single-lane.

mod compress;
mod constants;
#[cfg(test)]
mod tests;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
pub use compress::{
	Blake2sCompress2x, blake2s_compress, blake2s_compress_2x, blake2s_compress_2x_seq, ref_compress,
};
use constants::IV;

use crate::util::clear_high_bits;

/// One message block's padded words, plus the counter and flag values its compression needs.
struct BlockInput {
	/// The 16-word padded message block, one 32-bit value per wire.
	m: [Wire; 16],
	/// The low 32 bits of the byte counter after absorbing this block.
	t_lo: Wire,
	/// The finalization flag: all-ones if this is the final block, zero otherwise.
	last: Wire,
}

/// BLAKE2s hash function circuit for a fixed-length message.
///
/// This struct is a complete circuit that verifies a message of a fixed, compile-time-known
/// length produces a specific 256-bit digest.
///
/// The message bytes are packed little-endian into 64-bit words.
pub struct Blake2s {
	/// Message size in bytes this circuit supports.
	pub length: usize,
	/// Witness wires for the input message, packed little-endian into 64-bit words.
	pub message: Vec<Wire>,
	/// Witness wires for the expected 256-bit digest, as 8 32-bit words.
	pub digest: [Wire; 8],
}

impl Blake2s {
	/// Creates a new BLAKE2s circuit with witness variables.
	///
	/// The message length is fixed at circuit construction time, so it shapes the circuit
	/// rather than being a witness value itself.
	///
	/// # Arguments
	/// * `builder` - Circuit builder to add constraints to.
	/// * `length` - The exact message size, in bytes, this circuit will verify.
	///
	/// # Returns
	/// A struct holding the witness wires for the message and the expected digest.
	pub fn new_witness(builder: &mut CircuitBuilder, length: usize) -> Self {
		// One witness wire per 8 bytes of the message.
		let message: Vec<Wire> = (0..length.div_ceil(8))
			.map(|_| builder.add_witness())
			.collect();
		let digest = std::array::from_fn(|_| builder.add_witness());

		Self::build_circuit(builder, length, &message, digest);

		Self {
			length,
			message,
			digest,
		}
	}

	/// Builds the constraints that verify the message hashes to the expected digest.
	fn build_circuit(
		builder: &mut CircuitBuilder,
		length: usize,
		message: &[Wire],
		expected_digest: [Wire; 8],
	) {
		// A message that exactly fills whole blocks still needs at least one block.
		let num_blocks = length.div_ceil(64).max(1);
		let zero = builder.add_constant(Word(0));

		// The BLAKE2s-256 parameter block folds into the first IV word: unkeyed (key length
		// zero), 32-byte digest, sequential mode (fanout 1, depth 1).
		let init_state = [
			builder.add_constant_64((IV[0] ^ 0x01010020) as u64),
			builder.add_constant_64(IV[1] as u64),
			builder.add_constant_64(IV[2] as u64),
			builder.add_constant_64(IV[3] as u64),
			builder.add_constant_64(IV[4] as u64),
			builder.add_constant_64(IV[5] as u64),
			builder.add_constant_64(IV[6] as u64),
			builder.add_constant_64(IV[7] as u64),
		];

		// Every block's padded message words and counter/flag values, computed up front.
		//
		// That lets the compression chain below pair consecutive blocks without interleaving
		// padding logic.
		let blocks: Vec<BlockInput> = (0..num_blocks)
			.map(|block_idx| Self::block_input(builder, message, length, block_idx, zero))
			.collect();

		// Consecutive blocks chain: each one's input state is the previous one's output.
		//
		// So a pair of blocks is compressed through one parallel core, at roughly half the AND
		// cost of two single-lane compressions.
		//
		// The threaded state carries the pair's first compression in its high half.
		//
		// That half is left as it is, rather than masked off, since nothing downstream reads it
		// before the final digest:
		//
		// - A compression never lets a carry or a rotate cross bit 32, so the halves stay apart.
		// - The paired core takes an input state's low half only, through a left shift.
		//
		// So the low half of a result depends on the low halves of its inputs alone.
		let mut h = init_state;
		let mut block_idx = 0;
		while block_idx + 1 < num_blocks {
			let sub =
				builder.subcircuit(format!("blake2s_compress[{block_idx}..{}]", block_idx + 2));
			let a = &blocks[block_idx];
			let b = &blocks[block_idx + 1];
			h = blake2s_compress_2x_seq(
				&sub,
				h,
				[a.m, b.m],
				[a.t_lo, b.t_lo],
				[zero, zero],
				[a.last, b.last],
			);
			block_idx += 2;
		}
		if block_idx < num_blocks {
			let sub = builder.subcircuit(format!("blake2s_compress[{block_idx}]"));
			let b = &blocks[block_idx];
			h = if block_idx > 0 {
				// The trailing odd block has no partner.
				//
				// It still runs through the paired core with its second lane dead, so a
				// registered chip serves every compression uniformly.
				//
				// In gates the two cores emit the same circuit, so nothing changes without a
				// chip.
				blake2s_compress_2x(&sub, h, b.m, b.t_lo, zero, b.last)
			} else {
				// A single-block message keeps the single-lane core.
				//
				// Its empty high halves are what lets the digest skip the clearing below.
				blake2s_compress(&sub, h, b.m, b.t_lo, zero, b.last)
			};
		}

		// The escaping digest is the one place a clean high half is required.
		//
		// So clear it once here, rather than after every pair.
		//
		// A single-block message never enters the paired core, so its one-lane result already
		// has an empty high half and needs no clearing.
		let final_digest: [Wire; 8] = if num_blocks < 2 {
			h
		} else {
			std::array::from_fn(|i| clear_high_bits(builder, h[i], 32))
		};

		for i in 0..8 {
			builder.assert_eq("digest_match", final_digest[i], expected_digest[i]);
		}
	}

	/// Builds one block's padded message words, and its counter and finalization-flag values.
	///
	/// The high half of the byte counter is not part of the returned value.
	///
	/// It is always the zero constant, per this module's 4 GiB message-size limit, so every
	/// caller shares one wire for it.
	fn block_input(
		builder: &CircuitBuilder,
		message: &[Wire],
		length: usize,
		block_idx: usize,
		zero: Wire,
	) -> BlockInput {
		let mut m = [zero; 16];

		for word_idx in 0..16 {
			// The message is packed 8 bytes per wire, so two consecutive 32-bit words share
			// one 64-bit wire.
			let message_qword = *message.get(block_idx << 3 | word_idx >> 1).unwrap_or(&zero);

			// Take the low or the high 32 bits of that wire, depending on which of the pair
			// this word is.
			let message_dword = if word_idx % 2 == 0 {
				clear_high_bits(builder, message_qword, 32)
			} else {
				builder.shr(message_qword, 32)
			};

			// Mask off any bytes past the message's true length.
			//
			// A word entirely past the end becomes the zero constant.
			//
			// A word straddling the end keeps only its valid leading bytes.
			let first_byte_offset = block_idx * 64 + word_idx * 4;
			let padded_message_dword = if first_byte_offset + 4 > length {
				if first_byte_offset < length {
					let nonzero_bytes = (length - first_byte_offset) as u32;
					builder.band(
						message_dword,
						builder.add_constant(Word::ALL_ONE >> (64 - nonzero_bytes * 8)),
					)
				} else {
					zero
				}
			} else {
				message_dword
			};

			m[word_idx] = padded_message_dword;
		}

		// The final block is the one whose byte range contains the message's true length.
		//
		// Block 0 always counts as in range, so a zero-length message still has one block.
		let block_start = (block_idx * 64) as u64;
		let block_end = block_start + 64;
		let is_final_block =
			(block_idx == 0 || block_start < length as u64) && length as u64 <= block_end;

		// The byte counter after this block is the block's end offset, except on the final
		// block, where it is the message's true length instead.
		let t_lo = builder.add_constant_64(if is_final_block {
			length as u64
		} else {
			block_end
		});

		// The finalization flag is all-ones on the final block, and zero otherwise.
		let flag_value = builder.add_constant(Word(0xFFFFFFFF));
		let last = if is_final_block { flag_value } else { zero };

		BlockInput { m, t_lo, last }
	}

	/// Populates the message witness wires.
	///
	/// # Arguments
	/// * `witness` - Witness filler to populate.
	/// * `message` - The message bytes to hash.
	///
	/// # Panics
	/// * If `message.len()` does not equal the circuit's fixed message length.
	pub fn populate_message(&self, witness: &mut WitnessFiller, message: &[u8]) {
		assert!(
			message.len() == self.length,
			"Only messages of length {} supported while given {} bytes",
			self.length,
			message.len(),
		);

		for (i, bytes) in message.chunks(8).enumerate() {
			let mut le_bytes = [0; 8];
			le_bytes[..bytes.len()].copy_from_slice(bytes);
			witness[self.message[i]] = Word(u64::from_le_bytes(le_bytes))
		}
	}

	/// Populates the expected-digest witness wires.
	///
	/// # Arguments
	/// * `witness` - Witness filler to populate.
	/// * `digest` - The expected 32-byte BLAKE2s digest.
	pub fn populate_digest(&self, witness: &mut WitnessFiller, digest: &[u8; 32]) {
		for i in 0..8 {
			let word_bytes = &digest[i * 4..(i + 1) * 4];
			let word = u32::from_le_bytes(word_bytes.try_into().unwrap());
			witness[self.digest[i]] = Word(word as u64);
		}
	}
}
