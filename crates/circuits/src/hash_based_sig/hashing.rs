// Copyright 2026 The Binius Developers
// Copyright (c) 2026 leanEthereum
//! The XMSS hash layer: [`tweak_hash`] is the BLAKE3 keyed hash of the payload under `pp | tweak`,
//! truncated to 16 bytes, for chain steps, Merkle nodes, WOTS public keys and message encodings
//! alike.
//!
//! The 16-byte tweak makes every call site a distinct hash function, which is what separates the
//! many targets an attacker may aim at, and the public parameter separates users. Together they
//! are 32 bytes, exactly a BLAKE3 key, so the whole of the domain fits the key with nothing left
//! to pad — two distinct `(pp, tweak)` pairs are two distinct keys.
//!
//! Keying is BLAKE3's own domain separation: the key replaces the initial chaining value and every
//! compression carries `KEYED_HASH`. The payload is then the entire message, and the full
//! construction binds its length, so no digest extends into the digest of a longer payload.

use binius_frontend::{CircuitBuilder, Wire};

use super::{DIGEST_LEN, DIGEST_WIRES, Digest, PUBLIC_PARAM_LEN, PUBLIC_PARAM_WIRES, PublicParam};
use crate::{
	blake3::{KEY_BYTES, blake3_keyed_fixed, blake3_keyed_fixed_2x},
	fixed_byte_vec::ByteVec,
	util::{clear_high_bits, split_u32_words},
};

/// Tweak type for a chain step.
pub const TWEAK_TYPE_CHAIN: u8 = 0;
/// Tweak type for a WOTS public-key (Merkle leaf) hash.
pub const TWEAK_TYPE_WOTS_PK: u8 = 1;
/// Tweak type for an internal Merkle node.
pub const TWEAK_TYPE_MERKLE: u8 = 2;
/// Tweak type for the message encoding.
pub const TWEAK_TYPE_ENCODING: u8 = 3;

/// Tweak length in bytes.
pub const TWEAK_LEN: usize = 16;

/// Wires holding a tweak.
const TWEAK_WIRES: usize = TWEAK_LEN / 8;

/// A tweak: `[tweak_type (1) | sub_position (4) | index (4) | zeros (7)]`, little-endian.
pub type Tweak = [u8; TWEAK_LEN];

// The public parameter and the tweak fill a BLAKE3 key exactly, so the key needs no padding and
// distinct domains cannot collide in it.
const _: () = assert!(PUBLIC_PARAM_LEN + TWEAK_LEN == KEY_BYTES);

/// Builds a tweak.
///
/// `index` is the epoch (chain, WOTS public key, encoding) or the Merkle node index;
/// `sub_position` is the chain position or the Merkle level.
pub fn make_tweak(tweak_type: u8, sub_position: u32, index: u32) -> Tweak {
	let mut tweak = [0u8; TWEAK_LEN];
	tweak[0] = tweak_type;
	tweak[1..5].copy_from_slice(&sub_position.to_le_bytes());
	tweak[5..9].copy_from_slice(&index.to_le_bytes());
	tweak
}

/// The key a call site hashes under: `pp | tweak`, the full 32 bytes of a BLAKE3 key.
pub fn make_key(
	public_param: &PublicParam,
	tweak_type: u8,
	sub_position: u32,
	index: u32,
) -> [u8; KEY_BYTES] {
	let mut key = [0u8; KEY_BYTES];
	key[..PUBLIC_PARAM_LEN].copy_from_slice(public_param);
	key[PUBLIC_PARAM_LEN..].copy_from_slice(&make_tweak(tweak_type, sub_position, index));
	key
}

/// The BLAKE3 keyed hash of `payload` under `pp | tweak`, truncated to [`DIGEST_LEN`].
///
/// One compression for chain steps (16 bytes of payload), Merkle nodes (32) and the message
/// encoding (56); eleven for the WOTS public key's 672.
pub fn tweak_hash(
	public_param: &PublicParam,
	tweak_type: u8,
	sub_position: u32,
	index: u32,
	payload: &[u8],
) -> Digest {
	let key = make_key(public_param, tweak_type, sub_position, index);
	blake3::keyed_hash(&key, payload).as_bytes()[..DIGEST_LEN]
		.try_into()
		.expect("the slice is DIGEST_LEN bytes")
}

/// In-circuit form of [`tweak_hash`], returning the truncated digest as 64-bit little-endian
/// wires.
///
/// The tweak type is a circuit constant at every call site. The sub-position is a wire: a Merkle
/// node's level is still a constant fed in as one, but a chain step's position is computed, so a
/// chain can spend hashes only where its digit says it must. Nothing about it is free for the
/// prover to choose — the caller is what constrains it.
///
/// # Arguments
///
/// - `builder`: circuit builder.
/// - `public_param`: the per-signer parameter, eight bytes per wire.
/// - `tweak_type`: one of the `TWEAK_TYPE_*` constants.
/// - `sub_position`: chain position or Merkle level.
/// - `index`: epoch or Merkle node index. Only its low four bytes reach the tweak, matching the
///   `u32` the reference takes.
/// - `payload`: the hashed payload, eight bytes per wire.
///
/// # Returns
///
/// The 16-byte digest as [`DIGEST_WIRES`] 64-bit little-endian wires.
pub fn circuit_tweak_hash(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	tweak_type: u8,
	sub_position: Wire,
	index: Wire,
	payload: &[Wire],
) -> [Wire; DIGEST_WIRES] {
	let key = circuit_key(builder, public_param, tweak_type, sub_position, index);
	let len_bytes = payload.len() * 8;
	let message = split_u32_words(builder, payload, len_bytes / 4);
	truncate(builder, &blake3_keyed_fixed(builder, &message, len_bytes, &key))
}

/// The key as a byte vector: the public parameter, then the tweak.
fn circuit_key(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	tweak_type: u8,
	sub_position: Wire,
	index: Wire,
) -> ByteVec {
	let mut wires = Vec::with_capacity(PUBLIC_PARAM_WIRES + TWEAK_WIRES);
	wires.extend_from_slice(public_param);
	wires.extend_from_slice(&tweak_wires(builder, tweak_type, sub_position, index));
	ByteVec::new_const_len(builder, wires, KEY_BYTES)
}

/// Two independent tweak hashes evaluated as the two lanes of one core.
///
/// Two hashes of equal length run in lockstep — same block count, same block lengths, same flags,
/// same tree shape — so every compression of one has a partner in the other and the two share a
/// paired core. Here the lanes differ only in their key's sub-position and in their payload, which
/// is exactly how two chains at one step differ.
///
/// This is cheaper than two lone hashes, though not for the reason the lane count suggests. A lone
/// compression already uses both lanes, splitting its own seven rounds across them, so on rounds
/// alone the pair only trades eight for seven. The rest of the saving is the split itself: a lone
/// compression hints its mid-round state and then constrains that hint word for word, and a pair
/// has no split to pin.
///
/// # Panics
///
/// - If the two payloads differ in length.
pub fn circuit_tweak_hash_2x(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	tweak_type: u8,
	sub_positions: [Wire; 2],
	index: Wire,
	payloads: [&[Wire]; 2],
) -> [[Wire; DIGEST_WIRES]; 2] {
	assert_eq!(
		payloads[0].len(),
		payloads[1].len(),
		"both lanes must hash the same number of bytes"
	);

	let keys = sub_positions
		.map(|sub_position| circuit_key(builder, public_param, tweak_type, sub_position, index));
	let len_bytes = payloads[0].len() * 8;
	let messages = payloads.map(|payload| split_u32_words(builder, payload, len_bytes / 4));

	let digests = blake3_keyed_fixed_2x(
		builder,
		[&messages[0], &messages[1]],
		len_bytes,
		[&keys[0], &keys[1]],
	);
	digests.map(|digest| truncate(builder, &digest))
}

/// The tweak as 64-bit little-endian wires.
fn tweak_wires(
	builder: &CircuitBuilder,
	tweak_type: u8,
	sub_position: Wire,
	index: Wire,
) -> [Wire; TWEAK_WIRES] {
	// Bytes 0..8 hold the type, then the sub-position, then the index's low three bytes. Each
	// field is lifted to the top of the word and dropped into place, which selects its bytes and
	// carries everything above them off the top, so no field can reach into the next one's.
	let head = builder.add_constant_64(tweak_type as u64);
	let sub = builder.shr(builder.shl(sub_position, 32), 24);
	let word0 = builder.bxor(head, builder.bxor(sub, builder.shl(index, 40)));

	// Byte 8 is the index's top byte and bytes 9..16 are zero. Lifting that byte to the top of
	// the word and dropping it back to the bottom selects it and discards anything a caller
	// passes above bit 32, so the tweak stays the reference's 16 bytes rather than growing an
	// extra prover-chosen field.
	let word1 = builder.shr(builder.shl(index, 32), 56);

	[word0, word1]
}

/// The low [`DIGEST_LEN`] bytes of a BLAKE3 digest, repacked from 32-bit words into 64-bit wires.
fn truncate(builder: &CircuitBuilder, digest: &[Wire; 8]) -> [Wire; DIGEST_WIRES] {
	std::array::from_fn(|k| {
		let low = clear_high_bits(builder, digest[2 * k], 32);
		builder.bxor(low, builder.shl(digest[2 * k + 1], 32))
	})
}

#[cfg(test)]
mod tests {
	use binius_core::Word;
	use proptest::prelude::*;

	use super::*;
	use crate::hash_based_sig::{PUBLIC_PARAM_LEN, V};

	/// Builds the gadget over `payload`, and checks the circuit agrees with [`tweak_hash`].
	fn check(
		public_param: &PublicParam,
		tweak_type: u8,
		sub_position: u32,
		index: u32,
		payload: &[u8],
	) {
		assert_eq!(payload.len() % 8, 0, "payloads are a whole number of 64-bit wires");

		let b = CircuitBuilder::new();
		let param_w: [Wire; PUBLIC_PARAM_WIRES] = std::array::from_fn(|_| b.add_inout());
		let index_w = b.add_inout();
		let payload_w: Vec<Wire> = (0..payload.len() / 8).map(|_| b.add_inout()).collect();
		let sub_position_w = b.add_constant_64(sub_position as u64);
		let digest =
			circuit_tweak_hash(&b, &param_w, tweak_type, sub_position_w, index_w, &payload_w);
		let expected: [Wire; DIGEST_WIRES] = std::array::from_fn(|_| b.add_inout());
		for k in 0..DIGEST_WIRES {
			b.assert_eq("digest", digest[k], expected[k]);
		}

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&param_w, public_param);
		w[index_w] = Word::from_u64(index as u64);
		w.pack_bytes_le(&payload_w, payload);
		w.pack_bytes_le(
			&expected,
			&tweak_hash(public_param, tweak_type, sub_position, index, payload),
		);

		circuit.populate_wire_witness(&mut w).unwrap();
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.unwrap();
	}

	#[test]
	fn tweak_separates_everything() {
		let pp = [7u8; PUBLIC_PARAM_LEN];
		let x = [1u8; DIGEST_LEN];
		let base = tweak_hash(&pp, TWEAK_TYPE_CHAIN, 3, 5, &x);
		// A different type, position, index or parameter gives a different hash.
		assert_ne!(base, tweak_hash(&pp, TWEAK_TYPE_MERKLE, 3, 5, &x));
		assert_ne!(base, tweak_hash(&pp, TWEAK_TYPE_CHAIN, 4, 5, &x));
		assert_ne!(base, tweak_hash(&pp, TWEAK_TYPE_CHAIN, 3, 6, &x));
		assert_ne!(base, tweak_hash(&[8u8; PUBLIC_PARAM_LEN], TWEAK_TYPE_CHAIN, 3, 5, &x));
		// Hashing the exact byte string binds its length.
		let mut extended = [0u8; 2 * DIGEST_LEN];
		extended[..DIGEST_LEN].copy_from_slice(&x);
		assert_ne!(base, tweak_hash(&pp, TWEAK_TYPE_CHAIN, 3, 5, &extended));
	}

	#[test]
	fn tweak_layout_matches_the_reference() {
		// `[type(1) | sub_position(4) | index(4) | zeros(7)]`, little-endian.
		let tweak = make_tweak(TWEAK_TYPE_MERKLE, 0x0403_0201, 0x0807_0605);
		assert_eq!(
			tweak,
			[
				TWEAK_TYPE_MERKLE,
				1,
				2,
				3,
				4,
				5,
				6,
				7,
				8,
				0,
				0,
				0,
				0,
				0,
				0,
				0
			]
		);
	}

	#[test]
	fn circuit_matches_reference_at_every_call_site() {
		// The four payload shapes the scheme actually hashes: a chain value, a Merkle node's two
		// children, the padded message encoding input, and the WOTS public key's V chain tips.
		let pp = [3u8; PUBLIC_PARAM_LEN];
		let payload = |len: usize| -> Vec<u8> { (0..len).map(|i| (i * 37 + 11) as u8).collect() };
		check(&pp, TWEAK_TYPE_CHAIN, 17, 5, &payload(DIGEST_LEN));
		check(&pp, TWEAK_TYPE_MERKLE, 4, 9, &payload(2 * DIGEST_LEN));
		check(&pp, TWEAK_TYPE_ENCODING, 0, 5, &payload(64));
		check(&pp, TWEAK_TYPE_WOTS_PK, 0, 5, &payload(V * DIGEST_LEN));
	}

	#[test]
	fn circuit_matches_reference_at_the_index_extremes() {
		// The index straddles two wires in the tweak, so its top byte is the interesting one.
		let pp = [5u8; PUBLIC_PARAM_LEN];
		for index in [0, 1, u32::MAX, u32::MAX - 1, 1 << 24, (1 << 24) - 1] {
			check(&pp, TWEAK_TYPE_CHAIN, 0, index, &[9u8; DIGEST_LEN]);
		}
	}

	#[test]
	fn two_lane_hash_matches_the_reference_in_both_lanes() {
		// The paired core must agree with `tweak_hash` lane for lane, at the chain step's shape.
		let pp = [6u8; PUBLIC_PARAM_LEN];
		let payloads = [[1u8; DIGEST_LEN], [2u8; DIGEST_LEN]];
		let sub_positions = [17u32, 25];
		let index = 4242u32;

		let b = CircuitBuilder::new();
		let param_w: [Wire; PUBLIC_PARAM_WIRES] = std::array::from_fn(|_| b.add_inout());
		let index_w = b.add_inout();
		let payload_w: [[Wire; DIGEST_WIRES]; 2] =
			std::array::from_fn(|_| std::array::from_fn(|_| b.add_inout()));
		let digests = circuit_tweak_hash_2x(
			&b,
			&param_w,
			TWEAK_TYPE_CHAIN,
			sub_positions.map(|s| b.add_constant_64(s as u64)),
			index_w,
			[&payload_w[0], &payload_w[1]],
		);
		let expected: [[Wire; DIGEST_WIRES]; 2] =
			std::array::from_fn(|_| std::array::from_fn(|_| b.add_inout()));
		for lane in 0..2 {
			b.assert_eq_v(format!("lane[{lane}]"), digests[lane], expected[lane]);
		}

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&param_w, &pp);
		w[index_w] = Word::from_u64(index as u64);
		for lane in 0..2 {
			w.pack_bytes_le(&payload_w[lane], &payloads[lane]);
			w.pack_bytes_le(
				&expected[lane],
				&tweak_hash(&pp, TWEAK_TYPE_CHAIN, sub_positions[lane], index, &payloads[lane]),
			);
		}

		circuit.populate_wire_witness(&mut w).unwrap();
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.unwrap();
	}

	proptest! {
		#[test]
		fn circuit_matches_reference(
			tweak_type in 0u8..=3,
			sub_position in 0u32..=u32::MAX,
			index in 0u32..=u32::MAX,
			payload_wires in 1usize..=9,
			seed in any::<u64>(),
		) {
			use rand::{Rng, SeedableRng, rngs::StdRng};

			let mut rng = StdRng::seed_from_u64(seed);
			let mut public_param = [0u8; PUBLIC_PARAM_LEN];
			rng.fill_bytes(&mut public_param);
			let mut payload = vec![0u8; payload_wires * 8];
			rng.fill_bytes(&mut payload);

			check(&public_param, tweak_type, sub_position, index, &payload);
		}
	}
}
