// Copyright 2026 The Binius Developers
// Copyright (c) 2026 leanEthereum
//! WOTS (Winternitz one-time signature) with target-sum encoding.
//!
//! There are no checksum chains. Instead the signer grinds the signature randomness until the
//! encoding's digits sum to [`TARGET_SUM`], and a verifier that checks the sum knows no digit can
//! have been lowered without another being raised.

use std::iter;

use binius_frontend::{CircuitBuilder, Wire};
use rand::CryptoRng;

use super::{
	CHAIN_LENGTH, DIGEST_LEN, DIGEST_WIRES, Digest, MESSAGE_LEN, MESSAGE_WIRES, Message,
	PUBLIC_PARAM_WIRES, PublicParam, RANDOMNESS_LEN, RANDOMNESS_WIRES, Randomness, TARGET_SUM, V,
	W,
	hashing::{
		TWEAK_TYPE_CHAIN, TWEAK_TYPE_ENCODING, TWEAK_TYPE_WOTS_PK, circuit_tweak_hash,
		circuit_tweak_hash_2x, tweak_hash,
	},
};

/// Digits carried by each of the digest's two 64-bit words.
const DIGITS_PER_WORD: usize = V / 2;

/// The encoding hashes `message | randomness` zero-padded to this many bytes.
///
/// Fixing the payload length keeps the encoding two compressions and, because the hash binds the
/// exact byte string, keeps the padding out of the prover's hands.
const ENCODING_PAYLOAD_LEN: usize = 64;

const _: () = assert!(MESSAGE_LEN + RANDOMNESS_LEN <= ENCODING_PAYLOAD_LEN);
const _: () = assert!(2 * DIGITS_PER_WORD == V);

/// The target-sum encoding.
///
/// `D` is the encoding hash of `message | randomness`, truncated to 16 bytes. Each of its two
/// little-endian 64-bit words holds 21 digits of [`W`] bits: digit `i < 21` at bits `3i` of word
/// 0, digit `i >= 21` at bits `3(i - 21)` of word 1.
///
/// The encoding is valid exactly when the leftover top bit of *each* word (bits 63 and 127) is
/// zero and the digits sum to [`TARGET_SUM`]. Grinding the top bits to zero makes each word
/// exactly `sum(e_i * 2^{3i})` over its 21 digits, so both words decompose into digits with no
/// slack term.
///
/// Returns `None` when the randomness does not produce a valid encoding, which is the signal the
/// grinding loop retries on.
pub fn wots_encode(
	message: &Message,
	epoch: u32,
	public_param: &PublicParam,
	randomness: &Randomness,
) -> Option<[u8; V]> {
	let mut data = [0u8; ENCODING_PAYLOAD_LEN];
	data[..MESSAGE_LEN].copy_from_slice(message);
	data[MESSAGE_LEN..][..RANDOMNESS_LEN].copy_from_slice(randomness);
	let digest = tweak_hash(public_param, TWEAK_TYPE_ENCODING, 0, epoch, &data);

	if digest[7] >> 7 != 0 || digest[DIGEST_LEN - 1] >> 7 != 0 {
		return None; // the leftover top bit of each 64-bit word must be zero
	}
	let bit = |j: usize| (digest[j / 8] >> (j % 8)) & 1;
	let pos = |i: usize| {
		if i < DIGITS_PER_WORD {
			W * i
		} else {
			64 + W * (i - DIGITS_PER_WORD)
		}
	};
	let encoding: [u8; V] =
		std::array::from_fn(|i| (0..W).fold(0, |acc, k| acc | (bit(pos(i) + k) << k)));
	(encoding.iter().map(|&x| x as usize).sum::<usize>() == TARGET_SUM).then_some(encoding)
}

/// Draws randomness until it encodes validly.
///
/// The encoding is valid when both leftover bits are zero and the digits hit the target sum, so
/// grinding takes fewer than `2^15` attempts on average.
pub fn find_randomness_for_wots_encoding(
	message: &Message,
	epoch: u32,
	public_param: &PublicParam,
	rng: &mut impl CryptoRng,
) -> (Randomness, [u8; V]) {
	loop {
		let mut randomness = [0u8; RANDOMNESS_LEN];
		rng.fill_bytes(&mut randomness);
		if let Some(encoding) = wots_encode(message, epoch, public_param, &randomness) {
			return (randomness, encoding);
		}
	}
}

/// One chain step.
///
/// The position `chain_index * CHAIN_LENGTH + step` identifies the edge from chain value `step` to
/// `step + 1`, so no two edges anywhere in the instance share a tweak.
pub fn chain_step(
	public_param: &PublicParam,
	epoch: u32,
	chain_index: usize,
	step: usize,
	x: &Digest,
) -> Digest {
	let position = (chain_index * CHAIN_LENGTH + step) as u32;
	tweak_hash(public_param, TWEAK_TYPE_CHAIN, position, epoch, x)
}

/// Walks chain `chain_index` for `n` steps starting at chain value `start_step`.
pub fn iterate_hash(
	a: &Digest,
	n: usize,
	public_param: &PublicParam,
	epoch: u32,
	chain_index: usize,
	start_step: usize,
) -> Digest {
	(0..n).fold(*a, |acc, j| chain_step(public_param, epoch, chain_index, start_step + j, &acc))
}

/// Walks every chain from its tip to the public-key end.
pub fn recover_public_key(
	chain_tips: &[Digest; V],
	encoding: &[u8; V],
	epoch: u32,
	public_param: &PublicParam,
) -> [Digest; V] {
	std::array::from_fn(|i| {
		let digit = encoding[i] as usize;
		iterate_hash(&chain_tips[i], CHAIN_LENGTH - 1 - digit, public_param, epoch, i, digit)
	})
}

/// The Merkle leaf: the hash over the public parameter and the [`V`] concatenated chain ends.
pub fn wots_public_key_hash(
	public_param: &PublicParam,
	epoch: u32,
	chain_ends: &[Digest; V],
) -> Digest {
	let mut data = [0u8; V * DIGEST_LEN];
	for (chunk, end) in iter::zip(data.chunks_exact_mut(DIGEST_LEN), chain_ends) {
		chunk.copy_from_slice(end);
	}
	tweak_hash(public_param, TWEAK_TYPE_WOTS_PK, 0, epoch, &data)
}

/// In-circuit form of [`wots_encode`], returning the digits and constraining them to be a valid
/// encoding.
///
/// Both validity conditions are asserted rather than returned: an encoding whose leftover bits are
/// set, or whose digits miss the target sum, has no satisfying witness.
///
/// # Returns
///
/// The [`V`] digits, each a wire holding a value below [`CHAIN_LENGTH`].
pub fn circuit_wots_encode(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	epoch: Wire,
	message: &[Wire; MESSAGE_WIRES],
	randomness: &[Wire; RANDOMNESS_WIRES],
) -> [Wire; V] {
	let zero = builder.add_constant_64(0);

	// `message | randomness`, zero-padded to the fixed payload length.
	let mut payload = Vec::with_capacity(ENCODING_PAYLOAD_LEN / 8);
	payload.extend_from_slice(message);
	payload.extend_from_slice(randomness);
	payload.resize(ENCODING_PAYLOAD_LEN / 8, zero);

	let digest = circuit_tweak_hash(builder, public_param, TWEAK_TYPE_ENCODING, 0, epoch, &payload);

	// With `V * W = 126` of the digest's 128 bits spent on digits, the two leftover top bits are
	// what would otherwise let a word carry a slack term the digits do not account for.
	for (k, &word) in digest.iter().enumerate() {
		builder.assert_zero(format!("encoding_leftover_bit[{k}]"), builder.shr(word, 63));
	}

	// A digit is W bits from the middle of its word: lift them to the top of the word, then drop
	// them back to the bottom, so nothing above or below them survives.
	let digits: [Wire; V] = std::array::from_fn(|i| {
		let word = digest[i / DIGITS_PER_WORD];
		let shift = (W * (i % DIGITS_PER_WORD)) as u32;
		builder.shr(builder.shl(word, u64::BITS - shift - W as u32), u64::BITS - W as u32)
	});

	// The digits are each below CHAIN_LENGTH by construction, so the sum cannot overflow.
	let sum = digits
		.iter()
		.fold(zero, |acc, &digit| builder.iadd(acc, digit).0);
	builder.assert_eq("encoding_target_sum", sum, builder.add_constant_64(TARGET_SUM as u64));

	digits
}

/// In-circuit form of [`recover_public_key`].
///
/// Each chain evaluates all `CHAIN_LENGTH - 1` of its steps and takes the hashed value only past
/// its digit, because a step's tweak is a circuit constant and cannot be indexed by the digit. A
/// chain whose digit is `CHAIN_LENGTH - 1` therefore never advances, and its end is its tip, as
/// the scheme requires.
///
/// Chains are walked in pairs, both lanes of one compression per step. Chains are independent and
/// [`V`] is even, so every step of every chain finds a partner at the same step.
pub fn circuit_recover_public_key(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	epoch: Wire,
	chain_tips: &[[Wire; DIGEST_WIRES]; V],
	digits: &[Wire; V],
) -> [[Wire; DIGEST_WIRES]; V] {
	let chain_ends = (0..V / 2)
		.flat_map(|pair| {
			let (c0, c1) = (2 * pair, 2 * pair + 1);
			(0..CHAIN_LENGTH - 1).fold([chain_tips[c0], chain_tips[c1]], |current, step| {
				let next = circuit_tweak_hash_2x(
					builder,
					public_param,
					TWEAK_TYPE_CHAIN,
					[
						(c0 * CHAIN_LENGTH + step) as u32,
						(c1 * CHAIN_LENGTH + step) as u32,
					],
					epoch,
					[&current[0], &current[1]],
				);
				// The reference walks steps `digit..CHAIN_LENGTH - 2`, so this step applies
				// exactly when `digit <= step`.
				let past = builder.add_constant_64(step as u64 + 1);
				std::array::from_fn(|lane| {
					let advance = builder.icmp_ult(digits[2 * pair + lane], past);
					std::array::from_fn(|k| {
						builder.select(advance, next[lane][k], current[lane][k])
					})
				})
			})
		})
		.collect::<Vec<_>>();
	std::array::from_fn(|i| chain_ends[i])
}

/// In-circuit form of [`wots_public_key_hash`].
pub fn circuit_wots_public_key_hash(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	epoch: Wire,
	chain_ends: &[[Wire; DIGEST_WIRES]; V],
) -> [Wire; DIGEST_WIRES] {
	let payload = chain_ends.iter().flatten().copied().collect::<Vec<_>>();
	circuit_tweak_hash(builder, public_param, TWEAK_TYPE_WOTS_PK, 0, epoch, &payload)
}

#[cfg(test)]
mod tests {
	use binius_core::Word;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::hash_based_sig::PUBLIC_PARAM_LEN;

	/// A signature at `epoch`: random chain preimages walked to the digits the message encodes to.
	struct TestSignature {
		public_param: PublicParam,
		message: Message,
		randomness: Randomness,
		encoding: [u8; V],
		chain_tips: [Digest; V],
		chain_ends: [Digest; V],
	}

	impl TestSignature {
		fn generate(rng: &mut StdRng, epoch: u32) -> Self {
			let mut public_param = [0u8; PUBLIC_PARAM_LEN];
			rng.fill_bytes(&mut public_param);
			let mut message = [0u8; MESSAGE_LEN];
			rng.fill_bytes(&mut message);

			let (randomness, encoding) =
				find_randomness_for_wots_encoding(&message, epoch, &public_param, rng);

			// A signature's chain tip is the secret preimage walked as far as its digit; the
			// verifier walks the rest.
			let mut pre_images = [[0u8; DIGEST_LEN]; V];
			for pre_image in pre_images.iter_mut() {
				rng.fill_bytes(pre_image);
			}
			let chain_tips: [Digest; V] = std::array::from_fn(|i| {
				iterate_hash(&pre_images[i], encoding[i] as usize, &public_param, epoch, i, 0)
			});
			let chain_ends = recover_public_key(&chain_tips, &encoding, epoch, &public_param);

			Self {
				public_param,
				message,
				randomness,
				encoding,
				chain_tips,
				chain_ends,
			}
		}
	}

	#[test]
	fn encoding_is_valid_by_construction() {
		let mut rng = StdRng::seed_from_u64(0);
		let sig = TestSignature::generate(&mut rng, 7);
		assert_eq!(sig.encoding.iter().map(|&e| e as usize).sum::<usize>(), TARGET_SUM);
		assert!(sig.encoding.iter().all(|&e| (e as usize) < CHAIN_LENGTH));
	}

	#[test]
	fn a_chain_end_is_its_tip_at_the_last_digit() {
		// The one chain the verifier never advances.
		let pp = [4u8; PUBLIC_PARAM_LEN];
		let tip = [9u8; DIGEST_LEN];
		assert_eq!(iterate_hash(&tip, CHAIN_LENGTH - 1 - (CHAIN_LENGTH - 1), &pp, 3, 0, 7), tip);
	}

	/// Builds the encode-and-walk circuit, populates it from `sig`, and returns the result of
	/// checking the constraint system.
	fn run(sig: &TestSignature, epoch: u32) -> Result<(), String> {
		let b = CircuitBuilder::new();
		let param_w: [Wire; PUBLIC_PARAM_WIRES] = std::array::from_fn(|_| b.add_inout());
		let epoch_w = b.add_inout();
		let message_w: [Wire; MESSAGE_WIRES] = std::array::from_fn(|_| b.add_inout());
		let randomness_w: [Wire; RANDOMNESS_WIRES] = std::array::from_fn(|_| b.add_witness());
		let tips_w: [[Wire; DIGEST_WIRES]; V] =
			std::array::from_fn(|_| std::array::from_fn(|_| b.add_witness()));
		let leaf_w: [Wire; DIGEST_WIRES] = std::array::from_fn(|_| b.add_inout());

		let digits = circuit_wots_encode(&b, &param_w, epoch_w, &message_w, &randomness_w);
		let ends = circuit_recover_public_key(&b, &param_w, epoch_w, &tips_w, &digits);
		let leaf = circuit_wots_public_key_hash(&b, &param_w, epoch_w, &ends);
		b.assert_eq_v("leaf", leaf, leaf_w);

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&param_w, &sig.public_param);
		w[epoch_w] = Word::from_u64(epoch as u64);
		w.pack_bytes_le(&message_w, &sig.message);
		w.pack_bytes_le(&randomness_w, &sig.randomness);
		for (wires, tip) in tips_w.iter().zip(&sig.chain_tips) {
			w.pack_bytes_le(wires, tip);
		}
		w.pack_bytes_le(&leaf_w, &wots_public_key_hash(&sig.public_param, epoch, &sig.chain_ends));

		circuit
			.populate_wire_witness(&mut w)
			.map_err(|e| format!("populate: {e:?}"))?;
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.map_err(|e| format!("verify: {e:?}"))
	}

	#[test]
	fn circuit_recovers_the_public_key() {
		let mut rng = StdRng::seed_from_u64(1);
		let epoch = 12345;
		let sig = TestSignature::generate(&mut rng, epoch);
		run(&sig, epoch).unwrap();
	}

	#[test]
	fn circuit_digits_match_the_reference_encoding() {
		let mut rng = StdRng::seed_from_u64(2);
		let epoch = 9;
		let sig = TestSignature::generate(&mut rng, epoch);

		let b = CircuitBuilder::new();
		let param_w: [Wire; PUBLIC_PARAM_WIRES] = std::array::from_fn(|_| b.add_inout());
		let epoch_w = b.add_inout();
		let message_w: [Wire; MESSAGE_WIRES] = std::array::from_fn(|_| b.add_inout());
		let randomness_w: [Wire; RANDOMNESS_WIRES] = std::array::from_fn(|_| b.add_inout());
		let digits = circuit_wots_encode(&b, &param_w, epoch_w, &message_w, &randomness_w);
		let expected: [Wire; V] = std::array::from_fn(|_| b.add_inout());
		b.assert_eq_v("digits", digits, expected);

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&param_w, &sig.public_param);
		w[epoch_w] = Word::from_u64(epoch as u64);
		w.pack_bytes_le(&message_w, &sig.message);
		w.pack_bytes_le(&randomness_w, &sig.randomness);
		for (wire, &digit) in expected.iter().zip(&sig.encoding) {
			w[*wire] = Word::from_u64(digit as u64);
		}

		circuit.populate_wire_witness(&mut w).unwrap();
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.unwrap();
	}

	#[test]
	fn circuit_rejects_randomness_that_does_not_encode() {
		let mut rng = StdRng::seed_from_u64(3);
		let epoch = 4;
		let mut sig = TestSignature::generate(&mut rng, epoch);

		// Any randomness the grinder did not settle on fails one of the two conditions, so the
		// encoding constraints have no satisfying witness.
		let mut bad = sig.randomness;
		bad[0] ^= 0xFF;
		assert!(
			wots_encode(&sig.message, epoch, &sig.public_param, &bad).is_none(),
			"the tampered randomness happened to encode validly; pick another"
		);
		sig.randomness = bad;
		assert!(run(&sig, epoch).is_err(), "an invalid encoding must not verify");
	}

	#[test]
	fn circuit_rejects_a_tampered_chain_tip() {
		let mut rng = StdRng::seed_from_u64(4);
		let epoch = 4;
		let mut sig = TestSignature::generate(&mut rng, epoch);
		sig.chain_tips[0][0] ^= 0xFF;
		assert!(run(&sig, epoch).is_err(), "a tampered tip must not reach the public key");
	}

	#[test]
	fn circuit_rejects_a_signature_from_another_epoch() {
		let mut rng = StdRng::seed_from_u64(5);
		let epoch = 4;
		let sig = TestSignature::generate(&mut rng, epoch);
		// Every tweak carries the epoch, so the chains and the encoding both move with it.
		assert!(run(&sig, epoch + 1).is_err(), "an epoch it was not signed at must not verify");
	}
}
