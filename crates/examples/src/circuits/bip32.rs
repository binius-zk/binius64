// Copyright 2025 Irreducible Inc.
//! BIP32 hierarchical-deterministic key derivation as a Binius64 circuit gadget.
//!
//! [`bip32_derive_compressed`] derives a compressed secp256k1 public key at a BIP32 derivation
//! path, supporting both hardened and non-hardened child steps. The derivation is run for the full
//! maximum tree depth and the public key at the actual path depth is selected with a multiplexer,
//! so the circuit shape is independent of the (witness) depth.
//!
//! BIP32 spec: <https://en.bitcoin.it/wiki/BIP_0032>.
//!
//! # Word conventions
//!
//! HMAC-SHA512 (and SHA-512) consume and produce *big-endian* 64-bit words: word `i` holds
//! `u64::from_be_bytes(bytes[8*i .. 8*i + 8])`. A [`BigUint`] stores *little-endian* 64-bit limbs,
//! and limb values are plain integers. The numeric value of a SHA-512 word therefore equals the
//! corresponding 64-bit limb of the big-endian-parsed integer, so converting between a 256-bit
//! hash half and a [`BigUint`] is just a limb reversal — no per-byte swapping is required.

use std::array;

use anyhow::{Result, bail};
use binius_circuits::{
	bignum::{BigUint, select as select_biguint},
	bitcoin::p2pkh_signature::compress_pubkey,
	ecdsa::scalar_mul::scalar_mul,
	hmac::hmac_sha512_fixed,
	multiplexer::multi_wire_multiplex,
	secp256k1::{Secp256k1, Secp256k1Affine},
	sha256::sha256_fixed,
};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use bitcoin::{
	NetworkKind,
	bip32::{ChildNumber, Xpriv},
	secp256k1::{PublicKey, Secp256k1 as BtcSecp256k1},
};
use clap::Args;
use sha2::{Digest, Sha256};

use crate::ExampleCircuit;

/// Bit 31 of a derivation index marks a hardened child.
const HARDENED_BIT: u32 = 0x8000_0000;

/// Derive the BIP32 compressed secp256k1 public key at `depth` along `path`.
///
/// The maximum tree depth is the circuit parameter `path.len()`.
///
/// # Arguments
/// * `b` - circuit builder
/// * `seed` - the 512-bit BIP32 seed as eight big-endian 64-bit words
/// * `path` - `max_depth` derivation-index words. Only the low 32 bits are significant; bit 31 set
///   means a hardened child. Entries beyond `depth` are still derived but never selected.
/// * `depth` - the actual path depth (`<= max_depth`) selecting which level's public key to output
///
/// # Returns
/// The 33-byte compressed public key as nine four-byte big-endian words (the layout produced by
/// [`compress_pubkey`]).
///
/// # Assumptions
/// The negligible-probability BIP32 invalid-key cases (`parse256(I_L) >= n`, or a derived key equal
/// to zero) are *not* constrained; the gadget assumes a valid witness.
pub fn bip32_derive_compressed(
	b: &mut CircuitBuilder,
	seed: &[Wire; 8],
	path: &[Wire],
	depth: Wire,
) -> Vec<Wire> {
	let max_depth = path.len();
	let curve = Secp256k1::new(b);

	// Master key: I = HMAC-SHA512("Bitcoin seed", seed). HMAC zero-pads the key to the block size,
	// so the four trailing zero bytes of the second key word are exactly the padding the spec adds.
	let key_words = [
		b.add_constant_64(u64::from_be_bytes(*b"Bitcoin ")),
		b.add_constant_64(u64::from_be_bytes([b's', b'e', b'e', b'd', 0, 0, 0, 0])),
	];
	let master = hmac_sha512_fixed(b, &key_words, seed, 64);
	let mut k = il_scalar(&master);
	let mut c = ir_words(&master);

	// Compressed public key at every level 0..=max_depth: needed as the parent pubkey for any
	// non-hardened step, and as the candidates for the depth multiplexer.
	let mut serp_levels: Vec<Vec<Wire>> = Vec::with_capacity(max_depth + 1);

	for level in 0..max_depth {
		// Parent public key P = k * G.
		let point = scalar_mul(b, &curve, &k, Secp256k1Affine::generator(b));
		serp_levels.push(compress_pubkey(b, &point.x, &point.y));

		// Hardened iff bit 31 of the index is set (moved to the MSB for `select`).
		let is_hardened = b.shl(path[level], 32);

		// CKD data is `prefix || value[32] || ser32(index)` either way (37 bytes):
		// hardened => 0x00 || ser256(k_par); normal => (0x02|0x03) || x_be.
		let prefix_norm = compressed_prefix(b, &point.y);
		let prefix = b.select(is_hardened, b.add_constant(Word::ZERO), prefix_norm);
		// Select the 32-byte value field (the scalar `k` or the x-coordinate), then read it out as
		// big-endian words (limb `3 - i` is the i-th most significant word).
		let value_field = select_biguint(b, is_hardened, &k, &point.x);
		let value = [
			value_field.limbs[3],
			value_field.limbs[2],
			value_field.limbs[1],
			value_field.limbs[0],
		];

		let message = assemble_message(b, prefix, &value, path[level]);
		let i = hmac_sha512_fixed(b, &c, &message, 37);

		// k_child = (parse256(I_L) + k_par) mod n ; c_child = I_R.
		let il = il_scalar(&i);
		k = curve.f_scalar().add(b, &il, &k);
		c = ir_words(&i);
	}

	// Compressed public key for the deepest derived level.
	let point = scalar_mul(b, &curve, &k, Secp256k1Affine::generator(b));
	serp_levels.push(compress_pubkey(b, &point.x, &point.y));

	let refs: Vec<&[Wire]> = serp_levels.iter().map(Vec::as_slice).collect();
	multi_wire_multiplex(b, &refs, depth)
}

/// Interpret the first 32 bytes (`I_L`) of a SHA-512 output as a 256-bit scalar (four little-endian
/// limbs). SHA-512 words are big-endian, so `hash[3]` is the least significant limb.
fn il_scalar(hash: &[Wire; 8]) -> BigUint {
	BigUint {
		limbs: vec![hash[3], hash[2], hash[1], hash[0]],
	}
}

/// The last 32 bytes (`I_R`, the chain code) of a SHA-512 output, as four big-endian words ready to
/// be used directly as an HMAC-SHA512 key.
fn ir_words(hash: &[Wire; 8]) -> [Wire; 4] {
	[hash[4], hash[5], hash[6], hash[7]]
}

/// Compressed-pubkey prefix byte (`0x02` if `y` is even, `0x03` if odd), placed in the most
/// significant byte of the returned word.
fn compressed_prefix(b: &CircuitBuilder, y: &BigUint) -> Wire {
	let y_is_odd = b.shl(y.limbs[0], 63);
	let even = b.add_constant_64(0x02u64 << 56);
	let odd = b.add_constant_64(0x03u64 << 56);
	b.select(y_is_odd, odd, even)
}

/// Assemble the 37-byte CKD data `prefix || value[32] || ser32(index)` into five big-endian 64-bit
/// words.
///
/// `prefix` holds the prefix byte in its most significant byte; `value` is the 32-byte field as
/// four big-endian words; only the low 32 bits of `index` are used. The prefix occupies byte 0, so
/// `value` is shifted right by one byte and straddles word boundaries. The low three bytes of the
/// last word are left zero — SHA-512 masks them out for a 37-byte message anyway.
///
/// The operands combined for each word occupy disjoint byte ranges, so `bxor` (cheaper than `bor`
/// in this constraint system) is equivalent to a bitwise-or here.
fn assemble_message(b: &CircuitBuilder, prefix: Wire, value: &[Wire; 4], index: Wire) -> [Wire; 5] {
	let m0 = b.bxor(prefix, b.shr(value[0], 8));
	let m1 = b.bxor(b.shl(value[0], 56), b.shr(value[1], 8));
	let m2 = b.bxor(b.shl(value[1], 56), b.shr(value[2], 8));
	let m3 = b.bxor(b.shl(value[2], 56), b.shr(value[3], 8));
	let index_bytes = b.band(index, b.add_constant_64(0xFFFF_FFFF));
	let m4 = b.bxor(b.shl(value[3], 56), b.shl(index_bytes, 24));
	[m0, m1, m2, m3, m4]
}

/// Example circuit proving knowledge of a BIP32 seed and derivation path whose derived compressed
/// secp256k1 public key hashes (SHA-256) to a public digest.
///
/// The seed, path, and depth are private (witness); only the SHA-256 hash of the compressed public
/// key is public (inout).
pub struct Bip32Example {
	seed: [Wire; 8],
	path: Vec<Wire>,
	depth: Wire,
	expected_hash: [Wire; 8],
	max_depth: usize,
}

#[derive(Args, Debug, Clone)]
pub struct Params {
	/// Maximum BIP32 tree depth supported by the circuit.
	#[arg(long, default_value_t = 5)]
	pub max_depth: usize,
}

#[derive(Args, Debug, Clone)]
pub struct Instance {
	/// BIP32 seed as 128 hex chars (exactly 64 bytes). Defaults to a fixed test seed.
	#[arg(long)]
	pub seed: Option<String>,

	/// Derivation path as comma-separated children, e.g. "0',1,2'". A trailing `'` or `h` marks a
	/// hardened child. Empty selects the master key.
	#[arg(long, value_delimiter = ',', value_parser = parse_child, default_value = "0'")]
	pub path: Vec<u32>,
}

impl ExampleCircuit for Bip32Example {
	type Params = Params;
	type Instance = Instance;

	fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
		let max_depth = params.max_depth;
		let seed: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let path: Vec<Wire> = (0..max_depth).map(|_| builder.add_witness()).collect();
		let depth = builder.add_witness();

		let pubkey = bip32_derive_compressed(builder, &seed, &path, depth);

		// SHA-256 of the 33-byte compressed public key. The digest is the only public input.
		let digest = sha256_fixed(builder, &pubkey, 33);
		let expected_hash: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		for (idx, (&computed, &expected)) in digest.iter().zip(&expected_hash).enumerate() {
			builder.assert_eq(format!("pubkey_hash[{idx}]"), computed, expected);
		}

		Ok(Self {
			seed,
			path,
			depth,
			expected_hash,
			max_depth,
		})
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		if instance.path.len() > self.max_depth {
			bail!("path depth {} exceeds max_depth {}", instance.path.len(), self.max_depth);
		}

		let seed_bytes = match &instance.seed {
			Some(hex_str) => {
				let bytes = hex::decode(hex_str.trim_start_matches("0x"))
					.map_err(|e| anyhow::anyhow!("invalid seed hex: {e}"))?;
				let bytes: [u8; 64] = bytes
					.try_into()
					.map_err(|_| anyhow::anyhow!("seed must be exactly 64 bytes (512 bits)"))?;
				bytes
			}
			None => array::from_fn(|i| i as u8),
		};

		// Seed -> 8 big-endian words.
		for i in 0..8 {
			let word = u64::from_be_bytes(seed_bytes[8 * i..8 * i + 8].try_into().unwrap());
			w[self.seed[i]] = Word::from_u64(word);
		}

		// Path words, padding unused tail levels with index 0 (a normal child, never selected).
		for i in 0..self.max_depth {
			let idx = instance.path.get(i).copied().unwrap_or(0);
			w[self.path[i]] = Word::from_u64(idx as u64);
		}
		w[self.depth] = Word::from_u64(instance.path.len() as u64);

		// Reference derivation via the `bitcoin` crate, then SHA-256 the compressed public key.
		let pubkey = derive_compressed_pubkey(&seed_bytes, &instance.path)?;
		let hash: [u8; 32] = Sha256::digest(pubkey).into();
		let words = sha256_digest_words(&hash);
		for i in 0..8 {
			w[self.expected_hash[i]] = Word::from_u64(words[i]);
		}

		tracing::info!(
			"BIP32 compressed pubkey {} -> SHA-256 {} (depth {})",
			hex::encode(pubkey),
			hex::encode(hash),
			instance.path.len()
		);
		Ok(())
	}
}

/// Parse a single derivation-path child like "0", "44'", or "5h" into a 32-bit index with the
/// hardened bit set when suffixed.
fn parse_child(s: &str) -> Result<u32, String> {
	let (digits, hardened) = match s.strip_suffix(['\'', 'h', 'H']) {
		Some(rest) => (rest, true),
		None => (s, false),
	};
	let idx: u32 = digits
		.parse()
		.map_err(|e| format!("invalid child index '{s}': {e}"))?;
	if idx >= HARDENED_BIT {
		return Err(format!("child index {idx} out of range (must be < 2^31)"));
	}
	Ok(if hardened { idx | HARDENED_BIT } else { idx })
}

/// Reference BIP32 derivation oracle: returns the 33-byte compressed public key at `path`.
fn derive_compressed_pubkey(seed: &[u8; 64], path: &[u32]) -> Result<[u8; 33]> {
	let secp = BtcSecp256k1::new();
	let master = Xpriv::new_master(NetworkKind::Main, seed)
		.map_err(|e| anyhow::anyhow!("invalid master key: {e}"))?;
	let children: Vec<ChildNumber> = path
		.iter()
		.map(|&idx| child_number(idx))
		.collect::<Result<_>>()?;
	let derived = master
		.derive_priv(&secp, &children)
		.map_err(|e| anyhow::anyhow!("derivation failed: {e}"))?;
	let pubkey = PublicKey::from_secret_key(&secp, &derived.private_key);
	Ok(pubkey.serialize())
}

/// Convert a raw 32-bit index (hardened bit encoded in bit 31) into a `bitcoin` [`ChildNumber`].
fn child_number(idx: u32) -> Result<ChildNumber> {
	let child = if idx & HARDENED_BIT != 0 {
		ChildNumber::from_hardened_idx(idx & !HARDENED_BIT)
	} else {
		ChildNumber::from_normal_idx(idx)
	};
	child.map_err(|e| anyhow::anyhow!("invalid child index {idx}: {e}"))
}

/// Pack a 32-byte SHA-256 digest into eight 4-byte big-endian words (low 32 bits each), matching
/// the output layout of [`sha256_fixed`].
fn sha256_digest_words(digest: &[u8; 32]) -> [u64; 8] {
	array::from_fn(|i| u32::from_be_bytes(digest[4 * i..4 * i + 4].try_into().unwrap()) as u64)
}

#[cfg(test)]
mod tests {
	use binius_core::verify::verify_constraints;
	use binius_frontend::CircuitBuilder;

	use super::*;

	/// Number of 4-byte big-endian words in a compressed public key (33 bytes, zero-padded to 36).
	const COMPRESSED_PUBKEY_WORDS: usize = 9;

	/// Pack a 33-byte compressed public key into nine 4-byte big-endian words (last word
	/// zero-padded), matching the layout produced by `compress_pubkey`.
	fn compressed_pubkey_words(compressed: &[u8; 33]) -> [u64; COMPRESSED_PUBKEY_WORDS] {
		let mut padded = [0u8; 36];
		padded[..33].copy_from_slice(compressed);
		array::from_fn(|i| u32::from_be_bytes(padded[4 * i..4 * i + 4].try_into().unwrap()) as u64)
	}

	/// Build a standalone circuit around `bip32_derive_compressed`, populate it for `seed`/`path`,
	/// assert the derived pubkey equals the `bitcoin`-crate oracle, and verify all constraints.
	fn check_derivation(seed: &[u8; 64], path: &[u32], max_depth: usize) {
		assert!(path.len() <= max_depth);
		let builder = CircuitBuilder::new();

		let seed_wires: [Wire; 8] = array::from_fn(|_| builder.add_witness());
		let path_wires: Vec<Wire> = (0..max_depth).map(|_| builder.add_witness()).collect();
		let depth_wire = builder.add_witness();

		// `bip32_derive_compressed` needs `&mut`; the builder is consumed by `build` afterwards.
		let mut b = builder;
		let derived = bip32_derive_compressed(&mut b, &seed_wires, &path_wires, depth_wire);
		let builder = b;

		let expected_wires: Vec<Wire> = (0..derived.len()).map(|_| builder.add_inout()).collect();
		for (idx, (&computed, &expected)) in derived.iter().zip(&expected_wires).enumerate() {
			builder.assert_eq(format!("pubkey[{idx}]"), computed, expected);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();

		for i in 0..8 {
			let word = u64::from_be_bytes(seed[8 * i..8 * i + 8].try_into().unwrap());
			w[seed_wires[i]] = Word::from_u64(word);
		}
		for i in 0..max_depth {
			let idx = path.get(i).copied().unwrap_or(0);
			w[path_wires[i]] = Word::from_u64(idx as u64);
		}
		w[depth_wire] = Word::from_u64(path.len() as u64);

		let expected = derive_compressed_pubkey(seed, path).expect("oracle derivation");
		let words = compressed_pubkey_words(&expected);
		for i in 0..COMPRESSED_PUBKEY_WORDS {
			w[expected_wires[i]] = Word::from_u64(words[i]);
		}

		circuit
			.populate_wire_witness(&mut w)
			.expect("witness population");
		verify_constraints(circuit.constraint_system(), &w.into_value_vec())
			.expect("constraints satisfied");
	}

	fn test_seed() -> [u8; 64] {
		array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(1))
	}

	#[test]
	fn master_pubkey_depth_zero() {
		check_derivation(&test_seed(), &[], 3);
	}

	#[test]
	fn single_hardened_step() {
		check_derivation(&test_seed(), &[HARDENED_BIT], 3);
	}

	#[test]
	fn single_normal_step() {
		check_derivation(&test_seed(), &[7], 3);
	}

	#[test]
	fn mixed_path_full_depth() {
		// m/0'/1/2'/2/1000000000 (the BIP32 test-vector-1 path shape).
		let path = [HARDENED_BIT, 1, 2 | HARDENED_BIT, 2, 1_000_000_000];
		check_derivation(&test_seed(), &path, 5);
	}

	#[test]
	fn short_path_with_padding() {
		// depth 2 within a max-depth-5 circuit exercises the multiplexer and the ignored tail.
		check_derivation(&test_seed(), &[5 | HARDENED_BIT, 9], 5);
	}

	#[test]
	fn hardened_boundary_indices() {
		// Largest normal index and smallest hardened index sit either side of bit 31.
		check_derivation(&test_seed(), &[HARDENED_BIT - 1, HARDENED_BIT], 2);
	}

	#[test]
	fn parse_child_accepts_hardened_and_normal() {
		assert_eq!(parse_child("0").unwrap(), 0);
		assert_eq!(parse_child("44'").unwrap(), 44 | HARDENED_BIT);
		assert_eq!(parse_child("5h").unwrap(), 5 | HARDENED_BIT);
		assert!(parse_child("2147483648").is_err());
	}

	/// Full example: build the circuit, populate it for a path, and verify that the
	/// circuit-computed SHA-256 of the compressed pubkey matches the public digest.
	#[test]
	fn example_proves_pubkey_hash() {
		let mut builder = CircuitBuilder::new();
		let example = Bip32Example::build(Params { max_depth: 3 }, &mut builder)
			.expect("build example circuit");
		let circuit = builder.build();

		let mut w = circuit.new_witness_filler();
		let instance = Instance {
			seed: None,
			path: vec![HARDENED_BIT, 1],
		};
		example
			.populate_witness(instance, &mut w)
			.expect("populate witness");

		circuit
			.populate_wire_witness(&mut w)
			.expect("witness population");
		verify_constraints(circuit.constraint_system(), &w.into_value_vec())
			.expect("constraints satisfied");
	}
}
