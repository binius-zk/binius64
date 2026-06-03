// Copyright 2026 The Binius Developers
//! Bitcoin key/address helpers built on the `bitcoin` crate.
//!
//! The fixed circuit commits to `SHA-256(compressed_pubkey)`, which is the inner hash of
//! `HASH160 = RIPEMD160(SHA256(pubkey))`. Every address type supported here (P2PKH, native SegWit
//! P2WPKH, nested SegWit P2SH-P2WPKH) is derived from that `HASH160`, so the verifier can bind a
//! proof to an address by recomputing the address's `scriptPubKey` from the stored SHA-256 hash.
//! Taproot is intentionally unsupported: it commits to an EC-tweaked output key, not `HASH160`.

use anyhow::{Result, bail};
use bitcoin::{
	Address, CompressedPublicKey, Network, NetworkKind, PubkeyHash, ScriptBuf, WPubkeyHash,
	address::NetworkUnchecked,
	bip32::{ChildNumber, Xpriv},
	hashes::Hash,
	secp256k1::{PublicKey, Secp256k1},
};
use ripemd::{Digest as _, Ripemd160};
use sha2::Sha256;

/// Bit 31 of a derivation index marks a hardened child (matches `Bip32Example`).
pub const HARDENED_BIT: u32 = 0x8000_0000;

/// All address types the demo supports. Each maps to its standard BIP purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressType {
	/// Legacy pay-to-pubkey-hash (`1...`), BIP44 purpose 44'.
	P2pkh,
	/// Native SegWit pay-to-witness-pubkey-hash (`bc1q...`), BIP84 purpose 84'.
	P2wpkh,
	/// Nested SegWit P2WPKH wrapped in P2SH (`3...`), BIP49 purpose 49'.
	P2shwpkh,
}

impl AddressType {
	/// The three types in menu order.
	pub const ALL: [AddressType; 3] = [
		AddressType::P2pkh,
		AddressType::P2wpkh,
		AddressType::P2shwpkh,
	];

	/// BIP44/49/84 purpose index (without the hardened bit).
	pub fn purpose(self) -> u32 {
		match self {
			AddressType::P2pkh => 44,
			AddressType::P2wpkh => 84,
			AddressType::P2shwpkh => 49,
		}
	}

	/// Human-readable label for prompts.
	pub fn label(self) -> &'static str {
		match self {
			AddressType::P2pkh => "P2PKH (legacy, 1...)",
			AddressType::P2wpkh => "P2WPKH (native SegWit, bc1q...)",
			AddressType::P2shwpkh => "P2SH-P2WPKH (nested SegWit, 3...)",
		}
	}

	/// Detect the type of a parsed address. Returns `None` for unsupported types (e.g. Taproot).
	pub fn from_address(addr: &Address) -> Option<AddressType> {
		match addr.address_type()? {
			bitcoin::AddressType::P2pkh => Some(AddressType::P2pkh),
			bitcoin::AddressType::P2wpkh => Some(AddressType::P2wpkh),
			bitcoin::AddressType::P2sh => Some(AddressType::P2shwpkh),
			_ => None,
		}
	}
}

/// A BIP44-style path `m/purpose'/0'/account'/change/index` as raw indices (hardened bit set on the
/// first three levels). Depth 5, matching the circuit's `max_depth`.
pub fn bip44_path(ty: AddressType, account: u32, change: u32, index: u32) -> Vec<u32> {
	vec![
		ty.purpose() | HARDENED_BIT,
		HARDENED_BIT, // coin type 0' (Bitcoin mainnet)
		account | HARDENED_BIT,
		change,
		index,
	]
}

/// Format a raw-index path as a human-readable BIP32 string like `m/44'/0'/0'/0/5`.
pub fn format_path(path: &[u32]) -> String {
	let mut s = String::from("m");
	for &idx in path {
		if idx & HARDENED_BIT != 0 {
			s.push_str(&format!("/{}'", idx & !HARDENED_BIT));
		} else {
			s.push_str(&format!("/{idx}"));
		}
	}
	s
}

/// Parse a BIP32 path string like `44'/0'/0'/0/5` (optionally prefixed `m/`) into raw indices.
pub fn parse_path(s: &str) -> Result<Vec<u32>> {
	let s = s.trim().trim_start_matches("m/").trim_start_matches('m');
	if s.is_empty() {
		return Ok(Vec::new());
	}
	s.split('/')
		.map(|part| {
			let part = part.trim();
			let (digits, hardened) = match part.strip_suffix(['\'', 'h', 'H']) {
				Some(rest) => (rest, true),
				None => (part, false),
			};
			let idx: u32 = digits
				.parse()
				.map_err(|e| anyhow::anyhow!("invalid path element '{part}': {e}"))?;
			if idx >= HARDENED_BIT {
				bail!("path element {idx} out of range (must be < 2^31)");
			}
			Ok(if hardened { idx | HARDENED_BIT } else { idx })
		})
		.collect()
}

/// Convert a raw 32-bit index into a `bitcoin` [`ChildNumber`].
fn child_number(idx: u32) -> Result<ChildNumber> {
	let child = if idx & HARDENED_BIT != 0 {
		ChildNumber::from_hardened_idx(idx & !HARDENED_BIT)
	} else {
		ChildNumber::from_normal_idx(idx)
	};
	child.map_err(|e| anyhow::anyhow!("invalid child index {idx}: {e}"))
}

/// Derive the compressed secp256k1 public key at `path` from a 64-byte BIP32 seed.
pub fn derive_compressed_pubkey(seed: &[u8; 64], path: &[u32]) -> Result<CompressedPublicKey> {
	let secp = Secp256k1::new();
	let master = Xpriv::new_master(NetworkKind::Main, seed)
		.map_err(|e| anyhow::anyhow!("master key: {e}"))?;
	let children: Vec<ChildNumber> = path
		.iter()
		.map(|&idx| child_number(idx))
		.collect::<Result<_>>()?;
	let derived = master
		.derive_priv(&secp, &children)
		.map_err(|e| anyhow::anyhow!("derivation failed: {e}"))?;
	let pubkey = PublicKey::from_secret_key(&secp, &derived.private_key);
	Ok(CompressedPublicKey(pubkey))
}

/// SHA-256 of the 33-byte compressed public key — the circuit's public input.
pub fn pubkey_sha256(pubkey: &CompressedPublicKey) -> [u8; 32] {
	Sha256::digest(pubkey.to_bytes()).into()
}

/// `HASH160 = RIPEMD160(SHA256(pubkey))` computed from the stored SHA-256 digest.
pub fn hash160_from_sha256(sha256: &[u8; 32]) -> [u8; 20] {
	Ripemd160::digest(sha256).into()
}

/// Build the Bitcoin address of the given type for a public key.
pub fn address_for(ty: AddressType, pubkey: &CompressedPublicKey) -> Address {
	match ty {
		AddressType::P2pkh => Address::p2pkh(pubkey, Network::Bitcoin),
		AddressType::P2wpkh => Address::p2wpkh(pubkey, Network::Bitcoin),
		AddressType::P2shwpkh => Address::p2shwpkh(pubkey, Network::Bitcoin),
	}
}

/// The `scriptPubKey` of `ty` for a given `HASH160`. Address equality is decided by comparing
/// `scriptPubKey`s, which is exactly how a Bitcoin output binds to a key hash.
pub fn script_pubkey_for(ty: AddressType, hash160: &[u8; 20]) -> ScriptBuf {
	match ty {
		AddressType::P2pkh => ScriptBuf::new_p2pkh(&PubkeyHash::from_byte_array(*hash160)),
		AddressType::P2wpkh => ScriptBuf::new_p2wpkh(&WPubkeyHash::from_byte_array(*hash160)),
		AddressType::P2shwpkh => {
			// Nested SegWit: the P2SH redeem script is the P2WPKH program.
			let redeem = ScriptBuf::new_p2wpkh(&WPubkeyHash::from_byte_array(*hash160));
			ScriptBuf::new_p2sh(&redeem.script_hash())
		}
	}
}

/// Parse a mainnet Bitcoin address string.
pub fn parse_address(s: &str) -> Result<Address> {
	let unchecked: Address<NetworkUnchecked> = s
		.trim()
		.parse()
		.map_err(|e| anyhow::anyhow!("invalid Bitcoin address: {e}"))?;
	unchecked
		.require_network(Network::Bitcoin)
		.map_err(|e| anyhow::anyhow!("address is not a Bitcoin mainnet address: {e}"))
}

/// Check that `sha256` (the circuit's public input) corresponds to `address`. Returns the detected
/// address type on success.
pub fn bind_address_to_hash(address: &Address, sha256: &[u8; 32]) -> Result<AddressType> {
	let ty = AddressType::from_address(address).ok_or_else(|| {
		anyhow::anyhow!(
			"unsupported address type (only P2PKH, P2WPKH and P2SH-P2WPKH are supported)"
		)
	})?;
	let hash160 = hash160_from_sha256(sha256);
	let expected = script_pubkey_for(ty, &hash160);
	if address.script_pubkey() != expected {
		bail!("proof does not match this address (public-key hash mismatch)");
	}
	Ok(ty)
}

/// Scan up to `max_addresses` BIP44 candidate paths for `ty`, looking for one whose derived address
/// matches `target`. Returns the path on success.
pub fn scan_for_path(
	seed: &[u8; 64],
	ty: AddressType,
	target: &Address,
	max_addresses: usize,
) -> Result<Option<Vec<u32>>> {
	let target_spk = target.script_pubkey();
	let mut scanned = 0;
	// Walk accounts, then address index, with the receive/change chain (`change` in {0, 1}) varying
	// innermost so both chains are covered evenly within the address budget.
	'outer: for account in 0..u32::MAX {
		for index in 0..u32::MAX {
			for change in 0..2 {
				if scanned >= max_addresses {
					break 'outer;
				}
				scanned += 1;
				let path = bip44_path(ty, account, change, index);
				let pubkey = derive_compressed_pubkey(seed, &path)?;
				let hash160 = hash160_from_sha256(&pubkey_sha256(&pubkey));
				if script_pubkey_for(ty, &hash160) == target_spk {
					return Ok(Some(path));
				}
			}
		}
	}
	Ok(None)
}

#[cfg(test)]
mod tests {
	use bip39::Mnemonic;

	use super::*;

	fn test_seed() -> [u8; 64] {
		// A fixed mnemonic -> seed, so the test is deterministic and offline.
		let mnemonic = Mnemonic::parse_normalized(
			"abandon abandon abandon abandon abandon abandon abandon abandon \
			 abandon abandon abandon about",
		)
		.unwrap();
		mnemonic.to_seed("")
	}

	#[test]
	fn address_binding_round_trips_for_each_type() {
		let seed = test_seed();
		for ty in AddressType::ALL {
			let path = bip44_path(ty, 0, 0, 0);
			let pubkey = derive_compressed_pubkey(&seed, &path).unwrap();
			let address = address_for(ty, &pubkey);
			let sha256 = pubkey_sha256(&pubkey);
			let detected = bind_address_to_hash(&address, &sha256).unwrap();
			assert_eq!(detected, ty);
		}
	}

	#[test]
	fn binding_rejects_wrong_hash() {
		let seed = test_seed();
		let path = bip44_path(AddressType::P2pkh, 0, 0, 0);
		let pubkey = derive_compressed_pubkey(&seed, &path).unwrap();
		let address = address_for(AddressType::P2pkh, &pubkey);
		let mut wrong = pubkey_sha256(&pubkey);
		wrong[0] ^= 0xff;
		assert!(bind_address_to_hash(&address, &wrong).is_err());
	}

	#[test]
	fn scan_recovers_known_path() {
		let seed = test_seed();
		let ty = AddressType::P2wpkh;
		let path = bip44_path(ty, 0, 1, 7);
		let pubkey = derive_compressed_pubkey(&seed, &path).unwrap();
		let address = address_for(ty, &pubkey);
		let found = scan_for_path(&seed, ty, &address, 1000).unwrap();
		assert_eq!(found, Some(path));
	}

	#[test]
	fn path_string_round_trip() {
		let path = bip44_path(AddressType::P2pkh, 3, 1, 9);
		assert_eq!(format_path(&path), "m/44'/0'/3'/1/9");
		assert_eq!(parse_path("m/44'/0'/3'/1/9").unwrap(), path);
		assert_eq!(parse_path("44h/0h/3h/1/9").unwrap(), path);
	}
}
