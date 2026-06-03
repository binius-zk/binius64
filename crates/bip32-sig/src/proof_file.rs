// Copyright 2026 The Binius Developers
//! On-disk proof container.
//!
//! A proof file is simply the 32-byte SHA-256 hash of the compressed public key (the circuit's
//! public input, and the value that binds the proof to a Bitcoin address) followed by the raw
//! proof bytes. No length prefix is needed: the hash is fixed width and the proof is the remainder.

use std::{fs, path::Path};

use anyhow::{Result, bail};

/// Length of the SHA-256 public-key hash stored at the front of a proof file.
pub const PUBKEY_HASH_LEN: usize = 32;

/// Parsed contents of a proof file.
pub struct ProofFile {
	/// SHA-256 of the 33-byte compressed public key — the circuit's public input.
	pub pubkey_sha256: [u8; PUBKEY_HASH_LEN],
	/// The serialized signature-of-knowledge proof.
	pub proof: Vec<u8>,
}

impl ProofFile {
	/// Serialize to the on-disk byte layout (`hash || proof`).
	pub fn to_bytes(&self) -> Vec<u8> {
		let mut bytes = Vec::with_capacity(PUBKEY_HASH_LEN + self.proof.len());
		bytes.extend_from_slice(&self.pubkey_sha256);
		bytes.extend_from_slice(&self.proof);
		bytes
	}

	/// Parse the on-disk byte layout.
	pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
		if bytes.len() < PUBKEY_HASH_LEN {
			bail!("proof file too short: {} bytes (need at least {PUBKEY_HASH_LEN})", bytes.len());
		}
		let (hash, proof) = bytes.split_at(PUBKEY_HASH_LEN);
		Ok(Self {
			pubkey_sha256: hash.try_into().expect("split at PUBKEY_HASH_LEN"),
			proof: proof.to_vec(),
		})
	}

	/// Write the proof file to `path`.
	pub fn write(&self, path: &Path) -> Result<()> {
		fs::write(path, self.to_bytes())
			.map_err(|e| anyhow::anyhow!("failed to write proof file '{}': {e}", path.display()))
	}

	/// Read and parse a proof file from `path`.
	pub fn read(path: &Path) -> Result<Self> {
		let bytes = fs::read(path)
			.map_err(|e| anyhow::anyhow!("failed to read proof file '{}': {e}", path.display()))?;
		Self::from_bytes(&bytes)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn round_trip() {
		let file = ProofFile {
			pubkey_sha256: [7u8; 32],
			proof: vec![1, 2, 3, 4, 5],
		};
		let parsed = ProofFile::from_bytes(&file.to_bytes()).unwrap();
		assert_eq!(parsed.pubkey_sha256, file.pubkey_sha256);
		assert_eq!(parsed.proof, file.proof);
	}

	#[test]
	fn rejects_short_input() {
		assert!(ProofFile::from_bytes(&[0u8; 16]).is_err());
	}
}
