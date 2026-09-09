// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The demo circuits, one subcommand each.

use anyhow::Result;
use binius_examples::{Cli, circuits};

fn main() -> Result<()> {
	Cli::new("binius-examples")
		.circuit::<circuits::bip32::Bip32Example>(
			"bip32",
			"BIP32 HD compressed secp256k1 public key derivation from a seed and path",
		)
		.circuit::<circuits::bitcoin_header_chain::BitcoinHeaderChainExample>(
			"bitcoin_headers",
			"Bitcoin Header Chain Example",
		)
		.circuit::<circuits::bitcoin_p2pkh::BitcoinP2PKHExample>(
			"bitcoin_p2pkh",
			"Bitcoin P2PKH address validation example - proves knowledge of private key without \
			 revealing it",
		)
		.circuit::<circuits::blake2b::Blake2bExample>(
			"blake2b",
			"Blake2b hash function circuit example",
		)
		.circuit::<circuits::blake2s::Blake2sExample>(
			"blake2s",
			"Blake2s hash function circuit example",
		)
		.circuit::<circuits::blake3::Blake3Example>("blake3", "BLAKE3 hash example")
		.circuit::<circuits::blake3_compress::Blake3CompressExample>(
			"blake3_compress",
			"BLAKE3 compression benchmark (uses blake3_compress_2x under the hood)",
		)
		.circuit::<circuits::ec_msm::EcMsmExample>(
			"ec_msm",
			"secp256k1 multi-scalar multiplication example (Straus fixed-window)",
		)
		.circuit::<circuits::ethsign::EthSignExample>("ethsign", "Ethereum-style signing example")
		.circuit::<circuits::hashsign::HashBasedSigExample>(
			"hashsign",
			"Hash-based multi-signature (XMSS) verification example",
		)
		.circuit::<circuits::keccak::KeccakExample>(
			"keccak",
			"Keccak-256 hash function circuit example",
		)
		.circuit::<circuits::sha256::Sha256Example>("sha256", "SHA256 compression function example")
		.circuit::<circuits::sha3::Sha3Example>("sha3", "SHA3-256 hash function circuit example")
		.circuit::<circuits::sha3_512::Sha3_512Example>(
			"sha3_512",
			"SHA3-512 hash function circuit example",
		)
		.circuit::<circuits::sha512::Sha512Example>("sha512", "SHA512 compression function example")
		.circuit::<circuits::zklogin::ZkLoginExample>(
			"zklogin",
			"Circuit verifying knowledge of a valid OpenID Connect login",
		)
		.run()
}
