// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_examples::{Cli, circuits::bip32::Bip32Example};

fn main() -> Result<()> {
	Cli::<Bip32Example>::new("bip32")
		.about("BIP32 HD compressed secp256k1 public key derivation from a seed and path")
		.run()
}
