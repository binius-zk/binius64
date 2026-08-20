// Copyright 2026 The Binius Developers
use anyhow::Result;
use binius_examples::{Cli, circuits::sha3_512::Sha3_512Example};

fn main() -> Result<()> {
	Cli::<Sha3_512Example>::new("sha3_512")
		.about("SHA3-512 hash function circuit example")
		.run()
}
