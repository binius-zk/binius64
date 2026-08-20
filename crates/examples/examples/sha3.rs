// Copyright 2026 The Binius Developers
use anyhow::Result;
use binius_examples::{Cli, circuits::sha3::Sha3Example};

fn main() -> Result<()> {
	Cli::<Sha3Example>::new("sha3")
		.about("SHA3-256 hash function circuit example")
		.run()
}
