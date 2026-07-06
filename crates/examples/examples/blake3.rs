// Copyright 2026 The Binius Developers
use anyhow::Result;
use binius_examples::{Cli, circuits::blake3::Blake3Example};

fn main() -> Result<()> {
	Cli::<Blake3Example>::new("blake3")
		.about("BLAKE3 hash example")
		.run()
}
