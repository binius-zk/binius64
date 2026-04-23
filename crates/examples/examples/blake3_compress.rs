// Copyright 2025-2026 The Binius Developers

use anyhow::Result;
use binius_examples::{Cli, circuits::blake3_compress::Blake3CompressExample};

fn main() -> Result<()> {
	Cli::<Blake3CompressExample>::new("blake3_compress")
		.about("BLAKE3 compression benchmark (uses blake3_compress_2x under the hood)")
		.run()
}
