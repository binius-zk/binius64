// Copyright 2026 The Binius Developers
use anyhow::Result;
use binius_examples::{Cli, circuits::ec_msm::EcMsmExample};

fn main() -> Result<()> {
	Cli::<EcMsmExample>::new("ec_msm")
		.about("secp256k1 multi-scalar multiplication example (Straus fixed-window)")
		.run()
}
