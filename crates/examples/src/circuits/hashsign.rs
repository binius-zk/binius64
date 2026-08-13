// Copyright 2026 The Binius Developers
//! XMSS multi-signature verification.
//!
//! `n` signers with independent trees sign one message at one epoch, and the circuit verifies
//! every signature. The scheme's parameters are fixed, so the signer count is the only dial.

use anyhow::Result;
use binius_circuits::hash_based_sig::{
	MESSAGE_LEN, Message,
	aggregate::{MultiSigWires, circuit_xmss_multisig},
	xmss::generate_signature,
};
use binius_frontend::{CircuitBuilder, WitnessFiller};
use clap::Args;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::ExampleCircuit;

/// Fixed seed, so a benchmark run is comparable with the one before it.
const SEED: u64 = 42;

pub struct HashBasedSigExample {
	num_signers: usize,
	wires: MultiSigWires,
}

#[derive(Args, Debug, Clone)]
pub struct Params {
	/// Number of signers in the multi-signature
	#[arg(short = 'n', long, default_value_t = 3)]
	pub num_signers: usize,
}

#[derive(Args, Debug, Clone)]
pub struct Instance {}

impl ExampleCircuit for HashBasedSigExample {
	type Params = Params;
	type Instance = Instance;

	fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
		if params.num_signers == 0 {
			anyhow::bail!("num_signers must be positive");
		}

		let wires = MultiSigWires::new(builder, params.num_signers);
		circuit_xmss_multisig(builder, &wires);

		Ok(Self {
			num_signers: params.num_signers,
			wires,
		})
	}

	fn populate_witness(&self, _instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(SEED);

		let mut message: Message = [0u8; MESSAGE_LEN];
		rng.fill_bytes(&mut message);
		let mut epoch_bytes = [0u8; 4];
		rng.fill_bytes(&mut epoch_bytes);
		let epoch = u32::from_le_bytes(epoch_bytes);

		// Each signer has its own tree, so each generates its own key alongside its signature.
		let signatures = (0..self.num_signers)
			.map(|_| generate_signature(&mut rng, &message, epoch))
			.collect::<Vec<_>>();

		self.wires.populate(w, &message, epoch, &signatures);
		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!("{}s", params.num_signers))
	}
}
