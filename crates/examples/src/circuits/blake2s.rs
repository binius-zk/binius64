// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_circuits::blake2s::Blake2s;
use binius_frontend::{CircuitBuilder, WitnessFiller};
use blake2::{Blake2s256, Digest};
use clap::Args;

use super::utils;
use crate::ExampleCircuit;

/// Blake2s circuit example demonstrating the Blake2s hash function implementation
pub struct Blake2sExample {
	blake2s_gadget: Blake2s,
}

/// Circuit parameters that affect structure (compile-time configuration)
#[derive(Debug, Clone, Args)]
pub struct Params {
	/// Maximum message length in bytes that the circuit can handle.
	#[arg(long)]
	pub max_bytes: Option<usize>,
}

/// Instance data for witness population (runtime values)
#[derive(Debug, Clone, Args)]
#[group(multiple = false)]
pub struct Instance {
	/// Length of the randomly generated message, in bytes (defaults to 1024).
	#[arg(long)]
	pub message_len: Option<usize>,

	/// UTF-8 string to hash (if not provided, random bytes are generated)
	#[arg(long)]
	pub message_string: Option<String>,
}

impl ExampleCircuit for Blake2sExample {
	type Params = Params;
	type Instance = Instance;

	fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
		let max_bytes = utils::determine_hash_max_bytes_from_args(params.max_bytes)?;

		// Create the Blake2s gadget with witness wires
		let blake2s_gadget = Blake2s::new_witness(builder, max_bytes);

		Ok(Self { blake2s_gadget })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller<'_>) -> Result<()> {
		// Step 1: Get raw message bytes
		let message = utils::generate_message_bytes(instance.message_string, instance.message_len);

		// Step 2: Compute digest using reference implementation
		let mut hasher = Blake2s256::new();
		hasher.update(&message);
		let digest: [u8; 32] = hasher.finalize().into();

		// Step 3: Populate witness values (Blake2s doesn't use len_bytes)
		self.blake2s_gadget.populate_message(w, &message);
		self.blake2s_gadget.populate_digest(w, &digest);

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!(
			"{}b",
			params
				.max_bytes
				.unwrap_or(utils::DEFAULT_HASH_MESSAGE_BYTES)
		))
	}
}
