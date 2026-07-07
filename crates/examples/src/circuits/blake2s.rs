// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_circuits::blake2s::Blake2s;
use binius_frontend::{CircuitBuilder, WitnessFiller};
use blake2::{Blake2s256, Digest};

use super::utils::{self, HasherInstance, HasherMode, HasherParams};
use crate::ExampleCircuit;

/// Blake2s circuit example demonstrating the Blake2s hash function implementation
pub struct Blake2sExample {
	blake2s_gadget: Blake2s,
	mode: HasherMode,
}

impl ExampleCircuit for Blake2sExample {
	type Params = HasherParams;
	type Instance = HasherInstance;

	fn build(params: HasherParams, builder: &mut CircuitBuilder) -> Result<Self> {
		// TODO: pass `supports_variable = true` and wire up a variable-length Blake2s gadget here
		// once one exists in binius-circuits. Until then `--max-message-len` is rejected.
		let mode = utils::resolve_hasher_mode(&params, "Blake2s", false)?;
		let HasherMode::Fixed { len_bytes } = mode else {
			unreachable!("Blake2s only supports the fixed-length gadget")
		};

		// Build the fixed-length Blake2s gadget for exactly this message length.
		let blake2s_gadget = Blake2s::new_witness(builder, len_bytes);

		Ok(Self {
			blake2s_gadget,
			mode,
		})
	}

	fn populate_witness(&self, instance: HasherInstance, w: &mut WitnessFiller) -> Result<()> {
		let message = utils::resolve_hasher_message(&self.mode, &instance)?;

		// Compute digest using reference implementation
		let mut hasher = Blake2s256::new();
		hasher.update(&message);
		let digest: [u8; 32] = hasher.finalize().into();

		// Populate witness values (Blake2s doesn't use len_bytes)
		self.blake2s_gadget.populate_message(w, &message);
		self.blake2s_gadget.populate_digest(w, &digest);

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		utils::hasher_param_summary(params)
	}
}
