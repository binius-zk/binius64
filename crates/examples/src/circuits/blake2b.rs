// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_circuits::blake2b::{Blake2bCircuit, blake2b};
use binius_frontend::{CircuitBuilder, WitnessFiller};

use super::utils::{self, HasherInstance, HasherMode, HasherParams};
use crate::ExampleCircuit;

/// Blake2b circuit example demonstrating the Blake2b hash function implementation
pub struct Blake2bExample {
	blake2b_circuit: Blake2bCircuit,
	mode: HasherMode,
}

impl ExampleCircuit for Blake2bExample {
	type Params = HasherParams;
	type Instance = HasherInstance;

	fn build(params: HasherParams, builder: &mut CircuitBuilder) -> Result<Self> {
		// TODO: pass `supports_variable = true` and wire up a variable-length Blake2b gadget here
		// once one exists in binius-circuits. Until then `--max-message-len` is rejected.
		let mode = utils::resolve_hasher_mode(&params, "Blake2b", false)?;
		let HasherMode::Fixed { len_bytes } = mode else {
			unreachable!("Blake2b only supports the fixed-length gadget")
		};

		let blake2b_circuit = Blake2bCircuit::new_with_length(builder, len_bytes);

		Ok(Self {
			blake2b_circuit,
			mode,
		})
	}

	fn populate_witness(&self, instance: HasherInstance, w: &mut WitnessFiller) -> Result<()> {
		let message = utils::resolve_hasher_message(&self.mode, &instance)?;

		// Compute digest using reference implementation
		let expected_digest_vec = blake2b(&message, 64);
		let mut expected_digest = [0u8; 64];
		expected_digest.copy_from_slice(&expected_digest_vec);

		// Populate witness values (Blake2b doesn't use len_bytes)
		self.blake2b_circuit.populate_message(w, &message);
		self.blake2b_circuit.populate_digest(w, &expected_digest);

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		utils::hasher_param_summary(params)
	}
}
