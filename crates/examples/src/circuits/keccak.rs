// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
use std::array;

use anyhow::Result;
use binius_circuits::keccak::{Keccak256, N_WORDS_PER_DIGEST, fixed_length::keccak256};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use sha3::Digest;

use super::utils::{self, HasherInstance, HasherMode, HasherParams};
use crate::ExampleCircuit;

/// Keccak-256 hash circuit example
pub struct KeccakExample {
	circuit: KeccakCircuit,
	mode: HasherMode,
}

/// Either the fixed-length (default) or variable-length Keccak-256 circuit.
enum KeccakCircuit {
	/// Fixed-length gadget: message length is a compile-time constant.
	Fixed {
		message: Vec<Wire>,
		digest: [Wire; N_WORDS_PER_DIGEST],
	},
	/// Variable-length gadget: message length is a runtime witness.
	Variable(Keccak256),
}

impl ExampleCircuit for KeccakExample {
	type Params = HasherParams;
	type Instance = HasherInstance;

	fn build(params: HasherParams, builder: &mut CircuitBuilder) -> Result<Self> {
		let mode = utils::resolve_hasher_mode(&params, "Keccak-256", true)?;

		let circuit = match mode {
			// Fixed-length (default): message length is a compile-time constant.
			HasherMode::Fixed { len_bytes } => {
				let n_words = len_bytes.div_ceil(8);
				let message: Vec<Wire> = (0..n_words).map(|_| builder.add_inout()).collect();
				let computed_digest = keccak256(builder, &message, len_bytes);
				let digest: [Wire; N_WORDS_PER_DIGEST] = array::from_fn(|_| builder.add_inout());
				for i in 0..N_WORDS_PER_DIGEST {
					builder.assert_eq(format!("digest[{i}]"), computed_digest[i], digest[i]);
				}
				KeccakCircuit::Fixed { message, digest }
			}
			// Variable-length: message length is a runtime witness.
			HasherMode::Variable { max_len_bytes } => {
				let n_words = max_len_bytes.div_ceil(8);
				let len_bytes = builder.add_witness();
				let digest: [Wire; N_WORDS_PER_DIGEST] = array::from_fn(|_| builder.add_inout());
				let message = (0..n_words).map(|_| builder.add_inout()).collect();
				KeccakCircuit::Variable(Keccak256::new(builder, len_bytes, digest, message))
			}
		};

		Ok(Self { circuit, mode })
	}

	fn populate_witness(&self, instance: HasherInstance, w: &mut WitnessFiller) -> Result<()> {
		let message = utils::resolve_hasher_message(&self.mode, &instance)?;
		let mut hasher = sha3::Keccak256::new();
		hasher.update(&message);
		let digest: [u8; 32] = hasher.finalize().into();

		match &self.circuit {
			KeccakCircuit::Fixed {
				message: message_wires,
				digest: digest_wires,
			} => {
				// Message: 64-bit little-endian words, 8 bytes per wire.
				for (wire, word) in message_wires
					.iter()
					.zip(utils::pack_bytes_u64words(&message, false))
				{
					w[*wire] = word;
				}
				// Digest: 4 x 64-bit little-endian words.
				for (i, chunk) in digest.chunks(8).enumerate() {
					w[digest_wires[i]] = Word(u64::from_le_bytes(chunk.try_into().unwrap()));
				}
			}
			KeccakCircuit::Variable(gadget) => {
				gadget.populate_len_bytes(w, message.len());
				gadget.populate_message(w, &message);
				gadget.populate_digest(w, digest);
			}
		}

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		utils::hasher_param_summary(params)
	}
}
