// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
use std::array;

use anyhow::Result;
use binius_circuits::sha256::{Sha256, sha256_fixed};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use sha2::Digest;

use super::utils::{self, HasherInstance, HasherMode, HasherParams};
use crate::ExampleCircuit;

pub struct Sha256Example {
	circuit: Sha256Circuit,
	mode: HasherMode,
}

/// Either the fixed-length (default) or variable-length SHA-256 circuit.
enum Sha256Circuit {
	/// Fixed-length gadget: message length is a compile-time constant.
	Fixed {
		message: Vec<Wire>,
		digest: [Wire; 8],
	},
	/// Variable-length gadget: message length is a runtime witness.
	Variable(Sha256),
}

impl ExampleCircuit for Sha256Example {
	type Params = HasherParams;
	type Instance = HasherInstance;

	fn build(params: HasherParams, builder: &mut CircuitBuilder) -> Result<Self> {
		let mode = utils::resolve_hasher_mode(&params, "SHA-256", true)?;

		let circuit = match mode {
			// Fixed-length (default): message length is a compile-time constant.
			HasherMode::Fixed { len_bytes } => {
				let n_words = len_bytes.div_ceil(4);
				let message: Vec<Wire> = (0..n_words).map(|_| builder.add_inout()).collect();
				let computed_digest = sha256_fixed(builder, &message, len_bytes);
				let digest: [Wire; 8] = array::from_fn(|_| builder.add_inout());
				for i in 0..8 {
					builder.assert_eq(format!("digest[{i}]"), computed_digest[i], digest[i]);
				}
				Sha256Circuit::Fixed { message, digest }
			}
			// Variable-length: message length is a runtime witness.
			HasherMode::Variable { max_len_bytes } => {
				let max_words = max_len_bytes.div_ceil(8);
				let len_bytes = builder.add_witness();
				let digest: [Wire; 4] = array::from_fn(|_| builder.add_inout());
				let message = (0..max_words).map(|_| builder.add_inout()).collect();
				Sha256Circuit::Variable(Sha256::new(builder, len_bytes, digest, message))
			}
		};

		Ok(Self { circuit, mode })
	}

	fn populate_witness(&self, instance: HasherInstance, w: &mut WitnessFiller) -> Result<()> {
		let message = utils::resolve_hasher_message(&self.mode, &instance)?;
		let digest = sha2::Sha256::digest(&message);

		match &self.circuit {
			Sha256Circuit::Fixed {
				message: message_wires,
				digest: digest_wires,
			} => {
				// Message: 32-bit big-endian words, 4 bytes per wire, high 32 bits zero.
				for (wire, word) in message_wires
					.iter()
					.zip(utils::pack_bytes_u32words(&message, true))
				{
					w[*wire] = word;
				}
				// Digest: 8 x 32-bit big-endian words.
				for (i, chunk) in digest.chunks(4).enumerate() {
					w[digest_wires[i]] = Word(u32::from_be_bytes(chunk.try_into().unwrap()) as u64);
				}
			}
			Sha256Circuit::Variable(gadget) => {
				gadget.populate_len_bytes(w, message.len());
				gadget.populate_message(w, &message);
				gadget.populate_digest(w, digest.into());
			}
		}

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		utils::hasher_param_summary(params)
	}
}
