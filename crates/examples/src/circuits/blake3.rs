// Copyright 2026 The Binius Developers
use std::array;

use anyhow::Result;
use binius_circuits::blake3::{blake3_fixed, blake3_variable};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

use super::utils::{self, HasherInstance, HasherMode, HasherParams};
use crate::ExampleCircuit;

/// BLAKE3 circuit example, selecting the fixed- or variable-length hasher gadget from the params.
pub struct Blake3Example {
	message: Vec<Wire>,
	/// Length wire, present only in variable-length mode.
	len_bytes: Option<Wire>,
	digest: [Wire; 8],
	mode: HasherMode,
}

impl ExampleCircuit for Blake3Example {
	type Params = HasherParams;
	type Instance = HasherInstance;

	fn build(params: HasherParams, builder: &mut CircuitBuilder) -> Result<Self> {
		let mode = utils::resolve_hasher_mode(&params, "BLAKE3", true)?;

		let (message, len_bytes, computed_digest) = match mode {
			HasherMode::Fixed { len_bytes } => {
				let n_words = len_bytes.div_ceil(4);
				let message: Vec<Wire> = (0..n_words).map(|_| builder.add_inout()).collect();
				let digest = blake3_fixed(builder, &message, len_bytes);
				(message, None, digest)
			}
			HasherMode::Variable { max_len_bytes } => {
				let n_words = max_len_bytes.div_ceil(4);
				let message: Vec<Wire> = (0..n_words).map(|_| builder.add_inout()).collect();
				let len_wire = builder.add_inout();
				let digest = blake3_variable(builder, &message, len_wire, max_len_bytes);
				(message, Some(len_wire), digest)
			}
		};

		let digest: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq(format!("digest[{i}]"), computed_digest[i], digest[i]);
		}

		Ok(Self {
			message,
			len_bytes,
			digest,
			mode,
		})
	}

	fn populate_witness(&self, instance: HasherInstance, w: &mut WitnessFiller) -> Result<()> {
		let message_bytes = utils::resolve_hasher_message(&self.mode, &instance)?;

		// Message: 32-bit little-endian words, 4 bytes per wire, high 32 bits zero. In
		// variable-length mode the message wires past the actual length keep their default zero
		// value; the gadget masks them regardless.
		for (wire, word) in self
			.message
			.iter()
			.zip(utils::pack_bytes_u32words(&message_bytes, false))
		{
			w[*wire] = word;
		}

		if let Some(len_wire) = self.len_bytes {
			w[len_wire] = Word(message_bytes.len() as u64);
		}

		// Digest: 8 x 32-bit little-endian words.
		let expected = blake3::hash(&message_bytes);
		for (i, chunk) in expected.as_bytes().chunks(4).enumerate() {
			w[self.digest[i]] = Word(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
		}

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		utils::hasher_param_summary(params)
	}
}
