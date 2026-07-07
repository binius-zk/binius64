// Copyright 2026 The Binius Developers
use std::array;

use anyhow::{Result, ensure};
use binius_circuits::blake3::{CHUNK_BYTES, blake3_fixed};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

use super::utils::{self, HasherInstance, HasherMode, HasherParams};
use crate::ExampleCircuit;

/// BLAKE3 circuit example using the fixed-length (single-chunk) hasher gadget.
pub struct Blake3Example {
	message: Vec<Wire>,
	digest: [Wire; 8],
	mode: HasherMode,
}

impl ExampleCircuit for Blake3Example {
	type Params = HasherParams;
	type Instance = HasherInstance;

	fn build(params: HasherParams, builder: &mut CircuitBuilder) -> Result<Self> {
		// TODO: pass `supports_variable = true` and wire up a variable-length BLAKE3 gadget here
		// once one exists in binius-circuits. Until then `--max-message-len` is rejected.
		let mode = utils::resolve_hasher_mode(&params, "BLAKE3", false)?;
		let HasherMode::Fixed { len_bytes } = mode else {
			unreachable!("BLAKE3 only supports the fixed-length gadget")
		};

		// blake3_fixed is restricted to single-chunk inputs. Multi-chunk hashing needs BLAKE3's
		// tree construction, which the gadget does not yet support.
		ensure!(
			len_bytes <= CHUNK_BYTES,
			"BLAKE3 example is limited to single-chunk messages (<= {CHUNK_BYTES} bytes), got {len_bytes}"
		);

		let n_words = len_bytes.div_ceil(4);
		let message: Vec<Wire> = (0..n_words).map(|_| builder.add_inout()).collect();
		let computed_digest = blake3_fixed(builder, &message, len_bytes);
		let digest: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		for i in 0..8 {
			builder.assert_eq(format!("digest[{i}]"), computed_digest[i], digest[i]);
		}

		Ok(Self {
			message,
			digest,
			mode,
		})
	}

	fn populate_witness(&self, instance: HasherInstance, w: &mut WitnessFiller) -> Result<()> {
		let message_bytes = utils::resolve_hasher_message(&self.mode, &instance)?;

		// Message: 32-bit little-endian words, 4 bytes per wire, high 32 bits zero.
		for (wire, word) in self
			.message
			.iter()
			.zip(utils::pack_bytes_u32words(&message_bytes, false))
		{
			w[*wire] = word;
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
