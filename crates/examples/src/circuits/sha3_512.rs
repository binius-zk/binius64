// Copyright 2026 The Binius Developers
use std::array;

use anyhow::Result;
use binius_circuits::{
	fixed_byte_vec::ByteVec,
	sha3::{SHA3_512_DIGEST_WORDS, fixed_length::sha3_512, varlen::sha3_512_varlen},
};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use sha3::Digest;

use super::utils::{self, HasherInstance, HasherMode, HasherParams};
use crate::ExampleCircuit;

/// SHA3-512 hash circuit example.
pub struct Sha3_512Example {
	circuit: Sha3_512Circuit,
	mode: HasherMode,
}

/// Either the fixed-length (default) or variable-length SHA3-512 circuit.
enum Sha3_512Circuit {
	/// Fixed-length gadget: message length is a compile-time constant.
	Fixed {
		message: Vec<Wire>,
		digest: [Wire; SHA3_512_DIGEST_WORDS],
	},
	/// Variable-length gadget: message length is a runtime witness.
	Variable {
		message: ByteVec,
		digest: [Wire; SHA3_512_DIGEST_WORDS],
	},
}

impl ExampleCircuit for Sha3_512Example {
	type Params = HasherParams;
	type Instance = HasherInstance;

	fn build(params: HasherParams, builder: &mut CircuitBuilder) -> Result<Self> {
		let mode = utils::resolve_hasher_mode(&params, "SHA3-512", true)?;

		let circuit = match mode {
			// Fixed-length (default): message length is a compile-time constant.
			HasherMode::Fixed { len_bytes } => {
				let n_words = len_bytes.div_ceil(8);
				let message: Vec<Wire> = (0..n_words).map(|_| builder.add_inout()).collect();
				let computed_digest = sha3_512(builder, &message, len_bytes);
				let digest: [Wire; SHA3_512_DIGEST_WORDS] = array::from_fn(|_| builder.add_inout());
				for i in 0..SHA3_512_DIGEST_WORDS {
					builder.assert_eq(format!("digest[{i}]"), computed_digest[i], digest[i]);
				}
				Sha3_512Circuit::Fixed { message, digest }
			}
			// Variable-length: message length is a runtime witness.
			HasherMode::Variable { max_len_bytes } => {
				let n_words = max_len_bytes.div_ceil(8);
				let len_bytes = builder.add_witness();
				let data = (0..n_words).map(|_| builder.add_inout()).collect();
				let message = ByteVec::new(data, len_bytes);
				let digest: [Wire; SHA3_512_DIGEST_WORDS] = array::from_fn(|_| builder.add_inout());
				let computed_digest = sha3_512_varlen(builder, &message);
				for i in 0..SHA3_512_DIGEST_WORDS {
					builder.assert_eq(format!("digest[{i}]"), computed_digest[i], digest[i]);
				}
				Sha3_512Circuit::Variable { message, digest }
			}
		};

		Ok(Self { circuit, mode })
	}

	fn populate_witness(&self, instance: HasherInstance, w: &mut WitnessFiller) -> Result<()> {
		let message = utils::resolve_hasher_message(&self.mode, &instance)?;
		let digest: [u8; 64] = sha3::Sha3_512::digest(&message).into();

		match &self.circuit {
			Sha3_512Circuit::Fixed {
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
				// Digest: 8 x 64-bit little-endian words.
				for (i, chunk) in digest.chunks(8).enumerate() {
					w[digest_wires[i]] = Word(u64::from_le_bytes(chunk.try_into().unwrap()));
				}
			}
			Sha3_512Circuit::Variable {
				message: byte_vec,
				digest: digest_wires,
			} => {
				byte_vec.populate_data(w, &message);
				byte_vec.populate_len_bytes(w, message.len());
				for (i, chunk) in digest.chunks(8).enumerate() {
					w[digest_wires[i]] = Word(u64::from_le_bytes(chunk.try_into().unwrap()));
				}
			}
		}

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		utils::hasher_param_summary(params)
	}
}
