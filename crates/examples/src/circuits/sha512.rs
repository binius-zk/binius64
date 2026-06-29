// Copyright 2025 Irreducible Inc.
use std::array;

use anyhow::Result;
use binius_circuits::{fixed_byte_vec::ByteVec, sha512::sha512_varlen};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use clap::Args;
use sha2::Digest;

use super::utils;
use crate::ExampleCircuit;

pub struct Sha512Example {
	message: ByteVec,
	digest: [Wire; 8],
}

#[derive(Args, Debug, Clone)]
pub struct Params {
	/// Maximum message length in bytes that the circuit can handle.
	#[arg(long)]
	pub max_len_bytes: Option<usize>,

	/// Build circuit for exact message length (makes length a compile-time constant instead of
	/// runtime witness).
	#[arg(long, default_value_t = false)]
	pub exact_len: bool,
}

#[derive(Args, Debug, Clone)]
#[group(multiple = false)]
pub struct Instance {
	/// Length of the randomly generated message, in bytes (defaults to 1024).
	#[arg(long)]
	pub message_len: Option<usize>,

	/// UTF-8 string to hash (if not provided, random bytes are generated)
	#[arg(long)]
	pub message_string: Option<String>,
}

impl ExampleCircuit for Sha512Example {
	type Params = Params;
	type Instance = Instance;

	fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
		let max_len_bytes = utils::determine_hash_max_bytes_from_args(params.max_len_bytes)?;
		let max_len_words = max_len_bytes.div_ceil(8);

		let len_bytes = if params.exact_len {
			builder.add_constant_64(max_len_bytes as u64)
		} else {
			builder.add_inout()
		};
		let data: Vec<Wire> = (0..max_len_words).map(|_| builder.add_inout()).collect();
		let message = ByteVec::new(data, len_bytes);

		let digest: [Wire; 8] = array::from_fn(|_| builder.add_inout());
		let computed_digest = sha512_varlen(builder, &message);
		for i in 0..8 {
			builder.assert_eq(format!("digest[{i}]"), computed_digest[i], digest[i]);
		}

		Ok(Self { message, digest })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller<'_>) -> Result<()> {
		// Step 1: Get raw message bytes
		let message = utils::generate_message_bytes(instance.message_string, instance.message_len);

		// Step 2: Compute digest using reference implementation
		let digest = sha2::Sha512::digest(&message);

		// Step 3: Populate inout values
		self.message.populate_data(w, &message);
		self.message.populate_len_bytes(w, message.len());
		for (i, chunk) in digest.chunks(8).enumerate() {
			w[self.digest[i]] = Word(u64::from_be_bytes(chunk.try_into().unwrap()));
		}

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		let base = format!(
			"{}b",
			params
				.max_len_bytes
				.unwrap_or(utils::DEFAULT_HASH_MESSAGE_BYTES)
		);
		if params.exact_len {
			Some(format!("{}-exact", base))
		} else {
			Some(base)
		}
	}
}
