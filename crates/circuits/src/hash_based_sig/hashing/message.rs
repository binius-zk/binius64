// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::chain_blake3::{circuit_blake3_th, ref_blake3_th};
use crate::fixed_byte_vec::ByteVec;

/// Tweak separator for message hashing.
pub const MESSAGE_TWEAK: u8 = 0x02;

/// Computes the message hash, returning its 32-byte digest as four 64-bit little-endian wires.
///
/// Evaluates the BLAKE3 tweakable hash with:
/// - domain (chaining value) `param || 0x02 || epoch`,
/// - data (absorbed block) `nonce || message`.
///
/// The codeword coordinates are read out of this digest.
/// Binding the epoch into the domain makes the encoding epoch-dependent, matching the reference
/// message hash over the parameter, epoch, randomness, and message (eprint 2025/055, Construction
/// 5/6).
///
/// # Arguments
///
/// * `builder` - Circuit builder.
/// * `domain_param_wires` - Per-signer parameter, eight bytes per wire (little-endian).
/// * `domain_param_len` - Parameter length in bytes; at most 23 so the domain fits the 32-byte cv.
/// * `epoch` - Epoch (leaf index) at which the message is signed.
/// * `nonce_wires` - Nonce wires, eight bytes per wire.
/// * `nonce_len` - Nonce length in bytes.
/// * `message_wires` - Message wires, eight bytes per wire.
/// * `message_len` - Message length in bytes.
///
/// # Returns
///
/// The 32-byte digest as four 64-bit little-endian wires.
#[allow(clippy::too_many_arguments)]
pub fn circuit_message_hash(
	builder: &CircuitBuilder,
	domain_param_wires: Vec<Wire>,
	domain_param_len: usize,
	epoch: Wire,
	nonce_wires: Vec<Wire>,
	nonce_len: usize,
	message_wires: Vec<Wire>,
	message_len: usize,
) -> [Wire; 4] {
	let domain = vec![
		ByteVec::new_const_len(builder, domain_param_wires, domain_param_len),
		ByteVec::new_const_len(builder, vec![builder.add_constant_64(MESSAGE_TWEAK as u64)], 1),
		ByteVec::new_const_len(builder, vec![epoch], 8),
	];
	let data = vec![
		ByteVec::new_const_len(builder, nonce_wires, nonce_len),
		ByteVec::new_const_len(builder, message_wires, message_len),
	];
	circuit_blake3_th(builder, &domain, &data)
}

/// Reference (out-of-circuit) message hash, matching `circuit_message_hash` exactly.
///
/// # Arguments
///
/// * `param` - Per-signer parameter bytes.
/// * `epoch` - Epoch (leaf index), encoded as eight little-endian bytes in the domain.
/// * `nonce` - Nonce bytes.
/// * `message` - Message bytes.
pub fn hash_message(param: &[u8], epoch: u64, nonce: &[u8], message: &[u8]) -> [u8; 32] {
	let mut domain = Vec::with_capacity(param.len() + 1 + 8);
	domain.extend_from_slice(param);
	domain.push(MESSAGE_TWEAK);
	domain.extend_from_slice(&epoch.to_le_bytes());

	let mut data = Vec::with_capacity(nonce.len() + message.len());
	data.extend_from_slice(nonce);
	data.extend_from_slice(message);

	ref_blake3_th(&domain, &data)
}

#[cfg(test)]
mod tests {
	use binius_core::{Word, verify::verify_constraints};
	use binius_frontend::util::pack_bytes_into_wires_le;
	use proptest::prelude::*;

	use super::*;

	// Single source of truth for the in-circuit message hash: build the gadget, populate the
	// inputs, and return the evaluated 32-byte digest.
	fn run_circuit(
		domain_param_bytes: &[u8],
		epoch: u64,
		nonce_bytes: &[u8],
		message_bytes: &[u8],
	) -> [u8; 32] {
		let b = CircuitBuilder::new();

		// One wire per eight parameter/nonce/message bytes.
		let param: Vec<Wire> = (0..domain_param_bytes.len().div_ceil(8))
			.map(|_| b.add_inout())
			.collect();
		let epoch_w = b.add_inout();
		let nonce: Vec<Wire> = (0..nonce_bytes.len().div_ceil(8))
			.map(|_| b.add_inout())
			.collect();
		let message: Vec<Wire> = (0..message_bytes.len().div_ceil(8))
			.map(|_| b.add_inout())
			.collect();

		let digest = circuit_message_hash(
			&b,
			param.clone(),
			domain_param_bytes.len(),
			epoch_w,
			nonce.clone(),
			nonce_bytes.len(),
			message.clone(),
			message_bytes.len(),
		);
		// Expose the digest so the evaluated value can be read back.
		let out: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
		for k in 0..4 {
			b.assert_eq("msg_digest", digest[k], out[k]);
		}

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		pack_bytes_into_wires_le(&mut w, &param, domain_param_bytes);
		w[epoch_w] = Word::from_u64(epoch);
		pack_bytes_into_wires_le(&mut w, &nonce, nonce_bytes);
		pack_bytes_into_wires_le(&mut w, &message, message_bytes);
		// The reference value lets the constraint check pass; the test asserts it matches.
		let reference = hash_message(domain_param_bytes, epoch, nonce_bytes, message_bytes);
		pack_bytes_into_wires_le(&mut w, &out, &reference);

		circuit.populate_wire_witness(&mut w).unwrap();
		verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
		reference
	}

	#[test]
	fn matches_reference_18_byte_param() {
		// Real-parameter shape: 18-byte parameter, 23-byte nonce, 32-byte message.
		let digest = run_circuit(b"test_param_18bytes", 42, &[7u8; 23], &[9u8; 32]);
		// The digest depends on every input, so it is not all-zero.
		assert_ne!(digest, [0u8; 32]);
	}

	proptest! {
		#[test]
		fn matches_reference_property_based(
			param_len in 1usize..=23,
			nonce_len in 0usize..=40,
			message_len in 1usize..=64,
			epoch in 0u64..=u64::MAX,
			seed in any::<u64>(),
		) {
			use rand::{Rng, SeedableRng, rngs::StdRng};

			// Random inputs of the sampled lengths.
			let mut rng = StdRng::seed_from_u64(seed);
			let mut param = vec![0u8; param_len];
			rng.fill_bytes(&mut param);
			let mut nonce = vec![0u8; nonce_len];
			rng.fill_bytes(&mut nonce);
			let mut message = vec![0u8; message_len];
			rng.fill_bytes(&mut message);

			// The constraint system is satisfiable only if the gadget equals the reference.
			run_circuit(&param, epoch, &nonce, &message);
		}
	}
}
