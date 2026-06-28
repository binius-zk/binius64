// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::chain_blake3::{circuit_blake3_th, ref_blake3_th};
use crate::fixed_byte_vec::ByteVec;

/// Tweak separator for the one-time public-key (Merkle leaf) hash.
///
/// - Chain steps use `0x00`, internal tree nodes `0x01`, message hashing `0x02`.
/// - The leaf uses a distinct `0x03`.
/// - This keeps the leaf in a different hash domain from any internal tree node.
pub const PUBLIC_KEY_TWEAK: u8 = 0x03;

/// Computes the one-time public-key (Merkle leaf) hash, returning its 32-byte digest.
///
/// Evaluates the BLAKE3 tweakable hash with:
/// - domain (chaining value) `param || 0x03 || epoch`,
/// - data (absorbed blocks) the concatenation of the Winternitz chain ends.
///
/// The leaf is the larger-than-one-block case: the chain ends are absorbed two per compression.
/// Mixing the epoch into the domain binds the leaf to its position in the tree.
///
/// # Arguments
///
/// * `builder` - Circuit builder.
/// * `domain_param_wires` - Per-signer parameter, eight bytes per wire.
/// * `domain_param_len` - Parameter length in bytes; at most 23 so the domain fits the 32-byte cv.
/// * `epoch` - Epoch (leaf index) this public key sits at.
/// * `pk_hashes` - Chain-end hashes, 32 bytes each as four 64-bit little-endian wires.
///
/// # Returns
///
/// The 32-byte leaf digest as four 64-bit little-endian wires.
pub fn circuit_public_key_hash(
	builder: &CircuitBuilder,
	domain_param_wires: Vec<Wire>,
	domain_param_len: usize,
	epoch: Wire,
	pk_hashes: &[[Wire; 4]],
) -> [Wire; 4] {
	let domain = vec![
		ByteVec::new_const_len(builder, domain_param_wires, domain_param_len),
		ByteVec::new_const_len(builder, vec![builder.add_constant_64(PUBLIC_KEY_TWEAK as u64)], 1),
		ByteVec::new_const_len(builder, vec![epoch], 8),
	];
	let data: Vec<ByteVec> = pk_hashes
		.iter()
		.map(|pk| ByteVec::new_const_len(builder, pk.to_vec(), 32))
		.collect();
	circuit_blake3_th(builder, &domain, &data)
}

/// Reference (out-of-circuit) leaf hash, matching `circuit_public_key_hash` exactly.
///
/// # Arguments
///
/// * `param` - Per-signer parameter bytes.
/// * `epoch` - Epoch (leaf index), encoded as eight little-endian bytes in the domain.
/// * `pk_hashes` - The 32-byte chain-end hashes.
pub fn hash_public_key(param: &[u8], epoch: u64, pk_hashes: &[[u8; 32]]) -> [u8; 32] {
	let mut domain = Vec::with_capacity(param.len() + 1 + 8);
	domain.extend_from_slice(param);
	domain.push(PUBLIC_KEY_TWEAK);
	domain.extend_from_slice(&epoch.to_le_bytes());

	let mut data = Vec::with_capacity(pk_hashes.len() * 32);
	for pk in pk_hashes {
		data.extend_from_slice(pk);
	}

	ref_blake3_th(&domain, &data)
}

#[cfg(test)]
mod tests {
	use binius_core::{Word, verify::verify_constraints};
	use binius_frontend::util::pack_bytes_into_wires_le;
	use proptest::prelude::*;

	use super::*;

	// Build the gadget, populate the inputs, and return the evaluated 32-byte leaf digest.
	fn run_circuit(param_bytes: &[u8], epoch: u64, pk_hashes: &[[u8; 32]]) -> [u8; 32] {
		let b = CircuitBuilder::new();
		let param: Vec<Wire> = (0..param_bytes.len().div_ceil(8))
			.map(|_| b.add_inout())
			.collect();
		let epoch_w = b.add_inout();
		let pk_wires: Vec<[Wire; 4]> = (0..pk_hashes.len())
			.map(|_| std::array::from_fn(|_| b.add_inout()))
			.collect();

		let digest =
			circuit_public_key_hash(&b, param.clone(), param_bytes.len(), epoch_w, &pk_wires);
		let out: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
		for k in 0..4 {
			b.assert_eq("leaf_digest", digest[k], out[k]);
		}

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		pack_bytes_into_wires_le(&mut w, &param, param_bytes);
		w[epoch_w] = Word::from_u64(epoch);
		for (wires, bytes) in pk_wires.iter().zip(pk_hashes) {
			pack_bytes_into_wires_le(&mut w, wires, bytes);
		}
		let reference = hash_public_key(param_bytes, epoch, pk_hashes);
		pack_bytes_into_wires_le(&mut w, &out, &reference);

		circuit.populate_wire_witness(&mut w).unwrap();
		verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
		reference
	}

	#[test]
	fn matches_reference_single_chain_end() {
		// Smallest case: one chain end (one absorbed block).
		let digest = run_circuit(b"test_param_18bytes", 99, &[[7u8; 32]]);
		assert_ne!(digest, [0u8; 32]);
	}

	#[test]
	fn matches_reference_many_chain_ends() {
		// Winternitz with 72 chains: 72 * 32 bytes absorbed over many blocks.
		let pk_hashes: Vec<[u8; 32]> = (0..72u8).map(|i| [i; 32]).collect();
		let digest = run_circuit(b"test_param_18bytes", 0, &pk_hashes);
		assert_ne!(digest, [0u8; 32]);
	}

	proptest! {
		#[test]
		fn matches_reference_property_based(
			param_len in 1usize..=23,
			num_hashes in 1usize..=40,
			epoch in 0u64..=u64::MAX,
			seed in any::<u64>(),
		) {
			use rand::{Rng, SeedableRng, rngs::StdRng};

			// Random parameter and chain ends of the sampled counts.
			let mut rng = StdRng::seed_from_u64(seed);
			let mut param = vec![0u8; param_len];
			rng.fill_bytes(&mut param);
			let pk_hashes: Vec<[u8; 32]> = (0..num_hashes)
				.map(|_| {
					let mut h = [0u8; 32];
					rng.fill_bytes(&mut h);
					h
				})
				.collect();

			run_circuit(&param, epoch, &pk_hashes);
		}
	}
}
