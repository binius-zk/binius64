// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use crate::{concat::concat, fixed_byte_vec::ByteVec, keccak::Keccak256};

/// Verify a tweaked Keccak-256 circuit with custom terms.
///
/// This function provides the common setup for both message and chain tweaking,
/// which both follow the pattern: `Keccak256(domain_param || tweak_byte || additional_data)`
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `domain_param_wires` - The cryptographic domain parameter wires
/// * `domain_param_len` - The actual domain parameter length in bytes
/// * `tweak_byte` - The tweak byte value (MESSAGE_TWEAK or CHAIN_TWEAK)
/// * `additional_terms` - Additional concatenation terms after param and tweak
/// * `digest` - Output digest wires
///
/// # Returns
/// A `Keccak` instance that computes the tweaked hash
pub fn circuit_tweaked_keccak(
	builder: &CircuitBuilder,
	domain_param_wires: Vec<Wire>,
	domain_param_len: usize,
	tweak_byte: u8,
	additional_terms: Vec<ByteVec>,
	digest: [Wire; 4],
) -> Keccak256 {
	let mut terms = Vec::with_capacity(2 + additional_terms.len());
	terms.push(ByteVec::new(domain_param_wires, builder.add_constant_64(domain_param_len as u64)));
	terms.push(ByteVec::new(
		vec![builder.add_constant_64(tweak_byte as u64)],
		builder.add_constant_64(1),
	));
	terms.extend(additional_terms);

	let message = concat(builder, &terms);

	Keccak256::new(builder, message.len_bytes, digest, message.data)
}
