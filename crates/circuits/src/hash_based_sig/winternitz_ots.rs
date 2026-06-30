// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::{
	codeword::codeword,
	hashing::{circuit_chain_hash_blake3, circuit_chain_step_2x_blake3, circuit_message_hash},
};

/// Nonce length in bytes for the message-hash randomness.
///
/// - `nonce || message` is 32 + 32 = 64 bytes, exactly one BLAKE3 block.
/// - The message hash therefore stays a single compression.
/// - 256 bits clears the randomness-space requirement of eprint 2025/055 (Parameter Requirement 2).
/// - At lifetime 2^32 with up to ~10^5 grinding retries the quantum bound needs about 201 bits.
/// - 256 bits leaves margin at no extra hashing cost.
pub const NONCE_LENGTH_BYTES: usize = 32;

/// BLAKE3-256 output size in bytes
const MESSAGE_LENGTH_BYTES: usize = 32;

/// Number of 64-bit wires needed to represent a 32-byte hash (32 bytes / 8 bytes per wire)
const HASH_WIRES_COUNT: usize = 4;

/// Number of 64-bit wires holding the nonce (8 bytes per wire).
pub const NONCE_WIRES_COUNT: usize = NONCE_LENGTH_BYTES.div_ceil(8);

/// Verifies a Winternitz one-time signature by walking each hash chain end to end.
///
/// # Definitions
/// - `dimension`: number of chains.
/// - `chain_len`: length of each chain, `2^{coordinate_resolution_bits}`.
/// - `x_i`: the i-th codeword coordinate, with `0 <= x_i <= chain_len - 1`.
/// - The coordinates sum to the target sum.
/// - `r_i = (chain_len - 1) - x_i`: steps from the signature value `sig_i` to the public key
///   `pk_i`.
///
/// # Algorithm
///
/// Each chain walks forward from its signature value to its public-key end:
///
/// ```text
/// cur := sig_i                          // value at chain position x_i
/// for p in 1..=chain_len-1:
///     next := Th(cur, chain = i, position = p)   // one BLAKE3 compression
///     cur  := select(x_i < p, next, cur)         // advance only past position x_i
/// assert cur == pk_i
/// ```
///
/// - The chain index and position are circuit constants, so each step's tweak is fixed.
/// - The running value is threaded from the signature value to the public-key check.
/// - So the start and the end are bound to the same chain of hashes, with no existence search.
/// - When `x_i = chain_len - 1` the running value never advances.
/// - The final check is then `pk_i == sig_i`, as required.
/// - This implements `pk_i = Th^{r_i}(sig_i)` (eprint 2025/055, Construction 3, step 6).
///
/// # Performance
///
/// - Chains are processed two at a time as the two lanes of one BLAKE3 compression.
/// - Total work is `dimension * (chain_len - 1)` chain steps plus one message hash.
/// - Each step adds one unsigned comparison and four 64-bit selects beyond the compression.
///
/// # Arguments
/// - `domain_param`: per-signer public parameter, prefixed into every hash.
/// - `epoch`: leaf index of the signature, bound into the message and chain tweaks.
/// - `message`: the 32-byte message being signed, as four 64-bit words.
/// - `nonce`: nonce feeding the tweaked message hash.
/// - `signature_hashes`: starting value of each chain.
/// - `public_key_hashes`: expected end value of each chain.
/// - `spec`: Winternitz parameters.
///
/// All hashing is BLAKE3, whose digests are derived from the inputs, so this emits constraints
/// only and returns nothing.
#[allow(clippy::too_many_arguments)]
pub fn circuit_winternitz_ots(
	builder: &CircuitBuilder,
	domain_param: &[Wire],
	epoch: Wire,
	message: &[Wire],
	nonce: &[Wire],
	signature_hashes: &[[Wire; HASH_WIRES_COUNT]],
	public_key_hashes: &[[Wire; HASH_WIRES_COUNT]],
	spec: &WinternitzSpec,
) {
	assert!(
		spec.domain_param_len <= domain_param.len() * 8,
		"domain_param wires must have capacity for {} bytes, but only has capacity for {} bytes",
		spec.domain_param_len,
		domain_param.len() * 8
	);
	assert_eq!(
		message.len(),
		HASH_WIRES_COUNT,
		"message must be 32 bytes as {} wires",
		HASH_WIRES_COUNT
	);

	// Step 1: compute the tweaked message hash `param || 0x02 || epoch || nonce || message`.
	// The digest is derived from the inputs, so nothing here needs a witness value.
	let message_hash_output = circuit_message_hash(
		builder,
		domain_param.to_vec(),
		spec.domain_param_len,
		epoch,
		nonce.to_vec(),
		NONCE_LENGTH_BYTES,
		message.to_vec(),
		MESSAGE_LENGTH_BYTES,
	);

	// Step 2: Extract codeword coordinates x_i from the message hash.
	let message_hash_bytes = spec.message_hash_len;
	let message_hash_wires_needed = message_hash_bytes.div_ceil(8);
	let message_hash_for_codeword = &message_hash_output[..message_hash_wires_needed];

	let coordinates = codeword(
		builder,
		spec.dimension(),
		spec.coordinate_resolution_bits,
		spec.target_sum,
		message_hash_for_codeword,
	);

	assert_eq!(coordinates.len(), spec.dimension(), "Codeword dimension mismatch");
	assert_eq!(signature_hashes.len(), spec.dimension(), "Signature hashes count mismatch");
	assert_eq!(public_key_hashes.len(), spec.dimension(), "Public key hashes count mismatch");

	// Step 3: Verify every chain by walking it forward from sig_i to pk_i with BLAKE3 steps.
	// chain_index and position are u8 in the compact BLAKE3 tweak, so both must fit a byte.
	let chain_len = spec.chain_len();
	assert!(
		spec.dimension() <= 256,
		"BLAKE3 chain tweak encodes chain_index in one byte; dimension {} exceeds 256",
		spec.dimension()
	);
	assert!(
		chain_len <= 256,
		"BLAKE3 chain tweak encodes position in one byte; chain_len {chain_len} exceeds 256"
	);

	// Precompute the position constants `1..chain_len` once; reused across all chains.
	let position_consts: Vec<Wire> = (0..chain_len)
		.map(|p| builder.add_constant_64(p as u64))
		.collect();

	// Selection for one chain step, applied per 64-bit limb.
	// Below position x_c: keep the current value (still the signature value).
	// At position x_c and beyond: take the freshly hashed value, so the chain ends at pk_c.
	let advance = |builder: &CircuitBuilder,
	               coord: Wire,
	               p: usize,
	               next: [Wire; HASH_WIRES_COUNT],
	               cur: [Wire; HASH_WIRES_COUNT]|
	 -> [Wire; HASH_WIRES_COUNT] {
		let active = builder.icmp_ult(coord, position_consts[p]);
		std::array::from_fn(|limb| builder.select(active, next[limb], cur[limb]))
	};

	// Walk two independent chains in lockstep as the two lanes of one BLAKE3 compression.
	let dim = spec.dimension();
	for cc in (0..dim - 1).step_by(2) {
		let (c0, c1) = (cc, cc + 1);
		let mut cur0 = signature_hashes[c0];
		let mut cur1 = signature_hashes[c1];

		for position in 1..chain_len {
			let (next0, next1) = circuit_chain_step_2x_blake3(
				builder,
				domain_param,
				spec.domain_param_len,
				epoch,
				cur0,
				cur1,
				c0 as u8,
				c1 as u8,
				position as u8,
			);
			cur0 = advance(builder, coordinates[c0], position, next0, cur0);
			cur1 = advance(builder, coordinates[c1], position, next1, cur1);
		}

		builder.assert_eq_v(format!("wots_chain_end[{c0}]"), cur0, public_key_hashes[c0]);
		builder.assert_eq_v(format!("wots_chain_end[{c1}]"), cur1, public_key_hashes[c1]);
	}

	// Odd dimension: the final chain has no partner, so run it as a single compression.
	if dim % 2 == 1 {
		let c = dim - 1;
		let mut cur = signature_hashes[c];
		for position in 1..chain_len {
			let next = circuit_chain_hash_blake3(
				builder,
				domain_param,
				spec.domain_param_len,
				epoch,
				cur,
				c as u8,
				position as u8,
			);
			cur = advance(builder, coordinates[c], position, next, cur);
		}
		builder.assert_eq_v(format!("wots_chain_end[{c}]"), cur, public_key_hashes[c]);
	}
}

/// Specification for Winternitz OTS parameters
///
/// # Constraints
/// - `message_hash_len` must be <= 32 bytes (BLAKE3-256 output size)
/// - `coordinate_resolution_bits` must divide evenly into `message_hash_len * 8`
pub struct WinternitzSpec {
	/// Number of bytes from message hash to use (must be <= 32)
	pub message_hash_len: usize,
	/// Number of bits per coordinate in the codeword
	pub coordinate_resolution_bits: usize,
	/// Expected sum of all coordinates
	pub target_sum: u64,
	/// Size of the domain parameter in bytes
	pub domain_param_len: usize,
}

impl WinternitzSpec {
	/// Creates a new WinternitzSpec with validation
	pub fn new(
		message_hash_len: usize,
		coordinate_resolution_bits: usize,
		target_sum: u64,
		domain_param_len: usize,
	) -> Self {
		assert!(
			message_hash_len <= 32,
			"message_hash_len {} exceeds maximum of 32 bytes (BLAKE3-256 output size)",
			message_hash_len
		);
		assert!(
			(message_hash_len * 8).is_multiple_of(coordinate_resolution_bits),
			"coordinate_resolution_bits {} must divide evenly into message_hash_len * 8 = {}",
			coordinate_resolution_bits,
			message_hash_len * 8
		);

		Self {
			message_hash_len,
			coordinate_resolution_bits,
			target_sum,
			domain_param_len,
		}
	}

	/// Returns the number of coordinates/chains
	pub const fn dimension(&self) -> usize {
		self.message_hash_len * 8 / self.coordinate_resolution_bits
	}

	/// Returns the chain length (2^coordinate_resolution_bits)
	pub const fn chain_len(&self) -> usize {
		1 << self.coordinate_resolution_bits
	}

	/// Create a spec matching SPEC_1 from leansig-xmss
	pub fn spec_1() -> Self {
		Self::new(18, 2, 119, 18)
	}

	/// Create a spec matching SPEC_2 from leansig-xmss
	pub fn spec_2() -> Self {
		Self::new(18, 4, 297, 18)
	}
}

/// Result of successfully grinding a nonce that produces a valid target sum.
pub struct GrindResult {
	/// The extracted codeword coordinates from the message hash
	pub coords: Vec<u8>,
	/// The nonce value that achieved the target sum
	pub nonce: Vec<u8>,
}

/// Grind for a nonce whose codeword coordinates sum to the target value.
///
/// - Draws random nonces and hashes the tweaked message until the coordinates hit the target sum.
/// - The epoch is part of the message tweak, so the codeword depends on the epoch.
///
/// # Arguments
///
/// * `spec` - The Winternitz OTS specification containing dimension, resolution, and target sum
/// * `rng` - Random number generator for generating nonce candidates
/// * `param` - The cryptographic parameter
/// * `epoch` - The epoch (leaf index) at which the message is signed
/// * `message` - The message to be signed
///
/// # Returns
///
/// * `Some(GrindResult)` - the successful nonce and the resulting coordinates.
/// * `None` - failed to find a valid nonce within 1000 attempts.
pub fn grind_nonce(
	spec: &WinternitzSpec,
	rng: &mut rand::rngs::StdRng,
	param: &[u8],
	epoch: u64,
	message: &[u8],
) -> Option<GrindResult> {
	use rand::prelude::*;

	use super::{codeword::extract_coordinates, hashing::hash_message};

	let mut nonce = vec![0u8; NONCE_LENGTH_BYTES];
	for _ in 0..1000 {
		rng.fill_bytes(&mut nonce);
		let tweaked_message_hash = hash_message(param, epoch, &nonce, message);

		let coords = extract_coordinates(
			&tweaked_message_hash[..spec.message_hash_len],
			spec.dimension(),
			spec.coordinate_resolution_bits,
		);
		let coord_sum: usize = coords.iter().map(|&c| c as usize).sum();
		if coord_sum == spec.target_sum as usize {
			return Some(GrindResult { coords, nonce });
		}
	}
	None
}

#[cfg(test)]
mod tests {
	use binius_core::{Word, verify::verify_constraints};
	use binius_frontend::util::pack_bytes_into_wires_le;
	use rand::prelude::*;

	use super::*;
	use crate::hash_based_sig::hashing::hash_chain_blake3;

	/// Fixed epoch used by the Winternitz OTS test.
	const TEST_EPOCH: u64 = 5;

	#[test]
	fn test_circuit_winternitz_ots() {
		let spec = WinternitzSpec::spec_1();
		let builder = CircuitBuilder::new();

		// Inputs
		let domain_param: Vec<Wire> = (0..(spec.domain_param_len.div_ceil(8)))
			.map(|_| builder.add_inout())
			.collect();
		let epoch = builder.add_inout();
		let message: Vec<Wire> = (0..HASH_WIRES_COUNT).map(|_| builder.add_inout()).collect();
		let nonce: Vec<Wire> = (0..NONCE_WIRES_COUNT)
			.map(|_| builder.add_inout())
			.collect();

		let signature_hashes: Vec<[Wire; HASH_WIRES_COUNT]> = (0..spec.dimension())
			.map(|_| std::array::from_fn(|_| builder.add_inout()))
			.collect();
		let public_key_hashes: Vec<[Wire; HASH_WIRES_COUNT]> = (0..spec.dimension())
			.map(|_| std::array::from_fn(|_| builder.add_inout()))
			.collect();

		circuit_winternitz_ots(
			&builder,
			&domain_param,
			epoch,
			&message,
			&nonce,
			&signature_hashes,
			&public_key_hashes,
			&spec,
		);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();

		// Randomize inputs and grind a nonce for valid target sum
		let mut rng = StdRng::seed_from_u64(7);
		let mut domain_param_bytes = vec![0u8; spec.domain_param_len];
		rng.fill_bytes(&mut domain_param_bytes);
		let mut message_bytes = [0u8; MESSAGE_LENGTH_BYTES];
		rng.fill_bytes(&mut message_bytes);

		// Find coordinates via grinding for consistency with codeword sum
		let grind = grind_nonce(&spec, &mut rng, &domain_param_bytes, TEST_EPOCH, &message_bytes)
			.expect("Failed to find valid nonce");
		let mut nonce_bytes = grind.nonce;
		nonce_bytes.resize(NONCE_LENGTH_BYTES, 0);

		// Pack fixed inputs
		w[epoch] = Word::from_u64(TEST_EPOCH);
		pack_bytes_into_wires_le(&mut w, &domain_param, &domain_param_bytes);
		pack_bytes_into_wires_le(&mut w, &message, &message_bytes);
		pack_bytes_into_wires_le(&mut w, &nonce, &nonce_bytes);

		// Prepare signatures and derive each public key as the endpoint after
		// (chain_len - 1 - x_i) steps starting from sig_i, at positions x_i+1..chain_len-1.
		let mut sig_hashes = Vec::with_capacity(spec.dimension());
		let mut pk_hashes = Vec::with_capacity(spec.dimension());

		for chain_idx in 0..spec.dimension() {
			let mut sig = [0u8; MESSAGE_LENGTH_BYTES];
			rng.fill_bytes(&mut sig);
			sig_hashes.push(sig);

			let xi = grind.coords[chain_idx] as usize;
			let pk_hash = hash_chain_blake3(
				&domain_param_bytes,
				TEST_EPOCH as u32,
				chain_idx as u8,
				&sig,
				xi,
				spec.chain_len() - 1 - xi,
			);
			pk_hashes.push(pk_hash);

			pack_bytes_into_wires_le(&mut w, &signature_hashes[chain_idx], &sig);
			pack_bytes_into_wires_le(&mut w, &public_key_hashes[chain_idx], &pk_hashes[chain_idx]);
		}

		// Every hash is BLAKE3, derived from the inputs, so the evaluator fills all digests here.
		circuit.populate_wire_witness(&mut w).unwrap();
		let cs = circuit.constraint_system();
		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}
}
