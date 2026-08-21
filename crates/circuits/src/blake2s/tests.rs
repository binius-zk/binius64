// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_frontend::CircuitBuilder;
use blake2::{Blake2s256, Digest};
use rand::prelude::*;
use rstest::rstest;

use super::{Blake2s, Blake2sCompress2x};

#[rstest]
#[case(0, 1)] // Empty message - edge case
#[case(1, 1)] // Single byte - minimal non-empty
#[case(3, 2)] // 3 bytes - partial word (less than 4 bytes)
#[case(64, 16)] // 64 bytes - exactly one block
#[case(65, 17)] // 65 bytes - crosses block boundary
fn test_blake2s_circuit(#[case] message_len_bytes: usize, #[case] max_message_len_words: usize) {
	// Create test message with deterministic random bytes seeded by the length inputs
	let seed = ((message_len_bytes as u64) << 32) | (max_message_len_words as u64);
	let mut rng = StdRng::seed_from_u64(seed);
	let mut message = vec![0u8; message_len_bytes];
	rng.fill_bytes(&mut message);

	// Compute expected digest using reference implementation
	let mut hasher = Blake2s256::new();
	hasher.update(&message);
	let expected_digest = hasher.finalize();

	// Build circuit with specified max message length
	// Note: Blake2s expects length in bytes, not words
	let max_bytes = max_message_len_words * 4;
	assert!(
		message_len_bytes <= max_bytes,
		"Message length {} exceeds max capacity {} bytes",
		message_len_bytes,
		max_bytes
	);

	let mut builder = CircuitBuilder::new();
	let blake2s = Blake2s::new_witness(&mut builder, message_len_bytes);
	let circuit = builder.build();

	// Create and populate witness
	let mut witness = circuit.new_witness_filler();
	blake2s.populate_message(&mut witness, &message);
	blake2s.populate_digest(&mut witness, expected_digest.as_slice().try_into().unwrap());

	// Verify circuit accepts the witness
	circuit
		.populate_wire_witness(&mut witness)
		.expect("Circuit should accept valid witness");

	// Verify all constraints are satisfied
	let cs = circuit.constraint_system();
	cs.verify(&witness.into_value_vec())
		.expect("All constraints should be satisfied");
}

// Builds and checks a fixed-length BLAKE2s circuit for `message`, against the reference crate.
//
// Shared by the block-boundary sweep below, which needs message lengths beyond what a single
// `#[case]` list conveniently expresses: every block count from one block up through several
// pairs, in both parities.
fn check(message: &[u8]) {
	let expected: [u8; 32] = Blake2s256::digest(message).into();

	let mut builder = CircuitBuilder::new();
	let blake2s = Blake2s::new_witness(&mut builder, message.len());
	let circuit = builder.build();

	let mut witness = circuit.new_witness_filler();
	blake2s.populate_message(&mut witness, message);
	blake2s.populate_digest(&mut witness, &expected);

	circuit
		.populate_wire_witness(&mut witness)
		.unwrap_or_else(|e| panic!("Blake2s digest disagreed for len={}: {e:?}", message.len()));
	let cs = circuit.constraint_system();
	cs.verify(&witness.into_value_vec()).unwrap();
}

#[test]
fn block_boundaries() {
	// Lengths chosen to cover 1 through 6 blocks, both parities, and the byte immediately
	// either side of each block boundary.
	//
	// That sweep exercises every shape of the compression chain:
	//
	//     1 block     : the single-lane path, no pairing at all
	//     2, 4 blocks : whole pairs, no trailing block
	//     3, 5 blocks : whole pairs plus a trailing block through the dead-lane path
	for &len in &[
		0usize, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 319, 320, 321,
	] {
		let message: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
		check(&message);
	}
}

// Hashes `message` with the two-lane compression registered as a chip, and checks both the
// digest and the resulting constraint system.
//
// The digest wires are populated with the reference digest, so a disagreement fails to
// populate.
//
// What the chip adds is checked afterward: every compression the circuit makes has to be
// served by a chip instance that recomputes the same words.
fn check_with_compress_chip(message: &[u8]) {
	let mut builder = CircuitBuilder::new();
	builder.register_chip(Blake2sCompress2x, &[]);

	let blake2s = Blake2s::new_witness(&mut builder, message.len());
	let circuit = builder.build_m4();
	circuit.validate().unwrap();
	let cs = circuit.to_constraint_system();
	cs.validate().unwrap();

	let expected: [u8; 32] = Blake2s256::digest(message).into();

	let witness = circuit
		.generate_witness(|w| {
			// Populate the message bytes, the same way the plain circuit does.
			for (i, bytes) in message.chunks(8).enumerate() {
				let mut le_bytes = [0u8; 8];
				le_bytes[..bytes.len()].copy_from_slice(bytes);
				w[blake2s.message[i]] = Word(u64::from_le_bytes(le_bytes));
			}
			// Populate the expected digest, the same way the plain circuit does.
			for i in 0..8 {
				let word = u32::from_le_bytes(expected[i * 4..(i + 1) * 4].try_into().unwrap());
				w[blake2s.digest[i]] = Word(word as u64);
			}
		})
		.unwrap_or_else(|e| panic!("Blake2s digest disagreed for len={}: {e:?}", message.len()));

	witness.verify(&cs).unwrap();
}

// Every layer between the circuit's entry point and its two-lane compression is untouched by
// the chip.
//
// Block pairs, and the trailing odd block riding the paired core with a dead lane, land as chip
// calls because the builder holds the chip, not because anything in between was told to route
// through it.
//
// Lengths cover one pair, a pair plus a trailing block, two pairs, and two pairs plus a
// trailing block.
#[test]
fn a_registered_chip_serves_every_paired_compression() {
	for &len in &[128usize, 192, 256, 300] {
		let message: Vec<u8> = (0..len).map(|i| (i * 37 + 1) as u8).collect();
		check_with_compress_chip(&message);
	}
}

// A single-block message compresses single-lane only, and never reaches the paired gadget.
//
// So its chip goes uncalled, and the constraint system that leaves cannot be populated.
#[test]
fn a_chip_no_paired_compression_reaches_leaves_an_uncalled_chip() {
	let mut builder = CircuitBuilder::new();
	builder.register_chip(Blake2sCompress2x, &[]);
	Blake2s::new_witness(&mut builder, 4);

	let error = builder.build_m4().validate().unwrap_err();
	assert!(matches!(error, binius_frontend::CircuitM4Error::NeverCalled { .. }), "{error:?}");
}
