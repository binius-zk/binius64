// Copyright 2026 The Binius Developers

//! Checks of the in-circuit Merkle gadgets against the native committing scheme they mirror.

use std::array;

use binius_core::Word;
use binius_field::BinaryField128bGhash as B128;
use binius_frontend::{CircuitBuilder, CircuitStat, PopulateError, WitnessFiller};
use binius_hash::{
	CompressionFunction, hash_serialize,
	sha256::{Sha256Compression, Sha256HashSuite},
};
use binius_iop::merkle_tree::MerkleTreeScheme;
use binius_iop_prover::merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use digest::Output;
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use sha2::{Digest as _, Sha256};

use super::*;

/// The native committer whose trees and openings every gadget here is checked against.
type Prover = BinaryMerkleTreeProver<B128, Sha256HashSuite>;

/// The Fiat-Shamir hasher of the transcript an opening travels through.
type Challenger = HasherChallenger<Sha256>;

/// A given number of uniformly random field elements.
fn random_values(rng: &mut StdRng, n: usize) -> Vec<B128> {
	// Every test seeds its own generator, so a failure reproduces from the seed alone.
	(0..n).map(|_| B128::from(rng.random::<u128>())).collect()
}

/// A hash output as the plain 32 bytes the committing scheme moves around.
fn digest_bytes(digest: &Output<Sha256>) -> [u8; 32] {
	// Both forms hold the same bytes, so this only drops the hasher-specific wrapper.
	(*digest).into()
}

/// A given number of fresh witness elements.
fn element_wires(builder: &CircuitBuilder, n: usize) -> Vec<Element> {
	// Two wires an element: its serialized low half, then its high half.
	(0..n)
		.map(|_| array::from_fn(|_| builder.add_witness()))
		.collect()
}

/// A given number of fresh witness digests.
fn digest_wires(builder: &CircuitBuilder, n: usize) -> Vec<Digest> {
	// Four wires a digest, each carrying two of the message words a compression consumes.
	(0..n)
		.map(|_| array::from_fn(|_| builder.add_witness()))
		.collect()
}

/// Writes one field element into each element's pair of wires.
fn fill_elements(w: &mut WitnessFiller, wires: &[Element], values: &[B128]) {
	// A count mismatch would leave wires unwritten, which reads later as a wrong value.
	assert_eq!(wires.len(), values.len(), "one element per value");
	// Splitting a value across its two wires is the packing convention, not a choice here.
	for (element, value) in wires.iter().zip(values) {
		populate_element(w, element, u128::from(*value));
	}
}

/// Writes one 32-byte digest into each digest's wires.
fn fill_digests(w: &mut WitnessFiller, wires: &[Digest], bytes: &[[u8; 32]]) {
	// As with elements, an unwritten wire would masquerade as a corrupted digest.
	assert_eq!(wires.len(), bytes.len(), "one digest per byte string");
	// The byte order each wire holds is fixed by the digest convention.
	for (digest, bytes) in wires.iter().zip(bytes) {
		populate_digest(w, digest, bytes);
	}
}

/// Asserts every retained failure sits under the given assertion path, and that none were dropped.
///
/// Locating the failure is what separates "the circuit rejected this" from "the circuit rejected
/// this for the reason under test".
fn assert_failed_paths(error: &PopulateError, path: &str) {
	// Witness population may retain fewer failures than it found, and a dropped one could sit
	// anywhere else in the circuit.
	assert_eq!(
		error.total,
		error.failures.len(),
		"the failure list must be complete, so every path can be checked"
	);
	// A rejection that named no failing assertion would pass a negative test for free.
	assert!(error.total > 0, "an unsatisfied circuit must report at least one failure");
	for failure in &error.failures {
		// Gate names are hierarchical, so a prefix match pins the failure to one gadget.
		assert!(
			failure.path.starts_with(path),
			"unexpected failing assertion {:?}, expected one under {path:?}",
			failure.path
		);
		// A failure with no message would leave a real regression undiagnosable.
		assert!(!failure.detail.is_empty(), "a failure must carry a diagnostic");
	}
}

#[test]
fn compression_iv_is_the_domain_separator() {
	// Invariant: the state every inner node folds from is the hash of the scheme's separator
	// string, read as eight little-endian words.
	//
	// Why it is pinned: starting from the hash function's own initial state would let a node
	// digest coincide with a plain hash of the same 64 bytes.
	let separator: [u8; 32] = Sha256::digest(b"BINIUS SHA-256 COMPRESS").into();

	// Four bytes a word, read little-endian, which is the order the native scheme reads.
	let expected: [u32; 8] = array::from_fn(|i| {
		u32::from_le_bytes(
			separator[4 * i..4 * i + 4]
				.try_into()
				.expect("four bytes per word"),
		)
	});
	assert_eq!(COMPRESSION_IV, expected);
}

#[test]
fn element_words_split_the_serialization() {
	// Invariant: an element's sixteen serialized bytes split across two wires, low half first.
	//
	// Fixture state: the bytes 0x00 through 0x0f, in serialization order.
	//
	//     serialized :  00 01 02 03 04 05 06 07 | 08 09 0a 0b 0c 0d 0e 0f
	//     low wire   :  0x0706050403020100        bytes 0 to 7, little-endian
	//     high wire  :  0x0f0e0d0c0b0a0908        bytes 8 to 15, little-endian
	let value = 0x0f0e0d0c_0b0a0908_07060504_03020100u128;
	assert_eq!(element_words(value), [0x07060504_03020100, 0x0f0e0d0c_0b0a0908]);
}

#[test]
fn digest_words_carry_the_message_words() {
	// Invariant: each digest wire holds the two big-endian message words a compression reads for
	// eight consecutive digest bytes.
	//
	// Fixture state: the 32 bytes 0x00 through 0x1f, in digest order.
	//
	//     wire 0 low   :  bytes 00 01 02 03  ->  0x00010203
	//     wire 0 high  :  bytes 04 05 06 07  ->  0x04050607
	//     wire 3 low   :  bytes 18 19 1a 1b  ->  0x18191a1b
	//     wire 3 high  :  bytes 1c 1d 1e 1f  ->  0x1c1d1e1f
	let bytes: [u8; 32] = array::from_fn(|i| i as u8);
	let words = digest_words(&bytes);
	// The first and last wires bracket the packing, so a swapped half or byte order shows here.
	assert_eq!(words[0], 0x04050607_00010203);
	assert_eq!(words[3], 0x1c1d1e1f_18191a1b);
}

/// Checks the in-circuit hash of one leaf against the native scheme's hash of the same elements.
///
/// The comparison lives inside the circuit, as an equality assertion against a public claim.
/// So a mismatch fails witness population rather than a comparison made in Rust.
fn check_leaf_digest(values: &[B128]) {
	// Reference value: what the native scheme hashes these elements to.
	let expected = digest_bytes(&hash_serialize::<B128, Sha256>(values).expect("B128 serializes"));

	let builder = CircuitBuilder::new();
	// The leaf's elements are witness data, as a decommitted leaf would be.
	let wires = element_wires(&builder, values.len());
	// The native digest enters on public wires, so the assertion spans circuit against native.
	let claimed: Digest = array::from_fn(|_| builder.add_inout());
	builder.assert_eq_v("leaf", leaf_digest(&builder, &wires), claimed);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	// Only the inputs and the reference value are written; the digest wires are derived.
	fill_elements(&mut w, &wires, values);
	populate_digest(&mut w, &claimed, &expected);

	// Population evaluates every gate and assertion, so a wrong digest fails right here.
	circuit
		.populate_wire_witness(&mut w)
		.expect("the circuit must reproduce the native leaf digest");
	// Verification re-checks that same witness against the constraint system itself.
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("every constraint must hold");
}

#[test]
fn leaf_digest_matches_hash_serialize() {
	let mut rng = StdRng::seed_from_u64(0);

	// Invariant: the in-circuit leaf hash agrees with the native one at every leaf width.
	//
	// Fixture state: an element serializes to 16 bytes and a hash block is 64 bytes, so the
	// widths below straddle the block boundary in both directions.
	//
	//     1, 2 elements   :  16 and 32 bytes, the narrowest leaves
	//     3, 5 elements   :  48 and 80 bytes, so the padding lands mid-block
	//     4, 8, 16        :  64, 128 and 256 bytes, so the message fills whole blocks
	for leaf_size in [1, 2, 3, 4, 5, 8, 16] {
		check_leaf_digest(&random_values(&mut rng, leaf_size));
	}
}

#[test]
fn leaf_digest_2x_matches_hash_serialize() {
	let mut rng = StdRng::seed_from_u64(1);

	// Invariant: hashing two equal-width leaves in the two 32-bit lanes of one compression chain
	// gives each lane the digest that leaf would get on its own.
	//
	// Fixture state: widths of 1, 3, 4 and 16 elements, covering mid-block and block-aligned
	// padding at both the narrowest and a multi-block width.
	for leaf_size in [1, 3, 4, 16] {
		// Independent random leaves, so a value leaking between lanes cannot go unnoticed.
		let values = [
			random_values(&mut rng, leaf_size),
			random_values(&mut rng, leaf_size),
		];
		// Reference values: each leaf hashed natively on its own, with no notion of lanes.
		let expected = values.each_ref().map(|values| {
			digest_bytes(&hash_serialize::<B128, Sha256>(values).expect("B128 serializes"))
		});

		let builder = CircuitBuilder::new();
		// Equal widths in, so both padded messages hold the same number of blocks.
		let wires = [
			element_wires(&builder, leaf_size),
			element_wires(&builder, leaf_size),
		];
		// One public claim per lane, each carrying that lane's native digest.
		let claimed: [Digest; 2] = array::from_fn(|_| array::from_fn(|_| builder.add_inout()));
		let computed = leaf_digest_2x(&builder, [&wires[0], &wires[1]]);
		// Pinning the lanes separately is what rules out a swapped or duplicated lane.
		for lane in 0..2 {
			builder.assert_eq_v(format!("leaf[{lane}]"), computed[lane], claimed[lane]);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		// Both leaves' elements and both reference digests, the only hand-written wires.
		for lane in 0..2 {
			fill_elements(&mut w, &wires[lane], &values[lane]);
			populate_digest(&mut w, &claimed[lane], &expected[lane]);
		}

		// A lane that drifted from its own message fails one of the two assertions here.
		circuit
			.populate_wire_witness(&mut w)
			.expect("both lanes must reproduce their native leaf digest");
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.expect("every constraint must hold");
	}
}

#[test]
fn compress_node_matches_sha256_compression() {
	let mut rng = StdRng::seed_from_u64(2);

	// Invariant: an in-circuit inner node equals the native two-to-one compression of the same
	// pair of child digests.
	//
	// Fixture state: four random child pairs, each pair being the 64 bytes of one message block.
	for _ in 0..4 {
		// A node hash never inspects what its children hash, so random bytes are enough.
		let children: [[u8; 32]; 2] = array::from_fn(|_| array::from_fn(|_| rng.random()));
		// Reference value: the native compression over the two children concatenated.
		let expected = digest_bytes(
			&Sha256Compression::default().compress([children[0].into(), children[1].into()]),
		);

		let builder = CircuitBuilder::new();
		// Both children are witness data, as an authentication path supplies them.
		let inputs = digest_wires(&builder, 2);
		let claimed: Digest = array::from_fn(|_| builder.add_inout());
		// Left child first, which is the order the message block lays them out in.
		builder.assert_eq_v("node", compress_node(&builder, inputs[0], inputs[1]), claimed);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		fill_digests(&mut w, &inputs, &children);
		populate_digest(&mut w, &claimed, &expected);

		// A wrong byte order anywhere in the block or the output fails here.
		circuit
			.populate_wire_witness(&mut w)
			.expect("the circuit must reproduce the native node digest");
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.expect("every constraint must hold");
	}
}

#[test]
fn compress_node_2x_matches_sha256_compression() {
	let mut rng = StdRng::seed_from_u64(3);

	// Invariant: two inner nodes folded in the two lanes of one compression each equal the native
	// compression of their own children.
	//
	// Fixture state: four independent children, paired into two nodes, one per lane.
	//
	//     lane 0 :  node(child 0, child 1)
	//     lane 1 :  node(child 2, child 3)
	let children: [[u8; 32]; 4] = array::from_fn(|_| array::from_fn(|_| rng.random()));
	// Reference values: each node compressed natively, with no notion of lanes.
	let expected: [[u8; 32]; 2] = array::from_fn(|i| {
		digest_bytes(
			&Sha256Compression::default()
				.compress([children[2 * i].into(), children[2 * i + 1].into()]),
		)
	});

	let builder = CircuitBuilder::new();
	let inputs = digest_wires(&builder, 4);
	// One public claim per lane, so neither lane can borrow the other's answer.
	let claimed: [Digest; 2] = array::from_fn(|_| array::from_fn(|_| builder.add_inout()));
	let computed = compress_node_2x(&builder, [(inputs[0], inputs[1]), (inputs[2], inputs[3])]);
	for lane in 0..2 {
		builder.assert_eq_v(format!("node[{lane}]"), computed[lane], claimed[lane]);
	}

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	fill_digests(&mut w, &inputs, &children);
	for lane in 0..2 {
		populate_digest(&mut w, &claimed[lane], &expected[lane]);
	}

	// Merging the lanes and splitting them apart again must be lossless, or this fails.
	circuit
		.populate_wire_witness(&mut w)
		.expect("both lanes must reproduce their native node digest");
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("every constraint must hold");
}

/// Builds a circuit that folds a layer of digests to a claimed root, and reports the outcome.
///
/// Filling the witness is what runs the check.
/// So a layer that cannot reach the root surfaces as an error here rather than a panic.
fn run_layer(root: &[u8; 32], layer: &[[u8; 32]]) -> Result<(), PopulateError> {
	let builder = CircuitBuilder::new();
	// The root is public, since it is what the commitment pinned down.
	let root_wires: Digest = array::from_fn(|_| builder.add_inout());
	// The layer is witness data, since a prover is what hands it over.
	let layer_wires = digest_wires(&builder, layer.len());
	verify_layer(&builder, root_wires, &layer_wires);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	populate_digest(&mut w, &root_wires, root);
	fill_digests(&mut w, &layer_wires, layer);

	// Evaluating the fold is where a wrong layer is caught, so the error is returned to the caller.
	circuit.populate_wire_witness(&mut w)?;
	// A witness that populated must satisfy the constraint system, so this failing is a defect.
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("a satisfied circuit must verify");
	Ok(())
}

#[test]
fn verify_layer_accepts_prover_layers() {
	let mut rng = StdRng::seed_from_u64(4);
	let prover = Prover::new();

	// Invariant: a layer the native committer decommits folds back to the committed root, at every
	// depth the tree has.
	//
	// Fixture state: depth 3 tree, 8 leaves, 2 elements each, so 16 committed values.
	let (batch_size, log_len) = (2, 3);
	let data = random_values(&mut rng, batch_size << log_len);
	let (commitment, tree) = prover.commit(&data, batch_size);

	// Depth 0 is the root on its own, depth 3 the whole leaf layer, so both extremes are covered.
	for layer_depth in 0..=log_len {
		// The digests a native decommitment at this depth would send, one per node there.
		let layer = prover.layer(&tree, layer_depth);

		// Anchor the fixture: the native verifier accepts this layer at this depth.
		// Without it the circuit could agree with a committer that was itself wrong.
		prover
			.scheme()
			.verify_layer(&commitment.root, layer_depth, layer)
			.expect("the prover's own layer must verify natively");

		// The same digests as plain bytes, which is the form a witness carries.
		let bytes: Vec<[u8; 32]> = layer.iter().map(digest_bytes).collect();
		run_layer(&digest_bytes(&commitment.root), &bytes)
			.expect("the layer must fold to the committed root");
	}
}

#[test]
fn verify_layer_rejects_a_corrupted_digest() {
	let mut rng = StdRng::seed_from_u64(5);
	let prover = Prover::new();

	// Invariant: a layer with one altered entry cannot fold to the committed root.
	//
	// Fixture state: depth 2 tree, 4 leaves, 1 element each, layer at depth 2, so the layer is
	// the leaf layer and holds 4 digests.
	let (batch_size, log_len, layer_depth) = (1, 2, 2);
	let data = random_values(&mut rng, batch_size << log_len);
	let (commitment, tree) = prover.commit(&data, batch_size);

	let mut bytes: Vec<[u8; 32]> = prover
		.layer(&tree, layer_depth)
		.iter()
		.map(digest_bytes)
		.collect();
	// Mutation: flip the lowest bit of the second entry.
	//
	//     layer   :  [d_0, d_1, d_2, d_3]  ->  [d_0, d_1 ^ 1, d_2, d_3]
	//     level 1 :  node(d_0, d_1)        ->  a different left parent
	//     level 2 :  the top digest inherits the difference, so it misses the root
	bytes[1][0] ^= 1;

	let error = run_layer(&digest_bytes(&commitment.root), &bytes)
		.expect_err("a corrupted layer entry must not fold to the root");
	// The root comparison is the check that must catch this, not some unrelated assertion.
	assert_failed_paths(&error, ".verify_layer.root");
}

/// Builds a circuit that rebuilds a whole committed vector's root, and reports the outcome.
///
/// The values arrive in leaf order, with a fixed number of them under each leaf.
fn run_vector(data: &[B128], batch_size: usize, root: &[u8; 32]) -> Result<(), PopulateError> {
	let builder = CircuitBuilder::new();
	// The committed root is public and the committed values are witness data.
	let root_wires: Digest = array::from_fn(|_| builder.add_inout());
	let wires = element_wires(&builder, data.len());
	verify_vector(&builder, root_wires, &wires, batch_size);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	populate_digest(&mut w, &root_wires, root);
	fill_elements(&mut w, &wires, data);

	// Every leaf hash and every inner node is evaluated here, so a wrong value is caught here.
	circuit.populate_wire_witness(&mut w)?;
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("a satisfied circuit must verify");
	Ok(())
}

#[test]
fn verify_vector_accepts_prover_commitments() {
	let mut rng = StdRng::seed_from_u64(6);
	let prover = Prover::new();

	// Invariant: rebuilding the tree over the committed values in-circuit reaches the root the
	// native committer published.
	//
	// Fixture state: leaf counts from a single leaf up to 16, with leaf widths that land both
	// mid-block and on a block boundary.
	//
	//     1 leaf,   1 element    the single-leaf tree, where no node is folded
	//     1 leaf,   16 elements  a leaf spanning five hash blocks
	//     2 leaves, 1 element    one node above two leaves
	//     4 leaves, 3 elements   48-byte leaves, so the padding lands mid-block
	//     8 leaves, 2 elements   an even leaf count that pairs at every level
	//     16 leaves, 1 element   the widest tree here, four levels of folding
	for (log_len, batch_size) in [(0, 1), (0, 16), (1, 1), (2, 3), (3, 2), (4, 1)] {
		let data = random_values(&mut rng, batch_size << log_len);
		let (commitment, _) = prover.commit(&data, batch_size);
		// Anchor the fixture: the native verifier accepts this vector against this root.
		prover
			.scheme()
			.verify_vector(&commitment.root, &data, batch_size)
			.expect("the prover's own vector must verify natively");

		run_vector(&data, batch_size, &digest_bytes(&commitment.root))
			.expect("the gadget must rebuild the committed root");
	}
}

#[test]
fn verify_vector_rejects_a_corrupted_value() {
	let mut rng = StdRng::seed_from_u64(7);
	let prover = Prover::new();

	// Invariant: changing one committed value stops the rebuilt tree reaching the committed root.
	//
	// Fixture state: depth 2 tree, 4 leaves, 2 elements each, so 8 committed values.
	let (batch_size, log_len) = (2, 2);
	let mut data = random_values(&mut rng, batch_size << log_len);
	let (commitment, _) = prover.commit(&data, batch_size);

	// Mutation: add one to the fourth value, which is the second element of leaf 1.
	//
	//     leaf 1  :  hash(v_2, v_3)  ->  hash(v_2, v_3 + 1)
	//     level 1 :  node(leaf 0, leaf 1) changes with it
	//     root    :  differs, so the equality assertion fails
	data[3] += B128::from(1u128);
	let error = run_vector(&data, batch_size, &digest_bytes(&commitment.root))
		.expect_err("a corrupted value must not rebuild the root");
	assert_failed_paths(&error, ".verify_vector.root");
}

/// One native opening, in the byte form the gadget's witness needs.
struct Opening {
	/// The opened leaf's values.
	values: Vec<B128>,
	/// One sibling per level, from the leaf up to the decommitted layer.
	branch: Vec<[u8; 32]>,
	/// The decommitted layer, one digest per node at its own depth.
	layer: Vec<[u8; 32]>,
}

/// Commits a vector, then produces the native opening of one leaf down to a chosen layer depth.
///
/// The authentication path travels through a transcript, exactly as it reaches a verifier.
fn prove_opening(
	data: &[B128],
	batch_size: usize,
	log_len: usize,
	layer_depth: usize,
	index: usize,
) -> Opening {
	let prover = Prover::new();
	// A fresh tree over the data, so the opening below is one this tree really supports.
	let (_, tree) = prover.commit(data, batch_size);

	// The opening is written into a transcript and read back out, so the bytes are the ones a
	// verifier would actually see.
	let mut writer = ProverTranscript::new(Challenger::default());
	prover.prove_opening(&tree, layer_depth, index, &mut writer.message());
	let mut reader = writer.into_verifier();
	let mut message = reader.message();
	// One sibling per level between the leaf and the layer, in climbing order.
	let branch = (0..log_len - layer_depth)
		.map(|_| {
			digest_bytes(
				&message
					.read::<Output<Sha256>>()
					.expect("the prover wrote one sibling per level"),
			)
		})
		.collect();

	Opening {
		// The opened leaf's values are a contiguous run of the committed data.
		values: data[index * batch_size..(index + 1) * batch_size].to_vec(),
		branch,
		// The whole layer, since a verifier picks the entry itself rather than being told it.
		layer: prover
			.layer(&tree, layer_depth)
			.iter()
			.map(digest_bytes)
			.collect(),
	}
}

/// Builds a circuit that verifies one opening against its layer, and reports the outcome.
///
/// The claimed index is a parameter of its own, rather than being read off the opening.
/// That is what lets a negative test point the check at a leaf the path does not belong to.
fn run_opening(
	opening: &Opening,
	layer_depth: usize,
	tree_depth: usize,
	claimed_index: u64,
) -> Result<(), PopulateError> {
	let builder = CircuitBuilder::new();
	// The index is public, matching a caller that derives it from sampled challenge bits.
	let index = builder.add_inout();
	// Leaf values and siblings are witness data, the parts of an opening a prover supplies.
	let values = element_wires(&builder, opening.values.len());
	let branch = digest_wires(&builder, opening.branch.len());
	// The layer is public, since it is checked once against the root and shared by every query.
	let layer: Vec<Digest> = (0..opening.layer.len())
		.map(|_| array::from_fn(|_| builder.add_inout()))
		.collect();
	verify_opening(&builder, index, &values, layer_depth, tree_depth, &layer, &branch);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	// The index the witness claims, which need not be the one the path was produced for.
	w[index] = Word(claimed_index);
	fill_elements(&mut w, &values, &opening.values);
	fill_digests(&mut w, &branch, &opening.branch);
	fill_digests(&mut w, &layer, &opening.layer);

	// The leaf hash, the climb and the layer lookup are all evaluated here.
	circuit.populate_wire_witness(&mut w)?;
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("a satisfied circuit must verify");
	Ok(())
}

#[test]
fn verify_opening_accepts_prover_openings() {
	let mut rng = StdRng::seed_from_u64(8);

	// Invariant: every opening the native committer produces climbs to the layer entry its index
	// addresses.
	//
	// Fixture state: five tree shapes, each exercised at every layer depth and every leaf index.
	//
	//     1 leaf,    1 element    the single-leaf tree, where the layer is the leaf itself
	//     2 leaves,  16 elements  a leaf spanning five hash blocks
	//     4 leaves,  3 elements   48-byte leaves, so the padding lands mid-block
	//     8 leaves,  1 element    up to a three-level climb
	//     16 leaves, 2 elements   up to a four-level climb
	for (log_len, batch_size) in [(0, 1), (1, 16), (2, 3), (3, 1), (4, 2)] {
		let data = random_values(&mut rng, batch_size << log_len);
		// At depth 0 the layer is the root alone, so the climb spans the whole tree.
		// At the full depth the climb is empty and the leaf digest is looked up directly.
		for layer_depth in 0..=log_len {
			// Every index, so every pattern of left and right pair orderings is covered.
			for index in 0..(1usize << log_len) {
				let opening = prove_opening(&data, batch_size, log_len, layer_depth, index);
				// The claimed index is the true one, so this shape must be accepted.
				run_opening(&opening, layer_depth, log_len, index as u64).unwrap_or_else(|error| {
					panic!("opening {index} at layer {layer_depth} of 2^{log_len}: {error}")
				});
			}
		}
	}
}

#[test]
fn verify_opening_rejects_a_corrupted_sibling() {
	let mut rng = StdRng::seed_from_u64(9);

	// Invariant: an authentication path with one altered sibling cannot climb to the entry the
	// index addresses.
	//
	// Fixture state: depth 3 tree, 8 leaves, 2 elements each, layer at depth 1, index 5.
	// The climb is two levels, and index 5 in binary is 101, so the layer entry is 1.
	let (batch_size, log_len, layer_depth, index) = (2, 3, 1, 5);
	let data = random_values(&mut rng, batch_size << log_len);
	let mut opening = prove_opening(&data, batch_size, log_len, layer_depth, index);

	// Mutation: flip the lowest bit of the sibling used at the bottom level.
	//
	//     level 0 :  node(sibling ^ 1, leaf digest)  ->  a different parent
	//     level 1 :  the node above inherits the difference
	//     layer   :  the arriving digest no longer equals entry 1
	opening.branch[0][0] ^= 1;
	let error = run_opening(&opening, layer_depth, log_len, index as u64)
		.expect_err("a corrupted sibling must not climb to the layer");
	assert_failed_paths(&error, ".verify_opening.layer_digest");
}

#[test]
fn verify_opening_rejects_a_corrupted_leaf_value() {
	let mut rng = StdRng::seed_from_u64(10);

	// Invariant: an opening only verifies for the leaf values that were committed.
	//
	// Fixture state: depth 3 tree, 8 leaves, 2 elements each, layer at depth 1, index 5.
	let (batch_size, log_len, layer_depth, index) = (2, 3, 1, 5);
	let data = random_values(&mut rng, batch_size << log_len);
	let mut opening = prove_opening(&data, batch_size, log_len, layer_depth, index);

	// Mutation: add one to the first of the two values under the opened leaf.
	//
	//     leaf    :  hash(v_0, v_1)  ->  hash(v_0 + 1, v_1)
	//     climb   :  the siblings are untouched, so both levels carry the difference upward
	//     layer   :  the arriving digest no longer equals entry 1
	opening.values[0] += B128::from(1u128);
	let error = run_opening(&opening, layer_depth, log_len, index as u64)
		.expect_err("a corrupted leaf value must not climb to the layer");
	assert_failed_paths(&error, ".verify_opening.layer_digest");
}

#[test]
fn verify_opening_rejects_a_wrong_index() {
	let mut rng = StdRng::seed_from_u64(11);

	// Invariant: an opening verifies only at the index its path belongs to, since the index both
	// orders every pair on the climb and picks the layer entry.
	//
	// Fixture state: depth 3 tree, 8 leaves, 2 elements each, layer at depth 1, true index 5.
	// Index 5 is binary 101, so bits 0 and 1 order the climb and bit 2 addresses the layer.
	let (batch_size, log_len, layer_depth, index) = (2, 3, 1, 5);
	let data = random_values(&mut rng, batch_size << log_len);
	let opening = prove_opening(&data, batch_size, log_len, layer_depth, index);

	// Mutation: keep the path and the layer, but claim a different index.
	//
	//     0 = 000, 4 = 100 :  bit 0 differs, so level 0 orders its pair the other way
	//     7 = 111          :  bit 1 differs, so level 1 orders its pair the other way
	//     1 = 001          :  the climb still matches, but bit 2 picks the other layer entry
	for wrong in [0u64, 1, 4, 7] {
		let error = run_opening(&opening, layer_depth, log_len, wrong)
			.expect_err("an index the branch does not belong to must fail");
		// Whichever bit differs, the same final comparison against the layer catches it.
		assert_failed_paths(&error, ".verify_opening.layer_digest");
	}
}

/// Builds a circuit that verifies a pair of openings in shared cores, and reports the outcome.
///
/// Both openings come from one tree, so they climb to the same layer.
/// Each query keeps its own claimed index, so a negative case can move one and leave the other.
fn run_opening_2x(
	openings: [&Opening; 2],
	layer_depth: usize,
	tree_depth: usize,
	claimed_indices: [u64; 2],
) -> Result<(), PopulateError> {
	assert_eq!(openings[0].layer, openings[1].layer, "both openings must climb to one layer");

	let builder = CircuitBuilder::new();
	// The same wire roles as a single opening, one set per query.
	let indices: [Wire; 2] = array::from_fn(|_| builder.add_inout());
	let values = openings.map(|opening| element_wires(&builder, opening.values.len()));
	let branch = openings.map(|opening| digest_wires(&builder, opening.branch.len()));
	// The layer is shared, since it is checked once against the root and climbed to by both.
	let layer: Vec<Digest> = (0..openings[0].layer.len())
		.map(|_| array::from_fn(|_| builder.add_inout()))
		.collect();
	verify_opening_2x(
		&builder,
		indices,
		[&values[0], &values[1]],
		layer_depth,
		tree_depth,
		&layer,
		[&branch[0], &branch[1]],
	);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	for q in 0..2 {
		w[indices[q]] = Word(claimed_indices[q]);
		fill_elements(&mut w, &values[q], &openings[q].values);
		fill_digests(&mut w, &branch[q], &openings[q].branch);
	}
	fill_digests(&mut w, &layer, &openings[0].layer);

	// Both leaf hashes, both climbs and both layer lookups are evaluated here.
	circuit.populate_wire_witness(&mut w)?;
	circuit
		.constraint_system()
		.verify(&w.into_value_vec())
		.expect("a satisfied circuit must verify");
	Ok(())
}

/// What a differential case does to the pair of openings before verifying them.
///
/// Each mutation is one of the ones the single-opening negative tests use, aimed at one query.
#[derive(Clone, Copy, Debug)]
enum Tamper {
	/// Leaves both openings as the native committer produced them.
	None,
	/// Flips the lowest bit of the sibling one query uses at the bottom level.
	Sibling(usize),
	/// Adds one to the first value under one query's leaf.
	Value(usize),
	/// Verifies one query at an index its path does not belong to.
	Index(usize),
}

proptest! {
	// Three circuits a case, each one built and evaluated, so the cases are not cheap.
	#![proptest_config(ProptestConfig::with_cases(32))]

	#[test]
	fn verify_opening_2x_matches_two_solo_openings(
		// Trees of 2 to 8 leaves, holding 1 or 2 elements per leaf.
		log_len in 1usize..=3,
		batch_size in 1usize..=2,
		// Reduced below to a depth the tree has, since the two are drawn independently.
		layer_draw in 0usize..=3,
		// Likewise reduced to leaf indices, and free to coincide.
		first in 0usize..8,
		second in 0usize..8,
		tamper in prop_oneof![
			Just(Tamper::None),
			(0usize..2).prop_map(Tamper::Sibling),
			(0usize..2).prop_map(Tamper::Value),
			(0usize..2).prop_map(Tamper::Index),
		],
		// Varying the seed varies the committed values, not just the shape.
		seed: u64,
	) {
		// Invariant: verifying two openings in shared cores accepts exactly when verifying each on
		// its own accepts, and a rejection is attributed to the query that earned it.
		//
		// The second half is what rules out cross-wiring: a pair that mixed the two climbs could
		// still accept honest openings and reject tampered ones, but would blame the wrong query.
		let layer_depth = layer_draw.min(log_len);
		let indices = [first, second].map(|index| index % (1 << log_len));

		let mut rng = StdRng::seed_from_u64(seed);
		let data = random_values(&mut rng, batch_size << log_len);
		// One tree, opened at both indices, so both openings share a layer.
		let mut openings =
			indices.map(|index| prove_opening(&data, batch_size, log_len, layer_depth, index));
		let mut claimed = indices.map(|index| index as u64);

		match tamper {
			Tamper::None => {}
			// At the full layer depth the climb is empty, so there is no sibling to corrupt.
			Tamper::Sibling(q) => {
				if let Some(sibling) = openings[q].branch.first_mut() {
					sibling[0] ^= 1;
				}
			}
			Tamper::Value(q) => openings[q].values[0] += B128::from(1u128),
			// Flipping bit 0 names another leaf, whether it reorders a pair or moves the entry.
			Tamper::Index(q) => claimed[q] ^= 1,
		}

		// Reference behaviour: each query verified on its own, by the single-opening gadget.
		let solo: [Result<(), PopulateError>; 2] =
			array::from_fn(|q| run_opening(&openings[q], layer_depth, log_len, claimed[q]));
		let paired =
			run_opening_2x([&openings[0], &openings[1]], layer_depth, log_len, claimed);

		if solo.iter().all(Result::is_ok) {
			prop_assert!(
				paired.is_ok(),
				"the pair rejected two openings that each verify alone: {paired:?}"
			);
		} else {
			let error = paired.expect_err("the pair must reject what a solo verification rejects");
			// Only the layer comparison can catch any of these, and only for a failing query.
			assert_failed_paths(&error, ".verify_opening_2x.layer_digest");
			for (q, result) in solo.iter().enumerate() {
				if result.is_ok() {
					let path = format!(".verify_opening_2x.layer_digest[{q}]");
					prop_assert!(
						!error.failures.iter().any(|failure| failure.path.starts_with(&path)),
						"query {q} verifies alone, so the pair must not fail it: {:?}",
						error.failures
					);
				}
			}
		}
	}
}

/// AND constraints one climbed level of an opening costs.
///
/// The module docs derive the layer-depth trade-off from this number, so a change here means the
/// trade-off needs re-deriving.
const LEVEL_AND: usize = 738;

/// AND constraints one climbed level costs each query, when two openings share cores.
const PAIRED_LEVEL_AND: usize = 377;

/// AND constraints one inner node costs on its own core.
const NODE_AND: usize = 742;

/// AND constraints one inner node costs when it shares a core with a second node.
const PAIRED_NODE_AND: usize = 380;

/// The AND and BMUL constraint counts of a circuit the caller builds.
fn cost(build: impl FnOnce(&CircuitBuilder)) -> (usize, usize) {
	let builder = CircuitBuilder::new();
	// The caller emits the gadget under measurement, and nothing else.
	build(&builder);
	// Counting after building means constant folding and pruning are already accounted for.
	let stat = CircuitStat::collect(&builder.build());
	// The two columns are separate budgets, so neither can stand in for the other.
	(stat.n_and_constraints, stat.n_bmul_constraints)
}

/// The cost of verifying one opening of a given tree shape, layer depth and leaf width.
fn opening_cost(tree_depth: usize, layer_depth: usize, leaf_size: usize) -> (usize, usize) {
	cost(|builder| {
		// The same wire roles a real verifier would use, so the cost is the real one.
		let index = builder.add_inout();
		let values = element_wires(builder, leaf_size);
		// One sibling per level between the leaf and the layer.
		let branch = digest_wires(builder, tree_depth - layer_depth);
		// A layer at depth L holds 2^L entries, which is what the entry lookup ranges over.
		let layer: Vec<Digest> = (0..1usize << layer_depth)
			.map(|_| array::from_fn(|_| builder.add_inout()))
			.collect();
		verify_opening(builder, index, &values, layer_depth, tree_depth, &layer, &branch);
	})
}

#[test]
fn verify_opening_follows_the_documented_cost_model() {
	// Invariant: an opening's AND cost is affine in the number of climbed levels, and its select
	// cost is exactly the pair ordering plus the layer lookup.
	//
	// Fixture state: depth 20 trees with one-element leaves, at five layer depths.
	// Each shape is measured again one level deeper, so the difference isolates one level.
	for layer_depth in [0usize, 1, 4, 7, 8] {
		let (and, bmul) = opening_cost(20, layer_depth, 1);
		let (deeper_and, deeper_bmul) = opening_cost(21, layer_depth, 1);

		// Ordering a pair is eight selects, one per wire of each side.
		// Picking the layer entry is four selects per lookup node, and a 2^L-wide lookup has
		// 2^L - 1 of them, so this count is exact rather than a bound.
		assert_eq!(
			bmul,
			8 * (20 - layer_depth) + 4 * ((1 << layer_depth) - 1),
			"BMUL at layer depth {layer_depth}"
		);
		// A deeper tree at the same layer depth climbs one more level, hence orders one more pair.
		assert_eq!(deeper_bmul - bmul, 8, "one more level orders one more pair");

		// Per climbed level the AND cost is one node hash and nothing else, at every layer depth.
		assert_eq!(
			deeper_and - and,
			LEVEL_AND,
			"AND per climbed level at layer depth {layer_depth}"
		);

		// Printed so the figures quoted in the module docs can be re-read off a test run.
		println!("verify_opening depth=20 layer={layer_depth} leaf=1: AND={and} BMUL={bmul}");
	}

	// A wider leaf hashes more blocks, which moves the constant term but not the per-level slope.
	let (narrow, _) = opening_cost(20, 7, 1);
	let (wide, _) = opening_cost(20, 7, 16);
	println!("verify_opening depth=20 layer=7 leaf=16: AND={wide}");
	assert!(wide > narrow, "a sixteen-element leaf hashes more blocks than a one-element leaf");
}

/// The cost of verifying two openings of the same tree shape in shared cores.
fn opening_cost_2x(tree_depth: usize, layer_depth: usize, leaf_size: usize) -> (usize, usize) {
	cost(|builder| {
		// One set of wires per query, over the one layer both climbs land on.
		let indices: [Wire; 2] = array::from_fn(|_| builder.add_inout());
		let values = [(); 2].map(|()| element_wires(builder, leaf_size));
		let branch = [(); 2].map(|()| digest_wires(builder, tree_depth - layer_depth));
		let layer: Vec<Digest> = (0..1usize << layer_depth)
			.map(|_| array::from_fn(|_| builder.add_inout()))
			.collect();
		verify_opening_2x(
			builder,
			indices,
			[&values[0], &values[1]],
			layer_depth,
			tree_depth,
			&layer,
			[&branch[0], &branch[1]],
		);
	})
}

#[test]
fn verify_opening_2x_follows_the_documented_cost_model() {
	// Invariant: pairing two climbs halves what a level costs each of them, and changes nothing
	// about the selects, since ordering a pair and picking a layer entry stay per query.
	//
	// Fixture state: depth 20 trees with one-element leaves, at five layer depths.
	// Each shape is measured again one level deeper, so the difference isolates one level.
	for layer_depth in [0usize, 1, 4, 7, 8] {
		let (and, bmul) = opening_cost_2x(20, layer_depth, 1);
		let (deeper_and, deeper_bmul) = opening_cost_2x(21, layer_depth, 1);

		// Two queries, each ordering one pair per level and picking one layer entry.
		assert_eq!(
			bmul,
			2 * (8 * (20 - layer_depth) + 4 * ((1 << layer_depth) - 1)),
			"BMUL at layer depth {layer_depth}"
		);
		assert_eq!(deeper_bmul - bmul, 16, "one more level orders one more pair per query");

		// One shared node hash per level, so each query pays half of it.
		assert_eq!(
			(deeper_and - and) / 2,
			PAIRED_LEVEL_AND,
			"AND per climbed level per query at layer depth {layer_depth}"
		);

		// Printed so the figures quoted in the module docs can be re-read off a test run.
		println!("verify_opening_2x depth=20 layer={layer_depth} leaf=1: AND={and} BMUL={bmul}");
	}

	// Sharing cores must be what pays for itself, so a pair costs less than two lone climbs.
	let (paired, _) = opening_cost_2x(20, 7, 1);
	let (solo, _) = opening_cost(20, 7, 1);
	println!("two solo openings: AND={}, one paired: AND={paired}", 2 * solo);
	assert!(paired < 2 * solo, "a shared core must beat two separate ones");
}

#[test]
fn two_lane_packing_nearly_halves_a_node() {
	// Invariant: two inner nodes sharing one compression core cost about what one node costs on
	// its own, since a gate costs one constraint whatever its operand width.
	//
	//     one node  on its own core :  the low 32-bit lane works, the high one idles
	//     two nodes on a shared core:  both lanes work, one node each
	let (single, _) = cost(|builder| {
		// Two children in, one node digest out, pinned to a public claim.
		let inputs = digest_wires(builder, 2);
		let claimed: Digest = array::from_fn(|_| builder.add_inout());
		builder.assert_eq_v("node", compress_node(builder, inputs[0], inputs[1]), claimed);
	});
	let (paired, _) = cost(|builder| {
		// Four children give two independent nodes, which is what lets them share a core.
		let inputs = digest_wires(builder, 4);
		let claimed: [Digest; 2] = array::from_fn(|_| array::from_fn(|_| builder.add_inout()));
		let computed = compress_node_2x(builder, [(inputs[0], inputs[1]), (inputs[2], inputs[3])]);
		for lane in 0..2 {
			builder.assert_eq_v(format!("node[{lane}]"), computed[lane], claimed[lane]);
		}
	});
	println!("compress_node: AND={single}, compress_node_2x: AND={paired}");

	assert_eq!(single, NODE_AND, "AND per lone node");
	// Halving the pair's cost gives the per-node figure the module docs quote.
	assert_eq!(paired / 2, PAIRED_NODE_AND, "AND per node when two share a core");
}

#[test]
fn two_lane_packing_nearly_halves_a_single_block_leaf() {
	// Invariant: two narrow leaves hashed in separate lanes cost about what one costs alone.
	//
	// Fixture state: one element is 16 bytes, so a padded one-element leaf is a single block.
	// Equal block counts let the two lanes run end to end with no leftover single-lane core.
	let leaf_cost = |n_leaves: usize| {
		cost(|builder| {
			// One element per leaf, so the leaf count is also the element count.
			let values = element_wires(builder, n_leaves);
			// Every digest is pinned, or dead-code elimination prunes the hashing away.
			let claimed: Vec<Digest> = (0..n_leaves)
				.map(|_| array::from_fn(|_| builder.add_inout()))
				.collect();
			// A lone leaf uses the single-lane path, a pair the two-lane one.
			if n_leaves == 1 {
				builder.assert_eq_v("leaf", leaf_digest(builder, &values[..1]), claimed[0]);
			} else {
				let computed = leaf_digest_2x(builder, [&values[..1], &values[1..2]]);
				for lane in 0..2 {
					builder.assert_eq_v(format!("leaf[{lane}]"), computed[lane], claimed[lane]);
				}
			}
		})
	};
	let (single, _) = leaf_cost(1);
	let (paired, _) = leaf_cost(2);
	println!("leaf_digest: AND={single}, leaf_digest_2x: AND={paired}");

	// The one-eighth margin covers merging the two lanes and splitting them apart again.
	assert!(paired < single + single / 8, "two paired leaves must cost about one lone leaf");
}

#[test]
fn verify_layer_costs_one_paired_node_per_inner_node() {
	// Invariant: folding a layer to its root spends one shared-core node hash per inner node, and
	// no selects at all.
	//
	// Fixture state: layers 2, 16 and 128 digests wide, holding 1, 15 and 127 inner nodes above
	// them.
	for layer_depth in [1usize, 4, 7] {
		let (and, bmul) = cost(|builder| {
			// A public root claim over a layer of witness digests, as a verifier holds them.
			let root: Digest = array::from_fn(|_| builder.add_inout());
			let layer = digest_wires(builder, 1usize << layer_depth);
			verify_layer(builder, root, &layer);
		});
		println!("verify_layer layer={layer_depth}: AND={and} BMUL={bmul}");

		// Folding is pure hashing, with no pair to order, so no select gate can appear.
		assert_eq!(bmul, 0, "the fold uses no select gates");

		// Every node but the top one has a neighbour at its own level to share a core with.
		let n_nodes = (1usize << layer_depth) - 1;
		// A two-digest layer has only the top node, so its average is not a fair comparison.
		if n_nodes > 1 {
			assert!(
				and / n_nodes <= PAIRED_NODE_AND + PAIRED_NODE_AND / 8,
				"AND per node at layer depth {layer_depth} is {}",
				and / n_nodes
			);
		}
	}
}
