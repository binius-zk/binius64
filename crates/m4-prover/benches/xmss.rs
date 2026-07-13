// Copyright 2026 The Binius Developers
//! End-to-end M4 proving throughput for XMSS signature verification.
//!
//! One full XMSS verification runs per instance, and the whole batch is proved together, so the
//! throughput is signatures per second. The circuit is all-BLAKE3 and MUL-free, which the batched
//! M4 prover requires.
//!
//! A single valid signature is generated once and replicated across every instance. Proving cost
//! is data-independent, so this measures the same throughput as distinct signatures while keeping
//! setup to one nonce grind.
//!
//! Environment overrides:
//! - `LOG_INSTANCES`: base-2 log of the signature count (default 13 = 8192).
//! - `LOG_INV_RATE`: base-2 log of the inverse Reed-Solomon rate (default 1 = rate 1/2).
//! - `WOTS_SPEC`: Winternitz spec, 1 (w=2) or 2 (w=4) (default 1).
//! - `XMSS_TREE_HEIGHT`: Merkle tree height; the tree has 2^height leaves (default 13).

#[path = "utils/m4_bench.rs"]
mod m4_bench;

use std::{array, env};

use binius_circuits::hash_based_sig::{
	winternitz_ots::{NONCE_WIRES_COUNT, WinternitzSpec},
	witness_utils::ValidatorSignatureData,
	xmss::{XmssSignature, circuit_xmss},
};
use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::BatchWitnessFiller;
use criterion::{Criterion, criterion_group, criterion_main};
use m4_bench::bench_m4_proving;
use rand::prelude::*;

/// Base-2 logarithm of the signature count: 2^13 = 8192 signatures.
const DEFAULT_LOG_INSTANCES: usize = 13;

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching the hash benches.
const DEFAULT_LOG_INV_RATE: usize = 1;

/// Merkle tree height: the tree has 2^height leaves, one per epoch.
const DEFAULT_TREE_HEIGHT: usize = 13;

/// One signature verification per instance: the throughput count is signatures per second.
const SIGNATURES_PER_INSTANCE: u64 = 1;

/// The witness input wires of one XMSS verification instance.
///
/// Every wire is a witness input: the public data folds into the witness, since the batch table
/// forbids inout wires. The circuit's internal root-equality assertion keeps the computation alive.
struct XmssWires {
	/// The per-signer domain parameter, eight bytes per wire.
	domain_param: Vec<Wire>,
	/// The 32-byte message, four wires.
	message: Vec<Wire>,
	/// The committed Merkle root, four wires.
	root_hash: [Wire; 4],
	/// The signature witness: nonce, epoch, chain values, chain ends, and authentication path.
	signature: XmssSignature,
}

/// Packs little-endian bytes into 64-bit word wires, zero-filling any trailing wires.
///
/// Mirrors the single-instance packing helper, but writes into the batch filler one wire at a time.
fn pack_bytes_le(w: &mut BatchWitnessFiller<'_, '_>, wires: &[Wire], bytes: &[u8]) {
	assert!(bytes.len() <= wires.len() * 8, "bytes overflow the wire capacity");

	// Pack each 8-byte little-endian chunk into its word.
	for (&wire, chunk) in wires.iter().zip(bytes.chunks(8)) {
		let mut word = [0u8; 8];
		word[..chunk.len()].copy_from_slice(chunk);
		w[wire] = Word(u64::from_le_bytes(word));
	}
	// Zero any wire past the packed bytes.
	for &wire in &wires[bytes.len().div_ceil(8)..] {
		w[wire] = Word::ZERO;
	}
}

/// Builds one XMSS verification circuit with every input allocated as a witness.
///
/// With no inout wires the circuit is eligible for the batch witness table.
/// The verification asserts the recomputed root equals the committed root, so nothing is pruned.
fn build_xmss_circuit(spec: &WinternitzSpec, tree_height: usize) -> (Circuit, XmssWires) {
	let builder = CircuitBuilder::new();

	// The per-signer domain parameter is a byte string; each wire holds 8 of its bytes.
	let param_wire_count = spec.domain_param_len.div_ceil(8);
	let domain_param: Vec<Wire> = (0..param_wire_count)
		.map(|_| builder.add_witness())
		.collect();
	// The message digest is 32 bytes across four 64-bit wires.
	let message: Vec<Wire> = (0..4).map(|_| builder.add_witness()).collect();
	// The committed Merkle root is 32 bytes across four wires.
	let root_hash: [Wire; 4] = array::from_fn(|_| builder.add_witness());

	// The nonce feeds the message hash and fills four wires exactly.
	let nonce: Vec<Wire> = (0..NONCE_WIRES_COUNT)
		.map(|_| builder.add_witness())
		.collect();
	// The epoch is the signing leaf index within the tree.
	let epoch = builder.add_witness();
	// One chain value per Winternitz coordinate (the signature), each a four-wire digest.
	let signature_hashes: Vec<[Wire; 4]> = (0..spec.dimension())
		.map(|_| array::from_fn(|_| builder.add_witness()))
		.collect();
	// One chain end per coordinate (the one-time public key), each a four-wire digest.
	let public_key_hashes: Vec<[Wire; 4]> = (0..spec.dimension())
		.map(|_| array::from_fn(|_| builder.add_witness()))
		.collect();
	// One authentication-path node per tree level.
	let auth_path: Vec<[Wire; 4]> = (0..tree_height)
		.map(|_| array::from_fn(|_| builder.add_witness()))
		.collect();

	let signature = XmssSignature {
		nonce,
		epoch,
		signature_hashes,
		public_key_hashes,
		auth_path,
	};

	circuit_xmss(&builder, spec, &domain_param, &message, &signature, &root_hash);

	(
		builder.build(),
		XmssWires {
			domain_param,
			message,
			root_hash,
			signature,
		},
	)
}

/// The plaintext inputs and signature data of one valid XMSS signature.
struct SignatureData {
	/// The per-signer domain parameter bytes.
	param_bytes: Vec<u8>,
	/// The 32-byte message digest.
	message_bytes: [u8; 32],
	/// The signing leaf index within the tree.
	epoch: u32,
	/// The signed data: nonce, chain values, chain ends, Merkle root, and authentication path.
	data: ValidatorSignatureData,
}

/// Generates one valid XMSS signature: a random parameter, message, and epoch, then a signature.
///
/// The nonce grind and Merkle tree build happen here, once, outside any timed region.
fn generate_signature(spec: &WinternitzSpec, tree_height: usize) -> SignatureData {
	let mut rng = StdRng::seed_from_u64(0);

	let mut param_bytes = vec![0u8; spec.domain_param_len];
	rng.fill_bytes(&mut param_bytes);
	let mut message_bytes = [0u8; 32];
	rng.fill_bytes(&mut message_bytes);
	let epoch = rng.next_u32() % (1u32 << tree_height);

	let data = ValidatorSignatureData::generate(
		&mut rng,
		&param_bytes,
		&message_bytes,
		epoch,
		spec,
		tree_height,
	);

	SignatureData {
		param_bytes,
		message_bytes,
		epoch,
		data,
	}
}

/// Packs the signature into one instance's witness input wires.
fn fill_instance(wires: &XmssWires, sig: &SignatureData, w: &mut BatchWitnessFiller<'_, '_>) {
	// The public inputs, folded into the witness.
	pack_bytes_le(w, &wires.domain_param, &sig.param_bytes);
	pack_bytes_le(w, &wires.message, &sig.message_bytes);
	pack_bytes_le(w, &wires.root_hash, &sig.data.root);

	// The signature's nonce and epoch.
	pack_bytes_le(w, &wires.signature.nonce, &sig.data.nonce);
	w[wires.signature.epoch] = Word::from_u64(sig.epoch as u64);
	// One chain value per coordinate.
	for (dst, src) in wires
		.signature
		.signature_hashes
		.iter()
		.zip(&sig.data.signature_hashes)
	{
		pack_bytes_le(w, dst, src);
	}
	// One chain end per coordinate.
	for (dst, src) in wires
		.signature
		.public_key_hashes
		.iter()
		.zip(&sig.data.public_key_hashes)
	{
		pack_bytes_le(w, dst, src);
	}
	// One authentication-path node per tree level.
	for (dst, src) in wires.signature.auth_path.iter().zip(&sig.data.auth_path) {
		pack_bytes_le(w, dst, src);
	}
}

fn bench_prove_xmss(c: &mut Criterion) {
	// Batch size, code rate, spec, and tree height are environment-tunable for sweeping.
	let log_instances = env_usize("LOG_INSTANCES").unwrap_or(DEFAULT_LOG_INSTANCES);
	let log_inv_rate = env_usize("LOG_INV_RATE").unwrap_or(DEFAULT_LOG_INV_RATE);
	let tree_height = env_usize("XMSS_TREE_HEIGHT").unwrap_or(DEFAULT_TREE_HEIGHT);
	let spec = match env_usize("WOTS_SPEC") {
		Some(2) => WinternitzSpec::spec_2(),
		_ => WinternitzSpec::spec_1(),
	};

	// One circuit and one signature, replicated across the batch by the shared driver.
	let (circuit, wires) = build_xmss_circuit(&spec, tree_height);
	let sig = generate_signature(&spec, tree_height);

	bench_m4_proving(
		c,
		"xmss",
		&circuit,
		log_instances,
		log_inv_rate,
		SIGNATURES_PER_INSTANCE,
		|_, w| fill_instance(&wires, &sig, w),
	);
}

/// Reads a `usize` environment variable, returning `None` when unset or not a number.
fn env_usize(key: &str) -> Option<usize> {
	env::var(key).ok().and_then(|s| s.parse().ok())
}

criterion_group!(xmss, bench_prove_xmss);
criterion_main!(xmss);
