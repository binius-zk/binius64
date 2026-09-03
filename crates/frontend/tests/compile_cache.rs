// Copyright 2026 The Binius Developers

//! The compilation cache round-trips: a circuit assembled from its cache behaves exactly like
//! the circuit a full build produces, and a cache written for a different builder state, by a
//! different crate version, or corrupted in transit is refused.

use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, CompileCacheError, Hint, Options, Wire};
use binius_utils::serialization::SerializeBytes;

/// A witness-time computation the cache cannot serialize: hint handlers are closures, so the
/// loading builder must re-register them and the cached bytecode must resolve against the fresh
/// registry by stable hint id.
struct SplitHint;

impl Hint for SplitHint {
	const NAME: &'static str = "compile_cache_test::split";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(1, 2)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let x = inputs[0].as_u64();
		outputs[0] = Word::from_u64(x & 0xFFFF_FFFF);
		outputs[1] = Word::from_u64(x >> 32);
	}
}

/// A small circuit covering every wire kind and both gate bodies: constants, inouts, witnesses,
/// gate-created internal wires, a hint call, and assertions that commit the hint's outputs.
/// Returns the wires a test reads or writes.
fn build_test_circuit(b: &CircuitBuilder) -> (Wire, Wire, Wire, Wire) {
	let sub = b.subcircuit("cache_test");
	let x = sub.add_inout();
	let y = sub.add_witness();
	let k = sub.add_constant_64(0xDEAD_BEEF_CAFE_F00D);
	let masked = sub.band(x, k);
	let sum = sub.iadd_32(masked, y);
	let halves = sub.call_hint(SplitHint, &[], &[sum]);
	let (lo, hi) = (halves[0], halves[1]);
	// Recombine the hint's outputs and pin them to the value they split, so the hint wires are
	// committed and a wrong hint execution fails witness population.
	let shifted = sub.shl(hi, 32);
	let recombined = sub.bxor(lo, shifted);
	sub.assert_eq("split_recombines", recombined, sum);
	// Promote a gate-created wire into the public interface, so the cached inout list carries
	// a promoted wire as well as declared ones.
	sub.mark_inout(recombined);
	drop(sub);
	(x, y, lo, hi)
}

/// Fill the two input wires, populate, and read back the hint's output wires.
fn populate(circuit: &Circuit, wires: (Wire, Wire, Wire, Wire), x: u64, y: u64) -> (u64, u64) {
	let (x_wire, y_wire, lo, hi) = wires;
	let mut filler = circuit.new_witness_filler();
	filler[x_wire] = Word::from_u64(x);
	filler[y_wire] = Word::from_u64(y);
	circuit
		.populate_wire_witness(&mut filler)
		.expect("population succeeds");
	(filler[lo].as_u64(), filler[hi].as_u64())
}

/// Mirror of `compile_cache::payload_checksum`, for tests that splice a field and re-seal the
/// trailer so the header checks behind the checksum stay individually pinned.
fn refresh_checksum(bytes: &mut [u8]) {
	const PRIME: u64 = 0x0000_0100_0000_01b3;
	let body_len = bytes.len() - 8;
	let body = &bytes[..body_len];
	let mut h = 0xcbf2_9ce4_8422_2325_u64 ^ body.len() as u64;
	let mut chunks = body.chunks_exact(8);
	for chunk in &mut chunks {
		h = (h ^ u64::from_le_bytes(chunk.try_into().unwrap())).wrapping_mul(PRIME);
	}
	let mut tail = [0u8; 8];
	tail[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
	h = (h ^ u64::from_le_bytes(tail)).wrapping_mul(PRIME);
	bytes[body_len..].copy_from_slice(&h.to_le_bytes());
}

fn cs_bytes(circuit: &Circuit) -> Vec<u8> {
	let mut buf = Vec::new();
	circuit
		.constraint_system()
		.serialize(&mut buf)
		.expect("serializing a constraint system to a Vec cannot fail");
	buf
}

#[test]
fn cached_build_matches_full_build() {
	let full_builder = CircuitBuilder::new();
	let full_wires = build_test_circuit(&full_builder);
	let (full, cache) = full_builder
		.try_build_with_compile_cache()
		.expect("full build");

	let cached_builder = CircuitBuilder::new();
	let cached_wires = build_test_circuit(&cached_builder);
	let cached = cached_builder
		.try_build_from_compile_cache(&cache)
		.expect("the same construction accepts its own cache");

	// Requesting a cache must not perturb the build: a plain build of the same construction
	// produces the identical constraint system.
	let plain_builder = CircuitBuilder::new();
	build_test_circuit(&plain_builder);
	let plain = plain_builder.try_build().expect("plain build");
	assert_eq!(cs_bytes(&full), cs_bytes(&plain));

	// The constraint systems are byte-identical, so anything proven against one verifies
	// against the other.
	assert_eq!(cs_bytes(&full), cs_bytes(&cached));
	// The public interface is the same wires in the same order.
	assert_eq!(full.inout(), cached.inout());
	assert_eq!(full.n_eval_insn(), cached.n_eval_insn());

	// Witness generation agrees, hint execution included.
	let inputs = (0x0123_4567_89AB_CDEF, 0x0000_0000_1111_2222);
	assert_eq!(
		populate(&full, full_wires, inputs.0, inputs.1),
		populate(&cached, cached_wires, inputs.0, inputs.1)
	);
}

#[test]
fn optimized_away_gates_still_round_trip() {
	// Common-subexpression elimination rewrites this graph: the two identical `band` gates
	// collapse, and the survivors' operand wires are rewritten to canonical form. The cache
	// fingerprint is taken before the passes run, so the loading builder — which runs no
	// passes and still holds both gates — matches the cache all the same.
	let build = || {
		let b = CircuitBuilder::new();
		let x = b.add_inout();
		let out_io = b.add_inout();
		let k1 = b.add_constant_64(0xFF00_FF00_FF00_FF00);
		let dup_a = b.band(x, k1);
		let dup_b = b.band(x, k1);
		let mixed = b.iadd_32(dup_a, dup_b);
		let out = b.bxor(mixed, dup_a);
		b.assert_eq("out", out_io, out);
		(b, x, out_io)
	};

	let (full_builder, _, _) = build();
	let (full, cache) = full_builder
		.try_build_with_compile_cache()
		.expect("full build");

	let (cached_builder, x, out_io) = build();
	let cached = cached_builder
		.try_build_from_compile_cache(&cache)
		.expect("a CSE-rewritten circuit accepts its own cache");

	assert_eq!(cs_bytes(&full), cs_bytes(&cached));

	let input = 0x1234_5678_9ABC_DEF0_u64;
	let masked = input & 0xFF00_FF00_FF00_FF00;
	// `iadd_32` adds the two 32-bit halves independently, discarding each carry-out.
	let hi = ((masked >> 32) as u32).wrapping_add((masked >> 32) as u32) as u64;
	let lo = (masked as u32).wrapping_add(masked as u32) as u64;
	let expected = ((hi << 32) | lo) ^ masked;
	let mut filler = cached.new_witness_filler();
	filler[x] = Word::from_u64(input);
	filler[out_io] = Word::from_u64(expected);
	cached
		.populate_wire_witness(&mut filler)
		.expect("the deduplicated output pins correctly");

	let mut bad = cached.new_witness_filler();
	bad[x] = Word::from_u64(input);
	bad[out_io] = Word::from_u64(expected ^ 1);
	cached
		.populate_wire_witness(&mut bad)
		.expect_err("a wrong output must fail the assertion through the cached bytecode");
}

#[test]
fn cache_for_a_different_graph_is_refused() {
	let builder = CircuitBuilder::new();
	build_test_circuit(&builder);
	let (_, cache) = builder.try_build_with_compile_cache().expect("full build");

	// One extra gate: the counts and digest both move.
	let drifted = CircuitBuilder::new();
	let wires = build_test_circuit(&drifted);
	let _ = drifted.band(wires.0, wires.1);
	match drifted.try_build_from_compile_cache(&cache) {
		Err(CompileCacheError::StateMismatch { .. }) => {}
		Err(other) => panic!("expected StateMismatch, got {other}"),
		Ok(_) => panic!("a drifted graph accepted the cache"),
	}
}

#[test]
fn same_shape_different_constant_is_refused() {
	// Same wire and gate counts; only a constant's value differs. Only the digest catches it.
	let build_with = |c: u64| {
		let b = CircuitBuilder::new();
		let x = b.add_inout();
		let k = b.add_constant_64(c);
		let y = b.band(x, k);
		b.assert_eq("pin", y, y);
		b
	};
	let (_, cache) = build_with(1)
		.try_build_with_compile_cache()
		.expect("full build");
	match build_with(2).try_build_from_compile_cache(&cache) {
		Err(CompileCacheError::StateMismatch { .. }) => {}
		Err(other) => panic!("expected StateMismatch, got {other}"),
		Ok(_) => panic!("a different constant accepted the cache"),
	}
}

#[test]
fn corrupt_bytes_are_refused() {
	let builder = CircuitBuilder::new();
	build_test_circuit(&builder);
	let (_, cache) = builder.try_build_with_compile_cache().expect("full build");

	for bytes in [&[][..], &cache[..4], &cache[..cache.len() - 1]] {
		let fresh = CircuitBuilder::new();
		build_test_circuit(&fresh);
		match fresh.try_build_from_compile_cache(bytes) {
			Err(CompileCacheError::Deserialize(_)) => {}
			Err(other) => panic!("expected a deserialize error, got {other}"),
			Ok(_) => panic!("corrupt bytes accepted"),
		}
	}

	// A single bit flipped anywhere in the payload — mid-file and in the trailing checksum
	// itself — is refused by the checksum, not loaded as a subtly wrong circuit.
	for flip_at in [cache.len() / 2, cache.len() - 1] {
		let mut bytes = cache.clone();
		bytes[flip_at] ^= 0x01;
		let fresh = CircuitBuilder::new();
		build_test_circuit(&fresh);
		match fresh.try_build_from_compile_cache(&bytes) {
			Err(CompileCacheError::Deserialize(_)) => {}
			Err(other) => panic!("expected a deserialize error, got {other}"),
			Ok(_) => panic!("a flipped payload bit was accepted"),
		}
	}
}

#[test]
fn options_change_is_refused() {
	let (_, cache) = {
		let b = CircuitBuilder::new();
		build_test_circuit(&b);
		b.try_build_with_compile_cache().expect("full build")
	};
	// Scratch pooling changes nothing about construction — the gate graph is byte-identical —
	// but it decides the compiled layout and which wires a witness filler may read, so the
	// cache must be refused on the option alone.
	let mut opts = Options::default();
	opts.enable_scratch_pooling = false;
	let loader = CircuitBuilder::with_opts(opts);
	build_test_circuit(&loader);
	match loader.try_build_from_compile_cache(&cache) {
		Err(CompileCacheError::StateMismatch { .. }) => {}
		Err(other) => panic!("expected StateMismatch, got {other}"),
		Ok(_) => panic!("different builder options accepted the cache"),
	}
}

#[test]
fn force_commit_drift_is_refused() {
	let (_, cache) = {
		let b = CircuitBuilder::new();
		build_test_circuit(&b);
		b.try_build_with_compile_cache().expect("full build")
	};
	// Same graph, byte for byte; the only change is which wires must stay readable — which
	// changes the compiled layout, so the cache must be refused.
	let loader = CircuitBuilder::new();
	let (_, _, lo, _) = build_test_circuit(&loader);
	loader.force_commit(lo);
	match loader.try_build_from_compile_cache(&cache) {
		Err(CompileCacheError::StateMismatch { .. }) => {}
		Err(other) => panic!("expected StateMismatch, got {other}"),
		Ok(_) => panic!("a changed force-committed set accepted the cache"),
	}
}

#[test]
fn interned_constants_round_trip() {
	// With constant propagation enabled the passes fold gates and can intern constants the
	// construction never made, so the cached wire mapping runs past the constructed graph's
	// wire count — the one case where the mapping length and the fingerprint's wire count
	// differ.
	let opts = || {
		let mut opts = Options::default();
		opts.enable_constant_propagation = true;
		opts
	};
	let build = |b: &CircuitBuilder| {
		let x = b.add_inout();
		let out_io = b.add_inout();
		let k1 = b.add_constant_64(0x1111_2222_3333_4444);
		let k2 = b.add_constant_64(0x0F0F_0F0F_0F0F_0F0F);
		let folded = b.band(k1, k2);
		let out = b.bxor(x, folded);
		b.assert_eq("out", out_io, out);
		(x, out_io)
	};

	let full_builder = CircuitBuilder::with_opts(opts());
	build(&full_builder);
	let (full, cache) = full_builder
		.try_build_with_compile_cache()
		.expect("full build");

	let loader = CircuitBuilder::with_opts(opts());
	let (x, out_io) = build(&loader);
	let cached = loader
		.try_build_from_compile_cache(&cache)
		.expect("a constant-folding build accepts its own cache");

	assert_eq!(cs_bytes(&full), cs_bytes(&cached));
	let input = 0xAAAA_BBBB_CCCC_DDDD_u64;
	let expected = input ^ (0x1111_2222_3333_4444 & 0x0F0F_0F0F_0F0F_0F0F);
	let mut filler = cached.new_witness_filler();
	filler[x] = Word::from_u64(input);
	filler[out_io] = Word::from_u64(expected);
	cached
		.populate_wire_witness(&mut filler)
		.expect("the folded constant evaluates correctly from the cache");
}

#[test]
fn wrong_magic_and_version_are_refused() {
	let builder = CircuitBuilder::new();
	build_test_circuit(&builder);
	let (_, cache) = builder.try_build_with_compile_cache().expect("full build");

	// Byte 0 is the magic, byte 4 the format version; each flip is re-sealed under a fresh
	// checksum, and the error's field name is asserted, so the header check itself — not the
	// checksum, and not a drifted `refresh_checksum` mirror — does the refusing.
	use binius_utils::serialization::SerializationError;
	for (corrupt_at, expected) in [
		(0usize, "compile_cache::MAGIC"),
		(4, "compile_cache::CACHE_VERSION"),
	] {
		let mut bytes = cache.clone();
		bytes[corrupt_at] ^= 0xFF;
		refresh_checksum(&mut bytes);
		let fresh = CircuitBuilder::new();
		build_test_circuit(&fresh);
		match fresh.try_build_from_compile_cache(&bytes) {
			Err(CompileCacheError::Deserialize(SerializationError::InvalidConstruction {
				name,
			})) => {
				assert_eq!(name, expected);
			}
			Err(other) => panic!("expected the {expected} check to refuse, got {other}"),
			Ok(_) => panic!("corrupt header accepted"),
		}
	}

	// Trailing garbage after a valid cache is refused too.
	let mut bytes = cache;
	bytes.push(0);
	let fresh = CircuitBuilder::new();
	build_test_circuit(&fresh);
	match fresh.try_build_from_compile_cache(&bytes) {
		Err(CompileCacheError::Deserialize(_)) => {}
		Err(other) => panic!("expected a deserialize error, got {other}"),
		Ok(_) => panic!("trailing bytes accepted"),
	}
}

#[test]
fn different_crate_version_is_refused() {
	let builder = CircuitBuilder::new();
	build_test_circuit(&builder);
	let (_, cache) = builder.try_build_with_compile_cache().expect("full build");

	// Header layout: magic (4) ‖ format version (4) ‖ crate version as len-prefixed string.
	// Bump the last character of the crate version in place and re-seal the checksum, so the
	// version check itself — not the checksum — does the refusing.
	let mut bytes = cache;
	let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
	let last = 12 + len - 1;
	bytes[last] = bytes[last].wrapping_add(1);
	refresh_checksum(&mut bytes);
	let fresh = CircuitBuilder::new();
	build_test_circuit(&fresh);
	match fresh.try_build_from_compile_cache(&bytes) {
		Err(CompileCacheError::VersionMismatch { cached, current }) => {
			assert_eq!(current, env!("CARGO_PKG_VERSION"));
			assert_ne!(cached, current);
		}
		Err(other) => panic!("expected VersionMismatch, got {other}"),
		Ok(_) => panic!("a different crate version accepted the cache"),
	}
}

#[test]
fn renamed_paths_are_accepted() {
	// Same structure, different subcircuit name: path and assertion names are diagnostic
	// metadata, deliberately outside the digest, so the cache is accepted and failures report
	// under the loader's own names.
	let build_named = |name: &str| {
		let b = CircuitBuilder::new();
		let sub = b.subcircuit(name);
		let x = sub.add_inout();
		let out_io = sub.add_inout();
		let k = sub.add_constant_64(0x5555_5555_5555_5555);
		let out = sub.band(x, k);
		sub.assert_eq("pin", out_io, out);
		drop(sub);
		(b, x, out_io)
	};

	let (writer, _, _) = build_named("original");
	let (_, cache) = writer.try_build_with_compile_cache().expect("full build");

	let (loader, x, out_io) = build_named("renamed");
	let cached = loader
		.try_build_from_compile_cache(&cache)
		.expect("a renamed but structurally identical construction accepts the cache");

	let mut bad = cached.new_witness_filler();
	bad[x] = Word::from_u64(0xFFFF_0000_FFFF_0000);
	bad[out_io] = Word::from_u64(0);
	let err = cached
		.populate_wire_witness(&mut bad)
		.expect_err("the pinned output is wrong");
	let msg = format!("{err}");
	assert!(
		msg.contains("renamed.pin"),
		"assertion failures report under the loader's names, got: {msg}"
	);
}

#[test]
fn scratch_peak_live_round_trips() {
	let full_builder = CircuitBuilder::new();
	build_test_circuit(&full_builder);
	let (full, cache) = full_builder
		.try_build_with_compile_cache()
		.expect("full build");

	let loader = CircuitBuilder::new();
	build_test_circuit(&loader);
	let cached = loader
		.try_build_from_compile_cache(&cache)
		.expect("round trip");
	assert_eq!(full.scratch_peak_live(), cached.scratch_peak_live());
}
