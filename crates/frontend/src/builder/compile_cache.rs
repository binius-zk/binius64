// Copyright 2026 The Binius Developers

//! Byte-serialized compilation cache: skip the optimization passes and constraint building when
//! the same circuit is compiled again.
//!
//! [`CircuitBuilder::build`](super::CircuitBuilder::build) spends nearly all of its time in the
//! optimization passes and constraint building; constructing the gate graph itself is cheap. For
//! a fixed circuit compiled repeatedly — a prover service that starts one fresh process per
//! proof, say — that cost recurs with an unchanging result. This module serializes the *outputs*
//! of compilation (the constraint system, value-vector layout, wire mapping, inout list and
//! evaluation bytecode), so a later process can reconstruct the gate graph — re-registering
//! every hint handler along the way — and assemble the same [`Circuit`](crate::Circuit) by
//! deserializing the rest. The usage example lives on
//! [`CircuitBuilder::try_build_with_compile_cache`](super::CircuitBuilder::try_build_with_compile_cache),
//! the public face of this module.
//!
//! A cache is only valid for the exact builder state that produced it: the same code building
//! the same gate graph under the same options, so that the cached wire mapping and bytecode
//! address the same wires. The loader recomputes a digest of its freshly built state — wire
//! kinds, constant values, gate bodies, gate wires, immediates, dimensions, per-gate path
//! references, the builder options and the force-committed set — and refuses a cache whose
//! recorded digest differs; a cache written by a different published version of this crate is
//! refused outright, since the digest cannot see changes to the passes or the lowering. Within
//! one published version the header cannot tell two builds apart, so regenerate the cache
//! whenever the code is rebuilt — the natural arrangement is writing it from a build script,
//! which ties the cache's lifetime to the binary's. Under that discipline, drift between the
//! builder's state and the cache is an error rather than a miscompiled circuit, and a trailing
//! checksum over the serialized payload refuses a cache corrupted at rest or in transit. Path and
//! assertion names are not digested: they are diagnostic metadata, and the loaded circuit reports
//! failures under its own construction's names. The digest is an integrity check against accidental
//! divergence, not a security boundary: a cache file is trusted input, on the same footing as the
//! binary that loads it. Hint handlers are closures and are never serialized; they come from
//! the fresh builder's registry, and hint ids are name hashes
//! ([`hint_id_of`](crate::ir::hints::hint_id_of)), so the cached bytecode's references resolve
//! against it. The name hash is stable within a toolchain; a toolchain that changed it would
//! also change the digest of any hint-calling graph, refusing the cache rather than
//! mis-resolving a handler.

use binius_core::constraint_system::{ConstraintSystem, ValueIndex, ValueVecLayout};
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::Buf;
use cranelift_entity::{EntityRef, EntitySet, SecondaryMap};

use super::Options;
use crate::ir::{GateBody, GateGraph, Wire, WireKind};

/// Format magic, so a foreign or truncated file fails loudly instead of deserializing garbage.
/// Little-endian serialization makes the file begin with the literal bytes `B64C`.
const MAGIC: u32 = u32::from_le_bytes(*b"B64C");

/// Version of the cache layout below. Bump on any change to the format; there is no
/// cross-version compatibility, by design — a cache lives and dies with the build that wrote it.
const CACHE_VERSION: u32 = 1;

/// Why a compilation cache was refused.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CompileCacheError {
	/// The bytes do not decode as a compilation cache of this version.
	#[error("compilation cache: {0}")]
	Deserialize(#[from] SerializationError),
	/// The cache was written by a different published version of this crate. The digest cannot
	/// see changes to the passes or the lowering, so a version mismatch is refused outright.
	/// (Two builds of the same published version share a header; see the module docs.)
	#[error(
		"compilation cache was written by binius-frontend {cached}, this is {current}; \
		 rebuild the cache"
	)]
	VersionMismatch {
		cached: String,
		current: &'static str,
	},
	/// The bytes decode, but were written for a different builder state than this one —
	/// different construction code, parameters, options or force-committed wires.
	#[error(
		"compilation cache was written for a different builder state \
		 (cached {cached_wires} wires / {cached_gates} gates, digest {cached_digest:#018x}; \
		 built {built_wires} wires / {built_gates} gates, digest {built_digest:#018x})"
	)]
	StateMismatch {
		cached_wires: usize,
		cached_gates: usize,
		cached_digest: u64,
		built_wires: usize,
		built_gates: usize,
		built_digest: u64,
	},
	/// The bytes decode and pass the payload checksum, but the constraint system they carry
	/// fails its own validation. On the build path this state is unreachable; reaching it here
	/// means a deliberately altered cache file.
	#[error("compilation cache carries an invalid constraint system: {0}")]
	Validation(#[from] binius_core::error::ConstraintSystemError),
}

/// The deserialized compilation outputs, ready for
/// [`Circuit::new`](crate::Circuit)-style assembly.
pub(super) struct CompileCacheParts {
	/// Wire count of the constructed (pre-pass) graph — what the loading builder holds.
	pub n_wires: usize,
	/// Gate count of the constructed (pre-pass) graph.
	pub n_gates: usize,
	/// [`graph_digest`] of the constructed state: graph, options and force-committed set.
	pub digest: u64,
	pub constraint_system: ConstraintSystem,
	pub value_vec_layout: ValueVecLayout,
	pub wire_mapping: SecondaryMap<Wire, ValueIndex>,
	pub inout: Vec<Wire>,
	pub bytecode: Vec<u8>,
	pub n_eval_insn: usize,
	pub scratch_peak_live: usize,
	pub scratch_pooled: bool,
}

/// Fold the byte stream into one 64-bit word, one multiply per 8 bytes, so checksumming a
/// multi-gigabyte cache runs at memory speed. An integrity tripwire for the serialized payload
/// (bit flips at rest or in transit), same trust stance as the digest.
fn payload_checksum(bytes: &[u8]) -> u64 {
	const PRIME: u64 = 0x0000_0100_0000_01b3;
	let mut h = 0xcbf2_9ce4_8422_2325_u64 ^ bytes.len() as u64;
	let mut chunks = bytes.chunks_exact(8);
	for chunk in &mut chunks {
		h = (h ^ u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"))).wrapping_mul(PRIME);
	}
	let mut tail = [0u8; 8];
	tail[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
	(h ^ u64::from_le_bytes(tail)).wrapping_mul(PRIME)
}

/// FNV-1a over one 64-bit word. Stable by construction (no dependency on `Hash` impls or
/// hasher internals), which is what lets the digest compare across processes and builds of the
/// same code.
#[inline]
const fn fnv1a(hash: u64, word: u64) -> u64 {
	const PRIME: u64 = 0x0000_0100_0000_01b3;
	let mut h = hash;
	let bytes = word.to_le_bytes();
	let mut i = 0;
	while i < 8 {
		h ^= bytes[i] as u64;
		h = h.wrapping_mul(PRIME);
		i += 1;
	}
	h
}

/// Structural digest of the compilation inputs: every wire's kind (with constant values); every
/// gate's body, wires, immediates, dimensions and path references; the builder options; and the
/// force-committed set. Equal digests mean the same compilation for the cache's purposes: the
/// cached wire mapping and bytecode address wires and gates by these indices, the options decide
/// which passes produced the cached outputs, the force-committed set decides which wires stay
/// readable from a witness filler, and the per-gate path references are the ids the cached
/// bytecode symbolicates assertion failures with.
pub(super) fn graph_digest(
	graph: &GateGraph,
	opts: &Options,
	force_committed: &EntitySet<Wire>,
) -> u64 {
	const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
	let mut h = OFFSET_BASIS;
	// Exhaustive destructuring, no `..`: a field added to `Options` fails to compile here
	// rather than silently escaping the digest.
	let Options {
		enable_gate_fusion,
		enable_constant_propagation,
		enable_common_subexpression_elimination,
		enable_dead_code_elimination,
		enable_algebraic_folding,
		enable_scratch_pooling,
		enable_zero_propagation,
	} = *opts;
	for flag in [
		enable_gate_fusion,
		enable_constant_propagation,
		enable_common_subexpression_elimination,
		enable_dead_code_elimination,
		enable_algebraic_folding,
		enable_scratch_pooling,
		enable_zero_propagation,
	] {
		h = fnv1a(h, flag as u64);
	}
	h = fnv1a(h, force_committed.iter().count() as u64);
	for wire in force_committed.iter() {
		h = fnv1a(h, wire.index() as u64);
	}
	h = fnv1a(h, graph.wires.len() as u64);
	for (_, kind) in graph.wires.iter() {
		match kind {
			WireKind::Constant(word) => {
				h = fnv1a(h, 1);
				h = fnv1a(h, word.as_u64());
			}
			WireKind::Inout => h = fnv1a(h, 2),
			WireKind::Witness => h = fnv1a(h, 3),
			WireKind::Internal => h = fnv1a(h, 4),
			WireKind::Scratch => h = fnv1a(h, 5),
		}
	}
	h = fnv1a(h, graph.gates.len() as u64);
	for (gate_id, gate) in graph.gates.iter() {
		h = fnv1a(h, graph.gate_origin[gate_id].index() as u64);
		h = fnv1a(h, graph.assertion_names[gate_id].index() as u64);
		// Exhaustive destructuring, no `..`: a field added to `GateData` fails to compile
		// here rather than silently escaping the digest.
		let crate::ir::GateData {
			body,
			wires,
			immediates,
			dimensions,
		} = gate;
		match body {
			GateBody::Op(opcode) => {
				h = fnv1a(h, 1);
				h = fnv1a(h, *opcode as u64);
			}
			GateBody::Hint(hint_id) => {
				h = fnv1a(h, 2);
				h = fnv1a(h, *hint_id as u64);
			}
		}
		h = fnv1a(h, wires.len() as u64);
		for wire in wires {
			h = fnv1a(h, wire.index() as u64);
		}
		h = fnv1a(h, immediates.len() as u64);
		for imm in immediates {
			h = fnv1a(h, *imm as u64);
		}
		h = fnv1a(h, dimensions.len() as u64);
		for dim in dimensions {
			h = fnv1a(h, *dim as u64);
		}
	}
	h
}

/// The compilation outputs by reference, named at the call site so a transposed argument is a
/// compile error rather than a corrupt cache. The write side of [`CompileCacheParts`]'s
/// `deserialize`; field order there must match [`write()`] exactly.
pub(super) struct CacheWrite<'a> {
	pub n_wires: usize,
	pub n_gates: usize,
	pub digest: u64,
	pub n_mapped: usize,
	pub constraint_system: &'a ConstraintSystem,
	pub value_vec_layout: &'a ValueVecLayout,
	pub wire_mapping: &'a SecondaryMap<Wire, ValueIndex>,
	pub inout: &'a [Wire],
	pub bytecode: &'a [u8],
	pub n_eval_insn: usize,
	pub scratch_peak_live: usize,
	pub scratch_pooled: bool,
}

/// Serialize the compilation outputs, ending with a checksum of every preceding byte.
pub(super) fn write(parts: &CacheWrite<'_>, out: &mut Vec<u8>) -> Result<(), SerializationError> {
	let start = out.len();
	let mut write_buf = out;
	MAGIC.serialize(&mut write_buf)?;
	CACHE_VERSION.serialize(&mut write_buf)?;
	env!("CARGO_PKG_VERSION").serialize(&mut write_buf)?;
	parts.n_wires.serialize(&mut write_buf)?;
	parts.n_gates.serialize(&mut write_buf)?;
	parts.digest.serialize(&mut write_buf)?;
	parts.n_mapped.serialize(&mut write_buf)?;
	parts.constraint_system.serialize(&mut write_buf)?;
	parts.value_vec_layout.n_const.serialize(&mut write_buf)?;
	parts.value_vec_layout.n_inout.serialize(&mut write_buf)?;
	parts.value_vec_layout.n_witness.serialize(&mut write_buf)?;
	parts
		.value_vec_layout
		.n_internal
		.serialize(&mut write_buf)?;
	parts.value_vec_layout.n_scratch.serialize(&mut write_buf)?;
	// The wire mapping is dense over the post-pass graph's wires: one entry per wire, in
	// index order.
	for i in 0..parts.n_mapped {
		parts.wire_mapping[Wire::new(i)].serialize(&mut write_buf)?;
	}
	parts.inout.len().serialize(&mut write_buf)?;
	for wire in parts.inout {
		(wire.index() as u32).serialize(&mut write_buf)?;
	}
	parts.bytecode.serialize(&mut write_buf)?;
	parts.n_eval_insn.serialize(&mut write_buf)?;
	parts.scratch_peak_live.serialize(&mut write_buf)?;
	parts.scratch_pooled.serialize(&mut write_buf)?;
	let checksum = payload_checksum(&write_buf[start..]);
	checksum.serialize(write_buf)
}

impl CompileCacheParts {
	pub(super) fn deserialize(bytes: &[u8]) -> Result<Self, CompileCacheError> {
		// Peek the magic before checksumming, so pointing the loader at some other file reads
		// as "not a compilation cache" rather than a checksum failure over megabytes of the
		// wrong bytes.
		if bytes.len() < 4 || bytes[..4] != MAGIC.to_le_bytes() {
			return Err(SerializationError::InvalidConstruction {
				name: "compile_cache::MAGIC",
			}
			.into());
		}
		// The last 8 bytes checksum everything before them; verify before parsing, so a bit
		// flip anywhere in the payload is refused at the door rather than loading a subtly
		// wrong circuit.
		let Some(body_len) = bytes.len().checked_sub(8) else {
			return Err(SerializationError::InvalidConstruction {
				name: "compile_cache::checksum",
			}
			.into());
		};
		let (body, trailer) = bytes.split_at(body_len);
		let stored = u64::from_le_bytes(trailer.try_into().expect("8-byte trailer"));
		if payload_checksum(body) != stored {
			return Err(SerializationError::InvalidConstruction {
				name: "compile_cache::checksum",
			}
			.into());
		}
		let mut read_buf = body;
		let magic = u32::deserialize(&mut read_buf)?;
		if magic != MAGIC {
			return Err(SerializationError::InvalidConstruction {
				name: "compile_cache::MAGIC",
			}
			.into());
		}
		let version = u32::deserialize(&mut read_buf)?;
		if version != CACHE_VERSION {
			return Err(SerializationError::InvalidConstruction {
				name: "compile_cache::CACHE_VERSION",
			}
			.into());
		}
		let pkg_version = String::deserialize(&mut read_buf)?;
		if pkg_version != env!("CARGO_PKG_VERSION") {
			return Err(CompileCacheError::VersionMismatch {
				cached: pkg_version,
				current: env!("CARGO_PKG_VERSION"),
			});
		}
		let n_wires = usize::deserialize(&mut read_buf)?;
		let n_gates = usize::deserialize(&mut read_buf)?;
		let digest = u64::deserialize(&mut read_buf)?;
		let n_mapped = usize::deserialize(&mut read_buf)?;
		let constraint_system = ConstraintSystem::deserialize(&mut read_buf)?;
		let value_vec_layout = ValueVecLayout {
			n_const: usize::deserialize(&mut read_buf)?,
			n_inout: usize::deserialize(&mut read_buf)?,
			n_witness: usize::deserialize(&mut read_buf)?,
			n_internal: usize::deserialize(&mut read_buf)?,
			n_scratch: usize::deserialize(&mut read_buf)?,
		};
		// Scratch, like `value_vec_alloc`'s own default: a wire handle beyond the mapped range
		// must not alias a real word.
		let mut wire_mapping = SecondaryMap::with_default(ValueIndex::scratch(0));
		for i in 0..n_mapped {
			wire_mapping[Wire::new(i)] = ValueIndex::deserialize(&mut read_buf)?;
		}
		let n_inout = usize::deserialize(&mut read_buf)?;
		// A length field can't promise more than the buffer holds; check before preallocating,
		// so a truncated or corrupt file errors instead of reserving from a lie.
		if n_inout.saturating_mul(4) > read_buf.remaining() {
			return Err(SerializationError::InvalidConstruction {
				name: "compile_cache::n_inout",
			}
			.into());
		}
		let mut inout = Vec::with_capacity(n_inout);
		for _ in 0..n_inout {
			inout.push(Wire::new(u32::deserialize(&mut read_buf)? as usize));
		}
		let bytecode = Vec::<u8>::deserialize(&mut read_buf)?;
		let n_eval_insn = usize::deserialize(&mut read_buf)?;
		let scratch_peak_live = usize::deserialize(&mut read_buf)?;
		let scratch_pooled = bool::deserialize(&mut read_buf)?;
		if read_buf.remaining() != 0 {
			return Err(SerializationError::InvalidConstruction {
				name: "compile_cache::trailing",
			}
			.into());
		}
		Ok(CompileCacheParts {
			n_wires,
			n_gates,
			digest,
			constraint_system,
			value_vec_layout,
			wire_mapping,
			inout,
			bytecode,
			n_eval_insn,
			scratch_peak_live,
			scratch_pooled,
		})
	}
}
