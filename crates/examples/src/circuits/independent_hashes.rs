// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

//! Batches of independent hash primitive evaluations.
//!
//! Each circuit proves `n` independent compression-function or permutation evaluations. The
//! primitive outputs are exposed as public inout wires and asserted equal to the gadget
//! outputs: dead-code elimination keeps only gates that feed assertions or public IO, so
//! without observing the outputs the entire hash computation would be pruned and the
//! benchmarks would measure an empty circuit.

use anyhow::{Result, ensure};
use binius_circuits::{
	blake3::{blake3_compress, ref_compress},
	keccak::permutation::{Permutation, State as KeccakState},
	sha256::{State as Sha256State, populate_message_block, sha256_compress},
};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use clap::Args;
use rand::prelude::*;

use super::utils::DEFAULT_RANDOM_SEED;
use crate::ExampleCircuit;

/// Default number of logical compression/permutation primitives for CLI use.
///
/// Benchmark harnesses override this with environment variables.
pub const DEFAULT_NUM_PRIMITIVES: usize = 16;

/// SHA-256 initial hash value (FIPS 180-4), matching [`Sha256State::iv`].
const SHA256_IV: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[derive(Debug, Clone, Args)]
pub struct PrimitiveParams {
	/// Number of independent primitive evaluations.
	#[arg(short = 'n', long, default_value_t = DEFAULT_NUM_PRIMITIVES)]
	pub num_primitives: usize,
}

#[derive(Debug, Clone, Args)]
pub struct Instance {
	/// RNG seed for deterministic witness values.
	#[arg(long)]
	pub seed: Option<u64>,
}

/// Independent SHA-256 compression evaluations.
///
/// Each circuit component runs `compress(IV, m_i)` for a fresh 64-byte message block and
/// asserts the resulting digest against a public inout copy.
pub struct IndependentSha256Compressions {
	compressions: Vec<Sha256Compression>,
}

struct Sha256Compression {
	block: [Wire; 16],
	digest: [Wire; 8],
}

impl ExampleCircuit for IndependentSha256Compressions {
	type Params = PrimitiveParams;
	type Instance = Instance;

	fn build(params: PrimitiveParams, builder: &mut CircuitBuilder) -> Result<Self> {
		ensure!(params.num_primitives > 0, "num_primitives must be positive");

		let compressions = (0..params.num_primitives)
			.map(|i| {
				let block = std::array::from_fn(|_| builder.add_witness());
				let out = sha256_compress(builder, Sha256State::iv(builder), block);
				let digest = std::array::from_fn(|_| builder.add_inout());
				// The raw 64-bit equality holds because honest witness population
				// zero-extends every 32-bit input and the gadget preserves empty high
				// halves.
				builder.assert_eq_v(format!("sha256_compression_out[{i}]"), out.0, digest);
				Sha256Compression { block, digest }
			})
			.collect();

		Ok(Self { compressions })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(instance.seed.unwrap_or(DEFAULT_RANDOM_SEED));
		for compression in &self.compressions {
			let block_bytes = next_block(&mut rng);
			populate_message_block(w, &compression.block, block_bytes);

			let mut state = SHA256_IV;
			sha2::block_api::compress256(&mut state, &[block_bytes]);
			for (wire, value) in compression.digest.iter().zip(state) {
				w[*wire] = Word(value as u64);
			}
		}
		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!("{}c", params.num_primitives))
	}
}

/// Independent BLAKE3 compression evaluations.
///
/// Each circuit component runs one independent BLAKE3 compression function and asserts the
/// resulting chaining value against a public inout copy.
pub struct IndependentBlake3Compressions {
	compressions: Vec<Blake3Compression>,
}

struct Blake3Compression {
	cv: [Wire; 8],
	block: [Wire; 16],
	counter: Wire,
	block_len: Wire,
	flags: Wire,
	out_cv: [Wire; 8],
}

impl ExampleCircuit for IndependentBlake3Compressions {
	type Params = PrimitiveParams;
	type Instance = Instance;

	fn build(params: PrimitiveParams, builder: &mut CircuitBuilder) -> Result<Self> {
		ensure!(params.num_primitives > 0, "num_primitives must be positive");

		let compressions = (0..params.num_primitives)
			.map(|i| {
				let cv = std::array::from_fn(|_| builder.add_witness());
				let block = std::array::from_fn(|_| builder.add_witness());
				let counter = builder.add_witness();
				let block_len = builder.add_witness();
				let flags = builder.add_witness();
				let out = blake3_compress(builder, cv, block, counter, block_len, flags);
				let out_cv = std::array::from_fn(|_| builder.add_inout());
				// See the SHA-256 note on raw 64-bit equality over 32-bit lanes.
				builder.assert_eq_v(format!("blake3_compression_out[{i}]"), out, out_cv);
				Blake3Compression {
					cv,
					block,
					counter,
					block_len,
					flags,
					out_cv,
				}
			})
			.collect();

		Ok(Self { compressions })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(instance.seed.unwrap_or(DEFAULT_RANDOM_SEED));
		for compression in &self.compressions {
			let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
			let block: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
			let counter = rng.next_u64();
			let block_len = rng.next_u32() % 65;
			let flags = rng.next_u32();

			for (wire, value) in compression.cv.iter().zip(cv) {
				w[*wire] = Word(value as u64);
			}
			for (wire, value) in compression.block.iter().zip(block) {
				w[*wire] = Word(value as u64);
			}
			w[compression.counter] = Word(counter);
			w[compression.block_len] = Word(block_len as u64);
			w[compression.flags] = Word(flags as u64);

			let expected = ref_compress(&cv, &block, counter, block_len, flags);
			for (wire, value) in compression.out_cv.iter().zip(expected) {
				w[*wire] = Word(value as u64);
			}
		}
		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!("{}c", params.num_primitives))
	}
}

/// Independent Keccak-f\[1600\] permutation evaluations.
///
/// Each circuit component runs one independent permutation and asserts the output state
/// against a public inout copy.
pub struct IndependentKeccakPermutations {
	permutations: Vec<KeccakPermutation>,
}

struct KeccakPermutation {
	permutation: Permutation,
	output: [Wire; 25],
}

impl ExampleCircuit for IndependentKeccakPermutations {
	type Params = PrimitiveParams;
	type Instance = Instance;

	fn build(params: PrimitiveParams, builder: &mut CircuitBuilder) -> Result<Self> {
		ensure!(params.num_primitives > 0, "num_primitives must be positive");

		let permutations = (0..params.num_primitives)
			.map(|i| {
				let words = std::array::from_fn(|_| builder.add_witness());
				let permutation = Permutation::new(builder, KeccakState { words });
				let output = std::array::from_fn(|_| builder.add_inout());
				builder.assert_eq_v(
					format!("keccak_permutation_out[{i}]"),
					permutation.output_state.words,
					output,
				);
				KeccakPermutation {
					permutation,
					output,
				}
			})
			.collect();

		Ok(Self { permutations })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(instance.seed.unwrap_or(DEFAULT_RANDOM_SEED));
		for permutation in &self.permutations {
			let state: [u64; 25] = std::array::from_fn(|_| rng.next_u64());
			permutation.permutation.populate_state(w, state);

			let mut expected = state;
			tiny_keccak::keccakf(&mut expected);
			for (wire, value) in permutation.output.iter().zip(expected) {
				w[*wire] = Word(value);
			}
		}
		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!("{}p", params.num_primitives))
	}
}

fn next_block(rng: &mut StdRng) -> [u8; 64] {
	let mut block = [0; 64];
	for chunk in block.chunks_exact_mut(8) {
		chunk.copy_from_slice(&rng.next_u64().to_le_bytes());
	}
	block
}

#[cfg(test)]
mod tests {
	use binius_core::verify::verify_constraints;
	use binius_frontend::CircuitBuilder;

	use super::*;

	fn assert_builds_and_populates<E>()
	where
		E: ExampleCircuit<Params = PrimitiveParams, Instance = Instance>,
	{
		let mut builder = CircuitBuilder::new();
		let example = E::build(PrimitiveParams { num_primitives: 2 }, &mut builder).unwrap();
		let circuit = builder.build();
		let mut filler = circuit.new_witness_filler();

		example
			.populate_witness(Instance { seed: Some(7) }, &mut filler)
			.unwrap();
		circuit.populate_wire_witness(&mut filler).unwrap();
		verify_constraints(circuit.constraint_system(), &filler.into_value_vec()).unwrap();
	}

	#[test]
	fn independent_sha256_compressions_populate_valid_witness() {
		assert_builds_and_populates::<IndependentSha256Compressions>();
	}

	#[test]
	fn independent_blake3_compressions_populate_valid_witness() {
		assert_builds_and_populates::<IndependentBlake3Compressions>();
	}

	#[test]
	fn independent_keccak_permutations_populate_valid_witness() {
		assert_builds_and_populates::<IndependentKeccakPermutations>();
	}

	#[test]
	fn compression_benchmarks_accept_odd_counts() {
		let mut builder = CircuitBuilder::new();
		IndependentSha256Compressions::build(PrimitiveParams { num_primitives: 1 }, &mut builder)
			.unwrap();

		let mut builder = CircuitBuilder::new();
		IndependentBlake3Compressions::build(PrimitiveParams { num_primitives: 1 }, &mut builder)
			.unwrap();
	}

	#[test]
	fn compression_benchmarks_reject_zero_counts() {
		let mut builder = CircuitBuilder::new();
		assert!(
			IndependentSha256Compressions::build(
				PrimitiveParams { num_primitives: 0 },
				&mut builder
			)
			.is_err()
		);

		let mut builder = CircuitBuilder::new();
		assert!(
			IndependentBlake3Compressions::build(
				PrimitiveParams { num_primitives: 0 },
				&mut builder
			)
			.is_err()
		);
	}

	/// Guard against dead-code elimination pruning the hash logic: each primitive must emit
	/// a non-trivial number of AND constraints per evaluation.
	#[test]
	fn circuits_emit_hash_constraints() {
		fn and_count<E>() -> usize
		where
			E: ExampleCircuit<Params = PrimitiveParams, Instance = Instance>,
		{
			let mut builder = CircuitBuilder::new();
			E::build(PrimitiveParams { num_primitives: 1 }, &mut builder).unwrap();
			builder.build().constraint_system().and_constraints.len()
		}

		assert!(and_count::<IndependentSha256Compressions>() > 100);
		assert!(and_count::<IndependentBlake3Compressions>() > 100);
		assert!(and_count::<IndependentKeccakPermutations>() > 100);
	}

	/// Tampering with the expected output must fail the output assertions during wire
	/// witness evaluation, proving they are enforced.
	#[test]
	fn tampered_digest_fails_output_assertion() {
		let mut builder = CircuitBuilder::new();
		let example = IndependentSha256Compressions::build(
			PrimitiveParams { num_primitives: 1 },
			&mut builder,
		)
		.unwrap();
		let circuit = builder.build();
		let mut filler = circuit.new_witness_filler();
		example
			.populate_witness(Instance { seed: Some(7) }, &mut filler)
			.unwrap();

		let digest_wire = example.compressions[0].digest[0];
		filler[digest_wire] = Word(filler[digest_wire].0 ^ 1);
		assert!(circuit.populate_wire_witness(&mut filler).is_err());
	}
}
