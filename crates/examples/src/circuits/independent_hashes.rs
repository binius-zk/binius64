// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

use anyhow::{Result, ensure};
use binius_circuits::{
	blake3::blake3_compress,
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
/// Each circuit component runs `compress(IV, m_i)` for a fresh 64-byte message block.
pub struct IndependentSha256Compressions {
	blocks: Vec<[Wire; 16]>,
}

impl ExampleCircuit for IndependentSha256Compressions {
	type Params = PrimitiveParams;
	type Instance = Instance;

	fn build(params: PrimitiveParams, builder: &mut CircuitBuilder) -> Result<Self> {
		ensure!(params.num_primitives > 0, "num_primitives must be positive");

		let blocks = (0..params.num_primitives)
			.map(|_| {
				let block = std::array::from_fn(|_| builder.add_witness());
				let _out = sha256_compress(builder, Sha256State::iv(builder), block);
				block
			})
			.collect();

		Ok(Self { blocks })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(instance.seed.unwrap_or(DEFAULT_RANDOM_SEED));
		for block in &self.blocks {
			populate_message_block(w, block, next_block(&mut rng));
		}
		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!("{}c", params.num_primitives))
	}
}

/// Independent BLAKE3 compression evaluations.
///
/// Each circuit component runs one independent BLAKE3 compression function.
pub struct IndependentBlake3Compressions {
	compressions: Vec<Blake3Compression>,
}

struct Blake3Compression {
	cv: [Wire; 8],
	block: [Wire; 16],
	counter: Wire,
	block_len: Wire,
	flags: Wire,
}

impl ExampleCircuit for IndependentBlake3Compressions {
	type Params = PrimitiveParams;
	type Instance = Instance;

	fn build(params: PrimitiveParams, builder: &mut CircuitBuilder) -> Result<Self> {
		ensure!(params.num_primitives > 0, "num_primitives must be positive");

		let compressions = (0..params.num_primitives)
			.map(|_| {
				let cv = std::array::from_fn(|_| builder.add_witness());
				let block = std::array::from_fn(|_| builder.add_witness());
				let counter = builder.add_witness();
				let block_len = builder.add_witness();
				let flags = builder.add_witness();
				let _out = blake3_compress(builder, cv, block, counter, block_len, flags);
				Blake3Compression {
					cv,
					block,
					counter,
					block_len,
					flags,
				}
			})
			.collect();

		Ok(Self { compressions })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(instance.seed.unwrap_or(DEFAULT_RANDOM_SEED));
		for compression in &self.compressions {
			for wire in compression.cv {
				w[wire] = next_u32_word(&mut rng);
			}
			for wire in compression.block {
				w[wire] = next_u32_word(&mut rng);
			}
			w[compression.counter] = Word(rng.next_u64());
			w[compression.block_len] = Word((rng.next_u32() % 65) as u64);
			w[compression.flags] = next_u32_word(&mut rng);
		}
		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!("{}c", params.num_primitives))
	}
}

/// Independent Keccak-f\[1600\] permutation evaluations.
pub struct IndependentKeccakPermutations {
	permutations: Vec<Permutation>,
}

impl ExampleCircuit for IndependentKeccakPermutations {
	type Params = PrimitiveParams;
	type Instance = Instance;

	fn build(params: PrimitiveParams, builder: &mut CircuitBuilder) -> Result<Self> {
		ensure!(params.num_primitives > 0, "num_primitives must be positive");

		let permutations = (0..params.num_primitives)
			.map(|_| {
				let words = std::array::from_fn(|_| builder.add_witness());
				Permutation::new(builder, KeccakState { words })
			})
			.collect();

		Ok(Self { permutations })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(instance.seed.unwrap_or(DEFAULT_RANDOM_SEED));
		for permutation in &self.permutations {
			let state = std::array::from_fn(|_| rng.next_u64());
			permutation.populate_state(w, state);
		}
		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!("{}p", params.num_primitives))
	}
}

fn next_u32_word(rng: &mut StdRng) -> Word {
	Word(rng.next_u32() as u64)
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
}
