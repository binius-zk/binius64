// Copyright 2025-2026 The Binius Developers

use anyhow::Result;
use binius_circuits::blake3::blake3_compress_2x;
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use clap::Args;
use rand::{RngCore, SeedableRng, rngs::StdRng};

use super::utils::DEFAULT_RANDOM_SEED;
use crate::ExampleCircuit;

/// Default number of logical BLAKE3 compressions if none is specified.
const DEFAULT_NUM_COMPRESSIONS: usize = 16;

/// Benchmark for `blake3_compress_2x`: runs ceil(num_compressions / 2) parallel 2-lane
/// compressions chained through their chaining-value output.
pub struct Blake3CompressExample {
	initial_cv: [Wire; 8],
	pairs: Vec<PairInputs>,
}

struct PairInputs {
	block: [Wire; 16],
	counter_lo: Wire,
	counter_hi: Wire,
	block_len: Wire,
	flags: Wire,
}

#[derive(Debug, Clone, Args)]
pub struct Params {
	/// Total number of logical BLAKE3 compressions. The circuit issues
	/// `ceil(num_compressions / 2)` calls to `blake3_compress_2x`, running two
	/// compressions per call in the upper/lower 32-bit lanes.
	#[arg(long)]
	pub num_compressions: Option<usize>,
}

#[derive(Debug, Clone, Args)]
pub struct Instance {
	/// RNG seed for witness values.
	#[arg(long)]
	pub seed: Option<u64>,
}

impl ExampleCircuit for Blake3CompressExample {
	type Params = Params;
	type Instance = Instance;

	fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
		let num_compressions = params.num_compressions.unwrap_or(DEFAULT_NUM_COMPRESSIONS);
		let n_pairs = num_compressions.div_ceil(2);

		let initial_cv: [Wire; 8] = std::array::from_fn(|_| builder.add_witness());
		let mut cv = initial_cv;
		let mut pairs = Vec::with_capacity(n_pairs);

		for _ in 0..n_pairs {
			let block: [Wire; 16] = std::array::from_fn(|_| builder.add_witness());
			let counter_lo = builder.add_witness();
			let counter_hi = builder.add_witness();
			let block_len = builder.add_witness();
			let flags = builder.add_witness();

			cv = blake3_compress_2x(builder, cv, block, counter_lo, counter_hi, block_len, flags);

			pairs.push(PairInputs {
				block,
				counter_lo,
				counter_hi,
				block_len,
				flags,
			});
		}

		Ok(Self { initial_cv, pairs })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(instance.seed.unwrap_or(DEFAULT_RANDOM_SEED));
		let mut next = || Word(rng.next_u64());

		for wire in self.initial_cv {
			w[wire] = next();
		}
		for pair in &self.pairs {
			for b in pair.block {
				w[b] = next();
			}
			w[pair.counter_lo] = next();
			w[pair.counter_hi] = next();
			w[pair.block_len] = next();
			w[pair.flags] = next();
		}

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!(
			"{}c",
			params.num_compressions.unwrap_or(DEFAULT_NUM_COMPRESSIONS)
		))
	}
}
