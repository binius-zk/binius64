// Copyright 2026 The Binius Developers

use binius_compute::BufferPool;
use binius_core::word::Word;
use binius_field::{Ghash128b as B128, Random, arch::OptimalPackedB128};
use binius_iop::whir::WHIRLevel;
use binius_iop_prover::whir::induced_weight::InducedWeight;
use binius_math::ntt::{NeighborsLastMultiThread, domain_context::GaoMateerPreExpanded};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::current_num_threads};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// The three levels a 2^22-scalar message is proved over at 96-bit security.
///
/// Each entry is a message dimension, an inverse rate, and the queries drawn at that rate.
const LADDER: [(usize, usize, usize); 3] = [(18, 1, 232), (14, 4, 106), (10, 5, 101)];

/// The lane count of that ladder, which enters neither route.
const LOG_LANES: usize = 4;

/// Benchmarks both routes to a level's induced weight, at every shape of the shipped ladder.
///
/// The two are alternatives, so they are timed in one binary and meet one machine state.
/// Their ratio at a shape is the constant the selection rule between them encodes.
///
/// Throughput is the weight's own bytes, which both routes produce equally many of.
fn bench_induced_weight(c: &mut Criterion) {
	let mut rng = StdRng::seed_from_u64(0);
	let log_num_shares = log2_ceil_usize(current_num_threads());

	let mut group = c.benchmark_group("whir_induced_weight");

	for (log_msg_cols, log_inv_rate, n_queries) in LADDER {
		let level = WHIRLevel {
			log_msg_cols,
			log_lanes: LOG_LANES,
			log_inv_rate,
			n_queries,
		};

		// The transform the adjoint route runs its layers over, as the prover holds it.
		let domain_context = GaoMateerPreExpanded::<B128>::generate(level.log_codeword_len());
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		// Sampling is with replacement, exactly as the channel draws it.
		let indices = (0..n_queries)
			.map(|_| Word::from_u64(rng.random_range(0..1u64 << level.log_codeword_len())))
			.collect::<Vec<_>>();
		let weight = InducedWeight::new(&level, &ntt, &indices, B128::random(&mut rng));

		let shape = format!("cols=2^{log_msg_cols} rate=2^{log_inv_rate} t={n_queries}");
		group.throughput(Throughput::Bytes(((1 << log_msg_cols) * size_of::<B128>()) as u64));

		// One pool per arm, recycled across iterations, which is how the prover allocates.
		group.bench_function(BenchmarkId::new("rows", &shape), |b| {
			let pool = BufferPool::new();
			b.iter(|| weight.by_rows::<OptimalPackedB128, _>(&&pool));
		});
		group.bench_function(BenchmarkId::new("adjoint", &shape), |b| {
			let pool = BufferPool::new();
			b.iter(|| weight.by_adjoint::<OptimalPackedB128, _>(&&pool));
		});
	}

	group.finish();
}

criterion_group! {
	name = default;
	config = Criterion::default().sample_size(10).significance_level(0.01);
	targets = bench_induced_weight
}
criterion_main!(default);
