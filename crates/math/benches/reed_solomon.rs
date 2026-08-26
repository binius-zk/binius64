// Copyright 2026 The Binius Developers

use binius_compute::BufferPool;
use binius_field::{Ghash128b as B128, arch::OptimalPackedB128};
use binius_math::{
	ntt::{NeighborsLastMultiThread, domain_context::GaoMateerOnTheFly},
	reed_solomon::ReedSolomonCode,
	test_utils::random_field_buffer,
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::current_num_threads};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Message dimensions to bench, spanning the oracle sizes the prover commits.
const LOG_DIMS: [usize; 2] = [17, 20];

/// Inverse rates to bench: the rate the prover ships, and the one that minimizes the proof.
///
/// The rate axis needs no sweep, since its arithmetic cost is exact rather than empirical.
/// [`ReedSolomonCode::encode_batch`] skips the first `log_inv_rate` layers of its transform.
/// It therefore runs `log_dim` layers over a codeword of `2^(log_dim + log_inv_rate)`:
///
///     butterflies = log_dim * 2^(log_dim + log_inv_rate - 1)
///
/// The rate enters as a bare factor of `2^log_inv_rate`, with no log term on it.
/// One step down the rate is exactly twice the arithmetic, by construction.
///
/// Two points price that, and also check it.
/// Throughput holding flat between them is the doubling holding.
/// A shortfall is what the growing working set costs on top of the arithmetic.
///
/// `binius_iop::fri` pins the other half of the trade-off, the proof bytes each rate buys.
const LOG_INV_RATES: [usize; 2] = [1, 3];

/// Benchmarks [`ReedSolomonCode::encode_batch`], the encoder on the prover's commit path.
///
/// The NTT is the multi-threaded one that path uses, at one share per core.
/// The domain context is regenerated per rate, since the code length changes with the rate.
///
/// Buffers come from a [`BufferPool`], as they do in the prover.
/// The pool recycles the codeword across iterations, so what is timed is the encode.
/// Allocating afresh each iteration would instead charge first-touch page faults to the encode.
fn bench_encode(c: &mut Criterion) {
	let mut rng = rand::rng();
	let log_num_shares = log2_ceil_usize(current_num_threads());

	let mut group = c.benchmark_group("reed_solomon_encode");

	for log_dim in LOG_DIMS {
		let message = random_field_buffer::<OptimalPackedB128>(&mut rng, log_dim);

		for log_inv_rate in LOG_INV_RATES {
			let rs_code = ReedSolomonCode::<B128>::new(log_dim, log_inv_rate);
			let domain_context = GaoMateerOnTheFly::<B128>::generate(rs_code.log_len());
			let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

			// Codeword bytes, so the number reads as encode bandwidth across both axes.
			group.throughput(Throughput::Bytes((rs_code.len() * size_of::<B128>()) as u64));
			group.bench_function(
				BenchmarkId::new(
					format!("log_dim={log_dim}"),
					format!("log_inv_rate={log_inv_rate}"),
				),
				|b| {
					let pool = BufferPool::new();
					let alloc = &pool;
					b.iter(|| rs_code.encode_batch(&ntt, message.as_view(), 0, &alloc));
				},
			);
		}
	}

	group.finish();
}

criterion_group! {
	name = default;
	config = Criterion::default().sample_size(10).significance_level(0.01);
	targets = bench_encode
}
criterion_main!(default);
