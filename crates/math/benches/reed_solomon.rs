// Copyright 2026 The Binius Developers

use binius_compute::GlobalAllocator;
use binius_field::{BinaryField128bGhash as B128, arch::OptimalPackedB128};
use binius_math::{
	ntt::{NeighborsLastMultiThread, domain_context::GaoMateerOnTheFly},
	reed_solomon::ReedSolomonCode,
	test_utils::random_field_buffer,
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::current_num_threads};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Message dimensions to sweep, matching the oracle sizes the FRI rate sweep prices.
const LOG_DIMS: [usize; 2] = [17, 20];

/// Candidate inverse rates, matching the sweep `binius_iop::fri::estimate_by_rate` prices.
const LOG_INV_RATES: [usize; 6] = [1, 2, 3, 4, 5, 6];

/// Benchmarks [`ReedSolomonCode::encode_batch`] at each candidate Reed-Solomon rate.
///
/// This is the prover-cost half of the trade-off `binius_iop::fri::estimate_by_rate` prices.
/// That function reports how many proof bytes a rate saves; this reports what the rate costs.
/// One step lower doubles the codeword, so encode time should roughly double with it.
///
/// The NTT is the multi-threaded one the prover's commit path uses, at one share per core.
/// The domain context is regenerated per rate, since the code length changes with the rate.
fn bench_encode_by_rate(c: &mut Criterion) {
	let mut rng = rand::rng();
	let log_num_shares = log2_ceil_usize(current_num_threads());

	let mut group = c.benchmark_group("reed_solomon_encode_by_rate");
	group.sample_size(10);

	for log_dim in LOG_DIMS {
		let message = random_field_buffer::<OptimalPackedB128>(&mut rng, log_dim);

		for log_inv_rate in LOG_INV_RATES {
			let rs_code = ReedSolomonCode::<B128>::new(log_dim, log_inv_rate);
			let domain_context = GaoMateerOnTheFly::<B128>::generate(rs_code.log_len());
			let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

			// Codeword bytes, so throughput reads as encode bandwidth across rates.
			group.throughput(Throughput::Bytes((rs_code.len() * size_of::<B128>()) as u64));
			group.bench_function(
				BenchmarkId::new(
					format!("log_dim={log_dim}"),
					format!("log_inv_rate={log_inv_rate}"),
				),
				|b| b.iter(|| rs_code.encode_batch(&ntt, message.to_ref(), 0, &GlobalAllocator)),
			);
		}
	}

	group.finish();
}

criterion_group! {
	name = default;
	config = Criterion::default().sample_size(10).significance_level(0.01);
	targets = bench_encode_by_rate
}
criterion_main!(default);
