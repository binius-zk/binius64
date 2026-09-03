// Copyright 2026 The Binius Developers

//! The multithreaded additive NTT over the share counts a real proof runs at.
//!
//! The share count decides how many layers the shared phase covers, and the matrix bench pins it
//! too low for that phase to carry much of the transform.

use binius_field::PackedGhash4x128b;
use binius_math::{
	ntt::{AdditiveNTT, NeighborsLastMultiThread, domain_context::GaoMateerPreExpanded},
	test_utils::random_field_buffer,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

type P = PackedGhash4x128b;

/// The early skip a rate-1/2 encoder asks the transform for.
const SKIP_EARLY: usize = 1;

fn bench_shared_phase(c: &mut Criterion) {
	let mut rng = rand::rng();
	let mut group = c.benchmark_group("ntt_shared_phase");
	group.sample_size(20);

	for log_d in [20, 24] {
		let domain_context = GaoMateerPreExpanded::generate(log_d);
		group.throughput(Throughput::Bytes(16 << log_d));

		for log_num_shares in [3, 4, 5] {
			let ntt = NeighborsLastMultiThread::new(&domain_context, log_num_shares);
			let mut data = random_field_buffer::<P>(&mut rng, log_d);

			let parameter = format!("log_d={log_d}/log_num_shares={log_num_shares}");
			group.bench_function(BenchmarkId::from_parameter(parameter), |b| {
				b.iter(|| ntt.forward_transform(data.as_mut_view(), SKIP_EARLY, 0));
			});
		}
	}

	group.finish();
}

criterion_group!(default, bench_shared_phase);
criterion_main!(default);
