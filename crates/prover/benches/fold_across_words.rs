// Copyright 2026 The Binius Developers
use binius_core::word::Word;
use binius_field::arch::OptimalPackedB128;
use binius_math::test_utils::random_scalars;
use binius_prover::fold_word::fold_across_words;
use binius_verifier::config::B128;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

fn bench_fold_across_words(c: &mut Criterion) {
	let mut group = c.benchmark_group("fold_across_words");

	for log_n_words in [12, 16, 20] {
		let n_words = 1 << log_n_words;

		// Set throughput to measure words per second.
		group.throughput(Throughput::Elements(n_words as u64));

		group.bench_with_input(BenchmarkId::from_parameter(n_words), &n_words, |b, &n_words| {
			let mut rng = rand::rng();
			let words = (0..n_words)
				.map(|_| Word::from_u64(rng.random::<u64>()))
				.collect::<Vec<_>>();
			let point = random_scalars::<B128>(&mut rng, log_n_words);

			b.iter(|| fold_across_words::<_, OptimalPackedB128>(&words, &point));
		});
	}

	group.finish();
}

criterion_group!(benches, bench_fold_across_words);
criterion_main!(benches);
