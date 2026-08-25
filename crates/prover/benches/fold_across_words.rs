// Copyright 2026 The Binius Developers
use binius_core::word::Word;
use binius_math::test_utils::random_scalars;
use binius_prover::fold_word::WordAxisFolder;
use binius_verifier::config::B128;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

/// Word columns a single proof folds against one shared point.
///
/// The integer-multiplication protocol sends four, and binary multiplication sends six.
/// Six is the wider case, so it is the one benched.
const N_COLUMNS: usize = 6;

/// One column folded against one point, with the two drivers side by side.
///
/// Both compute the same array. The sequential driver leaves every core but one idle, so the pair
/// measures what dividing the chunk axis is worth.
fn bench_single_column(c: &mut Criterion) {
	let mut group = c.benchmark_group("fold_across_words");

	for log_n_words in [12, 16, 20] {
		let n_words = 1 << log_n_words;
		let mut rng = rand::rng();
		let words = (0..n_words)
			.map(|_| Word::from_u64(rng.random()))
			.collect::<Vec<_>>();
		let point = random_scalars::<B128>(&mut rng, log_n_words);
		let folder = WordAxisFolder::<B128>::new(&point);

		// Words per second, so the two drivers compare directly across sizes.
		group.throughput(Throughput::Elements(n_words as u64));

		group.bench_with_input(BenchmarkId::new("sequential", n_words), &n_words, |b, _| {
			b.iter(|| folder.fold(&words));
		});
		group.bench_with_input(BenchmarkId::new("parallel", n_words), &n_words, |b, _| {
			b.iter(|| folder.fold_par(&words));
		});
	}

	group.finish();
}

/// The shape a proof actually folds in: several columns against one shared point.
///
/// The folder is built once outside the loop either way, so this isolates the driver alone.
fn bench_shared_point_columns(c: &mut Criterion) {
	let mut group = c.benchmark_group("fold_across_words/shared_point");

	for log_n_words in [16, 20] {
		let n_words = 1 << log_n_words;
		let mut rng = rand::rng();

		// One independent column per output the protocol sends.
		let columns = (0..N_COLUMNS)
			.map(|_| {
				(0..n_words)
					.map(|_| Word::from_u64(rng.random()))
					.collect::<Vec<_>>()
			})
			.collect::<Vec<_>>();
		let point = random_scalars::<B128>(&mut rng, log_n_words);
		let folder = WordAxisFolder::<B128>::new(&point);

		// All six columns' words, so throughput counts the whole phase.
		group.throughput(Throughput::Elements((N_COLUMNS * n_words) as u64));

		group.bench_with_input(
			BenchmarkId::new(format!("{N_COLUMNS}col sequential"), n_words),
			&n_words,
			|b, _| {
				b.iter(|| {
					columns
						.iter()
						.map(|col| folder.fold(col))
						.collect::<Vec<_>>()
				});
			},
		);
		group.bench_with_input(
			BenchmarkId::new(format!("{N_COLUMNS}col parallel"), n_words),
			&n_words,
			|b, _| {
				b.iter(|| {
					columns
						.iter()
						.map(|col| folder.fold_par(col))
						.collect::<Vec<_>>()
				});
			},
		);
	}

	group.finish();
}

criterion_group!(benches, bench_single_column, bench_shared_point_columns);
criterion_main!(benches);
