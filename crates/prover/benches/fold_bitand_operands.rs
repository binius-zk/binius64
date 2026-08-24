// Copyright 2026 The Binius Developers
use binius_compute::BufferPool;
use binius_core::word::Word;
use binius_field::arch::OptimalPackedB128;
use binius_math::test_utils::random_scalars;
use binius_prover::fold_word::BitAxisFolder;
use binius_verifier::config::B128;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

/// Columns the AND zerocheck folds from two stored ones.
///
/// The constraint is `A & B == C`, and on a satisfying witness the third column is the AND of the
/// first two. So two columns are read and three are produced.
const N_COLUMNS: u64 = 3;

/// The fused fold, against the three separate folds it replaces.
///
/// The zerocheck stores only the two operand columns, so a caller folding them separately has to
/// materialize the third first. The fused one derives it in registers instead, and never writes it.
/// Both produce the same three folded columns, so the pair measures what the fusion is worth.
fn bench_fold_bitand_operands(c: &mut Criterion) {
	let mut group = c.benchmark_group("fold_bitand_operands");

	for log_n_words in [16, 20] {
		let n_words = 1 << log_n_words;
		let mut rng = rand::rng();

		// Two random operand columns, and the third the constraint derives from them.
		let a_words = (0..n_words)
			.map(|_| Word::from_u64(rng.random()))
			.collect::<Vec<_>>();
		let b_words = (0..n_words)
			.map(|_| Word::from_u64(rng.random()))
			.collect::<Vec<_>>();
		let weights = random_scalars::<B128>(&mut rng, Word::BITS);
		let folder = BitAxisFolder::new(&weights);
		let pool = BufferPool::new();
		let alloc = &pool;

		// Words across all three output columns, so throughput counts the whole phase.
		group.throughput(Throughput::Elements(N_COLUMNS * n_words as u64));

		group.bench_with_input(BenchmarkId::new("fused", n_words), &n_words, |b, _| {
			b.iter(|| {
				folder.fold_bitand_operands::<OptimalPackedB128, _>(&alloc, &a_words, &b_words)
			})
		});
		group.bench_with_input(BenchmarkId::new("separate", n_words), &n_words, |b, _| {
			b.iter(|| {
				// The derived column is not stored, so folding separately pays to build it.
				let c_words = std::iter::zip(&a_words, &b_words)
					.map(|(&a, &b)| a & b)
					.collect::<Vec<_>>();
				[
					folder.fold::<OptimalPackedB128, _>(&alloc, &a_words),
					folder.fold::<OptimalPackedB128, _>(&alloc, &b_words),
					folder.fold::<OptimalPackedB128, _>(&alloc, &c_words),
				]
			})
		});
	}

	group.finish();
}

criterion_group!(benches, bench_fold_bitand_operands);
criterion_main!(benches);
