// Copyright 2026 The Binius Developers

//! Small-input sweep of the parallel loops that carry a minimum task length.
//!
//! A minimum task length only binds below its threshold.
//! So the sizes here straddle it rather than sitting above it.
//!
//! Two runs measure what the floor buys, with no recompile between them:
//!
//! ```text
//!     BINIUS_TASK_TARGET_NS=1 BINIUS_MIN_TASK_BYTES=1 \
//!         cargo bench -p binius-math --bench task_size -- --save-baseline flooroff
//!     cargo bench -p binius-math --bench task_size -- --baseline flooroff
//! ```
//!
//! The first run floors every loop at one item, reproducing an unfloored loop.
//! The second uses the calibrated budgets.

use binius_field::arch::{OptimalB128, OptimalPackedB128};
use binius_math::{
	bit_reverse::bit_reverse_indices,
	inner_product::inner_product_par,
	multilinear::{eq::eq_ind_partial_eval, fold::fold_highest_var_inplace},
	test_utils::{random_field_buffer, random_scalars},
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

type F = OptimalB128;
type P = OptimalPackedB128;

/// Base-two logarithms of the sizes swept, spanning the floor from below to above.
///
/// A budget of 100 microseconds over a few nanoseconds per word floors the
/// arithmetic-bound loops in the low tens of thousands of words.
/// So 2^10 sits far below the floor and 2^20 far above it.
const LOG_LENS: [usize; 6] = [10, 12, 14, 16, 18, 20];

fn bench_fold(c: &mut Criterion) {
	let mut group = c.benchmark_group("task_size/fold_highest_var_inplace");
	let mut rng = StdRng::seed_from_u64(0);

	for n_vars in LOG_LENS {
		let buffer = random_field_buffer::<P>(&mut rng, n_vars);
		let challenge = random_scalars::<F>(&mut rng, 1)[0];

		group.bench_function(BenchmarkId::from_parameter(n_vars), |b| {
			// Folding overwrites its input, so each iteration starts from a fresh copy.
			// The clone is excluded from the timed region.
			b.iter_batched_ref(
				|| buffer.clone(),
				|buf| fold_highest_var_inplace(buf, challenge),
				criterion::BatchSize::LargeInput,
			);
		});
	}

	group.finish();
}

fn bench_inner_product(c: &mut Criterion) {
	let mut group = c.benchmark_group("task_size/inner_product_par");
	let mut rng = StdRng::seed_from_u64(0);

	for n_vars in LOG_LENS {
		let a = random_field_buffer::<P>(&mut rng, n_vars);
		let b_buf = random_field_buffer::<P>(&mut rng, n_vars);

		group.bench_function(BenchmarkId::from_parameter(n_vars), |b| {
			b.iter(|| inner_product_par::<F, P, _, _>(&a, &b_buf));
		});
	}

	group.finish();
}

fn bench_eq_ind(c: &mut Criterion) {
	let mut group = c.benchmark_group("task_size/eq_ind_partial_eval");
	let mut rng = StdRng::seed_from_u64(0);

	for n_vars in LOG_LENS {
		let point = random_scalars::<F>(&mut rng, n_vars);

		group.bench_function(BenchmarkId::from_parameter(n_vars), |b| {
			b.iter(|| eq_ind_partial_eval::<P>(&point));
		});
	}

	group.finish();
}

fn bench_bit_reverse_indices(c: &mut Criterion) {
	let mut group = c.benchmark_group("task_size/bit_reverse_indices");

	for log_len in LOG_LENS {
		let mut data = (0..1u64 << log_len).collect::<Vec<_>>();

		group.bench_function(BenchmarkId::from_parameter(log_len), |b| {
			// The permutation is its own inverse, so repeating it in place is stable.
			// No per-iteration setup is needed.
			b.iter(|| bit_reverse_indices(&mut data));
		});
	}

	group.finish();
}

criterion_group! {
	name = default;
	config = Criterion::default().sample_size(50).significance_level(0.01);
	targets = bench_fold, bench_inner_product, bench_eq_ind, bench_bit_reverse_indices
}
criterion_main!(default);
