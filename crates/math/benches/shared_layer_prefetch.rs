// Copyright 2026 The Binius Developers

//! In-process A/B for the prefetch hint added to `forward_shared_layer`'s inner loop.
//!
//! Both variants run in one binary so criterion alternates them within the same process.
//! Cross-process/rebuild comparisons on this machine swing by 2x on identical code, so only a
//! same-binary comparison is trustworthy here.
//!
//! This mirrors the exact loop shape in `crates/math/src/ntt/neighbors_last.rs`'s
//! `forward_shared_layer`, but works on two plain slices so it stays self-contained and does not
//! need any production visibility change.

use binius_field::{PackedBinaryGhash1x128b as P, Random};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

const PREFETCH_DISTANCE: usize = 16;

#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
	#[cfg(target_arch = "x86_64")]
	{
		use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
		// Safety: `_mm_prefetch` never faults, even for an invalid or out-of-bounds address.
		unsafe { _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0) };
	}
	#[cfg(target_arch = "aarch64")]
	{
		// Safety: `prfm` is a hint instruction that never raises a data abort.
		unsafe {
			std::arch::asm!(
				"prfm pldl1keep, [{ptr}]",
				ptr = in(reg) ptr,
				options(nostack, readonly, preserves_flags),
			);
		}
	}
	#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
	{
		let _ = ptr;
	}
}

fn butterfly_no_prefetch(chunk0: &mut [P], chunk1: &mut [P], twiddle: P) {
	for i in 0..chunk0.len() {
		let mut u = chunk0[i];
		let mut v = chunk1[i];
		u += v * twiddle;
		v += u;
		chunk0[i] = u;
		chunk1[i] = v;
	}
}

fn butterfly_with_prefetch(chunk0: &mut [P], chunk1: &mut [P], twiddle: P) {
	for i in 0..chunk0.len() {
		if let Some(j) = i
			.checked_add(PREFETCH_DISTANCE)
			.filter(|&j| j < chunk0.len())
		{
			prefetch_read(chunk0.as_ptr().wrapping_add(j));
			prefetch_read(chunk1.as_ptr().wrapping_add(j));
		}

		let mut u = chunk0[i];
		let mut v = chunk1[i];
		u += v * twiddle;
		v += u;
		chunk0[i] = u;
		chunk1[i] = v;
	}
}

fn bench_shared_layer_inner_loop(c: &mut Criterion) {
	let mut group = c.benchmark_group("shared_layer_inner_loop");

	// 2^17 packed elements per chunk, matching a realistic shared-layer chunk size at log_d=20
	// with 8 threads (log_num_shares=3): log_d_chunk = 20 - 4 = 16, chunk length = 2^16 / 1 (128b
	// packing) = 2^16 elements; 2^17 stresses a bit past that to keep both chunks DRAM-sized.
	let log_len = 17;
	let n = 1 << log_len;
	group.throughput(Throughput::Bytes((n * 2 * size_of::<P>()) as u64));

	let mut rng = StdRng::seed_from_u64(0);
	let twiddle = P::random(&mut rng);

	group.bench_function("no_prefetch", |b| {
		let mut chunk0: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();
		let mut chunk1: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();
		b.iter(|| butterfly_no_prefetch(&mut chunk0, &mut chunk1, twiddle));
	});

	group.bench_function("with_prefetch", |b| {
		let mut chunk0: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();
		let mut chunk1: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();
		b.iter(|| butterfly_with_prefetch(&mut chunk0, &mut chunk1, twiddle));
	});

	group.finish();
}

criterion_group!(benches, bench_shared_layer_inner_loop);
criterion_main!(benches);
