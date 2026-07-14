// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
#![allow(dead_code)]
#![allow(unused_imports)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::uint64x2_t;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m128i, __m256i};
use std::{array, hint::black_box};

use binius_arith_bench::{Underlier, ghash, ghash_sq, polyval};
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use rand::{
	Rng,
	distr::{Distribution, StandardUniform},
};

/// Runs the field throughput benchmark strategy in the Longfellow-ZK repo.
///
/// This increases the density of field multiplications compared to the amount of memory used per
/// iteration.
fn run_google_mul_benchmark<U, R>(
	group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
	name: &str,
	mul_fn: impl Fn(U, U) -> U,
	rng: &mut R,
	element_bits: usize,
) where
	U: Underlier,
	R: Rng,
{
	const N: usize = 64;

	let x = U::random(rng);
	let mut y = [U::ZERO; N];

	// Calculate throughput based on elements per underlier
	let elements_per_underlier = U::BITS / element_bits;
	group.throughput(Throughput::Elements(
		((N + N * N * (N + 1) + N) * elements_per_underlier) as u64,
	));

	group.bench_function(name, |b| {
		b.iter(|| {
			let mut x = black_box(x);
			for j in 0..N {
				y[j] = x;
				x = mul_fn(x, x);
			}
			for _i in 0..N * N {
				for j in 0..N {
					y[j] = mul_fn(y[j], x);
				}
				x = mul_fn(x, x);
			}
			for j in 0..N {
				x = mul_fn(y[j], x);
			}
			x
		})
	});
}

/// Generic benchmark helper for multiplication operations.
///
/// The benchmark strategy is to iterate over an array of field elements repeatedly, multiplying
/// strided pairs of elements in place. The stride is half of the length of the array, so that when
/// the array is large enough, the instructions can be efficiently pipelined. The number of passes
/// controls the density of multiplications per iteration.
/// Generic benchmark helper for multiplication operations
fn run_mul_benchmark<T, R>(
	group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
	name: &str,
	mul_fn: impl Fn(T, T) -> T,
	mut rng: R,
	element_bits: usize,
) where
	T: Underlier,
	R: Rng,
{
	/// The size of the field element array to process. Values too small limit pipelined
	/// parallelism.
	const BATCH_SIZE: usize = 16;
	/// The number of multiplications per element in the batch. Larger values increase the density
	/// of multiplications per iteration.
	const N_PASSES: usize = 64;

	// Generate random elements that are to be multiplied.
	let mut batch: [T; BATCH_SIZE] = array::from_fn(|_| T::random(&mut rng));

	// Calculate throughput based on elements per underlier
	let elements_per_underlier = T::BITS / element_bits;
	group.throughput(Throughput::Elements((BATCH_SIZE * N_PASSES * elements_per_underlier) as u64));

	group.bench_function(name, |b| {
		b.iter(|| {
			for _ in 0..N_PASSES {
				for i in 0..BATCH_SIZE {
					batch[i] = mul_fn(batch[i], batch[(i + BATCH_SIZE / 2) % BATCH_SIZE]);
				}
			}
		})
	});
}

/// Generic benchmark helper for unary operations (e.g. squaring, multiply-by-constant).
///
/// Similar to run_mul_benchmark but for operations that take a single operand.
fn run_unary_op_benchmark<T, R>(
	group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
	name: &str,
	op_fn: impl Fn(T) -> T,
	mut rng: R,
	element_bits: usize,
) where
	T: Underlier,
	R: Rng,
{
	/// The size of the field element array to process. Values too small limit pipelined
	/// parallelism.
	const BATCH_SIZE: usize = 16;
	/// The number of operations per element in the batch. Larger values increase the density
	/// of operations per iteration.
	const N_PASSES: usize = 64;

	// Generate random elements to apply the operation to.
	let mut batch: [T; BATCH_SIZE] = array::from_fn(|_| T::random(&mut rng));

	// Calculate throughput based on elements per underlier
	let elements_per_underlier = T::BITS / element_bits;
	group.throughput(Throughput::Elements((BATCH_SIZE * N_PASSES * elements_per_underlier) as u64));

	group.bench_function(name, |b| {
		b.iter(|| {
			for _ in 0..N_PASSES {
				for i in 0..BATCH_SIZE {
					batch[i] = op_fn(batch[i]);
				}
			}
		})
	});
}

/// Benchmark helper for an inner product of two fixed-size buffers.
///
/// This is the setting where a widening multiply pays off: the per-term products are
/// XOR-accumulated in their unreduced form and the single final reduction is skipped entirely (an
/// inner product needs only one reduction at the very end, negligible against the `2^log_len`
/// terms). The gap between a `wide_mul` variant and the corresponding reduce-every-term `mul`
/// isolates the cost of the amortized reduction.
///
/// `T` is the underlier of the operands (`u128` for the software path, a SIMD type for the packed
/// paths) and `W` the unreduced product accumulator — either a field element (reduce-every-term) or
/// a multi-limb product like `[u64; 4]` / `[__m128i; 3]`, all [`Underlier`]s (arrays via the
/// blanket impl), so accumulation is `W::xor`. `element_bits` is the field element size (e.g. 128
/// for GHASH, 64 for the base Monbijou field), so a `T`-wide underlier carries `T::BITS /
/// element_bits` inner-product terms.
fn run_inner_product_benchmark<T, W, R>(
	group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
	name: &str,
	wide_mul: impl Fn(T, T) -> W,
	mut rng: R,
	log_len: usize,
	element_bits: usize,
) where
	T: Underlier,
	W: Underlier,
	R: Rng,
{
	let len = 1usize << log_len;
	let a: Vec<T> = (0..len).map(|_| T::random(&mut rng)).collect();
	let b: Vec<T> = (0..len).map(|_| T::random(&mut rng)).collect();

	let elements_per_underlier = T::BITS / element_bits;
	group.throughput(Throughput::Elements((len * elements_per_underlier) as u64));
	group.bench_function(name, |bencher| {
		bencher.iter(|| {
			black_box(std::iter::zip(&a, &b).fold(W::ZERO, |acc, (&ai, &bi)| {
				W::xor(acc, wide_mul(black_box(ai), black_box(bi)))
			}))
		})
	});
}

/// Benchmark GF(2^8) multiplication
#[allow(unused_variables, unused_mut)]
fn bench_rijndael(c: &mut Criterion) {
	let mut group = c.benchmark_group("rijndael");
	let mut rng = rand::rng();

	// Benchmark GFNI __m128i
	#[cfg(all(target_feature = "gfni", target_feature = "sse2"))]
	{
		use binius_arith_bench::rijndael::gfni::mul;
		run_mul_benchmark(&mut group, "gfni::mul::<__m128i>", mul::<__m128i>, &mut rng, 8);
	}

	// Benchmark GFNI __m256i
	#[cfg(all(target_feature = "gfni", target_feature = "avx"))]
	{
		use binius_arith_bench::rijndael::gfni::mul;
		run_mul_benchmark(&mut group, "gfni::mul::<__m256i>", mul::<__m256i>, &mut rng, 8);
	}

	// Benchmark the branch-free Russian-peasant software path — the non-GFNI x86_64 fallback,
	// across SSE2/AVX2/AVX-512BW register widths.
	#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
	{
		use binius_arith_bench::rijndael::russian_peasant::mul;
		run_mul_benchmark(
			&mut group,
			"russian_peasant::mul::<__m128i>",
			mul::<__m128i>,
			&mut rng,
			8,
		);
	}

	#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
	{
		use binius_arith_bench::rijndael::russian_peasant::mul;
		run_mul_benchmark(
			&mut group,
			"russian_peasant::mul::<__m256i>",
			mul::<__m256i>,
			&mut rng,
			8,
		);
	}

	#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
	{
		use std::arch::x86_64::__m512i;

		use binius_arith_bench::rijndael::russian_peasant::mul;
		run_mul_benchmark(
			&mut group,
			"russian_peasant::mul::<__m512i>",
			mul::<__m512i>,
			&mut rng,
			8,
		);
	}

	// Benchmark vmull uint64x2_t
	#[cfg(target_arch = "aarch64")]
	{
		use binius_arith_bench::rijndael::vmull::mul;
		run_mul_benchmark(&mut group, "vmull::mul", mul, &mut rng, 8);
	}

	group.finish();
}

/// Benchmark GF(2^128) polynomial Montgomery multiplication using CLMUL instructions
#[allow(unused_imports, unused_variables, unused_mut)]
fn bench_polyval(c: &mut Criterion) {
	use binius_arith_bench::polyval::mul_clmul;

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("polyval");

	// Benchmark u128
	run_mul_benchmark(&mut group, "soft64::mul", polyval::soft64::mul, &mut rng, 128);

	// Benchmark __m128i
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_mul_benchmark(&mut group, "mul_clmul::<__m128i>", mul_clmul::<__m128i>, &mut rng, 128);
	}

	// Benchmark __m256i
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(&mut group, "mul_clmul::<__m256i>", mul_clmul::<__m256i>, &mut rng, 128);
	}

	// Benchmark uint64x2_t (AARCH64 NEON)
	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_clmul::uint64x2_t",
			mul_clmul::<uint64x2_t>,
			&mut rng,
			128,
		);
	}

	group.finish();
}

/// Benchmark GF(2^128) GHASH multiplication using CLMUL instructions
#[allow(unused_imports, unused_variables, unused_mut)]
fn bench_ghash(c: &mut Criterion) {
	use binius_arith_bench::ghash::{mul_clmul, square_clmul};

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("ghash");

	// Benchmark u128
	run_mul_benchmark(&mut group, "soft64::mul", ghash::soft64::mul, &mut rng, 128);
	run_unary_op_benchmark(&mut group, "soft64::square", ghash::soft64::square, &mut rng, 128);

	// Benchmark __m128i
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_mul_benchmark(&mut group, "mul_clmul::<__m128i>", mul_clmul::<__m128i>, &mut rng, 128);

		run_mul_benchmark(
			&mut group,
			"bit_sliced::mul_naive::<__m128i>",
			ghash::bit_sliced::mul_naive::<__m128i>,
			&mut rng,
			128,
		);

		run_mul_benchmark(
			&mut group,
			"bit_sliced::mul_katatsuba::<__m128i>",
			ghash::bit_sliced::mul_katatsuba::<__m128i>,
			&mut rng,
			128,
		);
	}

	// Benchmark __m256i
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(&mut group, "mul_clmul::<__m256i>", mul_clmul::<__m256i>, &mut rng, 128);
	}

	// Benchmark uint64x2_t (AARCH64 NEON)
	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_clmul::uint64x2_t",
			mul_clmul::<uint64x2_t>,
			&mut rng,
			128,
		);

		// Direct poly64x2_t multiplication: the two `mul_wide` variants share an identical
		// reduction, so the gap between these two benchmarks isolates the schoolbook (4 PMULL)
		// vs Karatsuba (3 PMULL) widening multiply.
		run_mul_benchmark(
			&mut group,
			"aarch64::mul_schoolbook",
			ghash::aarch64::mul_schoolbook,
			&mut rng,
			128,
		);

		run_mul_benchmark(
			&mut group,
			"aarch64::mul_karatsuba",
			ghash::aarch64::mul_karatsuba,
			&mut rng,
			128,
		);

		run_mul_benchmark(
			&mut group,
			"bit_sliced::mul_naive::<uint64x2_t>",
			ghash::bit_sliced::mul_naive::<uint64x2_t>,
			&mut rng,
			128,
		);

		run_mul_benchmark(
			&mut group,
			"bit_sliced::mul_katatsuba::<uint64x2_t>",
			ghash::bit_sliced::mul_katatsuba::<uint64x2_t>,
			&mut rng,
			128,
		);
	}

	// Benchmark bit-sliced GHASH
	run_mul_benchmark(
		&mut group,
		"bit_sliced::mul_naive::<u64>",
		ghash::bit_sliced::mul_naive::<u64>,
		&mut rng,
		128,
	);

	run_mul_benchmark(
		&mut group,
		"bit_sliced::mul_katatsuba::<u64>",
		ghash::bit_sliced::mul_katatsuba::<u64>,
		&mut rng,
		128,
	);

	// Benchmark squaring operations
	// Benchmark __m128i squaring
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_unary_op_benchmark(
			&mut group,
			"square_clmul::<__m128i>",
			square_clmul::<__m128i>,
			&mut rng,
			128,
		);
	}

	// Benchmark __m256i squaring
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_unary_op_benchmark(
			&mut group,
			"square_clmul::<__m256i>",
			square_clmul::<__m256i>,
			&mut rng,
			128,
		);
	}

	// Benchmark uint64x2_t squaring (AARCH64 NEON)
	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_unary_op_benchmark(
			&mut group,
			"square_clmul::uint64x2_t",
			square_clmul::<uint64x2_t>,
			&mut rng,
			128,
		);
	}

	// Benchmark mul_inv_x operations
	run_unary_op_benchmark(
		&mut group,
		"soft64::mul_inv_x",
		ghash::soft64::mul_inv_x,
		&mut rng,
		128,
	);

	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_unary_op_benchmark(
			&mut group,
			"mul_inv_x_clmul::<__m128i>",
			ghash::clmul::mul_inv_x::<__m128i>,
			&mut rng,
			128,
		);
	}

	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_unary_op_benchmark(
			&mut group,
			"mul_inv_x_clmul::<__m256i>",
			ghash::clmul::mul_inv_x::<__m256i>,
			&mut rng,
			128,
		);
	}

	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_unary_op_benchmark(
			&mut group,
			"mul_inv_x_clmul::uint64x2_t",
			ghash::clmul::mul_inv_x::<uint64x2_t>,
			&mut rng,
			128,
		);
	}

	group.finish();

	let mut group = c.benchmark_group("ghash_google");

	// Benchmark __m128i
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_google_mul_benchmark(
			&mut group,
			"mul_clmul::<__m128i>",
			mul_clmul::<__m128i>,
			&mut rng,
			128,
		);
	}

	// Benchmark __m256i
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_google_mul_benchmark(
			&mut group,
			"mul_clmul::<__m256i>",
			mul_clmul::<__m256i>,
			&mut rng,
			128,
		);
	}

	// Benchmark uint64x2_t (AARCH64 NEON)
	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_google_mul_benchmark(
			&mut group,
			"mul_clmul::uint64x2_t",
			mul_clmul::<uint64x2_t>,
			&mut rng,
			128,
		);
	}

	group.finish();
}

/// Benchmark GF(2^64) Monbijou multiplication, comparing the portable soft64 implementation with
/// the CLMUL implementations.
#[allow(unused_variables, unused_mut)]
fn bench_monbijou(c: &mut Criterion) {
	use binius_arith_bench::monbijou::{mul_clmul, soft64};

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("monbijou_64b");

	// Portable soft64 (no CLMUL/SIMD)
	run_mul_benchmark(&mut group, "soft64::mul", soft64::mul, &mut rng, 64);

	// Benchmark __m128i
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_mul_benchmark(&mut group, "mul_clmul::<__m128i>", mul_clmul::<__m128i>, &mut rng, 64);
	}

	// Benchmark __m256i
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(&mut group, "mul_clmul::<__m256i>", mul_clmul::<__m256i>, &mut rng, 64);
	}

	// Benchmark uint64x2_t (AARCH64 NEON)
	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_clmul::uint64x2_t",
			mul_clmul::<uint64x2_t>,
			&mut rng,
			64,
		);
	}

	group.finish();
}

/// Benchmark GF(2^128) Monbijou 128-bit extension field multiplication using CLMUL instructions,
/// comparing the packed representation (`mul_128b_clmul`) against the sliced representation
/// (`mul_sliced_128b_clmul`).
#[allow(unused_imports, unused_variables, unused_mut)]
fn bench_monbijou_128b(c: &mut Criterion) {
	use binius_arith_bench::monbijou::{mul_128b_clmul, mul_sliced_128b_clmul, soft64};

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("monbijou_128b");

	// Portable soft64 (no CLMUL/SIMD)
	run_mul_benchmark(&mut group, "soft64::mul_128b", soft64::mul_128b, &mut rng, 128);

	// Packed __m128i
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_128b_clmul::<__m128i>",
			mul_128b_clmul::<__m128i>,
			&mut rng,
			128,
		);
	}

	// Packed __m256i
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_128b_clmul::<__m256i>",
			mul_128b_clmul::<__m256i>,
			&mut rng,
			128,
		);
	}

	// Packed uint64x2_t (AARCH64 NEON)
	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_128b_clmul::uint64x2_t",
			mul_128b_clmul::<uint64x2_t>,
			&mut rng,
			128,
		);
	}

	// Sliced __m128i (a `[__m128i; 2]` holds two GF(2^128) elements)
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_sliced_128b_clmul::<__m128i>",
			mul_sliced_128b_clmul::<__m128i>,
			&mut rng,
			128,
		);
	}

	// Sliced __m256i (a `[__m256i; 2]` holds four GF(2^128) elements)
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_sliced_128b_clmul::<__m256i>",
			mul_sliced_128b_clmul::<__m256i>,
			&mut rng,
			128,
		);
	}

	// Sliced uint64x2_t (AARCH64 NEON)
	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_sliced_128b_clmul::uint64x2_t",
			mul_sliced_128b_clmul::<uint64x2_t>,
			&mut rng,
			128,
		);
	}

	group.finish();
}

/// Benchmark GF(2^192) Monbijou degree-3 extension field multiplication in the sliced
/// representation, comparing portable soft64 against the CLMUL implementation.
#[allow(unused_imports, unused_variables, unused_mut)]
fn bench_monbijou_192b(c: &mut Criterion) {
	use binius_arith_bench::monbijou::{mul_sliced_192b_clmul, soft64};

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("monbijou_192b");

	// Portable soft64 (no CLMUL/SIMD)
	run_mul_benchmark(&mut group, "soft64::mul_192b", soft64::mul_192b, &mut rng, 192);

	// Sliced __m128i (a `[__m128i; 3]` holds two GF(2^192) elements)
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_sliced_192b_clmul::<__m128i>",
			mul_sliced_192b_clmul::<__m128i>,
			&mut rng,
			192,
		);
	}

	// Sliced __m256i (a `[__m256i; 3]` holds four GF(2^192) elements)
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_sliced_192b_clmul::<__m256i>",
			mul_sliced_192b_clmul::<__m256i>,
			&mut rng,
			192,
		);
	}

	// Sliced uint64x2_t (AARCH64 NEON)
	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"mul_sliced_192b_clmul::uint64x2_t",
			mul_sliced_192b_clmul::<uint64x2_t>,
			&mut rng,
			192,
		);
	}

	group.finish();
}

/// Benchmark GHASH² (degree-2 extension of GHASH) sliced multiplication
#[allow(unused_variables, unused_mut)]
fn bench_ghash_sq(c: &mut Criterion) {
	let mut rng = rand::rng();

	let mut group = c.benchmark_group("ghash_sq");

	// Benchmark soft64
	run_mul_benchmark(
		&mut group,
		"soft64::mul_sliced",
		ghash_sq::soft64::mul_sliced,
		&mut rng,
		256,
	);

	run_unary_op_benchmark(
		&mut group,
		"soft64::square_sliced",
		ghash_sq::soft64::square_sliced,
		&mut rng,
		256,
	);

	// Benchmark __m128i
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_mul_benchmark(
			&mut group,
			"x86_64::mul_sliced::<__m128i>",
			ghash_sq::x86_64::mul_sliced::<__m128i>,
			&mut rng,
			256,
		);
	}

	// Benchmark __m256i sliced
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"x86_64::mul_sliced::<__m256i>",
			ghash_sq::x86_64::mul_sliced::<__m256i>,
			&mut rng,
			256,
		);
	}

	// Benchmark __m256i packed (single GHASH² element per register)
	#[cfg(all(
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"x86_64::mul_m256i",
			ghash_sq::x86_64::mul_m256i,
			&mut rng,
			256,
		);
	}

	#[cfg(all(
		target_feature = "pclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"x86_64::mul_m256i_as_m128i",
			ghash_sq::x86_64::mul_m256i_as_m128i,
			&mut rng,
			256,
		);
	}

	#[cfg(all(
		target_feature = "pclmulqdq",
		target_feature = "vpclmulqdq",
		target_feature = "avx2",
		target_feature = "sse2"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"x86_64::mul_m256i_hybrid",
			ghash_sq::x86_64::mul_m256i_hybrid,
			&mut rng,
			256,
		);
	}

	// Benchmark uint64x2_t (AARCH64 NEON)
	#[cfg(all(
		target_arch = "aarch64",
		target_feature = "neon",
		target_feature = "aes"
	))]
	{
		run_mul_benchmark(
			&mut group,
			"aarch64::mul_sliced::uint64x2_t",
			ghash_sq::aarch64::mul_sliced::<uint64x2_t>,
			&mut rng,
			256,
		);
	}

	group.finish();
}

/// Benchmark inner products over the GHASH field, contrasting the widening multiply (accumulate
/// unreduced, reduce once) against the reduce-every-term multiply.
#[allow(unused_imports, unused_variables, unused_mut)]
fn bench_ghash_inner_product(c: &mut Criterion) {
	/// Length of the inner product. Long enough that the single skipped final reduction is
	/// negligible.
	const LOG_LEN: usize = 10;

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("ghash_inner_product");

	// Baseline: reduce each product, accumulate the reduced GHASH elements.
	run_inner_product_benchmark(
		&mut group,
		"soft64::mul",
		ghash::soft64::mul,
		&mut rng,
		LOG_LEN,
		128,
	);

	// Widening: accumulate the unreduced four-limb products by XOR, skip the final reduction.
	run_inner_product_benchmark(
		&mut group,
		"soft64::mul_wide",
		ghash::soft64::mul_wide,
		&mut rng,
		LOG_LEN,
		128,
	);

	// CLMUL path (__m128i): the reduction is a larger fraction of a single fast pclmulqdq multiply,
	// so amortizing it across the inner product should help more than in the software path.
	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		use binius_arith_bench::ghash::clmul;

		run_inner_product_benchmark(
			&mut group,
			"clmul::mul::<__m128i>",
			clmul::mul::<__m128i>,
			&mut rng,
			LOG_LEN,
			128,
		);

		run_inner_product_benchmark(
			&mut group,
			"clmul::mul_wide::<__m128i>",
			clmul::mul_wide::<__m128i>,
			&mut rng,
			LOG_LEN,
			128,
		);
	}

	group.finish();
}

/// Benchmark inner products over the GHASH² field, contrasting the delayed-reduction widening
/// multiply (accumulate the two unreduced coefficients, reduce once at the end) against the
/// reduce-every-term sliced multiply.
#[allow(unused_imports, unused_variables, unused_mut)]
fn bench_ghash_sq_inner_product(c: &mut Criterion) {
	/// Length of the inner product. Long enough that the two skipped final reductions are
	/// negligible.
	const LOG_LEN: usize = 10;

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("ghash_sq_inner_product");

	// Baseline: reduce each product (two base-field reductions per term), accumulate the reduced
	// GHASH² elements.
	run_inner_product_benchmark(
		&mut group,
		"soft64::mul_sliced",
		ghash_sq::soft64::mul_sliced,
		&mut rng,
		LOG_LEN,
		128,
	);

	// Widening: accumulate the unreduced coefficients by XOR, reduce once at the very end.
	run_inner_product_benchmark(
		&mut group,
		"soft64::mul_wide_sliced",
		ghash_sq::soft64::mul_wide_sliced,
		&mut rng,
		LOG_LEN,
		128,
	);

	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_inner_product_benchmark(
			&mut group,
			"x86_64::mul_sliced::<__m128i>",
			ghash_sq::x86_64::mul_sliced::<__m128i>,
			&mut rng,
			LOG_LEN,
			128,
		);

		run_inner_product_benchmark(
			&mut group,
			"x86_64::mul_wide_sliced::<__m128i>",
			ghash_sq::x86_64::mul_wide_sliced::<__m128i>,
			&mut rng,
			LOG_LEN,
			128,
		);
	}

	group.finish();
}

/// Benchmark inner products over the base Monbijou field GF(2^64), contrasting the widening
/// multiply (accumulate the unreduced products, reduce once) against the reduce-every-term
/// multiply.
#[allow(unused_imports, unused_variables, unused_mut)]
fn bench_monbijou_inner_product(c: &mut Criterion) {
	use binius_arith_bench::monbijou::{clmul, soft64};

	/// Length of the inner product. Long enough that the single skipped final reduction is
	/// negligible.
	const LOG_LEN: usize = 10;

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("monbijou_inner_product");

	// Baseline: reduce each product, accumulate the reduced base-field elements.
	run_inner_product_benchmark(&mut group, "soft64::mul", soft64::mul, &mut rng, LOG_LEN, 64);

	// Widening: accumulate the unreduced [low, high] products by XOR, skip the final reduction.
	run_inner_product_benchmark(
		&mut group,
		"soft64::mul_wide",
		soft64::mul_wide,
		&mut rng,
		LOG_LEN,
		64,
	);

	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_inner_product_benchmark(
			&mut group,
			"clmul::mul::<__m128i>",
			clmul::mul::<__m128i>,
			&mut rng,
			LOG_LEN,
			64,
		);

		run_inner_product_benchmark(
			&mut group,
			"clmul::mul_wide::<__m128i>",
			clmul::mul_wide::<__m128i>,
			&mut rng,
			LOG_LEN,
			64,
		);
	}

	group.finish();
}

/// Benchmark inner products over the degree-3 Monbijou extension GF(2^192), contrasting the
/// delayed-reduction widening multiply (accumulate the six raw products, reduce once) against the
/// reduce-every-term sliced multiply.
#[allow(unused_imports, unused_variables, unused_mut)]
fn bench_monbijou_192b_inner_product(c: &mut Criterion) {
	use binius_arith_bench::monbijou::{clmul, soft64};

	/// Length of the inner product.
	const LOG_LEN: usize = 10;

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("monbijou_192b_inner_product");

	// Baseline: reduce each product; widening: accumulate the six raw products, reduce once.
	run_inner_product_benchmark(
		&mut group,
		"soft64::mul_192b",
		soft64::mul_192b,
		&mut rng,
		LOG_LEN,
		192,
	);
	run_inner_product_benchmark(
		&mut group,
		"soft64::mul_wide_192b",
		soft64::mul_wide_192b,
		&mut rng,
		LOG_LEN,
		192,
	);

	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		run_inner_product_benchmark(
			&mut group,
			"clmul::mul_sliced_192b::<__m128i>",
			clmul::mul_sliced_192b::<__m128i>,
			&mut rng,
			LOG_LEN,
			192,
		);

		run_inner_product_benchmark(
			&mut group,
			"clmul::mul_wide_sliced_192b::<__m128i>",
			clmul::mul_wide_sliced_192b::<__m128i>,
			&mut rng,
			LOG_LEN,
			192,
		);
	}

	group.finish();
}

/// Benchmark inner products over the degree-2 Monbijou extension GF(2^128), contrasting the
/// delayed-reduction widening multiply (accumulate the three raw products, reduce once) against the
/// reduce-every-term multiply, for both the packed (`u128`) and sliced representations.
#[allow(unused_imports, unused_variables, unused_mut)]
fn bench_monbijou_128b_inner_product(c: &mut Criterion) {
	use binius_arith_bench::monbijou::{clmul, soft64};

	/// Length of the inner product.
	const LOG_LEN: usize = 10;

	let mut rng = rand::rng();

	let mut group = c.benchmark_group("monbijou_128b_inner_product");

	// Portable soft64 (unsliced u128): reduce each product vs accumulate the three raw products.
	run_inner_product_benchmark(
		&mut group,
		"soft64::mul_128b",
		soft64::mul_128b,
		&mut rng,
		LOG_LEN,
		128,
	);
	run_inner_product_benchmark(
		&mut group,
		"soft64::mul_wide_128b",
		soft64::mul_wide_128b,
		&mut rng,
		LOG_LEN,
		128,
	);

	#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
	{
		// Packed representation (coefficients in the low/high halves of each lane).
		run_inner_product_benchmark(
			&mut group,
			"clmul::mul_128b::<__m128i>",
			clmul::mul_128b::<__m128i>,
			&mut rng,
			LOG_LEN,
			128,
		);
		run_inner_product_benchmark(
			&mut group,
			"clmul::mul_wide_128b::<__m128i>",
			clmul::mul_wide_128b::<__m128i>,
			&mut rng,
			LOG_LEN,
			128,
		);

		// Sliced representation (coefficients in separate registers).
		run_inner_product_benchmark(
			&mut group,
			"clmul::mul_sliced_128b::<__m128i>",
			clmul::mul_sliced_128b::<__m128i>,
			&mut rng,
			LOG_LEN,
			128,
		);
		run_inner_product_benchmark(
			&mut group,
			"clmul::mul_wide_sliced_128b::<__m128i>",
			clmul::mul_wide_sliced_128b::<__m128i>,
			&mut rng,
			LOG_LEN,
			128,
		);
	}

	group.finish();
}

criterion_group!(
	benches,
	bench_rijndael,
	bench_polyval,
	bench_ghash,
	bench_ghash_sq,
	bench_monbijou,
	bench_monbijou_128b,
	bench_monbijou_192b,
	bench_ghash_inner_product,
	bench_ghash_sq_inner_product,
	bench_monbijou_inner_product,
	bench_monbijou_128b_inner_product,
	bench_monbijou_192b_inner_product,
);
criterion_main!(benches);
