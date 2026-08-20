// Copyright 2026 The Binius Developers

//! Benchmark serializing a slice of field elements two ways.
//!
//! Every `SerializeBytes` implementor gets a per-element loop by default.
//! `BinaryField128bGhash` overrides it with a bulk byte copy instead.
//!
//! Every supported target here is little-endian.
//! `BinaryField128bGhash` is also `Pod`, so its serialized form is just its raw memory bytes.
//! The bulk path copies a whole slice at once instead of writing each element separately.

use std::hint::black_box;

use binius_field::BinaryField128bGhash;
use binius_utils::SerializeBytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn bench_serialize_slice(c: &mut Criterion) {
	let mut group = c.benchmark_group("ghash_serialize_slice");

	for &n in &[1 << 10, 1 << 14, 1 << 18] {
		let values: Vec<BinaryField128bGhash> = (0..n)
			.map(|_| BinaryField128bGhash::from(rand::random::<u128>()))
			.collect();

		group.throughput(Throughput::Bytes((n * 16) as u64));

		group.bench_function(format!("loop/n={n}"), |b| {
			b.iter(|| {
				let mut buf = Vec::with_capacity(n * 16);
				for value in black_box(&values) {
					value.serialize(&mut buf).unwrap();
				}
				black_box(buf)
			})
		});

		group.bench_function(format!("bulk/n={n}"), |b| {
			b.iter(|| {
				let mut buf = Vec::with_capacity(n * 16);
				BinaryField128bGhash::serialize_slice(black_box(&values), &mut buf).unwrap();
				black_box(buf)
			})
		});
	}

	group.finish();
}

criterion_group!(benches, bench_serialize_slice);
criterion_main!(benches);
