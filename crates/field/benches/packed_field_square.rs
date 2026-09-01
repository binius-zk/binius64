// Copyright 2024-2025 Irreducible Inc.

mod packed_field_utils;

use binius_field::{
	PackedBinaryField128x1b, PackedBinaryField256x1b, PackedBinaryField512x1b, PackedField,
	PackedGhash1x128b, PackedGhash2x128b, PackedGhash4x128b, PackedRijndael16x8b,
	PackedRijndael32x8b, PackedRijndael64x8b,
};
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

fn square_main<T: PackedField>(val: T) -> T {
	val.square()
}

benchmark_packed_operation!(
	op_name @ square,
	bench_type @ unary_op,
	strategies @ (
		(main, PackedField, square_main),
	)
);

criterion_main!(square);
