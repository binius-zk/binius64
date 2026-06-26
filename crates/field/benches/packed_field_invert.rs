// Copyright 2024-2025 Irreducible Inc.

mod packed_field_utils;

use binius_field::{
	PackedAESBinaryField16x8b, PackedAESBinaryField32x8b, PackedAESBinaryField64x8b,
	PackedBinaryField128x1b, PackedBinaryField256x1b, PackedBinaryField512x1b,
	PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b, PackedField,
};
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

fn invert_main<T: PackedField>(val: T) -> T {
	val.invert_or_zero()
}

benchmark_packed_operation!(
	op_name @ invert,
	bench_type @ unary_op,
	strategies @ (
		(main, PackedField, invert_main),
	)
);

criterion_main!(invert);
