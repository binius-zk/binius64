// Copyright 2024-2025 Irreducible Inc.

mod packed_field_utils;

use std::ops::Mul;

use binius_field::{
	PackedBinaryField128x1b, PackedBinaryField256x1b, PackedBinaryField512x1b, PackedGhash1x128b,
	PackedGhash2x128b, PackedGhash4x128b, PackedRijndael16x8b, PackedRijndael32x8b,
	PackedRijndael64x8b,
};
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

/// This trait is needed to specify `Mul` constraint only
trait SelfMul: Mul<Self, Output = Self> + Sized {}

impl<T: Mul<Self, Output = Self> + Sized> SelfMul for T {}

fn mul_main<T: SelfMul>(lhs: T, rhs: T) -> T {
	lhs * rhs
}

benchmark_packed_operation!(
	op_name @ multiply,
	bench_type @ binary_op,
	strategies @ (
		(main, SelfMul, mul_main),
	)
);

criterion_main!(multiply);
