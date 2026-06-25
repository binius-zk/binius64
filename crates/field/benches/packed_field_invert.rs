// Copyright 2024-2025 Irreducible Inc.

mod packed_field_utils;

use binius_field::{
	PackedAESBinaryField16x8b, PackedAESBinaryField32x8b, PackedAESBinaryField64x8b,
	PackedBinaryField128x1b, PackedBinaryField256x1b, PackedBinaryField512x1b,
	PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b, PackedField,
};
use cfg_if::cfg_if;
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

fn invert_main<T: PackedField>(val: T) -> T {
	val.invert_or_zero()
}

cfg_if! {
	if #[cfg(feature = "benchmark_alternative_strategies")] {
		use binius_field::{
			arch::{PackedStrategy, PairwiseStrategy, PairwiseTableStrategy},
			arithmetic_traits::TaggedInvertOrZero,
		};

		fn invert_pairwise<T: TaggedInvertOrZero<PairwiseStrategy>>(val: T) -> T {
			val.invert_or_zero()
		}

		fn invert_pairwise_table<T: TaggedInvertOrZero<PairwiseTableStrategy>>(val: T) -> T {
			val.invert_or_zero()
		}

		fn invert_packed<T: TaggedInvertOrZero<PackedStrategy>>(val: T) -> T {
			val.invert_or_zero()
		}

		benchmark_packed_operation!(
			op_name @ invert,
			bench_type @ unary_op,
			strategies @ (
				(main, PackedField, invert_main),
				(pairwise, TaggedInvertOrZero::<PairwiseStrategy>, invert_pairwise),
				(pairwise_table, TaggedInvertOrZero::<PairwiseTableStrategy>, invert_pairwise_table),
				(packed, TaggedInvertOrZero::<PackedStrategy>, invert_packed),
			)
		);
	} else {
		benchmark_packed_operation!(
			op_name @ invert,
			bench_type @ unary_op,
			strategies @ (
				(main, PackedField, invert_main),
			)
		);
	}
}

criterion_main!(invert);
