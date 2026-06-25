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
			arch::{Pairwise, PairwiseTable},
			arithmetic_traits::InvertOrZero,
		};
		use bytemuck::TransparentWrapper;

		/// Marker trait for packed types whose `Pairwise` wrapper supports inversion.
		trait PairwiseInvert: Sized {
			fn pairwise_invert(self) -> Self;
		}
		impl<T> PairwiseInvert for T
		where
			Pairwise<T>: InvertOrZero,
		{
			#[inline]
			fn pairwise_invert(self) -> Self {
				Pairwise::peel(InvertOrZero::invert_or_zero(Pairwise::wrap(self)))
			}
		}

		/// Marker trait for packed types whose `PairwiseTable` wrapper supports inversion.
		trait PairwiseTableInvert: Sized {
			fn pairwise_table_invert(self) -> Self;
		}
		impl<T> PairwiseTableInvert for T
		where
			PairwiseTable<T>: InvertOrZero,
		{
			#[inline]
			fn pairwise_table_invert(self) -> Self {
				PairwiseTable::peel(InvertOrZero::invert_or_zero(PairwiseTable::wrap(self)))
			}
		}

		fn invert_pairwise<T: PairwiseInvert>(val: T) -> T {
			val.pairwise_invert()
		}

		fn invert_pairwise_table<T: PairwiseTableInvert>(val: T) -> T {
			val.pairwise_table_invert()
		}

		benchmark_packed_operation!(
			op_name @ invert,
			bench_type @ unary_op,
			strategies @ (
				(main, PackedField, invert_main),
				(pairwise, PairwiseInvert, invert_pairwise),
				(pairwise_table, PairwiseTableInvert, invert_pairwise_table),
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
