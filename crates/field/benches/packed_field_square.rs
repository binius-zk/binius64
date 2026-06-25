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

fn square_main<T: PackedField>(val: T) -> T {
	val.square()
}

cfg_if! {
	if #[cfg(feature = "benchmark_alternative_strategies")] {
		use binius_field::{
			arch::{Pairwise, PairwiseTable},
			arithmetic_traits::Square,
		};
		use bytemuck::TransparentWrapper;

		/// Marker trait for packed types whose `Pairwise` wrapper supports squaring.
		trait PairwiseSquare: Sized {
			fn pairwise_square(self) -> Self;
		}
		impl<T> PairwiseSquare for T
		where
			Pairwise<T>: Square,
		{
			#[inline]
			fn pairwise_square(self) -> Self {
				Pairwise::peel(Square::square(Pairwise::wrap(self)))
			}
		}

		/// Marker trait for packed types whose `PairwiseTable` wrapper supports squaring.
		trait PairwiseTableSquare: Sized {
			fn pairwise_table_square(self) -> Self;
		}
		impl<T> PairwiseTableSquare for T
		where
			PairwiseTable<T>: Square,
		{
			#[inline]
			fn pairwise_table_square(self) -> Self {
				PairwiseTable::peel(Square::square(PairwiseTable::wrap(self)))
			}
		}

		fn square_pairwise<T: PairwiseSquare>(val: T) -> T {
			val.pairwise_square()
		}

		fn square_pairwise_table<T: PairwiseTableSquare>(val: T) -> T {
			val.pairwise_table_square()
		}

		benchmark_packed_operation!(
			op_name @ square,
			bench_type @ unary_op,
			strategies @ (
				(main, PackedField, square_main),
				(pairwise, PairwiseSquare, square_pairwise),
				(pairwise_table, PairwiseTableSquare, square_pairwise_table),
			)
		);
	} else {
		benchmark_packed_operation!(
			op_name @ square,
			bench_type @ unary_op,
			strategies @ (
				(main, PackedField, square_main),
			)
		);
	}
}

criterion_main!(square);
