// Copyright 2024-2025 Irreducible Inc.

use binius_field::{
	ExtensionField, Random,
	arch::{
		packed_128::*, packed_256::*, packed_512::*, packed_aes_128::*, packed_aes_256::*,
		packed_aes_512::*, packed_ghash_128::*, packed_ghash_256::*, packed_ghash_512::*,
	},
	linear_transformation::{
		FieldLinearTransformation, PackedTransformationFactory, Transformation,
	},
};
use cfg_if::cfg_if;
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

mod packed_field_utils;

trait TransformToSelfFactory: PackedTransformationFactory<Self> {}

impl<T: PackedTransformationFactory<Self>> TransformToSelfFactory for T {}

fn create_transformation_main<PT: TransformToSelfFactory>() -> impl Transformation<PT, PT> {
	let mut rng = rand::rng();
	let bases: Vec<_> = (0..PT::Scalar::DEGREE)
		.map(|_| PT::Scalar::random(&mut rng))
		.collect();
	let transformation = FieldLinearTransformation::<PT::Scalar, Vec<PT::Scalar>>::new(bases);

	PT::make_packed_transformation(transformation)
}

cfg_if! {
	if #[cfg(feature = "benchmark_alternative_strategies")] {
		use binius_field::{
			arch::{PackedStrategy,  PairwiseStrategy, SimdStrategy,},
			arithmetic_traits::TaggedPackedTransformationFactory
		};

		trait TaggedTransformToSelfFactory<Strategy>:
		TaggedPackedTransformationFactory<Strategy, Self>
		{
		}

		impl<Strategy, T: TaggedPackedTransformationFactory<Strategy, Self>>
		TaggedTransformToSelfFactory<Strategy> for T
		{
		}

		fn create_transformation_pairwise<PT: TaggedTransformToSelfFactory<PairwiseStrategy>>(
		) -> impl Transformation<PT, PT> {
			let mut rng = rand::rng();
			let bases: Vec<_> = (0..PT::Scalar::DEGREE)
				.map(|_| PT::Scalar::random(&mut rng))
				.collect();
			let transformation = FieldLinearTransformation::<PT::Scalar, Vec<PT::Scalar>>::new(bases);

			PT::make_packed_transformation(transformation)
		}

		fn create_transformation_packed<PT: TaggedTransformToSelfFactory<PackedStrategy>>(
		) -> impl Transformation<PT, PT> {
			let mut rng = rand::rng();
			let bases: Vec<_> = (0..PT::Scalar::DEGREE)
				.map(|_| PT::Scalar::random(&mut rng))
				.collect();
			let transformation = FieldLinearTransformation::<PT::Scalar, Vec<PT::Scalar>>::new(bases);

			PT::make_packed_transformation(transformation)
		}

		fn create_transformation_simd<PT: TaggedTransformToSelfFactory<SimdStrategy>>(
		) -> impl Transformation<PT, PT> {
			let mut rng = rand::rng();
			let bases: Vec<_> = (0..PT::Scalar::DEGREE)
				.map(|_| PT::Scalar::random(&mut rng))
				.collect();
			let transformation = FieldLinearTransformation::<PT::Scalar, Vec<PT::Scalar>>::new(bases);

			PT::make_packed_transformation(transformation)
		}

		benchmark_packed_operation!(
			op_name @ linear_transform,
			bench_type @ transformation,
			strategies @ (
				(main, TransformToSelfFactory, create_transformation_main),
				(pairwise, TaggedTransformToSelfFactory::<PairwiseStrategy>, create_transformation_pairwise),
				(packed, TaggedTransformToSelfFactory::<PackedStrategy>, create_transformation_packed),
				(simd, TaggedTransformToSelfFactory::<SimdStrategy>, create_transformation_simd),
			)
		);
	} else {
		benchmark_packed_operation!(
			op_name @ linear_transform,
			bench_type @ transformation,
			strategies @ (
				(main, TransformToSelfFactory, create_transformation_main),
			)
		);
	}
}

criterion_main!(linear_transform);
