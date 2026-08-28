// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

/// Defines a packed binary field over an underlier, with the scalar it packs and its arithmetic.
///
/// Squaring, inversion, the prepared multiply and the widening multiply are supplied per target
/// as strategies.
/// Multiplication is always the widening multiply followed by its reduction, so the two cannot
/// disagree.
macro_rules! define_packed_binary_field {
	(
		$name:ident, $scalar:path, $underlier:ident,
		($($square:tt)*),
		($($invert:tt)*),
		($($wide_mul:tt)*),
		($($prepared_mul:tt)*)
	) => {
		pub type $name = $crate::packed_fields::primitive::PackedPrimitiveType<$underlier, $scalar>;

		impl std::ops::Mul for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				<Self as $crate::arithmetic_traits::WideMul>::reduce(
					<Self as $crate::arithmetic_traits::WideMul>::wide_mul(self, rhs),
				)
			}
		}

		impl_square_with!($name @ $($square)*);

		impl_invert_with!($name @ $($invert)*);

		impl_prepared_mul_with!($name @ $($prepared_mul)*);

		impl $crate::arithmetic_traits::WideMul for $name {
			type Output =
				<$($wide_mul)* <$name> as $crate::arithmetic_traits::WideMul>::Output;

			#[inline]
			fn wide_mul(a: Self, b: Self) -> Self::Output {
				<$($wide_mul)* <$name> as $crate::arithmetic_traits::WideMul>::wide_mul(
					<$($wide_mul)* <$name> as ::bytemuck::TransparentWrapper<$name>>::wrap(a),
					<$($wide_mul)* <$name> as ::bytemuck::TransparentWrapper<$name>>::wrap(b),
				)
			}

			#[inline]
			fn reduce(wide: Self::Output) -> Self {
				<$($wide_mul)* <$name> as ::bytemuck::TransparentWrapper<$name>>::peel(
					<$($wide_mul)* <$name> as $crate::arithmetic_traits::WideMul>::reduce(wide),
				)
			}
		}
	};
}

pub(crate) use define_packed_binary_field;

pub(crate) use crate::arithmetic_traits::{
	impl_invert_with, impl_prepared_mul_with, impl_square_with,
};
