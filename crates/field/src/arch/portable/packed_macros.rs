// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

macro_rules! define_packed_binary_field {
	(
		$name:ident, $scalar:path, $underlier:ident,
		($($mul:tt)*),
		($($square:tt)*),
		($($invert:tt)*),
		($($wide_mul:tt)*)
	) => {
		// Define packed field types
		pub type $name = $crate::arch::PackedPrimitiveType<$underlier, $scalar>;

		// Serialization is provided by a single generic impl on `PackedPrimitiveType` (see
		// `packed.rs`), so no per-type impl is needed here.

		// Define multiplication
		impl_strategy!(impl_mul_with       $name, ($($mul)*));

		// Define square
		impl_strategy!(impl_square_with    $name, ($($square)*));

		// Define invert
		impl_strategy!(impl_invert_with    $name, ($($invert)*));

		// Define widening multiplication. Every packed field is a `WideMul` (it's a parent trait
		// of `PackedField`). The caller passes a wrapper struct (a `TransparentWrapper` around this
		// packed field) that carries the actual `WideMul` impl; here we forward to it by wrapping
		// the inputs and peeling the reduced result.
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

pub(crate) use crate::arithmetic_traits::{impl_invert_with, impl_mul_with, impl_square_with};

pub(crate) mod portable_macros {
	macro_rules! impl_strategy {
		($impl_macro:ident $name:ident, (None)) => {};
		// gfni condition: strategy types are in $crate::arch
		($impl_macro:ident $name:ident, (if gfni $strategy:tt else $fallback:tt)) => {
			cfg_if! {
				if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
					$impl_macro!($name @ $crate::arch::$strategy);
				} else {
					$impl_macro!($name @ $crate::arch::$fallback);
				}
			}
		};
		// Path to strategy in caller's scope
		($impl_macro:ident $name:ident, ($($strategy:tt)*)) => {
			$impl_macro!($name @ $($strategy)*);
		};
	}

	pub(crate) use impl_strategy;
}
