// Copyright 2024-2025 Irreducible Inc.

macro_rules! impl_broadcast {
	($name:ty, BinaryField1b) => {
		impl $crate::arithmetic_traits::Broadcast<BinaryField1b>
			for PackedPrimitiveType<$name, BinaryField1b>
		{
			#[inline]
			fn broadcast(scalar: BinaryField1b) -> Self {
				use $crate::underlier::{UnderlierWithBitOps, WithUnderlier};

				<Self as WithUnderlier>::Underlier::fill_with_bit(scalar.0.into()).into()
			}
		}
	};
	($name:ty, $scalar_type:path) => {
		impl $crate::arithmetic_traits::Broadcast<$scalar_type>
			for PackedPrimitiveType<$name, $scalar_type>
		{
			#[inline]
			fn broadcast(scalar: $scalar_type) -> Self {
				let mut value = <$name>::from(scalar.0);
				// For PackedBinaryField1x128b, the log bits is 7, so this is
				// an empty range. This is safe behavior.
				#[allow(clippy::reversed_empty_ranges)]
				for i in <$scalar_type as $crate::binary_field::BinaryField>::N_BITS.ilog2()
					..<$name>::BITS.ilog2()
				{
					value = value << (1 << i) | value;
				}

				value.into()
			}
		}
	};
}

macro_rules! impl_ops_for_zero_height {
	($name:ty) => {
		impl std::ops::Mul for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			#[inline]
			fn mul(self, b: Self) -> Self {
				crate::tracing::trace_multiplication!($name);

				(self.to_underlier() & b.to_underlier()).into()
			}
		}

		impl $crate::arithmetic_traits::MulAlpha for $name {
			#[inline]
			fn mul_alpha(self) -> Self {
				self
			}
		}

		impl $crate::arithmetic_traits::Square for $name {
			#[inline]
			fn square(self) -> Self {
				self
			}
		}

		impl $crate::arithmetic_traits::InvertOrZero for $name {
			#[inline]
			fn invert_or_zero(self) -> Self {
				self
			}
		}
	};
}

macro_rules! define_packed_binary_fields {
    (
        underlier: $underlier:ident,
        packed_fields: [
            $(
                packed_field {
                    name: $name:ident,
                    scalar: $scalar:ident,
                    alpha_idx: $alpha_idx:tt,
                    mul: ($($mul:tt)*),
                    square: ($($square:tt)*),
                    invert: ($($invert:tt)*),
                    mul_alpha: ($($mul_alpha:tt)*),
                    transform: ($($transform:tt)*),
                }
            ),* $(,)?
        ]
    ) => {
        $(
            define_packed_binary_field!(
                $name,
                $crate::$scalar,
                $underlier,
                $alpha_idx,
                ($($mul)*),
                ($($square)*),
                ($($invert)*),
                ($($mul_alpha)*),
                ($($transform)*)
            );
        )*
    };
}

macro_rules! define_packed_binary_field {
	(
		$name:ident, $scalar:path, $underlier:ident, $alpha_idx:tt,
		($($mul:tt)*),
		($($square:tt)*),
		($($invert:tt)*),
		($($mul_alpha:tt)*),
		($($transform:tt)*)
	) => {
		// Define packed field types
		pub type $name = PackedPrimitiveType<$underlier, $scalar>;

		// Define serialization and deserialization
		impl_serialize_deserialize_for_packed_binary_field!($name);

		// Define broadcast
		maybe_impl_broadcast!($underlier, $scalar);

		// Define operations for height 0
		maybe_impl_ops!($name, $alpha_idx);

		// Define constants
		maybe_impl_tower_constants!($scalar, $underlier, $alpha_idx);

		// Define multiplication
		impl_strategy!(impl_mul_with       $name, ($($mul)*));

		// Define square
		impl_strategy!(impl_square_with    $name, ($($square)*));

		// Define invert
		impl_strategy!(impl_invert_with    $name, ($($invert)*));

		// Define multiply by alpha
		impl_strategy!(impl_mul_alpha_with $name, ($($mul_alpha)*));

		// Define linear transformations
		impl_transformation!($name, ($($transform)*));
	};
}

macro_rules! impl_serialize_deserialize_for_packed_binary_field {
	($bin_type:ty) => {
		impl binius_utils::SerializeBytes for $bin_type {
			fn serialize(
				&self,
				write_buf: impl binius_utils::bytes::BufMut,
			) -> Result<(), binius_utils::SerializationError> {
				self.0.serialize(write_buf)
			}
		}

		impl binius_utils::DeserializeBytes for $bin_type {
			fn deserialize(
				read_buf: impl binius_utils::bytes::Buf,
			) -> Result<Self, binius_utils::SerializationError> {
				Ok(Self(
					binius_utils::DeserializeBytes::deserialize(read_buf)?,
					std::marker::PhantomData,
				))
			}
		}
	};
}

macro_rules! maybe_impl_ops {
	($name:ident, 0) => {
		impl_ops_for_zero_height!($name);
	};
	($name:ident, $other_idx:tt) => {};
}

pub(crate) use define_packed_binary_field;
pub(crate) use define_packed_binary_fields;
pub(crate) use impl_broadcast;
pub(crate) use impl_ops_for_zero_height;
pub(crate) use impl_serialize_deserialize_for_packed_binary_field;
pub(crate) use maybe_impl_ops;

pub(crate) mod portable_macros {
	macro_rules! maybe_impl_broadcast {
		($underlier:ty, $scalar:path) => {
			impl_broadcast!($underlier, $scalar);
		};
	}

	macro_rules! maybe_impl_tower_constants {
		($scalar:path, $underlier:ty, _) => {};
		($scalar:path, $underlier:ty, $alpha_idx:tt) => {
			impl_tower_constants!($scalar, $underlier, { alphas!($underlier, $alpha_idx) });
		};
	}

	macro_rules! impl_strategy {
		($impl_macro:ident $name:ident, (None)) => {};
		($impl_macro:ident $name:ident, (if $cond:ident $gfni_x86_strategy:tt else $fallback:tt)) => {
			cfg_if! {
				if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
					$impl_macro!($name => $crate::$gfni_x86_strategy);
				} else {
					$impl_macro!($name @ $crate::arch::$fallback);
				}
			}
		};
		($impl_macro:ident $name:ident, ($strategy:ident)) => {
			$impl_macro!($name @ $crate::arch::$strategy);
		};
	}

	macro_rules! impl_transformation {
		($name:ident, ($strategy:ident)) => {
			impl_transformation_with_strategy!($name, $crate::arch::$strategy);
		};
	}

	pub(crate) use impl_strategy;
	pub(crate) use impl_transformation;
	pub(crate) use maybe_impl_broadcast;
	pub(crate) use maybe_impl_tower_constants;
}
