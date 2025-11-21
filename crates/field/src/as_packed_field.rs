// Copyright 2024-2025 Irreducible Inc.

use crate::{
	BinaryField128bGhash, ExtensionField, Field, PackedBinaryGhash1x128b, PackedField,
	aes_field::*,
	arch::{packed_1::*, packed_aes_8::*},
	binary_field::*,
	underlier::{UnderlierType, WithUnderlier},
};

/// Trait that establishes correspondence between the scalar field and a packed field of the same
/// bit size with a single element.
///
/// E.g. `BinaryField64b` -> `PackedBinaryField1x64b`.
/// Note that the underlier of the packed field may be different.
/// E.g. `BinaryField128b` has u128 as underlier, while for x64 `PackedBinaryField1x128b`'s
/// underlier may be `M128`.
pub trait AsSinglePacked: Field {
	type Packed: PackedField<Scalar = Self>
		+ WithUnderlier<Underlier: From<Self::Underlier> + Into<Self::Underlier>>;

	fn to_single_packed(self) -> Self::Packed {
		assert_eq!(Self::Packed::WIDTH, 1);

		Self::Packed::set_single(self)
	}

	fn from_single_packed(value: Self::Packed) -> Self {
		assert_eq!(Self::Packed::WIDTH, 1);

		value.get(0)
	}
}

macro_rules! impl_as_single_packed_field {
	($field:ty, $packed_field:ty) => {
		impl AsSinglePacked for $field {
			type Packed = $packed_field;
		}
	};
}

impl_as_single_packed_field!(BinaryField1b, PackedBinaryField1x1b);
impl_as_single_packed_field!(AESTowerField8b, PackedAESBinaryField1x8b);
impl_as_single_packed_field!(BinaryField128bGhash, PackedBinaryGhash1x128b);

/// This trait represents correspondence (UnderlierType, Field) -> PackedField.
/// For example (u64, BinaryField16b) -> PackedBinaryField4x16b.
pub trait PackScalar<F: Field>: UnderlierType {
	type Packed: PackedField<Scalar = F> + WithUnderlier<Underlier = Self>;
}

/// Returns the packed field type for the scalar field `F` and underlier `U`.
pub type PackedType<U, F> = <U as PackScalar<F>>::Packed;

/// A trait to convert field to a same bit size packed field with some smaller scalar.
pub(crate) trait AsPackedField<Scalar: Field>: Field
where
	Self: ExtensionField<Scalar>,
{
	type Packed: PackedField<Scalar = Scalar>
		+ WithUnderlier<Underlier: From<Self::Underlier> + Into<Self::Underlier>>;
}

impl<Scalar, F> AsPackedField<Scalar> for F
where
	F: Field
		+ WithUnderlier<Underlier: PackScalar<Scalar>>
		+ AsSinglePacked
		+ ExtensionField<Scalar>,
	Scalar: Field,
{
	type Packed = <Self::Underlier as PackScalar<Scalar>>::Packed;
}
