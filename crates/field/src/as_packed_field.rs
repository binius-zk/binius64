// Copyright 2024-2025 Irreducible Inc.

use crate::{
	ExtensionField, Field, PackedField,
	underlier::{UnderlierType, WithUnderlier},
};

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
	F: Field + WithUnderlier<Underlier: PackScalar<Scalar>> + ExtensionField<Scalar>,
	Scalar: Field,
{
	type Packed = <Self::Underlier as PackScalar<Scalar>>::Packed;
}
