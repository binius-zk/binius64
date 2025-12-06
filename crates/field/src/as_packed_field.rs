// Copyright 2024-2025 Irreducible Inc.

use crate::{
	Field, PackedField,
	underlier::{UnderlierType, WithUnderlier},
};

/// This trait represents correspondence (UnderlierType, Field) -> PackedField.
/// For example (u64, BinaryField16b) -> PackedBinaryField4x16b.
pub trait PackScalar<F: Field>: UnderlierType {
	type Packed: PackedField<Scalar = F> + WithUnderlier<Underlier = Self>;
}

/// Returns the packed field type for the scalar field `F` and underlier `U`.
pub type PackedType<U, F> = <U as PackScalar<F>>::Packed;
