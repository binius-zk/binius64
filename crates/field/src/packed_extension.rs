// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	BinaryField, ExtensionField, PackedField, arch::PackedPrimitiveType, underlier::WithUnderlier,
};

/// The packed subfield type sharing the same underlier as the packed extension field type `P`.
///
/// `P` and `PackedSubfield<P, FSub>` have the same in-memory representation, differing only in the
/// scalar type and preserving the order of the smaller elements. This is what makes the reinterpret
/// casts ([`cast_bases_mut`], [`cast_base_mut`], [`cast_ext`]) sound: the `FSub` scalars of
/// `cast_bases_mut(exts)` are exactly the scalars yielded by
/// `exts.iter().flat_map(|ext| ext.into_iter_bases())`.
pub type PackedSubfield<P, FSub> = PackedPrimitiveType<<P as WithUnderlier>::Underlier, FSub>;

/// Reinterpret a mutable slice of packed extension field elements as a mutable slice of the
/// corresponding packed subfield elements.
///
/// The two slices share the same memory; `PackedSubfield<P, FSub>` has the same underlier as `P`.
pub fn cast_bases_mut<FSub, P>(packed: &mut [P]) -> &mut [PackedSubfield<P, FSub>]
where
	FSub: BinaryField,
	P: PackedField<Scalar: ExtensionField<FSub>> + WithUnderlier,
	PackedSubfield<P, FSub>: PackedField<Scalar = FSub>,
{
	PackedSubfield::<P, FSub>::from_underliers_ref_mut(P::to_underliers_ref_mut(packed))
}

/// Reinterpret a mutable reference to a packed extension field element as a mutable reference to
/// the corresponding packed subfield element.
pub fn cast_base_mut<FSub, P>(packed: &mut P) -> &mut PackedSubfield<P, FSub>
where
	FSub: BinaryField,
	P: PackedField<Scalar: ExtensionField<FSub>> + WithUnderlier,
	PackedSubfield<P, FSub>: PackedField<Scalar = FSub>,
{
	PackedSubfield::<P, FSub>::from_underlier_ref_mut(packed.to_underlier_ref_mut())
}

/// Reinterpret a packed subfield element as the corresponding packed extension field element.
pub fn cast_ext<FSub, P>(base: PackedSubfield<P, FSub>) -> P
where
	FSub: BinaryField,
	P: PackedField<Scalar: ExtensionField<FSub>> + WithUnderlier,
	PackedSubfield<P, FSub>: PackedField<Scalar = FSub>,
{
	P::from_underlier(base.to_underlier())
}
