// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Reading a packed extension field element as a packing of its subfield.
//!
//! A packed extension field element and its subfield packing are the same bytes.
//! They differ only in how wide a scalar those bytes are cut into.
//! So every cast here is a reinterpretation rather than a conversion, and compiles to nothing.
//!
//! The subfield scalars come out in [`ExtensionField`] basis order:
//!
//! ```text
//! cast_bases_mut(exts)  ==  exts.iter().flat_map(|ext| ext.iter_bases())
//! ```
//!
//! The bare names do not say what is being cast, so call sites read best qualified by the module:
//!
//! ```text
//! packed_extension::cast_base::<B1, _>(elem)
//! ```

use crate::{
	BinaryField, ExtensionField, PackedField, arch::PackedPrimitiveType, underlier::WithUnderlier,
};

/// The packing of `FSub` covering the same bits as the packed extension field type `P`.
///
/// A transparent wrapper over an underlier holds one contiguous bit string.
/// Cutting that same string into `FSub` scalars is exactly a [`PackedPrimitiveType`].
pub type PackedSubfield<P, FSub> = PackedPrimitiveType<<P as WithUnderlier>::Underlier, FSub>;

/// Reads a packed extension field element as a packed subfield element.
pub fn cast_base<FSub, P>(ext: P) -> PackedSubfield<P, FSub>
where
	FSub: BinaryField,
	P: PackedField<Scalar: ExtensionField<FSub>> + WithUnderlier,
	PackedSubfield<P, FSub>: PackedField<Scalar = FSub>,
{
	PackedSubfield::<P, FSub>::from_underlier(ext.to_underlier())
}

/// Reads a packed extension field element in place as a packed subfield element.
pub fn cast_base_mut<FSub, P>(packed: &mut P) -> &mut PackedSubfield<P, FSub>
where
	FSub: BinaryField,
	P: PackedField<Scalar: ExtensionField<FSub>> + WithUnderlier,
	PackedSubfield<P, FSub>: PackedField<Scalar = FSub>,
{
	PackedSubfield::<P, FSub>::from_underlier_ref_mut(packed.to_underlier_ref_mut())
}

/// Reads a packed subfield element as a packed extension field element.
pub fn cast_ext<FSub, P>(base: PackedSubfield<P, FSub>) -> P
where
	FSub: BinaryField,
	P: PackedField<Scalar: ExtensionField<FSub>> + WithUnderlier,
	PackedSubfield<P, FSub>: PackedField<Scalar = FSub>,
{
	P::from_underlier(base.to_underlier())
}

/// Reads a slice of packed extension field elements in place as packed subfield elements.
pub fn cast_bases_mut<FSub, P>(packed: &mut [P]) -> &mut [PackedSubfield<P, FSub>]
where
	FSub: BinaryField,
	P: PackedField<Scalar: ExtensionField<FSub>> + WithUnderlier,
	PackedSubfield<P, FSub>: PackedField<Scalar = FSub>,
{
	PackedSubfield::<P, FSub>::from_underliers_ref_mut(P::to_underliers_ref_mut(packed))
}
