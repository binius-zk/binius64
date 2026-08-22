// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	BinaryField, ExtensionField, PackedField, arch::PackedPrimitiveType, underlier::WithUnderlier,
};

/// A packed extension field that can also be read as a packing of its subfield `FSub`.
///
/// `Self` and [`Self::PackedSubfield`] cover the same bits in the same order.
/// They differ only in how wide a scalar those bits are cut into.
/// That is what makes all four casts free reinterpretations rather than conversions.
///
/// The subfield scalars come out in [`ExtensionField`] basis order:
///
/// ```text
/// P::cast_bases_mut(exts)  ==  exts.iter().flat_map(|ext| ext.iter_bases())
/// ```
pub trait PackedExtension<FSub: BinaryField>:
	PackedField<Scalar: ExtensionField<FSub>> + WithUnderlier
{
	/// The packing of `FSub` covering the same bits as `Self`.
	type PackedSubfield: PackedField<Scalar = FSub>;

	/// Reads a packed extension field element as a packed subfield element.
	fn cast_base(self) -> Self::PackedSubfield;

	/// Reads a packed extension field element in place as a packed subfield element.
	fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield;

	/// Reads a packed subfield element as a packed extension field element.
	fn cast_ext(base: Self::PackedSubfield) -> Self;

	/// Reads a slice of packed extension field elements in place as packed subfield elements.
	fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield];
}

/// A shorthand for [`PackedExtension::PackedSubfield`].
pub type PackedSubfield<P, FSub> = <P as PackedExtension<FSub>>::PackedSubfield;

/// A transparent wrapper over an underlier holds one contiguous bit string.
/// Cutting that same string into `FSub` scalars is exactly a [`PackedPrimitiveType`].
impl<FSub, P> PackedExtension<FSub> for P
where
	FSub: BinaryField,
	P: PackedField<Scalar: ExtensionField<FSub>> + WithUnderlier,
	PackedPrimitiveType<P::Underlier, FSub>: PackedField<Scalar = FSub>,
{
	type PackedSubfield = PackedPrimitiveType<P::Underlier, FSub>;

	#[inline]
	fn cast_base(self) -> Self::PackedSubfield {
		Self::PackedSubfield::from_underlier(self.to_underlier())
	}

	#[inline]
	fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield {
		Self::PackedSubfield::from_underlier_ref_mut(self.to_underlier_ref_mut())
	}

	#[inline]
	fn cast_ext(base: Self::PackedSubfield) -> Self {
		Self::from_underlier(base.to_underlier())
	}

	#[inline]
	fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield] {
		Self::PackedSubfield::from_underliers_ref_mut(Self::to_underliers_ref_mut(packed))
	}
}
