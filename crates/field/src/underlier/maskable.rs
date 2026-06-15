// Copyright 2026 The Binius Developers

use bytemuck::Zeroable;

use super::divisible::Divisible;

/// Branchless conditional lane selection for fields and packed fields.
///
/// `Maskable<T>` keeps a chosen subset of a value's `T`-lanes and zeroes the rest, against a
/// precomputed [`Self::Mask`]. It is the high-level, underlier-free primitive behind the branchless
/// masked accumulation in the shift protocol: a mask is built once with [`make_mask`] and reused
/// across many [`select`] calls, so the per-call cost is a single bitwise AND on a binary field.
///
/// It is a parent trait of [`Field`](crate::Field) (`Maskable<Self>`, a single lane) and
/// [`PackedField`](crate::PackedField) (`Maskable<Self::Scalar>`, one entry per packed lane),
/// mirroring how [`Divisible`] is a parent trait of both.
///
/// [`make_mask`]: Maskable::make_mask
/// [`select`]: Maskable::select
pub trait Maskable<T>: Divisible<T> + Copy + Zeroable {
	/// A precomputed mask selecting which `T`-lanes [`select`](Maskable::select) keeps.
	///
	/// `Send + Sync` so a precomputed mask table can be shared across threads (e.g. a parallel
	/// fold over a precomputed `Vec<Self::Mask>`).
	type Mask: Send + Sync;

	/// Builds a mask from per-lane boolean selectors, in LSB-to-MSB lane order (the same ordering
	/// as [`Divisible`]). Consumes at most [`Divisible::N`] selectors; any lane past the end of the
	/// iterator, or whose selector is `false`, is not selected.
	fn make_mask(selectors: impl Iterator<Item = bool>) -> Self::Mask;

	/// Returns a value keeping the lanes selected when `mask` was built and zeroing the rest.
	///
	/// For a `mask` built by `Self::make_mask(selectors)`, this equals
	///
	/// ```text
	/// Self::from_iter(zip(self.ref_iter(), selectors)
	///     .map(|(val, selected)| if selected { val } else { <zero> }))
	/// ```
	///
	/// but branchless: on a binary field each selected lane of the mask is all-ones and each
	/// unselected lane all-zeros, so this is a single bitwise AND.
	fn select(&self, mask: &Self::Mask) -> Self;
}
