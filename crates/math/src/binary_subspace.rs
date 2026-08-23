// Copyright 2024-2025 Irreducible Inc.

//! Binary subspaces: $\mathbb{F}_2$-linear spans of a binary field, enumerated in order.
//!
//! A subspace element is a subset-XOR of ordered basis elements, chosen by the bits of an index.
//!
//! Walking the elements in order is a binary-counter increment.
//! Each step XORs in or out only the basis elements whose bit changed.

use std::ops::Deref;

use binius_field::{BinaryField, BinaryField1b};

/// An $\mathbb{F}_2$-linear subspace of a binary field.
///
/// The subspace is the span of an ordered basis under XOR.
/// The basis order fixes an order on the subspace's own elements too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinarySubspace<F, Data: Deref<Target = [F]> = Vec<F>> {
	/// Ordered basis elements the subspace is spanned by.
	basis: Data,
}

impl<F: BinaryField, Data: Deref<Target = [F]>> BinarySubspace<F, Data> {
	/// Creates a new subspace from a vector of ordered basis elements.
	///
	/// This constructor does not check that the basis elements are linearly independent.
	pub const fn new_unchecked(basis: Data) -> Self {
		Self { basis }
	}

	/// Creates a new subspace isomorphic to this one, over a different field type.
	///
	/// Maps each basis element into `FIso` via `From`, keeping the same order.
	pub fn isomorphic<FIso>(&self) -> BinarySubspace<FIso>
	where
		FIso: BinaryField + From<F>,
	{
		BinarySubspace {
			// Convert every basis element into the target field, in the same order.
			basis: self.basis.iter().copied().map(Into::into).collect(),
		}
	}

	/// Returns the dimension of the subspace.
	pub fn dim(&self) -> usize {
		self.basis.len()
	}

	/// Returns the slice of ordered basis elements.
	pub fn basis(&self) -> &[F] {
		&self.basis
	}

	/// Returns the subspace element selected by `index`.
	///
	/// Bit `i` of `index` selects whether basis element `i` is included in the sum.
	/// Basis elements combine over $\mathbb{F}_2$, so "included" means XORed in.
	///
	/// # Arguments
	/// - `index`: which subspace element to return, in `0..2^dim`.
	///
	/// # Panics
	/// Panics if `index` is at least `2^dim`.
	pub fn get(&self, index: usize) -> F {
		assert!(index < 1 << self.dim(), "precondition: index must be less than 2^dim");

		element_at(&self.basis, index)
	}

	/// Returns an iterator over every element of the subspace, in index order.
	///
	/// # Panics
	/// Panics if the subspace's dimension is at least `usize::BITS`.
	/// An index that large would not fit in a `usize`.
	pub fn iter(&self) -> BinarySubspaceIterator<'_, F> {
		BinarySubspaceIterator::new(&self.basis)
	}
}

impl<F: BinaryField> BinarySubspace<F> {
	/// Creates a subspace spanned by the field's first `dim` default basis elements.
	///
	/// Uses a prefix of the field's own canonical $\mathbb{F}_2$ basis.
	/// So a smaller dimension is always a prefix of a larger one's basis.
	///
	/// # Panics
	/// Panics if `dim` is greater than `F::DEGREE`.
	pub fn with_dim(dim: usize) -> Self {
		assert!(dim <= F::DEGREE, "precondition: dim must be at most F::DEGREE");

		// Take the field's own first `dim` basis elements, in order.
		let basis = (0..dim).map(|i| F::basis(i)).collect();
		Self { basis }
	}

	/// Creates a smaller subspace using a prefix of this subspace's basis.
	///
	/// # Panics
	/// Panics if `dim` is greater than this subspace's own dimension.
	pub fn reduce_dim(&self, dim: usize) -> Self {
		assert!(dim <= self.dim(), "precondition: dim must be at most this subspace's dimension");

		Self {
			basis: self.basis[..dim].to_vec(),
		}
	}
}

/// Computes the subset-XOR of `basis` selected by the bits of `index`.
fn element_at<F: BinaryField>(basis: &[F], index: usize) -> F {
	basis
		.iter()
		.enumerate()
		// Keep basis_i when bit i of index is set.
		// Drop it (multiply by 0) otherwise.
		.map(|(i, &basis_i)| basis_i * BinaryField1b::from((index >> i) & 1 == 1))
		.sum()
}

/// Iterator over every element of a binary subspace, in index order.
///
/// Each element is a subset-XOR of the basis elements.
/// Stepping forward reuses the previous value instead of recomputing a full subset sum.
/// Skipping ahead computes the landing value directly instead.
#[derive(Debug, Clone)]
pub struct BinarySubspaceIterator<'a, F> {
	/// The subspace's ordered basis elements.
	basis: &'a [F],
	/// Index of the next element this iterator will yield.
	index: usize,
	/// The next element's value, precomputed so stepping never redoes a full sum.
	next: Option<F>,
}

impl<'a, F: BinaryField> BinarySubspaceIterator<'a, F> {
	/// Creates an iterator starting at index 0.
	///
	/// # Panics
	/// Panics if the basis has `usize::BITS` or more elements.
	/// An index that large would not fit in a `usize`.
	pub fn new(basis: &'a [F]) -> Self {
		assert!(basis.len() < usize::BITS as usize);

		// Index 0 selects no basis elements, so its value is always zero.
		Self {
			basis,
			index: 0,
			next: Some(F::ZERO),
		}
	}
}

impl<'a, F: BinaryField> Iterator for BinarySubspaceIterator<'a, F> {
	type Item = F;

	/// Advances to the next index.
	/// Reuses the previous value instead of recomputing a full subset sum.
	///
	/// # Algorithm
	/// Moving from `index` to `index + 1` is a binary-counter increment.
	/// A run of trailing 1-bits flips to 0, then the next 0-bit flips to 1.
	///
	/// A bit flip here means XOR-ing a basis element in or out.
	/// So the update only touches elements whose bit actually changed.
	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		let ret = self.next?;

		// Length of the trailing run of 1-bits.
		// Found with one hardware instruction, not a bit-by-bit scan.
		let ones = self.index.trailing_ones() as usize;

		// Undo every bit in that run: each one flips to 0, so XOR its basis element back out.
		let mut next = ret;
		for &basis_i in &self.basis[..ones] {
			next -= basis_i;
		}

		// The bit right after that run flips from 0 to 1: XOR its basis element in.
		// No such bit left in the basis means this was the last element.
		self.next = self.basis.get(ones).map(|&basis_i| next + basis_i);

		self.index += 1;
		Some(ret)
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		// Total element count is 2^dim.
		// Subtract how many indices are already past.
		let last = 1 << self.basis.len();
		let remaining = last - self.index;
		(remaining, Some(remaining))
	}

	/// Skips ahead `n` elements, computing the landing value directly.
	/// Never steps through everything in between.
	///
	/// Exhausts the iterator instead of panicking when `n` overflows the index.
	/// Also exhausts it if the landing point is past the last element.
	fn nth(&mut self, n: usize) -> Option<Self::Item> {
		match self.index.checked_add(n) {
			// Lands inside range: jump straight to that index's value.
			Some(new_index) if new_index < 1 << self.basis.len() => {
				self.index = new_index;
				self.next = Some(element_at(self.basis, new_index));
			}
			// Overflowed, or landed past the end: exhaust the iterator.
			_ => {
				self.index = 1 << self.basis.len();
				self.next = None;
			}
		}

		self.next()
	}
}

impl<'a, F: BinaryField> ExactSizeIterator for BinarySubspaceIterator<'a, F> {
	fn len(&self) -> usize {
		// Same total as the length hint, without the Option wrapper this trait doesn't need.
		let last = 1 << self.basis.len();
		last - self.index
	}
}

impl<'a, F: BinaryField> std::iter::FusedIterator for BinarySubspaceIterator<'a, F> {}

impl<F: BinaryField> Default for BinarySubspace<F> {
	/// The default subspace spans the whole field, using its full canonical basis.
	fn default() -> Self {
		// Every basis element of the field, in canonical order.
		let basis = (0..F::DEGREE).map(|i| F::basis(i)).collect();
		Self { basis }
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{AESTowerField8b as B8, Field, Ghash128b as B128};

	use super::*;

	#[test]
	fn test_default_binary_subspace_iterates_elements() {
		// The default basis for an 8-bit field is the powers of two.
		// So get(i) reconstructs i exactly, for every byte value.
		let subspace = BinarySubspace::<B8>::default();
		for i in 0..=255 {
			assert_eq!(subspace.get(i), B8::new(i as u8));
		}
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn test_binary_subspace_range_error() {
		// dim = 8, so valid indices are 0..256; 256 is one past the last valid index.
		let subspace = BinarySubspace::<B8>::default();
		let _ = subspace.get(256);
	}

	#[test]
	fn test_default_binary_subspace() {
		let subspace = BinarySubspace::<B8>::default();
		assert_eq!(subspace.dim(), 8);
		assert_eq!(subspace.basis().len(), 8);

		// The default basis is the field's own bits, in order: 1, 2, 4, ..., 128.
		assert_eq!(
			subspace.basis(),
			[
				B8::new(0b00000001),
				B8::new(0b00000010),
				B8::new(0b00000100),
				B8::new(0b00001000),
				B8::new(0b00010000),
				B8::new(0b00100000),
				B8::new(0b01000000),
				B8::new(0b10000000)
			]
		);

		// With that basis, index and value coincide: get(i) is just i itself.
		let expected_elements: [u8; 256] = (0..=255).collect::<Vec<_>>().try_into().unwrap();

		for (i, &expected) in expected_elements.iter().enumerate() {
			assert_eq!(subspace.get(i), B8::new(expected));
		}
	}

	#[test]
	fn test_with_dim_valid() {
		// A 3-dimensional subspace only uses the field's first 3 basis elements: 1, 2, 4.
		let subspace = BinarySubspace::<B8>::with_dim(3);
		assert_eq!(subspace.dim(), 3);
		assert_eq!(subspace.basis().len(), 3);

		assert_eq!(subspace.basis(), [B8::new(0b001), B8::new(0b010), B8::new(0b100)]);

		// So it spans exactly the 8 values expressible in 3 bits: 0..8.
		let expected_elements: [u8; 8] = [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111];

		for (i, &expected) in expected_elements.iter().enumerate() {
			assert_eq!(subspace.get(i), B8::new(expected));
		}
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn test_with_dim_invalid() {
		// B8 has degree 8, so dimension 10 is out of range.
		let _ = BinarySubspace::<B8>::with_dim(10);
	}

	#[test]
	fn test_reduce_dim_valid() {
		// Start from a 6-dimensional subspace, then keep only its first 4 basis elements.
		let subspace = BinarySubspace::<B8>::with_dim(6);
		let reduced = subspace.reduce_dim(4);
		assert_eq!(reduced.dim(), 4);
		assert_eq!(reduced.basis().len(), 4);

		// A prefix of the basis, so this matches with_dim(4) exactly.
		assert_eq!(
			reduced.basis(),
			[
				B8::new(0b0001),
				B8::new(0b0010),
				B8::new(0b0100),
				B8::new(0b1000)
			]
		);

		let expected_elements: [u8; 16] = (0..16).collect::<Vec<_>>().try_into().unwrap();

		for (i, &expected) in expected_elements.iter().enumerate() {
			assert_eq!(reduced.get(i), B8::new(expected));
		}
	}

	#[test]
	#[should_panic(expected = "precondition")]
	fn test_reduce_dim_invalid() {
		// Can't reduce to a larger dimension than the subspace already has.
		let subspace = BinarySubspace::<B8>::with_dim(4);
		let _ = subspace.reduce_dim(6);
	}

	#[test]
	fn test_isomorphic_conversion() {
		let subspace = BinarySubspace::<B8>::with_dim(3);
		// Re-express the same 3 basis elements as values of a much larger field.
		let iso_subspace: BinarySubspace<B128> = subspace.isomorphic();
		assert_eq!(iso_subspace.dim(), 3);
		assert_eq!(iso_subspace.basis().len(), 3);

		// Same basis values, just converted into B128 via From, in the same order.
		assert_eq!(
			iso_subspace.basis(),
			[
				B128::from(B8::new(0b001)),
				B128::from(B8::new(0b010)),
				B128::from(B8::new(0b100)),
			]
		);
	}

	#[test]
	fn test_iterate_subspace() {
		let subspace = BinarySubspace::<B8>::with_dim(3);
		// Collecting the iterator gives exactly the 8 elements of a 3-dim subspace.
		let elements: Vec<_> = subspace.iter().collect();
		assert_eq!(elements.len(), 8);

		let expected_elements: [u8; 8] = [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111];

		for (i, &expected) in expected_elements.iter().enumerate() {
			assert_eq!(elements[i], B8::new(expected));
		}
	}

	#[test]
	fn test_iterator_matches_get() {
		let subspace = BinarySubspace::<B8>::with_dim(5);

		// The incremental iterator and the direct per-index formula must always agree.
		for (i, elem) in subspace.iter().enumerate() {
			assert_eq!(elem, subspace.get(i), "Mismatch at index {}", i);
		}
	}

	#[test]
	#[allow(clippy::iter_nth_zero)]
	fn test_iterator_nth() {
		let subspace = BinarySubspace::<B8>::with_dim(4);

		// nth(0) behaves like next(): returns the very next element, advances by 1.
		let mut iter = subspace.iter();
		assert_eq!(iter.nth(0), Some(subspace.get(0)));
		assert_eq!(iter.nth(0), Some(subspace.get(1)));
		// Larger skips land on the element that many steps further along.
		assert_eq!(iter.nth(2), Some(subspace.get(4)));
		assert_eq!(iter.nth(5), Some(subspace.get(10)));

		// Landing exactly on the last valid index still works.
		let mut iter = subspace.iter();
		assert_eq!(iter.nth(15), Some(subspace.get(15)));
		// One more step past the end exhausts the iterator.
		assert_eq!(iter.nth(0), None);
	}

	#[test]
	fn test_iterator_nth_skips_efficiently() {
		let subspace = BinarySubspace::<B8>::with_dim(6);

		// Jump straight to index 30, without stepping through 0..30 first.
		let mut iter = subspace.iter();
		assert_eq!(iter.nth(30), Some(subspace.get(30)));
		// A plain next() afterward continues from exactly where the jump landed.
		assert_eq!(iter.next(), Some(subspace.get(31)));

		// A larger single jump works the same way.
		let mut iter = subspace.iter();
		assert_eq!(iter.nth(50), Some(subspace.get(50)));
	}

	#[test]
	fn test_iterator_size_hint() {
		let subspace = BinarySubspace::<B8>::with_dim(3);
		let mut iter = subspace.iter();

		// 3 dimensions means 8 elements total, all still ahead at the start.
		assert_eq!(iter.size_hint(), (8, Some(8)));
		iter.next();
		assert_eq!(iter.size_hint(), (7, Some(7)));
		// Skipping 3 ahead accounts for all 4 consumed elements at once.
		iter.nth(3);
		assert_eq!(iter.size_hint(), (3, Some(3)));
	}

	#[test]
	fn test_iterator_exact_size() {
		let subspace = BinarySubspace::<B8>::with_dim(4);
		let mut iter = subspace.iter();

		assert_eq!(iter.len(), 16);
		iter.next();
		assert_eq!(iter.len(), 15);
		iter.nth(5);
		assert_eq!(iter.len(), 9);
	}

	#[test]
	fn test_iterator_empty_subspace() {
		// Dimension 0 has exactly one element: the empty XOR-sum, zero.
		let subspace = BinarySubspace::<B8>::with_dim(0);
		let mut iter = subspace.iter();

		assert_eq!(iter.len(), 1);
		assert_eq!(iter.next(), Some(B8::ZERO));
		assert_eq!(iter.next(), None);
	}

	#[test]
	fn test_iterator_full_iteration() {
		// The full 8-bit field has 256 elements.
		// The iterator must produce all of them, matching get() at every index.
		let subspace = BinarySubspace::<B8>::default();
		let collected: Vec<_> = subspace.iter().collect();

		assert_eq!(collected.len(), 256);
		for (i, elem) in collected.iter().enumerate() {
			assert_eq!(*elem, subspace.get(i));
		}
	}

	#[test]
	fn test_iterator_partial_then_nth() {
		let subspace = BinarySubspace::<B8>::with_dim(5);
		let mut iter = subspace.iter();

		// Step through the first 3 elements one at a time.
		assert_eq!(iter.next(), Some(subspace.get(0)));
		assert_eq!(iter.next(), Some(subspace.get(1)));
		assert_eq!(iter.next(), Some(subspace.get(2)));

		// Jump ahead 5 more (landing on index 8), then continue normally.
		assert_eq!(iter.nth(5), Some(subspace.get(8)));
		assert_eq!(iter.next(), Some(subspace.get(9)));
	}

	#[test]
	fn test_iterator_clone() {
		let subspace = BinarySubspace::<B8>::with_dim(3);
		let mut iter1 = subspace.iter();

		iter1.next();
		iter1.next();

		// Cloning mid-iteration copies the current position, not a fresh start.
		let mut iter2 = iter1.clone();

		assert_eq!(iter1.next(), iter2.next());
		assert_eq!(iter1.collect::<Vec<_>>(), iter2.collect::<Vec<_>>());
	}
}
