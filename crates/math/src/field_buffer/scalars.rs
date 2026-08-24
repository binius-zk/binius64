// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Scalar-element iteration over a borrowed field buffer.

use std::ops::Deref;

use binius_field::{PackedField, packed::get_packed_slice_unchecked};

use super::FieldBuffer;

/// Iterator over the scalar elements of a borrowed buffer, in index order.
///
/// Reads one element out of the packed store per step, and stops at the logical length:
///
/// ```text
/// WIDTH = 4, log_len = 1
///
/// word:  [ s_0, s_1, dead, dead ]  ->  s_0, s_1
/// ```
///
/// A shared borrow is the only thing that yields this.
/// An element lives in a lane, so iteration copies it out rather than lending it.
/// A consuming form would hand back those same copies while destroying the buffer.
#[derive(Debug, Clone, Copy)]
pub struct Scalars<'a, P: PackedField> {
	/// The words holding the elements, exactly the buffer's live run.
	words: &'a [P],
	/// Index of the element to yield next.
	index: usize,
	/// Element count of the buffer, which is where iteration stops.
	len: usize,
}

impl<P: PackedField> Iterator for Scalars<'_, P> {
	type Item = P::Scalar;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if self.index == self.len {
			return None;
		}

		// Safety: the index is below the element count, and the words span every element.
		let scalar = unsafe { get_packed_slice_unchecked(self.words, self.index) };
		self.index += 1;
		Some(scalar)
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		let remaining = self.len - self.index;
		(remaining, Some(remaining))
	}
}

impl<P: PackedField> ExactSizeIterator for Scalars<'_, P> {}

impl<'a, P: PackedField, Data: Deref<Target = [P]>> IntoIterator for &'a FieldBuffer<P, Data> {
	type Item = P::Scalar;
	type IntoIter = Scalars<'a, P>;

	/// Iterates the buffer's scalar elements.
	///
	/// This is what puts a buffer in a `for` loop, in a collect, or in code generic over iterables.
	///
	/// The named scalar iterator remains the way hot code iterates.
	/// It takes the packed field's own specialized path, which this one cannot name.
	#[inline]
	fn into_iter(self) -> Self::IntoIter {
		Scalars {
			words: self.as_ref(),
			index: 0,
			len: self.len(),
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Field;

	use crate::{
		FieldBuffer,
		test_utils::{B128, Packed128b},
	};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn into_iter_over_a_borrow() {
		// Sums whatever it is handed, so it only compiles if a borrowed buffer is an iterable.
		fn sum_of<I: IntoIterator<Item = F>>(iterable: I) -> F {
			iterable
				.into_iter()
				.fold(F::ZERO, |acc, scalar| acc + scalar)
		}

		// Fixture state: 2 scalars in a 4-lane word, so 2 lanes are dead.
		//
		//     word = [s_0, s_1, dead, dead]
		let values = vec![F::new(10), F::new(20)];
		let buffer = FieldBuffer::<P>::from_values(&values);

		// A `for` loop over the borrow walks the live prefix, and never the dead lanes.
		let mut collected = Vec::new();
		for scalar in &buffer {
			collected.push(scalar);
		}
		assert_eq!(collected, values);
		assert_eq!(collected, buffer.iter_scalars().collect::<Vec<_>>());

		// The generic consumer sees the same 2 elements.
		assert_eq!(sum_of(&buffer), F::new(10) + F::new(20));

		// Above the packing width: 16 elements over 4 whole words.
		let values: Vec<F> = (0..16).map(F::new).collect();
		let buffer = FieldBuffer::<P>::from_values(&values);

		// The length is exact, so a collect can size its vector up front.
		let iter = IntoIterator::into_iter(&buffer);
		assert_eq!(iter.len(), 16);
		assert_eq!(iter.collect::<Vec<_>>(), values);
		assert_eq!(
			buffer.iter_scalars().collect::<Vec<_>>(),
			IntoIterator::into_iter(&buffer).collect::<Vec<_>>()
		);

		// A borrowed store iterates the same way an owned one does.
		let slice = buffer.as_view();
		assert_eq!(IntoIterator::into_iter(&slice).collect::<Vec<_>>(), values);
	}
}
