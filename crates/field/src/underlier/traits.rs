// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	fmt::Debug,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

use bytemuck::{NoUninit, TransparentWrapper, Zeroable};

use super::U1;
use crate::{Divisible, Random};

/// A fixed-length vector of bits, whose length is a power of two.
///
/// This is the storage a binary field element lives in.
/// An element is a bit pattern.
/// This trait is the interface for holding that pattern and moving it around.
///
/// The same bits can be read two ways, and the interface serves both:
///
/// ```text
/// BITS = 32
///
/// one field element    [ -------------- x -------------- ]
/// eight packed ones    [ x7 x6 x5 x4 x3 x2 x1 x0 ]
/// ```
///
/// Nothing here knows which reading is meant.
/// The bitwise operators act on every bit at once, so they are correct under either.
/// Addition in a binary field is exclusive or, which is why that operator is required.
///
/// # Why the length is a power of two
///
/// A value splits evenly in half, and each half splits again, down to single bits.
/// The two shuffling operations below walk that ladder one rung at a time.
/// A length like 24 bits would have no such ladder.
///
/// # Bit order
///
/// Bit 0 is the least significant.
/// Every diagram here lists the low end first.
/// That is the reverse of how a binary literal reads.
pub trait Underlier:
	Debug
	+ Default
	+ Eq
	+ Ord
	+ Copy
	+ Random
	+ NoUninit
	+ Zeroable
	+ Sized
	+ Send
	+ Sync
	+ 'static
	+ BitAnd<Self, Output = Self>
	+ BitAndAssign<Self>
	+ BitOr<Self, Output = Self>
	+ BitOrAssign<Self>
	+ BitXor<Self, Output = Self>
	+ BitXorAssign<Self>
	+ Not<Output = Self>
	+ Divisible<U1>
{
	/// Base-2 logarithm of the number of bits in a value.
	const LOG_BITS: usize;

	/// Number of bits in a value.
	///
	/// This can be fewer than the bits of the type that stores it.
	/// The one-, two-, and four-bit underliers each sit in a byte.
	/// Their spare high bits carry nothing.
	const BITS: usize = 1 << Self::LOG_BITS;

	/// Every bit clear.
	const ZERO: Self;

	/// Bit 0 set, every other bit clear.
	const ONE: Self;

	/// Every bit set.
	const ONES: Self;

	/// Exchanges alternating blocks of two values.
	///
	/// Cut both values into blocks of `2^log_block_len` bits, numbered from the low end.
	/// The first result takes the even-numbered blocks of each value, one after the other.
	/// The second result takes the odd-numbered ones the same way.
	///
	/// ```text
	/// BITS = 8, log_block_len = 1, so four blocks of two bits, low block first
	///
	///     self    [ a0 | a1 | a2 | a3 ]
	///     other   [ b0 | b1 | b2 | b3 ]
	///
	///     first   [ a0 | b0 | a2 | b2 ]
	///     second  [ a1 | b1 | a3 | b3 ]
	/// ```
	///
	/// This is one rung of the ladder that halves a value down to single bits.
	/// Repeating it at every rung is what the transpose below does.
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self);

	/// Separates two values into their even and odd blocks.
	///
	/// Cut both values into blocks of `2^log_block_len` bits, numbered from the low end.
	/// The first result collects every even-numbered block, this value's before the other's.
	/// The second result collects every odd-numbered block the same way.
	///
	/// ```text
	/// BITS = 8, log_block_len = 0, so eight blocks of one bit, low bit first
	///
	///     self    [ a0 a1 a2 a3 a4 a5 a6 a7 ]
	///     other   [ b0 b1 b2 b3 b4 b5 b6 b7 ]
	///
	///     first   [ a0 a2 a4 a6 b0 b2 b4 b6 ]
	///     second  [ a1 a3 a5 a7 b1 b3 b5 b7 ]
	/// ```
	///
	/// Lay the two values out as the two rows of a matrix whose entries are blocks.
	/// This reads that matrix out one column at a time, which is what makes it a transpose.
	///
	/// # Panics
	///
	/// Panics unless the block length is shorter than the whole value.
	fn transpose(mut self, mut other: Self, log_block_len: usize) -> (Self, Self) {
		assert!(log_block_len < Self::LOG_BITS);

		// Start at the widest block and halve it each round, exchanging at every rung.
		// After the last round every bit sits where its block index alone decides.
		// That is what turns a sequence of exchanges into a transpose.
		for log_block_len in (log_block_len..Self::LOG_BITS).rev() {
			(self, other) = self.interleave(other, log_block_len);
		}

		(self, other)
	}

	/// Builds a value by filling it with narrower ones, low slot first.
	///
	/// The two widths fix how many slots there are.
	/// The closure is called exactly that many times.
	#[inline]
	fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self
	where
		T: Underlier,
		Self: Divisible<T>,
	{
		Self::from_iter((0..<Self as Divisible<T>>::N).map(f))
	}
}

/// A type stored exactly as some underlier, and freely viewable as one.
///
/// A binary field element is a bit pattern with arithmetic attached.
/// The bits sit in an underlier, and the element type wraps it to give those bits meaning.
///
/// Declaring that wrapper transparent means the two share an address, a size, and a bit pattern:
///
/// ```text
///     element    [ bits ]  <- arithmetic attached
///     underlier  [ bits ]  <- same address, same size, nothing to convert
/// ```
///
/// So a value, a reference, or a whole slice can be viewed as the other side for free.
/// Viewing a slice matters most, since it lets bulk code work on plain bits without copying.
///
/// The wrapping alone would be expressible with conversions in both directions.
/// What those cannot give is the underlier's name.
/// Generic code needs that name to state bounds against it.
/// Carrying it as an associated type is what this trait adds.
///
/// # Safety
///
/// An implementor must have the same representation as the underlier it names.
/// That is what makes casting a reference in either direction sound.
pub unsafe trait UnderlierView:
	TransparentWrapper<Self::Underlier> + Sized + Zeroable + Copy + Send + Sync + 'static
{
	/// The underlier holding this type's bits.
	type Underlier: Underlier;

	/// Views this value as its underlier.
	#[inline]
	fn to_underlier(self) -> Self::Underlier {
		Self::peel(self)
	}

	/// Views a shared reference as one to its underlier.
	#[inline]
	fn to_underlier_ref(&self) -> &Self::Underlier {
		Self::peel_ref(self)
	}

	/// Views a mutable reference as one to its underlier.
	#[inline]
	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		Self::peel_mut(self)
	}

	/// Views a slice as a slice of underliers, without copying.
	#[inline]
	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		Self::peel_slice(val)
	}

	/// Views a mutable slice as a mutable slice of underliers, without copying.
	#[inline]
	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		Self::peel_slice_mut(val)
	}

	/// Reads an underlier as this type.
	#[inline]
	fn from_underlier(val: Self::Underlier) -> Self {
		Self::wrap(val)
	}

	/// Views a shared reference to an underlier as one to this type.
	#[inline]
	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		Self::wrap_ref(val)
	}

	/// Views a mutable reference to an underlier as one to this type.
	#[inline]
	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		Self::wrap_mut(val)
	}

	/// Views a slice of underliers as a slice of this type, without copying.
	#[inline]
	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		Self::wrap_slice(val)
	}

	/// Views a mutable slice of underliers as a mutable slice of this type, without copying.
	#[inline]
	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		Self::wrap_slice_mut(val)
	}

	/// Rewrites the bits through a function on the underlier, keeping this type on both ends.
	#[inline]
	fn mutate_underlier(self, f: impl FnOnce(Self::Underlier) -> Self::Underlier) -> Self {
		Self::from_underlier(f(self.to_underlier()))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::underlier::{U2, U4};

	#[test]
	fn test_from_fn() {
		assert_eq!(u32::from_fn(|_| U1::new(0)), 0);
		assert_eq!(u32::from_fn(|i| U1::new((i % 2) as u8)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U1::new(1)), u32::MAX);

		assert_eq!(u32::from_fn(|_| U2::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U2::new(1)), 0x55555555);
		assert_eq!(u32::from_fn(|_| U2::new(2)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U2::new(3)), u32::MAX);
		assert_eq!(u32::from_fn(|i| U2::new((i % 4) as u8)), 0xe4e4e4e4);

		assert_eq!(u32::from_fn(|_| U4::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U4::new(1)), 0x11111111);
		assert_eq!(u32::from_fn(|_| U4::new(8)), 0x88888888);
		assert_eq!(u32::from_fn(|_| U4::new(31)), 0xffffffff);
		assert_eq!(u32::from_fn(|i| U4::new(i as u8)), 0x76543210);

		assert_eq!(u32::from_fn(|_| 0u8), 0);
		assert_eq!(u32::from_fn(|_| 0xabu8), 0xabababab);
		assert_eq!(u32::from_fn(|_| 255u8), 0xffffffff);
		assert_eq!(u32::from_fn(|i| i as u8), 0x03020100);
	}

	/// Reads a value as its bits, low bit first, the way the diagrams above are drawn.
	fn bits(value: u8) -> [u8; 8] {
		std::array::from_fn(|i| (value >> i) & 1)
	}

	/// Packs bits given low bit first back into a value.
	fn pack(bits: [u8; 8]) -> u8 {
		bits.iter()
			.enumerate()
			.fold(0, |acc, (i, bit)| acc | (bit << i))
	}

	#[test]
	fn interleave_exchanges_alternating_blocks() {
		// Two values whose bits are all distinguishable by position.
		let a = 0b1010_1010u8;
		let b = 0b1100_1100u8;
		let (av, bv) = (bits(a), bits(b));

		// Blocks of one bit: the first result takes the even positions of each value in turn.
		//
		//     first   [ a0 b0 a2 b2 a4 b4 a6 b6 ]
		//     second  [ a1 b1 a3 b3 a5 b5 a7 b7 ]
		let (first, second) = a.interleave(b, 0);
		assert_eq!(first, pack([av[0], bv[0], av[2], bv[2], av[4], bv[4], av[6], bv[6]]));
		assert_eq!(second, pack([av[1], bv[1], av[3], bv[3], av[5], bv[5], av[7], bv[7]]));

		// Blocks of two bits: the same pattern, one rung up the ladder.
		//
		//     first   [ a0 a1 | b0 b1 | a4 a5 | b4 b5 ]
		let (first, second) = a.interleave(b, 1);
		assert_eq!(first, pack([av[0], av[1], bv[0], bv[1], av[4], av[5], bv[4], bv[5]]));
		assert_eq!(second, pack([av[2], av[3], bv[2], bv[3], av[6], av[7], bv[6], bv[7]]));
	}

	#[test]
	fn transpose_separates_even_blocks_from_odd() {
		let a = 0b1010_1010u8;
		let b = 0b1100_1100u8;
		let (av, bv) = (bits(a), bits(b));

		// Single-bit blocks: every even bit lands in the first result, this value's before the
		// other's, and every odd bit lands in the second.
		//
		//     first   [ a0 a2 a4 a6 b0 b2 b4 b6 ]
		//     second  [ a1 a3 a5 a7 b1 b3 b5 b7 ]
		let (first, second) = a.transpose(b, 0);
		assert_eq!(first, pack([av[0], av[2], av[4], av[6], bv[0], bv[2], bv[4], bv[6]]));
		assert_eq!(second, pack([av[1], av[3], av[5], av[7], bv[1], bv[3], bv[5], bv[7]]));
	}

	#[test]
	fn transpose_at_the_widest_block_is_a_single_exchange() {
		// One rung below the whole value leaves only one exchange to make, so the transpose and
		// the interleave agree there.
		let a = 0x3cu8;
		let b = 0xa5u8;
		assert_eq!(a.transpose(b, u8::LOG_BITS - 1), a.interleave(b, u8::LOG_BITS - 1));
	}

	#[test]
	#[should_panic(expected = "log_block_len < Self::LOG_BITS")]
	fn transpose_rejects_a_block_as_wide_as_the_value() {
		// A block covering the whole value has no rung to stand on.
		let _ = 0u8.transpose(0u8, u8::LOG_BITS);
	}
}
