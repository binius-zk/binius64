// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::{iter, mem::MaybeUninit};

use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use super::{ValueIndex, ValueVec};
use crate::word::Word;

/// A different variants of shifting a value.
///
/// Note that there is no shift left arithmetic because it is redundant.
///
/// The discriminant is stored in a single byte.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ShiftVariant {
	/// Shift logical left.
	Sll = 0,
	/// Shift logical right.
	Slr = 1,
	/// Shift arithmetic right.
	///
	/// This is similar to the logical shift right but instead of shifting in 0 bits it will
	/// replicate the sign bit.
	Sar = 2,
	/// Rotate right.
	///
	/// Rotates bits to the right, with bits shifted off the right end wrapping around to the left.
	Rotr = 3,
	/// Shift logical left on 32-bit halves.
	///
	/// Performs independent logical left shifts on the upper and lower 32-bit halves of the word.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	Sll32 = 4,
	/// Shift logical right on 32-bit halves.
	///
	/// Performs independent logical right shifts on the upper and lower 32-bit halves of the word.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	Srl32 = 5,
	/// Shift arithmetic right on 32-bit halves.
	///
	/// Performs independent arithmetic right shifts on the upper and lower 32-bit halves of the
	/// word. Sign extends each 32-bit half independently. Only uses the lower 5 bits of the shift
	/// amount (0-31).
	Sra32 = 6,
	/// Rotate right on 32-bit halves.
	///
	/// Performs independent rotate right operations on the upper and lower 32-bit halves of the
	/// word. Bits shifted off the right end wrap around to the left within each 32-bit half.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	Rotr32 = 7,
}

/// A callback that runs against one concrete word-level shift.
///
/// Resolving a variant once turns it into a closure over words, handed to the callback here.
/// Each variant gets its own specialized copy, so the callback's loop keeps no per-word branch.
trait ShiftKernel {
	/// What the callback produces.
	type Output;

	/// Runs the callback against `shift`, the resolved word operation.
	fn call(self, shift: impl Fn(Word) -> Word) -> Self::Output;
}

/// Applies a resolved shift to a single word.
struct ShiftOneWord {
	/// The word the shift is applied to.
	word: Word,
}

impl ShiftKernel for ShiftOneWord {
	type Output = Word;

	#[inline]
	fn call(self, shift: impl Fn(Word) -> Word) -> Word {
		// A single word carries no loop to specialize, so the resolved shift runs once.
		shift(self.word)
	}
}

/// Writes one shifted source word into each output cell, initializing it.
struct WriteShiftedWords<'a> {
	/// The uninitialized output cells.
	out: &'a mut [MaybeUninit<Word>],
	/// The source words to shift, one per output cell.
	src: &'a [Word],
}

impl ShiftKernel for WriteShiftedWords<'_> {
	type Output = ();

	#[inline]
	fn call(self, shift: impl Fn(Word) -> Word) {
		// Positions line up one to one, so the pair iterator stops at the shorter slice.
		for (out_i, &src_i) in iter::zip(self.out, self.src) {
			out_i.write(shift(src_i));
		}
	}
}

/// XORs one shifted source word into each output cell.
struct XorShiftedWords<'a> {
	/// The output cells, each holding a running XOR.
	out: &'a mut [Word],
	/// The source words to shift, one per output cell.
	src: &'a [Word],
}

impl ShiftKernel for XorShiftedWords<'_> {
	type Output = ();

	#[inline]
	fn call(self, shift: impl Fn(Word) -> Word) {
		// Positions line up one to one, so the pair iterator stops at the shorter slice.
		for (out_i, &src_i) in iter::zip(self.out, self.src) {
			*out_i = *out_i ^ shift(src_i);
		}
	}
}

impl ShiftVariant {
	/// Every variant, ordered so that the array index equals the discriminant.
	///
	/// Callers that must cover all variants iterate this: random fixtures, exhaustive checks.
	pub const ALL: [Self; 8] = [
		Self::Sll,
		Self::Slr,
		Self::Sar,
		Self::Rotr,
		Self::Sll32,
		Self::Srl32,
		Self::Sra32,
		Self::Rotr32,
	];

	/// Decodes a variant from its `u8` discriminant.
	///
	/// The discriminants match the `#[repr(u8)]` layout: `0..=7` map to the eight variants.
	/// Any other byte returns `None`.
	#[inline]
	pub const fn from_u8(byte: u8) -> Option<Self> {
		match byte {
			0 => Some(ShiftVariant::Sll),
			1 => Some(ShiftVariant::Slr),
			2 => Some(ShiftVariant::Sar),
			3 => Some(ShiftVariant::Rotr),
			4 => Some(ShiftVariant::Sll32),
			5 => Some(ShiftVariant::Srl32),
			6 => Some(ShiftVariant::Sra32),
			7 => Some(ShiftVariant::Rotr32),
			_ => None,
		}
	}

	/// Whether this variant operates on the two 32-bit halves independently.
	///
	/// - The `*32` family shifts each half on its own.
	/// - It reads only the lower 5 bits of the amount.
	/// - Every other variant acts on the whole 64-bit word.
	#[inline]
	pub const fn is_half_word(self) -> bool {
		matches!(
			self,
			ShiftVariant::Sll32 | ShiftVariant::Srl32 | ShiftVariant::Sra32 | ShiftVariant::Rotr32
		)
	}

	/// The exclusive upper bound on a valid shift amount for this variant.
	///
	/// - Half-word (`*32`) variants read only the lower 5 bits, so amounts run `0..32`.
	/// - Full-width variants take amounts `0..64`.
	///
	/// Construction, validation, and deserialization all enforce this same bound.
	/// A value that passes any of them therefore denotes the same shift everywhere.
	#[inline]
	pub const fn max_amount(self) -> usize {
		if self.is_half_word() { 32 } else { 64 }
	}

	/// Resolves this variant to its word-level operation and runs a callback against it.
	///
	/// This is the single place that says what a variant does to a word:
	/// - Logical left and logical right shift zeros in.
	/// - Arithmetic right replicates the sign bit.
	/// - Rotate wraps the bits that fall off one end around to the other.
	/// - The half-word forms apply the same operation to each 32-bit half on its own.
	///
	/// # Arguments
	///
	/// - `amount`: the shift amount in bits, below this variant's upper bound.
	/// - `kernel`: the callback to run against the resolved operation.
	///
	/// # Performance
	///
	/// The variant is decided here, once, ahead of whatever loop the callback runs.
	/// Each branch hands over a distinct zero-sized closure, leaving no branch in the callback.
	#[inline]
	fn dispatch<K: ShiftKernel>(self, amount: u32, kernel: K) -> K::Output {
		match self {
			ShiftVariant::Sll => kernel.call(move |word| word << amount),
			ShiftVariant::Slr => kernel.call(move |word| word >> amount),
			ShiftVariant::Sar => kernel.call(move |word| word.sar(amount)),
			ShiftVariant::Rotr => kernel.call(move |word| word.rotr(amount)),
			ShiftVariant::Sll32 => kernel.call(move |word| word.sll32(amount)),
			ShiftVariant::Srl32 => kernel.call(move |word| word.srl32(amount)),
			ShiftVariant::Sra32 => kernel.call(move |word| word.sra32(amount)),
			ShiftVariant::Rotr32 => kernel.call(move |word| word.rotr32(amount)),
		}
	}

	/// Applies this shift to a 64-bit word and returns the result.
	///
	/// Full-width variants act on the whole 64-bit word.
	/// The half-word variants act on the upper and lower 32-bit halves independently.
	///
	/// # Arguments
	/// - The word to shift.
	/// - The shift amount in bits.
	///
	/// # Performance
	///
	/// Which operation to run is decided on every call.
	/// To shift many words by one fixed variant, resolve the variant once instead.
	#[inline]
	pub fn apply(self, word: Word, amount: usize) -> Word {
		// The word-level operators count the amount in 32 bits.
		self.dispatch(amount as u32, ShiftOneWord { word })
	}

	/// Applies this shift to each source word and writes the result to the matching output cell.
	///
	/// Every output cell is written, so the caller need not initialize them first.
	/// Cells past the end of either slice are left alone.
	///
	/// # Arguments
	///
	/// - `out`: the cells to initialize, one per source word.
	/// - `src`: the words to shift.
	/// - `amount`: the shift amount in bits, below this variant's upper bound.
	#[inline]
	pub fn write_shifted(self, out: &mut [MaybeUninit<Word>], src: &[Word], amount: u32) {
		self.dispatch(amount, WriteShiftedWords { out, src })
	}

	/// Applies this shift to each source word and XORs the result into the matching output cell.
	///
	/// Cells past the end of either slice are left alone.
	///
	/// # Arguments
	///
	/// - `out`: the cells to accumulate into, one per source word.
	/// - `src`: the words to shift.
	/// - `amount`: the shift amount in bits, below this variant's upper bound.
	#[inline]
	pub fn xor_shifted(self, out: &mut [Word], src: &[Word], amount: u32) {
		self.dispatch(amount, XorShiftedWords { out, src })
	}
}

impl SerializeBytes for ShiftVariant {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		(*self as u8).serialize(write_buf)
	}
}

impl DeserializeBytes for ShiftVariant {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let index = u8::deserialize(read_buf)?;
		match index {
			0 => Ok(ShiftVariant::Sll),
			1 => Ok(ShiftVariant::Slr),
			2 => Ok(ShiftVariant::Sar),
			3 => Ok(ShiftVariant::Rotr),
			4 => Ok(ShiftVariant::Sll32),
			5 => Ok(ShiftVariant::Srl32),
			6 => Ok(ShiftVariant::Sra32),
			7 => Ok(ShiftVariant::Rotr32),
			_ => Err(SerializationError::UnknownEnumVariant {
				name: "ShiftVariant",
				index,
			}),
		}
	}
}

/// One shift: an operation paired with the distance it moves by.
///
/// The amount is always below the variant's [`max_amount`](ShiftVariant::max_amount), so a `Shift`
/// that exists denotes the same operation wherever it is read.
///
/// Every variant is the identity at amount 0, so the amount alone does not fix how the identity is
/// spelled. [`Shift::IDENTITY`] is the canonical spelling, and [`Self::is_canonical`] is what says
/// which spelling a constraint system may carry.
///
/// The amount is stored as a byte to keep the struct small: constraint systems hold millions of
/// these.
///
/// ```
/// use binius_core::{constraint_system::Shift, word::Word};
///
/// let word = Word::from_u64(0xf0);
/// assert_eq!(Shift::srl(4).apply(word), Word::from_u64(0x0f));
/// assert_eq!(Shift::IDENTITY.apply(word), word);
///
/// // Every variant is the identity at amount 0, but only one spelling is canonical.
/// assert!(Shift::rotr(0).is_identity());
/// assert!(!Shift::rotr(0).is_canonical());
/// ```
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Shift {
	/// The operation this shift performs.
	pub variant: ShiftVariant,
	/// The number of bits to shift by, below the variant's upper bound.
	pub amount: u8,
}

impl Shift {
	/// The canonical shift that leaves a word untouched.
	pub const IDENTITY: Self = Self {
		variant: ShiftVariant::Sll,
		amount: 0,
	};

	/// A shift of `amount` bits by `variant`.
	///
	/// # Panics
	///
	/// Panics if the amount is not below the variant's [`max_amount`](ShiftVariant::max_amount):
	/// 32 for the half-word (`*32`) variants, 64 for the rest.
	pub fn new(variant: ShiftVariant, amount: usize) -> Self {
		assert!(
			amount < variant.max_amount(),
			"shift amount n={amount} out of range for {variant:?}"
		);
		Self {
			variant,
			// An amount below 64 always fits in the byte-sized field.
			amount: amount as u8,
		}
	}

	/// Shift Left Logical.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn sll(amount: usize) -> Self {
		Self::new(ShiftVariant::Sll, amount)
	}

	/// Shift Right Logical.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn srl(amount: usize) -> Self {
		Self::new(ShiftVariant::Slr, amount)
	}

	/// Shift Right Arithmetic.
	///
	/// This is similar to the Shift Right Logical but instead of shifting in 0 bits it will
	/// replicate the sign bit.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn sar(amount: usize) -> Self {
		Self::new(ShiftVariant::Sar, amount)
	}

	/// Rotate Right.
	///
	/// Rotates bits to the right, with bits shifted off the right end wrapping around to the left.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn rotr(amount: usize) -> Self {
		Self::new(ShiftVariant::Rotr, amount)
	}

	/// Shift Left Logical on 32-bit halves.
	///
	/// Performs independent logical left shifts on the upper and lower 32-bit halves.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn sll32(amount: usize) -> Self {
		Self::new(ShiftVariant::Sll32, amount)
	}

	/// Shift Right Logical on 32-bit halves.
	///
	/// Performs independent logical right shifts on the upper and lower 32-bit halves.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn srl32(amount: usize) -> Self {
		Self::new(ShiftVariant::Srl32, amount)
	}

	/// Shift Right Arithmetic on 32-bit halves.
	///
	/// Performs independent arithmetic right shifts on the upper and lower 32-bit halves,
	/// sign extending each half independently.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn sra32(amount: usize) -> Self {
		Self::new(ShiftVariant::Sra32, amount)
	}

	/// Rotate Right on 32-bit halves.
	///
	/// Performs independent rotate right operations on the upper and lower 32-bit halves.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn rotr32(amount: usize) -> Self {
		Self::new(ShiftVariant::Rotr32, amount)
	}

	/// Whether this shift leaves every word untouched.
	///
	/// Every variant is the identity at amount 0, so this holds for more shifts than
	/// [`Shift::IDENTITY`] alone.
	#[inline]
	pub const fn is_identity(self) -> bool {
		self.amount == 0
	}

	/// Whether this is the canonical spelling of the operation it denotes.
	///
	/// Only the identity has more than one spelling, and [`Shift::IDENTITY`] is the one to use.
	/// Constraint systems carry canonical shifts only, so that two terms denoting the same shifted
	/// word compare equal.
	#[inline]
	pub const fn is_canonical(self) -> bool {
		!self.is_identity() || matches!(self.variant, ShiftVariant::Sll)
	}

	/// Applies this shift to a word and returns the result.
	///
	/// # Performance
	///
	/// Which operation to run is decided on every call. To shift many words by one fixed shift,
	/// resolve the variant once instead — see [`ShiftVariant::write_shifted`] and
	/// [`ShiftVariant::xor_shifted`].
	#[inline]
	pub fn apply(self, word: Word) -> Word {
		self.variant.apply(word, self.amount as usize)
	}
}

impl SerializeBytes for Shift {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.variant.serialize(&mut write_buf)?;
		// Keep the wire format a usize so serialized systems stay byte-compatible.
		(self.amount as usize).serialize(write_buf)
	}
}

impl DeserializeBytes for Shift {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let variant = ShiftVariant::deserialize(&mut read_buf)?;
		let amount = usize::deserialize(read_buf)?;

		// Reject any amount the variant cannot represent.
		// Half-word variants cap at 32, full-width at 64.
		// This mirrors the bound `Shift::new` enforces.
		// An amount below 64 always fits in the byte-sized field.
		if amount >= variant.max_amount() {
			return Err(SerializationError::InvalidConstruction {
				name: "Shift::amount",
			});
		}

		Ok(Shift {
			variant,
			amount: amount as u8,
		})
	}
}

/// Similar to [`ValueIndex`], but represents a value that has been shifted.
///
/// This is used in the operands to constraints like [`AndConstraint`](super::AndConstraint).
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ShiftedValueIndex {
	/// The index of this value in the input values vector.
	pub value_index: ValueIndex,
	/// The shift applied to the value.
	pub shift: Shift,
}

impl ShiftedValueIndex {
	/// Create a value index that just uses the specified value, unshifted.
	pub const fn plain(value_index: ValueIndex) -> Self {
		Self {
			value_index,
			shift: Shift::IDENTITY,
		}
	}

	/// Shift Left Logical by the given number of bits.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn sll(value_index: ValueIndex, amount: usize) -> Self {
		Self {
			value_index,
			shift: Shift::sll(amount),
		}
	}

	/// Shift Right Logical by the given number of bits.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn srl(value_index: ValueIndex, amount: usize) -> Self {
		Self {
			value_index,
			shift: Shift::srl(amount),
		}
	}

	/// Shift Right Arithmetic by the given number of bits.
	///
	/// This is similar to the Shift Right Logical but instead of shifting in 0 bits it will
	/// replicate the sign bit.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn sar(value_index: ValueIndex, amount: usize) -> Self {
		Self {
			value_index,
			shift: Shift::sar(amount),
		}
	}

	/// Rotate Right by the given number of bits.
	///
	/// Rotates bits to the right, with bits shifted off the right end wrapping around to the left.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub fn rotr(value_index: ValueIndex, amount: usize) -> Self {
		Self {
			value_index,
			shift: Shift::rotr(amount),
		}
	}

	/// Shift Left Logical on 32-bit halves by the given number of bits.
	///
	/// Performs independent logical left shifts on the upper and lower 32-bit halves.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn sll32(value_index: ValueIndex, amount: usize) -> Self {
		Self {
			value_index,
			shift: Shift::sll32(amount),
		}
	}

	/// Shift Right Logical on 32-bit halves by the given number of bits.
	///
	/// Performs independent logical right shifts on the upper and lower 32-bit halves.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn srl32(value_index: ValueIndex, amount: usize) -> Self {
		Self {
			value_index,
			shift: Shift::srl32(amount),
		}
	}

	/// Shift Right Arithmetic on 32-bit halves by the given number of bits.
	///
	/// Performs independent arithmetic right shifts on the upper and lower 32-bit halves.
	/// Sign extends each 32-bit half independently. Only uses the lower 5 bits of the shift amount
	/// (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn sra32(value_index: ValueIndex, amount: usize) -> Self {
		Self {
			value_index,
			shift: Shift::sra32(amount),
		}
	}

	/// Rotate Right on 32-bit halves by the given number of bits.
	///
	/// Performs independent rotate right operations on the upper and lower 32-bit halves.
	/// Bits shifted off the right end wrap around to the left within each 32-bit half.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub fn rotr32(value_index: ValueIndex, amount: usize) -> Self {
		Self {
			value_index,
			shift: Shift::rotr32(amount),
		}
	}

	/// Evaluates this term against a witness.
	///
	/// A term names one value and a shift to apply to it.
	/// It contributes one shifted word to the XOR that forms an operand.
	#[inline]
	pub fn eval(&self, witness: &ValueVec) -> Word {
		// Look up the referenced word, then apply this term's shift.
		self.shift.apply(witness[self.value_index])
	}
}

impl SerializeBytes for ShiftedValueIndex {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.value_index.serialize(&mut write_buf)?;
		self.shift.serialize(write_buf)
	}
}

impl DeserializeBytes for ShiftedValueIndex {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let value_index = ValueIndex::deserialize(&mut read_buf)?;
		let shift = Shift::deserialize(read_buf)?;
		Ok(ShiftedValueIndex { value_index, shift })
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;

	// What each variant means, spelled out over raw integers.
	// Independent of the word methods the implementation names, so a mis-wired variant fails here.
	fn reference_shift(variant: ShiftVariant, word: u64, amount: u32) -> u64 {
		// Half-word variants act on each 32-bit half on its own, reading only the low 5 bits.
		let halves = |op: fn(u32, u32) -> u32| {
			let amount = amount & 0x1F;
			op(word as u32, amount) as u64 | ((op((word >> 32) as u32, amount) as u64) << 32)
		};
		match variant {
			ShiftVariant::Sll => word << amount,
			ShiftVariant::Slr => word >> amount,
			ShiftVariant::Sar => (word as i64 >> amount) as u64,
			ShiftVariant::Rotr => word.rotate_right(amount),
			ShiftVariant::Sll32 => halves(|half, n| half << n),
			ShiftVariant::Srl32 => halves(|half, n| half >> n),
			ShiftVariant::Sra32 => halves(|half, n| ((half as i32) >> n) as u32),
			ShiftVariant::Rotr32 => halves(|half, n| half.rotate_right(n)),
		}
	}

	#[test]
	fn all_covers_every_discriminant_in_order() {
		// `ALL` is indexed by discriminant, and every entry decodes back to itself.
		for (discriminant, variant) in ShiftVariant::ALL.into_iter().enumerate() {
			assert_eq!(variant as usize, discriminant);
			assert_eq!(ShiftVariant::from_u8(discriminant as u8), Some(variant));
		}
		// The list is exhaustive: the next discriminant, and any byte above it, decode to nothing.
		assert_eq!(ShiftVariant::from_u8(ShiftVariant::ALL.len() as u8), None);
		assert_eq!(ShiftVariant::from_u8(255), None);
	}

	#[test]
	fn every_variant_is_the_identity_at_amount_zero() {
		// The batched witness builder leans on this: at amount 0 it copies instead of dispatching.
		for variant in ShiftVariant::ALL {
			for word in [
				Word::ZERO,
				Word::ONE,
				Word::ALL_ONE,
				Word(0x0123_4567_89AB_CDEF),
			] {
				assert_eq!(variant.apply(word, 0), word, "{variant:?} is not the identity at 0");
			}
		}
	}

	proptest! {
		// Invariant: each variant resolves to the operation it denotes, at every valid amount.
		#[test]
		fn dispatch_matches_the_reference_for_every_variant(word in any::<u64>()) {
			for variant in ShiftVariant::ALL {
				// Amounts run 0..max, so both extremes are covered.
				for amount in 0..variant.max_amount() {
					let expected = Word(reference_shift(variant, word, amount as u32));
					// Both entry points share one resolution step, so checking both pins it.
					prop_assert_eq!(variant.apply(Word(word), amount), expected);
					let kernel = ShiftOneWord { word: Word(word) };
					prop_assert_eq!(variant.dispatch(amount as u32, kernel), expected);
				}
			}
		}
	}

	#[test]
	fn slice_forms_stop_at_the_shorter_slice() {
		// Fixture state: 3 source words, 2 output cells, shifting left by 1.
		//
		//     src: [1, 2, 4]  ->  [2, 4, 8]
		//     out: [0, 0]         only two cells to fill, so the third result is dropped
		let src = [Word(1), Word(2), Word(4)];
		let mut out = [Word::ZERO; 2];
		ShiftVariant::Sll.xor_shifted(&mut out, &src, 1);
		assert_eq!(out, [Word(2), Word(4)]);

		// Fixture state: 2 source words, 3 output cells preset to 1.
		//
		//     src: [1, 2]  ->  [2, 4]
		//     out: [1, 1, 1]   XOR gives [3, 5, _], and the trailing cell keeps its value
		let mut out = [Word::ONE; 3];
		ShiftVariant::Sll.xor_shifted(&mut out, &src[..2], 1);
		assert_eq!(out, [Word(3), Word(5), Word::ONE]);
	}

	proptest! {
		// Invariant: the slice forms agree with the single-word form, cell by cell.
		#[test]
		fn slice_forms_match_the_single_word_form(src in prop::collection::vec(any::<u64>(), 1..24)) {
			let src: Vec<Word> = src.into_iter().map(Word).collect();
			for variant in ShiftVariant::ALL {
				for amount in 0..variant.max_amount() {
					let expected: Vec<Word> = src.iter().map(|&w| variant.apply(w, amount)).collect();

					// The write form fills cells that start out uninitialized.
					let mut out = vec![MaybeUninit::uninit(); src.len()];
					variant.write_shifted(&mut out, &src, amount as u32);
					// Safety: the call above wrote every cell, since the slices are the same length.
					let written: Vec<Word> =
						out.iter().map(|cell| unsafe { cell.assume_init() }).collect();
					prop_assert_eq!(&written, &expected);

					// Folding into zeroed cells reduces the XOR to the shift itself.
					let mut out = vec![Word::ZERO; src.len()];
					variant.xor_shifted(&mut out, &src, amount as u32);
					prop_assert_eq!(&out, &expected);

					// Folding the same term a second time cancels it, restoring the zeroes.
					variant.xor_shifted(&mut out, &src, amount as u32);
					prop_assert_eq!(out, vec![Word::ZERO; src.len()]);
				}
			}
		}
	}

	#[test]
	fn test_shift_variant_serialization_round_trip() {
		for variant in ShiftVariant::ALL {
			let mut buf = Vec::new();
			variant.serialize(&mut buf).unwrap();

			let deserialized = ShiftVariant::deserialize(&mut buf.as_slice()).unwrap();
			assert_eq!(variant, deserialized);
		}
	}

	#[test]
	fn test_shift_variant_unknown_variant() {
		// Create invalid variant index
		let mut buf = Vec::new();
		255u8.serialize(&mut buf).unwrap();

		let result = ShiftVariant::deserialize(&mut buf.as_slice());
		assert!(result.is_err());
		match result.unwrap_err() {
			SerializationError::UnknownEnumVariant { name, index } => {
				assert_eq!(name, "ShiftVariant");
				assert_eq!(index, 255);
			}
			_ => panic!("Expected UnknownEnumVariant error"),
		}
	}

	#[test]
	fn test_shifted_value_index_serialization_round_trip() {
		let shifted_value_index = ShiftedValueIndex::srl(ValueIndex::private(42), 23);

		let mut buf = Vec::new();
		shifted_value_index.serialize(&mut buf).unwrap();

		let deserialized = ShiftedValueIndex::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(shifted_value_index.value_index, deserialized.value_index);
		assert_eq!(shifted_value_index.shift.amount, deserialized.shift.amount);
		match (shifted_value_index.shift.variant, deserialized.shift.variant) {
			(ShiftVariant::Slr, ShiftVariant::Slr) => {}
			_ => panic!("ShiftVariant mismatch"),
		}
	}

	#[test]
	fn test_shifted_value_index_invalid_amount() {
		// Create a buffer with invalid shift amount (>= 64)
		let mut buf = Vec::new();
		ValueIndex::constant(0).serialize(&mut buf).unwrap();
		ShiftVariant::Sll.serialize(&mut buf).unwrap();
		64usize.serialize(&mut buf).unwrap(); // Invalid amount

		let result = ShiftedValueIndex::deserialize(&mut buf.as_slice());
		assert!(result.is_err());
		match result.unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "Shift::amount");
			}
			_ => panic!("Expected InvalidConstruction error"),
		}
	}

	#[test]
	fn test_max_amount_and_is_half_word() {
		// The four full-width variants come first, then the four half-word ones.
		let (full_width, half_word) = ShiftVariant::ALL.split_at(4);
		// Full-width variants take amounts up to 63.
		for &variant in full_width {
			assert!(!variant.is_half_word());
			assert_eq!(variant.max_amount(), 64);
		}
		// Half-word variants take amounts up to 31.
		for &variant in half_word {
			assert!(variant.is_half_word());
			assert_eq!(variant.max_amount(), 32);
		}
	}

	// Deserializes a raw (variant, amount) buffer, bypassing the constructors.
	// This lets out-of-range half-word amounts reach the deserialization path.
	fn deserialize_amount(
		shift_variant: ShiftVariant,
		amount: usize,
	) -> Result<ShiftedValueIndex, SerializationError> {
		let mut buf = Vec::new();
		ValueIndex::constant(0).serialize(&mut buf).unwrap();
		shift_variant.serialize(&mut buf).unwrap();
		amount.serialize(&mut buf).unwrap();
		ShiftedValueIndex::deserialize(&mut buf.as_slice())
	}

	#[test]
	fn test_deserialize_rejects_half_word_amount_at_or_above_32() {
		// 31 is the largest amount a half-word variant can carry.
		assert_eq!(
			deserialize_amount(ShiftVariant::Sll32, 31).unwrap(),
			ShiftedValueIndex {
				value_index: ValueIndex::constant(0),
				shift: Shift::sll32(31),
			}
		);
		// 32 exceeds the 5-bit range and must be rejected.
		match deserialize_amount(ShiftVariant::Sll32, 32).unwrap_err() {
			SerializationError::InvalidConstruction { name } => {
				assert_eq!(name, "Shift::amount");
			}
			other => panic!("Expected InvalidConstruction, got: {other:?}"),
		}
		// A full-width variant still accepts 32 and up to 63.
		assert_eq!(
			deserialize_amount(ShiftVariant::Sll, 32).unwrap(),
			ShiftedValueIndex {
				value_index: ValueIndex::constant(0),
				shift: Shift::sll(32),
			}
		);
		assert_eq!(
			deserialize_amount(ShiftVariant::Sll, 63).unwrap(),
			ShiftedValueIndex {
				value_index: ValueIndex::constant(0),
				shift: Shift::sll(63),
			}
		);
	}

	#[test]
	fn shifted_value_index_fits_in_a_word() {
		// Layout: value_index (u32, 4 bytes) + Shift (variant byte + amount byte).
		// Padded to the u32 alignment, that is 8 bytes.
		// Holding this at one word matters: systems carry millions of these on the prover hot path.
		assert_eq!(size_of::<ShiftedValueIndex>(), 8);
	}
}
