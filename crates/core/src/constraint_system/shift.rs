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

	/// Whether this variant wraps the bits it moves out, rather than discarding them.
	///
	/// A cyclic variant loses nothing, so any two of its shifts compose however far they move;
	/// every other variant drops what it carries past the end.
	#[inline]
	pub const fn is_cyclic(self) -> bool {
		matches!(self, ShiftVariant::Rotr | ShiftVariant::Rotr32)
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
	pub const fn new(variant: ShiftVariant, amount: usize) -> Self {
		// A const context cannot format, so the amount and variant are left to the panic location
		// rather than spelled into the message.
		assert!(amount < variant.max_amount(), "shift amount out of range for this variant");
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
	pub const fn sll(amount: usize) -> Self {
		Self::new(ShiftVariant::Sll, amount)
	}

	/// Shift Right Logical.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub const fn srl(amount: usize) -> Self {
		Self::new(ShiftVariant::Slr, amount)
	}

	/// Shift Right Arithmetic.
	///
	/// This is similar to the Shift Right Logical but instead of shifting in 0 bits it will
	/// replicate the sign bit.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub const fn sar(amount: usize) -> Self {
		Self::new(ShiftVariant::Sar, amount)
	}

	/// Rotate Right.
	///
	/// Rotates bits to the right, with bits shifted off the right end wrapping around to the left.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub const fn rotr(amount: usize) -> Self {
		Self::new(ShiftVariant::Rotr, amount)
	}

	/// Shift Left Logical on 32-bit halves.
	///
	/// Performs independent logical left shifts on the upper and lower 32-bit halves.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub const fn sll32(amount: usize) -> Self {
		Self::new(ShiftVariant::Sll32, amount)
	}

	/// Shift Right Logical on 32-bit halves.
	///
	/// Performs independent logical right shifts on the upper and lower 32-bit halves.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub const fn srl32(amount: usize) -> Self {
		Self::new(ShiftVariant::Srl32, amount)
	}

	/// Shift Right Arithmetic on 32-bit halves.
	///
	/// Performs independent arithmetic right shifts on the upper and lower 32-bit halves,
	/// sign extending each half independently.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub const fn sra32(amount: usize) -> Self {
		Self::new(ShiftVariant::Sra32, amount)
	}

	/// Rotate Right on 32-bit halves.
	///
	/// Performs independent rotate right operations on the upper and lower 32-bit halves.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub const fn rotr32(amount: usize) -> Self {
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

	/// Classifies the composition of two shifts, `outer` applied to the result of `inner`.
	///
	/// This is the merge rule for a shift sequence: it says whether the two collapse to one shift,
	/// clear the word, or genuinely need both slots.
	///
	/// Collapsing is not just a matter of adding amounts. Two shifts collapse when the second
	/// continues the first — which happens for more pairs than sharing a variant, since a shift
	/// that has already cleared the sign bit or carried every bit past the halfway point leaves
	/// the next shift nothing to distinguish. `chained` enumerates those, and `degenerate` the
	/// cases where one shift has flattened the word past the other's notice.
	///
	/// Reporting [`Composition::Pair`] where a collapse exists would cost a shift slot but never a
	/// wrong answer; the tests check against an independent bit-level model so that does not
	/// happen silently.
	///
	/// # Arguments
	///
	/// - `inner`: the shift applied first.
	/// - `outer`: the shift applied to its result.
	pub fn compose(inner: Shift, outer: Shift) -> Composition {
		// The identity leaves the other shift to stand alone, whichever side it is on.
		if inner.is_identity() {
			return Composition::Single(outer);
		}
		if outer.is_identity() {
			return Composition::Single(inner);
		}

		if let Some(single) = degenerate(inner, outer) {
			return Composition::Single(single);
		}
		match chained(inner, outer) {
			Some((variant, distance)) => chained_composition(variant, distance),
			None => Composition::Pair,
		}
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

/// What the composition of two shifts denotes.
///
/// Composing two shifts does not always need two: they may collapse to one shift, or clear the
/// word outright. A caller merging shifts has to tell those apart — the collapsed cases save a
/// shift slot, and the cleared case means the term contributes nothing and should be dropped
/// rather than encoded.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum Composition {
	/// The two collapse to this single shift, which may be [`Shift::IDENTITY`].
	Single(Shift),
	/// The two clear every bit, so the term they apply to is zero.
	Zero,
	/// The two do not collapse; both are needed, in the order they were given.
	Pair,
}

/// Two shifts that chain, as the one direction they move bits in and the distance they cover
/// together.
///
/// Chaining is what makes a composition collapse: the second shift continues the first rather than
/// undoing part of it, so only the total distance matters and the variant's own overflow rule
/// decides what comes out.
///
/// Returns `None` when the two do not chain, which is the common case.
fn chained(inner: Shift, outer: Shift) -> Option<(ShiftVariant, usize)> {
	let distance = inner.amount as usize + outer.amount as usize;
	// Two shifts of one variant always chain.
	if inner.variant == outer.variant {
		return Some((inner.variant, distance));
	}

	// Neither shift is the identity here, so every amount below is at least 1.
	Some((
		match (inner.variant, outer.variant) {
			// A logical right shift clears the sign bit, so an arithmetic one behind it has no
			// sign left to replicate and moves zeros in just the same.
			(ShiftVariant::Slr, ShiftVariant::Sar) => ShiftVariant::Slr,
			(ShiftVariant::Srl32, ShiftVariant::Sra32) => ShiftVariant::Srl32,

			// Once a full-width shift has carried everything past the halfway point, each 32-bit
			// half holds only bits from one side of the word, so a half-wise shift in the same
			// direction continues it as if it were full-width.
			(ShiftVariant::Sll, ShiftVariant::Sll32) if inner.amount >= 32 => ShiftVariant::Sll,
			(ShiftVariant::Sll32, ShiftVariant::Sll) if outer.amount >= 32 => ShiftVariant::Sll,
			(ShiftVariant::Slr, ShiftVariant::Srl32) if inner.amount >= 32 => ShiftVariant::Slr,
			(ShiftVariant::Srl32, ShiftVariant::Slr) if outer.amount >= 32 => ShiftVariant::Slr,
			(ShiftVariant::Sar, ShiftVariant::Sra32) if inner.amount >= 32 => ShiftVariant::Sar,
			(ShiftVariant::Sra32, ShiftVariant::Sar) if outer.amount >= 32 => ShiftVariant::Sar,

			// The same, where the half-wise shift is the arithmetic one: it needs both halves'
			// sign bits already cleared, which costs one more bit of travel than the cases above.
			(ShiftVariant::Slr, ShiftVariant::Sra32) if inner.amount >= 33 => ShiftVariant::Slr,
			(ShiftVariant::Srl32, ShiftVariant::Sar) if outer.amount >= 32 => ShiftVariant::Slr,

			_ => return None,
		},
		distance,
	))
}

/// The single shift a composition collapses to for reasons other than chaining.
///
/// These are the degenerate cases, where one shift has already flattened the word enough that the
/// other cannot tell the difference.
const fn degenerate(inner: Shift, outer: Shift) -> Option<Shift> {
	match (inner.variant, outer.variant) {
		// Shifted arithmetically all the way, a word is all zeros or all ones. Rotating a word of
		// one repeated bit leaves it alone.
		(ShiftVariant::Sar, ShiftVariant::Rotr | ShiftVariant::Rotr32) if inner.amount == 63 => {
			Some(inner)
		}
		(ShiftVariant::Sra32, ShiftVariant::Rotr32) if inner.amount == 31 => Some(inner),

		// Keeping only the top bit keeps the very bit an arithmetic right shift replicates, so
		// whatever that shift did before it does not show.
		(ShiftVariant::Sar | ShiftVariant::Sra32, ShiftVariant::Slr) if outer.amount == 63 => {
			Some(outer)
		}
		(ShiftVariant::Sra32, ShiftVariant::Srl32) if outer.amount == 31 => Some(outer),

		_ => None,
	}
}

/// What a chained shift of the given total distance comes to, once its own overflow rule applies.
///
/// This is where the variants differ: a logical shift runs out of word and clears it, an
/// arithmetic one saturates at the sign, and a rotation wraps.
fn chained_composition(variant: ShiftVariant, distance: usize) -> Composition {
	let width = variant.max_amount();
	match variant {
		// Bits carried past the end are gone; carry everything past it and nothing is left.
		ShiftVariant::Sll | ShiftVariant::Sll32 | ShiftVariant::Slr | ShiftVariant::Srl32 => {
			if distance < width {
				Composition::Single(Shift::new(variant, distance))
			} else {
				Composition::Zero
			}
		}
		// Every position past the shift reads the sign bit, so travel beyond the width adds
		// nothing.
		ShiftVariant::Sar | ShiftVariant::Sra32 => {
			Composition::Single(Shift::new(variant, distance.min(width - 1)))
		}
		// A rotation loses nothing, so a full turn is the identity.
		ShiftVariant::Rotr | ShiftVariant::Rotr32 => match distance % width {
			0 => Composition::Single(Shift::IDENTITY),
			distance => Composition::Single(Shift::new(variant, distance)),
		},
	}
}

/// Similar to [`ValueIndex`], but represents a value that has been shifted.
///
/// This is used in the operands to constraints like [`AndConstraint`](super::AndConstraint).
///
/// A term carries a *sequence* of two shifts rather than one. The inner shift, `shift_seq[0]`,
/// applies to the word first; the outer shift, `shift_seq[1]`, applies to its result. Two shifts
/// express maps no single shift can: clearing the low bits and returning the rest to where they
/// started needs both, since no one shift both drops bits and leaves the others in place.
///
/// # Canonical form
///
/// A lone shift goes in the inner slot, so `shift_seq[0].is_identity()` implies
/// `shift_seq[1].is_identity()`. That splits every term into three classes:
///
/// ```text
/// unshifted         s_1 = s_2 = 0     spelled Shift::IDENTITY twice
/// singly shifted    s_2 = 0 != s_1    the lone shift sits inner
/// doubly shifted    s_2 != 0          both slots carry work
/// ```
///
/// A doubly shifted term must not collapse: [`Shift::compose`] of the two reports
/// [`Composition::Pair`], never [`Composition::Single`] (the two merge into one shift) or
/// [`Composition::Zero`] (the two clear every bit, so the term is identically zero).
/// [`ConstraintSystem::validate`](super::ConstraintSystem::validate) enforces both rules.
///
/// The `[Shift; 2]` spelling is not itself canonical as a *map*: per the composition derivation,
/// 108,571 irreducible spellings denote only 74,341 distinct maps. Nothing in the reduction depends
/// on that normalization for correctness, and buying it back needs a table far larger than the few
/// dozen shifts a real constraint system uses.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ShiftedValueIndex {
	/// The index of this value in the input values vector.
	pub value_index: ValueIndex,
	/// The two shifts applied to the value, inner first.
	pub shift_seq: [Shift; 2],
}

impl ShiftedValueIndex {
	/// A value shifted by a sequence of two shifts, `shift_seq[0]` first.
	pub const fn new(value_index: ValueIndex, shift_seq: [Shift; 2]) -> Self {
		Self {
			value_index,
			shift_seq,
		}
	}

	/// A value shifted by one shift, which the canonical form places in the inner slot.
	pub const fn single(value_index: ValueIndex, shift: Shift) -> Self {
		Self::new(value_index, [shift, Shift::IDENTITY])
	}

	/// The shift applied first.
	#[inline]
	pub const fn inner(&self) -> Shift {
		self.shift_seq[0]
	}

	/// The shift applied to the inner shift's result.
	#[inline]
	pub const fn outer(&self) -> Shift {
		self.shift_seq[1]
	}

	/// Whether this term leaves its word untouched.
	///
	/// The canonical form puts a lone shift inner, so an identity inner shift settles it.
	#[inline]
	pub const fn is_unshifted(&self) -> bool {
		self.inner().is_identity()
	}

	/// Whether this term genuinely needs both shift slots.
	#[inline]
	pub const fn is_doubly_shifted(&self) -> bool {
		!self.outer().is_identity()
	}

	/// Create a value index that just uses the specified value, unshifted.
	pub const fn plain(value_index: ValueIndex) -> Self {
		Self::single(value_index, Shift::IDENTITY)
	}

	/// Shift Left Logical by the given number of bits.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub const fn sll(value_index: ValueIndex, amount: usize) -> Self {
		Self::single(value_index, Shift::sll(amount))
	}

	/// Shift Right Logical by the given number of bits.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub const fn srl(value_index: ValueIndex, amount: usize) -> Self {
		Self::single(value_index, Shift::srl(amount))
	}

	/// Shift Right Arithmetic by the given number of bits.
	///
	/// This is similar to the Shift Right Logical but instead of shifting in 0 bits it will
	/// replicate the sign bit.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub const fn sar(value_index: ValueIndex, amount: usize) -> Self {
		Self::single(value_index, Shift::sar(amount))
	}

	/// Rotate Right by the given number of bits.
	///
	/// Rotates bits to the right, with bits shifted off the right end wrapping around to the left.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 64.
	pub const fn rotr(value_index: ValueIndex, amount: usize) -> Self {
		Self::single(value_index, Shift::rotr(amount))
	}

	/// Shift Left Logical on 32-bit halves by the given number of bits.
	///
	/// Performs independent logical left shifts on the upper and lower 32-bit halves.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub const fn sll32(value_index: ValueIndex, amount: usize) -> Self {
		Self::single(value_index, Shift::sll32(amount))
	}

	/// Shift Right Logical on 32-bit halves by the given number of bits.
	///
	/// Performs independent logical right shifts on the upper and lower 32-bit halves.
	/// Only uses the lower 5 bits of the shift amount (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub const fn srl32(value_index: ValueIndex, amount: usize) -> Self {
		Self::single(value_index, Shift::srl32(amount))
	}

	/// Shift Right Arithmetic on 32-bit halves by the given number of bits.
	///
	/// Performs independent arithmetic right shifts on the upper and lower 32-bit halves.
	/// Sign extends each 32-bit half independently. Only uses the lower 5 bits of the shift amount
	/// (0-31).
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub const fn sra32(value_index: ValueIndex, amount: usize) -> Self {
		Self::single(value_index, Shift::sra32(amount))
	}

	/// Rotate Right on 32-bit halves by the given number of bits.
	///
	/// Performs independent rotate right operations on the upper and lower 32-bit halves.
	/// Bits shifted off the right end wrap around to the left within each 32-bit half.
	///
	/// # Panics
	/// Panics if the shift amount is greater than or equal to 32.
	pub const fn rotr32(value_index: ValueIndex, amount: usize) -> Self {
		Self::single(value_index, Shift::rotr32(amount))
	}

	/// Evaluates this term against a witness.
	///
	/// A term names one value and a sequence of two shifts to apply to it.
	/// It contributes one shifted word to the XOR that forms an operand.
	#[inline]
	pub fn eval(&self, witness: &ValueVec) -> Word {
		// Look up the referenced word, then apply the two shifts in sequence, inner first.
		let [inner, outer] = self.shift_seq;
		outer.apply(inner.apply(witness[self.value_index]))
	}
}

impl SerializeBytes for ShiftedValueIndex {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.value_index.serialize(&mut write_buf)?;
		for shift in &self.shift_seq {
			shift.serialize(&mut write_buf)?;
		}
		Ok(())
	}
}

impl DeserializeBytes for ShiftedValueIndex {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let value_index = ValueIndex::deserialize(&mut read_buf)?;
		let inner = Shift::deserialize(&mut read_buf)?;
		let outer = Shift::deserialize(read_buf)?;
		Ok(ShiftedValueIndex::new(value_index, [inner, outer]))
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

	/// The width the half-word (`*32`) variants act over.
	const HALF_WORD_BITS: usize = 32;

	/// The bit an output position reads, for the positions that read nothing.
	const READS_ZERO: u8 = u8::MAX;

	/// What a shift does to a word, as one input bit position per output bit position.
	///
	/// This is an independent model of a shift's meaning, written from the definitions rather than
	/// from the composition rules it is used to check. Entry `i` is the input bit that output bit
	/// `i` reads, or [`READS_ZERO`].
	fn bit_map(shift: Shift) -> [u8; Word::BITS] {
		let mut map = [READS_ZERO; Word::BITS];
		let amount = shift.amount as usize;
		let width = if shift.variant.is_half_word() {
			HALF_WORD_BITS
		} else {
			Word::BITS
		};
		for (half, positions) in map.chunks_exact_mut(width).enumerate() {
			let base = (half * width) as u8;
			for (out, slot) in positions.iter_mut().enumerate() {
				let read = match shift.variant {
					ShiftVariant::Sll | ShiftVariant::Sll32 => out.checked_sub(amount),
					ShiftVariant::Slr | ShiftVariant::Srl32 => {
						Some(out + amount).filter(|&read| read < width)
					}
					ShiftVariant::Sar | ShiftVariant::Sra32 => Some((out + amount).min(width - 1)),
					ShiftVariant::Rotr | ShiftVariant::Rotr32 => Some((out + amount) % width),
				};
				if let Some(read) = read {
					*slot = base + read as u8;
				}
			}
		}
		map
	}

	/// What composing two shifts denotes, worked out at the bit level.
	///
	/// Composition is function composition of the maps: output bit `i` reads whatever the inner
	/// shift put where the outer shift looks. The result is then matched against the alphabet.
	fn reference_composition(
		inner: Shift,
		outer: Shift,
		by_map: &std::collections::HashMap<[u8; Word::BITS], Shift>,
	) -> Composition {
		let (inner_map, outer_map) = (bit_map(inner), bit_map(outer));
		let mut composed = [READS_ZERO; Word::BITS];
		for (slot, &read) in iter::zip(&mut composed, &outer_map) {
			if read != READS_ZERO {
				*slot = inner_map[read as usize];
			}
		}

		if composed == [READS_ZERO; Word::BITS] {
			return Composition::Zero;
		}
		match by_map.get(&composed) {
			Some(&single) => Composition::Single(single),
			None => Composition::Pair,
		}
	}

	/// Every canonical shift, in the order the alphabet is enumerated.
	fn canonical_shifts() -> Vec<Shift> {
		ShiftVariant::ALL
			.into_iter()
			.flat_map(|variant| {
				(0..variant.max_amount()).map(move |amount| Shift::new(variant, amount))
			})
			.filter(|shift| shift.is_canonical())
			.collect()
	}

	#[test]
	fn bit_map_matches_applying_the_shift() {
		// The bit map is the oracle the composition rules are checked against, so it is pinned to
		// the word operations first.
		//
		// One-hot words pin every entry: setting only input bit `b` makes the output exactly the
		// positions whose map entry is `b`, so a map that reads the wrong bit shows up here.
		for shift in canonical_shifts() {
			let map = bit_map(shift);
			for bit in 0..Word::BITS {
				let expected = shift.apply(Word(1u64 << bit)).as_u64();
				let from_map = map
					.iter()
					.enumerate()
					.filter(|&(_, &read)| read as usize == bit)
					.map(|(out, _)| 1u64 << out)
					.fold(0u64, |acc, position| acc | position);
				assert_eq!(from_map, expected, "{shift:?} disagrees on input bit {bit}");
			}
		}
	}

	#[test]
	fn the_canonical_alphabet_has_no_two_spellings_of_one_shift() {
		// The alphabet is 4 full-width variants at 63 non-zero amounts, 4 half-word ones at 31,
		// and the identity: 4 * 63 + 4 * 31 + 1.
		let shifts = canonical_shifts();
		assert_eq!(shifts.len(), 4 * 63 + 4 * 31 + 1);

		// Every one of them denotes a distinct operation, which is what makes `Single` name one
		// shift unambiguously.
		let maps = shifts
			.iter()
			.map(|&shift| bit_map(shift))
			.collect::<std::collections::HashSet<_>>();
		assert_eq!(maps.len(), shifts.len());
	}

	#[test]
	fn compose_matches_the_bit_level_model() {
		// The casework in `compose` is a closed form for something the bits already determine.
		// This checks the two agree on every ordered pair of the alphabet — so a missing rule, or
		// one whose guard is off by a bit of travel, fails here rather than costing a shift slot
		// silently.
		let shifts = canonical_shifts();
		let by_map = shifts
			.iter()
			.map(|&shift| (bit_map(shift), shift))
			.collect::<std::collections::HashMap<_, _>>();
		for &inner in &shifts {
			for &outer in &shifts {
				assert_eq!(
					Shift::compose(inner, outer),
					reference_composition(inner, outer, &by_map),
					"composing {inner:?} then {outer:?}"
				);
			}
		}
	}

	#[test]
	fn compose_classifies_the_whole_alphabet_the_way_the_derivation_does() {
		// The split from the BINIUS-408 design pass, as a regression: of the 377^2 ordered pairs,
		// 23,046 collapse to one shift, 10,512 clear the word, and 108,571 need both slots. The
		// counts move only if the shift alphabet itself changes.
		let shifts = canonical_shifts();
		let mut single = 0;
		let mut zero = 0;
		let mut pair = 0;
		for &inner in &shifts {
			for &outer in &shifts {
				match Shift::compose(inner, outer) {
					Composition::Single(_) => single += 1,
					Composition::Zero => zero += 1,
					Composition::Pair => pair += 1,
				}
			}
		}
		assert_eq!(single + zero + pair, shifts.len() * shifts.len());
		assert_eq!((single, zero, pair), (23_046, 10_512, 108_571));
	}

	#[test]
	fn compose_catches_the_collapses_amount_arithmetic_misses() {
		// Saturating past the width is still a shift: every position already reads the sign bit.
		assert_eq!(
			Shift::compose(Shift::sar(5), Shift::sar(60)),
			Composition::Single(Shift::sar(63))
		);
		// Shifting the whole word out clears it, rather than shifting by the sum of the amounts.
		assert_eq!(Shift::compose(Shift::sll(40), Shift::sll(30)), Composition::Zero);
		// Rotations wrap, so a full turn is the identity.
		assert_eq!(
			Shift::compose(Shift::rotr(7), Shift::rotr(57)),
			Composition::Single(Shift::IDENTITY)
		);
		// The identity composes with anything, leaving the other shift alone.
		for shift in [Shift::rotr(9), Shift::sar(3), Shift::sll32(4)] {
			assert_eq!(Shift::compose(Shift::IDENTITY, shift), Composition::Single(shift));
			assert_eq!(Shift::compose(shift, Shift::IDENTITY), Composition::Single(shift));
		}
		// Clearing the low bits needs both slots: no single shift both drops bits and returns the
		// rest to where they started.
		assert_eq!(Shift::compose(Shift::srl(3), Shift::sll(3)), Composition::Pair);
	}

	#[test]
	fn compose_collapses_a_half_word_shift_into_a_full_width_one() {
		// Once a full-width shift has carried everything past the halfway point, a half-wise shift
		// in the same direction continues it — the case plain amount arithmetic over variants
		// misses, and the one an inlined `*32` gadget is most likely to produce.
		assert_eq!(
			Shift::compose(Shift::sll(32), Shift::sll32(1)),
			Composition::Single(Shift::sll(33))
		);
		assert_eq!(
			Shift::compose(Shift::sll32(1), Shift::sll(32)),
			Composition::Single(Shift::sll(33))
		);
		// The arithmetic half-wise shift needs one more bit of travel, since it wants both halves'
		// sign bits already clear.
		assert_eq!(
			Shift::compose(Shift::srl(33), Shift::sra32(1)),
			Composition::Single(Shift::srl(34))
		);
		assert_eq!(Shift::compose(Shift::srl(32), Shift::sra32(1)), Composition::Pair);
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
		assert_eq!(shifted_value_index.shift_seq, deserialized.shift_seq);
		match (deserialized.inner().variant, deserialized.outer().variant) {
			(ShiftVariant::Slr, ShiftVariant::Sll) => {}
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

	// Deserializes a raw (variant, amount) inner shift, bypassing the constructors.
	// This lets out-of-range half-word amounts reach the deserialization path.
	// The outer slot carries the identity, as the canonical form of a lone shift requires.
	fn deserialize_amount(
		shift_variant: ShiftVariant,
		amount: usize,
	) -> Result<ShiftedValueIndex, SerializationError> {
		let mut buf = Vec::new();
		ValueIndex::constant(0).serialize(&mut buf).unwrap();
		shift_variant.serialize(&mut buf).unwrap();
		amount.serialize(&mut buf).unwrap();
		Shift::IDENTITY.serialize(&mut buf).unwrap();
		ShiftedValueIndex::deserialize(&mut buf.as_slice())
	}

	#[test]
	fn test_deserialize_rejects_half_word_amount_at_or_above_32() {
		// 31 is the largest amount a half-word variant can carry.
		assert_eq!(
			deserialize_amount(ShiftVariant::Sll32, 31).unwrap(),
			ShiftedValueIndex::sll32(ValueIndex::constant(0), 31)
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
			ShiftedValueIndex::sll(ValueIndex::constant(0), 32)
		);
		assert_eq!(
			deserialize_amount(ShiftVariant::Sll, 63).unwrap(),
			ShiftedValueIndex::sll(ValueIndex::constant(0), 63)
		);
	}

	#[test]
	fn a_term_serialization_round_trips_both_shift_slots() {
		// The outer slot is on the wire too, so a doubly shifted term survives the round trip.
		// Clearing the low bits and returning the rest is the canonical example of a genuine pair.
		let term = ShiftedValueIndex::new(ValueIndex::private(7), [Shift::srl(3), Shift::sll(3)]);
		assert_eq!(Shift::compose(term.inner(), term.outer()), Composition::Pair);

		let mut buf = Vec::new();
		term.serialize(&mut buf).unwrap();
		assert_eq!(ShiftedValueIndex::deserialize(buf.as_slice()).unwrap(), term);
	}

	#[test]
	fn a_term_applies_its_two_shifts_inner_first() {
		// Order matters: `srl(4)` then `sll(4)` clears the word's low nibble, while the reverse
		// order clears its top one. A term that applied the outer shift first would swap the two.
		// The fixture sets bits in both nibbles so each pair drops something.
		let values = ValueVec::new_from_data(0, &[], &[Word::from_u64(0xf000_0000_0000_abcd)]);

		let clear_low =
			ShiftedValueIndex::new(ValueIndex::private(0), [Shift::srl(4), Shift::sll(4)]);
		assert_eq!(clear_low.eval(&values), Word::from_u64(0xf000_0000_0000_abc0));

		let clear_top =
			ShiftedValueIndex::new(ValueIndex::private(0), [Shift::sll(4), Shift::srl(4)]);
		assert_eq!(clear_top.eval(&values), Word::from_u64(0x0000_0000_0000_abcd));

		// A lone shift sits inner, and the identity outer leaves its result alone.
		assert_eq!(
			ShiftedValueIndex::srl(ValueIndex::private(0), 4).eval(&values),
			Word::from_u64(0x0f00_0000_0000_0abc)
		);
	}

	#[test]
	fn the_term_classes_read_off_the_shift_sequence() {
		let index = ValueIndex::private(0);

		// Unshifted: the canonical form spells the identity in both slots.
		let unshifted = ShiftedValueIndex::plain(index);
		assert!(unshifted.is_unshifted());
		assert!(!unshifted.is_doubly_shifted());

		// Singly shifted: the lone shift sits inner, so the term is not unshifted.
		let singly = ShiftedValueIndex::rotr(index, 5);
		assert!(!singly.is_unshifted());
		assert!(!singly.is_doubly_shifted());

		// Doubly shifted: the outer slot carries work of its own.
		let doubly = ShiftedValueIndex::new(index, [Shift::srl(3), Shift::sll(3)]);
		assert!(!doubly.is_unshifted());
		assert!(doubly.is_doubly_shifted());
	}

	#[test]
	fn shifted_value_index_fits_in_a_word() {
		// Layout: value_index (u32, 4 bytes) + two Shifts (variant byte + amount byte each).
		// That fills the u32 alignment exactly: 4 + 2 * 2 = 8 bytes.
		// Holding this at one word matters: systems carry millions of these on the prover hot path.
		assert_eq!(size_of::<ShiftedValueIndex>(), 8);
	}
}
