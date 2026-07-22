// Copyright 2025 Irreducible Inc.

//! The shift algebra shared by operands: a [`Shift`] applied to a [`Wire`].

use std::ops::Deref;

use binius_core::{
	constraint_system::{ShiftVariant, ShiftedValueIndex, ValueIndex},
	word::Word,
};
use cranelift_entity::{EntitySet, SecondaryMap};

use crate::compiler::Wire;

/// A single wire term of an operand, tagged with the shift to apply to it.
#[derive(Copy, Clone, Debug)]
pub struct ShiftedWire {
	/// The wire the shift applies to.
	pub wire: Wire,
	/// The shift folded into this term.
	pub shift: Shift,
}

impl ShiftedWire {
	/// Lowers this term to a core [`ShiftedValueIndex`] via the wire mapping.
	///
	/// A zero-amount shift is the identity, so it collapses to a plain value index.
	pub(super) fn to_shifted_value_index(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> ShiftedValueIndex {
		let idx = wire_mapping[self.wire];
		match self.shift {
			Shift::None => ShiftedValueIndex::plain(idx),
			Shift::Sll(0)
			| Shift::Sll32(0)
			| Shift::Srl(0)
			| Shift::Srl32(0)
			| Shift::Sar(0)
			| Shift::Sra32(0)
			| Shift::Rotr(0)
			| Shift::Rotr32(0) => ShiftedValueIndex::plain(idx),
			Shift::Sll(n) => ShiftedValueIndex::sll(idx, n as usize),
			Shift::Sll32(n) => ShiftedValueIndex::sll32(idx, n as usize),
			Shift::Srl(n) => ShiftedValueIndex::srl(idx, n as usize),
			Shift::Srl32(n) => ShiftedValueIndex::srl32(idx, n as usize),
			Shift::Sar(n) => ShiftedValueIndex::sar(idx, n as usize),
			Shift::Sra32(n) => ShiftedValueIndex::sra32(idx, n as usize),
			Shift::Rotr(n) => ShiftedValueIndex::rotr(idx, n as usize),
			Shift::Rotr32(n) => ShiftedValueIndex::rotr32(idx, n as usize),
		}
	}
}

/// An operand: an XOR of shifted-wire terms, stored per constraint position.
#[derive(Clone, Debug, Default)]
pub struct WireOperand(Vec<ShiftedWire>);

impl WireOperand {
	/// Creates an empty operand.
	pub const fn new() -> Self {
		Self(Vec::new())
	}

	/// Appends a shifted-wire term.
	pub fn push(&mut self, term: ShiftedWire) {
		self.0.push(term);
	}

	/// Mutable iterator over this operand's terms.
	///
	/// Lets the compiler retarget a term that reads a shift of the all-ones word.
	/// The term is pointed at the all-ones word itself, carrying an adjusted shift.
	pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, ShiftedWire> {
		self.0.iter_mut()
	}

	/// Lowers the whole operand to core `ShiftedValueIndex` terms.
	pub(super) fn into_value_indices(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> Vec<ShiftedValueIndex> {
		self.0
			.into_iter()
			.map(|term| term.to_shifted_value_index(wire_mapping))
			.collect()
	}

	/// Inserts every wire this operand references into `used_set`.
	pub(super) fn mark_used(&self, used_set: &mut EntitySet<Wire>) {
		for term in &self.0 {
			used_set.insert(term.wire);
		}
	}
}

impl Deref for WireOperand {
	type Target = [ShiftedWire];

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl<'a> IntoIterator for &'a WireOperand {
	type Item = &'a ShiftedWire;
	type IntoIter = std::slice::Iter<'a, ShiftedWire>;

	fn into_iter(self) -> Self::IntoIter {
		self.0.iter()
	}
}

impl FromIterator<ShiftedWire> for WireOperand {
	fn from_iter<I: IntoIterator<Item = ShiftedWire>>(iter: I) -> Self {
		Self(iter.into_iter().collect())
	}
}

impl From<Vec<ShiftedWire>> for WireOperand {
	fn from(terms: Vec<ShiftedWire>) -> Self {
		Self(terms)
	}
}

/// A shift folded into an operand term.
///
/// The `*32` variants act half-wise on the two 32-bit lanes of a 64-bit word;
/// the others act on the whole word. The amount is the shift distance in bits.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum Shift {
	/// No shift; the wire is used as-is.
	None,
	/// Logical left shift of the whole word.
	Sll(u32),
	/// Half-wise logical left shift of each 32-bit lane.
	Sll32(u32),
	/// Logical right shift of the whole word.
	Srl(u32),
	/// Half-wise logical right shift of each 32-bit lane.
	Srl32(u32),
	/// Arithmetic right shift of the whole word.
	Sar(u32),
	/// Half-wise arithmetic right shift of each 32-bit lane.
	Sra32(u32),
	/// Rotate-right of the whole word.
	Rotr(u32),
	/// Half-wise rotate-right of each 32-bit lane.
	Rotr32(u32),
}

impl Shift {
	/// Decomposes a word into a single shift of the all-ones word, if one exists.
	///
	/// The all-ones word is `0xFFFF_FFFF_FFFF_FFFF`.
	/// Shifting it yields exactly the words whose set bits form one run anchored at an end.
	///
	/// - A low run `0b0...01...1` is the all-ones word shifted right by `k`, with `k` zeros above.
	/// - A high run `0b1...10...0` is the all-ones word shifted left by `k`, with `k` zeros below.
	///
	/// Returns nothing in three cases:
	/// - the zero word.
	/// - the all-ones word itself, since a zero-amount shift is the identity, not a useful alias.
	/// - any word whose set bits are not anchored at an end.
	pub const fn of_all_one(word: Word) -> Option<Self> {
		let v = word.0;
		// The zero word and the all-ones word are not proper (nonzero-amount) shifts of ALL_ONE.
		if v == 0 || v == u64::MAX {
			return None;
		}
		// Low run: v = 2^n - 1  <=>  v & (v + 1) == 0, with n = popcount(v) ones.
		// Then v = ALL_ONE >> (64 - n): the top 64 - n bits are shifted out.
		if v & (v + 1) == 0 {
			let n = v.count_ones();
			return Some(Shift::Srl(64 - n));
		}
		// High run: !v is a low run, so v = ALL_ONE << trailing_zeros(v).
		let complement = !v;
		if complement & (complement + 1) == 0 {
			return Some(Shift::Sll(v.trailing_zeros()));
		}
		None
	}

	/// The shift kind and bit amount this shift denotes, or nothing for the identity.
	///
	/// A zero amount is the identity on every variant, so it maps to nothing.
	/// The amount is always below 64, so it fits in a byte.
	pub const fn as_variant_amount(self) -> Option<(ShiftVariant, u8)> {
		match self {
			Shift::None
			| Shift::Sll(0)
			| Shift::Sll32(0)
			| Shift::Srl(0)
			| Shift::Srl32(0)
			| Shift::Sar(0)
			| Shift::Sra32(0)
			| Shift::Rotr(0)
			| Shift::Rotr32(0) => None,
			Shift::Sll(n) => Some((ShiftVariant::Sll, n as u8)),
			Shift::Sll32(n) => Some((ShiftVariant::Sll32, n as u8)),
			Shift::Srl(n) => Some((ShiftVariant::Slr, n as u8)),
			Shift::Srl32(n) => Some((ShiftVariant::Srl32, n as u8)),
			Shift::Sar(n) => Some((ShiftVariant::Sar, n as u8)),
			Shift::Sra32(n) => Some((ShiftVariant::Sra32, n as u8)),
			Shift::Rotr(n) => Some((ShiftVariant::Rotr, n as u8)),
			Shift::Rotr32(n) => Some((ShiftVariant::Rotr32, n as u8)),
		}
	}

	/// Folds `rhs` applied after `lhs` into a single equivalent shift.
	///
	/// Returns `None` when the two shifts cannot be one shift:
	/// - `None` is the identity, so it composes with anything.
	/// - Like-kind shifts add their amounts.
	/// - Rotations are cyclic, so the sum wraps modulo the width and always composes.
	/// - A shift (not a rotation) whose amounts sum past the width shifts everything out to zero,
	///   which is not expressible as one shift, so it returns `None`.
	/// - Different kinds never compose.
	pub const fn compose(lhs: Shift, rhs: Shift) -> Option<Self> {
		// A shift whose amounts sum to `>= width` clears the word, which is not
		// expressible as a single shift, so those arms return `None`.
		// Rotations are cyclic, so their sum wraps modulo the width instead.
		match (lhs, rhs) {
			(Shift::None, shift) | (shift, Shift::None) => Some(shift),
			(Shift::Sll(a), Shift::Sll(b)) if a + b < 64 => Some(Shift::Sll(a + b)),
			(Shift::Sll32(a), Shift::Sll32(b)) if a + b < 32 => Some(Shift::Sll32(a + b)),
			(Shift::Srl(a), Shift::Srl(b)) if a + b < 64 => Some(Shift::Srl(a + b)),
			(Shift::Srl32(a), Shift::Srl32(b)) if a + b < 32 => Some(Shift::Srl32(a + b)),
			(Shift::Sar(a), Shift::Sar(b)) if a + b < 64 => Some(Shift::Sar(a + b)),
			(Shift::Sra32(a), Shift::Sra32(b)) if a + b < 32 => Some(Shift::Sra32(a + b)),
			(Shift::Rotr(a), Shift::Rotr(b)) => Some(Shift::Rotr((a + b) % 64)),
			(Shift::Rotr32(a), Shift::Rotr32(b)) => Some(Shift::Rotr32((a + b) % 32)),
			_ => None,
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::{ShiftVariant, ValueIndex};
	use cranelift_entity::{EntityRef, SecondaryMap};

	use super::Shift;
	use crate::compiler::{
		Wire,
		constraint_builder::{ConstraintBuilder, expr},
	};

	#[test]
	fn of_all_one_detects_end_anchored_runs() {
		use binius_core::word::Word;

		// The zero word and the all-ones word have no proper shift form.
		// A zero-amount shift is the identity, which is not a useful alias.
		assert_eq!(Shift::of_all_one(Word::ZERO), None);
		assert_eq!(Shift::of_all_one(Word::ALL_ONE), None);

		// Three common single-bit-run constants and their shifts.
		assert_eq!(Shift::of_all_one(Word::ONE), Some(Shift::Srl(63)));
		assert_eq!(Shift::of_all_one(Word::MASK_32), Some(Shift::Srl(32)));
		assert_eq!(Shift::of_all_one(Word::MSB_ONE), Some(Shift::Sll(63)));

		// Low runs of ones are right shifts: k = 64 - popcount.
		assert_eq!(Shift::of_all_one(Word(0xFF)), Some(Shift::Srl(56)));
		assert_eq!(Shift::of_all_one(Word(0x7FFF_FFFF_FFFF_FFFF)), Some(Shift::Srl(1)));

		// High runs of ones are left shifts: k = trailing_zeros.
		assert_eq!(Shift::of_all_one(Word(0xFF00_0000_0000_0000)), Some(Shift::Sll(56)));
		assert_eq!(Shift::of_all_one(Word(0xFFFF_FFFF_FFFF_FFFE)), Some(Shift::Sll(1)));

		// Ones that are not anchored at an end have no single-shift form.
		assert_eq!(Shift::of_all_one(Word(0xFF00)), None);
		assert_eq!(Shift::of_all_one(Word(0x8000_0000)), None);
		assert_eq!(Shift::of_all_one(Word(0b110)), None);
	}

	#[test]
	fn as_variant_amount_maps_nonzero_shifts() {
		// The identity and every zero-amount shift carry no work.
		assert_eq!(Shift::None.as_variant_amount(), None);
		assert_eq!(Shift::Sll(0).as_variant_amount(), None);
		assert_eq!(Shift::Rotr32(0).as_variant_amount(), None);

		// Nonzero shifts map to their core variant and amount.
		assert_eq!(Shift::Sll(63).as_variant_amount(), Some((ShiftVariant::Sll, 63)));
		assert_eq!(Shift::Srl(32).as_variant_amount(), Some((ShiftVariant::Slr, 32)));
		assert_eq!(Shift::Sar(7).as_variant_amount(), Some((ShiftVariant::Sar, 7)));
		assert_eq!(Shift::Rotr(5).as_variant_amount(), Some((ShiftVariant::Rotr, 5)));
		assert_eq!(Shift::Sll32(3).as_variant_amount(), Some((ShiftVariant::Sll32, 3)));
	}

	#[test]
	fn of_all_one_round_trips_through_shift_application() {
		use binius_core::word::Word;

		// Every detected shift, applied to ALL_ONE, must reproduce the original word.
		for w in [
			Word::ONE,
			Word::MASK_32,
			Word::MSB_ONE,
			Word(0xFF),
			Word(0xFF00_0000_0000_0000),
			Word(0x0000_FFFF_FFFF_FFFF),
		] {
			let (variant, amount) = Shift::of_all_one(w).unwrap().as_variant_amount().unwrap();
			assert_eq!(variant.apply(Word::ALL_ONE, amount as usize), w);
		}
	}

	#[test]
	fn rotr_zero_folds_to_plain_via_linear() {
		// A rotr-by-0 term must lower to a plain value index; a rotr-by-n>0 must stay native.
		let mut wire_mapping = SecondaryMap::new();
		let wire_a = Wire::new(0);
		let wire_b = Wire::new(1);
		let wire_c = Wire::new(2);
		let all_one_wire = Wire::new(3);

		wire_mapping[wire_a] = ValueIndex(0);
		wire_mapping[wire_b] = ValueIndex(1);
		wire_mapping[wire_c] = ValueIndex(2);
		wire_mapping[all_one_wire] = ValueIndex(3);

		// c = rotr(a, 0) ^ b  ->  rotr(0) collapses to plain(a).
		{
			let mut builder = ConstraintBuilder::new();
			builder
				.linear()
				.rhs(expr::xor2(expr::rotr(wire_a, 0), wire_b))
				.dst(wire_c)
				.build();

			let (and_constraints, imul_constraints, _bmul_constraints) =
				builder.build(&wire_mapping, all_one_wire);

			// Linear lowers to `(a ^ b) & all_one = c`.
			assert_eq!(and_constraints.len(), 1);
			assert_eq!(imul_constraints.len(), 0);

			let and_c = &and_constraints[0];

			assert_eq!(and_c.a.len(), 2);
			assert!(
				and_c
					.a
					.iter()
					.any(|svi| svi.value_index == ValueIndex(0) && svi.amount == 0)
			);
			assert!(
				and_c
					.a
					.iter()
					.any(|svi| svi.value_index == ValueIndex(1) && svi.amount == 0)
			);

			assert_eq!(and_c.b.len(), 1);
			assert_eq!(and_c.b[0].value_index, ValueIndex(3));
			assert_eq!(and_c.b[0].amount, 0);

			assert_eq!(and_c.c.len(), 1);
			assert_eq!(and_c.c[0].value_index, ValueIndex(2));
			assert_eq!(and_c.c[0].amount, 0);
		}

		// c = rotr(a, 5) ^ b  ->  native rotr(a, 5).
		{
			let mut builder = ConstraintBuilder::new();
			builder
				.linear()
				.rhs(expr::xor2(expr::rotr(wire_a, 5), wire_b))
				.dst(wire_c)
				.build();

			let (and_constraints, imul_constraints, _bmul_constraints) =
				builder.build(&wire_mapping, all_one_wire);

			assert_eq!(and_constraints.len(), 1);
			assert_eq!(imul_constraints.len(), 0);

			let and_c = &and_constraints[0];
			assert_eq!(and_c.a.len(), 2);
			assert!(and_c.a.iter().any(|svi| {
				svi.value_index == ValueIndex(0)
					&& svi.amount == 5
					&& matches!(svi.shift_variant, ShiftVariant::Rotr)
			}));
			assert!(
				and_c
					.a
					.iter()
					.any(|svi| svi.value_index == ValueIndex(1) && svi.amount == 0)
			);
		}
	}

	#[test]
	fn rotr_folds_inside_and_operand() {
		// The same rotr(0)->plain and rotr(n)->native folding must hold inside an AND operand.
		let mut wire_mapping = SecondaryMap::new();
		let wire_a = Wire::new(0);
		let wire_b = Wire::new(1);
		let wire_c = Wire::new(2);
		let all_one_wire = Wire::new(3);

		wire_mapping[wire_a] = ValueIndex(0);
		wire_mapping[wire_b] = ValueIndex(1);
		wire_mapping[wire_c] = ValueIndex(2);
		wire_mapping[all_one_wire] = ValueIndex(3);

		// a & rotr(b, 0) = c  ->  b stays plain.
		{
			let mut builder = ConstraintBuilder::new();
			builder
				.and()
				.a(wire_a)
				.b(expr::rotr(wire_b, 0))
				.c(wire_c)
				.build();

			let (and_constraints, _, _) = builder.build(&wire_mapping, all_one_wire);

			assert_eq!(and_constraints.len(), 1);
			let and_c = &and_constraints[0];

			assert_eq!(and_c.a.len(), 1);
			assert_eq!(and_c.a[0].value_index, ValueIndex(0));
			assert_eq!(and_c.a[0].amount, 0);

			assert_eq!(and_c.b.len(), 1);
			assert_eq!(and_c.b[0].value_index, ValueIndex(1));
			assert_eq!(and_c.b[0].amount, 0);

			assert_eq!(and_c.c.len(), 1);
			assert_eq!(and_c.c[0].value_index, ValueIndex(2));
			assert_eq!(and_c.c[0].amount, 0);
		}

		// a & rotr(b, 8) = c  ->  b keeps native rotr(8).
		{
			let mut builder = ConstraintBuilder::new();
			builder
				.and()
				.a(wire_a)
				.b(expr::rotr(wire_b, 8))
				.c(wire_c)
				.build();

			let (and_constraints, _, _) = builder.build(&wire_mapping, all_one_wire);

			assert_eq!(and_constraints.len(), 1);
			let and_c = &and_constraints[0];
			assert_eq!(and_c.b.len(), 1);
			assert!(and_c.b.iter().any(|svi| {
				svi.value_index == ValueIndex(1)
					&& svi.amount == 8
					&& matches!(svi.shift_variant, ShiftVariant::Rotr)
			}));
		}
	}
}
