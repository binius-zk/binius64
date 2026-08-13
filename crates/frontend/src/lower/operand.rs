// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The shift algebra shared by operands: a core [`Shift`] applied to a [`Wire`].

use std::ops::Index;

use binius_core::constraint_system::{Composition, Shift, ShiftedValueIndex, ValueIndex};
use cranelift_entity::{EntitySet, SecondaryMap};

use crate::ir::Wire;

/// A single wire term of an operand, tagged with the shifts to apply to it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShiftedWire {
	/// The wire the shifts apply to.
	pub wire: Wire,
	/// The shifts folded into this term, innermost first, matching
	/// [`ShiftedValueIndex::shift_seq`].
	///
	/// A term the builder produces carries one shift, in the inner slot. The second slot is filled
	/// only by gate fusion, and only for a chain that [`Shift::compose`] cannot collapse — see
	/// [`push_inner`].
	pub shift_seq: [Shift; 2],
}

impl ShiftedWire {
	/// A term carrying one shift, which the canonical form places in the inner slot.
	pub const fn single(wire: Wire, shift: Shift) -> Self {
		Self {
			wire,
			shift_seq: [shift, Shift::IDENTITY],
		}
	}

	/// The one shift this term carries.
	///
	/// ## Preconditions
	///
	/// * the term is singly shifted, which every term the builder produces is — a second slot is
	///   only ever filled by gate fusion, over its own output
	pub fn sole_shift(&self) -> Shift {
		let [inner, outer] = self.shift_seq;
		assert!(outer.is_identity(), "precondition: the term carries one shift");
		inner
	}

	/// Lowers this term to a core [`ShiftedValueIndex`] via the wire mapping.
	pub(super) fn to_shifted_value_index(
		self,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
	) -> ShiftedValueIndex {
		ShiftedValueIndex::new(wire_mapping[self.wire], self.shift_seq)
	}
}

/// What folding one more shift inside a shift sequence comes to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PushInner {
	/// The sequence with the shift folded in, filling no more slots than the two a term has.
	Seq([Shift; 2]),
	/// The shifts clear every bit between them, so the term they apply to is identically zero.
	///
	/// An operand is an XOR, so such a term contributes nothing and is dropped rather than spelled.
	Zero,
	/// The shift would need a third slot, which no term can carry.
	///
	/// Gate fusion's commit set refuses to inline a path that would reach here, so a caller that
	/// respects its decision never sees this.
	OverBudget,
}

/// The canonical sequence carrying one shift: the shift inner, and an identity spelled one way.
///
/// [`Shift::compose`] reports a cancellation with whichever variant produced it, so an identity
/// can arrive spelled `rotr(0)` or `srl(0)`. Letting that through would give one transformation
/// several spellings, and the prover keys a row of its shift tables on the spelling — so equal
/// transformations spelled differently cost it a row each instead of sharing one.
const fn lone(shift: Shift) -> [Shift; 2] {
	if shift.is_identity() {
		[Shift::IDENTITY, Shift::IDENTITY]
	} else {
		[shift, Shift::IDENTITY]
	}
}

/// Folds `shift` into `seq` as the new innermost shift, as greedily as the shifts compose.
///
/// Inlining a definition applies the consumer's accumulated shifts to a term that already carries
/// one of its own, and that term's shift lands innermost:
///
/// ```text
///     definition:  y = sll(x, 4)
///     consumer:    ... ^ sll(y, 3) ^ ...        seq = [sll(3), id]
///     substituted: ... ^ sll(x, 7) ^ ...        push_inner(seq, sll(4), _) -> [sll(7), id]
/// ```
///
/// A slot is spent only where [`Shift::compose`] genuinely cannot collapse two shifts, so a
/// sequence reaches its second slot only where one shift could not have done the job. Keeping to
/// that matters beyond tidiness: the prover carries one row of its shift tables per distinct
/// sequence, so a pair left unmerged is a row it did not need.
///
/// # Arguments
///
/// - `slots`: how many shifts a term may carry. A sequence needing more is refused rather than
///   truncated, since dropping a shift would change what the term denotes.
pub fn push_inner(seq: [Shift; 2], shift: Shift, slots: usize) -> PushInner {
	let [inner, outer] = seq;

	// Fold the new shift into the inner slot first, which is where it applies.
	let inner = match Shift::compose(shift, inner) {
		Composition::Single(merged) => merged,
		// Clearing the inner word leaves the outer shift nothing to carry.
		Composition::Zero => return PushInner::Zero,
		// The two need a slot each. That is three shifts for two slots unless the outer slot is
		// still free, and both halves are non-identity here or one would have absorbed the other.
		Composition::Pair if slots >= 2 && outer.is_identity() => {
			return PushInner::Seq([shift, inner]);
		}
		Composition::Pair => return PushInner::OverBudget,
	};

	// Merging inward can bring the two slots within reach of each other, so try again outward.
	// A full-width shift past the halfway point is the case that does it: it carries every bit
	// clear of its own half, after which a half-word shift continues it as if it were full-width.
	//
	//     [sll(1), sll32(11)]  is a genuine pair — 1 is not past the halfway point
	//     fold sll(31) inside  -> [sll(32), sll32(11)], whose halves now chain into sll(43)
	match Shift::compose(inner, outer) {
		Composition::Single(collapsed) => PushInner::Seq(lone(collapsed)),
		Composition::Zero => PushInner::Zero,
		Composition::Pair => PushInner::Seq([inner, outer]),
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

	/// The terms this operand XORs together.
	pub fn as_slice(&self) -> &[ShiftedWire] {
		&self.0
	}

	/// The number of terms.
	pub const fn len(&self) -> usize {
		self.0.len()
	}

	/// Whether the operand has no terms.
	///
	/// An empty XOR is the constant zero, so such an operand contributes nothing.
	pub const fn is_empty(&self) -> bool {
		self.0.is_empty()
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

impl Index<usize> for WireOperand {
	type Output = ShiftedWire;

	fn index(&self, term: usize) -> &Self::Output {
		&self.0[term]
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

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::{ShiftVariant, ValueIndex};
	use cranelift_entity::{EntityRef, SecondaryMap};

	use super::*;
	use crate::lower::{ConstraintBuilder, expr};

	#[test]
	fn rotr_zero_folds_to_plain_via_linear() {
		// A rotr-by-0 term must lower to a plain value index; a rotr-by-n>0 must stay native.
		let mut wire_mapping = SecondaryMap::with_default(ValueIndex::scratch(0));
		let wire_a = Wire::new(0);
		let wire_b = Wire::new(1);
		let wire_c = Wire::new(2);
		let all_one_wire = Wire::new(3);

		wire_mapping[wire_a] = ValueIndex::private(0);
		wire_mapping[wire_b] = ValueIndex::private(1);
		wire_mapping[wire_c] = ValueIndex::private(2);
		wire_mapping[all_one_wire] = ValueIndex::private(3);

		// c = rotr(a, 0) ^ b  ->  rotr(0) collapses to plain(a).
		{
			let mut builder = ConstraintBuilder::new();
			builder.linear(expr::xor2(expr::rotr(wire_a, 0), wire_b), wire_c);

			let (zero_constraints, and_constraints, imul_constraints, _bmul_constraints) =
				builder.build(&wire_mapping);

			// Linear lowers to the ZERO constraint `a ^ b ^ c = 0`.
			assert_eq!(zero_constraints.len(), 1);
			assert_eq!(and_constraints.len(), 0);
			assert_eq!(imul_constraints.len(), 0);

			let val = zero_constraints[0].val();
			assert_eq!(val.len(), 3);
			assert!(
				val.iter()
					.any(|svi| svi.value_index == ValueIndex::private(0) && svi.is_unshifted())
			);
			assert!(
				val.iter()
					.any(|svi| svi.value_index == ValueIndex::private(1) && svi.is_unshifted())
			);
			// The destination joins the operand rather than sitting in its own `c`.
			assert!(
				val.iter()
					.any(|svi| svi.value_index == ValueIndex::private(2) && svi.is_unshifted())
			);
		}

		// c = rotr(a, 5) ^ b  ->  native rotr(a, 5).
		{
			let mut builder = ConstraintBuilder::new();
			builder.linear(expr::xor2(expr::rotr(wire_a, 5), wire_b), wire_c);

			let (zero_constraints, and_constraints, imul_constraints, _bmul_constraints) =
				builder.build(&wire_mapping);

			assert_eq!(zero_constraints.len(), 1);
			assert_eq!(and_constraints.len(), 0);
			assert_eq!(imul_constraints.len(), 0);

			let val = zero_constraints[0].val();
			assert_eq!(val.len(), 3);
			assert!(val.iter().any(|svi| {
				svi.value_index == ValueIndex::private(0)
					&& svi.inner().amount == 5
					&& matches!(svi.inner().variant, ShiftVariant::Rotr)
			}));
			assert!(
				val.iter()
					.any(|svi| svi.value_index == ValueIndex::private(1) && svi.is_unshifted())
			);
		}
	}

	#[test]
	fn rotr_folds_inside_and_operand() {
		// The same rotr(0)->plain and rotr(n)->native folding must hold inside an AND operand.
		let mut wire_mapping = SecondaryMap::with_default(ValueIndex::scratch(0));
		let wire_a = Wire::new(0);
		let wire_b = Wire::new(1);
		let wire_c = Wire::new(2);
		let all_one_wire = Wire::new(3);

		wire_mapping[wire_a] = ValueIndex::private(0);
		wire_mapping[wire_b] = ValueIndex::private(1);
		wire_mapping[wire_c] = ValueIndex::private(2);
		wire_mapping[all_one_wire] = ValueIndex::private(3);

		// a & rotr(b, 0) = c  ->  b stays plain.
		{
			let mut builder = ConstraintBuilder::new();
			builder.and(wire_a, expr::rotr(wire_b, 0), wire_c);

			let (_, and_constraints, _, _) = builder.build(&wire_mapping);

			assert_eq!(and_constraints.len(), 1);
			let and_c = &and_constraints[0];

			assert_eq!(and_c.a().len(), 1);
			assert_eq!(and_c.a()[0].value_index, ValueIndex::private(0));
			assert_eq!(and_c.a()[0].inner().amount, 0);

			assert_eq!(and_c.b().len(), 1);
			assert_eq!(and_c.b()[0].value_index, ValueIndex::private(1));
			assert_eq!(and_c.b()[0].inner().amount, 0);

			assert_eq!(and_c.c().len(), 1);
			assert_eq!(and_c.c()[0].value_index, ValueIndex::private(2));
			assert_eq!(and_c.c()[0].inner().amount, 0);
		}

		// a & rotr(b, 8) = c  ->  b keeps native rotr(8).
		{
			let mut builder = ConstraintBuilder::new();
			builder.and(wire_a, expr::rotr(wire_b, 8), wire_c);

			let (_, and_constraints, _, _) = builder.build(&wire_mapping);

			assert_eq!(and_constraints.len(), 1);
			let and_c = &and_constraints[0];
			assert_eq!(and_c.b().len(), 1);
			assert!(and_c.b().iter().any(|svi| {
				svi.value_index == ValueIndex::private(1)
					&& svi.inner().amount == 8
					&& matches!(svi.inner().variant, ShiftVariant::Rotr)
			}));
		}
	}

	/// The identity is what an unshifted term's sequence holds in both slots.
	const NONE: [Shift; 2] = [Shift::IDENTITY, Shift::IDENTITY];

	/// The budget the flip will run at, which is what these cases are about.
	const TWO: usize = 2;

	#[test]
	fn push_inner_merges_as_greedily_as_the_shifts_compose() {
		// Two shifts of one variant chain into one, so the sequence keeps its second slot free.
		// Spending a slot here would cost the prover a second shift-table row for a
		// transformation one shift already spells.
		assert_eq!(
			push_inner([Shift::sll(3), Shift::IDENTITY], Shift::sll(5), TWO),
			PushInner::Seq([Shift::sll(8), Shift::IDENTITY])
		);
		// The identity folds into anything, from either side.
		assert_eq!(
			push_inner(NONE, Shift::rotr(7), TWO),
			PushInner::Seq([Shift::rotr(7), Shift::IDENTITY])
		);
		assert_eq!(
			push_inner([Shift::rotr(7), Shift::IDENTITY], Shift::IDENTITY, TWO),
			PushInner::Seq([Shift::rotr(7), Shift::IDENTITY])
		);
	}

	#[test]
	fn push_inner_fills_the_second_slot_in_application_order() {
		// A genuine pair: shifting a field up and arithmetically back down sign-extends it, and no
		// single shift does that. The one that applies first lands in the inner slot.
		let (inner, outer) = (Shift::sll(40), Shift::sar(40));
		assert_eq!(
			push_inner([outer, Shift::IDENTITY], inner, TWO),
			PushInner::Seq([inner, outer])
		);
	}

	#[test]
	fn push_inner_drops_a_term_the_shifts_clear() {
		// Shifting a word 126 places off one end leaves nothing, so the term is identically zero.
		assert_eq!(
			push_inner([Shift::sll(63), Shift::IDENTITY], Shift::sll(63), TWO),
			PushInner::Zero
		);
		// The outer slot cannot bring it back: a shift of nothing is nothing.
		assert_eq!(
			push_inner([Shift::sll(63), Shift::rotr(9)], Shift::sll(63), TWO),
			PushInner::Zero
		);
	}

	#[test]
	fn push_inner_returns_a_cancelled_pair_to_a_lone_shift() {
		// Rotating one way and back cancels, leaving the outer shift standing alone — which
		// belongs in the inner slot, since that is where the canonical form puts a lone shift.
		assert_eq!(
			push_inner([Shift::rotr(1), Shift::sar(7)], Shift::rotr(63), TWO),
			PushInner::Seq([Shift::sar(7), Shift::IDENTITY])
		);
		// With nothing outside either, the term is left unshifted, spelled the canonical way
		// rather than as the rotate that cancelled.
		assert_eq!(
			push_inner([Shift::rotr(1), Shift::IDENTITY], Shift::rotr(63), TWO),
			PushInner::Seq(NONE)
		);
	}

	#[test]
	fn push_inner_refuses_a_third_slot() {
		// Both slots hold shifts that do not chain, so a third non-composing shift has nowhere to
		// go. Reporting that is what lets the caller trust the commit set rather than silently
		// dropping the shift, which would change what the term denotes.
		let full = [Shift::sll(40), Shift::sar(40)];
		assert_eq!(push_inner(full, Shift::rotr(9), TWO), PushInner::OverBudget);
		// A shift that *does* chain with the inner slot still fits, full sequence or not.
		assert_eq!(
			push_inner(full, Shift::sll(5), TWO),
			PushInner::Seq([Shift::sll(45), Shift::sar(40)])
		);
	}

	#[test]
	fn every_reachable_sequence_is_one_the_constraint_system_accepts() {
		// Emitting a sequence core would reject is the compiler's failure, not core's, so the
		// property is that *no reachable sequence* is rejectable. Reachable means: start from what
		// the builder emits — one shift, inner slot — and fold in shifts until nothing new appears.
		// That closure is exactly the set inlining can produce, however deep a chain it walks.
		//
		// The two faults being ruled out are the ones `ConstraintSystem::operand_fault` names:
		// a lone shift sitting in the outer slot, and a "pair" whose halves collapse — the latter
		// being what a naive "didn't compose, so append" would emit.
		let alphabet = [
			Shift::IDENTITY,
			Shift::sll(1),
			Shift::sll(40),
			Shift::srl(8),
			Shift::srl(63),
			Shift::sar(7),
			Shift::sar(40),
			Shift::rotr(1),
			Shift::rotr(63),
			Shift::sll32(11),
			Shift::sra32(11),
			Shift::rotr32(5),
		];

		let mut reachable: std::collections::BTreeSet<[Shift; 2]> = alphabet
			.iter()
			.map(|&shift| [shift, Shift::IDENTITY])
			.collect();
		loop {
			let mut next = reachable.clone();
			for &seq in &reachable {
				for &shift in &alphabet {
					if let PushInner::Seq(folded) = push_inner(seq, shift, TWO) {
						next.insert(folded);
					}
				}
			}
			if next.len() == reachable.len() {
				break;
			}
			reachable = next;
		}

		// A doubly shifted sequence was reached, or this proves nothing about the second slot.
		assert!(reachable.iter().any(|[_, outer]| !outer.is_identity()));

		for seq in reachable {
			let [inner, outer] = seq;
			if outer.is_identity() {
				assert!(inner.is_canonical(), "one transformation, two spellings: {seq:?}");
			} else {
				assert!(!inner.is_identity(), "a lone shift belongs in the inner slot: {seq:?}");
				assert_eq!(
					Shift::compose(inner, outer),
					Composition::Pair,
					"a pair that collapses should have been merged: {seq:?}"
				);
			}
		}
	}
}
