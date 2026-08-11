// Copyright 2026 The Binius Developers

//! The GHASH-field element a circuit-building channel carries.

use std::{
	array,
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
	rc::{Rc, Weak},
};

use binius_field::{
	BinaryField128bGhash as B128, ExtensionField, Field, FieldOps,
	arithmetic_traits::{InvertOrZero, Square},
};
use binius_frontend::{CircuitBuilder, Wire};

use crate::shared::Shared;

/// An element of `GF(2^128)` that is either fixed while the circuit is built or carried by a pair
/// of wires.
///
/// The wire pair is the `(lo, hi)` split the frontend's `bmul` gate takes, so one field
/// multiplication is one BMUL constraint and addition is two XORs, which the constraint system
/// absorbs into its operands for free.
///
/// A `Constant` folds while the circuit is built. That matters more than it looks: the verifier's
/// arithmetic is full of build-time constants — subspace bases, Lagrange weights, eq-indicator
/// evaluations at fixed points — and folding them costs no constraints at all.
#[derive(Clone)]
pub enum SymbolicElem {
	Constant(B128),
	Wires {
		shared: Weak<Shared>,
		lo: Wire,
		hi: Wire,
	},
}

impl SymbolicElem {
	/// Constructs a wire-backed element anchored to the shared builder.
	pub fn wires(shared: &Rc<Shared>, lo: Wire, hi: Wire) -> Self {
		Self::Wires {
			shared: Rc::downgrade(shared),
			lo,
			hi,
		}
	}

	/// The shared builder this element is anchored to, if it is wire-backed.
	pub fn shared(&self) -> Option<Rc<Shared>> {
		match self {
			Self::Constant(_) => None,
			Self::Wires { shared, .. } => Some(
				shared
					.upgrade()
					.expect("a SymbolicElem outlived the channel that created it"),
			),
		}
	}

	/// Lowers to a `(lo, hi)` wire pair, materializing a `Constant` on the builder.
	pub fn to_wires(&self, builder: &CircuitBuilder) -> (Wire, Wire) {
		match self {
			Self::Constant(value) => {
				let value = u128::from(*value);
				(
					builder.add_constant_64(value as u64),
					builder.add_constant_64((value >> 64) as u64),
				)
			}
			Self::Wires { lo, hi, .. } => (*lo, *hi),
		}
	}

	/// Combines two elements, folding at the field level when both are constants and otherwise
	/// running `gate` over the wire pairs on the shared builder.
	fn combine(
		&self,
		rhs: &Self,
		fold: impl Fn(B128, B128) -> B128,
		gate: impl Fn(&CircuitBuilder, (Wire, Wire), (Wire, Wire)) -> (Wire, Wire),
	) -> Self {
		let shared = match (self, rhs) {
			(Self::Constant(a), Self::Constant(b)) => return Self::Constant(fold(*a, *b)),
			(Self::Wires { shared, .. }, _) | (_, Self::Wires { shared, .. }) => shared,
		};
		let Some(owner) = shared.upgrade() else {
			panic!("a SymbolicElem outlived the channel that created it");
		};
		let builder = owner.builder();
		let (lo, hi) = gate(builder, self.to_wires(builder), rhs.to_wires(builder));
		Self::wires(&owner, lo, hi)
	}
}

// In characteristic 2 negation is the identity.
impl Neg for SymbolicElem {
	type Output = Self;

	fn neg(self) -> Self {
		self
	}
}

impl Add<&Self> for SymbolicElem {
	type Output = Self;

	fn add(self, rhs: &Self) -> Self {
		self.combine(
			rhs,
			|a, b| a + b,
			|builder, (a_lo, a_hi), (b_lo, b_hi)| {
				(builder.bxor(a_lo, b_lo), builder.bxor(a_hi, b_hi))
			},
		)
	}
}

impl Mul<&Self> for SymbolicElem {
	type Output = Self;

	fn mul(self, rhs: &Self) -> Self {
		// A constant factor of zero or one settles the product without a constraint, which is
		// worth catching: the verifier multiplies by eq-indicator terms and Lagrange weights that
		// are often one or zero at build time.
		if matches!(&self, Self::Constant(c) if *c == B128::ZERO)
			|| matches!(rhs, Self::Constant(c) if *c == B128::ZERO)
		{
			return Self::Constant(B128::ZERO);
		}
		if matches!(rhs, Self::Constant(c) if *c == B128::ONE) {
			return self;
		}
		if matches!(&self, Self::Constant(c) if *c == B128::ONE) {
			return rhs.clone();
		}
		self.combine(
			rhs,
			|a, b| a * b,
			|builder, (a_lo, a_hi), (b_lo, b_hi)| builder.bmul(a_lo, a_hi, b_lo, b_hi),
		)
	}
}

impl Sub<&Self> for SymbolicElem {
	type Output = Self;

	// Subtraction is addition in characteristic 2, which is what the shared `combine` records.
	#[allow(clippy::suspicious_arithmetic_impl)]
	fn sub(self, rhs: &Self) -> Self {
		self + rhs
	}
}

macro_rules! by_value {
	($trait:ident, $method:ident) => {
		impl $trait for SymbolicElem {
			type Output = Self;

			fn $method(self, rhs: Self) -> Self {
				self.$method(&rhs)
			}
		}
	};
}
by_value!(Add, add);
by_value!(Sub, sub);
by_value!(Mul, mul);

macro_rules! assign {
	($trait:ident, $method:ident, $op:ident) => {
		impl $trait for SymbolicElem {
			fn $method(&mut self, rhs: Self) {
				*self = self.clone().$op(&rhs);
			}
		}
		impl $trait<&Self> for SymbolicElem {
			fn $method(&mut self, rhs: &Self) {
				*self = self.clone().$op(rhs);
			}
		}
	};
}
assign!(AddAssign, add_assign, add);
assign!(SubAssign, sub_assign, sub);
assign!(MulAssign, mul_assign, mul);

impl Sum for SymbolicElem {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(B128::ZERO), |acc, x| acc + x)
	}
}

impl<'a> Sum<&'a Self> for SymbolicElem {
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(B128::ZERO), |acc, x| acc + x)
	}
}

impl Product for SymbolicElem {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(B128::ONE), |acc, x| acc * x)
	}
}

impl<'a> Product<&'a Self> for SymbolicElem {
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(B128::ONE), |acc, x| acc * x)
	}
}

impl Square for SymbolicElem {
	fn square(self) -> Self {
		self.clone() * &self
	}
}

impl InvertOrZero for SymbolicElem {
	fn invert_or_zero(self) -> Self {
		let Some(owner) = self.shared() else {
			// A constant inverts while the circuit is built.
			let Self::Constant(value) = self else {
				unreachable!("only a constant has no shared builder")
			};
			return Self::Constant(value.invert_or_zero());
		};
		let builder = owner.builder();
		let (lo, hi) = self.to_wires(builder);
		// The value still arrives as a hint, since a circuit cannot divide. What follows is what
		// makes it binding.
		let out = builder.call_hint(crate::hints::InvertOrZeroHint, &[], &[lo, hi]);
		constrain_inverse(builder, (lo, hi), (out[0], out[1]));
		Self::wires(&owner, out[0], out[1])
	}
}

/// Pins a hinted inverse to its input, so a prover cannot supply anything else.
///
/// # Why both assertions are needed
///
/// Write `p` for the product of the value and its claimed inverse, then assert:
///
/// ```text
///     x * p == x        at x != 0 this divides down to x * inv == 1
///     inv * p == inv    at x == 0 this reads inv * 0 == inv, forcing inv == 0
/// ```
///
/// Neither suffices alone, and the cheaper combined form is unsound:
///
/// - Keeping only the first admits any claimed inverse at `x = 0`, since both sides vanish.
/// - Adding the two into one assertion admits `inv = x`, since in characteristic 2 that also
///   vanishes on both sides.
///
/// # Cost
///
/// 3 BMUL constraints, and 4 ZERO constraints for the two asserted wire pairs.
fn constrain_inverse(builder: &CircuitBuilder, x: (Wire, Wire), inv: (Wire, Wire)) {
	// The product both assertions are phrased over, computed once and reused.
	let p = builder.bmul(x.0, x.1, inv.0, inv.1);

	// First assertion: scaling that product by the value returns the value itself.
	let scaled = builder.bmul(x.0, x.1, p.0, p.1);
	builder.assert_eq_v("x_times_p", [scaled.0, scaled.1], [x.0, x.1]);

	// Second assertion: scaling it by the claimed inverse returns the claimed inverse.
	let reflected = builder.bmul(inv.0, inv.1, p.0, p.1);
	builder.assert_eq_v("inv_times_p", [reflected.0, reflected.1], [inv.0, inv.1]);
}

impl From<B128> for SymbolicElem {
	fn from(value: B128) -> Self {
		Self::Constant(value)
	}
}

impl FieldOps for SymbolicElem {
	type Scalar = B128;

	fn zero() -> Self {
		Self::Constant(B128::ZERO)
	}

	fn one() -> Self {
		Self::Constant(B128::ONE)
	}

	fn square_transpose<FSub: Field>(elems: &mut [Self])
	where
		B128: ExtensionField<FSub>,
	{
		let degree = <B128 as ExtensionField<FSub>>::DEGREE;
		assert_eq!(elems.len(), degree); // precondition
		if degree == 1 {
			return;
		}
		// The network below reads bit `j` of a wire pair as subfield coefficient `j`, which is only
		// the decomposition asked for when the subfield is the bit field.
		assert_eq!(
			degree,
			B128::ORDER_EXPONENT,
			"square_transpose over B128 is implemented for the trivial and the bit subfield only"
		);
		// That reading also depends on the basis being the monomial one, so confirm it rather than
		// assume it.
		assert!(
			(0..degree).all(|i| {
				<B128 as ExtensionField<FSub>>::basis(i) == <B128 as From<u128>>::from(1u128 << i)
			}),
			"the bit transpose assumes basis(i) = X^i"
		);

		// All-constant inputs transpose while the circuit is built.
		let Some(owner) = elems.iter().find_map(Self::shared) else {
			let mut values = elems
				.iter()
				.map(|elem| match elem {
					Self::Constant(value) => *value,
					Self::Wires { .. } => unreachable!("no element is wire-backed here"),
				})
				.collect::<Vec<_>>();
			<B128 as ExtensionField<FSub>>::square_transpose(&mut values);
			for (elem, value) in elems.iter_mut().zip(values) {
				*elem = Self::Constant(value);
			}
			return;
		};

		// One subcircuit gathers all 768 constraints under a single path.
		let builder = owner.builder().subcircuit("square_transpose");
		// Elements settled at build time become constant rows of the matrix at this point.
		let words = elems
			.iter()
			.map(|elem| elem.to_wires(&builder))
			.collect::<Vec<_>>();

		// Row i of the matrix is element i's 128 bits, so the four 64 x 64 blocks are the low and
		// high wires of the low and high halves of the input:
		//
		//     block 0 = low  wires of elements 0 to 63     = A
		//     block 1 = high wires of elements 0 to 63     = B
		//     block 2 = low  wires of elements 64 to 127   = C
		//     block 3 = high wires of elements 64 to 127   = D
		let mut blocks: [[Wire; BLOCK]; 4] = array::from_fn(|block| {
			// Blocks are counted row half first, then which wire of the pair.
			let (half, part) = (block / 2, block % 2);
			array::from_fn(|row| {
				let (lo, hi) = words[half * BLOCK + row];
				if part == 0 { lo } else { hi }
			})
		});
		// Each block is an independent 64 x 64 transpose over its own 64 wires.
		for block in &mut blocks {
			transpose_64(&builder, block);
		}
		let [a, b, c, d] = blocks;

		// The off-diagonal blocks trade places, which is only a question of which wire is read:
		//
		//     [ A  B ]  ->  [ A^T  C^T ]
		//     [ C  D ]      [ B^T  D^T ]
		for row in 0..BLOCK {
			elems[row] = Self::wires(&owner, a[row], c[row]);
			elems[BLOCK + row] = Self::wires(&owner, b[row], d[row]);
		}
	}
}

/// Rows in one of the four square blocks the 128 x 128 transpose splits into.
const BLOCK: usize = binius_core::word::Word::BITS;

/// Transposes in place the 64 x 64 bit matrix whose row `i` is wire `i` and column `j` is bit `j`.
///
/// # Algorithm
///
/// A `2h x 2h` block transposes by swapping its two off-diagonal `h x h` halves, then transposing
/// each of its four `h x h` sub-blocks:
///
/// ```text
///     [ P  Q ]  swap   [ P  R ]  recurse   [ P^T  R^T ]
///     [ R  S ]  ---->  [ Q  S ]  ------>   [ Q^T  S^T ]
/// ```
///
/// Six stages take `h` from 32 down to 1.
/// Each stage visits every wire pair straddling a block's midline and trades the bits their two
/// halves must exchange:
///
/// ```text
///     t    = ((row >> h) ^ mate) & MASK      MASK selects bit p where p mod 2h < h
///     mate = mate ^ t
///     row  = row ^ (t << h)
/// ```
///
/// # Cost
///
/// - The mask is the only non-linear step, so one exchange costs one AND constraint.
/// - A stage exchanges 32 pairs, so the six stages together cost 192.
fn transpose_64(builder: &CircuitBuilder, rows: &mut [Wire; BLOCK]) {
	// Start with the two largest sub-blocks and halve them every stage, down to single bits.
	let mut h = BLOCK / 2;
	while h >= 1 {
		// One mask per stage, shared by all 32 exchanges in it.
		let mask = builder.add_constant_64(low_mask(h));
		// Blocks of this stage begin at multiples of 2h, each contributing h straddling pairs.
		for base in (0..BLOCK).step_by(2 * h) {
			for row in base..base + h {
				// A row and its mate h rows down hold the two halves that must change places.
				let (r0, r1) = (rows[row], rows[row + h]);
				// Line the upper row's high half up with the lower row, then keep only the bits
				// belonging to the sub-block being swapped.
				let exchanged = builder.bxor(builder.shr(r0, h as u32), r1);
				let t = builder.band(exchanged, mask);
				// XOR is its own inverse, so folding that difference into both rows swaps them.
				rows[row] = builder.bxor(r0, builder.shl(t, h as u32));
				rows[row + h] = builder.bxor(r1, t);
			}
		}
		h /= 2;
	}
}

/// The mask selecting bit `p` exactly where `p mod 2h < h`: `h` ones, `h` zeros, repeating.
///
/// At `h = 8` this is `0x00ff00ff00ff00ff`, and at `h = 1` it is `0x5555555555555555`.
const fn low_mask(h: usize) -> u64 {
	// One run of h ones, at the bottom of the word.
	let mut mask = (1u64 << h) - 1;
	// Doubling the stride doubles the number of runs, so the word fills in six steps at worst
	// rather than one step per run.
	let mut stride = 2 * h;
	while stride < BLOCK {
		mask |= mask << stride;
		stride *= 2;
	}
	mask
}

#[cfg(test)]
mod tests {
	use std::rc::Rc;

	use binius_core::word::Word;
	use binius_field::BinaryField1b as B1;
	use binius_frontend::{CircuitStat, PopulateError};
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{merkle::element_words, shared::Shared};

	/// Builds the inverse constraints over an inverse the caller chooses, and reports what filling
	/// said about them.
	///
	/// Both sides enter on public wires, so the claimed inverse is the caller's rather than the
	/// hint's. That is what lets a test play the part of a prover supplying a forgery.
	fn run_inverse_claim(x: B128, inv: B128) -> Result<(), PopulateError> {
		let shared = Rc::new(Shared::new());
		let builder = shared.builder();
		let x_wires = [builder.add_inout(), builder.add_inout()];
		let inv_wires = [builder.add_inout(), builder.add_inout()];
		constrain_inverse(builder, (x_wires[0], x_wires[1]), (inv_wires[0], inv_wires[1]));

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for (wires, value) in [(x_wires, x), (inv_wires, inv)] {
			for (wire, word) in wires.iter().zip(element_words(u128::from(value))) {
				w[*wire] = Word(word);
			}
		}
		circuit.populate_wire_witness(&mut w)
	}

	#[test]
	fn the_inverse_constraints_admit_exactly_the_inverse() {
		// Invariant: the two assertions together pin the inverse, and each rules out a forgery the
		// other would let through.
		let mut rng = StdRng::seed_from_u64(0);
		let x = B128::from(rng.random::<u128>());

		// The honest witness, at a random value and at zero.
		run_inverse_claim(x, x.invert_or_zero()).expect("the true inverse must be admitted");
		run_inverse_claim(B128::ZERO, B128::ZERO).expect("zero inverts to zero");

		// Mutation: the two forgeries a weaker scheme admits.
		//
		//     inv = 0 at x = 0   the first assertion alone cannot see it, since both sides vanish
		//     inv = x            the tempting combined assertion cannot see it, for the same reason
		run_inverse_claim(B128::ZERO, B128::ONE).expect_err("zero has no nonzero inverse");
		run_inverse_claim(x, x).expect_err("a value is not its own inverse");
	}

	#[test]
	fn square_transpose_matches_native_at_one_and_per_exchange() {
		// Invariant: the in-circuit transpose agrees with the field's own, and the mask on each
		// exchange is its only non-linear step.
		//
		// Fixture state: 128 elements on public wires, six stages of 32 exchanges in each of the
		// four 64 x 64 blocks.
		const DEGREE: usize = B128::ORDER_EXPONENT;
		const EXCHANGES: usize = 6 * (BLOCK / 2) * 4;

		let mut rng = StdRng::seed_from_u64(1);
		let values = (0..DEGREE)
			.map(|_| B128::from(rng.random::<u128>()))
			.collect::<Vec<_>>();
		// The reference answer, so the circuit is checked against the field rather than itself.
		let mut expected = values.clone();
		<B128 as ExtensionField<B1>>::square_transpose(&mut expected);

		let shared = Rc::new(Shared::new());
		let builder = shared.builder();
		let mut fill = Vec::new();
		let mut elems = values
			.iter()
			.map(|&value| {
				let wires = [builder.add_inout(), builder.add_inout()];
				fill.push((wires, value));
				SymbolicElem::wires(&shared, wires[0], wires[1])
			})
			.collect::<Vec<_>>();

		SymbolicElem::square_transpose::<B1>(&mut elems);

		// Each output is pinned to a public claim, so a wrong transpose fails population rather
		// than a Rust comparison.
		for (i, (elem, &want)) in elems.iter().zip(&expected).enumerate() {
			let (lo, hi) = elem.to_wires(builder);
			let claimed = [builder.add_inout(), builder.add_inout()];
			builder.assert_eq_v(format!("transposed[{i}]"), [lo, hi], claimed);
			fill.push((claimed, want));
		}

		let circuit = builder.build();
		let stat = CircuitStat::collect(&circuit);
		let mut w = circuit.new_witness_filler();
		for (wires, value) in fill {
			for (wire, word) in wires.iter().zip(element_words(u128::from(value))) {
				w[*wire] = Word(word);
			}
		}
		circuit
			.populate_wire_witness(&mut w)
			.expect("the transpose must reproduce the native one");

		assert_eq!(stat.n_and_constraints, EXCHANGES);
		assert_eq!(stat.n_bmul_constraints, 0, "a bit transpose needs no field multiplication");
	}
}
