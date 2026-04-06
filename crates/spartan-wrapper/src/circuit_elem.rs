// Copyright 2026 The Binius Developers

//! Generic field element types for building constraint systems or generating witnesses.
//!
//! [`CircuitElem<B>`] and [`CircuitWire<B>`] are parameterized over a [`CircuitBuilder`] backend,
//! allowing the same arithmetic to drive symbolic constraint recording ([`ConstraintBuilder`]) or
//! concrete witness evaluation ([`WitnessGenerator`]).
//!
//! [`CircuitBuilder`]: binius_spartan_frontend::circuit_builder::CircuitBuilder
//! [`ConstraintBuilder`]: binius_spartan_frontend::circuit_builder::ConstraintBuilder
//! [`WitnessGenerator`]: binius_spartan_frontend::circuit_builder::WitnessGenerator

use std::{
	cell::RefCell,
	iter::{Product, Sum},
	mem,
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
	rc::{Rc, Weak},
};

use binius_field::{
	BinaryField128bGhash as B128, ExtensionField, Field,
	arithmetic_traits::{InvertOrZero, Square},
	field::FieldOps,
};
use binius_spartan_frontend::circuit_builder::CircuitBuilder;

/// An opaque wire in a circuit builder, carrying a weak reference to the builder.
///
/// The weak reference allows the channel to reclaim sole ownership of the builder in `finish()`
/// via `Rc::try_unwrap`, even while `CircuitWire` values are still alive.
pub struct CircuitWire<B: CircuitBuilder> {
	builder: Weak<RefCell<B>>,
	wire: B::Wire,
}

impl<B: CircuitBuilder> Clone for CircuitWire<B> {
	fn clone(&self) -> Self {
		Self {
			builder: self.builder.clone(),
			wire: self.wire,
		}
	}
}

impl<B: CircuitBuilder> CircuitWire<B> {
	pub(crate) fn new(builder: &Rc<RefCell<B>>, wire: B::Wire) -> Self {
		Self {
			builder: Rc::downgrade(builder),
			wire,
		}
	}

	pub(crate) fn wire(&self) -> B::Wire {
		self.wire
	}

	fn upgrade(&self) -> Rc<RefCell<B>> {
		self.builder
			.upgrade()
			.expect("channel has been consumed by finish()")
	}
}

/// A field element that is either a known constant or a wire in a circuit builder.
///
/// When the builder is a [`ConstraintBuilder`], arithmetic on wires records constraints
/// symbolically. When the builder is a [`WitnessGenerator`], arithmetic computes concrete values
/// and populates the witness.
///
/// [`ConstraintBuilder`]: binius_spartan_frontend::circuit_builder::ConstraintBuilder
/// [`WitnessGenerator`]: binius_spartan_frontend::circuit_builder::WitnessGenerator
pub enum CircuitElem<B: CircuitBuilder> {
	Constant(B128),
	Wire(CircuitWire<B>),
}

impl<B: CircuitBuilder> Clone for CircuitElem<B> {
	fn clone(&self) -> Self {
		match self {
			CircuitElem::Constant(c) => CircuitElem::Constant(*c),
			CircuitElem::Wire(w) => CircuitElem::Wire(w.clone()),
		}
	}
}

impl<B: CircuitBuilder<Field = B128>> CircuitElem<B> {
	/// Returns the builder Rc if this is a Wire variant.
	fn builder_rc(&self) -> Option<Rc<RefCell<B>>> {
		match self {
			CircuitElem::Constant(_) => None,
			CircuitElem::Wire(w) => Some(w.upgrade()),
		}
	}

	/// Given two CircuitElems, return the builder that at least one of them references.
	fn resolve_builder(a: &Self, b: &Self) -> Rc<RefCell<B>> {
		match (a.builder_rc(), b.builder_rc()) {
			(Some(a), Some(b)) => {
				assert!(Rc::ptr_eq(&a, &b), "CircuitElem wires reference different builders");
				b
			}
			(Some(b), None) | (None, Some(b)) => b,
			(None, None) => panic!("cannot resolve builder: both operands are constants"),
		}
	}

	/// Convert this element to a wire, allocating a constant wire if necessary.
	fn to_wire(&self, builder: &mut B) -> B::Wire {
		match self {
			CircuitElem::Constant(val) => builder.constant(*val),
			CircuitElem::Wire(w) => w.wire,
		}
	}

	fn make_wire(rc: &Rc<RefCell<B>>, wire: B::Wire) -> Self {
		CircuitElem::Wire(CircuitWire {
			builder: Rc::downgrade(rc),
			wire,
		})
	}
}

// In characteristic 2, negation is identity.
impl<B: CircuitBuilder<Field = B128>> Neg for CircuitElem<B> {
	type Output = Self;

	fn neg(self) -> Self {
		self
	}
}

impl<B: CircuitBuilder<Field = B128>> Add for CircuitElem<B> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		match (&self, &rhs) {
			(CircuitElem::Constant(a), CircuitElem::Constant(b)) => CircuitElem::Constant(*a + *b),
			_ => {
				if matches!(&self, CircuitElem::Constant(c) if *c == B128::ZERO) {
					return rhs;
				}
				if matches!(&rhs, CircuitElem::Constant(c) if *c == B128::ZERO) {
					return self;
				}
				let rc = Self::resolve_builder(&self, &rhs);
				let mut builder = rc.borrow_mut();
				let a_wire = self.to_wire(&mut builder);
				let b_wire = rhs.to_wire(&mut builder);
				let out = builder.add(a_wire, b_wire);
				Self::make_wire(&rc, out)
			}
		}
	}
}

impl<B: CircuitBuilder<Field = B128>> Sub for CircuitElem<B> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		match (&self, &rhs) {
			(CircuitElem::Constant(a), CircuitElem::Constant(b)) => CircuitElem::Constant(*a + *b),
			_ => {
				if matches!(&self, CircuitElem::Constant(c) if *c == B128::ZERO) {
					return rhs;
				}
				if matches!(&rhs, CircuitElem::Constant(c) if *c == B128::ZERO) {
					return self;
				}
				let rc = Self::resolve_builder(&self, &rhs);
				let mut builder = rc.borrow_mut();
				let a_wire = self.to_wire(&mut builder);
				let b_wire = rhs.to_wire(&mut builder);
				let out = builder.sub(a_wire, b_wire);
				Self::make_wire(&rc, out)
			}
		}
	}
}

impl<B: CircuitBuilder<Field = B128>> Mul for CircuitElem<B> {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		match (&self, &rhs) {
			(CircuitElem::Constant(a), CircuitElem::Constant(b)) => CircuitElem::Constant(*a * *b),
			_ => {
				if matches!(&self, CircuitElem::Constant(c) if *c == B128::ZERO) {
					return CircuitElem::Constant(B128::ZERO);
				}
				if matches!(&rhs, CircuitElem::Constant(c) if *c == B128::ZERO) {
					return CircuitElem::Constant(B128::ZERO);
				}
				if matches!(&self, CircuitElem::Constant(c) if *c == B128::ONE) {
					return rhs;
				}
				if matches!(&rhs, CircuitElem::Constant(c) if *c == B128::ONE) {
					return self;
				}
				let rc = Self::resolve_builder(&self, &rhs);
				let mut builder = rc.borrow_mut();
				let a_wire = self.to_wire(&mut builder);
				let b_wire = rhs.to_wire(&mut builder);
				let out = builder.mul(a_wire, b_wire);
				Self::make_wire(&rc, out)
			}
		}
	}
}

// By-reference variants: clone and delegate.

impl<B: CircuitBuilder<Field = B128>> Add<&Self> for CircuitElem<B> {
	type Output = Self;

	fn add(self, rhs: &Self) -> Self {
		self + rhs.clone()
	}
}

impl<B: CircuitBuilder<Field = B128>> Sub<&Self> for CircuitElem<B> {
	type Output = Self;

	fn sub(self, rhs: &Self) -> Self {
		self - rhs.clone()
	}
}

impl<B: CircuitBuilder<Field = B128>> Mul<&Self> for CircuitElem<B> {
	type Output = Self;

	fn mul(self, rhs: &Self) -> Self {
		self * rhs.clone()
	}
}

// Assign variants — use mem::replace to avoid requiring B: Clone.

impl<B: CircuitBuilder<Field = B128>> AddAssign for CircuitElem<B> {
	fn add_assign(&mut self, rhs: Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B128::ZERO));
		*self = lhs + rhs;
	}
}

impl<B: CircuitBuilder<Field = B128>> SubAssign for CircuitElem<B> {
	fn sub_assign(&mut self, rhs: Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B128::ZERO));
		*self = lhs - rhs;
	}
}

impl<B: CircuitBuilder<Field = B128>> MulAssign for CircuitElem<B> {
	fn mul_assign(&mut self, rhs: Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B128::ZERO));
		*self = lhs * rhs;
	}
}

impl<B: CircuitBuilder<Field = B128>> AddAssign<&Self> for CircuitElem<B> {
	fn add_assign(&mut self, rhs: &Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B128::ZERO));
		*self = lhs + rhs.clone();
	}
}

impl<B: CircuitBuilder<Field = B128>> SubAssign<&Self> for CircuitElem<B> {
	fn sub_assign(&mut self, rhs: &Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B128::ZERO));
		*self = lhs - rhs.clone();
	}
}

impl<B: CircuitBuilder<Field = B128>> MulAssign<&Self> for CircuitElem<B> {
	fn mul_assign(&mut self, rhs: &Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B128::ZERO));
		*self = lhs * rhs.clone();
	}
}

// Sum and Product

impl<B: CircuitBuilder<Field = B128>> Sum for CircuitElem<B> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(B128::ZERO), |acc, x| acc + x)
	}
}

impl<'a, B: CircuitBuilder<Field = B128>> Sum<&'a CircuitElem<B>> for CircuitElem<B> {
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(B128::ZERO), |acc, x| acc + x)
	}
}

impl<B: CircuitBuilder<Field = B128>> Product for CircuitElem<B> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(B128::ONE), |acc, x| acc * x)
	}
}

impl<'a, B: CircuitBuilder<Field = B128>> Product<&'a CircuitElem<B>> for CircuitElem<B> {
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(B128::ONE), |acc, x| acc * x)
	}
}

impl<B: CircuitBuilder<Field = B128>> Square for CircuitElem<B> {
	fn square(self) -> Self {
		match &self {
			CircuitElem::Constant(c) => CircuitElem::Constant(c.square()),
			_ => {
				let copy = self.clone();
				self * copy
			}
		}
	}
}

impl<B: CircuitBuilder<Field = B128>> InvertOrZero for CircuitElem<B> {
	fn invert_or_zero(self) -> Self {
		match &self {
			CircuitElem::Constant(c) => CircuitElem::Constant(c.invert_or_zero()),
			CircuitElem::Wire(w) => {
				let rc = w.upgrade();
				let mut builder = rc.borrow_mut();
				let wire = w.wire;

				// Allocate the inverse wire via hint.
				let [inv_wire] = builder.hint([wire], |[x]| [x.invert_or_zero()]);

				// Constrain wire * inverse = one.
				let product = builder.mul(wire, inv_wire);
				let one = builder.constant(B128::ONE);
				builder.assert_eq(product, one);

				Self::make_wire(&rc, inv_wire)
			}
		}
	}
}

impl<B: CircuitBuilder<Field = B128>> FieldOps for CircuitElem<B> {
	type Scalar = B128;

	fn zero() -> Self {
		CircuitElem::Constant(B128::ZERO)
	}

	fn one() -> Self {
		CircuitElem::Constant(B128::ONE)
	}

	fn square_transpose<FSub: Field>(_elems: &mut [Self])
	where
		Self::Scalar: ExtensionField<FSub>,
	{
		unimplemented!("square_transpose is not used in symbolic verifier context")
	}
}
