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
	ExtensionField, Field,
	arithmetic_traits::{InvertOrZero, Square},
	field::FieldOps,
};
use binius_spartan_frontend::circuit_builder::CircuitBuilder;

use super::gadgets;

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
	Constant(B::Field),
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

impl<B: CircuitBuilder> CircuitElem<B> {
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
impl<B: CircuitBuilder> Neg for CircuitElem<B> {
	type Output = Self;

	fn neg(self) -> Self {
		self
	}
}

impl<B: CircuitBuilder> Add for CircuitElem<B> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		match (&self, &rhs) {
			(CircuitElem::Constant(a), CircuitElem::Constant(b)) => CircuitElem::Constant(*a + *b),
			_ => {
				if matches!(&self, CircuitElem::Constant(c) if *c == B::Field::ZERO) {
					return rhs;
				}
				if matches!(&rhs, CircuitElem::Constant(c) if *c == B::Field::ZERO) {
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

impl<B: CircuitBuilder> Sub for CircuitElem<B> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		match (&self, &rhs) {
			(CircuitElem::Constant(a), CircuitElem::Constant(b)) => CircuitElem::Constant(*a + *b),
			_ => {
				if matches!(&self, CircuitElem::Constant(c) if *c == B::Field::ZERO) {
					return rhs;
				}
				if matches!(&rhs, CircuitElem::Constant(c) if *c == B::Field::ZERO) {
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

impl<B: CircuitBuilder> Mul for CircuitElem<B> {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		match (&self, &rhs) {
			(CircuitElem::Constant(a), CircuitElem::Constant(b)) => CircuitElem::Constant(*a * *b),
			_ => {
				if matches!(&self, CircuitElem::Constant(c) if *c == B::Field::ZERO) {
					return CircuitElem::Constant(B::Field::ZERO);
				}
				if matches!(&rhs, CircuitElem::Constant(c) if *c == B::Field::ZERO) {
					return CircuitElem::Constant(B::Field::ZERO);
				}
				if matches!(&self, CircuitElem::Constant(c) if *c == B::Field::ONE) {
					return rhs;
				}
				if matches!(&rhs, CircuitElem::Constant(c) if *c == B::Field::ONE) {
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

impl<B: CircuitBuilder> Add<&Self> for CircuitElem<B> {
	type Output = Self;

	fn add(self, rhs: &Self) -> Self {
		self + rhs.clone()
	}
}

impl<B: CircuitBuilder> Sub<&Self> for CircuitElem<B> {
	type Output = Self;

	fn sub(self, rhs: &Self) -> Self {
		self - rhs.clone()
	}
}

impl<B: CircuitBuilder> Mul<&Self> for CircuitElem<B> {
	type Output = Self;

	fn mul(self, rhs: &Self) -> Self {
		self * rhs.clone()
	}
}

// Assign variants — use mem::replace to avoid requiring B: Clone.

impl<B: CircuitBuilder> AddAssign for CircuitElem<B> {
	fn add_assign(&mut self, rhs: Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B::Field::ZERO));
		*self = lhs + rhs;
	}
}

impl<B: CircuitBuilder> SubAssign for CircuitElem<B> {
	fn sub_assign(&mut self, rhs: Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B::Field::ZERO));
		*self = lhs - rhs;
	}
}

impl<B: CircuitBuilder> MulAssign for CircuitElem<B> {
	fn mul_assign(&mut self, rhs: Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B::Field::ZERO));
		*self = lhs * rhs;
	}
}

impl<B: CircuitBuilder> AddAssign<&Self> for CircuitElem<B> {
	fn add_assign(&mut self, rhs: &Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B::Field::ZERO));
		*self = lhs + rhs.clone();
	}
}

impl<B: CircuitBuilder> SubAssign<&Self> for CircuitElem<B> {
	fn sub_assign(&mut self, rhs: &Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B::Field::ZERO));
		*self = lhs - rhs.clone();
	}
}

impl<B: CircuitBuilder> MulAssign<&Self> for CircuitElem<B> {
	fn mul_assign(&mut self, rhs: &Self) {
		let lhs = mem::replace(self, CircuitElem::Constant(B::Field::ZERO));
		*self = lhs * rhs.clone();
	}
}

// Sum and Product

impl<B: CircuitBuilder> Sum for CircuitElem<B> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(B::Field::ZERO), |acc, x| acc + x)
	}
}

impl<'a, B: CircuitBuilder> Sum<&'a CircuitElem<B>> for CircuitElem<B> {
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(B::Field::ZERO), |acc, x| acc + x)
	}
}

impl<B: CircuitBuilder> Product for CircuitElem<B> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(B::Field::ONE), |acc, x| acc * x)
	}
}

impl<'a, B: CircuitBuilder> Product<&'a CircuitElem<B>> for CircuitElem<B> {
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(B::Field::ONE), |acc, x| acc * x)
	}
}

impl<B: CircuitBuilder> Square for CircuitElem<B> {
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

impl<B: CircuitBuilder> InvertOrZero for CircuitElem<B> {
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
				let one = builder.constant(B::Field::ONE);
				builder.assert_eq(product, one);

				Self::make_wire(&rc, inv_wire)
			}
		}
	}
}

impl<B: CircuitBuilder> FieldOps for CircuitElem<B> {
	type Scalar = B::Field;

	fn zero() -> Self {
		CircuitElem::Constant(B::Field::ZERO)
	}

	fn one() -> Self {
		CircuitElem::Constant(B::Field::ONE)
	}

	fn square_transpose<FSub: Field>(elems: &mut [Self])
	where
		Self::Scalar: ExtensionField<FSub>,
	{
		let degree = <B::Field as ExtensionField<FSub>>::DEGREE;
		assert_eq!(elems.len(), degree);

		if degree == 1 {
			return;
		}

		// Fast path: transpose concretely when all elements are constants.
		if elems.iter().all(|e| matches!(e, CircuitElem::Constant(_))) {
			let mut vals = elems
				.iter()
				.map(|e| match e {
					CircuitElem::Constant(c) => *c,
					CircuitElem::Wire(_) => unreachable!(),
				})
				.collect::<Vec<_>>();
			<B::Field as ExtensionField<FSub>>::square_transpose(&mut vals);
			for (e, v) in elems.iter_mut().zip(vals) {
				*e = CircuitElem::Constant(v);
			}
			return;
		}

		// At least one element is a wire. Delegate to the gadget.
		let rc = elems
			.iter()
			.find_map(|e| e.builder_rc())
			.expect("at least one wire exists (not all-constants)");
		let mut builder = rc.borrow_mut();

		let input_wires = elems
			.iter()
			.map(|e| e.to_wire(&mut *builder))
			.collect::<Vec<_>>();

		let outputs = gadgets::square_transpose::<_, FSub>(&mut *builder, &input_wires);

		drop(builder);

		for (e, out_wire) in elems.iter_mut().zip(outputs) {
			*e = Self::make_wire(&rc, out_wire);
		}
	}
}
