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
	marker::PhantomData,
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
	rc::Weak,
};

use binius_field::{
	ExtensionField, Field,
	arithmetic_traits::{InvertOrZero, Square},
	field::FieldOps,
};
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator, WitnessWire},
	constraint_system::ConstraintWire,
};

use super::gadgets;

pub trait CircuitWire<F: Field>: Sized {
	type Builder: CircuitBuilder<Field = F>;

	fn constant(builder: &mut Self::Builder, val: F) -> Self {
		let [ret] = Self::combine(builder, [], |_| [val], |builder, _| [builder.constant(val)]);
		ret
	}

	fn combine<const NIn: usize, const NOut: usize>(
		builder: &mut Self::Builder,
		wires: [&Self; NIn],
		f_op: impl Fn([F; NIn]) -> [F; NOut],
		op: impl Fn(
			&mut Self::Builder,
			[<Self::Builder as CircuitBuilder>::Wire; NIn],
		) -> [<Self::Builder as CircuitBuilder>::Wire; NOut],
	) -> [Self; NOut];
}

#[derive(Debug)]
pub enum BuilderWire<F> {
	Constant(F),
	Wire(ConstraintWire),
}

impl<F: Field> CircuitWire<F> for BuilderWire<F> {
	type Builder = ConstraintBuilder<F>;

	fn combine<const NIn: usize, const NOut: usize>(
		builder: &mut Self::Builder,
		wires: [&Self; NIn],
		f_op: impl Fn([F; NIn]) -> [F; NOut],
		builder_op: impl Fn(&mut Self::Builder, [ConstraintWire; NIn]) -> [ConstraintWire; NOut],
	) -> [Self; NOut] {
		let inner_constants = array_util::try_map(wires, |wire| {
			if let Self::Constant(val) = wire {
				Some(*val)
			} else {
				None
			}
		});

		if let Some(inner_constants) = inner_constants {
			f_op(inner_constants).map(Self::Constant)
		} else {
			let inner_wires = wires.map(|wire| match wire {
				Self::Constant(val) => builder.constant(*val),
				Self::Wire(wire) => *wire,
			});
			builder_op(builder, inner_wires).map(Self::Wire)
		}
	}
}

#[derive(Debug, Clone, Copy)]
pub enum WrappedWire<F> {
	Constant(F),
	Decrypted(F),
	Encrypted,
}

impl<F: Field> CircuitWire<F> for WrappedWire<F> {
	type Builder = NoopBuilder<F>;

	fn combine<const NIn: usize, const NOut: usize>(
		_builder: &mut Self::Builder,
		wires: [&Self; NIn],
		f_op: impl Fn([F; NIn]) -> [F; NOut],
		_builder_op: impl Fn(&mut Self::Builder, [(); NIn]) -> [(); NOut],
	) -> [Self; NOut] {
		let inner_values = array_util::try_map(wires, |wire| match wire {
			Self::Constant(val) | Self::Decrypted(val) => Some(*val),
			Self::Encrypted => None,
		});
		if let Some(inner_values) = inner_values {
			let ret_values = f_op(inner_values);

			let all_constant = wires.iter().all(|wire| matches!(wire, Self::Constant(_)));
			if all_constant {
				// If all inputs are constant, then compute and propagate the constant value.
				ret_values.map(Self::Constant)
			} else {
				// If all inputs are decrypted or constant, then compute and propagate the decrypted
				// value.
				ret_values.map(Self::Decrypted)
			}
		} else {
			// If any inputs are encrypted, all outputs are encrypted.
			[Self::Encrypted; NOut]
		}
	}
}

#[derive(Debug, Default)]
pub struct NoopBuilder<F>(PhantomData<F>);

impl<F: Field> CircuitBuilder for NoopBuilder<F> {
	type Wire = ();
	type Field = F;

	fn assert_zero(&mut self, _wire: Self::Wire) {}

	fn constant(&mut self, _val: Self::Field) -> Self::Wire {
		()
	}

	fn add(&mut self, _lhs: Self::Wire, _rhs: Self::Wire) -> Self::Wire {
		()
	}

	fn mul(&mut self, _lhs: Self::Wire, _rhs: Self::Wire) -> Self::Wire {
		()
	}

	fn hint<H: Fn([Self::Field; IN]) -> [Self::Field; OUT], const IN: usize, const OUT: usize>(
		&mut self,
		_inputs: [Self::Wire; IN],
		_f: H,
	) -> [Self::Wire; OUT] {
		[(); OUT]
	}
}

#[derive(Debug, Clone, Copy)]
pub enum WitnessGenWire<'a, F: Field> {
	Constant(F),
	Wire(WitnessWire<F>, PhantomData<&'a ()>),
}

impl<'a, F: Field> WitnessGenWire<'a, F> {
	pub fn wire(wire: WitnessWire<F>) -> Self {
		Self::Wire(wire, PhantomData)
	}
}

impl<'a, F: Field> CircuitWire<F> for WitnessGenWire<'a, F> {
	type Builder = WitnessGenerator<'a, F>;

	fn combine<const NIn: usize, const NOut: usize>(
		builder: &mut Self::Builder,
		wires: [&Self; NIn],
		f_op: impl Fn([F; NIn]) -> [F; NOut],
		builder_op: impl Fn(&mut Self::Builder, [WitnessWire<F>; NIn]) -> [WitnessWire<F>; NOut],
	) -> [Self; NOut] {
		let inner_constants = array_util::try_map(wires, |wire| {
			if let Self::Constant(val) = wire {
				Some(*val)
			} else {
				None
			}
		});

		if let Some(inner_constants) = inner_constants {
			f_op(inner_constants).map(Self::Constant)
		} else {
			let inner_wires = wires.map(|wire| match wire {
				Self::Constant(val) => builder.constant(*val),
				Self::Wire(wire, _) => *wire,
			});
			builder_op(builder, inner_wires).map(|val| Self::Wire(val, PhantomData))
		}
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
#[derive(Debug, Clone)]
pub enum CircuitElem<F: Field, W: CircuitWire<F>> {
	Constant(F),
	Wire {
		builder: Weak<RefCell<W::Builder>>,
		wire: W,
	},
}

impl<F, W> CircuitElem<F, W>
where
	F: Field,
	W: CircuitWire<F>,
{
	fn combine<const NIn: usize, const NOut: usize>(
		elems: [&Self; NIn],
		f_op: impl Fn([F; NIn]) -> [F; NOut],
		builder_op: impl Fn(
			&mut W::Builder,
			[<W::Builder as CircuitBuilder>::Wire; NIn],
		) -> [<W::Builder as CircuitBuilder>::Wire; NOut],
	) -> [Self; NOut] {
		let builder = elems.iter().find_map(|elem| match elem {
			Self::Wire { builder, .. } => Some(builder),
			_ => None,
		});

		if let Some(builder_ptr) = builder {
			let Some(builder) = builder_ptr.upgrade() else {
				panic!("combine cannot be called on a CircuitElem after the channel is closed");
			};
			let mut builder = builder.borrow_mut();
			let inner_wires = elems.map(|elem| match elem {
				Self::Constant(val) => OwnedOrRef::Owned(W::constant(&mut *builder, *val)),
				Self::Wire {
					builder: other_builder_ptr,
					wire,
				} => {
					assert!(
						Weak::ptr_eq(builder_ptr, other_builder_ptr),
						"all combined CircuitElems must come from the same channel"
					);
					OwnedOrRef::Ref(wire)
				}
			});
			let inner_wire_refs = inner_wires.each_ref().map(AsRef::as_ref);
			W::combine(&mut *builder, inner_wire_refs, f_op, builder_op).map(|wire| Self::Wire {
				builder: builder_ptr.clone(),
				wire,
			})
		} else {
			let inner_constants = elems.map(|elem| {
				let Self::Constant(val) = elem else {
					unreachable!(
						"the enum has only two variants; none of them are Wire; thus all must be Constant"
					);
				};
				*val
			});
			f_op(inner_constants).map(Self::Constant)
		}
	}
}

// In characteristic 2, negation is identity.
// TODO: For the sake of purity, it would be nice for CircuitBuilder to have a neg method
impl<F: Field, W: CircuitWire<F>> Neg for CircuitElem<F, W> {
	type Output = Self;

	fn neg(self) -> Self {
		self
	}
}

impl<F: Field, W: CircuitWire<F>> Add for CircuitElem<F, W> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		self + &rhs
	}
}

impl<F: Field, W: CircuitWire<F>> Sub for CircuitElem<F, W> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		self - &rhs
	}
}

impl<F: Field, W: CircuitWire<F>> Mul for CircuitElem<F, W> {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		self * &rhs
	}
}

// By-reference variants: clone and delegate.

impl<F, W> Add<&Self> for CircuitElem<F, W>
where
	F: Field,
	W: CircuitWire<F>,
{
	type Output = Self;

	fn add(self, rhs: &Self) -> Self {
		&self + rhs
	}
}

impl<F, W> Sub<&Self> for CircuitElem<F, W>
where
	F: Field,
	W: CircuitWire<F>,
{
	type Output = Self;

	fn sub(self, rhs: &Self) -> Self {
		&self - rhs
	}
}

impl<F, W> Mul<&Self> for CircuitElem<F, W>
where
	F: Field,
	W: CircuitWire<F>,
{
	type Output = Self;

	fn mul(self, rhs: &Self) -> Self {
		&self * rhs
	}
}

impl<F, W> Add for &CircuitElem<F, W>
where
	F: Field,
	W: CircuitWire<F>,
{
	type Output = CircuitElem<F, W>;

	fn add(self, rhs: Self) -> Self::Output {
		let [ret] = CircuitElem::combine(
			[self, rhs],
			|[lhs, rhs]| [lhs + rhs],
			|builder, [lhs, rhs]| [builder.add(lhs, rhs)],
		);
		ret
	}
}

impl<F, W> Sub for &CircuitElem<F, W>
where
	F: Field,
	W: CircuitWire<F>,
{
	type Output = CircuitElem<F, W>;

	fn sub(self, rhs: Self) -> Self::Output {
		let [ret] = CircuitElem::combine(
			[self, rhs],
			|[lhs, rhs]| [lhs - rhs],
			|builder, [lhs, rhs]| [builder.sub(lhs, rhs)],
		);
		ret
	}
}

impl<F, W> Mul for &CircuitElem<F, W>
where
	F: Field,
	W: CircuitWire<F>,
{
	type Output = CircuitElem<F, W>;

	fn mul(self, rhs: Self) -> Self::Output {
		let [ret] = CircuitElem::combine(
			[self, rhs],
			|[lhs, rhs]| [lhs * rhs],
			|builder, [lhs, rhs]| [builder.mul(lhs, rhs)],
		);
		ret
	}
}

// Assign variants — use mem::replace to avoid requiring B: Clone.

impl<F: Field, W: CircuitWire<F>> AddAssign for CircuitElem<F, W> {
	fn add_assign(&mut self, rhs: Self) {
		*self = &*self + &rhs;
	}
}

impl<F: Field, W: CircuitWire<F>> SubAssign for CircuitElem<F, W> {
	fn sub_assign(&mut self, rhs: Self) {
		*self = &*self - &rhs;
	}
}

impl<F: Field, W: CircuitWire<F>> MulAssign for CircuitElem<F, W> {
	fn mul_assign(&mut self, rhs: Self) {
		*self = &*self * &rhs;
	}
}

impl<F: Field, W: CircuitWire<F>> AddAssign<&Self> for CircuitElem<F, W> {
	fn add_assign(&mut self, rhs: &Self) {
		*self = &*self + rhs;
	}
}

impl<F: Field, W: CircuitWire<F>> SubAssign<&Self> for CircuitElem<F, W> {
	fn sub_assign(&mut self, rhs: &Self) {
		*self = &*self - rhs;
	}
}

impl<F: Field, W: CircuitWire<F>> MulAssign<&Self> for CircuitElem<F, W> {
	fn mul_assign(&mut self, rhs: &Self) {
		*self = &*self * rhs;
	}
}

// Sum and Product

impl<F: Field, W: CircuitWire<F>> Sum for CircuitElem<F, W> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(F::ZERO), |acc, x| acc + x)
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

impl<F: Field, B: CircuitBuilder<Field = F>> From<F> for CircuitElem<B> {
	fn from(val: F) -> Self {
		CircuitElem::Constant(val)
	}
}

#[derive(Debug)]
enum OwnedOrRef<'a, T> {
	Owned(T),
	Ref(&'a T),
}

impl<'a, T> AsRef<T> for OwnedOrRef<'a, T> {
	fn as_ref(&self) -> &T {
		match self {
			Self::Owned(ret) => ret,
			Self::Ref(ret) => ret,
		}
	}
}
