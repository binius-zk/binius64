// Copyright 2026 The Binius Developers

//! Generic field element over a pluggable [`CircuitBuilder`] backend.
//!
//! [`CircuitElem<F, B>`] is a field element that is either a known `Constant` or a `Wire` in a
//! [`CircuitBuilder`] `B`. The arithmetic-trait impls on [`CircuitElem`] are written once and
//! reused across all backends; each operation either folds constants at the `F` level or delegates
//! to the builder's `add`/`mul`/`hint`/… on the wire type `B::Wire`. The backends are the frontend
//! builders themselves:
//!
//! - [`ConstraintBuilder`] — symbolic constraint recording (used by
//!   [`IronSpartanBuilderChannel`](super::builder_channel::IronSpartanBuilderChannel)).
//! - [`InstanceGenerator`] — reconstructs the public-input vector during verification (used by
//!   [`ZKWrappedVerifierChannel`](super::zk_wrapped_channel::ZKWrappedVerifierChannel)).
//! - [`WitnessGenerator`] — concrete evaluation that fills a witness (used by
//!   `binius_spartan_prover::wrapper::ReplayChannel`).
//!
//! [`CircuitBuilder`]: binius_spartan_frontend::circuit_builder::CircuitBuilder
//! [`ConstraintBuilder`]: binius_spartan_frontend::circuit_builder::ConstraintBuilder
//! [`InstanceGenerator`]: binius_spartan_frontend::circuit_builder::InstanceGenerator
//! [`WitnessGenerator`]: binius_spartan_frontend::circuit_builder::WitnessGenerator

use std::{
	cell::RefCell,
	fmt,
	iter::{Product, Sum},
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

/// A field element that is either a known constant or a wire in a [`CircuitBuilder`] `B`.
///
/// The `Wire` variant holds a [`Weak`] reference to the shared builder; it must outlive any
/// operation performed on the element. Arithmetic over all-`Constant` operands folds at the `F`
/// level without touching a builder; any `Wire` operand routes the operation through `B`.
pub enum CircuitElem<F: Field, B: CircuitBuilder<Field = F>> {
	Constant(F),
	Wire {
		builder: Weak<RefCell<B>>,
		wire: B::Wire,
	},
}

// Manual `Debug`: the derived impl would bound the type parameter `B: Debug`, but the field is the
// associated type `B::Wire`, so we bound that instead.
impl<F: Field, B: CircuitBuilder<Field = F>> fmt::Debug for CircuitElem<F, B>
where
	B::Wire: fmt::Debug,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Constant(c) => f.debug_tuple("Constant").field(c).finish(),
			Self::Wire { wire, .. } => f.debug_struct("Wire").field("wire", wire).finish(),
		}
	}
}

// Manual `Clone` that does not require `B: Clone` (the derived impl would, even though
// `Weak<T>: Clone` for any `T` and `B::Wire: Copy`).
impl<F: Field, B: CircuitBuilder<Field = F>> Clone for CircuitElem<F, B> {
	fn clone(&self) -> Self {
		match self {
			Self::Constant(c) => Self::Constant(*c),
			Self::Wire { builder, wire } => Self::Wire {
				builder: builder.clone(),
				wire: *wire,
			},
		}
	}
}

impl<F, B> CircuitElem<F, B>
where
	F: Field,
	B: CircuitBuilder<Field = F>,
{
	/// Construct a [`Self::Wire`] anchored to a shared builder via a [`Weak`] reference.
	pub fn wire(builder: &Rc<RefCell<B>>, wire: B::Wire) -> Self {
		Self::Wire {
			builder: Rc::downgrade(builder),
			wire,
		}
	}

	/// Lowers this element to a wire on `builder`, materializing a `Constant` via
	/// [`CircuitBuilder::constant`]. A `Wire`'s backing builder is assumed to be `builder`; callers
	/// mixing elements from different channels must check that themselves (as [`Self::combine`]
	/// does).
	pub fn to_wire(&self, builder: &mut B) -> B::Wire {
		match self {
			Self::Constant(val) => builder.constant(*val),
			Self::Wire { wire, .. } => *wire,
		}
	}

	/// Combine `elems` under an operation. If every input is a `Constant`, fold at the `F` level
	/// via `f_op` (no builder is touched). Otherwise convert constants to wires on the shared
	/// builder and run `builder_op` over the wires.
	pub fn combine<const IN: usize, const OUT: usize>(
		elems: [&Self; IN],
		f_op: impl Fn([F; IN]) -> [F; OUT],
		builder_op: impl Fn(&mut B, [B::Wire; IN]) -> [B::Wire; OUT],
	) -> [Self; OUT] {
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
				Self::Constant(val) => builder.constant(*val),
				Self::Wire {
					builder: other_builder_ptr,
					wire,
				} => {
					assert!(
						Weak::ptr_eq(builder_ptr, other_builder_ptr),
						"all combined CircuitElems must come from the same channel"
					);
					*wire
				}
			});
			builder_op(&mut builder, inner_wires).map(|wire| Self::Wire {
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

	/// Variable-arity sibling of [`Self::combine`].
	///
	/// `f_op` and `builder_op` must return a `Vec` of length `n_out`; checked via
	/// `debug_assert_eq!`.
	pub fn combine_varlen(
		elems: &[&Self],
		n_out: usize,
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		builder_op: impl FnOnce(&mut B, &[B::Wire]) -> Vec<B::Wire>,
	) -> Vec<Self> {
		let builder = elems.iter().find_map(|elem| match elem {
			Self::Wire { builder, .. } => Some(builder),
			_ => None,
		});

		if let Some(builder_ptr) = builder {
			let Some(builder) = builder_ptr.upgrade() else {
				panic!(
					"combine_varlen cannot be called on a CircuitElem after the channel is closed"
				);
			};
			let mut builder = builder.borrow_mut();
			let inner_wires = elems
				.iter()
				.map(|elem| match elem {
					Self::Constant(val) => builder.constant(*val),
					Self::Wire {
						builder: other_builder_ptr,
						wire,
					} => {
						assert!(
							Weak::ptr_eq(builder_ptr, other_builder_ptr),
							"all combined CircuitElems must come from the same channel"
						);
						*wire
					}
				})
				.collect::<Vec<_>>();
			let result = builder_op(&mut builder, &inner_wires);
			debug_assert_eq!(result.len(), n_out);
			result
				.into_iter()
				.map(|wire| Self::Wire {
					builder: builder_ptr.clone(),
					wire,
				})
				.collect()
		} else {
			let inner_constants = elems
				.iter()
				.map(|elem| {
					let Self::Constant(val) = elem else {
						unreachable!(
							"no Wire variant exists in elems; all entries must be Constant"
						);
					};
					*val
				})
				.collect::<Vec<_>>();
			let result = f_op(&inner_constants);
			debug_assert_eq!(result.len(), n_out);
			result.into_iter().map(Self::Constant).collect()
		}
	}
}

// In characteristic 2, negation is identity.
// TODO: For the sake of purity, it would be nice for CircuitBuilder to have a neg method
impl<F: Field, B: CircuitBuilder<Field = F>> Neg for CircuitElem<F, B> {
	type Output = Self;

	fn neg(self) -> Self {
		self
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> Add for CircuitElem<F, B> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		self + &rhs
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> Sub for CircuitElem<F, B> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		self - &rhs
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> Mul for CircuitElem<F, B> {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		self * &rhs
	}
}

// By-reference variants: clone and delegate.

impl<F, B> Add<&Self> for CircuitElem<F, B>
where
	F: Field,
	B: CircuitBuilder<Field = F>,
{
	type Output = Self;

	fn add(self, rhs: &Self) -> Self {
		&self + rhs
	}
}

impl<F, B> Sub<&Self> for CircuitElem<F, B>
where
	F: Field,
	B: CircuitBuilder<Field = F>,
{
	type Output = Self;

	fn sub(self, rhs: &Self) -> Self {
		&self - rhs
	}
}

impl<F, B> Mul<&Self> for CircuitElem<F, B>
where
	F: Field,
	B: CircuitBuilder<Field = F>,
{
	type Output = Self;

	fn mul(self, rhs: &Self) -> Self {
		&self * rhs
	}
}

impl<F, B> Add for &CircuitElem<F, B>
where
	F: Field,
	B: CircuitBuilder<Field = F>,
{
	type Output = CircuitElem<F, B>;

	fn add(self, rhs: Self) -> Self::Output {
		let [ret] = CircuitElem::combine(
			[self, rhs],
			|[lhs, rhs]| [lhs + rhs],
			|builder, [lhs, rhs]| [builder.add(lhs, rhs)],
		);
		ret
	}
}

impl<F, B> Sub for &CircuitElem<F, B>
where
	F: Field,
	B: CircuitBuilder<Field = F>,
{
	type Output = CircuitElem<F, B>;

	fn sub(self, rhs: Self) -> Self::Output {
		let [ret] = CircuitElem::combine(
			[self, rhs],
			|[lhs, rhs]| [lhs - rhs],
			|builder, [lhs, rhs]| [builder.sub(lhs, rhs)],
		);
		ret
	}
}

impl<F, B> Mul for &CircuitElem<F, B>
where
	F: Field,
	B: CircuitBuilder<Field = F>,
{
	type Output = CircuitElem<F, B>;

	fn mul(self, rhs: Self) -> Self::Output {
		// Short-circuit `wire * 0 = 0` so the wrapper does not allocate a multiplication
		// constraint that pins a wire to zero.
		if matches!(self, CircuitElem::Constant(c) if *c == F::ZERO)
			|| matches!(rhs, CircuitElem::Constant(c) if *c == F::ZERO)
		{
			return CircuitElem::Constant(F::ZERO);
		}
		let [ret] = CircuitElem::combine(
			[self, rhs],
			|[lhs, rhs]| [lhs * rhs],
			|builder, [lhs, rhs]| [builder.mul(lhs, rhs)],
		);
		ret
	}
}

// Assign variants.

impl<F: Field, B: CircuitBuilder<Field = F>> AddAssign for CircuitElem<F, B> {
	fn add_assign(&mut self, rhs: Self) {
		*self = &*self + &rhs;
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> SubAssign for CircuitElem<F, B> {
	fn sub_assign(&mut self, rhs: Self) {
		*self = &*self - &rhs;
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> MulAssign for CircuitElem<F, B> {
	fn mul_assign(&mut self, rhs: Self) {
		*self = &*self * &rhs;
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> AddAssign<&Self> for CircuitElem<F, B> {
	fn add_assign(&mut self, rhs: &Self) {
		*self = &*self + rhs;
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> SubAssign<&Self> for CircuitElem<F, B> {
	fn sub_assign(&mut self, rhs: &Self) {
		*self = &*self - rhs;
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> MulAssign<&Self> for CircuitElem<F, B> {
	fn mul_assign(&mut self, rhs: &Self) {
		*self = &*self * rhs;
	}
}

// Sum and Product

impl<F: Field, B: CircuitBuilder<Field = F>> Sum for CircuitElem<F, B> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(CircuitElem::Constant(F::ZERO), |acc, x| acc + x)
	}
}

impl<'a, F: Field, B: CircuitBuilder<Field = F>> Sum<&'a Self> for CircuitElem<F, B> {
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(F::ZERO), |acc, x| acc + x)
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> Product for CircuitElem<F, B> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(F::ONE), |acc, x| acc * x)
	}
}

impl<'a, F: Field, B: CircuitBuilder<Field = F>> Product<&'a Self> for CircuitElem<F, B> {
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(F::ONE), |acc, x| acc * x)
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> Square for CircuitElem<F, B> {
	fn square(self) -> Self {
		let [ret] = Self::combine([&self], |[x]| [x.square()], |builder, [x]| [builder.mul(x, x)]);
		ret
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> InvertOrZero for CircuitElem<F, B> {
	fn invert_or_zero(self) -> Self {
		let [ret] = Self::combine(
			[&self],
			|[x]| [x.invert_or_zero()],
			|builder, [x]| {
				let [inv] = builder.hint([x], |[v]| [v.invert_or_zero()]);
				let one = builder.constant(F::ONE);
				let product = builder.mul(x, inv);
				builder.assert_eq(product, one);
				[inv]
			},
		);
		ret
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> FieldOps for CircuitElem<F, B> {
	type Scalar = F;

	fn zero() -> Self {
		Self::Constant(F::ZERO)
	}

	fn one() -> Self {
		Self::Constant(F::ONE)
	}

	fn square_transpose<FSub: Field>(elems: &mut [Self])
	where
		Self::Scalar: ExtensionField<FSub>,
	{
		let degree = <F as ExtensionField<FSub>>::DEGREE;
		assert_eq!(elems.len(), degree);

		if degree == 1 {
			return;
		}

		let inputs = elems.iter().collect::<Vec<_>>();
		let outputs = Self::combine_varlen(
			&inputs,
			degree,
			|vals| {
				let mut out = vals.to_vec();
				<F as ExtensionField<FSub>>::square_transpose(&mut out);
				out
			},
			|builder, wires| gadgets::square_transpose::<_, FSub>(builder, wires),
		);
		for (e, out) in elems.iter_mut().zip(outputs) {
			*e = out;
		}
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> From<F> for CircuitElem<F, B> {
	fn from(val: F) -> Self {
		Self::Constant(val)
	}
}
