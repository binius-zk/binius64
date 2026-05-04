// Copyright 2026 The Binius Developers

//! Generic field element types over a pluggable circuit-wire backend.
//!
//! [`CircuitElem<F, W>`] is a field element parameterized by a [`CircuitWire<F>`] impl. The
//! [`CircuitWire`] trait hides backend-specific wire-combination logic behind a single `combine`
//! operation, so the arithmetic trait impls on [`CircuitElem`] are written once and reused across
//! three backends:
//!
//! - [`BuilderWire`] over [`ConstraintBuilder`] — symbolic constraint recording (used by
//!   [`IronSpartanBuilderChannel`]).
//! - [`WitnessGenWire`] over [`WitnessGenerator`] — concrete evaluation that fills a witness
//!   (used by [`ReplayChannel`]).
//! - [`WrappedWire`] over [`NoopBuilder`] — no constraint recording; values are tracked as
//!   `Constant` / `Decrypted` / `Encrypted` to distinguish what the verifier does and does not
//!   know concretely (used by [`ZKWrappedVerifierChannel`]).
//!
//! [`CircuitBuilder`]: binius_spartan_frontend::circuit_builder::CircuitBuilder
//! [`ConstraintBuilder`]: binius_spartan_frontend::circuit_builder::ConstraintBuilder
//! [`WitnessGenerator`]: binius_spartan_frontend::circuit_builder::WitnessGenerator
//! [`IronSpartanBuilderChannel`]: super::channel::IronSpartanBuilderChannel
//! [`ReplayChannel`]: super::channel::ReplayChannel
//! [`ZKWrappedVerifierChannel`]: super::zk_wrapped_channel::ZKWrappedVerifierChannel

use std::{
	cell::RefCell,
	iter::{Product, Sum},
	marker::PhantomData,
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
	rc::{Rc, Weak},
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

/// Backend-specific logic for combining wires under arithmetic operations.
///
/// One method (`combine`, plus its variable-arity sibling `combine_varlen`) suffices for the
/// arithmetic-trait impls on [`CircuitElem`]. Each impl decides whether to short-circuit constant
/// inputs into a concrete value or to delegate to its underlying [`CircuitBuilder`].
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

	/// Variable-arity version of [`Self::combine`].
	///
	/// Used by operations whose input/output sizes are determined at runtime (e.g. delegating to
	/// a `&[B::Wire]`-taking gadget like `square_transpose`).
	fn combine_varlen(
		builder: &mut Self::Builder,
		wires: &[&Self],
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		op: impl FnOnce(
			&mut Self::Builder,
			&[<Self::Builder as CircuitBuilder>::Wire],
		) -> Vec<<Self::Builder as CircuitBuilder>::Wire>,
	) -> Vec<Self>;
}

/// [`CircuitWire`] backend over [`ConstraintBuilder`] — used by [`IronSpartanBuilderChannel`] to
/// record arithmetic as constraints in a constraint system.
///
/// [`IronSpartanBuilderChannel`]: super::channel::IronSpartanBuilderChannel
#[derive(Debug, Clone, Copy)]
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

	fn combine_varlen(
		builder: &mut Self::Builder,
		wires: &[&Self],
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		builder_op: impl FnOnce(&mut Self::Builder, &[ConstraintWire]) -> Vec<ConstraintWire>,
	) -> Vec<Self> {
		let inner_constants = wires
			.iter()
			.map(|wire| match wire {
				Self::Constant(val) => Some(*val),
				Self::Wire(_) => None,
			})
			.collect::<Option<Vec<_>>>();

		if let Some(inner_constants) = inner_constants {
			f_op(&inner_constants)
				.into_iter()
				.map(Self::Constant)
				.collect()
		} else {
			let inner_wires = wires
				.iter()
				.map(|wire| match wire {
					Self::Constant(val) => builder.constant(*val),
					Self::Wire(wire) => *wire,
				})
				.collect::<Vec<_>>();
			builder_op(builder, &inner_wires)
				.into_iter()
				.map(Self::Wire)
				.collect()
		}
	}
}

/// [`CircuitWire`] backend used by [`ZKWrappedVerifierChannel`].
///
/// The wrapped channel records no constraints of its own — its [`Self::Builder`] is the no-op
/// [`NoopBuilder`]. The variants instead distinguish what the verifier knows about each value:
///
/// - `Constant` — known at compile time.
/// - `Decrypted` — known at runtime (a sampled challenge or a value derived only from sampled
///   challenges and constants).
/// - `Encrypted` — flows through the inner channel as ciphertext; the F value is not exposed to
///   circuit logic.
///
/// `combine` propagates concretely known values when possible and produces `Encrypted` whenever
/// any input is `Encrypted`.
///
/// [`ZKWrappedVerifierChannel`]: super::zk_wrapped_channel::ZKWrappedVerifierChannel
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

	fn combine_varlen(
		builder: &mut Self::Builder,
		wires: &[&Self],
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		builder_op: impl FnOnce(&mut Self::Builder, &[()]) -> Vec<()>,
	) -> Vec<Self> {
		let inner_values = wires
			.iter()
			.map(|wire| match wire {
				Self::Constant(val) | Self::Decrypted(val) => Some(*val),
				Self::Encrypted => None,
			})
			.collect::<Option<Vec<_>>>();
		if let Some(inner_values) = inner_values {
			let ret_values = f_op(&inner_values);
			let all_constant = wires.iter().all(|wire| matches!(wire, Self::Constant(_)));
			if all_constant {
				ret_values.into_iter().map(Self::Constant).collect()
			} else {
				ret_values.into_iter().map(Self::Decrypted).collect()
			}
		} else {
			// Run the no-op builder_op to discover the output arity, then mark all as encrypted.
			let inner_wires = vec![(); wires.len()];
			builder_op(builder, &inner_wires)
				.into_iter()
				.map(|()| Self::Encrypted)
				.collect()
		}
	}
}

/// Trivial [`CircuitBuilder`] with `Wire = ()`. Provides a target for the `Weak<RefCell<…>>`
/// references stored in [`CircuitElem::Wire`] when the channel doesn't need to record real
/// constraints (i.e. [`ZKWrappedVerifierChannel`]).
///
/// [`ZKWrappedVerifierChannel`]: super::zk_wrapped_channel::ZKWrappedVerifierChannel
#[derive(Debug, Default)]
pub struct NoopBuilder<F>(PhantomData<F>);

impl<F: Field> CircuitBuilder for NoopBuilder<F> {
	type Wire = ();
	type Field = F;

	fn assert_zero(&mut self, _wire: Self::Wire) {}

	fn constant(&mut self, _val: Self::Field) -> Self::Wire {}

	fn add(&mut self, _lhs: Self::Wire, _rhs: Self::Wire) -> Self::Wire {}

	fn mul(&mut self, _lhs: Self::Wire, _rhs: Self::Wire) -> Self::Wire {}

	fn hint<H: Fn([Self::Field; IN]) -> [Self::Field; OUT], const IN: usize, const OUT: usize>(
		&mut self,
		_inputs: [Self::Wire; IN],
		_f: H,
	) -> [Self::Wire; OUT] {
		[(); OUT]
	}
}

/// [`CircuitWire`] backend over [`WitnessGenerator`] — used by [`ReplayChannel`] to evaluate
/// arithmetic concretely while filling private witness wires.
///
/// [`ReplayChannel`]: super::channel::ReplayChannel
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

	fn combine_varlen(
		builder: &mut Self::Builder,
		wires: &[&Self],
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		builder_op: impl FnOnce(&mut Self::Builder, &[WitnessWire<F>]) -> Vec<WitnessWire<F>>,
	) -> Vec<Self> {
		let inner_constants = wires
			.iter()
			.map(|wire| match wire {
				Self::Constant(val) => Some(*val),
				Self::Wire(_, _) => None,
			})
			.collect::<Option<Vec<_>>>();

		if let Some(inner_constants) = inner_constants {
			f_op(&inner_constants)
				.into_iter()
				.map(Self::Constant)
				.collect()
		} else {
			let inner_wires = wires
				.iter()
				.map(|wire| match wire {
					Self::Constant(val) => builder.constant(*val),
					Self::Wire(wire, _) => *wire,
				})
				.collect::<Vec<_>>();
			builder_op(builder, &inner_wires)
				.into_iter()
				.map(|val| Self::Wire(val, PhantomData))
				.collect()
		}
	}
}

/// A field element that is either a known constant or a wire in a [`CircuitBuilder`].
///
/// The behaviour of arithmetic on `Wire` values is determined by the `W: CircuitWire<F>` impl —
/// see the module docs for the available backends. The `Wire` variant holds a [`Weak`] reference
/// to the shared builder; it must outlive any operation performed on the element.
#[derive(Debug)]
pub enum CircuitElem<F: Field, W: CircuitWire<F>> {
	Constant(F),
	Wire {
		builder: Weak<RefCell<W::Builder>>,
		wire: W,
	},
}

// Manual `Clone` impl that does not require `W::Builder: Clone` (the derived impl would, even
// though `Weak<T>: Clone` for any `T`).
impl<F: Field, W: CircuitWire<F> + Clone> Clone for CircuitElem<F, W> {
	fn clone(&self) -> Self {
		match self {
			Self::Constant(c) => Self::Constant(*c),
			Self::Wire { builder, wire } => Self::Wire {
				builder: builder.clone(),
				wire: wire.clone(),
			},
		}
	}
}

impl<F, W> CircuitElem<F, W>
where
	F: Field,
	W: CircuitWire<F>,
{
	/// Construct a [`Self::Wire`] anchored to a shared builder via a [`Weak`] reference.
	pub fn wire(builder: &Rc<RefCell<W::Builder>>, wire: W) -> Self {
		Self::Wire {
			builder: Rc::downgrade(builder),
			wire,
		}
	}

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

	/// Variable-arity sibling of [`Self::combine`].
	fn combine_varlen(
		elems: &[&Self],
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		builder_op: impl FnOnce(
			&mut W::Builder,
			&[<W::Builder as CircuitBuilder>::Wire],
		) -> Vec<<W::Builder as CircuitBuilder>::Wire>,
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
				})
				.collect::<Vec<_>>();
			let inner_wire_refs = inner_wires.iter().map(AsRef::as_ref).collect::<Vec<_>>();
			W::combine_varlen(&mut *builder, &inner_wire_refs, f_op, builder_op)
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
			f_op(&inner_constants)
				.into_iter()
				.map(Self::Constant)
				.collect()
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

impl<'a, F: Field, W: CircuitWire<F>> Sum<&'a Self> for CircuitElem<F, W> {
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(F::ZERO), |acc, x| acc + x)
	}
}

impl<F: Field, W: CircuitWire<F>> Product for CircuitElem<F, W> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(F::ONE), |acc, x| acc * x)
	}
}

impl<'a, F: Field, W: CircuitWire<F>> Product<&'a Self> for CircuitElem<F, W> {
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(F::ONE), |acc, x| acc * x)
	}
}

impl<F: Field, W: CircuitWire<F>> Square for CircuitElem<F, W> {
	fn square(self) -> Self {
		let [ret] = Self::combine(
			[&self],
			|[x]| [x.square()],
			|builder, [x]| [builder.mul(x, x)],
		);
		ret
	}
}

impl<F: Field, W: CircuitWire<F>> InvertOrZero for CircuitElem<F, W> {
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

impl<F: Field, W: CircuitWire<F> + Clone> FieldOps for CircuitElem<F, W> {
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

impl<F: Field, W: CircuitWire<F>> From<F> for CircuitElem<F, W> {
	fn from(val: F) -> Self {
		Self::Constant(val)
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
