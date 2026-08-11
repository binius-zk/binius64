// Copyright 2026 The Binius Developers

//! The GHASH-field element a circuit-building channel carries.

use std::{
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
	rc::{Rc, Weak},
};

use binius_field::{
	BinaryField128bGhash as B128, ExtensionField, Field, FieldOps,
	arithmetic_traits::{InvertOrZero, Square},
};
use binius_frontend::{CircuitBuilder, Wire};

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
pub enum Elem {
	Constant(B128),
	Wires {
		builder: Weak<CircuitBuilder>,
		lo: Wire,
		hi: Wire,
	},
}

impl Elem {
	/// Constructs a wire-backed element anchored to a shared builder.
	pub fn wires(builder: &Rc<CircuitBuilder>, lo: Wire, hi: Wire) -> Self {
		Self::Wires {
			builder: Rc::downgrade(builder),
			lo,
			hi,
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
		let builder = match (self, rhs) {
			(Self::Constant(a), Self::Constant(b)) => return Self::Constant(fold(*a, *b)),
			(Self::Wires { builder, .. }, _) | (_, Self::Wires { builder, .. }) => builder,
		};
		let Some(shared) = builder.upgrade() else {
			panic!("an Elem outlived the channel that created it");
		};
		let (lo, hi) = gate(&shared, self.to_wires(&shared), rhs.to_wires(&shared));
		Self::wires(&shared, lo, hi)
	}
}

// In characteristic 2 negation is the identity.
impl Neg for Elem {
	type Output = Self;

	fn neg(self) -> Self {
		self
	}
}

impl Add<&Self> for Elem {
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

impl Mul<&Self> for Elem {
	type Output = Self;

	fn mul(self, rhs: &Self) -> Self {
		// A product with a zero factor is zero without a constraint, which is worth catching: the
		// verifier multiplies by eq-indicator terms that are often zero at build time.
		if matches!(&self, Self::Constant(c) if *c == B128::ZERO)
			|| matches!(rhs, Self::Constant(c) if *c == B128::ZERO)
		{
			return Self::Constant(B128::ZERO);
		}
		self.combine(
			rhs,
			|a, b| a * b,
			|builder, (a_lo, a_hi), (b_lo, b_hi)| builder.bmul(a_lo, a_hi, b_lo, b_hi),
		)
	}
}

impl Sub<&Self> for Elem {
	type Output = Self;

	// Subtraction is addition in characteristic 2, which is what the shared `combine` records.
	#[allow(clippy::suspicious_arithmetic_impl)]
	fn sub(self, rhs: &Self) -> Self {
		self + rhs
	}
}

macro_rules! by_value {
	($trait:ident, $method:ident) => {
		impl $trait for Elem {
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
		impl $trait for Elem {
			fn $method(&mut self, rhs: Self) {
				*self = self.clone().$op(&rhs);
			}
		}
		impl $trait<&Self> for Elem {
			fn $method(&mut self, rhs: &Self) {
				*self = self.clone().$op(rhs);
			}
		}
	};
}
assign!(AddAssign, add_assign, add);
assign!(SubAssign, sub_assign, sub);
assign!(MulAssign, mul_assign, mul);

impl Sum for Elem {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(B128::ZERO), |acc, x| acc + x)
	}
}

impl<'a> Sum<&'a Self> for Elem {
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(B128::ZERO), |acc, x| acc + x)
	}
}

impl Product for Elem {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(B128::ONE), |acc, x| acc * x)
	}
}

impl<'a> Product<&'a Self> for Elem {
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::Constant(B128::ONE), |acc, x| acc * x)
	}
}

impl Square for Elem {
	fn square(self) -> Self {
		self.clone() * &self
	}
}

impl InvertOrZero for Elem {
	fn invert_or_zero(self) -> Self {
		// The inverse is hinted and then checked, the way the Spartan wrapper does it, so this
		// needs the hint plumbing rather than a gate.
		todo!("invert_or_zero: hint the inverse and constrain the product")
	}
}

impl From<B128> for Elem {
	fn from(value: B128) -> Self {
		Self::Constant(value)
	}
}

impl FieldOps for Elem {
	type Scalar = B128;

	fn zero() -> Self {
		Self::Constant(B128::ZERO)
	}

	fn one() -> Self {
		Self::Constant(B128::ONE)
	}

	fn square_transpose<FSub: Field>(_elems: &mut [Self])
	where
		B128: ExtensionField<FSub>,
	{
		// Ring switching calls this over `B1`, where `basis(i) = 1 << i` makes it a 128x128
		// bit-matrix transpose of the wire pairs: the 64-bit block swap is free rewiring and each
		// 64x64 block is the standard shift-and-mask network.
		todo!("square_transpose: bit-matrix transpose gadget")
	}
}
