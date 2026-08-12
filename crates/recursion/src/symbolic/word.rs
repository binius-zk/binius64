// Copyright 2026 The Binius Developers

//! The 64-bit word a circuit-building channel carries.

use std::{
	ops::{Shl, Shr},
	rc::{Rc, Weak},
};

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use crate::shared::Shared;

/// A 64-bit word that is either fixed while the circuit is built or carried by a wire.
///
/// This is the `Word` associated type of
/// [`WordIPVerifierChannel`](binius_ip::channel::WordIPVerifierChannel). The trait requires
/// `From<Word> + Shr<u32>` on it rather than offering channel methods, so the operations have to be
/// available without a channel in hand — hence the [`Weak`] handle back to the builder, which is
/// the same shape `CircuitElem` uses in the Spartan wrapper.
///
/// A `Constant` folds while the circuit is built and costs nothing. The FRI query indices, which
/// arrive from `sample_bits`, are `Wire`s.
#[derive(Clone)]
pub enum SymbolicWord {
	Constant(Word),
	Wire { shared: Weak<Shared>, wire: Wire },
}

impl SymbolicWord {
	/// Constructs a wire-backed word anchored to the shared builder.
	pub fn wire(shared: &Rc<Shared>, wire: Wire) -> Self {
		Self::Wire {
			shared: Rc::downgrade(shared),
			wire,
		}
	}

	/// The value, when the word was fixed while the circuit was built.
	///
	/// A caller reads this to settle an operation in Rust rather than spend gates on it.
	pub const fn value(&self) -> Option<Word> {
		match self {
			Self::Constant(word) => Some(*word),
			Self::Wire { .. } => None,
		}
	}

	/// Lowers to a wire, materializing a `Constant` on the builder.
	pub fn to_wire(&self, builder: &CircuitBuilder) -> Wire {
		match self {
			Self::Constant(word) => builder.add_constant_64(word.as_u64()),
			Self::Wire { wire, .. } => *wire,
		}
	}

	/// Applies a shift, folding a constant and emitting a gate otherwise.
	fn shift(
		self,
		amount: u32,
		gate: impl Fn(&CircuitBuilder, Wire, u32) -> Wire,
		fold: impl Fn(Word, u32) -> Word,
	) -> Self {
		match self {
			Self::Constant(word) => Self::Constant(fold(word, amount)),
			Self::Wire { shared, wire } => {
				let Some(owner) = shared.upgrade() else {
					panic!("a SymbolicWord outlived the channel that created it");
				};
				let shifted = gate(owner.builder(), wire, amount);
				Self::wire(&owner, shifted)
			}
		}
	}
}

impl From<Word> for SymbolicWord {
	fn from(word: Word) -> Self {
		Self::Constant(word)
	}
}

impl Shr<u32> for SymbolicWord {
	type Output = Self;

	fn shr(self, rhs: u32) -> Self {
		self.shift(rhs, CircuitBuilder::shr, |word, rhs| word >> rhs)
	}
}

/// Shifting left is how a caller moves a chosen bit into the most significant position, where a
/// `select` gate reads it.
impl Shl<u32> for SymbolicWord {
	type Output = Self;

	fn shl(self, rhs: u32) -> Self {
		self.shift(rhs, CircuitBuilder::shl, |word, rhs| word << rhs)
	}
}
