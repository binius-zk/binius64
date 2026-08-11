// Copyright 2026 The Binius Developers

//! The 64-bit word a circuit-building channel carries.

use std::{
	ops::Shr,
	rc::{Rc, Weak},
};

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
pub enum Word {
	Constant(binius_core::word::Word),
	Wire { shared: Weak<Shared>, wire: Wire },
}

impl Word {
	/// Constructs a wire-backed word anchored to the shared builder.
	pub fn wire(shared: &Rc<Shared>, wire: Wire) -> Self {
		Self::Wire {
			shared: Rc::downgrade(shared),
			wire,
		}
	}

	/// Lowers to a wire, materializing a `Constant` on the builder.
	pub fn to_wire(&self, builder: &CircuitBuilder) -> Wire {
		match self {
			Self::Constant(word) => builder.add_constant_64(word.as_u64()),
			Self::Wire { wire, .. } => *wire,
		}
	}

	/// Returns bit `i` moved into the most significant position, where a `select` gate reads it.
	///
	/// The bits below it are whatever the shift carried up and are ignored, so this is one gate
	/// rather than a broadcast. `single_wire_multiplex` selects the same way.
	pub fn bit_at_msb(&self, builder: &CircuitBuilder, i: usize) -> Wire {
		let wire = self.to_wire(builder);
		builder.shl(wire, (binius_core::word::Word::BITS - 1 - i) as u32)
	}
}

impl From<binius_core::word::Word> for Word {
	fn from(word: binius_core::word::Word) -> Self {
		Self::Constant(word)
	}
}

impl Shr<u32> for Word {
	type Output = Self;

	fn shr(self, rhs: u32) -> Self {
		match self {
			Self::Constant(word) => Self::Constant(word >> rhs),
			Self::Wire { shared, wire } => {
				let Some(owner) = shared.upgrade() else {
					panic!("a Word outlived the channel that created it");
				};
				let shifted = owner.builder().shr(wire, rhs);
				Self::wire(&owner, shifted)
			}
		}
	}
}
