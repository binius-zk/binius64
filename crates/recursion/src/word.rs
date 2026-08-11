// Copyright 2026 The Binius Developers

//! The 64-bit word a circuit-building channel carries.

use std::{
	ops::Shr,
	rc::{Rc, Weak},
};

use binius_frontend::{CircuitBuilder, Wire};

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
	Wire {
		builder: Weak<CircuitBuilder>,
		wire: Wire,
	},
}

impl Word {
	/// Constructs a wire-backed word anchored to a shared builder.
	pub fn wire(builder: &Rc<CircuitBuilder>, wire: Wire) -> Self {
		Self::Wire {
			builder: Rc::downgrade(builder),
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

	/// Returns the bit at position `i` as an all-ones or all-zeros mask word.
	///
	/// Masks rather than a single bit because that is what `select` consumes: the frontend's
	/// `select` gate reads the most significant bit of its condition.
	pub fn bit_mask(&self, _i: usize) -> Wire {
		// Needs the bit broadcast to the whole word, which is a gadget rather than a single gate.
		todo!("bit extraction gadget")
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
			Self::Wire { builder, wire } => {
				let Some(shared) = builder.upgrade() else {
					panic!("a Word outlived the channel that created it");
				};
				let shifted = shared.shr(wire, rhs);
				Self::Wire {
					builder: Rc::downgrade(&shared),
					wire: shifted,
				}
			}
		}
	}
}
