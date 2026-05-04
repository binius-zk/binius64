// Copyright 2026 The Binius Developers

//! [`IronSpartanBuilderChannel`]: an [`IPVerifierChannel`] that symbolically executes a verifier
//! and records the computation as constraints on a [`ConstraintBuilder`].

use std::{
	cell::RefCell,
	rc::{Rc, Weak},
};

use binius_field::Field;
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder},
	constraint_system::ConstraintWire,
};

use super::circuit_elem::{CircuitElem, CircuitWire};

/// [`CircuitWire`] backend over [`ConstraintBuilder`] — used by [`IronSpartanBuilderChannel`] to
/// record arithmetic as constraints in a constraint system.
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
		n_out: usize,
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
			let result = f_op(&inner_constants);
			debug_assert_eq!(result.len(), n_out);
			result.into_iter().map(Self::Constant).collect()
		} else {
			let inner_wires = wires
				.iter()
				.map(|wire| match wire {
					Self::Constant(val) => builder.constant(*val),
					Self::Wire(wire) => *wire,
				})
				.collect::<Vec<_>>();
			let result = builder_op(builder, &inner_wires);
			debug_assert_eq!(result.len(), n_out);
			result.into_iter().map(Self::Wire).collect()
		}
	}
}

/// A channel that symbolically executes a verifier, building up an IronSpartan constraint system.
///
/// Instead of performing actual verification, this channel records all operations as constraints
/// in a [`ConstraintBuilder`]. The typical usage pattern is:
///
/// 1. Create an `IronSpartanBuilderChannel` from a [`ConstraintBuilder`]
/// 2. Run the verifier on the channel (e.g., `verify_iop`)
/// 3. The channel's `finish()` method returns the [`ConstraintBuilder`] with all recorded
///    constraints
pub struct IronSpartanBuilderChannel<F: Field> {
	builder: Rc<RefCell<ConstraintBuilder<F>>>,
}

impl<F: Field> IronSpartanBuilderChannel<F> {
	/// Creates a new builder channel that takes ownership of the given constraint builder.
	pub fn new(builder: ConstraintBuilder<F>) -> Self {
		Self {
			builder: Rc::new(RefCell::new(builder)),
		}
	}

	fn alloc_inout_elem(&self) -> CircuitElem<F, BuilderWire<F>> {
		let wire = self.builder.borrow_mut().alloc_inout();
		CircuitElem::wire(&self.builder, BuilderWire::Wire(wire))
	}

	fn alloc_precommit_elem(&self) -> CircuitElem<F, BuilderWire<F>> {
		let wire = self.builder.borrow_mut().alloc_precommit();
		CircuitElem::wire(&self.builder, BuilderWire::Wire(wire))
	}

	/// Consumes the channel and returns the underlying [`ConstraintBuilder`].
	///
	/// This must be called after all `CircuitElem` values derived from this channel have been
	/// dropped, as it requires sole ownership of the builder via `Rc::try_unwrap`.
	pub fn finish(self) -> ConstraintBuilder<F> {
		Rc::try_unwrap(self.builder)
			.expect("CircuitElem values should only hold Weak references")
			.into_inner()
	}
}

impl<F: Field> IPVerifierChannel<F> for IronSpartanBuilderChannel<F> {
	type Elem = CircuitElem<F, BuilderWire<F>>;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		// For each element that the inner prover sends, the wrapped prover allocates a one-time-pad
		// encryption key in the precommit segment and encrypts the underlying value before sending.
		// Here the verifier gets the encryption key from the precommit segment and decrypts.
		let inout = self.alloc_inout_elem();
		let key = self.alloc_precommit_elem();
		Ok(inout - key)
	}

	fn sample(&mut self) -> Self::Elem {
		self.alloc_inout_elem()
	}

	fn observe_one(&mut self, _val: F) -> Self::Elem {
		self.alloc_inout_elem()
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			CircuitElem::Constant(c)
			| CircuitElem::Wire {
				wire: BuilderWire::Constant(c),
				..
			} => {
				if c == F::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			CircuitElem::Wire {
				builder,
				wire: BuilderWire::Wire(wire),
			} => {
				assert!(Weak::ptr_eq(&Rc::downgrade(&self.builder), &builder));
				self.builder.borrow_mut().assert_zero(wire);
				Ok(())
			}
		}
	}
}

impl<F: Field> IOPVerifierChannel<F> for IronSpartanBuilderChannel<F> {
	type Oracle = ();

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&[]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, binius_iop::channel::Error> {
		Ok(())
	}

	fn verify_oracle_relations<'a>(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'a, Self::Oracle, Self::Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		// For each oracle opening, the prover sends the decrypted evaluation. The outer verifier
		// checks in the circuit equality of this value with the expected expression over encrypted
		// values.
		for relation in oracle_relations {
			let decrypted_claim = self.alloc_inout_elem();
			self.assert_zero(relation.claim - decrypted_claim)?;
		}
		Ok(())
	}
}
