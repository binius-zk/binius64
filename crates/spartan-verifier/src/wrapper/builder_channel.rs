// Copyright 2026 The Binius Developers

//! [`IronSpartanBuilderChannel`]: an [`IPVerifierChannel`] that symbolically executes a verifier
//! and records the computation as constraints on a [`ConstraintBuilder`].

use std::{
	array,
	cell::{Cell, RefCell},
	iter::repeat_with,
	rc::{Rc, Weak},
};

use binius_field::Field;
use binius_iop::channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder},
	constraint_system::ConstraintWire,
	gate::{EvalFn, GateSequence, RecWire, Val},
};

use super::circuit_elem::{CircuitElem, CircuitWire};

/// [`CircuitWire`] backend over [`ConstraintBuilder`] — used by [`IronSpartanBuilderChannel`] to
/// record arithmetic as constraints in a constraint system.
#[derive(Debug, Clone)]
pub enum BuilderWire<F> {
	Constant(F),
	/// A lazy InOut value: `cw` is the outer constraint-system wire, allocated only when the wire
	/// is materialized (mixed with a private value). `rec` is its recorded-gate-sequence wire id
	/// (BINIUS-43), assigned when recording is enabled (always, via the builder channel).
	InOut {
		cw: Rc<Cell<Option<ConstraintWire>>>,
		rec: Option<RecWire>,
	},
	Private(ConstraintWire),
}

impl<F: Field> BuilderWire<F> {
	fn lazy_inout(rec: Option<RecWire>) -> Self {
		Self::InOut {
			cw: Rc::new(Cell::new(None)),
			rec,
		}
	}

	fn materialize(builder: &mut ConstraintBuilder<F>, wire: &Self) -> ConstraintWire {
		match wire {
			Self::Constant(val) => builder.constant(*val),
			Self::InOut { cw, rec } => {
				if let Some(wire) = cw.get() {
					wire
				} else {
					let wire = builder.alloc_inout();
					cw.set(Some(wire));
					if let (Some(rec), Some(recorder)) = (rec, builder.recorder_mut()) {
						recorder.materialize(*rec, wire);
					}
					wire
				}
			}
			Self::Private(wire) => *wire,
		}
	}
}

impl<F: Field> CircuitWire<F> for BuilderWire<F> {
	type Builder = ConstraintBuilder<F>;

	fn combine<const IN: usize, const OUT: usize>(
		builder: &mut Self::Builder,
		wires: [&Self; IN],
		f_op: impl Fn([F; IN]) -> [F; OUT],
		builder_op: impl Fn(&mut Self::Builder, [ConstraintWire; IN]) -> [ConstraintWire; OUT],
	) -> [Self; OUT] {
		let inner_constants = array_util::try_map(wires, |wire| match wire {
			Self::Constant(val) => Some(*val),
			_ => None,
		});

		if let Some(inner_values) = inner_constants {
			f_op(inner_values).map(Self::Constant)
		} else {
			let is_inout = wires
				.iter()
				.all(|wire| matches!(wire, Self::Constant(_) | Self::InOut { .. }));
			if is_inout {
				// Public-only result: a fresh lazy InOut per output. Allocate a recorder wire id so
				// the public gate (recorded by `record_public_gate`) and later `materialize` can
				// reference it.
				array::from_fn(|_| {
					let rec = builder.recorder_mut().map(|r| r.alloc_rec());
					Self::lazy_inout(rec)
				})
			} else {
				// If any of the inputs are private wires, then lazily materialize in inout wires
				// and compute the result within the circuit.
				let inner_wires = wires.map(|wire| Self::materialize(builder, wire));
				builder_op(builder, inner_wires).map(Self::Private)
			}
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
				_ => None,
			})
			.collect::<Option<Vec<_>>>();

		if let Some(inner_constants) = inner_constants {
			let result = f_op(&inner_constants);
			debug_assert_eq!(result.len(), n_out);
			result.into_iter().map(Self::Constant).collect()
		} else {
			let is_inout = wires
				.iter()
				.all(|wire| matches!(wire, Self::Constant(_) | Self::InOut { .. }));
			if is_inout {
				repeat_with(|| {
					let rec = builder.recorder_mut().map(|r| r.alloc_rec());
					Self::lazy_inout(rec)
				})
				.take(n_out)
				.collect()
			} else {
				// If any of the inputs are private wires, then lazily materialize in inout wires
				// and compute the result within the circuit.
				let inner_wires = wires
					.iter()
					.map(|wire| Self::materialize(builder, wire))
					.collect::<Vec<_>>();
				let result = builder_op(builder, &inner_wires);
				debug_assert_eq!(result.len(), n_out);
				result.into_iter().map(Self::Private).collect()
			}
		}
	}

	fn record_public_gate(
		builder: &mut Self::Builder,
		ins: &[&Self],
		outs: &[&Self],
		f: EvalFn<F>,
	) {
		// Only the public-only path is recorded here: every output is a freshly-allocated lazy
		// InOut wire carrying a recorder id. Private results (their gates are recorded by the
		// ConstraintBuilder primitives) and constant-folded results carry no `rec` and are skipped.
		let mut out_recs = Vec::with_capacity(outs.len());
		for out in outs {
			match out {
				Self::InOut { rec: Some(rec), .. } => out_recs.push(*rec),
				_ => return,
			}
		}
		let in_vals = ins
			.iter()
			.map(|wire| match wire {
				Self::Constant(val) => Val::Const(*val),
				Self::InOut { rec: Some(rec), .. } => Val::Wire(*rec),
				Self::InOut { rec: None, .. } => {
					unreachable!("public input InOut wire is missing its recorder id")
				}
				Self::Private(_) => unreachable!("public op cannot have a private input"),
			})
			.collect::<Vec<_>>();
		if let Some(recorder) = builder.recorder_mut() {
			recorder.push_public(in_vals, out_recs, f);
		}
	}
}

/// A channel that symbolically executes a verifier, building up an IronSpartan constraint system.
///
/// Instead of performing actual verification, this channel records all operations as constraints
/// in a [`ConstraintBuilder`]. The typical usage pattern is:
///
/// 1. Construct a fresh [`IronSpartanBuilderChannel`] via [`Self::new`]
/// 2. Run the verifier on the channel (e.g., `verify_iop`)
/// 3. The channel's `finish()` method returns the [`ConstraintBuilder`] with all recorded
///    constraints
pub struct IronSpartanBuilderChannel<F: Field> {
	builder: Rc<RefCell<ConstraintBuilder<F>>>,
}

impl<F: Field> Default for IronSpartanBuilderChannel<F> {
	fn default() -> Self {
		Self::new()
	}
}

impl<F: Field> IronSpartanBuilderChannel<F> {
	/// Creates a new builder channel backed by a fresh [`ConstraintBuilder`] with gate recording
	/// enabled (BINIUS-43): the symbolic build records a [`GateSequence`] alongside the constraint
	/// system, retrievable via [`Self::finish_with_gates`].
	pub fn new() -> Self {
		let mut builder = ConstraintBuilder::new();
		builder.enable_recording();
		Self {
			builder: Rc::new(RefCell::new(builder)),
		}
	}

	fn alloc_inout_elem(&self) -> CircuitElem<F, BuilderWire<F>> {
		// Each recv/sample/observe value is an external channel input in the recorded sequence.
		let rec = self.builder.borrow_mut().recorder_mut().map(|r| r.input());
		CircuitElem::wire(&self.builder, BuilderWire::lazy_inout(rec))
	}

	fn alloc_precommit_elem(&self) -> CircuitElem<F, BuilderWire<F>> {
		let mut builder = self.builder.borrow_mut();
		let wire = builder.alloc_precommit();
		if let Some(recorder) = builder.recorder_mut() {
			recorder.precommit(wire);
		}
		drop(builder);
		CircuitElem::wire(&self.builder, BuilderWire::Private(wire))
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

	/// Like [`Self::finish`], but also returns the recorded [`GateSequence`] (BINIUS-43).
	pub fn finish_with_gates(self) -> (ConstraintBuilder<F>, GateSequence<F>) {
		let mut builder = Rc::try_unwrap(self.builder)
			.expect("CircuitElem values should only hold Weak references")
			.into_inner();
		let gates = builder
			.take_recording()
			.expect("recording is enabled in IronSpartanBuilderChannel::new");
		(builder, gates)
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
				wire: BuilderWire::InOut { .. },
				..
			} => {
				// Nothing to do here. The value can be checked directly in
				// ZKWrappedVerifierChannel.
				Ok(())
			}
			CircuitElem::Wire {
				builder,
				wire: BuilderWire::Private(wire),
			} => {
				assert!(Weak::ptr_eq(&Rc::downgrade(&self.builder), &builder));
				self.builder.borrow_mut().assert_zero(wire);
				Ok(())
			}
		}
	}

	fn compute_public_value(
		&mut self,
		inputs: &[Self::Elem],
		f: impl FnOnce(&[F]) -> F,
	) -> Self::Elem {
		let input_refs = inputs.iter().collect::<Vec<_>>();
		let outs = CircuitElem::combine_varlen(
			&input_refs,
			1,
			move |inputs| vec![f(inputs)],
			|_, _| {
				// Self::Elem::combine_varlen will only call the builder_op closure if any inputs
				// are non-public.
				panic!("compute_public_value: input is not public")
			},
		);
		outs.into_iter()
			.next()
			.expect("combine_varlen returns Vec with len = n_out; n_out = 1")
	}

	fn compute_public_value_recorded(
		&mut self,
		inputs: &[Self::Elem],
		f: impl FnOnce(&[F]) -> F + 'static,
	) -> Self::Elem {
		// The result is a single public (InOut) value; record it as a `Public` gate so replay can
		// recompute it from the (public) inputs, then return a fresh lazy InOut wire labelled with
		// the recorded output id.
		let in_vals = inputs
			.iter()
			.map(|elem| match elem {
				CircuitElem::Constant(c)
				| CircuitElem::Wire {
					wire: BuilderWire::Constant(c),
					..
				} => Val::Const(*c),
				CircuitElem::Wire {
					wire: BuilderWire::InOut { rec: Some(rec), .. },
					..
				} => Val::Wire(*rec),
				_ => panic!("compute_public_value: input is not public"),
			})
			.collect::<Vec<_>>();

		let mut builder = self.builder.borrow_mut();
		let rec = builder.recorder_mut().map(|recorder| {
			let outs = recorder.public(in_vals, 1, Box::new(move |xs: &[F]| vec![f(xs)]));
			outs[0]
		});
		drop(builder);
		CircuitElem::wire(&self.builder, BuilderWire::lazy_inout(rec))
	}
}

impl<'r, F: Field> IOPVerifierChannel<'r, F> for IronSpartanBuilderChannel<F> {
	type Oracle = ();

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&[]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, binius_iop::channel::Error> {
		Ok(())
	}

	fn verify_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'r, Self::Oracle, Self::Elem>>,
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
