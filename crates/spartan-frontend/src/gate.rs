// Copyright 2026 The Binius Developers

//! Recorded gate sequence for prover-side witness generation.
//!
//! When the inner verifier is symbolically executed to build the outer wrapper constraint system
//! (via [`ConstraintBuilder`]), the same traversal can be recorded as a flat [`GateSequence`]: one
//! [`Gate`] per derived-computation step (an `add`, `mul`, hint, or `compute_public_value`), keyed
//! on the (pre-elimination) [`ConstraintWire`]s the [`ConstraintBuilder`] allocated. The gates
//! capture exactly the inner verifier's "verify" arithmetic.
//!
//! The prover replays this sequence in [`ZKWrappedProverChannel::finish`] to fill the outer
//! witness: it allocates the inout (transcript) and precommit (one-time-pad key) wires on its own
//! [`WitnessGenerator`] and writes their values directly as the inner proof runs, then
//! [`WitnessGenerator::apply_gates`](crate::circuit_builder::WitnessGenerator::apply_gates) replays
//! the recorded gates to compute the derived and private wires — substituting for a re-execution of
//! the inner verifier. The external inputs are not recorded as gates; only the derived computation
//! is.
//!
//! The recording is done once (during the setup-time symbolic build) and replayed on every prove
//! call, so [`Gate::Generic`] closures are [`Fn`] and own their captures (`'static`).
//!
//! [`ConstraintBuilder`]: crate::circuit_builder::ConstraintBuilder
//! [`WitnessGenerator`]: crate::circuit_builder::WitnessGenerator
//! [`ZKWrappedProverChannel::finish`]: https://docs.rs/binius-spartan-prover

use binius_field::Field;

use crate::{
	circuit_builder::{
		CircuitBuilder, CircuitBuilderWithAlloc, ConstraintBuilder, ConstraintSystemIR,
	},
	constraint_system::ConstraintWire,
};

/// The boxed closure a [`Gate::Generic`] evaluates: it maps recorded input values to output
/// values. `Fn` and `'static` (owns its captures), since the gate sequence is recorded once and
/// replayed on every prove call.
pub type GateEval<F> = Box<dyn Fn(&[F]) -> Vec<F>>;

/// A single recorded derived-computation step, referencing the [`ConstraintWire`]s allocated for it
/// during the symbolic build.
///
/// The wire ids are the *pre-elimination* ids the [`ConstraintBuilder`] produced; replay
/// ([`WitnessGenerator::apply_gates`](crate::circuit_builder::WitnessGenerator::apply_gates)) looks
/// each input wire up by id and writes each output's value into the corresponding witness slot, or
/// holds it aside if the wire was pruned by wire-elimination but is still consumed by a later gate.
///
/// External inputs (inout transcript wires and precommit keys) are *not* gates — the prover channel
/// writes their values directly. Only the verifier's derived arithmetic is recorded here.
pub enum Gate<F: Field> {
	/// `out = lhs + rhs`.
	Add {
		lhs: ConstraintWire,
		rhs: ConstraintWire,
		out: ConstraintWire,
	},
	/// `out = lhs * rhs`.
	Mul {
		lhs: ConstraintWire,
		rhs: ConstraintWire,
		out: ConstraintWire,
	},
	/// Outputs computed by an arbitrary closure over the input values. Covers hints (e.g.
	/// `invert_or_zero`, `square_transpose`'s coefficient extraction) and `compute_public_value`.
	///
	/// `eval` is `Fn` and `'static` (it owns its captures): the sequence is recorded once and
	/// replayed on every prove call.
	Generic {
		inputs: Vec<ConstraintWire>,
		outputs: Vec<ConstraintWire>,
		eval: GateEval<F>,
	},
}

/// A flat, topologically-ordered recording of the derived-computation steps performed during a
/// symbolic build. Replay it with
/// [`WitnessGenerator::apply_gates`](crate::circuit_builder::WitnessGenerator::apply_gates).
#[derive(Default)]
pub struct GateSequence<F: Field> {
	gates: Vec<Gate<F>>,
}

impl<F: Field> GateSequence<F> {
	pub fn new() -> Self {
		Self { gates: Vec::new() }
	}

	pub fn push(&mut self, gate: Gate<F>) {
		self.gates.push(gate);
	}

	pub fn gates(&self) -> &[Gate<F>] {
		&self.gates
	}

	pub fn len(&self) -> usize {
		self.gates.len()
	}

	pub fn is_empty(&self) -> bool {
		self.gates.is_empty()
	}
}

/// A [`CircuitBuilder`] that records a [`GateSequence`] while delegating wire allocation and
/// constraint recording to an inner [`ConstraintBuilder`].
///
/// Running the same circuit logic on a `GateRecordingConstraintBuilder` that you would on a
/// [`ConstraintBuilder`] yields both the constraint system IR (via [`Self::into_parts`]) and a gate
/// sequence whose wire ids match it exactly. The inout/precommit allocations record no gate (the
/// prover channel supplies their values directly); the derived-computation ops (`add`, `mul`,
/// `hint`, `hint_varsize`) each record a gate.
pub struct GateRecordingConstraintBuilder<F: Field> {
	inner: ConstraintBuilder<F>,
	gates: GateSequence<F>,
}

impl<F: Field> GateRecordingConstraintBuilder<F> {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self {
			inner: ConstraintBuilder::new(),
			gates: GateSequence::new(),
		}
	}

	/// Consumes the builder, returning the underlying constraint system IR (normally discarded —
	/// the layout comes from the setup-time build) and the recorded gate sequence.
	pub fn into_parts(self) -> (ConstraintSystemIR<F>, GateSequence<F>) {
		(self.inner.build(), self.gates)
	}
}

impl<F: Field> CircuitBuilderWithAlloc for GateRecordingConstraintBuilder<F> {
	// No gate is recorded for either — the prover channel writes the transcript value /
	// one-time-pad key into these wires directly at proving time.
	fn alloc_inout(&mut self) -> Self::Wire {
		self.inner.alloc_inout()
	}

	fn alloc_precommit(&mut self) -> Self::Wire {
		self.inner.alloc_precommit()
	}
}

impl<F: Field> CircuitBuilder for GateRecordingConstraintBuilder<F> {
	type Wire = ConstraintWire;
	type Field = F;

	fn assert_zero(&mut self, wire: Self::Wire) {
		// Assertions allocate no wire and are not replayed (witness validation against the
		// constraint system catches violations), so nothing is recorded.
		self.inner.assert_zero(wire);
	}

	fn assert_eq(&mut self, lhs: Self::Wire, rhs: Self::Wire) {
		self.inner.assert_eq(lhs, rhs);
	}

	fn constant(&mut self, val: Self::Field) -> Self::Wire {
		// Constant wires carry their value in the layout; no gate is needed.
		self.inner.constant(val)
	}

	fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		let out = self.inner.add(lhs, rhs);
		self.gates.push(Gate::Add { lhs, rhs, out });
		out
	}

	fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		let out = self.inner.mul(lhs, rhs);
		self.gates.push(Gate::Mul { lhs, rhs, out });
		out
	}

	fn hint<
		H: Fn([Self::Field; IN]) -> [Self::Field; OUT] + 'static,
		const IN: usize,
		const OUT: usize,
	>(
		&mut self,
		inputs: [Self::Wire; IN],
		f: H,
	) -> [Self::Wire; OUT] {
		// The inner `ConstraintBuilder::hint` ignores its closure (it only allocates wires), so
		// pass a no-op and keep the real `f` to box into the recorded gate's eval.
		let outputs = self.inner.hint(inputs, |_| [F::ZERO; OUT]);
		self.gates.push(Gate::Generic {
			inputs: inputs.to_vec(),
			outputs: outputs.to_vec(),
			eval: Box::new(move |vals: &[F]| {
				let arr: [F; IN] = vals.try_into().expect("Generic hint input arity matches");
				f(arr).to_vec()
			}),
		});
		outputs
	}

	fn hint_varsize(
		&mut self,
		inputs: &[Self::Wire],
		out_len: usize,
		f: impl Fn(&[F]) -> Vec<F> + 'static,
	) -> Vec<Self::Wire> {
		// Same as `hint`: the inner builder ignores the closure (allocates wires only); keep the
		// real `f` to box into the recorded gate's eval.
		let outputs = self
			.inner
			.hint_varsize(inputs, out_len, |_: &[F]| Vec::<F>::new());
		self.gates.push(Gate::Generic {
			inputs: inputs.to_vec(),
			outputs: outputs.clone(),
			eval: Box::new(f),
		});
		outputs
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128bGhash as B128, Field, arithmetic_traits::InvertOrZero};

	use super::*;
	use crate::{
		circuit_builder::WitnessGenerator,
		constraint_system::{ConstraintSystem, WitnessLayout},
		wire_elimination::{CostModel, run_wire_elimination},
	};

	/// A circuit mixing a derived mul, a derived hint, a derived add, and a private mul (against a
	/// precommit input) — exercises all gate kinds plus wire-elimination of intermediates.
	fn mixed_circuit<Builder: CircuitBuilder>(
		builder: &mut Builder,
		a: Builder::Wire,
		b: Builder::Wire,
		s: Builder::Wire,
		expected: Builder::Wire,
	) {
		let d = builder.mul(a, b);
		let [d_inv] = builder.hint([d], |[x]| [x.invert_or_zero()]);
		let one_check = builder.mul(d, d_inv);
		let e = builder.add(one_check, b);
		let p = builder.mul(e, s);
		builder.assert_eq(p, expected);
	}

	fn mixed_expected(a: B128, b: B128, s: B128) -> B128 {
		let d = a * b;
		let one_check = d * d.invert_or_zero();
		(one_check + b) * s
	}

	#[test]
	fn test_apply_gates_matches_witness_generator() {
		let a_val = B128::new(3);
		let b_val = B128::new(5);
		let s_val = B128::new(7);
		let expected_val = mixed_expected(a_val, b_val, s_val);

		// Record the circuit (inout a, b; precommit s; inout expected), capturing the gate
		// sequence.
		let mut rb = GateRecordingConstraintBuilder::<B128>::new();
		let a = rb.alloc_inout();
		let b = rb.alloc_inout();
		let s = rb.alloc_precommit();
		let expected = rb.alloc_inout();
		mixed_circuit(&mut rb, a, b, s, expected);
		let (ir, gates) = rb.into_parts();
		let (cs, layout) = run_compile(ir);

		// Replay: write the external inputs (inout + precommit) directly, then apply the gates to
		// fill the derived and private wires — the prover-channel path.
		let mut wg = WitnessGenerator::new(&layout);
		wg.write_inout(a, a_val);
		wg.write_inout(b, b_val);
		wg.write_precommit(s, s_val);
		wg.write_inout(expected, expected_val);
		wg.apply_gates(&gates);
		let witness = wg.build().expect("witness generation should succeed");
		cs.validate(&witness);

		// Compare against the witness the direct WitnessGenerator path (running the circuit)
		// produces.
		let mut wg2 = WitnessGenerator::new(&layout);
		let a_w = wg2.write_inout(a, a_val);
		let b_w = wg2.write_inout(b, b_val);
		let s_w = wg2.write_precommit(s, s_val);
		let expected_w = wg2.write_inout(expected, expected_val);
		mixed_circuit(&mut wg2, a_w, b_w, s_w, expected_w);
		let expected_witness = wg2.build().expect("witness generation should succeed");

		assert_eq!(witness.public(), expected_witness.public());
		assert_eq!(witness.precommit(), expected_witness.precommit());
		assert_eq!(witness.private(), expected_witness.private());
	}

	#[test]
	fn test_apply_gates_is_repeatable() {
		// A stored gate sequence is replayed on every prove call, so replay must be idempotent (the
		// `Fn` evals must not consume state).
		let mut rb = GateRecordingConstraintBuilder::<B128>::new();
		let a = rb.alloc_inout();
		let b = rb.alloc_inout();
		let _ = rb.mul(a, b);
		let (ir, gates) = rb.into_parts();
		let (_cs, layout) = run_compile(ir);

		let replay = |a_val, b_val| {
			let mut wg = WitnessGenerator::new(&layout);
			wg.write_inout(a, a_val);
			wg.write_inout(b, b_val);
			wg.apply_gates(&gates);
			wg.build().expect("witness generation should succeed")
		};

		let w1 = replay(B128::new(3), B128::new(5));
		let w2 = replay(B128::new(9), B128::new(2));
		// Same inputs give the same witness (the sequence reads its inputs fresh, no consumed
		// state); different inputs give different witnesses.
		assert_eq!(w1.public(), replay(B128::new(3), B128::new(5)).public());
		assert_ne!(w1.public(), w2.public());
	}

	#[test]
	fn test_apply_gates_compute_public() {
		// A compute_public wire whose value is an opaque function of two inout inputs, then
		// asserted equal to a third inout. Mirrors how `compute_public_value` materializes a
		// derived public.
		let x_val = B128::new(11);
		let y_val = B128::new(13);
		let prod = x_val * y_val + B128::ONE;

		let mut rb = GateRecordingConstraintBuilder::<B128>::new();
		let x = rb.alloc_inout();
		let y = rb.alloc_inout();
		let z = rb.alloc_inout();
		let computed = rb.hint_varsize(&[x, y], 1, |vals| vec![vals[0] * vals[1] + B128::ONE])[0];
		rb.assert_eq(computed, z);
		let (ir, gates) = rb.into_parts();
		let (cs, layout) = run_compile(ir);

		// Only the three inout wires (x, y, z) are written externally; the compute_public wire is a
		// derived public computed by its Generic gate.
		let mut wg = WitnessGenerator::new(&layout);
		wg.write_inout(x, x_val);
		wg.write_inout(y, y_val);
		wg.write_inout(z, prod);
		wg.apply_gates(&gates);
		let witness = wg.build().expect("witness generation should succeed");
		cs.validate(&witness);
	}

	// The IR already came from GateRecordingConstraintBuilder, so run the same wire-elimination +
	// finalize pipeline `compile` would.
	fn run_compile(ir: ConstraintSystemIR<B128>) -> (ConstraintSystem<B128>, WitnessLayout<B128>) {
		let ir = run_wire_elimination(CostModel::default(), ir);
		ir.finalize()
	}
}
