// Copyright 2026 The Binius Developers

//! Recorded gate sequence and its replay into an outer witness (BINIUS-43).
//!
//! Instead of re-running the inner IOP verifier to populate the outer constraint system's witness
//! (the job of `binius_spartan_prover::wrapper::ReplayChannel`), the symbolic build records a flat
//! [`GateSequence`] of the high-level field operations it performs. Replaying that sequence against
//! concrete inputs reproduces every wire value, so the prover can fill the witness — and the
//! verifier can derive the public (InOut) values — without a second pass through the verifier.
//!
//! # Wire identities
//!
//! Every value produced during the symbolic build is assigned a [`RecWire`] (a dense `usize`
//! index, independent of constraint-system wire allocation and of the lazy InOut materialization
//! scheme). Operations reference their inputs via [`Val`], which is either an earlier [`RecWire`]
//! or an inline constant.
//!
//! A [`RecWire`] that ends up backed by a real constraint-system wire also carries a
//! [`ConstraintWire`]:
//!
//! * private wires allocated by `ConstraintBuilder::{add, mul, hint}` ([`Gate::Add`],
//!   [`Gate::Mul`], [`Gate::Hint`]),
//! * the precommit (OTP key) wire ([`Gate::Precommit`]),
//! * an InOut wire that the symbolic build materialized lazily ([`Gate::Materialize`]).
//!
//! Replay computes a value for *every* `RecWire` (so later gates can consume it as an input), but
//! only writes a value into the witness when the wire's [`ConstraintWire`] is present in the
//! [`WitnessLayout`]. Wires pruned by `wire_elimination` are absent from the layout
//! ([`WitnessLayout::get`] returns `None`), so their writes are silently dropped — exactly as the
//! existing `WitnessGenerator::write_value` does today. This keeps replay correct across the
//! post-symbolic-build wire-elimination pass even though the gate sequence references
//! pre-elimination wire ids.

use binius_field::Field;
use binius_spartan_frontend::constraint_system::{
	ConstraintWire, Witness, WitnessLayout, WitnessSegment,
};

/// Dense index of a value produced during the symbolic build.
pub type RecWire = usize;

/// An operand of a [`Gate`]: either an earlier recorded wire or an inline constant.
#[derive(Debug, Clone, Copy)]
pub enum Val<F> {
	Wire(RecWire),
	Const(F),
}

/// Field-level evaluation of a recorded operation: maps input values to output values.
///
/// Captured at recording time for operations whose computation is not a single inline arithmetic
/// step (public-only short-circuits and hints).
pub type EvalFn<F> = Box<dyn Fn(&[F]) -> Vec<F>>;

/// A single recorded high-level operation.
///
/// Boxed closures (`Gate::Public`, `Gate::Hint`) carry the field-level computation needed to
/// reproduce output values at replay time; the simple arithmetic gates are evaluated inline.
pub enum Gate<F> {
	/// An external value entering the channel (`recv_one` / `sample` / `observe_one`). Its value is
	/// supplied from the replay's interaction stream, in recording order.
	Input { out: RecWire },
	/// The precommit one-time-pad key. Its value is supplied from the replay's key stream, in
	/// recording order, and written to the precommit witness segment.
	Precommit { out: RecWire, cw: ConstraintWire },
	/// Public-only computation (the symbolic build's `is_inout` short-circuit): `outs = f(ins)`.
	/// The result is a lazy InOut value with no constraint-system wire yet; it gains one only via a
	/// later [`Gate::Materialize`].
	Public {
		ins: Vec<Val<F>>,
		outs: Vec<RecWire>,
		f: EvalFn<F>,
	},
	/// `out = a + b`, allocating a private wire (`ConstraintBuilder::add`). In characteristic 2
	/// this also covers subtraction.
	Add {
		a: Val<F>,
		b: Val<F>,
		out: RecWire,
		cw: ConstraintWire,
	},
	/// `out = a * b`, allocating a private wire (`ConstraintBuilder::mul`).
	Mul {
		a: Val<F>,
		b: Val<F>,
		out: RecWire,
		cw: ConstraintWire,
	},
	/// A hint (`ConstraintBuilder::hint`): allocates one private wire per output and fills them
	/// with `f(ins)`.
	Hint {
		ins: Vec<Val<F>>,
		outs: Vec<(RecWire, ConstraintWire)>,
		f: EvalFn<F>,
	},
	/// A lazy public ([`Gate::Public`] / [`Gate::Input`]) wire was materialized into an InOut
	/// constraint-system wire; write its already-computed value to the InOut segment.
	Materialize { rec: RecWire, cw: ConstraintWire },
}

/// A flat, topologically-ordered recording of the symbolic build's field operations.
#[derive(Default)]
pub struct GateSequence<F> {
	/// Number of distinct [`RecWire`] ids referenced by the gates.
	pub n_wires: usize,
	pub gates: Vec<Gate<F>>,
}

impl<F: Field> GateSequence<F> {
	pub fn new() -> Self {
		Self {
			n_wires: 0,
			gates: Vec::new(),
		}
	}

	/// Allocate a fresh [`RecWire`].
	pub fn alloc(&mut self) -> RecWire {
		let id = self.n_wires;
		self.n_wires += 1;
		id
	}

	pub fn push(&mut self, gate: Gate<F>) {
		self.gates.push(gate);
	}

	/// Replay the sequence into a [`Witness`].
	///
	/// `inputs` supplies the [`Gate::Input`] values in recording order (for the prover, the
	/// recorded encrypted interaction values); `keys` supplies the [`Gate::Precommit`] OTP keys in
	/// recording order. `layout` is the *post-elimination* witness layout — writes to wires it
	/// does not contain are dropped.
	pub fn replay(
		&self,
		layout: &WitnessLayout<F>,
		mut inputs: impl Iterator<Item = F>,
		mut keys: impl Iterator<Item = F>,
	) -> Witness<F> {
		let mut values: Vec<F> = vec![F::ZERO; self.n_wires];

		let mut public: Vec<F> = vec![F::ZERO; layout.public_size()];
		public[..layout.n_constants()].copy_from_slice(layout.constants());
		let mut precommit: Vec<F> = vec![F::ZERO; layout.precommit_size()];
		let mut private: Vec<F> = vec![F::ZERO; layout.private_size()];

		let val = |values: &[F], v: &Val<F>| -> F {
			match v {
				Val::Wire(id) => values[*id],
				Val::Const(c) => *c,
			}
		};

		let write = |public: &mut [F],
		             precommit: &mut [F],
		             private: &mut [F],
		             cw: &ConstraintWire,
		             value: F| {
			if let Some(index) = layout.get(cw) {
				match index.segment {
					WitnessSegment::Public => public[index.index as usize] = value,
					WitnessSegment::Precommit => precommit[index.index as usize] = value,
					WitnessSegment::Private => private[index.index as usize] = value,
				}
			}
		};

		for gate in &self.gates {
			match gate {
				Gate::Input { out } => {
					values[*out] = inputs.next().expect("interaction stream exhausted");
				}
				Gate::Precommit { out, cw } => {
					let value = keys.next().expect("key stream exhausted");
					values[*out] = value;
					write(&mut public, &mut precommit, &mut private, cw, value);
				}
				Gate::Public { ins, outs, f } => {
					let in_vals = ins.iter().map(|v| val(&values, v)).collect::<Vec<_>>();
					let out_vals = f(&in_vals);
					debug_assert_eq!(out_vals.len(), outs.len());
					for (out, value) in outs.iter().zip(out_vals) {
						values[*out] = value;
					}
				}
				Gate::Add { a, b, out, cw } => {
					let value = val(&values, a) + val(&values, b);
					values[*out] = value;
					write(&mut public, &mut precommit, &mut private, cw, value);
				}
				Gate::Mul { a, b, out, cw } => {
					let value = val(&values, a) * val(&values, b);
					values[*out] = value;
					write(&mut public, &mut precommit, &mut private, cw, value);
				}
				Gate::Hint { ins, outs, f } => {
					let in_vals = ins.iter().map(|v| val(&values, v)).collect::<Vec<_>>();
					let out_vals = f(&in_vals);
					debug_assert_eq!(out_vals.len(), outs.len());
					for ((out, cw), value) in outs.iter().zip(out_vals) {
						values[*out] = value;
						write(&mut public, &mut precommit, &mut private, cw, value);
					}
				}
				Gate::Materialize { rec, cw } => {
					write(&mut public, &mut precommit, &mut private, cw, values[*rec]);
				}
			}
		}

		Witness::new(public, precommit, private)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128bGhash as B128, Field};
	use binius_spartan_frontend::constraint_system::{ConstraintWire, WitnessLayout};

	use super::*;

	/// Build a small sequence modelling: receive encrypted `a`, precommit key `k`, decrypt
	/// `p = a - k` (private), then square `q = p * p` (private). Replay and check the witness.
	fn build_seq() -> (GateSequence<B128>, B128, B128) {
		let a_enc = B128::new(7);
		let k = B128::new(3);

		let mut seq = GateSequence::new();
		let w_a = seq.alloc(); // encrypted input (lazy InOut)
		let w_k = seq.alloc(); // precommit key
		let w_p = seq.alloc(); // plaintext = a - k (private 0)
		let w_q = seq.alloc(); // p*p (private 1)

		seq.push(Gate::Input { out: w_a });
		seq.push(Gate::Precommit {
			out: w_k,
			cw: ConstraintWire::precommit(0),
		});
		// The encrypted value is materialized into InOut wire 0 when consumed by the private sub.
		seq.push(Gate::Materialize {
			rec: w_a,
			cw: ConstraintWire::inout(0),
		});
		// p = a - k (== a + k in char 2), private wire 0.
		seq.push(Gate::Add {
			a: Val::Wire(w_a),
			b: Val::Wire(w_k),
			out: w_p,
			cw: priv_wire(0),
		});
		// q = p * p, private wire 1.
		seq.push(Gate::Mul {
			a: Val::Wire(w_p),
			b: Val::Wire(w_p),
			out: w_q,
			cw: priv_wire(1),
		});

		(seq, a_enc, k)
	}

	fn priv_wire(id: u32) -> ConstraintWire {
		ConstraintWire::private(id)
	}

	#[test]
	fn replay_fills_witness() {
		let (seq, a_enc, k) = build_seq();
		// constants = [ONE]; 1 inout; 1 precommit; 2 private, all alive.
		let layout = WitnessLayout::<B128>::sparse(vec![B128::ONE], 1, 1, &[true, true]);

		let witness = seq.replay(&layout, [a_enc].into_iter(), [k].into_iter());

		let p = a_enc + k;
		let q = p * p;
		// public = [const ONE, inout a_enc]
		assert_eq!(witness.public()[1], a_enc);
		assert_eq!(witness.precommit()[0], k);
		assert_eq!(witness.private()[0], p);
		assert_eq!(witness.private()[1], q);
	}

	#[test]
	fn replay_drops_pruned_wire() {
		let (seq, a_enc, k) = build_seq();
		// Prune private wire 1 (q): private_alive = [true, false].
		let layout = WitnessLayout::<B128>::sparse(vec![B128::ONE], 1, 1, &[true, false]);

		// Replay must not panic and must still produce p in private[0]; q's write is dropped.
		let witness = seq.replay(&layout, [a_enc].into_iter(), [k].into_iter());
		let p = a_enc + k;
		assert_eq!(witness.private().len(), 1);
		assert_eq!(witness.private()[0], p);
	}
}
