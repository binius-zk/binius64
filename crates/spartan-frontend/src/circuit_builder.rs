// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, backtrace::Backtrace, collections::HashMap, mem};

use binius_field::Field;
use bytemuck::zeroed_vec;
use smallvec::{SmallVec, smallvec};

use crate::constraint_system::{
	ConstraintSystem, ConstraintWire, MulConstraint, Operand, WireKind, Witness, WitnessIndex,
	WitnessLayout, WitnessSegment,
};

/// Common interface for circuit construction and witness generation.
///
/// This trait enables writing circuit logic once that works for both abstract circuit
/// expression ([`ConstraintBuilder`]) and concrete witness generation ([`WitnessGenerator`]).
/// The same function can build constraints symbolically or evaluate them with concrete values.
pub trait CircuitBuilder {
	type Wire: Copy;
	type Field: Field;

	fn assert_zero(&mut self, wire: Self::Wire);

	fn assert_eq(&mut self, lhs: Self::Wire, rhs: Self::Wire) {
		let diff = self.add(lhs, rhs);
		self.assert_zero(diff);
	}

	fn constant(&mut self, val: Self::Field) -> Self::Wire;

	fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

	// Addition is subtraction in characteristic 2.
	fn sub(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		self.add(lhs, rhs)
	}

	fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

	fn hint<H: Fn([Self::Field; IN]) -> [Self::Field; OUT], const IN: usize, const OUT: usize>(
		&mut self,
		inputs: [Self::Wire; IN],
		f: H,
	) -> [Self::Wire; OUT];

	/// Runtime-arity variant of [`hint`](Self::hint): takes a slice of input wires and an
	/// `out_len`, with a closure mapping the input values to `out_len` output values. Emits no
	/// constraints; the outputs are public-derivable (allocated in the derived segment) iff every
	/// input is, matching [`hint`](Self::hint). The closure is invoked at most once —
	/// value-generating builders run it, the symbolic builder skips it.
	fn hint_varsize<H: FnOnce(&[Self::Field]) -> Vec<Self::Field>>(
		&mut self,
		inputs: &[Self::Wire],
		out_len: usize,
		f: H,
	) -> Vec<Self::Wire>;
}

#[derive(Debug)]
pub struct WireAllocator {
	n_wires: u32,
	kind: WireKind,
}

impl WireAllocator {
	pub const fn new(kind: WireKind) -> Self {
		WireAllocator { n_wires: 0, kind }
	}

	pub const fn alloc(&mut self) -> ConstraintWire {
		let wire = ConstraintWire {
			kind: self.kind,
			id: self.n_wires,
		};
		self.n_wires += 1;
		wire
	}
}

// TODO: Add string labels for constraints to make validation easier.

// Witness values are a permuted subset of the wire values.
// Need a way to fingerprint a constraint system.

/// Status of a private wire during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireStatus {
	/// Wire is unknown - not yet analyzed for elimination
	Unknown,
	/// Wire is pinned - explicitly kept alive
	Pinned,
	/// Wire has been pruned (eliminated)
	Pruned,
}

/// Intermediate representation of a constraint system that can be manipulated and optimized.
///
/// This IR is used during circuit construction and optimization passes like wire elimination.
/// It tracks wire allocators, constants, and constraints, along with metadata about which
/// private wires are still alive (not eliminated by optimization).
#[derive(Debug)]
pub struct ConstraintSystemIR<F: Field> {
	pub(crate) constant_alloc: WireAllocator,
	pub(crate) public_alloc: WireAllocator,
	pub(crate) derived_alloc: WireAllocator,
	pub(crate) precommit_alloc: WireAllocator,
	pub(crate) private_alloc: WireAllocator,
	pub(crate) constants: HashMap<F, u32>,
	pub(crate) zero_constraints: Vec<Operand<ConstraintWire>>,
	pub(crate) mul_constraints: Vec<MulConstraint<ConstraintWire>>,
	/// Tracks the status of private wires (Unknown, Pinned, or Pruned).
	/// Index corresponds to private wire ID. Initially all wires are Unknown.
	pub(crate) private_wires_status: Vec<WireStatus>,
}

impl<F: Field> ConstraintSystemIR<F> {
	/// Finalize the IR into a ConstraintSystem and WitnessLayout by converting remaining
	/// zero constraints to MulConstraints, computing the final witness layout, and mapping all
	/// ConstraintWires to WitnessIndices.
	///
	/// Internally looks up or allocates a constant wire with value 1, used to convert
	/// zero constraints of the form `A = 0` into MulConstraints `A * 1 = 0`.
	pub fn finalize(mut self) -> (ConstraintSystem<F>, WitnessLayout<F>) {
		// Look up or allocate a constant wire for ONE
		let one_id = self
			.constants
			.entry(F::ONE)
			.or_insert_with(|| self.constant_alloc.alloc().id);
		let one_wire = ConstraintWire {
			kind: WireKind::Constant,
			id: *one_id,
		};

		// Capture the zero constant wire id before consuming self.constants
		let zero_const_id: Option<u32> = self.constants.get(&F::ZERO).copied();

		// Convert constants HashMap to Vec
		let mut constants = zeroed_vec(self.constant_alloc.n_wires as usize);
		for (val, id) in self.constants {
			constants[id as usize] = val;
		}

		// Replace all remaining zero constraints with mul constraints
		let one_operand = Operand::from(one_wire);
		let zero_operand = Operand::default();
		for operand in mem::take(&mut self.zero_constraints) {
			if !operand.is_empty() {
				self.mul_constraints.push(MulConstraint {
					a: operand,
					b: one_operand.clone(),
					c: zero_operand.clone(),
				});
			}
		}

		// Create private_alive array from wire status (invert pruned logic)
		let private_alive: Vec<bool> = self
			.private_wires_status
			.iter()
			.map(|&status| !matches!(status, WireStatus::Pruned))
			.collect();

		// A derived wire needs a public slot iff it is referenced by a surviving constraint. Scan
		// the (now fully folded) mul constraints for derived ids; intermediate derived wires that
		// feed only other derived wires are referenced nowhere and get no slot — the generators
		// compute them inline and `layout.get` returns `None`.
		let mut derived_alive = vec![false; self.derived_alloc.n_wires as usize];
		for MulConstraint { a, b, c } in &self.mul_constraints {
			for operand in [a, b, c] {
				for wire in operand.wires() {
					if wire.kind == WireKind::Derived {
						derived_alive[wire.id as usize] = true;
					}
				}
			}
		}

		// Create WitnessLayout
		let layout = WitnessLayout::sparse(
			constants.clone(),
			self.public_alloc.n_wires,
			self.precommit_alloc.n_wires,
			&derived_alive,
			&private_alive,
		);

		// Map all ConstraintWire to WitnessIndex, filtering out zero constant wires.
		let map_operand = |operand: &Operand<ConstraintWire>| -> Operand<WitnessIndex> {
			let indices: SmallVec<[WitnessIndex; 4]> = operand
				.wires()
				.iter()
				.filter(|wire| !(wire.kind == WireKind::Constant && zero_const_id == Some(wire.id)))
				.filter_map(|wire| layout.get(wire))
				.collect();
			Operand::new(indices)
		};

		let mul_constraints = self
			.mul_constraints
			.iter()
			.map(|constraint| MulConstraint {
				a: map_operand(&constraint.a),
				b: map_operand(&constraint.b),
				c: map_operand(&constraint.c),
			})
			.collect();

		// Map one_wire to WitnessIndex
		let one_wire_index = layout
			.get(&one_wire)
			.expect("one_wire constant should exist in layout")
			.index;

		let cs = ConstraintSystem::new(
			constants,
			layout.n_inout() as u32,
			layout.n_precommit() as u32,
			layout.n_private() as u32,
			layout.log_public(),
			mul_constraints,
			one_wire_index,
		);

		(cs, layout)
	}
}

/// Builds constraint systems symbolically by recording operations as constraints.
///
/// Implements [`CircuitBuilder`] with [`ConstraintWire`] as the wire type. Operations like
/// `add` and `mul` allocate new wires and record constraints without evaluating values.
#[derive(Debug)]
pub struct ConstraintBuilder<F: Field> {
	ir: ConstraintSystemIR<F>,
}

impl<F: Field> ConstraintBuilder<F> {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		ConstraintBuilder {
			ir: ConstraintSystemIR {
				constant_alloc: WireAllocator::new(WireKind::Constant),
				public_alloc: WireAllocator::new(WireKind::InOut),
				derived_alloc: WireAllocator::new(WireKind::Derived),
				precommit_alloc: WireAllocator::new(WireKind::Precommit),
				private_alloc: WireAllocator::new(WireKind::Private),
				constants: HashMap::new(),
				zero_constraints: Vec::new(),
				mul_constraints: Vec::new(),
				private_wires_status: Vec::new(),
			},
		}
	}

	pub const fn alloc_inout(&mut self) -> ConstraintWire {
		self.ir.public_alloc.alloc()
	}

	pub const fn alloc_precommit(&mut self) -> ConstraintWire {
		self.ir.precommit_alloc.alloc()
	}

	pub fn build(self) -> ConstraintSystemIR<F> {
		self.ir
	}
}

impl<F: Field> CircuitBuilder for ConstraintBuilder<F> {
	type Wire = ConstraintWire;
	type Field = F;

	fn assert_zero(&mut self, wire: Self::Wire) {
		self.ir.zero_constraints.push(wire.into())
	}

	fn assert_eq(&mut self, lhs: Self::Wire, rhs: Self::Wire) {
		self.ir
			.zero_constraints
			.push(Operand::new(smallvec![lhs, rhs]));
	}

	fn constant(&mut self, val: F) -> Self::Wire {
		let id = self
			.ir
			.constants
			.entry(val)
			.or_insert_with(|| self.ir.constant_alloc.alloc().id);
		ConstraintWire {
			kind: WireKind::Constant,
			id: *id,
		}
	}

	fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		// If both inputs are public-derivable, the sum is too: allocate a derived wire in the
		// public segment and emit no constraint. The verifier recomputes it itself.
		if lhs.kind.is_public() && rhs.kind.is_public() {
			return self.ir.derived_alloc.alloc();
		}
		let out = self.ir.private_alloc.alloc();
		self.ir.private_wires_status.push(WireStatus::Unknown);
		self.ir
			.zero_constraints
			.push(Operand::new(smallvec![lhs, rhs, out]));
		out
	}

	fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		// If both inputs are public-derivable, the product is too: allocate a derived wire and emit
		// no multiplication constraint.
		if lhs.kind.is_public() && rhs.kind.is_public() {
			return self.ir.derived_alloc.alloc();
		}
		let out = self.ir.private_alloc.alloc();
		self.ir.private_wires_status.push(WireStatus::Unknown);
		self.ir.mul_constraints.push(MulConstraint {
			a: lhs.into(),
			b: rhs.into(),
			c: out.into(),
		});
		out
	}

	fn hint<H: Fn([F; IN]) -> [F; OUT], const IN: usize, const OUT: usize>(
		&mut self,
		inputs: [Self::Wire; IN],
		_f: H,
	) -> [Self::Wire; OUT] {
		// A hint over only public-derivable inputs is itself public-derivable. (Hints emit no
		// constraints regardless; only the allocator choice changes.)
		let derived = inputs.iter().all(|wire| wire.kind.is_public());
		array::from_fn(|_| {
			if derived {
				self.ir.derived_alloc.alloc()
			} else {
				let wire = self.ir.private_alloc.alloc();
				self.ir.private_wires_status.push(WireStatus::Unknown);
				wire
			}
		})
	}

	fn hint_varsize<H: FnOnce(&[F]) -> Vec<F>>(
		&mut self,
		inputs: &[Self::Wire],
		out_len: usize,
		_f: H,
	) -> Vec<Self::Wire> {
		// A hint over only public-derivable inputs is itself public-derivable. (Hints emit no
		// constraints regardless; only the allocator choice changes.)
		let derived = inputs.iter().all(|wire| wire.kind.is_public());
		(0..out_len)
			.map(|_| {
				if derived {
					self.ir.derived_alloc.alloc()
				} else {
					let wire = self.ir.private_alloc.alloc();
					self.ir.private_wires_status.push(WireStatus::Unknown);
					wire
				}
			})
			.collect()
	}
}

#[derive(Debug)]
pub struct WitnessError {
	pub backtrace: Backtrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessWire<F: Field> {
	val: F,
	segment: WitnessSegment,
}

impl<F: Field> WitnessWire<F> {
	#[inline]
	pub const fn val(self) -> F {
		self.val
	}

	/// Whether this wire's value lives in the public segment, i.e. is public-derivable. Drives the
	/// derived-vs-private allocation choice, matching [`ConstraintBuilder`].
	#[inline]
	fn is_public(self) -> bool {
		self.segment == WitnessSegment::Public
	}
}

/// Generates witness values by evaluating circuit operations with concrete field elements.
///
/// Implements [`CircuitBuilder`] with [`WitnessWire`] as the wire type. Operations like
/// `add` and `mul` compute actual field values and populate the witness array. Captures
/// the first constraint violation as an error for debugging.
#[derive(Debug)]
pub struct WitnessGenerator<'a, F: Field> {
	derived_alloc: WireAllocator,
	private_alloc: WireAllocator,
	public: Vec<F>,
	precommit: Vec<F>,
	private: Vec<F>,
	layout: &'a WitnessLayout<F>,
	first_error: Option<Backtrace>,
}

impl<'a, F: Field> WitnessGenerator<'a, F> {
	pub fn new(layout: &'a WitnessLayout<F>) -> Self {
		let mut public = zeroed_vec(layout.public_size());
		public[..layout.constants.len()].copy_from_slice(&layout.constants);

		let precommit = zeroed_vec(layout.precommit_size());
		let private = zeroed_vec(layout.private_size());

		Self {
			derived_alloc: WireAllocator::new(WireKind::Derived),
			private_alloc: WireAllocator::new(WireKind::Private),
			public,
			precommit,
			private,
			layout,
			first_error: None,
		}
	}

	/// Allocates the output wire of an op, choosing the derived or private allocator from the input
	/// kinds (matching [`ConstraintBuilder`]), and writes its value.
	fn alloc_op_value(&mut self, all_derivable: bool, value: F) -> WitnessWire<F> {
		let wire = if all_derivable {
			self.derived_alloc.alloc()
		} else {
			self.private_alloc.alloc()
		};
		self.write_value(wire, value)
	}

	fn write_value(&mut self, wire: ConstraintWire, value: F) -> WitnessWire<F> {
		if let Some(index) = self.layout.get(&wire) {
			match index.segment {
				WitnessSegment::Public => self.public[index.index as usize] = value,
				WitnessSegment::Precommit => self.precommit[index.index as usize] = value,
				WitnessSegment::Private => self.private[index.index as usize] = value,
			}
		}
		WitnessWire {
			val: value,
			segment: wire.kind.segment(),
		}
	}

	pub fn write_inout(&mut self, wire: ConstraintWire, value: F) -> WitnessWire<F> {
		assert_eq!(wire.kind, WireKind::InOut);
		self.write_value(wire, value)
	}

	pub fn write_precommit(&mut self, wire: ConstraintWire, value: F) -> WitnessWire<F> {
		assert_eq!(wire.kind, WireKind::Precommit);
		self.write_value(wire, value)
	}

	pub fn build(self) -> Result<Witness<F>, WitnessError> {
		if let Some(backtrace) = self.first_error {
			Err(WitnessError { backtrace })
		} else {
			Ok(Witness::new(self.public, self.precommit, self.private))
		}
	}

	pub const fn error(&self) -> Option<&Backtrace> {
		self.first_error.as_ref()
	}

	fn record_error(&mut self) {
		if self.first_error.is_none() {
			self.first_error = Some(Backtrace::capture());
		}
	}
}

impl<'a, F: Field> CircuitBuilder for WitnessGenerator<'a, F> {
	type Wire = WitnessWire<F>;
	type Field = F;

	fn assert_zero(&mut self, wire: Self::Wire) {
		if wire.val() != F::ZERO {
			self.record_error();
		}
	}

	fn assert_eq(&mut self, lhs: Self::Wire, rhs: Self::Wire) {
		if lhs.val() != rhs.val() {
			self.record_error();
		}
	}

	fn constant(&mut self, val: F) -> Self::Wire {
		WitnessWire {
			val,
			segment: WitnessSegment::Public,
		}
	}

	fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		let all_derivable = lhs.is_public() && rhs.is_public();
		self.alloc_op_value(all_derivable, lhs.val() + rhs.val())
	}

	fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		let all_derivable = lhs.is_public() && rhs.is_public();
		self.alloc_op_value(all_derivable, lhs.val() * rhs.val())
	}

	fn hint<H: Fn([F; IN]) -> [F; OUT], const IN: usize, const OUT: usize>(
		&mut self,
		inputs: [Self::Wire; IN],
		f: H,
	) -> [Self::Wire; OUT] {
		let all_derivable = inputs.iter().all(|wire| wire.is_public());
		f(inputs.map(WitnessWire::val)).map(|value| self.alloc_op_value(all_derivable, value))
	}

	fn hint_varsize<H: FnOnce(&[F]) -> Vec<F>>(
		&mut self,
		inputs: &[Self::Wire],
		out_len: usize,
		f: H,
	) -> Vec<Self::Wire> {
		let all_derivable = inputs.iter().all(|wire| wire.is_public());
		let input_vals: Vec<F> = inputs.iter().map(|wire| wire.val()).collect();
		let outputs = f(&input_vals);
		debug_assert_eq!(
			outputs.len(),
			out_len,
			"hint_varsize closure returned wrong output count"
		);
		outputs
			.into_iter()
			.map(|value| self.alloc_op_value(all_derivable, value))
			.collect()
	}
}

/// Wire type for [`InstanceGenerator`]: holds the field value of a public wire (a constant, inout,
/// or derived value the verifier can compute) and is `None` for private/precommit wires, whose
/// values the verifier does not know. A wire being `Some` mirrors [`WireKind::is_public`], so the
/// derived-vs-private branching stays aligned with the other builders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicWire<F: Field>(Option<F>);

impl<F: Field> PublicWire<F> {
	/// The wire's value if it is public-derivable (constant, inout, or derived), else `None` for a
	/// private/precommit wire whose value the verifier does not know.
	#[inline]
	pub const fn value(self) -> Option<F> {
		self.0
	}
}

/// Reconstructs the public input vector `[constants | inout | derived]` on the verifier side by
/// re-running the circuit function.
///
/// Implements [`CircuitBuilder`] with [`PublicWire`] as the wire type, structurally mirroring
/// [`WitnessGenerator`], but it only computes the *public* segment. The verifier lacks the secret
/// (precommit) values, so private wires carry no value; by the soundness invariant a derived wire
/// never reads a secret, so every derived value is computable from the real public inputs alone.
/// Derived wires are allocated in the same order as [`ConstraintBuilder`], keeping their ids
/// aligned so [`WitnessLayout::get`] resolves each to the correct public slot.
#[derive(Debug)]
pub struct InstanceGenerator<'a, F: Field> {
	derived_alloc: WireAllocator,
	public: Vec<F>,
	layout: &'a WitnessLayout<F>,
}

impl<'a, F: Field> InstanceGenerator<'a, F> {
	pub fn new(layout: &'a WitnessLayout<F>) -> Self {
		let mut public = zeroed_vec(layout.public_size());
		public[..layout.constants.len()].copy_from_slice(&layout.constants);

		Self {
			derived_alloc: WireAllocator::new(WireKind::Derived),
			public,
			layout,
		}
	}

	/// Writes a real inout value into its public slot.
	pub fn write_inout(&mut self, wire: ConstraintWire, value: F) -> PublicWire<F> {
		assert_eq!(wire.kind, WireKind::InOut);
		self.write_public(wire, value)
	}

	/// Records a precommit wire as private (value unknown to the verifier). Per the soundness
	/// invariant a secret never feeds a derived value, so the missing value is inert.
	pub fn placeholder_precommit(&mut self, wire: ConstraintWire) -> PublicWire<F> {
		assert_eq!(wire.kind, WireKind::Precommit);
		PublicWire(None)
	}

	/// Writes `value` to the wire's public slot if it has one (i.e. it is alive), then returns the
	/// wire carrying its value. Wires with no slot (pruned intermediates) are computed inline only.
	fn write_public(&mut self, wire: ConstraintWire, value: F) -> PublicWire<F> {
		if let Some(index) = self.layout.get(&wire) {
			debug_assert_eq!(index.segment, WitnessSegment::Public);
			self.public[index.index as usize] = value;
		}
		PublicWire(Some(value))
	}

	/// The reconstructed public vector `[constants | inout | derived]` so far, of length
	/// `1 << layout.log_public()`. Once every inout and alive-derived wire has been written, this
	/// is the final public vector; reading it by reference (rather than consuming via
	/// [`Self::build`]) lets callers keep the generator alive for wires that still reference it.
	pub fn public(&self) -> &[F] {
		&self.public
	}

	/// Returns the reconstructed public vector `[constants | inout | derived]`, of length
	/// `1 << layout.log_public()`, ready to pass to `Verifier::verify`.
	pub fn build(self) -> Vec<F> {
		self.public
	}
}

impl<'a, F: Field> CircuitBuilder for InstanceGenerator<'a, F> {
	type Wire = PublicWire<F>;
	type Field = F;

	// Assertions are enforced as ordinary public-segment constraints by the verifier, so they are
	// no-ops here. They must not allocate, to stay wire-id-aligned with the other builders.
	fn assert_zero(&mut self, _wire: Self::Wire) {}

	fn assert_eq(&mut self, _lhs: Self::Wire, _rhs: Self::Wire) {}

	fn constant(&mut self, val: F) -> Self::Wire {
		PublicWire(Some(val))
	}

	fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		match (lhs.0, rhs.0) {
			(Some(lhs), Some(rhs)) => {
				let wire = self.derived_alloc.alloc();
				self.write_public(wire, lhs + rhs)
			}
			_ => PublicWire(None),
		}
	}

	fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
		match (lhs.0, rhs.0) {
			(Some(lhs), Some(rhs)) => {
				let wire = self.derived_alloc.alloc();
				self.write_public(wire, lhs * rhs)
			}
			_ => PublicWire(None),
		}
	}

	fn hint<H: Fn([F; IN]) -> [F; OUT], const IN: usize, const OUT: usize>(
		&mut self,
		inputs: [Self::Wire; IN],
		f: H,
	) -> [Self::Wire; OUT] {
		// Only invoke `f` when every input is public (its value is known); a non-derived hint would
		// receive value-less inputs, so `f` is skipped to avoid panics on missing values.
		if inputs.iter().all(|wire| wire.0.is_some()) {
			let values = inputs.map(|wire| wire.0.expect("every input checked public above"));
			f(values).map(|value| {
				let wire = self.derived_alloc.alloc();
				self.write_public(wire, value)
			})
		} else {
			[PublicWire(None); OUT]
		}
	}

	fn hint_varsize<H: FnOnce(&[F]) -> Vec<F>>(
		&mut self,
		inputs: &[Self::Wire],
		out_len: usize,
		f: H,
	) -> Vec<Self::Wire> {
		// Only invoke `f` when every input is public (its value is known); otherwise the outputs
		// are private (value-less to the verifier) and allocate no derived wires, matching the
		// other builders' private branch.
		if inputs.iter().all(|wire| wire.0.is_some()) {
			let input_vals: Vec<F> = inputs
				.iter()
				.map(|wire| wire.0.expect("every input checked public above"))
				.collect();
			let outputs = f(&input_vals);
			debug_assert_eq!(
				outputs.len(),
				out_len,
				"hint_varsize closure returned wrong output count"
			);
			outputs
				.into_iter()
				.map(|value| {
					let wire = self.derived_alloc.alloc();
					self.write_public(wire, value)
				})
				.collect()
		} else {
			vec![PublicWire(None); out_len]
		}
	}
}

#[cfg(test)]
mod tests {
	use std::iter::successors;

	use binius_field::{BinaryField128bGhash as B128, Field, PackedField};

	use super::*;

	fn fibonacci<Builder: CircuitBuilder>(
		builder: &mut Builder,
		x0: Builder::Wire,
		x1: Builder::Wire,
		n: usize,
	) -> Builder::Wire {
		if n == 0 {
			return x0;
		}

		let (_xnsub1, xn) = successors(Some((x0, x1)), |&(a, b)| {
			let next = builder.mul(a, b);
			Some((b, next))
		})
		.nth(n - 1)
		.expect("closure always returns Some");

		xn
	}

	#[test]
	fn test_fibonacci() {
		let mut constraint_builder = ConstraintBuilder::new();
		let x0 = constraint_builder.alloc_inout();
		let x1 = constraint_builder.alloc_inout();
		let xn = constraint_builder.alloc_inout();
		let out = fibonacci(&mut constraint_builder, x0, x1, 20);
		constraint_builder.assert_eq(out, xn);
		let ir = constraint_builder.build();
		let (constraint_system, layout) = ir.finalize();

		let mut witness_generator = WitnessGenerator::new(&layout);
		let x0 = witness_generator.write_inout(x0, B128::ONE);
		let x1 = witness_generator.write_inout(x1, B128::MULTIPLICATIVE_GENERATOR);
		let xn = witness_generator.write_inout(xn, B128::MULTIPLICATIVE_GENERATOR.pow(6765));
		let out = fibonacci(&mut witness_generator, x0, x1, 20);
		witness_generator.assert_eq(out, xn);
		let witness = witness_generator.build().unwrap();

		constraint_system.validate(&witness);
	}

	#[test]
	fn test_fibonacci_with_precommit() {
		let mut constraint_builder = ConstraintBuilder::new();
		let x0 = constraint_builder.alloc_inout();
		let x1 = constraint_builder.alloc_inout();
		let xn = constraint_builder.alloc_precommit();
		let out = fibonacci(&mut constraint_builder, x0, x1, 20);
		constraint_builder.assert_eq(out, xn);
		let ir = constraint_builder.build();
		let (constraint_system, layout) = ir.finalize();

		let mut witness_generator = WitnessGenerator::new(&layout);
		let x0 = witness_generator.write_inout(x0, B128::ONE);
		let x1 = witness_generator.write_inout(x1, B128::MULTIPLICATIVE_GENERATOR);
		let xn = witness_generator.write_precommit(xn, B128::MULTIPLICATIVE_GENERATOR.pow(6765));
		let out = fibonacci(&mut witness_generator, x0, x1, 20);
		witness_generator.assert_eq(out, xn);
		let witness = witness_generator.build().unwrap();

		constraint_system.validate(&witness);
	}

	#[test]
	fn test_assertion_failure_captured() {
		let mut constraint_builder = ConstraintBuilder::new();
		let x = constraint_builder.alloc_inout();
		let y = constraint_builder.alloc_inout();
		let sum = constraint_builder.add(x, y);
		let expected = constraint_builder.alloc_inout();
		constraint_builder.assert_eq(sum, expected);
		let ir = constraint_builder.build();
		let (_constraint_system, layout) = ir.finalize();

		let mut witness_generator = WitnessGenerator::new(&layout);
		let x_val = B128::new(5);
		let y_val = B128::new(7);
		let wrong_expected = B128::new(99); // Incorrect: should be 5 + 7 = 2 in binary field

		let x_wire = witness_generator.write_inout(x, x_val);
		let y_wire = witness_generator.write_inout(y, y_val);
		let sum_wire = witness_generator.add(x_wire, y_wire);
		let expected_wire = witness_generator.write_inout(expected, wrong_expected);
		witness_generator.assert_eq(sum_wire, expected_wire);

		assert!(witness_generator.build().is_err());
	}

	#[test]
	fn test_zero_constant_not_in_final_operands() {
		use crate::{compiler::compile, constraint_system::WitnessSegment};

		// Build a circuit where a zero constant is added to a wire; after compilation the
		// zero constant term must be absent from all mul-constraint operands.
		let mut builder = ConstraintBuilder::new();
		let x = builder.alloc_inout();
		let zero = builder.constant(B128::ZERO);
		let sum = builder.add(x, zero);
		let y = builder.alloc_inout();
		builder.assert_eq(sum, y);

		let (cs, _layout) = compile(builder);

		let zero_constant_indices: std::collections::HashSet<u32> = cs
			.constants()
			.iter()
			.enumerate()
			.filter(|&(_, c)| *c == B128::ZERO)
			.map(|(i, _)| i as u32)
			.collect();

		for constraint in cs.mul_constraints() {
			for operand in [&constraint.a, &constraint.b, &constraint.c] {
				for wire_idx in operand.wires() {
					assert!(
						!(wire_idx.segment == WitnessSegment::Public
							&& zero_constant_indices.contains(&wire_idx.index)),
						"zero constant found in compiled operand at WitnessIndex::public({})",
						wire_idx.index
					);
				}
			}
		}
	}

	/// A circuit mixing derived wires (a derived mul, a derived hint, a derived add) with a private
	/// wire (a mul against a precommit input). Used by the sync test below.
	///
	/// `expected` must equal `((a * b) * (a * b).invert_or_zero() + b) * s` (see
	/// [`mixed_expected`]), so that the witness is satisfiable.
	fn mixed_circuit<Builder: CircuitBuilder>(
		builder: &mut Builder,
		a: Builder::Wire,
		b: Builder::Wire,
		s: Builder::Wire,
		expected: Builder::Wire,
	) {
		use binius_field::arithmetic_traits::InvertOrZero;

		let d = builder.mul(a, b); // derived (a, b public-derivable)
		let [d_inv] = builder.hint([d], |[x]| [x.invert_or_zero()]); // derived hint
		let one_check = builder.mul(d, d_inv); // derived (== 1 when d != 0)
		let e = builder.add(one_check, b); // derived
		let p = builder.mul(e, s); // private (s is precommit)
		builder.assert_eq(p, expected);
	}

	// Computes the `expected` inout value for `mixed_circuit` from concrete inputs.
	fn mixed_expected(a: B128, b: B128, s: B128) -> B128 {
		use binius_field::arithmetic_traits::InvertOrZero;
		let d = a * b;
		let one_check = d * d.invert_or_zero();
		(one_check + b) * s
	}

	#[test]
	fn test_instance_generator_syncs_with_witness() {
		use crate::compiler::compile;

		let a_val = B128::new(3);
		let b_val = B128::new(5);
		let s_val = B128::new(7);
		let expected_val = mixed_expected(a_val, b_val, s_val);

		let mut cb = ConstraintBuilder::new();
		let a = cb.alloc_inout();
		let b = cb.alloc_inout();
		let s = cb.alloc_precommit();
		let expected = cb.alloc_inout();
		mixed_circuit(&mut cb, a, b, s, expected);
		let (cs, layout) = compile(cb);

		// Witness generation (prover side) fills the full witness including the derived publics.
		let mut wg = WitnessGenerator::new(&layout);
		let a_w = wg.write_inout(a, a_val);
		let b_w = wg.write_inout(b, b_val);
		let s_w = wg.write_precommit(s, s_val);
		let expected_w = wg.write_inout(expected, expected_val);
		mixed_circuit(&mut wg, a_w, b_w, s_w, expected_w);
		let witness = wg.build().expect("witness generation should succeed");
		cs.validate(&witness);

		// Instance generation (verifier side) reconstructs only the public segment, without the
		// secret `s`. It must match the witness's public segment exactly.
		let mut ig = InstanceGenerator::new(&layout);
		let a_i = ig.write_inout(a, a_val);
		let b_i = ig.write_inout(b, b_val);
		let s_i = ig.placeholder_precommit(s);
		let expected_i = ig.write_inout(expected, expected_val);
		mixed_circuit(&mut ig, a_i, b_i, s_i, expected_i);
		let public = ig.build();

		assert_eq!(public, witness.public());
	}

	#[test]
	fn test_derived_elision_no_private_wires() {
		use crate::compiler::compile;

		// A circuit that is a pure function of inout: x -> x^2 -> x^3, asserting x^3 == y.
		let mut cb = ConstraintBuilder::new();
		let x = cb.alloc_inout();
		let y = cb.alloc_inout();
		let x2 = cb.mul(x, x); // derived (intermediate, feeds only x3)
		let x3 = cb.mul(x2, x); // derived (referenced by the assert)
		cb.assert_eq(x3, y);
		assert_eq!(x2.kind, WireKind::Derived);
		assert_eq!(x3.kind, WireKind::Derived);
		let (cs, layout) = compile(cb);

		// No private wires or mul constraints from the chain; x2 is a pruned intermediate, so only
		// x3 occupies a derived public slot.
		assert_eq!(cs.n_private(), 0);
		assert_eq!(layout.n_derived(), 1);

		let x_val = B128::new(9);
		let y_val = x_val * x_val * x_val;

		let mut wg = WitnessGenerator::new(&layout);
		let x_w = wg.write_inout(x, x_val);
		let y_w = wg.write_inout(y, y_val);
		let x2_w = wg.mul(x_w, x_w);
		let x3_w = wg.mul(x2_w, x_w);
		wg.assert_eq(x3_w, y_w);
		let witness = wg.build().expect("witness generation should succeed");
		cs.validate(&witness);
	}

	#[test]
	fn test_derived_wire_in_constraint_maps_to_public() {
		use crate::{compiler::compile, constraint_system::WitnessSegment};

		// `e` is derived but consumed by a mul constraint that also references the private `p`.
		let mut cb = ConstraintBuilder::new();
		let a = cb.alloc_inout();
		let b = cb.alloc_inout();
		let s = cb.alloc_precommit();
		let e = cb.add(a, b); // derived
		let p = cb.mul(e, s); // private; mul constraint references derived `e`
		cb.assert_zero(p);
		assert_eq!(e.kind, WireKind::Derived);
		let (cs, layout) = compile(cb);

		// The derived wire resolves to a slot in the public segment.
		let e_index = layout.get(&e).expect("alive derived wire must have a slot");
		assert_eq!(e_index.segment, WitnessSegment::Public);

		// Some surviving mul constraint references that public index.
		let references_e = cs.mul_constraints().iter().any(|c| {
			[&c.a, &c.b, &c.c]
				.iter()
				.any(|operand| operand.wires().contains(&e_index))
		});
		assert!(references_e, "derived wire's public index should appear in a mul constraint");

		// And a consistent witness validates.
		let a_val = B128::new(3);
		let b_val = B128::new(5);
		let s_val = B128::ZERO; // makes p = e * 0 = 0 so assert_zero(p) holds
		let mut wg = WitnessGenerator::new(&layout);
		let a_w = wg.write_inout(a, a_val);
		let b_w = wg.write_inout(b, b_val);
		let s_w = wg.write_precommit(s, s_val);
		let e_w = wg.add(a_w, b_w);
		let p_w = wg.mul(e_w, s_w);
		wg.assert_zero(p_w);
		let witness = wg.build().expect("witness generation should succeed");
		cs.validate(&witness);
	}

	#[test]
	fn test_pruned_derived_intermediate_has_no_slot() {
		use crate::compiler::compile;

		// x2 feeds only the derived x3; only x3 is referenced by a surviving constraint.
		let mut cb = ConstraintBuilder::<B128>::new();
		let x = cb.alloc_inout();
		let y = cb.alloc_inout();
		let x2 = cb.mul(x, x); // derived intermediate
		let x3 = cb.mul(x2, x); // derived, referenced by assert
		cb.assert_eq(x3, y);
		let (_cs, layout) = compile(cb);

		// The intermediate has no public slot; the alive one does.
		assert!(layout.get(&x2).is_none(), "pruned derived intermediate must have no slot");
		assert!(layout.get(&x3).is_some(), "referenced derived wire must have a slot");
	}

	#[test]
	fn test_hint_varsize_syncs_with_witness() {
		use crate::compiler::compile;

		// From two public inouts `a`, `b`, a runtime-arity hint yields `[a + b, a * b]`; the first
		// output (a derived wire) is asserted equal to an inout `expected`. The second output feeds
		// nothing and is pruned. The instance generator must reproduce the witness public segment.
		fn circuit<Builder: CircuitBuilder>(
			builder: &mut Builder,
			a: Builder::Wire,
			b: Builder::Wire,
			expected: Builder::Wire,
		) {
			let outs =
				builder.hint_varsize(&[a, b], 2, |vals| vec![vals[0] + vals[1], vals[0] * vals[1]]);
			builder.assert_eq(outs[0], expected);
		}

		let a_val = B128::new(3);
		let b_val = B128::new(5);
		let expected_val = a_val + b_val;

		let mut cb = ConstraintBuilder::new();
		let a = cb.alloc_inout();
		let b = cb.alloc_inout();
		let expected = cb.alloc_inout();
		circuit(&mut cb, a, b, expected);
		let (cs, layout) = compile(cb);

		let mut wg = WitnessGenerator::new(&layout);
		let a_w = wg.write_inout(a, a_val);
		let b_w = wg.write_inout(b, b_val);
		let expected_w = wg.write_inout(expected, expected_val);
		circuit(&mut wg, a_w, b_w, expected_w);
		let witness = wg.build().expect("witness generation should succeed");
		cs.validate(&witness);

		let mut ig = InstanceGenerator::new(&layout);
		let a_i = ig.write_inout(a, a_val);
		let b_i = ig.write_inout(b, b_val);
		let expected_i = ig.write_inout(expected, expected_val);
		circuit(&mut ig, a_i, b_i, expected_i);
		let public = ig.build();

		assert_eq!(public, witness.public());
	}
}
