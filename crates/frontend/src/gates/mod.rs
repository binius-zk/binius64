// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The gate kinds: what each carries, what it constrains, and how it evaluates.
//!
//! One impl states all three, so a kind's shape and the code reading it sit together.
//! A table maps an opcode to the kind serving it.

use crate::{
	eval_form::BytecodeBuilder,
	ir::{Gate, GateData, GateGraph, GateParam, Wire, hints::HintRegistry, path::PathSpec},
	lower::ConstraintBuilder,
};

pub mod opcode;

pub use opcode::{Opcode, OpcodeShape};

pub mod assert_eq;
pub mod assert_eq_cond;
pub mod assert_false;
pub mod assert_non_zero;
pub mod assert_true;
pub mod assert_zero;
pub mod band;
pub mod bmul;
pub mod bor;
pub mod bxor;
pub mod bxor_multi;
pub mod fax;
pub mod iadd32;
pub mod iadd32_cin_cout;
pub mod iadd_cin_cout;
pub mod icmp_eq;
pub mod icmp_ult;
pub mod imul;
pub mod isub_bin_bout;
pub mod select;
pub mod shift;

/// One kind of gate: what it carries, what it constrains, and how it evaluates.
///
/// The shape settles the arity, so the code destructuring against it belongs in the same impl.
pub trait GateKind {
	/// The wires and immediates a gate of this kind carries.
	const SHAPE: OpcodeShape;

	/// The shape for the dimensions a gate was emitted with.
	///
	/// Only a kind whose arity is a function of its dimensions overrides this.
	fn shape(dimensions: &[usize]) -> OpcodeShape {
		debug_assert!(dimensions.is_empty(), "a fixed-shape gate carries no dimensions");
		Self::SHAPE
	}

	/// The constraints this gate contributes, over its own wires.
	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder);

	/// The instructions that derive this gate's outputs at witness time.
	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder);
}

/// What a gate's emission needs beyond its own wires.
///
/// Every kind takes one, so an assertion reads its path through the signature all gates share.
pub struct EmitCtx<'a> {
	resolve: &'a dyn Fn(Wire) -> u32,
	path: PathSpec,
}

impl EmitCtx<'_> {
	/// The register the interpreter addresses this wire by.
	pub fn reg(&self, wire: Wire) -> u32 {
		(self.resolve)(wire)
	}

	/// The circuit path an assertion failure of this gate is reported under.
	pub const fn path(&self) -> PathSpec {
		self.path
	}
}

/// The three functions dispatch needs of a gate kind, as one row of the table.
///
/// A trait object cannot carry an associated const, so the row holds function pointers.
struct GateVTable {
	/// The opcode this row serves, which pins the row to its slot.
	opcode: Opcode,
	shape: fn(&[usize]) -> OpcodeShape,
	constrain: fn(GateParam<'_>, &mut ConstraintBuilder),
	emit: fn(GateParam<'_>, EmitCtx<'_>, &mut BytecodeBuilder),
}

/// The row serving one gate kind.
const fn row<G: GateKind>(opcode: Opcode) -> GateVTable {
	GateVTable {
		opcode,
		shape: G::shape,
		constrain: G::constrain,
		emit: G::emit,
	}
}

/// Every gate kind, indexed by its opcode.
///
/// - The hint gate has no row: its shape lives in the hint registry, not in a type.
/// - It is the last opcode declared, so it indexes one past the table.
/// - A lookup reads that as absent, and the tests below pin both facts.
static GATE_KINDS: [GateVTable; 21] = [
	row::<band::Band>(Opcode::Band),
	row::<bxor::Bxor>(Opcode::Bxor),
	row::<bxor_multi::BxorMulti>(Opcode::BxorMulti),
	row::<bor::Bor>(Opcode::Bor),
	row::<fax::Fax>(Opcode::Fax),
	row::<select::Select>(Opcode::Select),
	row::<iadd_cin_cout::IaddCinCout>(Opcode::IaddCinCout),
	row::<iadd32::Iadd32>(Opcode::Iadd32),
	row::<iadd32_cin_cout::Iadd32CinCout>(Opcode::Iadd32CinCout),
	row::<isub_bin_bout::IsubBinBout>(Opcode::IsubBinBout),
	row::<imul::Imul>(Opcode::Imul),
	row::<bmul::Bmul>(Opcode::Bmul),
	row::<shift::Shift>(Opcode::Shift),
	row::<icmp_ult::IcmpUlt>(Opcode::IcmpUlt),
	row::<icmp_eq::IcmpEq>(Opcode::IcmpEq),
	row::<assert_eq::AssertEq>(Opcode::AssertEq),
	row::<assert_zero::AssertZero>(Opcode::AssertZero),
	row::<assert_non_zero::AssertNonZero>(Opcode::AssertNonZero),
	row::<assert_false::AssertFalse>(Opcode::AssertFalse),
	row::<assert_true::AssertTrue>(Opcode::AssertTrue),
	row::<assert_eq_cond::AssertEqCond>(Opcode::AssertEqCond),
];

impl Opcode {
	/// The gate kind serving this opcode, or `None` for the hint gate.
	fn vtable(self) -> Option<&'static GateVTable> {
		let row = GATE_KINDS.get(self as usize)?;
		debug_assert_eq!(
			row.opcode, self,
			"the gate table row does not serve the opcode indexing it"
		);
		Some(row)
	}

	/// The shape a gate of this opcode carries, for the dimensions it was emitted with.
	///
	/// # Panics
	///
	/// Panics for the hint gate, whose shape the hint registry settles instead.
	/// Reading a gate's shape rather than its opcode's covers that case too.
	pub fn shape(self, dimensions: &[usize]) -> OpcodeShape {
		let vtable = self
			.vtable()
			.expect("Opcode::Hint shape requires the HintRegistry; use GateData::shape instead");
		(vtable.shape)(dimensions)
	}
}

/// Appends the constraints one gate contributes.
///
/// A hint contributes none: the surrounding circuit constrains the value it computes.
pub fn constrain(gate: Gate, graph: &GateGraph, builder: &mut ConstraintBuilder) {
	let data = &graph.gates[gate];
	if let Some(vtable) = data.opcode.vtable() {
		(vtable.constrain)(data.gate_param(), builder);
	}
}

/// Emits the instructions that derive one gate's outputs at witness time.
pub fn emit_gate_bytecode(
	gate: Gate,
	graph: &GateGraph,
	builder: &mut BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32 + Copy,
	hint_registry: &HintRegistry,
) {
	let data = &graph.gates[gate];
	match data.opcode.vtable() {
		Some(vtable) => {
			let ctx = EmitCtx {
				resolve: &wire_to_reg,
				path: graph.assertion_names[gate],
			};
			(vtable.emit)(data.gate_param(), ctx, builder);
		}
		None => emit_hint(data, builder, wire_to_reg, hint_registry),
	}
}

/// Emits a hint call, the one gate with no kind of its own.
///
/// The hint already lives in the registry, put there when the circuit called it.
/// The gate carries only its id and the dimensions the caller passed.
///
/// A hint's shape is not known statically, so the wires are sliced with the arity the registry
/// reports rather than through the usual accessor.
fn emit_hint(
	data: &GateData,
	builder: &mut BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
	hint_registry: &HintRegistry,
) {
	let hint_id = data.immediates[0];
	let (n_in, n_out) = hint_registry.shape(hint_id, &data.dimensions);
	let input_regs: Vec<u32> = data.wires[..n_in].iter().map(|&w| wire_to_reg(w)).collect();
	let output_regs: Vec<u32> = data.wires[n_in..n_in + n_out]
		.iter()
		.map(|&w| wire_to_reg(w))
		.collect();
	builder.emit_hint(hint_id, &data.dimensions, &input_regs, &output_regs);
}

#[cfg(test)]
mod tests {
	use super::*;

	// Invariant: a row sits at the slot its own opcode indexes.
	// A misplaced row would constrain one gate as another, and no circuit test would say why.
	#[test]
	fn every_row_sits_at_the_slot_its_opcode_indexes() {
		for (slot, row) in GATE_KINDS.iter().enumerate() {
			assert_eq!(row.opcode as usize, slot, "{:?} is not at slot {slot}", row.opcode);
		}
	}

	// Invariant: the hint is the one opcode without a row, and it is declared last.
	// A new opcode appended after it would silently take its slot.
	#[test]
	fn the_hint_is_the_one_opcode_with_no_row() {
		assert_eq!(Opcode::Hint as usize, GATE_KINDS.len());
		assert!(Opcode::Hint.vtable().is_none());
		for row in &GATE_KINDS {
			assert!(row.opcode.vtable().is_some(), "{:?} has no row", row.opcode);
		}
	}
}
