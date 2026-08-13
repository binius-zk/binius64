// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use crate::{
	eval_form::BytecodeBuilder,
	ir::{Gate, GateData, GateGraph, Wire, hints::HintRegistry},
	lower::ConstraintBuilder,
};

pub mod opcode;

pub use opcode::Opcode;

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

pub fn constrain(gate: Gate, graph: &GateGraph, builder: &mut ConstraintBuilder) {
	let data = &graph.gates[gate];
	match data.opcode {
		Opcode::Band => band::constrain(data, builder),
		Opcode::Bxor => bxor::constrain(data, builder),
		Opcode::BxorMulti => bxor_multi::constrain(data, builder),
		Opcode::Bor => bor::constrain(data, builder),
		Opcode::Fax => fax::constrain(data, builder),
		Opcode::Select => select::constrain(data, builder),
		Opcode::IaddCinCout => iadd_cin_cout::constrain(data, builder),
		Opcode::Iadd32 => iadd32::constrain(data, builder),
		Opcode::Iadd32CinCout => iadd32_cin_cout::constrain(data, builder),
		Opcode::IsubBinBout => isub_bin_bout::constrain(data, builder),
		Opcode::Shift => shift::constrain(data, builder),
		Opcode::AssertEq => assert_eq::constrain(data, builder),
		Opcode::AssertZero => assert_zero::constrain(data, builder),
		Opcode::AssertNonZero => assert_non_zero::constrain(data, builder),
		Opcode::AssertFalse => assert_false::constrain(data, builder),
		Opcode::AssertTrue => assert_true::constrain(data, builder),
		Opcode::AssertEqCond => assert_eq_cond::constrain(data, builder),
		Opcode::Imul => imul::constrain(data, builder),
		Opcode::Bmul => bmul::constrain(data, builder),
		Opcode::IcmpUlt => icmp_ult::constrain(data, builder),
		Opcode::IcmpEq => icmp_eq::constrain(data, builder),
		// Hints do not introduce constraints
		Opcode::Hint => (),
	}
}

/// Emit bytecode for a single gate
pub fn emit_gate_bytecode(
	gate: Gate,
	graph: &GateGraph,
	builder: &mut BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32 + Copy,
	hint_registry: &HintRegistry,
) {
	let data = &graph.gates[gate];

	// The assertion gates take the path their failure is reported under; the rest do not.
	let path = graph.assertion_names[gate];
	match data.opcode {
		Opcode::Band => band::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::Bxor => bxor::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::BxorMulti => bxor_multi::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::Bor => bor::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::Fax => fax::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::Select => select::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::IaddCinCout => iadd_cin_cout::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::Iadd32 => iadd32::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::Iadd32CinCout => iadd32_cin_cout::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::IsubBinBout => isub_bin_bout::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::Shift => shift::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::Imul => imul::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::Bmul => bmul::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::IcmpUlt => icmp_ult::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::IcmpEq => icmp_eq::emit_eval_bytecode(data, builder, wire_to_reg),
		Opcode::AssertEq => assert_eq::emit_eval_bytecode(data, path, builder, wire_to_reg),
		Opcode::AssertZero => assert_zero::emit_eval_bytecode(data, path, builder, wire_to_reg),
		Opcode::AssertNonZero => {
			assert_non_zero::emit_eval_bytecode(data, path, builder, wire_to_reg)
		}
		Opcode::AssertEqCond => {
			assert_eq_cond::emit_eval_bytecode(data, path, builder, wire_to_reg)
		}
		Opcode::AssertFalse => assert_false::emit_eval_bytecode(data, path, builder, wire_to_reg),
		Opcode::AssertTrue => assert_true::emit_eval_bytecode(data, path, builder, wire_to_reg),
		Opcode::Hint => emit_hint(data, builder, wire_to_reg, hint_registry),
	}
}

/// Emits a hint call, the one gate with no module of its own.
///
/// The hint itself already lives in the registry, put there by `CircuitBuilder::call_hint`.
/// The gate carries only its id, in `immediates[0]`, plus the user dimensions in `dimensions`.
///
/// `gate_param()` would panic for a hint, since its shape is not known statically.
/// So the wires are sliced directly, using the shape read back from the registry.
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
