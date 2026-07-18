// Copyright 2025-2026 The Binius Developers
//! Constant-amount shift and rotate.
//!
//! Returns `z = shift(x, n)` for one of the eight shift/rotate variants.
//!
//! # Immediates
//!
//! - `immediates[0]`: the [`ShiftVariant`] discriminant selecting the operation.
//! - `immediates[1]`: the shift amount `n`.
//!
//! # Constraints
//!
//! The gate generates 1 linear constraint:
//! - `shift(x, n) = z`
//!
//! The shift is folded into a constraint operand for free, so no AND constraint is spent.

use binius_core::constraint_system::ShiftVariant;

use crate::compiler::{
	constraint_builder::{ConstraintBuilder, WireExprTerm, expr},
	gate::opcode::OpcodeShape,
	gate_graph::{Gate, GateData, GateParam, Wire},
};

pub const fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[],
		n_in: 1,
		n_out: 1,
		n_aux: 0,
		n_scratch: 0,
		n_imm: 2,
	}
}

/// Decodes the variant immediate into its [`ShiftVariant`].
///
/// The builder always emits a discriminant in `0..=7`, so an out-of-range value is a bug.
const fn variant_of(imm: u32) -> ShiftVariant {
	ShiftVariant::from_u8(imm as u8).expect("shift gate carries a valid ShiftVariant discriminant")
}

/// Builds the shifted-operand term for the given variant and amount.
const fn shifted_term(variant: ShiftVariant, x: Wire, n: u32) -> WireExprTerm {
	match variant {
		ShiftVariant::Sll => expr::sll(x, n),
		ShiftVariant::Slr => expr::srl(x, n),
		ShiftVariant::Sar => expr::sar(x, n),
		ShiftVariant::Rotr => expr::rotr(x, n),
		ShiftVariant::Sll32 => expr::sll32(x, n),
		ShiftVariant::Srl32 => expr::srl32(x, n),
		ShiftVariant::Sra32 => expr::sra32(x, n),
		ShiftVariant::Rotr32 => expr::rotr32(x, n),
	}
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam {
		inputs,
		outputs,
		imm,
		..
	} = data.gate_param();
	let [x] = inputs else { unreachable!() };
	let [z] = outputs else { unreachable!() };
	let [variant, n] = imm else { unreachable!() };

	// Constraint: shift(x, n) = z, with the shift folded into the operand.
	let term = shifted_term(variant_of(*variant), *x, *n);
	builder.linear().rhs(term).dst(*z).build();
}

pub fn emit_eval_bytecode(
	_gate: Gate,
	data: &GateData,
	builder: &mut crate::compiler::eval_form::BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam {
		inputs,
		outputs,
		imm,
		..
	} = data.gate_param();
	let [x] = inputs else { unreachable!() };
	let [z] = outputs else { unreachable!() };
	let [variant, n] = imm else { unreachable!() };

	// Emit a single shift instruction carrying the variant and amount.
	builder.emit_shift(wire_to_reg(*z), wire_to_reg(*x), variant_of(*variant), *n as u8);
}
