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

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::{GateParam, Wire},
	lower::{ConstraintBuilder, WireExprTerm, expr},
};

/// One word shifted or rotated by a constant amount.
pub struct Shift;

impl GateKind for Shift {
	const SHAPE: OpcodeShape = OpcodeShape::new(1, 1).with_imm(2);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x] = gate.in_wires();
		let [z] = gate.out_wires();
		let [variant, n] = gate.imms();

		// shift(x, n) = z, with the shift folded into the operand.
		cb.linear(shifted_term(variant_of(variant), x, n), z);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x] = gate.in_wires();
		let [z] = gate.out_wires();
		let [variant, n] = gate.imms();

		// One instruction carrying the variant and the amount.
		bc.emit_shift(ctx.reg(z), ctx.reg(x), variant_of(variant), n as u8);
	}
}

/// Decodes the variant immediate.
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
