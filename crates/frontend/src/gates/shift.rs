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
///
/// A zero amount is the identity in every variant, and the identity has one canonical
/// spelling ([`Shift::IDENTITY`](binius_core::constraint_system::Shift::IDENTITY), i.e. `Sll 0`)
/// — see [`Shift::is_canonical`](binius_core::constraint_system::Shift::is_canonical). Emitting
/// `Slr 0` or `Sar 0` here would put a non-canonical shift in an operand, which
/// `ConstraintSystem::validate` rejects.
const fn shifted_term(variant: ShiftVariant, x: Wire, n: u32) -> WireExprTerm {
	if n == 0 {
		return WireExprTerm::Wire(x);
	}
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

#[cfg(test)]
mod tests {
	use binius_core::{Word, constraint_system::ShiftVariant};

	use crate::{CircuitBuilder, Options};

	/// A shift by zero is the identity in every variant, and the constraint system carries
	/// only the canonical spelling of it. Gate fusion used to hide a non-canonical one by
	/// rebuilding the operand at the consumer, so this builds with fusion off.
	#[test]
	fn a_zero_amount_shift_lowers_to_the_canonical_identity() {
		for variant in [
			ShiftVariant::Sll,
			ShiftVariant::Slr,
			ShiftVariant::Sar,
			ShiftVariant::Rotr,
			ShiftVariant::Sll32,
			ShiftVariant::Srl32,
			ShiftVariant::Sra32,
			ShiftVariant::Rotr32,
		] {
			let opts = Options {
				enable_gate_fusion: false,
				..Options::default()
			};
			let b = CircuitBuilder::with_opts(opts);
			let x = b.add_witness();
			let shifted = match variant {
				ShiftVariant::Sll => b.shl(x, 0),
				ShiftVariant::Slr => b.shr(x, 0),
				ShiftVariant::Sar => b.sar(x, 0),
				ShiftVariant::Rotr => b.rotr(x, 0),
				ShiftVariant::Sll32 => b.sll32(x, 0),
				ShiftVariant::Srl32 => b.srl32(x, 0),
				ShiftVariant::Sra32 => b.sra32(x, 0),
				ShiftVariant::Rotr32 => b.rotr32(x, 0),
			};
			b.assert_eq("shifted", shifted, b.add_constant(Word::ZERO));
			let circuit = b.build();
			circuit.constraint_system().validate().unwrap_or_else(|e| {
				panic!("{variant:?} by 0 lowered to a non-canonical shift: {e:?}")
			});
		}
	}
}
