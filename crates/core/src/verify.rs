// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Routines for checking whether the
//! [constraint system][`crate::constraint_system::ConstraintSystem`] is satisfied with the given
//! [value vector][`ValueVec`].

use crate::{
	constraint_system::{
		AndConstraint, BmulConstraint, ConstraintSystem, ImulConstraint, ValueIndex, ValueVec,
		ZeroConstraint,
	},
	word::Word,
};

/// Verifies that a ZERO constraint is satisfied: VAL = 0
pub fn verify_zero_constraint(
	witness: &ValueVec,
	constraint: &ZeroConstraint,
) -> Result<(), String> {
	let Word(val) = witness.eval_operand(constraint.val());

	if val != 0 {
		Err(format!("ZERO constraint failed: {val:016x} (expected 0)"))
	} else {
		Ok(())
	}
}

/// Verifies that an AND constraint is satisfied: (A & B) ^ C = 0
pub fn verify_and_constraint(witness: &ValueVec, constraint: &AndConstraint) -> Result<(), String> {
	let Word(a) = witness.eval_operand(constraint.a());
	let Word(b) = witness.eval_operand(constraint.b());
	let Word(c) = witness.eval_operand(constraint.c());

	let result = (a & b) ^ c;
	if result != 0 {
		Err(format!(
			"AND constraint failed: ({a:016x} & {b:016x}) ^ {c:016x} = {result:016x} (expected 0)",
		))
	} else {
		Ok(())
	}
}

/// Verifies that an IMUL constraint is satisfied: A * B = (HI << 64) | LO
pub fn verify_imul_constraint(
	witness: &ValueVec,
	constraint: &ImulConstraint,
) -> Result<(), String> {
	let Word(a) = witness.eval_operand(constraint.a());
	let Word(b) = witness.eval_operand(constraint.b());
	let Word(lo) = witness.eval_operand(constraint.lo());
	let Word(hi) = witness.eval_operand(constraint.hi());

	let a_val = a as u128;
	let b_val = b as u128;
	let product = a_val * b_val;

	let expected_lo = (product & 0xFFFFFFFFFFFFFFFF) as u64;
	let expected_hi = (product >> 64) as u64;

	if lo != expected_lo || hi != expected_hi {
		Err(format!(
			"IMUL constraint failed: {a:016x} * {b:016x} = {hi:016x}{lo:016x} (expected {expected_hi:016x}{expected_lo:016x})",
		))
	} else {
		Ok(())
	}
}

/// Multiplies two elements of the GHASH field: `GF(2^128)` with reduction polynomial
/// `X^128 + X^7 + X^2 + X + 1`, bit `i` carrying the coefficient of `X^i`.
///
/// Shift-and-add over `u128` — an independent restatement of the gate semantics for checking,
/// not a call into the field crate's multiplier.
fn ghash_mul(a: u128, b: u128) -> u128 {
	let mut acc = 0u128;
	let mut shifted = a;
	for i in 0..128 {
		if (b >> i) & 1 == 1 {
			acc ^= shifted;
		}
		let overflow = shifted >> 127;
		shifted <<= 1;
		if overflow == 1 {
			shifted ^= 0x87;
		}
	}
	acc
}

/// Verifies that a BMUL constraint is satisfied: A * B = C in the GHASH field, each element
/// carried by a `(lo, hi)` pair of words.
pub fn verify_bmul_constraint(
	witness: &ValueVec,
	constraint: &BmulConstraint,
) -> Result<(), String> {
	let Word(a_lo) = witness.eval_operand(constraint.a_lo());
	let Word(a_hi) = witness.eval_operand(constraint.a_hi());
	let Word(b_lo) = witness.eval_operand(constraint.b_lo());
	let Word(b_hi) = witness.eval_operand(constraint.b_hi());
	let Word(c_lo) = witness.eval_operand(constraint.c_lo());
	let Word(c_hi) = witness.eval_operand(constraint.c_hi());

	let a = (a_lo as u128) | ((a_hi as u128) << 64);
	let b = (b_lo as u128) | ((b_hi as u128) << 64);
	let c = (c_lo as u128) | ((c_hi as u128) << 64);

	let expected = ghash_mul(a, b);
	if c != expected {
		Err(format!(
			"BMUL constraint failed: {a:032x} * {b:032x} = {c:032x} (expected {expected:032x})",
		))
	} else {
		Ok(())
	}
}

/// Verifies all constraints in a constraint system are satisfied by the witness
pub fn verify_constraints(cs: &ConstraintSystem, witness: &ValueVec) -> Result<(), String> {
	cs.validate_shape()
		.map_err(|e| format!("ValueVec shape validation failed: {e}"))?;

	// First check that the witness correctly populated the constants section.
	for (index, constant) in cs.constants.iter().enumerate() {
		if witness[ValueIndex(index as u32)] != *constant {
			return Err(format!(
				"Constant at index {index} does not match expected value {:016x} in value vec",
				constant.as_u64()
			));
		}
	}
	for (i, constraint) in cs.zero_constraints.iter().enumerate() {
		verify_zero_constraint(witness, constraint)
			.map_err(|e| format!("ZERO constraint {i} failed: {e}"))?;
	}
	for (i, constraint) in cs.and_constraints.iter().enumerate() {
		verify_and_constraint(witness, constraint)
			.map_err(|e| format!("AND constraint {i} failed: {e}"))?;
	}
	for (i, constraint) in cs.imul_constraints.iter().enumerate() {
		verify_imul_constraint(witness, constraint)
			.map_err(|e| format!("IMUL constraint {i} failed: {e}"))?;
	}
	for (i, constraint) in cs.bmul_constraints.iter().enumerate() {
		verify_bmul_constraint(witness, constraint)
			.map_err(|e| format!("BMUL constraint {i} failed: {e}"))?;
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::constraint_system::ShiftedValueIndex;

	/// Products pinned against `BinaryField128bGhash`: `X = 2`, coefficients ascend with bit
	/// index, and `X^127 * X` wraps to the reduction polynomial.
	#[test]
	fn ghash_mul_matches_field_arithmetic() {
		assert_eq!(ghash_mul(1, 0x0123_4567), 0x0123_4567);
		assert_eq!(ghash_mul(2, 2), 4);
		assert_eq!(ghash_mul(1 << 63, 2), 1 << 64);
		assert_eq!(ghash_mul(1 << 127, 2), 0x87);
		// Computed with `BinaryField128bGhash`.
		assert_eq!(
			ghash_mul(0x0123456789abcdef_fedcba9876543210, 0x0f1e2d3c4b5a6978_8796a5b4c3d2e1f0),
			0x7f2984f784967f5a_7b881bf2b700d768
		);
	}

	/// A shape whose six inout words carry one BMUL constraint's operands directly.
	///
	///     [ _ _ | a_lo a_hi b_lo b_hi c_lo c_hi ][ p p p p p p p p ]
	///       0 1   2    3    4    5    6    7       8 ...        15
	fn one_bmul_system() -> ConstraintSystem {
		ConstraintSystem {
			constants: vec![],
			n_const_pad: 2,
			n_inout: 6,
			n_inout_pad: 0,
			n_private: 8,
			n_private_pad: 0,
			zero_constraints: vec![],
			and_constraints: vec![],
			imul_constraints: vec![],
			bmul_constraints: vec![BmulConstraint(std::array::from_fn(|i| {
				vec![ShiftedValueIndex::plain(ValueIndex(2 + i as u32))]
			}))],
		}
	}

	fn one_bmul_witness(a: u128, b: u128, c: u128) -> ValueVec {
		let split = |x: u128| [Word::from_u64(x as u64), Word::from_u64((x >> 64) as u64)];
		let [a_lo, a_hi] = split(a);
		let [b_lo, b_hi] = split(b);
		let [c_lo, c_hi] = split(c);
		let public = [Word::ZERO, Word::ZERO, a_lo, a_hi, b_lo, b_hi, c_lo, c_hi];
		ValueVec::new_from_data(&public, &[Word::ZERO; 8])
	}

	#[test]
	fn bmul_constraint_accepts_correct_product() {
		let a = 0x0123456789abcdef_fedcba9876543210;
		let b = 0x0f1e2d3c4b5a6978_8796a5b4c3d2e1f0;
		let witness = one_bmul_witness(a, b, ghash_mul(a, b));
		assert!(verify_constraints(&one_bmul_system(), &witness).is_ok());
	}

	#[test]
	fn bmul_constraint_rejects_wrong_product() {
		let a = 0x0123456789abcdef_fedcba9876543210;
		let b = 0x0f1e2d3c4b5a6978_8796a5b4c3d2e1f0;
		let witness = one_bmul_witness(a, b, ghash_mul(a, b) ^ 1);
		let err = verify_constraints(&one_bmul_system(), &witness).unwrap_err();
		assert!(err.contains("BMUL constraint 0 failed"), "{err}");
	}
}
