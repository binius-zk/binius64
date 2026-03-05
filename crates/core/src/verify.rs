// Copyright 2025 Irreducible Inc.
//! Routines for checking whether the
//! [constraint system][`crate::constraint_system::ConstraintSystem`] is satisfied with the given
//! [value vector][`ValueVec`].

use crate::{
	constraint_system::{
		AndConstraint, ConstraintSystem, MulConstraint, ShiftVariant, ShiftedValueIndex, ValueVec,
	},
	word::Word,
};

/// Evaluates a shifted value given a word
#[inline]
pub fn eval_shifted_word(word: Word, shift_variant: ShiftVariant, amount: usize) -> Word {
	match shift_variant {
		ShiftVariant::Sll => word << (amount as u32),
		ShiftVariant::Slr => word >> (amount as u32),
		ShiftVariant::Sar => word.sar(amount as u32),
		ShiftVariant::Rotr => word.rotr(amount as u32),
		ShiftVariant::Sll32 => word.sll32(amount as u32),
		ShiftVariant::Srl32 => word.srl32(amount as u32),
		ShiftVariant::Sra32 => word.sra32(amount as u32),
		ShiftVariant::Rotr32 => word.rotr32(amount as u32),
	}
}

/// Evaluates an operand (XOR of shifted values) using a ValueVec
pub fn eval_operand(witness: &ValueVec, operand: &[ShiftedValueIndex]) -> Word {
	operand.iter().fold(Word::ZERO, |acc, sv| {
		let word = witness[sv.value_index];
		let shifted_word = eval_shifted_word(word, sv.shift_variant, sv.amount);
		acc ^ shifted_word
	})
}

/// Verifies that an AND constraint is satisfied: (A & B) ^ C = 0
pub fn verify_and_constraint(witness: &ValueVec, constraint: &AndConstraint) -> Result<(), String> {
	let Word(a) = eval_operand(witness, &constraint.a);
	let Word(b) = eval_operand(witness, &constraint.b);
	let Word(c) = eval_operand(witness, &constraint.c);

	let result = (a & b) ^ c;
	if result != 0 {
		Err(format!(
			"AND constraint failed: ({a:016x} & {b:016x}) ^ {c:016x} = {result:016x} (expected 0)",
		))
	} else {
		Ok(())
	}
}

/// Verifies that a MUL constraint is satisfied: A * B = (HI << 64) | LO
pub fn verify_mul_constraint(witness: &ValueVec, constraint: &MulConstraint) -> Result<(), String> {
	let Word(a) = eval_operand(witness, &constraint.a);
	let Word(b) = eval_operand(witness, &constraint.b);
	let Word(lo) = eval_operand(witness, &constraint.lo);
	let Word(hi) = eval_operand(witness, &constraint.hi);

	let a_val = a as u128;
	let b_val = b as u128;
	let product = a_val * b_val;

	let expected_lo = (product & 0xFFFFFFFFFFFFFFFF) as u64;
	let expected_hi = (product >> 64) as u64;

	if lo != expected_lo || hi != expected_hi {
		Err(format!(
			"MUL constraint failed: {a:016x} * {b:016x} = {hi:016x}{lo:016x} (expected {expected_hi:016x}{expected_lo:016x})",
		))
	} else {
		Ok(())
	}
}

/// Verifies all constraints in a constraint system are satisfied by the witness
pub fn verify_constraints(cs: &ConstraintSystem, witness: &ValueVec) -> Result<(), String> {
	cs.value_vec_layout
		.validate()
		.map_err(|e| format!("ValueVec layout validation failed: {e}"))?;

	// First check that the witness correctly populated the constants section.
	for (index, constant) in cs.constants.iter().enumerate() {
		if witness.get(index) != *constant {
			return Err(format!(
				"Constant at index {index} does not match expected value {:016x} in value vec",
				constant.as_u64()
			));
		}
	}
	for (i, constraint) in cs.and_constraints.iter().enumerate() {
		verify_and_constraint(witness, constraint)
			.map_err(|e| format!("AND constraint {i} failed: {e}"))?;
	}
	for (i, constraint) in cs.mul_constraints.iter().enumerate() {
		verify_mul_constraint(witness, constraint)
			.map_err(|e| format!("MUL constraint {i} failed: {e}"))?;
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::constraint_system::{ValueIndex, ValueVecLayout};

	fn test_layout() -> ValueVecLayout {
		ValueVecLayout {
			n_const: 1,
			n_inout: 0,
			offset_inout: 1,
			n_witness: 5,
			n_internal: 0,
			offset_witness: 2,
			committed_total_len: 8,
			n_scratch: 0,
		}
	}

	fn test_witness(values: &[u64]) -> ValueVec {
		let layout = test_layout();
		let mut vv = ValueVec::new(layout);

		for (i, &v) in values.iter().enumerate() {
			vv.set(i, Word(v));
		}

		vv
	}

	fn sv(index: u32, variant: ShiftVariant, amount: usize) -> ShiftedValueIndex {
		ShiftedValueIndex {
			value_index: ValueIndex(index),
			shift_variant: variant,
			amount,
		}
	}

	fn plain(index: u32) -> ShiftedValueIndex {
		ShiftedValueIndex::plain(ValueIndex(index))
	}

	#[test]
	fn eval_shifted_word_all_variants() {
		let w = Word(0xDEADBEEF_CAFEBABE);

		assert_eq!(eval_shifted_word(w, ShiftVariant::Sll, 4), w << 4);

		assert_eq!(eval_shifted_word(w, ShiftVariant::Slr, 4), w >> 4);

		assert_eq!(eval_shifted_word(w, ShiftVariant::Sar, 4), w.sar(4));

		assert_eq!(eval_shifted_word(w, ShiftVariant::Rotr, 4), w.rotr(4));

		assert_eq!(eval_shifted_word(w, ShiftVariant::Sll32, 4), w.sll32(4));

		assert_eq!(eval_shifted_word(w, ShiftVariant::Srl32, 4), w.srl32(4));

		assert_eq!(eval_shifted_word(w, ShiftVariant::Sra32, 4), w.sra32(4));

		assert_eq!(eval_shifted_word(w, ShiftVariant::Rotr32, 4), w.rotr32(4));
	}

	#[test]
	fn eval_shifted_word_zero_amount_is_identity() {
		let w = Word(0x1234567890ABCDEF);

		let variants = [
			ShiftVariant::Sll,
			ShiftVariant::Slr,
			ShiftVariant::Sar,
			ShiftVariant::Rotr,
			ShiftVariant::Sll32,
			ShiftVariant::Srl32,
			ShiftVariant::Sra32,
			ShiftVariant::Rotr32,
		];

		for variant in variants {
			assert_eq!(
				eval_shifted_word(w, variant, 0),
				w,
				"shift by 0 should be identity for {variant:?}"
			);
		}
	}

	#[test]
	fn eval_operand_empty_is_zero() {
		let witness = test_witness(&[0, 0, 0, 0, 0, 0, 0, 0]);

		assert_eq!(eval_operand(&witness, &[]), Word::ZERO);
	}

	#[test]
	fn eval_operand_single_plain() {
		let witness = test_witness(&[0, 0, 0xABCD, 0, 0, 0, 0, 0]);

		let operand = vec![plain(2)];

		assert_eq!(eval_operand(&witness, &operand), Word(0xABCD));
	}

	#[test]
	fn eval_operand_xor_of_two() {
		let witness = test_witness(&[0, 0, 0xFF00, 0x0FF0, 0, 0, 0, 0]);

		let operand = vec![plain(2), plain(3)];

		assert_eq!(eval_operand(&witness, &operand), Word(0xFF00 ^ 0x0FF0));
	}

	#[test]
	fn eval_operand_with_shifts() {
		let witness = test_witness(&[0, 0, 0x01, 0, 0, 0, 0, 0]);

		let operand = vec![sv(2, ShiftVariant::Sll, 8)];

		assert_eq!(eval_operand(&witness, &operand), Word(0x0100));
	}

	#[test]
	fn eval_operand_xor_cancellation() {
		let witness = test_witness(&[0, 0, 0xDEAD, 0, 0, 0, 0, 0]);

		let operand = vec![plain(2), plain(2)];

		assert_eq!(eval_operand(&witness, &operand), Word::ZERO);
	}

	#[test]
	fn verify_and_constraint_satisfied() {
		// 0xFF00 & 0x0FF0 = 0x0F00
		let witness = test_witness(&[0, 0, 0xFF00, 0x0FF0, 0x0F00, 0, 0, 0]);

		let constraint = AndConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			c: vec![plain(4)],
		};

		assert!(verify_and_constraint(&witness, &constraint).is_ok());
	}

	#[test]
	fn verify_and_constraint_violated() {
		// 0xFF00 & 0x0FF0 = 0x0F00, but c = 0xBAAD
		let witness = test_witness(&[0, 0, 0xFF00, 0x0FF0, 0xBAAD, 0, 0, 0]);

		let constraint = AndConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			c: vec![plain(4)],
		};

		let result = verify_and_constraint(&witness, &constraint);

		assert!(result.is_err());

		assert!(result.unwrap_err().contains("AND constraint failed"));
	}

	#[test]
	fn verify_and_constraint_with_shifted_operands() {
		// a = w[2] << 4 = 0x10 << 4 = 0x100
		// b = w[3]      = 0xFFF
		// c = a & b     = 0x100
		let witness = test_witness(&[0, 0, 0x10, 0xFFF, 0x100, 0, 0, 0]);

		let constraint = AndConstraint {
			a: vec![sv(2, ShiftVariant::Sll, 4)],
			b: vec![plain(3)],
			c: vec![plain(4)],
		};

		assert!(verify_and_constraint(&witness, &constraint).is_ok());
	}

	#[test]
	fn verify_and_constraint_zeros() {
		// 0 & anything = 0
		let witness = test_witness(&[0, 0, 0, 0xFFFFFFFFFFFFFFFF, 0, 0, 0, 0]);

		let constraint = AndConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			c: vec![plain(4)],
		};

		assert!(verify_and_constraint(&witness, &constraint).is_ok());
	}

	#[test]
	fn verify_mul_constraint_satisfied() {
		// 3 * 7 = 21 (hi=0, lo=21)
		let witness = test_witness(&[0, 0, 3, 7, 21, 0, 0, 0]);

		let constraint = MulConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			lo: vec![plain(4)],
			hi: vec![plain(5)],
		};

		assert!(verify_mul_constraint(&witness, &constraint).is_ok());
	}

	#[test]
	fn verify_mul_constraint_violated() {
		// 3 * 7 = 21, but lo = 99
		let witness = test_witness(&[0, 0, 3, 7, 99, 0, 0, 0]);

		let constraint = MulConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			lo: vec![plain(4)],
			hi: vec![plain(5)],
		};

		let result = verify_mul_constraint(&witness, &constraint);

		assert!(result.is_err());

		assert!(result.unwrap_err().contains("MUL constraint failed"));
	}

	#[test]
	fn verify_mul_constraint_large_product() {
		// 0xFFFFFFFFFFFFFFFF * 0xFFFFFFFFFFFFFFFF
		// = 0xFFFFFFFFFFFFFFFE_0000000000000001
		let a: u64 = 0xFFFFFFFFFFFFFFFF;
		let b: u64 = 0xFFFFFFFFFFFFFFFF;
		let product = (a as u128) * (b as u128);
		let lo = product as u64;
		let hi = (product >> 64) as u64;

		let witness = test_witness(&[0, 0, a, b, lo, hi, 0, 0]);

		let constraint = MulConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			lo: vec![plain(4)],
			hi: vec![plain(5)],
		};

		assert!(verify_mul_constraint(&witness, &constraint).is_ok());
	}

	#[test]
	fn verify_mul_constraint_by_zero() {
		let witness = test_witness(&[0, 0, 42, 0, 0, 0, 0, 0]);

		let constraint = MulConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			lo: vec![plain(4)],
			hi: vec![plain(5)],
		};

		assert!(verify_mul_constraint(&witness, &constraint).is_ok());
	}

	#[test]
	fn verify_mul_constraint_by_one() {
		let witness = test_witness(&[0, 0, 12345, 1, 12345, 0, 0, 0]);

		let constraint = MulConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			lo: vec![plain(4)],
			hi: vec![plain(5)],
		};

		assert!(verify_mul_constraint(&witness, &constraint).is_ok());
	}

	#[test]
	fn verify_constraints_full_system() {
		let layout = test_layout();

		// constant at index 0 = 42
		let constants = vec![Word(42)];

		// AND: w[2] & w[3] = w[4]
		// 0xF0 & 0xFF = 0xF0
		let and_constraint = AndConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			c: vec![plain(4)],
		};

		// MUL: w[5] * w[6] = hi:w[7] lo:w[2]
		// We reuse w[2] for lo. 2 * 3 = 6.
		// But w[2] = 0xF0, so let's pick different values.
		// Use: w[5]=2, w[6]=3 => lo=6, hi=0. lo at w[2]? No, separate indices.
		// Simpler: just one AND constraint, no MUL.
		let cs = ConstraintSystem::new(constants, layout, vec![and_constraint], vec![]);

		let mut witness = test_witness(&[42, 0, 0xF0, 0xFF, 0xF0, 0, 0, 0]);
		witness.set(0, Word(42));

		assert!(verify_constraints(&cs, &witness).is_ok());
	}

	#[test]
	fn verify_constraints_constant_mismatch() {
		let layout = test_layout();

		let constants = vec![Word(42)];

		let cs = ConstraintSystem::new(constants, layout, vec![], vec![]);

		let witness = test_witness(&[99, 0, 0, 0, 0, 0, 0, 0]);

		let result = verify_constraints(&cs, &witness);

		assert!(result.is_err());

		assert!(result.unwrap_err().contains("Constant at index 0"));
	}

	#[test]
	fn verify_constraints_and_then_mul() {
		let layout = test_layout();

		let constants = vec![Word(0)];

		// AND: w[2] & w[3] = w[4] => 0xFF & 0x0F = 0x0F
		let and_c = AndConstraint {
			a: vec![plain(2)],
			b: vec![plain(3)],
			c: vec![plain(4)],
		};

		// MUL: w[5] * w[6] = hi:0 lo:w[2] => but we need separate.
		// w[5]=3, w[6]=5 => lo=15(w[2]), hi=0(w[0]) ... w[2] is 0xFF.
		// Let's just use: w[5]*w[6] => lo=w[4], hi=w[0]
		// w[5]=3, w[6]=5 => lo=15, hi=0. But w[4]=0x0F=15, w[0]=0. Works!
		let mul_c = MulConstraint {
			a: vec![plain(5)],
			b: vec![plain(6)],
			lo: vec![plain(4)],
			hi: vec![plain(0)],
		};

		let cs = ConstraintSystem::new(constants, layout, vec![and_c], vec![mul_c]);

		// w[0]=0(const), w[1]=0(pad), w[2]=0xFF, w[3]=0x0F, w[4]=0x0F, w[5]=3, w[6]=5, w[7]=0
		let witness = test_witness(&[0, 0, 0xFF, 0x0F, 0x0F, 3, 5, 0]);

		assert!(verify_constraints(&cs, &witness).is_ok());
	}
}
