// Copyright 2025 Irreducible Inc.
//! Variable-amount shift gadgets.
//!
//! `CircuitBuilder` only exposes shifts by a compile-time-constant amount. This module provides
//! [`var_sll`], [`var_srl`], and [`var_sra`] — barrel-shifter gadgets that shift by a runtime
//! [`Wire`] amount.
//!
//! ## Precondition
//!
//! The shift amount `shift` is treated as a `shift_bits`-bit unsigned integer; bits above
//! position `shift_bits - 1` are ignored. Callers must ensure `shift < 2^shift_bits`; this is
//! not checked by the gadget.
//!
//! ## Cost
//!
//! Each gadget emits `3 * shift_bits` AND constraints.

use binius_frontend::{CircuitBuilder, Wire};

/// Variable-amount logical left shift.
///
/// Returns `x << shift`, where the low `shift_bits` bits of `shift` are the shift amount.
///
/// # Panics
///
/// Panics if `shift_bits > 6`.
pub fn var_sll(b: &CircuitBuilder, x: Wire, shift: Wire, shift_bits: usize) -> Wire {
	var_shift(b, x, shift, shift_bits, CircuitBuilder::shl)
}

/// Variable-amount logical right shift.
///
/// Returns `x >> shift`, where the low `shift_bits` bits of `shift` are the shift amount.
///
/// # Panics
///
/// Panics if `shift_bits > 6`.
pub fn var_srl(b: &CircuitBuilder, x: Wire, shift: Wire, shift_bits: usize) -> Wire {
	var_shift(b, x, shift, shift_bits, CircuitBuilder::shr)
}

/// Variable-amount arithmetic right shift.
///
/// Returns `x SAR shift` (sign-extending), where the low `shift_bits` bits of `shift` are the
/// shift amount.
///
/// # Panics
///
/// Panics if `shift_bits > 6`.
pub fn var_sra(b: &CircuitBuilder, x: Wire, shift: Wire, shift_bits: usize) -> Wire {
	var_shift(b, x, shift, shift_bits, CircuitBuilder::sar)
}

fn var_shift(
	b: &CircuitBuilder,
	x: Wire,
	shift: Wire,
	shift_bits: usize,
	step: impl Fn(&CircuitBuilder, Wire, u32) -> Wire,
) -> Wire {
	assert!(shift_bits <= 6, "shift_bits={shift_bits} > 6 (max for 64-bit word)");
	let mut result = x;
	for i in 0..shift_bits {
		// Move bit i of `shift` into the MSB position so `select` reads it as the condition.
		let cond = b.shl(shift, 63 - i as u32);
		let shifted = step(b, result, 1u32 << i);
		result = b.select(cond, shifted, result);
	}
	result
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;
	use binius_frontend::Circuit;
	use proptest::prelude::*;

	use super::*;

	type Gadget = fn(&CircuitBuilder, Wire, Wire, usize) -> Wire;

	fn build_circuit(gadget: Gadget, shift_bits: usize) -> (Circuit, Wire, Wire, Wire) {
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let shift = builder.add_witness();
		let output = builder.add_witness();
		let computed = gadget(&builder, x, shift, shift_bits);
		builder.assert_eq("var_shift_result", computed, output);
		let circuit = builder.build();
		(circuit, x, shift, output)
	}

	fn check_ok(gadget: Gadget, shift_bits: usize, x_val: u64, shift_val: u64, expected: u64) {
		let (circuit, x, shift, output) = build_circuit(gadget, shift_bits);
		let mut w = circuit.new_witness_filler();
		w[x] = Word(x_val);
		w[shift] = Word(shift_val);
		w[output] = Word(expected);
		circuit.populate_wire_witness(&mut w).unwrap_or_else(|e| {
			panic!(
				"populate failed: x=0x{x_val:016x} shift={shift_val} expected=0x{expected:016x}: {e:?}"
			)
		});
	}

	fn ref_sll(x: u64, s: u64) -> u64 {
		if s >= 64 { 0 } else { x << s }
	}
	fn ref_srl(x: u64, s: u64) -> u64 {
		if s >= 64 { 0 } else { x >> s }
	}
	fn ref_sra(x: u64, s: u64) -> u64 {
		let s = s.min(63);
		((x as i64) >> s) as u64
	}

	// Static edge-case fixtures.
	const X_FIXTURES: &[u64] = &[
		0,
		1,
		0xFFFF_FFFF_FFFF_FFFF,
		0x8000_0000_0000_0000, // MSB only — important for sra
		0x0123_4567_89AB_CDEF,
		0xDEAD_BEEF_CAFE_F00D,
		0x5555_5555_5555_5555,
	];

	#[test]
	fn var_sll_fixtures() {
		for shift_bits in [0, 1, 3, 6] {
			let max_shift = if shift_bits == 0 {
				1
			} else {
				1u64 << shift_bits
			};
			for &x_val in X_FIXTURES {
				for shift_val in 0..max_shift {
					check_ok(var_sll, shift_bits, x_val, shift_val, ref_sll(x_val, shift_val));
				}
			}
		}
	}

	#[test]
	fn var_srl_fixtures() {
		for shift_bits in [0, 1, 3, 6] {
			let max_shift = if shift_bits == 0 {
				1
			} else {
				1u64 << shift_bits
			};
			for &x_val in X_FIXTURES {
				for shift_val in 0..max_shift {
					check_ok(var_srl, shift_bits, x_val, shift_val, ref_srl(x_val, shift_val));
				}
			}
		}
	}

	#[test]
	fn var_sra_fixtures() {
		for shift_bits in [0, 1, 3, 6] {
			let max_shift = if shift_bits == 0 {
				1
			} else {
				1u64 << shift_bits
			};
			for &x_val in X_FIXTURES {
				for shift_val in 0..max_shift {
					check_ok(var_sra, shift_bits, x_val, shift_val, ref_sra(x_val, shift_val));
				}
			}
		}
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(64))]

		#[test]
		fn var_sll_random(x_val in any::<u64>(), shift_val in 0u64..64) {
			check_ok(var_sll, 6, x_val, shift_val, ref_sll(x_val, shift_val));
		}

		#[test]
		fn var_srl_random(x_val in any::<u64>(), shift_val in 0u64..64) {
			check_ok(var_srl, 6, x_val, shift_val, ref_srl(x_val, shift_val));
		}

		#[test]
		fn var_sra_random(x_val in any::<u64>(), shift_val in 0u64..64) {
			check_ok(var_sra, 6, x_val, shift_val, ref_sra(x_val, shift_val));
		}
	}

	#[test]
	fn var_sll_zero_shift_bits_is_identity() {
		// shift_bits = 0: the gadget should pass x through unchanged regardless of `shift`.
		check_ok(var_sll, 0, 0xDEAD_BEEF_CAFE_F00D, 0, 0xDEAD_BEEF_CAFE_F00D);
		check_ok(var_srl, 0, 0xDEAD_BEEF_CAFE_F00D, 0, 0xDEAD_BEEF_CAFE_F00D);
		check_ok(var_sra, 0, 0xDEAD_BEEF_CAFE_F00D, 0, 0xDEAD_BEEF_CAFE_F00D);
	}

	#[test]
	fn rejects_incorrect_output() {
		let (circuit, x, shift, output) = build_circuit(var_sll, 6);
		let mut w = circuit.new_witness_filler();
		w[x] = Word(0x1);
		w[shift] = Word(4);
		w[output] = Word(0x20); // wrong — should be 0x10
		assert!(circuit.populate_wire_witness(&mut w).is_err());
	}

	#[test]
	#[should_panic(expected = "shift_bits=7")]
	fn rejects_excessive_shift_bits() {
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let shift = builder.add_witness();
		let _ = var_sll(&builder, x, shift, 7);
	}
}
