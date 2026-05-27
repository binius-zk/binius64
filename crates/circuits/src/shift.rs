// Copyright 2025 Irreducible Inc.
//! Variable-amount shift gadgets.
//!
//! `CircuitBuilder` only exposes shifts by a compile-time-constant amount. This module provides
//! barrel-shifter gadgets that shift by a runtime [`Wire`] amount:
//!
//! - [`var_sll`], [`var_srl`], [`var_sra`] — shift by the low 6 bits of `shift` (full bit range for
//!   a 64-bit word).
//! - [`var_sll_blocks`], [`var_srl_blocks`], [`var_sra_blocks`] — most general: the actual
//!   bit-shift is `shift * 2^block_bits`, with `shift` treated as a `shift_bits`-bit unsigned
//!   integer. Useful when the caller knows the shift is a multiple of a fixed power of two, or when
//!   only a narrow range of shifts is possible.
//! - [`var_sll_bytes`], [`var_srl_bytes`], [`var_sra_bytes`] — byte-granularity shifts (shift in
//!   `0..8` bytes).
//!
//! ## Precondition
//!
//! For `*_blocks`, the shift amount is treated as a `shift_bits`-bit unsigned integer; bits above
//! position `shift_bits - 1` are ignored. Callers must ensure `shift < 2^shift_bits`; this is
//! not checked by the gadget. The other variants impose the same precondition with `shift_bits`
//! fixed (6 for the base variants, 3 for the byte variants).
//!
//! ## Cost
//!
//! `*_blocks`: `3 * shift_bits` AND constraints. Base variants: 18. Byte variants: 9.

use binius_frontend::{CircuitBuilder, Wire};

/// Variable-amount logical left shift.
///
/// Returns `x << shift`, reading the low 6 bits of `shift` as the shift amount.
pub fn var_sll(b: &CircuitBuilder, x: Wire, shift: Wire) -> Wire {
	var_sll_blocks(b, x, shift, 6, 0)
}

/// Variable-amount logical right shift.
///
/// Returns `x >> shift`, reading the low 6 bits of `shift` as the shift amount.
pub fn var_srl(b: &CircuitBuilder, x: Wire, shift: Wire) -> Wire {
	var_srl_blocks(b, x, shift, 6, 0)
}

/// Variable-amount arithmetic right shift.
///
/// Returns `x SAR shift` (sign-extending), reading the low 6 bits of `shift` as the shift amount.
pub fn var_sra(b: &CircuitBuilder, x: Wire, shift: Wire) -> Wire {
	var_sra_blocks(b, x, shift, 6, 0)
}

/// Variable-amount logical left shift by blocks of `2^block_bits` bits.
///
/// Returns `x << (shift * 2^block_bits)`, where the low `shift_bits` bits of `shift` are the
/// block count.
///
/// # Panics
///
/// Panics if `shift_bits + block_bits > 6`.
pub fn var_sll_blocks(
	b: &CircuitBuilder,
	x: Wire,
	shift: Wire,
	shift_bits: usize,
	block_bits: usize,
) -> Wire {
	var_shift_blocks(b, x, shift, shift_bits, block_bits, CircuitBuilder::shl)
}

/// Variable-amount logical right shift by blocks of `2^block_bits` bits.
///
/// Returns `x >> (shift * 2^block_bits)`, where the low `shift_bits` bits of `shift` are the
/// block count.
///
/// # Panics
///
/// Panics if `shift_bits + block_bits > 6`.
pub fn var_srl_blocks(
	b: &CircuitBuilder,
	x: Wire,
	shift: Wire,
	shift_bits: usize,
	block_bits: usize,
) -> Wire {
	var_shift_blocks(b, x, shift, shift_bits, block_bits, CircuitBuilder::shr)
}

/// Variable-amount arithmetic right shift by blocks of `2^block_bits` bits.
///
/// Returns `x SAR (shift * 2^block_bits)` (sign-extending), where the low `shift_bits` bits of
/// `shift` are the block count.
///
/// # Panics
///
/// Panics if `shift_bits + block_bits > 6`.
pub fn var_sra_blocks(
	b: &CircuitBuilder,
	x: Wire,
	shift: Wire,
	shift_bits: usize,
	block_bits: usize,
) -> Wire {
	var_shift_blocks(b, x, shift, shift_bits, block_bits, CircuitBuilder::sar)
}

/// Variable-amount logical left shift by whole bytes.
///
/// Returns `x << (shift * 8)`. The low 3 bits of `shift` are the byte count (range `0..8`).
pub fn var_sll_bytes(b: &CircuitBuilder, x: Wire, shift: Wire) -> Wire {
	var_sll_blocks(b, x, shift, 3, 3)
}

/// Variable-amount logical right shift by whole bytes.
///
/// Returns `x >> (shift * 8)`. The low 3 bits of `shift` are the byte count (range `0..8`).
pub fn var_srl_bytes(b: &CircuitBuilder, x: Wire, shift: Wire) -> Wire {
	var_srl_blocks(b, x, shift, 3, 3)
}

/// Variable-amount arithmetic right shift by whole bytes.
///
/// Returns `x SAR (shift * 8)` (sign-extending). The low 3 bits of `shift` are the byte count
/// (range `0..8`).
pub fn var_sra_bytes(b: &CircuitBuilder, x: Wire, shift: Wire) -> Wire {
	var_sra_blocks(b, x, shift, 3, 3)
}

fn var_shift_blocks(
	b: &CircuitBuilder,
	x: Wire,
	shift: Wire,
	shift_bits: usize,
	block_bits: usize,
	step: impl Fn(&CircuitBuilder, Wire, u32) -> Wire,
) -> Wire {
	assert!(
		shift_bits + block_bits <= 6,
		"shift_bits={shift_bits} + block_bits={block_bits} > 6 (max for 64-bit word)"
	);
	let mut result = x;
	for i in 0..shift_bits {
		// Move bit i of `shift` into the MSB position so `select` reads it as the condition.
		let cond = b.shl(shift, 63 - i as u32);
		let shifted = step(b, result, 1u32 << (i + block_bits));
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

	type Gadget = fn(&CircuitBuilder, Wire, Wire) -> Wire;
	type BlocksGadget = fn(&CircuitBuilder, Wire, Wire, usize, usize) -> Wire;

	fn build_circuit(gadget: Gadget) -> (Circuit, Wire, Wire, Wire) {
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let shift = builder.add_witness();
		let output = builder.add_witness();
		let computed = gadget(&builder, x, shift);
		builder.assert_eq("var_shift_result", computed, output);
		let circuit = builder.build();
		(circuit, x, shift, output)
	}

	fn build_blocks_circuit(
		gadget: BlocksGadget,
		shift_bits: usize,
		block_bits: usize,
	) -> (Circuit, Wire, Wire, Wire) {
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let shift = builder.add_witness();
		let output = builder.add_witness();
		let computed = gadget(&builder, x, shift, shift_bits, block_bits);
		builder.assert_eq("var_shift_result", computed, output);
		let circuit = builder.build();
		(circuit, x, shift, output)
	}

	fn check_ok(gadget: Gadget, x_val: u64, shift_val: u64, expected: u64) {
		let (circuit, x, shift, output) = build_circuit(gadget);
		fill_and_check(&circuit, x, shift, output, x_val, shift_val, expected);
	}

	fn check_ok_blocks(
		gadget: BlocksGadget,
		shift_bits: usize,
		block_bits: usize,
		x_val: u64,
		shift_val: u64,
		expected: u64,
	) {
		let (circuit, x, shift, output) = build_blocks_circuit(gadget, shift_bits, block_bits);
		fill_and_check(&circuit, x, shift, output, x_val, shift_val, expected);
	}

	fn fill_and_check(
		circuit: &Circuit,
		x: Wire,
		shift: Wire,
		output: Wire,
		x_val: u64,
		shift_val: u64,
		expected: u64,
	) {
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
		for &x_val in X_FIXTURES {
			for shift_val in 0u64..64 {
				check_ok(var_sll, x_val, shift_val, ref_sll(x_val, shift_val));
			}
		}
	}

	#[test]
	fn var_srl_fixtures() {
		for &x_val in X_FIXTURES {
			for shift_val in 0u64..64 {
				check_ok(var_srl, x_val, shift_val, ref_srl(x_val, shift_val));
			}
		}
	}

	#[test]
	fn var_sra_fixtures() {
		for &x_val in X_FIXTURES {
			for shift_val in 0u64..64 {
				check_ok(var_sra, x_val, shift_val, ref_sra(x_val, shift_val));
			}
		}
	}

	const BLOCK_CONFIGS: &[(usize, usize)] = &[
		(0, 0),
		(0, 6),
		(3, 0),
		(3, 3),
		(6, 0),
		(1, 5),
		(2, 4),
		(4, 2),
	];

	#[test]
	fn var_sll_blocks_fixtures() {
		for &(shift_bits, block_bits) in BLOCK_CONFIGS {
			let max_shift = if shift_bits == 0 {
				1
			} else {
				1u64 << shift_bits
			};
			for &x_val in X_FIXTURES {
				for shift_val in 0..max_shift {
					let effective = shift_val << block_bits;
					check_ok_blocks(
						var_sll_blocks,
						shift_bits,
						block_bits,
						x_val,
						shift_val,
						ref_sll(x_val, effective),
					);
				}
			}
		}
	}

	#[test]
	fn var_srl_blocks_fixtures() {
		for &(shift_bits, block_bits) in BLOCK_CONFIGS {
			let max_shift = if shift_bits == 0 {
				1
			} else {
				1u64 << shift_bits
			};
			for &x_val in X_FIXTURES {
				for shift_val in 0..max_shift {
					let effective = shift_val << block_bits;
					check_ok_blocks(
						var_srl_blocks,
						shift_bits,
						block_bits,
						x_val,
						shift_val,
						ref_srl(x_val, effective),
					);
				}
			}
		}
	}

	#[test]
	fn var_sra_blocks_fixtures() {
		for &(shift_bits, block_bits) in BLOCK_CONFIGS {
			let max_shift = if shift_bits == 0 {
				1
			} else {
				1u64 << shift_bits
			};
			for &x_val in X_FIXTURES {
				for shift_val in 0..max_shift {
					let effective = shift_val << block_bits;
					check_ok_blocks(
						var_sra_blocks,
						shift_bits,
						block_bits,
						x_val,
						shift_val,
						ref_sra(x_val, effective),
					);
				}
			}
		}
	}

	#[test]
	fn var_sll_bytes_fixtures() {
		for &x_val in X_FIXTURES {
			for shift_val in 0u64..8 {
				check_ok(var_sll_bytes, x_val, shift_val, ref_sll(x_val, shift_val * 8));
			}
		}
	}

	#[test]
	fn var_srl_bytes_fixtures() {
		for &x_val in X_FIXTURES {
			for shift_val in 0u64..8 {
				check_ok(var_srl_bytes, x_val, shift_val, ref_srl(x_val, shift_val * 8));
			}
		}
	}

	#[test]
	fn var_sra_bytes_fixtures() {
		for &x_val in X_FIXTURES {
			for shift_val in 0u64..8 {
				check_ok(var_sra_bytes, x_val, shift_val, ref_sra(x_val, shift_val * 8));
			}
		}
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(64))]

		#[test]
		fn var_sll_random(x_val in any::<u64>(), shift_val in 0u64..64) {
			check_ok(var_sll, x_val, shift_val, ref_sll(x_val, shift_val));
		}

		#[test]
		fn var_srl_random(x_val in any::<u64>(), shift_val in 0u64..64) {
			check_ok(var_srl, x_val, shift_val, ref_srl(x_val, shift_val));
		}

		#[test]
		fn var_sra_random(x_val in any::<u64>(), shift_val in 0u64..64) {
			check_ok(var_sra, x_val, shift_val, ref_sra(x_val, shift_val));
		}

		#[test]
		fn var_sll_bytes_random(x_val in any::<u64>(), shift_val in 0u64..8) {
			check_ok(var_sll_bytes, x_val, shift_val, ref_sll(x_val, shift_val * 8));
		}

		#[test]
		fn var_srl_bytes_random(x_val in any::<u64>(), shift_val in 0u64..8) {
			check_ok(var_srl_bytes, x_val, shift_val, ref_srl(x_val, shift_val * 8));
		}

		#[test]
		fn var_sra_bytes_random(x_val in any::<u64>(), shift_val in 0u64..8) {
			check_ok(var_sra_bytes, x_val, shift_val, ref_sra(x_val, shift_val * 8));
		}
	}

	#[test]
	fn rejects_incorrect_output() {
		let (circuit, x, shift, output) = build_circuit(var_sll);
		let mut w = circuit.new_witness_filler();
		w[x] = Word(0x1);
		w[shift] = Word(4);
		w[output] = Word(0x20); // wrong — should be 0x10
		assert!(circuit.populate_wire_witness(&mut w).is_err());
	}

	#[test]
	#[should_panic(expected = "shift_bits=4 + block_bits=3")]
	fn rejects_excessive_total_bits() {
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let shift = builder.add_witness();
		let _ = var_sll_blocks(&builder, x, shift, 4, 3);
	}
}
