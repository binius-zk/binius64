// Copyright 2025-2026 The Binius Developers
//! Shared bytecode execution core for the circuit interpreters.
//!
//! Both interpreters run the same bytecode through the same opcode dispatch.
//! They differ only in where a decoded instruction reads and writes:
//! - single instance: one value vector.
//! - batched: one column per instance.
//!
//! That difference is captured by one trait over the execution context.
//! Every instruction is applied across all instances the context holds.
//! The single-instance form is then the degenerate case of one instance.
//! The dispatch loop, the opcode handlers, and the bytecode readers live here once.

use binius_core::{Word, constraint_system::ShiftVariant};
use binius_field::BinaryField128bGhash;

use crate::compiler::{hints::HintRegistry, pathspec::PathSpec};

/// Multiplies two GHASH field elements ($\mathbb{F}_{2^{128}}$), each carried by a `(lo, hi)` pair
/// of words — `lo` holds the coefficients of $1, X, \ldots, X^{63}$ and `hi` those of
/// $X^{64}, \ldots, X^{127}$ — and returns the product in the same `(lo, hi)` form.
fn ghash_mul(a_lo: Word, a_hi: Word, b_lo: Word, b_hi: Word) -> (Word, Word) {
	let to_field = |lo: Word, hi: Word| {
		BinaryField128bGhash::from((lo.as_u64() as u128) | ((hi.as_u64() as u128) << 64))
	};
	let product = u128::from(to_field(a_lo, a_hi) * to_field(b_lo, b_hi));
	(Word::from_u64(product as u64), Word::from_u64((product >> 64) as u64))
}

/// The values one bytecode program evaluates against.
///
/// A context holds some number of independent instances of one circuit.
/// A register names a value-vector index.
/// Reading or writing a register targets that index within one chosen instance.
/// So an instruction is applied once per instance.
pub trait EvalContext {
	/// The number of independent instances evaluated in lockstep.
	fn n_instances(&self) -> usize;

	/// Reads the register at index `reg` within instance `instance`.
	fn load(&self, reg: u32, instance: usize) -> Word;

	/// Writes `value` to the register at index `reg` within instance `instance`.
	fn store(&mut self, reg: u32, instance: usize, value: Word);

	/// Records an assertion violation for one instance.
	///
	/// The index is local to this context.
	/// A context that covers a stripe of a larger batch remaps it to a global index.
	fn note_assertion_failure(&mut self, instance: usize, path_spec: PathSpec, message: String);
}

/// A bytecode program together with a cursor into it.
///
/// The cursor advances as the dispatch loop consumes opcodes and operands.
/// One executor drives one pass over the bytecode.
/// The interpreters build a fresh executor per run, so the cursor starts at the first instruction.
pub struct Executor<'a> {
	bytecode: &'a [u8],
	hints: &'a HintRegistry,
	pc: usize,
}

impl<'a> Executor<'a> {
	pub const fn new(bytecode: &'a [u8], hints: &'a HintRegistry) -> Self {
		Self {
			bytecode,
			hints,
			pc: 0,
		}
	}

	/// Evaluates the whole program against the context, filling every instance's wires.
	///
	/// The constant and input registers must already be populated for every instance.
	/// Assertion violations are recorded on the context, not raised here.
	/// So the caller decides how to turn them into an error.
	///
	/// # Panics
	///
	/// Panics on an unknown opcode, which can only happen if the bytecode is malformed.
	pub fn run<C: EvalContext>(&mut self, ctx: &mut C) {
		while self.pc < self.bytecode.len() {
			let opcode = self.read_u8();
			match opcode {
				// Bitwise operations
				0x01 => self.exec_band(ctx),
				0x02 => self.exec_bor(ctx),
				0x03 => self.exec_bxor(ctx),
				0x05 => self.exec_select(ctx),
				0x06 => self.exec_bxor_multi(ctx),
				0x07 => self.exec_fax(ctx),

				// Shifts
				0x10 => self.exec_shift(ctx),

				// Arithmetic
				0x20 => self.exec_iadd_cout(ctx),
				0x21 => self.exec_iadd_cin_cout(ctx),
				0x23 => self.exec_isub_bin_bout(ctx),
				0x30 => self.exec_imul(ctx),
				0x31 => self.exec_bmul(ctx),

				// 32-bit operations
				0x40 => self.exec_iadd32_cin_cout(ctx),
				0x46 => self.exec_iadd32_cout(ctx),

				// Masks
				0x50 => self.exec_mask_low(ctx),
				0x51 => self.exec_mask_high(ctx),

				// Assertions
				0x60 => self.exec_assert_eq(ctx),
				0x61 => self.exec_assert_eq_cond(ctx),
				0x62 => self.exec_assert_zero(ctx),
				0x63 => self.exec_assert_non_zero(ctx),
				0x64 => self.exec_assert_false(ctx),
				0x65 => self.exec_assert_true(ctx),

				// Hint calls
				0x80 => self.exec_hint(ctx),

				_ => panic!("Unknown opcode: {:#x} at pc={}", opcode, self.pc - 1),
			}
		}
	}

	// Bitwise operations
	fn exec_band<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src1, i) & ctx.load(src2, i);
			ctx.store(dst, i, val);
		}
	}

	fn exec_bor<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src1, i) | ctx.load(src2, i);
			ctx.store(dst, i, val);
		}
	}

	fn exec_bxor<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src1, i) ^ ctx.load(src2, i);
			ctx.store(dst, i, val);
		}
	}

	fn exec_select<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let cond = self.read_reg();
		let t = self.read_reg();
		let f = self.read_reg();
		for i in 0..ctx.n_instances() {
			// Select t if MSB(cond) is 1, otherwise select f.
			let val = if ctx.load(cond, i).is_msb_true() {
				ctx.load(t, i)
			} else {
				ctx.load(f, i)
			};
			ctx.store(dst, i, val);
		}
	}

	fn exec_bxor_multi<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let n = self.read_u32() as usize;
		// Read the source registers once; they are shared across every instance.
		let srcs = (0..n).map(|_| self.read_reg()).collect::<Vec<_>>();
		for i in 0..ctx.n_instances() {
			let mut val = Word::ZERO;
			for &src in &srcs {
				val = val ^ ctx.load(src, i);
			}
			ctx.store(dst, i, val);
		}
	}

	fn exec_fax<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let src3 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let val = (ctx.load(src1, i) & ctx.load(src2, i)) ^ ctx.load(src3, i);
			ctx.store(dst, i, val);
		}
	}

	// Shifts
	fn exec_shift<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src = self.read_reg();
		// The builder only ever emits a valid discriminant, so the decode cannot fail.
		let variant =
			ShiftVariant::from_u8(self.read_u8()).expect("bytecode carries a valid shift variant");
		let amount = self.read_u8() as u32;
		// The variant is fixed for this instruction, so dispatch on it once, not per word.
		//
		// Each arm is then a branch-free tight loop, the shape Keccak's many rotations need.
		match variant {
			ShiftVariant::Sll => Self::shift_each(ctx, dst, src, |w| w << amount),
			ShiftVariant::Slr => Self::shift_each(ctx, dst, src, |w| w >> amount),
			ShiftVariant::Sar => Self::shift_each(ctx, dst, src, |w| w.sar(amount)),
			ShiftVariant::Rotr => Self::shift_each(ctx, dst, src, |w| w.rotr(amount)),
			ShiftVariant::Sll32 => Self::shift_each(ctx, dst, src, |w| w.sll32(amount)),
			ShiftVariant::Srl32 => Self::shift_each(ctx, dst, src, |w| w.srl32(amount)),
			ShiftVariant::Sra32 => Self::shift_each(ctx, dst, src, |w| w.sra32(amount)),
			ShiftVariant::Rotr32 => Self::shift_each(ctx, dst, src, |w| w.rotr32(amount)),
		}
	}

	/// Applies one fixed word-level shift across every instance.
	///
	/// The op is a distinct zero-sized closure per call site.
	/// So the compiler monomorphizes this into a branch-free tight loop with the shift inlined.
	#[inline]
	fn shift_each<C: EvalContext>(ctx: &mut C, dst: u32, src: u32, op: impl Fn(Word) -> Word) {
		for i in 0..ctx.n_instances() {
			ctx.store(dst, i, op(ctx.load(src, i)));
		}
	}

	// Arithmetic operations
	fn exec_iadd_cout<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let (sum, cout) = ctx
				.load(src1, i)
				.iadd_cin_cout(ctx.load(src2, i), Word::ZERO);
			ctx.store(dst_sum, i, sum);
			ctx.store(dst_cout, i, cout);
		}
	}

	fn exec_iadd_cin_cout<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let cin = self.read_reg();
		for i in 0..ctx.n_instances() {
			let cin_bit = ctx.load(cin, i) >> 63; // Use MSB as carry bit
			let (sum, cout) = ctx.load(src1, i).iadd_cin_cout(ctx.load(src2, i), cin_bit);
			ctx.store(dst_sum, i, sum);
			ctx.store(dst_cout, i, cout);
		}
	}

	fn exec_isub_bin_bout<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_diff = self.read_reg();
		let dst_bout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let bin = self.read_reg();
		for i in 0..ctx.n_instances() {
			let bin_bit = ctx.load(bin, i) >> 63; // Use MSB as borrow bit
			let (diff, bout) = ctx.load(src1, i).isub_bin_bout(ctx.load(src2, i), bin_bit);
			ctx.store(dst_diff, i, diff);
			ctx.store(dst_bout, i, bout);
		}
	}

	fn exec_imul<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_hi = self.read_reg();
		let dst_lo = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let (hi, lo) = ctx.load(src1, i).imul(ctx.load(src2, i));
			ctx.store(dst_hi, i, hi);
			ctx.store(dst_lo, i, lo);
		}
	}

	/// GHASH-field multiply: `(c_lo, c_hi) = (a_lo, a_hi) * (b_lo, b_hi)` in
	/// $\mathbb{F}_{2^{128}}$.
	///
	/// Operands are read in the order `dst_lo, dst_hi, a_lo, a_hi, b_lo, b_hi`, matching
	/// [`BytecodeBuilder::emit_bmul`](super::builder::BytecodeBuilder::emit_bmul).
	fn exec_bmul<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_lo = self.read_reg();
		let dst_hi = self.read_reg();
		let a_lo = self.read_reg();
		let a_hi = self.read_reg();
		let b_lo = self.read_reg();
		let b_hi = self.read_reg();
		for i in 0..ctx.n_instances() {
			let (lo, hi) = ghash_mul(
				ctx.load(a_lo, i),
				ctx.load(a_hi, i),
				ctx.load(b_lo, i),
				ctx.load(b_hi, i),
			);
			ctx.store(dst_lo, i, lo);
			ctx.store(dst_hi, i, hi);
		}
	}

	// 32-bit operations
	fn exec_iadd32_cin_cout<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let cin = self.read_reg();
		for i in 0..ctx.n_instances() {
			let (sum, cout) = ctx
				.load(src1, i)
				.iadd32_cin_cout(ctx.load(src2, i), ctx.load(cin, i));
			ctx.store(dst_sum, i, sum);
			ctx.store(dst_cout, i, cout);
		}
	}

	fn exec_iadd32_cout<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let (sum, cout) = ctx.load(src1, i).iadd_cout_32(ctx.load(src2, i));
			ctx.store(dst_sum, i, sum);
			ctx.store(dst_cout, i, cout);
		}
	}

	// Mask operations
	fn exec_mask_low<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let n_bits = self.read_u8();
		// The mask depends only on the immediate, so build it once for the whole batch.
		let mask = if n_bits >= 64 {
			Word::ALL_ONE
		} else {
			Word::from_u64((1u64 << n_bits) - 1)
		};
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i) & mask;
			ctx.store(dst, i, val);
		}
	}

	fn exec_mask_high<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let n_bits = self.read_u8();
		// The mask depends only on the immediate, so build it once for the whole batch.
		let mask = if n_bits >= 64 {
			Word::ALL_ONE
		} else {
			Word::from_u64(!((1u64 << (64 - n_bits)) - 1))
		};
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i) & mask;
			ctx.store(dst, i, val);
		}
	}

	// Assertions
	fn exec_assert_eq<C: EvalContext>(&mut self, ctx: &mut C) {
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		for i in 0..ctx.n_instances() {
			let val1 = ctx.load(src1, i);
			let val2 = ctx.load(src2, i);
			if val1 != val2 {
				ctx.note_assertion_failure(i, path_spec, format!("{val1:?} != {val2:?}"));
			}
		}
	}

	fn exec_assert_eq_cond<C: EvalContext>(&mut self, ctx: &mut C) {
		let cond = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		for i in 0..ctx.n_instances() {
			if ctx.load(cond, i).is_msb_true() {
				let val1 = ctx.load(src1, i);
				let val2 = ctx.load(src2, i);
				if val1 != val2 {
					ctx.note_assertion_failure(
						i,
						path_spec,
						format!("conditional assert: {val1:?} != {val2:?}"),
					);
				}
			}
		}
	}

	fn exec_assert_zero<C: EvalContext>(&mut self, ctx: &mut C) {
		let src = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i);
			if val != Word::ZERO {
				ctx.note_assertion_failure(i, path_spec, format!("{val:?} != 0"));
			}
		}
	}

	fn exec_assert_non_zero<C: EvalContext>(&mut self, ctx: &mut C) {
		let src = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i);
			if val == Word::ZERO {
				ctx.note_assertion_failure(i, path_spec, format!("{val:?} == 0"));
			}
		}
	}

	fn exec_assert_false<C: EvalContext>(&mut self, ctx: &mut C) {
		let src = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i);
			if val.is_msb_true() {
				ctx.note_assertion_failure(i, path_spec, format!("{val:?} MSB is true"));
			}
		}
	}

	fn exec_assert_true<C: EvalContext>(&mut self, ctx: &mut C) {
		let src = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i);
			if val.is_msb_false() {
				ctx.note_assertion_failure(i, path_spec, format!("{val:?} MSB is false"));
			}
		}
	}

	// Hint execution
	fn exec_hint<C: EvalContext>(&mut self, ctx: &mut C) {
		let hint_id = self.read_u32();

		// Read dimensions
		let n_dimensions = self.read_u16() as usize;
		let mut dimensions = Vec::with_capacity(n_dimensions);
		for _ in 0..n_dimensions {
			dimensions.push(self.read_u32() as usize);
		}

		let n_inputs = self.read_u16() as usize;
		let n_outputs = self.read_u16() as usize;

		// Read the input and output registers once; they are shared across every instance.
		let input_regs = (0..n_inputs).map(|_| self.read_reg()).collect::<Vec<_>>();
		let output_regs = (0..n_outputs).map(|_| self.read_reg()).collect::<Vec<_>>();

		let mut inputs = vec![Word::ZERO; n_inputs];
		let mut outputs = vec![Word::ZERO; n_outputs];
		for i in 0..ctx.n_instances() {
			for (input, &reg) in inputs.iter_mut().zip(&input_regs) {
				*input = ctx.load(reg, i);
			}
			self.hints
				.execute(hint_id, &dimensions, &inputs, &mut outputs);
			for (&reg, &output) in output_regs.iter().zip(&outputs) {
				ctx.store(reg, i, output);
			}
		}
	}

	// Bytecode reading helpers
	fn read_u8(&mut self) -> u8 {
		let val = self.bytecode[self.pc];
		self.pc += 1;
		val
	}

	fn read_u16(&mut self) -> u16 {
		let val = u16::from_le_bytes([self.bytecode[self.pc], self.bytecode[self.pc + 1]]);
		self.pc += 2;
		val
	}

	fn read_u32(&mut self) -> u32 {
		let val = u32::from_le_bytes([
			self.bytecode[self.pc],
			self.bytecode[self.pc + 1],
			self.bytecode[self.pc + 2],
			self.bytecode[self.pc + 3],
		]);
		self.pc += 4;
		val
	}

	fn read_reg(&mut self) -> u32 {
		self.read_u32()
	}
}
