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
use binius_field::Ghash128b;
use smallvec::{SmallVec, smallvec};

use super::opcode::EvalOpcode;
use crate::ir::{hints::HintRegistry, path::PathSpec};

/// Multiplies two GHASH field elements ($\mathbb{F}_{2^{128}}$), each carried by a `(lo, hi)` pair
/// of words — `lo` holds the coefficients of $1, X, \ldots, X^{63}$ and `hi` those of
/// $X^{64}, \ldots, X^{127}$ — and returns the product in the same `(lo, hi)` form.
pub(super) fn ghash_mul(a_lo: Word, a_hi: Word, b_lo: Word, b_hi: Word) -> (Word, Word) {
	let to_field =
		|lo: Word, hi: Word| Ghash128b::from((lo.as_u64() as u128) | ((hi.as_u64() as u128) << 64));
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

	/// Applies `op` to every instance, reading the `srcs` registers and writing the `dsts`.
	///
	/// Every destination register must differ from every source register.
	fn map<const D: usize, const S: usize, F>(&mut self, dsts: [u32; D], srcs: [u32; S], op: F)
	where
		F: Fn([Word; S]) -> [Word; D],
	{
		for i in 0..self.n_instances() {
			let out = op(srcs.map(|reg| self.load(reg, i)));
			for (dst, value) in dsts.into_iter().zip(out) {
				self.store(dst, i, value);
			}
		}
	}

	/// Folds `src` into `dst` across every instance.
	///
	/// `dst` must differ from `src`.
	fn update<F>(&mut self, dst: u32, src: u32, op: F)
	where
		F: Fn(Word, Word) -> Word,
	{
		for i in 0..self.n_instances() {
			let value = op(self.load(dst, i), self.load(src, i));
			self.store(dst, i, value);
		}
	}

	/// Records a failure against `path_spec` for every instance whose `srcs` satisfy `fails`.
	///
	/// `message` renders the same words into the detail line of that failure.
	/// It runs only for a failing instance, so it may allocate freely.
	fn check<const S: usize, P, M>(
		&mut self,
		srcs: [u32; S],
		path_spec: PathSpec,
		fails: P,
		message: M,
	) where
		P: Fn([Word; S]) -> bool,
		M: Fn([Word; S]) -> String,
	{
		for i in 0..self.n_instances() {
			let words = srcs.map(|reg| self.load(reg, i));
			if fails(words) {
				self.note_assertion_failure(i, path_spec, message(words));
			}
		}
	}

	/// Records an assertion violation for one instance.
	///
	/// The index is local to this context.
	/// A context that covers a stripe of a larger batch remaps it to a global index.
	fn note_assertion_failure(&mut self, instance: usize, path_spec: PathSpec, message: String);
}

/// A bytecode program together with a cursor into it.
///
/// The cursor advances as the dispatch loop consumes opcodes and operands.
/// One executor drives one pass over the bytecode, and [`Self::run`] consumes it.
/// So a cursor left at the end of the program can never be run a second time.
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
	pub fn run<C: EvalContext>(mut self, ctx: &mut C) {
		while self.pc < self.bytecode.len() {
			let byte = self.read_u8();
			let opcode = EvalOpcode::from_byte(byte)
				.unwrap_or_else(|| panic!("Unknown opcode: {byte:#x} at pc={}", self.pc - 1));

			// Matching the enum rather than the byte makes the dispatch exhaustive.
			// So a new opcode without a handler here does not compile.
			match opcode {
				// Bitwise operations
				EvalOpcode::Band => self.exec_band(ctx),
				EvalOpcode::Bor => self.exec_bor(ctx),
				EvalOpcode::Bxor => self.exec_bxor(ctx),
				EvalOpcode::Select => self.exec_select(ctx),
				EvalOpcode::BxorMulti => self.exec_bxor_multi(ctx),
				EvalOpcode::Fax => self.exec_fax(ctx),

				// Shifts
				EvalOpcode::Shift => self.exec_shift(ctx),

				// Arithmetic
				EvalOpcode::IaddCinCout => self.exec_iadd_cin_cout(ctx),
				EvalOpcode::IaddCarry => self.exec_iadd_carry(ctx),
				EvalOpcode::IsubBinBout => self.exec_isub_bin_bout(ctx),
				EvalOpcode::Imul => self.exec_imul(ctx),
				EvalOpcode::Bmul => self.exec_bmul(ctx),

				// 32-bit operations
				EvalOpcode::Iadd32CinCout => self.exec_iadd32_cin_cout(ctx),
				EvalOpcode::Iadd32Cout => self.exec_iadd32_cout(ctx),

				// Assertions
				EvalOpcode::AssertEq => self.exec_assert_eq(ctx),
				EvalOpcode::AssertEqCond => self.exec_assert_eq_cond(ctx),
				EvalOpcode::AssertZero => self.exec_assert_zero(ctx),
				EvalOpcode::AssertNonZero => self.exec_assert_non_zero(ctx),
				EvalOpcode::AssertFalse => self.exec_assert_false(ctx),
				EvalOpcode::AssertTrue => self.exec_assert_true(ctx),

				// Hint calls
				EvalOpcode::Hint => self.exec_hint(ctx),
			}
		}
	}

	// Bitwise operations
	fn exec_band<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		ctx.map([dst], [src1, src2], |[a, b]| [a & b]);
	}

	fn exec_bor<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		ctx.map([dst], [src1, src2], |[a, b]| [a | b]);
	}

	fn exec_bxor<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		ctx.map([dst], [src1, src2], |[a, b]| [a ^ b]);
	}

	fn exec_select<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let cond = self.read_reg();
		let t = self.read_reg();
		let f = self.read_reg();
		// Select t if MSB(cond) is 1, otherwise select f.
		ctx.map([dst], [cond, t, f], |[cond, t, f]| [if cond.is_msb_true() { t } else { f }]);
	}

	fn exec_bxor_multi<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let n = self.read_u32() as usize;
		// Read the source registers once; they are shared across every instance.
		// Most multi-way exclusive-ors are narrow, so the common case stays on the stack.
		let srcs = (0..n)
			.map(|_| self.read_reg())
			.collect::<SmallVec<[u32; 8]>>();
		// The destination doubles as the accumulator, so it is seeded before the fold runs.
		match srcs.split_first() {
			None => ctx.map([dst], [], |[]| [Word::ZERO]),
			Some((&first, rest)) => {
				ctx.map([dst], [first], |[a]| [a]);
				for &src in rest {
					ctx.update(dst, src, |acc, w| acc ^ w);
				}
			}
		}
	}

	fn exec_fax<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let src3 = self.read_reg();
		ctx.map([dst], [src1, src2, src3], |[a, b, c]| [(a & b) ^ c]);
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
		ctx.map([dst], [src], |[w]| [op(w)]);
	}

	// Arithmetic operations
	fn exec_iadd_cin_cout<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let cin = self.read_reg();
		ctx.map([dst_sum, dst_cout], [src1, src2, cin], |[a, b, cin]| {
			// The carry in is the MSB of its word.
			let (sum, cout) = a.iadd_cin_cout(b, cin >> 63);
			[sum, cout]
		});
	}

	/// Carry word of `src1 + src2`, discarding the sum.
	fn exec_iadd_carry<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		// No carry in, and the sum is dropped rather than stored.
		ctx.map([dst_cout], [src1, src2], |[a, b]| [a.iadd_cin_cout(b, Word::ZERO).1]);
	}

	fn exec_isub_bin_bout<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_diff = self.read_reg();
		let dst_bout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let bin = self.read_reg();
		ctx.map([dst_diff, dst_bout], [src1, src2, bin], |[a, b, bin]| {
			// The borrow in is the MSB of its word.
			let (diff, bout) = a.isub_bin_bout(b, bin >> 63);
			[diff, bout]
		});
	}

	fn exec_imul<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_hi = self.read_reg();
		let dst_lo = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		ctx.map([dst_hi, dst_lo], [src1, src2], |[a, b]| {
			let (hi, lo) = a.imul(b);
			[hi, lo]
		});
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
		ctx.map([dst_lo, dst_hi], [a_lo, a_hi, b_lo, b_hi], |[a_lo, a_hi, b_lo, b_hi]| {
			let (lo, hi) = ghash_mul(a_lo, a_hi, b_lo, b_hi);
			[lo, hi]
		});
	}

	// 32-bit operations
	fn exec_iadd32_cin_cout<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let cin = self.read_reg();
		ctx.map([dst_sum, dst_cout], [src1, src2, cin], |[a, b, cin]| {
			let (sum, cout) = a.iadd32_cin_cout(b, cin);
			[sum, cout]
		});
	}

	fn exec_iadd32_cout<C: EvalContext>(&mut self, ctx: &mut C) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		ctx.map([dst_sum, dst_cout], [src1, src2], |[a, b]| {
			let (sum, cout) = a.iadd_cout_32(b);
			[sum, cout]
		});
	}

	// Assertions
	fn exec_assert_eq<C: EvalContext>(&mut self, ctx: &mut C) {
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let path_spec = self.read_path_spec();
		ctx.check([src1, src2], path_spec, |[a, b]| a != b, |[a, b]| format!("{a:?} != {b:?}"));
	}

	fn exec_assert_eq_cond<C: EvalContext>(&mut self, ctx: &mut C) {
		let cond = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let path_spec = self.read_path_spec();
		ctx.check(
			[cond, src1, src2],
			path_spec,
			|[cond, a, b]| cond.is_msb_true() && a != b,
			|[_, a, b]| format!("conditional assert: {a:?} != {b:?}"),
		);
	}

	fn exec_assert_zero<C: EvalContext>(&mut self, ctx: &mut C) {
		let src = self.read_reg();
		let path_spec = self.read_path_spec();
		ctx.check([src], path_spec, |[v]| v != Word::ZERO, |[v]| format!("{v:?} != 0"));
	}

	fn exec_assert_non_zero<C: EvalContext>(&mut self, ctx: &mut C) {
		let src = self.read_reg();
		let path_spec = self.read_path_spec();
		ctx.check([src], path_spec, |[v]| v == Word::ZERO, |[v]| format!("{v:?} == 0"));
	}

	fn exec_assert_false<C: EvalContext>(&mut self, ctx: &mut C) {
		let src = self.read_reg();
		let path_spec = self.read_path_spec();
		ctx.check([src], path_spec, |[v]| v.is_msb_true(), |[v]| format!("{v:?} MSB is true"));
	}

	fn exec_assert_true<C: EvalContext>(&mut self, ctx: &mut C) {
		let src = self.read_reg();
		let path_spec = self.read_path_spec();
		ctx.check([src], path_spec, |[v]| v.is_msb_false(), |[v]| format!("{v:?} MSB is false"));
	}

	// Hint execution
	fn exec_hint<C: EvalContext>(&mut self, ctx: &mut C) {
		let hint_id = self.read_u32();

		// Read dimensions
		let n_dimensions = self.read_u32() as usize;
		let mut dimensions: SmallVec<[usize; 4]> = SmallVec::with_capacity(n_dimensions);
		for _ in 0..n_dimensions {
			dimensions.push(self.read_u32() as usize);
		}

		let n_inputs = self.read_u32() as usize;
		let n_outputs = self.read_u32() as usize;

		// Read the input and output registers once; they are shared across every instance.
		let input_regs = (0..n_inputs)
			.map(|_| self.read_reg())
			.collect::<SmallVec<[u32; 8]>>();
		let output_regs = (0..n_outputs)
			.map(|_| self.read_reg())
			.collect::<SmallVec<[u32; 8]>>();

		let mut inputs: SmallVec<[Word; 8]> = smallvec![Word::ZERO; n_inputs];
		let mut outputs: SmallVec<[Word; 8]> = smallvec![Word::ZERO; n_outputs];
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

	// Bytecode reading helpers.
	//
	// Each takes its bytes as one slice.
	// So a multi-byte value costs a single bounds check, not one per byte.
	//
	// This is the innermost decode of witness filling.
	fn read_u8(&mut self) -> u8 {
		let val = self.bytecode[self.pc];
		self.pc += 1;
		val
	}

	/// Reads the next `N` bytes as an array, advancing the cursor past them.
	#[inline]
	fn read_bytes<const N: usize>(&mut self) -> [u8; N] {
		let bytes: [u8; N] = self.bytecode[self.pc..self.pc + N]
			.try_into()
			.expect("the slice is exactly N bytes long");
		self.pc += N;
		bytes
	}

	fn read_u32(&mut self) -> u32 {
		u32::from_le_bytes(self.read_bytes())
	}

	fn read_reg(&mut self) -> u32 {
		self.read_u32()
	}

	/// Decodes the `u32` an assertion instruction carries on the wire back into a path spec.
	fn read_path_spec(&mut self) -> PathSpec {
		PathSpec::from_u32(self.read_u32())
	}
}
