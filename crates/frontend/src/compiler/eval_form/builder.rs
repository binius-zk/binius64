// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Bytecode builder for generating evaluation instructions

use binius_core::constraint_system::ShiftVariant;

/// Builder for constructing bytecode during circuit compilation
pub struct BytecodeBuilder {
	bytecode: Vec<u8>,
	n_eval_insn: usize,
}

impl BytecodeBuilder {
	pub const fn new() -> Self {
		Self {
			bytecode: Vec::new(),
			n_eval_insn: 0,
		}
	}

	// Bitwise operations
	pub fn emit_band(&mut self, dst: u32, src1: u32, src2: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x01);
		self.emit_reg(dst);
		self.emit_reg(src1);
		self.emit_reg(src2);
	}

	pub fn emit_bor(&mut self, dst: u32, src1: u32, src2: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x02);
		self.emit_reg(dst);
		self.emit_reg(src1);
		self.emit_reg(src2);
	}

	pub fn emit_bxor(&mut self, dst: u32, src1: u32, src2: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x03);
		self.emit_reg(dst);
		self.emit_reg(src1);
		self.emit_reg(src2);
	}

	pub fn emit_bxor_multi(&mut self, dst: u32, srcs: &[u32]) {
		self.n_eval_insn += 1;
		self.emit_u8(0x06); // Opcode for multi-way XOR
		self.emit_reg(dst);
		self.emit_u32(srcs.len() as u32);
		for &src in srcs {
			self.emit_reg(src);
		}
	}

	pub fn emit_fax(&mut self, dst: u32, src1: u32, src2: u32, src3: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x07);
		self.emit_reg(dst);
		self.emit_reg(src1);
		self.emit_reg(src2);
		self.emit_reg(src3);
	}

	pub fn emit_select(&mut self, dst: u32, cond: u32, t: u32, f: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x05);
		self.emit_reg(dst);
		self.emit_reg(cond);
		self.emit_reg(t);
		self.emit_reg(f);
	}

	/// One instruction covering every shift and rotate variant.
	///
	/// Layout: `[0x10][dst reg][src reg][variant u8][amount u8]`.
	/// The variant byte selects the shift operation.
	/// The amount byte is the shift count in bits.
	pub fn emit_shift(&mut self, dst: u32, src: u32, variant: ShiftVariant, amount: u8) {
		self.n_eval_insn += 1;
		self.emit_u8(0x10);
		self.emit_reg(dst);
		self.emit_reg(src);
		self.emit_u8(variant as u8);
		self.emit_u8(amount);
	}

	// Arithmetic with carry
	pub fn emit_iadd_cout(&mut self, dst_sum: u32, dst_cout: u32, src1: u32, src2: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x20);
		self.emit_reg(dst_sum);
		self.emit_reg(dst_cout);
		self.emit_reg(src1);
		self.emit_reg(src2);
	}

	pub fn emit_iadd_cin_cout(
		&mut self,
		dst_sum: u32,
		dst_cout: u32,
		src1: u32,
		src2: u32,
		cin: u32,
	) {
		self.n_eval_insn += 1;
		self.emit_u8(0x21);
		self.emit_reg(dst_sum);
		self.emit_reg(dst_cout);
		self.emit_reg(src1);
		self.emit_reg(src2);
		self.emit_reg(cin);
	}

	pub fn emit_isub_bin_bout(
		&mut self,
		dst_diff: u32,
		dst_bout: u32,
		src1: u32,
		src2: u32,
		bin: u32,
	) {
		self.n_eval_insn += 1;
		self.emit_u8(0x23);
		self.emit_reg(dst_diff);
		self.emit_reg(dst_bout);
		self.emit_reg(src1);
		self.emit_reg(src2);
		self.emit_reg(bin);
	}

	// Multiply
	pub fn emit_imul(&mut self, dst_hi: u32, dst_lo: u32, src1: u32, src2: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x30);
		self.emit_reg(dst_hi);
		self.emit_reg(dst_lo);
		self.emit_reg(src1);
		self.emit_reg(src2);
	}

	/// GHASH-field multiply: `(dst_lo, dst_hi) = (a_lo, a_hi) * (b_lo, b_hi)` in
	/// $\mathbb{F}_{2^{128}}$, where each field element is carried by a `(lo, hi)` pair of words.
	//
	// The first non-test caller is the BMUL gate (BINIUS-298); until then this is only reached from
	// tests, so silence the dead-code lint here rather than crate-wide.
	#[allow(dead_code)]
	pub fn emit_bmul(
		&mut self,
		dst_lo: u32,
		dst_hi: u32,
		a_lo: u32,
		a_hi: u32,
		b_lo: u32,
		b_hi: u32,
	) {
		self.n_eval_insn += 1;
		self.emit_u8(0x31);
		self.emit_reg(dst_lo);
		self.emit_reg(dst_hi);
		self.emit_reg(a_lo);
		self.emit_reg(a_hi);
		self.emit_reg(b_lo);
		self.emit_reg(b_hi);
	}

	// 32-bit operations
	pub fn emit_iadd32_cin_cout(
		&mut self,
		dst_sum: u32,
		dst_cout: u32,
		src1: u32,
		src2: u32,
		cin: u32,
	) {
		self.n_eval_insn += 1;
		self.emit_u8(0x40);
		self.emit_reg(dst_sum);
		self.emit_reg(dst_cout);
		self.emit_reg(src1);
		self.emit_reg(src2);
		self.emit_reg(cin);
	}

	pub fn emit_iadd32_cout(&mut self, dst_sum: u32, dst_cout: u32, src1: u32, src2: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x46);
		self.emit_reg(dst_sum);
		self.emit_reg(dst_cout);
		self.emit_reg(src1);
		self.emit_reg(src2);
	}

	// Assertions
	pub fn emit_assert_eq(&mut self, src1: u32, src2: u32, error_id: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x60);
		self.emit_reg(src1);
		self.emit_reg(src2);
		self.emit_u32(error_id);
	}

	pub fn emit_assert_eq_cond(&mut self, cond: u32, src1: u32, src2: u32, error_id: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x61);
		self.emit_reg(cond);
		self.emit_reg(src1);
		self.emit_reg(src2);
		self.emit_u32(error_id);
	}

	pub fn emit_assert_zero(&mut self, src: u32, error_id: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x62);
		self.emit_reg(src);
		self.emit_u32(error_id);
	}

	pub fn emit_assert_non_zero(&mut self, src: u32, error_id: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x63);
		self.emit_reg(src);
		self.emit_u32(error_id);
	}

	pub fn emit_assert_false(&mut self, src: u32, error_id: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x64);
		self.emit_reg(src);
		self.emit_u32(error_id);
	}

	pub fn emit_assert_true(&mut self, src: u32, error_id: u32) {
		self.n_eval_insn += 1;
		self.emit_u8(0x65);
		self.emit_reg(src);
		self.emit_u32(error_id);
	}

	// Hint calls
	pub fn emit_hint(
		&mut self,
		hint_id: u32,
		dimensions: &[usize],
		inputs: &[u32],
		outputs: &[u32],
	) {
		self.n_eval_insn += 1;
		self.emit_u8(0x80);
		self.emit_u32(hint_id);
		self.emit_u16(dimensions.len() as u16);
		for &dim in dimensions {
			self.emit_u32(dim as u32);
		}
		self.emit_u16(inputs.len() as u16);
		self.emit_u16(outputs.len() as u16);
		for &input in inputs {
			self.emit_reg(input);
		}
		for &output in outputs {
			self.emit_reg(output);
		}
	}

	// Low-level emitters
	fn emit_u8(&mut self, val: u8) {
		self.bytecode.push(val);
	}

	fn emit_u16(&mut self, val: u16) {
		self.bytecode.extend_from_slice(&val.to_le_bytes());
	}

	fn emit_u32(&mut self, val: u32) {
		self.bytecode.extend_from_slice(&val.to_le_bytes());
	}

	fn emit_reg(&mut self, reg: u32) {
		self.emit_u32(reg);
	}

	pub fn finalize(self) -> (Vec<u8>, usize) {
		(self.bytecode, self.n_eval_insn)
	}
}

impl Default for BytecodeBuilder {
	fn default() -> Self {
		Self::new()
	}
}
