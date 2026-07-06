// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Bytecode interpreter for circuit evaluation

use binius_core::{ValueVec, Word};

use crate::compiler::{
	circuit::PopulateError,
	hints::HintRegistry,
	pathspec::{PathSpec, PathSpecTree},
};

const MAX_ASSERTION_FAILURES: usize = 100;

/// Assertion failure information
pub struct AssertionFailure {
	pub path_spec: PathSpec,
	pub message: String,
}

/// Raw shareable view over a [`ValueVec`]'s backing words.
///
/// Safety contract: parallel executions may share a store only when each task writes a disjoint set
/// of value indices, and any concurrently shared reads are from already-populated input or constant
/// values.
#[derive(Clone, Copy)]
pub(super) struct ValueStore {
	ptr: *mut Word,
	len: usize,
}

// SAFETY: `ValueStore` only becomes `Send`/`Sync` under the disjoint-writer contract documented on
// the type. The parallel evaluator derives those disjoint sets from gate-graph connected
// components while holding an exclusive `&mut ValueVec` for the entire run.
unsafe impl Send for ValueStore {}
unsafe impl Sync for ValueStore {}

impl ValueStore {
	pub(super) fn new(value_vec: &mut ValueVec) -> Self {
		let values = value_vec.as_mut_slice();
		Self {
			ptr: values.as_mut_ptr(),
			len: values.len(),
		}
	}

	#[inline]
	fn get(self, index: u32) -> Word {
		let index = index as usize;
		assert!(index < self.len, "value index {index} out of bounds");
		// SAFETY: The index is bounds-checked above. The caller must uphold the store's aliasing
		// contract.
		unsafe { *self.ptr.add(index) }
	}

	#[inline]
	fn set(self, index: u32, value: Word) {
		let index = index as usize;
		assert!(index < self.len, "value index {index} out of bounds");
		// SAFETY: The index is bounds-checked above. The caller must uphold the store's aliasing
		// contract.
		unsafe {
			*self.ptr.add(index) = value;
		}
	}
}

/// Execution context holds access to witness values during execution.
pub struct ExecutionContext {
	store: ValueStore,
	/// Assertion failures recorded during the evaluation of the circuit.
	///
	/// This list is capped by [`MAX_ASSERTION_FAILURES`].
	assertion_failures: Vec<AssertionFailure>,
	/// The total number of assert violations recorded.
	assertion_count: usize,
}

impl ExecutionContext {
	pub const fn new(store: ValueStore) -> Self {
		Self {
			store,
			assertion_failures: Vec::new(),
			assertion_count: 0,
		}
	}

	pub fn into_failures(self) -> (Vec<AssertionFailure>, usize) {
		(self.assertion_failures, self.assertion_count)
	}

	/// Record an assertion failure with the given path spec and message.
	///
	/// Note that this assertion might be discarded in case there is already too many recorded
	/// assertions.
	#[cold]
	fn note_assertion_failure(&mut self, path_spec: PathSpec, message: String) {
		self.assertion_count += 1;
		if self.assertion_failures.len() < MAX_ASSERTION_FAILURES {
			self.assertion_failures
				.push(AssertionFailure { path_spec, message });
		}
	}

	/// Check assertions and return error if any failed
	pub fn check_assertions(
		self,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), PopulateError> {
		if !self.assertion_failures.is_empty() {
			let messages = if let Some(tree) = path_spec_tree {
				// Symbolicate the path specs
				self.assertion_failures
					.into_iter()
					.map(|f| {
						let mut path = String::new();
						tree.stringify(f.path_spec, &mut path);
						if path.is_empty() {
							f.message
						} else {
							format!("{}: {}", path, f.message)
						}
					})
					.collect()
			} else {
				// No tree provided, just use messages as-is
				self.assertion_failures
					.into_iter()
					.map(|f| f.message)
					.collect()
			};

			Err(PopulateError {
				messages,
				total_count: self.assertion_count,
			})
		} else {
			Ok(())
		}
	}
}

pub(super) fn report_failures(
	mut failures: Vec<AssertionFailure>,
	total_count: usize,
	path_spec_tree: Option<&PathSpecTree>,
) -> Result<(), PopulateError> {
	if failures.is_empty() {
		return Ok(());
	}

	failures.truncate(MAX_ASSERTION_FAILURES);
	let messages = if let Some(tree) = path_spec_tree {
		failures
			.into_iter()
			.map(|f| {
				let mut path = String::new();
				tree.stringify(f.path_spec, &mut path);
				if path.is_empty() {
					f.message
				} else {
					format!("{}: {}", path, f.message)
				}
			})
			.collect()
	} else {
		failures.into_iter().map(|f| f.message).collect()
	};

	Err(PopulateError {
		messages,
		total_count,
	})
}

pub struct Interpreter<'a> {
	bytecode: &'a [u8],
	hints: &'a HintRegistry,
	pc: usize,
}

impl<'a> Interpreter<'a> {
	pub const fn new(bytecode: &'a [u8], hints: &'a HintRegistry) -> Self {
		Self {
			bytecode,
			hints,
			pc: 0,
		}
	}

	pub fn run_with_value_vec(
		&mut self,
		value_vec: &mut ValueVec,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), PopulateError> {
		let mut ctx = ExecutionContext::new(ValueStore::new(value_vec));
		self.run(&mut ctx)?;
		ctx.check_assertions(path_spec_tree)
	}

	pub fn run_with_store(
		&mut self,
		store: ValueStore,
	) -> Result<(Vec<AssertionFailure>, usize), PopulateError> {
		let mut ctx = ExecutionContext::new(store);
		self.run(&mut ctx)?;
		Ok(ctx.into_failures())
	}

	pub fn run(&mut self, ctx: &mut ExecutionContext) -> Result<(), PopulateError> {
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
				0x10 => self.exec_sll(ctx),
				0x11 => self.exec_slr(ctx),
				0x12 => self.exec_sar(ctx),

				// Arithmetic
				0x20 => self.exec_iadd_cout(ctx),
				0x21 => self.exec_iadd_cin_cout(ctx),
				0x23 => self.exec_isub_bin_bout(ctx),
				0x30 => self.exec_imul(ctx),
				0x31 => self.exec_smul(ctx),

				// 32-bit operations
				0x40 => self.exec_iadd32_cin_cout(ctx),
				0x41 => self.exec_rotr32(ctx),
				0x42 => self.exec_srl32(ctx),
				0x43 => self.exec_rotr(ctx),
				0x44 => self.exec_sll32(ctx),
				0x45 => self.exec_sra32(ctx),
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
		Ok(())
	}

	// Bitwise operations
	fn exec_band(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let val = self.load(ctx, src1) & self.load(ctx, src2);
		self.store(ctx, dst, val);
	}

	fn exec_bor(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let val = self.load(ctx, src1) | self.load(ctx, src2);
		self.store(ctx, dst, val);
	}

	fn exec_bxor(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let val = self.load(ctx, src1) ^ self.load(ctx, src2);
		self.store(ctx, dst, val);
	}

	fn exec_select(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let cond = self.read_reg();
		let t = self.read_reg();
		let f = self.read_reg();
		// Select t if MSB(cond) is 1, otherwise select f
		let cond_val = self.load(ctx, cond);
		let val = if cond_val.is_msb_true() {
			self.load(ctx, t)
		} else {
			self.load(ctx, f)
		};
		self.store(ctx, dst, val);
	}

	fn exec_bxor_multi(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let n = self.read_u32() as usize;
		let mut val = Word::ZERO;
		for _ in 0..n {
			let src = self.read_reg();
			val = val ^ self.load(ctx, src);
		}
		self.store(ctx, dst, val);
	}

	fn exec_fax(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let src3 = self.read_reg();
		let val = (self.load(ctx, src1) & self.load(ctx, src2)) ^ self.load(ctx, src3);
		self.store(ctx, dst, val);
	}

	// Shifts
	fn exec_sll(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		let val = self.load(ctx, src) << shift;
		self.store(ctx, dst, val);
	}

	fn exec_slr(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		let val = self.load(ctx, src) >> shift;
		self.store(ctx, dst, val);
	}

	fn exec_sar(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		let val = self.load(ctx, src).sar(shift);
		self.store(ctx, dst, val);
	}

	// Arithmetic operations
	fn exec_iadd_cout(&mut self, ctx: &mut ExecutionContext) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let (sum, cout) = self
			.load(ctx, src1)
			.iadd_cin_cout(self.load(ctx, src2), Word::ZERO);
		self.store(ctx, dst_sum, sum);
		self.store(ctx, dst_cout, cout);
	}

	fn exec_iadd_cin_cout(&mut self, ctx: &mut ExecutionContext) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let cin = self.read_reg();
		let cin_bit = self.load(ctx, cin) >> 63; // Use MSB as carry bit
		let (sum, cout) = self
			.load(ctx, src1)
			.iadd_cin_cout(self.load(ctx, src2), cin_bit);
		self.store(ctx, dst_sum, sum);
		self.store(ctx, dst_cout, cout);
	}

	fn exec_isub_bin_bout(&mut self, ctx: &mut ExecutionContext) {
		let dst_diff = self.read_reg();
		let dst_bout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let bin = self.read_reg();
		let bin_bit = self.load(ctx, bin) >> 63; // Use MSB as borrow bit
		let (diff, bout) = self
			.load(ctx, src1)
			.isub_bin_bout(self.load(ctx, src2), bin_bit);
		self.store(ctx, dst_diff, diff);
		self.store(ctx, dst_bout, bout);
	}

	fn exec_imul(&mut self, ctx: &mut ExecutionContext) {
		let dst_hi = self.read_reg();
		let dst_lo = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let (hi, lo) = self.load(ctx, src1).imul(self.load(ctx, src2));
		self.store(ctx, dst_hi, hi);
		self.store(ctx, dst_lo, lo);
	}

	fn exec_smul(&mut self, ctx: &mut ExecutionContext) {
		let dst_hi = self.read_reg();
		let dst_lo = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let (hi, lo) = self.load(ctx, src1).smul(self.load(ctx, src2));
		self.store(ctx, dst_hi, hi);
		self.store(ctx, dst_lo, lo);
	}

	// 32-bit operations
	fn exec_iadd32_cin_cout(&mut self, ctx: &mut ExecutionContext) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let cin = self.read_reg();
		let (sum, cout) = self
			.load(ctx, src1)
			.iadd32_cin_cout(self.load(ctx, src2), self.load(ctx, cin));
		self.store(ctx, dst_sum, sum);
		self.store(ctx, dst_cout, cout);
	}

	fn exec_iadd32_cout(&mut self, ctx: &mut ExecutionContext) {
		let dst_sum = self.read_reg();
		let dst_cout = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let (sum, cout) = self.load(ctx, src1).iadd_cout_32(self.load(ctx, src2));
		self.store(ctx, dst_sum, sum);
		self.store(ctx, dst_cout, cout);
	}

	fn exec_rotr32(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let rotate = self.read_u8() as u32;
		let val = self.load(ctx, src).rotr32(rotate);
		self.store(ctx, dst, val);
	}

	fn exec_srl32(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		let val = self.load(ctx, src).srl32(shift);
		self.store(ctx, dst, val);
	}

	fn exec_sll32(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		let val = self.load(ctx, src).sll32(shift);
		self.store(ctx, dst, val);
	}

	fn exec_sra32(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		let val = self.load(ctx, src).sra32(shift);
		self.store(ctx, dst, val);
	}

	fn exec_rotr(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let rotate = self.read_u8() as u32;
		let val = self.load(ctx, src).rotr(rotate);
		self.store(ctx, dst, val);
	}

	// Mask operations
	fn exec_mask_low(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let n_bits = self.read_u8();
		let mask = if n_bits >= 64 {
			Word::ALL_ONE
		} else {
			Word::from_u64((1u64 << n_bits) - 1)
		};
		let val = self.load(ctx, src) & mask;
		self.store(ctx, dst, val);
	}

	fn exec_mask_high(&mut self, ctx: &mut ExecutionContext) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let n_bits = self.read_u8();
		let mask = if n_bits >= 64 {
			Word::ALL_ONE
		} else {
			Word::from_u64(!((1u64 << (64 - n_bits)) - 1))
		};
		let val = self.load(ctx, src) & mask;
		self.store(ctx, dst, val);
	}

	// Assertions
	fn exec_assert_eq(&mut self, ctx: &mut ExecutionContext) {
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		let val1 = self.load(ctx, src1);
		let val2 = self.load(ctx, src2);

		if val1 != val2 {
			ctx.note_assertion_failure(path_spec, format!("{val1:?} != {val2:?}"));
		}
	}

	fn exec_assert_eq_cond(&mut self, ctx: &mut ExecutionContext) {
		let cond = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		let cond_val = self.load(ctx, cond);

		if cond_val.is_msb_true() {
			let val1 = self.load(ctx, src1);
			let val2 = self.load(ctx, src2);

			if val1 != val2 {
				ctx.note_assertion_failure(
					path_spec,
					format!("conditional assert: {val1:?} != {val2:?}"),
				);
			}
		}
	}

	fn exec_assert_zero(&mut self, ctx: &mut ExecutionContext) {
		let src = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		let val = self.load(ctx, src);

		if val != Word::ZERO {
			ctx.note_assertion_failure(path_spec, format!("{val:?} != 0"));
		}
	}

	fn exec_assert_non_zero(&mut self, ctx: &mut ExecutionContext) {
		let src = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		let val = self.load(ctx, src);

		if val == Word::ZERO {
			ctx.note_assertion_failure(path_spec, format!("{val:?} == 0"));
		}
	}

	fn exec_assert_false(&mut self, ctx: &mut ExecutionContext) {
		let src = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		let val = self.load(ctx, src);

		if val.is_msb_true() {
			ctx.note_assertion_failure(path_spec, format!("{val:?} MSB is true"));
		}
	}

	fn exec_assert_true(&mut self, ctx: &mut ExecutionContext) {
		let src = self.read_reg();
		let error_id = self.read_u32();
		let path_spec = PathSpec::from_u32(error_id);

		let val = self.load(ctx, src);

		if val.is_msb_false() {
			ctx.note_assertion_failure(path_spec, format!("{val:?} MSB is false"));
		}
	}

	// Hint execution
	fn exec_hint(&mut self, ctx: &mut ExecutionContext) {
		let hint_id = self.read_u32();

		// Read dimensions
		let n_dimensions = self.read_u16() as usize;
		let mut dimensions = Vec::with_capacity(n_dimensions);
		for _ in 0..n_dimensions {
			dimensions.push(self.read_u32() as usize);
		}

		let n_inputs = self.read_u16() as usize;
		let n_outputs = self.read_u16() as usize;

		// Collect input values
		let mut inputs = Vec::with_capacity(n_inputs);
		for _ in 0..n_inputs {
			let reg = self.read_reg();
			inputs.push(self.load(ctx, reg));
		}

		// Prepare output buffer
		let mut outputs = vec![Word::ZERO; n_outputs];

		self.hints
			.execute(hint_id, &dimensions, &inputs, &mut outputs);

		// Store outputs
		for output_val in outputs {
			let dst = self.read_reg();
			self.store(ctx, dst, output_val);
		}
	}

	fn load(&self, ctx: &ExecutionContext, reg: u32) -> Word {
		ctx.store.get(reg)
	}

	fn store(&self, ctx: &mut ExecutionContext, reg: u32, value: Word) {
		ctx.store.set(reg, value);
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
