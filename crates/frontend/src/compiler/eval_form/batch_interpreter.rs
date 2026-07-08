// Copyright 2025-2026 The Binius Developers
//! Batched bytecode interpreter for circuit evaluation.
//!
//! This is the structure-of-arrays counterpart to [`Interpreter`](super::interpreter::Interpreter).
//! It evaluates the same bytecode over `n` independent instances of one circuit at once.
//!
//! The value vector is transposed into a 2D array whose rows are value-vector indices (wires) and
//! whose columns are instances. A "register" is therefore a whole row, and executing an
//! instruction applies its scalar operation across the entire row — every instance in one pass.
//! This is the memory order the batch prover wants downstream.
//!
//! ```text
//!                  instance 0   instance 1   ...   instance n-1
//!   value index 0 [   w        |   w        | ... |   w        ]   <- one row = one register
//!   value index 1 [   w        |   w        | ... |   w        ]
//!         ...
//! ```

use binius_core::Word;
use binius_utils::strided_array::StridedArray2DViewMut;

use crate::compiler::{
	circuit::PopulateError,
	hints::HintRegistry,
	pathspec::{PathSpec, PathSpecTree},
};

/// The cap on how many assertion failures are retained across the whole batch.
///
/// This mirrors the single-instance interpreter's cap. Failures past it are counted but not stored.
const MAX_ASSERTION_FAILURES: usize = 100;

/// A single assertion failure, tagged with the instance whose values violated it.
struct InstanceAssertionFailure {
	instance: usize,
	path_spec: PathSpec,
	message: String,
}

/// The failure of batch witness population, attributed to a single instance.
///
/// When several instances fail, this reports the one with the lowest index, matching what
/// populating that instance on its own would produce.
#[derive(Debug)]
pub struct BatchPopulateError {
	/// The index of the reported instance, the lowest-numbered failing instance.
	pub instance: usize,
	/// The assertion failures recorded for that instance.
	pub source: PopulateError,
}

/// Execution context holding the transposed value array during batch evaluation.
struct BatchExecutionContext<'a, 'v> {
	/// Rows are value-vector indices; columns are instances.
	values: &'a mut StridedArray2DViewMut<'v, Word>,
	/// The global instance index represented by local column 0.
	instance_offset: usize,
	/// Assertion failures recorded during evaluation, capped by [`MAX_ASSERTION_FAILURES`].
	failures: Vec<InstanceAssertionFailure>,
	/// The total number of assertion violations recorded, across all instances.
	total_count: usize,
	/// The lowest-indexed instance that has failed an assertion, tracked even past the cap.
	min_failing_instance: Option<usize>,
}

impl<'a, 'v> BatchExecutionContext<'a, 'v> {
	const fn new(values: &'a mut StridedArray2DViewMut<'v, Word>, instance_offset: usize) -> Self {
		Self {
			values,
			instance_offset,
			failures: Vec::new(),
			total_count: 0,
			min_failing_instance: None,
		}
	}

	/// The number of instances, i.e. the number of columns in the value array.
	const fn n_instances(&self) -> usize {
		self.values.width()
	}

	#[inline]
	fn load(&self, reg: u32, instance: usize) -> Word {
		self.values[(reg as usize, instance)]
	}

	#[inline]
	fn store(&mut self, reg: u32, instance: usize, value: Word) {
		self.values[(reg as usize, instance)] = value;
	}

	/// Record an assertion failure for `instance`.
	///
	/// The failure may be dropped from the stored list once the cap is reached, but it always
	/// updates the count and the lowest-failing-instance tracker.
	#[cold]
	fn note_assertion_failure(&mut self, instance: usize, path_spec: PathSpec, message: String) {
		let instance = self.instance_offset + instance;
		self.total_count += 1;
		self.min_failing_instance = Some(
			self.min_failing_instance
				.map_or(instance, |m| m.min(instance)),
		);
		if self.failures.len() < MAX_ASSERTION_FAILURES {
			self.failures.push(InstanceAssertionFailure {
				instance,
				path_spec,
				message,
			});
		}
	}

	/// Turn recorded failures into an error attributed to the lowest-failing instance.
	fn check_assertions(
		self,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		let Some(instance) = self.min_failing_instance else {
			return Ok(());
		};

		// Collect and symbolicate just the reported instance's messages.
		let mut messages = Vec::new();
		let mut total_count = 0;
		for failure in self.failures.into_iter().filter(|f| f.instance == instance) {
			total_count += 1;
			let message = if let Some(tree) = path_spec_tree {
				let mut path = String::new();
				tree.stringify(failure.path_spec, &mut path);
				if path.is_empty() {
					failure.message
				} else {
					format!("{}: {}", path, failure.message)
				}
			} else {
				failure.message
			};
			messages.push(message);
		}

		Err(BatchPopulateError {
			instance,
			source: PopulateError {
				messages,
				total_count,
			},
		})
	}
}

/// Bytecode interpreter that evaluates one circuit over many instances at once.
pub struct BatchInterpreter<'a> {
	bytecode: &'a [u8],
	hints: &'a HintRegistry,
	pc: usize,
}

impl<'a> BatchInterpreter<'a> {
	pub const fn new(bytecode: &'a [u8], hints: &'a HintRegistry) -> Self {
		Self {
			bytecode,
			hints,
			pc: 0,
		}
	}

	/// Evaluate the bytecode over the transposed value array, filling every instance's wires.
	///
	/// The constant and input rows must already be populated for every instance. Returns an error
	/// naming the lowest-indexed instance whose assignment fails an assertion.
	pub fn run(
		&mut self,
		values: &mut StridedArray2DViewMut<'_, Word>,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		self.run_with_instance_offset(values, 0, path_spec_tree)
	}

	/// Evaluate the bytecode over a view whose local column 0 corresponds to `instance_offset`.
	pub(crate) fn run_with_instance_offset(
		&mut self,
		values: &mut StridedArray2DViewMut<'_, Word>,
		instance_offset: usize,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		let mut ctx = BatchExecutionContext::new(values, instance_offset);
		while self.pc < self.bytecode.len() {
			let opcode = self.read_u8();
			match opcode {
				// Bitwise operations
				0x01 => self.exec_band(&mut ctx),
				0x02 => self.exec_bor(&mut ctx),
				0x03 => self.exec_bxor(&mut ctx),
				0x05 => self.exec_select(&mut ctx),
				0x06 => self.exec_bxor_multi(&mut ctx),
				0x07 => self.exec_fax(&mut ctx),

				// Shifts
				0x10 => self.exec_sll(&mut ctx),
				0x11 => self.exec_slr(&mut ctx),
				0x12 => self.exec_sar(&mut ctx),

				// Arithmetic
				0x20 => self.exec_iadd_cout(&mut ctx),
				0x21 => self.exec_iadd_cin_cout(&mut ctx),
				0x23 => self.exec_isub_bin_bout(&mut ctx),
				0x30 => self.exec_imul(&mut ctx),

				// 32-bit operations
				0x40 => self.exec_iadd32_cin_cout(&mut ctx),
				0x41 => self.exec_rotr32(&mut ctx),
				0x42 => self.exec_srl32(&mut ctx),
				0x43 => self.exec_rotr(&mut ctx),
				0x44 => self.exec_sll32(&mut ctx),
				0x45 => self.exec_sra32(&mut ctx),
				0x46 => self.exec_iadd32_cout(&mut ctx),

				// Masks
				0x50 => self.exec_mask_low(&mut ctx),
				0x51 => self.exec_mask_high(&mut ctx),

				// Assertions
				0x60 => self.exec_assert_eq(&mut ctx),
				0x61 => self.exec_assert_eq_cond(&mut ctx),
				0x62 => self.exec_assert_zero(&mut ctx),
				0x63 => self.exec_assert_non_zero(&mut ctx),
				0x64 => self.exec_assert_false(&mut ctx),
				0x65 => self.exec_assert_true(&mut ctx),

				// Hint calls
				0x80 => self.exec_hint(&mut ctx),

				_ => panic!("Unknown opcode: {:#x} at pc={}", opcode, self.pc - 1),
			}
		}
		ctx.check_assertions(path_spec_tree)
	}

	// Bitwise operations
	fn exec_band(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src1, i) & ctx.load(src2, i);
			ctx.store(dst, i, val);
		}
	}

	fn exec_bor(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src1, i) | ctx.load(src2, i);
			ctx.store(dst, i, val);
		}
	}

	fn exec_bxor(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src1 = self.read_reg();
		let src2 = self.read_reg();
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src1, i) ^ ctx.load(src2, i);
			ctx.store(dst, i, val);
		}
	}

	fn exec_select(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let cond = self.read_reg();
		let t = self.read_reg();
		let f = self.read_reg();
		for i in 0..ctx.n_instances() {
			// Select t if MSB(cond) is 1, otherwise select f
			let val = if ctx.load(cond, i).is_msb_true() {
				ctx.load(t, i)
			} else {
				ctx.load(f, i)
			};
			ctx.store(dst, i, val);
		}
	}

	fn exec_bxor_multi(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_fax(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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
	fn exec_sll(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i) << shift;
			ctx.store(dst, i, val);
		}
	}

	fn exec_slr(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i) >> shift;
			ctx.store(dst, i, val);
		}
	}

	fn exec_sar(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i).sar(shift);
			ctx.store(dst, i, val);
		}
	}

	// Arithmetic operations
	fn exec_iadd_cout(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_iadd_cin_cout(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_isub_bin_bout(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_imul(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	// 32-bit operations
	fn exec_iadd32_cin_cout(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_iadd32_cout(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_rotr32(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let rotate = self.read_u8() as u32;
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i).rotr32(rotate);
			ctx.store(dst, i, val);
		}
	}

	fn exec_srl32(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i).srl32(shift);
			ctx.store(dst, i, val);
		}
	}

	fn exec_sll32(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i).sll32(shift);
			ctx.store(dst, i, val);
		}
	}

	fn exec_sra32(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let shift = self.read_u8() as u32;
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i).sra32(shift);
			ctx.store(dst, i, val);
		}
	}

	fn exec_rotr(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let rotate = self.read_u8() as u32;
		for i in 0..ctx.n_instances() {
			let val = ctx.load(src, i).rotr(rotate);
			ctx.store(dst, i, val);
		}
	}

	// Mask operations
	fn exec_mask_low(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let n_bits = self.read_u8();
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

	fn exec_mask_high(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
		let dst = self.read_reg();
		let src = self.read_reg();
		let n_bits = self.read_u8();
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
	fn exec_assert_eq(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_assert_eq_cond(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_assert_zero(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_assert_non_zero(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_assert_false(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

	fn exec_assert_true(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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
	fn exec_hint(&mut self, ctx: &mut BatchExecutionContext<'_, '_>) {
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

#[cfg(test)]
mod tests {
	use binius_core::Word;
	use binius_utils::strided_array::StridedArray2DViewMut;

	use crate::compiler::CircuitBuilder;

	// The batched interpreter must reproduce, for every instance, exactly what the single-instance
	// interpreter produces for the same inputs. This is the core equivalence guarantee.
	#[test]
	fn batched_matches_scalar_per_instance() {
		// A circuit that exercises a spread of opcodes plus a constant, with only witness inputs
		// and force-committed outputs (no inout wires — the M4 setting).
		let builder = CircuitBuilder::new();
		let a = builder.add_witness();
		let b = builder.add_witness();
		let k = builder.add_constant_64(0x0123_4567_89ab_cdef);
		let c = builder.band(a, b);
		let d = builder.bxor(a, k);
		let (sum, _cout) = builder.iadd(a, b);
		let e = builder.rotr(b, 7);
		let f = builder.bor(c, e);
		builder.force_commit(c);
		builder.force_commit(d);
		builder.force_commit(sum);
		builder.force_commit(f);
		let circuit = builder.build();

		let layout = circuit.constraint_system().value_vec_layout.clone();
		assert_eq!(layout.n_inout, 0, "fixture should have no inout wires");
		let combined = layout.combined_len();
		let full_len = combined + layout.n_scratch;
		let n = 8usize;

		// Distinct inputs per instance.
		let inputs: Vec<(u64, u64)> = (0..n)
			.map(|i| {
				let i = i as u64;
				(i.wrapping_mul(0x9e37_79b9_7f4a_7c15), i ^ 0x0000_0000_dead_beef)
			})
			.collect();

		// Single-instance reference: populate each instance on its own.
		let scalar: Vec<Vec<Word>> = inputs
			.iter()
			.map(|&(x, y)| {
				let mut filler = circuit.new_witness_filler();
				filler[a] = Word(x);
				filler[b] = Word(y);
				circuit.populate_wire_witness(&mut filler).unwrap();
				filler.value_vec().combined_witness().to_vec()
			})
			.collect();

		// Batched: fill the input rows for every instance, then evaluate all at once.
		let a_row = circuit.witness_index(a).0 as usize;
		let b_row = circuit.witness_index(b).0 as usize;
		let mut data = vec![Word::ZERO; full_len * n];
		let mut view = StridedArray2DViewMut::without_stride(&mut data, full_len, n).unwrap();
		for (instance, &(x, y)) in inputs.iter().enumerate() {
			view[(a_row, instance)] = Word(x);
			view[(b_row, instance)] = Word(y);
		}
		circuit.populate_wire_witness_batched(&mut view).unwrap();

		// Every instance's committed prefix must equal the single-instance witness.
		for instance in 0..n {
			for row in 0..combined {
				assert_eq!(
					view[(row, instance)],
					scalar[instance][row],
					"mismatch at row {row}, instance {instance}"
				);
			}
		}
	}

	// A batched run must flag the lowest-indexed instance whose inputs violate an assertion.
	#[test]
	fn batched_reports_lowest_failing_instance() {
		// Assert a == b; instances where a != b fail.
		let builder = CircuitBuilder::new();
		let a = builder.add_witness();
		let b = builder.add_witness();
		builder.assert_eq("a_eq_b", a, b);
		let circuit = builder.build();

		let layout = circuit.constraint_system().value_vec_layout.clone();
		let full_len = layout.combined_len() + layout.n_scratch;
		let n = 4usize;

		// Instances 2 and 3 violate a == b; instance 2 is the lowest.
		let inputs = [(1u64, 1u64), (7, 7), (4, 5), (9, 8)];
		let a_row = circuit.witness_index(a).0 as usize;
		let b_row = circuit.witness_index(b).0 as usize;
		let mut data = vec![Word::ZERO; full_len * n];
		let mut view = StridedArray2DViewMut::without_stride(&mut data, full_len, n).unwrap();
		for (instance, &(x, y)) in inputs.iter().enumerate() {
			view[(a_row, instance)] = Word(x);
			view[(b_row, instance)] = Word(y);
		}

		let err = circuit
			.populate_wire_witness_batched(&mut view)
			.expect_err("instances 2 and 3 violate a == b");
		assert_eq!(err.instance, 2);
		assert_eq!(err.source.total_count, 1);
		assert_eq!(
			err.source.messages,
			vec![".a_eq_b: Word(0x0000000000000004) != Word(0x0000000000000005)".to_string()]
		);
	}
}
