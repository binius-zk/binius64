// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Circuit statistics module for analyzing constraint counts and circuit complexity.

use std::fmt;

use binius_core::{ConstraintSystem, Operand, ShiftedValueIndex};
use itertools::chain;
use rustc_hash::FxHashSet;

use crate::compiler::circuit::Circuit;

/// Various stats of a circuit that affect the prover performance.
pub struct CircuitStat {
	/// Number of gates in the circuit.
	pub n_gates: usize,
	/// Number of instructions in the evaluation form of circuit.
	///
	/// Directly proportional to performance of witness filling.
	pub n_eval_insn: usize,
	/// Number of AND constraints in the circuit.
	///
	/// Affects performance of AND reduction.
	pub n_and_constraints: usize,
	/// Number of IMUL constraints in the circuit.
	///
	/// Affects performance of intmul reduction phase.
	pub n_imul_constraints: usize,
	/// Number of BMUL constraints in the circuit.
	///
	/// Affects performance of binmul reduction phase.
	pub n_bmul_constraints: usize,
	/// Number of distinct value indices with non-zero shift in the circuit.
	///
	/// Every use of a value with a distinct type and amount is counted here.
	///
	/// Affects performance of shift reduction phase.
	pub distinct_shifted_value_indices: usize,
	/// Number of distinct value indices with zero shift in the circuit.
	///
	/// Affects performance of shift reduction phase.
	pub distinct_unshifted_value_indices: usize,
	/// Length of the value vector.
	///
	/// Affects performance of committing.
	pub value_vec_len: usize,
	/// Number of constant values used by the circuit.
	pub n_const: usize,
	/// Number of public input values in the circuit.
	pub n_inout: usize,
	/// Number of private input values in the circuit.
	pub n_witness: usize,
	/// Number of internal values in the circuit.
	///
	/// Internal values are values produced by gates.
	pub n_internal: usize,
	/// Number of scratch values in the circuit.
	///
	/// Those values are not committed, those only exist during witness generation.
	pub n_scratch: usize,
	/// Smallest scratch segment this circuit could run with.
	///
	/// This is the largest number of uncommitted values alive at the same time.
	/// It equals the segment length when slots are shared, and is a lower bound on it otherwise.
	pub scratch_peak_live: usize,
	/// Allocated size for AND constraints (power of 2)
	pub and_allocated: usize,
	/// Allocated size for IMUL constraints (power of 2)
	pub imul_allocated: usize,
	/// Allocated size for BMUL constraints (power of 2)
	pub bmul_allocated: usize,
	/// Allocated size for public section (power of 2)
	pub public_allocated: usize,
	/// Allocated size for private section.
	///
	/// This is the space available for witness and internal values. Note that unlike
	/// `public_allocated` and the total committed length, this is NOT necessarily a
	/// power of two. It's simply the difference between the total committed length
	/// (power of 2) and the public section size (power of 2). For example, if total
	/// is 8192 and public is 128, private is 8064.
	pub private_allocated: usize,
}

impl CircuitStat {
	/// Creates a new `CircuitStat` instance by collecting statistics from the given circuit.
	pub fn collect(circuit: &Circuit) -> Self {
		let cs = circuit.constraint_system();

		// Counts as the circuit compiled them, before the prover pads anything.
		let n_and_constraints = cs.n_and_constraints();
		let n_imul_constraints = cs.n_imul_constraints();
		let n_bmul_constraints = cs.n_bmul_constraints();
		let (distinct_shifted_value_indices, distinct_unshifted_value_indices) =
			traverse_constraint_system(cs);

		// Sizes the prover pads each constraint set to before proving.
		//
		// - Every set is rounded up to a power of two.
		// - Rounding up from zero gives one, so an empty AND set still occupies a single slot.
		// - An empty multiply set instead stays at zero, letting its reduction be skipped whole.
		let pad = |n: usize| n.next_power_of_two();
		let pad_or_skip = |n: usize| if n == 0 { 0 } else { pad(n) };
		let and_allocated = pad(n_and_constraints);
		let imul_allocated = pad_or_skip(n_imul_constraints);
		let bmul_allocated = pad_or_skip(n_bmul_constraints);

		// The public section size is already determined by the layout
		let layout = circuit.value_vec_layout();
		let n_const = layout.n_const;
		let n_inout = layout.n_inout;
		let public_allocated = layout.offset_witness;
		// The committed values are not padded to a power of two in the layout, but the prover
		// commits to a power-of-two-length witness polynomial, so report that padded size.
		let total_allocated = layout.combined_len().next_power_of_two();
		let private_allocated = total_allocated - public_allocated;

		Self {
			n_gates: circuit.n_gates(),
			n_eval_insn: circuit.n_eval_insn(),
			n_and_constraints,
			n_imul_constraints,
			n_bmul_constraints,
			value_vec_len: total_allocated,
			distinct_shifted_value_indices,
			distinct_unshifted_value_indices,
			n_const,
			n_inout,
			n_witness: layout.n_witness,
			n_internal: layout.n_internal,
			n_scratch: layout.n_scratch,
			scratch_peak_live: circuit.scratch_peak_live(),
			and_allocated,
			imul_allocated,
			bmul_allocated,
			public_allocated,
			private_allocated,
		}
	}
}

impl fmt::Display for CircuitStat {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		// Helper to format numbers with commas
		fn fmt_num(n: usize) -> String {
			let s = n.to_string();
			let mut result = String::new();
			for (i, c) in s.chars().rev().enumerate() {
				if i > 0 && i % 3 == 0 {
					result.push(',');
				}
				result.push(c);
			}
			result.chars().rev().collect()
		}

		// Helper to create a simple progress bar
		fn progress_bar(used: usize, total: usize) -> String {
			let percent = (used as f64 / total as f64 * 100.0) as usize;
			let filled = percent / 10;
			let mut bar = String::from("[");
			for i in 0..10 {
				if i < filled {
					bar.push('▓');
				} else {
					bar.push('░');
				}
			}
			bar.push(']');
			bar
		}

		// Helper to get log2 of a power of 2
		const fn log2(n: usize) -> u32 {
			n.trailing_zeros()
		}

		// Use pre-calculated values
		let public_used = self.n_const + self.n_inout;
		let private_used = self.n_witness + self.n_internal;
		let total_used = public_used + private_used;

		// Gates & Instructions
		writeln!(f, "Gates & Instructions")?;
		writeln!(f, "├─ Number of gates: {}", fmt_num(self.n_gates))?;
		writeln!(f, "└─ Number of evaluation instructions: {}", fmt_num(self.n_eval_insn))?;
		writeln!(f)?;

		// Constraints
		writeln!(f, "Constraints")?;
		let and_percent = self.n_and_constraints as f64 / self.and_allocated as f64 * 100.0;
		let and_spare = self.and_allocated - self.n_and_constraints;
		writeln!(
			f,
			"├─ AND constraints: {} used ({:.1}% of 2^{})",
			fmt_num(self.n_and_constraints),
			and_percent,
			log2(self.and_allocated)
		)?;
		writeln!(
			f,
			"│  {} spare: {}",
			progress_bar(self.n_and_constraints, self.and_allocated),
			fmt_num(and_spare)
		)?;

		let imul_percent = if self.imul_allocated > 0 {
			self.n_imul_constraints as f64 / self.imul_allocated as f64 * 100.0
		} else {
			0.0
		};
		let imul_spare = self.imul_allocated - self.n_imul_constraints;
		// A circuit with no IMUL constraints allocates nothing (the IntMul reduction is skipped),
		// so there is no power-of-two allocation to report — unlike AND, which is always padded to
		// at least one.
		let imul_allocation = if self.imul_allocated == 0 {
			"0".to_string()
		} else {
			format!("2^{}", log2(self.imul_allocated))
		};
		writeln!(
			f,
			"├─ IMUL constraints: {} used ({:.1}% of {})",
			fmt_num(self.n_imul_constraints),
			imul_percent,
			imul_allocation
		)?;
		writeln!(
			f,
			"│  {} spare: {}",
			progress_bar(self.n_imul_constraints, self.imul_allocated),
			fmt_num(imul_spare)
		)?;

		let bmul_percent = if self.bmul_allocated > 0 {
			self.n_bmul_constraints as f64 / self.bmul_allocated as f64 * 100.0
		} else {
			0.0
		};
		let bmul_spare = self.bmul_allocated - self.n_bmul_constraints;
		// A circuit with no BMUL constraints allocates nothing (the BinMul reduction is skipped),
		// so there is no power-of-two allocation to report — unlike AND, which is always padded to
		// at least one.
		let bmul_allocation = if self.bmul_allocated == 0 {
			"0".to_string()
		} else {
			format!("2^{}", log2(self.bmul_allocated))
		};
		writeln!(
			f,
			"├─ BMUL constraints: {} used ({:.1}% of {})",
			fmt_num(self.n_bmul_constraints),
			bmul_percent,
			bmul_allocation
		)?;
		writeln!(
			f,
			"│  {} spare: {}",
			progress_bar(self.n_bmul_constraints, self.bmul_allocated),
			fmt_num(bmul_spare)
		)?;
		writeln!(
			f,
			"└─ Distinct value indices: {}",
			fmt_num(self.distinct_shifted_value_indices + self.distinct_unshifted_value_indices)
		)?;
		writeln!(
			f,
			"   ├─ Distinct shifted value indices: {}",
			fmt_num(self.distinct_shifted_value_indices)
		)?;
		writeln!(
			f,
			"   └─ Distinct unshifted value indices: {}",
			fmt_num(self.distinct_unshifted_value_indices)
		)?;
		writeln!(f)?;

		// Value Vector
		writeln!(f, "Value Vector")?;

		// Public Section
		let public_percent = public_used as f64 / self.public_allocated as f64 * 100.0;
		let public_spare = self.public_allocated - public_used;
		writeln!(
			f,
			"├─ Public Section: {} used ({:.1}% of 2^{})",
			fmt_num(public_used),
			public_percent,
			log2(self.public_allocated)
		)?;
		writeln!(
			f,
			"│  {} spare: {}",
			progress_bar(public_used, self.public_allocated),
			fmt_num(public_spare)
		)?;
		writeln!(f, "│  ├─ Constants: {}", fmt_num(self.n_const))?;
		writeln!(f, "│  └─ Inout: {}", fmt_num(self.n_inout))?;

		// Private Section (no allocated size shown since it's not a power of 2)
		let private_percent = private_used as f64 / self.private_allocated as f64 * 100.0;
		let private_spare = self.private_allocated - private_used;
		writeln!(
			f,
			"├─ Private Section: {} used ({:.1}%)",
			fmt_num(private_used),
			private_percent
		)?;
		writeln!(
			f,
			"│  {} spare: {}",
			progress_bar(private_used, self.private_allocated),
			fmt_num(private_spare)
		)?;
		writeln!(f, "│  ├─ Witness: {}", fmt_num(self.n_witness))?;
		writeln!(f, "│  └─ Internal: {}", fmt_num(self.n_internal))?;

		// Total Committed
		let total_percent = total_used as f64 / self.value_vec_len as f64 * 100.0;
		let total_spare = self.value_vec_len - total_used;
		writeln!(
			f,
			"├─ Total Committed: {} used ({:.1}% of 2^{})",
			fmt_num(total_used),
			total_percent,
			log2(self.value_vec_len)
		)?;
		writeln!(
			f,
			"│  {} spare: {}",
			progress_bar(total_used, self.value_vec_len),
			fmt_num(total_spare)
		)?;

		// Report the segment length alongside the floor it could reach if its slots were shared.
		// Recording both pins the lifetime analysis, so a regression in it shows up here.
		writeln!(
			f,
			"└─ Scratch (uncommitted): {} (peak live: {})",
			fmt_num(self.n_scratch),
			fmt_num(self.scratch_peak_live)
		)?;
		writeln!(f)?;

		Ok(())
	}
}

/// Traverses the constraint system and returns the number of distinct value indices that
/// are shifted and unshifted, respectively.
fn traverse_constraint_system(cs: &ConstraintSystem) -> (usize, usize) {
	let mut cx = Cx::default();
	let operands = chain!(
		cs.and_constraints.iter().flat_map(|c| &c.0),
		cs.imul_constraints.iter().flat_map(|c| &c.0),
		cs.bmul_constraints.iter().flat_map(|c| &c.0),
	);
	for operand in operands {
		cx.visit_operand(operand);
	}
	(cx.shifted_terms.len(), cx.unshifted_terms.len())
}

/// The distinct terms seen so far, split by whether they carry a shift.
#[derive(Default)]
struct Cx {
	shifted_terms: FxHashSet<ShiftedValueIndex>,
	unshifted_terms: FxHashSet<ShiftedValueIndex>,
}

impl Cx {
	/// Records every term of one operand.
	fn visit_operand(&mut self, operand: &Operand) {
		for term in operand {
			if term.amount == 0 {
				self.unshifted_terms.insert(*term);
			} else {
				self.shifted_terms.insert(*term);
			}
		}
	}
}
