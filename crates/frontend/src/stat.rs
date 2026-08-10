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
	/// Number of ZERO constraints in the circuit.
	///
	/// Affects performance of the shift reduction only: the Zero reduction itself carries no
	/// sumcheck.
	pub n_zero_constraints: usize,
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
	/// Length of the committed trace in words, a power of two.
	///
	/// Affects performance of committing.
	pub committed_allocated: usize,
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
	/// Allocated size for ZERO constraints (power of 2, or zero when there are none)
	pub zero_allocated: usize,
	/// Allocated size for AND constraints (power of 2)
	pub and_allocated: usize,
	/// Allocated size for IMUL constraints (power of 2)
	pub imul_allocated: usize,
	/// Allocated size for BMUL constraints (power of 2)
	pub bmul_allocated: usize,
}

impl CircuitStat {
	/// Creates a new `CircuitStat` instance by collecting statistics from the given circuit.
	pub fn collect(circuit: &Circuit) -> Self {
		let cs = circuit.constraint_system();

		// Counts as the circuit compiled them, before the prover pads anything.
		let n_zero_constraints = cs.n_zero_constraints();
		let n_and_constraints = cs.n_and_constraints();
		let n_imul_constraints = cs.n_imul_constraints();
		let n_bmul_constraints = cs.n_bmul_constraints();
		let (distinct_shifted_value_indices, distinct_unshifted_value_indices) =
			traverse_constraint_system(cs);

		// Sizes the prover pads each constraint set to before proving.
		//
		// - Every set is rounded up to a power of two.
		// - Rounding up from zero gives one, so an empty AND set still occupies a single slot.
		// - An empty ZERO or multiply set instead stays at zero, letting its reduction be skipped
		//   whole.
		let pad = |n: usize| n.next_power_of_two();
		let pad_or_skip = |n: usize| if n == 0 { 0 } else { pad(n) };
		let zero_allocated = pad_or_skip(n_zero_constraints);
		let and_allocated = pad(n_and_constraints);
		let imul_allocated = pad_or_skip(n_imul_constraints);
		let bmul_allocated = pad_or_skip(n_bmul_constraints);

		// The value counts come from the layout. Neither segment is padded, so each holds exactly
		// the values the circuit declared and there is no spare capacity to report against.
		let layout = circuit.value_vec_layout();
		let n_const = layout.n_const;
		let n_inout = layout.n_inout;
		// The prover commits only the hidden segment, zero-extended to the word-index space the
		// shift reduction runs over. That padding is the one the circuit author can still fill.
		let committed_allocated = 1 << cs.log_segment_words();

		Self {
			n_gates: circuit.n_gates(),
			n_eval_insn: circuit.n_eval_insn(),
			n_zero_constraints,
			n_and_constraints,
			n_imul_constraints,
			n_bmul_constraints,
			committed_allocated,
			distinct_shifted_value_indices,
			distinct_unshifted_value_indices,
			n_const,
			n_inout,
			n_witness: layout.n_witness,
			n_internal: layout.n_internal,
			n_scratch: layout.n_scratch,
			scratch_peak_live: circuit.scratch_peak_live(),
			zero_allocated,
			and_allocated,
			imul_allocated,
			bmul_allocated,
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

		// Gates & Instructions
		writeln!(f, "Gates & Instructions")?;
		writeln!(f, "├─ Number of gates: {}", fmt_num(self.n_gates))?;
		writeln!(f, "└─ Number of evaluation instructions: {}", fmt_num(self.n_eval_insn))?;
		writeln!(f)?;

		// Reports one constraint set: how many rows it uses out of the power of two the prover pads
		// it to. A set whose reduction is skipped when empty allocates nothing, so it reports an
		// allocation of `0` rather than a power of two — unlike AND, which is always padded to at
		// least one row.
		fn constraint_line(
			f: &mut fmt::Formatter,
			name: &str,
			used: usize,
			allocated: usize,
		) -> fmt::Result {
			let percent = if allocated > 0 {
				used as f64 / allocated as f64 * 100.0
			} else {
				0.0
			};
			let allocation = if allocated == 0 {
				"0".to_string()
			} else {
				format!("2^{}", log2(allocated))
			};
			writeln!(
				f,
				"├─ {name} constraints: {} used ({percent:.1}% of {allocation})",
				fmt_num(used)
			)?;
			writeln!(f, "│  {} spare: {}", progress_bar(used, allocated), fmt_num(allocated - used))
		}

		// Constraints
		writeln!(f, "Constraints")?;
		constraint_line(f, "ZERO", self.n_zero_constraints, self.zero_allocated)?;
		constraint_line(f, "AND", self.n_and_constraints, self.and_allocated)?;
		constraint_line(f, "IMUL", self.n_imul_constraints, self.imul_allocated)?;
		constraint_line(f, "BMUL", self.n_bmul_constraints, self.bmul_allocated)?;
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

		// Public Section. The segment holds exactly the values the circuit declared, and the
		// verifier knows them, so there is neither padding nor a commitment to measure it against.
		writeln!(f, "├─ Public Section: {} (not committed)", fmt_num(public_used))?;
		writeln!(f, "│  ├─ Constants: {}", fmt_num(self.n_const))?;
		writeln!(f, "│  └─ Inout: {}", fmt_num(self.n_inout))?;

		// Private Section, likewise exactly the declared values.
		writeln!(f, "├─ Private Section: {}", fmt_num(private_used))?;
		writeln!(f, "│  ├─ Witness: {}", fmt_num(self.n_witness))?;
		writeln!(f, "│  └─ Internal: {}", fmt_num(self.n_internal))?;

		// Committed Trace: the hidden segment zero-extended to the shift reduction's word-index
		// space. This is the only padded length left, and the one more values can grow into for
		// free.
		let committed_percent = private_used as f64 / self.committed_allocated as f64 * 100.0;
		let committed_spare = self.committed_allocated - private_used;
		writeln!(
			f,
			"├─ Committed Trace: {} used ({:.1}% of 2^{})",
			fmt_num(private_used),
			committed_percent,
			log2(self.committed_allocated)
		)?;
		writeln!(
			f,
			"│  {} spare: {}",
			progress_bar(private_used, self.committed_allocated),
			fmt_num(committed_spare)
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
		cs.zero_constraints.iter().flat_map(|c| &c.0),
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
			if term.shift.is_identity() {
				self.unshifted_terms.insert(*term);
			} else {
				self.shifted_terms.insert(*term);
			}
		}
	}
}
