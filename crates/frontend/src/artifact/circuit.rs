// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The artifact a build hands back: a constraint system plus what it takes to fill a witness.

use binius_core::{
	ValueTable,
	constraint_system::{ConstraintSystem, ValueIndex, ValueVec, ValueVecLayout},
	word::Word,
};
use binius_utils::strided_array::StridedArray2DViewMut;
use cranelift_entity::SecondaryMap;

use crate::{
	artifact::{
		dump::dump_composition,
		witness::{BatchWitnessFiller, PopulateError, WitnessFiller},
	},
	eval_form::{BatchPopulateError, EvalForm},
	ir::{
		GateBody, Wire,
		path::{PathSpec, PathSpecTree},
	},
	pass::BuiltGates,
};

/// Default number of instance columns evaluated by one parallel witness-generation task.
const DEFAULT_PARALLEL_STRIPE_WIDTH: usize = 1024;

/// An artifact that represents a built circuit.
///
/// The difference from [`ConstraintSystem`] is that a circuit retains enough information to
/// perform circuit evaluation to generate internal witness values.
pub struct Circuit {
	path_spec_tree: PathSpecTree,
	gate_records: Vec<(PathSpec, GateBody)>,
	constraint_system: ConstraintSystem,
	value_vec_layout: ValueVecLayout,
	wire_mapping: SecondaryMap<Wire, ValueIndex>,
	inout: Vec<Wire>,
	eval_form: EvalForm,
	scratch_peak_live: usize,
}

impl Circuit {
	/// Creates a new circuit with the given shared data and wire mapping. Only used during building
	/// by the circuit builder.
	pub(crate) fn new(
		built_gates: BuiltGates,
		constraint_system: ConstraintSystem,
		value_vec_layout: ValueVecLayout,
		wire_mapping: SecondaryMap<Wire, ValueIndex>,
		inout: Vec<Wire>,
		eval_form: EvalForm,
		scratch_peak_live: usize,
	) -> Self {
		let BuiltGates {
			path_spec_tree,
			gate_records,
		} = built_gates;
		Self {
			path_spec_tree,
			gate_records,
			constraint_system,
			value_vec_layout,
			wire_mapping,
			inout,
			eval_form,
			scratch_peak_live,
		}
	}

	/// Returns the wires forming the circuit's public interface, in inout segment order.
	///
	/// Inout value `i` of the value vector is the word wire `inout()[i]` holds, so this is the
	/// reverse of [`Self::witness_index`] over that segment. A caller assembling the positional
	/// public-input vector a verifier takes reads it straight off this slice.
	///
	/// The segment is ordered by wire creation, so a wire promoted with
	/// [`CircuitBuilder::mark_inout`](crate::CircuitBuilder::mark_inout) follows the declared inout
	/// wires only if its gate created it later; among promotions the order is the one their gates
	/// ran in, not the order they were promoted in.
	pub fn inout(&self) -> &[Wire] {
		&self.inout
	}

	/// Returns the smallest scratch segment this circuit could run with.
	///
	/// This is the largest number of uncommitted temporaries alive at the same time.
	/// It is what the segment shrinks to once slots are shared.
	/// It is reported whether or not sharing is on, so the unused headroom stays visible.
	pub const fn scratch_peak_live(&self) -> usize {
		self.scratch_peak_live
	}

	/// For the given wire, returns its index in the witness vector.
	#[inline(always)]
	pub fn witness_index(&self, wire: Wire) -> ValueIndex {
		self.wire_mapping[wire]
	}

	/// For the given wire, returns the row it occupies in a transposed value array.
	///
	/// This is the wire's flat position in the value vector, counting the scratch tail, which is
	/// how [`Self::populate_wire_witness_batched`] numbers the rows it fills.
	#[inline(always)]
	pub fn witness_row(&self, wire: Wire) -> usize {
		self.value_vec_layout.word_offset(self.witness_index(wire))
	}

	/// Creates a new witness filler for this circuit.
	pub fn new_witness_filler(&self) -> WitnessFiller<'_> {
		WitnessFiller {
			circuit: self,
			value_vec: ValueVec::new(&self.value_vec_layout),
		}
	}

	/// Populates non-input values (wires) in the witness.
	///
	/// Specifically, this will evaluate the circuit gate-by-gate and save the results in the
	/// witness vector.
	///
	/// This function expects that the input wires are already filled. The input wires are
	///
	/// - [`CircuitBuilder::add_inout`],
	/// - [`CircuitBuilder::add_witness`] that were not created by the gates,
	///
	/// The wires created by [`CircuitBuilder::add_constant`] (and its convenience methods)
	/// are automatically populated by this function as well. So is a wire promoted with
	/// [`CircuitBuilder::mark_inout`]: it is public but gate-derived, so the caller leaves it
	/// unset and reads the computed value back afterwards.
	///
	/// # Errors
	///
	/// Returns [`PopulateError`] when any assertion fails.
	/// Each failure names the circuit path the assertion was declared under.
	/// Evaluation runs to completion first, so every violation is reported at once.
	///
	/// [`CircuitBuilder::add_constant`]: crate::CircuitBuilder::add_constant
	/// [`CircuitBuilder::add_inout`]: crate::CircuitBuilder::add_inout
	/// [`CircuitBuilder::add_witness`]: crate::CircuitBuilder::add_witness
	/// [`CircuitBuilder::mark_inout`]: crate::CircuitBuilder::mark_inout
	pub fn populate_wire_witness(&self, w: &mut WitnessFiller) -> Result<(), PopulateError> {
		// Fill the constant part from the witness.
		for (index, constant) in self.constraint_system.constants.iter().enumerate() {
			w.value_vec[ValueIndex::constant(index as u32)] = *constant;
		}

		// Execute the evaluation form - it modifies the ValueVec in place
		// Pass the PathSpecTree for assertion error symbolication
		self.eval_form
			.evaluate(&mut w.value_vec, Some(&self.path_spec_tree))?;

		Ok(())
	}

	/// Populates non-input values for a batch of instances at once.
	///
	/// This is the structure-of-arrays counterpart to [`Self::populate_wire_witness`]. `values` is
	/// the transposed value array: rows are value-vector indices (in the same order a single
	/// instance's [`ValueVec`] uses) and columns are instances. Its height must be the full
	/// value-vector length (including scratch) and its width is the instance count.
	///
	/// The caller must fill each instance's input rows first — the witness wires and any declared
	/// inout wires, but not a wire promoted with [`CircuitBuilder::mark_inout`], which its gate
	/// derives. This function fills the constant rows (broadcasting each constant across every
	/// instance) and then evaluates the circuit gate-by-gate for all instances.
	///
	/// # Errors
	///
	/// If any instance is not satisfiable, returns an error naming the lowest-indexed failing
	/// instance and its assertion failures.
	///
	/// [`CircuitBuilder::mark_inout`]: crate::CircuitBuilder::mark_inout
	pub fn populate_wire_witness_batched(
		&self,
		values: &mut StridedArray2DViewMut<'_, Word>,
	) -> Result<(), BatchPopulateError> {
		// Broadcast each constant into its row across every instance. The constants are the same
		// for all instances, so this fills the constant rows uniformly.
		let n_instances = values.width();
		for (index, &constant) in self.constraint_system.constants.iter().enumerate() {
			for instance in 0..n_instances {
				values[(index, instance)] = constant;
			}
		}

		// Evaluate the bytecode across all instances, symbolicating assertion failures.
		self.eval_form
			.evaluate_batched(values, Some(&self.path_spec_tree))
	}

	/// Populates non-input values for a batch of instances split into vertical stripes.
	///
	/// This is the parallel counterpart to [`Self::populate_wire_witness_batched`]. Constants are
	/// broadcast once over the full value array, then the bytecode interpreter runs independently
	/// on disjoint instance-column stripes of at most `stripe_width` columns.
	///
	/// # Errors
	///
	/// If any instance is not satisfiable, returns an error naming a failing instance and its
	/// assertion failures. The reported instance is not guaranteed to be the lowest failing
	/// instance across all stripes.
	pub(crate) fn populate_wire_witness_batched_parallel(
		&self,
		mut values: StridedArray2DViewMut<'_, Word>,
		stripe_width: usize,
	) -> Result<(), BatchPopulateError> {
		assert!(stripe_width > 0, "stripe width must be positive");

		// Broadcast each constant into its row across every instance. The constants are the same
		// for all instances, so this fills the constant rows uniformly before the stripes split.
		let n_instances = values.width();
		for (index, &constant) in self.constraint_system.constants.iter().enumerate() {
			for instance in 0..n_instances {
				values[(index, instance)] = constant;
			}
		}

		// Evaluate independent instance stripes in parallel, symbolicating assertion failures.
		self.eval_form
			.evaluate_batched_parallel(values, stripe_width, Some(&self.path_spec_tree))
	}

	/// Returns the constraint system for this circuit.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		&self.constraint_system
	}

	/// Returns the layout of the value vector this circuit fills.
	pub const fn value_vec_layout(&self) -> &ValueVecLayout {
		&self.value_vec_layout
	}

	/// Returns the number of gates in this circuit.
	///
	/// Depending on what type of gates this circuit uses, the number of constraints might be
	/// significantly larger.
	pub const fn n_gates(&self) -> usize {
		self.gate_records.len()
	}

	/// Returns the number of evaluation instructions in this circuit.
	pub const fn n_eval_insn(&self) -> usize {
		self.eval_form.n_eval_insn()
	}

	/// Returns a string with a JSON dump that is useful to profile the circuit.
	pub fn simple_json_dump(&self) -> String {
		dump_composition(&self.path_spec_tree, &self.gate_records)
	}

	/// Builds the batch witness in wire-major order, populating all `2^log_instances` instances.
	///
	/// The instances are independent. For each, `fill` sets the input wires; the batched
	/// interpreter then derives every remaining wire, filling all instances of one wire at a time.
	///
	/// # Arguments
	///
	/// - `log_instances`: base-2 logarithm of the instance count.
	/// - `fill`: sets the input wires of instance `i`, for `i` in `0..2^log_instances`. It must
	///   assign every witness input and every inout wire on each call.
	///
	/// # Errors
	///
	/// Returns an error naming the lowest-indexed instance whose inputs do not satisfy the circuit.
	pub fn populate_batch<F>(
		&self,
		log_instances: usize,
		fill: F,
	) -> Result<ValueTable, BatchPopulateError>
	where
		F: Fn(usize, &mut BatchWitnessFiller<'_, '_>),
	{
		self.populate_batch_with(log_instances, None, fill)
	}

	/// Builds the batch witness in wire-major order, evaluating instance stripes in parallel.
	///
	/// This is the parallel counterpart to [`Self::populate_batch`]. Input filling is still
	/// performed once up front, then circuit evaluation runs over disjoint vertical instance
	/// stripes.
	///
	/// # Errors
	///
	/// Returns an error naming a failing instance whose inputs do not satisfy the circuit. The
	/// reported instance is not guaranteed to be the lowest failing instance across all stripes.
	pub fn populate_batch_parallel<F>(
		&self,
		log_instances: usize,
		fill: F,
	) -> Result<ValueTable, BatchPopulateError>
	where
		F: Fn(usize, &mut BatchWitnessFiller<'_, '_>),
	{
		self.populate_batch_parallel_with_stripe_width(
			log_instances,
			DEFAULT_PARALLEL_STRIPE_WIDTH,
			fill,
		)
	}

	/// Builds the batch witness in parallel using a caller-provided stripe width.
	///
	/// This is exposed for benchmarking stripe widths. Production callers should use
	/// [`Self::populate_batch_parallel`].
	///
	/// # Errors
	///
	/// Returns an error naming a failing instance whose inputs do not satisfy the circuit. The
	/// reported instance is not guaranteed to be the lowest failing instance across all stripes.
	///
	/// # Panics
	///
	/// Panics if `stripe_width == 0`.
	pub fn populate_batch_parallel_with_stripe_width<F>(
		&self,
		log_instances: usize,
		stripe_width: usize,
		fill: F,
	) -> Result<ValueTable, BatchPopulateError>
	where
		F: Fn(usize, &mut BatchWitnessFiller<'_, '_>),
	{
		assert!(stripe_width > 0, "stripe width must be positive");
		self.populate_batch_with(log_instances, Some(stripe_width), fill)
	}

	fn populate_batch_with<F>(
		&self,
		log_instances: usize,
		parallel_stripe_width: Option<usize>,
		fill: F,
	) -> Result<ValueTable, BatchPopulateError>
	where
		F: Fn(usize, &mut BatchWitnessFiller<'_, '_>),
	{
		let layout = self.value_vec_layout().clone();
		let n_instances = 1usize << log_instances;

		// The transient working buffer spans the full value vector — constants, inputs, internal
		// values, and scratch — for every instance, in wire-major order.
		let full_len = layout.combined_len() + layout.n_scratch;
		let mut working = vec![Word::ZERO; full_len << log_instances];

		{
			let mut values =
				StridedArray2DViewMut::without_stride(&mut working, full_len, n_instances)
					.expect("full_len * n_instances == working.len() by construction");

			// The caller assigns each instance's witness input wires into that instance's column.
			for instance in 0..n_instances {
				let mut filler = BatchWitnessFiller::new(self, &mut values, instance);
				fill(instance, &mut filler);
			}

			if let Some(stripe_width) = parallel_stripe_width {
				// Broadcast the constants once, then evaluate disjoint instance stripes in
				// parallel.
				self.populate_wire_witness_batched_parallel(values, stripe_width)?;
			} else {
				// Broadcast the constants and evaluate every instance's remaining wires.
				self.populate_wire_witness_batched(&mut values)?;
			}
		}

		// Keep the hidden segment: rows `[offset_inout, combined_len)`, the inout values followed
		// by the private ones. In the wire-major working buffer these rows are contiguous, so
		// this is a single slice of the words. The constants and scratch are dropped.
		let start = layout.offset_inout() << log_instances;
		let end = layout.combined_len() << log_instances;
		let data = working[start..end].to_vec();

		Ok(ValueTable::from_hidden_words(layout, log_instances, data))
	}
}

#[cfg(test)]
mod tests {
	use std::ops::IndexMut;

	use binius_core::{ValueVec, constraint_system::InoutSegment};
	use proptest::prelude::*;

	use super::*;
	use crate::{AssertionFailure, CircuitBuilder};

	/// The constant the mix circuit XORs its first input against.
	const MIX_K: u64 = 0x0123_4567_89ab_cdef;

	// A circuit deriving four words from two public inputs and a constant, each promoted to a
	// public output. The promotions are what keep the derivations alive under dead-code
	// elimination.
	struct MixCircuit {
		circuit: Circuit,
		a: Wire,
		b: Wire,
	}

	impl MixCircuit {
		// Assigns one instance's inputs; the circuit derives its public outputs.
		fn fill<F: IndexMut<Wire, Output = Word>>(&self, filler: &mut F, a: u64, b: u64) {
			filler[self.a] = Word(a);
			filler[self.b] = Word(b);
		}
	}

	fn mix_circuit() -> MixCircuit {
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		let k = builder.add_constant_64(MIX_K);

		let and = builder.band(a, b);
		let xor = builder.bxor(a, k);
		let (sum, _cout) = builder.iadd(a, b);
		let rot = builder.rotr(b, 7);
		let or = builder.bor(and, rot);

		for wire in [and, xor, sum, or] {
			builder.mark_inout(wire);
		}

		MixCircuit {
			circuit: builder.build(),
			a,
			b,
		}
	}

	// Populate one instance on its own through the ordinary single-instance flow.
	fn reference_value_vec(c: &MixCircuit, a: u64, b: u64) -> ValueVec {
		let mut filler = c.circuit.new_witness_filler();
		c.fill(&mut filler, a, b);
		c.circuit.populate_wire_witness(&mut filler).unwrap();
		filler.into_value_vec()
	}

	#[test]
	fn shape_matches_layout() {
		let c = mix_circuit();
		let log_instances = 3;
		let table = c
			.circuit
			.populate_batch(log_instances, |i, w| {
				c.fill(w, i as u64, i as u64 + 1);
			})
			.unwrap();

		let layout = c.circuit.value_vec_layout();
		assert_eq!(table.log_instances(), log_instances);
		assert_eq!(table.n_instances(), 8);
		let n_hidden_words = c
			.circuit
			.constraint_system()
			.n_hidden_words(InoutSegment::Hidden);
		assert_eq!(table.n_hidden_words(), n_hidden_words);
		assert_eq!(table.as_words().len(), n_hidden_words * 8);
		// The committed rows are the inout values the layout stores, then the private ones.
		assert_eq!(n_hidden_words, layout.n_inout + layout.n_private());
	}

	#[test]
	fn every_instance_satisfies_the_constraint_system() {
		let c = mix_circuit();
		let constants = &c.circuit.constraint_system().constants;

		let table = c
			.circuit
			.populate_batch(2, |i, w| {
				c.fill(w, i as u64 * 0x9e37_79b9, i as u64 ^ 0xdead);
			})
			.unwrap();

		for i in 0..table.n_instances() {
			let vv = table.instance_value_vec(i, constants);
			c.circuit
				.constraint_system()
				.verify(&vv)
				.unwrap_or_else(|e| panic!("instance {i} failed verification: {e}"));
		}
	}

	#[test]
	fn single_instance_batch_matches_reference() {
		let c = mix_circuit();
		let constants = &c.circuit.constraint_system().constants;

		let table = c
			.circuit
			.populate_batch(0, |_, w| {
				c.fill(w, 0xABCD, 0x0F0F);
			})
			.unwrap();

		assert_eq!(table.n_instances(), 1);
		let reference = reference_value_vec(&c, 0xABCD, 0x0F0F);
		// The reconstructed instance equals the reference's committed witness, word for word.
		let reconstructed = table.instance_value_vec(0, constants);
		assert_eq!(reconstructed.combined_witness(), reference.combined_witness());
	}

	proptest! {
		// Invariant: every batch instance equals the single-instance witness for the same inputs.
		#[test]
		fn batch_instances_match_single_instance_reference(
			inputs in prop::collection::vec((any::<u64>(), any::<u64>()), 4),
		) {
			let c = mix_circuit();
			let constants = c.circuit.constraint_system().constants.clone();

			let table = c.circuit.populate_batch(2, |i, w| {
				let (a, b) = inputs[i];
				c.fill(w, a, b);
			})
			.unwrap();

			for (i, &(a, b)) in inputs.iter().enumerate() {
				let reference = reference_value_vec(&c, a, b);
				let reconstructed = table.instance_value_vec(i, &constants);
				prop_assert_eq!(reconstructed.combined_witness(), reference.combined_witness());
			}
		}
	}

	#[test]
	fn parallel_population_matches_serial_for_varied_stripe_widths() {
		let c = mix_circuit();
		let log_instances = 3;
		let fill = |i: usize, w: &mut BatchWitnessFiller<'_, '_>| {
			c.fill(
				w,
				(i as u64).wrapping_mul(0x9e37_79b9),
				(i as u64).rotate_left(17) ^ 0xdead_beef,
			);
		};

		let serial = c.circuit.populate_batch(log_instances, fill).unwrap();

		let default_parallel = c
			.circuit
			.populate_batch_parallel(log_instances, fill)
			.unwrap();
		assert_eq!(default_parallel.as_words(), serial.as_words());

		for stripe_width in [1, 2, 3, 8, 64] {
			let parallel = c
				.circuit
				.populate_batch_parallel_with_stripe_width(log_instances, stripe_width, fill)
				.unwrap();

			assert_eq!(
				parallel.as_words(),
				serial.as_words(),
				"stripe width {stripe_width} changed the populated table"
			);
		}
	}

	#[test]
	fn unsatisfiable_instance_reports_its_index() {
		// A circuit that asserts a == b; instances where they differ fail.
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		builder.assert_eq("a_eq_b", a, b);
		let circuit = builder.build();

		// Instance 2 violates a == b; the others satisfy it.
		let result = circuit.populate_batch(2, |i, w| {
			w[a] = Word(i as u64);
			w[b] = Word(if i == 2 { 99 } else { i as u64 });
		});

		let err = result.expect_err("instance 2 violates a == b");
		assert_eq!(err.instance, 2);
		assert_eq!(err.source.total, 1);
		assert_eq!(
			err.source.failures,
			vec![AssertionFailure {
				path: ".a_eq_b".to_string(),
				detail: "Word(0x0000000000000002) != Word(0x0000000000000063)".to_string(),
			}]
		);
	}

	#[test]
	fn parallel_unsatisfiable_instance_reports_global_index_across_stripes() {
		// A circuit that asserts a == b; instances where they differ fail.
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		builder.assert_eq("a_eq_b", a, b);
		let circuit = builder.build();

		// Instance 5 is in the third two-column stripe. Reporting a local stripe index would
		// incorrectly return 1 instead of the global instance index 5.
		let result = circuit.populate_batch_parallel_with_stripe_width(3, 2, |i, w| {
			w[a] = Word(i as u64);
			w[b] = Word(if i == 5 { 99 } else { i as u64 });
		});

		let err = result.expect_err("instance 5 violates a == b");
		assert_eq!(err.instance, 5);
		assert_eq!(err.source.total, 1);
		assert_eq!(
			err.source.failures,
			vec![AssertionFailure {
				path: ".a_eq_b".to_string(),
				detail: "Word(0x0000000000000005) != Word(0x0000000000000063)".to_string(),
			}]
		);
	}

	#[test]
	fn parallel_failure_diagnostics_report_global_instance_across_stripes() {
		// A circuit that asserts a == b; instances where they differ fail.
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		builder.assert_eq("a_eq_b", a, b);
		let circuit = builder.build();

		// Instances 5 and 7 fail in different two-column stripes. The parallel path may report
		// either stripe depending on scheduling, but it must report a global instance index and
		// diagnostics for that instance rather than aggregating unrelated stripes.
		let fill = |i: usize, w: &mut BatchWitnessFiller<'_, '_>| {
			w[a] = Word(i as u64);
			w[b] = Word(if i == 5 || i == 7 { 99 } else { i as u64 });
		};
		let parallel = circuit
			.populate_batch_parallel_with_stripe_width(3, 2, fill)
			.expect_err("instances fail");

		assert!(parallel.instance == 5 || parallel.instance == 7);
		assert_eq!(parallel.source.total, 1);
		assert_eq!(
			parallel.source.failures,
			vec![AssertionFailure {
				path: ".a_eq_b".to_string(),
				detail: format!("Word(0x{0:016x}) != Word(0x0000000000000063)", parallel.instance),
			}]
		);
	}

	// Inout wires are committed, so they lead the stored rows: row `i` is inout value `i`, and the
	// private values follow. Each instance carries its own inout words.
	#[test]
	fn inout_wires_lead_the_committed_rows() {
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		// A private wire, so the table carries private rows behind the inout ones.
		let w = builder.add_witness();
		let and = builder.band(a, b);
		let mixed = builder.bxor(and, w);
		builder.mark_inout(and);
		builder.mark_inout(mixed);
		let circuit = builder.build();

		let log_instances = 2;
		let table = circuit
			.populate_batch(log_instances, |i, f| {
				f[a] = Word(i as u64);
				f[b] = Word(i as u64 + 0x100);
				f[w] = Word(i as u64 ^ 0xbeef);
			})
			.unwrap();

		// The inout values lead the private ones, all of them committed.
		let layout = circuit.value_vec_layout();
		assert_eq!(layout.n_inout, 4);
		assert!(layout.n_private() > 0, "the fixture must carry private rows too");
		assert_eq!(table.n_hidden_words(), layout.n_inout + layout.n_private());

		// The inout rows hold what the filler assigned, one column per instance.
		let inout_row =
			|row: usize| &table.as_words()[row << log_instances..(row + 1) << log_instances];
		assert_eq!(inout_row(0), [Word(0), Word(1), Word(2), Word(3)]);
		assert_eq!(inout_row(1), [Word(0x100), Word(0x101), Word(0x102), Word(0x103)]);

		// And each instance reconstructs to a witness the constraint system accepts, inout
		// values included.
		let constants = &circuit.constraint_system().constants;
		for i in 0..table.n_instances() {
			let vv = table.instance_value_vec(i, constants);
			assert_eq!(vv[circuit.witness_index(a)], Word(i as u64));
			assert_eq!(vv[circuit.witness_index(b)], Word(i as u64 + 0x100));
			circuit
				.constraint_system()
				.verify(&vv)
				.unwrap_or_else(|e| panic!("instance {i} failed verification: {e}"));
		}
	}
}
