// Copyright 2026 The Binius Developers

//! Populating a [`ValueTable`] by evaluating one circuit over many independent instances.
//!
//! The table itself is [`binius_core`]'s, alongside the [`ValueVec`](binius_core::ValueVec) it
//! batches. Filling one takes a circuit to evaluate, which is what puts these entry points here.

use std::ops::{Index, IndexMut};

use binius_core::{ValueTable, Word};
use binius_utils::strided_array::StridedArray2DViewMut;

use crate::{BatchPopulateError, Circuit, Wire};

/// Default number of instance columns evaluated by one parallel witness-generation task.
const DEFAULT_PARALLEL_STRIPE_WIDTH: usize = 1024;

// The single-instance API lives in `compiler::circuit`; this block deliberately keeps the batched
// population in its own module rather than growing that one.
#[allow(clippy::multiple_inherent_impl)]
impl Circuit {
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
				let mut filler = BatchWitnessFiller {
					circuit: self,
					values: &mut values,
					instance,
				};
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

/// Assigns witness input wires of one instance into a [`ValueTable`] working buffer.
///
/// Indexing by [`Wire`] targets that wire's row in the instance's column, mirroring the
/// single-instance [`WitnessFiller`](crate::WitnessFiller).
pub struct BatchWitnessFiller<'a, 'v> {
	circuit: &'a Circuit,
	values: &'a mut StridedArray2DViewMut<'v, Word>,
	instance: usize,
}

impl Index<Wire> for BatchWitnessFiller<'_, '_> {
	type Output = Word;

	fn index(&self, wire: Wire) -> &Self::Output {
		&self.values[(self.circuit.witness_row(wire), self.instance)]
	}
}

impl IndexMut<Wire> for BatchWitnessFiller<'_, '_> {
	fn index_mut(&mut self, wire: Wire) -> &mut Self::Output {
		let row = self.circuit.witness_row(wire);
		&mut self.values[(row, self.instance)]
	}
}

#[cfg(test)]
mod tests {
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
