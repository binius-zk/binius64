// Copyright 2026 The Binius Developers

use std::ops::{Index, IndexMut};

use binius_core::{
	constraint_system::{ValueVec, ValueVecLayout},
	word::Word,
};
use binius_frontend::{BatchPopulateError, Circuit, Wire};
use binius_utils::strided_array::StridedArray2DViewMut;

/// The witness for a batch of `2^k` independent instances of one circuit, in wire-major order.
///
/// This is the transpose of [`ValueTable`](crate::ValueTable). Where `ValueTable` stores each
/// instance's words back to back (instance-major), `ValueTable2` groups each wire's values across
/// all instances (wire-major):
///
/// ```text
///                 instance 0   instance 1   ...   instance K-1
///   wire 0       [   w        |   w        | ... |   w        ]   <- one row
///   wire 1       [   w        |   w        | ... |   w        ]
///        ...
/// ```
///
/// Each row is one wire and each column is one instance, so a batched interpreter can advance every
/// instance of a wire in a single pass. Transposing the value table this way is expected to pay off
/// throughout the rest of the M4 protocol.
///
/// # Differences from `ValueTable`
///
/// This table is specialized for the M4 accelerator setting, where a circuit plugs into a larger
/// system rather than exposing public inputs and outputs:
///
/// 1. **No inout wires.** The circuit's [`ValueVecLayout`] must have `n_inout == 0`. Output wires
///    are kept alive with
///    [`CircuitBuilder::force_commit`](binius_frontend::CircuitBuilder::force_commit) so that
///    dead-code elimination does not drop the circuit. Inout wires are inappropriate here, so
///    [`Self::populate`] rejects them.
/// 2. **Constants are not stored.** The constants live once on the constraint system as a
///    `Vec<Word>`, not replicated per instance. Only the hidden segment — the witness and internal
///    words — is stored here. Witness generation reads the constants from that separate bank.
///
/// So the stored data holds exactly the hidden segment of every instance: `n_hidden_words` rows by
/// `2^log_instances` columns.
#[derive(Clone, Debug)]
pub struct ValueTable2 {
	/// The per-instance value layout, shared by every instance in the batch.
	layout: ValueVecLayout,
	/// The base-2 logarithm of the instance count.
	log_instances: usize,
	/// The hidden words of every wire, in wire-major order.
	///
	/// Row `r` (for `r` in `0..n_hidden_words`) holds the `2^log_instances` values of hidden wire
	/// `r` — value index `offset_witness + r` — one per instance. The rows are laid out
	/// contiguously, so the length is `n_hidden_words << log_instances`.
	data: Vec<Word>,
}

impl ValueTable2 {
	/// Builds the batch witness in wire-major order, populating all `2^log_instances` instances.
	///
	/// The instances are independent. For each, `fill` sets the witness input wires; the batched
	/// interpreter then derives every remaining wire, filling all instances of one wire at a time.
	///
	/// # Arguments
	///
	/// - `circuit`: the single-instance circuit. Its layout must have no inout wires.
	/// - `log_instances`: base-2 logarithm of the instance count.
	/// - `fill`: sets the witness input wires of instance `i`, for `i` in `0..2^log_instances`. It
	///   must assign every witness input on each call.
	///
	/// # Panics
	///
	/// Panics if the circuit's layout has any inout wires (`n_inout != 0`).
	///
	/// # Errors
	///
	/// Returns an error naming the lowest-indexed instance whose inputs do not satisfy the circuit.
	pub fn populate<F>(
		circuit: &Circuit,
		log_instances: usize,
		fill: F,
	) -> Result<Self, BatchPopulateError>
	where
		F: Fn(usize, &mut Batch2WitnessFiller<'_, '_>),
	{
		let layout = circuit.constraint_system().value_vec_layout.clone();
		assert_eq!(
			layout.n_inout, 0,
			"ValueTable2 requires a constraint system with no inout wires; \
			 use force_commit on output wires instead"
		);

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
				let mut filler = Batch2WitnessFiller {
					circuit,
					values: &mut values,
					instance,
				};
				fill(instance, &mut filler);
			}

			// Broadcast the constants and evaluate every instance's remaining wires.
			circuit.populate_wire_witness_batched(&mut values)?;
		}

		// Keep only the hidden segment: rows `[offset_witness, combined_len)`. In the wire-major
		// working buffer these rows are contiguous, so this is a single slice of the words. The
		// constants and scratch are dropped.
		let start = layout.offset_witness << log_instances;
		let end = layout.combined_len() << log_instances;
		let data = working[start..end].to_vec();

		Ok(Self {
			layout,
			log_instances,
			data,
		})
	}

	/// The base-2 logarithm of the number of instances.
	pub const fn log_instances(&self) -> usize {
		self.log_instances
	}

	/// The number of instances in the batch.
	pub const fn n_instances(&self) -> usize {
		1usize << self.log_instances
	}

	/// The per-instance value layout shared by every instance.
	pub const fn layout(&self) -> &ValueVecLayout {
		&self.layout
	}

	/// The number of hidden words per instance: the witness and internal words stored in each row.
	pub const fn n_hidden_words(&self) -> usize {
		self.layout.n_hidden_words
	}

	/// The whole batch as one flat, wire-major word buffer.
	///
	/// Row `r` (hidden wire `r`) occupies `data[r << log_instances .. (r + 1) << log_instances]`,
	/// holding that wire's value in every instance.
	pub fn as_words(&self) -> &[Word] {
		&self.data
	}

	/// Reconstructs one instance as a standalone single-instance value vector.
	///
	/// Because the constants are not stored in the table, the caller supplies them (they live on
	/// the constraint system). The result is bit-for-bit what populating this instance on its own
	/// would produce, so it can be fed directly to single-instance constraint checking.
	///
	/// # Panics
	///
	/// Panics if the index is not below the instance count, or if `constants` does not match the
	/// layout's constant count.
	pub fn instance_value_vec(&self, instance: usize, constants: &[Word]) -> ValueVec {
		assert!(instance < self.n_instances(), "instance index out of range");
		assert_eq!(
			constants.len(),
			self.layout.n_const,
			"constants length must match the layout's constant count"
		);

		// The public segment holds the constants at the front, then zero padding up to its
		// power-of-two length. There are no inout wires.
		let mut public = vec![Word::ZERO; self.layout.offset_witness];
		public[..constants.len()].copy_from_slice(constants);

		// Gather this instance's column of hidden words across every row.
		let private = (0..self.n_hidden_words())
			.map(|row| self.data[(row << self.log_instances) + instance])
			.collect::<Vec<_>>();

		ValueVec::new_from_data(self.layout.clone(), public, private)
			.expect("public and private lengths match the layout by construction")
	}
}

/// Assigns witness input wires of one instance into a [`ValueTable2`] working buffer.
///
/// Indexing by [`Wire`] targets that wire's row in the instance's column, mirroring the
/// single-instance [`WitnessFiller`](binius_frontend::WitnessFiller).
pub struct Batch2WitnessFiller<'a, 'v> {
	circuit: &'a Circuit,
	values: &'a mut StridedArray2DViewMut<'v, Word>,
	instance: usize,
}

impl Index<Wire> for Batch2WitnessFiller<'_, '_> {
	type Output = Word;

	fn index(&self, wire: Wire) -> &Self::Output {
		&self.values[(self.circuit.witness_index(wire).0 as usize, self.instance)]
	}
}

impl IndexMut<Wire> for Batch2WitnessFiller<'_, '_> {
	fn index_mut(&mut self, wire: Wire) -> &mut Self::Output {
		let row = self.circuit.witness_index(wire).0 as usize;
		&mut self.values[(row, self.instance)]
	}
}

#[cfg(test)]
mod tests {
	use binius_core::verify::verify_constraints;
	use binius_frontend::{CircuitBuilder, Wire};
	use proptest::prelude::*;

	use super::*;

	// A circuit that computes several derived words from two witness inputs and a constant, with no
	// inout wires. Every observable output is force-committed so dead-code elimination keeps it.
	struct MixCircuit {
		circuit: Circuit,
		a: Wire,
		b: Wire,
	}

	fn mix_circuit() -> MixCircuit {
		let builder = CircuitBuilder::new();
		let a = builder.add_witness();
		let b = builder.add_witness();
		let k = builder.add_constant_64(0x0123_4567_89ab_cdef);

		let and = builder.band(a, b);
		let xor = builder.bxor(a, k);
		let (sum, _cout) = builder.iadd(a, b);
		let rot = builder.rotr(b, 7);
		let or = builder.bor(and, rot);

		builder.force_commit(and);
		builder.force_commit(xor);
		builder.force_commit(sum);
		builder.force_commit(or);

		MixCircuit {
			circuit: builder.build(),
			a,
			b,
		}
	}

	// Populate one instance on its own through the ordinary single-instance flow.
	fn reference_value_vec(c: &MixCircuit, a: u64, b: u64) -> ValueVec {
		let mut filler = c.circuit.new_witness_filler();
		filler[c.a] = Word(a);
		filler[c.b] = Word(b);
		c.circuit.populate_wire_witness(&mut filler).unwrap();
		filler.into_value_vec()
	}

	#[test]
	fn shape_matches_layout() {
		let c = mix_circuit();
		let log_instances = 3;
		let table = ValueTable2::populate(&c.circuit, log_instances, |i, w| {
			w[c.a] = Word(i as u64);
			w[c.b] = Word(i as u64 + 1);
		})
		.unwrap();

		let layout = &c.circuit.constraint_system().value_vec_layout;
		assert_eq!(table.log_instances(), log_instances);
		assert_eq!(table.n_instances(), 8);
		assert_eq!(table.n_hidden_words(), layout.n_hidden_words);
		assert_eq!(table.as_words().len(), layout.n_hidden_words * 8);
	}

	#[test]
	fn every_instance_satisfies_the_constraint_system() {
		let c = mix_circuit();
		let constants = &c.circuit.constraint_system().constants;

		let table = ValueTable2::populate(&c.circuit, 2, |i, w| {
			w[c.a] = Word(i as u64 * 0x9e37_79b9);
			w[c.b] = Word(i as u64 ^ 0xdead);
		})
		.unwrap();

		for i in 0..table.n_instances() {
			let vv = table.instance_value_vec(i, constants);
			verify_constraints(c.circuit.constraint_system(), &vv)
				.unwrap_or_else(|e| panic!("instance {i} failed verification: {e}"));
		}
	}

	#[test]
	fn single_instance_batch_matches_reference() {
		let c = mix_circuit();
		let constants = &c.circuit.constraint_system().constants;

		let table = ValueTable2::populate(&c.circuit, 0, |_, w| {
			w[c.a] = Word(0xABCD);
			w[c.b] = Word(0x0F0F);
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

			let table = ValueTable2::populate(&c.circuit, 2, |i, w| {
				let (a, b) = inputs[i];
				w[c.a] = Word(a);
				w[c.b] = Word(b);
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
	fn unsatisfiable_instance_reports_its_index() {
		// A circuit that asserts a == b; instances where they differ fail.
		let builder = CircuitBuilder::new();
		let a = builder.add_witness();
		let b = builder.add_witness();
		builder.assert_eq("a_eq_b", a, b);
		let circuit = builder.build();

		// Instance 2 violates a == b; the others satisfy it.
		let result = ValueTable2::populate(&circuit, 2, |i, w| {
			w[a] = Word(i as u64);
			w[b] = Word(if i == 2 { 99 } else { i as u64 });
		});

		let err = result.expect_err("instance 2 violates a == b");
		assert_eq!(err.instance, 2);
		assert_eq!(err.source.total_count, 1);
		assert_eq!(
			err.source.messages,
			vec![".a_eq_b: Word(0x0000000000000002) != Word(0x0000000000000063)".to_string()]
		);
	}

	#[test]
	#[should_panic(expected = "no inout wires")]
	fn rejects_circuits_with_inout_wires() {
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_witness();
		let and = builder.band(a, b);
		builder.force_commit(and);
		let circuit = builder.build();

		let _ = ValueTable2::populate(&circuit, 1, |_, w| {
			w[a] = Word(1);
			w[b] = Word(1);
		});
	}
}
