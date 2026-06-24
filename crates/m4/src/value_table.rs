// Copyright 2025 Irreducible Inc.

use std::{error, fmt};

use binius_core::{
	constraint_system::{ValueVec, ValueVecLayout},
	word::Word,
};
use binius_frontend::{Circuit, PopulateError, WitnessFiller};
use binius_utils::rayon::prelude::*;

/// The witness for a batch of `2^k` independent instances of one circuit.
///
/// This batch holds `K = 2^k` instances of the same circuit at once.
/// Every instance shares the one layout, and only the values differ.
///
/// The instances are stored back to back in one flat buffer, in instance-major order.
/// The instance index occupies the high-order positions of the buffer.
/// The word index within an instance occupies the low-order positions.
///
/// ```text
///        instance 0          instance 1             instance K-1
///    [ committed words ][ committed words ] ... [ committed words ]
///    \____ stride ____/
///
///    stride = committed words of one instance, with no scratch
/// ```
///
/// A later batch commitment reads this buffer directly as a multilinear over `k + m_0` variables.
/// - The top `k` variables index the batch.
/// - The bottom `m_0` variables index the values within one instance.
///
/// This is the block-diagonal shape the batch prover exploits.
///
/// Only committed words are kept.
/// The transient scratch space that circuit evaluation needs is dropped during population.
#[derive(Clone, Debug)]
pub struct ValueTable {
	/// The per-instance value layout, shared by every instance in the batch.
	layout: ValueVecLayout,
	/// The base-2 logarithm of the instance count.
	///
	/// The batch always holds a power-of-two number of instances.
	/// So the batch dimension is a clean hypercube whose dimension equals this value.
	log_instances: usize,
	/// The committed words of every instance, concatenated in instance-major order.
	///
	/// The length is the instance count times the committed-word count of one instance.
	data: Vec<Word>,
}

impl ValueTable {
	/// Builds the batch witness, populating all `2^log_instances` instances in parallel.
	///
	/// The instances are independent.
	/// For each, the closure sets its input wires and circuit evaluation fills the rest.
	///
	/// # Arguments
	///
	/// - `circuit`: the single-instance circuit, evaluated once per instance.
	/// - `log_instances`: base-2 logarithm of the instance count.
	/// - `fill`: sets the input wires of instance `i`, for `i` in `0..2^log_instances`. It sets the
	///   inputs evaluation cannot derive: public inputs/outputs and free witnesses. It must assign
	///   every input on each call, since the witness vector is reused between instances.
	///
	/// # Errors
	///
	/// Returns the index of the first instance whose inputs do not satisfy the circuit.
	pub fn populate<F>(
		circuit: &Circuit,
		log_instances: usize,
		fill: F,
	) -> Result<Self, PopulateInstanceError>
	where
		F: Fn(usize, &mut WitnessFiller<'_>) + Sync,
	{
		// Every instance shares the single-instance layout exactly.
		let layout = circuit.constraint_system().value_vec_layout.clone();

		// The committed-word count of one instance, with no scratch.
		// This is the gap between consecutive instances in the flat buffer.
		let stride = layout.committed_total_len;

		// Number of instances in the batch: a power of two by construction.
		let n_instances = 1usize << log_instances;

		// Back-to-back storage for every instance's committed words, zero-initialized.
		//
		//     [ instance 0 | instance 1 | ... | instance K-1 ]
		//      \_ stride _/
		let mut data = vec![Word::ZERO; n_instances * stride];

		// Populate each instance into its own slice of the buffer, concurrently.
		// The chunks are disjoint, so the instances never contend for the same words.
		data.par_chunks_mut(stride).enumerate().try_for_each_init(
			// A witness vector reused across this thread's instances, including transient scratch
			// space.
			|| circuit.new_witness_filler(),
			|filler, (instance, chunk)| {
				// The caller assigns this instance's input wires.
				// Different instances generally receive different inputs.
				fill(instance, filler);

				// Evaluate the circuit gate by gate to derive the remaining committed values.
				// A failed assertion means this instance's inputs do not satisfy the circuit.
				// Record which instance failed.
				circuit
					.populate_wire_witness(filler)
					.map_err(|source| PopulateInstanceError { instance, source })?;

				// Keep only the committed words.
				// The scratch space stays in the filler for the next instance.
				//
				//     filler value vec: [ committed words | scratch ]
				//     chunk           : [ committed words ]
				chunk.copy_from_slice(filler.value_vec().combined_witness());

				Ok(())
			},
		)?;

		Ok(Self {
			layout,
			log_instances,
			data,
		})
	}

	/// The base-2 logarithm of the number of instances.
	pub fn log_instances(&self) -> usize {
		self.log_instances
	}

	/// The number of instances in the batch.
	pub fn n_instances(&self) -> usize {
		1usize << self.log_instances
	}

	/// The per-instance value layout shared by every instance.
	pub fn layout(&self) -> &ValueVecLayout {
		&self.layout
	}

	/// The number of committed words occupied by a single instance.
	pub fn instance_stride(&self) -> usize {
		self.layout.committed_total_len
	}

	/// The committed words of one instance.
	///
	/// The words are in per-instance order: the public segment first, then the remaining values.
	/// This matches one instance's committed witness exactly.
	///
	/// # Panics
	///
	/// Panics if the index is not below the instance count.
	pub fn instance(&self, instance: usize) -> &[Word] {
		// Reject out-of-range instance indices up front with a clear message.
		assert!(instance < self.n_instances(), "instance index out of range");

		// Instance i occupies the half-open word range [i * stride, (i + 1) * stride).
		let stride = self.instance_stride();
		let start = instance * stride;
		&self.data[start..start + stride]
	}

	/// The whole batch as one flat, instance-major word buffer.
	///
	/// This is the buffer a batch commitment reads.
	/// The instance index selects the high-order positions.
	/// The word index within an instance selects the low-order positions.
	pub fn as_words(&self) -> &[Word] {
		&self.data
	}

	/// Reconstructs one instance as a standalone single-instance value vector.
	///
	/// The result is bit-for-bit what populating this instance on its own would produce.
	/// So it can be fed directly to single-instance constraint checking.
	///
	/// # Panics
	///
	/// Panics if the index is not below the instance count.
	pub fn instance_value_vec(&self, instance: usize) -> ValueVec {
		// The committed words of this instance, in per-instance order.
		let words = self.instance(instance);

		// The public segment is the prefix.
		// The remaining committed words follow it in order.
		//
		//     words: [ public segment | remaining values ]
		//             \_ public len _/
		let (public, private) = words.split_at(self.layout.offset_witness);

		// Rebuild the single-instance value vector from the two segments.
		// Their lengths sum to one instance's committed length by construction.
		// So reconstruction never fails here.
		// The constructor only rejects a mismatched total length.
		ValueVec::new_from_data(self.layout.clone(), public.to_vec(), private.to_vec())
			.expect("public and private lengths sum to the committed layout length")
	}
}

/// The failure of a single instance during batch witness population.
///
/// It records which instance failed, so the caller can locate the bad inputs.
/// It wraps the underlying assertion failure from evaluating that instance.
#[derive(Debug)]
pub struct PopulateInstanceError {
	/// The index of the instance that failed, in `0..2^log_instances`.
	pub instance: usize,
	/// The assertion failure raised while evaluating that instance.
	pub source: PopulateError,
}

impl fmt::Display for PopulateInstanceError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// Lead with the instance index, then defer to the underlying failure.
		write!(f, "instance {} failed to populate: {}", self.instance, self.source)
	}
}

impl error::Error for PopulateInstanceError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		// Expose the wrapped assertion failure for error-chain walking.
		Some(&self.source)
	}
}

#[cfg(test)]
mod tests {
	use binius_core::verify::verify_constraints;
	use binius_frontend::{CircuitBuilder, Wire};
	use proptest::prelude::*;

	use super::*;

	// A circuit that asserts `z == x & y` over three public words.
	//
	//     inputs : x, y, z   (all inout)
	//     gate   : and = x & y   (internal wire)
	//     assert : and == z
	//
	// An instance is satisfiable exactly when its assignment sets z = x & y.
	struct AndCircuit {
		circuit: Circuit,
		x: Wire,
		y: Wire,
		z: Wire,
	}

	fn and_circuit() -> AndCircuit {
		// Build the three public wires and the single AND gate.
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let y = builder.add_inout();
		let z = builder.add_inout();
		let and = builder.band(x, y);

		// The only constraint: the gate output must equal the claimed output word.
		builder.assert_eq("z_eq_x_and_y", and, z);

		AndCircuit {
			circuit: builder.build(),
			x,
			y,
			z,
		}
	}

	// Populate one instance on its own through the ordinary single-instance flow.
	// This is the reference the batch must reproduce.
	fn reference_value_vec(c: &AndCircuit, x: u64, y: u64) -> ValueVec {
		// Assign the inputs of a lone instance: z is chosen to satisfy the circuit.
		let mut filler = c.circuit.new_witness_filler();
		filler[c.x] = Word(x);
		filler[c.y] = Word(y);
		filler[c.z] = Word(x & y);

		// Derive the internal values, then extract the committed witness.
		c.circuit.populate_wire_witness(&mut filler).unwrap();
		filler.into_value_vec()
	}

	#[test]
	fn shape_matches_layout_and_instances_validate() {
		let c = and_circuit();

		// Fixture state: 2^3 = 8 instances with distinct, satisfying inputs.
		let log_instances = 3;
		let table = ValueTable::populate(&c.circuit, log_instances, |i, w| {
			// Instance i gets inputs (i, i + 1) and the matching AND output.
			let x = i as u64;
			let y = i as u64 + 1;
			w[c.x] = Word(x);
			w[c.y] = Word(y);
			w[c.z] = Word(x & y);
		})
		.unwrap();

		// Shape: 8 instances, batch dimension of 3, stride equal to one committed witness.
		let stride = c
			.circuit
			.constraint_system()
			.value_vec_layout
			.committed_total_len;
		assert_eq!(table.log_instances(), log_instances);
		assert_eq!(table.n_instances(), 8);
		assert_eq!(table.instance_stride(), stride);
		assert_eq!(table.as_words().len(), 8 * stride);

		// Every reconstructed instance satisfies the single-instance constraint system.
		for i in 0..table.n_instances() {
			let vv = table.instance_value_vec(i);
			verify_constraints(c.circuit.constraint_system(), &vv)
				.unwrap_or_else(|e| panic!("instance {i} failed verification: {e}"));
		}
	}

	#[test]
	fn flat_buffer_is_instances_concatenated() {
		let c = and_circuit();

		// Fixture state: 4 instances; record each instance slice as we go.
		let table = ValueTable::populate(&c.circuit, 2, |i, w| {
			let x = i as u64;
			let y = i as u64 + 1;
			w[c.x] = Word(x);
			w[c.y] = Word(y);
			w[c.z] = Word(x & y);
		})
		.unwrap();

		// Invariant: the flat buffer is exactly instance 0 ++ instance 1 ++ ... ++ instance K-1.
		//
		//     as_words(): [ instance(0) | instance(1) | instance(2) | instance(3) ]
		let stride = table.instance_stride();
		for i in 0..table.n_instances() {
			let from_flat = &table.as_words()[i * stride..(i + 1) * stride];
			assert_eq!(table.instance(i), from_flat);
		}
	}

	#[test]
	fn single_instance_batch_is_degenerate_but_valid() {
		let c = and_circuit();

		// Fixture state: log_instances = 0 → exactly one instance (K = 1).
		let table = ValueTable::populate(&c.circuit, 0, |_, w| {
			w[c.x] = Word(0xABCD);
			w[c.y] = Word(0x0F0F);
			w[c.z] = Word(0xABCD & 0x0F0F);
		})
		.unwrap();

		// The batch collapses to a single instance whose witness matches the reference flow.
		assert_eq!(table.n_instances(), 1);
		let reference = reference_value_vec(&c, 0xABCD, 0x0F0F);
		assert_eq!(table.instance(0), reference.combined_witness());
	}

	#[test]
	fn unsatisfiable_instance_reports_its_index() {
		let c = and_circuit();

		// Fixture state: 4 instances, all satisfying except instance 2.
		//
		// Mutation: instance 2 claims z = x & y XOR 1, which violates z == x & y.
		//
		//     instance 2: x = 2, y = 3, z = (2 & 3) ^ 1   → assertion fails
		let result = ValueTable::populate(&c.circuit, 2, |i, w| {
			let x = i as u64;
			let y = i as u64 + 1;
			w[c.x] = Word(x);
			w[c.y] = Word(y);
			let correct = x & y;
			w[c.z] = Word(if i == 2 { correct ^ 1 } else { correct });
		});

		// Population fails on instance 2, and the error pins down both the index and the cause.
		//
		//     instance 2: band(2, 3) = 2, but z was set to 3, so the assertion 2 == 3 fails.
		let err = result.expect_err("instance 2 violates the AND constraint");

		// The failing instance is reported exactly.
		assert_eq!(err.instance, 2);

		// Exactly one assertion failed, naming the constraint and the mismatched words.
		assert_eq!(err.source.total_count, 1);
		assert_eq!(
			err.source.messages,
			vec![".z_eq_x_and_y: Word(0x0000000000000002) != Word(0x0000000000000003)".to_string()]
		);
	}

	proptest! {
		// Invariant: every batch instance equals the single-instance witness for the same inputs.
		//
		//     batch instance i  ==  single-instance witness for inputs[i]
		#[test]
		fn batch_instances_match_single_instance_reference(
			inputs in prop::collection::vec((any::<u64>(), any::<u64>()), 4),
		) {
			let c = and_circuit();

			// Build the 4-instance batch, feeding instance i its sampled (x, y) pair.
			let table = ValueTable::populate(&c.circuit, 2, |i, w| {
				let (x, y) = inputs[i];
				w[c.x] = Word(x);
				w[c.y] = Word(y);
				w[c.z] = Word(x & y);
			})
			.unwrap();

			// Each instance must equal the independently-built reference, word for word.
			for (i, &(x, y)) in inputs.iter().enumerate() {
				let reference = reference_value_vec(&c, x, y);
				prop_assert_eq!(table.instance(i), reference.combined_witness());
			}
		}
	}
}
