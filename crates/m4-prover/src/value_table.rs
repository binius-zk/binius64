// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{error, fmt};

use binius_core::{
	constraint_system::{ValueVec, ValueVecLayout},
	word::Word,
};
use binius_field::PackedField;
use binius_frontend::{Circuit, PopulateError, WitnessFiller};
use binius_iop_prover::channel::IOPProverChannel;
use binius_m4_verifier::BatchCommitLayout;
use binius_math::FieldBuffer;
use binius_utils::rayon::prelude::*;
use binius_verifier::config::B128;

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

		// The public and committed words of one instance, with no scratch.
		// This is the gap between consecutive instances in the flat buffer.
		let stride = layout.combined_len();

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

	/// The number of public and committed words occupied by a single instance.
	pub const fn instance_stride(&self) -> usize {
		self.layout.combined_len()
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

	/// The committed-multilinear layout for this batch.
	///
	/// The verifier derives the same layout, so both sides agree on the committed size.
	pub fn commit_layout(&self) -> BatchCommitLayout {
		BatchCommitLayout::new(self.instance_stride(), self.log_instances())
	}

	/// Packs the batch witness into the multilinear committed as the trace oracle.
	///
	/// Each instance is zero-padded up to a power-of-two word count.
	/// So the instance index becomes the high-order coordinates of the committed multilinear.
	///
	/// ```text
	///   [ instance 0 ][ instance 1 ] ... [ instance K-1 ]
	///   each block padded to 2^log_instance_words words
	/// ```
	///
	/// Every packed element is built in one parallel pass over the table.
	/// No zero-padded word buffer is materialized in between.
	/// The result equals the single-instance packer applied to each padded instance.
	pub fn pack<P>(&self) -> FieldBuffer<P>
	where
		P: PackedField<Scalar = B128>,
	{
		let layout = self.commit_layout();

		// Element index layout: high bits select the instance, low bits the element within it.
		let total_elems = 1usize << layout.log_witness_elems;
		let log_instance_elems = layout.log_witness_elems - layout.log_instances;
		let elem_mask = (1usize << log_instance_elems) - 1;

		// The unpadded, instance-major word buffer feeds the packing directly.
		let words = self.as_words();
		let stride = self.instance_stride();

		// Build every packed field element in parallel, with no padded intermediate buffer.
		// One element packs two consecutive words of one instance, little-endian.
		let n_packed = 1usize << layout.log_witness_elems.saturating_sub(P::LOG_WIDTH);
		let mut values = Vec::with_capacity(n_packed);
		(0..n_packed)
			.into_par_iter()
			.map(|packed_index| {
				P::from_scalars((0..P::WIDTH).map(|lane| {
					// The batch-wide element index this lane carries.
					let elem = (packed_index << P::LOG_WIDTH) | lane;

					// Lanes past the real elements are the commitment's zero padding.
					if elem >= total_elems {
						return B128::new(0);
					}

					// Split into the instance and the element within that instance.
					let instance = elem >> log_instance_elems;
					let local = elem & elem_mask;

					// The two words this element packs, zeroed past the instance's real words.
					let base = instance * stride;
					let lo = 2 * local;
					let w0 = if lo < stride { words[base + lo].0 } else { 0 };
					let w1 = if lo + 1 < stride {
						words[base + lo + 1].0
					} else {
						0
					};
					B128::new(((w1 as u128) << 64) | (w0 as u128))
				}))
			})
			.collect_into_vec(&mut values);

		FieldBuffer::new(layout.log_witness_elems, values.into_boxed_slice())
	}

	/// Commits the batch witness as the trace oracle on the given channel.
	///
	/// Returns the oracle handle the later opening proof refers to.
	pub fn commit<P, Channel>(&self, channel: &mut Channel) -> Channel::Oracle
	where
		P: PackedField<Scalar = B128>,
		Channel: IOPProverChannel<P>,
	{
		channel.send_oracle(self.pack::<P>().to_ref())
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
	use binius_field::{PackedBinaryGhash1x128b, Random};
	use binius_frontend::{CircuitBuilder, Wire};
	use binius_iop::channel::OracleSpec;
	use binius_iop_prover::naive_channel::NaiveProverChannel;
	use binius_math::multilinear::evaluate::evaluate;
	use binius_transcript::ProverTranscript;
	use binius_verifier::config::StdChallenger;
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	// A width-1 packed field keeps one scalar per element, so scalar iteration is layout-clean.
	type P = PackedBinaryGhash1x128b;

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
			.combined_len();
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

	#[test]
	fn instance_is_the_high_order_block_of_the_commitment() {
		let c = and_circuit();

		// Fixture state: 2^2 = 4 instances with distinct, satisfying inputs.
		let inputs: [(u64, u64); 4] = [(1, 3), (5, 6), (9, 12), (0xFF, 0x0F)];
		let table = ValueTable::populate(&c.circuit, 2, |i, w| {
			let (x, y) = inputs[i];
			w[c.x] = Word(x);
			w[c.y] = Word(y);
			w[c.z] = Word(x & y);
		})
		.unwrap();

		// The committed multilinear, as a flat list of scalars.
		let batched: Vec<B128> = table.pack::<P>().iter_scalars().collect();

		// One instance occupies `2^log_witness_elems` scalars when committed on its own.
		let block = 1usize << BatchCommitLayout::new(table.instance_stride(), 0).log_witness_elems;

		// Invariant: block i equals instance i committed on its own.
		//
		//     batched: [ block 0 | block 1 | block 2 | block 3 ]
		//     block i  ==  pack(single-instance table for inputs[i])
		for (i, &(x, y)) in inputs.iter().enumerate() {
			let single_table = ValueTable::populate(&c.circuit, 0, |_, w| {
				w[c.x] = Word(x);
				w[c.y] = Word(y);
				w[c.z] = Word(x & y);
			})
			.unwrap();

			let single: Vec<B128> = single_table.pack::<P>().iter_scalars().collect();

			assert_eq!(single.len(), block);
			assert_eq!(
				&batched[i * block..(i + 1) * block],
				&single[..],
				"instance {i} is not the high-order block of the committed table"
			);
		}
	}

	#[test]
	fn commit_matches_the_verifier_oracle_spec() {
		let c = and_circuit();

		// Fixture state: 2^2 = 4 instances.
		let log_instances = 2;
		let table = ValueTable::populate(&c.circuit, log_instances, |i, w| {
			let x = i as u64;
			let y = i as u64 + 1;
			w[c.x] = Word(x);
			w[c.y] = Word(y);
			w[c.z] = Word(x & y);
		})
		.unwrap();

		// The verifier sizes the oracle from the constraint system, without zero-knowledge.
		// The prover packs the buffer from the table.
		// `send_oracle` asserts both agree on size: the batched-FRI sizing invariant.
		let layout =
			BatchCommitLayout::for_constraint_system(c.circuit.constraint_system(), log_instances);
		let spec = OracleSpec::new(layout.log_witness_elems);

		let mut transcript = ProverTranscript::<StdChallenger>::default();
		let mut channel = NaiveProverChannel::<B128, _>::new(&mut transcript, vec![spec]);

		let _oracle = table.commit::<P, _>(&mut channel);

		// All declared oracles were committed.
		channel.finish();
	}

	#[test]
	fn pack_matches_single_instance_packer() {
		let c = and_circuit();

		// Fixture state: 2^2 = 4 instances.
		let table = ValueTable::populate(&c.circuit, 2, |i, w| {
			let x = i as u64;
			let y = i as u64 + 1;
			w[c.x] = Word(x);
			w[c.y] = Word(y);
			w[c.z] = Word(x & y);
		})
		.unwrap();

		// Reference: lay each instance into a zero-padded block, then run the base packer.
		let layout = table.commit_layout();
		let block = layout.padded_instance_words();
		let mut padded = vec![Word::ZERO; table.n_instances() * block];
		for instance in 0..table.n_instances() {
			let src = table.instance(instance);
			padded[instance * block..instance * block + src.len()].copy_from_slice(src);
		}
		let reference =
			binius_prover::pack_witness::<P>(layout.log_witness_elems, &padded).unwrap();

		// Invariant: the single-pass packer matches the base packer byte for byte.
		// This pins the word-to-element layout to the base prover's, with no silent drift.
		assert_eq!(table.pack::<P>().as_ref(), reference.as_ref());
	}

	// Packs one instance's words the way the committer packs within an instance:
	// two little-endian words per element, zero-padded past the instance's real words.
	//
	// Independent of the batched packer, so the guardrail cross-checks the layout.
	fn pack_instance_elems(words: &[Word], n_elems: usize) -> Vec<B128> {
		(0..n_elems)
			.map(|e| {
				// Element e packs words 2e and 2e+1.
				// Missing words are the commitment's zero-padding.
				let w0 = words.get(2 * e).map_or(0, |w| w.0);
				let w1 = words.get(2 * e + 1).map_or(0, |w| w.0);
				B128::new(((w1 as u128) << 64) | (w0 as u128))
			})
			.collect()
	}

	proptest! {
		#[test]
		fn committed_mle_is_block_diagonal_in_the_instance(seed in any::<u64>()) {
			// Invariant: the committed multilinear is block-diagonal in the instance index.
			// - The top `k` variables index the instance.
			// - The bottom `m_0` variables index the words within one instance.
			//
			//     committed(r_lo || r_hi) == sum_i eq(r_hi, i) * instance_i(r_lo)
			//
			// Evaluating both sides at a uniform random point pins the layout.
			// A wire-major order, or any instance/word split drift, breaks the identity.

			let c = and_circuit();

			// K = 4 instances with distinct, satisfying inputs.
			let table = ValueTable::populate(&c.circuit, 2, |i, w| {
				let x = i as u64;
				let y = i as u64 + 1;
				w[c.x] = Word(x);
				w[c.y] = Word(y);
				w[c.z] = Word(x & y);
			})
			.unwrap();

			let layout = table.commit_layout();
			let committed = table.pack::<P>();

			// A uniform random point over the committed element hypercube.
			let mut rng = StdRng::seed_from_u64(seed);
			let r: Vec<B128> = (0..layout.log_witness_elems)
				.map(|_| B128::random(&mut rng))
				.collect();

			// Left side: evaluate the committed multilinear directly at r.
			let lhs = evaluate(&committed, &r);

			// Split r into the low (within-instance) and high (instance) coordinates.
			//
			//     r = [ r_lo ............ | r_hi ......... ]
			//          \_ m_0 elem bits _/  \_k inst bits_/
			let log_lo = layout.log_witness_elems - layout.log_instances;
			let (r_lo, r_hi) = r.split_at(log_lo);

			// Evaluate each instance's own packed multilinear at the low coordinates.
			// Then combine across instances at the high coordinates.
			// That combine is itself a multilinear evaluation: sum_i eq(r_hi, i) * s_i.
			let per_instance: Vec<B128> = (0..table.n_instances())
				.map(|i| {
					let elems = pack_instance_elems(table.instance(i), 1 << log_lo);
					evaluate(&FieldBuffer::<P>::from_values(&elems), r_lo)
				})
				.collect();
			let rhs = evaluate(&FieldBuffer::<P>::from_values(&per_instance), r_hi);

			// Agreement confirms the instance index occupies the high coordinates.
			prop_assert_eq!(lhs, rhs);
		}
	}
}
