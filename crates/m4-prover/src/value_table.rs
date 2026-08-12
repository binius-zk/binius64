// Copyright 2026 The Binius Developers

//! Committing a [`ValueTable`] as the trace oracle.
//!
//! The table is [`binius_core`]'s and populating one is the frontend's, but neither knows about
//! commitments. These are the two operations that do, so they live with the prover.

use binius_compute::Allocator;
use binius_core::ValueTable;
use binius_field::PackedField;
use binius_m4_verifier::BatchCommitLayout;
use binius_math::FieldVec;
use binius_verifier::config::B128;

/// The committed-multilinear layout for a batch.
///
/// The verifier derives the same layout from the constraint system.
/// So both sides agree on the committed size.
pub fn commit_layout(table: &ValueTable) -> BatchCommitLayout {
	BatchCommitLayout::new(table.n_hidden_words(), table.log_instances())
}

/// Packs the wire-major hidden buffer into the multilinear committed as the trace oracle.
///
/// Two little-endian words pack into one field element.
/// The element sequence is zero-padded up to the committed element count.
/// The instance index occupies the low coordinates, the hidden-word index the high coordinates.
/// Only the hidden segment is committed; the shared constants are not part of the oracle.
/// The packed buffer is drawn from `alloc`.
pub fn pack_table<P, A>(table: &ValueTable, alloc: &A) -> FieldVec<P, A>
where
	P: PackedField<Scalar = B128>,
	A: Allocator,
{
	// The stored buffer is already the committed word sequence, wire-major.
	// The base packer zero-pads it up to `2^log_witness_elems` elements.
	let layout = commit_layout(table);
	binius_prover::pack_witness::<P, A>(alloc, layout.log_witness_elems, table.as_words())
		.expect("the hidden buffer fits in 2^log_witness_elems field elements by construction")
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_core::{Word, constraint_system::InoutSegment};
	use binius_field::PackedBinaryGhash1x128b;
	use binius_frontend::CircuitBuilder;
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{
		N_INPUT_WORDS, crc64_circuit, crc64_iso_reference, populate_crc64_witness,
	};

	// A large batch of independent random instances populates to the documented shape.
	// Each instance reconstructs to a witness that satisfies the circuit.
	// Its output word is the reference CRC of that instance's own inputs.
	#[test]
	fn populate_batch_of_random_instances() {
		let c = crc64_circuit();

		// A batch of 2^10 instances, each with an independent random message.
		let log_instances = 10;
		let n_instances = 1usize << log_instances;

		// Sample every instance's inputs up front so the fill closure is a pure lookup and the
		// reference check below sees the same words.
		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();

		let table = populate_crc64_witness(&c, &inputs);
		let constants = &c.circuit.constraint_system().constants;

		// Shape: 2^10 instances, one hidden-word row per committed word.
		let n_hidden_words = c
			.circuit
			.constraint_system()
			.n_hidden_words(InoutSegment::Hidden);
		assert_eq!(table.log_instances(), log_instances);
		assert_eq!(table.n_instances(), n_instances);
		assert_eq!(table.n_hidden_words(), n_hidden_words);
		assert_eq!(table.as_words().len(), n_hidden_words * n_instances);

		// Spot-check a few instances: each reconstructs to a valid single-instance witness whose
		// output word is the reference CRC of its inputs.
		let output_index = c.circuit.witness_index(c.output);
		for i in [0, 1, 42, n_instances - 1] {
			let vv = table.instance_value_vec(i, constants);
			c.circuit
				.constraint_system()
				.verify(&vv)
				.unwrap_or_else(|e| panic!("instance {i} failed verification: {e}"));
			assert_eq!(vv[output_index], Word(crc64_iso_reference(&inputs[i])));
		}
	}

	#[test]
	fn pack_lays_out_hidden_words_wire_major() {
		type P = PackedBinaryGhash1x128b;

		// A circuit deriving a few words from two inout inputs, so the table carries several rows.
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		let and = builder.band(a, b);
		let (sum, _cout) = builder.iadd(a, b);
		builder.mark_inout(and);
		builder.mark_inout(sum);
		let circuit = builder.build();

		// Fixture state: 4 instances with distinct witness inputs.
		let table = circuit
			.populate_batch(2, |i, w| {
				w[a] = Word((i as u64).wrapping_mul(0x9e37_79b9));
				w[b] = Word(i as u64 ^ 0xdead);
			})
			.unwrap();

		let layout = commit_layout(&table);
		let packed: Vec<B128> = pack_table::<P, _>(&table, &GlobalAllocator)
			.iter_scalars()
			.collect();

		// The committed scalars are the wire-major buffer taken two little-endian words per
		// element. Indices past the stored words are the commitment's zero padding.
		let words = table.as_words();
		let total_elems = 1usize << layout.log_witness_elems;
		let expected: Vec<B128> = (0..total_elems)
			.map(|e| {
				let w0 = words.get(2 * e).map_or(0, |w| w.0);
				let w1 = words.get(2 * e + 1).map_or(0, |w| w.0);
				B128::new(((w1 as u128) << 64) | (w0 as u128))
			})
			.collect();

		assert_eq!(packed.len(), total_elems);
		assert_eq!(packed, expected);
	}
}
