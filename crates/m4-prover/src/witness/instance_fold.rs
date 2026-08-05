// Copyright 2026 The Binius Developers

//! The committed witness with its instance axis collapsed at a challenge point.

use binius_compute::{Allocator, VecLike};
use binius_core::word::Word;
use binius_field::BinaryField;
use binius_prover::fold_word::WordFolder;
use binius_utils::rayon::prelude::*;

use crate::ValueTable;

/// A committed witness word after folding its bits into the field.
///
/// Each 64-bit word contributes one field element per bit position, so a folded word is the oblong
/// representation of that word: its bit axis expanded to full field elements.
pub type FoldedWord<F> = [F; Word::BITS];

/// Folds the committed witness of a batch value table along the instance axis.
///
/// The committed witness has three axes:
/// - the bits within each 64-bit word.
/// - the committed words within one instance.
/// - the instances themselves.
///
/// This collapses the instance axis by the equality-indicator weights of `r_rho`.
/// What remains is a multilinear over the other two axes.
///
/// For committed word `w` and bit `b`, the output element is
///
/// ```text
/// out[w][b] = sum_rho eq(r_rho, rho) * bit_b(word[rho][w])
/// ```
///
/// so each set bit contributes its instance's equality weight to a full field element.
///
/// The bit axis occupies the low coordinates and the word axis the high coordinates.
/// The result is a multilinear over `Word::LOG_BITS + log2(n_committed)` variables:
///
/// ```text
/// index = w * Word::BITS + b     (b occupies the low Word::LOG_BITS coordinates)
/// ```
///
/// The table stores exactly the committed (hidden) words, so nothing here is excluded.
/// The constants and public words live once on the constraint system, folded separately.
///
/// The wire-major layout makes this cheap: one wire's values across every instance are stored
/// contiguously, so each word position is a plain sub-slice rather than a strided gather.
///
/// Every word position folds against the same instance point `r_rho`.
/// So the lookup tables and per-chunk weights are built once and shared across all word positions.
/// The word positions are independent, so the fold runs in parallel, one output word per task.
///
/// # Panics
///
/// Panics if `r_rho.len()` does not equal the batch dimension.
pub fn fold_instances<F: BinaryField, A: Allocator>(
	table: &ValueTable,
	r_rho: &[F],
	alloc: &A,
) -> A::Vec<FoldedWord<F>> {
	assert_eq!(r_rho.len(), table.log_instances(), "r_rho must match the batch dimension");

	// Build the instance-fold tables once; the lookups and weights depend only on r_rho.
	let folder = WordFolder::<F>::new(r_rho);

	// Each output element holds one committed word position:
	//     out[w][b] = sum_rho eq(r_rho, rho) * bit_b(word[rho][w]).
	// The word positions are independent, so fold them in parallel, one output element per task.
	//
	// The table stores exactly `n_hidden_words << log_instances` words, so chunking it by the
	// instance count yields exactly one chunk per output element. `zip` truncates to the shorter
	// side, so this equality is what makes the loop below cover every element; assert it rather
	// than rely on a `ValueTable` invariant stated in another module.
	let n_words = table.n_hidden_words();
	debug_assert_eq!(table.as_words().len(), n_words << table.log_instances());
	let mut folded = alloc.alloc::<FoldedWord<F>>(n_words);
	table
		.as_words()
		.par_chunks(1 << table.log_instances())
		.zip(folded.spare_capacity_mut())
		.for_each(|(instance_words, out)| {
			out.write(folder.fold(instance_words));
		});
	// SAFETY: the chunks and the output elements are in one-to-one correspondence by the length
	// equality asserted above, so the loop writes each of the `n_words` elements exactly once.
	unsafe { folded.set_len(n_words) };
	folded
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::PackedBinaryGhash1x128b;
	use binius_math::{
		multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
		test_utils::random_scalars,
	};
	use binius_prover::fold_word::fold_words;
	use binius_utils::checked_arithmetics::checked_log_2;
	use binius_verifier::config::B128;
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{
		N_INPUT_WORDS, crc64_circuit, evaluate_folded_witness, populate_crc64_witness,
	};

	// Folding the batch over the instance axis and then evaluating over the (bit, word) axes agrees
	// with folding each word's bits first and then evaluating over the (word, instance) axes.
	//
	// Both routes compute the same triple sum, just associated differently:
	//
	//     sum_{rho, w, b} eq(r_rho, rho) * eq(r_wire, w) * eq(r_bit, b) * bit_b(word[rho][w])
	//
	// The evaluation point `r` is fresh and unrelated to the reduction's own r_z / r_x challenges;
	// its low Word::LOG_BITS coordinates are the bit axis and its high coordinates are the word
	// axis, matching the layout `fold_instances` produces.
	#[test]
	fn fold_instances_commutes_with_evaluation() {
		type P = PackedBinaryGhash1x128b;

		let c = crc64_circuit();
		let mut rng = StdRng::seed_from_u64(0);

		// Cover every chunk regime of the per-column fold.
		// A sub-chunk batch (< CHUNK_SIZE instances), exactly one chunk, and several chunks.
		for log_instances in [3, Word::LOG_BITS, Word::LOG_BITS + 2] {
			let n_instances = 1usize << log_instances;

			let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
				.map(|_| std::array::from_fn(|_| rng.random()))
				.collect();
			let table = populate_crc64_witness(&c, &inputs);
			let constants = &c.circuit.constraint_system().constants;

			// The committed witness segment, whose word count fixes the word (x) axis.
			let layout = table.layout();
			let offset = layout.offset_witness;
			let n_committed = layout.combined_len() - offset;
			let log_committed = checked_log_2(n_committed);

			// The instance-fold point, and a fresh point over the (bit, word) axes.
			let r_rho = random_scalars::<B128>(&mut rng, log_instances);
			let r = random_scalars::<B128>(&mut rng, Word::LOG_BITS + log_committed);

			// Route A: fold the instance axis, giving one FoldedWord per committed word, then
			// evaluate that (bit, word) multilinear at r.
			let folded = fold_instances::<B128, _>(&table, &r_rho, &GlobalAllocator);
			let (r_bit, r_wire) = r.split_at(Word::LOG_BITS);
			let lhs = evaluate_folded_witness(&folded, r_bit, r_wire);

			// Route B: fold each word's bits by the tensor expansion of the bit coordinates, then
			// evaluate the resulting (word, instance) multilinear over the word and instance axes.
			let bit_tensor = eq_ind_partial_eval_scalars::<B128>(r_bit);

			// Gather the committed words of every instance, instance-major: index = rho *
			// n_committed + w. Each instance is reconstructed independently of the fold under test.
			let mut committed = Vec::with_capacity(n_instances * n_committed);
			for rho in 0..n_instances {
				let vv = table.instance_value_vec(rho, constants);
				committed.extend_from_slice(&vv.combined_witness()[offset..]);
			}
			let folded_words = fold_words::<B128, P, _>(&GlobalAllocator, &committed, &bit_tensor);

			let mut point = r_wire.to_vec();
			point.extend_from_slice(&r_rho);
			let rhs = evaluate(&folded_words, &point);

			assert_eq!(lhs, rhs, "mismatch at log_instances = {log_instances}");
		}
	}
}
