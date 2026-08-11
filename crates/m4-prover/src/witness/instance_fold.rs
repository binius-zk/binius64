// Copyright 2026 The Binius Developers

//! The committed witness with its instance axis collapsed at a challenge point.

use binius_compute::{Allocator, VecLike};
use binius_core::{constraint_system::Shift, word::Word};
use binius_field::{BinaryField, PackedField};
use binius_math::{FieldVec, inner_product::inner_product};
use binius_prover::fold_word::WordFolder;
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};

use crate::ValueTable;

/// One 64-bit word with its bit axis expanded into full field elements.
///
/// Each bit position becomes one element, so the word is carried in oblong form.
pub type FoldedWord<F> = [F; Word::BITS];

/// Applies a shift to a word whose bits have been folded over the instance axis.
///
/// A shift cannot act on field elements the way it acts on bits, but it does not need to.
/// Every variant reads at most one source bit per output position, so a shift is a permutation with
/// holes over those positions.
/// Folding over instances is linear in the bits, so the two commute:
///
/// ```text
/// fold(op(w, s))[k] = fold(w)[source_bits[k]]
/// ```
///
/// which makes this a **gather**: 64 moves, with a zero wherever the output position reads nothing.
/// Going through the shift indicator would be a 64x64 matrix of field multiplications for the same
/// answer.
///
/// # Arguments
///
/// - `word`: the word's bits, already folded over the instance axis.
/// - `shift`: the shift to apply.
pub fn shift_folded_word<F: BinaryField>(word: &FoldedWord<F>, shift: Shift) -> FoldedWord<F> {
	// The common case copies straight through, with no source map to build.
	if shift.is_identity() {
		return *word;
	}
	let source_bits = shift.source_bits();
	std::array::from_fn(|out| match source_bits[out] {
		Some(source) => word[source as usize],
		// The shift moved a zero into this position, and zero folds to zero whatever the point.
		None => F::ZERO,
	})
}

/// The committed witness of a batch, with its instance axis collapsed at a point.
///
/// # Overview
///
/// The committed witness spans three axes:
///
/// - the bits within each 64-bit word;
/// - the committed words of one instance;
/// - the instances of the batch.
///
/// Collapsing the instance axis leaves a multilinear over the remaining two:
///
/// ```text
/// index = w * Word::BITS + b        b occupies the low Word::LOG_BITS coordinates
/// ```
///
/// So the bit axis takes the low coordinates and the word axis the high ones.
///
/// The committed word count need not be a power of two.
/// Anything that produces a multilinear from it zero-pads the word axis up to one.
pub struct FoldedWitness<F: Send, A: Allocator> {
	/// The oblong words, one per committed word of a single instance, in wire order.
	words: A::Vec<FoldedWord<F>>,
}

impl<F: BinaryField, A: Allocator> FoldedWitness<F, A> {
	/// Collapses the instance axis of a batch witness at a challenge point.
	///
	/// # Overview
	///
	/// Every instance is weighted by its equality indicator, and the weighted bits are summed:
	///
	/// ```text
	/// out[w][b] = sum_rho eq(r_rho, rho) * bit_b(word[rho][w])
	/// ```
	///
	/// So each set bit contributes its own instance's weight to a full field element.
	///
	/// Only the hidden segment is stored in the table, so only it is folded here.
	/// The shared constants live on the constraint system and fold separately.
	///
	/// # Performance
	///
	/// The wire-major layout stores one wire's instances contiguously.
	/// Each word position therefore reads a plain sub-slice, not a strided gather.
	/// The positions are independent, so they fold in parallel, one task per output word.
	///
	/// # Panics
	///
	/// Panics if the point width does not match the batch dimension.
	pub fn fold_instances(table: &ValueTable, r_rho: &[F], alloc: &A) -> Self {
		// Invariant: one challenge coordinate per instance-axis variable.
		assert_eq!(r_rho.len(), table.log_instances(), "r_rho must match the batch dimension");

		// The bit lookup tables and per-chunk weights depend only on the point.
		// Build them once here, then share them across every word position.
		let folder = WordFolder::<F>::new(r_rho);
		let n_words = table.n_hidden_words();

		// Invariant: the stored words split evenly into one chunk per committed word.
		//
		//     stored:      n_words * 2^log_instances
		//     chunked by:  2^log_instances -> exactly n_words chunks
		debug_assert_eq!(table.as_words().len(), n_words << table.log_instances());

		// One output element per committed word position.
		let mut words = alloc.alloc::<FoldedWord<F>>(n_words);
		table
			.as_words()
			// One chunk is one word position across every instance of the batch.
			.par_chunks(1 << table.log_instances())
			.zip(words.spare_capacity_mut())
			.for_each(|(instance_words, out)| {
				// Collapse those instances into 64 elements, one per bit position.
				out.write(folder.fold(instance_words));
			});

		// SAFETY: chunks and output elements correspond one-to-one by the equality above, so the
		// loop writes each of the `n_words` elements exactly once.
		unsafe { words.set_len(n_words) };

		Self { words }
	}

	/// The oblong words, one per committed word, in wire order.
	pub fn words(&self) -> &[FoldedWord<F>] {
		&self.words
	}

	/// The number of word-axis variables, after padding the word count to a power of two.
	///
	/// This is the width of the word half of any point the witness is evaluated at.
	pub fn log_padded_words(&self) -> usize {
		log2_ceil_usize(self.words.len())
	}

	/// Contracts the bit axis at a point, leaving a multilinear over the word axis.
	///
	/// # Overview
	///
	/// Each word's bits are contracted against the supplied weights:
	///
	/// ```text
	/// out[w] = sum_b weights[b] * word_w[b]
	/// ```
	///
	/// This is the partial evaluation of the witness along its bit axis.
	/// The word axis is zero-padded up to a power of two.
	///
	/// # Panics
	///
	/// Panics unless there is exactly one weight per bit of a word.
	pub fn fold_bits<P>(&self, r_j_tensor: &[F], alloc: &A) -> FieldVec<P, A>
	where
		P: PackedField<Scalar = F>,
	{
		// Invariant: one weight per bit position.
		// The contraction pairs both sides with a length-checked zip, which would otherwise panic
		// from inside the arithmetic without naming the cause.
		assert_eq!(r_j_tensor.len(), Word::BITS, "r_j_tensor must hold one weight per word bit");

		// A multilinear needs a power-of-two word axis; the real words fill the front of it.
		let padded_len = 1 << self.log_padded_words();
		let mut scalars = alloc.alloc::<F>(padded_len);

		// One scalar per committed word: that word's bits contracted at the point.
		scalars.extend(
			self.words
				.iter()
				.map(|word| inner_product(word.iter().copied(), r_j_tensor.iter().copied())),
		);

		// Everything past the real words is the multilinear's zero padding.
		scalars.resize(padded_len, F::ZERO);

		// Pack the scalars into the buffer shape the sumcheck consumes.
		FieldVec::<P, A>::from_values_in(alloc, &scalars)
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_field::{Field, PackedBinaryGhash1x128b, Random};
	use binius_frontend::{CircuitBuilder, Wire};
	use binius_math::{
		multilinear::{
			eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars},
			evaluate::evaluate,
		},
		test_utils::random_scalars,
	};
	use binius_prover::fold_word::fold_words;
	use binius_utils::checked_arithmetics::checked_log_2;
	use binius_verifier::config::B128;
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{N_INPUT_WORDS, crc64_circuit, populate_crc64_witness};

	impl<F: BinaryField, A: Allocator> FoldedWitness<F, A> {
		/// Evaluates the witness at a point split into a bit half and a word half.
		///
		/// # Overview
		///
		/// - Contract the bit axis at the first half of the point.
		/// - Contract the resulting per-word multilinear at the second half.
		///
		/// The word axis is zero-padded out to the width of the word half.
		/// Those padded positions are zero, so they drop out of the result.
		///
		/// # Why this exists
		///
		/// This contracts scalar by scalar, where the production path packs words into field lanes.
		/// Agreeing with it is what pins that packed path.
		///
		/// # Panics
		///
		/// Panics if the word half is too narrow to index every committed word.
		pub fn evaluate(&self, r_j: &[F], r_y: &[F]) -> F {
			// Expand both halves of the point into equality weights.
			let r_j_tensor = eq_ind_partial_eval::<F>(r_j);
			let r_y_tensor = eq_ind_partial_eval::<F>(r_y);
			let n_padded = r_y_tensor.as_ref().len();

			// Invariant: the word half addresses at least every real word.
			assert!(self.words.len() <= n_padded, "r_y must index every committed word");

			// Contract each word's bits, then extend the axis with the zeros the padding stands
			// for.
			let per_word = self
				.words
				.iter()
				.map(|word| {
					inner_product(word.iter().copied(), r_j_tensor.as_ref().iter().copied())
				})
				.chain(iter::repeat(F::ZERO))
				.take(n_padded);

			// Contract the per-word multilinear at the word half of the point.
			inner_product(per_word, r_y_tensor.as_ref().iter().copied())
		}
	}

	// A batch of 2^3 instances of a circuit whose committed word count is `3 * n_gates`: each gate
	// commits two inputs and the conjunction it promotes to a public output.
	//
	// A multiple of three is a power of two only for a hand-picked gate count.
	// That is what lets a caller choose whether the word axis needs padding.
	fn and_chain_table(n_gates: usize) -> ValueTable {
		// Each gate consumes two fresh input words and promotes its result to a public output.
		// That promotion is what keeps the gate alive under dead-code elimination.
		let builder = CircuitBuilder::new();
		let inputs: Vec<Wire> = (0..2 * n_gates).map(|_| builder.add_inout()).collect();
		for pair in inputs.chunks_exact(2) {
			builder.mark_inout(builder.band(pair[0], pair[1]));
		}
		let circuit = builder.build();

		// Every instance gets its own seed, so no two instances share an input.
		ValueTable::populate(&circuit, 3, |i, w| {
			let mut rng = StdRng::seed_from_u64(i as u64);
			for &wire in &inputs {
				w[wire] = Word(rng.random());
			}
		})
		.unwrap()
	}

	#[test]
	fn fold_instances_commutes_with_evaluation() {
		type P = PackedBinaryGhash1x128b;

		// Invariant: the order of the two folds does not matter.
		// Both routes compute one triple sum, associated differently:
		//
		//     sum_{rho, w, b} eq(r_rho, rho) * eq(r_wire, w) * eq(r_bit, b) * bit_b(word[rho][w])
		//
		//     route A: fold instances first  -> evaluate over (bit, word)
		//     route B: fold bits first       -> evaluate over (word, instance)
		let c = crc64_circuit();
		let mut rng = StdRng::seed_from_u64(0);

		// Fixture state: three batch sizes covering every chunk regime of the per-column fold.
		// Below one chunk, exactly one chunk, and several chunks.
		for log_instances in [3, Word::LOG_BITS, Word::LOG_BITS + 2] {
			let n_instances = 1usize << log_instances;

			let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
				.map(|_| std::array::from_fn(|_| rng.random()))
				.collect();
			let table = populate_crc64_witness(&c, &inputs);
			let constants = &c.circuit.constraint_system().constants;

			// The committed segment runs from the inout offset to the end of the value vector.
			// Its length fixes the width of the word axis.
			let offset = table.layout().offset_inout();
			let n_committed = table.n_hidden_words();
			assert_eq!(n_committed, table.layout().combined_len() - offset);
			let log_committed = checked_log_2(n_committed);

			// The instance-fold point, and a fresh point over the (bit, word) axes.
			// This point is unrelated to the reduction's own challenges.
			let r_rho = random_scalars::<B128>(&mut rng, log_instances);
			let r = random_scalars::<B128>(&mut rng, Word::LOG_BITS + log_committed);

			// Route A: collapse instances, then evaluate the (bit, word) multilinear.
			// The low coordinates are the bit axis, matching the layout the fold produces.
			let folded = FoldedWitness::<B128, _>::fold_instances(&table, &r_rho, &GlobalAllocator);
			let (r_bit, r_wire) = r.split_at(Word::LOG_BITS);
			let lhs = folded.evaluate(r_bit, r_wire);

			// Route B: collapse each word's bits against the bit coordinates first.
			let bit_tensor = eq_ind_partial_eval_scalars::<B128>(r_bit);

			// Gather every instance's committed words, instance-major:
			//
			//     index = rho * n_committed + w
			//
			// Each instance is reconstructed on its own, independently of the fold under test.
			let mut committed = Vec::with_capacity(n_instances * n_committed);
			for rho in 0..n_instances {
				let vv = table.instance_value_vec(rho, constants);
				committed.extend_from_slice(&vv.combined_witness()[offset..]);
			}
			let folded_words = fold_words::<B128, P, _>(&GlobalAllocator, &committed, &bit_tensor);

			// Route B evaluates over (word, instance), so the point is reordered to match.
			let mut point = r_wire.to_vec();
			point.extend_from_slice(&r_rho);
			let rhs = evaluate(&folded_words, &point);

			assert_eq!(lhs, rhs, "mismatch at log_instances = {log_instances}");
		}
	}

	#[test]
	fn shifting_a_folded_row_matches_folding_the_shifted_words() {
		// Invariant: shifting a folded row equals folding the shifted words.
		//
		// The shift is a permutation with holes over the bit positions, and the fold is linear in
		// those bits, so the two commute. That is what turns forming an intermediate row into a
		// gather rather than a 64x64 field product against the shift indicator.
		//
		// Every canonical shift is covered, so a wrong source map cannot hide.
		use binius_core::{ShiftVariant, constraint_system::Shift};

		let mut rng = StdRng::seed_from_u64(4);

		// One word per instance, drawn at random, and the point the instance axis folds at.
		const LOG_INSTANCES: usize = 4;
		let words = (0..1 << LOG_INSTANCES)
			.map(|_| Word(rng.random::<u64>()))
			.collect::<Vec<_>>();
		let r_rho = random_scalars::<B128>(&mut rng, LOG_INSTANCES);
		let eq = eq_ind_partial_eval::<B128>(&r_rho);

		// Folds a batch of words over the instance axis, one field element per bit position.
		let fold = |words: &[Word]| -> FoldedWord<B128> {
			let mut folded = [B128::ZERO; Word::BITS];
			for (word, &weight) in iter::zip(words, eq.as_ref()) {
				for (bit, slot) in folded.iter_mut().enumerate() {
					if (word.0 >> bit) & 1 == 1 {
						*slot += weight;
					}
				}
			}
			folded
		};

		let folded = fold(&words);
		for variant in ShiftVariant::ALL {
			for amount in 0..variant.max_amount() {
				let shift = Shift::new(variant, amount);

				// Route A: shift every word, then fold.
				let shifted_words = words
					.iter()
					.map(|&word| shift.apply(word))
					.collect::<Vec<_>>();
				let expected = fold(&shifted_words);

				// Route B: fold once, then gather through the shift's source map.
				let gathered = shift_folded_word(&folded, shift);

				assert_eq!(
					gathered, expected,
					"{shift:?} disagrees with folding the shifted words"
				);
			}
		}
	}

	#[test]
	#[ignore = "reports a measurement rather than asserting a property"]
	fn gather_versus_indicator_contraction() {
		// What the gather buys over contracting against the shift indicator.
		//
		//     gather   : read the source map, move 64 elements
		//     reference: sum shift-ind(k, j, s) * row[j] over all 64 source positions
		//
		// The reference is the 64x64 field product the at-most-one-source-bit property lets the
		// outer phase skip. Both routes form the same row, which is asserted before the timings.
		use std::time::Instant;

		use binius_core::{ShiftVariant, constraint_system::Shift};

		let mut rng = StdRng::seed_from_u64(9);
		// One folded row per key the outer phase would touch, at a realistic count for sha256's
		// private segment.
		const N_ROWS: usize = 8192;
		let rows = (0..N_ROWS)
			.map(|_| std::array::from_fn(|_| B128::random(&mut rng)))
			.collect::<Vec<FoldedWord<B128>>>();
		// A spread of variants, so neither route is measured on one alone.
		let shifts = ShiftVariant::ALL
			.into_iter()
			.map(|variant| Shift::new(variant, 7))
			.collect::<Vec<_>>();

		// The reference: contract the row against the shift indicator over every source position.
		let by_indicator = |row: &FoldedWord<B128>, shift: Shift| -> FoldedWord<B128> {
			let source_bits = shift.source_bits();
			std::array::from_fn(|out| {
				(0..Word::BITS)
					.map(|source| {
						let indicator = if source_bits[out] == Some(source as u8) {
							B128::ONE
						} else {
							B128::ZERO
						};
						indicator * row[source]
					})
					.sum()
			})
		};

		// Time both, touching the results so neither is optimized away.
		let mut checksum = [B128::ZERO; 2];
		let mut elapsed = [std::time::Duration::ZERO; 2];
		for (index, route) in [
			&by_indicator as &dyn Fn(&FoldedWord<B128>, Shift) -> FoldedWord<B128>,
			&|row, shift| shift_folded_word(row, shift),
		]
		.into_iter()
		.enumerate()
		{
			let start = Instant::now();
			for (row, &shift) in iter::zip(&rows, shifts.iter().cycle()) {
				checksum[index] += route(row, shift)[0];
			}
			elapsed[index] = start.elapsed();
		}

		// Both routes must agree, or the timings compare different computations.
		assert_eq!(checksum[0], checksum[1]);
		let [indicator, gather] = elapsed;
		println!("intermediate rows: {N_ROWS}");
		println!("  indicator contraction: {indicator:?}");
		println!("  gather:                {gather:?}");
		println!("  speedup:               {:.1}x", indicator.as_secs_f64() / gather.as_secs_f64());
	}

	#[test]
	fn fold_bits_matches_the_scalar_contraction() {
		let mut rng = StdRng::seed_from_u64(1);

		// Invariant: packing the bit contraction into field lanes changes no evaluation.
		// The packed buffer also carries zero padding, which must not shift the result.
		//
		// Fixture state: two witnesses, one per word-axis regime.
		//
		//     CRC-64:    1024 committed words  -> already a power of two, no padding
		//     AND chain:    6 committed words  -> zero-padded up to 8
		let c = crc64_circuit();
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..8)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();
		let unpadded = populate_crc64_witness(&c, &inputs);
		let padded = and_chain_table(2);

		// Pin which fixture is which, so neither regime can silently stop being covered.
		assert!(unpadded.n_hidden_words().is_power_of_two(), "fixture must skip the padding");
		assert!(!padded.n_hidden_words().is_power_of_two(), "fixture must reach the padding");

		for table in [unpadded, padded] {
			let r_rho = random_scalars::<B128>(&mut rng, table.log_instances());
			let folded = FoldedWitness::<B128, _>::fold_instances(&table, &r_rho, &GlobalAllocator);

			// The bit point, and a word point as wide as the padded word axis.
			let r_j = random_scalars::<B128>(&mut rng, Word::LOG_BITS);
			let r_y = random_scalars::<B128>(&mut rng, folded.log_padded_words());
			let r_j_tensor = eq_ind_partial_eval::<B128>(&r_j);

			// The packed contraction, evaluated at the word point, against the scalar reference.
			let packed =
				folded.fold_bits::<PackedBinaryGhash1x128b>(r_j_tensor.as_ref(), &GlobalAllocator);
			assert_eq!(
				evaluate(&packed, &r_y),
				folded.evaluate(&r_j, &r_y),
				"mismatch at {} committed words",
				table.n_hidden_words()
			);
		}
	}

	#[test]
	#[should_panic(expected = "r_j_tensor must hold one weight per word bit")]
	fn fold_bits_rejects_a_tensor_that_is_not_one_weight_per_bit() {
		// Fixture state: two instances of the smallest witness that populates.
		let c = crc64_circuit();
		let table = populate_crc64_witness(&c, &[[0; N_INPUT_WORDS], [1; N_INPUT_WORDS]]);
		let folded =
			FoldedWitness::<B128, _>::fold_instances(&table, &[B128::new(7)], &GlobalAllocator);

		// Mutation: one coordinate short of the bit axis.
		//
		//     weights supplied: 2^5 = 32
		//     bits per word:    2^6 = 64
		//
		// The contraction pairs the two sides, so the width check must reject this up front.
		let short = eq_ind_partial_eval::<B128>(&[B128::new(3); Word::LOG_BITS - 1]);
		let _ = folded.fold_bits::<PackedBinaryGhash1x128b>(short.as_ref(), &GlobalAllocator);
	}
}
