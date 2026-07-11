// Copyright 2025 Irreducible Inc.

//! The batched BitAnd-check witness built from a populated batch value table.

use std::mem::MaybeUninit;

use binius_core::{
	constraint_system::{AndConstraint, ShiftedValueIndex},
	word::Word,
};
use binius_field::{AESTowerField8b as B8, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::BinarySubspace;
use binius_prover::and_reduction::prover::OblongZerocheckProver;
use binius_utils::{checked_arithmetics::checked_log_2, rayon::prelude::*};
use binius_verifier::{
	config::{B128, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES},
	protocols::bitand::AndCheckOutput,
};

use crate::ValueTable;

/// Instance columns processed by one parallel witness-assembly task.
///
/// A task streams contiguous sub-rows of this many instances out of the wire-major table.
/// Wide enough that each row read amortizes over many instances.
/// Small enough to spread across cores on realistic batch sizes.
const STRIPE_WIDTH: usize = 256;

/// The operand columns of the BitAnd check for a whole batch of instances.
///
/// The BitAnd check works on three columns of words, one row per AND constraint.
/// It enforces the bitwise relation `A & B == C` on every row.
///
/// This holds those three columns for a batch of `K = 2^log_instances` instances at once.
/// The rows are stacked in instance-major order, with the instance index on the high coordinates:
///
/// ```text
///         instance 0         instance 1            instance K-1
///   A: [ a_0 .. a_{n-1} ][ a_0 .. a_{n-1} ] ... [ a_0 .. a_{n-1} ]
///   B: [ b_0 .. b_{n-1} ][ b_0 .. b_{n-1} ] ... [ b_0 .. b_{n-1} ]
///   C: [ c_0 .. c_{n-1} ][ c_0 .. c_{n-1} ] ... [ c_0 .. c_{n-1} ]
///       \___ n_and ___/
/// ```
///
/// The row index splits cleanly into the instance and the per-instance constraint:
///
/// ```text
/// row = instance * n_and + local_constraint
/// ```
///
/// - The high `log_instances` bits select the instance.
/// - The low bits select the constraint within that instance.
/// - The reduction reads each column as a multilinear over `log_instances + log(n_and)` bits.
///
/// `n_and` is a power of two when the constraints come from a prepared constraint system.
/// So the batch forms a clean hypercube whose high coordinates are the instance index.
#[derive(Clone, Debug)]
pub struct BatchAndCheckWitness {
	/// Operand `A` of every constraint of every instance, instance-major.
	a: Vec<Word>,
	/// Operand `B` of every constraint of every instance, instance-major.
	b: Vec<Word>,
	/// Operand `C` of every constraint of every instance, instance-major.
	c: Vec<Word>,
}

impl BatchAndCheckWitness {
	/// Builds the batched BitAnd witness from a populated wire-major batch table.
	///
	/// Every AND constraint is evaluated against every instance.
	/// This produces `K * n_and` rows laid out instance-major.
	/// Row `instance * n_and + j` is constraint `j` of that instance.
	///
	/// An operand is a XOR of shifted committed values:
	///
	/// ```text
	/// operand(instance) = XOR_t shift_t( value(index_t, instance) )
	/// ```
	///
	/// A value index splits by the witness offset:
	/// - below it: a public word, the same constant in every instance.
	/// - at or above it: a hidden word read from the table.
	///
	/// One index's hidden words across all instances form one contiguous row of the buffer.
	/// So a term streams that row, XOR-ing its shifted words into the column.
	/// This replaces the per-instance gather the instance-major layout required.
	///
	/// # Arguments
	///
	/// - `table`: the wire-major batch witness holding every instance's hidden words.
	/// - `constants`: the circuit's constant words, shared by every instance.
	/// - `and_constraints`: the per-instance AND constraints, shared by every instance.
	///
	/// Pass constraints from a prepared constraint system, so their count is a power of two.
	///
	/// # Panics
	///
	/// Panics if the constraint count or the instance count is not a power of two.
	pub fn build(
		table: &ValueTable,
		constants: &[Word],
		and_constraints: &[AndConstraint],
	) -> Self {
		// Rows per instance, and total rows across the batch.
		let n_and = and_constraints.len();
		let n_instances = table.n_instances();
		let total = n_instances * n_and;

		// Both dimensions are powers of two, so both are at least 1.
		// - The row count `K * n_and` is then at least 1, so the witness is never empty.
		// - The chunk size used below is then never zero, so the parallel split is well-defined.
		assert!(n_and.is_power_of_two(), "constraint count must be a power of two");
		assert!(n_instances.is_power_of_two(), "instance count must be a power of two");

		// One column each for the three operands, laid out instance-major.
		// The columns are left uninitialized here.
		// Each task zeroes its own chunk, fusing the zero-fill into the write pass.
		let mut a = Vec::<Word>::with_capacity(total);
		let mut b = Vec::<Word>::with_capacity(total);
		let mut c = Vec::<Word>::with_capacity(total);

		// The witness offset splits public words (below) from hidden rows (at or above).
		let offset = table.layout().offset_witness;
		let log_instances = table.log_instances();
		let data = table.as_words();

		// Each task owns a contiguous stripe of instances.
		// Its output blocks are those instances' rows of the three instance-major columns.
		// Reads stay contiguous: within a stripe, one hidden row is a contiguous sub-slice.
		let block = STRIPE_WIDTH.min(n_instances) * n_and;
		{
			let a_spare = &mut a.spare_capacity_mut()[..total];
			let b_spare = &mut b.spare_capacity_mut()[..total];
			let c_spare = &mut c.spare_capacity_mut()[..total];
			a_spare
				.par_chunks_mut(block)
				.zip(b_spare.par_chunks_mut(block))
				.zip(c_spare.par_chunks_mut(block))
				.enumerate()
				.for_each(|(stripe, ((a_block, b_block), c_block))| {
					// Zero this chunk, then view it as an initialized slice to accumulate into.
					let a_block = init_zeroed(a_block);
					let b_block = init_zeroed(b_block);
					let c_block = init_zeroed(c_block);

					// The first global instance of this stripe.
					// The instances in this stripe, derived from the block length.
					let base = stripe * STRIPE_WIDTH;
					let width = a_block.len() / n_and;

					for (j, constraint) in and_constraints.iter().enumerate() {
						let ctx = OperandContext {
							data,
							constants,
							offset,
							log_instances,
							base,
							width,
							n_and,
							j,
						};
						ctx.accumulate(a_block, &constraint.a);
						ctx.accumulate(b_block, &constraint.b);
						ctx.accumulate(c_block, &constraint.c);
					}
				});
		}

		// The stripes partition `[0, total)` and each zeroed its whole range.
		// So all `total` elements of every column are initialized.
		//
		// SAFETY: every element in `0..total` was written above.
		unsafe {
			a.set_len(total);
			b.set_len(total);
			c.set_len(total);
		}

		Self { a, b, c }
	}

	/// Operand `A` column, `K * n_and` rows in instance-major order.
	pub fn a(&self) -> &[Word] {
		&self.a
	}

	/// Operand `B` column, `K * n_and` rows in instance-major order.
	pub fn b(&self) -> &[Word] {
		&self.b
	}

	/// Operand `C` column, `K * n_and` rows in instance-major order.
	pub fn c(&self) -> &[Word] {
		&self.c
	}

	/// Consumes the witness into its three operand columns `(A, B, C)`.
	///
	/// This is the shape the AND reduction destructures to drive its sumcheck.
	pub fn into_columns(self) -> (Vec<Word>, Vec<Word>, Vec<Word>) {
		(self.a, self.b, self.c)
	}

	/// Proves the batched BitAnd check: `A & B == C` on every row of every instance.
	///
	/// The check is the univariate-skip zerocheck of the constraint polynomial:
	///
	/// ```text
	/// A(Z, X) * B(Z, X) - C(Z, X) == 0   for all rows (Z, X)
	/// ```
	///
	/// `Z` is the bit index within a 64-bit word.
	/// `X` is the row index.
	///
	/// The batch carries no special structure for this step.
	/// The stacked rows are one flat hypercube.
	/// So the batch is just a larger single zerocheck.
	///
	/// ```text
	///     row = instance * n_and + local_constraint
	///   X bits = log_instances + log(n_and)
	/// ```
	///
	/// This reuses the single-instance kernel verbatim, only over `K * n_and` rows.
	/// The block-diagonal batch structure is exploited later, in the lincheck, not here.
	///
	/// The reduction folds the three bit-columns into one multilinear evaluation point.
	/// It returns the claimed evaluations of `A`, `B`, `C` at that point.
	/// A later shift reduction ties those claims back to the committed witness.
	///
	/// # Type parameters
	///
	/// - `P`: the packed field for the SIMD multilinear sumcheck rounds.
	///
	/// # Arguments
	///
	/// - `channel`: the prover channel that records messages and draws Fiat-Shamir challenges.
	///
	/// # Returns
	///
	/// The reduced claim, holding:
	/// - The claimed `A`, `B`, `C` evaluations.
	/// - The univariate (bit-index) challenge.
	/// - The multilinear evaluation point reached by the sumcheck.
	pub fn prove<P, Channel>(self, channel: &mut Channel) -> AndCheckOutput<B128>
	where
		P: PackedField<Scalar = B128>,
		Channel: IPProverChannel<B128>,
	{
		// The univariate-skip domain spans one extra dimension above the 64-bit word.
		// This is the same skip parameter the single-instance check uses.
		let prover_message_domain = BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1);

		let (a, b, c) = self.into_columns();

		// X has `log_instances + log(n_and)` coordinates: the row count is a power of two.
		let log_total_constraints = checked_log_2(a.len());

		// Pin the first few zerocheck coordinates to fixed small-field elements (friendly
		// challenges).
		// Draw the rest from the large field.
		// The prover and verifier pin and draw the same split, in the same order.
		//
		//     coordinates = [ pinned small-field | sampled large-field ]
		//                    \____ at most 3 ___/
		let n_extra_zerocheck_challenges =
			log_total_constraints.saturating_sub(PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len());
		let big_field_zerocheck_challenges = channel.sample_many(n_extra_zerocheck_challenges);

		let prover = OblongZerocheckProver::<_, P>::new(
			log_total_constraints,
			a,
			b,
			c,
			big_field_zerocheck_challenges,
			prover_message_domain.isomorphic(),
		);

		prover.prove_with_channel(channel)
	}
}

/// Zeroes an uninitialized word chunk and returns it as an initialized slice.
///
/// Every element is set to `Word::ZERO`, so the caller can XOR operand terms into it.
/// Chunks whose constraints have empty operands then stay zero, as the reduction expects.
fn init_zeroed(chunk: &mut [MaybeUninit<Word>]) -> &mut [Word] {
	for slot in chunk.iter_mut() {
		slot.write(Word::ZERO);
	}
	// Every element was just written.
	// `MaybeUninit<Word>` shares the layout of `Word`.
	//
	// SAFETY: the whole chunk is initialized, so reinterpreting it as `Word` is sound.
	unsafe { &mut *(chunk as *mut [MaybeUninit<Word>] as *mut [Word]) }
}

/// The placement of one operand column for one constraint over one instance stripe.
///
/// It names where to read source words: the wire-major table and the constant bank.
/// It names where to write them: one constraint's column across the stripe's instances.
struct OperandContext<'a> {
	/// The wire-major hidden words: row `r` spans all instances of hidden value `offset + r`.
	data: &'a [Word],
	/// The circuit's constant words, shared by every instance.
	constants: &'a [Word],
	/// The value index at which hidden words begin.
	/// Public words lie below it.
	offset: usize,
	/// The base-2 logarithm of the instance count, i.e. the stride of one hidden row.
	log_instances: usize,
	/// The first global instance of this stripe.
	base: usize,
	/// The number of instances this stripe spans.
	width: usize,
	/// The number of constraints, i.e. the stride between one instance's columns and the next.
	n_and: usize,
	/// The constraint index whose column this operand fills.
	j: usize,
}

impl OperandContext<'_> {
	/// Accumulates one operand into its constraint's column across this stripe's instances.
	///
	/// For each term the operand XORs, the shifted source word is added into every instance:
	///
	/// ```text
	/// out[local * n_and + j] ^= shift_t( value(index_t, base + local) )   for local in 0..width
	/// ```
	///
	/// A hidden term streams one contiguous sub-row.
	/// A public term broadcasts one shared word.
	fn accumulate(&self, out: &mut [Word], operand: &[ShiftedValueIndex]) {
		for sv in operand {
			let idx = sv.value_index.0 as usize;
			let amount = sv.amount as usize;
			if idx >= self.offset {
				// Hidden word: this stripe's slice of row `idx - offset` is contiguous.
				let row = idx - self.offset;
				let src = &self.data[(row << self.log_instances) + self.base..][..self.width];
				for (local, &word) in src.iter().enumerate() {
					let slot = &mut out[local * self.n_and + self.j];
					*slot = *slot ^ sv.shift_variant.apply(word, amount);
				}
			} else {
				// Public word: the same constant enters every instance of this stripe.
				// Indices past the constant bank are layout padding, read as zero.
				let word = self.constants.get(idx).copied().unwrap_or(Word::ZERO);
				let shifted = sv.shift_variant.apply(word, amount);
				for local in 0..self.width {
					let slot = &mut out[local * self.n_and + self.j];
					*slot = *slot ^ shifted;
				}
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_core::constraint_system::ValueVec;
	use binius_field::{
		PackedBinaryGhash1x128b,
		linear_transformation::{
			BytewiseLookupTransformationFactory, LinearTransformationFactory,
			OutputWrappingTransformationFactory,
		},
	};
	use binius_frontend::{Circuit, CircuitBuilder, Wire};
	use binius_ip::channel::Error as ChannelError;
	use binius_m4_verifier::verify_bitand_reduction;
	use binius_math::{
		FieldBuffer, multilinear::evaluate::evaluate, univariate::lagrange_evals_scalars,
	};
	use binius_prover::fold_word::fold_words_with_transform;
	use binius_transcript::{ProverTranscript, VerifierTranscript};
	use binius_verifier::{
		Error as VerifierError, config::StdChallenger, protocols::bitand::SKIPPED_VARS,
	};
	use proptest::prelude::*;

	use super::*;

	/// A width-1 packed field keeps one scalar per element, so the SIMD sumcheck rounds stay
	/// simple.
	type P = PackedBinaryGhash1x128b;

	/// Recomputes the true multilinear evaluation of one bit-column at the reduction's point.
	///
	/// This mirrors the single-instance kernel's own consistency check:
	///
	///     fold the column at the bit-index challenge z
	///     evaluate the folded multilinear at the sumcheck point
	///
	/// The result must equal the eval the reduction claimed for that column.
	fn fold_eval_column(col: &[Word], z_challenge: B128, eval_point: &[B128]) -> B128 {
		// The univariate domain is the skip domain with the extension dimension dropped.
		let univariate_domain = BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1)
			.isomorphic::<B128>()
			.reduce_dim(SKIPPED_VARS);
		let lagrange = lagrange_evals_scalars(&univariate_domain, z_challenge);
		let transform =
			OutputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory)
				.create(&lagrange);
		let folded: FieldBuffer<B128> = fold_words_with_transform(&transform, col);
		evaluate(&folded, eval_point)
	}

	// The prepared per-instance AND constraints, padded to a power of two.
	// This mirrors how the prover feeds constraints downstream.
	fn table_constraints(c: &AndCircuit) -> Vec<AndConstraint> {
		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		cs.and_constraints
	}

	// The circuit's constant words, shared by every instance.
	// This circuit declares none, so the slice is empty; operands read only hidden words.
	fn constants(c: &AndCircuit) -> &[Word] {
		&c.circuit.constraint_system().constants
	}

	// A circuit asserting `z == (x & y) ^ w`, over four witness words.
	//
	//     inputs : x, y, w, z   (all witness — the wire-major table admits no inout)
	//     gate   : and = x & y
	//     assert : and ^ w == z
	struct AndCircuit {
		circuit: Circuit,
		x: Wire,
		y: Wire,
		w: Wire,
		z: Wire,
	}

	fn and_circuit() -> AndCircuit {
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let y = builder.add_witness();
		let w = builder.add_witness();
		let z = builder.add_witness();
		let and = builder.band(x, y);
		let lhs = builder.bxor(and, w);
		builder.assert_eq("z_eq_x_and_y_xor_w", lhs, z);
		AndCircuit {
			circuit: builder.build(),
			x,
			y,
			w,
			z,
		}
	}

	// Populate one instance per input tuple; the instance count is the tuple count.
	//
	// Each tuple is `(x, y, w)`, the three free inputs.
	// The output is derived as `z = (x & y) ^ w`, so every tuple satisfies the circuit.
	//
	// `w` is an arbitrary mask that only feeds the XOR, never the AND.
	// So a tuple like `(1, 3, 7)` means `x=1, y=3, w=7`, not `1 & 3 = 7`.
	fn populate_table(c: &AndCircuit, inputs: &[(u64, u64, u64)]) -> ValueTable {
		let log_instances = inputs.len().ilog2() as usize;
		ValueTable::populate(&c.circuit, log_instances, |i, filler| {
			let (x, y, w) = inputs[i];
			filler[c.x] = Word(x);
			filler[c.y] = Word(y);
			filler[c.w] = Word(w);
			filler[c.z] = Word((x & y) ^ w);
		})
		.unwrap()
	}

	// The reference for one instance: the core operand evaluator on its reconstructed value vec.
	// This is exactly what the single-instance BitAnd witness builder computes.
	fn reference_rows(
		and_constraints: &[AndConstraint],
		vv: &ValueVec,
	) -> (Vec<Word>, Vec<Word>, Vec<Word>) {
		let mut a = Vec::new();
		let mut b = Vec::new();
		let mut c = Vec::new();
		for constraint in and_constraints {
			a.push(vv.eval_operand(&constraint.a));
			b.push(vv.eval_operand(&constraint.b));
			c.push(vv.eval_operand(&constraint.c));
		}
		(a, b, c)
	}

	#[test]
	fn columns_are_instance_major_with_the_expected_shape() {
		let c = and_circuit();

		// Fixture state: 2^2 = 4 instances with distinct, satisfying inputs.
		let inputs = [
			(1, 3, 7),
			(5, 6, 0),
			(9, 12, 0xFF),
			(0xABCD, 0x0F0F, 0x1234),
		];
		let table = populate_table(&c, &inputs);

		let and_constraints = &table_constraints(&c);
		let witness = BatchAndCheckWitness::build(&table, constants(&c), and_constraints);

		// Shape: K * n_and rows, with K = 4.
		let n_and = and_constraints.len();
		assert_eq!(witness.a().len(), 4 * n_and);
		assert_eq!(witness.b().len(), 4 * n_and);
		assert_eq!(witness.c().len(), 4 * n_and);

		// Invariant: row `instance * n_and + j` is constraint `j` of that instance.
		// Each instance's block equals the single-instance reference for its inputs.
		for instance in 0..table.n_instances() {
			let vv = table.instance_value_vec(instance, constants(&c));
			let (a_ref, b_ref, c_ref) = reference_rows(and_constraints, &vv);

			let start = instance * n_and;
			assert_eq!(&witness.a()[start..start + n_and], a_ref.as_slice());
			assert_eq!(&witness.b()[start..start + n_and], b_ref.as_slice());
			assert_eq!(&witness.c()[start..start + n_and], c_ref.as_slice());
		}
	}

	#[test]
	fn and_relation_holds_on_every_row() {
		let c = and_circuit();

		// Fixture state: 4 satisfying instances, each tuple `(x, y, w)`.
		let table = populate_table(&c, &[(1, 3, 7), (5, 6, 0), (9, 12, 0xFF), (0xF0, 0x0F, 1)]);
		let witness = BatchAndCheckWitness::build(&table, constants(&c), &table_constraints(&c));

		// The single AND constraint is `and = x & y`, so each row is `A=x`, `B=y`, `C=x&y`.
		// A satisfying witness therefore makes `A & B == C` hold on every row.
		//
		// Padded rows have empty operands, so `0 & 0 == 0` holds there too.
		for ((a, b), c) in witness.a().iter().zip(witness.b()).zip(witness.c()) {
			assert_eq!(a.0 & b.0, c.0);
		}
	}

	#[test]
	fn single_instance_batch_matches_the_reference() {
		let c = and_circuit();

		// Fixture state: log_instances = 0 → exactly one instance (K = 1).
		let table = populate_table(&c, &[(0xABCD, 0x0F0F, 0x55)]);
		let and_constraints = table_constraints(&c);
		let witness = BatchAndCheckWitness::build(&table, constants(&c), &and_constraints);

		// The degenerate batch reproduces the single-instance BitAnd columns exactly.
		let vv = table.instance_value_vec(0, constants(&c));
		let (a_ref, b_ref, c_ref) = reference_rows(&and_constraints, &vv);
		assert_eq!(witness.a(), a_ref.as_slice());
		assert_eq!(witness.b(), b_ref.as_slice());
		assert_eq!(witness.c(), c_ref.as_slice());
	}

	#[test]
	#[should_panic(expected = "constraint count must be a power of two")]
	fn build_rejects_non_power_of_two_constraint_count() {
		let c = and_circuit();

		// Fixture state: a valid batch with one instance (K = 1).
		let table = populate_table(&c, &[(1, 3, 7)]);

		// Invariant: the per-instance constraint count must be a power of two.
		//
		// Mutation: hand the builder 3 constraints.
		//
		//     3 is not a power of two → build asserts on the count before reading any row.
		//
		// The operands are empty, so the panic is the count check, never an out-of-range index.
		let three = vec![AndConstraint::default(); 3];
		let _ = BatchAndCheckWitness::build(&table, constants(&c), &three);
	}

	proptest! {
		// Invariant: every batch row equals the single-instance reference for that instance.
		//
		//     witness[instance * n_and + j]  ==  eval_operand(instance value vec, constraint j)
		//
		// This pins the batched, slice-based evaluator to the core value-vec evaluator.
		// And since each instance is satisfying, the AND relation `A & B == C` holds on every row.
		#[test]
		fn batch_rows_match_single_instance_reference(
			inputs in prop::collection::vec((any::<u64>(), any::<u64>(), any::<u64>()), 4),
		) {
			let c = and_circuit();
			let table = populate_table(&c, &inputs);

			let and_constraints = table_constraints(&c);
			let n_and = and_constraints.len();
			let witness = BatchAndCheckWitness::build(&table, constants(&c),&and_constraints);

			for instance in 0..table.n_instances() {
				let vv = table.instance_value_vec(instance, constants(&c));
				let (a_ref, b_ref, c_ref) = reference_rows(&and_constraints, &vv);

				let start = instance * n_and;
				prop_assert_eq!(&witness.a()[start..start + n_and], a_ref.as_slice());
				prop_assert_eq!(&witness.b()[start..start + n_and], b_ref.as_slice());
				prop_assert_eq!(&witness.c()[start..start + n_and], c_ref.as_slice());
			}

			// The built columns satisfy the AND constraint on every row, padding included.
			for ((a, b), c) in witness.a().iter().zip(witness.b()).zip(witness.c()) {
				prop_assert_eq!(a.0 & b.0, c.0);
			}
		}
	}

	#[test]
	fn build_spans_multiple_instance_stripes() {
		let c = and_circuit();

		// Fixture state: 2 * STRIPE_WIDTH instances, so the build spans more than one stripe.
		// The second stripe starts at a nonzero base offset.
		// That exercises the stripe index arithmetic the small-batch tests never reach.
		let n = 2 * STRIPE_WIDTH;
		let inputs: Vec<(u64, u64, u64)> = (0..n as u64)
			.map(|i| (i.wrapping_mul(0x9e37_79b9), i ^ 0xdead, i.rotate_left(7)))
			.collect();
		let table = populate_table(&c, &inputs);
		let and_constraints = table_constraints(&c);
		let n_and = and_constraints.len();
		let witness = BatchAndCheckWitness::build(&table, constants(&c), &and_constraints);

		// Every instance's block equals its independent single-instance reference.
		// This includes instances at or beyond STRIPE_WIDTH, which only the second stripe produces.
		for instance in 0..table.n_instances() {
			let vv = table.instance_value_vec(instance, constants(&c));
			let (a_ref, b_ref, c_ref) = reference_rows(&and_constraints, &vv);
			let start = instance * n_and;
			assert_eq!(&witness.a()[start..start + n_and], a_ref.as_slice());
			assert_eq!(&witness.b()[start..start + n_and], b_ref.as_slice());
			assert_eq!(&witness.c()[start..start + n_and], c_ref.as_slice());
		}
	}

	#[test]
	fn small_batch_round_trips_with_only_pinned_challenges() {
		let c = and_circuit();

		// Fixture state: K = 4 instances, n_and = 2 → 8 rows → log_total = 3.
		// The row count log equals the pinned-challenge count.
		// So every coordinate is pinned and no large-field challenge is drawn.
		let table = populate_table(&c, &[(1, 3, 7), (5, 6, 0), (9, 12, 0xFF), (0xF0, 0x0F, 1)]);
		let and_constraints = table_constraints(&c);
		let witness = BatchAndCheckWitness::build(&table, constants(&c), &and_constraints);
		let log_total = checked_log_2(witness.a().len());

		// Prover and verifier agree on the reduced claim over the batched columns.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prove_output = witness.prove::<P, _>(&mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verify_output = verify_bitand_reduction(log_total, &mut verifier_transcript).unwrap();
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");

		assert_eq!(prove_output, verify_output);
	}

	#[test]
	fn tampered_proof_is_rejected() {
		let c = and_circuit();

		// Fixture state: 16 satisfying instances → 32 rows → log_total = 5.
		let inputs: Vec<(u64, u64, u64)> = (0..16u64)
			.map(|i| (i, i.wrapping_mul(3) + 1, i ^ 0xAB))
			.collect();
		let table = populate_table(&c, &inputs);
		let and_constraints = table_constraints(&c);
		let witness = BatchAndCheckWitness::build(&table, constants(&c), &and_constraints);
		let log_total = checked_log_2(witness.a().len());

		// Produce a faithful proof.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let _ = witness.prove::<P, _>(&mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Mutation: flip a bit in the prover's first message, the univariate round evaluations.
		proof[0] ^= 1;

		// The verifier redraws a different univariate challenge from the tampered message.
		// The final consistency check then no longer holds.
		// So verification fails.
		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let err = verify_bitand_reduction(log_total, &mut verifier_transcript).unwrap_err();

		// The closing check is `A_eval * B_eval - C_eval == sumcheck_eval`.
		// The tampered message moves the claim, so this equality no longer holds.
		// The channel rejects the non-zero assertion, the protocol's terminal check.
		assert_matches!(err, VerifierError::Channel(ChannelError::InvalidAssert));
	}

	proptest! {
		#[test]
		fn reduction_round_trips_and_claims_match_columns(
			inputs in prop::collection::vec((any::<u64>(), any::<u64>(), any::<u64>()), 16),
		) {
			// Invariant: the reduction round-trips, and its output claims equal the true column MLEs.
			//
			//     prover  : zerocheck A*B - C over K*n_and rows, claim (a_eval, b_eval, c_eval) at p
			//     verifier: replay, reach the same claim
			//     check   : fold each column at z and evaluate at p == the claimed eval
			//
			// The third check pins the reduction to the actual columns.
			// Internal transcript agreement alone would not catch a wrong claim point.

			let c = and_circuit();
			let table = populate_table(&c, &inputs);
			let and_constraints = table_constraints(&c);
			let witness = BatchAndCheckWitness::build(&table, constants(&c),&and_constraints);

			// Keep the columns so the claimed evals can be checked against them.
			let a_cols = witness.a().to_vec();
			let b_cols = witness.b().to_vec();
			let c_cols = witness.c().to_vec();
			let log_total = checked_log_2(witness.a().len());

			let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
			let prove_output = witness.prove::<P, _>(&mut prover_transcript);

			let mut verifier_transcript = prover_transcript.into_verifier();
			let verify_output =
				verify_bitand_reduction(log_total, &mut verifier_transcript).unwrap();
			verifier_transcript.finalize().expect("no trailing proof data");

			// Both sides reach the same reduced claim.
			prop_assert_eq!(&prove_output, &verify_output);

			// Each claimed eval is the column's multilinear, folded at z and evaluated at the point.
			let AndCheckOutput {
				a_eval,
				b_eval,
				c_eval,
				z_challenge,
				eval_point,
			} = verify_output;
			prop_assert_eq!(fold_eval_column(&a_cols, z_challenge, &eval_point), a_eval);
			prop_assert_eq!(fold_eval_column(&b_cols, z_challenge, &eval_point), b_eval);
			prop_assert_eq!(fold_eval_column(&c_cols, z_challenge, &eval_point), c_eval);
		}
	}
}
