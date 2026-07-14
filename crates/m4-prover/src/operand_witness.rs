// Copyright 2025 Irreducible Inc.

//! The batched operand-column witnesses built from a populated batch value table.
//!
//! Every AND and MUL constraint is projected to its fixed-arity operand columns (`[A, B, C]` for
//! AND, `[A, B, HI, LO]` for MUL), one column per operand, stacked over every instance in the
//! batch. [`build_operation_witness`] is the shared arity-generic core: it projects one operand
//! per constraint into one column. [`BatchAndCheckWitness`] builds the three AND columns and
//! drives the AND-check zerocheck; [`build_intmul_witness`] builds the four IntMul columns, not
//! yet consumed downstream.

use std::{iter, mem::MaybeUninit, ptr};

use binius_core::{
	ValueIndex,
	constraint_system::{AndConstraint, MulConstraint, Operand, ShiftVariant, ShiftedValueIndex},
	word::Word,
};
use binius_field::{AESTowerField8b as B8, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::BinarySubspace;
use binius_prover::and_reduction::prover::OblongZerocheckProver;
use binius_utils::{
	checked_arithmetics::{checked_log_2, log2_strict_usize},
	rayon::{self, prelude::*},
};
use binius_verifier::{
	config::{B128, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES},
	protocols::bitand::AndCheckOutput,
};

use crate::ValueTable;

/// The operand columns of the BitAnd check for a whole batch of instances.
///
/// The BitAnd check works on three columns of words, one row per AND constraint.
/// It enforces the bitwise relation `A & B == C` on every row.
///
/// This holds those three columns for a batch of `K = 2^log_instances` instances at once.
/// The rows are stacked in constraint-major order, with the constraint index on the high
/// coordinates:
///
/// ```text
///      constraint 0         constraint 1          constraint n_and-1
///   A: [ a_0 .. a_{K-1} ][ a_0 .. a_{K-1} ] ... [ a_0 .. a_{K-1} ]
///   B: [ b_0 .. b_{K-1} ][ b_0 .. b_{K-1} ] ... [ b_0 .. b_{K-1} ]
///   C: [ c_0 .. c_{K-1} ][ c_0 .. c_{K-1} ] ... [ c_0 .. c_{K-1} ]
///       \____ K ____/
/// ```
///
/// The row index splits cleanly into the constraint and the per-constraint instance:
///
/// ```text
/// row = local_constraint * K + instance
/// ```
///
/// - The high bits select the constraint.
/// - The low `log_instances` bits select the instance within that constraint.
/// - The reduction reads each column as a multilinear over `log_instances + log(n_and)` bits.
///
/// `n_and` is a power of two when the constraints come from a prepared constraint system.
/// So the batch forms a clean hypercube whose low coordinates are the instance index.
#[derive(Clone, Debug)]
pub struct BatchAndCheckWitness {
	/// Operand `A` of every constraint of every instance, constraint-major.
	a: Vec<Word>,
	/// Operand `B` of every constraint of every instance, constraint-major.
	b: Vec<Word>,
	/// Operand `C` of every constraint of every instance, constraint-major.
	c: Vec<Word>,
}

impl BatchAndCheckWitness {
	/// Builds the batched BitAnd witness from a populated wire-major batch table.
	///
	/// Every AND constraint contributes one row per instance to each of the three operand
	/// columns `A`, `B`, `C`, laid out constraint-major. This delegates to the arity-generic
	/// `build_operation_witness`, projecting each constraint to its three operands.
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
		let a_operand_iter = and_constraints.par_iter().map(|constraint| &constraint.a);
		let b_operand_iter = and_constraints.par_iter().map(|constraint| &constraint.b);
		let (a, b) = rayon::join(
			|| build_operation_witness(table, constants, a_operand_iter),
			|| build_operation_witness(table, constants, b_operand_iter),
		);

		// Instead of constructing c from the original word Vec, it's cheaper to compute it from the
		// constraint relation.
		let c = (a.as_slice(), b.as_slice())
			.into_par_iter()
			.map(|(&a_i, &b_i)| a_i & b_i)
			.collect();

		Self { a, b, c }
	}

	/// Operand `A` column, `K * n_and` rows in constraint-major order.
	pub fn a(&self) -> &[Word] {
		&self.a
	}

	/// Operand `B` column, `K * n_and` rows in constraint-major order.
	pub fn b(&self) -> &[Word] {
		&self.b
	}

	/// Operand `C` column, `K * n_and` rows in constraint-major order.
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
	///     row = local_constraint * K + instance
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
	/// - `prover_message_domain`: the univariate-skip domain, one dimension above the 64-bit word.
	///   The caller passes it so it matches the shift reduction's domain by construction.
	/// - `channel`: the prover channel that records messages and draws Fiat-Shamir challenges.
	///
	/// # Returns
	///
	/// The reduced claim, holding:
	/// - The claimed `A`, `B`, `C` evaluations.
	/// - The univariate (bit-index) challenge.
	/// - The multilinear evaluation point reached by the sumcheck.
	pub fn prove<P, Channel>(
		self,
		prover_message_domain: &BinarySubspace<B8>,
		channel: &mut Channel,
	) -> AndCheckOutput<B128>
	where
		P: PackedField<Scalar = B128>,
		Channel: IPProverChannel<B128>,
	{
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

/// Builds the operand-column witness of a batched fixed-arity operation over every instance.
///
/// This is the arity-generic core shared by every per-operation witness: BitAnd projects each
/// constraint to its three operands `[A, B, C]`; IntMul projects to its four `[A, B, HI, LO]`.
/// Every constraint contributes one row per instance to each operand column, laid out
/// constraint-major exactly as [`BatchAndCheckWitness`] documents:
///
/// ```text
/// row = local_constraint * n_instances + instance
/// ```
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
/// So a term streams that row, XOR-ing its shifted words into the column, which lands exactly on
/// one constraint's contiguous instance block, matching the constraint-major output layout.
///
/// # Arguments
///
/// - `table`: the wire-major batch witness holding every instance's hidden words.
/// - `constants`: the circuit's constant words, shared by every instance.
/// - `operands`: one operand per constraint, in order; the returned column follows that same order.
///   Pass operands from a prepared constraint system, so their count is a power of two.
///
/// # Panics
///
/// Panics if the constraint count is not a power of two.
pub fn build_operation_witness<'a>(
	table: &ValueTable,
	constants: &[Word],
	operands: impl IndexedParallelIterator<Item = &'a Operand>,
) -> Vec<Word> {
	// Rows per instance, and total rows across the batch.
	let log_constraints = log2_strict_usize(operands.len());
	let log_instances = table.log_instances();

	let table_words = table.as_words();
	let witness_offset = ValueIndex(table.layout().offset_witness as u32);

	let mut out = Vec::<Word>::with_capacity(1 << (log_instances + log_constraints));

	operands
		.zip(out.spare_capacity_mut().par_chunks_mut(1 << log_instances))
		.for_each(|(operand, out_chunk)| {
			let mut shifted_indices_iter = operand.iter();

			if let Some(shifted_index_0) = shifted_indices_iter.next() {
				write_shifted_values(
					out_chunk,
					shifted_index_0,
					constants,
					table_words,
					witness_offset,
					log_instances,
				);

				// out_chunk is fully initialized in the block above.
				let out_chunk = unsafe { out_chunk.assume_init_mut() };

				for shifted_index in shifted_indices_iter {
					accum_shifted_values(
						out_chunk,
						shifted_index,
						constants,
						table_words,
						witness_offset,
						log_instances,
					);
				}
			} else {
				// When the operand is empty, write 0 words to the stripe.
				// Safety: out_chunk is a valid slice of Word, writing zero bytes writes zero words.
				unsafe { ptr::write_bytes(out_chunk.as_mut_ptr(), 0, out_chunk.len()) };
			}
		});

	// The stripes partition `[0, total)` and each zeroed its whole range.
	// So all `total` elements of every column are initialized.
	//
	// SAFETY: every element in `0..total` of every column was written above.
	unsafe { out.set_len(1 << (log_instances + log_constraints)) };
	out
}

/// Writes `shift(src[i])` into `out_chunk[i]` for every `i`, initializing each cell.
///
/// `shift` is a concrete per-variant closure bound once by the caller's match on
/// [`ShiftVariant`], so each call site monomorphizes to a loop with the shift operation inlined,
/// rather than branching on the variant every iteration.
#[inline]
fn write_shifted_words(
	out_chunk: &mut [MaybeUninit<Word>],
	src: &[Word],
	shift: impl Fn(Word) -> Word,
) {
	for (out_i, &src_i) in iter::zip(out_chunk, src) {
		out_i.write(shift(src_i));
	}
}

/// Writes one shifted value into every element of `out_chunk`, initializing it.
///
/// This is the first term of an operand's accumulation: it initializes the cell rather than
/// XOR-ing into it, so the caller need not zero `out_chunk` first. Use [`accum_shifted_values`]
/// for every subsequent term.
///
/// # Arguments
///
/// - `out_chunk`: the uninitialized output cells to write, one per instance in this stripe.
/// - `shifted_index`: the shifted value index to write.
/// - `constants`: the circuit's constant words, shared by every instance.
/// - `table_words`: the wire-major batch witness's hidden words.
/// - `witness_offset`: the value index at which hidden words begin; public words lie below it.
/// - `log_instances`: the base-2 logarithm of the instance count, i.e. one hidden row's stride.
fn write_shifted_values(
	out_chunk: &mut [MaybeUninit<Word>],
	shifted_index: &ShiftedValueIndex,
	constants: &[Word],
	table_words: &[Word],
	witness_offset: ValueIndex,
	log_instances: usize,
) {
	let ShiftedValueIndex {
		value_index,
		shift_variant,
		amount: shift_amount,
	} = *shifted_index;
	let amount = shift_amount as u32;

	if value_index < witness_offset {
		let constant = constants[value_index.0 as usize];
		let shifted_constant = shift_variant.apply(constant, shift_amount as usize);
		for out_i in &mut *out_chunk {
			out_i.write(shifted_constant);
		}
	} else {
		let index = value_index.0 as usize - witness_offset.0 as usize;
		let src = &table_words[(index << log_instances)..((index + 1) << log_instances)];

		if amount == 0 {
			// Safety:
			// * out_chunk.len() == src.len()
			// * MaybeUninit<Word> has the same memory repr as Word
			unsafe {
				ptr::copy_nonoverlapping(
					src.as_ptr(),
					out_chunk.as_mut_ptr() as *mut Word,
					out_chunk.len(),
				)
			};
		} else {
			match shift_variant {
				ShiftVariant::Sll => write_shifted_words(out_chunk, src, |w| w << amount),
				ShiftVariant::Slr => write_shifted_words(out_chunk, src, |w| w >> amount),
				ShiftVariant::Sar => write_shifted_words(out_chunk, src, |w| w.sar(amount)),
				ShiftVariant::Rotr => write_shifted_words(out_chunk, src, |w| w.rotr(amount)),
				ShiftVariant::Sll32 => write_shifted_words(out_chunk, src, |w| w.sll32(amount)),
				ShiftVariant::Srl32 => write_shifted_words(out_chunk, src, |w| w.srl32(amount)),
				ShiftVariant::Sra32 => write_shifted_words(out_chunk, src, |w| w.sra32(amount)),
				ShiftVariant::Rotr32 => write_shifted_words(out_chunk, src, |w| w.rotr32(amount)),
			}
		}
	}
}

/// XORs `shift(src[i])` into `out_chunk[i]` for every `i`.
///
/// `shift` is a concrete per-variant closure bound once by the caller's match on
/// [`ShiftVariant`], so each call site monomorphizes to a loop with the shift operation inlined,
/// rather than branching on the variant every iteration.
#[inline]
fn xor_shifted_words(out_chunk: &mut [Word], src: &[Word], shift: impl Fn(Word) -> Word) {
	for (out_i, &src_i) in iter::zip(out_chunk, src) {
		*out_i = *out_i ^ shift(src_i);
	}
}

/// XORs one shifted value into every element of `out_chunk`.
///
/// Use this for every term of an operand's accumulation after the first; see
/// [`write_shifted_values`] for the first term, which initializes `out_chunk` instead.
///
/// # Arguments
///
/// - `out_chunk`: the initialized output cells to accumulate into, one per instance in this stripe.
/// - `shifted_index`: the shifted value index to accumulate.
/// - `constants`: the circuit's constant words, shared by every instance.
/// - `table_words`: the wire-major batch witness's hidden words.
/// - `witness_offset`: the value index at which hidden words begin; public words lie below it.
/// - `log_instances`: the base-2 logarithm of the instance count, i.e. one hidden row's stride.
fn accum_shifted_values(
	out_chunk: &mut [Word],
	shifted_index: &ShiftedValueIndex,
	constants: &[Word],
	table_words: &[Word],
	witness_offset: ValueIndex,
	log_instances: usize,
) {
	let ShiftedValueIndex {
		value_index,
		shift_variant,
		amount: shift_amount,
	} = *shifted_index;
	let amount = shift_amount as u32;

	if value_index < witness_offset {
		let constant = constants[value_index.0 as usize];
		let shifted_constant = shift_variant.apply(constant, shift_amount as usize);
		for out_i in &mut *out_chunk {
			*out_i = *out_i ^ shifted_constant;
		}
	} else {
		let index = value_index.0 as usize - witness_offset.0 as usize;
		let src = &table_words[(index << log_instances)..((index + 1) << log_instances)];

		if amount == 0 {
			for (out_i, &src_i) in iter::zip(&mut *out_chunk, src) {
				*out_i = *out_i ^ src_i;
			}
		} else {
			match shift_variant {
				ShiftVariant::Sll => xor_shifted_words(out_chunk, src, |w| w << amount),
				ShiftVariant::Slr => xor_shifted_words(out_chunk, src, |w| w >> amount),
				ShiftVariant::Sar => xor_shifted_words(out_chunk, src, |w| w.sar(amount)),
				ShiftVariant::Rotr => xor_shifted_words(out_chunk, src, |w| w.rotr(amount)),
				ShiftVariant::Sll32 => xor_shifted_words(out_chunk, src, |w| w.sll32(amount)),
				ShiftVariant::Srl32 => xor_shifted_words(out_chunk, src, |w| w.srl32(amount)),
				ShiftVariant::Sra32 => xor_shifted_words(out_chunk, src, |w| w.sra32(amount)),
				ShiftVariant::Rotr32 => xor_shifted_words(out_chunk, src, |w| w.rotr32(amount)),
			}
		}
	}
}

/// Builds the batched IntMul operand witness from a populated wire-major batch table.
///
/// Every MUL constraint contributes one row per instance to each of the four operand columns
/// `A`, `B`, `HI`, `LO`, laid out constraint-major. This delegates to the arity-generic
/// `build_operation_witness`, projecting each constraint to its four operands.
///
/// # Arguments
///
/// - `table`: the wire-major batch witness holding every instance's hidden words.
/// - `constants`: the circuit's constant words, shared by every instance.
/// - `mul_constraints`: the per-instance MUL constraints, shared by every instance.
///
/// Pass constraints from a prepared constraint system, so their count is a power of two.
///
/// # Panics
///
/// Panics if the constraint count is not a power of two.
pub fn build_intmul_witness(
	table: &ValueTable,
	constants: &[Word],
	mul_constraints: &[MulConstraint],
) -> [Vec<Word>; 4] {
	let a_operand_iter = mul_constraints.par_iter().map(|constraint| &constraint.a);
	let b_operand_iter = mul_constraints.par_iter().map(|constraint| &constraint.b);
	let hi_operand_iter = mul_constraints.par_iter().map(|constraint| &constraint.hi);
	let lo_operand_iter = mul_constraints.par_iter().map(|constraint| &constraint.lo);
	let ((a, b), (hi, lo)) = rayon::join(
		|| {
			rayon::join(
				|| build_operation_witness(table, constants, a_operand_iter),
				|| build_operation_witness(table, constants, b_operand_iter),
			)
		},
		|| {
			rayon::join(
				|| build_operation_witness(table, constants, hi_operand_iter),
				|| build_operation_witness(table, constants, lo_operand_iter),
			)
		},
	);
	[a, b, hi, lo]
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

	// The univariate-skip domain the AND-check runs over: one dimension above the 64-bit word.
	fn message_domain() -> BinarySubspace<B8> {
		BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1)
	}

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

	// The reference columns across the whole batch, transposed to the batch witness's
	// constraint-major layout: `columns.a[j * n_instances + instance]` is constraint `j`'s
	// operand `A` evaluated on `instance`.
	fn reference_columns(
		table: &ValueTable,
		constants: &[Word],
		and_constraints: &[AndConstraint],
	) -> (Vec<Word>, Vec<Word>, Vec<Word>) {
		let n_and = and_constraints.len();
		let n_instances = table.n_instances();
		let mut a = vec![Word::ZERO; n_and * n_instances];
		let mut b = vec![Word::ZERO; n_and * n_instances];
		let mut c = vec![Word::ZERO; n_and * n_instances];
		for instance in 0..n_instances {
			let vv = table.instance_value_vec(instance, constants);
			let (a_ref, b_ref, c_ref) = reference_rows(and_constraints, &vv);
			for j in 0..n_and {
				a[j * n_instances + instance] = a_ref[j];
				b[j * n_instances + instance] = b_ref[j];
				c[j * n_instances + instance] = c_ref[j];
			}
		}
		(a, b, c)
	}

	#[test]
	fn columns_are_constraint_major_with_the_expected_shape() {
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

		// Invariant: row `j * n_instances + instance` is constraint `j` of that instance.
		let (a_ref, b_ref, c_ref) = reference_columns(&table, constants(&c), and_constraints);
		assert_eq!(witness.a(), a_ref.as_slice());
		assert_eq!(witness.b(), b_ref.as_slice());
		assert_eq!(witness.c(), c_ref.as_slice());
	}

	// A circuit computing one unsigned 64×64→128 product, with both result words committed.
	//
	//     inputs : x, y   (witness)
	//     gate   : (hi, lo) = imul(x, y)   → 1 MUL constraint (+ 1 AND security check)
	//
	// `force_commit` makes `hi` and `lo` hidden words, so the MUL operands read them from the
	// table.
	struct MulCircuit {
		circuit: Circuit,
		x: Wire,
		y: Wire,
	}

	fn mul_circuit() -> MulCircuit {
		let builder = CircuitBuilder::new();
		let x = builder.add_witness();
		let y = builder.add_witness();
		let (hi, lo) = builder.imul(x, y);
		builder.force_commit(hi);
		builder.force_commit(lo);
		MulCircuit {
			circuit: builder.build(),
			x,
			y,
		}
	}

	// Populate one instance per input pair; the circuit derives the two product words.
	fn populate_mul_table(c: &MulCircuit, inputs: &[(u64, u64)]) -> ValueTable {
		let log_instances = inputs.len().ilog2() as usize;
		ValueTable::populate(&c.circuit, log_instances, |i, filler| {
			let (x, y) = inputs[i];
			filler[c.x] = Word(x);
			filler[c.y] = Word(y);
		})
		.unwrap()
	}

	// The arity-4 IntMul witness lays out its four operand columns [A, B, HI, LO] constraint-major,
	// each row matching the single-instance operand evaluator. This is the batched IntMul witness
	// the reduction will consume once the IntMul check is wired in.
	#[test]
	fn intmul_operand_columns_match_the_single_instance_reference() {
		let c = mul_circuit();
		let constants = &c.circuit.constraint_system().constants;

		// Fixture state: 2^2 = 4 instances with distinct inputs, some carrying into `hi`.
		let inputs = [
			(1, 1),
			(0xFFFF_FFFF, 0xFFFF_FFFF),
			(0x1_0000_0000, 0x1_0000_0000),
			(0xDEAD_BEEF_CAFE_BABE, 0x0123_4567_89AB_CDEF),
		];
		let table = populate_mul_table(&c, &inputs);

		// The prepared per-instance MUL constraints, padded to a power of two.
		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		let mul_constraints = &cs.mul_constraints;
		assert!(!mul_constraints.is_empty(), "the circuit must emit a MUL constraint");

		let [a, b, hi, lo] = build_intmul_witness(&table, constants, mul_constraints);

		// Shape: K * n_mul rows, with K = 4.
		let n_mul = mul_constraints.len();
		for col in [&a, &b, &hi, &lo] {
			assert_eq!(col.len(), 4 * n_mul);
		}

		// Invariant: row `j * n_instances + instance` is constraint `j` of that instance, and each
		// of the four columns equals the single-instance reference for its inputs.
		let n_instances = table.n_instances();
		for instance in 0..n_instances {
			let vv = table.instance_value_vec(instance, constants);
			for (j, con) in mul_constraints.iter().enumerate() {
				let idx = j * n_instances + instance;
				assert_eq!(a[idx], vv.eval_operand(&con.a));
				assert_eq!(b[idx], vv.eval_operand(&con.b));
				assert_eq!(hi[idx], vv.eval_operand(&con.hi));
				assert_eq!(lo[idx], vv.eval_operand(&con.lo));
			}
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
	#[should_panic(expected = "Not a power of two")]
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
		//     witness[j * n_instances + instance]  ==  eval_operand(instance value vec, constraint j)
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
			let witness = BatchAndCheckWitness::build(&table, constants(&c),&and_constraints);

			let (a_ref, b_ref, c_ref) = reference_columns(&table, constants(&c), &and_constraints);
			prop_assert_eq!(witness.a(), a_ref.as_slice());
			prop_assert_eq!(witness.b(), b_ref.as_slice());
			prop_assert_eq!(witness.c(), c_ref.as_slice());

			// The built columns satisfy the AND constraint on every row, padding included.
			for ((a, b), c) in witness.a().iter().zip(witness.b()).zip(witness.c()) {
				prop_assert_eq!(a.0 & b.0, c.0);
			}
		}
	}

	#[test]
	fn build_spans_multiple_instance_stripes() {
		let c = and_circuit();

		// Fixture state: enough instances that the per-constraint parallel chunks span multiple
		// rayon tasks, exercising the multi-chunk path the small-batch tests never reach.
		const LOG_INSTANCES: usize = 9;
		let n = 1 << LOG_INSTANCES;
		let inputs: Vec<(u64, u64, u64)> = (0..n as u64)
			.map(|i| (i.wrapping_mul(0x9e37_79b9), i ^ 0xdead, i.rotate_left(7)))
			.collect();
		let table = populate_table(&c, &inputs);
		let and_constraints = table_constraints(&c);
		let witness = BatchAndCheckWitness::build(&table, constants(&c), &and_constraints);

		// Every instance's contribution equals its independent single-instance reference.
		// This includes instances at or beyond STRIPE_WIDTH, which only the second stripe produces.
		let (a_ref, b_ref, c_ref) = reference_columns(&table, constants(&c), &and_constraints);
		assert_eq!(witness.a(), a_ref.as_slice());
		assert_eq!(witness.b(), b_ref.as_slice());
		assert_eq!(witness.c(), c_ref.as_slice());
	}

	#[test]
	fn build_matches_reference_with_shifted_operands() {
		let c = and_circuit();

		// Fixture state: 4 instances with distinct inputs.
		let table =
			populate_table(&c, &[(1, 3, 7), (0xF0F0, 0x0FF0, 0xAA), (5, 6, 9), (0xFFFF, 1, 2)]);

		// Hand-craft constraints that carry real shifts on hidden operands.
		// The circuit compiler emits unshifted operands here, so the shifted branch of the
		// accumulator would otherwise go untested by this module.
		//
		// The `c` operand is deliberately unrelated to `a & b`: `BatchAndCheckWitness::build`
		// derives `c` from `a` and `b` rather than evaluating the constraint's `c` operand, so this
		// fixture need not satisfy the AND relation to exercise `a` and `b`'s shift handling.
		let x = c.circuit.witness_index(c.x);
		let y = c.circuit.witness_index(c.y);
		let z = c.circuit.witness_index(c.z);
		let and_constraints = vec![
			AndConstraint {
				// A mixes a shifted and an unshifted term, exercising both accumulator paths.
				a: vec![ShiftedValueIndex::sll(x, 3), ShiftedValueIndex::plain(y)],
				b: vec![ShiftedValueIndex::srl(y, 5)],
				c: vec![ShiftedValueIndex::sar(z, 7)],
			},
			// An empty operand set: its column must stay at the zeroed initial value.
			AndConstraint::default(),
		];

		// Sanity: at least one operand term really is shifted, so the else-branch runs.
		let shifted = and_constraints
			.iter()
			.flat_map(|con| [&con.a, &con.b])
			.any(|op| {
				op.iter()
					.any(|sv| sv.shift_variant != ShiftVariant::Sll || sv.amount != 0)
			});
		assert!(shifted, "fixture must contain a shifted operand");

		let witness = BatchAndCheckWitness::build(&table, constants(&c), &and_constraints);

		// `a` and `b` equal the shift-aware value-vec reference for the same constraints.
		let (a_ref, b_ref, _) = reference_columns(&table, constants(&c), &and_constraints);
		assert_eq!(witness.a(), a_ref.as_slice());
		assert_eq!(witness.b(), b_ref.as_slice());

		// `c` is defined as `a & b`, not evaluated from the constraint's `c` operand.
		for ((a, b), c) in witness.a().iter().zip(witness.b()).zip(witness.c()) {
			assert_eq!(a.0 & b.0, c.0);
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
		let prove_output = witness.prove::<P, _>(&message_domain(), &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verify_output =
			verify_bitand_reduction(log_total, &message_domain(), &mut verifier_transcript)
				.unwrap();
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
		let _ = witness.prove::<P, _>(&message_domain(), &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Mutation: flip a bit in the prover's first message, the univariate round evaluations.
		proof[0] ^= 1;

		// The verifier redraws a different univariate challenge from the tampered message.
		// The final consistency check then no longer holds.
		// So verification fails.
		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let err = verify_bitand_reduction(log_total, &message_domain(), &mut verifier_transcript)
			.unwrap_err();

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
			let prove_output = witness.prove::<P, _>(&message_domain(), &mut prover_transcript);

			let mut verifier_transcript = prover_transcript.into_verifier();
			let verify_output =
				verify_bitand_reduction(log_total, &message_domain(), &mut verifier_transcript).unwrap();
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
