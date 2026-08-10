// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The batched operand columns of one operation, built from a populated batch value table.

use std::{iter, mem::MaybeUninit, ptr};

use binius_compute::{Allocator, VecLike};
use binius_core::{
	ValueSegment,
	constraint_system::{Operand, Shift, ShiftVariant, ShiftedValueIndex},
	word::Word,
};
use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, FieldVec};
use binius_prover::fold_word::fold_words;
use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	rayon::{prelude::*, task_size::IndexedParallelIteratorExt},
};
use binius_verifier::config::B128;

use crate::ValueTable;

/// The operand columns of one batched fixed-arity operation.
///
/// # Overview
///
/// Each operation projects every constraint to a fixed array of operands:
///
/// ```text
/// AND     [A, B]
/// IMUL    [A, B, LO, HI]
/// BMUL    [A_LO, A_HI, B_LO, B_HI, C_LO, C_HI]
/// ```
///
/// One column holds one operand position, evaluated on every instance of the batch.
///
/// # Layout
///
/// Every column is constraint-major, so one constraint's instances are contiguous:
///
/// ```text
/// row = local_constraint * n_instances + instance
/// ```
///
/// The constraint count need not be a power of two.
///
/// - The columns are zero-padded up to the next one.
/// - A zero row satisfies every operation, so the padding needs no further handling.
pub struct OperandColumns<A: Allocator, const ARITY: usize> {
	/// One column per operand position, in the constraint type's storage order.
	columns: [A::Vec<Word>; ARITY],
	/// The base-2 logarithm of the instance count, which is the stride of the constraint axis.
	log_instances: usize,
}

impl<A: Allocator, const ARITY: usize> OperandColumns<A, ARITY> {
	/// Builds the leading columns of an operation, one per operand position.
	///
	/// # Overview
	///
	/// Every constraint is projected to its operand array.
	///
	/// One column is then built per operand position, all of them in parallel.
	///
	/// Fewer columns than the constraint arity may be built, leaving trailing operands unread.
	///
	/// The AND check uses that to skip its third operand, which its reduction derives instead.
	///
	/// # Arguments
	///
	/// - `table`: the wire-major batch witness holding every instance's hidden words.
	/// - `constants`: the circuit's constant words, shared by every instance.
	/// - `constraints`: the per-instance constraints, shared by every instance.
	/// - `alloc`: the allocator backing the columns.
	///
	/// # Panics
	///
	/// Panics if more columns are requested than the constraints carry operands.
	pub fn build<C, const CONSTRAINT_ARITY: usize>(
		table: &ValueTable,
		constants: &[Word],
		constraints: &[C],
		alloc: &A,
	) -> Self
	where
		C: AsRef<[Operand; CONSTRAINT_ARITY]> + Sync,
	{
		const {
			assert!(
				ARITY <= CONSTRAINT_ARITY,
				"cannot build more columns than the constraint arity"
			);
		}

		// A term reads from the same banks for every column, so resolve them once.
		let values = ValueWords::new(table, constants);

		// The operand positions are independent, so build one column per position in parallel.
		let columns = (0..ARITY)
			.into_par_iter()
			.map(|op_idx| {
				// Project every constraint to the operand sitting at this position.
				values.build_column(
					constraints
						.par_iter()
						.map(move |constraint| &constraint.as_ref()[op_idx]),
					alloc,
				)
			})
			.collect::<Vec<_>>()
			.try_into()
			.unwrap_or_else(|_| unreachable!("the source iterator has ARITY elements"));

		Self {
			columns,
			log_instances: table.log_instances(),
		}
	}

	/// The columns as plain word slices, dropping whichever allocator produced them.
	pub fn as_slices(&self) -> [&[Word]; ARITY] {
		self.columns.each_ref().map(|column| &**column)
	}

	/// Folds one column over its bit and constraint axes, leaving the instance axis.
	///
	/// # Overview
	///
	/// A column is constraint-major, so collapsing its two other axes leaves one per instance:
	///
	/// ```text
	/// M[rho] = sum_{local, j} lagrange[j] * r_x_tensor[local] * bit_j(column[local * K + rho])
	/// ```
	///
	/// - The Lagrange weights fold each word over its bit axis at the shared univariate challenge.
	/// - The constraint tensor folds the constraint axis, and is shared across the operands.
	///
	/// Evaluating the result at the operation's instance point gives that operand's oblong claim.
	///
	/// A sumcheck can then transport that claim to a point shared with the other operations.
	///
	/// # Panics
	///
	/// Panics if the constraint tensor does not cover the column's padded constraint axis.
	pub fn rho_multilinear<P>(
		&self,
		operand: usize,
		lagrange: &[B128],
		r_x_tensor: &[B128],
		alloc: &A,
	) -> FieldVec<P, A>
	where
		P: PackedField<Scalar = B128>,
	{
		let column = &self.columns[operand];

		// Invariant: the tensor carries one weight per padded constraint row.
		assert_eq!(
			r_x_tensor.len() << self.log_instances,
			column.len(),
			"the constraint tensor must cover the padded constraint axis"
		);

		// Fold each word's bits at the univariate challenge, giving one scalar per row.
		// Scalars keep the row indexing flat for the constraint fold below.
		let folded_rows = fold_words::<B128, B128, _>(alloc, column, lagrange);
		let folded_rows = folded_rows.as_ref();

		// One packed element per parallel task, each lane holding one instance.
		// Lanes past the instance count are the multilinear's zero padding.
		let n_instances = 1usize << self.log_instances;
		let packed_len = 1usize << self.log_instances.saturating_sub(P::LOG_WIDTH);
		let mut packed = alloc.alloc::<P>(packed_len);
		packed
			.spare_capacity_mut()
			.par_iter_mut()
			.enumerate()
			.for_each(|(packed_index, slot)| {
				slot.write(P::from_scalars((0..P::WIDTH).map(|lane| {
					let instance = (packed_index << P::LOG_WIDTH) | lane;
					if instance < n_instances {
						// Constraint `local` of this instance sits at row `local * K + rho`.
						// So the constraint axis is the strided one.
						r_x_tensor
							.iter()
							.enumerate()
							.map(|(local, &weight)| {
								weight * folded_rows[local * n_instances + instance]
							})
							.sum()
					} else {
						B128::ZERO
					}
				})));
			});
		// SAFETY: the parallel loop writes every packed slot exactly once.
		unsafe { packed.set_len(packed_len) };

		FieldBuffer::new(self.log_instances, packed)
	}
}

impl<A: Allocator> OperandColumns<A, 2> {
	/// Appends the AND check's third column, derived word-by-word from the first two.
	///
	/// # Overview
	///
	/// The check itself stores only its two input operands.
	///
	/// On a satisfying witness the third is their conjunction.
	/// Deriving it is therefore cheaper than building it from the constraints.
	///
	/// Only the re-randomization reads it, so it is materialized on that path alone.
	///
	/// # Performance
	///
	/// One conjunction per row is a single instruction.
	///
	/// The cost is streaming three word columns, two read and one written.
	pub fn with_derived_and(self, alloc: &A) -> OperandColumns<A, 3> {
		let Self {
			columns: [a, b],
			log_instances,
		} = self;

		// The parallel zip stops at the shortest of its three sides.
		// Equal input lengths are therefore what make the output fully written.
		debug_assert_eq!(a.len(), b.len());
		let n_rows = a.len();

		let mut c = alloc.alloc::<Word>(n_rows);
		(&a[..], &b[..], c.spare_capacity_mut())
			.into_par_iter()
			.with_min_task_bytes::<[Word; 3]>()
			.for_each(|(&a_i, &b_i, out)| {
				out.write(a_i & b_i);
			});
		// SAFETY: the loop above writes each of the `n_rows` entries exactly once.
		unsafe { c.set_len(n_rows) };

		OperandColumns {
			columns: [a, b, c],
			log_instances,
		}
	}
}

/// The batch's value words, addressed the way an operand term addresses them.
///
/// # Overview
///
/// A value index splits at the inout offset:
///
/// - below it, a constant: the same word in every instance;
/// - at or above it, a hidden word: one per instance, read from the table.
///
/// Both banks are carried here, together with the inout count that positions the private rows
/// behind the inout ones.
///
/// So a term resolves without its caller re-deriving that split at each use.
#[derive(Clone, Copy)]
struct ValueWords<'a> {
	/// The circuit's constant words, shared by every instance.
	constants: &'a [Word],
	/// Every instance's hidden words, wire-major: the inout rows, then the private ones.
	hidden: &'a [Word],
	/// The number of inout values, which is where the private rows start.
	n_inout: usize,
	/// The base-2 logarithm of the instance count, which is one hidden row's stride.
	log_instances: usize,
}

/// The words one operand term contributes, one per instance of the batch.
enum TermWords<'a> {
	/// One already-shifted constant, repeated across every instance.
	Splat(Word),
	/// One hidden row, taken as it is.
	Plain(&'a [Word]),
	/// One hidden row, shifted as it is streamed.
	Shifted {
		/// The unshifted row, one word per instance.
		row: &'a [Word],
		/// Which shift to apply to each word.
		variant: ShiftVariant,
		/// How far to shift, always non-zero.
		amount: u32,
	},
}

impl<'a> ValueWords<'a> {
	/// Borrows a populated table and its circuit's constants as one addressable value space.
	fn new(table: &'a ValueTable, constants: &'a [Word]) -> Self {
		Self {
			constants,
			hidden: table.as_words(),
			n_inout: table.layout().n_inout,
			log_instances: table.log_instances(),
		}
	}

	/// Builds one operand column of a batched operation, over every instance.
	///
	/// # Overview
	///
	/// Every constraint contributes one row per instance, laid out constraint-major:
	///
	/// ```text
	/// row = local_constraint * n_instances + instance
	/// ```
	///
	/// So one constraint's instances land on one contiguous stripe of the column.
	///
	/// The stripes are independent, so they are evaluated in parallel, one task each.
	///
	/// # Arguments
	///
	/// - `operands`: one operand per constraint, in the order the column follows.
	/// - `alloc`: the allocator backing the column.
	///
	/// # Returns
	///
	/// The column, its constraint axis zero-padded up to a power of two.
	fn build_column<A: Allocator>(
		&self,
		operands: impl IndexedParallelIterator<Item = &'a Operand>,
		alloc: &A,
	) -> A::Vec<Word> {
		// The constraint axis is rounded up to a power of two.
		// The column therefore ends in `(n_padded - n_constraints) << log_instances` zero rows.
		let n_constraints = operands.len();
		let total = 1 << (self.log_instances + log2_ceil_usize(n_constraints));

		let mut out = alloc.alloc::<Word>(total);
		// The allocator may hand back more capacity than asked for.
		// Bound the spare slice to the row count before splitting off the padding.
		let (constraint_rows, padding_rows) =
			out.spare_capacity_mut()[..total].split_at_mut(n_constraints << self.log_instances);

		// Zeroing a `MaybeUninit<Word>` initializes it.
		// `Word` is `repr(transparent)` over `u64`, whose all-zero bit pattern is zero.
		bytemuck::fill_zeroes(padding_rows);

		// One stripe per constraint, each holding that constraint's instances.
		operands
			.zip(constraint_rows.par_chunks_mut(1 << self.log_instances))
			.for_each(|(operand, stripe)| self.write_operand(stripe, operand));

		// SAFETY: the stripes partition `[0, n_constraints << log_instances)`.
		// Each is written in full, and everything past them was zeroed above.
		// So all `total` elements are initialized.
		unsafe { out.set_len(total) };
		out
	}

	/// Writes one operand into a stripe, initializing every cell.
	///
	/// # Overview
	///
	/// An operand is a XOR of shifted values:
	///
	/// ```text
	/// operand(instance) = XOR_t shift_t( value(index_t, instance) )
	/// ```
	///
	/// The first term initializes the stripe, so it never has to be zeroed first.
	///
	/// Every later term accumulates into it.
	fn write_operand(&self, out: &mut [MaybeUninit<Word>], operand: &Operand) {
		let mut terms = operand.iter();

		let Some(first) = terms.next() else {
			// An empty operand contributes zero to every instance.
			bytemuck::fill_zeroes(out);
			return;
		};

		self.write_term(out, first);

		// SAFETY: the write above set every cell of the stripe.
		let out = unsafe { out.assume_init_mut() };
		for term in terms {
			self.accumulate_term(out, term);
		}
	}

	/// Writes one term into every cell of a stripe, initializing it.
	fn write_term(&self, out: &mut [MaybeUninit<Word>], term: &ShiftedValueIndex) {
		match self.term_words(term) {
			TermWords::Splat(word) => {
				for cell in out {
					cell.write(word);
				}
			}
			// SAFETY:
			// * the row holds one word per instance, so it is exactly as long as the stripe;
			// * `MaybeUninit<Word>` has the same memory representation as `Word`.
			TermWords::Plain(row) => unsafe {
				ptr::copy_nonoverlapping(row.as_ptr(), out.as_mut_ptr() as *mut Word, out.len())
			},
			TermWords::Shifted {
				row,
				variant,
				amount,
			} => variant.write_shifted(out, row, amount),
		}
	}

	/// XORs one term into every cell of an initialized stripe.
	fn accumulate_term(&self, out: &mut [Word], term: &ShiftedValueIndex) {
		match self.term_words(term) {
			TermWords::Splat(word) => {
				for cell in out {
					*cell = *cell ^ word;
				}
			}
			TermWords::Plain(row) => {
				for (cell, &word) in iter::zip(out, row) {
					*cell = *cell ^ word;
				}
			}
			TermWords::Shifted {
				row,
				variant,
				amount,
			} => variant.xor_shifted(out, row, amount),
		}
	}

	/// Resolves one term to the words it contributes.
	fn term_words(&self, term: &ShiftedValueIndex) -> TermWords<'a> {
		let &ShiftedValueIndex {
			value_index,
			shift: Shift { variant, amount },
		} = term;

		let row_index = match value_index.segment() {
			// A constant names one word, shifted once and shared by every instance.
			ValueSegment::Constant => {
				let constant = self.constants[value_index.index() as usize];
				return TermWords::Splat(variant.apply(constant, amount as usize));
			}
			// An inout or private index names one wire, whose instances are one contiguous row.
			// The table stores the hidden segment, so the inout rows lead and the private ones
			// follow them.
			ValueSegment::InOut => value_index.index() as usize,
			ValueSegment::Private => self.n_inout + value_index.index() as usize,
			// A scratch word is never committed, and `ConstraintSystem::validate` rejects an
			// operand naming one.
			ValueSegment::Scratch => panic!("a constraint system has no Scratch values"),
		};
		let row = &self.hidden
			[(row_index << self.log_instances)..((row_index + 1) << self.log_instances)];

		// Invariant: shifting by zero returns the word untouched, whichever variant is named.
		// So the unshifted case covers every variant.
		// The shifted case then never sees a zero amount.
		match u32::from(amount) {
			0 => TermWords::Plain(row),
			amount => TermWords::Shifted {
				row,
				variant,
				amount,
			},
		}
	}
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_compute::{BufferPool, GlobalAllocator};
	use binius_core::constraint_system::{AndConstraint, Shift, ShiftVariant, ValueVec};
	use binius_field::{AESTowerField8b as B8, PackedBinaryGhash1x128b, Random};
	use binius_frontend::{Circuit, CircuitBuilder, Wire};
	use binius_ip::channel::Error as ChannelError;
	use binius_math::{
		BinarySubspace, FieldBuffer, multilinear::evaluate::evaluate,
		univariate::lagrange_evals_scalars,
	};
	use binius_prover::{and_reduction, fold_word::BitAxisFolder};
	use binius_transcript::{ProverTranscript, VerifierTranscript};
	use binius_utils::checked_arithmetics::checked_log_2;
	use binius_verifier::{
		Error as VerifierError,
		config::{B128, StdChallenger},
		protocols::bitand::{AndCheckOutput, SKIPPED_VARS},
		verify_bitand_reduction,
	};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	/// A width-1 packed field keeps one scalar per element, so the SIMD sumcheck rounds stay
	/// simple.
	type P = PackedBinaryGhash1x128b;

	// Four instances of the AND circuit, with distinct satisfying inputs.
	//
	// Shared by the tests that only need columns to fold, not a particular fixture.
	fn four_instance_and_table() -> (AndCircuit, ValueTable) {
		let c = and_circuit();
		let table = populate_table(&c, &[(1, 3, 7), (5, 6, 0), (9, 12, 0xFF), (0xF0, 0x0F, 1)]);
		(c, table)
	}

	#[test]
	fn with_derived_and_appends_the_conjunction() {
		// Invariant: the third column is the word-by-word conjunction of the first two.
		// The two inputs pass through untouched.
		//
		// Fixture state: 4 instances, the AND circuit's own constraints.
		let (c, table) = four_instance_and_table();
		let and_constraints = table_constraints(&c);
		let columns =
			OperandColumns::build(&table, constants(&c), &and_constraints, &GlobalAllocator);

		// Snapshot the inputs, since deriving the third column consumes the value.
		let [a_before, b_before] = columns.as_slices().map(<[Word]>::to_vec);

		let derived = columns.with_derived_and(&GlobalAllocator);
		let [a, b, c_column] = derived.as_slices();

		// The two input columns are carried over unchanged.
		assert_eq!(a, &a_before[..]);
		assert_eq!(b, &b_before[..]);

		// Every row of the third column is the conjunction of the other two.
		let expected: Vec<Word> = iter::zip(a, b).map(|(&a_i, &b_i)| a_i & b_i).collect();
		assert_eq!(c_column, &expected[..]);

		// Sanity: the fixture leaves at least one non-zero row, so the check is not vacuous.
		assert!(c_column.iter().any(|&word| word != Word::ZERO));
	}

	#[test]
	fn rho_multilinear_matches_the_direct_triple_sum() {
		// Invariant: folding the bit and constraint axes leaves one element per instance.
		//
		//     M[rho] = sum_{local, j} lagrange[j] * tensor[local] * bit_j(column[local*K + rho])
		//
		// Fixture state: 4 instances, random weights on both folded axes.
		let (c, table) = four_instance_and_table();
		let and_constraints = table_constraints(&c);
		let columns = OperandColumns::<_, 2>::build(
			&table,
			constants(&c),
			&and_constraints,
			&GlobalAllocator,
		);

		let n_instances = table.n_instances();
		let n_padded = columns.as_slices()[0].len() / n_instances;

		// Independent weights per axis, so a swapped index cannot pass by coincidence.
		let mut rng = StdRng::seed_from_u64(0);
		let lagrange: Vec<B128> = (0..Word::BITS).map(|_| B128::random(&mut rng)).collect();
		let r_x_tensor: Vec<B128> = (0..n_padded).map(|_| B128::random(&mut rng)).collect();

		for operand in 0..2 {
			let column = columns.as_slices()[operand];
			let folded =
				columns.rho_multilinear::<P>(operand, &lagrange, &r_x_tensor, &GlobalAllocator);
			let got: Vec<B128> = folded.iter_scalars().collect();

			// One element per instance, each the triple sum over that instance's column entries.
			assert_eq!(got.len(), n_instances);
			for (rho, &value) in got.iter().enumerate() {
				let mut expected = B128::ZERO;
				for (local, &weight) in r_x_tensor.iter().enumerate() {
					let word = column[local * n_instances + rho];
					for (j, &basis) in lagrange.iter().enumerate() {
						// A set bit contributes both of its axis weights.
						if (word.0 >> j) & 1 == 1 {
							expected += weight * basis;
						}
					}
				}
				assert_eq!(value, expected, "operand {operand}, instance {rho}");
			}
		}
	}

	#[test]
	#[should_panic(expected = "the constraint tensor must cover the padded constraint axis")]
	fn rho_multilinear_rejects_a_tensor_of_the_wrong_width() {
		let (c, table) = four_instance_and_table();
		let and_constraints = table_constraints(&c);
		let columns = OperandColumns::<_, 2>::build(
			&table,
			constants(&c),
			&and_constraints,
			&GlobalAllocator,
		);

		// Mutation: one weight short of the padded constraint axis.
		//
		//     rows in the column:  n_padded * n_instances
		//     weights supplied:    n_padded - 1
		//
		// The fold indexes the column by the tensor, so a short one would drop constraint rows.
		let n_padded = columns.as_slices()[0].len() / table.n_instances();
		let lagrange = vec![B128::ZERO; Word::BITS];
		let short = vec![B128::ZERO; n_padded - 1];
		let _ = columns.rho_multilinear::<P>(0, &lagrange, &short, &GlobalAllocator);
	}

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
		let lagrange = lagrange_evals_scalars(&univariate_domain, &z_challenge);
		let folded: FieldBuffer<B128> = BitAxisFolder::new(&lagrange).fold(&GlobalAllocator, col);
		evaluate(&folded, eval_point)
	}

	// The per-instance AND constraints, at their true (unpadded) count.
	// This mirrors how the prover feeds constraints downstream.
	fn table_constraints(c: &AndCircuit) -> Vec<AndConstraint> {
		let cs = c.circuit.constraint_system().clone();
		cs.validate().unwrap();
		cs.and_constraints
	}

	// The circuit's constant words, shared by every instance.
	// This circuit declares none, so the slice is empty; operands read only hidden words.
	fn constants(c: &AndCircuit) -> &[Word] {
		&c.circuit.constraint_system().constants
	}

	// One AND constraint whose `A` and `B` operands evaluate to a non-zero word on every instance
	// of `table`, chosen with the reference evaluator rather than the builder under test.
	fn all_nonzero_constraint(c: &AndCircuit, table: &ValueTable) -> AndConstraint {
		table_constraints(c)
			.into_iter()
			.find(|constraint| {
				(0..table.n_instances()).all(|instance| {
					let vv = table.instance_value_vec(instance, constants(c));
					vv.eval_operand(constraint.a()) != Word::ZERO
						&& vv.eval_operand(constraint.b()) != Word::ZERO
				})
			})
			.expect("the fixture has an AND constraint with non-zero A and B on every instance")
	}

	// A circuit asserting `z == (x & y) ^ w`, over four witness words.
	//
	//     inputs : x, y, w, z   (all witness wires)
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
	//
	// Only the first two columns exist; the third is derived rather than stored.
	fn reference_rows(and_constraints: &[AndConstraint], vv: &ValueVec) -> (Vec<Word>, Vec<Word>) {
		let mut a = Vec::new();
		let mut b = Vec::new();
		for constraint in and_constraints {
			a.push(vv.eval_operand(constraint.a()));
			b.push(vv.eval_operand(constraint.b()));
		}
		(a, b)
	}

	// The reference columns across the whole batch, transposed to the batch witness's
	// constraint-major layout: `columns.a[j * n_instances + instance]` is constraint `j`'s
	// operand `A` evaluated on `instance`.
	fn reference_columns(
		table: &ValueTable,
		constants: &[Word],
		and_constraints: &[AndConstraint],
	) -> (Vec<Word>, Vec<Word>) {
		let n_and = and_constraints.len();
		let n_instances = table.n_instances();
		let mut a = vec![Word::ZERO; n_and * n_instances];
		let mut b = vec![Word::ZERO; n_and * n_instances];
		for instance in 0..n_instances {
			let vv = table.instance_value_vec(instance, constants);
			let (a_ref, b_ref) = reference_rows(and_constraints, &vv);
			for j in 0..n_and {
				a[j * n_instances + instance] = a_ref[j];
				b[j * n_instances + instance] = b_ref[j];
			}
		}
		(a, b)
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
		let columns =
			OperandColumns::build(&table, constants(&c), and_constraints, &GlobalAllocator);
		let [a, b] = columns.as_slices();

		// Shape: K * n_and rows, with K = 4.
		let n_and = and_constraints.len();
		assert_eq!(a.len(), 4 * n_and);
		assert_eq!(b.len(), 4 * n_and);

		// Invariant: row `j * n_instances + instance` is constraint `j` of that instance.
		let (a_ref, b_ref) = reference_columns(&table, constants(&c), and_constraints);
		assert_eq!(a, a_ref);
		assert_eq!(b, b_ref);
	}

	// A circuit computing one unsigned 64×64→128 product, with both result words committed.
	//
	//     inputs : x, y   (witness)
	//     gate   : (hi, lo) = imul(x, y)   → 1 IMUL constraint (+ 1 AND security check)
	//
	// `force_commit` makes `hi` and `lo` hidden words, so the IMUL operands read them from the
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

	// The four IntMul columns are laid out constraint-major, in the order [A, B, LO, HI].
	// Every row matches what the single-instance operand evaluator produces.
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

		// The per-instance IMUL constraints, at their true (unpadded) count.
		let cs = c.circuit.constraint_system().clone();
		cs.validate().unwrap();
		let imul_constraints = &cs.imul_constraints;
		assert!(!imul_constraints.is_empty(), "the circuit must emit an IMUL constraint");

		let columns = OperandColumns::build(&table, constants, imul_constraints, &GlobalAllocator);
		let [a, b, lo, hi] = columns.as_slices();

		// Shape: K * n_imul rows, with K = 4.
		let n_imul = imul_constraints.len();
		for col in [&a, &b, &hi, &lo] {
			assert_eq!(col.len(), 4 * n_imul);
		}

		// Invariant: row `j * n_instances + instance` is constraint `j` of that instance, and each
		// of the four columns equals the single-instance reference for its inputs.
		let n_instances = table.n_instances();
		for instance in 0..n_instances {
			let vv = table.instance_value_vec(instance, constants);
			for (j, con) in imul_constraints.iter().enumerate() {
				let idx = j * n_instances + instance;
				assert_eq!(a[idx], vv.eval_operand(con.a()));
				assert_eq!(b[idx], vv.eval_operand(con.b()));
				assert_eq!(hi[idx], vv.eval_operand(con.hi()));
				assert_eq!(lo[idx], vv.eval_operand(con.lo()));
			}
		}
	}

	// A circuit computing one GHASH-field product `(a_lo, a_hi) * (b_lo, b_hi)`, with both product
	// words committed.
	//
	//     inputs : a_lo, a_hi, b_lo, b_hi   (witness)
	//     gate   : (c_lo, c_hi) = bmul(a_lo, a_hi, b_lo, b_hi)   → 1 BMUL constraint
	//
	// `force_commit` makes `c_lo` and `c_hi` hidden words, so the BMUL operands read them from the
	// table.
	struct BinMulCircuit {
		circuit: Circuit,
		a_lo: Wire,
		a_hi: Wire,
		b_lo: Wire,
		b_hi: Wire,
	}

	fn binmul_circuit() -> BinMulCircuit {
		let builder = CircuitBuilder::new();
		let a_lo = builder.add_witness();
		let a_hi = builder.add_witness();
		let b_lo = builder.add_witness();
		let b_hi = builder.add_witness();
		let (c_lo, c_hi) = builder.bmul(a_lo, a_hi, b_lo, b_hi);
		builder.force_commit(c_lo);
		builder.force_commit(c_hi);
		BinMulCircuit {
			circuit: builder.build(),
			a_lo,
			a_hi,
			b_lo,
			b_hi,
		}
	}

	// Populate one instance per input tuple; the circuit derives the two product words.
	fn populate_binmul_table(c: &BinMulCircuit, inputs: &[(u64, u64, u64, u64)]) -> ValueTable {
		let log_instances = inputs.len().ilog2() as usize;
		ValueTable::populate(&c.circuit, log_instances, |i, filler| {
			let (a_lo, a_hi, b_lo, b_hi) = inputs[i];
			filler[c.a_lo] = Word(a_lo);
			filler[c.a_hi] = Word(a_hi);
			filler[c.b_lo] = Word(b_lo);
			filler[c.b_hi] = Word(b_hi);
		})
		.unwrap()
	}

	// The six BinMul columns are laid out constraint-major, in operand order.
	// Every row matches what the single-instance operand evaluator produces.
	#[test]
	fn binmul_operand_columns_match_the_single_instance_reference() {
		let c = binmul_circuit();
		let constants = &c.circuit.constraint_system().constants;

		// Fixture state: 2^2 = 4 instances with distinct GHASH-field operands.
		let inputs = [
			(1, 0, 1, 0),
			(0xFFFF_FFFF_FFFF_FFFF, 0x0123_4567_89AB_CDEF, 0xDEAD_BEEF_CAFE_BABE, 0x1),
			(0x1234, 0x5678, 0x9ABC, 0xDEF0),
			(
				0xAAAA_AAAA_AAAA_AAAA,
				0x5555_5555_5555_5555,
				0xF0F0_F0F0_F0F0_F0F0,
				0x0F0F_0F0F_0F0F_0F0F,
			),
		];
		let table = populate_binmul_table(&c, &inputs);

		// The per-instance BMUL constraints, at their true (unpadded) count.
		let cs = c.circuit.constraint_system().clone();
		cs.validate().unwrap();
		let bmul_constraints = &cs.bmul_constraints;
		assert!(!bmul_constraints.is_empty(), "the circuit must emit a BMUL constraint");

		let columns = OperandColumns::build(&table, constants, bmul_constraints, &GlobalAllocator);
		let [a_lo, a_hi, b_lo, b_hi, c_lo, c_hi] = columns.as_slices();

		// Shape: K * n_binmul rows, with K = 4.
		let n_binmul = bmul_constraints.len();
		for col in [&a_lo, &a_hi, &b_lo, &b_hi, &c_lo, &c_hi] {
			assert_eq!(col.len(), 4 * n_binmul);
		}

		// Invariant: row `j * n_instances + instance` is constraint `j` of that instance, and each
		// of the six columns equals the single-instance reference for its inputs.
		let n_instances = table.n_instances();
		for instance in 0..n_instances {
			let vv = table.instance_value_vec(instance, constants);
			for (j, con) in bmul_constraints.iter().enumerate() {
				let idx = j * n_instances + instance;
				assert_eq!(a_lo[idx], vv.eval_operand(con.a_lo()));
				assert_eq!(a_hi[idx], vv.eval_operand(con.a_hi()));
				assert_eq!(b_lo[idx], vv.eval_operand(con.b_lo()));
				assert_eq!(b_hi[idx], vv.eval_operand(con.b_hi()));
				assert_eq!(c_lo[idx], vv.eval_operand(con.c_lo()));
				assert_eq!(c_hi[idx], vv.eval_operand(con.c_hi()));
			}
		}
	}

	#[test]
	fn single_instance_batch_matches_the_reference() {
		let c = and_circuit();

		// Fixture state: log_instances = 0 → exactly one instance (K = 1).
		let table = populate_table(&c, &[(0xABCD, 0x0F0F, 0x55)]);
		let and_constraints = table_constraints(&c);
		let columns =
			OperandColumns::build(&table, constants(&c), &and_constraints, &GlobalAllocator);
		let [a, b] = columns.as_slices();

		// The degenerate batch reproduces the single-instance BitAnd columns exactly.
		let vv = table.instance_value_vec(0, constants(&c));
		let (a_ref, b_ref) = reference_rows(&and_constraints, &vv);
		assert_eq!(a, a_ref);
		assert_eq!(b, b_ref);
	}

	#[test]
	fn build_zero_pads_a_non_power_of_two_constraint_count() {
		let c = and_circuit();

		// Fixture state: a batch of 2 instances (K = 2).
		let table = populate_table(&c, &[(1, 3, 7), (5, 6, 0)]);
		let n_instances = table.n_instances();

		// Invariant: the constraint axis is rounded up to a power of two, so 3 constraints yield 4
		// constraint stripes, the last of which is all zeros.
		//
		// All three constraints repeat one whose `A` and `B` are non-zero on every instance.
		// Every leading stripe therefore carries non-zero words.
		//
		// Asserting that pins the split point.
		// A column zeroed past the wrong offset fails here, rather than passing the zero check.
		let constraints = vec![all_nonzero_constraint(&c, &table); 3];
		let columns = OperandColumns::build(&table, constants(&c), &constraints, &GlobalAllocator);
		let [a, b] = columns.as_slices();

		for col in [&a, &b] {
			assert_eq!(col.len(), 4 * n_instances);
			assert!(
				col[..3 * n_instances]
					.iter()
					.all(|&word| word != Word::ZERO)
			);
			assert!(
				col[3 * n_instances..]
					.iter()
					.all(|&word| word == Word::ZERO)
			);
		}
	}

	// One drawn shift term: which of the circuit's four wires to read, and how to shift it.
	// The amount stays in `1..max_amount`, so every draw really shifts and stays in range.
	fn any_shift_term() -> impl Strategy<Value = (usize, ShiftVariant, u8)> {
		(0usize..4, prop::sample::select(ShiftVariant::ALL.to_vec())).prop_flat_map(
			|(wire, shift_variant)| {
				(1..shift_variant.max_amount())
					.prop_map(move |amount| (wire, shift_variant, amount as u8))
			},
		)
	}

	proptest! {
		// Invariant: every batch row equals the single-instance reference, for all eight shifts.
		//
		// The compiler emits only unshifted operands, so the fixed fixtures reach three shifts.
		// Drawing the shift covers the other five: rotate, and the four half-word forms.
		//
		// Fixture state: 4 instances, 1 constraint.
		//
		//     operand A: 1..4 drawn terms  → always initializes its column
		//     operand B: 0..4 drawn terms  → may be empty, pinning that column at zero
		#[test]
		fn shifted_operands_match_the_reference_for_every_variant(
			inputs in prop::collection::vec((any::<u64>(), any::<u64>(), any::<u64>()), 4),
			a_terms in prop::collection::vec(any_shift_term(), 1..4),
			b_terms in prop::collection::vec(any_shift_term(), 0..4),
		) {
			let c = and_circuit();
			let table = populate_table(&c, &inputs);

			// A drawn wire slot in 0..4 names one of the circuit's four witness words.
			let wires = [c.x, c.y, c.w, c.z].map(|wire| c.circuit.witness_index(wire));
			// An operand is the XOR of its terms, so a term list becomes one operand directly.
			let to_operand = |terms: &[(usize, ShiftVariant, u8)]| -> Operand {
				terms
					.iter()
					.map(|&(wire, variant, amount)| ShiftedValueIndex {
						value_index: wires[wire],
						shift: Shift { variant, amount },
					})
					.collect()
			};

			// The third operand is never evaluated by the builder, so it stays empty.
			// The fixture therefore need not satisfy the AND relation to exercise the shifts.
			let and_constraints =
				vec![AndConstraint([to_operand(&a_terms), to_operand(&b_terms), Operand::new()])];

			let columns =
				OperandColumns::build(&table, constants(&c), &and_constraints, &GlobalAllocator);
			let [a, b] = columns.as_slices();

			// Both columns match the shift-aware value-vec evaluator on the same constraints.
			let (a_ref, b_ref) = reference_columns(&table, constants(&c), &and_constraints);
			prop_assert_eq!(a, &a_ref[..]);
			prop_assert_eq!(b, &b_ref[..]);
		}
	}

	proptest! {
		// Invariant: every batch row equals the single-instance reference for that instance.
		//
		//     witness[j * n_instances + instance]  ==  eval_operand(instance value vec, constraint j)
		//
		// This pins the batched, slice-based evaluator to the core value-vec evaluator.
		//
		// The columns are drawn from a `BufferPool`, so this is the case that covers the
		// `A::Vec = PoolVec` instantiation; the rest of the module's tests use `GlobalAllocator`.
		#[test]
		fn batch_rows_match_single_instance_reference(
			inputs in prop::collection::vec((any::<u64>(), any::<u64>(), any::<u64>()), 4),
		) {
			let c = and_circuit();
			let table = populate_table(&c, &inputs);

			let pool = BufferPool::new();
			let and_constraints = table_constraints(&c);
			let columns = OperandColumns::build(&table, constants(&c), &and_constraints, &&pool);
			let [a, b] = columns.as_slices();

			let (a_ref, b_ref) = reference_columns(&table, constants(&c), &and_constraints);
			prop_assert_eq!(a, &a_ref[..]);
			prop_assert_eq!(b, &b_ref[..]);
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
		let columns =
			OperandColumns::build(&table, constants(&c), &and_constraints, &GlobalAllocator);
		let [a, b] = columns.as_slices();

		// Every instance's contribution equals its independent single-instance reference.
		// This includes instances at or beyond STRIPE_WIDTH, which only the second stripe produces.
		let (a_ref, b_ref) = reference_columns(&table, constants(&c), &and_constraints);
		assert_eq!(a, a_ref);
		assert_eq!(b, b_ref);
	}

	#[test]
	fn build_matches_reference_with_shifted_operands() {
		let c = and_circuit();

		// Fixture state: 4 instances with distinct inputs.
		let table =
			populate_table(&c, &[(1, 3, 7), (0xF0F0, 0x0FF0, 0xAA), (5, 6, 9), (0xFFFF, 1, 2)]);

		// Hand-craft constraints that carry real shifts on hidden operands.
		//
		// The circuit compiler emits only unshifted operands here.
		// The shifted branch would otherwise go untested by this module.
		//
		// The third operand is deliberately unrelated to the conjunction of the first two.
		// The builder never evaluates it, so the fixture need not satisfy the AND relation.
		let x = c.circuit.witness_index(c.x);
		let y = c.circuit.witness_index(c.y);
		let z = c.circuit.witness_index(c.z);
		let and_constraints = vec![
			AndConstraint([
				// A mixes a shifted and an unshifted term, exercising both accumulator paths.
				vec![ShiftedValueIndex::sll(x, 3), ShiftedValueIndex::plain(y)],
				vec![ShiftedValueIndex::srl(y, 5)],
				vec![ShiftedValueIndex::sar(z, 7)],
			]),
			// An empty operand set: its column must stay at the zeroed initial value.
			AndConstraint::default(),
		];

		// Sanity: at least one operand term really is shifted, so the else-branch runs.
		let shifted = and_constraints
			.iter()
			.flat_map(|con| [con.a(), con.b()])
			.any(|op| op.iter().any(|sv| !sv.shift.is_identity()));
		assert!(shifted, "fixture must contain a shifted operand");

		let columns =
			OperandColumns::build(&table, constants(&c), &and_constraints, &GlobalAllocator);
		let [a, b] = columns.as_slices();

		// `a` and `b` equal the shift-aware value-vec reference for the same constraints.
		let (a_ref, b_ref) = reference_columns(&table, constants(&c), &and_constraints);
		assert_eq!(a, a_ref);
		assert_eq!(b, b_ref);
	}

	#[test]
	fn small_batch_round_trips_with_only_pinned_challenges() {
		let c = and_circuit();

		// Fixture state: K = 4 instances, n_and = 2 → 8 rows → log_total = 3.
		// The row count log equals the pinned-challenge count.
		// So every coordinate is pinned and no large-field challenge is drawn.
		let table = populate_table(&c, &[(1, 3, 7), (5, 6, 0), (9, 12, 0xFF), (0xF0, 0x0F, 1)]);
		let and_constraints = table_constraints(&c);
		let columns =
			OperandColumns::build(&table, constants(&c), &and_constraints, &GlobalAllocator);
		let [a, b] = columns.as_slices();
		let log_total = checked_log_2(a.len());

		// Prover and verifier agree on the reduced claim over the batched columns.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prove_output = and_reduction::prove::<_, B128, P, _, _>(
			[a, b],
			&mut prover_transcript,
			&GlobalAllocator,
		);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verify_output = verify_bitand_reduction(
			log_total,
			&message_domain().isomorphic::<B128>(),
			&mut verifier_transcript,
		)
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
		let columns =
			OperandColumns::build(&table, constants(&c), &and_constraints, &GlobalAllocator);
		let [a, b] = columns.as_slices();
		let log_total = checked_log_2(a.len());

		// Produce a faithful proof.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let _ = and_reduction::prove::<_, B128, P, _, _>(
			[a, b],
			&mut prover_transcript,
			&GlobalAllocator,
		);
		let mut proof = prover_transcript.finalize();

		// Mutation: flip a bit in the prover's first message, the univariate round evaluations.
		proof[0] ^= 1;

		// The verifier redraws a different univariate challenge from the tampered message.
		// The final consistency check then no longer holds.
		// So verification fails.
		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let err = verify_bitand_reduction(
			log_total,
			&message_domain().isomorphic::<B128>(),
			&mut verifier_transcript,
		)
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
			let columns =
				OperandColumns::build(&table, constants(&c), &and_constraints, &GlobalAllocator);
			let [a, b] = columns.as_slices();

			// The columns outlive the reduction, so the claimed evals can be checked against them.
			//
			// No third column is stored, and the reduction folds the conjunction of the first two.
			// Materialize that same derived column, so the third claim is pinned to it.
			let c_cols: Vec<Word> = iter::zip(a, b)
				.map(|(&a, &b)| a & b)
				.collect();
			let log_total = checked_log_2(a.len());

			let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
			let prove_output = and_reduction::prove::<_, B128, P, _, _>(columns.as_slices(), &mut prover_transcript, &GlobalAllocator);

			let mut verifier_transcript = prover_transcript.into_verifier();
			let verify_output =
				verify_bitand_reduction(log_total, &message_domain().isomorphic::<B128>(), &mut verifier_transcript).unwrap();
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
			prop_assert_eq!(fold_eval_column(a, z_challenge, &eval_point), a_eval);
			prop_assert_eq!(fold_eval_column(b, z_challenge, &eval_point), b_eval);
			prop_assert_eq!(fold_eval_column(&c_cols, z_challenge, &eval_point), c_eval);
		}
	}
}
