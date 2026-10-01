// Copyright 2026 The Binius Developers

use binius_core::constraint_system::{ConstraintSystem, InoutSegment, Operand};
use binius_field::{BinaryField, FieldOps, util::FieldFn};
use binius_math::{
	line::extrapolate_line,
	multilinear::{
		eq::eq_ind_zero,
		sparse::{
			SparseBitVector, evaluate_sparse_b1_multilinear, evaluate_sparse_b1_multilinear_native,
		},
	},
};
use getset::{CopyGetters, Getters};

use super::LOG_SHIFT_COUNT;
use crate::reduction::{LOG_OPERANDS, log_constraint_point};

/// The address bits below the word index: the operand column, then the inner and outer shift.
const LOG_TERM_BITS: usize = LOG_OPERANDS + 2 * LOG_SHIFT_COUNT;

/// The wiring matrix as two bit vectors, one per committed segment, with the widths of the point
/// they are read at.
///
/// Each set bit is one operand term, at the address
///
/// ```text
/// bits, low to high:  operand (LOG_OPERANDS) | inner shift (9) | outer shift (9) | word | constraint
/// ```
///
/// A shift slot is [`Shift::index`](binius_core::constraint_system::Shift::index). The word index
/// is relative to its segment, `log_public_words` wide for the public vector and
/// `log_segment_words` wide for the hidden one. The constraint index is `log_constraint_point`
/// wide. The operand column runs over each operation's operands in `[zero, bitand, intmul,
/// binmul]` order. A term listed twice cancels.
///
/// The bit order is the order of the point the shift reduction evaluates the vectors at, so this
/// is the one place that fixes the layout.
///
/// # The wiring multilinear
///
/// As a [`FieldFn`], this evaluates the two vectors joined on the segment selector:
///
/// ```text
/// W = extrapolate_line(eq(0, r_y[log_public_words..]) · P(point_pub), H(point_hid), r_segment)
///
/// point_hid = operand_batch | r_s_inner | r_v_inner | r_s_outer | r_v_outer | r_y | r_x
/// point_pub = the same with r_y cut to log_public_words
/// ```
///
/// Its input is `r_segment` followed by `point_hid`. The public segment spans a prefix of the
/// word-index space, so its point drops the coordinates above it, which read as zero.
///
/// The bit-index factors every shift scalar is scaled by are left out: they depend on prover
/// messages, so [`check_eval`](super::check_eval) multiplies them in outside.
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct WiringInfo {
	/// The terms whose word lies in the public segment.
	#[getset(get = "pub")]
	public_segment: SparseBitVector,
	/// The terms whose word lies in the hidden segment.
	#[getset(get = "pub")]
	hidden_segment: SparseBitVector,
	/// The word-index variables the public segment spans.
	#[getset(get_copy = "pub")]
	log_public_words: usize,
	/// The word-index variables the shift reduction runs over, the wider of the two segments'
	/// spans: the length of `r_y`.
	#[getset(get_copy = "pub")]
	log_segment_words: usize,
	/// The constraint-index variables: the length of `r_x`.
	#[getset(get_copy = "pub")]
	log_constraint_point: usize,
}

impl WiringInfo {
	/// Lays out the wiring matrix of `cs`, with its inout values in `inout`.
	///
	/// # Panics
	///
	/// Panics if the hidden address does not fit a `u64` index below `2^63`: that is,
	/// `LOG_OPERANDS + 18 + log_constraint_point + log_segment_words < 64`, or
	/// `log_constraint_point + log_segment_words ≤ 41`.
	pub fn new(cs: &ConstraintSystem, inout: InoutSegment) -> Self {
		let log_public_words = cs.log_public_words(inout);
		let log_segment_words = cs.log_segment_words(inout);
		let log_constraint_point = log_constraint_point(cs);
		let log_words = [log_public_words, log_segment_words];
		let log_lens = log_words.map(|log_words| LOG_TERM_BITS + log_words + log_constraint_point);
		assert!(
			log_lens[1] < u64::BITS as usize,
			"the wiring address needs {} bits; at most 63 fit",
			log_lens[1]
		);

		// Each operand term's segment and address, in the column order the reduction batches.
		let n_public_words = cs.n_public_words(inout);
		let mut indices = [Vec::new(), Vec::new()];
		let mut push = |column: usize, constraint_index: usize, terms: &Operand| {
			for term in terms {
				let word = cs.word_offset(term.value_index);
				let (segment, word) = if word < n_public_words {
					(0, word)
				} else {
					(1, word - n_public_words)
				};
				let index = column as u64
					| (term.inner().index() as u64) << LOG_OPERANDS
					| (term.outer().index() as u64) << (LOG_OPERANDS + LOG_SHIFT_COUNT)
					| (word as u64) << LOG_TERM_BITS
					| (constraint_index as u64) << (LOG_TERM_BITS + log_words[segment]);
				indices[segment].push(index);
			}
		};

		/// Walks one operation, its operands at the columns from `first_column` on, and returns the
		/// column after its last.
		fn walk<C: AsRef<[Operand; ARITY]>, const ARITY: usize>(
			first_column: usize,
			constraints: &[C],
			push: &mut impl FnMut(usize, usize, &Operand),
		) -> usize {
			for (constraint_index, constraint) in constraints.iter().enumerate() {
				for (operand_index, terms) in constraint.as_ref().iter().enumerate() {
					push(first_column + operand_index, constraint_index, terms);
				}
			}
			first_column + ARITY
		}

		let column = walk(0, &cs.zero_constraints, &mut push);
		let column = walk(column, &cs.and_constraints, &mut push);
		let column = walk(column, &cs.imul_constraints, &mut push);
		walk(column, &cs.bmul_constraints, &mut push);

		let [public, hidden] = indices;
		let [public_log_len, hidden_log_len] = log_lens;
		Self {
			public_segment: SparseBitVector::new(public_log_len, public),
			hidden_segment: SparseBitVector::new(hidden_log_len, hidden),
			log_public_words,
			log_segment_words,
			log_constraint_point,
		}
	}

	/// Joins the two segments' evaluations at `r_segment`, each vector read by `eval`.
	fn eval<E: FieldOps>(&self, vals: &[E], eval: impl Fn(&SparseBitVector, &[E]) -> E) -> E {
		let (r_segment, point_hid) = vals.split_first().expect("the input leads with r_segment");
		assert_eq!(point_hid.len(), self.hidden_segment.log_len());

		let public_end = LOG_TERM_BITS + self.log_public_words;
		let r_y_end = LOG_TERM_BITS + self.log_segment_words;
		let point_pub = [&point_hid[..public_end], &point_hid[r_y_end..]].concat();
		let public_eval =
			eq_ind_zero(&point_hid[public_end..r_y_end]) * eval(&self.public_segment, &point_pub);
		let hidden_eval = eval(&self.hidden_segment, point_hid);
		extrapolate_line(public_eval, hidden_eval, r_segment.clone())
	}
}

impl<F: BinaryField> FieldFn<F> for WiringInfo {
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, vals: &[E]) -> E {
		self.eval(vals, evaluate_sparse_b1_multilinear)
	}

	fn call_native(&self, vals: &[F]) -> F {
		self.eval(vals, evaluate_sparse_b1_multilinear_native)
	}
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_core::{
		ShiftVariant,
		constraint_system::{
			AndConstraint, BmulConstraint, ImulConstraint, Shift, ShiftedValueIndex, ValueIndex,
			ValueSegment, ZeroConstraint,
		},
		word::Word,
	};
	use binius_field::Field;
	use binius_math::{multilinear::eq::eq_ind_partial_eval_scalars, test_utils::random_scalars};
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::config::B128;

	/// A system whose terms read every value segment, through two shifts each.
	///
	/// IMUL is optional, so the columns after it only line up if an empty run is still skipped.
	fn random_system(rng: &mut StdRng, with_imul: bool) -> ConstraintSystem {
		let constants = vec![Word::ZERO, Word::ONE, Word::ALL_ONE];
		let n_inout = 5;
		let n_private = 12;
		let term = |rng: &mut StdRng| {
			let value_index = match rng.random_range(0..3) {
				0 => ValueIndex::constant(rng.random_range(0..constants.len()) as u32),
				1 => ValueIndex::inout(rng.random_range(0..n_inout) as u32),
				_ => ValueIndex::private(rng.random_range(0..n_private) as u32),
			};
			let shift = |rng: &mut StdRng| Shift {
				variant: [ShiftVariant::Sll, ShiftVariant::Sar, ShiftVariant::Rotr32]
					[rng.random_range(0..3)],
				amount: rng.random_range(0..Word::BITS) as u8,
			};
			ShiftedValueIndex::new(value_index, [shift(rng), shift(rng)])
		};
		let operand = |rng: &mut StdRng| (0..rng.random_range(0..3)).map(|_| term(rng)).collect();

		ConstraintSystem {
			zero_constraints: (0..5)
				.map(|_| ZeroConstraint(array::from_fn(|_| operand(rng))))
				.collect(),
			and_constraints: (0..7)
				.map(|_| AndConstraint(array::from_fn(|_| operand(rng))))
				.collect(),
			imul_constraints: if with_imul {
				(0..3)
					.map(|_| ImulConstraint(array::from_fn(|_| operand(rng))))
					.collect()
			} else {
				Vec::new()
			},
			bmul_constraints: (0..3)
				.map(|_| BmulConstraint(array::from_fn(|_| operand(rng))))
				.collect(),
			constants,
			n_inout,
			n_private,
		}
	}

	/// The wiring multilinear as the tensor's defining sum: one weight table per axis, one product
	/// per term.
	///
	/// The value axis is the full `r_y | r_segment` indicator, the hidden segment in its high half,
	/// so it holds the padded reading against the builder's two cut segments.
	fn contract(cs: &ConstraintSystem, inout: InoutSegment, vals: &[B128]) -> B128 {
		let (&r_segment, point) = vals.split_first().unwrap();
		let r_y_len = cs.log_segment_words(inout);
		let (inner, rest) = point.split_at(LOG_OPERANDS + LOG_SHIFT_COUNT);
		let (outer, rest) = rest.split_at(LOG_SHIFT_COUNT);
		let (r_y, r_x) = rest.split_at(r_y_len);

		let inner = eq_ind_partial_eval_scalars(inner);
		let outer = eq_ind_partial_eval_scalars(outer);
		let value = eq_ind_partial_eval_scalars(&[r_y, &[r_segment]].concat());
		let constraint = eq_ind_partial_eval_scalars(r_x);

		let hidden = 1 << r_y_len;
		let address = |value_index: ValueIndex| {
			let index = value_index.index() as usize;
			match (value_index.segment(), inout) {
				(ValueSegment::Constant, _) => index,
				(ValueSegment::InOut, InoutSegment::Public) => cs.n_const() + index,
				(ValueSegment::InOut, InoutSegment::Hidden) => hidden + index,
				(ValueSegment::Private, InoutSegment::Public) => hidden + index,
				(ValueSegment::Private, InoutSegment::Hidden) => hidden + cs.n_inout + index,
				(ValueSegment::Scratch, _) => unreachable!(),
			}
		};

		// Each operation's first operand column, named rather than accumulated.
		let operations: [(usize, Vec<&[Operand]>); 4] = [
			(0, cs.zero_constraints.iter().map(|c| &c.0[..]).collect()),
			(1, cs.and_constraints.iter().map(|c| &c.0[..]).collect()),
			(4, cs.imul_constraints.iter().map(|c| &c.0[..]).collect()),
			(8, cs.bmul_constraints.iter().map(|c| &c.0[..]).collect()),
		];
		let mut acc = B128::ZERO;
		for (first_column, constraints) in operations {
			for (constraint_index, operands) in constraints.into_iter().enumerate() {
				for (operand_index, terms) in operands.iter().enumerate() {
					for term in terms {
						let column = first_column + operand_index;
						acc += constraint[constraint_index]
							* inner[(term.inner().index() << LOG_OPERANDS) | column]
							* outer[term.outer().index()]
							* value[address(term.value_index)];
					}
				}
			}
		}
		acc
	}

	#[test]
	fn the_wiring_bits_evaluate_to_the_contracted_tensor() {
		let mut rng = StdRng::seed_from_u64(31);
		for with_imul in [false, true] {
			let cs = random_system(&mut rng, with_imul);
			for inout in [InoutSegment::Public, InoutSegment::Hidden] {
				let wiring = WiringInfo::new(&cs, inout);
				let vals = random_scalars::<B128>(&mut rng, 1 + wiring.hidden_segment().log_len());

				let got = FieldFn::<B128>::call::<B128>(&wiring, &vals);
				// A vacuous claim would let a broken layout pass unnoticed.
				assert_ne!(got, B128::ZERO);
				assert_eq!(got, contract(&cs, inout, &vals), "{inout:?}, imul: {with_imul}");
				assert_eq!(got, FieldFn::<B128>::call_native(&wiring, &vals));
			}
		}
	}

	#[test]
	fn a_repeated_term_cancels() {
		// Two copies of one term sit at one address, and a bit set twice is clear.
		let term = ShiftedValueIndex::new(ValueIndex::private(1), [Shift::srl(7), Shift::srl(3)]);
		let mut cs = empty_system(0, 2);
		cs.and_constraints = vec![AndConstraint([vec![term, term], vec![term], vec![]])];
		let wiring = WiringInfo::new(&cs, InoutSegment::Public);
		assert!(wiring.public_segment().indices().is_empty());
		assert_eq!(wiring.hidden_segment().indices().len(), 1);
	}

	/// A system of `2^log_constraints` empty ZERO constraints over `2^log_words` private words.
	fn empty_system(log_constraints: usize, log_words: usize) -> ConstraintSystem {
		ConstraintSystem {
			constants: Vec::new(),
			n_inout: 0,
			n_private: 1 << log_words,
			zero_constraints: vec![ZeroConstraint::default(); 1 << log_constraints],
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		}
	}

	#[test]
	fn the_widest_address_that_fits_is_63_bits() {
		let wiring = WiringInfo::new(&empty_system(15, 26), InoutSegment::Public);
		assert_eq!(wiring.hidden_segment().log_len(), 63);
	}

	#[test]
	#[should_panic(expected = "the wiring address needs 64 bits")]
	fn a_system_whose_address_does_not_fit_is_rejected() {
		let _ = WiringInfo::new(&empty_system(16, 26), InoutSegment::Public);
	}
}
