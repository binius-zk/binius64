// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	iter,
	ops::{Deref, Range},
};

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		ProveSingleOutput, bivariate_product_evaluator::BivariateProductEvaluator,
		bivariate_product_prover, common::SumcheckProver, prove_single, round_evals::RoundEvals,
		round_evaluator::SharedSumcheckProver,
	},
};
use binius_math::{FieldBuffer, FieldVec, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};
use binius_verifier::protocols::shift::LOG_SHIFT_COUNT;
use bytemuck::zeroed_vec;
use itertools::izip;
use tracing::instrument;

use super::{
	SegmentWords,
	claims::PreparedOperatorClaims,
	key_collection::{DenseShiftEncoding, KeyCollection, KeySegment},
	monster::shift_operator_table,
	outer::OuterShiftStage,
};

/// The number of variables the shift-and-bit phases of the reduction span.
///
/// The axes run, from the low index positions up: the bit position within a word, then the inner
/// shift slot, then the outer one. The `g` multilinear spans all of them; no single table the
/// prover forms does, which is what the outer rounds exist to avoid.
pub const PHASE_1_LOG_LEN: usize = Word::LOG_BITS + LOG_SHIFT_ROWS;

/// The number of variables of the phase-1 multilinears that index the shift.
///
/// A term names a sequence of two shifts, so the row axis is two slots wide. It is indexed
/// **outer-major**, since the rounds peel the outer shift first.
pub const LOG_SHIFT_ROWS: usize = 2 * LOG_SHIFT_COUNT;

/// The number of variables one shift operator table spans.
///
/// Such a table holds one row of `Word::BITS` weights per `(variant, amount)` pair of a single
/// slot, the bit position running within a row.
pub const SHIFT_OPERATOR_LOG_LEN: usize = Word::LOG_BITS + LOG_SHIFT_COUNT;

/// What phase 1 leaves the phases after it: its sumcheck's challenge point, split into the axes
/// it binds, and the two evaluations it reduced to.
#[derive(Debug, Clone)]
pub struct Phase1Output<F> {
	/// The bit position within a word.
	pub r_j: Vec<F>,
	/// The inner shift's amount.
	pub r_s_inner: Vec<F>,
	/// The inner shift's variant, called the operation in the paper.
	pub r_v_inner: Vec<F>,
	/// The outer shift's amount.
	pub r_s_outer: Vec<F>,
	/// The outer shift's variant.
	pub r_v_outer: Vec<F>,
	/// The oblong weights carried through the outer shift, at the outer challenge point.
	///
	/// This is the terminal fold of the outer rounds' table, and it is the weight vector the
	/// rounds binding the intermediate bit index run against.
	pub psi: Vec<F>,
	/// The evaluation claim the next phase proves, called gamma in the paper: the product of the
	/// two evaluations below.
	pub gamma: F,
	/// `g` at the bound shift and bit point, the sum over the word index of the constraint matrix
	/// against the witness. It is the scalar the bit-index phases carry through their rounds, so
	/// the sum never has to be formed a second time.
	pub g_eval: F,
}

/// Proves the first phase of the shift reduction.
///
/// Builds the g multilinear and runs one sumcheck over its product with `h`, which is never formed
/// as a table — see [`run_phase_1_sumcheck`].
///
/// # Arguments
///
/// - `oblong_weights`: the oblong weights of the reduction's first factor, one per bit position.
///   `h` is those weights pushed through both shift slots.
#[instrument(skip_all, name = "prover_phase_1")]
pub fn prove_phase_1<F, P, Channel, A>(
	key_collection: &KeyCollection,
	words: SegmentWords<'_>,
	prepared: &PreparedOperatorClaims<F>,
	oblong_weights: &[F],
	channel: &mut Channel,
	alloc: &A,
) -> Phase1Output<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	// Accumulate the g rows of the public and hidden segments separately, then collect both. The
	// public words are the prefix of `words`, and each segment's key ranges are segment-relative.
	let public = build_g::<_, P>(words.public, &key_collection.public, prepared);
	let hidden = build_g::<_, P>(words.hidden, &key_collection.hidden, prepared);
	let g = SparseShiftRows::from_segments([
		(&public, &key_collection.public.dense_shift_enc),
		(&hidden, &key_collection.hidden.dense_shift_enc),
	]);

	run_phase_1_sumcheck::<_, P, _, _>(g, oblong_weights, prepared.batched_eval(), channel, alloc)
}

/// The number of packed elements one row of `Word::BITS` scalars occupies.
const fn row_len<P: PackedField>() -> usize {
	assert!(
		P::LOG_WIDTH <= Word::LOG_BITS,
		"a row of `Word::BITS` scalars must be a whole number of packed elements"
	);
	Word::BITS >> P::LOG_WIDTH
}

/// The rows of the phase-1 `g` multilinear that a constraint system's shifts reach.
///
/// `g` spans one row of `Word::BITS` scalars per `(shift variant, shift amount)` pair, of which a
/// constraint system names a few dozen out of the `1 << LOG_SHIFT_ROWS` the space holds — 16 for a
/// SHA256 circuit, 40 for Keccak. Only the named rows are stored, each tagged with its row index.
///
/// `g` is the sum of its stored rows, so **rows at a repeated index add up and need not be
/// deduplicated**. That is what lets the two key segments' rows simply concatenate, and what keeps
/// the list a flat run of the same length for the whole protocol.
#[derive(Debug, Clone)]
pub struct SparseShiftRows<P: PackedField> {
	/// The row index of each stored row, in `values` order and below `1 << log_rows`. Repeatable.
	indices: Vec<u32>,
	/// The stored rows end to end, [`row_len`] packed elements each, in `indices` order.
	values: Vec<P>,
	/// The number of row-index variables the sumcheck has yet to bind.
	log_rows: usize,
}

impl<P: PackedField> SparseShiftRows<P> {
	/// Collects the rows two key segments accumulated, tagged with the shift index each sits at.
	///
	/// Each segment's rows arrive in its own dense shift index order, which its encoding decodes to
	/// the shift indices this list keys on. The two segments concatenate: a shift both of them use
	/// contributes one row each at the same index, and the two add up where `g` is read.
	///
	/// # Panics
	///
	/// Panics unless each segment's rows number exactly what its encoding accounts for.
	pub fn from_segments(segments: [(&[P], &DenseShiftEncoding); 2]) -> Self {
		let mut indices = Vec::new();
		let mut values = Vec::new();
		for (rows, dense_shift_enc) in segments {
			assert_eq!(
				rows.len(),
				dense_shift_enc.len() * row_len::<P>(),
				"a segment holds one row per shift its encoding names"
			);
			indices.extend(dense_shift_enc.shift_indices().map(|index| index as u32));
			values.extend_from_slice(rows);
		}

		Self::new(indices, values, LOG_SHIFT_ROWS)
	}

	/// Collects stored rows sitting at the given indices of a row space `log_rows` variables wide.
	///
	/// Indices repeat freely: `g` is the sum of its rows, so two rows at one index add up where it
	/// is read.
	///
	/// # Panics
	///
	/// Panics unless there is one `row_len` run of values per index, and every index is below
	/// the row space.
	pub fn new(indices: Vec<u32>, values: Vec<P>, log_rows: usize) -> Self {
		assert_eq!(
			values.len(),
			indices.len() * row_len::<P>(),
			"the values hold one row per index"
		);
		assert!(
			indices
				.iter()
				.all(|&index| (index as usize) < 1 << log_rows),
			"every index names a row of the space"
		);

		Self {
			indices,
			values,
			log_rows,
		}
	}

	/// The number of row-index variables the sumcheck has yet to bind.
	pub(crate) const fn log_rows(&self) -> usize {
		self.log_rows
	}

	/// The stored rows, each with the row index it sits at.
	pub(crate) fn rows(&self) -> impl Iterator<Item = (usize, &[P])> {
		iter::zip(&self.indices, self.values.chunks_exact(row_len::<P>()))
			.map(|(&index, row)| (index as usize, row))
	}

	/// The index-space bit the next round binds, separating the two halves of the row space.
	///
	/// ## Preconditions
	///
	/// * `self.log_rows() >= 1`
	pub(crate) fn half(&self) -> usize {
		assert!(self.log_rows > 0, "precondition: a row-index variable remains to bind");
		1 << (self.log_rows - 1)
	}

	/// Binds the highest row-index variable to a challenge.
	///
	/// Folding is linear in `g`, so a row keeps its identity across it: it is scaled by the
	/// challenge weight of the half it sits in and moved down into the folded index space. The list
	/// therefore stays the same length, and nothing has to be paired up or merged.
	///
	/// ## Preconditions
	///
	/// * `self.log_rows() >= 1`
	pub(crate) fn fold(&mut self, challenge: P::Scalar) {
		let half = self.half();
		let lower_weight = P::broadcast(P::Scalar::ONE - challenge);
		let upper_weight = P::broadcast(challenge);

		let row_len = row_len::<P>();
		for (position, index) in self.indices.iter_mut().enumerate() {
			let row = &mut self.values[position * row_len..][..row_len];
			if *index as usize & half == 0 {
				row.iter_mut().for_each(|value| *value *= lower_weight);
			} else {
				row.iter_mut().for_each(|value| *value *= upper_weight);
				*index ^= half as u32;
			}
		}

		self.log_rows -= 1;
	}

	/// What is left of `g` once every row-index variable is bound: one dense row over the bit
	/// position, the sum of the stored rows now that they all sit at the one remaining index.
	///
	/// ## Preconditions
	///
	/// * `self.log_rows() == 0`
	fn into_bit_multilinear<A: Allocator>(self, alloc: &A) -> FieldVec<P, A> {
		assert_eq!(self.log_rows, 0, "precondition: every row-index variable is bound");

		let mut multilinear = FieldBuffer::zeros_in(alloc, Word::LOG_BITS);
		for (_, row) in self.rows() {
			for (slot, &value) in iter::zip(multilinear.as_mut(), row) {
				*slot += value;
			}
		}
		multilinear
	}
}

/// Proves the phase-1 sumcheck: the product of `g` and `h` summed over the hypercube.
///
/// `g` is zero outside the rows a constraint system's shifts name, but *within* a named row it is
/// dense — every one of the `Word::BITS` bit positions carries a value. Its sparsity is in long
/// strides rather than scattered points, so the prover changes strategy halfway:
///
/// - The [`LOG_SHIFT_ROWS`] rounds binding the row index read the stored rows alone
///   ([`SparseShiftRows`]), so their cost follows the shifts the system names rather than the 512
///   the space holds. `h` stays dense across them and folds as any multilinear does, which is what
///   keeps this simple: a round reads `h` at whichever row a live `g` row pairs with, so nothing
///   has to work out in advance which rows of `h` a later round will reach.
/// - Once the row index is bound, both multilinears are one dense row over the bit position and no
///   sparsity is left to exploit. The shared [`bivariate_product_prover`] runs the remaining
///   `Word::LOG_BITS` rounds.
///
/// Every round message is the one a dense prover over the whole of `g` would have sent, so the
/// transcript is unchanged and the verifier is untouched.
///
/// This adapts `SparseDenseProductSumcheckProver` to that stride structure, and inherits its
/// central simplification: both sampled evaluations and the fold are linear in `g`, so a stored
/// row contributes on its own and rows never have to be paired across the split or merged. That
/// prover keys on single points, which costs a pass over `n_shifts * Word::BITS` entries in
/// *every* round; keying on rows lets a round walk whole packed rows instead.
pub struct Phase1SumcheckProver<'alloc, A: Allocator, P: PackedField> {
	alloc: &'alloc A,
	/// The stage the protocol is in.
	///
	/// Always `Some` between calls. [`SumcheckProver::fold`] takes it out for the moment it moves
	/// the row stage's buffers into the bit stage's prover.
	stage: Option<Stage<'alloc, A, P>>,
}

/// Which half of the protocol the prover is in.
enum Stage<'alloc, A: Allocator, P: PackedField> {
	/// Binding the row index, over the rows `g` stores.
	Rows {
		g: SparseShiftRows<P>,
		h: FieldVec<P, A>,
		/// This round's sum claim.
		claim: P::Scalar,
		/// The round polynomial, once `execute` has produced it and while it awaits the challenge
		/// that reduces it to the next claim.
		coeffs: Option<RoundCoeffs<P::Scalar>>,
	},
	/// Binding the bit position, both multilinears now one dense row.
	Bits(SharedSumcheckProver<'alloc, A, P, BivariateProductEvaluator>),
}

impl<'alloc, A: Allocator, F: Field, P: PackedField<Scalar = F>>
	Phase1SumcheckProver<'alloc, A, P>
{
	/// Creates a prover for the claim that `g * h` sums to `sum` over the hypercube.
	///
	/// The outer shift slot is already bound by the time this runs, so `h` is the shift operator
	/// over the weights those rounds left behind, and `g`'s remaining row index is the inner slot.
	///
	/// # Panics
	///
	/// Panics unless `g` and `h` both span one shift slot and the bit position.
	pub fn new(g: SparseShiftRows<P>, h: FieldVec<P, A>, sum: F, alloc: &'alloc A) -> Self {
		assert_eq!(h.log_len(), SHIFT_OPERATOR_LOG_LEN, "h spans one shift slot");
		assert_eq!(g.log_rows(), LOG_SHIFT_COUNT, "g's rows span one shift slot");

		Self {
			alloc,
			stage: Some(Stage::Rows {
				g,
				h,
				claim: sum,
				coeffs: None,
			}),
		}
	}
}

impl<A: Allocator, F: Field, P: PackedField<Scalar = F>> SumcheckProver<F>
	for Phase1SumcheckProver<'_, A, P>
{
	fn n_vars(&self) -> usize {
		match self.stage.as_ref().expect("the stage is set between calls") {
			// `h` spans both axes through the row rounds, so its length is what remains.
			Stage::Rows { h, .. } => h.log_len(),
			Stage::Bits(prover) => prover.n_vars(),
		}
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		match self.stage.as_mut().expect("the stage is set between calls") {
			Stage::Rows {
				g,
				h,
				claim,
				coeffs,
			} => {
				let round_coeffs = shift_round_coeffs(g, h, *claim);
				*coeffs = Some(round_coeffs.clone());
				vec![round_coeffs]
			}
			Stage::Bits(prover) => prover.execute(),
		}
	}

	fn fold(&mut self, challenge: F) {
		// Taken out so the row stage's buffers can move into the bit stage's prover below.
		let stage = self.stage.take().expect("the stage is set between calls");
		self.stage = Some(match stage {
			Stage::Rows {
				mut g,
				mut h,
				coeffs,
				..
			} => {
				let claim = coeffs
					.expect("execute is called before fold")
					.evaluate(&challenge);
				g.fold(challenge);
				fold_highest_var_inplace(&mut h, challenge);

				if g.log_rows() > 0 {
					Stage::Rows {
						g,
						h,
						claim,
						coeffs: None,
					}
				} else {
					// The row index is bound. What is left of each multilinear is one dense row
					// over the bit position, which the shared prover handles from here.
					Stage::Bits(bivariate_product_prover(
						self.alloc,
						[g.into_bit_multilinear(self.alloc), h],
						claim,
					))
				}
			}
			Stage::Bits(mut prover) => {
				prover.fold(challenge);
				Stage::Bits(prover)
			}
		});
	}

	fn finish(self) -> Vec<F> {
		match self.stage.expect("the stage is set between calls") {
			Stage::Rows { .. } => panic!("finish called before the row index was bound"),
			// The columns went in as `[g, h]`, so the evaluations come out in that order.
			Stage::Bits(prover) => prover.finish(),
		}
	}
}

/// Runs the phase 1 sumcheck protocol for shift constraint verification.
///
/// One sumcheck over the product of `g` and `h`, multilinears of [`PHASE_1_LOG_LEN`] variables,
/// binding the outer shift slot, then the inner one, then the bit position. Each slot's variant is
/// the high
/// [`LOG_SHIFT_VARIANT_COUNT`](binius_verifier::protocols::shift::LOG_SHIFT_VARIANT_COUNT)
/// variables of its run, so the sumcheck folds the variant axis rather than the prover summing a
/// separate claim per variant.
///
/// The `g` multilinear carries the witness and the batching randomness; `h` says what each shift
/// *sequence* does at the univariate challenge point. A table of `h` would hold `2^24` entries and
/// is never formed:
///
/// - The [`LOG_SHIFT_COUNT`] rounds binding the outer slot run against [`OuterShiftStage`], which
///   derives each live row from a folded `2^15` table one slice at a time.
/// - Those rounds leave the weights `psi`, and `h` over what remains is the shift operator applied
///   to them — a `2^15` table, which the remaining rounds fold as an ordinary multilinear through
///   [`Phase1SumcheckProver`].
///
/// The outer rounds are driven here rather than folded into [`Phase1SumcheckProver`] because
/// `prove_single` consumes its prover, and `psi` has to outlive it: the rounds binding the
/// intermediate bit index run against those same weights.
///
/// # Arguments
///
/// - `oblong_weights`: the oblong weights of the reduction's first factor, one per bit position.
/// - `sum`: the claim being proved, as the operations' batched evaluations give it. This is the
///   same value the verifier feeds its own sumcheck, so the prover proves the statement it was
///   handed rather than whatever `g · h` happens to come to.
///
/// The two agree exactly when the witness satisfies the constraint system — which is what this
/// reduction exists to establish, so it is not something the prover can check on its way in. On an
/// unsatisfying witness they differ, and that difference travels through both phases to the
/// reduction's final evaluation check, which is where verification fails.
///
/// # Returns
///
/// The challenge point split into the axes the rounds bound, the weights the next phase runs
/// against, and the evaluations this one reduced to.
#[instrument(skip_all, name = "run_sumcheck")]
pub fn run_phase_1_sumcheck<
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
>(
	mut g: SparseShiftRows<P>,
	oblong_weights: &[F],
	sum: F,
	channel: &mut Channel,
	alloc: &A,
) -> Phase1Output<F> {
	assert_eq!(g.log_rows(), LOG_SHIFT_ROWS, "g's rows span both shift slots");

	// The rounds binding the outer slot, driven a round at a time.
	let mut outer = OuterShiftStage::new(alloc, oblong_weights);
	let mut claim = sum;
	let mut outer_point = Vec::with_capacity(LOG_SHIFT_COUNT);
	for _ in 0..LOG_SHIFT_COUNT {
		let round_coeffs = outer.round_coeffs(&g, claim);
		channel.send_many(round_coeffs.clone().truncate().coeffs());
		let challenge = channel.sample();
		claim = round_coeffs.evaluate(&challenge);
		outer.fold(challenge);
		g.fold(challenge);
		outer_point.push(challenge);
	}
	let psi = outer.psi().to_vec();

	// What is left is one shift slot and the bit position, against the operator over those weights.
	let h = shift_operator_table(alloc, &psi);
	let ProveSingleOutput {
		multilinear_evals,
		mut challenges,
	} = prove_single(Phase1SumcheckProver::new(g, h, claim, alloc), channel);

	// Reverse each run of challenges into its evaluation point, whose coordinates then run in
	// increasing order of significance: the bit position within a word, then the inner shift slot's
	// amount and variant, then the outer slot's.
	challenges.reverse();
	assert_eq!(challenges.len(), SHIFT_OPERATOR_LOG_LEN);
	let mut r_j = challenges;
	let r_v_inner = r_j.split_off(Word::LOG_BITS * 2);
	let r_s_inner = r_j.split_off(Word::LOG_BITS);

	outer_point.reverse();
	let mut r_s_outer = outer_point;
	let r_v_outer = r_s_outer.split_off(Word::LOG_BITS);

	let [g_eval, h_eval] = multilinear_evals
		.try_into()
		.expect("prover has 2 multilinear polynomials");

	Phase1Output {
		r_j,
		r_s_inner,
		r_v_inner,
		r_s_outer,
		r_v_outer,
		psi,
		gamma: g_eval * h_eval,
		g_eval,
	}
}

/// Computes one round message of the phase-1 sumcheck over the row index.
///
/// The round polynomial is sampled at 1 and at infinity, as [`RoundEvals`] documents; the claim
/// supplies its value at 0. Both are linear in `g`, so each stored row contributes on its own and
/// rows facing each other across the split never have to be paired:
///
/// ```text
/// R(1)   = sum_v G_1(v) H_1(v)             row (i, c) adds <c, h[i]>, upper half only
/// R(inf) = sum_v (G_0 + G_1)(H_0 + H_1)    row (i, c) adds <c, h[i] + h[i ^ half]>, either half
/// ```
///
/// `h` is dense, so the row facing a stored row is always there to read.
fn shift_round_coeffs<F, P, Data>(
	g: &SparseShiftRows<P>,
	h: &FieldBuffer<P, Data>,
	claim: F,
) -> RoundCoeffs<F>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Data: Deref<Target = [P]>,
{
	let half = g.half();
	let row_len = row_len::<P>();
	let h_rows = h.as_ref();

	// The per-point products accumulate in unreduced (wide) form and reduce once at the end.
	let mut y_1 = <P as WideMul>::Output::default();
	let mut y_inf = <P as WideMul>::Output::default();
	for (index, row) in g.rows() {
		let own = &h_rows[index * row_len..][..row_len];
		let facing = &h_rows[(index ^ half) * row_len..][..row_len];

		for i in 0..row_len {
			// The infinity evaluation reads H(0) + H(1), the same sum from either half, so the
			// row's own half only decides the evaluation at 1.
			if index & half != 0 {
				y_1 += P::wide_mul(row[i], own[i]);
			}
			y_inf += P::wide_mul(row[i], own[i] + facing[i]);
		}
	}

	// A row is a whole number of packed elements, so every lane of the accumulators is live.
	let sum_lanes = |wide| P::reduce(wide).iter().sum::<F>();
	RoundEvals([sum_lanes(y_1), sum_lanes(y_inf)]).interpolate(claim)
}

/// Accumulates the phase-1 "g" rows of one key segment.
///
/// This builds the rows of the g multilinear used in phase 1 of the shift protocol, over the words
/// of a single value-vector segment (public or hidden).
///
/// The value vector's public and hidden words participate through their own [`KeySegment`], so a
/// caller accumulates each segment's rows with the matching words and merges the two results with
/// [`SparseShiftRows::from_segments`].
///
/// # Construction Process
///
/// 1. **Parallel Processing**: Words are processed in parallel chunks for efficiency
/// 2. **Key Processing**: For each word, iterate through its associated keys in the segment
/// 3. **Accumulation**: For each key, accumulate its contribution weighted by the r_x' tensor
/// 4. **Word Expansion**: Expand each witness word bitwise to populate the g rows
/// 5. **Lambda Weighting**: Apply lambda powers to weight different operand positions
///
/// # Returns
///
/// One row of `Word::BITS` scalars per shift the segment uses, addressed by the keys' dense shift
/// index. The segment's [`DenseShiftEncoding`] decodes that index to the `(variant, amount)` pair
/// the row belongs to, and so to the row's place in the full shift space.
///
/// # Usage
///
/// Used in phase 1 to construct the constant size g multilinear
/// that will participate in the phase 1 sumcheck protocol.
#[instrument(skip_all, name = "build_g")]
pub fn build_g<F: Field, P: PackedField<Scalar = F>>(
	words: &[Word],
	segment: &KeySegment,
	prepared: &PreparedOperatorClaims<F>,
) -> Box<[P]> {
	// One row of `Word::BITS` scalars per shift the segment uses.
	let row_len = row_len::<P>();
	let acc_size = segment.dense_shift_enc.len() * row_len;

	const {
		assert!(
			P::WIDTH <= 8,
			"the optimizations below work only when the width of `P` is less than 8 (which is true for all packed 128b fields we use for now)"
		);
	}

	// Map from a u8 with `P::WIDTH` meaningful bits to the lane mask selecting exactly those lanes,
	// precomputed once and reused across every accumulator below.
	let packed_masks_map = (0..1 << P::WIDTH)
		.map(|i| P::make_mask((0..P::WIDTH).map(|bit_index| (i >> bit_index) & 1 == 1)))
		.collect::<Vec<_>>();
	// A mask for low `P::WIDTH` bits.
	let low_bits_mask = (1u8 << P::WIDTH) - 1;

	// Each word carries the keys named by the segment-relative range at its position.
	words
		.par_iter()
		.zip(segment.key_ranges.par_iter())
		.with_min_task(WorkPerItem::FieldMuls)
		.fold(
			|| zeroed_vec::<P>(acc_size).into_boxed_slice(),
			|mut multilinears, (word, Range { start, end })| {
				let keys = &segment.keys[*start as usize..*end as usize];

				for key in keys {
					let operator_data = &prepared[key.operation];

					let acc = key.accumulate(
						&segment.constraint_indices,
						operator_data.r_x_prime_tensor.as_ref(),
						&operator_data.lambda_powers,
					);
					let acc_packed = P::broadcast(acc);

					// The following loop is an optimized version of the following
					// for i in 0..Word::BITS {
					//     if get_bit(word, i) {
					//         values[start + i] += acc;
					//     }
					// }
					debug_assert!(
						(key.dense_shift_idx as usize) < segment.dense_shift_enc.len(),
						"a key indexes the dense shift encoding of its own segment"
					);
					let start = key.dense_shift_idx as usize * row_len;
					let values = &mut multilinears[start..start + row_len];
					let values_per_byte = Word::BYTES >> P::LOG_WIDTH;
					let mut remaining_word = word.0;
					let mut byte_index = 0;
					while remaining_word != 0 {
						let byte = remaining_word as u8;
						let byte_values =
							&mut values[byte_index * values_per_byte..][..values_per_byte];
						for value_index in 0..(8 >> P::LOG_WIDTH) {
							unsafe {
								let packed_mask_index =
									((byte >> (value_index * P::WIDTH)) & low_bits_mask) as usize;

								// Safety:
								// - `packed_masks_map` is guaranteed to have enough elements to be
								//   indexed with a `P::WIDTH`-bits value.
								let packed_mask = packed_masks_map.get_unchecked(packed_mask_index);

								// Safety:
								// - `values` is guaranteed to be (8 >> P::LOG_WIDTH) elements long
								//   due to the chunking
								// - `value_index` is always in bounds because we iterate over 0..(8
								//   >> P::LOG_WIDTH)
								*byte_values.get_unchecked_mut(value_index) +=
									acc_packed.select(packed_mask);
							}
						}
						remaining_word >>= 8;
						byte_index += 1;
					}
				}

				multilinears
			},
		)
		// A merge seeded with a partial that already exists never touches a buffer of zeros.
		// An identity would allocate and zero one accumulator per merge, then add all of it.
		.reduce_with(|mut acc, local| {
			izip!(acc.iter_mut(), local.iter()).for_each(|(acc, local)| {
				*acc += *local;
			});
			acc
		})
		// An empty word list yields no partials at all, and its rows are zero.
		.unwrap_or_else(|| zeroed_vec::<P>(acc_size).into_boxed_slice())
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_core::constraint_system::{
		AndConstraint, ConstraintSystem, InoutSegment, Shift, ShiftedValueIndex, ValueIndex,
	};
	use binius_field::{BinaryField128bGhash, Field, PackedBinaryGhash2x128b};
	use binius_math::{inner_product::inner_product_buffers, test_utils::random_scalars};
	use binius_transcript::ProverTranscript;
	use binius_verifier::config::StdChallenger;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::protocols::shift::key_collection::build_key_collection;

	type F = BinaryField128bGhash;

	impl<P: PackedField> SparseShiftRows<P> {
		/// Spreads the rows over the space they still span, as a dense multilinear. Rows at a
		/// repeated index add up; every row the system does not name stays zero.
		///
		/// The prover never needs this — [`Phase1SumcheckProver`] reads the rows where they are.
		/// It is the dense `g` these tests measure the sparse rounds against.
		fn scatter<A: Allocator>(&self, alloc: &A) -> FieldVec<P, A> {
			let row_len = row_len::<P>();
			let mut g = FieldBuffer::zeros_in(alloc, self.log_rows + Word::LOG_BITS);
			for (index, row) in self.rows() {
				// A row is a whole number of packed elements, so it lands at row alignment.
				let slots = &mut g.as_mut()[index * row_len..][..row_len];
				for (slot, &value) in iter::zip(slots, row) {
					*slot += value;
				}
			}
			g
		}
	}

	/// A system whose two segments name overlapping but distinct shifts.
	///
	/// The public segment names `(Sll, 0)` and `(Slr, 3)`; the hidden one `(Sll, 0)`, `(Sar, 7)`
	/// and `(Rotr, 1)`. So `(Sll, 0)` is the shift the merge has to sum across segments.
	fn overlapping_shift_system() -> ConstraintSystem {
		let public = ValueIndex::constant(1);
		let hidden = ValueIndex::private(1);
		ConstraintSystem {
			constants: vec![Word::ZERO; 4],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: vec![AndConstraint([
				vec![
					ShiftedValueIndex::plain(public),
					ShiftedValueIndex::srl(public, 3),
				],
				vec![ShiftedValueIndex::sar(hidden, 7)],
				vec![
					ShiftedValueIndex::rotr(hidden, 1),
					ShiftedValueIndex::plain(hidden),
				],
			])],
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		}
	}

	/// The two segments' rows concatenate, each tagged with the shift index it sits at.
	///
	/// A shift both segments name appears twice rather than being merged — `g` is the sum of its
	/// rows, so the two add up wherever it is read, and nothing has to deduplicate them.
	#[test]
	fn from_segments_concatenates_the_two_encodings() {
		let key_collection =
			build_key_collection(&overlapping_shift_system(), InoutSegment::Public);

		// Fill each segment's rows with a distinct constant per row, so a row's value says which
		// segment it came from.
		let segment_rows = |enc: &DenseShiftEncoding, base: u128| {
			(0..enc.len() * Word::BITS)
				.map(|i| F::new(base + (i / Word::BITS) as u128))
				.collect::<Vec<F>>()
		};
		let public = segment_rows(&key_collection.public.dense_shift_enc, 0x100);
		let hidden = segment_rows(&key_collection.hidden.dense_shift_enc, 0x200);

		let g = SparseShiftRows::from_segments([
			(&public, &key_collection.public.dense_shift_enc),
			(&hidden, &key_collection.hidden.dense_shift_enc),
		]);

		// The public segment's two shifts, then the hidden segment's three. Every term here is
		// singly shifted, so its outer slot is the identity and its quadruple index is its inner
		// shift's. `(Sll, 0)` is the first row of both segments, so index 0 appears twice.
		let row_index = |shift: Shift| shift.index() as u32;
		assert_eq!(
			g.indices,
			[
				row_index(Shift::IDENTITY),
				row_index(Shift::srl(3)),
				row_index(Shift::IDENTITY),
				row_index(Shift::sar(7)),
				row_index(Shift::rotr(1)),
			]
		);

		// Each row is the one its own segment accumulated, untouched by the other's.
		let row = |position: usize| &g.values[position * Word::BITS..][..Word::BITS];
		for (position, expected) in [0x100, 0x101, 0x200, 0x201, 0x202].into_iter().enumerate() {
			assert!(row(position).iter().all(|&value| value == F::new(expected)));
		}

		// Where `g` is read, the two rows at the identity add up.
		let at = |shift: Shift| {
			g.rows()
				.filter(|&(index, _)| index == shift.index())
				.map(|(_, row)| row[0])
				.sum::<F>()
		};
		assert_eq!(at(Shift::IDENTITY), F::new(0x100) + F::new(0x200));
		assert_eq!(at(Shift::srl(3)), F::new(0x101));
		assert_eq!(at(Shift::sar(7)), F::new(0x201));
		assert_eq!(at(Shift::rotr(1)), F::new(0x202));
	}

	/// A sequence is placed outer-major, so the outer slot lands where the first rounds bind it.
	#[test]
	fn a_sequence_is_keyed_outer_major() {
		let hidden = ValueIndex::private(1);
		let sequence = [Shift::srl(3), Shift::sll(5)];
		let cs = ConstraintSystem {
			constants: vec![Word::ZERO; 4],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: vec![AndConstraint([
				vec![ShiftedValueIndex::new(hidden, sequence)],
				Vec::new(),
				Vec::new(),
			])],
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		};

		let key_collection = build_key_collection(&cs, InoutSegment::Public);
		let [inner, outer] = sequence;
		assert_eq!(
			key_collection
				.hidden
				.dense_shift_enc
				.shift_indices()
				.collect::<Vec<_>>(),
			[outer.index() << LOG_SHIFT_COUNT | inner.index()]
		);
	}

	/// The scatter puts every row where its shift index names, and leaves the rest zero.
	///
	/// The outer rounds are past by the time the dense reference below is taken, so the rows span
	/// one shift slot here rather than the quadruple `from_segments` keys on.
	#[test]
	fn scatter_places_rows_at_their_shift_index() {
		let indices = [Shift::IDENTITY, Shift::sar(7), Shift::rotr(1)]
			.map(|shift| shift.index() as u32)
			.to_vec();
		let values = (0..indices.len() * Word::BITS)
			.map(|i| F::new(1 + (i / Word::BITS) as u128))
			.collect::<Vec<F>>();
		let rows = SparseShiftRows::<F>::new(indices.clone(), values, LOG_SHIFT_COUNT);

		let g = rows.scatter(&GlobalAllocator);
		assert_eq!(g.log_len(), SHIFT_OPERATOR_LOG_LEN);

		// Exactly the named rows are non-zero, and each holds what the list held.
		for (row, &shift_index) in indices.iter().enumerate() {
			let offset = shift_index as usize * Word::BITS;
			for bit in 0..Word::BITS {
				assert_eq!(g.get(offset + bit), F::new(1 + row as u128));
			}
		}
		for row in (0..1 << LOG_SHIFT_COUNT).filter(|row| !indices.contains(&(*row as u32))) {
			assert!((0..Word::BITS).all(|bit| g.get(row * Word::BITS + bit) == F::ZERO));
		}
	}

	/// Runs the sparse row and bit rounds the dense way: one bivariate-product prover over the
	/// scattered `g` and the whole of `h`, with no round of it special-cased.
	///
	/// This is what [`Phase1SumcheckProver`] replaces, kept here as the reference its round
	/// messages have to reproduce.
	fn run_dense_reference<P: PackedField<Scalar = F>>(
		g: &SparseShiftRows<P>,
		h: FieldVec<P, GlobalAllocator>,
		sum: F,
		channel: &mut ProverTranscript<StdChallenger>,
	) -> (Vec<F>, F) {
		let prover =
			bivariate_product_prover(&GlobalAllocator, [g.scatter(&GlobalAllocator), h], sum);

		let ProveSingleOutput {
			multilinear_evals,
			mut challenges,
		} = prove_single(prover, channel);
		challenges.reverse();

		let [g_eval, h_eval] = multilinear_evals
			.try_into()
			.expect("prover has 2 multilinear polynomials");

		(challenges, g_eval * h_eval)
	}

	/// The `g` and `h` the row and bit rounds run over, from pseudo-random weights.
	///
	/// The outer rounds are past by this point, so `g`'s rows sit at the inner shift each sequence
	/// names. They carry arbitrary values rather than ones a witness produces: the sumcheck is a
	/// statement about whatever multilinears it is handed, so what the rows hold does not bear on
	/// whether the sparse rounds reproduce the dense ones.
	fn phase_1_multilinears<P: PackedField<Scalar = F>>(
		cs: &ConstraintSystem,
		seed: u64,
	) -> (SparseShiftRows<P>, FieldVec<P, GlobalAllocator>) {
		let mut rng = StdRng::seed_from_u64(seed);
		let key_collection = build_key_collection(cs, InoutSegment::Public);

		let mut indices = Vec::new();
		let mut values = Vec::new();
		for segment in [&key_collection.public, &key_collection.hidden] {
			for [inner, _] in segment.dense_shift_enc.iter() {
				indices.push(inner.index() as u32);
				values.extend((0..row_len::<P>()).map(|_| P::random(&mut rng)));
			}
		}
		let g = SparseShiftRows::new(indices, values, LOG_SHIFT_COUNT);

		// The weights `h` is built from are arbitrary here for the same reason `g`'s rows are.
		let h = shift_operator_table(&GlobalAllocator, &random_scalars::<F>(&mut rng, Word::BITS));

		(g, h)
	}

	/// The sparse rounds send exactly what the dense prover sends.
	///
	/// The two provers sum the same product over the same hypercube, so every round message — and
	/// therefore the whole transcript, the challenges it draws, and the evaluation it reduces to —
	/// must agree. This is what lets the verifier stay untouched.
	fn assert_sparse_matches_dense<P: PackedField<Scalar = F>>(cs: &ConstraintSystem, seed: u64) {
		let (g, h) = phase_1_multilinears::<P>(cs, seed);
		// The true sum, so the test exercises a sumcheck a verifier would accept.
		let sum = inner_product_buffers(&g.scatter(&GlobalAllocator), &h);

		let mut sparse_transcript = ProverTranscript::<StdChallenger>::default();
		let ProveSingleOutput {
			multilinear_evals,
			challenges: mut sparse_challenges,
		} = prove_single(
			Phase1SumcheckProver::new(g.clone(), h.clone(), sum, &GlobalAllocator),
			&mut sparse_transcript,
		);
		sparse_challenges.reverse();
		let [g_eval, h_eval] = multilinear_evals
			.try_into()
			.expect("prover has 2 multilinear polynomials");

		let mut dense_transcript = ProverTranscript::<StdChallenger>::default();
		let (dense_challenges, dense_eval) = run_dense_reference(&g, h, sum, &mut dense_transcript);

		assert_eq!(sparse_challenges, dense_challenges);
		assert_eq!(g_eval * h_eval, dense_eval);
		assert_eq!(sparse_transcript.finalize(), dense_transcript.finalize());
	}

	#[test]
	fn sparse_rounds_match_the_dense_prover() {
		// Both packing widths the prover is instantiated at: the m4 prover drives phase 1 with
		// scalars, the single-instance prover with a packed field.
		assert_sparse_matches_dense::<F>(&overlapping_shift_system(), 0);
		assert_sparse_matches_dense::<PackedBinaryGhash2x128b>(&overlapping_shift_system(), 1);
	}

	/// A constraint system that constrains nothing names no shift, so `g` stores no row at all.
	///
	/// The sparse rounds then have nothing to scan, which must still leave the transcript the one
	/// a dense prover over the zero `g` would have written.
	#[test]
	fn sparse_rounds_match_the_dense_prover_with_an_empty_g() {
		let cs = ConstraintSystem {
			constants: vec![Word::ZERO; 4],
			n_inout: 0,
			n_private: 4,
			zero_constraints: Vec::new(),
			and_constraints: Vec::new(),
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		};

		let key_collection = build_key_collection(&cs, InoutSegment::Public);
		assert!(key_collection.public.dense_shift_enc.is_empty());
		assert!(key_collection.hidden.dense_shift_enc.is_empty());

		assert_sparse_matches_dense::<PackedBinaryGhash2x128b>(&cs, 2);
	}
}
