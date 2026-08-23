// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, ops::Deref};

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
use binius_verifier::protocols::shift::LOG_SHIFT_COUNT;
use tracing::instrument;

use super::{
	key_collection::DenseShiftEncoding, monster::shift_operator_table, outer::OuterShiftStage,
	shift_ind::ShiftChallenge,
};

/// The number of variables the shift-and-bit phases of the reduction span: the bit position
/// within a word, the inner shift slot, and the outer shift slot.
///
/// No table the prover ever builds spans all three axes at once — avoiding that table is why
/// the outer-slot rounds exist.
pub const PHASE_1_LOG_LEN: usize = Word::LOG_BITS + LOG_SHIFT_ROWS;

/// The number of variables of the row-index axis: two shift slots, since a term names two
/// shifts applied in sequence.
///
/// The slots are ordered outer-major, so the rounds binding this axis peel the outer shift
/// first.
pub const LOG_SHIFT_ROWS: usize = 2 * LOG_SHIFT_COUNT;

/// The number of variables one shift-weight table spans: one row of weights per (shift
/// variant, shift amount) pair, one weight per bit position within a word.
pub const SHIFT_OPERATOR_LOG_LEN: usize = Word::LOG_BITS + LOG_SHIFT_COUNT;

/// The output of the first proving phase.
///
/// The challenge point the phase's rounds bound, split into its axes, plus the two
/// evaluations the reduction still needs to carry forward.
#[derive(Debug, Clone)]
pub struct Phase1Output<F> {
	/// The bit position within a word.
	pub r_j: Vec<F>,
	/// The inner shift's amount and variant.
	pub inner: ShiftChallenge<F>,
	/// The outer shift's amount and variant.
	pub outer: ShiftChallenge<F>,
	/// The weights carried through the outer shift, at the outer challenge point.
	///
	/// Read by the next phase's bit-index rounds.
	pub psi: Vec<F>,
	/// The evaluation claim the next phase proves: the product of the two evaluations below.
	pub gamma: F,
	/// The witness-and-batching multilinear, evaluated at the bound shift and bit point.
	///
	/// Carried through the remaining phases' rounds, so it is never recomputed.
	pub g_eval: F,
}

/// The number of packed elements one row of `Word::BITS` scalars occupies.
pub(super) const fn row_len<P: PackedField>() -> usize {
	assert!(
		P::LOG_WIDTH <= Word::LOG_BITS,
		"a row of `Word::BITS` scalars must be a whole number of packed elements"
	);
	Word::BITS >> P::LOG_WIDTH
}

/// The nonzero rows of the witness-and-batching multilinear that a constraint system's shifts
/// reach: one row of weights per (shift variant, shift amount) pair actually named, each
/// tagged with its row index.
///
/// A constraint system only ever names a few dozen pairs — 16 for a SHA-256 circuit, 40 for
/// Keccak.
///
/// Rows at a repeated index add up wherever the multilinear is read, so two segments' rows can
/// simply concatenate here with no deduplication step.
#[derive(Debug, Clone)]
pub struct SparseShiftRows<P: PackedField> {
	/// The row index of each stored row, in the same order as the row values below.
	///
	/// An index can repeat.
	indices: Vec<u32>,
	/// The stored rows end to end, in the same order as the indices above.
	values: Vec<P>,
	/// The number of row-index variables the sumcheck has yet to bind.
	log_rows: usize,
}

impl<P: PackedField> SparseShiftRows<P> {
	/// Collects the rows two key segments accumulated, tagged with the shift index each sits at.
	///
	/// Each segment's rows arrive in its own dense encoding order, which decodes each position
	/// back to the shift index this list keys on.
	/// The two segments' rows simply concatenate: a shift both use appears twice, and the two
	/// rows add up wherever the multilinear is read later.
	///
	/// # Panics
	///
	/// Panics unless each segment's row count matches what its encoding accounts for.
	pub fn from_segments(segments: [(&[P], &DenseShiftEncoding); 2]) -> Self {
		let mut indices = Vec::new();
		let mut values = Vec::new();

		// Each segment contributes its own rows, tagged with its own shift indices.
		for (rows, dense_shift_enc) in segments {
			assert_eq!(
				rows.len(),
				dense_shift_enc.len() * row_len::<P>(),
				"a segment holds one row per shift its encoding names"
			);
			// Decode each stored row's position back to the shift index it belongs to.
			indices.extend(dense_shift_enc.shift_indices().map(|index| index as u32));
			// Rows just concatenate: a shift both segments use simply appears twice.
			values.extend_from_slice(rows);
		}

		Self::new(indices, values, LOG_SHIFT_ROWS)
	}

	/// Collects stored rows sitting at the given indices of a row space `log_rows` variables
	/// wide.
	///
	/// An index can repeat: rows at a repeated index add up wherever the multilinear is read
	/// later.
	///
	/// # Panics
	///
	/// Panics unless there is one row of values per index and every index fits the row space.
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
	/// # Panics
	///
	/// Panics unless at least one row-index variable remains to bind.
	pub(crate) fn half(&self) -> usize {
		assert!(self.log_rows > 0, "precondition: a row-index variable remains to bind");
		1 << (self.log_rows - 1)
	}

	/// Binds the highest row-index variable to a challenge.
	///
	/// Folding is linear, so a row keeps its identity across it: it is scaled by the challenge
	/// weight of its half, then moved down into the folded index space.
	/// The list stays the same length, with nothing paired up or merged.
	///
	/// # Panics
	///
	/// Panics unless at least one row-index variable remains to bind.
	pub(crate) fn fold(&mut self, challenge: P::Scalar) {
		let half = self.half();
		let lower_weight = P::broadcast(P::Scalar::ONE - challenge);
		let upper_weight = P::broadcast(challenge);

		let row_len = row_len::<P>();
		// Every stored row moves on its own: the fold is linear, so no row needs its
		// counterpart from the other half to update.
		for (position, index) in self.indices.iter_mut().enumerate() {
			let row = &mut self.values[position * row_len..][..row_len];
			if *index as usize & half == 0 {
				// Lower half: scale by `1 - challenge` and keep the same index.
				row.iter_mut().for_each(|value| *value *= lower_weight);
			} else {
				// Upper half: scale by `challenge` and fold the index into the lower half.
				row.iter_mut().for_each(|value| *value *= upper_weight);
				*index ^= half as u32;
			}
		}

		self.log_rows -= 1;
	}

	/// Collapses every remaining row-index variable into one dense row over the bit position:
	/// the sum of the stored rows, now that they all sit at the same index.
	///
	/// # Panics
	///
	/// Panics unless every row-index variable is already bound.
	fn into_bit_multilinear<A: Allocator>(self, alloc: &A) -> FieldVec<P, A> {
		assert_eq!(self.log_rows, 0, "precondition: every row-index variable is bound");

		let mut multilinear = FieldBuffer::zeros_in(alloc, Word::LOG_BITS);
		// Every stored row now sits at the same index, so they all add into one dense row.
		for (_, row) in self.rows() {
			for (slot, &value) in iter::zip(multilinear.as_mut(), row) {
				*slot += value;
			}
		}
		multilinear
	}

	/// Computes one round message of the phase-1 sumcheck over the row index.
	///
	/// Sampled at 1 and at infinity, with the claim supplying the value at 0.
	/// Both samples are linear in the stored rows, so each row contributes independently and
	/// facing rows across the split never have to be paired up.
	///
	/// ```text
	/// R(1)   = sum_v G_1(v) H_1(v)             row (i, c) adds <c, h[i]>, upper half only
	/// R(inf) = sum_v (G_0 + G_1)(H_0 + H_1)    row (i, c) adds <c, h[i] + h[i ^ half]>, either half
	/// ```
	fn round_coeffs<F, Data>(&self, h: &FieldBuffer<P, Data>, claim: F) -> RoundCoeffs<F>
	where
		F: Field,
		P: PackedField<Scalar = F>,
		Data: Deref<Target = [P]>,
	{
		let half = self.half();
		let row_len = row_len::<P>();
		let h_rows = h.as_ref();

		// The per-point products accumulate in unreduced (wide) form and reduce once at the
		// end.
		let mut y_1 = <P as WideMul>::Output::default();
		let mut y_inf = <P as WideMul>::Output::default();
		for (index, row) in self.rows() {
			let own = &h_rows[index * row_len..][..row_len];
			let facing = &h_rows[(index ^ half) * row_len..][..row_len];

			for i in 0..row_len {
				// The infinity evaluation reads H(0) + H(1), the same sum from either half.
				// So a row's own half only decides its contribution to the evaluation at 1.
				if index & half != 0 {
					y_1 += P::wide_mul(row[i], own[i]);
				}
				y_inf += P::wide_mul(row[i], own[i] + facing[i]);
			}
		}

		// A row is a whole number of packed elements, so every lane of the accumulators is
		// live.
		let sum_lanes = |wide| P::reduce(wide).iter().sum::<F>();
		RoundEvals([sum_lanes(y_1), sum_lanes(y_inf)]).interpolate(claim)
	}

	/// Runs the phase-1 sumcheck over the product of this row list and a weight table.
	///
	/// This row list is zero outside the rows a constraint system's shifts name, dense within
	/// a named row.
	/// A weight table holding one row per possible shift sequence would need `2^24` entries
	/// and is never built:
	///
	/// - The rounds binding the outer shift slot read only the stored rows, so their cost follows
	///   the shifts the constraint system names, not the whole space.
	/// - Once the outer slot is bound, both multilinears are one dense row, and a shared
	///   dense-product prover runs the remaining rounds.
	///
	/// Every round message matches what a fully dense prover would send.
	///
	/// The outer rounds run here, rather than inside that dense-product prover, because the
	/// weights they leave behind must outlive it: the next phase's rounds run against those
	/// same weights.
	///
	/// # Arguments
	///
	/// - `oblong_weights`: the weights of the reduction's first factor, one per bit position.
	/// - `sum`: the claim being proved, which must equal the true product exactly when the witness
	///   satisfies the constraint system.
	///
	/// # Returns
	///
	/// The challenge point split into its axes, the leftover weights, and the two evaluations
	/// this phase reduced to.
	#[instrument(skip_all, name = "run_sumcheck")]
	pub fn run_phase_1_sumcheck<F, Channel, A>(
		mut self,
		oblong_weights: &[F],
		sum: F,
		channel: &mut Channel,
		alloc: &A,
	) -> Phase1Output<F>
	where
		F: BinaryField,
		P: PackedField<Scalar = F>,
		Channel: IPProverChannel<F>,
		A: Allocator,
	{
		assert_eq!(self.log_rows(), LOG_SHIFT_ROWS, "the row list spans both shift slots");

		// Phase 1a: bind the outer shift slot, one round at a time.
		//
		// Each round asks the outer-slot driver for the round polynomial, sends it, samples
		// the challenge, then folds both the driver and the row list by that challenge.
		let mut outer = OuterShiftStage::new(alloc, oblong_weights);
		let mut claim = sum;
		let mut outer_point = Vec::with_capacity(LOG_SHIFT_COUNT);
		for _ in 0..LOG_SHIFT_COUNT {
			let round_coeffs = outer.round_coeffs(&self, claim);
			channel.send_many(round_coeffs.clone().truncate().coeffs());
			let challenge = channel.sample();
			claim = round_coeffs.evaluate(&challenge);
			outer.fold(challenge);
			self.fold(challenge);
			outer_point.push(challenge);
		}
		// The weights the outer rounds leave behind: what the next phase runs against.
		let psi = outer.psi().to_vec();

		// Phase 1b: bind the inner shift slot and the bit position.
		//
		// What is left is one shift slot and the bit position, against the weight table built
		// from the outer rounds' leftover weights.
		let h = shift_operator_table(alloc, &psi);

		// The row list itself becomes the sparse half of the row-and-bit sumcheck prover.
		let g = self;
		let ProveSingleOutput {
			multilinear_evals,
			mut challenges,
		} = prove_single(Phase1SumcheckProver::new(g, h, claim, alloc), channel);

		// The rounds bind coordinates from the most significant one down.
		// Reversing recovers the evaluation point in increasing order of significance: the bit
		// position, then the inner slot's amount and variant, then the outer slot's.
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
			inner: ShiftChallenge::new(r_s_inner, r_v_inner),
			outer: ShiftChallenge::new(r_s_outer, r_v_outer),
			psi,
			gamma: g_eval * h_eval,
			g_eval,
		}
	}
}

/// A sumcheck prover for the product of a sparse row list and a dense weight table, both
/// spanning one shift slot and the bit position.
///
/// The row list is zero outside the rows a constraint system's shifts name, dense within a
/// named row.
/// So this prover changes strategy halfway through:
///
/// - While the row index has unbound variables, each round reads only the stored rows, at a cost
///   that follows how many shifts the constraint system names, not the whole row space.
/// - Once the row index is bound, both multilinears are one dense row, and a shared dense-product
///   prover takes over.
///
/// Every round message matches what a fully dense prover would send.
///
/// # Performance
///
/// Both the sampled evaluations and the fold are linear in the sparse row list, so each
/// stored row contributes independently, walking whole packed rows rather than individual
/// scalar points.
pub struct Phase1SumcheckProver<'alloc, A: Allocator, P: PackedField> {
	alloc: &'alloc A,
	/// The stage the protocol is in.
	///
	/// Always holds a value between calls, only briefly emptied while the row-stage buffers
	/// move into the bit-stage prover.
	stage: Option<Stage<'alloc, A, P>>,
}

/// Which half of the protocol the prover is in.
enum Stage<'alloc, A: Allocator, P: PackedField> {
	/// Binding the row index, over the rows the sparse list stores.
	Rows {
		g: SparseShiftRows<P>,
		h: FieldVec<P, A>,
		/// This round's sum claim.
		claim: P::Scalar,
		/// The round polynomial.
		///
		/// Set once the round's message is produced, and cleared again once the challenge
		/// that reduces it to the next claim arrives.
		coeffs: Option<RoundCoeffs<P::Scalar>>,
	},
	/// Binding the bit position, with both multilinears now one dense row.
	Bits(SharedSumcheckProver<'alloc, A, P, BivariateProductEvaluator>),
}

impl<'alloc, A: Allocator, F: Field, P: PackedField<Scalar = F>>
	Phase1SumcheckProver<'alloc, A, P>
{
	/// Creates a prover for the claim that the two multilinears' product sums to the given
	/// value over the hypercube.
	///
	/// The outer shift slot is already bound by the time this runs, so the dense multilinear
	/// is the weight table over the weights those rounds left behind, and the row list's
	/// remaining row index is the inner slot.
	///
	/// # Panics
	///
	/// Panics unless both multilinears span exactly one shift slot and the bit position.
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
			// The weight table spans both axes through the row rounds.
			// So its length is what remains.
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
				// Row stage: compute this round's message from the stored rows alone, and
				// hold onto it until the challenge that reduces the claim arrives.
				let round_coeffs = g.round_coeffs(h, *claim);
				*coeffs = Some(round_coeffs.clone());
				vec![round_coeffs]
			}
			// Bit stage: delegate to the shared dense-product prover.
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
				// Fold both multilinears by the same challenge.
				// The sparse rows move into their half's slot of the shrunken row space.
				// The dense weight table folds its highest variable the ordinary way.
				g.fold(challenge);
				fold_highest_var_inplace(&mut h, challenge);

				if g.log_rows() > 0 {
					// The row index still has unbound variables: stay in the row stage.
					Stage::Rows {
						g,
						h,
						claim,
						coeffs: None,
					}
				} else {
					// The row index is bound.
					// What is left of each multilinear is one dense row over the bit
					// position, which the shared prover handles from here.
					Stage::Bits(bivariate_product_prover(
						self.alloc,
						[g.into_bit_multilinear(self.alloc), h],
						claim,
					))
				}
			}
			Stage::Bits(mut prover) => {
				// Bit stage: delegate the fold to the shared dense-product prover.
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
	use crate::protocols::shift::KeyCollection;

	type F = BinaryField128bGhash;

	impl<P: PackedField> SparseShiftRows<P> {
		/// Spreads the rows over the space they still span, as a dense multilinear.
		///
		/// Rows at a repeated index add up.
		/// Every row the constraint system does not name stays zero.
		///
		/// Nothing in the actual proving path needs this.
		/// It exists only so these tests have a dense reference to check the sparse rounds
		/// against.
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
			KeyCollection::build(&overlapping_shift_system(), InoutSegment::Public);

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

		let key_collection = KeyCollection::build(&cs, InoutSegment::Public);
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

	/// Runs the same claim through a single dense product-sumcheck prover, with no round handled
	/// specially.
	///
	/// This is the reference the sparse-and-dense hybrid prover's round messages have to
	/// reproduce.
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
		let key_collection = KeyCollection::build(cs, InoutSegment::Public);

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

		let key_collection = KeyCollection::build(&cs, InoutSegment::Public);
		assert!(key_collection.public.dense_shift_enc.is_empty());
		assert!(key_collection.hidden.dense_shift_enc.is_empty());

		assert_sparse_matches_dense::<PackedBinaryGhash2x128b>(&cs, 2);
	}
}
