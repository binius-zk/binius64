// Copyright 2026 The Binius Developers

//! The rounds binding the outer slot of a shift sequence.
//!
//! A shifted value index names a word together with two shifts applied in sequence, and the shift
//! reduction peels them from the output end inward. These are the rounds that peel the outer one.

use std::iter;

use binius_compute::Allocator;
use binius_core::{ShiftVariant, word::Word};
use binius_field::{BinaryField, Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_ip_prover::sumcheck::round_evals::RoundEvals;
use binius_math::{FieldVec, multilinear::fold::fold_highest_var_inplace};
use binius_verifier::protocols::shift::{LOG_SHIFT_COUNT, SHIFT_COUNT};

use super::{
	monster::{shift_operator_row, shift_operator_table},
	phase_1::SparseShiftRows,
};

/// The `(variant, amount)` pair a shift index names.
///
/// This inverts [`Shift::index`](binius_core::constraint_system::Shift::index) over the reduction's
/// index space rather than over well-formed shifts: the amount axis spans `Word::BITS` for every
/// variant, so a half-word (`*32`) variant's index may carry an amount no `Shift` of that variant
/// could hold. Such an index is a hypercube vertex the sumcheck ranges over all the same.
///
/// # Panics
///
/// Panics unless the index is below [`SHIFT_COUNT`].
fn decode_shift(index: usize) -> (ShiftVariant, usize) {
	assert!(index < SHIFT_COUNT, "a shift index names one slot's spelling");
	let variant = ShiftVariant::from_u8((index >> Word::LOG_BITS) as u8)
		.expect("an index below SHIFT_COUNT has a variant field below the variant count");
	(variant, index % Word::BITS)
}

/// The sumcheck rounds binding the outer shift of a sequence, against the folded oblong table.
///
/// The reduction's `h` factor spans 24 variables — the bit position and both shift slots — so its
/// value table would hold `2^24` entries. It is never formed. Instead, writing `T` for the shift
/// operator ([`shift_operator_table`]) and `d` for the oblong weights,
///
/// ```text
///     eta := T[d]                                    2^15 entries
///     h(J, s_2, o_2, s_1, o_1) = T[eta(., s_2, o_2)](J, s_1, o_1)
/// ```
///
/// so `h` is reached by applying `T` to the oblong weights, taking the outer slice, and applying
/// `T` again. This stage holds `eta` and folds it as the rounds bind the outer slot; each round's
/// `h` rows are derived from the folded table, one slice at a time. Folding `eta` first and
/// applying `T` after gives the same answer as folding `h`, because `T` is linear in its weights.
///
/// # Why the outer slot binds first
///
/// Two independent reasons:
///
/// - **Correctness.** The two indicator matrices do not commute — `sra` is the obstruction, since
///   it is the one shift whose vacated positions all read a single input bit. Nesting `T` inside
///   `T` composes them in the order the slots are bound, so only binding the outer slot first
///   computes `h` rather than its transpose-order counterpart.
/// - **Cost.** The push-through is a *shift* only while the inner pair is still a cube index. Under
///   the opposite order the outer indicator would arrive folded to a dense `2^6 x 2^6` matrix, and
///   each live shift quadruple would cost a matrix-vector product in place of a shift.
pub struct OuterShiftStage<F: Field, A: Allocator> {
	/// `eta`, the oblong weights pushed through one shift, folded over the outer variables bound
	/// so far.
	///
	/// The axes run, from the low index positions up: the intermediate bit index, then the outer
	/// shift amount, then the outer shift variant. So binding the highest variable takes the outer
	/// variant before the outer amount, which is the order the reduction's rounds run in.
	eta: FieldVec<F, A>,
}

impl<F: BinaryField, A: Allocator> OuterShiftStage<F, A> {
	/// Pushes the oblong weights through every shift, ready for the first round.
	///
	/// # Panics
	///
	/// Panics unless the weights hold one entry per bit position of a word.
	pub fn new(alloc: &A, oblong_weights: &[F]) -> Self {
		Self {
			eta: shift_operator_table::<F, F, A>(alloc, oblong_weights),
		}
	}

	/// The number of outer-index variables the stage has yet to bind.
	pub const fn n_vars_remaining(&self) -> usize {
		self.eta.log_len() - Word::LOG_BITS
	}

	/// The folded table at one outer index: `eta(., outer)` over the intermediate bit index.
	fn stride(&self, outer: usize) -> &[F] {
		&self.eta.as_ref()[outer * Word::BITS..][..Word::BITS]
	}

	/// The weights the inner rounds run against: `eta` at the bound outer point.
	///
	/// The terminal fold of `eta` *is* the partial evaluation
	/// `sum_i d(i) * shift-ind~(i, K, r_s2, r_o2)`, the multilinear extension commuting with the
	/// finite sum over `i`. So the inner rounds need no division and no second pass.
	///
	/// ## Preconditions
	///
	/// * `self.n_vars_remaining() == 0`
	pub fn psi(&self) -> &[F] {
		assert_eq!(self.n_vars_remaining(), 0, "precondition: every outer variable is bound");
		self.eta.as_ref()
	}

	/// Binds the highest outer variable to a challenge.
	///
	/// ## Preconditions
	///
	/// * `self.n_vars_remaining() >= 1`
	pub fn fold(&mut self, challenge: F) {
		assert!(self.n_vars_remaining() > 0, "precondition: an outer variable remains to bind");
		fold_highest_var_inplace(&mut self.eta, challenge);
	}

	/// Computes one round message: the degree-2 round polynomial binding the next outer variable.
	///
	/// The round polynomial is sampled at 1 and at infinity, as [`RoundEvals`] documents; the claim
	/// supplies its value at 0. Both are linear in `g`, so each stored row contributes on its own
	/// and rows facing each other across the split never have to be paired:
	///
	/// ```text
	/// R(1)   = sum_v G_1(v) H_1(v)             row (i, c) adds <c, h[i]>, upper half only
	/// R(inf) = sum_v (G_0 + G_1)(H_0 + H_1)    row (i, c) adds <c, h[i] + h[i ^ half]>, either half
	/// ```
	///
	/// Unlike the rounds that follow, the `h` rows are not read from a table but derived: a row's
	/// is one slice of the shift operator applied to one stride of the folded `eta`, which costs
	/// `O(2^6)`. A round therefore costs `O(2^6 * n_shift)` in the number of live shift quadruples,
	/// with no charge proportional to the space they are drawn from.
	///
	/// ## Preconditions
	///
	/// * `g`'s row index is a shift quadruple, the outer slot above the inner one
	/// * `g` and this stage have the same number of outer variables left to bind
	pub fn round_coeffs<P: PackedField<Scalar = F>>(
		&self,
		g: &SparseShiftRows<P>,
		claim: F,
	) -> RoundCoeffs<F> {
		assert_eq!(
			g.log_rows(),
			self.n_vars_remaining() + LOG_SHIFT_COUNT,
			"precondition: the rows and the folded table agree on the outer index"
		);

		// The bit this round binds is an outer one, since the outer slot sits above the inner one
		// in a quadruple. Dropping the inner slot off it leaves the bit that indexes eta's strides.
		let half = g.half();
		let facing_half = half >> LOG_SHIFT_COUNT;

		// One scratch row per side, rewritten for each stored row: `shift_operator_row` writes
		// every cell, so nothing carries over between rows and neither needs an allocation.
		let mut own = [F::ZERO; Word::BITS];
		let mut facing = [F::ZERO; Word::BITS];

		let (mut y_1, mut y_inf) = (F::ZERO, F::ZERO);
		for (index, row) in g.rows() {
			// A row and the row facing it share an inner slot and differ in the outer bit being
			// bound, so one slice of the operator serves both.
			let (variant, amount) = decode_shift(index % SHIFT_COUNT);
			let outer = index >> LOG_SHIFT_COUNT;
			shift_operator_row(variant, amount, &mut own, self.stride(outer));
			shift_operator_row(variant, amount, &mut facing, self.stride(outer ^ facing_half));

			// The infinity evaluation reads H(0) + H(1), the same sum from either half, so the
			// row's own half only decides the evaluation at 1.
			let in_upper_half = index & half != 0;
			for (value, (&own_j, &facing_j)) in
				iter::zip(P::iter_slice(row), iter::zip(&own, &facing))
			{
				if in_upper_half {
					y_1 += value * own_j;
				}
				y_inf += value * (own_j + facing_j);
			}
		}

		RoundEvals([y_1, y_inf]).interpolate(claim)
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_core::constraint_system::Shift;
	use binius_field::{BinaryField128bGhash, Random};
	use binius_math::test_utils::random_scalars;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	type F = BinaryField128bGhash;

	/// Whether output bit `out` of `variant` at `amount` reads input bit `in_bit`.
	///
	/// Read off the word operation itself: shifting a word with only bit `in_bit` set leaves bits
	/// exactly where that bit is read. This is the shift indicator, straight from its definition
	/// and independent of every table the reduction builds.
	fn reads_input_bit(variant: ShiftVariant, out: usize, in_bit: usize, amount: usize) -> bool {
		let shifted = variant.apply(binius_core::word::Word(1u64 << in_bit), amount);
		(shifted.as_u64() >> out) & 1 == 1
	}

	/// The `h` rows of one inner shift, over every outer index, straight from the definition:
	///
	/// ```text
	/// h(j) = sum_{i, k} d(i) * shift-ind(i, k, outer) * shift-ind(k, j, inner)
	/// ```
	///
	/// This is the double contraction the stage computes by nesting the shift operator. It is
	/// evaluated only where `g` is supported, so no `2^24` table is ever formed — and no
	/// multiplications are needed, since each indicator selects rather than scales.
	fn reference_rows(d: &[F], inner: Shift) -> Vec<Vec<F>> {
		let (inner_variant, inner_amount) = (inner.variant, inner.amount as usize);
		(0..SHIFT_COUNT)
			.map(|outer_index| {
				let (outer_variant, outer_amount) = decode_shift(outer_index);
				// eta at this outer index: the oblong weights carried to the intermediate word.
				let eta = (0..Word::BITS)
					.map(|k| {
						(0..Word::BITS)
							.filter(|&i| reads_input_bit(outer_variant, i, k, outer_amount))
							.map(|i| d[i])
							.sum::<F>()
					})
					.collect::<Vec<F>>();
				// And on down to the witness bit.
				(0..Word::BITS)
					.map(|j| {
						(0..Word::BITS)
							.filter(|&k| reads_input_bit(inner_variant, k, j, inner_amount))
							.map(|k| eta[k])
							.sum::<F>()
					})
					.collect()
			})
			.collect()
	}

	/// A dense reference for the rounds, folded directly rather than derived from a folded `eta`.
	///
	/// One `(g, h)` pair per inner shift the fixture uses, each spanning the whole outer index. The
	/// outer rounds never mix inner indices and `g` is zero at the inner shifts absent here, so
	/// this is exact rather than a restriction.
	struct Reference {
		/// Per inner shift, the `g` rows over the remaining outer index.
		g: Vec<Vec<Vec<F>>>,
		/// Per inner shift, the `h` rows over the remaining outer index.
		h: Vec<Vec<Vec<F>>>,
	}

	impl Reference {
		/// The sum the rounds start from.
		fn sum(&self) -> F {
			iter::zip(&self.g, &self.h)
				.flat_map(|(g, h)| iter::zip(g, h))
				.flat_map(|(g_row, h_row)| iter::zip(g_row, h_row))
				.map(|(&g, &h)| g * h)
				.sum()
		}

		/// This round's message, by brute force over every entry.
		fn round_coeffs(&self, claim: F) -> RoundCoeffs<F> {
			let half = self.g[0].len() / 2;
			let (mut y_1, mut y_inf) = (F::ZERO, F::ZERO);
			for (g, h) in iter::zip(&self.g, &self.h) {
				for lower in 0..half {
					let upper = lower + half;
					for j in 0..Word::BITS {
						y_1 += g[upper][j] * h[upper][j];
						y_inf += (g[lower][j] + g[upper][j]) * (h[lower][j] + h[upper][j]);
					}
				}
			}
			RoundEvals([y_1, y_inf]).interpolate(claim)
		}

		/// Binds the highest outer variable of both tables.
		fn fold(&mut self, challenge: F) {
			let fold = |rows: &mut Vec<Vec<F>>| {
				let half = rows.len() / 2;
				for lower in 0..half {
					for j in 0..Word::BITS {
						let (low, high) = (rows[lower][j], rows[lower + half][j]);
						rows[lower][j] = low + challenge * (low + high);
					}
				}
				rows.truncate(half);
			};
			self.g.iter_mut().for_each(&fold);
			self.h.iter_mut().for_each(&fold);
		}
	}

	/// The inner shifts of the fixture, one `g` row per outer index the fixture stores them at.
	///
	/// Every variant appears, and every case whose operator slice is not a plain move of the
	/// weights: `sra` and `sra32` pile several weights onto one position, in either slot.
	fn fixture() -> (Vec<F>, Vec<(Shift, Vec<Shift>)>) {
		let mut rng = StdRng::seed_from_u64(0);
		let d = random_scalars::<F>(&mut rng, Word::BITS);
		let quadruples = vec![
			// A sign extension: shift a field up to the top and arithmetically back down.
			(Shift::sll(40), vec![Shift::sar(40)]),
			// The sign bit in the inner slot instead, under two different outer shifts.
			(Shift::sar(7), vec![Shift::srl(3), Shift::rotr(19)]),
			// A rotate under a rotate, which wraps from both ends.
			(Shift::rotr(1), vec![Shift::rotr(63), Shift::sll(9)]),
			// The half-word family, including its own sign-extension case.
			(Shift::sll32(11), vec![Shift::sra32(11), Shift::rotr32(5)]),
			(Shift::srl32(3), vec![Shift::sll32(30)]),
			// An unshifted inner slot, which the reduction reaches as the identity spelling.
			(Shift::IDENTITY, vec![Shift::IDENTITY, Shift::srl(17)]),
		];
		(d, quadruples)
	}

	/// The stage's round messages are the ones a prover folding `h` itself would send.
	///
	/// This is the property the whole construction rests on: deriving each row from a folded `eta`
	/// gives the same round polynomial as folding the fully formed `h`, round after round, because
	/// the shift operator is linear in its weights.
	#[test]
	fn round_messages_match_a_directly_folded_reference() {
		let mut rng = StdRng::seed_from_u64(1);
		let (d, quadruples) = fixture();

		// The sparse rows, keyed on the quadruple with the outer slot above the inner one.
		let mut indices = Vec::new();
		let mut values = Vec::new();
		let mut reference_g = Vec::new();
		for (inner, outers) in &quadruples {
			let mut rows = vec![vec![F::ZERO; Word::BITS]; SHIFT_COUNT];
			for outer in outers {
				let row = random_scalars::<F>(&mut rng, Word::BITS);
				indices.push((outer.index() << LOG_SHIFT_COUNT | inner.index()) as u32);
				values.extend_from_slice(&row);
				// Rows at a repeated index add up, which the reference has to mirror.
				for (slot, value) in iter::zip(&mut rows[outer.index()], row) {
					*slot += value;
				}
			}
			reference_g.push(rows);
		}
		let log_rows = 2 * LOG_SHIFT_COUNT;
		let mut g = SparseShiftRows::<F>::new(indices, values, log_rows);

		let mut reference = Reference {
			g: reference_g,
			h: quadruples
				.iter()
				.map(|(inner, _)| reference_rows(&d, *inner))
				.collect(),
		};

		let mut stage = OuterShiftStage::new(&GlobalAllocator, &d);
		assert_eq!(stage.n_vars_remaining(), LOG_SHIFT_COUNT);

		// A non-degenerate fixture, or matching round messages would prove nothing.
		let mut claim = reference.sum();
		assert_ne!(claim, F::ZERO);

		for _ in 0..LOG_SHIFT_COUNT {
			let coeffs = stage.round_coeffs(&g, claim);
			assert_eq!(coeffs, reference.round_coeffs(claim));

			let challenge = F::random(&mut rng);
			claim = coeffs.evaluate(&challenge);
			stage.fold(challenge);
			g.fold(challenge);
			reference.fold(challenge);
		}

		// The stage hands the inner rounds `eta` at the bound outer point, which is what the
		// reference's own `h` was folded down to for the inner shift that leaves it untouched.
		assert_eq!(stage.n_vars_remaining(), 0);
		let identity_slot = quadruples
			.iter()
			.position(|(inner, _)| *inner == Shift::IDENTITY)
			.expect("the fixture carries an identity inner slot");
		assert_eq!(stage.psi(), reference.h[identity_slot][0].as_slice());
	}

	/// The stage never reads a table indexed by anything but the intermediate bit and the outer
	/// slot, so its cost is fixed however many quadruples are live.
	#[test]
	fn the_folded_table_stays_the_size_of_one_shift_slot() {
		let (d, _) = fixture();
		let mut stage = OuterShiftStage::<F, _>::new(&GlobalAllocator, &d);

		let mut expected = Word::LOG_BITS + LOG_SHIFT_COUNT;
		assert_eq!(stage.eta.log_len(), expected);
		for _ in 0..LOG_SHIFT_COUNT {
			stage.fold(F::ONE);
			expected -= 1;
			assert_eq!(stage.eta.log_len(), expected);
		}
		assert_eq!(stage.psi().len(), Word::BITS);
	}
}
