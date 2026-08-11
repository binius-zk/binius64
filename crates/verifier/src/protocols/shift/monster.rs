// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_core::{
	constraint_system::{Operand, Shift},
	word::Word,
};
use binius_field::{
	BinaryField, FieldOps, WideMul,
	util::{FieldFn, powers},
};
use binius_math::{
	inner_product::inner_product_scalars, multilinear::eq::eq_ind_partial_eval_scalars,
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};

use super::{
	SHIFT_COUNT, SHIFT_VARIANT_COUNT,
	shift_ind::{partial_eval_phi, partial_eval_sigmas, partial_eval_sigmas_transpose},
};

/// Contracts each shift variant's indicator against a weight per output bit position.
///
/// This is the verifier's version of the h-parts evaluation.
/// Instead of building full multilinear polynomials, it computes their evaluations directly.
/// For each variant it returns
///
/// ```text
/// sum_i l_tilde[i] * shift-ind_op(i, r_j, r_s)
/// ```
///
/// The weights are an arbitrary vector over the `Word::BITS` output bit positions.
/// That is what lets one kernel serve both weight families the reduction needs.
/// See [`evaluate_inner_h_op`] for the second.
pub fn evaluate_h_op<E: FieldOps>(l_tilde: &[E], r_j: &[E], r_s: &[E]) -> [E; SHIFT_VARIANT_COUNT] {
	assert_eq!(l_tilde.len(), Word::BITS);
	assert_eq!(r_j.len(), Word::LOG_BITS);
	assert_eq!(r_s.len(), Word::LOG_BITS);

	// Use helper functions to compute shift indicator helpers for 64-bit shifts
	let (sigma, sigma_prime) = partial_eval_sigmas(r_j, r_s);
	let sigma_transpose = partial_eval_sigmas_transpose(r_j, r_s);
	let phi = partial_eval_phi(r_s);
	let j_product: E = r_j.iter().cloned().product();

	// Use helper functions to compute shift indicator helpers for 32-bit shifts
	let (sigma32, sigma32_prime) = partial_eval_sigmas(&r_j[..5], &r_s[..5]);
	let sigma32_transpose = partial_eval_sigmas_transpose(&r_j[..5], &r_s[..5]);
	let phi32 = partial_eval_phi(&r_s[..5]);
	let j_product32: E = r_j[..5].iter().cloned().product();

	// Compute final results
	let sll = inner_product_scalars(l_tilde.iter().cloned(), sigma_transpose);
	let srl = inner_product_scalars(l_tilde.iter().cloned(), sigma.iter().cloned());
	// sra == ∑ᵢ L̃(i) ⋅ (srlᵢ + ∏ₖ rⱼ[k] ⋅ φᵢ)
	//     == ∑ᵢ L̃(i) ⋅ srlᵢ + ∏ₖ rⱼ[k] ⋅ [ ∑ᵢ L̃(i) ⋅ φᵢ ]
	//     == srl + ∏ₖ rⱼ[k] ⋅ [ ∑ᵢ L̃(i) ⋅ φᵢ ]
	let sra = srl.clone() + j_product * inner_product_scalars(l_tilde.iter().cloned(), phi);
	let rotr = inner_product_scalars(
		l_tilde.iter().cloned(),
		iter::zip(&sigma, &sigma_prime).map(|(s_i, s_prime_i)| s_i.clone() + s_prime_i),
	);

	let r_j_rest_tensor = eq_ind_partial_eval_scalars(&r_j[5..]);
	let chunk_size = 1 << 5; // 32

	let sll32 = inner_product_scalars(
		l_tilde.chunks(chunk_size).map(|chunk| {
			inner_product_scalars(chunk.iter().cloned(), sigma32_transpose.iter().cloned())
		}),
		r_j_rest_tensor.iter().cloned(),
	);
	let srl32 = inner_product_scalars(
		l_tilde
			.chunks(chunk_size)
			.map(|chunk| inner_product_scalars(chunk.iter().cloned(), sigma32.iter().cloned())),
		r_j_rest_tensor.iter().cloned(),
	);
	let sra32 = srl32.clone()
		+ inner_product_scalars(
			l_tilde.chunks(chunk_size).map(|chunk| {
				j_product32.clone()
					* inner_product_scalars(chunk.iter().cloned(), phi32.iter().cloned())
			}),
			r_j_rest_tensor.iter().cloned(),
		);
	let rotr32 = inner_product_scalars(
		l_tilde.chunks(chunk_size).map(|chunk| {
			inner_product_scalars(
				chunk.iter().cloned(),
				iter::zip(&sigma32, &sigma32_prime).map(|(s_i, s_prime_i)| s_i.clone() + s_prime_i),
			)
		}),
		r_j_rest_tensor,
	);

	[sll, srl, sra, rotr, sll32, srl32, sra32, rotr32]
}

/// Evaluates the inner-phase shift-indicator weight at the intermediate point.
///
/// The three-phase reduction carries two weight families, one per shift slot:
///
/// ```text
/// outer   H(K, S_2) = sum_i delta_D(r_ihat, ihat) * shift-ind_op2(i, K, S_2)
/// inner   h(J, S_1) = shift-ind_op1(r_k, J, S_1)
/// ```
///
/// The outer family contracts against the oblong weights `delta_D(r_ihat, ihat)`.
/// The inner family is the bare indicator at the intermediate point the outer phase produced.
/// It carries no oblong factor, because that phase already summed the output bit index out.
///
/// Both are the same contraction under different weights.
/// A multilinear extension in the output-bit argument *is* the equality-weighted sum over it:
///
/// ```text
/// shift-ind~_op1(r_k, J, S_1) = sum_i eq~(r_k, i) * shift-ind_op1(i, J, S_1)
/// ```
///
/// So this is the shared kernel with the equality indicator of `r_k` in place of the Lagrange
/// weights.
///
/// # Arguments
///
/// - `r_k`: the intermediate bit point, `Word::LOG_BITS` coordinates, from the outer phase.
/// - `r_j`: the source bit point, `Word::LOG_BITS` coordinates.
/// - `r_s`: the inner shift amount point, `Word::LOG_BITS` coordinates.
///
/// # Returns
///
/// One evaluation per shift variant, in [`ShiftVariant::ALL`](binius_core::ShiftVariant::ALL)
/// order. The caller folds them over the variant axis, as it folds the outer family's.
pub fn evaluate_inner_h_op<E: FieldOps>(
	r_k: &[E],
	r_j: &[E],
	r_s: &[E],
) -> [E; SHIFT_VARIANT_COUNT] {
	assert_eq!(r_k.len(), Word::LOG_BITS);

	// Weighting by the equality indicator of `r_k` turns the contraction into an evaluation of
	// the indicator's multilinear extension there.
	let eq_r_k = eq_ind_partial_eval_scalars(r_k);
	evaluate_h_op(&eq_r_k, r_j, r_s)
}

/// A [`FieldFn`] evaluating one operation's monster multilinear polynomial.
///
/// The monster multilinear encodes all `ARITY`-operand constraints of a single operation (BitAnd,
/// IntMul or BinMul) into one polynomial:
///
/// $$
/// \sum_{\text{m_idx} \in \text{enumerate(operands)}}
///     \lambda^{\text{m_idx}+1}
///     \sum_{\text{op}} h_{\text{op}}(r_j, r_s) \cdot M_{\text{m}, \text{op}}(r_x', r_y, r_s)
/// $$
///
/// where `m_idx` indexes the operand position (0 to `ARITY - 1`), `op` ranges over the shift
/// variants, `h_op` is the shift selector polynomial, and `M_{m,op}` is the multilinear extension
/// of the operand values.
///
/// The `FieldFn` input is the flat slice built by [`encode_operation_input`]: the constraint
/// challenge `r_x'`, then the batching coefficient `lambda`, then the shared shift scalars, then
/// the word-index tensor `r_y`. [`FieldFn::call`] evaluates generically over any `E`;
/// [`FieldFn::call_native`] takes the `WideMul`-accelerated base-field path.
pub struct OperationEvalFn<'a, C, const ARITY: usize> {
	/// The operation's constraints, each exposing its `ARITY` operands as an array in storage
	/// order.
	constraints: &'a [C],
	/// The number of constants the constraints may name.
	n_constants: usize,
	/// The number of inout values the constraints may name.
	n_inout: usize,
	/// The number of private values the constraints may name.
	n_hidden: usize,
}

impl<'a, C, const ARITY: usize> OperationEvalFn<'a, C, ARITY> {
	/// Wraps an operation's constraints for monster-multilinear evaluation.
	///
	/// The three counts are the segment lengths of the system holding the constraints. They are
	/// what the word-index tensor is cut along when the input is split back apart, so
	/// [`encode_operation_input`] must be given runs of matching lengths.
	pub const fn new(
		constraints: &'a [C],
		n_constants: usize,
		n_inout: usize,
		n_hidden: usize,
	) -> Self {
		Self {
			constraints,
			n_constants,
			n_inout,
			n_hidden,
		}
	}

	/// Splits the flat [`FieldFn`] input into its sections.
	///
	/// The `r_x'` section has `ceil(log2(constraints.len()))` entries — the reductions run over the
	/// constraint count rounded up to a power of two — so the split needs no state beyond the
	/// constraints.
	///
	/// Two shift-scalar tables follow, one per slot of a term's shift sequence.
	/// Each is [`SHIFT_COUNT`] entries wide; see [`ShiftScalars`] for why the weight splits that
	/// way.
	///
	/// The word-index tensor arrives as one run per value segment, in
	/// [`ValueSegment`](binius_core::constraint_system::ValueSegment) order, and
	/// comes back as an array indexed by that segment. An operand term is then read at
	/// `r_y_tensor[segment][index]`, which is the term's own `(segment, index)` pair — no address
	/// arithmetic in between. The runs hold only the words a constraint can name, so the padding
	/// between sections never reaches the input.
	fn split_input<'i, E>(
		&self,
		input: &'i [E],
	) -> (&'i [E], &'i E, ShiftScalars<'i, E>, [&'i [E]; 3]) {
		let n_vars = log2_ceil_usize(self.constraints.len());
		let (r_x_prime, rest) = input.split_at(n_vars);
		let (lambda, rest) = rest.split_first().expect("input encodes lambda");
		let (inner, rest) = rest.split_at(SHIFT_COUNT);
		let (outer, rest) = rest.split_at(SHIFT_COUNT);
		let shift_scalars = ShiftScalars {
			inner: inner
				.try_into()
				.expect("input encodes the inner shift scalars"),
			outer: outer
				.try_into()
				.expect("input encodes the outer shift scalars"),
		};

		let (constants, rest) = rest.split_at(self.n_constants);
		let (inout, rest) = rest.split_at(self.n_inout);
		let (hidden, _) = rest.split_at(self.n_hidden);

		(r_x_prime, lambda, shift_scalars, [constants, inout, hidden])
	}
}

/// The shift-sequence weight tables, one per slot of the sequence.
///
/// A term's sequence weight factorizes across its two slots:
///
/// ```text
/// eq(r_v1, v_1) * eq(r_s1, s_1)  *  eq(r_v2, v_2) * eq(r_s2, s_2)
/// \_______ inner table _______/     \_______ outer table ______/
/// ```
///
/// One table per slot holds the weights at `2 * SHIFT_COUNT` = 1,024 entries.
/// Keying a single table on the whole sequence would need `SHIFT_COUNT^2` = 262,144.
/// Fanning that out over the operand batching coefficients reaches roughly 1.5M multiplications at
/// BMUL's arity of six.
///
/// The cost of the split is one extra multiply per term.
pub struct ShiftScalars<'a, E> {
	/// The weight of each spelling the inner shift slot can take, with the operand batching
	/// coefficients yet to be fanned in.
	pub inner: &'a [E; SHIFT_COUNT],
	/// The weight of each spelling the outer shift slot can take.
	pub outer: &'a [E; SHIFT_COUNT],
}

// Two shared slices copy freely whatever `E` is. Deriving these would demand `E: Copy`, which the
// generic evaluation path does not have.
impl<E> Clone for ShiftScalars<'_, E> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<E> Copy for ShiftScalars<'_, E> {}

/// Places a shift in its slot's weight table.
///
/// The variant indexes runs of `Word::BITS`, and the amount indexes within a run.
/// The prover's shift multilinears use the same layout.
#[inline]
const fn shift_index(shift: Shift) -> usize {
	shift.variant as usize * Word::BITS + shift.amount as usize
}

impl<F, C, const ARITY: usize> FieldFn<F> for OperationEvalFn<'_, C, ARITY>
where
	F: BinaryField,
	C: AsRef<[Operand; ARITY]> + Sync,
{
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, input: &[E]) -> E {
		let (r_x_prime, lambda, shift_scalars, r_y_tensor) = self.split_input(input);

		let r_x_prime_tensor = eq_ind_partial_eval_scalars(r_x_prime);
		// The batching coefficients fan into the inner table only, holding it to
		// `SHIFT_COUNT * arity`; the outer weight multiplies in per term.
		let operand_shift_scalars =
			operand_shift_scalar_table(shift_scalars.inner, lambda.clone(), ARITY);

		// One contribution per constraint.
		// Each term is weighted by its two slots' shift scalars and its word-index tensor entry.
		// The running sum then scales by the constraint-index tensor entry.
		//
		// The tensor covers the padded constraint count, so the zip stops at the last real
		// constraint; padding rows carry no operand terms and contribute nothing.
		let mut eval = E::zero();
		for (constraint, r_x_prime_entry) in iter::zip(self.constraints, &r_x_prime_tensor) {
			let mut constraint_eval = E::zero();
			for (operand_id, operand) in constraint.as_ref().iter().enumerate() {
				for svi in operand {
					let inner = shift_index(svi.inner()) * ARITY + operand_id;
					let outer = shift_index(svi.outer());
					constraint_eval += operand_shift_scalars[inner].clone()
						* &shift_scalars.outer[outer]
						* &r_y_tensor[svi.value_index.segment() as usize]
							[svi.value_index.index() as usize];
				}
			}
			eval += constraint_eval * r_x_prime_entry;
		}

		eval
	}

	/// Native fast path over the base field `F`.
	///
	/// Produces the identical result, but defers the `GF(2^128)` reductions: the per-constraint
	/// contributions accumulate into a single *unreduced* wide element, reduced exactly once at the
	/// end (reduction is `F`-linear, so this equals reducing each per-constraint product and
	/// summing). The generic [`call`](FieldFn::call) can't do this because `E: FieldOps` does not
	/// imply `WideMul`.
	fn call_native(&self, input: &[F]) -> F {
		let (r_x_prime, lambda, shift_scalars, r_y_tensor) = self.split_input(input);

		let r_x_prime_tensor = eq_ind_partial_eval_scalars(r_x_prime);
		let operand_shift_scalars = operand_shift_scalar_table(shift_scalars.inner, *lambda, ARITY);

		// One unreduced wide product per constraint. The constraints partition cleanly across
		// rayon: each produces a single wide element and they are summed, so there is no large
		// per-task accumulator. The single final reduction is `F`-linear. The tensor covers the
		// padded constraint count, so the zip stops at the last real constraint; the padding rows
		// have no operand terms and contribute nothing.
		let eval = self
			.constraints
			.par_iter()
			.zip(r_x_prime_tensor.par_iter())
			.map(|(constraint, &r_x_prime_entry)| {
				let mut constraint_eval = F::ZERO;
				for (operand_id, operand) in constraint.as_ref().iter().enumerate() {
					for svi in operand {
						let inner = shift_index(svi.inner()) * ARITY + operand_id;
						let outer = shift_index(svi.outer());
						constraint_eval += operand_shift_scalars[inner]
							* shift_scalars.outer[outer]
							* r_y_tensor[svi.value_index.segment() as usize]
								[svi.value_index.index() as usize];
					}
				}
				F::wide_mul(constraint_eval, r_x_prime_entry)
			})
			.sum::<<F as WideMul>::Output>();
		F::reduce(eval)
	}
}

/// Builds the flat [`FieldFn`] input consumed by [`OperationEvalFn`].
///
/// Concatenates `r_x_prime ++ [lambda] ++ inner_shift_scalars ++ outer_shift_scalars ++
/// r_y_tensor`. `OperationEvalFn::split_input` is the inverse; it recovers the `r_x'` length from
/// the constraint count, so only `lambda` and the two fixed-length shift-scalar tables need a known
/// position.
pub fn encode_operation_input<E: Clone>(
	r_x_prime: &[E],
	lambda: E,
	shift_scalars: ShiftScalars<'_, E>,
	r_y_tensor: [&[E]; 3],
) -> Vec<E> {
	let n_words = r_y_tensor
		.iter()
		.map(|segment| segment.len())
		.sum::<usize>();
	let mut input = Vec::with_capacity(r_x_prime.len() + 1 + 2 * SHIFT_COUNT + n_words);
	input.extend_from_slice(r_x_prime);
	input.push(lambda);
	// The inner table leads the outer one, which is the order `split_input` cuts them back apart.
	input.extend_from_slice(shift_scalars.inner);
	input.extend_from_slice(shift_scalars.outer);
	// One run per value segment, in `ValueSegment` order, which is how `split_input` cuts them
	// back apart.
	for segment in r_y_tensor {
		input.extend_from_slice(segment);
	}
	input
}

/// Folds the operand batching coefficients (λ powers) into the inner slot's shift scalars,
/// producing a table indexed by `(variant, amount, operand_id)` whose entry is
/// `inner[variant * Word::BITS + amount] · λ^{operand_id + 1}`.
///
/// The fan-out stays on this one table.
/// A term's outer-slot weight multiplies in where the term is read.
/// So the table is `SHIFT_COUNT * arity` entries rather than `SHIFT_COUNT^2 * arity`.
fn operand_shift_scalar_table<E: FieldOps>(
	shift_scalars: &[E; SHIFT_COUNT],
	lambda: E,
	arity: usize,
) -> Vec<E> {
	let lambda_powers = powers(lambda).skip(1).take(arity).collect::<Vec<_>>();
	let mut table = Vec::with_capacity(shift_scalars.len() * arity);
	for shift_scalar in shift_scalars {
		for lambda_power in &lambda_powers {
			table.push(shift_scalar.clone() * lambda_power);
		}
	}
	table
}

#[cfg(test)]
mod tests {
	use binius_core::{
		ShiftVariant,
		constraint_system::{AndConstraint, Shift, ShiftedValueIndex, ValueIndex},
	};
	use binius_field::{BinaryField128bGhash, Field, Random};
	use binius_math::{
		BinarySubspace,
		test_utils::{index_to_hypercube_point, random_scalars},
		univariate::lagrange_evals_scalars,
	};
	use rand::prelude::*;

	use super::*;

	/// Builds `n_constraints` random arity-3 constraints (like `AndConstraint`), constraint-major:
	/// one array of operands per constraint.
	///
	/// Every term names a private word, so the constant and inout runs of the word-index tensor
	/// stay empty.
	fn random_and_constraints(
		rng: &mut StdRng,
		n_constraints: usize,
		n_words: usize,
	) -> Vec<AndConstraint> {
		let shift_variants = [
			ShiftVariant::Sll,
			ShiftVariant::Slr,
			ShiftVariant::Sar,
			ShiftVariant::Rotr,
			ShiftVariant::Sll32,
			ShiftVariant::Srl32,
			ShiftVariant::Sra32,
			ShiftVariant::Rotr32,
		];
		(0..n_constraints)
			.map(|_| {
				AndConstraint(std::array::from_fn(|_| {
					(0..rng.random_range(0..=3))
						.map(|_| {
							// The reduction reads the inner slot, so the fixture leaves the outer
							// one at the identity.
							ShiftedValueIndex::single(
								ValueIndex::private(rng.random_range(0..n_words) as u32),
								Shift {
									variant: shift_variants
										[rng.random_range(0..SHIFT_VARIANT_COUNT)],
									amount: rng.random_range(0..Word::BITS) as u8,
								},
							)
						})
						.collect()
				}))
			})
			.collect()
	}

	#[test]
	fn shift_index_places_a_shift_by_variant_then_amount() {
		// Both slot tables share this layout, and so do the prover's phase-1 multilinears: runs of
		// `Word::BITS` amounts, one run per variant. A table indexed the other way round would
		// weight a term by another shift's scalar.
		assert_eq!(shift_index(Shift::IDENTITY), 0);
		assert_eq!(shift_index(Shift::sll(5)), 5);
		assert_eq!(shift_index(Shift::srl(0)), Word::BITS);
		assert_eq!(shift_index(Shift::srl(3)), Word::BITS + 3);
		assert_eq!(shift_index(Shift::rotr32(31)), ShiftVariant::Rotr32 as usize * Word::BITS + 31);

		// Every spelling lands inside its table.
		for variant in ShiftVariant::ALL {
			for amount in 0..variant.max_amount() {
				assert!(shift_index(Shift::new(variant, amount)) < SHIFT_COUNT);
			}
		}
	}

	#[test]
	fn evaluate_monster_scales_by_the_outer_slot_weight() {
		// Invariant: the outer slot's weight reaches the evaluation, and reaches it as a factor.
		//
		// Every term the fixture builds carries an identity outer shift, so index 0 is the only
		// entry read. Scaling it must scale the whole evaluation by the same factor.
		type F = BinaryField128bGhash;
		let mut rng = StdRng::seed_from_u64(7);

		let n_words = 40usize;
		let constraints = random_and_constraints(&mut rng, 32, n_words);
		let r_x_prime = random_scalars::<F>(&mut rng, 5);
		let lambda = F::random(&mut rng);
		let inner: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
		let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
		let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

		let eval_fn = OperationEvalFn::new(&constraints, 0, 0, n_words);
		let eval_with_outer = |outer: &[F; SHIFT_COUNT]| {
			let shift_scalars = ShiftScalars {
				inner: &inner,
				outer,
			};
			let input = encode_operation_input(&r_x_prime, lambda, shift_scalars, r_y_tensor);
			eval_fn.call_native(&input)
		};

		// The identity-selecting table, which is what the two-phase reduction supplies.
		let mut identity_selecting = [F::ZERO; SHIFT_COUNT];
		identity_selecting[0] = F::ONE;
		let baseline = eval_with_outer(&identity_selecting);
		// A non-degenerate fixture, or the scaling below proves nothing.
		assert_ne!(baseline, F::ZERO);

		// Scaling the entry every term reads scales the evaluation by the same factor.
		let scale = F::random(&mut rng);
		let mut scaled = [F::ZERO; SHIFT_COUNT];
		scaled[0] = scale;
		assert_eq!(eval_with_outer(&scaled), baseline * scale);

		// Zeroing it kills the evaluation, so no term slipped past the outer factor.
		assert_eq!(eval_with_outer(&[F::ZERO; SHIFT_COUNT]), F::ZERO);

		// The weight of a shift the fixture never names must not enter. Only index 0 is read, so
		// filling every other entry changes nothing.
		let mut noise = [F::random(&mut rng); SHIFT_COUNT];
		noise[0] = F::ONE;
		assert_eq!(eval_with_outer(&noise), baseline);
	}

	/// The native `WideMul` variant must produce exactly the same result as the generic
	/// evaluation (deferred reduction is `F`-linear). Covers a power-of-two constraint count and a
	/// non-power-of-two one, whose `r_x'` tensor runs past the last constraint.
	#[test]
	fn evaluate_monster_native_matches_generic() {
		type F = BinaryField128bGhash;
		let mut rng = StdRng::seed_from_u64(3);

		let n_words = 40usize;
		for n_constraints in [64usize, 37] {
			let constraints = random_and_constraints(&mut rng, n_constraints, n_words);

			let r_x_prime = random_scalars::<F>(&mut rng, log2_ceil_usize(n_constraints));
			let lambda = F::random(&mut rng);
			// Both slots draw random weights, so a path that dropped the outer factor, or read
			// it from the inner table, would disagree with the other.
			let inner: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
			let outer: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
			let shift_scalars = ShiftScalars {
				inner: &inner,
				outer: &outer,
			};
			let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
			let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

			let eval_fn = OperationEvalFn::new(&constraints, 0, 0, n_words);
			let input = encode_operation_input(&r_x_prime, lambda, shift_scalars, r_y_tensor);
			let generic = eval_fn.call::<F>(&input);
			let native = eval_fn.call_native(&input);
			assert_eq!(generic, native, "n_constraints = {n_constraints}");
		}
	}

	/// Appending all-zero padding constraints must not change the evaluation: a padding constraint
	/// has no operand terms, so it contributes nothing. This is what lets the constraint system
	/// keep its true count while the reductions run over the padded one.
	///
	/// [`FieldFn::call`] and [`FieldFn::call_native`] walk the constraints over independent zips,
	/// so both are checked.
	#[test]
	fn evaluate_monster_ignores_zero_padding_constraints() {
		type F = BinaryField128bGhash;
		let mut rng = StdRng::seed_from_u64(5);

		let n_words = 40usize;
		let n_constraints = 21usize;
		let constraints = random_and_constraints(&mut rng, n_constraints, n_words);
		let padded = constraints
			.iter()
			.cloned()
			.chain(iter::repeat_n(
				AndConstraint::default(),
				n_constraints.next_power_of_two() - n_constraints,
			))
			.collect::<Vec<_>>();

		let r_x_prime = random_scalars::<F>(&mut rng, log2_ceil_usize(n_constraints));
		let lambda = F::random(&mut rng);
		let inner: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
		let outer: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
		let shift_scalars = ShiftScalars {
			inner: &inner,
			outer: &outer,
		};
		let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
		let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

		let input = encode_operation_input(&r_x_prime, lambda, shift_scalars, r_y_tensor);
		assert_eq!(
			OperationEvalFn::new(&constraints, 0, 0, n_words).call::<F>(&input),
			OperationEvalFn::new(&padded, 0, 0, n_words).call::<F>(&input)
		);
		assert_eq!(
			OperationEvalFn::new(&constraints, 0, 0, n_words).call_native(&input),
			OperationEvalFn::new(&padded, 0, 0, n_words).call_native(&input)
		);
	}

	#[test]
	fn test_evaluate_h_op_hypercube_vertices() {
		// Property-based test: for random i, j, s in {0..63}, with challenge being
		// the i-th element of the subspace, the outputs must match indicator relations
		// over integers:
		// - sll == 1 iff j + s == i
		// - srl == 1 iff i + s == j
		// - sra == 1 iff i + s == j || i + s >= 64 && j == 63
		// - rotr == 1 iff (i + s) % 64 == j
		let mut rng = StdRng::seed_from_u64(0);
		let subspace = BinarySubspace::<BinaryField128bGhash>::with_dim(Word::LOG_BITS);

		// Run a reasonable number of random trials
		for _trial in 0..1024 {
			let i = rng.random_range(0..64);
			let j = rng.random_range(0..64);
			let s = rng.random_range(0..64);

			let challenge = subspace.get(i);
			let l_tilde = lagrange_evals_scalars(&subspace, &challenge);

			let r_j = index_to_hypercube_point::<BinaryField128bGhash>(Word::LOG_BITS, j);
			let r_s = index_to_hypercube_point::<BinaryField128bGhash>(Word::LOG_BITS, s);

			let [sll, srl, sra, rotr, sll32, srl32, sra32, rotr32] =
				evaluate_h_op(&l_tilde, &r_j, &r_s);

			let expected_sll = j + s == i;
			let expected_srl = i + s == j;
			let expected_sra = (i + s).min(63) == j;
			let expected_rotr = (i + s) % 64 == j;

			let i_hi = i / 32;
			let i_lo = i % 32;
			let j_hi = j / 32;
			let j_lo = j % 32;
			let s_lo = s % 32;

			let expected_sll32 = i_hi == j_hi && j_lo + s_lo == i_lo;
			let expected_srl32 = i_hi == j_hi && i_lo + s_lo == j_lo;
			let expected_sra32 = i_hi == j_hi && (i_lo + s_lo).min(31) == j_lo;
			let expected_rotr32 = i_hi == j_hi && (i_lo + s_lo) % 32 == j_lo;

			let to_field = |b: bool| {
				if b {
					BinaryField128bGhash::ONE
				} else {
					BinaryField128bGhash::ZERO
				}
			};

			assert_eq!(sll, to_field(expected_sll), "sll failed for i={i}, j={j}, s={s}");
			assert_eq!(srl, to_field(expected_srl), "srl failed for i={i}, j={j}, s={s}");
			assert_eq!(sra, to_field(expected_sra), "sra failed for i={i}, j={j}, s={s}");
			assert_eq!(rotr, to_field(expected_rotr), "rotr failed for i={i}, j={j}, s={s}");
			assert_eq!(sll32, to_field(expected_sll32), "sll32 failed for i={i}, j={j}, s={s}");
			assert_eq!(srl32, to_field(expected_srl32), "srl32 failed for i={i}, j={j}, s={s}");
			assert_eq!(sra32, to_field(expected_sra32), "sra32 failed for i={i}, j={j}, s={s}");
			assert_eq!(rotr32, to_field(expected_rotr32), "rotr32 failed for i={i}, j={j}, s={s}");
		}
	}

	#[test]
	fn inner_h_op_at_a_vertex_is_the_integer_indicator() {
		// The inner weight is the indicator's multilinear extension in its output-bit argument. At
		// a hypercube vertex an extension agrees with the function it extends, so driving all
		// three points to vertices must reproduce the integer conditions exactly.
		//
		// This is the outer family's vertex test with the oblong weights replaced by `eq(r_k, .)`,
		// which is the one substitution `evaluate_inner_h_op` makes.
		type F = BinaryField128bGhash;
		let mut rng = StdRng::seed_from_u64(11);

		for _trial in 0..1024 {
			let k = rng.random_range(0..64);
			let j = rng.random_range(0..64);
			let s = rng.random_range(0..64);

			let r_k = index_to_hypercube_point::<F>(Word::LOG_BITS, k);
			let r_j = index_to_hypercube_point::<F>(Word::LOG_BITS, j);
			let r_s = index_to_hypercube_point::<F>(Word::LOG_BITS, s);

			let [sll, srl, sra, rotr, sll32, srl32, sra32, rotr32] =
				evaluate_inner_h_op(&r_k, &r_j, &r_s);

			// Full-width variants: output position `k` reads source position `j`.
			let expected_sll = j + s == k;
			let expected_srl = k + s == j;
			let expected_sra = (k + s).min(63) == j;
			let expected_rotr = (k + s) % 64 == j;

			// Half-word variants act within a half and read only the low 5 bits of the amount.
			let (k_hi, k_lo) = (k / 32, k % 32);
			let (j_hi, j_lo) = (j / 32, j % 32);
			let s_lo = s % 32;

			let expected_sll32 = k_hi == j_hi && j_lo + s_lo == k_lo;
			let expected_srl32 = k_hi == j_hi && k_lo + s_lo == j_lo;
			let expected_sra32 = k_hi == j_hi && (k_lo + s_lo).min(31) == j_lo;
			let expected_rotr32 = k_hi == j_hi && (k_lo + s_lo) % 32 == j_lo;

			let to_field = |bit: bool| if bit { F::ONE } else { F::ZERO };

			assert_eq!(sll, to_field(expected_sll), "sll at k={k}, j={j}, s={s}");
			assert_eq!(srl, to_field(expected_srl), "srl at k={k}, j={j}, s={s}");
			assert_eq!(sra, to_field(expected_sra), "sra at k={k}, j={j}, s={s}");
			assert_eq!(rotr, to_field(expected_rotr), "rotr at k={k}, j={j}, s={s}");
			assert_eq!(sll32, to_field(expected_sll32), "sll32 at k={k}, j={j}, s={s}");
			assert_eq!(srl32, to_field(expected_srl32), "srl32 at k={k}, j={j}, s={s}");
			assert_eq!(sra32, to_field(expected_sra32), "sra32 at k={k}, j={j}, s={s}");
			assert_eq!(rotr32, to_field(expected_rotr32), "rotr32 at k={k}, j={j}, s={s}");
		}
	}

	#[test]
	fn inner_h_op_of_an_identity_shift_is_the_equality_indicator() {
		// Invariant: an identity shift weights by plain equality between the two bit points.
		//
		// A degenerate slot contributes eq(r_k, r_j) and nothing else, so composing it with the
		// other slot's weight leaves that weight standing alone.
		//
		// Every variant is the identity at amount zero, so all eight must agree — including the
		// half-word forms, which move nothing within either half.
		type F = BinaryField128bGhash;
		let mut rng = StdRng::seed_from_u64(13);

		let r_k = random_scalars::<F>(&mut rng, Word::LOG_BITS);
		let r_j = random_scalars::<F>(&mut rng, Word::LOG_BITS);
		// The amount point at the hypercube's zero vertex fixes the shift amount to zero.
		let r_s = vec![F::ZERO; Word::LOG_BITS];

		// eq(r_k, r_j) over `Word::LOG_BITS` coordinates, built independently of the indicator.
		let expected = iter::zip(&r_k, &r_j)
			.map(|(&r_k_b, &r_j_b)| r_k_b * r_j_b + (F::ONE - r_k_b) * (F::ONE - r_j_b))
			.product::<F>();

		for (variant, weight) in iter::zip(ShiftVariant::ALL, evaluate_inner_h_op(&r_k, &r_j, &r_s))
		{
			assert_eq!(
				weight, expected,
				"{variant:?} at amount zero is not the equality indicator"
			);
		}
	}

	#[test]
	fn inner_h_op_is_multilinear_in_each_point() {
		// The reduction binds `r_j` and `r_s` by sumcheck and reads `r_k` from the outer phase, so
		// the weight has to be multilinear in all three for those bindings to be sound.
		type F = BinaryField128bGhash;
		let mut rng = StdRng::seed_from_u64(17);

		let r_k = random_scalars::<F>(&mut rng, Word::LOG_BITS);
		let r_j = random_scalars::<F>(&mut rng, Word::LOG_BITS);
		let r_s = random_scalars::<F>(&mut rng, Word::LOG_BITS);

		// Interpolating between a point's two boolean settings must reproduce the point itself.
		let check_linear_in = |point: &[F], rebuild: &dyn Fn(&[F]) -> [F; SHIFT_VARIANT_COUNT]| {
			for coordinate in 0..point.len() {
				let mut at_zero = point.to_vec();
				at_zero[coordinate] = F::ZERO;
				let mut at_one = point.to_vec();
				at_one[coordinate] = F::ONE;

				let [result_0, result_1, result] = [&at_zero[..], &at_one[..], point].map(rebuild);
				for variant in 0..SHIFT_VARIANT_COUNT {
					let interpolated = result_0[variant] * (F::ONE - point[coordinate])
						+ result_1[variant] * point[coordinate];
					assert_eq!(
						result[variant], interpolated,
						"not linear in coordinate {coordinate}"
					);
				}
			}
		};

		check_linear_in(&r_k, &|r_k| evaluate_inner_h_op(r_k, &r_j, &r_s));
		check_linear_in(&r_j, &|r_j| evaluate_inner_h_op(&r_k, r_j, &r_s));
		check_linear_in(&r_s, &|r_s| evaluate_inner_h_op(&r_k, &r_j, r_s));
	}

	#[test]
	fn test_evaluate_h_op_multilinearity() {
		// Test that the function is multilinear in each variable
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random evaluation points
		let challenge = BinaryField128bGhash::random(&mut rng);
		let subspace = BinarySubspace::<BinaryField128bGhash>::with_dim(Word::LOG_BITS);
		let l_tilde = lagrange_evals_scalars(&subspace, &challenge);
		let r_j = random_scalars::<BinaryField128bGhash>(&mut rng, Word::LOG_BITS);
		let r_s = random_scalars::<BinaryField128bGhash>(&mut rng, Word::LOG_BITS);

		// Check linearity in each variable
		for i in 0..Word::LOG_BITS {
			// Check r_j[i]
			let mut r_j_at_0 = r_j.clone();
			r_j_at_0[i] = BinaryField128bGhash::ZERO;
			let mut r_j_at_1 = r_j.clone();
			r_j_at_1[i] = BinaryField128bGhash::ONE;
			let [result_0, result_1, result_y] = [&r_j_at_0, &r_j_at_1, &r_j]
				.map(|r_j_variant| evaluate_h_op(&l_tilde, r_j_variant, &r_s));
			for variant in 0..SHIFT_VARIANT_COUNT {
				let expected = result_0[variant] * (BinaryField128bGhash::ONE - r_j[i])
					+ result_1[variant] * r_j[i];
				assert_eq!(result_y[variant], expected, "Not linear in r_j[{i}]");
			}

			// Check r_s[i]
			let mut r_s_at_0 = r_s.clone();
			r_s_at_0[i] = BinaryField128bGhash::ZERO;
			let mut r_s_at_1 = r_s.clone();
			r_s_at_1[i] = BinaryField128bGhash::ONE;
			let [result_0, result_1, result_y] = [&r_s_at_0, &r_s_at_1, &r_s]
				.map(|r_s_variant| evaluate_h_op(&l_tilde, &r_j, r_s_variant));
			for variant in 0..SHIFT_VARIANT_COUNT {
				let expected = result_0[variant] * (BinaryField128bGhash::ONE - r_s[i])
					+ result_1[variant] * r_s[i];
				assert_eq!(result_y[variant], expected, "Not linear in r_s[{i}]");
			}
		}
	}
}
