// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_core::{constraint_system::Operand, word::Word};
use binius_field::{BinaryField, FieldOps, WideMul, util::powers};
use binius_math::{
	inner_product::inner_product_scalars, multilinear::eq::eq_ind_partial_eval_scalars,
};
use binius_utils::rayon::prelude::*;

use super::{
	SHIFT_VARIANT_COUNT,
	shift_ind::{partial_eval_phi, partial_eval_sigmas, partial_eval_sigmas_transpose},
};

/// Evaluates the three h multilinear polynomials (corresponding to SLL, SRL, SRA) at challenge
/// points.
///
/// This is the verifier's version of the h-parts evaluation - instead of building
/// full multilinear polynomials, it directly computes their evaluations.
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

/// Evaluates the monster multilinear polynomial for a constraint operation.
///
/// The monster multilinear encodes all constraints of a given type (AND or IMUL) into
/// a single polynomial. For an operation with multiple operands, it computes:
///
/// $$
/// \sum_{\text{m_idx} \in \text{enumerate(operands)}}
///     \lambda^{\text{m_idx}+1}
///     \sum_{\text{op}} h_{\text{op}}(r_j, r_s) \cdot M_{\text{m}, \text{op}}(r_x', r_y, r_s)
/// $$
///
/// where:
/// - `m_idx` indexes the operand position (0 to `operand_vecs.len() - 1`)
/// - `op` ranges over the shift variants (logical/arithmetic shifts)
/// - `h_op` is the shift selector polynomial
/// - `M_m,op` is the multilinear extension of the operand values
///
/// # Arguments
///
/// * `operand_vecs` - Vector of operand vectors, one per operand position in the constraint
/// * `r_x_prime` - The constraint challenge `r_x'`; its equality indicator tensor is expanded here.
/// * `lambda` - Random coefficient for batching operand evaluations
/// * `shift_scalars` - Tensor product of the shift-selector evaluations `h_op` and the shift-amount
///   equality indicator `eq(r_s)`, indexed by `variant * Word::BITS + amount`. Shared across
///   operations.
/// * `r_y_tensor` - Equality indicator tensor expansion of the word index challenge `r_y`
///
/// # Returns
///
/// The evaluation of the monster multilinear at the given challenge points.
pub fn evaluate_monster_multilinear_for_operation<F, E>(
	operand_vecs: &[Vec<&Operand>],
	r_x_prime: &[E],
	lambda: E,
	shift_scalars: &[E; SHIFT_VARIANT_COUNT * Word::BITS],
	r_y_tensor: &[E],
) -> E
where
	F: BinaryField,
	E: FieldOps<Scalar = F> + From<F>,
{
	let arity = operand_vecs.len();
	for operands in operand_vecs {
		assert_eq!(operands.len(), (1 << r_x_prime.len()));
	}

	let r_x_prime_tensor = eq_ind_partial_eval_scalars(r_x_prime);
	let operand_shift_scalars = operand_shift_scalar_table(shift_scalars, lambda, arity);

	// Accumulate one contribution per constraint. Within a constraint, each shifted-value term over
	// all operands is weighted by its operand shift scalar and the word-index tensor entry; the
	// running sum is then scaled by the constraint-index tensor entry.
	let mut eval = E::zero();
	for (constraint, r_x_prime_entry) in r_x_prime_tensor.iter().enumerate() {
		let mut constraint_eval = E::zero();
		for (operand_id, operand_vec) in operand_vecs.iter().enumerate() {
			for svi in operand_vec[constraint] {
				let variant = svi.shift_variant as usize;
				let index = (variant * Word::BITS + svi.amount as usize) * arity + operand_id;
				constraint_eval +=
					operand_shift_scalars[index].clone() * &r_y_tensor[svi.value_index.0 as usize];
			}
		}
		eval += constraint_eval * r_x_prime_entry;
	}

	eval
}

/// Folds the operand batching coefficients (λ powers) into the shared shift scalars, producing a
/// table indexed by `(variant, amount, operand_id)` whose entry is
/// `shift_scalars[variant * Word::BITS + amount] · λ^{operand_id + 1}` — the scalar that
/// multiplies each shifted-value term.
fn operand_shift_scalar_table<E: FieldOps>(
	shift_scalars: &[E; SHIFT_VARIANT_COUNT * Word::BITS],
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

/// Native, `WideMul`-accelerated variant of [`evaluate_monster_multilinear_for_operation`] over the
/// base field `F`.
///
/// Produces the identical result, but defers the `GF(2^128)` reductions: the per-constraint
/// contributions accumulate into a single *unreduced* wide element, reduced exactly once at the end
/// (reduction is `F`-linear, so this equals reducing each per-constraint product and summing). The
/// generic path can't do this because `E: FieldOps` does not imply `WideMul`.
pub(crate) fn evaluate_monster_multilinear_for_operation_native<F: BinaryField>(
	operand_vecs: &[Vec<&Operand>],
	r_x_prime: &[F],
	lambda: F,
	shift_scalars: &[F; SHIFT_VARIANT_COUNT * Word::BITS],
	r_y_tensor: &[F],
) -> F {
	let arity = operand_vecs.len();
	for operands in operand_vecs {
		assert_eq!(operands.len(), 1 << r_x_prime.len());
	}

	let r_x_prime_tensor = eq_ind_partial_eval_scalars(r_x_prime);
	let operand_shift_scalars = operand_shift_scalar_table(shift_scalars, lambda, arity);

	// One unreduced wide product per constraint. The constraints partition cleanly across rayon:
	// each produces a single wide element and they are summed, so there is no large per-task
	// accumulator. The single final reduction is `F`-linear.
	let eval = r_x_prime_tensor
		.par_iter()
		.enumerate()
		.map(|(constraint, &r_x_prime_entry)| {
			let mut constraint_eval = F::ZERO;
			for (operand_id, operand_vec) in operand_vecs.iter().enumerate() {
				for svi in operand_vec[constraint] {
					let variant = svi.shift_variant as usize;
					let index = (variant * Word::BITS + svi.amount as usize) * arity + operand_id;
					constraint_eval +=
						operand_shift_scalars[index] * r_y_tensor[svi.value_index.0 as usize];
				}
			}
			F::wide_mul(constraint_eval, r_x_prime_entry)
		})
		.sum::<<F as WideMul>::Output>();
	F::reduce(eval)
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128bGhash, Field, Random};
	use binius_math::{
		BinarySubspace,
		test_utils::{index_to_hypercube_point, random_scalars},
		univariate::lagrange_evals_scalars,
	};
	use rand::prelude::*;

	use super::*;

	/// The native `WideMul` variant must produce exactly the same result as the generic
	/// evaluation (deferred reduction is `F`-linear).
	#[test]
	fn evaluate_monster_native_matches_generic() {
		use binius_core::{
			ShiftVariant,
			constraint_system::{ShiftedValueIndex, ValueIndex},
		};

		type F = BinaryField128bGhash;
		let mut rng = StdRng::seed_from_u64(3);

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
		let n_words = 40usize;
		let log_constraints = 6usize;
		let n_constraints = 1usize << log_constraints;
		let arity = 3usize;

		let owned: Vec<Vec<Operand>> = (0..arity)
			.map(|_| {
				(0..n_constraints)
					.map(|_| {
						(0..rng.random_range(0..=3))
							.map(|_| ShiftedValueIndex {
								value_index: ValueIndex(rng.random_range(0..n_words) as u32),
								shift_variant: shift_variants
									[rng.random_range(0..SHIFT_VARIANT_COUNT)],
								amount: rng.random_range(0..Word::BITS) as u8,
							})
							.collect()
					})
					.collect()
			})
			.collect();
		let operand_vecs: Vec<Vec<&Operand>> = owned.iter().map(|v| v.iter().collect()).collect();

		let r_x_prime = random_scalars::<F>(&mut rng, log_constraints);
		let lambda = F::random(&mut rng);
		let shift_scalars: [F; SHIFT_VARIANT_COUNT * Word::BITS] =
			std::array::from_fn(|_| F::random(&mut rng));
		let r_y_tensor = random_scalars::<F>(&mut rng, n_words);

		let generic = evaluate_monster_multilinear_for_operation::<F, F>(
			&operand_vecs,
			&r_x_prime,
			lambda,
			&shift_scalars,
			&r_y_tensor,
		);
		let native = evaluate_monster_multilinear_for_operation_native::<F>(
			&operand_vecs,
			&r_x_prime,
			lambda,
			&shift_scalars,
			&r_y_tensor,
		);
		assert_eq!(generic, native);
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
			let l_tilde = lagrange_evals_scalars(&subspace, challenge);

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
	fn test_evaluate_h_op_multilinearity() {
		// Test that the function is multilinear in each variable
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random evaluation points
		let challenge = BinaryField128bGhash::random(&mut rng);
		let subspace = BinarySubspace::<BinaryField128bGhash>::with_dim(Word::LOG_BITS);
		let l_tilde = lagrange_evals_scalars(&subspace, challenge);
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
