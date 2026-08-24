// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The basis one variable contributes, and everything a cube derives from it.

use std::iter;

use binius_field::{Field, field::FieldOps};

use super::Expansion;

/// A hypercube of coefficients for multilinear polynomials.
///
/// A cube is fixed by the two-element basis `(b_0, b_1)` that each of its variables contributes.
/// That basis is a pair of linear polynomials, so a cube is a choice between two of them.
/// Everything below is derived from that choice, and shared by both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hypercube {
	/// The Boolean cube `{0, 1}^n`, whose per-variable basis is `(1 - X, X)`.
	///
	/// That basis is the pair of Lagrange polynomials on the two vertices `0` and `1`.
	/// So the coefficient indexed by a vertex is the multilinear's evaluation at that vertex.
	One,

	/// The infinity cube `{0, inf}^n`, whose per-variable basis is `(1, X)`.
	///
	/// The vertex `inf` selects a multilinear's leading coefficient in that variable.
	/// So the coefficient indexed by a vertex `v` belongs to the monomial
	///
	/// ```text
	/// prod_{i : v_i = inf} X_i
	/// ```
	Inf,
}

impl Hypercube {
	/// Evaluates the basis of one variable at a coordinate.
	///
	/// Returns `(b_0(r), b_1(r))` for the coordinate `r`.
	#[inline(always)]
	pub fn basis<F: FieldOps>(self, coord: &F) -> [F; 2] {
		match self {
			Self::One => [F::one() - coord, coord.clone()],
			Self::Inf => [F::one(), coord.clone()],
		}
	}

	/// Scales the basis of one variable by a value.
	///
	/// Returns `(v * b_0(r), v * b_1(r))` for the value `v` and the coordinate `r`.
	/// This is the inner loop of every expansion.
	/// So each arm beats the two multiplications that scaling the basis directly costs.
	#[inline(always)]
	pub fn expand_var<F: FieldOps>(self, value: &F, coord: &F) -> [F; 2] {
		match self {
			Self::One => {
				// Both halves share the product `value * coord`, so one multiplication covers both.
				let prod = value.clone() * coord;
				[value.clone() - &prod, prod]
			}
			// The constant basis polynomial is one, so the low half is the value untouched.
			Self::Inf => [value.clone(), value.clone() * coord],
		}
	}

	/// Strips one variable's basis factor from the two halves of an expansion.
	///
	/// The halves hold `v * b_0(r)` and `v * b_1(r)` for the stripped variable's coordinate `r`.
	/// The low half is overwritten with `v`.
	///
	/// Recovering `v` is one fixed linear combination of the two halves:
	///
	/// ```text
	/// sum_i w_i * v * b_i(r) = v    where    sum_i w_i * b_i(X) = 1
	/// ```
	///
	/// Those weights are unique and free of `r`, so the same combination works at any coordinate.
	#[inline(always)]
	pub fn contract_var<F: FieldOps>(self, lo: &mut F, hi: &F) {
		match self {
			// The two basis polynomials sum to one, so both recovery weights are one.
			Self::One => *lo += hi,
			// The low half already holds the value, so the weights are one and zero.
			// Contracting a variable is therefore free for this cube.
			Self::Inf => {}
		}
	}

	/// Evaluates the equality indicator of one variable.
	///
	/// ```text
	/// eq(X, Y) = sum_i b_i(X) * b_i(Y)
	/// ```
	///
	/// Each arm is a closed form, cheaper than pairing the two bases term by term.
	#[inline(always)]
	pub fn eq_one_var<F: FieldOps>(self, x: F, y: F) -> F {
		match self {
			Self::One => {
				// Over characteristic two the `2 * X * Y` term vanishes, so
				//
				//     X * Y + (1 - X) * (1 - Y)  =  X + Y + 1
				//
				// The condition is a compile-time constant, so only one arm is ever generated.
				if F::Scalar::CHARACTERISTIC == 2 {
					x + y + F::one()
				} else {
					let one = F::one();
					x.clone() * y.clone() + (one.clone() - x) * (one - y)
				}
			}
			Self::Inf => F::one() + x * y,
		}
	}

	/// Begins the expansion of a point over this cube.
	///
	/// The seed and the storage are chosen on the returned value, which then computes.
	pub fn expand<F: FieldOps>(self, point: &[F]) -> Expansion<'_, F> {
		Expansion::new(self, point)
	}

	/// Evaluates the equality indicator multilinear at a pair of points.
	///
	/// This is the `2n`-variate multilinear
	///
	/// ```text
	/// eq(X_0, ..., X_{n-1}, Y_0, ..., Y_{n-1}) = prod_i sum_j b_j(X_i) * b_j(Y_i)
	/// ```
	pub fn eq_ind<F: FieldOps>(self, x: &[F], y: &[F]) -> F {
		assert_eq!(x.len(), y.len(), "pre-condition: x and y must be the same length");
		// The indicator factors over the variables, so one per-variable product suffices.
		iter::zip(x, y)
			.map(|(x, y)| self.eq_one_var(x.clone(), y.clone()))
			.product()
	}

	/// Evaluates the equality indicator multilinear with one operand fixed to all zeros.
	///
	/// Only the constant basis polynomial survives at a zero coordinate:
	///
	/// ```text
	/// eq(0^n, Y_0, ..., Y_{n-1}) = prod_i b_0(Y_i)
	/// ```
	pub fn eq_ind_zero<F: FieldOps>(self, point: &[F]) -> F {
		// The linear basis polynomial is multiplied by a zero coordinate, so it drops out.
		point
			.iter()
			.map(|y| {
				let [y_0, _] = self.basis(y);
				y_0
			})
			.product()
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_utils::rayon::task_size::min_len_for_bytes;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::{
		multilinear::MultilinearMut,
		test_utils::{B128, Packed128b, index_to_hypercube_point, random_scalars},
	};

	type P = Packed128b;
	type F = B128;

	/// Both bases, so a shared property is checked once against each of them.
	const CUBES: [Hypercube; 2] = [Hypercube::One, Hypercube::Inf];

	#[test]
	fn expand_var_matches_scaled_basis() {
		let mut rng = StdRng::seed_from_u64(0);

		// Each arm saves a multiplication over scaling the basis the plain way.
		// So both must land on the same pair.
		let [value, coord] = [(); 2].map(|_| random_scalars::<F>(&mut rng, 1)[0]);
		for cube in CUBES {
			assert_eq!(
				cube.expand_var(&value, &coord),
				cube.basis(&coord).map(|b_i| b_i * value),
				"mismatch for {cube:?}"
			);
		}
	}

	#[test]
	fn contract_var_inverts_expand_var() {
		let mut rng = StdRng::seed_from_u64(0);

		// Expanding a value by a coordinate and contracting it back must be the identity.
		let [value, coord] = [(); 2].map(|_| random_scalars::<F>(&mut rng, 1)[0]);
		for cube in CUBES {
			let [mut lo, hi] = cube.expand_var(&value, &coord);
			cube.contract_var(&mut lo, &hi);
			assert_eq!(lo, value, "mismatch for {cube:?}");
		}
	}

	#[test]
	fn eq_one_var_matches_basis_definition() {
		let mut rng = StdRng::seed_from_u64(0);

		// Each arm is a closed form that skips the two basis evaluations.
		// So pin both against the generic pairing of the bases.
		let [x, y] = [(); 2].map(|_| random_scalars::<F>(&mut rng, 1)[0]);
		for cube in CUBES {
			let [x_0, x_1] = cube.basis(&x);
			let [y_0, y_1] = cube.basis(&y);
			assert_eq!(cube.eq_one_var(x, y), x_0 * y_0 + x_1 * y_1, "mismatch for {cube:?}");
		}
	}

	#[test]
	fn one_cube_eq_ind_zero_is_the_product_of_complements() {
		let mut rng = StdRng::seed_from_u64(0);

		// The constant basis polynomial of this cube is `1 - Y`.
		for n_vars in 0..5 {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let expected: F = point.iter().map(|&r| F::ONE - r).product();
			assert_eq!(Hypercube::One.eq_ind_zero(&point), expected);

			// The same value as evaluating the full indicator against an all-zero operand.
			assert_eq!(
				Hypercube::One.eq_ind_zero(&point),
				Hypercube::One.eq_ind(&vec![F::ZERO; n_vars], &point)
			);
		}
	}

	#[test]
	fn inf_cube_eq_ind_zero_is_one() {
		let mut rng = StdRng::seed_from_u64(0);

		// Every monomial of positive degree vanishes at zero, leaving the constant one.
		for n_vars in [0, 1, 5] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			assert_eq!(Hypercube::Inf.eq_ind_zero::<F>(&point), F::ONE);

			// The same value as evaluating the full indicator against an all-zero operand.
			assert_eq!(
				Hypercube::Inf.eq_ind_zero::<F>(&point),
				Hypercube::Inf.eq_ind::<F>(&vec![F::ZERO; n_vars], &point)
			);
		}
	}

	#[test]
	fn one_cube_expansion_holds_the_indicator_at_every_vertex() {
		let mut rng = StdRng::seed_from_u64(0);

		// The defining property of this cube: coefficients are evaluations.
		// So the coefficient at an index is the indicator evaluated at that index's vertex.
		let n_vars = 5;
		let point = random_scalars(&mut rng, n_vars);
		let expansion = Hypercube::One.expand(&point).build::<P>();

		for index in 0..1 << n_vars {
			let vertex = index_to_hypercube_point(n_vars, index);
			assert_eq!(expansion.get(index), Hypercube::One.eq_ind::<F>(&point, &vertex));
		}
	}

	#[test]
	fn one_cube_expansion_of_the_empty_point() {
		// The empty point has no variables, so its expansion is the single coefficient one.
		let result = Hypercube::One.expand(&[]).build::<P>();
		assert_eq!(result.log_len(), 0);
		assert_eq!(result.len(), 1);
		assert_eq!(result.get(0), F::ONE);
	}

	#[test]
	fn one_cube_expansion_of_one_coordinate_is_the_basis() {
		// One coordinate expands to the basis `(1 - r_0, r_0)` itself.
		let r0 = F::new(2);
		let result = Hypercube::One.expand(&[r0]).build::<P>();
		assert_eq!(result.log_len(), 1);
		assert_eq!(result.len(), 2);
		assert_eq!(result.get(0), F::ONE - r0);
		assert_eq!(result.get(1), r0);
	}

	#[test]
	fn one_cube_expansion_of_two_coordinates() {
		// Two coordinates: the four products of one factor drawn from each basis.
		let r0 = F::new(2);
		let r1 = F::new(3);
		let result = Hypercube::One.expand(&[r0, r1]).build::<P>();
		assert_eq!(result.log_len(), 2);
		assert_eq!(result.len(), 4);

		// The variable index is the bit position, so `r_0` varies fastest.
		let expected = vec![
			(F::ONE - r0) * (F::ONE - r1),
			r0 * (F::ONE - r1),
			(F::ONE - r0) * r1,
			r0 * r1,
		];
		assert_eq!(result.iter_scalars().collect::<Vec<F>>(), expected);
	}

	#[test]
	fn one_cube_expansion_of_three_coordinates_fills_one_packed_word() {
		// Three coordinates span exactly one full packed word at this packing width.
		let r0 = F::new(2);
		let r1 = F::new(3);
		let r2 = F::new(5);
		let result = Hypercube::One.expand(&[r0, r1, r2]).build::<P>();
		assert_eq!(result.log_len(), 3);
		assert_eq!(result.len(), 8);

		let expected = vec![
			(F::ONE - r0) * (F::ONE - r1) * (F::ONE - r2),
			r0 * (F::ONE - r1) * (F::ONE - r2),
			(F::ONE - r0) * r1 * (F::ONE - r2),
			r0 * r1 * (F::ONE - r2),
			(F::ONE - r0) * (F::ONE - r1) * r2,
			r0 * (F::ONE - r1) * r2,
			(F::ONE - r0) * r1 * r2,
			r0 * r1 * r2,
		];
		assert_eq!(result.iter_scalars().collect::<Vec<F>>(), expected);
	}

	/// The expansion of a point, straight from the definition of the tensor of bases `(1, r_i)`.
	fn inf_cube_reference(point: &[F]) -> Vec<F> {
		// The coefficient at an index is the product of the coordinates its set bits select.
		(0..1 << point.len())
			.map(|index| {
				point
					.iter()
					.enumerate()
					.filter(|(i, _)| index >> i & 1 == 1)
					.map(|(_, r_i)| *r_i)
					.product()
			})
			.collect()
	}

	/// Evaluates the multilinear whose monomial coefficients are given, at a point.
	fn eval_monomial_basis(coeffs: &[F], point: &[F]) -> F {
		// The coefficient at an index belongs to the monomial its set bits select.
		coeffs
			.iter()
			.enumerate()
			.map(|(index, coeff)| {
				*coeff
					* point
						.iter()
						.enumerate()
						.filter(|(i, _)| index >> i & 1 == 1)
						.map(|(_, x_i)| *x_i)
						.product::<F>()
			})
			.sum()
	}

	#[test]
	fn inf_cube_expansion_matches_the_tensor_of_bases() {
		let mut rng = StdRng::seed_from_u64(0);

		// Sizes span the empty point up to a 256-coefficient cube.
		for n_vars in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let expansion = Hypercube::Inf.expand(&point).build::<P>();
			let expansion_scalars = expansion.iter_scalars().collect::<Vec<_>>();
			assert_eq!(expansion_scalars, inf_cube_reference(&point), "mismatch at {n_vars} vars");
		}
	}

	#[test]
	fn inf_cube_expansion_holds_the_monomial_coefficients_of_the_indicator() {
		let mut rng = StdRng::seed_from_u64(0);

		// The defining property of this cube: coefficients are monomial coefficients.
		//
		//     expansion of r  ->  the monomial coefficients of eq(X, r)
		//
		// So reading the expansion in the monomial basis at any x must give the indicator there.
		for n_vars in [0, 1, 2, 5] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let coeffs = Hypercube::Inf.expand(&point).build_scalars();

			let x = random_scalars::<F>(&mut rng, n_vars);
			assert_eq!(eval_monomial_basis(&coeffs, &x), Hypercube::Inf.eq_ind::<F>(&x, &point));
		}
	}

	#[test]
	fn inf_cube_expansion_is_the_evaluation_functional() {
		let mut rng = StdRng::seed_from_u64(0);

		// Read the other way round, the expansion of a point is the functional that evaluates
		// any multilinear at that point, given the multilinear's monomial coefficients.
		for n_vars in [0, 1, 2, 5] {
			let point = random_scalars::<F>(&mut rng, n_vars);
			let coeffs = random_scalars::<F>(&mut rng, 1 << n_vars);

			let expansion = Hypercube::Inf.expand(&point).build_scalars();
			let inner_product = iter::zip(&coeffs, &expansion)
				.map(|(c, e)| *c * e)
				.sum::<F>();
			assert_eq!(inner_product, eval_monomial_basis(&coeffs, &point));
		}
	}

	#[test]
	fn repeated_truncation_matches_expansion_of_the_prefix() {
		let mut rng = StdRng::seed_from_u64(0);

		// Truncate the same buffer over and over, by a shrinking number of variables each time.
		//
		//     reductions 4, 3, 2, 1, 0  ->  10 variables spent in total
		let reductions = 4;
		let n_vars = reductions * (reductions + 1) / 2;
		let point = random_scalars(&mut rng, n_vars);

		let mut eq_ind = Hypercube::One.expand(&point).build::<P>();
		let mut log_n_values = n_vars;

		for reduction in (0..=reductions).rev() {
			let truncated_log_n_values = log_n_values - reduction;
			eq_ind.eq_ind_truncate_low(Hypercube::One, truncated_log_n_values);

			// Each step must match a direct expansion of the surviving prefix of the point.
			let eq_ind_ref = Hypercube::One
				.expand(&point[..truncated_log_n_values])
				.build::<P>();
			assert_eq!(eq_ind_ref.len(), eq_ind.len());
			for i in 0..eq_ind.len() {
				assert_eq!(eq_ind.get(i), eq_ind_ref.get(i));
			}

			log_n_values = truncated_log_n_values;
		}

		// The last reduction is by zero variables, so the sequence ends at the empty point.
		assert_eq!(log_n_values, 0);
	}

	#[test]
	fn truncation_above_the_split_threshold_matches_the_inline_path() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: a round splits across threads only once it exceeds the minimum task size.
		// Below that it runs inline, leaving the parallel path unexercised.
		//
		//     words read in the first round = 2^(n_vars - 1) / scalars per word
		//     pick the smallest n_vars whose first round reaches the minimum
		//
		// Every other truncation test here is smaller, so this one covers the split.
		let min_len = min_len_for_bytes::<[P; 2]>();
		let n_vars = (2 * min_len * P::WIDTH).next_power_of_two().ilog2() as usize;
		let point = random_scalars::<F>(&mut rng, n_vars);

		// Strip the top variable and compare against a direct expansion of the prefix.
		let mut truncated = Hypercube::One.expand(&point).build::<P>();
		truncated.eq_ind_truncate_low(Hypercube::One, n_vars - 1);
		assert_eq!(truncated, Hypercube::One.expand(&point[..n_vars - 1]).build::<P>());
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(16))]

		#[test]
		fn truncation_strips_trailing_variables(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);

			// Truncating to any length must equal expanding that prefix of the point directly.
			for truncated_log_len in 0..=log_n {
				for cube in CUBES {
					let mut truncated = cube.expand(&point).build::<P>();
					truncated.eq_ind_truncate_low(cube, truncated_log_len);
					prop_assert_eq!(
						truncated,
						cube.expand(&point[..truncated_log_len]).build::<P>()
					);
				}
			}
		}
	}
}
