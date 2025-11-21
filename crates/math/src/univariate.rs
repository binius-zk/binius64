// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, Field};
use itertools::izip;

use super::{BinarySubspace, FieldBuffer};

/// Evaluate a univariate polynomial specified by its monomial coefficients.
///
/// # Arguments
/// * `coeffs` - Slice of coefficients ordered from low-degree terms to high-degree terms
/// * `x` - Point at which to evaluate the polynomial
pub fn evaluate_univariate<F: Field>(coeffs: &[F], x: F) -> F {
	let Some((&highest_degree, rest)) = coeffs.split_last() else {
		return F::ZERO;
	};

	// Evaluate using Horner's method
	rest.iter()
		.rev()
		.fold(highest_degree, |acc, &coeff| acc * x + coeff)
}

/// Optimized Lagrange evaluation for power-of-2 domains in binary fields.
///
/// Computes the Lagrange polynomial evaluations L̃(z, i) for a power-of-2 domain at point `z`.
/// Uses the provided binary subspace as the evaluation domain.
///
/// # Key Optimization
/// For power-of-2 domains, all barycentric weights are identical due to the additive group
/// structure. For each i ∈ {0, ..., 2^k - 1}, the set {i ⊕ j | j ≠ i} = {1, ..., 2^k - 1}.
/// This allows us to:
/// 1. Compute a single barycentric weight w = 1 / ∏_{j=1}^{n-1} j
/// 2. Use prefix/suffix products to avoid redundant computation
/// 3. Replace inversions with multiplications for better performance
///
/// # Complexity
/// - Time: O(n) where n = subspace size, using 4n - 2 multiplications and 1 inversion
/// - Space: O(n) for prefix/suffix arrays
///
/// # Parameters
/// - `subspace`: The binary subspace defining the evaluation domain
/// - `z`: The evaluation point
///
/// # Returns
/// A vector of Lagrange polynomial evaluations, one for each domain element
pub fn lagrange_evals<F: BinaryField>(subspace: &BinarySubspace<F>, z: F) -> FieldBuffer<F> {
	let domain: Vec<F> = subspace.iter().collect();
	let n = domain.len();

	// Compute single barycentric weight for the additive subgroup
	// All points have the same weight due to subgroup structure
	let w = domain[1..]
		.iter()
		.fold(F::ONE, |acc, &d| acc * d)
		.invert()
		.unwrap_or(F::ONE);

	// Compute prefix products: prefix[i] = ∏_{j=0}^{i-1} (z - domain[j])
	let mut prefixes = vec![F::ONE; n];
	for i in 1..n {
		prefixes[i] = prefixes[i - 1] * (z - domain[i - 1]);
	}

	// Compute suffix products: suffix[i] = ∏_{j=i+1}^{n-1} (z - domain[j])
	let mut suffixes = vec![F::ONE; n];
	for i in (0..n - 1).rev() {
		suffixes[i] = suffixes[i + 1] * (z - domain[i + 1]);
	}

	// Combine prefix, suffix, and weight: L_i(z) = prefix[i] * suffix[i] * w
	let mut result = vec![F::ZERO; n];
	for i in 0..n {
		result[i] = prefixes[i] * suffixes[i] * w;
	}

	FieldBuffer::new(subspace.dim(), result.into_boxed_slice())
		.expect("result.len() == 2^subspace.dim()")
}

/// A domain that univariate polynomials may be evaluated on.
///
/// An evaluation domain of size d + 1 together with polynomial values on that domain uniquely
/// defines a degree <= d polynomial.
#[derive(Debug, Clone)]
pub struct EvaluationDomain<F: Field> {
	points: Vec<F>,
	weights: Vec<F>,
}

impl<F: Field> EvaluationDomain<F> {
	/// Create a new evaluation domain from a set of points.
	///
	/// # Arguments
	/// * `points` - The points that define the domain
	///
	/// # Panics
	/// * If any points are repeated (not distinct)
	pub fn from_points(points: Vec<F>) -> Self {
		let weights = compute_barycentric_weights(&points);
		Self { points, weights }
	}

	pub fn size(&self) -> usize {
		self.points.len()
	}

	pub fn points(&self) -> &[F] {
		self.points.as_slice()
	}

	/// Compute a vector of Lagrange polynomial evaluations in $O(N)$ at a given point `x`.
	///
	/// For an evaluation domain consisting of points $x_i$ Lagrange polynomials $L_i(x)$
	/// are defined by
	///
	/// $$L_i(x) = \prod_{j \neq i}\frac{x - \pi_j}{\pi_i - \pi_j}$$
	pub fn lagrange_evals(&self, x: F) -> Vec<F> {
		let n = self.size();

		let mut result = vec![F::ONE; n];

		// Multiply the product suffixes
		for i in (1..n).rev() {
			result[i - 1] = result[i] * (x - self.points[i]);
		}

		let mut prefix = F::ONE;

		// Multiply the product prefixes and weights
		for (result_i, &point, &weight) in izip!(&mut result, &self.points, &self.weights) {
			*result_i *= prefix * weight;
			prefix *= x - point;
		}

		result
	}

	/// Evaluate the unique interpolated polynomial at any point `x`.
	///
	/// Computational complexity is $O(n)$, for a domain of size $n$.
	pub fn extrapolate(&self, values: &[F], x: F) -> F {
		assert_eq!(values.len(), self.size()); // precondition

		let (ret, _) = izip!(values, &self.points, &self.weights).fold(
			(F::ZERO, F::ONE),
			|(acc, prod), (&value, &point, &weight)| {
				let term = x - point;
				let next_acc = acc * term + prod * value * weight;
				(next_acc, prod * term)
			},
		);

		ret
	}
}

/// Compute the Barycentric weights for a sequence of unique points.
///
/// The [Barycentric] weight $w_i$ for point $x_i$ is calculated as:
/// $$w_i = \prod_{j \neq i} \frac{1}{x_i - x_j}$$
///
/// These weights are used in the Lagrange interpolation formula:
/// $$L(x) = \sum_{i=0}^{n-1} f(x_i) \cdot \frac{w_i}{x - x_i} \cdot \prod_{j=0}^{n-1} (x - x_j)$$
///
/// # Preconditions
/// * All points in the input slice must be distinct, otherwise this function panics.
///
/// [Barycentric]: <https://en.wikipedia.org/wiki/Lagrange_polynomial#Barycentric_form>
fn compute_barycentric_weights<F: Field>(points: &[F]) -> Vec<F> {
	let n = points.len();
	(0..n)
		.map(|i| {
			// TODO: We could use batch inversion here, but it's not a bottleneck
			let product = (0..n)
				.filter(|&j| j != i)
				.map(|j| points[i] - points[j])
				.product::<F>();
			product
				.invert()
				.expect("precondition: all points are distinct; invert only fails on 0")
		})
		.collect()
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128bGhash, Field, Random, util::powers};
	use rand::prelude::*;

	use super::*;
	use crate::{
		BinarySubspace,
		inner_product::inner_product,
		line::extrapolate_line_packed,
		test_utils::{B128, random_scalars},
	};

	fn evaluate_univariate_with_powers<F: Field>(coeffs: &[F], x: F) -> F {
		inner_product(coeffs.iter().copied(), powers(x).take(coeffs.len()))
	}

	type F = BinaryField128bGhash;

	#[test]
	fn test_evaluate_univariate_against_reference() {
		let mut rng = StdRng::seed_from_u64(0);

		for n_coeffs in [0, 1, 2, 5, 10] {
			let coeffs = random_scalars(&mut rng, n_coeffs);
			let x = F::random(&mut rng);
			assert_eq!(
				evaluate_univariate(&coeffs, x),
				evaluate_univariate_with_powers(&coeffs, x)
			);
		}
	}

	#[test]
	fn test_lagrange_evals() {
		let mut rng = StdRng::seed_from_u64(0);

		// Test mathematical properties across different domain sizes
		for log_domain_size in [3, 4, 5, 6] {
			// Create subspace for this test
			let subspace = BinarySubspace::<F>::with_dim(log_domain_size).unwrap();
			let domain: Vec<F> = subspace.iter().collect();

			// Test 1: Partition of Unity - Lagrange polynomials sum to 1
			let eval_point = F::random(&mut rng);
			let lagrange_coeffs = lagrange_evals(&subspace, eval_point);
			let sum: F = lagrange_coeffs.as_ref().iter().copied().sum();
			assert_eq!(
				sum,
				F::ONE,
				"Partition of unity failed for domain size {}",
				1 << log_domain_size
			);

			// Test 2: Interpolation Property - L_i(x_j) = δ_ij
			for (j, &domain_point) in domain.iter().enumerate() {
				let lagrange_at_domain = lagrange_evals(&subspace, domain_point);
				for (i, &coeff) in lagrange_at_domain.as_ref().iter().enumerate() {
					let expected = if i == j { F::ONE } else { F::ZERO };
					assert_eq!(
						coeff, expected,
						"Interpolation property failed: L_{i}({j}) ≠ {expected}"
					);
				}
			}
		}

		// Test 3: Polynomial Interpolation Accuracy
		let log_domain_size = 6;
		let subspace = BinarySubspace::<F>::with_dim(log_domain_size).unwrap();
		let domain: Vec<F> = subspace.iter().collect();
		let coeffs = random_scalars(&mut rng, 10);

		// Evaluate polynomial at domain points
		let domain_evals: Vec<F> = domain
			.iter()
			.map(|&point| evaluate_univariate(&coeffs, point))
			.collect();

		// Test interpolation at random point
		let test_point = F::random(&mut rng);
		let lagrange_coeffs = lagrange_evals(&subspace, test_point);
		let interpolated =
			inner_product(domain_evals.iter().copied(), lagrange_coeffs.iter_scalars());
		let direct = evaluate_univariate(&coeffs, test_point);

		assert_eq!(interpolated, direct, "Polynomial interpolation accuracy failed");
	}

	#[test]
	fn test_random_extrapolate() {
		let mut rng = StdRng::seed_from_u64(0);
		let degree = 6;

		let domain = EvaluationDomain::from_points(random_scalars(&mut rng, degree + 1));

		let coeffs = random_scalars(&mut rng, degree + 1);

		let values = domain
			.points()
			.iter()
			.map(|&x| evaluate_univariate(&coeffs, x))
			.collect::<Vec<_>>();

		let x = B128::random(&mut rng);
		let expected_y = evaluate_univariate(&coeffs, x);
		assert_eq!(domain.extrapolate(&values, x), expected_y);
	}

	#[test]
	fn test_extrapolate_line() {
		let mut rng = StdRng::seed_from_u64(0);
		for _ in 0..10 {
			let x0 = B128::random(&mut rng);
			let x1 = B128::random(&mut rng);
			// Use a smaller field element for z to test the subfield scalar multiplication
			let z = B128::from(rng.next_u64() as u128);
			assert_eq!(extrapolate_line_packed(x0, x1, z), x0 + (x1 - x0) * z);
		}
	}

	#[test]
	fn test_evaluation_domain_lagrange_evals() {
		let mut rng = StdRng::seed_from_u64(0);

		// Create a small domain
		let domain_points: Vec<B128> = (0..10).map(|_| B128::random(&mut rng)).collect();
		let evaluation_domain = EvaluationDomain::from_points(domain_points.clone());

		// Create random values for interpolation
		let values: Vec<B128> = (0..10).map(|_| B128::random(&mut rng)).collect();

		// Test point
		let z = B128::random(&mut rng);

		// Compute extrapolation
		let extrapolated = evaluation_domain.extrapolate(values.as_slice(), z);

		// Compute using Lagrange coefficients
		let lagrange_coeffs = evaluation_domain.lagrange_evals(z);
		let lagrange_eval = inner_product(lagrange_coeffs, values);

		assert_eq!(lagrange_eval, extrapolated);
	}
}
