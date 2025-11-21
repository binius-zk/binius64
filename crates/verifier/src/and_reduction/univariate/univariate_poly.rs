// Copyright 2025 Irreducible Inc.
use binius_field::{BinaryField, Field};
use binius_math::BinarySubspace;

use super::univariate_lagrange::lexicographic_lagrange_numerators;
use crate::and_reduction::univariate::univariate_lagrange::lexicographic_lagrange_denominator;

/// Trait for univariate polynomials that can be evaluated at challenge points.
///
/// This trait abstracts over different representations of univariate polynomials,
/// enabling polymorphic evaluation at points from potentially larger extension fields.
///
/// # Type Parameters
/// * `FChallenge` - The field type of evaluation points
pub trait UnivariatePolyIsomorphic<FChallenge: Field> {
	/// Evaluates the polynomial at a given challenge point.
	///
	/// # Arguments
	/// * `challenge` - The point at which to evaluate the polynomial
	///
	/// # Returns
	/// The polynomial evaluation p(challenge) as a field element
	fn evaluate_at_challenge(&self, challenge: FChallenge) -> FChallenge;
}

/// A univariate polynomial represented in Lagrange basis over a binary subspace.
///
/// # Representation
/// The polynomial is stored as coefficients in the Lagrange basis corresponding to
/// evaluation points from the binary subspace.
///
/// # Mathematical Background
/// For a polynomial p(x) and basis points B = {b₀, b₁, ..., b_{n-1}} from the binary subspace,
/// the Lagrange representation stores coefficients [c₀, c₁, ..., c_{n-1}] such that:
///
/// p(x) = Σᵢ cᵢ · Lᵢ(x)
///
/// where Lᵢ(x) is the i-th Lagrange basis polynomial that equals 1 at bᵢ and 0 at all other basis
/// points.
///
/// # Field Isomorphism Support
/// The struct supports evaluation in fields isomorphic to extension fields through the type system:
/// - Coefficients are stored in field F
/// - Evaluation can occur in any field that F can be embedded into
///
/// # Type Parameters
/// * `F` - The coefficient field (must be a binary field)
pub struct GenericPo2UnivariatePoly<F: Field + BinaryField> {
	univariate_lagrange_coeffs: Vec<F>,
	log_degree_lt: usize,
	domain: BinarySubspace<F>,
}

impl<F: Field + BinaryField> GenericPo2UnivariatePoly<F> {
	/// Creates a new univariate polynomial from Lagrange basis coefficients.
	///
	/// # Arguments
	/// * `univariate_lagrange_coeffs` - Coefficients in Lagrange basis ordered by the binary
	///   subspace iteration order
	/// * `domain` - The binary subspace defining the domain
	///
	/// # Panics
	/// Panics if the number of coefficients is not a power of 2.
	pub fn new(univariate_lagrange_coeffs: Vec<F>, domain: BinarySubspace<F>) -> Self {
		let degree_lt = univariate_lagrange_coeffs.len();
		let log_degree_lt = degree_lt.trailing_zeros() as usize;

		// panic if length is not a po2
		assert_eq!(degree_lt, 1 << log_degree_lt);

		Self {
			univariate_lagrange_coeffs,
			log_degree_lt,
			domain,
		}
	}

	/// Returns the degree bound of the polynomial.
	///
	/// For a polynomial in Lagrange basis over n points, the degree is at most n-1.
	/// This method returns n, which is the smallest degree bound.
	///
	/// # Returns
	/// The number of basis points (2^log_degree_lt)
	pub fn degree_lt(&self) -> usize {
		1 << self.log_degree_lt
	}

	/// Returns an iterator over the Lagrange basis coefficients.
	///
	/// The coefficients are ordered by the binary subspace iteration order.
	///
	/// # Returns
	/// An iterator yielding references to the Lagrange coefficients
	pub fn iter(&self) -> impl Iterator<Item = &F> {
		self.univariate_lagrange_coeffs.iter()
	}

	fn evaluate_lagrange_common<FChallenge: Field>(
		&self,
		numerators: impl Iterator<Item = FChallenge>,
		coeffs_in_eval_field: impl Iterator<Item = FChallenge>,
		denominator_inv_in_eval_field: FChallenge,
	) -> FChallenge {
		numerators
			.zip(coeffs_in_eval_field)
			.map(|(basis_vec_eval, coeff)| basis_vec_eval * coeff)
			.sum::<FChallenge>()
			* denominator_inv_in_eval_field
	}
}

impl<FCoeffs, FChallenge> UnivariatePolyIsomorphic<FChallenge> for GenericPo2UnivariatePoly<FCoeffs>
where
	FCoeffs: Field + BinaryField,
	FChallenge: Field + From<FCoeffs>,
{
	fn evaluate_at_challenge(&self, challenge: FChallenge) -> FChallenge {
		let evals_of_lagrange_basis_vectors_not_yet_divide_by_denominator =
			lexicographic_lagrange_numerators::<FCoeffs, FChallenge>(challenge, &self.domain);

		let denominator_inv =
			lexicographic_lagrange_denominator::<FCoeffs, FChallenge>(&self.domain)
				.invert_or_zero();

		self.evaluate_lagrange_common(
			evals_of_lagrange_basis_vectors_not_yet_divide_by_denominator.into_iter(),
			self.iter().map(|coeff| FChallenge::from(*coeff)),
			denominator_inv,
		)
	}
}

#[cfg(test)]
mod test {
	use binius_field::{AESTowerField8b, Random};
	use binius_math::BinarySubspace;
	use itertools::Itertools;
	use rand::{SeedableRng, rngs::StdRng};

	use super::GenericPo2UnivariatePoly;
	use crate::{
		and_reduction::{
			univariate::univariate_poly::UnivariatePolyIsomorphic,
			utils::constants::{ROWS_PER_HYPERCUBE_VERTEX, SKIPPED_VARS},
		},
		config::B128,
		protocols::sumcheck::RoundCoeffs,
	};

	#[test]
	fn univariate_po2_sanity_check() {
		let mut rng = StdRng::from_seed([0; 32]);

		let monomial_basis_coeffs = (0..ROWS_PER_HYPERCUBE_VERTEX)
			.map(|_| AESTowerField8b::random(&mut rng))
			.collect_vec();
		let monomial_basis_poly = RoundCoeffs(monomial_basis_coeffs.clone());

		let monomial_basis_coeffs_isomorphic = monomial_basis_coeffs
			.into_iter()
			.map(B128::from)
			.collect_vec();

		let monomial_basis_isomorphic = RoundCoeffs(monomial_basis_coeffs_isomorphic);

		let domain = BinarySubspace::<AESTowerField8b>::with_dim(SKIPPED_VARS).unwrap();
		let v = domain
			.iter()
			.map(|x| monomial_basis_poly.evaluate(x))
			.collect();

		let poly = GenericPo2UnivariatePoly::<AESTowerField8b>::new(v, domain);

		let random_point = B128::random(&mut rng);

		assert_eq!(
			poly.evaluate_at_challenge(random_point),
			monomial_basis_isomorphic.evaluate(random_point)
		);
	}
}
