// Copyright 2025 Irreducible Inc.
use binius_field::{BinaryField, Field};
use binius_math::BinarySubspace;

fn products_excluding_one_element<F: Field>(input: &[F]) -> Vec<F> {
	let mut results = vec![F::ONE; input.len()];
	for i in (0..(input.len() - 1)).rev() {
		results[i] = results[i + 1] * input[i + 1];
	}

	let mut forward_product = F::ONE;

	for i in 1..input.len() {
		forward_product *= input[i - 1];
		results[i] *= forward_product;
	}

	results
}

/// Computes the common denominator for Lagrange interpolation polynomials
/// in the univariate case over a binary subspace.
///
/// For a binary subspace, this computes the product of all nonzero elements
/// in the subspace.
///
/// # Arguments
/// * `binary_subspace` - The binary subspace defining the domain
///
/// # Returns
/// The product of all nonzero elements in the binary subspace
pub fn lexicographic_lagrange_denominator<
	FDomain: Field + BinaryField,
	FOutput: Field + From<FDomain>,
>(
	binary_subspace: &BinarySubspace<FDomain>,
) -> FOutput {
	FOutput::from(
		binary_subspace
			.iter()
			.filter(|x| *x != FDomain::ZERO)
			.product::<FDomain>(),
	)
}

/// Computes the numerators for Lagrange interpolation polynomials evaluated at a given point.
///
/// For a binary subspace and evaluation point z, this computes the numerators for each
/// Lagrange basis polynomial L_i(z). Specifically, for each basis point i in the subspace,
/// it computes the product of (z - j) for all j ≠ i.
///
/// This function supports field extension where the domain field can be embedded into
/// the output field.
///
/// # Arguments
/// * `eval_point` - The point at which to evaluate the Lagrange numerators
/// * `binary_subspace` - The binary subspace defining the domain
///
/// # Returns
/// A vector of numerators, one for each basis polynomial
pub fn lexicographic_lagrange_numerators<
	FDomain: Field + BinaryField,
	FOutput: Field + From<FDomain>,
>(
	eval_point: FOutput,
	binary_subspace: &BinarySubspace<FDomain>,
) -> Vec<FOutput> {
	let basis_point_differences: Vec<_> = binary_subspace
		.iter()
		.map(|i| eval_point - FOutput::from(i))
		.collect();

	products_excluding_one_element(&basis_point_differences)
}

/// Computes the Lagrange basis polynomial evaluations at a given point for a binary subspace.
///
/// This function computes the values of all Lagrange basis polynomials L_i(z) evaluated at
/// the given point z, where each L_i is the unique polynomial of degree < n that satisfies:
/// - L_i(x_i) = 1
/// - L_i(x_j) = 0 for all j ≠ i
///
/// For a binary subspace with basis points {x_0, x_1, ..., x_{n-1}}, the Lagrange basis
/// polynomial L_i(z) is given by:
///
/// L_i(z) = ∏_{j≠i} (z - x_j) / ∏_{j≠i} (x_i - x_j)
///
/// This function efficiently computes all L_i(z) values by:
/// 1. Computing the numerators ∏_{j≠i} (z - x_j) for each i
/// 2. Computing the common denominator (which is the same for all basis polynomials in a binary
///    subspace)
/// 3. Dividing each numerator by the common denominator
///
/// # Arguments
/// * `eval_point` - The point z at which to evaluate all Lagrange basis polynomials
/// * `binary_subspace` - The binary subspace defining the interpolation domain
///
/// # Returns
/// A vector containing L_i(z) for each basis point x_i in lexicographic order
pub fn lexicographic_lagrange_basis_vectors<
	FDomain: Field + BinaryField,
	FOutput: Field + From<FDomain>,
>(
	eval_point: FOutput,
	binary_subspace: &BinarySubspace<FDomain>,
) -> Vec<FOutput> {
	let mut result = lexicographic_lagrange_numerators(eval_point, binary_subspace);
	let inverse_denominator: FOutput =
		lexicographic_lagrange_denominator::<FDomain, FOutput>(binary_subspace).invert_or_zero();

	result
		.iter_mut()
		.for_each(|res| *res *= inverse_denominator);

	result
}
