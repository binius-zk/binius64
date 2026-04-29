// Copyright 2026 The Binius Developers

//! Circuit gadgets — functions generic over [`CircuitBuilder`] that take input wires and return
//! output wires.

use binius_field::{ExtensionField, Field};
use binius_spartan_frontend::circuit_builder::CircuitBuilder;

/// Transpose the subfield decomposition of extension field elements.
///
/// Given `d` input wires representing extension field elements (where `d` is the degree of
/// `B::Field` over `FSub`), returns `d` output wires containing the transposed elements.
///
/// Each input element decomposes as `input[i] = sum_j coeffs[i][j] * basis(j)` where
/// `coeffs[i][j]` are in `FSub`. The output satisfies `output[j] = sum_i coeffs[i][j] * basis(i)`,
/// i.e., `output[j].get_base(i) == input[i].get_base(j)`.
///
/// The gadget:
/// 1. Hints the `d × d` matrix of subfield coefficients
/// 2. Constrains each coefficient to lie in `FSub` via the Frobenius endomorphism
/// 3. Constrains that the coefficients reconstruct each input element
/// 4. Computes the transposed output as basis linear combinations
///
/// # Panics
///
/// * If `inputs.len() != <B::Field as ExtensionField<FSub>>::DEGREE`
/// * If `B::Field::CHARACTERISTIC != 2`
pub fn square_transpose<B: CircuitBuilder, FSub: Field>(
	builder: &mut B,
	inputs: &[B::Wire],
) -> Vec<B::Wire>
where
	B::Field: ExtensionField<FSub>,
{
	assert_eq!(
		B::Field::CHARACTERISTIC,
		2,
		"square_transpose gadget is only implemented for characteristic 2"
	);

	let degree = <B::Field as ExtensionField<FSub>>::DEGREE;
	assert_eq!(inputs.len(), degree);

	if degree == 1 {
		return inputs.to_vec();
	}

	// An element c of B::Field is in FSub iff c^(2^k) = c, where k = log_2(|FSub|).
	// Since |B::Field| = 2^ORDER_EXPONENT and [B::Field : FSub] = degree, we have
	// k = ORDER_EXPONENT / degree.
	let n_frobenius_squarings = B::Field::ORDER_EXPONENT / degree;

	// Hint the d×d matrix of subfield coefficients.
	// Each element decomposes as: inputs[i] = sum_j coeffs[i][j] * basis(j)
	// where coeffs[i][j] is in FSub (embedded in B::Field).
	let coeffs = (0..degree)
		.map(|i| {
			(0..degree)
				.map(|j| {
					let [c] = builder.hint([inputs[i]], move |[x]| {
						[B::Field::from(
							<B::Field as ExtensionField<FSub>>::get_base(&x, j),
						)]
					});
					c
				})
				.collect::<Vec<_>>()
		})
		.collect::<Vec<_>>();

	// Frobenius subfield membership check: for each coefficient c, verify
	// c^(2^k) = c by squaring k times and asserting equality.
	for row in &coeffs {
		for &c in row {
			let mut powered = c;
			for _ in 0..n_frobenius_squarings {
				powered = builder.mul(powered, powered);
			}
			builder.assert_eq(powered, c);
		}
	}

	// Reconstruction check: verify that the hinted coefficients actually
	// decompose each original element.
	// For each i: sum_j coeffs[i][j] * basis(j) == inputs[i]
	for i in 0..degree {
		let reconstructed = basis_linear_combination::<_, FSub>(builder, coeffs[i].iter().copied());
		builder.assert_eq(reconstructed, inputs[i]);
	}

	// Compute transposed elements: out[j] = sum_i coeffs[i][j] * basis(i)
	(0..degree)
		.map(|j| basis_linear_combination::<_, FSub>(builder, coeffs.iter().map(|row| row[j])))
		.collect::<Vec<_>>()
}

/// Compute the linear combination `sum_i scalars[i] * basis(i)` where `basis(i)` are the
/// extension field basis elements of `B::Field` over `FSub`.
fn basis_linear_combination<B: CircuitBuilder, FSub: Field>(
	builder: &mut B,
	scalars: impl ExactSizeIterator<Item = B::Wire>,
) -> B::Wire
where
	B::Field: ExtensionField<FSub>,
{
	assert_eq!(scalars.len(), <B::Field as ExtensionField<FSub>>::DEGREE);

	// basis(0) is always ONE, so the first term is just scalars[0].
	let mut scalars = scalars.enumerate();
	let (_, first) = scalars.next().expect("degree must be at least 1");
	scalars.fold(first, |sum, (j, scalar)| {
		let basis = builder.constant(<B::Field as ExtensionField<FSub>>::basis(j));
		let term = builder.mul(scalar, basis);
		builder.add(sum, term)
	})
}
