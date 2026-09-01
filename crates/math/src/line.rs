// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Extrapolation of the line through two points.

use binius_field::{PackedField, field::FieldOps};

/// Extrapolates a line through two points.
///
/// The two points are `(0, x_0)` and `(1, x_1)`.
/// The line through them, evaluated at the parameter `z`, is
///
/// ```text
/// x_0 + (x_1 - x_0) * z
/// ```
///
/// The two points are also the halves a variable splits a multilinear into.
/// Read that way, this binds that variable to `z`.
#[inline]
pub fn extrapolate_line<F: FieldOps>(x0: F, x1: F, z: F) -> F {
	// The line is affine in `z`, so one multiplication and two additions suffice.
	x0.clone() + (x1 - x0) * z
}

/// Extrapolates a line through two points, at a preprocessed parameter.
///
/// The line is the same one the plain form names.
///
/// Binding a variable runs over a whole buffer at one parameter, so preprocessing it once pays for
/// every pair.
#[inline]
pub fn extrapolate_line_preprocessed<P: PackedField>(x0: P, x1: P, z: &impl Fn(P) -> P) -> P {
	x0 + z(x1 - x0)
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Random, field::FieldOps};
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{B128, Packed128b};

	type P = Packed128b;
	type F = B128;

	#[test]
	fn extrapolate_line_reads_the_endpoints_at_zero_and_one() {
		let mut rng = StdRng::seed_from_u64(0);

		// The line is pinned by its two endpoints, which is what fixes the argument order:
		//
		//     z = 0  ->  x_0
		//     z = 1  ->  x_1
		let x0 = F::random(&mut rng);
		let x1 = F::random(&mut rng);
		assert_eq!(extrapolate_line(x0, x1, F::ZERO), x0);
		assert_eq!(extrapolate_line(x0, x1, F::ONE), x1);

		// A packed field is a vector of independent lanes, and the same must hold lane by lane.
		let x0_packed = P::random(&mut rng);
		let x1_packed = P::random(&mut rng);
		assert_eq!(extrapolate_line(x0_packed, x1_packed, P::zero()), x0_packed);
		assert_eq!(extrapolate_line(x0_packed, x1_packed, P::one()), x1_packed);
	}

	#[test]
	fn extrapolate_line_matches_the_closed_form() {
		let mut rng = StdRng::seed_from_u64(0);

		// Away from the endpoints the value is the closed form of the line through them.
		// Drawing `z` from a proper subfield also covers the cheaper subfield multiplication.
		for _ in 0..10 {
			let x0 = F::random(&mut rng);
			let x1 = F::random(&mut rng);
			let z = F::from(rng.next_u64() as u128);
			assert_eq!(extrapolate_line(x0, x1, z), x0 + (x1 - x0) * z);
		}
	}

	#[test]
	fn extrapolate_line_preprocessed_matches_the_plain_form() {
		let mut rng = StdRng::seed_from_u64(0);

		// Preprocessing the parameter changes its representation, not the line it names.
		for _ in 0..10 {
			let x0 = P::random(&mut rng);
			let x1 = P::random(&mut rng);
			let z = F::random(&mut rng);
			assert_eq!(
				extrapolate_line_preprocessed(x0, x1, &P::preprocess_mul(z)),
				extrapolate_line(x0, x1, P::broadcast(z))
			);
		}
	}
}
