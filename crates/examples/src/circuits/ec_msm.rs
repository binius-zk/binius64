// Copyright 2026 The Binius Developers
use std::iter;

use anyhow::Result;
use binius_circuits::{
	bignum::{BigUint, assert_eq},
	ecdsa::scalar_mul::msm_strauss_endo,
	secp256k1::{N_LIMBS, Secp256k1, Secp256k1Affine},
};
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, WitnessFiller};
use clap::Args;
use k256::{
	ProjectivePoint, Scalar, U256,
	elliptic_curve::{ops::MulByGenerator, scalar::FromUintUnchecked, sec1::ToEncodedPoint},
};
use rand::prelude::*;

use crate::ExampleCircuit;

/// Example circuit that computes a multi-scalar multiplication `Σ_i scalars[i] · points[i]` over
/// secp256k1 with a statically-known number of points, using the GLV fixed-window (Straus)
/// algorithm (see [`msm_strauss_endo`]).
///
/// The points and scalars are public inputs (the points are constrained to lie on the curve), and
/// the claimed result is a public input that the circuit checks against the in-circuit MSM.
pub struct EcMsmExample {
	scalars: Vec<BigUint>,
	points: Vec<AffinePoint>,
	expected: AffinePoint,
}

/// Public input wires for an affine secp256k1 point (assumed not to be the point at infinity).
struct AffinePoint {
	x: BigUint,
	y: BigUint,
}

impl AffinePoint {
	fn new_inout(builder: &mut CircuitBuilder) -> Self {
		Self {
			x: BigUint::new_inout(builder, N_LIMBS),
			y: BigUint::new_inout(builder, N_LIMBS),
		}
	}
}

#[derive(Args, Debug, Clone)]
pub struct Params {
	/// Number of (scalar, point) pairs in the multi-scalar multiplication
	#[arg(short = 'n', long, default_value_t = 2, value_parser = clap::value_parser!(u16).range(1..))]
	pub n_points: u16,
	/// Window size in bits for the Straus algorithm (4 is typically optimal)
	#[arg(short = 'w', long, default_value_t = 4, value_parser = clap::value_parser!(u16).range(1..64))]
	pub window: u16,
}

#[derive(Args, Debug, Clone)]
pub struct Instance {}

impl ExampleCircuit for EcMsmExample {
	type Params = Params;
	type Instance = Instance;

	fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
		let n_points = params.n_points as usize;
		let curve = Secp256k1::new(builder);

		let scalars = (0..n_points)
			.map(|_| BigUint::new_inout(builder, N_LIMBS))
			.collect::<Vec<_>>();
		let points = (0..n_points)
			.map(|_| AffinePoint::new_inout(builder))
			.collect::<Vec<_>>();

		// Build affine inputs and constrain each to be a valid curve point.
		let affine_points = points
			.iter()
			.map(|p| {
				let point = Secp256k1Affine {
					x: p.x.clone(),
					y: p.y.clone(),
					is_point_at_infinity: builder.add_constant(Word::ZERO),
				};
				curve.assert_on_curve(builder, &point);
				point
			})
			.collect::<Vec<_>>();

		let window = params.window as usize;
		let result = msm_strauss_endo(builder, &curve, window, &scalars, &affine_points);

		let expected = AffinePoint::new_inout(builder);
		assert_eq(builder, "msm_result_x", &result.x, &expected.x);
		assert_eq(builder, "msm_result_y", &result.y, &expected.y);

		Ok(Self {
			scalars,
			points,
			expected,
		})
	}

	fn populate_witness(&self, _instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(42);
		let mut expected = ProjectivePoint::IDENTITY;

		for (scalar, point) in iter::zip(&self.scalars, &self.points) {
			// Random 256-bit multiplier scalar.
			let mut scalar_bytes = [0u8; 32];
			rng.fill(&mut scalar_bytes);
			let k256_scalar = Scalar::from_uint_unchecked(U256::from_be_slice(&scalar_bytes));

			// Random curve point, generated as `g^r` so that it is guaranteed on-curve.
			let mut point_seed = [0u8; 32];
			rng.fill(&mut point_seed);
			let r = Scalar::from_uint_unchecked(U256::from_be_slice(&point_seed));
			let p = ProjectivePoint::mul_by_generator(&r);
			expected += p * k256_scalar;

			scalar.populate_limbs(w, &le_limbs(&num_bigint::BigUint::from_bytes_be(&scalar_bytes)));
			populate_point(w, &p, &point.x, &point.y);
		}

		populate_point(w, &expected, &self.expected.x, &self.expected.y);

		Ok(())
	}

	fn param_summary(params: &Self::Params) -> Option<String> {
		Some(format!("{}p-w{}", params.n_points, params.window))
	}
}

/// Populate the `x` and `y` limb wires from a k256 projective point's affine coordinates.
fn populate_point(w: &mut WitnessFiller, p: &ProjectivePoint, x: &BigUint, y: &BigUint) {
	let bytes = p.to_affine().to_encoded_point(false).to_bytes();
	// Uncompressed SEC1 encoding: `0x04 || x (32 bytes) || y (32 bytes)`.
	x.populate_limbs(w, &le_limbs(&num_bigint::BigUint::from_bytes_be(&bytes[1..33])));
	y.populate_limbs(w, &le_limbs(&num_bigint::BigUint::from_bytes_be(&bytes[33..65])));
}

/// Decompose a big integer into exactly `N_LIMBS` little-endian 64-bit limbs (zero-padded).
fn le_limbs(value: &num_bigint::BigUint) -> Vec<u64> {
	let mut limbs = value.to_u64_digits();
	assert!(limbs.len() <= N_LIMBS, "value exceeds {N_LIMBS} limbs");
	limbs.resize(N_LIMBS, 0);
	limbs
}
