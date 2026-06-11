// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire, util::all_true};

use super::scalar_mul::{MSM_WINDOW, msm_strauss_endo};
use crate::{
	bignum::{BigUint, biguint_lt},
	secp256k1::{Secp256k1, Secp256k1Affine},
};

/// "Bitcoin style" verification of ECDSA signatures over secp256k1
///
/// # Arguments
/// * `pk` - public key, a curve point in affine representation, asserted to be a valid curve point
/// * `z`  - hash of signed message as an integer
/// * `r`  - R part of the signature, the x coordinate of the nonce point
/// * `s`  - S part of the signature
///
/// # Assertions
/// The public key `pk` is asserted to be a valid curve point
///
/// # Outputs
/// A boolean wire equal to all-1 if the signature is valid, all-0 otherwise.
pub fn verify(
	b: &CircuitBuilder,
	pk: Secp256k1Affine,
	z: &BigUint,
	r: &BigUint,
	s: &BigUint,
) -> Wire {
	let curve = Secp256k1::new(b);

	// secp256k1 is cofactor-1 and PAI is not on it.
	curve.assert_on_curve(b, &pk);

	let f_scalar = curve.f_scalar();

	let valid_r = b.band(b.bnot(r.is_zero(b)), biguint_lt(b, r, f_scalar.modulus()));
	let valid_s = b.band(b.bnot(s.is_zero(b)), biguint_lt(b, s, f_scalar.modulus()));

	// u1 = z / s, u2 = r / s. `div` requires reduced dividends: `z` (the message hash) and
	// `r` must be in `[0, n)`. `valid_s` gates the shared divisor `s`.
	let u1 = f_scalar.div(b, z, s, valid_s);
	let u2 = f_scalar.div(b, r, s, valid_s);

	// Recover the candidate nonce point as the multi-scalar multiplication `u1*G + u2*PK`.
	let g = Secp256k1Affine::generator(b);
	let nonce = msm_strauss_endo(b, &curve, MSM_WINDOW, &[u1, u2], &[g, pk]);
	let nonce_not_pai = b.bnot(nonce.is_point_at_infinity);
	let r_diff = curve.f_p().sub(b, &nonce.x, r);

	let conditions = [valid_r, valid_s, nonce_not_pai, r_diff.is_zero(b)];
	all_true(b, conditions)
}
