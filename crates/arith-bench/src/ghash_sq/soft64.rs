// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication using the soft64 GHASH implementation.

use crate::ghash;

/// Multiply packed GHASH² elements in sliced representation using soft64 arithmetic.
pub fn mul_sliced(x: [u128; 2], y: [u128; 2]) -> [u128; 2] {
	super::sliced::mul_sliced(x, y, ghash::soft64::mul, ghash::soft64::mul_inv_x)
}
