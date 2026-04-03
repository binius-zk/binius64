// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication and squaring using x86_64 CLMUL instructions.

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

use crate::{PackedUnderlier, Underlier, ghash, underlier::OpsClmul};

/// Multiply packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn mul_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: [U; 2], y: [U; 2]) -> [U; 2] {
	super::sliced::mul_sliced(x, y, ghash::clmul::mul, ghash::clmul::mul_inv_x)
}

/// Square packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn square_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: [U; 2]) -> [U; 2] {
	super::sliced::square_sliced(x, ghash::clmul::square, ghash::clmul::mul_inv_x)
}

/// Multiply a GHASH² element stored as a pair of 128-bit registers.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2"
))]
#[inline]
pub fn mul_m128i(x: [__m128i; 2], y: [__m128i; 2]) -> [__m128i; 2] {
	mul_sliced(x, y)
}

/// Square a GHASH² element stored as a pair of 128-bit registers.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2"
))]
#[inline]
pub fn square_m128i(x: [__m128i; 2]) -> [__m128i; 2] {
	square_sliced(x)
}

/// Multiply packed GHASH² elements in 256-bit registers.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
	target_feature = "sse2"
))]
#[inline]
pub fn mul_m256i(x: __m256i, y: __m256i) -> __m256i {
	let t0_t2 = ghash::clmul::mul(x, y);
	let x0_y0 = unsafe { _mm256_permute2x128_si256::<0x20>(x, y) };
	let x1_y1 = unsafe { _mm256_permute2x128_si256::<0x31>(x, y) };
	let xxor_yxor = Underlier::xor(x0_y0, x1_y1);
	let invx = <__m256i as PackedUnderlier<u128>>::broadcast(ghash::INV_X);

	let invx_xxor = unsafe { _mm256_permute2x128_si256::<0x20>(invx, xxor_yxor) };
	let t2_yxor = unsafe { _mm256_permute2x128_si256::<0x31>(t0_t2, xxor_yxor) };
	let t2invx_t1 = ghash::clmul::mul(invx_xxor, t2_yxor);
	let t0_t0 = unsafe { _mm256_permute2x128_si256::<0x00>(t0_t2, t0_t2) };
	Underlier::xor(t0_t0, t2invx_t1)
}

/// Multiply packed GHASH² elements in 256-bit registers.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2",
	target_feature = "avx2"
))]
#[inline]
pub fn mul_m256i_as_m128i(x: __m256i, y: __m256i) -> __m256i {
	let x0 = unsafe { _mm256_extracti128_si256::<0>(x) };
	let x1 = unsafe { _mm256_extracti128_si256::<1>(x) };
	let y0 = unsafe { _mm256_extracti128_si256::<0>(y) };
	let y1 = unsafe { _mm256_extracti128_si256::<1>(y) };

	let [z0, z1] = mul_sliced([x0, x1], [y0, y1]);

	unsafe { _mm256_set_m128i(z1, z0) }
}

/// Multiply packed GHASH² elements in 256-bit registers.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
))]
#[inline]
pub fn mul_m256i_hybrid(x: __m256i, y: __m256i) -> __m256i {
	let t0_t2 = ghash::clmul::mul(x, y);

	let x0 = unsafe { _mm256_extracti128_si256::<0>(x) };
	let x1 = unsafe { _mm256_extracti128_si256::<1>(x) };
	let y0 = unsafe { _mm256_extracti128_si256::<0>(y) };
	let y1 = unsafe { _mm256_extracti128_si256::<1>(y) };
	let t0 = unsafe { _mm256_extracti128_si256::<0>(t0_t2) };
	let t2 = unsafe { _mm256_extracti128_si256::<1>(t0_t2) };

	let t1 = ghash::clmul::mul(Underlier::xor(x0, x1), Underlier::xor(y0, y1));

	let z0 = Underlier::xor(t0, ghash::clmul::mul_inv_x(t2));
	let z1 = Underlier::xor(t1, t0);
	unsafe { _mm256_set_m128i(z1, z0) }
}

#[cfg(test)]
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
	target_feature = "sse2"
))]
mod tests {
	use std::arch::x86_64::*;

	use proptest::prelude::*;

	use super::*;

	fn arb_m256i() -> impl Strategy<Value = __m256i> {
		(any::<u128>(), any::<u128>()).prop_map(|(low, high)| unsafe {
			let low_m128 = std::mem::transmute::<u128, __m128i>(low);
			let high_m128 = std::mem::transmute::<u128, __m128i>(high);
			_mm256_set_m128i(high_m128, low_m128)
		})
	}

	proptest! {
		#[test]
		fn test_mul_m256i_matches_reference(
			x in arb_m256i(),
			y in arb_m256i(),
		) {
			let expected = mul_m256i_as_m128i(x, y);
			let result = mul_m256i(x, y);
			prop_assert!(
				Underlier::is_equal(result, expected),
				"mul_m256i does not match mul_m256i_as_m128i"
			);
		}

		#[test]
		fn test_mul_m256i_hybrid_matches_reference(
			x in arb_m256i(),
			y in arb_m256i(),
		) {
			let expected = mul_m256i_as_m128i(x, y);
			let result = mul_m256i_hybrid(x, y);
			prop_assert!(
				Underlier::is_equal(result, expected),
				"mul_m256i_hybrid does not match mul_m256i_as_m128i"
			);
		}
	}
}
