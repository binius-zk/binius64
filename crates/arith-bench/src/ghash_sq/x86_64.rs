// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication and squaring using x86_64 CLMUL instructions.

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

use crate::{PackedUnderlier, Underlier, ghash, underlier::OpsClmul};

/// Multiply packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn mul_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: [U; 2], y: [U; 2]) -> [U; 2] {
	super::sliced::mul_sliced(
		x,
		y,
		ghash::clmul::mul_wide,
		ghash::clmul::reduce,
		ghash::clmul::mul_x_wide,
	)
}

/// Widening (unreduced) multiply of packed GHASH² elements in sliced representation using CLMUL
/// arithmetic. Returns the three raw GHASH products (`[U; 3]` each); see
/// [`super::sliced::mul_wide_sliced`] and reduce with [`reduce_sliced`].
#[inline]
pub fn mul_wide_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(
	x: [U; 2],
	y: [U; 2],
) -> [[U; 3]; 3] {
	super::sliced::mul_wide_sliced(x, y, ghash::clmul::mul_wide)
}

/// Reduce the three raw products from [`mul_wide_sliced`] into a GHASH² element using CLMUL
/// arithmetic; see [`super::sliced::reduce_sliced`].
#[inline]
pub fn reduce_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(t: [[U; 3]; 3]) -> [U; 2] {
	super::sliced::reduce_sliced(t, ghash::clmul::reduce, ghash::clmul::mul_x_wide)
}

/// Square packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn square_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: [U; 2]) -> [U; 2] {
	super::sliced::square_sliced(x, ghash::clmul::square, ghash::clmul::mul_x)
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

/// Widening (unreduced) schoolbook multiply of two GHASH² elements packed in a 256-bit register.
///
/// Returns the raw products as `[corners, cross]`: `corners = [t0, t2]` packs `t0 = x0·y0` in
/// lane 0 and `t2 = x1·y1` in lane 1 (one 256-bit `mul_wide`), and `cross = [x1·y0, x0·y1]` is a
/// second `mul_wide` of the lane-swapped `x` against `y`. All four are raw base-field widening
/// products; the cross-term sum, multiply-by-`X`, and base-field reduction are F2-linear and
/// deferred to [`reduce_m256i`], so an inner product XOR-accumulates the products and reduces once.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
	target_feature = "sse2"
))]
#[inline]
pub fn mul_m256i_wide(x: __m256i, y: __m256i) -> [[__m256i; 3]; 2] {
	// Lane 0 = t0 = x0·y0, lane 1 = t2 = x1·y1 (unreduced).
	let corners = ghash::clmul::mul_wide(x, y);
	// Swap the 128-bit lanes of x to [x1, x0], so multiplying by y gives the cross products
	// lane 0 = x1·y0, lane 1 = x0·y1 (unreduced).
	let x_swapped = unsafe { _mm256_permute2x128_si256::<0x01>(x, x) };
	let cross = ghash::clmul::mul_wide(x_swapped, y);
	[corners, cross]
}

/// Reduce the raw products from [`mul_m256i_wide`] into a GHASH² element packed in a 256-bit
/// register.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
	target_feature = "sse2"
))]
#[inline]
pub fn reduce_m256i([corners, cross]: [[__m256i; 3]; 2]) -> __m256i {
	// Reduce the corners together (lane 0 = t0, lane 1 = t2), then extract the two base elements.
	let t0_t2 = ghash::clmul::reduce(corners);
	let t0 = unsafe { _mm256_extracti128_si256::<0>(t0_t2) };
	let t2 = unsafe { _mm256_extracti128_si256::<1>(t0_t2) };

	// Sum the two cross products x1·y0 + x0·y1 by extracting and XORing the lanes, then reduce.
	let cross_lo: [__m128i; 3] =
		std::array::from_fn(|i| unsafe { _mm256_extracti128_si256::<0>(cross[i]) });
	let cross_hi: [__m128i; 3] =
		std::array::from_fn(|i| unsafe { _mm256_extracti128_si256::<1>(cross[i]) });
	let cross_sum = ghash::clmul::reduce(Underlier::xor(cross_lo, cross_hi));

	// Y² = X·Y + X, so z0 = t0 + X·t2 and z1 = (x0·y1 + x1·y0) + X·t2.
	let x_t2 = ghash::clmul::mul_x(t2);
	let z0 = Underlier::xor(t0, x_t2);
	let z1 = Underlier::xor(cross_sum, x_t2);
	unsafe { _mm256_set_m128i(z1, z0) }
}

/// Multiply packed GHASH² elements in 256-bit registers with a schoolbook widening multiply.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
	target_feature = "sse2"
))]
#[inline]
pub fn mul_m256i(x: __m256i, y: __m256i) -> __m256i {
	reduce_m256i(mul_m256i_wide(x, y))
}

/// Widening (unreduced) multiply of two GHASH² elements packed in a 256-bit register, using only
/// 128-bit CLMUL by extracting the lanes.
///
/// Returns the three raw base-field products `[t0, t1, t2]` in sliced form (`[[__m128i; 3]; 3]`);
/// see [`mul_wide_sliced`]. Reduce with [`reduce_m256i_as_m128i`].
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2",
	target_feature = "avx2"
))]
#[inline]
pub fn mul_m256i_as_m128i_wide(x: __m256i, y: __m256i) -> [[__m128i; 3]; 3] {
	let x0 = unsafe { _mm256_extracti128_si256::<0>(x) };
	let x1 = unsafe { _mm256_extracti128_si256::<1>(x) };
	let y0 = unsafe { _mm256_extracti128_si256::<0>(y) };
	let y1 = unsafe { _mm256_extracti128_si256::<1>(y) };

	mul_wide_sliced([x0, x1], [y0, y1])
}

/// Reduce the raw products from [`mul_m256i_as_m128i_wide`] into a GHASH² element packed in a
/// 256-bit register.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2",
	target_feature = "avx2"
))]
#[inline]
pub fn reduce_m256i_as_m128i(t: [[__m128i; 3]; 3]) -> __m256i {
	let [z0, z1] = reduce_sliced(t);
	unsafe { _mm256_set_m128i(z1, z0) }
}

/// Multiply packed GHASH² elements in 256-bit registers, using only 128-bit CLMUL.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2",
	target_feature = "avx2"
))]
#[inline]
pub fn mul_m256i_as_m128i(x: __m256i, y: __m256i) -> __m256i {
	reduce_m256i_as_m128i(mul_m256i_as_m128i_wide(x, y))
}

/// Widening (unreduced) multiply of two GHASH² elements packed in a 256-bit register, forming the
/// corner products `t0`/`t2` with one 256-bit `mul_wide` and the middle product `t1` with a 128-bit
/// `mul_wide` on the extracted lanes.
///
/// Returns `(corners, middle)`: `corners = [t0, t2]` packs `t0 = x0·y0` in lane 0 and `t2 = x1·y1`
/// in lane 1, and `middle` is `t1 = (x0+x1)·(y0+y1)` as a 128-bit widening product. Both are raw
/// base-field products; reduce with [`reduce_m256i_hybrid`].
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
))]
#[inline]
pub fn mul_m256i_hybrid_wide(x: __m256i, y: __m256i) -> ([__m256i; 3], [__m128i; 3]) {
	// Lane 0 = t0 = x0·y0, lane 1 = t2 = x1·y1 (unreduced).
	let corners = ghash::clmul::mul_wide(x, y);

	let x0 = unsafe { _mm256_extracti128_si256::<0>(x) };
	let x1 = unsafe { _mm256_extracti128_si256::<1>(x) };
	let y0 = unsafe { _mm256_extracti128_si256::<0>(y) };
	let y1 = unsafe { _mm256_extracti128_si256::<1>(y) };

	// t1 = (x0+x1)·(y0+y1) (unreduced).
	let middle = ghash::clmul::mul_wide(Underlier::xor(x0, x1), Underlier::xor(y0, y1));
	(corners, middle)
}

/// Reduce the raw products from [`mul_m256i_hybrid_wide`] into a GHASH² element packed in a 256-bit
/// register.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
))]
#[inline]
pub fn reduce_m256i_hybrid((corners, middle): ([__m256i; 3], [__m128i; 3])) -> __m256i {
	// Reduce the corners together (lane 0 = t0, lane 1 = t2), then extract the two base elements.
	let t0_t2 = ghash::clmul::reduce(corners);
	let t0 = unsafe { _mm256_extracti128_si256::<0>(t0_t2) };
	let t2 = unsafe { _mm256_extracti128_si256::<1>(t0_t2) };
	let t1 = ghash::clmul::reduce(middle);

	// Y² = X·Y + X, so z0 = t0 + X·t2 and z1 = (t1 + t0 + t2) + X·t2.
	let x_t2 = ghash::clmul::mul_x(t2);
	let z0 = Underlier::xor(t0, x_t2);
	let z1 = Underlier::xor(z0, Underlier::xor(t1, t2));
	unsafe { _mm256_set_m128i(z1, z0) }
}

/// Multiply packed GHASH² elements in 256-bit registers, mixing a 256-bit corner multiply with a
/// 128-bit middle multiply.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
))]
#[inline]
pub fn mul_m256i_hybrid(x: __m256i, y: __m256i) -> __m256i {
	reduce_m256i_hybrid(mul_m256i_hybrid_wide(x, y))
}

// CLMUL sliced multiply cross-checks that run wherever `pclmulqdq` is available (the tests below
// additionally need `vpclmulqdq`/`avx2` for the packed-256 variants).
#[cfg(test)]
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2"
))]
mod clmul_tests {
	use std::arch::x86_64::__m128i;

	use proptest::prelude::*;

	use super::*;
	use crate::ghash_sq::soft64;

	fn to_u(x: u128) -> __m128i {
		<__m128i as PackedUnderlier<u128>>::broadcast(x)
	}

	fn from_u(x: __m128i) -> u128 {
		<__m128i as PackedUnderlier<u128>>::get(x, 0)
	}

	proptest! {
		// The CLMUL delayed-reduction sliced multiply agrees with the soft64 reference.
		#[test]
		fn test_clmul_mul_sliced_matches_soft64(
			a in any::<[u128; 2]>(), b in any::<[u128; 2]>(),
		) {
			let got = mul_sliced::<__m128i>([to_u(a[0]), to_u(a[1])], [to_u(b[0]), to_u(b[1])]);
			prop_assert_eq!([from_u(got[0]), from_u(got[1])], soft64::mul_sliced(a, b));
		}

		// Deferred reduction: accumulating the three raw products by XOR and calling reduce_sliced
		// once equals summing the reduced products.
		#[test]
		fn test_clmul_wide_sliced_deferred_reduction(
			a1 in any::<[u128; 2]>(), b1 in any::<[u128; 2]>(),
			a2 in any::<[u128; 2]>(), b2 in any::<[u128; 2]>(),
		) {
			let mul = |a: [u128; 2], b: [u128; 2]| {
				mul_wide_sliced::<__m128i>([to_u(a[0]), to_u(a[1])], [to_u(b[0]), to_u(b[1])])
			};
			let acc = <[[__m128i; 3]; 3]>::xor(mul(a1, b1), mul(a2, b2));
			let reduced = reduce_sliced::<__m128i>(acc);
			let sum = <[u128; 2]>::xor(soft64::mul_sliced(a1, b1), soft64::mul_sliced(a2, b2));
			prop_assert_eq!([from_u(reduced[0]), from_u(reduced[1])], sum);
		}
	}
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
	use crate::test_utils::multiplication_tests::test_square_equals_mul;

	fn arb_m128i() -> impl Strategy<Value = __m128i> {
		any::<u128>().prop_map(|val| unsafe { std::mem::transmute::<u128, __m128i>(val) })
	}

	fn arb_m128i_pair() -> impl Strategy<Value = [__m128i; 2]> {
		(arb_m128i(), arb_m128i()).prop_map(|(a, b)| [a, b])
	}

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

		#[test]
		fn test_square_m128i_matches_mul(
			x in arb_m128i_pair(),
		) {
			test_square_equals_mul(x, mul_m128i, square_m128i, "GHASH²");
		}
	}
}
