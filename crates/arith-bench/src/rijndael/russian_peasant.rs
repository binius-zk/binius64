// Copyright 2026 The Binius Developers
//! GF(2^8) Rijndael field multiplication via a branch-free bit-twiddling algorithm, vectorized
//! per-byte across a SIMD underlier.
//!
//! Unlike [`gfni::mul`](super::gfni::mul), which lowers to a single hardware `gf2p8mul`
//! instruction, this path uses no GFNI — making it a candidate software fallback for x86_64 CPUs
//! without GFNI (and, with the same idea, for Wasm). The per-byte kernel is the constant-time
//! Russian-peasant multiply from Vault's Shamir secret-sharing implementation:
//!
//! ```text
//! r = 0
//! for i in (0..8).rev():
//!     r = (-(b >> i & 1) & a) ^ (-(r >> 7) & 0x1b) ^ (r + r)
//! ```
//!
//! Every operation (byte doubling, sign-bit mask, AND, XOR) is lane-parallel, so the scalar
//! kernel vectorizes directly. The lane primitives it needs — 8-bit add, a per-byte "sign mask",
//! and a 16-bit lane shift — are abstracted behind [`RijndaelBytewise`] so the same [`mul`] runs
//! over `__m128i` (SSE2), `__m256i` (AVX2), and `__m512i` (AVX-512BW). The two's-complement masks
//! `-(x & 1)` become signed byte compares (or AVX-512 mask registers), `r + r` becomes an 8-bit
//! add, and testing bit `i` of each byte is a left shift that moves it into the sign bit.

use seq_macro::seq;

use crate::underlier::{PackedUnderlier, Underlier};

/// Byte-lane primitives for the Russian-peasant GF(2^8) multiply, specialized per SIMD underlier.
pub trait RijndaelBytewise: Underlier + PackedUnderlier<u8> {
	/// Lane-wise wrapping 8-bit addition. Adding a value to itself is a byte-lane `<< 1` (with the
	/// carry out of bit 7 dropped — the reduction step accounts for it).
	fn add_epi8(a: Self, b: Self) -> Self;

	/// Per-byte mask: `0xff` in lanes whose bit 7 is set, `0x00` otherwise.
	fn sign_mask_epi8(a: Self) -> Self;

	/// Logical left shift of each 16-bit lane by `count` (a small runtime amount, 0..8). Used to
	/// move bit `i` of each byte into the byte's sign bit so
	/// [`sign_mask_epi8`](Self::sign_mask_epi8) can broadcast it. A runtime count keeps one
	/// primitive across widths, since the AVX-512 immediate-shift intrinsic uses a different
	/// const-generic type than the SSE/AVX ones.
	fn sll_epi16(a: Self, count: u32) -> Self;
}

/// Multiply packed GF(2^8) Rijndael elements (the byte lanes of `U`) using the branch-free
/// Russian-peasant algorithm, with the AES reduction polynomial x^8 + x^4 + x^3 + x + 1.
#[inline]
pub fn mul<U: RijndaelBytewise>(a: U, b: U) -> U {
	// Low byte of the reduction polynomial (x^4 + x^3 + x + 1), XORed in whenever a byte doubling
	// carries out of bit 7.
	let poly = <U as PackedUnderlier<u8>>::broadcast(0x1b);

	let mut r = U::ZERO;
	// MSB-first double-and-add. On step `k` we test bit `i = 7 - k` of `b`: shifting `b` left by
	// `k` moves that bit into each byte's sign bit, so `sign_mask_epi8` broadcasts it to a
	// full-byte mask.
	seq!(k in 0..8 {
		// Conditional reduction from the *previous* r: XOR 0x1b where bit 7 of r is set.
		let reduce = U::and(U::sign_mask_epi8(r), poly);
		// Byte-lane `r << 1` (the bit carried out of bit 7 is accounted for by `reduce`).
		let dbl = U::add_epi8(r, r);
		// Broadcast bit `7 - k` of each byte of `b` to a full-byte mask, then select `a`.
		let addend = U::and(U::sign_mask_epi8(U::sll_epi16(b, k)), a);
		r = U::xor(U::xor(addend, reduce), dbl);
	});
	r
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
impl RijndaelBytewise for std::arch::x86_64::__m128i {
	#[inline]
	fn add_epi8(a: Self, b: Self) -> Self {
		unsafe { std::arch::x86_64::_mm_add_epi8(a, b) }
	}

	#[inline]
	fn sign_mask_epi8(a: Self) -> Self {
		unsafe { std::arch::x86_64::_mm_cmpgt_epi8(std::arch::x86_64::_mm_setzero_si128(), a) }
	}

	#[inline]
	fn sll_epi16(a: Self, count: u32) -> Self {
		use std::arch::x86_64::{_mm_cvtsi32_si128, _mm_sll_epi16};
		unsafe { _mm_sll_epi16(a, _mm_cvtsi32_si128(count as i32)) }
	}
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl RijndaelBytewise for std::arch::x86_64::__m256i {
	#[inline]
	fn add_epi8(a: Self, b: Self) -> Self {
		unsafe { std::arch::x86_64::_mm256_add_epi8(a, b) }
	}

	#[inline]
	fn sign_mask_epi8(a: Self) -> Self {
		unsafe {
			std::arch::x86_64::_mm256_cmpgt_epi8(std::arch::x86_64::_mm256_setzero_si256(), a)
		}
	}

	#[inline]
	fn sll_epi16(a: Self, count: u32) -> Self {
		use std::arch::x86_64::{_mm_cvtsi32_si128, _mm256_sll_epi16};
		unsafe { _mm256_sll_epi16(a, _mm_cvtsi32_si128(count as i32)) }
	}
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl RijndaelBytewise for std::arch::x86_64::__m512i {
	#[inline]
	fn add_epi8(a: Self, b: Self) -> Self {
		unsafe { std::arch::x86_64::_mm512_add_epi8(a, b) }
	}

	#[inline]
	fn sign_mask_epi8(a: Self) -> Self {
		// AVX-512 signed compares yield an opmask register; broadcast each byte's sign bit back to
		// a full-byte vector mask.
		unsafe { std::arch::x86_64::_mm512_movm_epi8(std::arch::x86_64::_mm512_movepi8_mask(a)) }
	}

	#[inline]
	fn sll_epi16(a: Self, count: u32) -> Self {
		use std::arch::x86_64::{_mm_cvtsi32_si128, _mm512_sll_epi16};
		unsafe { _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32)) }
	}
}

#[cfg(test)]
mod tests {
	#[allow(unused_imports)]
	use proptest::{collection::vec, prelude::*};

	#[allow(unused_imports)]
	use super::*;

	/// Reference GF(2^8) Rijndael multiply (classic LSB-first shift-and-add). Deliberately a
	/// different algorithm than the MSB-first SIMD kernel under test, so it catches transcription
	/// bugs rather than sharing them.
	#[allow(dead_code)]
	fn mul_scalar(mut a: u8, mut b: u8) -> u8 {
		let mut r = 0u8;
		for _ in 0..8 {
			if b & 1 != 0 {
				r ^= a;
			}
			let carry = a & 0x80;
			a <<= 1;
			if carry != 0 {
				a ^= 0x1b;
			}
			b >>= 1;
		}
		r
	}

	/// Pack the byte slices into `U`, run the SIMD `mul`, and check every lane against the scalar
	/// oracle.
	#[allow(dead_code)]
	fn check<U: RijndaelBytewise>(a: &[u8], b: &[u8]) -> Result<(), TestCaseError> {
		let mut pa = U::ZERO;
		let mut pb = U::ZERO;
		for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
			pa = pa.set(i, x);
			pb = pb.set(i, y);
		}
		let pr = mul(pa, pb);
		for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
			prop_assert_eq!(pr.get(i), mul_scalar(x, y), "lane {}", i);
		}
		Ok(())
	}

	proptest! {
		#[test]
		#[cfg(target_feature = "sse4.1")]
		fn mul_m128i(a in vec(any::<u8>(), 16), b in vec(any::<u8>(), 16)) {
			check::<std::arch::x86_64::__m128i>(&a, &b)?;
		}

		#[test]
		#[cfg(target_feature = "avx2")]
		fn mul_m256i(a in vec(any::<u8>(), 32), b in vec(any::<u8>(), 32)) {
			check::<std::arch::x86_64::__m256i>(&a, &b)?;
		}

		#[test]
		#[cfg(target_feature = "avx512bw")]
		fn mul_m512i(a in vec(any::<u8>(), 64), b in vec(any::<u8>(), 64)) {
			check::<std::arch::x86_64::__m512i>(&a, &b)?;
		}
	}
}
