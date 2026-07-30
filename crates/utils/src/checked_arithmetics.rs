// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 Authors

/// Division implementation that fails in case when `a`` isn't divisible by `b`
pub const fn checked_int_div(a: usize, b: usize) -> usize {
	let result = a / b;
	assert!(b * result == a);

	result
}

/// Computes binary logarithm of `val`.
/// If `val` is not a power of 2, returns `None`.
#[inline]
#[must_use]
pub const fn strict_log_2(val: usize) -> Option<usize> {
	if val == 0 {
		return None;
	}

	let pow = val.trailing_zeros();
	if val.wrapping_shr(pow) == 1 {
		Some(pow as usize)
	} else {
		None
	}
}

/// Computes the binary logarithm of `val`.
///
/// `#[track_caller]` puts the panic at the call site rather than inside this function.
/// A `const fn` cannot format a panic message, so the offending value is not reported.
///
/// # Panics
/// Panics if `val` is not a power of two, zero included.
#[inline]
#[must_use]
#[track_caller]
pub const fn checked_log_2(val: usize) -> usize {
	strict_log_2(val).expect("value is not a power of two")
}

/// Computes the binary logarithm of $n$ rounded up to the nearest integer.
///
/// When $n$ is 0, this function returns 0. Otherwise, it returns $\lceil \log_2 n \rceil$.
#[must_use]
pub const fn log2_ceil_usize(n: usize) -> usize {
	min_bits(n.saturating_sub(1))
}

/// Returns the number of bits needed to represent $n$.
///
/// When $n$ is 0, this function returns 0. Otherwise, it returns $\lfloor \log_2 n \rfloor + 1$.
#[must_use]
pub const fn min_bits(n: usize) -> usize {
	(usize::BITS - n.leading_zeros()) as usize
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_checked_int_div_success() {
		assert_eq!(checked_int_div(6, 1), 6);
		assert_eq!(checked_int_div(6, 2), 3);
		assert_eq!(checked_int_div(6, 6), 1);
	}

	#[test]
	#[should_panic]
	const fn test_checked_int_div_fail() {
		_ = checked_int_div(5, 2);
	}

	// Number of bits in `n`, counted one shift at a time.
	fn min_bits_ref(n: usize) -> usize {
		let mut bits = 0;
		let mut rest = n;
		while rest > 0 {
			bits += 1;
			rest >>= 1;
		}
		bits
	}

	// Smallest `k` with `2^k >= n`, found by counting up.
	fn log2_ceil_ref(n: usize) -> usize {
		let mut k = 0;
		while (1usize << k) < n {
			k += 1;
		}
		k
	}

	#[test]
	fn test_checked_log2_success() {
		assert_eq!(checked_log_2(1), 0);
		assert_eq!(checked_log_2(2), 1);
		assert_eq!(checked_log_2(4), 2);
		assert_eq!(checked_log_2(64), 6);
		assert_eq!(checked_log_2(1 << 63), 63);
	}

	#[test]
	#[should_panic]
	const fn test_checked_log2_fail() {
		_ = checked_log_2(6)
	}

	#[test]
	#[should_panic(expected = "value is not a power of two")]
	fn test_checked_log2_zero_panics() {
		_ = checked_log_2(0);
	}

	// Callers put `checked_log_2` in associated-const initializers, so const-ness is load-bearing.
	const _: () = assert!(checked_log_2(64) == 6);

	#[test]
	fn test_strict_log_2_boundaries() {
		assert_eq!(strict_log_2(0), None);
		assert_eq!(strict_log_2(1), Some(0));
		assert_eq!(strict_log_2(2), Some(1));
		assert_eq!(strict_log_2(3), None);
		assert_eq!(strict_log_2(1 << 63), Some(63));
		assert_eq!(strict_log_2(usize::MAX), None);
	}

	#[test]
	fn test_min_bits_boundaries() {
		assert_eq!(min_bits(0), 0);
		assert_eq!(min_bits(1), 1);
		assert_eq!(min_bits(2), 2);
		assert_eq!(min_bits(3), 2);
		assert_eq!(min_bits(1 << 63), 64);
		assert_eq!(min_bits(usize::MAX), 64);
	}

	#[test]
	fn test_log2_ceil_usize_boundaries() {
		assert_eq!(log2_ceil_usize(0), 0);
		assert_eq!(log2_ceil_usize(1), 0);
		assert_eq!(log2_ceil_usize(2), 1);
		assert_eq!(log2_ceil_usize(3), 2);
		// The last exact power of two, then the first value that needs one more bit.
		assert_eq!(log2_ceil_usize(1 << 63), 63);
		assert_eq!(log2_ceil_usize((1 << 63) + 1), 64);
		assert_eq!(log2_ceil_usize(usize::MAX), 64);
	}

	// Exhaustive over the low 2^20, which beats sampling on a domain this small.
	#[test]
	fn test_bit_counts_match_reference() {
		for n in 0..1usize << 20 {
			assert_eq!(min_bits(n), min_bits_ref(n), "min_bits({n})");
			assert_eq!(log2_ceil_usize(n), log2_ceil_ref(n), "log2_ceil_usize({n})");
		}
	}

	// The three functions agree where their domains overlap: rounding a power of two up is a no-op,
	// and `min_bits` of a power of two is one more than its logarithm.
	#[test]
	fn test_powers_of_two_agree() {
		for log in 0..usize::BITS as usize {
			let n = 1usize << log;
			assert_eq!(checked_log_2(n), log);
			assert_eq!(log2_ceil_usize(n), log);
			assert_eq!(min_bits(n), log + 1);
		}
	}
}
