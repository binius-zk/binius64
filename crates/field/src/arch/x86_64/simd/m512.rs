// Copyright 2024-2025 Irreducible Inc.

use core::arch::x86_64::*;

use super::simd_arithmetic::TowerSimdType;
use crate::arch::x86_64::m512::M512;

impl TowerSimdType for M512 {
	#[inline(always)]
	fn set_epi_64(val: i64) -> Self {
		unsafe { _mm512_set1_epi64(val) }.into()
	}
}
