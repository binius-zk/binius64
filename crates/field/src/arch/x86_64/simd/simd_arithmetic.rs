// Copyright 2024-2025 Irreducible Inc.

use crate::underlier::UnderlierType;

/// SIMD underlier operations backing the x86_64 GFNI field arithmetic.
///
/// Only exercised when the `gfni` target feature is enabled.
/// On other x86_64 builds nothing calls it, hence the `dead_code` allow.
#[allow(dead_code)]
pub trait TowerSimdType: Sized + Copy + UnderlierType {
	/// Broadcast a 64-bit value across every 64-bit lane.
	fn set_epi_64(val: i64) -> Self;
}
