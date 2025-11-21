// Copyright 2025 Irreducible Inc.
//! SIMD binary field operations for the Rijndael field.
//!
//! This module implements arithmetic operations in the GF(2^8) binary field used by
//! the AES/Rijndael S-Box. This is the finite field with 256 elements defined by the
//! reduction polynomial x^8 + x^4 + x^3 + x + 1.
//!
//! The implementation leverages Galois Field New Instructions (GFNI) available on modern
//! Intel processors (Ice Lake and newer) for highly efficient field multiplication and
//! inversion operations. GFNI provides dedicated hardware support for GF(2^8) arithmetic,
//! making these operations significantly faster than software implementations.
//!
//! The Rijndael field is fundamental to AES encryption, where it's used in the SubBytes
//! transformation (S-Box) to provide non-linearity in the cipher.

use crate::underlier::{OpsGfni, PackedUnderlier, Underlier};

pub fn mul_gfni<U: Underlier + OpsGfni>(a: U, b: U) -> U {
	OpsGfni::gf2p8mul(a, b)
}

pub fn sqr_gfni<U: Underlier + OpsGfni>(x: U) -> U {
	OpsGfni::gf2p8mul(x, x)
}

pub fn inv_gfni<U: Underlier + OpsGfni + PackedUnderlier<u64>>(x: U) -> U {
	#[rustfmt::skip]
	pub const IDENTITY_MAP: u64 = u64::from_le_bytes([
		0b10000000,
		0b01000000,
		0b00100000,
		0b00010000,
		0b00001000,
		0b00000100,
		0b00000010,
		0b00000001,
	]);

	let identity_map = <U as PackedUnderlier<u64>>::broadcast(IDENTITY_MAP);
	OpsGfni::gf2p8affineinv::<0>(x, identity_map)
}
