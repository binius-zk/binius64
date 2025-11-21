// Copyright 2024-2025 Irreducible Inc.

use super::packed_scaled::packed_scaled_field;
use crate::arch::packed_aes_256::*;

packed_scaled_field!(PackedAESBinaryField64x8b = [PackedAESBinaryField32x8b; 2]);
