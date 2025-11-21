// Copyright 2024-2025 Irreducible Inc.

use super::packed_scaled::packed_scaled_field;
use crate::arch::packed_256::*;

packed_scaled_field!(PackedBinaryField512x1b = [PackedBinaryField256x1b; 2]);
