// Copyright 2024-2025 Irreducible Inc.

use super::packed_scaled::packed_scaled_field;
use crate::PackedAESBinaryField16x8b;

packed_scaled_field!(PackedAESBinaryField32x8b = [PackedAESBinaryField16x8b; 2]);
