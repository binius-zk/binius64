// Copyright 2023-2025 Irreducible Inc.

use super::packed_scaled::packed_scaled_field;
use crate::PackedBinaryField128x1b;

packed_scaled_field!(PackedBinaryField256x1b = [PackedBinaryField128x1b; 2]);
