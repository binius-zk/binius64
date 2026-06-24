// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{arch::M256, underlier::ScaledUnderlier};

pub type M512 = ScaledUnderlier<M256, 2>;
