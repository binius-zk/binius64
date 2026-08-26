// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Re-exports `rayon`, plus task-sizing and thread-pool-configuration helpers built on top of it.

pub use rayon::*;

pub mod config;
pub mod task_size;
