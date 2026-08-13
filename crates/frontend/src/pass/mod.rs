// Copyright 2026 The Binius Developers

//! Transformations over the gate graph, each rewriting it in place or reporting what it found.

pub mod const_prop;
pub mod cse;
pub mod dce;
pub mod fusion;
pub mod layout;
pub mod zero_fold;
