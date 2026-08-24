// Copyright 2026 The Binius Developers

//! Proving one committed Ligerito level.
//!
//! The verifier side lives in [`binius_iop::ligerito`], where the protocol is described.
//! This module holds [`LevelProver`], which commits a level's message and then opens it.

mod opening;

pub use opening::LevelProver;
