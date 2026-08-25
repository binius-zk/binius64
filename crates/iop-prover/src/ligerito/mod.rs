// Copyright 2026 The Binius Developers

//! Prover for the Ligerito polynomial commitment scheme.
//!
//! The verifier side lives in [`binius_iop::ligerito`], where the protocol is described.
//! This module holds the prover that commits a message and then opens it.
//! Beside it sits the channel that puts that prover behind the IOP prover channel trait.

pub mod channel;
pub mod compiler;
mod induced_weight;
pub(crate) mod opening;

pub use opening::LigeritoProver;
