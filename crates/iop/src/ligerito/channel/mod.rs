// Copyright 2026 The Binius Developers

//! Ligerito implementation of the IOP verifier channel.
//!
//! One file per concern: the oracle handle here, the queued relation beside it, and the channel
//! that turns a queue of relations into one ladder opening.

mod relation;
mod verifier;

pub use verifier::LigeritoVerifierChannel;

/// A handle to the oracle a Ligerito channel opens.
///
/// The inner field is private, so the only way to hold one is to have received the commitment.
/// A ladder opens exactly one committed message, so the handle carries nothing else.
#[derive(Debug, Clone, Copy)]
pub struct LigeritoOracle(());
