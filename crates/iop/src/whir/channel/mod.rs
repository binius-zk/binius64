// Copyright 2026 The Binius Developers

//! WHIR implementation of the IOP verifier channel.
//!
//! One file per concern.
//! The oracle handle is here, and the pieces the channel assembles sit beside it.

mod relation;
mod verifier;

pub use verifier::WHIRVerifierChannel;

/// A handle to one of the oracles a WHIR channel opens.
///
/// The inner field is private, so the only way to hold one is to have received the commitment.
/// It names the position the commitment arrived in, which is the position its relations queue at.
#[derive(Debug, Clone, Copy)]
pub struct WHIROracle {
	/// The order this oracle's commitment was received in.
	index: usize,
}
