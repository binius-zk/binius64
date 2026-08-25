// Copyright 2026 The Binius Developers

//! Ligerito implementation of the IOP prover channel.
//!
//! One file per concern.
//! The oracle handle is here, and the pieces the channel assembles sit beside it.

mod combined_message;
mod committed_oracle;
mod mask;
mod prover;
mod relation;

pub use prover::LigeritoProverChannel;

/// A handle to one of the oracles a Ligerito channel opens.
///
/// The inner field is private, so the only way to hold one is to have sent the commitment.
/// It names the position the commitment was sent in, which is the position its relations queue at.
#[derive(Debug, Clone, Copy)]
pub struct LigeritoOracle {
	/// The order this oracle's commitment was sent in.
	index: usize,
}
