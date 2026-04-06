// Copyright 2026 The Binius Developers

//! Spartan wrapper prover for ZK-wrapped IOP proving.
//!
//! This crate provides [`ZKWrappedProverChannel`], the prover-side counterpart to
//! [`ZKWrappedVerifierChannel`]. It wraps a [`BaseFoldZKProverChannel`] and records an
//! [`InteractionLog`] of all channel operations. The [`finish`] method replays the interaction
//! through a [`ReplayChannel`] to fill the outer witness, then runs the outer IOP prover.
//!
//! [`ZKWrappedVerifierChannel`]: binius_spartan_wrapper::ZKWrappedVerifierChannel
//! [`BaseFoldZKProverChannel`]: binius_iop_prover::basefold_zk_channel::BaseFoldZKProverChannel
//! [`InteractionLog`]: binius_spartan_wrapper::interaction::InteractionLog
//! [`ReplayChannel`]: binius_spartan_wrapper::ReplayChannel
//! [`finish`]: ZKWrappedProverChannel::finish

mod zk_wrapped_prover_channel;

pub use zk_wrapped_prover_channel::ZKWrappedProverChannel;
