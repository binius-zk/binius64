// Copyright 2026 The Binius Developers

//! Spartan wrapper prover for ZK-wrapped IOP proving.
//!
//! This crate provides [`ZKWrappedProverChannel`], the prover-side counterpart to
//! [`ZKWrappedVerifierChannel`]. It wraps a [`BaseFoldZKProverChannel`] and records all channel
//! operations. After the inner proof is run through the channel, [`finish`] replays the
//! interaction through a [`ReplayChannel`] to fill the outer witness, then runs the outer IOP
//! prover.
//!
//! [`ZKWrappedVerifierChannel`]: binius_spartan_verifier::wrapper::ZKWrappedVerifierChannel
//! [`BaseFoldZKProverChannel`]: binius_iop_prover::basefold_zk_channel::BaseFoldZKProverChannel
//! [`ReplayChannel`]: binius_spartan_verifier::wrapper::ReplayChannel
//! [`finish`]: ZKWrappedProverChannel::finish

mod zk_wrapped_prover_channel;

pub use zk_wrapped_prover_channel::ZKWrappedProverChannel;
