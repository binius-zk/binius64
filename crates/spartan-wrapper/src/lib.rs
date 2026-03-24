// Copyright 2026 The Binius Developers

//! Spartan wrapper for symbolically executing IOP verifiers to build constraint systems.
//!
//! This crate provides [`IronSpartanBuilderChannel`], an implementation of [`IPVerifierChannel`]
//! that symbolically executes a verifier and records the computation as an IronSpartan constraint
//! system via [`ConstraintBuilder`].
//!
//! [`IPVerifierChannel`]: binius_ip::channel::IPVerifierChannel
//! [`ConstraintBuilder`]: binius_spartan_frontend::circuit_builder::ConstraintBuilder

mod build_elem;
mod channel;

pub use build_elem::{BuildElem, BuildWire};
pub use channel::IronSpartanBuilderChannel;
