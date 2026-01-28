// Copyright 2026 The Binius Developers

//! Interactive Polynomial (IP) protocol data structures and verification for Binius64.
//!
//! This crate provides the core data structures and verifier-side implementations for
//! interactive polynomial protocols used in Binius64, including sumcheck, prodcheck,
//! and multilinear evaluation claims.
//!
//! # When to use this crate
//!
//! This crate is primarily used internally by `binius_verifier` and `binius_iop`.
//! Direct use is needed when implementing custom verification logic or working with
//! the IP protocol layer directly.
//!
//! # Key types
//!
//! - [`MultilinearEvalClaim`] - A claim that a multilinear polynomial evaluates to a value
//! - [`MultilinearRationalEvalClaim`] - A rational evaluation claim from IOP opening
//! - [`sumcheck`] - Sumcheck protocol verification
//! - [`prodcheck`] - Product check protocol verification
//! - [`channel`] - IP verifier channel traits
//!
//! # Related crates
//!
//! - `binius_ip_prover` - Prover-side IP implementations
//! - `binius_iop` - Higher-level IOP protocols built on IP

#![warn(rustdoc::missing_crate_level_docs)]

use binius_field::Field;

pub mod channel;
pub mod fracaddcheck;
pub mod mlecheck;
pub mod prodcheck;
pub mod sumcheck;

/// A claim that a multilinear polynomial evaluates to a specific value at a point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearEvalClaim<F: Field> {
	/// The evaluation of the multilinear.
	pub eval: F,
	/// The evaluation point.
	pub point: Vec<F>,
}

/// A claim that a multilinear polynomial evaluation satisfies a rational equation.
///
/// This represents the output of an IOP opening protocol where the verification equation is:
/// ```text
/// eval_numerator == eval_denominator * transparent_poly_eval(point)
/// ```
///
/// The caller must compute `transparent_poly_eval(point)` based on the protocol context
/// and verify the equation holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearRationalEvalClaim<F: Field> {
	/// The numerator of the rational claim.
	pub eval_numerator: F,
	/// The denominator: evaluation of the committed polynomial at `point`.
	pub eval_denominator: F,
	/// The evaluation point.
	pub point: Vec<F>,
}
