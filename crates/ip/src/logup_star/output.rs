// Copyright 2026 The Binius Developers

//! The reduced output claims of a logUp* verification.

/// The reduced output claims of a logUp* verification.
///
/// Each claim must be verified separately by the caller.
/// Verifying them is out of scope here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogupOutput<F> {
	/// The `m`-coordinate point shared by the table and pushforward evaluation claims.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluation of the table multilinear `T` at the table point.
	pub table_eval_claim: F,
	/// The claimed evaluation of the pushforward multilinear `Y` at the table point.
	pub pushforward_eval_claim: F,
	/// The `n`-coordinate point shared by the index evaluation claims.
	pub index_eval_point: Vec<F>,
	/// The claimed evaluations of the per-looker index multilinears `I_j` at the index point.
	pub index_eval_claims: Vec<F>,
}
