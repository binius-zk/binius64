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
	/// The point the index evaluation claims are drawn from, of `max_j n_j` coordinates.
	///
	/// Looker `j`, whose column has `n_j` variables, is claimed at the **last `n_j`** coordinates.
	/// Lookers of equal length therefore all share the whole point; a shorter looker's point is a
	/// suffix of a longer one's, because the batch pads each instance at its low coordinates.
	pub index_eval_point: Vec<F>,
	/// The claimed evaluations of the per-looker index multilinears `I_j`, each at its own suffix
	/// of [`Self::index_eval_point`].
	pub index_eval_claims: Vec<F>,
}
