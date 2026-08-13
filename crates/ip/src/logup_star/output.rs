// Copyright 2026 The Binius Developers

//! The reduced output claims of a logUp* verification.

/// The reduced output claims of a logUp* verification.
///
/// Each claim must be verified separately by the caller.
/// Verifying them is out of scope here.
///
/// The two sides each carry **one** point, shared by every table, and read it from opposite ends.
/// That is not a choice: the pushforward reduction pads a table at its high variables, while the
/// fractional-addition batch pads a looker at its low ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogupOutput<F> {
	/// The point the table and pushforward evaluation claims are drawn from, of `max m`
	/// coordinates.
	///
	/// A table over `m` variables is claimed at the **first `m`** coordinates. Tables of equal
	/// size therefore all share the whole point; a smaller table's point is a prefix of a larger
	/// one's.
	pub table_eval_point: Vec<F>,
	/// The point the index evaluation claims are drawn from, of `max n` coordinates.
	///
	/// A looker whose column has `n` variables is claimed at the **last `n`** coordinates. Lookers
	/// of equal length therefore all share the whole point; a shorter looker's point is a suffix
	/// of a longer one's, because the batch pads each instance at its low coordinates.
	pub index_eval_point: Vec<F>,
	/// One entry per table, in the order the tables were given.
	pub tables: Vec<LogupTableOutput<F>>,
}

/// The reduced claims belonging to one table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogupTableOutput<F> {
	/// The claimed evaluation of the table multilinear `T` at its prefix of
	/// [`LogupOutput::table_eval_point`].
	pub eval_claim: F,
	/// The claimed evaluation of the pushforward multilinear `Y` at the same prefix.
	pub pushforward_claim: F,
	/// The claimed evaluations of this table's lookers' index multilinears `I`, in its own looker
	/// order, each at that looker's own suffix of [`LogupOutput::index_eval_point`].
	pub index_eval_claims: Vec<F>,
}

/// The open claims of a logUp* verification that leaves the table side unclosed.
///
/// This is what the reduction holds before the pushforward sumcheck runs.
/// Each table keeps two claims on its pushforward and none on the table itself:
///
/// ```text
///     <Y_t, eq_{z_t}> = Y_t(z_t)      the fractional-addition leaf claim
///     <Y_t, T_t>      = e_t           the product claim
/// ```
///
/// See [`super::verify_reduction_transparent`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogupTransparentOutput<F> {
	/// The point the index evaluation claims are drawn from, of `max n` coordinates.
	///
	/// A looker whose column has `n` variables is claimed at the **last `n`** coordinates, exactly
	/// as in [`LogupOutput::index_eval_point`].
	pub index_eval_point: Vec<F>,
	/// One entry per table, in the order the tables were given.
	pub tables: Vec<LogupTransparentTableOutput<F>>,
}

/// The open claims belonging to one table whose table side is left unclosed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogupTransparentTableOutput<F> {
	/// The fractional-addition leaf point `z_t`, of this table's own `m_t` coordinates.
	pub pushforward_eval_point: Vec<F>,
	/// The leaf claim `Y_t(z_t)` on this table's pushforward.
	pub pushforward_eval_claim: F,
	/// The product claim `e_t = <T_t, Y_t>`, an inner product over this table's `m_t`-variable
	/// cube.
	///
	/// It is the gamma-combination of the claims of the lookers that read this table.
	/// The reduction never reads a table, so this claim arrives unchecked: opening it is what
	/// proves the lookups.
	pub product_claim: F,
	/// The claimed evaluations of this table's lookers' index multilinears `I`, in its own looker
	/// order, each at that looker's own suffix of [`LogupTransparentOutput::index_eval_point`].
	pub index_eval_claims: Vec<F>,
}
