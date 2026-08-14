// Copyright 2026 The Binius Developers

//! logUp* verification with the pushforwards committed as oracles.
//!
//! The bare reduction returns a claimed evaluation of each table's pushforward `Y = I_* eq_r`.
//! It never binds them to commitments, so those claims cannot be checked on their own.
//! This layer receives one `Y` oracle per table over the IOP channel and returns the relations that
//! open them.
//!
//! The receives precede the reduction, so its logUp challenges bind the received oracles.
//! The tables `T` and indexes `I` stay the caller's oracles, so their claims are returned
//! unchanged. The pushforwards are the only oracles this protocol introduces, per
//! [Soukhanov25, Section 3].
//!
//! [`verify_transparent`] is the variant for tables the verifier evaluates itself.
//! It opens each `Y` against both `eq_z` and the table, and so needs no pushforward sumcheck.
//!
//! [Soukhanov25]: <https://eprint.iacr.org/2025/946>

use binius_field::{BinaryField1b, ExtensionField, Field};
use binius_ip::logup_star as reduction;
use binius_math::multilinear::eq::eq_ind;
use itertools::izip;

use crate::channel::{Error as ChannelError, IOPVerifierChannel, TransparentEvalFn};

/// An error raised while verifying a committed logUp* reduction.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// The underlying logUp* reduction failed.
	#[error("logUp* reduction error: {0}")]
	Reduction(#[from] reduction::Error),
	/// Receiving the pushforward oracle commitment failed.
	#[error("IOP channel error: {0}")]
	Channel(#[from] ChannelError),
}

/// The reduced claims of a committed logUp* verification.
///
/// The table and index claims are left for the caller to open against its own commitments.
/// The pushforward claims are opened against the commitments received here, through the channel.
pub struct LogupProof<Elem> {
	/// The point the table claims are drawn from, spanning the widest table.
	///
	/// A table over `m` variables is claimed at its **first `m`** coordinates.
	pub table_eval_point: Vec<Elem>,
	/// The point the index claims are drawn from, spanning the deepest looker.
	///
	/// A looker over `n` variables is claimed at its **last `n`** coordinates.
	pub index_eval_point: Vec<Elem>,
	/// One entry per table, in the order the tables were given.
	pub tables: Vec<reduction::LogupTableOutput<Elem>>,
}

/// Verify a logUp* reduction whose pushforwards are committed as oracles.
///
/// This wraps [`binius_ip::logup_star::verify_reduction`] with the pushforward commitments. The
/// looker batching challenge is sampled first — the prover needs it to build the pushforwards —
/// then one `Y` oracle is received per table, in table order, before the reduction, so the logUp
/// challenges bind every commitment. The relations `<Y, eq_r> = Y(r)` at each table's reduced point
/// are opened through the channel, which may defer the actual openings to `finish()`.
///
/// One oracle per table is the simple arrangement; the pushforwards could instead be concatenated
/// into a single oracle, which is left for later.
///
/// # Arguments
///
/// * `tables` - One [`binius_ip::logup_star::TableLookup`] per table, by value. The lookers'
///   evaluation points may differ in length, from each other and from the table.
/// * `channel` - The IOP verifier channel carrying the `Y` commitments.
///
/// # Errors
///
/// Returns an error when a pushforward commitment is missing or the reduction identity fails.
pub fn verify<'a, F, C>(
	tables: impl IntoIterator<Item = reduction::TableLookup<'a, C::Elem>>,
	channel: &mut C,
) -> Result<LogupProof<C::Elem>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IOPVerifierChannel<F>,
	C::Elem: From<F> + 'a,
{
	// The tables are walked twice — once to receive their commitments, once to open them — so the
	// iterator is materialized once here.
	let tables = tables.into_iter().collect::<Vec<_>>();
	// Only the variable counts are needed after the reduction consumes the tables.
	let table_n_vars = tables.iter().map(|table| table.n_vars).collect::<Vec<_>>();

	// Sample the looker batching challenge before the commitments: the prover needs gamma to build
	// the pushforwards it commits.
	let gamma = channel.sample();

	// Receive the pushforward commitments next, so the reduction's logUp challenges bind them.
	//
	//     Y for a table over m variables has 2^m entries, so its message length is m.
	//     Y is witness-dependent (it scatters the numerators by the secret indexes), so it may be
	//     masked.
	let oracles = table_n_vars
		.iter()
		.map(|&n_vars| channel.recv_oracle(n_vars, true))
		.collect::<Result<Vec<_>, _>>()?;

	// Run the bare reduction over the same channel, viewed as an IP channel.
	let output = reduction::verify_reduction::<F, C>(&gamma, tables, channel)?;

	// Open each pushforward relation through the channel; a deferring channel (e.g. BaseFold)
	// batches them with every other queued relation in `finish()`.
	//
	//     <Y, eq_r> = Y(r) = that table's pushforward claim
	//
	// BaseFold reduces each inner product to a challenge point, where the transparent is eq(r, .).
	// A table's own point is the first m coordinates of the shared reduced point.
	for (oracle, &n_vars, table_output) in izip!(oracles, &table_n_vars, &output.tables) {
		let point = output.table_eval_point[..n_vars].to_vec();
		channel.verify_oracle_relation(
			oracle,
			Box::new(move |challenge: &[C::Elem]| eq_ind(&point, challenge)),
			table_output.pushforward_claim.clone(),
		)?;
	}

	Ok(LogupProof {
		table_eval_point: output.table_eval_point,
		index_eval_point: output.index_eval_point,
		tables: output.tables,
	})
}

/// One transparent table, the lookers that read it, and the closure evaluating its MLE.
pub struct TransparentTableLookup<'a, Elem> {
	/// The table's variable count and the claims of the lookers reading it.
	pub lookup: reduction::TableLookup<'a, Elem>,
	/// Evaluates the table's multilinear extension at a point of `lookup.n_vars` coordinates.
	pub table_eval: TransparentEvalFn<Elem>,
}

/// The reduced claims of a committed logUp* verification over transparent tables.
///
/// Nothing is left on the tables or the pushforwards: both of a table's claims are opened here,
/// against the one `Y` commitment. Only the index claims are the caller's to verify.
pub struct LogupTransparentProof<Elem> {
	/// The point the index claims are drawn from, spanning the deepest looker.
	///
	/// A looker over `n` variables is claimed at its **last `n`** coordinates.
	pub index_eval_point: Vec<Elem>,
	/// One entry per table, in the order the tables were given, holding that table's lookers'
	/// index claims in its own looker order.
	pub index_eval_claims: Vec<Vec<Elem>>,
}

/// Verify a logUp* reduction over transparent tables, with the pushforwards committed as oracles.
///
/// This wraps [`binius_ip::logup_star::verify_reduction_transparent`] the way [`verify`] wraps the
/// committed-table reduction. The reduction leaves two claims on each pushforward instead of one,
/// and both are opened here through the channel:
///
/// ```text
///     <Y, eq_z> = Y(z)      the fractional-addition leaf claim
///     <Y, T>    = e         the product claim, against the caller's transparent table
/// ```
///
/// The two are queued in that order, so the prover must queue them the same way.
/// A channel that batches an oracle's relations folds them into one opening.
///
/// # Arguments
///
/// * `tables` - One [`TransparentTableLookup`] per table, by value.
/// * `channel` - The IOP verifier channel carrying the `Y` commitments.
///
/// # Errors
///
/// Returns an error when a pushforward commitment is missing or the reduction identity fails.
pub fn verify_transparent<'a, F, C>(
	tables: impl IntoIterator<Item = TransparentTableLookup<'a, C::Elem>>,
	channel: &mut C,
) -> Result<LogupTransparentProof<C::Elem>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IOPVerifierChannel<F>,
	C::Elem: From<F> + 'a,
{
	// The reduction consumes the lookups, so the transparent closures are split off up front.
	let (lookups, table_evals): (Vec<_>, Vec<_>) = tables
		.into_iter()
		.map(|table| (table.lookup, table.table_eval))
		.unzip();
	let table_n_vars = lookups.iter().map(|table| table.n_vars).collect::<Vec<_>>();

	// Sample gamma, then receive the commitments, exactly as [`verify`] does: the prover needs
	// gamma to build the pushforwards, and the reduction's logUp challenges must bind them.
	let gamma = channel.sample();
	let oracles = table_n_vars
		.iter()
		.map(|&n_vars| channel.recv_oracle(n_vars, true))
		.collect::<Result<Vec<_>, _>>()?;

	let output = reduction::verify_reduction_transparent::<F, C>(&gamma, lookups, channel)?;

	// Open both of a table's claims against its one pushforward commitment. The table side never
	// reaches the caller: the product relation weighs Y directly against the transparent T.
	for (oracle, table_eval, table) in izip!(oracles, table_evals, &output.tables) {
		let point = table.pushforward_eval_point.clone();
		channel.verify_oracle_relation(
			oracle.clone(),
			Box::new(move |challenge: &[C::Elem]| eq_ind(&point, challenge)),
			table.pushforward_eval_claim.clone(),
		)?;
		channel.verify_oracle_relation(oracle, table_eval, table.product_claim.clone())?;
	}

	Ok(LogupTransparentProof {
		index_eval_point: output.index_eval_point,
		index_eval_claims: output
			.tables
			.into_iter()
			.map(|table| table.index_eval_claims)
			.collect(),
	})
}
