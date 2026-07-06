// Copyright 2026 The Binius Developers

//! logUp* verification with the pushforward committed as an oracle.
//!
//! The bare reduction returns a claimed evaluation of the pushforward `Y = I_* eq_r`.
//! It never binds `Y` to a commitment, so that claim cannot be checked on its own.
//! This layer receives the `Y` oracle over the IOP channel and returns the relation that opens it.
//!
//! The receive precedes the reduction, so the logUp challenge binds the received `Y`.
//! The table `T` and index `I` stay the caller's oracles, so their claims are returned unchanged.
//! `Y` is the only oracle this protocol introduces, per [Soukhanov25, Section 3].
//!
//! [Soukhanov25]: <https://eprint.iacr.org/2025/946>

use binius_field::{BinaryField1b, ExtensionField, Field};
use binius_ip::logup_star as reduction;
use binius_math::multilinear::eq::eq_ind;

use crate::channel::{Error as ChannelError, IOPVerifierChannel, OracleLinearRelation};

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
/// The pushforward claim is opened against the commitment received here, through the channel.
pub struct LogupProof<Elem> {
	/// The `m`-coordinate point shared by the table and pushforward claims.
	pub table_eval_point: Vec<Elem>,
	/// The claimed evaluation of the table `T` at the point.
	pub table_eval_claim: Elem,
	/// The `n`-coordinate point shared by the index claims.
	pub index_eval_point: Vec<Elem>,
	/// The claimed evaluations of the per-looker embedded index columns at the point.
	pub index_eval_claims: Vec<Elem>,
}

/// Verify a logUp* reduction whose pushforward is committed as an oracle.
///
/// This wraps [`binius_ip::logup_star::verify_reduction`] with the pushforward commitment: the
/// looker batching challenge is sampled first (the prover builds the combined pushforward from
/// it), then the `Y` oracle is received before the reduction, so the logUp challenge binds the
/// commitment. The relation `<Y, eq_r'> = Y(r')` at the reduced table point `r'` is opened
/// through the channel, which may defer the actual opening to `finish()`.
///
/// # Arguments
///
/// * `table_n_vars` - The number of table variables `m` (`2^m` entries).
/// * `lookers` - The looker claims; every evaluation point must have the same length `n`.
/// * `channel` - The IOP verifier channel carrying the `Y` commitment.
///
/// # Errors
///
/// Returns an error when the pushforward commitment is missing or the reduction identity fails.
pub fn verify<F, C>(
	table_n_vars: usize,
	lookers: &[reduction::LookerClaim<'_, C::Elem>],
	channel: &mut C,
) -> Result<LogupProof<C::Elem>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IOPVerifierChannel<F>,
	C::Elem: From<F>,
{
	// Sample the looker batching challenge before the commitment: the prover needs gamma to build
	// the combined pushforward it commits.
	let gamma = channel.sample();

	// Receive the pushforward Y commitment next, so the reduction's logUp challenge binds it.
	//
	//     Y has 2^m entries, so its message length is table_n_vars.
	//     Y is witness-dependent (it scatters the numerators by the secret indexes), so it may be
	//     masked.
	let oracle = channel.recv_oracle(table_n_vars, true)?;

	// Run the bare reduction over the same channel, viewed as an IP channel.
	let output = reduction::verify_reduction::<F, C>(gamma, table_n_vars, lookers, channel)?;

	// Open the pushforward relation through the channel; a deferring channel (e.g. BaseFold)
	// batches it with every other queued relation in `finish()`.
	//
	//     <Y, eq_r'> = Y(r') = pushforward_eval_claim
	//
	// BaseFold reduces this inner product to a challenge point, where the transparent is eq(r', .).
	let point = output.table_eval_point.clone();
	channel.verify_oracle_relations([OracleLinearRelation {
		oracle,
		transparent: Box::new(move |challenge: &[C::Elem]| eq_ind(&point, challenge)),
		claim: output.pushforward_eval_claim,
	}])?;

	Ok(LogupProof {
		table_eval_point: output.table_eval_point,
		table_eval_claim: output.table_eval_claim,
		index_eval_point: output.index_eval_point,
		index_eval_claims: output.index_eval_claims,
	})
}
