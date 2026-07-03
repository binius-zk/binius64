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
/// The pushforward claim is packaged as an oracle relation against the commitment received here.
pub struct LogupProof<'a, Oracle, F> {
	/// The `m`-coordinate point shared by the table and pushforward claims.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluation of the table `T` at the point.
	pub table_eval_claim: F,
	/// The `n`-coordinate point of the index claim.
	pub index_eval_point: Vec<F>,
	/// The claimed evaluation of the embedded index column at its point.
	pub index_eval_claim: F,
	/// The oracle relation `<Y, eq_{table_eval_point}> = Y(table_eval_point)` for the pushforward.
	pub pushforward: OracleLinearRelation<'a, Oracle, F>,
}

/// Verify a logUp* reduction whose pushforward is committed as an oracle.
///
/// This wraps [`binius_ip::logup_star::verify`] with the pushforward commitment.
/// It receives the `Y` oracle before the reduction, so the logUp challenge binds the commitment.
/// It then returns the relation that opens `Y` at the reduced point.
///
/// The returned relation asserts `<Y, eq_r'> = Y(r')` at the reduced table point `r'`.
/// Its transparent polynomial is the equality indicator at `r'`.
/// The caller batches this relation with the table and index openings into one channel open.
///
/// # Arguments
///
/// * `table_n_vars` - The number of table variables `m` (`2^m` entries).
/// * `eval_claim` - The claimed evaluation `e` of the looked-up vector.
/// * `eval_point` - The `n`-coordinate evaluation point, whose length defines `n`.
/// * `channel` - The IOP verifier channel carrying the `Y` commitment.
///
/// # Errors
///
/// Returns an error when the pushforward commitment is missing or the reduction identity fails.
pub fn verify<'a, F, C>(
	table_n_vars: usize,
	eval_claim: F,
	eval_point: &[F],
	channel: &mut C,
) -> Result<LogupProof<'a, C::Oracle, F>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IOPVerifierChannel<'a, F, Elem = F>,
{
	// Receive the pushforward Y commitment first, so the reduction's logUp challenge binds it.
	//
	//     Y has 2^m entries, so its message length is table_n_vars.
	//     Y is witness-dependent (it scatters eq_r by the secret index), so it may be masked.
	let oracle = channel.recv_oracle(table_n_vars, true)?;

	// Run the bare reduction over the same channel, viewed as an IP channel.
	let output = reduction::verify::<F, C>(table_n_vars, eval_claim, eval_point, channel)?;

	// The pushforward relation opens Y at the reduced point.
	//
	//     <Y, eq_r'> = Y(r') = pushforward_eval_claim
	//
	// BaseFold reduces this inner product to a challenge point, where the transparent is eq(r', .).
	let point = output.table_eval_point.clone();
	let pushforward = OracleLinearRelation {
		oracle,
		transparent: Box::new(move |challenge: &[F]| eq_ind(&point, challenge)),
		claim: output.pushforward_eval_claim,
	};

	Ok(LogupProof {
		table_eval_point: output.table_eval_point,
		table_eval_claim: output.table_eval_claim,
		index_eval_point: output.index_eval_point,
		index_eval_claim: output.index_eval_claim,
		pushforward,
	})
}
