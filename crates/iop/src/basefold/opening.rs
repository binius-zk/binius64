// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The core BaseFold opening protocol on the verifier side.

use binius_field::BinaryField;
use binius_ip::sumcheck::{RoundCoeffs, RoundProof};
use binius_utils::checked_arithmetics::log2_ceil_usize;

use super::error::Error;
use crate::{
	fri::{FRIFoldVerifier, FRIParams, verify::FRIQueryVerifier},
	merkle_channel::MerkleIPVerifierChannel,
};

/// Verifies the single BaseFold sumcheck opening of the Batched ZK BaseFold construction
/// (binius.xyz#128, step 4).
///
/// This is the verifier counterpart of `binius_iop_prover::basefold::prove_sumcheck_basefold`. The
/// claim is `Σ_{X,Z} G(X, Z) · T(X, Z) = eval_claim`, where `G` is the multilinear the FRI opens:
/// the γ-folded round oracles, concatenated along the oracle-index variables `Z` (`log_n_oracles`
/// of them) with `G(X, Z) = Σ_r M^γ_r↑(X) · eq(r, Z)`. `T` is the verifier-computable multiplicand.
///
/// The sumcheck binds the variables in the order the FRI folder consumes its challenges: the mask
/// fold γ (no round polynomial, and only if some round is masked), then the `Z` variables, then
/// the `X` variables, high to low. Each round polynomial is degree 2 and is received before the
/// FRI round is processed, so that a commitment the FRI sends in the same round lands after it.
///
/// ## Arguments
///
/// * `codeword_commitments` - one per oracle, in the same order as [`FRIParams::input_oracles`], as
///   commitment handles previously received over `channel`.
/// * `eval_claim` - the target `Σ_{X,Z} G · T`.
/// * `n_vars` - the number of variables `X`, the largest round oracle's.
/// * `batch_challenge` - the mask fold challenge `γ`, if any round is masked.
/// * `multiplicand_eval` - evaluates `T(ρ, ζ)`. It receives `ρ` in variable order (low to high) and
///   `ζ` in the order sampled, which is the bit order of the oracle index.
/// * `channel` - the Merkle channel carrying all prover interaction: round coefficients,
///   challenges, commitments, and query openings.
///
/// The final consistency check `G(ρ, ζ) · T(ρ, ζ) = final sum` is asserted internally, so `Ok`
/// means the opening verified.
pub fn verify_sumcheck_basefold<F, Channel>(
	fri_params: &FRIParams<F>,
	codeword_commitments: &[Channel::Commitment],
	eval_claim: Channel::Elem,
	n_vars: usize,
	batch_challenge: Option<Channel::Elem>,
	multiplicand_eval: impl FnOnce(&[Channel::Elem], &[Channel::Elem]) -> Channel::Elem,
	channel: &mut Channel,
) -> Result<(), Error>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F>,
	Channel::Elem: From<F>,
{
	// The round polynomial of a product of two multilinears is degree 2.
	const DEGREE: usize = 2;

	let log_n_oracles = log2_ceil_usize(codeword_commitments.len());
	let n_inner = usize::from(batch_challenge.is_some());
	let mut fri_challenges = Vec::with_capacity(n_inner + log_n_oracles + n_vars);
	let mut fri_fold_verifier = FRIFoldVerifier::new(fri_params);

	// Inner (unbatch) round: fold every interleaved (M_r ‖ ω_r) ZK codeword at the masking
	// challenge. It carries no round polynomial.
	if let Some(gamma) = batch_challenge {
		fri_fold_verifier.process_round(channel)?;
		fri_challenges.push(gamma);
	}

	// The sumcheck rounds: `log_n_oracles` rounds over `Z`, then `n_vars` over `X`.
	let mut sum = eval_claim;
	for _ in 0..log_n_oracles + n_vars {
		let round_proof = RoundProof(RoundCoeffs(channel.recv_many(DEGREE)?));
		fri_fold_verifier.process_round(channel)?;

		let challenge = channel.sample();
		sum = round_proof.recover(sum).evaluate(&challenge);
		fri_challenges.push(challenge);
	}

	fri_fold_verifier.process_round(channel)?;
	let round_commitments = fri_fold_verifier.finalize();

	let fri_verifier = FRIQueryVerifier::new_batch(
		fri_params,
		codeword_commitments,
		&round_commitments,
		&fri_challenges,
	);
	let final_fri_value = fri_verifier.verify(channel)?;

	// `Z` challenges come first, then `X` high to low; `ρ` is indexed low to high.
	let (zeta, rho) = fri_challenges[n_inner..].split_at(log_n_oracles);
	let rho = rho.iter().rev().cloned().collect::<Vec<_>>();
	let multiplicand = multiplicand_eval(&rho, zeta);

	channel.assert_zero(sum - final_fri_value * multiplicand)?;

	Ok(())
}
