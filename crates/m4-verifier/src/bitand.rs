// Copyright 2025 Irreducible Inc.

//! Verifier for the batched BitAnd reduction of the data-parallel M4 proof system.

use binius_field::AESTowerField8b as B8;
use binius_ip::channel::IPVerifierChannel;
use binius_math::BinarySubspace;
use binius_verifier::{
	Error,
	config::{B128, LOG_WORD_SIZE_BITS, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES},
	protocols::bitand::{AndCheckOutput, verify_with_channel},
};

/// Verifies the batched BitAnd check: `A & B == C` on every row of every instance.
///
/// This mirrors the prover side, the univariate-skip zerocheck of:
///
/// ```text
/// A(Z, X) * B(Z, X) - C(Z, X) == 0   for all rows (Z, X)
/// ```
///
/// `Z` is the bit index within a 64-bit word.
/// `X` is the row index.
///
/// The stacked batch is one flat hypercube.
/// So verification is the single-instance check at a larger row count.
/// The block-diagonal batch structure is exploited later, in the lincheck.
///
/// # Arguments
///
/// - `log_total_constraints`: base-2 logarithm of the total row count `K * n_and`.
/// - `channel`: the verifier channel that reads messages and redraws Fiat-Shamir challenges.
///
/// The total row count is the instance count times the per-instance constraint count.
/// Both are powers of two, so its logarithm is `log_instances + log(n_and)`.
///
/// # Returns
///
/// The reduced claim, holding:
/// - The claimed `A`, `B`, `C` evaluations.
/// - The univariate (bit-index) challenge.
/// - The multilinear evaluation point reached by the sumcheck.
///
/// # Errors
///
/// Returns an error if any sumcheck round message or the final consistency check fails.
pub fn verify_bitand_reduction<C>(
	log_total_constraints: usize,
	channel: &mut C,
) -> Result<AndCheckOutput<B128>, Error>
where
	C: IPVerifierChannel<B128, Elem = B128>,
{
	// The univariate-skip domain: the full B8 subspace, lifted to B128.
	// It is then cut to one dimension above the 64-bit word.
	// The prover sends round-message evaluations over exactly this domain.
	let eval_domain = BinarySubspace::<B8>::default()
		.isomorphic::<B128>()
		.reduce_dim(LOG_WORD_SIZE_BITS + 1);

	// The first few zerocheck coordinates are pinned to fixed small-field elements.
	// The prover pins the same prefix, so both sides agree on this split.
	//
	//     coordinates = [ pinned small-field | sampled large-field ]
	//                    \____ at most 3 ___/
	let small_field_zerocheck_challenges = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES
		.into_iter()
		.take(log_total_constraints)
		.map(B128::from)
		.collect::<Vec<_>>();

	// The remaining coordinates are drawn from the large field, in the same order as the prover.
	let big_field_zerocheck_challenges =
		channel.sample_many(log_total_constraints - small_field_zerocheck_challenges.len());

	let zerocheck_challenges = small_field_zerocheck_challenges
		.into_iter()
		.chain(big_field_zerocheck_challenges)
		.collect::<Vec<_>>();

	verify_with_channel(&zerocheck_challenges, channel, &eval_domain)
}
