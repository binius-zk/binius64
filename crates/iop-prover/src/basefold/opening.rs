// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The core BaseFold opening protocol on the prover side.

use std::ops::Deref;

use binius_compute::Allocator;
use binius_field::{BinaryField, PackedField};
use binius_ip_prover::sumcheck::{bivariate_product_prover, common::SumcheckProver};
use binius_math::{FieldVec, ntt::AdditiveNTT};

use crate::{fri::FRIFoldProver, merkle_channel::MerkleIPProverChannel};

/// Proves the single BaseFold sumcheck opening of the Batched ZK BaseFold construction
/// (binius.xyz#128, step 4), running a bivariate-product sumcheck on `<G, T>` alongside a single
/// combined FRI.
///
/// `G` is the multilinear the FRI opens: the γ-folded round oracles, concatenated along the
/// oracle-index variables `Z` with `G(X, Z) = Σ_r M^γ_r↑(X) · eq(r, Z)`. `T` is the multiplicand.
/// Both are buffers of `n_vars + log_n_oracles` variables, `X` low and `Z` high.
///
/// The sumcheck binds high to low, which is the order the FRI folder consumes its challenges
/// as `[γ ++ Z ++ X]`. The `Z` variables are bound in the order the FRI's outer tensor indexes
/// its oracles, so the oracle index `r` sits at chunk `bit_reverse(r)` of `G`.
///
/// ## Arguments
///
/// * `combined` - the multilinear `G`
/// * `multiplicand` - the multilinear `T`
/// * `eval_claim` - the target `Σ_{X,Z} G · T`
/// * `batch_challenge` - the masking challenge `γ`; folds each interleaved `[M_r ‖ ω_r]` codeword
///   in the FRI inner (unbatch) round, which carries no round polynomial
/// * `fri_folder` - the combined FRI fold prover, with `n_rounds == n_inner + combined.log_len()`,
///   where `n_inner` is 1 with a `batch_challenge` and 0 without
/// * `channel` - the Merkle channel carrying all prover interaction: round coefficients,
///   challenges, commitments, and query openings
///
/// The final FRI value equals `G` at the final point.
/// The verifier asserts that it matches the final sumcheck value when it runs the matching opening.
pub fn prove_sumcheck_basefold<A, F, P, NTT, Channel, Data>(
	combined: FieldVec<P, A>,
	multiplicand: FieldVec<P, A>,
	eval_claim: F,
	batch_challenge: Option<F>,
	mut fri_folder: FRIFoldProver<'_, F, P, NTT, Channel::Commitment, Data>,
	channel: &mut Channel,
	alloc: &A,
) where
	A: Allocator,
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
	Data: Deref<Target = [P]>,
{
	let _scope = tracing::debug_span!("Basefold sumcheck ZK (batched)").entered();

	let n_rounds = combined.log_len();
	let n_inner = usize::from(batch_challenge.is_some());
	assert_eq!(n_rounds + n_inner, fri_folder.n_rounds());

	// Inner (unbatch) round: fold every interleaved (M_r ‖ ω_r) ZK codeword at the masking
	// challenge.
	if let Some(gamma) = batch_challenge {
		fri_folder.receive_challenge(gamma);
	}

	let mut sumcheck = bivariate_product_prover(alloc, [combined, multiplicand], eval_claim);
	for _ in 0..n_rounds {
		let round_coeffs = sumcheck
			.execute()
			.pop()
			.expect("the bivariate-product prover proves exactly one claim");

		// Send the round coefficients first: on a commitment round, `execute_fold_round` commits
		// the folded codeword and writes its root, which must land after the coefficients.
		channel.send_many(round_coeffs.truncate().coeffs());
		fri_folder.execute_fold_round(channel);

		let challenge = channel.sample();
		sumcheck.fold(challenge);
		fri_folder.receive_challenge(challenge);
	}

	fri_folder.execute_fold_round(channel);
	fri_folder.finish_proof(channel);
}
