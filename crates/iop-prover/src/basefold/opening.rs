// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The core BaseFold opening protocol on the prover side.

use std::ops::Deref;

use binius_compute::Allocator;
use binius_field::{BinaryField, PackedField};
use binius_ip_prover::sumcheck::{bivariate_product_prover, common::SumcheckProver};
use binius_math::{bit_reverse::bit_reverse_packed, ntt::AdditiveNTT};

use crate::{
	basefold::index_rounds::{IndexNode, bind_oracle_index_variables},
	fri::FRIFoldProver,
	merkle_channel::MerkleIPProverChannel,
};

/// Proves the single BaseFold sumcheck opening of the Batched ZK BaseFold construction
/// (binius.xyz#128, step 4), running a bivariate-product sumcheck on `<G, T>` alongside a single
/// combined FRI.
///
/// `G` is the multilinear the FRI opens: the γ-folded round oracles, concatenated along the
/// oracle-index variables `Z` with `G(X, Z) = Σ_r M^γ_r↑(X) · eq(r, Z)`. `T` is the multiplicand.
/// They are never materialized over `Z`: the first `log_n_oracles` rounds bind `Z` directly on the
/// per-round buffers (see [`index_rounds`](super::index_rounds)), leaving one merged `G` and `T` of
/// `n_vars` variables for the remaining rounds.
///
/// The sumcheck binds low to high, which is the order the FRI folder consumes its challenges as
/// `[γ ++ Z ++ X]`: the encoder takes entry `i` as the coefficient of `X_i`, so the first fold
/// binds the lowest variable. The `Z` variables are the lowest, in the natural order of the oracle
/// index, with a round's own variables above them.
///
/// The index rounds and the bivariate-product prover below bind high to low. Binding the
/// bit-reversed buffers high to low gives the same round polynomials as binding the originals low
/// to high, so each node's message and transparent are bit-reversed over their own variables once
/// on entry, and the nodes are in natural order.
///
/// ## Arguments
///
/// * `nodes` - per round oracle, in oracle-index order, its γ-folded message `M^γ_r`, transparent
///   `T_r`, both in natural order, and the log of the message's repeat period; padded with `None`
///   to a power of two
/// * `eval_claim` - the target `Σ_{X,Z} G · T`
/// * `batch_challenge` - the masking challenge `γ`; folds each interleaved `[M_r ‖ ω_r]` codeword
///   in the FRI inner (unbatch) round, which carries no round polynomial
/// * `fri_folder` - the combined FRI fold prover, with `n_rounds == n_inner + log_n_oracles +
///   n_vars`, where `n_inner` is 1 with a `batch_challenge` and 0 without
/// * `channel` - the Merkle channel carrying all prover interaction: round coefficients,
///   challenges, commitments, and query openings
///
/// The final FRI value equals `G` at the final point.
/// The verifier asserts that it matches the final sumcheck value when it runs the matching opening.
pub fn prove_sumcheck_basefold<A, F, P, NTT, Channel, Data>(
	mut nodes: Vec<Option<IndexNode<P, A>>>,
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

	let n_vars = nodes
		.iter()
		.flatten()
		.map(|node| node.message.log_len())
		.max()
		.expect("at least one oracle");
	let log_n_oracles = nodes.len().trailing_zeros() as usize;
	let n_inner = usize::from(batch_challenge.is_some());
	assert_eq!(n_vars + log_n_oracles + n_inner, fri_folder.n_rounds());

	// Bit-reverse each node, so that binding high to low is binding the natural order low to high.
	for node in nodes.iter_mut().flatten() {
		bit_reverse_packed(node.message.as_mut_view());
		bit_reverse_packed(node.transparent.as_mut_view());
	}

	// Inner (unbatch) round: fold every interleaved (M_r ‖ ω_r) ZK codeword at the masking
	// challenge.
	if let Some(gamma) = batch_challenge {
		fri_folder.receive_challenge(gamma);
	}

	// The oracle-index rounds, on the per-round buffers.
	let (merged, claim) = bind_oracle_index_variables(nodes, eval_claim, |round_coeffs| {
		// Send the round coefficients first: on a commitment round, `execute_fold_round` commits
		// the folded codeword and writes its root, which must land after the coefficients.
		channel.send_many(round_coeffs.clone().truncate().coeffs());
		fri_folder.execute_fold_round(channel);

		let challenge = channel.sample();
		fri_folder.receive_challenge(challenge);
		challenge
	});
	let IndexNode {
		message,
		transparent,
		..
	} = merged;

	let mut sumcheck = bivariate_product_prover(alloc, [message, transparent], claim);
	for _ in 0..n_vars {
		let round_coeffs = sumcheck
			.execute()
			.pop()
			.expect("the bivariate-product prover proves exactly one claim");

		channel.send_many(round_coeffs.truncate().coeffs());
		fri_folder.execute_fold_round(channel);

		let challenge = channel.sample();
		sumcheck.fold(challenge);
		fri_folder.receive_challenge(challenge);
	}

	fri_folder.execute_fold_round(channel);
	fri_folder.finish_proof(channel);
}
