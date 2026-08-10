// Copyright 2026 The Binius Developers

use binius_field::BinaryField;

use super::common::FRIParams;
use crate::merkle_tree::MerkleTreeScheme;

/// Computes the exact byte-size of a FRI proof (including the initial commitment) without running
/// the prover.
///
/// This accounts for:
/// - **Message channel**: the initial codeword commitment and all round commitments (digests
///   observed by Fiat-Shamir).
/// - **Decommitment channel**: the terminal codeword, Merkle layer digests, per-query branch
///   digests, and per-query coset field values.
pub fn proof_size<F, VCS>(params: &FRIParams<F>, vcs: &VCS) -> usize
where
	F: BinaryField,
	VCS: MerkleTreeScheme<F>,
{
	let digest_size = std::mem::size_of::<VCS::Digest>();

	// Serialized byte-size of a single field element.
	let value_size = {
		let mut buf = Vec::new();
		F::default()
			.serialize(&mut buf)
			.expect("default element can be serialized to a resizable buffer");
		buf.len()
	};

	let n_test_queries = params.n_test_queries();

	// One digest per input oracle, one per fold round, one for the terminal codeword.
	let commitment_msg_size = (params.input_oracles().len() + params.n_oracles()) * digest_size;

	// Terminal codeword sent in the clear: 2^(log_terminal_dim + log_inv_rate) field elements.
	let log_terminal_dim = params.n_final_challenges();
	let log_inv_rate = params.rs_code().log_inv_rate();
	let terminate_codeword_size = (1 << (log_terminal_dim + log_inv_rate)) * value_size;

	let mut merkle_sizes = 0;
	let mut coset_values_size = 0;

	// Per query, an oracle sends one coset of `2^arity` elements and a Merkle branch.
	// The layer depth must be chosen for the tree it indexes.
	let mut open = |log_n_cosets: usize, arity: usize| {
		let layer_depth = vcs.optimal_verify_layer(n_test_queries, log_n_cosets);
		merkle_sizes += vcs.proof_size(1 << log_n_cosets, n_test_queries, layer_depth);
		coset_values_size += n_test_queries * (1 << arity) * value_size;
	};

	// Input oracles are opened one after another, each against its own commitment.
	// So a batch of N sends N multi-proofs, not one.
	//
	// An oracle's codeword sits `log_lift` below the reduced dimension.
	let log_dim = params.rs_code().log_dim();
	for spec in params.input_oracles() {
		open(log_dim - spec.log_lift + log_inv_rate, spec.log_batch_size());
	}

	// Then one per fold round.
	// The outer oracle-combine challenges cost nothing: they recombine values already opened.
	let mut log_n_cosets = params.index_bits();
	for &arity in params.fold_arities() {
		log_n_cosets -= arity;
		open(log_n_cosets, arity);
	}

	commitment_msg_size + terminate_codeword_size + merkle_sizes + coset_values_size
}
