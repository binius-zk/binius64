// Copyright 2026 The Binius Developers

use binius_field::BinaryField;

use super::common::{FRIParams, calculate_n_test_queries};
use crate::{channel::OracleSpec, merkle_tree::MerkleTreeScheme};

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

/// One candidate Reed-Solomon rate, together with what a proof at that rate costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateEstimate {
	/// The binary logarithm of the inverse Reed-Solomon code rate.
	pub log_inv_rate: usize,
	/// The number of test queries needed to hit the target security level at this rate.
	pub n_test_queries: usize,
	/// The exact byte-size of the proof at this rate, as reported by [`proof_size`].
	pub proof_size: usize,
}

/// Estimates the proof size at each candidate Reed-Solomon rate.
///
/// A larger `log_inv_rate` needs fewer test queries, which shrinks the proof.
/// It also lengthens every codeword, which costs the prover proportionally more encoding work.
/// This prices the bytes only, so the encoding cost is left to benchmarks.
///
/// Each candidate is priced independently.
/// Its test-query count comes from [`calculate_n_test_queries`].
/// Its fold arities are re-optimized for that rate by [`FRIParams::optimal_for_batch`].
///
/// ## Arguments
///
/// * `merkle_scheme` - the Merkle tree scheme used for commitments.
/// * `oracles` - the oracles to batch, as passed to [`FRIParams::optimal_for_batch`].
/// * `security_bits` - the target soundness threshold of the query phase, in bits.
/// * `log_inv_rates` - the candidate rates to price.
///
/// ## Returns
///
/// One [`RateEstimate`] per candidate rate, in the order the candidates were yielded.
///
/// ## Preconditions
///
/// * `oracles` is non-empty.
pub fn estimate_by_rate<F, MerkleScheme>(
	merkle_scheme: &MerkleScheme,
	oracles: &[OracleSpec],
	security_bits: usize,
	log_inv_rates: impl IntoIterator<Item = usize>,
) -> Vec<RateEstimate>
where
	F: BinaryField,
	MerkleScheme: MerkleTreeScheme<F>,
{
	log_inv_rates
		.into_iter()
		.map(|log_inv_rate| {
			let n_test_queries = calculate_n_test_queries(security_bits, log_inv_rate);
			// `optimal_for_batch`'s own estimate omits the commitment digests.
			// Price the parameters it returns with `proof_size`, which counts the whole proof.
			let (params, _) = FRIParams::<F>::optimal_for_batch(
				merkle_scheme,
				oracles,
				log_inv_rate,
				n_test_queries,
			);
			RateEstimate {
				log_inv_rate,
				n_test_queries,
				proof_size: proof_size(&params, merkle_scheme),
			}
		})
		.collect()
}

/// Returns the candidate rate from [`estimate_by_rate`] with the smallest proof size.
///
/// Ties go to the first candidate yielded.
/// With candidates in increasing `log_inv_rate` order, that is the cheapest of them to encode.
///
/// ## Arguments
///
/// * `merkle_scheme` - the Merkle tree scheme used for commitments.
/// * `oracles` - the oracles to batch, as passed to [`FRIParams::optimal_for_batch`].
/// * `security_bits` - the target soundness threshold of the query phase, in bits.
/// * `log_inv_rates` - the candidate rates to price.
///
/// ## Panics
///
/// Panics if `log_inv_rates` yields no candidates.
///
/// ## Preconditions
///
/// * `oracles` is non-empty.
pub fn best_rate<F, MerkleScheme>(
	merkle_scheme: &MerkleScheme,
	oracles: &[OracleSpec],
	security_bits: usize,
	log_inv_rates: impl IntoIterator<Item = usize>,
) -> RateEstimate
where
	F: BinaryField,
	MerkleScheme: MerkleTreeScheme<F>,
{
	estimate_by_rate::<F, _>(merkle_scheme, oracles, security_bits, log_inv_rates)
		.into_iter()
		.min_by_key(|estimate| estimate.proof_size)
		.expect("log_inv_rates yields at least one candidate rate")
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField128bGhash as B128;
	use binius_hash::StdHashSuite;

	use super::*;
	use crate::merkle_tree::BinaryMerkleTreeScheme;

	type TestMerkleScheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

	fn test_merkle_scheme() -> TestMerkleScheme {
		BinaryMerkleTreeScheme::new()
	}

	const SECURITY_BITS: usize = 100;

	// One non-ZK oracle over rates 1/2 down to 1/64.
	// Each row is `(log_msg_len, log_inv_rate, n_test_queries, proof_size)`.
	#[test]
	fn pinned_proof_size_by_rate() {
		let merkle_scheme = test_merkle_scheme();

		// Sweep in candidate order, so the observed table is directly comparable to the pinned one.
		let mut observed = Vec::new();
		for log_msg_len in [17usize, 20, 24] {
			let oracles = [OracleSpec::new(log_msg_len)];
			for estimate in
				estimate_by_rate::<B128, _>(&merkle_scheme, &oracles, SECURITY_BITS, 1..=6)
			{
				observed.push((
					log_msg_len,
					estimate.log_inv_rate,
					estimate.n_test_queries,
					estimate.proof_size,
				));
			}
		}

		// The query count falls monotonically with the rate, but the proof size bottoms out and
		// climbs again, because the terminal codeword and each opened coset grow with the rate.
		assert_eq!(
			observed,
			[
				(17, 1, 241, 211200),
				(17, 2, 148, 165504),
				(17, 3, 121, 160000),
				(17, 4, 110, 166080),
				(17, 5, 105, 178560),
				(17, 6, 103, 192832),
				(20, 1, 241, 318720),
				(20, 2, 148, 240000),
				(20, 3, 121, 222208),
				(20, 4, 110, 225376),
				(20, 5, 105, 237888),
				(20, 6, 103, 255488),
				(24, 1, 241, 488896),
				(24, 2, 148, 352416),
				(24, 3, 121, 319264),
				(24, 4, 110, 317504),
				(24, 5, 105, 329376),
				(24, 6, 103, 348608),
			]
		);
	}

	// A ZK oracle pins `log_batch_size = 1`, taking the fixed-batch-size branch of the selection.
	// The shorter non-ZK oracle beside it exercises lifting.
	#[test]
	fn pinned_proof_size_by_rate_zk_batch() {
		let merkle_scheme = test_merkle_scheme();
		let oracles = [OracleSpec::new_zk(24), OracleSpec::new(21)];

		let estimates = estimate_by_rate::<B128, _>(&merkle_scheme, &oracles, SECURITY_BITS, 1..=6);

		let expected = [
			(1, 241, 755984),
			(2, 148, 532256),
			(3, 121, 476592),
			(4, 110, 468320),
			(5, 105, 480432),
			(6, 103, 503536),
		];
		let observed = estimates
			.iter()
			.map(|estimate| (estimate.log_inv_rate, estimate.n_test_queries, estimate.proof_size))
			.collect::<Vec<_>>();
		assert_eq!(observed, expected);
	}

	// The unique-decoding query counts at 100 bits, which are what a rate change actually buys.
	// They are quoted outside this crate, so pin them against silent drift.
	#[test]
	fn pinned_n_test_queries_at_100_bits() {
		let observed = (1..=5)
			.map(|log_inv_rate| calculate_n_test_queries(SECURITY_BITS, log_inv_rate))
			.collect::<Vec<_>>();
		assert_eq!(observed, vec![241, 148, 121, 110, 105]);
	}

	// `estimate_by_rate` must be exactly the loop a caller would otherwise write by hand.
	#[test]
	fn estimate_by_rate_matches_manual_loop() {
		let merkle_scheme = test_merkle_scheme();
		let digest_size = size_of::<<TestMerkleScheme as MerkleTreeScheme<B128>>::Digest>();

		let batches = [
			vec![OracleSpec::new(20)],
			vec![OracleSpec::new_zk(20)],
			vec![
				OracleSpec::new(18),
				OracleSpec::new_zk(14),
				OracleSpec::new(9),
			],
		];

		for oracles in &batches {
			let estimates =
				estimate_by_rate::<B128, _>(&merkle_scheme, oracles, SECURITY_BITS, 1..=6);

			for estimate in &estimates {
				let n_test_queries = calculate_n_test_queries(SECURITY_BITS, estimate.log_inv_rate);
				assert_eq!(estimate.n_test_queries, n_test_queries);

				let (params, optimizer_estimate) = FRIParams::<B128>::optimal_for_batch(
					&merkle_scheme,
					oracles,
					estimate.log_inv_rate,
					n_test_queries,
				);
				assert_eq!(estimate.proof_size, proof_size(&params, &merkle_scheme));

				// `optimal_for_batch` returns the arity-search cost, which omits the commitment
				// digests: one per input oracle, one per oracle sent during the fold rounds.
				let digests = (oracles.len() + params.n_oracles()) * digest_size;
				assert_eq!(optimizer_estimate + digests, estimate.proof_size);
			}
		}
	}

	// `best_rate` is the minimum over the same sweep, and ties go to the first candidate.
	#[test]
	fn best_rate_minimizes_the_sweep() {
		let merkle_scheme = test_merkle_scheme();

		for log_msg_len in [0, 1, 8, 17, 20, 24] {
			let oracles = [OracleSpec::new(log_msg_len)];
			let estimates =
				estimate_by_rate::<B128, _>(&merkle_scheme, &oracles, SECURITY_BITS, 1..=6);
			let expected = estimates
				.iter()
				.copied()
				.min_by_key(|estimate| estimate.proof_size)
				.expect("the sweep is non-empty");

			let best = best_rate::<B128, _>(&merkle_scheme, &oracles, SECURITY_BITS, 1..=6);
			assert_eq!(best, expected, "log_msg_len={log_msg_len}");

			// The winner really is a minimum, and no earlier candidate ties it.
			for estimate in &estimates {
				assert!(best.proof_size <= estimate.proof_size);
				if estimate.log_inv_rate < best.log_inv_rate {
					assert!(estimate.proof_size > best.proof_size);
				}
			}
		}
	}

	// Boundaries on the candidate iterator.
	// An empty `oracles` slice is not a case, since it violates `optimal_for_batch`'s precondition.
	#[test]
	fn estimate_by_rate_candidate_boundaries() {
		let merkle_scheme = test_merkle_scheme();
		let oracles = [OracleSpec::new(16)];

		// No candidates in, no estimates out.
		let empty = estimate_by_rate::<B128, _>(&merkle_scheme, &oracles, SECURITY_BITS, []);
		assert_eq!(empty, vec![]);

		// A single candidate is priced, and is trivially its own best rate.
		let single = estimate_by_rate::<B128, _>(&merkle_scheme, &oracles, SECURITY_BITS, [3]);
		assert_eq!(single.len(), 1);
		assert_eq!(single[0].log_inv_rate, 3);
		assert_eq!(single[0].n_test_queries, 121);
		assert_eq!(best_rate::<B128, _>(&merkle_scheme, &oracles, SECURITY_BITS, [3]), single[0]);
	}

	// `best_rate` has nothing to return when the sweep is empty.
	#[test]
	#[should_panic(expected = "log_inv_rates yields at least one candidate rate")]
	fn best_rate_panics_without_candidates() {
		let merkle_scheme = test_merkle_scheme();
		best_rate::<B128, _>(&merkle_scheme, &[OracleSpec::new(16)], SECURITY_BITS, []);
	}
}
