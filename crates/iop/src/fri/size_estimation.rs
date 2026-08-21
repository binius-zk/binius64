// Copyright 2026 The Binius Developers

//! Pricing a FRI proof across candidate Reed-Solomon rates.

use binius_field::BinaryField;

use super::common::FRIParams;
use crate::{channel::OracleSpec, merkle_tree::MerkleTreeScheme};

/// One candidate Reed-Solomon rate, together with what a proof at that rate costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateEstimate {
	/// The binary logarithm of the inverse Reed-Solomon code rate.
	pub log_inv_rate: usize,
	/// The number of test queries needed to hit the target security level at this rate.
	pub n_test_queries: usize,
	/// The exact byte-size of the proof at this rate, counting every part of it.
	pub proof_size: usize,
}

impl RateEstimate {
	/// Prices a proof at one candidate Reed-Solomon rate.
	///
	/// The fold arities are re-optimized for this rate by [`FRIParams::optimal_for_batch`].
	/// That optimizer's own estimate omits the commitment digests, so the parameters it returns are
	/// re-priced here with [`FRIParams::proof_size`], which counts the whole proof.
	///
	/// ## Arguments
	///
	/// * `merkle_scheme` - the Merkle tree scheme used for commitments.
	/// * `oracles` - the oracles to batch, as passed to [`FRIParams::optimal_for_batch`].
	/// * `log_inv_rate` - the binary logarithm of the inverse code rate to price.
	/// * `n_test_queries` - the number of test queries at that rate.
	///
	/// ## Preconditions
	///
	/// * `oracles` is non-empty.
	pub fn for_rate<F, MerkleScheme>(
		merkle_scheme: &MerkleScheme,
		oracles: &[OracleSpec],
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> Self
	where
		F: BinaryField,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		let (params, _) =
			FRIParams::<F>::optimal_for_batch(merkle_scheme, oracles, log_inv_rate, n_test_queries);
		Self {
			log_inv_rate,
			n_test_queries,
			proof_size: params.proof_size(merkle_scheme),
		}
	}

	/// Prices a proof at every candidate rate, one [`Self::for_rate`] per candidate.
	///
	/// A larger `log_inv_rate` needs fewer test queries, which shrinks the proof.
	/// It also lengthens every codeword, which costs the prover proportionally more encoding work.
	/// This prices the bytes only, so the encoding cost is left to benchmarks.
	///
	/// The query count arrives as a rate-indexed closure rather than a fixed formula.
	/// Rate is not the only lever on it: so is the proximity-testing regime.
	/// Pass [`calculate_n_test_queries`](super::calculate_n_test_queries) to price the
	/// unique-decoding regime this repo ships.
	///
	/// ## Arguments
	///
	/// * `merkle_scheme` - the Merkle tree scheme used for commitments.
	/// * `oracles` - the oracles to batch, as passed to [`FRIParams::optimal_for_batch`].
	/// * `n_queries` - the test-query count to use at a given `log_inv_rate`.
	/// * `log_inv_rates` - the candidate rates to price.
	///
	/// ## Returns
	///
	/// One estimate per candidate rate, in the order the candidates were yielded.
	///
	/// ## Preconditions
	///
	/// * `oracles` is non-empty.
	pub fn sweep<F, MerkleScheme>(
		merkle_scheme: &MerkleScheme,
		oracles: &[OracleSpec],
		n_queries: impl Fn(usize) -> usize,
		log_inv_rates: impl IntoIterator<Item = usize>,
	) -> Vec<Self>
	where
		F: BinaryField,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		log_inv_rates
			.into_iter()
			.map(|log_inv_rate| {
				Self::for_rate(merkle_scheme, oracles, log_inv_rate, n_queries(log_inv_rate))
			})
			.collect()
	}

	/// The candidate rate with the smallest proof, or `None` when there were no candidates.
	///
	/// Ties go to the first candidate yielded.
	/// With candidates in ascending `log_inv_rate` order, that is the cheapest of them to encode.
	///
	/// ## Arguments
	///
	/// Those of [`Self::sweep`].
	///
	/// ## Preconditions
	///
	/// * `oracles` is non-empty.
	pub fn best<F, MerkleScheme>(
		merkle_scheme: &MerkleScheme,
		oracles: &[OracleSpec],
		n_queries: impl Fn(usize) -> usize,
		log_inv_rates: impl IntoIterator<Item = usize>,
	) -> Option<Self>
	where
		F: BinaryField,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		Self::sweep(merkle_scheme, oracles, n_queries, log_inv_rates)
			.into_iter()
			.min_by_key(|estimate| estimate.proof_size)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b as B128;
	use binius_hash::StdHashSuite;

	use super::*;
	use crate::{fri::calculate_n_test_queries, merkle_tree::BinaryMerkleTreeScheme};

	type TestMerkleScheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

	fn test_merkle_scheme() -> TestMerkleScheme {
		BinaryMerkleTreeScheme::new()
	}

	const SECURITY_BITS: usize = 100;

	/// The query count of the unique-decoding regime this repo ships, as a rate-indexed closure.
	fn udr(log_inv_rate: usize) -> usize {
		calculate_n_test_queries(SECURITY_BITS, log_inv_rate)
	}

	// One non-ZK oracle over rates 1/2 down to 1/64.
	// Each row is `(log_msg_len, log_inv_rate, n_test_queries, proof_size)`.
	#[test]
	fn pinned_proof_size_by_rate() {
		let merkle_scheme = test_merkle_scheme();

		// Sweep in candidate order, so the observed table is directly comparable to the pinned one.
		let mut observed = Vec::new();
		for log_msg_len in [17usize, 20, 24] {
			let oracles = [OracleSpec::new(log_msg_len)];
			for estimate in RateEstimate::sweep::<B128, _>(&merkle_scheme, &oracles, udr, 1..=6) {
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

		let estimates = RateEstimate::sweep::<B128, _>(&merkle_scheme, &oracles, udr, 1..=6);

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

	// `sweep` must be exactly the loop a caller would otherwise write by hand.
	#[test]
	fn sweep_matches_a_manual_loop() {
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
			let estimates = RateEstimate::sweep::<B128, _>(&merkle_scheme, oracles, udr, 1..=6);

			for estimate in &estimates {
				let n_test_queries = calculate_n_test_queries(SECURITY_BITS, estimate.log_inv_rate);
				assert_eq!(estimate.n_test_queries, n_test_queries);

				let (params, optimizer_estimate) = FRIParams::<B128>::optimal_for_batch(
					&merkle_scheme,
					oracles,
					estimate.log_inv_rate,
					n_test_queries,
				);
				assert_eq!(estimate.proof_size, params.proof_size(&merkle_scheme));

				// `optimal_for_batch` returns the arity-search cost, which omits the commitment
				// digests: one per input oracle, one per oracle sent during the fold rounds.
				let digests = (oracles.len() + params.n_oracles()) * digest_size;
				assert_eq!(optimizer_estimate + digests, estimate.proof_size);
			}
		}
	}

	// `best` is the minimum over the same sweep, and ties go to the first candidate.
	#[test]
	fn best_minimizes_the_sweep() {
		let merkle_scheme = test_merkle_scheme();

		for log_msg_len in [0, 1, 8, 17, 20, 24] {
			let oracles = [OracleSpec::new(log_msg_len)];
			let estimates = RateEstimate::sweep::<B128, _>(&merkle_scheme, &oracles, udr, 1..=6);
			let expected = estimates
				.iter()
				.copied()
				.min_by_key(|estimate| estimate.proof_size)
				.expect("the sweep is non-empty");

			let best = RateEstimate::best::<B128, _>(&merkle_scheme, &oracles, udr, 1..=6)
				.expect("the sweep is non-empty");
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
	fn sweep_candidate_boundaries() {
		let merkle_scheme = test_merkle_scheme();
		let oracles = [OracleSpec::new(16)];

		// No candidates in, no estimates out.
		let empty = RateEstimate::sweep::<B128, _>(&merkle_scheme, &oracles, udr, []);
		assert_eq!(empty, vec![]);

		// A single candidate is priced, and is trivially its own best rate.
		let single = RateEstimate::sweep::<B128, _>(&merkle_scheme, &oracles, udr, [3]);
		assert_eq!(single.len(), 1);
		assert_eq!(single[0].log_inv_rate, 3);
		assert_eq!(single[0].n_test_queries, 121);
		assert_eq!(
			RateEstimate::best::<B128, _>(&merkle_scheme, &oracles, udr, [3]),
			Some(single[0])
		);

		// And no candidates means no winner, rather than a panic.
		assert_eq!(RateEstimate::best::<B128, _>(&merkle_scheme, &oracles, udr, []), None);
	}

	// The query count is a parameter so that a different proximity-testing regime can be priced
	// without touching this module. Halving it must shrink the proof at every rate.
	#[test]
	fn a_smaller_query_count_prices_a_smaller_proof() {
		let merkle_scheme = test_merkle_scheme();
		let oracles = [OracleSpec::new(20)];

		let baseline = RateEstimate::sweep::<B128, _>(&merkle_scheme, &oracles, udr, 1..=6);
		let halved =
			RateEstimate::sweep::<B128, _>(&merkle_scheme, &oracles, |r| udr(r) / 2, 1..=6);

		for (base, half) in std::iter::zip(&baseline, &halved) {
			assert_eq!(base.log_inv_rate, half.log_inv_rate);
			assert_eq!(half.n_test_queries, base.n_test_queries / 2);
			assert!(half.proof_size < base.proof_size, "rate 1/{}", 1 << base.log_inv_rate);
		}
	}
}
