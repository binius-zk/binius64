// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::BinaryField;
use binius_math::{ntt::AdditiveNTT, reed_solomon::ReedSolomonCode};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use getset::{CopyGetters, Getters};

use super::error::Error;
use crate::merkle_tree::MerkleTreeScheme;

/// Parameters for an FRI interleaved code proximity protocol.
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct FRIParams<F> {
	/// The Reed-Solomon code the verifier is testing proximity to.
	#[getset(get = "pub")]
	rs_code: ReedSolomonCode<F>,
	/// log2 the interleaved batch size.
	#[getset(get_copy = "pub")]
	log_batch_size: usize,
	/// The reduction arities between each oracle sent to the verifier.
	fold_arities: Vec<usize>,
	/// log2 the dimension of the terminal codeword.
	log_terminal_dim: usize,
	/// The number oracle consistency queries required during the query phase.
	#[getset(get_copy = "pub")]
	n_test_queries: usize,
}

impl<F> FRIParams<F>
where
	F: BinaryField,
{
	pub fn new(
		rs_code: ReedSolomonCode<F>,
		log_batch_size: usize,
		fold_arities: Vec<usize>,
		n_test_queries: usize,
	) -> Result<Self, Error> {
		let fold_arities_sum = fold_arities.iter().sum();
		let log_terminal_dim = rs_code
			.log_dim()
			.checked_sub(fold_arities_sum)
			.ok_or(Error::InvalidFoldAritySequence)?;

		Ok(Self {
			rs_code,
			log_batch_size,
			fold_arities,
			log_terminal_dim,
			n_test_queries,
		})
	}

	/// Choose commit parameters based on protocol parameters, using a constant fold arity.
	///
	/// ## Arguments
	///
	/// * `log_msg_len` - the binary logarithm of the length of the message to commit.
	/// * `security_bits` - the target security level in bits.
	/// * `log_inv_rate` - the binary logarithm of the inverse Reed–Solomon code rate.
	/// * `arity` - the folding arity.
	pub fn choose_with_constant_fold_arity(
		ntt: &impl AdditiveNTT<Field = F>,
		log_msg_len: usize,
		security_bits: usize,
		log_inv_rate: usize,
		arity: usize,
	) -> Result<Self, Error> {
		assert!(arity > 0);

		let log_batch_size = log_msg_len.min(arity);
		let log_dim = log_msg_len - log_batch_size;
		let rs_code = ReedSolomonCode::with_ntt_subspace(ntt, log_dim, log_inv_rate)?;
		let n_test_queries = calculate_n_test_queries(security_bits, log_inv_rate);

		// TODO: Use BinaryMerkleTreeScheme to estimate instead of log2_ceil_usize
		let cap_height = log2_ceil_usize(n_test_queries);
		let log_terminal_len = cap_height.clamp(log_inv_rate, rs_code.log_len());

		let quotient = (rs_code.log_len() - log_terminal_len) / arity;
		let remainder = (rs_code.log_len() - log_terminal_len) % arity;
		let mut fold_arities = vec![arity; quotient];
		if remainder != 0 {
			fold_arities.push(remainder);
		}

		// here is the down-to-earth explanation of what we're doing: we want the terminal
		// codeword's log-length to be at least as large as the Merkle cap height. note that
		// `total_vars + log_inv_rate - sum(fold_arities)` is exactly the log-length of the
		// terminal codeword; we want this number to be ≥ cap height. so fold_arities will repeat
		// `arity` the maximal number of times possible, while maintaining that `total_vars +
		// log_inv_rate - sum(fold_arities) ≥ cap_height` stays true. this arity-selection
		// strategy can be characterized as: "terminate as late as you can, while maintaining that
		// no Merkle cap is strictly smaller than `cap_height`." this strategy does attain that
		// property: the Merkle path height of the last non-terminal codeword will equal the
		// log-length of the terminal codeword, which is ≥ cap height by fiat. moreover, if we
		// terminated later than we are above, then this would stop being true. imagine what would
		// happen if we took the above terminal codeword and continued folding. in that case, we
		// would Merklize this word, again with the coset-bundling trick; the post-bundling path
		// height would thus be `total_vars + log_inv_rate - sum(fold_arities) - arity`. but we
		// already agreed (by the maximality of the number of times we subtracted `arity`) that
		// the above number will be < cap_height. in other words, its Merkle cap will be
		// short. equivalently: this is the latest termination for which the `min` in
		// `optimal_verify_layer` will never trigger; i.e., we will have log2_ceil_usize(n_queries)
		// ≤ tree_depth there. it can be shown that this strategy beats any strategy which
		// terminates later than it does (in other words, by doing this, we are NOT terminating
		// TOO early!). this doesn't mean that we should't terminate EVEN earlier (maybe we
		// should). but this approach is conservative and simple; and it's easy to show that you
		// won't lose by doing this.

		// see https://github.com/IrreducibleOSS/binius/pull/300 for proof of this fact
		Self::new(rs_code, log_batch_size, fold_arities, n_test_queries)
	}

	pub fn n_fold_rounds(&self) -> usize {
		self.log_msg_len()
	}

	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		// One for the batched codeword commitment, and one for each subsequent one.
		1 + self.fold_arities.len()
	}

	/// Number of bits in the query indices sampled during the query phase.
	pub fn index_bits(&self) -> usize {
		self.rs_code.log_len()
	}

	/// Number of folding challenges the verifier sends after receiving the last oracle.
	pub fn n_final_challenges(&self) -> usize {
		self.log_terminal_dim
	}

	/// The reduction arities between each oracle sent to the verifier.
	pub fn fold_arities(&self) -> &[usize] {
		&self.fold_arities
	}

	/// The binary logarithm of the length of the initial oracle.
	pub fn log_len(&self) -> usize {
		self.rs_code.log_len() + self.log_batch_size()
	}

	/// The binary logarithm of the length of the initial message.
	pub fn log_msg_len(&self) -> usize {
		self.rs_code.log_dim() + self.log_batch_size()
	}
}

/// This layer allows minimizing the proof size.
pub fn vcs_optimal_layers_depths_iter<'a, F, VCS>(
	fri_params: &'a FRIParams<F>,
	vcs: &'a VCS,
) -> impl Iterator<Item = usize> + 'a
where
	VCS: MerkleTreeScheme<F>,
	F: BinaryField,
{
	iter::once(fri_params.log_batch_size())
		.chain(fri_params.fold_arities().iter().copied())
		.scan(fri_params.log_len(), |log_n_cosets, arity| {
			*log_n_cosets -= arity;
			Some(vcs.optimal_verify_layer(fri_params.n_test_queries(), *log_n_cosets))
		})
}

/// Calculates the number of test queries required to achieve a target soundness error.
///
/// This chooses a number of test queries so that the soundness error of the FRI query phase is
/// at most $2^{-t}$, where $t$ is the threshold `security_bits`. This _does not_ account for the
/// soundness error from the FRI folding phase or any other protocols, only the query phase. This
/// sets the proximity parameter for FRI to the code's unique decoding radius. See [DP24],
/// Section 5.2, for concrete soundness analysis.
///
/// Throws [`Error::ParameterError`] if the security level is unattainable given the code
/// parameters.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub fn calculate_n_test_queries(security_bits: usize, log_inv_rate: usize) -> usize {
	let rate = 2.0f64.powi(-(log_inv_rate as i32));
	let per_query_err = 0.5 * (1f64 + rate);
	(security_bits as f64 / -per_query_err.log2()).ceil() as usize
}

/// Heuristic for estimating the optimal FRI folding arity that minimizes proof size.
///
/// `log_block_length` is the binary logarithm of the  block length of the Reed–Solomon code.
pub fn estimate_optimal_arity(
	log_block_length: usize,
	digest_size: usize,
	field_size: usize,
) -> usize {
	(1..=log_block_length)
		.map(|arity| {
			(
				// for given arity, return a tuple (arity, estimate of query_proof_size).
				// this estimate is basd on the following approximation of a single
				// query_proof_size, where $\vartheta$ is the arity: $\big((n-\vartheta) +
				// (n-2\vartheta) + \ldots\big)\text{digest_size} +
				// \frac{n-\vartheta}{\vartheta}2^{\vartheta}\text{field_size}.$
				arity,
				((log_block_length) / 2 * digest_size + (1 << arity) * field_size)
					* (log_block_length - arity)
					/ arity,
			)
		})
		// now scan and terminate the iterator when query_proof_size increases.
		.scan(None, |old: &mut Option<(usize, usize)>, new| {
			let should_continue = !matches!(*old, Some(ref old) if new.1 > old.1);
			*old = Some(new);
			should_continue.then_some(new)
		})
		.last()
		.map(|(arity, _)| arity)
		.unwrap_or(1)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_calculate_n_test_queries() {
		let security_bits = 96;
		let n_test_queries = calculate_n_test_queries(security_bits, 1);
		assert_eq!(n_test_queries, 232);

		let n_test_queries = calculate_n_test_queries(security_bits, 2);
		assert_eq!(n_test_queries, 142);
	}

	#[test]
	fn test_estimate_optimal_arity() {
		let field_size = 128;
		for log_block_length in 22..35 {
			let digest_size = 256;
			assert_eq!(estimate_optimal_arity(log_block_length, digest_size, field_size), 4);
		}

		for log_block_length in 22..28 {
			let digest_size = 1024;
			assert_eq!(estimate_optimal_arity(log_block_length, digest_size, field_size), 6);
		}
	}
}
