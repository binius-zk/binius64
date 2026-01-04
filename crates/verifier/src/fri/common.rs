// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::{BinaryField, Field};
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

	/// Create parameters that minimize proof size.
	///
	/// This uses dynamic programming to find the optimal sequence of fold arities that minimizes
	/// the total proof size, accounting for both leaf data and Merkle proof overhead.
	///
	/// ## Arguments
	///
	/// * `ntt` - the additive NTT used for Reed-Solomon encoding.
	/// * `merkle_scheme` - the Merkle tree scheme used for commitments.
	/// * `log_msg_len` - the binary logarithm of the length of the message to commit.
	/// * `log_batch_size` - if `Some`, fixes the batch size; if `None`, the batch size is chosen
	///   optimally along with the fold arities.
	/// * `log_inv_rate` - the binary logarithm of the inverse Reed–Solomon code rate.
	/// * `n_test_queries` - the number of test queries for the FRI protocol.
	///
	/// ## Preconditions
	///
	/// * If `log_batch_size` is `Some(b)`, then `b <= log_msg_len`.
	/// * `ntt.log_domain_size() >= log_msg_len - log_batch_size.unwrap_or(0) + log_inv_rate`.
	///
	/// TODO: This should accept an optional argument for `security_bits`. When it is provided, the
	/// parameters should guarantee that the codeword length is not so large as to violate the
	/// required soundness error probability. In other words, this means bounding the maximum
	/// codeword length based on the field size and security bits.
	pub fn with_min_proof_size<NTT, MerkleScheme>(
		ntt: &NTT,
		merkle_scheme: &MerkleScheme,
		log_msg_len: usize,
		log_batch_size: Option<usize>,
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> Result<Self, Error>
	where
		NTT: AdditiveNTT<Field = F>,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		let (log_batch_size, fold_arities) = match log_batch_size {
			Some(log_batch_size) => {
				assert!(log_batch_size <= log_msg_len); // precondition
				let fold_arities = find_fold_arities_for_min_proof_size(
					merkle_scheme,
					log_msg_len - log_batch_size,
					log_inv_rate,
					n_test_queries,
				);
				(log_batch_size, fold_arities)
			}
			None => {
				let mut fold_arities = find_fold_arities_for_min_proof_size(
					merkle_scheme,
					log_msg_len,
					log_inv_rate,
					n_test_queries,
				);
				let log_batch_size = if !fold_arities.is_empty() {
					fold_arities.remove(0)
				} else {
					// Edge case: fold to log_dim = 0 code.
					log_msg_len
				};
				(log_batch_size, fold_arities)
			}
		};

		let log_dim = log_msg_len - log_batch_size;
		let rs_code = ReedSolomonCode::with_ntt_subspace(ntt, log_dim, log_inv_rate)?;
		Self::new(rs_code, log_batch_size, fold_arities, n_test_queries)
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

fn find_fold_arities_for_min_proof_size<F, MerkleScheme>(
	merkle_scheme: &MerkleScheme,
	log_msg_len: usize,
	log_inv_rate: usize,
	n_test_queries: usize,
) -> Vec<usize>
where
	F: Field,
	MerkleScheme: MerkleTreeScheme<F>,
{
	// This algorithm uses a dynamic programming approach to determine the sequence of arities that
	// minimizes proof size. For each i in [0, log_msg_len], we determine the minimum proof size
	// attainable when for a batched codeword with message size 2^i. This is determined by
	// minimizing over the first reduction arity, using the values already determined for the
	// smaller values of i.

	// This vec maps log_msg_len values to the minimum proof size attainable for a batched FRI
	// protocol committing a message with that length.
	let mut min_sizes = Vec::<Entry>::with_capacity(log_msg_len);

	#[derive(Debug)]
	struct Entry {
		// The minimum proof size attainable for the indexed value of i.
		proof_size: usize,
		// The first reduction arity to achieve the minimum proof size. If the value is none, then
		// the best reduction sequence is to skip all folding and send the full codeword.
		arity: Option<usize>,
	}

	// The byte-size of an element.
	let value_size = {
		let mut buf = Vec::new();
		F::default()
			.serialize(&mut buf)
			.expect("default element can be serialized to a resizable buffer");
		buf.len()
	};

	for i in 0..=log_msg_len {
		// Length of the batched codeword.
		let log_code_len = i + log_inv_rate;

		let mut last_entry = None;
		for arity in 1..=i {
			// The additional proof bytes for the reduction by arity.
			let reduction_proof_size = {
				// Each queried coset contains 2^arity values.
				let leaf_size = value_size << arity;
				// One coset per test query.
				let leaves_size = leaf_size * n_test_queries;

				// Size of the Merkle multi-proof.
				let optimal_layer =
					merkle_scheme.optimal_verify_layer(n_test_queries, log_code_len);
				let merkle_size = merkle_scheme
					.proof_size(1 << log_code_len, n_test_queries, optimal_layer)
					.expect("layer computed with optimal_layer must be valid");
				leaves_size + merkle_size
			};

			let reduced_proof_size = min_sizes[i - arity].proof_size;
			let proof_size = reduction_proof_size + reduced_proof_size;
			let replace = last_entry
				.as_ref()
				.is_none_or(|last_entry: &Entry| proof_size <= last_entry.proof_size);
			if replace {
				last_entry = Some(Entry {
					proof_size,
					arity: Some(arity),
				});
			} else {
				// The proof size function is concave with respect to arity. Break as soon is it
				// ascends.
				break;
			}
		}

		// Determine the proof size if this is the terminal codeword. In that case, the proof simply
		// consists of the 2^(i + log_inv_rate) leaf values.
		let terminal_proof_size = value_size << log_code_len;
		let terminal_entry = Entry {
			proof_size: terminal_proof_size,
			arity: None,
		};

		let optimal_entry = if let Some(last_entry) = last_entry
			&& last_entry.proof_size < terminal_entry.proof_size
		{
			last_entry
		} else {
			terminal_entry
		};

		min_sizes.push(optimal_entry);
	}

	let mut fold_arities = Vec::with_capacity(log_msg_len);

	let mut i = log_msg_len;
	let mut entry = &min_sizes[i];
	while let Some(arity) = entry.arity {
		fold_arities.push(arity);
		i -= arity;
		entry = &min_sizes[i];
	}
	fold_arities
}

#[cfg(test)]
mod tests {
	use binius_math::ntt::{NeighborsLastReference, domain_context::GaoMateerOnTheFly};

	use super::*;
	use crate::{config::B128, hash::StdCompression, merkle_tree::BinaryMerkleTreeScheme};

	type StdDigest = sha2::Sha256;
	type TestMerkleScheme = BinaryMerkleTreeScheme<B128, StdDigest, StdCompression>;

	fn test_merkle_scheme() -> TestMerkleScheme {
		BinaryMerkleTreeScheme::new(StdCompression::default())
	}

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

	#[test]
	fn test_find_fold_arities_for_min_proof_size() {
		let merkle_scheme = test_merkle_scheme();
		let log_inv_rate = 2;
		let n_test_queries = 128;

		// log_msg_len = 0: no folding needed, terminal codeword is optimal
		let arities = find_fold_arities_for_min_proof_size::<B128, _>(
			&merkle_scheme,
			0,
			log_inv_rate,
			n_test_queries,
		);
		assert_eq!(arities, vec![]);

		// log_msg_len = 3: no folding needed, terminal codeword is optimal
		let arities = find_fold_arities_for_min_proof_size::<B128, _>(
			&merkle_scheme,
			3,
			log_inv_rate,
			n_test_queries,
		);
		assert_eq!(arities, vec![]);

		// log_msg_len = 24
		let arities = find_fold_arities_for_min_proof_size::<B128, _>(
			&merkle_scheme,
			24,
			log_inv_rate,
			n_test_queries,
		);
		assert_eq!(arities, vec![4, 4, 4, 4]);
	}

	#[test]
	fn test_with_min_proof_size() {
		let merkle_scheme = test_merkle_scheme();
		let log_inv_rate = 2;
		let n_test_queries = 128;

		let ntt = NeighborsLastReference {
			domain_context: GaoMateerOnTheFly::<B128>::generate(24 + log_inv_rate),
		};

		// log_msg_len = 0
		{
			let fri_params = FRIParams::with_min_proof_size(
				&ntt,
				&merkle_scheme,
				0,
				None,
				log_inv_rate,
				n_test_queries,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[]);
			assert_eq!(fri_params.log_batch_size(), 0);
		}

		// log_msg_len = 3
		{
			let fri_params = FRIParams::with_min_proof_size(
				&ntt,
				&merkle_scheme,
				3,
				None,
				log_inv_rate,
				n_test_queries,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[]);
			assert_eq!(fri_params.log_batch_size(), 3);
		}

		// log_msg_len = 24
		{
			let fri_params = FRIParams::with_min_proof_size(
				&ntt,
				&merkle_scheme,
				24,
				None,
				log_inv_rate,
				n_test_queries,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[4, 4, 4]);
			assert_eq!(fri_params.log_batch_size(), 4);
		}
	}

	#[test]
	fn test_with_min_proof_size_fixed_batch_size() {
		let merkle_scheme = test_merkle_scheme();
		let log_inv_rate = 2;
		let n_test_queries = 128;

		let ntt = NeighborsLastReference {
			domain_context: GaoMateerOnTheFly::<B128>::generate(24 + log_inv_rate),
		};

		// log_msg_len = 3
		{
			let fri_params = FRIParams::with_min_proof_size(
				&ntt,
				&merkle_scheme,
				3,
				Some(1),
				log_inv_rate,
				n_test_queries,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[]);
			assert_eq!(fri_params.log_batch_size(), 1);
		}

		// log_msg_len = 24
		{
			let fri_params = FRIParams::with_min_proof_size(
				&ntt,
				&merkle_scheme,
				24,
				Some(1),
				log_inv_rate,
				n_test_queries,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[4, 4, 4, 3]);
			assert_eq!(fri_params.log_batch_size(), 1);
		}
	}
}
