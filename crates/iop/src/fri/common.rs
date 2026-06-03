// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{BinaryField, Field};
use binius_math::{ntt::DomainContext, reed_solomon::ReedSolomonCode};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use getset::{CopyGetters, Getters};

use super::error::Error;
use crate::merkle_tree::MerkleTreeScheme;

/// Parameters for an FRI interleaved code proximity protocol.
///
/// ## Invariants
///
/// The dimension of the first-round (reduced) FRI oracle is
/// `rs_code.log_dim() == log_terminal_dim + sum(fold_arities)`. For all oracle specs in
/// `input_oracles`:
/// - `log_batch_size <= log_msg_len`
/// - `log_msg_len <= rs_code.log_dim() + log_batch_size` (equivalently, `log_msg_len <=
///   log_terminal_dim + sum(fold_arities) + log_batch_size`)
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct FRIParams<F> {
	/// The Reed-Solomon code the verifier is testing proximity to.
	#[getset(get = "pub")]
	rs_code: ReedSolomonCode<F>,
	/// Guaranteed to be non-empty.
	input_oracles: Vec<OracleSpec>,
	/// log2 the maximum message length of all input oracles, after lifting each to the reduced
	/// dimension. Equals `rs_code.log_dim() + max(input_oracles.log_batch_size)`.
	max_log_msg_len: usize,
	/// log2 ceiling of the number of input oracles.
	log_n_oracles: usize,
	/// The reduction arities between each oracle sent to the verifier.
	fold_arities: Vec<usize>,
	/// log2 the dimension of the terminal codeword.
	log_terminal_dim: usize,
	/// The number oracle consistency queries required during the query phase.
	#[getset(get_copy = "pub")]
	n_test_queries: usize,
}

#[derive(Debug, Clone)]
pub struct OracleSpec {
	/// log2 the message length, i.e. the Reed–Solomon code dimension plus the batch size.
	pub log_msg_len: usize,
	/// log2 the interleaved batch size.
	pub log_batch_size: usize,
}

/// A partially specified oracle specification.
///
/// The partial specification may have a flexible batch size, to be decided when the [`FRIParams`]
/// are instantiated.
#[derive(Debug, Clone)]
pub struct PartialOracleSpec {
	/// log2 the message length, i.e. the Reed–Solomon code dimension plus the batch size.
	pub log_msg_len: usize,
	/// log2 the interleaved batch size.
	///
	/// This field is Some if the log_batch_size is fixed, and None if it's flexible.
	pub log_batch_size: Option<usize>,
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

		let oracle_spec = OracleSpec {
			log_msg_len: rs_code.log_dim() + log_batch_size,
			log_batch_size,
		};
		let log_n_oracles = 0;
		let max_log_msg_len = oracle_spec.log_msg_len;
		Ok(Self {
			rs_code,
			input_oracles: vec![oracle_spec],
			max_log_msg_len,
			log_n_oracles,
			fold_arities,
			log_terminal_dim,
			n_test_queries,
		})
	}

	/// Create parameters using the given arity selection strategy.
	///
	/// ## Arguments
	///
	/// * `domain_context` - the domain context providing subspaces for the Reed-Solomon code.
	/// * `merkle_scheme` - the Merkle tree scheme used for commitments.
	/// * `log_msg_len` - the binary logarithm of the length of the message to commit.
	/// * `log_batch_size` - if `Some`, fixes the batch size; if `None`, the batch size is chosen
	///   optimally along with the fold arities.
	/// * `log_inv_rate` - the binary logarithm of the inverse Reed–Solomon code rate.
	/// * `n_test_queries` - the number of test queries for the FRI protocol.
	/// * `strategy` - the strategy for selecting fold arities.
	///
	/// ## Preconditions
	///
	/// * If `log_batch_size` is `Some(b)`, then `b <= log_msg_len`.
	/// * `domain_context.log_domain_size() >= log_msg_len - log_batch_size.unwrap_or(0) +
	///   log_inv_rate`.
	pub fn with_strategy<DC, MerkleScheme, Strategy>(
		domain_context: &DC,
		merkle_scheme: &MerkleScheme,
		log_msg_len: usize,
		log_batch_size: Option<usize>,
		log_inv_rate: usize,
		n_test_queries: usize,
		strategy: &Strategy,
	) -> Result<Self, Error>
	where
		DC: DomainContext<Field = F>,
		MerkleScheme: MerkleTreeScheme<F>,
		Strategy: AritySelectionStrategy,
	{
		let (log_batch_size, fold_arities) = choose_batch_size_and_arities::<F, _, _>(
			merkle_scheme,
			log_msg_len,
			log_batch_size,
			log_inv_rate,
			n_test_queries,
			strategy,
		);

		let log_dim = log_msg_len - log_batch_size;
		let rs_code =
			ReedSolomonCode::with_domain_context_subspace(domain_context, log_dim, log_inv_rate);
		Self::new(rs_code, log_batch_size, fold_arities, n_test_queries)
	}

	/// Create parameters for a batch of input oracles, minimizing the estimated proof size.
	///
	/// The input oracles may have differing message lengths. Each oracle is reduced into a common
	/// first-round FRI oracle, whose dimension is chosen to minimize the estimated proof size; the
	/// per-oracle batch sizes and the subsequent fold arities are chosen along with it.
	///
	/// Returns the parameters together with the estimated proof size in bytes.
	///
	/// ## Arguments
	///
	/// * `domain_context` - the domain context providing subspaces for the Reed-Solomon code.
	/// * `merkle_scheme` - the Merkle tree scheme used for commitments.
	/// * `oracles` - the input oracles. An oracle's `log_batch_size` may be `Some` to fix it, or
	///   `None` to have it chosen optimally.
	/// * `log_inv_rate` - the binary logarithm of the inverse Reed–Solomon code rate.
	/// * `n_test_queries` - the number of test queries for the FRI protocol.
	///
	/// ## Preconditions
	///
	/// * `oracles` is non-empty.
	/// * For each oracle with a fixed `log_batch_size`, `log_batch_size <= log_msg_len`.
	/// * `domain_context.log_domain_size()` is large enough for the chosen reduced dimension plus
	///   `log_inv_rate`.
	pub fn optimal_for_batch<DC, MerkleScheme>(
		domain_context: &DC,
		merkle_scheme: &MerkleScheme,
		oracles: &[PartialOracleSpec],
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> (Self, usize)
	where
		DC: DomainContext<Field = F>,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		assert!(!oracles.is_empty()); // precondition
		for oracle in oracles {
			if let Some(log_batch_size) = oracle.log_batch_size {
				// precondition
				assert!(log_batch_size <= oracle.log_msg_len);
			}
		}

		let log_n_oracles = log2_ceil_usize(oracles.len());

		let ChooseBatchSizeAndAritiesOutput {
			proof_size,
			reduced_log_msg_len,
			oracle_specs,
			fold_arities,
		} = choose_batch_size_and_arities_multi(merkle_scheme, oracles, log_inv_rate, n_test_queries);

		// After lifting, every input oracle sits at the reduced dimension with its own interleaved
		// batch, so the effective maximum message length is the reduced dimension plus the largest
		// batch size. This is what the first fold must consume (the inner, per-oracle interleave
		// folds) before the `log_n_oracles` outer folds that batch the oracles together. Without
		// lifting this equals `max(oracle.log_msg_len)`.
		let max_log_batch_size = oracle_specs
			.iter()
			.map(|spec| spec.log_batch_size)
			.max()
			.expect("precondition: oracles is not empty");
		let max_log_msg_len = reduced_log_msg_len + max_log_batch_size;

		let rs_code = ReedSolomonCode::with_domain_context_subspace(
			domain_context,
			reduced_log_msg_len,
			log_inv_rate,
		);

		let fold_arities_sum = fold_arities.iter().sum::<usize>();
		let log_terminal_dim = rs_code.log_dim() - fold_arities_sum;

		let params = Self {
			rs_code,
			input_oracles: oracle_specs,
			max_log_msg_len,
			log_n_oracles,
			fold_arities,
			log_terminal_dim,
			n_test_queries,
		};
		(params, proof_size)
	}

	/// Number of folding rounds in the FRI protocol.
	///
	/// This is the largest input message length, plus `log_n_oracles` extra rounds that fold the
	/// distinct input oracles together into the batched codeword.
	pub fn n_fold_rounds(&self) -> usize {
		self.max_log_msg_len + self.log_n_oracles
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

	/// The specifications of the input oracles batched into the first-round FRI oracle.
	pub fn input_oracles(&self) -> &[OracleSpec] {
		&self.input_oracles
	}

	/// The arity of the reduction to the first round oracle.
	pub fn log_batch_size(&self) -> usize {
		self.log_msg_len() - self.rs_code().log_dim()
	}

	/// The binary logarithm of the length of the initial oracle.
	pub fn log_len(&self) -> usize {
		self.log_msg_len() + self.rs_code().log_inv_rate()
	}

	/// The binary logarithm of the length of the initial message.
	///
	/// This includes the `log_n_oracles` extra rounds used to fold the distinct input oracles
	/// together, so it equals [`Self::n_fold_rounds`].
	pub fn log_msg_len(&self) -> usize {
		self.max_log_msg_len + self.log_n_oracles
	}
}

fn choose_batch_size_and_arities<F, MerkleScheme, Strategy>(
	merkle_scheme: &MerkleScheme,
	log_msg_len: usize,
	log_batch_size: Option<usize>,
	log_inv_rate: usize,
	n_test_queries: usize,
	strategy: &Strategy,
) -> (usize, Vec<usize>)
where
	F: BinaryField,
	MerkleScheme: MerkleTreeScheme<F>,
	Strategy: AritySelectionStrategy,
{
	match log_batch_size {
		Some(log_batch_size) => {
			assert!(log_batch_size <= log_msg_len); // precondition
			let fold_arities = strategy.choose_arities::<F, _>(
				merkle_scheme,
				log_msg_len - log_batch_size,
				log_inv_rate,
				n_test_queries,
			);
			(log_batch_size, fold_arities)
		}
		None => {
			let mut fold_arities = strategy.choose_arities::<F, _>(
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
	}
}

struct ChooseBatchSizeAndAritiesOutput {
	proof_size: usize,
	reduced_log_msg_len: usize,
	oracle_specs: Vec<OracleSpec>,
	fold_arities: Vec<usize>,
}

fn choose_batch_size_and_arities_multi<F, MerkleScheme>(
	merkle_scheme: &MerkleScheme,
	oracles: &[PartialOracleSpec],
	log_inv_rate: usize,
	n_test_queries: usize,
) -> ChooseBatchSizeAndAritiesOutput
where
	F: BinaryField,
	MerkleScheme: MerkleTreeScheme<F>,
{
	// We want to determine the dimension of the first folded FRI oracle, which we'll call the
	// "reduced" oracle. This is reduced_log_msg_len. For each input oracle, we will determine a
	// log_batch_size. Some input oracles have a fixed log_batch_size, and some have a flexible one
	// (determined by whether log_batch_size is Some or None). For all input oracles, we need their
	// log_batch_size <= log_msg_len and log_msg_len <= reduced_log_msg_len + log_batch_size. We
	// allow log_msg_len to be less than reduced_log_msg_len + log_batch_size because we can lift
	// Reed-Solomon encoded oracles.

	// First, figure out lower and upper bounds on the reduced_log_msg_len. If there are any input
	// oracles with a fixed log_batch_size, then their dimension lower bounds the reduced oracle
	// dimension.
	let min_reduced_log_msg_len = oracles
		.iter()
		.filter_map(|oracle| {
			let log_batch_size = oracle.log_batch_size?;
			Some(oracle.log_msg_len - log_batch_size)
		})
		.max()
		.unwrap_or(0);
	// The upper bound is then the log_msg_len of the largest flexible oracle, if larger.
	let max_reduced_log_msg_len = oracles
		.iter()
		.filter(|oracle| oracle.log_batch_size.is_none())
		.map(|oracle| oracle.log_msg_len)
		.max()
		.unwrap_or(0)
		.max(min_reduced_log_msg_len);

	let optimizer = ReductionOptimizer::<F, _>::new(merkle_scheme, n_test_queries);
	let min_sizes = optimizer.compute_optimal_arities(max_reduced_log_msg_len, log_inv_rate);

	// Compute the contribution of the oracles with a fixed batch size to the initial fold reduction
	// size. It's not necessary to compute this for the purpose of parameter minimization, but it's
	// nice to get the resulting proof size estimate from this function.
	let fixed_reduction_size = oracles
		.iter()
		.filter_map(|oracle| {
			oracle.log_batch_size.map(|log_batch_size| {
				optimizer
					.compute_layer_reduction_size(oracle.log_msg_len + log_inv_rate, log_batch_size)
			})
		})
		.sum::<usize>();

	let (reduced_log_msg_len, proof_size) = min_concave(
		(min_reduced_log_msg_len..=max_reduced_log_msg_len).rev(),
		|reduced_log_msg_len| {
			// Compute the reduction sizes of the oracles with a flexible batch size, assuming
			// the first FRI round oracle has dimension reduced_log_msg_len.
			let non_fixed_reduction_size = oracles
				.iter()
				.filter(|oracle| oracle.log_batch_size.is_none())
				.map(|oracle| {
					optimizer.compute_layer_reduction_size(
						oracle.log_msg_len + log_inv_rate,
						oracle.log_msg_len.saturating_sub(reduced_log_msg_len),
					)
				})
				.sum::<usize>();

			let reduction_size = fixed_reduction_size + non_fixed_reduction_size;
			let reduced_proof_size = min_sizes[reduced_log_msg_len].proof_size;
			reduction_size + reduced_proof_size
		},
	)
	.expect("range is non-empty because it's inclusive of an upper bound >= the lower bound");

	let oracle_specs = oracles
		.iter()
		.map(|oracle| {
			let log_batch_size = oracle
				.log_batch_size
				.unwrap_or_else(|| oracle.log_msg_len.saturating_sub(reduced_log_msg_len));
			OracleSpec {
				log_msg_len: oracle.log_msg_len,
				log_batch_size,
			}
		})
		.collect();

	let fold_arities =
		optimizer.optimizer_entries_to_fold_arities(&min_sizes[..reduced_log_msg_len + 1]);
	ChooseBatchSizeAndAritiesOutput {
		proof_size,
		reduced_log_msg_len,
		oracle_specs,
		fold_arities,
	}
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

/// Strategy for selecting fold arities in the FRI protocol.
pub trait AritySelectionStrategy {
	fn choose_arities<F, MerkleScheme>(
		&self,
		merkle_scheme: &MerkleScheme,
		log_msg_len: usize,
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> Vec<usize>
	where
		F: Field,
		MerkleScheme: MerkleTreeScheme<F>;
}

#[derive(Debug)]
struct ReductionOptimizerEntry {
	// The minimum proof size attainable for the indexed value of i.
	proof_size: usize,
	// The first reduction arity to achieve the minimum proof size. If the value is none,
	// then the best reduction sequence is to skip all folding and send the full codeword.
	arity: Option<usize>,
}

struct ReductionOptimizer<'a, F, MTScheme> {
	merkle_scheme: &'a MTScheme,
	n_test_queries: usize,
	/// The byte-size of a serialized field element, cached at construction.
	value_size: usize,
	_marker: PhantomData<F>,
}

impl<'a, F, MTScheme> ReductionOptimizer<'a, F, MTScheme>
where
	F: Field,
	MTScheme: MerkleTreeScheme<F>,
{
	fn new(merkle_scheme: &'a MTScheme, n_test_queries: usize) -> Self {
		// The byte-size of a serialized field element.
		let value_size = {
			let mut buf = Vec::new();
			F::default()
				.serialize(&mut buf)
				.expect("default element can be serialized to a resizable buffer");
			buf.len()
		};
		Self {
			merkle_scheme,
			n_test_queries,
			value_size,
			_marker: PhantomData,
		}
	}

	fn compute_layer_reduction_size(&self, log_code_len: usize, arity: usize) -> usize {
		// Each queried coset contains 2^arity values.
		let leaf_size = self.value_size << arity;
		// One coset per test query.
		let leaves_size = leaf_size * self.n_test_queries;

		// Size of the Merkle multi-proof.
		let optimal_layer = self
			.merkle_scheme
			.optimal_verify_layer(self.n_test_queries, log_code_len);
		let merkle_size = self
			.merkle_scheme
			.proof_size(1 << log_code_len, self.n_test_queries, optimal_layer)
			.expect("layer computed with optimal_layer must be valid");

		leaves_size + merkle_size
	}

	fn compute_optimal_arities(
		&self,
		log_msg_len: usize,
		log_inv_rate: usize,
	) -> Vec<ReductionOptimizerEntry> {
		type Entry = ReductionOptimizerEntry;

		// This algorithm uses a dynamic programming approach to determine the sequence of arities
		// that minimizes proof size. For each i in [0, log_msg_len], we determine the minimum
		// proof size attainable when for a batched codeword with message size 2^i. This is
		// determined by minimizing over the first reduction arity, using the values already
		// determined for the smaller values of i.

		// This vec maps log_msg_len values to the minimum proof size attainable for a batched FRI
		// protocol committing a message with that length.
		let mut min_sizes = Vec::<Entry>::with_capacity(log_msg_len + 1);

		for i in 0..=log_msg_len {
			// Length of the batched codeword.
			let log_code_len = i + log_inv_rate;

			let non_terminal_entry = min_concave(1..=i, |arity| {
				// The additional proof bytes for the reduction by arity.
				let reduction_proof_size = self.compute_layer_reduction_size(log_code_len, arity);
				let reduced_proof_size = min_sizes[i - arity].proof_size;
				reduction_proof_size + reduced_proof_size
			})
			.map(|(arity, proof_size)| Entry {
				arity: Some(arity),
				proof_size,
			});

			// Determine the proof size if this is the terminal codeword. In that case, the proof
			// simply consists of the 2^(i + log_inv_rate) leaf values.
			let terminal_proof_size = self.value_size << log_code_len;
			let terminal_entry = Entry {
				proof_size: terminal_proof_size,
				arity: None,
			};

			let optimal_entry = if let Some(non_terminal_entry) = non_terminal_entry
				&& non_terminal_entry.proof_size < terminal_entry.proof_size
			{
				non_terminal_entry
			} else {
				terminal_entry
			};

			min_sizes.push(optimal_entry);
		}

		min_sizes
	}

	fn optimizer_entries_to_fold_arities(
		&self,
		min_sizes: &[ReductionOptimizerEntry],
	) -> Vec<usize> {
		let mut fold_arities = Vec::new();

		let mut i = min_sizes.len() - 1;
		let mut entry = &min_sizes[i];
		while let Some(arity) = entry.arity {
			fold_arities.push(arity);
			i -= arity;
			entry = &min_sizes[i];
		}
		fold_arities
	}
}

/// Minimizes `f` over the values yielded by `params`, returning the minimizing argument and value.
///
/// This assumes `f` is unimodal (quasi-convex) over the iteration order: non-increasing up to the
/// minimum and non-decreasing afterwards. It scans in order and stops as soon as `f` strictly
/// increases. On ties it keeps the later argument. Returns `None` if `params` is empty.
fn min_concave<A: Copy, B: Ord>(
	mut params: impl Iterator<Item = A>,
	f: impl Fn(A) -> B,
) -> Option<(A, B)> {
	let mut min_a = params.next()?;
	let mut min_b = f(min_a);
	for a in params {
		let b = f(a);
		if b <= min_b {
			min_a = a;
			min_b = b;
		} else {
			// The function f is concave in the sequence of params, so break if it begins
			// increasing.
			break;
		}
	}
	Some((min_a, min_b))
}

/// Strategy that minimizes proof size using dynamic programming.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinProofSizeStrategy;

impl AritySelectionStrategy for MinProofSizeStrategy {
	fn choose_arities<F, MerkleScheme>(
		&self,
		merkle_scheme: &MerkleScheme,
		log_msg_len: usize,
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> Vec<usize>
	where
		F: Field,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		let optimizer = ReductionOptimizer::<F, _>::new(merkle_scheme, n_test_queries);
		let min_sizes = optimizer.compute_optimal_arities(log_msg_len, log_inv_rate);
		optimizer.optimizer_entries_to_fold_arities(&min_sizes)
	}
}

/// Strategy that uses a constant fold arity.
#[derive(Debug, Clone, Copy)]
pub struct ConstantArityStrategy {
	/// The fold arity to use for each reduction step.
	pub arity: usize,
}

impl ConstantArityStrategy {
	/// Creates a new strategy with the given arity.
	pub fn new(arity: usize) -> Self {
		Self { arity }
	}

	/// Creates a strategy with an estimated optimal arity.
	///
	/// Uses a heuristic to estimate the optimal FRI folding arity that minimizes proof size.
	///
	/// ## Arguments
	///
	/// * `_merkle_scheme` - the Merkle tree scheme (used to infer digest size)
	/// * `approx_log_code_len` - approximate log2 of the codeword length
	pub fn with_optimal_arity<F, MerkleScheme>(
		_merkle_scheme: &MerkleScheme,
		approx_log_code_len: usize,
	) -> Self
	where
		F: Field,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		let digest_size = std::mem::size_of::<MerkleScheme::Digest>() * 8;
		let field_size = std::mem::size_of::<F>() * 8;

		// Estimate optimal arity using a heuristic based on the approximation of a single
		// query_proof_size, where θ is the arity:
		// ((n-θ) + (n-2θ) + ...) * digest_size + ((n-θ)/θ) * 2^θ * field_size
		let arity = (1..=approx_log_code_len)
			.map(|arity| {
				(
					arity,
					((approx_log_code_len) / 2 * digest_size + (1 << arity) * field_size)
						* (approx_log_code_len - arity)
						/ arity,
				)
			})
			// Scan and terminate when query_proof_size increases.
			.scan(None, |old: &mut Option<(usize, usize)>, new| {
				let should_continue = !matches!(*old, Some(ref old) if new.1 > old.1);
				*old = Some(new);
				should_continue.then_some(new)
			})
			.last()
			.map(|(arity, _)| arity)
			.unwrap_or(1);

		Self { arity }
	}
}

impl AritySelectionStrategy for ConstantArityStrategy {
	fn choose_arities<F, MerkleScheme>(
		&self,
		merkle_scheme: &MerkleScheme,
		log_msg_len: usize,
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> Vec<usize>
	where
		F: Field,
		MerkleScheme: MerkleTreeScheme<F>,
	{
		let log_code_len = log_msg_len + log_inv_rate;
		let cap_height = merkle_scheme.optimal_verify_layer(n_test_queries, log_code_len);
		let log_terminal_len = cap_height.max(log_inv_rate);

		let mut fold_arities = Vec::new();
		let mut i = log_code_len;
		while i > log_terminal_len {
			if let Some(next_i) = i.checked_sub(self.arity) {
				fold_arities.push(self.arity);
				i = next_i;
			} else {
				break;
			}
		}
		fold_arities
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField128bGhash as B128;
	use binius_hash::StdHashSuite;
	use binius_math::ntt::{
		AdditiveNTT, NeighborsLastReference, domain_context::GaoMateerOnTheFly,
	};

	use super::*;
	use crate::merkle_tree::BinaryMerkleTreeScheme;

	type TestMerkleScheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

	fn test_merkle_scheme() -> TestMerkleScheme {
		BinaryMerkleTreeScheme::new()
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
	fn test_min_proof_size_strategy() {
		let merkle_scheme = test_merkle_scheme();
		let log_inv_rate = 2;
		let n_test_queries = 128;
		let strategy = MinProofSizeStrategy;

		// log_msg_len = 0: no folding needed, terminal codeword is optimal
		let arities =
			strategy.choose_arities::<B128, _>(&merkle_scheme, 0, log_inv_rate, n_test_queries);
		assert_eq!(arities, vec![]);

		// log_msg_len = 3: no folding needed, terminal codeword is optimal
		let arities =
			strategy.choose_arities::<B128, _>(&merkle_scheme, 3, log_inv_rate, n_test_queries);
		assert_eq!(arities, vec![]);

		// log_msg_len = 24
		let arities =
			strategy.choose_arities::<B128, _>(&merkle_scheme, 24, log_inv_rate, n_test_queries);
		assert_eq!(arities, vec![4, 4, 4, 4]);
	}

	#[test]
	fn test_with_strategy_min_proof_size() {
		let merkle_scheme = test_merkle_scheme();
		let log_inv_rate = 2;
		let n_test_queries = 128;

		let ntt = NeighborsLastReference {
			domain_context: GaoMateerOnTheFly::<B128>::generate(24 + log_inv_rate),
		};

		// log_msg_len = 0
		{
			let fri_params = FRIParams::with_strategy(
				ntt.domain_context(),
				&merkle_scheme,
				0,
				None,
				log_inv_rate,
				n_test_queries,
				&MinProofSizeStrategy,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[]);
			assert_eq!(fri_params.log_batch_size(), 0);
		}

		// log_msg_len = 3
		{
			let fri_params = FRIParams::with_strategy(
				ntt.domain_context(),
				&merkle_scheme,
				3,
				None,
				log_inv_rate,
				n_test_queries,
				&MinProofSizeStrategy,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[]);
			assert_eq!(fri_params.log_batch_size(), 3);
		}

		// log_msg_len = 24
		{
			let fri_params = FRIParams::with_strategy(
				ntt.domain_context(),
				&merkle_scheme,
				24,
				None,
				log_inv_rate,
				n_test_queries,
				&MinProofSizeStrategy,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[4, 4, 4]);
			assert_eq!(fri_params.log_batch_size(), 4);
		}
	}

	#[test]
	fn test_optimal_for_batch_three_oracles() {
		let merkle_scheme = test_merkle_scheme();
		let log_inv_rate = 2;
		let n_test_queries = 128;

		let ntt = NeighborsLastReference {
			domain_context: GaoMateerOnTheFly::<B128>::generate(16 + log_inv_rate),
		};

		let oracles = vec![
			PartialOracleSpec {
				log_msg_len: 10,
				log_batch_size: Some(1),
			},
			PartialOracleSpec {
				log_msg_len: 12,
				log_batch_size: Some(1),
			},
			PartialOracleSpec {
				log_msg_len: 16,
				log_batch_size: None,
			},
		];

		let (fri_params, proof_size) = FRIParams::optimal_for_batch(
			ntt.domain_context(),
			&merkle_scheme,
			&oracles,
			log_inv_rate,
			n_test_queries,
		);

		// The reduced oracle dimension is the dimension of the first FRI round oracle, equal to
		// log_terminal_dim + sum(fold_arities).
		let reduced_log_msg_len = fri_params.rs_code().log_dim();
		assert_eq!(
			reduced_log_msg_len,
			fri_params.log_terminal_dim + fri_params.fold_arities().iter().sum::<usize>()
		);

		// Each input oracle satisfies the FRIParams invariants.
		assert_eq!(fri_params.input_oracles.len(), oracles.len());
		for (spec, partial) in fri_params.input_oracles.iter().zip(&oracles) {
			assert_eq!(spec.log_msg_len, partial.log_msg_len);
			if let Some(log_batch_size) = partial.log_batch_size {
				// Fixed batch sizes are preserved.
				assert_eq!(spec.log_batch_size, log_batch_size);
			}
			// log_batch_size <= log_msg_len
			assert!(spec.log_batch_size <= spec.log_msg_len);
			// log_msg_len <= log_terminal_dim + sum(fold_arities) + log_batch_size
			assert!(spec.log_msg_len <= reduced_log_msg_len + spec.log_batch_size);
		}

		// The largest input oracle is flexible, so its batch size is folded down exactly to the
		// reduced dimension (no lifting).
		assert_eq!(fri_params.input_oracles[2].log_batch_size, 16 - reduced_log_msg_len);

		// Pin the estimated proof size to detect unintended changes in the optimizer.
		assert_eq!(proof_size, 229376);
	}

	#[test]
	fn test_with_strategy_fixed_batch_size() {
		let merkle_scheme = test_merkle_scheme();
		let log_inv_rate = 2;
		let n_test_queries = 128;

		let ntt = NeighborsLastReference {
			domain_context: GaoMateerOnTheFly::<B128>::generate(24 + log_inv_rate),
		};

		// log_msg_len = 3
		{
			let fri_params = FRIParams::with_strategy(
				ntt.domain_context(),
				&merkle_scheme,
				3,
				Some(1),
				log_inv_rate,
				n_test_queries,
				&MinProofSizeStrategy,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[]);
			assert_eq!(fri_params.log_batch_size(), 1);
		}

		// log_msg_len = 24
		{
			let fri_params = FRIParams::with_strategy(
				ntt.domain_context(),
				&merkle_scheme,
				24,
				Some(1),
				log_inv_rate,
				n_test_queries,
				&MinProofSizeStrategy,
			)
			.unwrap();
			assert_eq!(fri_params.fold_arities(), &[4, 4, 4, 3]);
			assert_eq!(fri_params.log_batch_size(), 1);
		}
	}
}
