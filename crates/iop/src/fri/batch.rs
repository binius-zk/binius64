// Copyright 2026 The Binius Developers

use binius_field::BinaryField;
use binius_math::{line::extrapolate_line, multilinear, ntt::DomainContext};
use binius_transcript::TranscriptReader;
use binius_utils::DeserializeBytes;
use bytes::Buf;

use crate::merkle_tree::{Commitment, MerkleTreeScheme};
/// A virtual oracle for a code proximity test.
///
/// The interactive code proximity tests used in this project (eg. FRI) commit to a codeword and
/// then interactively fold it with random challenges. This trait represents the resulting *virtual
/// oracle*: the folded codeword, whose values are not committed directly but are instead recovered
/// on demand by opening the committed oracle at the queried indices and applying the folding. An
/// implementation therefore holds the committed oracle along with the folding challenges, and
/// verifies decommitted prover advice in order to evaluate the virtual oracle at queried locations.
pub trait ProxTestOracle<F: BinaryField> {
	/// The base-2 logarithm of the length of the virtual oracle.
	///
	/// The virtual oracle is defined by the committed oracle and the folding challenges. Indices
	/// passed to [`Self::open_queries`] must lie in the range `0..2^self.log_len()`.
	fn log_len(&self) -> usize;

	/// Opens queried locations on the virtual oracle.
	///
	/// This has a batch interface for verifying multiple queries because opening multiple Merkle
	/// tree locations at once amortizes the proof size.
	///
	/// ## Preconditions
	/// The `indices` must lie in the range `0..2^self.log_len()`.
	///
	/// ## Returns
	/// The values of the virtual oracle at the queried indices. The virtual oracle is defined by
	/// the committed oracle and the folding challenges.
	fn open_queries<B: Buf>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptReader<B>,
	) -> Result<Vec<F>, Error>;
}

/// A [ProxTestOracle] implementation for a [Brakedown]-style interleaved code proximity check.
///
/// [Brakedown]: <https://dl.acm.org/doi/10.1007/978-3-031-38545-2_7>
pub struct BrakedownOracle<F, MTScheme>
where
	MTScheme: MerkleTreeScheme<F>,
{
	challenges: Vec<F>,
	commitment: Commitment<MTScheme::Digest>,
	merkle_scheme: MTScheme,
	/// log2 the lift factor (oracle padding). The committed codeword is virtually duplicated
	/// `2^log_lift` times to reach the common first-round length; a query at global index `k`
	/// reads the committed codeword at `k >> log_lift`. Zero when no lifting is needed.
	log_lift: usize,
}

impl<F: BinaryField, MTScheme: MerkleTreeScheme<F>> BrakedownOracle<F, MTScheme> {
	/// Constructs a new oracle from the committed interleaved codeword and the folding challenges.
	///
	/// `log_lift` is the oracle-padding lift factor (the committed codeword is virtually duplicated
	/// `2^log_lift` times to reach the common first-round length); pass `0` when no lifting is
	/// needed.
	pub fn new(
		challenges: Vec<F>,
		commitment: Commitment<MTScheme::Digest>,
		merkle_scheme: MTScheme,
		log_lift: usize,
	) -> Self {
		Self {
			challenges,
			commitment,
			merkle_scheme,
			log_lift,
		}
	}
}

impl<F: BinaryField, MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>> ProxTestOracle<F>
	for BrakedownOracle<F, MTScheme>
{
	fn log_len(&self) -> usize {
		// The virtual (lifted) oracle length: the committed tree depth duplicated `2^log_lift`
		// times.
		self.commitment.depth + self.log_lift
	}

	fn open_queries<B: Buf>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptReader<B>,
	) -> Result<Vec<F>, Error> {
		assert!(indices.iter().all(|&index| index < 1 << self.log_len())); // precondition
		// Translate each query on the virtual lifted oracle into a query on the committed codeword
		// by dropping the low `log_lift` bits (the duplicated copies).
		let lifted_indices = indices
			.iter()
			.map(|&index| index >> self.log_lift)
			.collect::<Vec<_>>();
		verify_query_openings(
			&self.merkle_scheme,
			&self.commitment,
			self.challenges.len(),
			&lifted_indices,
			advice,
		)?
		.map(|opening| {
			let (_index, values) = opening?;
			// Fold the coset using a multilinear tensor fold over the challenges.
			Ok(multilinear::evaluate::evaluate_inplace_scalars(values, &self.challenges))
		})
		.collect()
	}
}

/// A [ProxTestOracle] bundling several separately committed [BrakedownOracle]s.
///
/// The bundled oracles wrap interleaved codewords of equal folded length that the prover batched
/// into a single folded codeword via the outer-challenge tensor expansion. Their query openings are
/// read sequentially, one oracle's full decommitment after another, and the per-query folded values
/// are combined as `\sum_i values_i[q] * outer_tensor[i]`, where
/// `outer_tensor = eq_ind_partial_eval(outer_challenges)`. This mirrors the prover's
/// `BatchBrakedownFolder::fold`.
pub struct BatchBrakedownOracle<F, MTScheme>
where
	MTScheme: MerkleTreeScheme<F>,
{
	oracles: Vec<BrakedownOracle<F, MTScheme>>,
	outer_challenges: Vec<F>,
}

impl<F: BinaryField, MTScheme: MerkleTreeScheme<F>> BatchBrakedownOracle<F, MTScheme> {
	/// Constructs a batch oracle from the per-commitment oracles and the batching challenges.
	pub fn new(oracles: Vec<BrakedownOracle<F, MTScheme>>, outer_challenges: Vec<F>) -> Self {
		assert!(!oracles.is_empty()); // precondition
		Self {
			oracles,
			outer_challenges,
		}
	}
}

impl<F: BinaryField, MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>> ProxTestOracle<F>
	for BatchBrakedownOracle<F, MTScheme>
{
	fn log_len(&self) -> usize {
		// Every bundled oracle reports the same virtual (lifted) length: the common first-round
		// codeword length they are batched into.
		debug_assert!(
			self.oracles
				.iter()
				.all(|oracle| oracle.log_len() == self.oracles[0].log_len())
		);
		self.oracles[0].log_len()
	}

	fn open_queries<B: Buf>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptReader<B>,
	) -> Result<Vec<F>, Error> {
		// Read each bundled oracle's openings in commit order (matching the prover), then combine
		// across oracles by the outer-challenge tensor expansion:
		// combined[q] = \sum_i values_i[q] * outer_tensor[i].
		let outer_tensor = multilinear::eq::eq_ind_partial_eval::<F>(&self.outer_challenges);
		let mut combined = vec![F::ZERO; indices.len()];
		for (oracle, &scalar) in self.oracles.iter().zip(outer_tensor.as_ref()) {
			let values = oracle.open_queries(indices, advice)?;
			for (acc, value) in combined.iter_mut().zip(values) {
				*acc += value * scalar;
			}
		}
		Ok(combined)
	}
}

/// A [ProxTestOracle] implementation for a FRI-style code proximity check.
///
/// Note that this is distinct from the full FRI query-phase verifier in the `verify` module. This
/// one only verifies the openings of a single committed oracle and folds each opened coset into a
/// single value using FRI folding.
pub struct FRIOracle<F, MTScheme, DC>
where
	MTScheme: MerkleTreeScheme<F>,
	DC: DomainContext<Field = F>,
{
	challenges: Vec<F>,
	commitment: Commitment<MTScheme::Digest>,
	merkle_scheme: MTScheme,
	domain_context: DC,
}

impl<F, MTScheme, DC> FRIOracle<F, MTScheme, DC>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F>,
	DC: DomainContext<Field = F>,
{
	/// Constructs a new oracle from a committed oracle, its folding challenges, and the domain
	/// context providing the FRI fold twiddles.
	pub fn new(
		challenges: Vec<F>,
		commitment: Commitment<MTScheme::Digest>,
		merkle_scheme: MTScheme,
		domain_context: DC,
	) -> Self {
		Self {
			challenges,
			commitment,
			merkle_scheme,
			domain_context,
		}
	}

	/// The base-2 log of the size of each coset opened from the committed oracle.
	fn coset_log_size(&self) -> usize {
		self.challenges.len()
	}

	/// Folds an opened coset into a single value.
	///
	/// The committed oracle's codeword length in log terms is the Merkle tree depth plus the number
	/// of folding challenges (one coset per leaf), which is the `log_len` consumed by
	/// [`fold_coset`].
	fn fold_coset(&self, chunk_index: usize, values: Vec<F>) -> F {
		fold_coset(
			&self.domain_context,
			self.commitment.depth + self.challenges.len(),
			chunk_index,
			&self.challenges,
			values,
		)
	}
}

/// Folds a coset of a codeword into a single value with the given folding challenges.
///
/// This implements the fold operation from Definition 4.6 of [DP24], reading twiddle factors from
/// the domain context. `log_len` is the base-2 log of the length of the codeword the coset belongs
/// to; the twiddle layer is absolute within the full NTT domain and decreases with each challenge.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub fn fold_coset<F, DC>(
	domain_context: &DC,
	mut log_len: usize,
	chunk_index: usize,
	challenges: &[F],
	mut values: Vec<F>,
) -> F
where
	F: BinaryField,
	DC: DomainContext<Field = F>,
{
	let mut log_size = challenges.len();
	for &challenge in challenges {
		for index_offset in 0..1 << (log_size - 1) {
			// Perform the inverse additive NTT butterfly, then extrapolate the resulting line at
			// the folding challenge.
			let mut u = values[index_offset << 1];
			let mut v = values[(index_offset << 1) | 1];
			let twiddle =
				domain_context.twiddle(log_len - 1, (chunk_index << (log_size - 1)) | index_offset);
			v += u;
			u += v * twiddle;
			values[index_offset] = extrapolate_line(u, v, challenge);
		}

		log_len -= 1;
		log_size -= 1;
	}

	values[0]
}

impl<F, MTScheme, DC> FRIOracle<F, MTScheme, DC>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	DC: DomainContext<Field = F>,
{
	/// Opens queried locations on the base codeword, reducing claims about it to the virtual
	/// oracle.
	///
	/// Whereas [`Self::open_queries`] indexes the virtual oracle, this indexes the base codeword:
	/// each index lies in the range `0..2^(self.log_len() + self.coset_log_size())` and splits into
	/// the high `self.log_len()` bits (the coset index into the committed oracle) and the low
	/// `self.coset_log_size()` bits (the offset within the coset). For each query, the opened coset
	/// value at that offset is checked against `claims[i]`, after which the coset is folded into
	/// the virtual oracle value as in [`Self::open_queries`].
	///
	/// ## Preconditions
	/// `claims` must have the same length as `indices`, and the indices must lie in the range
	/// `0..2^(self.log_len() + self.coset_log_size())`.
	///
	/// ## Returns
	/// The values of the virtual oracle at the queried coset indices.
	pub fn reduce_queries<B: Buf>(
		&self,
		indices: &[usize],
		claims: &[F],
		advice: &mut TranscriptReader<B>,
	) -> Result<Vec<F>, Error> {
		assert_eq!(indices.len(), claims.len()); // precondition
		assert!(
			indices
				.iter()
				.all(|&index| index < 1 << (self.log_len() + self.coset_log_size()))
		); // precondition

		let coset_log_size = self.coset_log_size();
		let coset_mask = (1 << coset_log_size) - 1;
		let coset_indices = indices
			.iter()
			.map(|&index| index >> coset_log_size)
			.collect::<Vec<_>>();

		verify_query_openings(
			&self.merkle_scheme,
			&self.commitment,
			coset_log_size,
			&coset_indices,
			advice,
		)?
		.zip(indices.iter().zip(claims))
		.map(|(opening, (&index, &claim))| {
			let (coset_index, values) = opening?;
			// Verify the claimed base-codeword value against the opened coset.
			if values[index & coset_mask] != claim {
				return Err(Error::ClaimMismatch { index });
			}
			Ok(self.fold_coset(coset_index, values))
		})
		.collect()
	}
}

impl<F, MTScheme, DC> ProxTestOracle<F> for FRIOracle<F, MTScheme, DC>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	DC: DomainContext<Field = F>,
{
	fn log_len(&self) -> usize {
		self.commitment.depth
	}

	fn open_queries<B: Buf>(
		&self,
		indices: &[usize],
		advice: &mut TranscriptReader<B>,
	) -> Result<Vec<F>, Error> {
		assert!(indices.iter().all(|&index| index < 1 << self.log_len())); // precondition
		verify_query_openings(
			&self.merkle_scheme,
			&self.commitment,
			self.challenges.len(),
			indices,
			advice,
		)?
		.map(|opening| {
			let (index, values) = opening?;
			Ok(self.fold_coset(index, values))
		})
		.collect()
	}
}

/// Verifies the Merkle openings shared by the [ProxTestOracle] implementations.
///
/// First decommits and verifies the optimal internal layer of the Merkle tree, then returns a lazy
/// iterator that, for each queried index, reads the opened coset of `1 << coset_log_size` values
/// from the advice and verifies its opening against that layer. Each item is the queried index
/// paired with the verified coset values.
fn verify_query_openings<'a, F, MTScheme, B>(
	merkle_scheme: &'a MTScheme,
	commitment: &'a Commitment<MTScheme::Digest>,
	coset_log_size: usize,
	indices: &'a [usize],
	advice: &'a mut TranscriptReader<B>,
) -> Result<impl Iterator<Item = Result<(usize, Vec<F>), Error>> + 'a, Error>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	B: Buf,
{
	let tree_depth = commitment.depth;
	let layer_depth = merkle_scheme.optimal_verify_layer(indices.len(), tree_depth);
	let layer_digests = advice.read_vec(1 << layer_depth)?;
	merkle_scheme.verify_layer(&commitment.root, layer_depth, &layer_digests)?;

	let openings = indices.iter().map(move |&index| {
		// Receive the codeword symbols of the coset at the index and verify their consistency with
		// the commitment.
		let values = advice.read_scalar_slice::<F>(1 << coset_log_size)?;
		merkle_scheme.verify_opening(
			index,
			&values,
			layer_depth,
			tree_depth,
			&layer_digests,
			advice,
		)?;
		Ok((index, values))
	});
	Ok(openings)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("claimed value at index {index} does not match the committed codeword")]
	ClaimMismatch { index: usize },
	#[error("Merkle tree error: {0}")]
	Merkle(#[from] crate::merkle_tree::Error),
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
}
