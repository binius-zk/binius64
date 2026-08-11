// Copyright 2026 The Binius Developers

use std::iter;

use binius_field::{BinaryField, FieldOps, util::expand_subset_sums};
use binius_ip::channel::WordIPVerifierChannel;
use binius_math::{line::extrapolate_line, multilinear, ntt::DomainContext};

use crate::merkle_channel::MerkleIPVerifierChannel;
/// A virtual oracle for a code proximity test.
///
/// The interactive code proximity tests used in this project (eg. FRI) commit to a codeword and
/// then interactively fold it with random challenges. This trait represents the resulting *virtual
/// oracle*: the folded codeword, whose values are not committed directly but are instead recovered
/// on demand by opening the committed oracle at the queried indices and applying the folding. An
/// implementation therefore holds a handle to the committed oracle along with the folding
/// challenges, and receives openings over a Merkle channel in order to evaluate the virtual oracle
/// at queried locations.
///
/// The oracle is parameterized by the element type `E` of the channel it is opened over, since the
/// folding challenges it holds were sampled from that channel.
pub trait ProxTestOracle<F: BinaryField, E> {
	/// The Merkle commitment handle for the committed oracle.
	type Commitment;

	/// Opens queried locations on the virtual oracle.
	///
	/// This has a batch interface for verifying multiple queries because opening multiple Merkle
	/// tree locations at once amortizes the proof size.
	///
	/// ## Preconditions
	/// The `indices` must address the virtual oracle, which spans the committed tree depth
	/// duplicated over the lift factor. The channel guarantees the bound by masking what
	/// [`WordIPVerifierChannel::sample_bits`] returns.
	///
	/// ## Returns
	/// The values of the virtual oracle at the queried indices. The virtual oracle is defined by
	/// the committed oracle and the folding challenges.
	fn open_queries<Channel>(
		&self,
		indices: &[Channel::Word],
		channel: &mut Channel,
	) -> Result<Vec<E>, Error>
	where
		Channel: MerkleIPVerifierChannel<F, Commitment = Self::Commitment, Elem = E>;
}

/// A [ProxTestOracle] implementation for a [Brakedown]-style interleaved code proximity check.
///
/// [Brakedown]: <https://dl.acm.org/doi/10.1007/978-3-031-38545-2_7>
pub struct BrakedownOracle<E, C> {
	challenges: Vec<E>,
	commitment: C,
	/// log2 the lift factor (oracle padding). The committed codeword is virtually duplicated
	/// `2^log_lift` times to reach the common first-round length; a query at global index `k`
	/// reads the committed codeword at `k >> log_lift`. Zero when no lifting is needed.
	log_lift: usize,
}

impl<E, C> BrakedownOracle<E, C> {
	/// Constructs a new oracle from the committed interleaved codeword and the folding challenges.
	///
	/// `log_lift` is the oracle-padding lift factor (the committed codeword is virtually
	/// duplicated `2^log_lift` times to reach the common first-round length); pass `0` when no
	/// lifting is needed.
	pub const fn new(challenges: Vec<E>, commitment: C, log_lift: usize) -> Self {
		Self {
			challenges,
			commitment,
			log_lift,
		}
	}
}

impl<F: BinaryField, E: FieldOps<Scalar = F>, C> ProxTestOracle<F, E> for BrakedownOracle<E, C> {
	type Commitment = C;

	fn open_queries<Channel>(
		&self,
		indices: &[Channel::Word],
		channel: &mut Channel,
	) -> Result<Vec<E>, Error>
	where
		Channel: MerkleIPVerifierChannel<F, Commitment = C, Elem = E>,
	{
		// Translate each query on the virtual lifted oracle into a query on the committed codeword
		// by dropping the low `log_lift` bits (the duplicated copies).
		let lifted_indices = indices
			.iter()
			.map(|index| index.clone() >> self.log_lift as u32)
			.collect::<Vec<_>>();
		let values = channel.recv_openings(&self.commitment, &lifted_indices)?;
		Ok(values
			.chunks(1 << self.challenges.len())
			.map(|coset| {
				// Fold the coset using a multilinear tensor fold over the challenges.
				multilinear::evaluate::evaluate_inplace_scalars(coset.to_vec(), &self.challenges)
			})
			.collect())
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
pub struct BatchBrakedownOracle<E, C> {
	oracles: Vec<BrakedownOracle<E, C>>,
	outer_challenges: Vec<E>,
}

impl<E, C> BatchBrakedownOracle<E, C> {
	/// Constructs a batch oracle from the per-commitment oracles and the batching challenges.
	pub fn new(oracles: Vec<BrakedownOracle<E, C>>, outer_challenges: Vec<E>) -> Self {
		assert!(!oracles.is_empty()); // precondition
		Self {
			oracles,
			outer_challenges,
		}
	}
}

impl<F: BinaryField, E: FieldOps<Scalar = F>, C> ProxTestOracle<F, E>
	for BatchBrakedownOracle<E, C>
{
	type Commitment = C;

	fn open_queries<Channel>(
		&self,
		indices: &[Channel::Word],
		channel: &mut Channel,
	) -> Result<Vec<E>, Error>
	where
		Channel: MerkleIPVerifierChannel<F, Commitment = C, Elem = E>,
	{
		// Receive each bundled oracle's openings in commit order (matching the prover), then
		// combine across oracles by the outer-challenge tensor expansion:
		// combined[q] = \sum_i values_i[q] * outer_tensor[i].
		let outer_tensor = multilinear::eq::eq_ind_partial_eval_scalars(&self.outer_challenges);
		let mut combined = vec![E::zero(); indices.len()];
		for (oracle, scalar) in self.oracles.iter().zip(&outer_tensor) {
			let values = oracle.open_queries(indices, channel)?;
			for (acc, value) in combined.iter_mut().zip(values) {
				*acc += value * scalar;
			}
		}
		Ok(combined)
	}
}

/// A single FRI reduction: one committed oracle, and the fold of each opened coset.
///
/// Note that this is distinct from the full FRI query-phase verifier in the `verify` module. This
/// one only verifies the openings of a single committed oracle and folds each opened coset into a
/// single value using FRI folding.
///
/// Unlike a [`ProxTestOracle`], this reduces *claims* about the base codeword rather than opening
/// the virtual oracle outright — see [`Self::reduce_queries`].
pub struct FRIOracle<E, C, DC> {
	challenges: Vec<E>,
	commitment: C,
	/// The depth of the committed Merkle tree.
	depth: usize,
	domain_context: DC,
}

impl<E, C, DC> FRIOracle<E, C, DC> {
	/// Constructs a new oracle from a committed oracle, its folding challenges, and the domain
	/// context providing the FRI fold twiddles.
	///
	/// `depth` is the depth of the committed Merkle tree.
	pub const fn new(challenges: Vec<E>, commitment: C, depth: usize, domain_context: DC) -> Self {
		Self {
			challenges,
			commitment,
			depth,
			domain_context,
		}
	}

	/// The base-2 log of the size of each coset opened from the committed oracle.
	const fn coset_log_size(&self) -> usize {
		self.challenges.len()
	}
}

impl<F, E, C, DC> FRIOracle<E, C, DC>
where
	F: BinaryField,
	E: FieldOps<Scalar = F> + From<F>,
	DC: DomainContext<Field = F>,
{
	/// Folds an opened coset into a single value.
	///
	/// The committed oracle's codeword length in log terms is the Merkle tree depth plus the number
	/// of folding challenges (one coset per leaf), which is the `log_len` consumed by
	/// [`fold_coset`].
	fn fold_coset<Channel>(
		&self,
		chunk_index: &Channel::Word,
		values: Vec<E>,
		channel: &mut Channel,
	) -> E
	where
		Channel: WordIPVerifierChannel<F, Elem = E>,
	{
		fold_coset(
			&self.domain_context,
			self.depth + self.challenges.len(),
			chunk_index,
			&self.challenges,
			values,
			channel,
		)
	}

	/// Opens queried locations on the base codeword, reducing claims about it to the virtual
	/// oracle.
	///
	/// An index addresses the base codeword, and splits in two:
	///
	/// ```text
	///     high depth bits              the coset index into the committed oracle
	///     low coset_log_size() bits    the offset within that coset
	/// ```
	///
	/// Each query checks the opened coset at its offset against the matching claim, which is
	/// asserted over the channel. The coset then folds into the virtual oracle value.
	///
	/// ## Preconditions
	/// `claims` must have the same length as `indices`.
	///
	/// ## Returns
	/// The values of the virtual oracle at the queried coset indices.
	pub fn reduce_queries<Channel>(
		&self,
		indices: &[Channel::Word],
		claims: &[E],
		channel: &mut Channel,
	) -> Result<Vec<E>, Error>
	where
		Channel: MerkleIPVerifierChannel<F, Commitment = C, Elem = E>,
	{
		assert_eq!(indices.len(), claims.len()); // precondition

		let coset_log_size = self.coset_log_size();
		let coset_indices = indices
			.iter()
			.map(|index| index.clone() >> coset_log_size as u32)
			.collect::<Vec<_>>();

		let values = channel.recv_openings(&self.commitment, &coset_indices)?;
		iter::zip(values.chunks(1 << coset_log_size), iter::zip(&coset_indices, indices))
			.zip(claims)
			.map(|((coset, (coset_index, index)), claim)| {
				// Check the claimed base-codeword value against the opened coset. `select` reads
				// the offset from the low `coset_log_size` bits of the index, which is the
				// coset mask.
				let opened = channel.select(coset, index);
				channel.assert_zero(opened - claim.clone())?;
				Ok(self.fold_coset(coset_index, coset.to_vec(), channel))
			})
			.collect()
	}
}

/// Folds a coset of a codeword into a single value with the given folding challenges.
///
/// This implements the fold operation from Definition 4.6 of [DP24], reading twiddle factors from
/// the domain context. `log_len` is the base-2 log of the length of the codeword the coset belongs
/// to; the twiddle layer is absolute within the full NTT domain and decreases with each challenge.
///
/// `chunk_index` names the coset within that codeword and is a channel word, so it need not be
/// known when the protocol is described. A twiddle is read as a sum over the set bits of its block
/// index: `DomainContext::twiddle` is `F2`-linear in that argument and vanishes at zero, so the
/// twiddle of a block is the sum of the twiddles of the individual bits. The bits below the coset
/// offset width are known here, and the rest come from `chunk_index` via
/// [`WordIPVerifierChannel::subset_sum`].
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub fn fold_coset<F, E, DC, Channel>(
	domain_context: &DC,
	mut log_len: usize,
	chunk_index: &Channel::Word,
	challenges: &[E],
	mut values: Vec<E>,
	channel: &mut Channel,
) -> E
where
	F: BinaryField,
	E: FieldOps<Scalar = F> + From<F>,
	DC: DomainContext<Field = F>,
	Channel: WordIPVerifierChannel<F, Elem = E>,
{
	let mut log_size = challenges.len();
	for challenge in challenges {
		let layer = log_len - 1;
		// The twiddle of a block index is the sum of the basis elements its set bits select. That
		// basis is the layer's subspace above the normalizing element, which `twiddle` itself
		// indexes into.
		let layer_subspace = domain_context.subspace(layer + 1);
		let twiddle_basis = &layer_subspace.basis()[1..];

		// A block index is `(chunk_index << shift) | index_offset`, so its low `shift` bits are the
		// offset within the coset and the rest are the coset index.
		let shift = log_size - 1;
		let chunk_basis = twiddle_basis[shift..]
			.iter()
			.copied()
			.map(E::from)
			.collect::<Vec<_>>();
		let chunk_twiddle = channel.subset_sum(&chunk_basis, chunk_index);
		let offset_twiddles = expand_subset_sums(&twiddle_basis[..shift]);

		for index_offset in 0..1 << shift {
			let twiddle = chunk_twiddle.clone() + E::from(offset_twiddles[index_offset]);

			// Perform the inverse additive NTT butterfly, then extrapolate the resulting line at
			// the folding challenge.
			let mut u = values[index_offset << 1].clone();
			let v = values[(index_offset << 1) | 1].clone() + &u;
			u += v.clone() * twiddle;
			values[index_offset] = extrapolate_line(u, v, challenge.clone());
		}

		log_len -= 1;
		log_size -= 1;
	}

	values[0].clone()
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Merkle channel error: {0}")]
	Channel(#[from] crate::merkle_channel::Error),
	#[error("IP channel error: {0}")]
	IPChannel(#[from] binius_ip::channel::Error),
}
