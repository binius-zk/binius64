// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{BinaryField, ExtensionField};
use binius_math::{line::extrapolate_line, ntt::AdditiveNTT};

use super::{FRIParams, error::Error};
use crate::merkle_channel::MerkleIPVerifierChannel;

/// Calculate fold of `values` at `index` with `r` random coefficient.
///
/// See [DP24], Def. 3.6.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[inline]
fn fold_pair<F, FS, NTT>(ntt: &NTT, round: usize, index: usize, values: (F, F), r: F) -> F
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
	NTT: AdditiveNTT<Field = FS>,
{
	// Perform inverse additive NTT butterfly
	let t = ntt.twiddle(round - 1, index);
	let (mut u, mut v) = values;
	v += u;
	u += v * t;
	extrapolate_line(u, v, r)
}

/// Calculate FRI fold of `values` at a `chunk_index` with random folding challenges.
///
/// Folds a coset of a Reed–Solomon codeword into a single value using the FRI folding algorithm.
/// The coset has size $2^n$, where $n$ is the number of challenges.
///
/// See [DP24], Def. 3.6 and Lemma 3.9 for more details.
///
/// NB: This method is on a hot path and does not perform any allocations or
/// precondition checks.
///
/// ## Arguments
///
/// * `math` - the NTT instance, used to look up the twiddle values.
/// * `log_len` - the binary logarithm of the code length.
/// * `chunk_index` - the index of the chunk, of size $2^n$, in the full codeword.
/// * `values` - mutable slice of values to fold, modified in place.
/// * `challenges` - the sequence of folding challenges, with length $n$.
///
/// ## Pre-conditions
///
/// - `challenges.len() <= log_len`.
/// - `log_len <= math.log_domain_size()`, so that the NTT domain is large enough.
/// - `values.len() == 1 << challenges.len()`.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[inline]
pub fn fold_chunk<F, FS, NTT>(
	ntt: &NTT,
	mut log_len: usize,
	chunk_index: usize,
	values: &mut [F],
	challenges: &[F],
) -> F
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
	NTT: AdditiveNTT<Field = FS>,
{
	let mut log_size = challenges.len();

	// Preconditions
	debug_assert!(log_size <= log_len);
	debug_assert!(log_len <= ntt.log_domain_size());
	debug_assert_eq!(values.len(), 1 << log_size);

	// FRI-fold the values in place.
	for &challenge in challenges {
		// Fold the (2i) and (2i+1)th cells of the scratch buffer in-place into the i-th cell
		for index_offset in 0..1 << (log_size - 1) {
			let pair = (values[index_offset << 1], values[(index_offset << 1) | 1]);
			values[index_offset] = fold_pair(
				ntt,
				log_len,
				(chunk_index << (log_size - 1)) | index_offset,
				pair,
				challenge,
			)
		}

		log_len -= 1;
		log_size -= 1;
	}

	values[0]
}

/// One commitment the prover is expected to send during the fold phase.
struct ExpectedCommitment {
	/// The fold round this commitment arrives in.
	round: usize,
	/// Number of field elements packed into a single leaf of the Merkle tree.
	leaf_size: usize,
	/// Base-2 logarithm of the number of leaves in the Merkle tree.
	depth: usize,
}

/// Receives the prover's fold-phase commitments in the order the protocol sends them.
///
/// The fold phase spends one round per variable of the committed message.
/// Most rounds only absorb a folding challenge and carry no message from the prover.
///
/// A round that completes a fold shrinks the codeword.
/// The prover commits to that new codeword, and this type reads it off the channel.
pub struct FRIFoldVerifier<'a, F, C>
where
	F: BinaryField,
{
	/// Every commitment the fold phase expects, ordered by the round it arrives in.
	commit_schedule: Vec<ExpectedCommitment>,
	/// Commitments read off the channel so far, in arrival order.
	round_commitments: Vec<C>,
	/// The round about to be processed, counted from zero.
	curr_round: usize,
	/// Number of rounds the fold phase runs.
	n_rounds: usize,
	_phantom: std::marker::PhantomData<&'a F>,
}

impl<'a, F, C> FRIFoldVerifier<'a, F, C>
where
	F: BinaryField,
	C: Clone,
{
	/// Derives the schedule of expected commitments from the protocol parameters.
	pub fn new(params: &'a FRIParams<F>) -> Self {
		// One commitment per codeword the fold phase sends: batched, one per fold, then terminal.
		let expected_oracles = params.n_oracles();

		// A fold absorbs one challenge per round, then commits to the codeword it produced.
		// So commitments land on the running total of the fold widths, never in between.
		//
		// Concrete run: batch width 4, folds of arity 4 then 3, over a 2^18-symbol codeword.
		//
		//     rounds 0..3   batch fold        commit at 4    leaf 2^4   2^14 leaves
		//     rounds 4..7   fold of arity 4   commit at 8    leaf 2^3   2^11 leaves
		//     rounds 8..10  fold of arity 3   commit at 11   leaf 2^9   2^2  leaves
		//
		// A leaf holds the whole coset that the *next* fold reads.
		// So each tree is shorter than its own codeword by that fold's arity.
		let mut commit_schedule = Vec::with_capacity(expected_oracles);

		// The opening fold absorbs the batching challenges.
		// So nothing is committed until it ends.
		let mut round = params.log_batch_size();

		// Ahead of any fold, the tree holds one leaf per codeword symbol.
		let mut depth = params.index_bits();

		for &arity in params.fold_arities() {
			// This fold reads cosets of its own arity.
			// So packing one per leaf costs the tree that many levels.
			depth -= arity;
			commit_schedule.push(ExpectedCommitment {
				round,
				leaf_size: 1 << arity,
				depth,
			});

			// The next commitment waits for this fold to absorb every one of its challenges.
			round += arity;
		}

		// The terminal codeword is sent in full rather than folded again.
		// So its leaf spans a whole message, not a coset, leaving one leaf per rate position.
		commit_schedule.push(ExpectedCommitment {
			round,
			leaf_size: 1 << params.n_final_challenges(),
			depth: params.rs_code().log_inv_rate(),
		});

		Self {
			commit_schedule,
			round_commitments: Vec::with_capacity(expected_oracles),
			curr_round: 0,
			// The opening round precedes the first challenge.
			// So the phase runs one round longer than it has challenges.
			n_rounds: params.n_fold_rounds() + 1,
			_phantom: std::marker::PhantomData,
		}
	}

	/// The next commitment still awaited, or nothing once every one of them has arrived.
	fn next_commitment(&self) -> Option<&ExpectedCommitment> {
		// Commitments arrive in schedule order.
		// So the number received so far is the read cursor.
		self.commit_schedule.get(self.round_commitments.len())
	}

	/// Advances one fold round, reading a commitment off the channel if this round carries one.
	///
	/// # Returns
	///
	/// The commitment received, or nothing when this round carries none.
	///
	/// # Panics
	///
	/// Panics once the fold phase has already run all of its rounds.
	pub fn process_round<Channel>(&mut self, channel: &mut Channel) -> Result<Option<C>, Error>
	where
		Channel: MerkleIPVerifierChannel<F, Commitment = C>,
	{
		// The fold phase has a length fixed by the parameters.
		// Stepping past it would read a message the prover never sent.
		assert!(
			self.curr_round < self.n_rounds(),
			"precondition: process_round must not be called more than n_rounds() times"
		);

		// Only the round named by the next scheduled commitment carries one.
		let commitment = match self.next_commitment() {
			Some(&ExpectedCommitment {
				round,
				leaf_size,
				depth,
			}) if round == self.curr_round => {
				// Demanding the scheduled shape pins the prover to the prescribed layout.
				// A mis-shaped tree is rejected here rather than at query time.
				let commitment = channel.recv_merkle_commitment(leaf_size, depth)?;
				self.round_commitments.push(commitment.clone());
				Some(commitment)
			}
			_ => None,
		};

		self.curr_round += 1;
		Ok(commitment)
	}

	/// Reports whether the round about to be processed carries a commitment.
	pub fn needs_commitment(&self) -> bool {
		// A commitment is due only when the schedule names this exact round.
		self.next_commitment()
			.is_some_and(|expected| expected.round == self.curr_round)
	}

	/// Reports whether every fold round has been processed.
	pub const fn is_complete(&self) -> bool {
		self.curr_round == self.n_rounds()
	}

	/// Consumes the verifier and yields the commitments it collected, in arrival order.
	///
	/// # Panics
	///
	/// Panics unless every fold round has been processed.
	/// Stopping early would drop commitments the query phase needs to open.
	pub fn finalize(self) -> Vec<C> {
		// Every scheduled commitment lands on some round, so an unfinished phase is a short list.
		assert!(
			self.is_complete(),
			"precondition: all fold rounds must be processed before finalize"
		);

		self.round_commitments
	}

	/// The round about to be processed, counted from zero.
	pub const fn current_round(&self) -> usize {
		self.curr_round
	}

	/// Number of rounds the fold phase runs.
	pub const fn n_rounds(&self) -> usize {
		self.n_rounds
	}
}
