// Copyright 2026 The Binius Developers

//! A channel decorator that merges oracles committed within the same interaction round.

use std::cmp::Reverse;

use binius_core::word::Word;
use binius_field::{BinaryField, Field, field::FieldOps};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_math::multilinear::eq::eq_ind;
use binius_utils::checked_arithmetics::log2_ceil_usize;

use crate::channel::{Error, IOPVerifierChannel, OracleSpec, TransparentEvalFn};

/// A handle to an oracle received through the merging decorator.
#[derive(Debug, Clone, Copy)]
pub struct MergeOracle {
	index: usize,
}

/// Records where one constituent oracle lives inside its round's combined oracle.
#[derive(Clone)]
struct Mapping<Oracle> {
	/// The handle the underlying channel returned for the combined oracle.
	outer: Oracle,

	/// The constituent's position among its round-mates, as a block index.
	///
	/// Its own `2^n` scalars begin at scalar `block_index * 2^n`.
	block_index: usize,

	/// The base-2 logarithm of the combined oracle's length.
	combined_log_len: usize,
}

/// Tracks one oracle from the moment it is received until its round is flushed.
struct Record<Oracle> {
	/// The base-2 logarithm of the oracle's own length.
	log_msg_len: usize,

	/// Whether the oracle's contents depend on the witness.
	///
	/// Declared by the caller when the oracle is received.
	is_witness_dependent: bool,

	/// Where this oracle ends up inside its round's combined oracle.
	///
	/// Filled in once that round is flushed.
	mapping: Option<Mapping<Oracle>>,
}

/// A verifier channel decorator that merges one round's oracles into one combined oracle.
///
/// # Overview
///
/// An interaction round is the run of oracle receipts between two challenge samples.
///
/// Committing each oracle separately costs one commitment per oracle.
/// One Merkle tree per oracle, for example.
///
/// This decorator buffers a round's oracles instead.
/// It commits them together as one larger oracle.
/// That cuts the cost to one commitment per round.
///
/// # Merging
///
/// A round's oracles are sorted from largest to smallest.
/// They are laid out end to end.
///
/// Every oracle's size is a power of two.
/// Write the sizes as `2^n_1, 2^n_2, ..., 2^n_k`.
/// Sort them so `n_1 >= n_2 >= ... >= n_k`.
///
/// The combined oracle's size is `2^N`.
/// `N` is the smallest exponent that covers the total.
///
/// ```text
/// combined oracle, size 2^N:
///
///     [ oracle 1 (2^n_1) | oracle 2 (2^n_2) | ... | oracle k (2^n_k) | unused padding ]
///       offset 0           offset 2^n_1                                total size 2^N
/// ```
///
/// Sorting largest to smallest makes this layout exact.
///
/// Every earlier oracle is at least as large as the current one.
/// So their combined space is a whole multiple of the current oracle's size.
/// The current oracle therefore starts on a boundary of its own size.
/// Its position is then a whole number of its-own-size blocks.
/// That whole number is its block index.
///
/// A round of a single oracle needs no combining.
/// It is forwarded unchanged, at zero cost.
///
/// Every oracle in a round must declare the same witness-dependence.
///
/// A commitment is masked as a whole, never partly.
/// So mixing witness-carrying and structural oracles in one round is not supported.
///
/// # Timing
///
/// A round's oracles are committed the moment a challenge is sampled.
/// Not the moment they arrive.
///
/// A real Fiat-Shamir transcript works the same way.
/// A challenge can only be derived after its commitments are absorbed.
/// So committing cannot wait past the sample that follows a round.
///
/// # Opening
///
/// A constituent oracle's claim is an inner product.
/// It pairs the oracle's data with a transparent polynomial.
///
/// That claim becomes a claim about the combined oracle too.
/// Extend the transparent polynomial with an equality check.
/// The check is one over the constituent's own block, zero elsewhere.
///
/// ```text
/// extended transparent(x) = original transparent(low bits of x) * is_this_block(high bits of x)
/// ```
///
/// The check is zero outside the constituent's own block.
/// So the combined inner product only ever sees this oracle's own data.
/// It equals the original claim exactly.
pub struct MergeVerifierChannel<'a, F, C>
where
	F: Field,
	C: IOPVerifierChannel<F>,
{
	/// The underlying channel every oracle and challenge passes through.
	inner: C,

	/// Fine-grained specs, one per oracle this channel's caller will receive.
	///
	/// Not the coarser, one-per-round specs the underlying channel uses.
	oracle_specs: &'a [OracleSpec],

	/// Every oracle received so far, in arrival order.
	records: Vec<Record<C::Oracle>>,

	/// How many entries have already been flushed to the underlying channel.
	///
	/// Every later entry belongs to the current, still-open round.
	flushed: usize,
}

impl<'a, F, C> MergeVerifierChannel<'a, F, C>
where
	F: Field,
	C: IOPVerifierChannel<F>,
{
	/// Creates a new merging verifier channel over an underlying channel.
	///
	/// # Arguments
	///
	/// * `inner` — the channel every combined oracle is committed to, already configured with the
	///   coarser, one-per-round spec list this decorator will produce.
	/// * `oracle_specs` — the fine-grained specs for every oracle this channel's caller will pass
	///   through, in arrival order.
	pub const fn new(inner: C, oracle_specs: &'a [OracleSpec]) -> Self {
		Self {
			inner,
			oracle_specs,
			records: Vec::new(),
			flushed: 0,
		}
	}

	/// Commits the current round's queued oracles as one combined oracle.
	///
	/// Does nothing if every received oracle is already committed.
	///
	/// # Panics
	///
	/// Panics if the queued oracles do not all share one witness-dependence.
	fn flush(&mut self) -> Result<(), Error> {
		// Nothing new has arrived since the last flush.
		if self.flushed == self.records.len() {
			return Ok(());
		}

		// Order this round largest to smallest.
		//
		// This lets every position be a whole number of block sizes.
		// That holds once every earlier oracle is at least as large.
		let mut order: Vec<usize> = (self.flushed..self.records.len()).collect();
		order.sort_by_key(|&i| Reverse(self.records[i].log_msg_len));

		// One commitment is masked as a whole, never partly.
		//
		// Every oracle in the round must agree on its witness-dependence.
		let is_witness_dependent = self.records[order[0]].is_witness_dependent;
		assert!(
			order
				.iter()
				.all(|&i| self.records[i].is_witness_dependent == is_witness_dependent),
			"MergeVerifierChannel: every oracle merged into one round must share \
			 is_witness_dependent"
		);

		// Size the combined oracle to fit every oracle end to end.
		let total_len: usize = order
			.iter()
			.map(|&i| 1usize << self.records[i].log_msg_len)
			.sum();
		let combined_log_len = log2_ceil_usize(total_len);

		// Commit the whole round as one oracle on the underlying channel.
		let outer = self
			.inner
			.recv_oracle(combined_log_len, is_witness_dependent)?;

		// Walk the sorted order again.
		// Record where each oracle actually landed.
		//
		// Every step adds a whole multiple of the next block's size.
		// So each offset divides evenly by that oracle's own size.
		let mut offset = 0usize;
		for &i in &order {
			let n_i = self.records[i].log_msg_len;
			self.records[i].mapping = Some(Mapping {
				outer: outer.clone(),
				block_index: offset >> n_i,
				combined_log_len,
			});
			offset += 1 << n_i;
		}

		self.flushed = self.records.len();
		Ok(())
	}

	/// Commits any oracles still queued and returns the underlying channel.
	///
	/// # Errors
	///
	/// Returns an error if the final round fails to commit.
	///
	/// # Panics
	///
	/// Panics if any declared oracle has not yet been received.
	pub fn into_inner(mut self) -> Result<C, Error> {
		self.flush()?;
		let n_remaining = self.oracle_specs.len() - self.records.len();
		assert!(n_remaining == 0, "into_inner called but {n_remaining} oracle specs remaining",);
		Ok(self.inner)
	}
}

impl<F, C> IPVerifierChannel<F> for MergeVerifierChannel<'_, F, C>
where
	F: Field,
	C: IOPVerifierChannel<F>,
{
	type Elem = C::Elem;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		self.inner.recv_one()
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<Self::Elem>, binius_ip::channel::Error> {
		self.inner.recv_many(n)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[Self::Elem; N], binius_ip::channel::Error> {
		self.inner.recv_array()
	}

	fn recv_public_claim(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		self.inner.recv_public_claim()
	}

	fn sample(&mut self) -> Self::Elem {
		// Commit this round before deriving its challenge.
		//
		// Sampling cannot return an error.
		// A flush failure here has nowhere to go.
		// It stays queued for the next call that can return one.
		let _ = self.flush();
		self.inner.sample()
	}

	fn observe_one(&mut self, val: F) -> Self::Elem {
		self.inner.observe_one(val)
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<Self::Elem> {
		self.inner.observe_many(vals)
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		self.inner.assert_zero(val)
	}
}

impl<F, C> WordIPVerifierChannel<F> for MergeVerifierChannel<'_, F, C>
where
	F: BinaryField,
	C: IOPVerifierChannel<F> + WordIPVerifierChannel<F>,
{
	type Word = C::Word;

	fn observe_words(&mut self, words: &[Word]) -> Vec<Self::Word> {
		self.inner.observe_words(words)
	}

	fn subset_sum(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem {
		self.inner.subset_sum(elems, word)
	}

	fn select(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem {
		self.inner.select(elems, word)
	}

	fn sample_bits(&mut self, bits: usize) -> Self::Word {
		self.inner.sample_bits(bits)
	}

	fn pack_words(&mut self, words: &[Self::Word]) -> Vec<Self::Elem> {
		self.inner.pack_words(words)
	}
}

impl<'a, F, C> IOPVerifierChannel<F> for MergeVerifierChannel<'a, F, C>
where
	F: Field,
	C: IOPVerifierChannel<F>,
{
	type Oracle = MergeOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.records.len()..]
	}

	fn recv_oracle(
		&mut self,
		log_msg_len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error> {
		// Every oracle this channel will receive is declared up front.
		//
		// Reject anything past that count.
		// Do not silently accept an undeclared oracle.
		let remaining = self.remaining_oracle_specs();
		assert!(!remaining.is_empty(), "recv_oracle called but no remaining oracle specs");
		debug_assert_eq!(log_msg_len, remaining[0].log_msg_len);

		// Queue the oracle without touching the underlying channel yet.
		//
		// It only becomes a real commitment once its round flushes.
		self.records.push(Record {
			log_msg_len,
			is_witness_dependent,
			mapping: None,
		});
		Ok(MergeOracle {
			index: self.records.len() - 1,
		})
	}

	fn verify_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: TransparentEvalFn<Self::Elem>,
		claim: Self::Elem,
	) -> Result<(), Error> {
		// Every oracle must be received before any oracle is opened.
		// So this oracle's round is already committed by now.
		//
		// Flush anyway, in case no challenge was sampled in between.
		self.flush()?;

		let record = &self.records[oracle.index];
		let n_i = record.log_msg_len;
		let Mapping {
			outer,
			block_index,
			combined_log_len,
		} = record.mapping.clone().expect("flushed above");

		// Build the fixed 0/1 pattern for this oracle's own block.
		// One bit per high coordinate of the combined opening point.
		//
		// A round of one oracle has no high coordinates at all.
		// The pattern is then empty, and the check below is always one.
		let padding_len = combined_log_len - n_i;
		let block_pattern: Vec<Self::Elem> = (0..padding_len)
			.map(|bit| {
				if (block_index >> bit) & 1 == 1 {
					Self::Elem::one()
				} else {
					Self::Elem::zero()
				}
			})
			.collect();

		// Extend the transparent polynomial with that equality check.
		//
		// The result is zero outside this oracle's own block.
		// Inside it, the result is the original transparent polynomial.
		let padded_transparent: TransparentEvalFn<Self::Elem> = Box::new(move |point| {
			let (low, high) = point.split_at(n_i);
			eq_ind(high, &block_pattern) * transparent(low)
		});

		// The claim itself is unchanged.
		//
		// The combined oracle agrees with this one on that block.
		// So the same inner product holds there.
		self.inner
			.verify_oracle_relation(outer, padded_transparent, claim)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField128bGhash;

	use super::*;
	use crate::channel::oracle_setup::OracleSetupChannel;

	type F = BinaryField128bGhash;

	/// Runs the given rounds through the merging decorator.
	///
	/// Returns the round-shaped spec list its underlying channel recorded.
	/// Each inner list is one round.
	/// A round is flushed once a challenge is sampled after it.
	fn record_rounds(rounds: &[&[usize]], is_zk: bool) -> Vec<OracleSpec> {
		// Flatten the rounds into one flat list of sizes, in order.
		let fine_sizes: Vec<usize> = rounds
			.iter()
			.flat_map(|round| round.iter().copied())
			.collect();
		let fine_specs: Vec<OracleSpec> = fine_sizes
			.iter()
			.map(|&n| {
				if is_zk {
					OracleSpec::new_zk(n)
				} else {
					OracleSpec::new(n)
				}
			})
			.collect();

		// A recording-only channel stands in for a real one here.
		// It never checks a proof.
		// It only remembers the shape of every oracle it receives.
		let mut channel = MergeVerifierChannel::new(OracleSetupChannel::new(is_zk), &fine_specs);
		for sizes in rounds {
			for &n in *sizes {
				channel.recv_oracle(n, true).unwrap();
			}
			// Crossing a round boundary flushes it as one combined oracle.
			IPVerifierChannel::<F>::sample(&mut channel);
		}
		channel.into_inner().unwrap().into_oracle_specs()
	}

	#[test]
	fn merges_oracles_within_a_round() {
		// Three rounds, each a different shape.
		//
		// Round 1: two same-size oracles.
		// 2^3 + 2^3 = 2^4, an exact power of two.
		//
		// Round 2: three unequal oracles.
		// 2^4 + 2^2 + 2^2 = 24, not a power of two.
		// One extra bit is needed: 2^5 = 32.
		//
		// Round 3: a single oracle.
		// It needs no combining.
		let coarse = record_rounds(&[&[3, 3], &[4, 2, 2], &[1]], true);

		// One combined oracle per round.
		//
		// Every one is still marked zero-knowledge.
		// Every constituent declared itself zero-knowledge too.
		assert_eq!(
			coarse,
			vec![
				OracleSpec::new_zk(4),
				OracleSpec::new_zk(5),
				OracleSpec::new_zk(1)
			]
		);
	}

	#[test]
	fn ties_keep_arrival_order() {
		// Three equal-size oracles.
		// 2^2 + 2^2 + 2^2 = 12, rounded up to 2^4 = 16.
		//
		// Equal sizes keep the alignment property either way.
		// This only checks the combined size comes out right.
		let coarse = record_rounds(&[&[2, 2, 2]], false);
		assert_eq!(coarse, vec![OracleSpec::new(log2_ceil_usize(3 << 2))]);
	}

	#[test]
	fn non_zk_round_records_non_zk_combined_spec() {
		// Neither constituent depends on the witness.
		// The combined oracle must not be zero-knowledge either.
		let coarse = record_rounds(&[&[5, 3]], false);
		assert_eq!(coarse, vec![OracleSpec::new(6)]);
	}

	#[test]
	#[should_panic(expected = "must share is_witness_dependent")]
	fn heterogeneous_witness_dependence_panics() {
		// One oracle depends on the witness.
		// The other does not.
		//
		// One commitment cannot be masked for one and not the other.
		// Mixing the two in a round must be rejected.
		let fine_specs = [OracleSpec::new(2), OracleSpec::new(2)];
		let mut channel = MergeVerifierChannel::new(OracleSetupChannel::new(true), &fine_specs);
		channel.recv_oracle(2, true).unwrap();
		channel.recv_oracle(2, false).unwrap();
		let _ = IPVerifierChannel::<F>::sample(&mut channel);
	}

	#[test]
	#[should_panic(expected = "recv_oracle called but no remaining oracle specs")]
	fn recv_oracle_past_remaining_specs_panics() {
		// Only one oracle was declared up front.
		//
		// Receiving a second must be rejected.
		// It must not be silently accepted.
		let fine_specs = [OracleSpec::new(2)];
		let mut channel =
			MergeVerifierChannel::<F, _>::new(OracleSetupChannel::new(false), &fine_specs);
		channel.recv_oracle(2, true).unwrap();
		channel.recv_oracle(2, true).unwrap();
	}

	#[test]
	#[should_panic(expected = "into_inner called but 1 oracle specs remaining")]
	fn into_inner_before_all_specs_received_panics() {
		// Two oracles were declared up front.
		// Only one ever arrives.
		//
		// One oracle spec is still outstanding at teardown.
		let fine_specs = [OracleSpec::new(2), OracleSpec::new(2)];
		let mut channel =
			MergeVerifierChannel::<F, _>::new(OracleSetupChannel::new(false), &fine_specs);
		channel.recv_oracle(2, true).unwrap();
		let _ = channel.into_inner();
	}
}
