// Copyright 2026 The Binius Developers

//! A channel decorator that merges oracles committed within the same interaction round.

use binius_core::word::Word;
use binius_field::{BinaryField, Field, field::FieldOps};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_math::multilinear::eq::eq_ind;

use crate::channel::{
	Error, IOPVerifierChannel, OracleSchedule, OracleSpec, TransparentEvalFn, layout_round,
};

/// A handle to an oracle received through the merging decorator.
#[derive(Debug, Clone, Copy)]
pub struct MergeOracle {
	index: usize,
}

/// Where one oracle lives inside its round's combined oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
	/// The index of the round the oracle is committed in.
	pub round: usize,

	/// The oracle's position among its round-mates, as a block index.
	///
	/// Its own `2^n` scalars begin at scalar `block_index * 2^n`.
	pub block_index: usize,

	/// The base-2 logarithm of the combined oracle's length.
	pub combined_log_len: usize,
}

/// Lays out every round of a schedule, returning one placement per oracle in arrival order.
///
/// Each round's largest oracle hosts the rest past its content, as described on
/// [`MergeVerifierChannel`].
pub fn place_oracles(schedule: &OracleSchedule) -> Vec<Placement> {
	let mut placements = Vec::with_capacity(schedule.specs().len());
	for (round, specs) in schedule.rounds().enumerate() {
		let (block_indices, merged) = layout_round(specs);
		placements.extend(block_indices.into_iter().map(|block_index| Placement {
			round,
			block_index,
			combined_log_len: merged.log_msg_len,
		}));
	}
	assert_eq!(placements.len(), schedule.specs().len(), "every oracle must be in a closed round");
	placements
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
/// [`place_oracles`] lays each round out.
///
/// The largest oracle is the host, at the start of the combined oracle.
/// The rest are tenants, largest first, each at the first block of its own size past everything
/// placed before it, starting at the host's content length.
///
/// ```text
/// combined oracle, size 2^N:
///
///     [ host content | tenant 1 | tenant 2 | ... | tenant k | unused padding ]
///       offset 0       past host.len                          total size 2^N
/// ```
///
/// Tenants fill the host's padding first; `N` exceeds the host's size only if they do not fit.
///
/// Each tenant starts on a boundary of its own size.
/// Its position is then a whole number of its-own-size blocks.
/// That whole number is its block index.
///
/// A round of a single oracle needs no combining.
/// It is forwarded unchanged, at zero cost.
///
/// A round is witness-dependent as soon as any of its oracles is.
///
/// A commitment is masked as a whole, never partly.
/// So a structural oracle sharing a round with a witness-carrying one is masked too.
///
/// # Timing
///
/// A round's oracles are committed the moment its last oracle arrives.
///
/// The [`OracleSchedule`] says where each round ends, so no challenge sample is needed to find
/// the boundary.
/// A real Fiat-Shamir transcript can therefore absorb the commitment before the next challenge.
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
///
/// A host's block holds its tenants too, in its padding.
/// Padding is the channel's to fill, so the host's claims already hold over them.
pub struct MergeVerifierChannel<'a, F, C>
where
	F: Field,
	C: IOPVerifierChannel<F>,
{
	/// The underlying channel every oracle and challenge passes through.
	inner: C,

	/// The fine-grained oracles this channel's caller will receive, grouped into rounds.
	schedule: &'a OracleSchedule,

	/// Where each oracle of the schedule lands, in arrival order.
	placements: Vec<Placement>,

	/// The handle the underlying channel returned for each committed round.
	outers: Vec<C::Oracle>,

	/// One spec per round, as the underlying channel receives it.
	///
	/// A round is masked as a whole, so it is witness-dependent iff its spec is zero-knowledge.
	round_specs: Vec<OracleSpec>,

	/// How many oracles have been received so far.
	n_received: usize,
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
	/// * `inner` — the channel every combined oracle is committed to, already configured with
	///   `schedule.merged_specs()`.
	/// * `schedule` — every oracle this channel's caller will pass through, grouped into rounds.
	///
	/// # Panics
	///
	/// Panics if `inner` is not configured with `schedule.merged_specs()`.
	pub fn new(inner: C, schedule: &'a OracleSchedule) -> Self {
		let round_specs = schedule.merged_specs();
		assert_eq!(
			inner.remaining_oracle_specs(),
			round_specs,
			"inner channel must be configured with the schedule's merged specs"
		);
		Self {
			inner,
			schedule,
			placements: place_oracles(schedule),
			outers: Vec::new(),
			round_specs,
			n_received: 0,
		}
	}

	/// Returns the underlying channel.
	///
	/// # Panics
	///
	/// Panics if any declared oracle has not yet been received.
	pub fn into_inner(self) -> C {
		let n_remaining = self.placements.len() - self.n_received;
		assert!(n_remaining == 0, "into_inner called but {n_remaining} oracle specs remaining",);
		self.inner
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
		&self.schedule.specs()[self.n_received..]
	}

	fn recv_oracle(
		&mut self,
		len: usize,
		log_msg_len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error> {
		// Every oracle this channel will receive is declared up front.
		//
		// Reject anything past that count.
		// Do not silently accept an undeclared oracle.
		let remaining = self.remaining_oracle_specs();
		assert!(!remaining.is_empty(), "recv_oracle called but no remaining oracle specs");
		let spec = remaining[0];
		assert_eq!(log_msg_len, spec.log_msg_len, "oracle size must match its spec");
		assert_eq!(len, spec.len, "oracle content length must match its spec");

		// A spec is zero-knowledge iff the protocol is and the oracle is witness-dependent.
		//
		// The protocol-level flag is not known here, so only one direction can be checked.
		assert!(
			!spec.is_zk || is_witness_dependent,
			"a zero-knowledge oracle spec must be received as witness-dependent"
		);

		let index = self.n_received;
		self.n_received += 1;

		// The last oracle of its round commits the whole round as one oracle.
		let Placement {
			round,
			combined_log_len,
			..
		} = self.placements[index];
		let is_last_of_round = self
			.placements
			.get(index + 1)
			.is_none_or(|next| next.round != round);
		if is_last_of_round {
			// One commitment is masked as a whole, never partly.
			//
			// So the combined oracle is witness-dependent as soon as any constituent is.
			// A structural oracle sharing the round is masked along with it, which costs
			// randomness but never correctness.
			let outer = self.inner.recv_oracle(
				self.round_specs[round].len,
				combined_log_len,
				self.round_specs[round].is_zk,
			)?;
			self.outers.push(outer);
		}

		Ok(MergeOracle { index })
	}

	fn verify_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: TransparentEvalFn<Self::Elem>,
		claim: Self::Elem,
	) -> Result<(), Error> {
		let n_i = self.schedule.specs()[oracle.index].log_msg_len;
		let Placement {
			round,
			block_index,
			combined_log_len,
		} = self.placements[oracle.index];
		let outer = self.outers[round].clone();

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
	use binius_field::Ghash128b;
	use binius_hash::StdDigest;
	use binius_transcript::{VerifierTranscript, fiat_shamir::HasherChallenger};

	use super::*;
	use crate::channel::naive::NaiveVerifierChannel;

	type F = Ghash128b;

	/// Builds a schedule from rounds of oracle sizes, in log2.
	fn schedule(rounds: &[&[usize]]) -> OracleSchedule {
		let mut schedule = OracleSchedule::new();
		for sizes in rounds {
			for &n in *sizes {
				schedule.push(OracleSpec::new(n));
			}
			schedule.end_round();
		}
		schedule
	}

	#[test]
	fn placements_sort_each_round_largest_first() {
		// Round 0: 2^2 + 2^4 + 2^2 = 24, rounded up to 2^5.
		// The 2^4 oracle goes first, then the two 2^2 oracles in arrival order.
		//
		// Round 1: a lone oracle, forwarded with no combining at all.
		let placements = place_oracles(&schedule(&[&[2, 4, 2], &[1]]));
		let place = |round, block_index, combined_log_len| Placement {
			round,
			block_index,
			combined_log_len,
		};
		assert_eq!(
			placements,
			[
				place(0, 4, 5),
				place(0, 0, 5),
				place(0, 5, 5),
				place(1, 0, 1)
			]
		);
	}

	#[test]
	fn placements_fill_the_host_padding() {
		// The plain Spartan round: precommit 2^10, private 2^17 filled to 98332, mask 2^9.
		//
		// The private oracle hosts: the precommit goes at block ⌈98332 / 2^10⌉ = 97, ending at
		// 98 * 2^10, and the mask at block 98 * 2 = 196.
		let mut schedule = OracleSchedule::new();
		schedule.push(OracleSpec::new(10));
		schedule.push(OracleSpec {
			len: 98332,
			..OracleSpec::new(17)
		});
		schedule.push(OracleSpec::new(9));
		schedule.end_round();
		let place = |block_index| Placement {
			round: 0,
			block_index,
			combined_log_len: 17,
		};
		assert_eq!(place_oracles(&schedule), [place(97), place(0), place(196)]);
		assert_eq!(
			schedule.merged_specs(),
			[OracleSpec {
				len: 197 << 9,
				..OracleSpec::new(17)
			}]
		);
	}

	#[test]
	#[should_panic(expected = "recv_oracle called but no remaining oracle specs")]
	fn recv_oracle_past_remaining_specs_panics() {
		// Only one oracle was declared up front.
		//
		// Receiving a second must be rejected.
		// It must not be silently accepted.
		let schedule = schedule(&[&[2]]);
		let merged_specs = schedule.merged_specs();
		let mut transcript =
			VerifierTranscript::new(HasherChallenger::<StdDigest>::default(), vec![0; 4 * 16]);
		let mut channel = MergeVerifierChannel::new(
			NaiveVerifierChannel::<F, _>::new(&mut transcript, &merged_specs),
			&schedule,
		);
		channel.recv_oracle(1 << 2, 2, true).unwrap();
		channel.recv_oracle(1 << 2, 2, true).unwrap();
	}

	#[test]
	#[should_panic(expected = "a zero-knowledge oracle spec must be received as witness-dependent")]
	fn zk_spec_received_as_structural_panics() {
		// The schedule declares a masked oracle.
		// Receiving it as independent of the witness contradicts that.
		let mut schedule = OracleSchedule::new();
		schedule.push(OracleSpec::new_zk(2));
		schedule.end_round();
		let merged_specs = schedule.merged_specs();
		let mut transcript =
			VerifierTranscript::new(HasherChallenger::<StdDigest>::default(), Vec::new());
		let mut channel = MergeVerifierChannel::new(
			NaiveVerifierChannel::<F, _>::new(&mut transcript, &merged_specs),
			&schedule,
		);
		let _ = channel.recv_oracle(1 << 2, 2, false);
	}

	#[test]
	#[should_panic(expected = "into_inner called but 1 oracle specs remaining")]
	fn into_inner_before_all_specs_received_panics() {
		// Two oracles were declared up front.
		// Only one ever arrives.
		//
		// One oracle spec is still outstanding at teardown.
		let schedule = schedule(&[&[2, 2]]);
		let merged_specs = schedule.merged_specs();
		let mut transcript =
			VerifierTranscript::new(HasherChallenger::<StdDigest>::default(), Vec::new());
		let mut channel = MergeVerifierChannel::new(
			NaiveVerifierChannel::<F, _>::new(&mut transcript, &merged_specs),
			&schedule,
		);
		channel.recv_oracle(1 << 2, 2, true).unwrap();
		let _ = channel.into_inner();
	}
}
