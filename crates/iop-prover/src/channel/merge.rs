// Copyright 2026 The Binius Developers

//! A channel decorator that merges oracles committed within the same interaction round.
//!
//! This is the prover-side half of a matching pair.
//! Whatever it commits here must be received by a matching verifier-side decorator.

use std::{cmp::Reverse, ops::DerefMut};

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_iop::channel::OracleSpec;
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_math::{FieldBuffer, FieldSlice, FieldVec};
use binius_utils::checked_arithmetics::log2_ceil_usize;

use crate::channel::IOPProverChannel;

/// A handle to an oracle sent through the merging decorator.
#[derive(Debug, Clone, Copy)]
pub struct MergeOracle {
	index: usize,
}

/// Records where one constituent oracle lives inside its round's combined oracle.
#[derive(Clone, Copy)]
struct Mapping {
	/// Which round's combined oracle this constituent belongs to.
	group: usize,

	/// The constituent's position among its round-mates, as a block index.
	///
	/// Its own `2^n` scalars begin at scalar `block_index * 2^n`.
	block_index: usize,

	/// The base-2 logarithm of the combined oracle's length.
	combined_log_len: usize,
}

/// Tracks one oracle from the moment it is sent until its round is flushed.
struct Record {
	/// The base-2 logarithm of the oracle's own length.
	log_msg_len: usize,

	/// Where this oracle ends up inside its round's combined oracle.
	///
	/// Filled in once that round is flushed.
	mapping: Option<Mapping>,
}

/// The combined oracle actually committed for one round.
///
/// Kept alive until every constituent oracle is finalized.
struct Group<P: PackedField, A: Allocator, Oracle> {
	/// The handle the underlying channel returned for the combined oracle.
	outer: Oracle,

	/// The combined witness data.
	///
	/// Kept until every constituent has been handed back.
	/// Then forwarded to the underlying channel exactly once, in their place.
	buffer: Option<FieldVec<P, A>>,

	/// How many constituent oracles make up this round.
	n_members: usize,

	/// How many of those constituents have been handed back so far.
	n_finalized: usize,
}

/// A prover channel decorator that merges one round's oracles into one combined oracle.
///
/// # Overview
///
/// An interaction round is the run of oracles sent between two challenge samples.
///
/// Committing each oracle separately costs one commitment per oracle.
/// One Merkle tree per oracle, for example.
///
/// This decorator buffers a round's oracles instead.
/// It commits them together as one larger oracle.
/// That cuts the cost to one commitment per round.
///
/// A round's oracles are sorted from largest to smallest, then laid out end to end.
/// That ordering makes every oracle's position exact.
///
/// Every earlier oracle is at least as large as the current one.
/// So their combined space is a whole multiple of the current oracle's size.
///
/// A round of a single oracle needs no combining.
/// It is forwarded unchanged, at zero cost.
///
/// A round is masked as a whole, never partly.
///
/// So a round carrying any witness data is masked in full by the underlying channel.
/// The verifier-side decorator is where that choice is made.
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
/// A verifier only ever holds a formula for a transparent polynomial.
///
/// This side holds the actual coefficients instead.
/// A constituent's own transparent polynomial becomes one for the combined oracle.
/// Write its values into a zero buffer, at the constituent's own position.
/// Every other position stays zero.
///
/// That placement is the same polynomial a verifier reaches by formula.
/// One side holds it as explicit values.
/// The other evaluates it on demand.
pub struct MergeProverChannel<'a, P, A, C>
where
	P: PackedField,
	A: Allocator,
	C: IOPProverChannel<P, A>,
{
	/// The underlying channel every oracle, challenge, and opening passes through.
	inner: C,

	/// Fine-grained specs, one per oracle this channel's caller will send.
	///
	/// Not the coarser, one-per-round specs the underlying channel uses.
	oracle_specs: &'a [OracleSpec],

	/// The allocator this channel draws its combined and padded buffers from.
	alloc: A,

	/// Buffers received for the current round, not yet sent to the underlying channel.
	pending: Vec<FieldVec<P, A>>,

	/// Every oracle sent so far, in arrival order.
	records: Vec<Record>,

	/// Every round committed so far, in commit order.
	groups: Vec<Group<P, A, C::Oracle>>,
}

impl<'a, P, A, C> MergeProverChannel<'a, P, A, C>
where
	P: PackedField,
	A: Allocator,
	C: IOPProverChannel<P, A>,
{
	/// Creates a new merging prover channel over an underlying channel.
	///
	/// # Arguments
	///
	/// * `inner` — the channel every combined oracle is committed to, already configured with the
	///   coarser, one-per-round spec list this decorator will produce.
	/// * `oracle_specs` — the fine-grained specs for every oracle this channel's caller will pass
	///   through, in arrival order.
	/// * `alloc` — where this channel draws its combined and padded buffers from.
	pub const fn new(inner: C, oracle_specs: &'a [OracleSpec], alloc: A) -> Self {
		Self {
			inner,
			oracle_specs,
			alloc,
			pending: Vec::new(),
			records: Vec::new(),
			groups: Vec::new(),
		}
	}

	/// Commits the current round's queued buffers as one combined oracle.
	///
	/// Does nothing if every sent oracle is already committed.
	fn flush(&mut self) {
		// Nothing new has arrived since the last flush.
		if self.pending.is_empty() {
			return;
		}
		let first_index = self.records.len() - self.pending.len();

		// Order this round largest to smallest.
		//
		// This lets every position be a whole number of block sizes.
		// That holds once every earlier buffer is at least as large.
		let mut order: Vec<usize> = (0..self.pending.len()).collect();
		order.sort_by_key(|&k| Reverse(self.pending[k].log_len()));

		// Size the combined oracle to fit every buffer end to end.
		let total_len: usize = order
			.iter()
			.map(|&k| 1usize << self.pending[k].log_len())
			.sum();
		let combined_log_len = log2_ceil_usize(total_len);

		// Lay every buffer into the combined data.
		// Record where each one landed.
		//
		// Every step adds a whole multiple of the next block's size.
		// So each offset divides evenly by that buffer's own size.
		let mut combined = FieldBuffer::zeros_in(&self.alloc, combined_log_len);
		let mut block_indices = vec![0usize; self.pending.len()];
		let mut offset = 0usize;
		for &k in &order {
			let n_k = self.pending[k].log_len();
			let block_index = offset >> n_k;
			block_indices[k] = block_index;
			place_block(&mut combined, self.pending[k].as_view(), block_index);
			offset += 1 << n_k;
		}

		// Commit the whole round as one oracle on the underlying channel.
		//
		// The combined data stays alive in its own round record.
		// It is still needed once every constituent is handed back.
		let outer = self.inner.send_oracle(combined.as_view());
		let group_index = self.groups.len();
		let n_members = self.pending.len();
		self.groups.push(Group {
			outer,
			buffer: Some(combined),
			n_members,
			n_finalized: 0,
		});

		// Record where each constituent oracle landed.
		for k in 0..n_members {
			self.records[first_index + k].mapping = Some(Mapping {
				group: group_index,
				block_index: block_indices[k],
				combined_log_len,
			});
		}

		self.pending.clear();
	}

	/// Commits any oracles still queued and returns the underlying channel.
	///
	/// # Panics
	///
	/// Panics if any declared oracle has not yet been sent.
	pub fn into_inner(mut self) -> C {
		self.flush();
		let n_remaining = self.oracle_specs.len() - self.records.len();
		assert!(n_remaining == 0, "into_inner called but {n_remaining} oracle specs remaining",);
		self.inner
	}
}

/// Writes one buffer into a fixed-size block of another.
///
/// Every other position of the destination is left untouched.
///
/// # Panics
///
/// Panics if the block does not fit at the given index.
fn place_block<P, Data>(dst: &mut FieldBuffer<P, Data>, src: FieldSlice<'_, P>, block_index: usize)
where
	P: PackedField,
	Data: DerefMut<Target = [P]>,
{
	// The destination block starts at a whole multiple of the source's own length.
	//
	// A block index of zero starts at the very first scalar.
	let n = src.log_len();
	let offset = block_index << n;
	assert!(offset + (1 << n) <= dst.len(), "pre-condition: the block must fit in the destination");

	// Copy every scalar across.
	// Everywhere else in the destination stays as it was.
	for i in 0..1usize << n {
		dst.set(offset + i, src.get(i));
	}
}

impl<F, P, A, C> IPProverChannel<F> for MergeProverChannel<'_, P, A, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
	A: Allocator,
	C: IOPProverChannel<P, A>,
{
	fn send_one(&mut self, elem: F) {
		self.inner.send_one(elem);
	}

	fn send_many(&mut self, elems: &[F]) {
		self.inner.send_many(elems);
	}

	fn observe_one(&mut self, val: F) {
		self.inner.observe_one(val);
	}

	fn observe_many(&mut self, vals: &[F]) {
		self.inner.observe_many(vals);
	}

	fn sample(&mut self) -> F {
		// Commit this round before deriving its challenge.
		//
		// A real transcript must absorb a commitment first.
		// Only then can it derive a challenge that depends on it.
		self.flush();
		self.inner.sample()
	}
}

impl<F, P, A, C> WordIPProverChannel<F> for MergeProverChannel<'_, P, A, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
	A: Allocator,
	C: IOPProverChannel<P, A> + WordIPProverChannel<F>,
{
	type Word = C::Word;

	fn observe_words(&mut self, words: &[Self::Word]) {
		self.inner.observe_words(words);
	}

	fn sample_bits(&mut self, bits: usize) -> Self::Word {
		// A sampled word is a challenge like any other.
		//
		// So this round's commitment must be absorbed before it is drawn.
		self.flush();
		self.inner.sample_bits(bits)
	}
}

impl<'a, F, P, A, C> IOPProverChannel<P, A> for MergeProverChannel<'a, P, A, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
	A: Allocator,
	C: IOPProverChannel<P, A>,
{
	type Oracle = MergeOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.records.len()..]
	}

	fn send_oracle(&mut self, buffer: FieldSlice<'_, P>) -> Self::Oracle {
		// Every oracle this channel will send is declared up front.
		//
		// Reject anything past that count.
		// Do not silently accept an undeclared oracle.
		let remaining = self.remaining_oracle_specs();
		assert!(!remaining.is_empty(), "send_oracle called but no remaining oracle specs");
		debug_assert_eq!(buffer.log_len(), remaining[0].log_msg_len);

		// Copy the data into a buffer this channel owns.
		//
		// The caller's buffer is only borrowed for this call.
		// Its round may not commit until a later challenge sample.
		self.pending
			.push(FieldBuffer::from_view_in(&self.alloc, buffer));
		self.records.push(Record {
			log_msg_len: buffer.log_len(),
			mapping: None,
		});
		MergeOracle {
			index: self.records.len() - 1,
		}
	}

	fn prove_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: FieldVec<P, A>,
		claim: P::Scalar,
	) {
		// Every oracle must be sent before any oracle is opened.
		// So this oracle's round is already committed by now.
		//
		// Flush anyway, in case no challenge was sampled in between.
		self.flush();

		let record = &self.records[oracle.index];
		let n_i = record.log_msg_len;
		assert_eq!(
			transparent.log_len(),
			n_i,
			"transparent log_len mismatch: expected {n_i}, got {}",
			transparent.log_len()
		);
		let Mapping {
			group,
			block_index,
			combined_log_len,
		} = record.mapping.expect("flushed above");

		// Place the constituent's transparent polynomial into a zero buffer.
		// Use the width of the combined oracle, at this oracle's own block.
		//
		// Everywhere outside that block reads as zero.
		// So the produced inner product equals the original claim exactly.
		let mut padded = FieldBuffer::zeros_in(&self.alloc, combined_log_len);
		place_block(&mut padded, transparent.as_view(), block_index);

		let outer = self.groups[group].outer.clone();
		self.inner.prove_oracle_relation(outer, padded, claim);
	}

	fn finalize_oracle(&mut self, oracle: Self::Oracle, _buffer: FieldVec<P, A>) {
		// The buffer handed back here must equal the one already sent.
		// This channel already copied that data at commit time.
		// So the copy handed back now is simply discarded.
		self.flush();

		let record = &self.records[oracle.index];
		let group_index = record.mapping.expect("flushed above").group;

		// Once every constituent is handed back, the round is done.
		// Its combined buffer can now reach the underlying channel.
		let group = &mut self.groups[group_index];
		group.n_finalized += 1;
		if group.n_finalized == group.n_members {
			let combined_buffer = group
				.buffer
				.take()
				.expect("group buffer present until every constituent is finalized");
			let outer = group.outer.clone();
			self.inner.finalize_oracle(outer, combined_buffer);
		}
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_field::{
		BinaryField, Field, Ghash128b, PackedBinaryGhash1x128b, PackedBinaryGhash4x128b,
		PackedField,
	};
	use binius_hash::StdDigest;
	use binius_iop::channel::{
		IOPVerifierChannel, OracleSpec, merge::MergeVerifierChannel, naive::NaiveVerifierChannel,
	};
	use binius_ip::channel::IPVerifierChannel;
	use binius_ip_prover::channel::IPProverChannel;
	use binius_math::{
		FieldBuffer,
		multilinear::{Multilinear, hypercube::Hypercube},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use proptest::prelude::*;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::{IOPProverChannel, MergeProverChannel};
	use crate::channel::naive::NaiveProverChannel;

	type StdChallenger = HasherChallenger<StdDigest>;

	/// Generates a random buffer of a given size.
	///
	/// Also returns an independent transparent polynomial.
	/// And the claim their inner product produces.
	fn generate_oracle_data<F, P, R: Rng>(
		rng: &mut R,
		n_vars: usize,
	) -> (FieldBuffer<P>, FieldBuffer<P>, F)
	where
		F: BinaryField,
		P: PackedField<Scalar = F>,
	{
		let buffer = random_field_buffer::<P>(&mut *rng, n_vars);
		let point = random_scalars::<F>(&mut *rng, n_vars);
		let transparent = Hypercube::One.expand(&point).build::<P>();
		let claim = buffer.inner_product(&transparent);
		(buffer, transparent, claim)
	}

	/// Runs a full prove-then-verify round trip over oracles grouped into rounds.
	///
	/// Each inner slice of `rounds` lists one round's oracle sizes, in log2.
	/// Every one is sent, or received, before one challenge is sampled.
	///
	/// If `tamper` is set, the first oracle's claim is corrupted.
	/// Verification must then reject the whole round trip.
	fn run_merge_round_trip<P>(rounds: &[&[usize]], tamper: bool)
	where
		P: PackedField<Scalar = Ghash128b>,
	{
		type F = Ghash128b;

		let mut rng = StdRng::seed_from_u64(0);

		// Flatten the rounds into one flat list of sizes, in order.
		// Both sides' bookkeeping expects that same list.
		// Generate independent witness data for each one.
		let fine_sizes: Vec<usize> = rounds
			.iter()
			.flat_map(|round| round.iter().copied())
			.collect();
		let fine_specs: Vec<OracleSpec> = fine_sizes.iter().map(|&n| OracleSpec::new(n)).collect();
		let data: Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> = fine_sizes
			.iter()
			.map(|&n| generate_oracle_data::<F, P, _>(&mut rng, n))
			.collect();

		// The expected result of merging.
		//
		// One combined oracle per round.
		// Sized to the smallest power of two that fits the total.
		let coarse_specs: Vec<OracleSpec> = rounds
			.iter()
			.map(|sizes| {
				let total: usize = sizes.iter().map(|&n| 1usize << n).sum();
				OracleSpec::new(log2_ceil_usize(total))
			})
			.collect();

		// Prover side.
		//
		// Send every oracle round by round.
		// Sample a challenge between rounds.
		// Each round commits as it goes.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let naive_prover = NaiveProverChannel::new(&mut prover_transcript, coarse_specs.clone());
		let mut merge_prover = MergeProverChannel::new(naive_prover, &fine_specs, GlobalAllocator);

		let mut oracles = Vec::new();
		let mut index = 0;
		for sizes in rounds {
			for _ in *sizes {
				let (buffer, _, _) = &data[index];
				oracles.push(merge_prover.send_oracle(buffer.as_view()));
				index += 1;
			}
			IPProverChannel::sample(&mut merge_prover);
		}
		// Prove every oracle's claim first.
		// Then hand back every oracle's own witness data.
		// That matches the order a real prover would follow.
		for (&oracle, (_, transparent, claim)) in iter::zip(&oracles, &data) {
			merge_prover.prove_oracle_relation(oracle, transparent.clone(), *claim);
		}
		for (&oracle, (buffer, _, _)) in iter::zip(&oracles, &data) {
			merge_prover.finalize_oracle(oracle, buffer.clone());
		}
		merge_prover.into_inner().finish();

		// Verifier side.
		//
		// Mirror the exact same round boundaries.
		// Both sides then sample from the same transcript positions.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let naive_verifier = NaiveVerifierChannel::new(&mut verifier_transcript, &coarse_specs);
		let mut merge_verifier = MergeVerifierChannel::new(naive_verifier, &fine_specs);

		let mut v_oracles = Vec::new();
		for sizes in rounds {
			for &n in *sizes {
				v_oracles.push(merge_verifier.recv_oracle(n, true).unwrap());
			}
			IPVerifierChannel::sample(&mut merge_verifier);
		}
		for (position, (&oracle, (_, transparent, claim))) in
			iter::zip(&v_oracles, &data).enumerate()
		{
			let transparent = transparent.clone();
			// Corrupt only the first oracle's claim.
			// Only do this when tampering is requested.
			// Every other claim is left untouched.
			let claim = if tamper && position == 0 {
				*claim + F::ONE
			} else {
				*claim
			};
			merge_verifier
				.verify_oracle_relation(
					oracle,
					Box::new(move |point: &[F]| {
						let eq = Hypercube::One.expand(point).build::<P>();
						transparent.inner_product(&eq)
					}),
					claim,
				)
				.expect("verification only ever queues a relation, it does not check it here");
		}
		merge_verifier.into_inner().unwrap().finish();
	}

	#[test]
	fn single_oracle_round_trip() {
		// A single round holding a single oracle.
		// The degenerate case, where merging has nothing to do.
		run_merge_round_trip::<PackedBinaryGhash1x128b>(&[&[6]], false);
	}

	#[test]
	fn multi_round_round_trip() {
		// Three rounds, each a different shape.
		//
		// Round 1: two equal-size oracles.
		// An exact power-of-two total.
		//
		// Round 2: three unequal oracles.
		// A non-power-of-two total, so padding is required.
		//
		// Round 3: a single oracle, the degenerate case.
		run_merge_round_trip::<PackedBinaryGhash1x128b>(&[&[3, 3], &[4, 2, 2], &[1]], false);
	}

	#[test]
	fn multi_round_round_trip_narrow_packing() {
		// The same three rounds, under a wider packing width.
		//
		// Some oracles are narrower than one packed field element.
		// Placement then writes into part of one, not a whole chunk.
		const {
			assert!(
				PackedBinaryGhash4x128b::LOG_WIDTH > 0,
				"the fixture needs sub-packed-width oracle sizes to appear"
			);
		};
		run_merge_round_trip::<PackedBinaryGhash4x128b>(&[&[3, 3], &[4, 2, 2], &[1]], false);
	}

	#[test]
	fn zero_variable_oracle_round_trip() {
		// A round mixing two single-scalar oracles with a larger one.
		// The smallest possible oracle size.
		run_merge_round_trip::<PackedBinaryGhash1x128b>(&[&[0, 0, 3]], false);
	}

	#[test]
	#[should_panic(expected = "NaiveVerifierChannel: inner product verification failed")]
	fn tampered_claim_is_rejected() {
		// Corrupting one oracle's claim must fail the whole round trip.
		// It must not be silently absorbed by the merge.
		run_merge_round_trip::<PackedBinaryGhash1x128b>(&[&[3, 3], &[4, 2, 2]], true);
	}

	#[test]
	fn multiple_relations_on_merged_oracle() {
		type F = Ghash128b;
		type P = PackedBinaryGhash1x128b;

		// Two oracles, merged into a single round.
		// Each carries two independent claims, rather than just one.
		let mut rng = StdRng::seed_from_u64(0);
		let fine_specs = vec![OracleSpec::new(4), OracleSpec::new(3)];
		let (buffer_1, _, _) = generate_oracle_data::<F, P, _>(&mut rng, 4);
		let (buffer_2, _, _) = generate_oracle_data::<F, P, _>(&mut rng, 3);

		let relations_1: Vec<(FieldBuffer<P>, F)> = (0..2)
			.map(|_| {
				let point = random_scalars::<F>(&mut rng, 4);
				let transparent = Hypercube::One.expand(&point).build::<P>();
				let claim = buffer_1.inner_product(&transparent);
				(transparent, claim)
			})
			.collect();
		let relations_2: Vec<(FieldBuffer<P>, F)> = (0..2)
			.map(|_| {
				let point = random_scalars::<F>(&mut rng, 3);
				let transparent = Hypercube::One.expand(&point).build::<P>();
				let claim = buffer_2.inner_product(&transparent);
				(transparent, claim)
			})
			.collect();

		// One round, sized to fit both oracles.
		// 2^4 + 2^3 = 24, rounded up to 2^5.
		let total: usize = (1usize << 4) + (1usize << 3);
		let coarse_specs = vec![OracleSpec::new(log2_ceil_usize(total))];

		// Prover side.
		//
		// Both oracles arrive in the same round.
		// All four claims are proved before either oracle is finalized.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let naive_prover = NaiveProverChannel::new(&mut prover_transcript, coarse_specs.clone());
		let mut merge_prover = MergeProverChannel::new(naive_prover, &fine_specs, GlobalAllocator);

		let oracle_1 = merge_prover.send_oracle(buffer_1.as_view());
		let oracle_2 = merge_prover.send_oracle(buffer_2.as_view());
		for (transparent, claim) in &relations_1 {
			merge_prover.prove_oracle_relation(oracle_1, transparent.clone(), *claim);
		}
		for (transparent, claim) in &relations_2 {
			merge_prover.prove_oracle_relation(oracle_2, transparent.clone(), *claim);
		}
		merge_prover.finalize_oracle(oracle_1, buffer_1);
		merge_prover.finalize_oracle(oracle_2, buffer_2);
		merge_prover.into_inner().finish();

		// Verifier side.
		//
		// The same two oracles, each checked against its own two claims.
		// In the same order the prover produced them.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let naive_verifier = NaiveVerifierChannel::new(&mut verifier_transcript, &coarse_specs);
		let mut merge_verifier = MergeVerifierChannel::new(naive_verifier, &fine_specs);

		let v_oracle_1 = merge_verifier.recv_oracle(4, true).unwrap();
		let v_oracle_2 = merge_verifier.recv_oracle(3, true).unwrap();
		for (transparent, claim) in relations_1 {
			merge_verifier
				.verify_oracle_relation(
					v_oracle_1,
					Box::new(move |point: &[F]| {
						let eq = Hypercube::One.expand(point).build::<P>();
						transparent.inner_product(&eq)
					}),
					claim,
				)
				.unwrap();
		}
		for (transparent, claim) in relations_2 {
			merge_verifier
				.verify_oracle_relation(
					v_oracle_2,
					Box::new(move |point: &[F]| {
						let eq = Hypercube::One.expand(point).build::<P>();
						transparent.inner_product(&eq)
					}),
					claim,
				)
				.unwrap();
		}
		merge_verifier.into_inner().unwrap().finish();
	}

	proptest! {
		#[test]
		fn round_trip_proptest(
			rounds in prop::collection::vec(prop::collection::vec(0usize..5, 1..5), 1..5),
		) {
			// Random round shapes.
			//
			// 1 to 4 rounds, each holding 1 to 4 oracles sized 2^0 to 2^4.
			//
			// This goes far beyond the hand-picked shapes above.
			// It stress-tests the alignment argument the merge relies on.
			let round_refs: Vec<&[usize]> = rounds.iter().map(Vec::as_slice).collect();
			run_merge_round_trip::<PackedBinaryGhash1x128b>(&round_refs, false);
		}
	}
}
