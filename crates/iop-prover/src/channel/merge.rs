// Copyright 2026 The Binius Developers

//! A channel decorator that merges oracles committed within the same interaction round.
//!
//! This is the prover-side half of a matching pair.
//! Whatever it commits here must be received by a matching verifier-side decorator.

use std::ops::DerefMut;

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_iop::channel::{
	OracleSchedule, OracleSpec,
	merge::{Placement, place_oracles},
};
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_math::{FieldBuffer, FieldSlice, FieldVec, StructuredBuffer};

use crate::channel::IOPProverChannel;

/// A handle to an oracle sent through the merging decorator.
#[derive(Debug, Clone, Copy)]
pub struct MergeOracle {
	index: usize,
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
/// It is forwarded unchanged, at zero cost: its buffer moves to the underlying channel, and takes
/// and returns of it pass straight through.
///
/// A round is masked as a whole, never partly.
///
/// So a round carrying any witness data is masked in full by the underlying channel.
/// The verifier-side decorator is where that choice is made.
///
/// # Timing
///
/// A round's oracles are committed the moment its last oracle arrives.
///
/// The [`OracleSchedule`] says where each round ends, so no challenge sample is needed to find
/// the boundary.
/// A real Fiat-Shamir transcript can therefore absorb the commitment before the next challenge.
///
/// The underlying channel owns the combined buffer once it is committed.
/// The constituents' own buffers stay with this channel, which takes each one out once and drops it
/// when it comes back.
///
/// # Opening
///
/// A verifier only ever holds a formula for a transparent polynomial.
///
/// This side holds the actual coefficients instead.
/// A constituent's own transparent polynomial becomes one for the combined oracle.
/// It is the constituent's values at the constituent's own block, and zero everywhere else.
/// This side forwards it to the underlying channel as a zero-padded structure, never writing the
/// zeros.
///
/// That placement is the same polynomial a verifier reaches by formula.
/// One side holds the block as explicit values.
/// The other evaluates it on demand.
pub struct MergeProverChannel<'a, P, A, C>
where
	P: PackedField,
	A: Allocator,
	C: IOPProverChannel<P, A>,
{
	/// The underlying channel every oracle, challenge, and opening passes through.
	inner: C,

	/// The fine-grained oracles this channel's caller will send, grouped into rounds.
	schedule: &'a OracleSchedule,

	/// Where each oracle of the schedule lands, in arrival order.
	placements: Vec<Placement>,

	/// The allocator this channel draws its combined buffers from.
	alloc: A,

	/// The open round's combined buffer, filled in as its oracles arrive.
	///
	/// Allocated on the round's first oracle and committed on its last.
	open: Option<FieldVec<P, A>>,

	/// How many oracles have been sent so far.
	n_sent: usize,

	/// The underlying channel's handle for every round committed so far, in commit order.
	groups: Vec<C::Oracle>,

	/// Every oracle's own buffer, in arrival order, while this channel holds it.
	///
	/// `None` for an oracle forwarded as a round of its own, and for one taken out.
	originals: Vec<Option<FieldVec<P, A>>>,
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
	/// * `inner` — the channel every combined oracle is committed to, already configured with
	///   `schedule.merged_specs()`.
	/// * `schedule` — every oracle this channel's caller will pass through, grouped into rounds.
	/// * `alloc` — where this channel draws its combined buffers from.
	///
	/// # Panics
	///
	/// Panics if `inner` is not configured with `schedule.merged_specs()`.
	pub fn new(inner: C, schedule: &'a OracleSchedule, alloc: A) -> Self {
		assert_eq!(
			inner.remaining_oracle_specs(),
			schedule.merged_specs(),
			"inner channel must be configured with the schedule's merged specs"
		);
		Self {
			inner,
			schedule,
			placements: place_oracles(schedule),
			alloc,
			open: None,
			n_sent: 0,
			groups: Vec::new(),
			originals: Vec::new(),
		}
	}

	/// Whether the oracle was forwarded to the underlying channel as a round of its own.
	fn is_forwarded(&self, oracle: MergeOracle) -> bool {
		self.placements[oracle.index].combined_log_len
			== self.schedule.specs()[oracle.index].log_msg_len
	}

	/// Returns the underlying channel.
	///
	/// # Panics
	///
	/// Panics if any declared oracle has not yet been sent.
	pub fn into_inner(self) -> C {
		let n_remaining = self.placements.len() - self.n_sent;
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

	// Copy the block across.
	// Everywhere else in the destination stays as it was.
	if n >= P::LOG_WIDTH {
		// A source at least one packed word wide occupies whole words, so copy them across.
		let n_words = 1 << (n - P::LOG_WIDTH);
		let word_offset = block_index * n_words;
		dst.as_mut()[word_offset..word_offset + n_words].copy_from_slice(src.as_ref());
	} else {
		// A source narrower than a packed word shares one, so place its scalars individually.
		for i in 0..1usize << n {
			dst.set(offset + i, src.get(i));
		}
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
		&self.schedule.specs()[self.n_sent..]
	}

	fn send_oracle(&mut self, buffer: FieldVec<P, A>) -> Self::Oracle {
		// Every oracle this channel will send is declared up front.
		//
		// Reject anything past that count.
		// Do not silently accept an undeclared oracle.
		let remaining = self.remaining_oracle_specs();
		assert!(!remaining.is_empty(), "send_oracle called but no remaining oracle specs");
		assert_eq!(buffer.log_len(), remaining[0].log_msg_len, "oracle size must match its spec");

		let index = self.n_sent;
		self.n_sent += 1;
		let Placement {
			round,
			block_index,
			combined_log_len,
		} = self.placements[index];

		// A round of its own is forwarded unchanged.
		let oracle = MergeOracle { index };
		if self.is_forwarded(oracle) {
			self.groups.push(self.inner.send_oracle(buffer));
			self.originals.push(None);
			return oracle;
		}

		// Copy the data into its block of the round's combined buffer.
		//
		// Its round may not commit until a later oracle arrives.
		let combined = self
			.open
			.get_or_insert_with(|| FieldBuffer::zeros_in(&self.alloc, combined_log_len));
		place_block(combined, buffer.as_view(), block_index);
		self.originals.push(Some(buffer));

		// The last oracle of its round commits the whole round as one oracle.
		let is_last_of_round = self
			.placements
			.get(index + 1)
			.is_none_or(|next| next.round != round);
		if is_last_of_round {
			let combined = self
				.open
				.take()
				.expect("open round buffer was just inserted");
			self.groups.push(self.inner.send_oracle(combined));
		}

		oracle
	}

	fn prove_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: StructuredBuffer<P, A::Vec<P>>,
		claim: P::Scalar,
	) {
		let n_i = self.schedule.specs()[oracle.index].log_msg_len;
		assert_eq!(
			transparent.log_len(),
			n_i,
			"transparent log_len mismatch: expected {n_i}, got {}",
			transparent.log_len()
		);
		let Placement {
			round,
			block_index,
			combined_log_len,
		} = self.placements[oracle.index];

		// The constituent's transparent is zero outside the constituent's own block of the
		// combined oracle. So the inner product equals the original claim exactly.
		let padded = StructuredBuffer::ZeroPadded {
			inner: Box::new(transparent),
			log_n_blocks: combined_log_len - n_i,
			index: block_index,
		};

		let outer = self.groups[round].clone();
		self.inner.prove_oracle_relation(outer, padded, claim);
	}

	fn take_oracle(&mut self, oracle: Self::Oracle) -> FieldVec<P, A> {
		let round = self.placements[oracle.index].round;
		assert!(round < self.groups.len(), "oracle {} is in a round still open", oracle.index);
		if self.is_forwarded(oracle) {
			// The round was forwarded, so its buffer is the underlying channel's.
			self.inner.take_oracle(self.groups[round].clone())
		} else {
			self.originals[oracle.index]
				.take()
				.unwrap_or_else(|| panic!("oracle {} is already taken", oracle.index))
		}
	}

	fn return_oracle(&mut self, oracle: Self::Oracle, buffer: FieldVec<P, A>) {
		// A merged oracle's copy in the combined buffer is what gets opened, so its own buffer is
		// no longer needed and is dropped here.
		if self.is_forwarded(oracle) {
			let outer = self.groups[self.placements[oracle.index].round].clone();
			self.inner.return_oracle(outer, buffer);
		}
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_field::{
		BinaryField, Field, Ghash128b, PackedField, PackedGhash1x128b, PackedGhash4x128b,
	};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		basefold::compiler::BaseFoldVerifierCompiler,
		channel::{
			IOPVerifierChannel, OracleSchedule, OracleSpec, merge::MergeVerifierChannel,
			naive::NaiveVerifierChannel,
		},
		fri::MinProofSizeStrategy,
		merkle_tree::BinaryMerkleTreeScheme,
	};
	use binius_ip::channel::IPVerifierChannel;
	use binius_ip_prover::channel::IPProverChannel;
	use binius_math::{
		FieldBuffer,
		inner_product::inner_product_buffers,
		multilinear::eq::eq_ind_partial_eval,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use proptest::prelude::*;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::{IOPProverChannel, MergeProverChannel};
	use crate::{basefold::compiler::BaseFoldProverCompiler, channel::naive::NaiveProverChannel};

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
		let transparent = eq_ind_partial_eval::<P>(&point);
		let claim = inner_product_buffers(&buffer, &transparent);
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
		let data: Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> = fine_sizes
			.iter()
			.map(|&n| generate_oracle_data::<F, P, _>(&mut rng, n))
			.collect();

		// The round layout both sides are driven by.
		//
		// The underlying channels see one combined oracle per round.
		let mut schedule = OracleSchedule::new();
		for sizes in rounds {
			for &n in *sizes {
				schedule.push(OracleSpec::new(n));
			}
			schedule.end_round();
		}
		let coarse_specs = schedule.merged_specs();

		// Prover side.
		//
		// Send every oracle round by round.
		// Sample a challenge between rounds.
		// Each round commits as it goes.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let naive_prover = NaiveProverChannel::new(&mut prover_transcript, coarse_specs.clone());
		let mut merge_prover = MergeProverChannel::new(naive_prover, &schedule, GlobalAllocator);

		let mut oracles = Vec::new();
		let mut index = 0;
		for sizes in rounds {
			for _ in *sizes {
				let (buffer, _, _) = &data[index];
				oracles.push(merge_prover.send_oracle(buffer.clone()));
				index += 1;
			}
			IPProverChannel::sample(&mut merge_prover);
		}
		// Every oracle's buffer comes back as sent, whether its round was forwarded or merged.
		for (&oracle, (buffer, _, _)) in iter::zip(&oracles, &data) {
			let taken = merge_prover.take_oracle(oracle);
			assert_eq!(&taken, buffer);
			merge_prover.return_oracle(oracle, taken);
		}
		for (&oracle, (_, transparent, claim)) in iter::zip(&oracles, &data) {
			merge_prover.prove_oracle_relation(oracle, transparent.clone().into(), *claim);
		}
		merge_prover.into_inner().finish();

		// Verifier side.
		//
		// Mirror the exact same round boundaries.
		// Both sides then sample from the same transcript positions.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let naive_verifier = NaiveVerifierChannel::new(&mut verifier_transcript, &coarse_specs);
		let mut merge_verifier = MergeVerifierChannel::new(naive_verifier, &schedule);

		let mut v_oracles = Vec::new();
		for sizes in rounds {
			for &n in *sizes {
				v_oracles.push(merge_verifier.recv_oracle(1 << n, n, true).unwrap());
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
						let eq = eq_ind_partial_eval::<P>(point);
						inner_product_buffers(&transparent, &eq)
					}),
					claim,
				)
				.expect("verification only ever queues a relation, it does not check it here");
		}
		merge_verifier.into_inner().finish();
	}

	#[test]
	fn single_oracle_round_trip() {
		// A single round holding a single oracle.
		// The degenerate case, where merging has nothing to do.
		run_merge_round_trip::<PackedGhash1x128b>(&[&[6]], false);
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
		run_merge_round_trip::<PackedGhash1x128b>(&[&[3, 3], &[4, 2, 2], &[1]], false);
	}

	#[test]
	fn multi_round_round_trip_narrow_packing() {
		// The same three rounds, under a wider packing width.
		//
		// Some oracles are narrower than one packed field element.
		// Placement then writes into part of one, not a whole chunk.
		const {
			assert!(
				PackedGhash4x128b::LOG_WIDTH > 0,
				"the fixture needs sub-packed-width oracle sizes to appear"
			);
		};
		run_merge_round_trip::<PackedGhash4x128b>(&[&[3, 3], &[4, 2, 2], &[1]], false);
	}

	#[test]
	fn zero_variable_oracle_round_trip() {
		// A round mixing two single-scalar oracles with a larger one.
		// The smallest possible oracle size.
		run_merge_round_trip::<PackedGhash1x128b>(&[&[0, 0, 3]], false);
	}

	#[test]
	#[should_panic(expected = "NaiveVerifierChannel: inner product verification failed")]
	fn tampered_claim_is_rejected() {
		// Corrupting one oracle's claim must fail the whole round trip.
		// It must not be silently absorbed by the merge.
		run_merge_round_trip::<PackedGhash1x128b>(&[&[3, 3], &[4, 2, 2]], true);
	}

	#[test]
	fn multiple_relations_on_merged_oracle() {
		type F = Ghash128b;
		type P = PackedGhash1x128b;

		// Two oracles, merged into a single round.
		// Each carries two independent claims, rather than just one.
		let mut rng = StdRng::seed_from_u64(0);
		let mut schedule = OracleSchedule::new();
		schedule.push(OracleSpec::new(4));
		schedule.push(OracleSpec::new(3));
		schedule.end_round();
		let (buffer_1, _, _) = generate_oracle_data::<F, P, _>(&mut rng, 4);
		let (buffer_2, _, _) = generate_oracle_data::<F, P, _>(&mut rng, 3);

		let relations_1: Vec<(FieldBuffer<P>, F)> = (0..2)
			.map(|_| {
				let point = random_scalars::<F>(&mut rng, 4);
				let transparent = eq_ind_partial_eval::<P>(&point);
				let claim = inner_product_buffers(&buffer_1, &transparent);
				(transparent, claim)
			})
			.collect();
		let relations_2: Vec<(FieldBuffer<P>, F)> = (0..2)
			.map(|_| {
				let point = random_scalars::<F>(&mut rng, 3);
				let transparent = eq_ind_partial_eval::<P>(&point);
				let claim = inner_product_buffers(&buffer_2, &transparent);
				(transparent, claim)
			})
			.collect();

		// One round, sized to fit both oracles.
		// 2^4 + 2^3 = 24, rounded up to 2^5.
		let coarse_specs = schedule.merged_specs();
		assert_eq!(coarse_specs, [OracleSpec::new(5)]);

		// Prover side.
		//
		// Both oracles arrive in the same round.
		// All four claims are proved before the channel finishes.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let naive_prover = NaiveProverChannel::new(&mut prover_transcript, coarse_specs.clone());
		let mut merge_prover = MergeProverChannel::new(naive_prover, &schedule, GlobalAllocator);

		let oracle_1 = merge_prover.send_oracle(buffer_1);
		let oracle_2 = merge_prover.send_oracle(buffer_2);
		for (transparent, claim) in &relations_1 {
			merge_prover.prove_oracle_relation(oracle_1, transparent.clone().into(), *claim);
		}
		for (transparent, claim) in &relations_2 {
			merge_prover.prove_oracle_relation(oracle_2, transparent.clone().into(), *claim);
		}
		merge_prover.into_inner().finish();

		// Verifier side.
		//
		// The same two oracles, each checked against its own two claims.
		// In the same order the prover produced them.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let naive_verifier = NaiveVerifierChannel::new(&mut verifier_transcript, &coarse_specs);
		let mut merge_verifier = MergeVerifierChannel::new(naive_verifier, &schedule);

		let v_oracle_1 = merge_verifier.recv_oracle(1 << 4, 4, true).unwrap();
		let v_oracle_2 = merge_verifier.recv_oracle(1 << 3, 3, true).unwrap();
		for (transparent, claim) in relations_1 {
			merge_verifier
				.verify_oracle_relation(
					v_oracle_1,
					Box::new(move |point: &[F]| {
						let eq = eq_ind_partial_eval::<P>(point);
						inner_product_buffers(&transparent, &eq)
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
						let eq = eq_ind_partial_eval::<P>(point);
						inner_product_buffers(&transparent, &eq)
					}),
					claim,
				)
				.unwrap();
		}
		merge_verifier.into_inner().finish();
	}

	/// Runs a prove-then-verify round trip of the merge channels over BaseFold, rather than the
	/// naive channel.
	///
	/// BaseFold accumulates each forwarded relation into its own block of the combined oracle.
	/// Every oracle carries two relations, so both the whole-oracle accumulator of a single-oracle
	/// round and the zero-started accumulator of a merged round are exercised.
	fn run_merge_over_basefold<P>(rounds: &[&[usize]]) -> bool
	where
		P: PackedField<Scalar = Ghash128b>,
	{
		type F = Ghash128b;
		const LOG_INV_RATE: usize = 1;
		const N_TEST_QUERIES: usize = 32;

		let mut rng = StdRng::seed_from_u64(0);
		let mut schedule = OracleSchedule::new();
		for sizes in rounds {
			for &n in *sizes {
				schedule.push(OracleSpec::new_zk(n));
			}
			schedule.end_round();
		}
		let data = schedule
			.specs()
			.iter()
			.map(|spec| {
				let buffer = random_field_buffer::<P>(&mut rng, spec.log_msg_len);
				let relations = (0..2)
					.map(|_| {
						let point = random_scalars::<F>(&mut rng, spec.log_msg_len);
						let transparent = eq_ind_partial_eval::<P>(&point);
						let claim = inner_product_buffers(&buffer, &transparent);
						(transparent, claim)
					})
					.collect::<Vec<_>>();
				(buffer, relations)
			})
			.collect::<Vec<_>>();

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&BinaryMerkleTreeScheme::<F, StdHashSuite>::new(),
			schedule.merged_specs(),
			LOG_INV_RATE,
			N_TEST_QUERIES,
			&MinProofSizeStrategy,
		);
		let ntt = NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(
			verifier_compiler.max_log_domain_size(),
		));
		let prover_compiler =
			BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

		// Prover side.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let basefold_prover = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
				&mut prover_transcript,
				StdRng::seed_from_u64(1),
				GlobalAllocator,
			);
		let mut merge_prover = MergeProverChannel::new(basefold_prover, &schedule, GlobalAllocator);
		let mut oracles = Vec::new();
		let mut data_iter = data.iter();
		for sizes in rounds {
			for (buffer, _) in data_iter.by_ref().take(sizes.len()) {
				oracles.push(merge_prover.send_oracle(buffer.clone()));
			}
			IPProverChannel::sample(&mut merge_prover);
		}
		for (&oracle, (_, relations)) in iter::zip(&oracles, &data) {
			for (transparent, claim) in relations {
				merge_prover.prove_oracle_relation(oracle, transparent.clone().into(), *claim);
			}
		}
		merge_prover.into_inner().finish();

		// Verifier side.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let basefold_verifier = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(
				&mut verifier_transcript,
			);
		let mut merge_verifier = MergeVerifierChannel::new(basefold_verifier, &schedule);
		let mut v_oracles = Vec::new();
		for sizes in rounds {
			for &n in *sizes {
				v_oracles.push(merge_verifier.recv_oracle(1 << n, n, true).unwrap());
			}
			IPVerifierChannel::sample(&mut merge_verifier);
		}
		for (&oracle, (_, relations)) in iter::zip(&v_oracles, data) {
			for (transparent, claim) in relations {
				merge_verifier
					.verify_oracle_relation(
						oracle,
						Box::new(move |point: &[F]| {
							let eq = eq_ind_partial_eval::<P>(point);
							inner_product_buffers(&transparent, &eq)
						}),
						claim,
					)
					.expect("verification only ever queues a relation, it does not check it here");
			}
		}
		merge_verifier.into_inner().finish().is_ok()
	}

	#[test]
	fn merge_over_basefold_round_trip() {
		let rounds: &[&[usize]] = &[&[4, 2, 2], &[5], &[3, 3, 0]];
		assert!(run_merge_over_basefold::<PackedGhash1x128b>(rounds));

		// Under the wider packing, the size-0 and size-2 blocks are narrower than a packed word.
		assert!(run_merge_over_basefold::<PackedGhash4x128b>(rounds));
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
			run_merge_round_trip::<PackedGhash1x128b>(&round_refs, false);

			// The wider packing also mixes word-aligned oracles with sub-word ones,
			// which is where the two placement paths meet.
			run_merge_round_trip::<PackedGhash4x128b>(&round_refs, false);
		}
	}
}
