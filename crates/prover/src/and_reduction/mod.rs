// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::Deref;

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{AESTowerField8b as B8, BinaryField, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::BinarySubspace;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{
	config::PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES, protocols::bitand::AndCheckOutput,
};

use self::prover::OblongZerocheckProver;

mod ntt_lookup;
pub mod prover;
pub mod sumcheck_round_messages;
pub use ntt_lookup::NTTLookup;

/// Proves the AND constraint reduction over the two operand columns `A` and `B`.
///
/// This wraps [`OblongZerocheckProver`], the univariate-skip zerocheck kernel, so both the
/// single-instance prover and the M4 batch prover route their AND check through one entry point.
/// The `C` operand is never passed: the reduction derives `C = A & B` word-by-word, which is sound
/// because folding is F2-linear on word bits (see [`OblongZerocheckProver::new`]).
///
/// The columns are generic over their backing store `Data` (anything that dereferences to
/// `[Word]`), so pooled buffers and plain `Vec<Word>` are both accepted and moved into the kernel.
/// The univariate-skip domain is built internally as
/// `BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1)`, matching the domain the shift reduction
/// folds its bit axis over.
///
/// The two columns must have equal length, but that length need not be a power of two: the
/// constraint axis is the next power of two and the rows past the columns' end read as zero. Such a
/// row forces the derived `C = A & B` to zero as well, so `A * B - C` vanishes on it and the
/// reduction skips it rather than folding zeros. Passing an already-padded column is therefore
/// equivalent, just slower.
///
/// See [`binius_verifier::protocols::bitand`] for the protocol specification and
/// [`AndCheckOutput`] for the output shape.
pub fn prove<A, F, PChallenge, Channel, Data>(
	columns: [Data; 2],
	channel: &mut Channel,
	alloc: &A,
) -> AndCheckOutput<F>
where
	A: Allocator,
	F: BinaryField + From<B8>,
	PChallenge: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	Data: Deref<Target = [Word]>,
{
	// The univariate-skip domain spans one dimension above the 64-bit word.
	let prover_message_domain = BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1);
	let [a, b] = columns;

	// The column length is the row count: one row per constraint for the single-instance prover,
	// one per (instance, constraint) pair for the M4 batch prover. The constraint axis rounds that
	// count up to a power of two, and the rows in between read as zero.
	assert_eq!(a.len(), b.len(), "the operand columns must have equal length");
	let log_constraint_count = log2_ceil_usize(a.len());

	// Pin the first few zerocheck coordinates to fixed small-field elements (friendly challenges),
	// and draw the rest from the large field. The prover and verifier pin and draw the same split,
	// in the same order.
	let n_extra_zerocheck_challenges =
		log_constraint_count.saturating_sub(PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len());
	let big_field_zerocheck_challenges = channel.sample_many(n_extra_zerocheck_challenges);

	let prover = OblongZerocheckProver::<_, PChallenge, _>::new(
		log_constraint_count,
		a,
		b,
		big_field_zerocheck_challenges,
		&prover_message_domain,
	);

	prover.prove_with_channel(channel, alloc)
}

#[cfg(test)]
mod tests {
	use std::{array, iter::repeat_with};

	use binius_compute::GlobalAllocator;
	use binius_field::arch::OptimalPackedB128;
	use binius_transcript::{ProverTranscript, VerifierTranscript};
	use binius_verifier::{
		config::{B128, StdChallenger},
		verify_bitand_reduction,
	};
	use rand::prelude::*;

	use super::*;

	// A column whose length is not a power of two reduces exactly as the same column zero-padded up
	// to one: byte-identical transcript, identical reduced claim, and the claim still verifies
	// against the padded row count the verifier derives from the constraint system.
	//
	// This is the guarantee that lets the M4 prover stop materializing the padding rows
	// (BINIUS-388) without moving anything on the wire.
	#[test]
	fn unpadded_columns_reduce_identically_to_padded_ones() {
		let mut rng = StdRng::seed_from_u64(0);

		// Row counts crossing the round-1 window of 128 words and the fold's packed width: inside a
		// single window, one word past a whole window, and several windows plus a partial one.
		for n_rows in [3, 100, 129, 1000] {
			let [a, b] = array::from_fn(|_| {
				repeat_with(|| Word(rng.random()))
					.take(n_rows)
					.collect::<Vec<_>>()
			});

			let log_rows = log2_ceil_usize(n_rows);
			let pad = |words: &[Word]| {
				let mut padded = words.to_vec();
				padded.resize(1 << log_rows, Word::ZERO);
				padded
			};

			// One proof per shape, each from a fresh transcript over the same challenger seed.
			let run = |columns: [Vec<Word>; 2]| {
				let mut transcript = ProverTranscript::new(StdChallenger::default());
				let output = prove::<_, B128, OptimalPackedB128, _, _>(
					columns,
					&mut transcript,
					&GlobalAllocator,
				);
				(output, transcript.finalize())
			};
			let (unpadded_output, unpadded_proof) = run([a.clone(), b.clone()]);
			let (padded_output, padded_proof) = run([pad(&a), pad(&b)]);

			assert_eq!(unpadded_output, padded_output, "claim differs at n_rows = {n_rows}");
			assert_eq!(unpadded_proof, padded_proof, "transcript differs at n_rows = {n_rows}");

			let mut verifier_transcript =
				VerifierTranscript::new(StdChallenger::default(), unpadded_proof);
			let verify_output = verify_bitand_reduction(
				log_rows,
				&BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1).isomorphic::<B128>(),
				&mut verifier_transcript,
			)
			.unwrap();
			verifier_transcript
				.finalize()
				.expect("no trailing proof data");
			assert_eq!(unpadded_output, verify_output, "verifier disagrees at n_rows = {n_rows}");
		}
	}
}
