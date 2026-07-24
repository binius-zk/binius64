// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::Deref;

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{AESTowerField8b as B8, BinaryField, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::BinarySubspace;
use binius_utils::checked_arithmetics::checked_log_2;
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

	let log_constraint_count = checked_log_2(a.len());

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
		prover_message_domain,
	);

	prover.prove_with_channel(channel, alloc)
}
