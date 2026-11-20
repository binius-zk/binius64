// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::Deref;

use binius_alloc::{Allocator, CollectIntoAllocVec, VecLike};
use binius_field::{BinaryField, PackedField};
use binius_iop::fri::FRIParams;
use binius_math::{FieldBuffer, FieldSlice, ntt::AdditiveNTT, reed_solomon::ReedSolomonCode};
use binius_utils::{
	rand::par_rand,
	rayon::{prelude::*, task_size::IndexedParallelIteratorExt},
};
use rand::{CryptoRng, rngs::StdRng};

/// Reverses the order of the lanes of a packed element.
fn reverse_lanes<P: PackedField>(word: P) -> P {
	P::from_fn(|lane| word.get(P::WIDTH - 1 - lane))
}

/// Reed-Solomon encodes one input oracle's interleaved message, in reversed order.
///
/// The committed message is `rev(message)` over all `message.log_len()` bits, `rev(m)[i] =
/// m[2^log_len - 1 - i]`, batch bits included. The reversal is done in the copy this encode makes
/// anyway, so `message` itself is neither modified nor required in reversed order. The FRI that
/// opens the codeword folds with the complement `1 - ρ` of each challenge that binds a message
/// variable, which opens the natural-order `message` at `ρ`.
///
/// `params` are the (possibly batched) FRI parameters and `oracle_index` selects which oracle of
/// [`FRIParams::input_oracles`] is encoded. The oracle is Reed–Solomon encoded at its own
/// dimension — which may be smaller than the batched code's reduced dimension — over the same
/// subspace and rate; the lift to the reduced dimension is applied later during the combined fold.
///
/// The returned codeword is committed by sending it over a Merkle channel (via
/// `MerkleIPProverChannel::send_merkle_commitment`) with one interleaved coset of
/// `2^log_batch_size` scalars per leaf.
///
/// ## Arguments
///
/// * `params` - the (possibly batched) FRI protocol parameters.
/// * `oracle_index` - the index into [`FRIParams::input_oracles`] of the oracle being encoded.
/// * `ntt` - the additive NTT for Reed-Solomon encoding.
/// * `message` - the interleaved message to encode, in natural order.
/// * `alloc` - the allocator the codeword is drawn from.
///
/// ## Preconditions
///
/// * `message.log_len()` must equal the oracle's committed message length,
///   `params.rs_code().log_dim() - log_lift + log_batch_size`.
pub fn encode_interleaved<F, P, NTT, A>(
	params: &FRIParams<F>,
	oracle_index: usize,
	ntt: &NTT,
	message: FieldSlice<'_, P>,
	alloc: &A,
) -> FieldBuffer<P, A::Vec<P>>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	A: Allocator,
{
	let oracle_spec = &params.input_oracles()[oracle_index];
	let log_batch_size = oracle_spec.log_batch_size();
	// The oracle's own codeword dimension is the reduced dimension minus its lift; its committed
	// (interleaved) message length is that plus the batch size.
	let oracle_log_dim = params.rs_code().log_dim() - oracle_spec.log_lift;

	assert_eq!(
		message.log_len(),
		oracle_log_dim + log_batch_size,
		"precondition: interleaved message length must match the oracle's spec"
	);

	// Write the reversed message into a store with room for the codeword, then encode in place.
	let log_output_len = oracle_log_dim + log_batch_size + params.rs_code().log_inv_rate();
	let mut words = alloc.alloc::<P>(1 << log_output_len.saturating_sub(P::LOG_WIDTH));
	let source = message.as_ref();
	if message.log_len() < P::LOG_WIDTH {
		// The message is the low `2^log_len` lanes of a single word, whose other lanes are dead.
		let len = 1 << message.log_len();
		let word = source[0];
		words.push(P::from_fn(|lane| word.get(if lane < len { len - 1 - lane } else { lane })));
	} else {
		// Word `k` of the result is word `n - 1 - k` of the message, with its lanes reversed.
		let n_words = source.len();
		words.spare_capacity_mut()[..n_words]
			.par_iter_mut()
			.with_min_task_bytes::<P>()
			.enumerate()
			.for_each(|(k, out)| {
				out.write(reverse_lanes(source[n_words - 1 - k]));
			});
		// SAFETY: the loop wrote each of the `n_words` slots exactly once.
		unsafe { words.set_len(n_words) };
	}

	encode_in_place(params, oracle_index, ntt, FieldBuffer::new(message.log_len(), words))
}

/// Reed-Solomon encodes the oracle's committed message, which `message` already holds.
///
/// ## Preconditions
///
/// * `message.log_len()` must equal the oracle's `log_dim() + log_batch_size`.
/// * The store of `message` must have room for the codeword.
fn encode_in_place<F, P, NTT, Data>(
	params: &FRIParams<F>,
	oracle_index: usize,
	ntt: &NTT,
	message: FieldBuffer<P, Data>,
) -> FieldBuffer<P, Data>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Data: VecLike<P>,
{
	let oracle_spec = &params.input_oracles()[oracle_index];
	let log_batch_size = oracle_spec.log_batch_size();
	let oracle_log_dim = params.rs_code().log_dim() - oracle_spec.log_lift;

	// Encode this oracle at its own dimension (≤ the batched code's reduced dimension), at the
	// same rate as the batched code. Both codes evaluate over the Gao-Mateer basis, the shorter
	// one over a prefix of the longer's, which is how the per-oracle codeword stays consistent
	// with the combined fold's lift. `encode_batch_in_place` checks the NTT agrees with that
	// domain.
	let rs_code = ReedSolomonCode::new(oracle_log_dim, params.rs_code().log_inv_rate());

	let _scope = tracing::debug_span!(
		"Reed–Solomon Encode",
		log_batch_size,
		log_dim = rs_code.log_dim(),
		log_inv_rate = rs_code.log_inv_rate(),
		field_bits = F::N_BITS,
	)
	.entered();

	rs_code.encode_batch_in_place(ntt, message, log_batch_size)
}

/// Output of [`encode_masked`]: the codeword of the interleaved (message, mask) pairs and the
/// generated mask.
///
/// Both buffers come from the allocator the encode was given, so a pooled prover holds neither on
/// the global heap.
#[derive(Debug)]
pub struct MaskedCodeword<P: PackedField, Data: Deref<Target = [P]> = Vec<P>> {
	/// The Reed-Solomon encoding of the interleaved (message ‖ mask) buffer.
	pub codeword: FieldBuffer<P, Data>,
	/// The generated random mask, of equal length to the message.
	pub mask: FieldBuffer<P, Data>,
}

/// Zips two packed elements: returns `[z_0, z_1]` where `z_0 || z_1` is `a_0, b_0, a_1, b_1, ...`.
fn zip_packed<P: PackedField>([a, b]: [P; 2]) -> [P; 2] {
	let (z_0, z_1) =
		(0..P::LOG_WIDTH).fold((a, b), |(a, b), log_block_len| a.interleave(b, log_block_len));
	[z_0, z_1]
}

/// Generates a random mask, interleaves it with the message, and Reed-Solomon encodes, in reversed
/// order.
///
/// This is used for zero-knowledge FRI commitments. The function generates a random mask of
/// equal length to the input message and interleaves `(rev(message)_i, rev(mask)_i)` as the
/// interleaved message (with `log_batch_size = 1`, the selector between message and mask being
/// the low bit and left uncomplemented), where `rev(m)[i] = m[2^log_len - 1 - i]`. It then
/// performs Reed-Solomon encoding. The reversal happens in the interleaving copy; see
/// [`encode_interleaved`]. The returned `mask` and the
/// input `message` stay in natural order.
///
/// ## Arguments
///
/// * `params` - the (possibly batched) FRI parameters.
/// * `oracle_index` - the index into [`FRIParams::input_oracles`] of the oracle being encoded; its
///   spec must have `log_batch_size == 1` and `log_msg_len - 1 == message.log_len()`.
/// * `ntt` - the additive NTT for Reed-Solomon encoding
/// * `message` - the raw message to encode (not doubled), in natural order
/// * `rng` - cryptographic RNG for mask generation
/// * `alloc` - the allocator the mask and the codeword are drawn from
pub fn encode_masked<F, P, NTT, A>(
	params: &FRIParams<F>,
	oracle_index: usize,
	ntt: &NTT,
	message: FieldSlice<'_, P>,
	mut rng: impl CryptoRng,
	alloc: &A,
) -> MaskedCodeword<P, A::Vec<P>>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	A: Allocator,
{
	let oracle_spec = &params.input_oracles()[oracle_index];
	assert_eq!(oracle_spec.log_batch_size(), 1, "encode_masked requires log_batch_size == 1");
	// With batch size 1, the oracle's own codeword dimension (reduced dimension minus its lift) is
	// exactly the bare message length; the mask is interleaved internally.
	let oracle_log_dim = params.rs_code().log_dim() - oracle_spec.log_lift;
	assert_eq!(
		oracle_log_dim,
		message.log_len(),
		"encode_masked requires the oracle's message dimension to match the message length"
	);

	// Generate random mask of equal length to message.
	let log_len = message.log_len();
	let packed_len = 1usize << log_len.saturating_sub(P::LOG_WIDTH);

	let gen_mask_scope = tracing::debug_span!("Generate random mask").entered();
	let mask_values =
		par_rand::<StdRng, _, _>(packed_len, &mut rng, P::random).collect_into_alloc_vec(alloc);
	let mask = FieldBuffer::new(log_len, mask_values);
	drop(gen_mask_scope);

	// Interleave `(rev(m)_i, rev(ω)_i)`, so entry `2i` is the reversed message and `2i + 1` the
	// reversed mask, and the batch index is the low bit. The store has room for the codeword.
	let log_output_len = log_len + 1 + params.rs_code().log_inv_rate();
	let mut combined_values = alloc.alloc::<P>(1 << log_output_len.saturating_sub(P::LOG_WIDTH));
	if log_len < P::LOG_WIDTH {
		let len = message.len();
		combined_values.push(P::from_scalars(
			(0..len).flat_map(|i| [message.get(len - 1 - i), mask.get(len - 1 - i)]),
		));
	} else {
		let _scope = tracing::debug_span!("Interleave message and mask").entered();
		// Output words `2k` and `2k + 1` zip input words `n - 1 - k` of message and mask, each with
		// its lanes reversed.
		let message_words = message.as_ref();
		let mask_words = mask.as_ref();
		let n_words = message_words.len();
		let len = 2 * n_words;
		// The allocator may hand back more capacity than requested, so bound the spare slice.
		combined_values.spare_capacity_mut()[..len]
			.par_chunks_exact_mut(2)
			.with_min_task_bytes::<[P; 2]>()
			.enumerate()
			.for_each(|(k, out)| {
				let [even, odd] = zip_packed([
					reverse_lanes(message_words[n_words - 1 - k]),
					reverse_lanes(mask_words[n_words - 1 - k]),
				]);
				out[0].write(even);
				out[1].write(odd);
			});
		// SAFETY: the chunks cover all `len` slots, and each was written exactly once.
		unsafe { combined_values.set_len(len) };
	}
	let combined = FieldBuffer::new(log_len + 1, combined_values);

	let codeword = encode_in_place(params, oracle_index, ntt, combined);

	MaskedCodeword { codeword, mask }
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_alloc::GlobalAllocator;
	use binius_field::{Ghash128b, PackedGhash1x128b, PackedGhash2x128b, PackedGhash4x128b};
	use binius_iop::fri::FRIParams;
	use binius_math::{
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::random_field_buffer,
	};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	/// `rev(m)[i] = m[2^log_len - 1 - i]`, by scalars.
	fn reversed<P: PackedField>(buffer: &FieldBuffer<P>) -> FieldBuffer<P> {
		FieldBuffer::from_values(
			&buffer
				.iter_scalars()
				.collect::<Vec<_>>()
				.into_iter()
				.rev()
				.collect::<Vec<_>>(),
		)
	}

	/// The reversal splits on whether the message spans whole packed elements, so this runs three
	/// packed widths over lengths on both sides of each.
	#[test]
	fn test_encode_interleaved_commits_the_reversed_message() {
		fn check<P: PackedField<Scalar = Ghash128b>>(log_dim: usize, log_batch_size: usize) {
			let mut rng = StdRng::seed_from_u64(log_dim as u64);
			let log_inv_rate = 1;
			let params = FRIParams::new(
				ReedSolomonCode::new(log_dim, log_inv_rate),
				log_batch_size,
				vec![],
				3,
			);
			let ntt =
				NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(log_dim + log_inv_rate));

			let message = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);
			let codeword =
				encode_interleaved(&params, 0, &ntt, message.as_view(), &GlobalAllocator);
			let expected = params.rs_code().encode_batch(
				&ntt,
				reversed(&message).as_view(),
				log_batch_size,
				&GlobalAllocator,
			);
			assert_eq!(codeword, expected, "P::LOG_WIDTH={}, log_dim={log_dim}", P::LOG_WIDTH);
		}

		for log_dim in 0..=6 {
			check::<PackedGhash1x128b>(log_dim, 0);
			check::<PackedGhash2x128b>(log_dim, 0);
			check::<PackedGhash4x128b>(log_dim, 0);
			check::<PackedGhash4x128b>(log_dim, 2);
		}
	}

	#[test]
	fn test_encode_masked_commits_the_interleaved_reversed_message_and_mask() {
		fn check<P: PackedField<Scalar = Ghash128b>>(log_dim: usize) {
			let mut rng = StdRng::seed_from_u64(log_dim as u64);
			let log_inv_rate = 1;
			let params = FRIParams::new(ReedSolomonCode::new(log_dim, log_inv_rate), 1, vec![], 3);
			let ntt =
				NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(log_dim + log_inv_rate));

			let message = random_field_buffer::<P>(&mut rng, log_dim);
			let MaskedCodeword { codeword, mask } =
				encode_masked(&params, 0, &ntt, message.as_view(), &mut rng, &GlobalAllocator);

			// Entry `2i` is the reversed message and `2i + 1` the reversed mask.
			let interleaved =
				iter::zip(reversed(&message).iter_scalars(), reversed(&mask).iter_scalars())
					.flat_map(|(message_i, mask_i)| [message_i, mask_i])
					.collect::<Vec<_>>();
			let expected = params.rs_code().encode_batch(
				&ntt,
				FieldBuffer::<P>::from_values(&interleaved).as_view(),
				1,
				&GlobalAllocator,
			);
			assert_eq!(codeword, expected, "P::LOG_WIDTH={}, log_dim={log_dim}", P::LOG_WIDTH);
		}

		for log_dim in 0..=6 {
			check::<PackedGhash1x128b>(log_dim);
			check::<PackedGhash2x128b>(log_dim);
			check::<PackedGhash4x128b>(log_dim);
		}
	}

	#[test]
	fn test_encode_masked() {
		type P = PackedGhash1x128b;

		let mut rng = StdRng::seed_from_u64(42);

		let log_dim = 6;
		let log_inv_rate = 1;
		let log_batch_size = 1;
		let n_test_queries = 3;

		let domain_context = GaoMateerOnTheFly::generate(log_dim + log_inv_rate);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		let params = FRIParams::new(
			ReedSolomonCode::new(log_dim, log_inv_rate),
			log_batch_size,
			vec![2, 2, 2],
			n_test_queries,
		);

		assert_eq!(params.log_batch_size(), 1);
		assert_eq!(params.rs_code().log_dim(), log_dim);

		let message = random_field_buffer::<P>(&mut rng, log_dim);

		let output: MaskedCodeword<P> =
			encode_masked(&params, 0, &ntt, message.as_view(), &mut rng, &GlobalAllocator);

		// Verify mask has correct dimensions.
		assert_eq!(output.mask.log_len(), log_dim);

		// Verify the codeword has expected length (log_dim + log_batch_size + log_inv_rate).
		assert_eq!(output.codeword.log_len(), log_dim + log_batch_size + log_inv_rate);
	}
}
