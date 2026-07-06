// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_field::{BinaryField, PackedField};
use binius_iop::fri::FRIParams;
use binius_math::{FieldBuffer, FieldSlice, ntt::AdditiveNTT, reed_solomon::ReedSolomonCode};
use binius_utils::{rand::par_rand, rayon::prelude::*};
use rand::{CryptoRng, rngs::StdRng};

/// Reed-Solomon encodes one input oracle's interleaved message.
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
/// * `message` - the interleaved message to encode.
///
/// ## Preconditions
///
/// * `message.log_len()` must equal the oracle's committed message length,
///   `params.rs_code().log_dim() - log_lift + log_batch_size`.
pub fn encode_interleaved<F, P, NTT>(
	params: &FRIParams<F>,
	oracle_index: usize,
	ntt: &NTT,
	message: FieldSlice<P>,
) -> FieldBuffer<P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
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

	// Encode this oracle at its own dimension (≤ the batched code's reduced dimension), over the
	// same subspace and rate as the batched code. `encode_batch` checks the subspace matches the
	// NTT, which is how the per-oracle codeword stays consistent with the combined fold's lift.
	let rs_code =
		ReedSolomonCode::with_ntt_subspace(ntt, oracle_log_dim, params.rs_code().log_inv_rate());

	let _scope = tracing::debug_span!(
		"FRI Encode",
		log_batch_size,
		log_dim = rs_code.log_dim(),
		log_inv_rate = rs_code.log_inv_rate(),
		field_bits = F::N_BITS,
	)
	.entered();

	tracing::debug_span!("Reed-Solomon Encode")
		.in_scope(|| rs_code.encode_batch(ntt, message.to_ref(), log_batch_size))
}

/// Output of [`encode_masked`]: the interleaved (message ‖ mask) codeword and the generated mask.
#[derive(Debug)]
pub struct MaskedCodeword<P: PackedField> {
	/// The Reed-Solomon encoding of the interleaved (message ‖ mask) buffer.
	pub codeword: FieldBuffer<P>,
	/// The generated random mask, of equal length to the message.
	pub mask: FieldBuffer<P>,
}

/// Generates a random mask, interleaves it with the message, and Reed-Solomon encodes.
///
/// This is used for zero-knowledge FRI commitments. The function generates a random mask of
/// equal length to the input message, concatenates `message || mask` as the interleaved message
/// (with `log_batch_size = 1`), and performs Reed-Solomon encoding. The returned codeword is
/// committed by sending it over a Merkle channel, like [`encode_interleaved`]'s.
///
/// ## Arguments
///
/// * `params` - the (possibly batched) FRI parameters.
/// * `oracle_index` - the index into [`FRIParams::input_oracles`] of the oracle being encoded; its
///   spec must have `log_batch_size == 1` and `log_msg_len - 1 == message.log_len()`.
/// * `ntt` - the additive NTT for Reed-Solomon encoding
/// * `message` - the raw message to encode (not doubled)
/// * `rng` - cryptographic RNG for mask generation
pub fn encode_masked<F, P, NTT>(
	params: &FRIParams<F>,
	oracle_index: usize,
	ntt: &NTT,
	message: FieldSlice<P>,
	mut rng: impl CryptoRng,
) -> MaskedCodeword<P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
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
	let mask = FieldBuffer::<P>::new(
		log_len,
		par_rand::<StdRng, _, _>(packed_len, &mut rng, P::random).collect(),
	);

	let combined_values = if log_len < P::LOG_WIDTH {
		let combined_value =
			P::from_scalars(iter::chain(message.iter_scalars(), mask.iter_scalars()));
		vec![combined_value]
	} else {
		// TODO: Ideally, encoding should not allocate and copy the memory into a temp buffer.
		let mut combined_values: Vec<P> = Vec::with_capacity(2 * packed_len);
		combined_values.extend_from_slice(message.as_ref());
		combined_values.extend_from_slice(mask.as_ref());
		combined_values
	};
	let combined = FieldBuffer::new(log_len + 1, combined_values.into_boxed_slice());

	let codeword = encode_interleaved(params, oracle_index, ntt, combined.to_ref());

	MaskedCodeword { codeword, mask }
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128bGhash as B128, PackedBinaryGhash1x128b};
	use binius_hash::StdHashSuite;
	use binius_iop::fri::FRIParams;
	use binius_math::{
		BinarySubspace,
		ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
		test_utils::random_field_buffer,
	};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::merkle_tree::prover::BinaryMerkleTreeProver;

	#[test]
	fn test_encode_masked() {
		type F = B128;
		type P = PackedBinaryGhash1x128b;

		let mut rng = StdRng::seed_from_u64(42);

		let log_dim = 6;
		let log_inv_rate = 1;
		let log_batch_size = 1;
		let n_test_queries = 3;

		let merkle_prover = BinaryMerkleTreeProver::<F, StdHashSuite>::new();

		let subspace = BinarySubspace::with_dim(log_dim + log_inv_rate);
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		let params = FRIParams::with_strategy(
			ntt.domain_context(),
			merkle_prover.scheme(),
			log_dim + log_batch_size,
			Some(log_batch_size),
			log_inv_rate,
			n_test_queries,
			&binius_iop::fri::ConstantArityStrategy::new(2),
		);

		assert_eq!(params.log_batch_size(), 1);
		assert_eq!(params.rs_code().log_dim(), log_dim);

		let message = random_field_buffer::<P>(&mut rng, log_dim);

		let output: MaskedCodeword<P> = encode_masked(&params, 0, &ntt, message.to_ref(), &mut rng);

		// Verify mask has correct dimensions.
		assert_eq!(output.mask.log_len(), log_dim);

		// Verify the codeword has expected length (log_dim + log_batch_size + log_inv_rate).
		assert_eq!(output.codeword.log_len(), log_dim + log_batch_size + log_inv_rate);
	}
}
