// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! [Reed–Solomon] codes over binary fields.
//!
//! See [`ReedSolomonCode`] for details.

use std::{iter, marker::PhantomData};

use binius_compute::Allocator;
use binius_field::{BinaryField, PackedField};
use getset::CopyGetters;

use super::{
	FieldBuffer, FieldSlice, FieldSliceMut, binary_subspace::BinarySubspace, ntt::AdditiveNTT,
};
use crate::{
	bit_reverse::bit_reverse_packed,
	ntt::{DomainContext, domain_context::GaoMateerOnTheFly},
};

/// [Reed–Solomon] codes over binary fields.
///
/// The Reed–Solomon code admits an efficient encoding algorithm over binary fields due to [LCH14].
/// The additive NTT encoding algorithm encodes messages interpreted as the coefficients of a
/// polynomial in a non-standard, novel polynomial basis and the codewords are the polynomial
/// evaluations over a linear subspace of the field. See the [binius-math] crate for more details.
///
/// [Reed–Solomon]: <https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction>
/// [LCH14]: <https://arxiv.org/abs/1404.3458>
#[derive(Debug, Clone, CopyGetters)]
pub struct ReedSolomonCode<F> {
	log_dimension: usize,
	#[get_copy = "pub"]
	log_inv_rate: usize,
	_marker: PhantomData<F>,
}

impl<F: BinaryField> ReedSolomonCode<F> {
	/// A code of the given dimension and rate, evaluated over the Gao-Mateer basis.
	///
	/// The evaluation domain is not a parameter: it is the Gao-Mateer basis of `log_dimension +
	/// log_inv_rate`, the same one [`GaoMateerOnTheFly`] and [`GaoMateerPreExpanded`] generate. A
	/// verifier can therefore rebuild the domain from the code's shape alone, without being told
	/// which basis the prover encoded over.
	///
	/// [`GaoMateerOnTheFly`]: crate::ntt::domain_context::GaoMateerOnTheFly
	/// [`GaoMateerPreExpanded`]: crate::ntt::domain_context::GaoMateerPreExpanded
	pub const fn new(log_dimension: usize, log_inv_rate: usize) -> Self {
		Self {
			log_dimension,
			log_inv_rate,
			_marker: PhantomData,
		}
	}

	/// The evaluation domain: the Gao-Mateer basis of [`Self::log_len`] dimensions.
	///
	/// Derived on demand rather than stored, so there is no way for it to disagree with the
	/// domain a prover or verifier generates from the same dimension.
	pub fn subspace(&self) -> BinarySubspace<F> {
		GaoMateerOnTheFly::<F>::generate(self.log_len()).subspace(self.log_len())
	}

	/// The dimension.
	pub const fn dim(&self) -> usize {
		1 << self.dim_bits()
	}

	pub const fn log_dim(&self) -> usize {
		self.log_dimension
	}

	pub const fn log_len(&self) -> usize {
		self.log_dimension + self.log_inv_rate
	}

	/// The block length.
	#[allow(clippy::len_without_is_empty)]
	pub const fn len(&self) -> usize {
		1 << (self.log_dimension + self.log_inv_rate)
	}

	/// The base-2 log of the dimension.
	const fn dim_bits(&self) -> usize {
		self.log_dimension
	}

	/// The reciprocal of the rate, ie. `self.len() / self.dim()`.
	pub const fn inv_rate(&self) -> usize {
		1 << self.log_inv_rate
	}

	/// Encodes a message with an interleaved Reed–Solomon code.
	///
	/// This function interprets the message as a batch of independent vectors and applies an
	/// interleaved Reed–Solomon.
	///
	/// ## Preconditions
	///
	/// * `data.log_len()` must equal `log_dim() + log_batch_size`.
	/// * The NTT subspace must match the code's subspace.
	///
	/// ## Postconditions
	///
	/// * All elements in the output buffer are initialized with the encoded codeword.
	pub fn encode_batch<P, NTT, A>(
		&self,
		ntt: &NTT,
		data: FieldSlice<'_, P>,
		log_batch_size: usize,
		alloc: &A,
	) -> FieldBuffer<P, A::Vec<P>>
	where
		P: PackedField<Scalar = F>,
		NTT: AdditiveNTT<Field = F> + Sync,
		A: Allocator,
	{
		assert_eq!(
			ntt.subspace(self.log_len()),
			self.subspace(),
			"precondition: NTT subspace must match code subspace"
		);
		assert_eq!(
			data.log_len(),
			self.log_dim() + log_batch_size,
			"precondition: data.log_len() must equal log_dim() + log_batch_size"
		);

		let _scope = tracing::trace_span!(
			"Reed-Solomon encode",
			log_len = self.log_len(),
			log_batch_size = log_batch_size,
			symbol_bits = F::N_BITS,
		)
		.entered();

		// The forward transform below skips its first `log_inv_rate` layers.
		// Each skipped layer would butterfly a coefficient with a zero pad:
		//
		//     u += v * twiddle; v += u;   with v = 0   =>   (c, 0) -> (c, c)
		//
		// That is one doubling per layer, so repeating the message does the skipped work.
		let log_output_len = self.log_dim() + log_batch_size + self.log_inv_rate;
		let mut output = FieldBuffer::from_view_with_capacity_in(alloc, data, log_output_len);

		// Permute the message once, then repeat it, so every copy inherits the permutation.
		bit_reverse_packed(output.as_mut_view());
		output.repeat_extend(log_output_len);

		ntt.forward_transform(output.as_mut_view(), self.log_inv_rate, log_batch_size);
		output
	}

	/// The adjoint of the interleaved encoder, taking a codeword weight back to a message weight.
	///
	/// The encoder is a linear map, so it has a transpose.
	/// Writing `E` for the encoder, this applies `E^T`, which is defined by
	///
	/// ```text
	///     <E^T a, m> = <a, E m>
	/// ```
	///
	/// for every weight `a` over codeword positions and every message `m`.
	///
	/// A caller reaches for this holding a sparse weight over codeword positions.
	/// What it wants back is the message weight that sparse weight induces.
	/// Expanding one generator row per nonzero costs `2^log_dim` per row.
	/// This costs one encode however many nonzeros there are.
	///
	/// # Algorithm
	///
	/// The encoder is three steps, so its transpose is those three steps reversed:
	///
	/// ```text
	///     encode   =  transform  after  repeat  after  bit-reverse
	///     adjoint  =  bit-reverse  after  sum-of-repeats  after  transposed transform
	/// ```
	///
	/// Bit reversal is a permutation that is its own inverse, so it transposes to itself.
	/// Repeating a message `2^log_inv_rate` times transposes to summing those repeats back down.
	///
	/// ## Preconditions
	///
	/// * `weights.log_len()` must equal `log_len() + log_batch_size`.
	/// * The NTT subspace must match the code's subspace.
	pub fn encode_batch_transpose<P, NTT, A>(
		&self,
		ntt: &NTT,
		mut weights: FieldSliceMut<'_, P>,
		log_batch_size: usize,
		alloc: &A,
	) -> FieldBuffer<P, A::Vec<P>>
	where
		P: PackedField<Scalar = F>,
		NTT: AdditiveNTT<Field = F> + Sync,
		A: Allocator,
	{
		assert_eq!(
			ntt.subspace(self.log_len()),
			self.subspace(),
			"precondition: NTT subspace must match code subspace"
		);
		assert_eq!(
			weights.log_len(),
			self.log_len() + log_batch_size,
			"precondition: weights.log_len() must equal log_len() + log_batch_size"
		);

		let _scope = tracing::trace_span!(
			"Reed-Solomon encode transpose",
			log_len = self.log_len(),
			log_batch_size = log_batch_size,
			symbol_bits = F::N_BITS,
		)
		.entered();

		// The transform runs at the same skips the encoder uses, with its layers reversed.
		ntt.transpose_transform(weights.as_mut_view(), self.log_inv_rate, log_batch_size);

		// The encoder repeated the message to fill the codeword, so the adjoint sums the repeats.
		let log_msg_len = self.log_dim() + log_batch_size;
		let mut folded = FieldBuffer::zeros_in(alloc, log_msg_len);
		for repeat in weights.chunks(log_msg_len) {
			for (accumulator, &word) in iter::zip(folded.as_mut(), repeat.as_ref()) {
				*accumulator += word;
			}
		}

		// The encoder permuted the message before repeating it.
		// A bit reversal is its own transpose, so the same permutation undoes it here.
		bit_reverse_packed(folded.as_mut_view());
		folded
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{
		BinaryField, Ghash128b, PackedBinaryGhash1x128b, PackedBinaryGhash4x128b, PackedField,
	};
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		FieldBuffer,
		bit_reverse::reverse_bits,
		inner_product::inner_product,
		ntt::{NeighborsLastReference, domain_context::GaoMateerPreExpanded},
		test_utils::random_field_buffer,
	};

	fn test_encode_batch_helper<P: PackedField>(
		log_dim: usize,
		log_inv_rate: usize,
		log_batch_size: usize,
	) where
		P::Scalar: BinaryField,
	{
		let mut rng = StdRng::seed_from_u64(0);

		let rs_code = ReedSolomonCode::<P::Scalar>::new(log_dim, log_inv_rate);

		// The code's domain is the Gao-Mateer basis of its length, so the NTT generates the same.
		let domain_context = GaoMateerPreExpanded::<P::Scalar>::generate(rs_code.log_len());
		let ntt = NeighborsLastReference {
			domain_context: &domain_context,
		};

		// Generate random message buffer
		let message = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);

		// Test the new encode_batch interface
		let encoded_buffer =
			rs_code.encode_batch(&ntt, message.as_view(), log_batch_size, &GlobalAllocator);

		// Method 2: Reference implementation - apply NTT with zero-padded coefficients to the
		// bit-reversal permuted message.
		let mut reference_buffer = FieldBuffer::zeros(rs_code.log_len() + log_batch_size);
		for (i, val) in message.iter_scalars().enumerate() {
			let bits = (rs_code.log_dim() + log_batch_size) as u32;
			reference_buffer.set(reverse_bits(i, bits), val);
		}

		// Perform large NTT with zero-padded coefficients.
		ntt.forward_transform(reference_buffer.as_mut_view(), 0, log_batch_size);

		// Compare results
		assert_eq!(
			encoded_buffer.as_ref(),
			reference_buffer.as_ref(),
			"encode_batch_inplace result differs from reference NTT implementation"
		);
	}

	#[test]
	fn test_encode_batch_above_packing_width() {
		// Test with PackedBinaryGhash1x128b
		test_encode_batch_helper::<PackedBinaryGhash1x128b>(4, 2, 0);
		test_encode_batch_helper::<PackedBinaryGhash1x128b>(6, 2, 1);
		test_encode_batch_helper::<PackedBinaryGhash1x128b>(8, 3, 2);

		// Test with PackedBinaryGhash4x128b
		test_encode_batch_helper::<PackedBinaryGhash4x128b>(4, 2, 0);
		test_encode_batch_helper::<PackedBinaryGhash4x128b>(6, 2, 1);
		test_encode_batch_helper::<PackedBinaryGhash4x128b>(8, 3, 2);
	}

	#[test]
	fn test_encode_batch_below_packing_width() {
		// Test where message length is less than the packing width and codeword length is greater.
		test_encode_batch_helper::<PackedBinaryGhash4x128b>(1, 2, 0);
	}

	/// Pins the codeword-duplication identity that underlies Lifted FRI (oracle padding).
	///
	/// Lifting a message `π` of dimension `m` to a larger dimension `M = m + η` zero-pads it on
	/// the most-significant hypercube coordinates (`ZeroPadMSB_η`). The novel-basis / bit-reversed
	/// encoding turns this into a *duplication* of the codeword: encoding the lifted message over
	/// the dimension-`M` code yields each entry of the dimension-`m` codeword repeated `2^η` times.
	/// This test asserts the contiguous form `Enc_M(ZeroPadMSB_η(π))[j] == Enc_m(π)[j >> η]`, which
	/// is the index translation Lifted FRI's prover and verifier rely on.
	fn test_lift_duplicate_identity_helper<P: PackedField>(
		log_dim_small: usize,
		log_dim_large: usize,
		log_inv_rate: usize,
	) where
		P::Scalar: BinaryField,
	{
		assert!(log_dim_small <= log_dim_large);
		let eta = log_dim_large - log_dim_small;

		let mut rng = StdRng::seed_from_u64(0);

		// One shared NTT covers the larger code. Both codes evaluate over the Gao-Mateer basis, and
		// the smaller one's is a prefix of the larger one's, which is what the shared twiddles
		// expect -- a property the codes now have by construction rather than by wiring.
		let domain_context =
			GaoMateerPreExpanded::<P::Scalar>::generate(log_dim_large + log_inv_rate);
		let ntt = NeighborsLastReference {
			domain_context: &domain_context,
		};

		let rs_small = ReedSolomonCode::new(log_dim_small, log_inv_rate);
		let rs_large = ReedSolomonCode::new(log_dim_large, log_inv_rate);

		// Random message for the small code.
		let msg_small = random_field_buffer::<P>(&mut rng, log_dim_small);

		// ZeroPadMSB lift: the small message occupies the low `2^log_dim_small` hypercube values,
		// the high coordinates are zero.
		let mut msg_large = FieldBuffer::<P>::zeros(log_dim_large);
		for (i, val) in msg_small.iter_scalars().enumerate() {
			msg_large.set(i, val);
		}

		let enc_small = rs_small.encode_batch(&ntt, msg_small.as_view(), 0, &GlobalAllocator);
		let enc_large = rs_large.encode_batch(&ntt, msg_large.as_view(), 0, &GlobalAllocator);

		let small_scalars = enc_small.iter_scalars().collect::<Vec<_>>();
		let large_scalars = enc_large.iter_scalars().collect::<Vec<_>>();
		assert_eq!(small_scalars.len(), 1 << (log_dim_small + log_inv_rate));
		assert_eq!(large_scalars.len(), 1 << (log_dim_large + log_inv_rate));

		for (j, &large) in large_scalars.iter().enumerate() {
			assert_eq!(
				large,
				small_scalars[j >> eta],
				"lift identity failed at index {j} (eta = {eta})"
			);
		}
	}

	#[test]
	fn test_lift_duplicate_identity() {
		// eta = 0 degrades to plain equality.
		test_lift_duplicate_identity_helper::<PackedBinaryGhash1x128b>(6, 6, 2);
		// Non-trivial lifts of varying sizes.
		test_lift_duplicate_identity_helper::<PackedBinaryGhash1x128b>(4, 6, 2);
		test_lift_duplicate_identity_helper::<PackedBinaryGhash1x128b>(2, 8, 1);
		test_lift_duplicate_identity_helper::<PackedBinaryGhash1x128b>(0, 4, 3);
		// Same lifts with a wider packing width.
		test_lift_duplicate_identity_helper::<PackedBinaryGhash4x128b>(4, 8, 2);
	}

	/// Checks the adjoint identity at one shape, over the packing `P`.
	fn assert_encode_adjoint<P: PackedField>(
		log_dim: usize,
		log_inv_rate: usize,
		log_batch_size: usize,
		seed: u64,
	) where
		P::Scalar: BinaryField,
	{
		let code = ReedSolomonCode::<P::Scalar>::new(log_dim, log_inv_rate);
		let domain_context = GaoMateerPreExpanded::<P::Scalar>::generate(code.log_len());
		let ntt = NeighborsLastReference {
			domain_context: &domain_context,
		};
		let mut rng = StdRng::seed_from_u64(seed);

		// `a` weighs codeword positions, `m` is a message the encoder could be handed.
		let a = random_field_buffer::<P>(&mut rng, code.log_len() + log_batch_size);
		let m = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);

		// Left side: pair the message with the weight the codeword weight induces on it.
		// The adjoint runs in place, so it gets a copy and `a` stays readable below.
		let mut scratch = a.clone();
		let induced = code.encode_batch_transpose(
			&ntt,
			scratch.as_mut_view(),
			log_batch_size,
			&GlobalAllocator,
		);
		let left = inner_product(induced.iter_scalars(), m.iter_scalars());

		// Right side: pair the codeword weight with the encoding itself.
		let encoded = code.encode_batch(&ntt, m.as_view(), log_batch_size, &GlobalAllocator);
		let right = inner_product(a.iter_scalars(), encoded.iter_scalars());

		assert_eq!(left, right, "log_dim={log_dim} rate={log_inv_rate} batch={log_batch_size}");
	}

	proptest! {
		/// The adjoint identity is the whole specification of the transposed encoder.
		///
		/// It is basis free, so one check covers an error in any of the three steps.
		/// Those are the layer order, the direction the repeats are summed, and the permutation.
		#[test]
		fn transposed_encoding_is_the_adjoint_of_encoding(seed: u64) {
			// A dimension of 0 is a one-element message, the smallest a code can carry.
			// A batch of 0 is the single-lane case the induced basis uses.
			for log_dim in 0..6 {
				for log_inv_rate in 1..4 {
					for log_batch_size in 0..3 {
						// One lane per word and four lanes per word take different chunk paths.
						assert_encode_adjoint::<PackedBinaryGhash1x128b>(
							log_dim, log_inv_rate, log_batch_size, seed,
						);
						assert_encode_adjoint::<PackedBinaryGhash4x128b>(
							log_dim, log_inv_rate, log_batch_size, seed,
						);
					}
				}
			}
		}
	}

	#[test]
	#[should_panic(expected = "weights.log_len() must equal log_len() + log_batch_size")]
	fn transposed_encoding_rejects_a_weight_of_the_wrong_width() {
		let code = ReedSolomonCode::<Ghash128b>::new(3, 1);
		let domain_context = GaoMateerPreExpanded::generate(code.log_len());
		let ntt = NeighborsLastReference {
			domain_context: &domain_context,
		};
		// The codeword has 2^4 positions, so a weight of 2^3 cannot be one over them.
		let mut weights = FieldBuffer::<PackedBinaryGhash1x128b>::zeros(3);
		code.encode_batch_transpose(&ntt, weights.as_mut_view(), 0, &GlobalAllocator);
	}
}
