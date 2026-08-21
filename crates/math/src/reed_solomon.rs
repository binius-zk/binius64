// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! [Reed–Solomon] codes over binary fields.
//!
//! See [`ReedSolomonCode`] for details.

use std::{marker::PhantomData, ptr};

use binius_compute::{Allocator, VecLike};
use binius_field::{BinaryField, PackedField};
use binius_utils::rayon::{prelude::*, task_size::task_chunk_len};
use getset::CopyGetters;

use super::{
	FieldBuffer, FieldSlice, FieldSliceMut, binary_subspace::BinarySubspace, ntt::AdditiveNTT,
};
use crate::{
	bit_reverse::{bit_reverse_indices, bit_reverse_packed},
	ntt::{DomainContext, NeighborsLastMultiThread, domain_context::GaoMateerOnTheFly},
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
		data: FieldSlice<P>,
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

		// Repeat the message to fill the entire buffer.
		let log_output_len = self.log_dim() + log_batch_size + self.log_inv_rate;
		let output_data = if data.log_len() < P::LOG_WIDTH {
			let mut scalars = data.iter_scalars().collect::<Vec<_>>();
			bit_reverse_indices(&mut scalars);
			let elem_0 = P::from_scalars(scalars.into_iter().cycle());
			let len = 1 << log_output_len.saturating_sub(P::LOG_WIDTH);
			let mut output = alloc.alloc::<P>(len);
			output.resize(len, elem_0);
			output
		} else {
			// The forward transform below skips its first `log_inv_rate` layers.
			// Each skipped layer would butterfly a coefficient with a zero pad:
			//
			//     u += v * twiddle; v += u;   with v = 0   =>   (c, 0) -> (c, c)
			//
			// That is one doubling per layer, so repeating the message does the skipped work.
			let output_packed_len = 1 << (log_output_len - P::LOG_WIDTH);

			// A run is the words one worker copies at a time.
			// It is a power of two at most the message length, so it divides the message evenly.
			let run = data
				.as_ref()
				.len()
				.min(task_chunk_len::<P>().next_power_of_two());

			repeated_message_buffer(data, output_packed_len, run, alloc)
		};
		let mut output = FieldBuffer::new(log_output_len, output_data);

		ntt.forward_transform(output.to_mut(), self.log_inv_rate, log_batch_size);
		output
	}

	/// Same encoding as [`Self::encode_batch`], but invokes `on_chunk_ready` the instant each
	/// independent post-shared-layer chunk of the codeword finishes transforming.
	///
	/// A caller that wants to start downstream work on a finished region of the codeword, without
	/// waiting for the whole encode to complete, hooks in here.
	/// See [`NeighborsLastMultiThread::forward_transform_with_callback`] for the chunking this
	/// callback rides on.
	///
	/// ## Preconditions
	///
	/// Same as [`Self::encode_batch`].
	pub fn encode_batch_with_callback<P, DC, A>(
		&self,
		ntt: &NeighborsLastMultiThread<DC>,
		data: FieldSlice<P>,
		log_batch_size: usize,
		alloc: &A,
		on_chunk_ready: impl Fn(usize, &[P]) + Sync,
	) -> FieldBuffer<P, A::Vec<P>>
	where
		P: PackedField<Scalar = F>,
		DC: DomainContext<Field = F> + Sync,
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
			"Reed-Solomon encode (pipelined)",
			log_len = self.log_len(),
			log_batch_size = log_batch_size,
			symbol_bits = F::N_BITS,
		)
		.entered();

		let log_output_len = self.log_dim() + log_batch_size + self.log_inv_rate;
		let output_data = if data.log_len() < P::LOG_WIDTH {
			let mut scalars = data.iter_scalars().collect::<Vec<_>>();
			bit_reverse_indices(&mut scalars);
			let elem_0 = P::from_scalars(scalars.into_iter().cycle());
			let len = 1 << log_output_len.saturating_sub(P::LOG_WIDTH);
			let mut output = alloc.alloc::<P>(len);
			output.resize(len, elem_0);
			on_chunk_ready(0, &output);
			return FieldBuffer::new(log_output_len, output);
		} else {
			let output_packed_len = 1 << (log_output_len - P::LOG_WIDTH);
			let run = data
				.as_ref()
				.len()
				.min(task_chunk_len::<P>().next_power_of_two());
			repeated_message_buffer(data, output_packed_len, run, alloc)
		};
		let mut output = FieldBuffer::new(log_output_len, output_data);

		ntt.forward_transform_with_callback(
			output.to_mut(),
			self.log_inv_rate,
			log_batch_size,
			on_chunk_ready,
		);
		output
	}
}

/// Builds a buffer of `total` words holding the bit-reversed message, repeated.
///
/// # Overview
///
/// ```text
///     [msg]  ->  [rev msg | rev msg | rev msg | rev msg]
/// ```
///
/// The leading copy is filled from the message, then permuted.
/// Every later copy is filled from that leading copy, so the permutation runs once.
/// Both fills split into runs, so more than one worker carries a long message.
///
/// # Arguments
///
/// * `msg` - the message to repeat
/// * `total` - word count of the returned buffer
/// * `run` - words one worker copies at a time
///
/// # Preconditions
///
/// * `msg` holds at least one whole word, and `total` is a multiple of its word count
/// * `run` is a power of two, and at most the message's word count
fn repeated_message_buffer<P: PackedField, A: Allocator>(
	msg: FieldSlice<P>,
	total: usize,
	run: usize,
	alloc: &A,
) -> A::Vec<P> {
	let msg_len = msg.as_ref().len();
	debug_assert!(msg_len.is_power_of_two());
	debug_assert!(run.is_power_of_two() && run <= msg_len);
	debug_assert_eq!(total % msg_len, 0);

	let mut output = alloc.alloc::<P>(total);

	// Copy the message into the leading copy.
	let head = &mut output.spare_capacity_mut()[..msg_len];
	(head.par_chunks_mut(run), msg.as_ref().par_chunks(run))
		.into_par_iter()
		.for_each(|(dst, src)| {
			// SAFETY:
			// - The two runs have equal length: one position, two buffers of one length.
			// - They live in different buffers, so they cannot overlap.
			unsafe {
				ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast::<P>(), src.len());
			}
		});
	// SAFETY: the loop above wrote every one of the leading `msg_len` words.
	unsafe { output.set_len(msg_len) };

	// Permute the leading copy, so the copies below inherit it.
	bit_reverse_packed(FieldSliceMut::from_slice(msg.log_len(), &mut output));

	// The source is read through an address rather than a borrow.
	// That is what lets the workers read the leading copy while the rest is held mutably.
	let msg_ptr = output.as_ptr() as usize;
	let tail = &mut output.spare_capacity_mut()[..total - msg_len];
	tail.par_chunks_mut(run).enumerate().for_each(|(i, dst)| {
		// Run `i` of the tail sits one whole message past run `i` of the buffer.
		// The message length is a power of two, so masking that position gives its source.
		let src_offset = (i * run) & (msg_len - 1);

		// SAFETY:
		// - The offset is masked into the leading copy, and a run divides it evenly.
		// - So the source run lies inside the leading copy, which is initialized.
		// - Nothing writes the leading copy here, and the destination lies past it.
		// - The address stays live because the buffer outlives this loop.
		unsafe {
			let msg = msg_ptr as *const P;
			ptr::copy_nonoverlapping(msg.add(src_offset), dst.as_mut_ptr().cast::<P>(), dst.len());
		}
	});

	// SAFETY: the loop above wrote every word between the leading copy and `total`.
	unsafe { output.set_len(total) };

	output
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{
		BinaryField, PackedBinaryGhash1x128b, PackedBinaryGhash4x128b, PackedField,
	};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		FieldBuffer,
		bit_reverse::reverse_bits,
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
			rs_code.encode_batch(&ntt, message.to_ref(), log_batch_size, &GlobalAllocator);

		// Method 2: Reference implementation - apply NTT with zero-padded coefficients to the
		// bit-reversal permuted message.
		let mut reference_buffer = FieldBuffer::zeros(rs_code.log_len() + log_batch_size);
		for (i, val) in message.iter_scalars().enumerate() {
			let bits = (rs_code.log_dim() + log_batch_size) as u32;
			reference_buffer.set(reverse_bits(i, bits), val);
		}

		// Perform large NTT with zero-padded coefficients.
		ntt.forward_transform(reference_buffer.to_mut(), 0, log_batch_size);

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

		let enc_small = rs_small.encode_batch(&ntt, msg_small.to_ref(), 0, &GlobalAllocator);
		let enc_large = rs_large.encode_batch(&ntt, msg_large.to_ref(), 0, &GlobalAllocator);

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

	// One serial copy of the message, one permutation, then a chain of doublings.
	// That is the plain form of the same construction, so it pins the run-split form.
	fn repeated_message_buffer_reference<P: PackedField>(
		msg: FieldSlice<P>,
		total: usize,
	) -> Vec<P> {
		let mut output = Vec::with_capacity(total);
		output.extend_from_slice(msg.as_ref());

		bit_reverse_packed(FieldSliceMut::from_slice(msg.log_len(), output.as_mut_slice()));

		while output.len() < total {
			output.extend_from_within(..);
		}
		output
	}

	#[test]
	fn test_repeated_message_buffer_matches_the_doubling_chain() {
		let mut rng = StdRng::seed_from_u64(0);

		// Fixture state: messages of 1 to 16 words, each repeated 1 to 8 times over.
		for log_msg in 0..5 {
			for log_copies in 0..4 {
				let msg = random_field_buffer::<PackedBinaryGhash1x128b>(&mut rng, log_msg);
				let total = (1 << log_msg) << log_copies;
				let expected = repeated_message_buffer_reference(msg.to_ref(), total);

				// Every run width a caller can pass, from one word up to the whole message.
				// A run under the message length is what splits a fill across workers.
				// The encoder only reaches that on a message above one mebibyte.
				for log_run in 0..=log_msg {
					let built = repeated_message_buffer(
						msg.to_ref(),
						total,
						1 << log_run,
						&GlobalAllocator,
					);
					assert_eq!(built, expected, "log_msg={log_msg} log_run={log_run}");
				}
			}
		}
	}
}
