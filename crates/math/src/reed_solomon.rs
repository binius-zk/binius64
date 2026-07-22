// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! [Reed–Solomon] codes over binary fields.
//!
//! See [`ReedSolomonCode`] for details.

use std::{mem::MaybeUninit, ptr, slice::from_raw_parts_mut};

use binius_field::{BinaryField, PackedField};
use binius_utils::rayon::prelude::*;
use getset::{CopyGetters, Getters};

use super::{
	FieldBuffer, FieldSlice, FieldSliceMut, binary_subspace::BinarySubspace, ntt::AdditiveNTT,
};
use crate::{
	bit_reverse::{bit_reverse_indices, bit_reverse_packed},
	ntt::DomainContext,
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
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct ReedSolomonCode<F> {
	#[get = "pub"]
	subspace: BinarySubspace<F>,
	log_dimension: usize,
	#[get_copy = "pub"]
	log_inv_rate: usize,
}

impl<F: BinaryField> ReedSolomonCode<F> {
	pub fn new(log_dimension: usize, log_inv_rate: usize) -> Self {
		let subspace = BinarySubspace::with_dim(log_dimension + log_inv_rate);
		Self::with_subspace(subspace, log_dimension, log_inv_rate)
	}

	pub fn with_ntt_subspace(
		ntt: &impl AdditiveNTT<Field = F>,
		log_dimension: usize,
		log_inv_rate: usize,
	) -> Self {
		Self::with_domain_context_subspace(ntt.domain_context(), log_dimension, log_inv_rate)
	}

	pub fn with_domain_context_subspace(
		domain_context: &impl DomainContext<Field = F>,
		log_dimension: usize,
		log_inv_rate: usize,
	) -> Self {
		let subspace_dim = log_dimension + log_inv_rate;
		assert!(
			subspace_dim <= domain_context.log_domain_size(),
			"precondition: subspace dimension must not exceed domain context log size"
		);
		let subspace = domain_context.subspace(subspace_dim);
		Self::with_subspace(subspace, log_dimension, log_inv_rate)
	}

	pub fn with_subspace(
		subspace: BinarySubspace<F>,
		log_dimension: usize,
		log_inv_rate: usize,
	) -> Self {
		assert_eq!(
			subspace.dim(),
			log_dimension + log_inv_rate,
			"precondition: subspace dimension must equal log_dimension + log_inv_rate"
		);
		Self {
			subspace,
			log_dimension,
			log_inv_rate,
		}
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
	pub fn encode_batch<P, NTT>(
		&self,
		ntt: &NTT,
		data: FieldSlice<P>,
		log_batch_size: usize,
	) -> FieldBuffer<P>
	where
		P: PackedField<Scalar = F>,
		NTT: AdditiveNTT<Field = F> + Sync,
	{
		assert_eq!(
			ntt.subspace(self.log_len()),
			self.subspace,
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

		let log_output_len = self.log_dim() + log_batch_size + self.log_inv_rate;

		// Sub-packed messages replicate scalar-wise.
		// The packed seed below cannot apply at this size.
		if data.log_len() < P::LOG_WIDTH {
			let mut scalars = data.iter_scalars().collect::<Vec<_>>();
			bit_reverse_indices(&mut scalars);
			let elem_0 = P::from_scalars(scalars.into_iter().cycle());
			let output_data = vec![elem_0; 1 << log_output_len.saturating_sub(P::LOG_WIDTH)];
			let mut output = FieldBuffer::new(log_output_len, output_data);
			ntt.forward_transform(output.to_mut(), self.log_inv_rate, log_batch_size);
			return output;
		}

		let mut output_data = Vec::with_capacity(1 << (log_output_len - P::LOG_WIDTH));

		output_data.extend_from_slice(data.as_ref());

		// Bit-reverse permute the message.
		bit_reverse_packed(FieldSliceMut::from_slice(data.log_len(), output_data.as_mut_slice()));

		// The zero-padded coefficient vector turns the early NTT layers into copies:
		// a butterfly against a zero half writes the live half into both outputs.
		// So the state entering the first executed layer is the message repeated once per coset.
		// The codeword is the remaining layers applied to that repetition.
		//
		// The fused seed goes one layer further in the same pass.
		// Every coset block holds identical data at the first executed layer.
		// So its butterflies can read the single message copy directly.
		// Each coset's post-layer state is written once.
		// The repetition is never materialized:
		//
		//     replicate-then-transform:  write 2^n, then read 2^n + write 2^n for the layer
		//     fused seed:                read 2^m once, write 2^n
		//
		// The seed needs one executable layer and a half-coset of at least one packed element.
		// Other shapes replicate as plain copies and keep the full early-layer skip.
		let fused_seed =
			self.log_inv_rate > 0 && self.log_dim() > 0 && data.log_len() > P::LOG_WIDTH;

		if fused_seed {
			seed_first_layer(
				ntt.domain_context(),
				&mut output_data,
				data.log_len() - P::LOG_WIDTH,
				self.log_inv_rate,
			);
		} else {
			let log_msg_len_packed = data.log_len() - P::LOG_WIDTH;
			output_data
				.spare_capacity_mut()
				.par_chunks_exact_mut(1 << log_msg_len_packed)
				.enumerate()
				.for_each(|(i, output_chunk)| unsafe {
					let dst_ptr = output_chunk.as_mut_ptr();

					// TODO(https://github.com/rust-lang/rust/issues/81944):
					// Improve unsafe code with Vec::split_at_spare_mut when stable

					// Safety:
					// - log_output_len == log_msg_len_packed + self.log_inv_rate
					// - i + 1 is in the range 1..1 << self.log_inv_rate
					// - dst_ptr is disjoint from src_ptr and within the Vec capacity
					let src_ptr = dst_ptr.sub((i + 1) << log_msg_len_packed);
					ptr::copy_nonoverlapping(src_ptr, dst_ptr, 1 << log_msg_len_packed);
				});
		}

		unsafe {
			// Safety: the seed or the replicating copies initialize every element above.
			output_data.set_len(1 << (log_output_len - P::LOG_WIDTH));
		}

		let mut output = FieldBuffer::new(log_output_len, output_data);

		// The seed already computed the first executed layer.
		// So the transform resumes one layer later.
		let skip_early = self.log_inv_rate + usize::from(fused_seed);
		ntt.forward_transform(output.to_mut(), skip_early, log_batch_size);
		output
	}
}

/// Base-2 logarithm of the seed's per-task chunk length, in packed elements.
///
/// A task holds one low-half and one high-half source chunk hot while writing every coset.
/// At 128-bit elements, two chunks of `2^10` elements occupy 32 KiB — comfortably L1-resident.
const LOG_SEED_CHUNK: usize = 10;

/// Writes every coset block of the codeword at its state after one NTT layer.
///
/// # Overview
///
/// The buffer's initialized prefix holds the bit-reversed message: coset block 0.
/// At the first executed layer, every coset block would hold that same data.
/// So the layer's butterflies read the single message copy and write each coset directly:
///
/// ```text
///     msg low half  a ──┬──> u_c = a + t_c * b     (t_c = layer twiddle of coset c)
///     msg high half b ──┴──> v_c = u_c + b
///
///     coset c block:  [ u_c ... | v_c ... ]        one write per output element
/// ```
///
/// The repeated-message intermediate is never materialized.
/// The message is read once.
/// Each source chunk stays cache-hot across all cosets.
///
/// After this call, the remaining layers run with one extra early layer skipped.
/// The output equals running the skipped layer inside the transform, butterfly for butterfly.
///
/// # Preconditions
///
/// * The buffer's length is `2^log_msg_len_packed` and holds the bit-reversed message.
/// * The buffer's capacity is `2^(log_msg_len_packed + log_inv_rate)`.
/// * `log_msg_len_packed >= 1`, so a half-coset spans at least one packed element.
/// * `log_inv_rate >= 1`.
/// * The layer at index `log_inv_rate` is executable in the domain.
fn seed_first_layer<P: PackedField>(
	domain_context: &impl DomainContext<Field = P::Scalar>,
	output_data: &mut Vec<P>,
	log_msg_len_packed: usize,
	log_inv_rate: usize,
) {
	debug_assert!(log_msg_len_packed >= 1);
	debug_assert!(log_inv_rate >= 1);
	debug_assert_eq!(output_data.len(), 1 << log_msg_len_packed);
	debug_assert!(output_data.capacity() >= 1 << (log_msg_len_packed + log_inv_rate));

	let copies = 1 << log_inv_rate;
	// Packed lengths of one coset block and of its half, the butterfly stride.
	let coset_packed = 1 << log_msg_len_packed;
	let half_packed = 1 << (log_msg_len_packed - 1);

	// One broadcast twiddle per coset: block c of the first executed layer.
	let twiddles = (0..copies)
		.map(|c| P::broadcast(domain_context.twiddle(log_inv_rate, c)))
		.collect::<Vec<_>>();

	// Partition the half-coset index space into cache-sized chunks, one task each.
	let chunk_len = half_packed.min(1 << LOG_SEED_CHUNK);
	let n_tasks = half_packed / chunk_len;

	let base_ptr = output_data.as_mut_ptr();

	// Hand each task its exclusive slice set.
	// A task owns the two source chunks of coset 0 (initialized).
	// It also owns the matching destination chunks of every other coset (uninitialized spare).
	//
	// Safety:
	// - The slice at (coset c, half h, task k) spans chunk_len elements starting at
	//
	//       c * coset_packed + h * half_packed + k * chunk_len
	//
	// - Distinct (c, h, k) triples therefore map to disjoint element ranges.
	// - All slices lie within the vector's capacity (checked by the debug assert above).
	// - Coset 0 slices cover initialized elements.
	// - Other cosets' slices stay behind `MaybeUninit` and are only written.
	// - `output_data` is not accessed again until the parallel loop below completes.
	let tasks = (0..n_tasks)
		.map(|k| {
			let offset = k * chunk_len;
			let src_u = unsafe { from_raw_parts_mut(base_ptr.add(offset), chunk_len) };
			let src_v =
				unsafe { from_raw_parts_mut(base_ptr.add(half_packed + offset), chunk_len) };
			let dsts = (1..copies)
				.map(|c| {
					let u_start = c * coset_packed + offset;
					let v_start = c * coset_packed + half_packed + offset;
					let dst_u = unsafe {
						from_raw_parts_mut(
							base_ptr.add(u_start).cast::<MaybeUninit<P>>(),
							chunk_len,
						)
					};
					let dst_v = unsafe {
						from_raw_parts_mut(
							base_ptr.add(v_start).cast::<MaybeUninit<P>>(),
							chunk_len,
						)
					};
					(dst_u, dst_v)
				})
				.collect::<Vec<_>>();
			(src_u, src_v, dsts)
		})
		.collect::<Vec<_>>();

	tasks.into_par_iter().for_each(|(src_u, src_v, dsts)| {
		// The other cosets first: coset 0 still holds the untouched message source.
		for (c, (dst_u, dst_v)) in dsts.into_iter().enumerate() {
			let twiddle = twiddles[c + 1];
			for j in 0..chunk_len {
				// The layer butterfly: u += v * twiddle; v += u.
				let mut u = src_u[j];
				let v0 = src_v[j];
				u += v0 * twiddle;
				dst_u[j].write(u);
				dst_v[j].write(v0 + u);
			}
		}

		// Coset 0 last, in place: its block is the source the copies just read.
		let twiddle = twiddles[0];
		for j in 0..chunk_len {
			let mut u = src_u[j];
			let v0 = src_v[j];
			u += v0 * twiddle;
			src_u[j] = u;
			src_v[j] = v0 + u;
		}
	});
}

#[cfg(test)]
mod tests {
	use binius_field::{
		BinaryField, PackedBinaryGhash1x128b, PackedBinaryGhash4x128b, PackedField,
	};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		FieldBuffer,
		bit_reverse::reverse_bits,
		ntt::{NeighborsLastReference, domain_context::GenericPreExpanded},
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

		// Create NTT with matching subspace
		let subspace = rs_code.subspace().clone();
		let domain_context = GenericPreExpanded::<P::Scalar>::generate_from_subspace(&subspace);
		let ntt = NeighborsLastReference {
			domain_context: &domain_context,
		};

		// Generate random message buffer
		let message = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);

		// Test the new encode_batch interface
		let encoded_buffer = rs_code.encode_batch(&ntt, message.to_ref(), log_batch_size);

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

	#[test]
	fn test_encode_batch_seed_edge_shapes() {
		// The fused seed needs one executable layer, one coset copy, and a packed half-coset.
		// These shapes sit on each of those boundaries.

		// Rate 1: no coset copies at all, nothing to seed.
		test_encode_batch_helper::<PackedBinaryGhash1x128b>(4, 0, 1);
		// Dimension 1: no NTT layer executes, the codeword is pure repetition.
		test_encode_batch_helper::<PackedBinaryGhash1x128b>(0, 2, 0);
		// Message length exactly the packing width: the half-coset is sub-packed.
		test_encode_batch_helper::<PackedBinaryGhash4x128b>(2, 2, 0);
		// Smallest seeded shape: one packed element per half-coset.
		test_encode_batch_helper::<PackedBinaryGhash1x128b>(1, 1, 0);
		// Half-coset larger than one seed chunk: the seed splits into several tasks.
		test_encode_batch_helper::<PackedBinaryGhash1x128b>(12, 1, 1);
	}

	// Pins one shape of the fused-seed encode across every NTT driver.
	//
	// The reference driver is itself pinned to the zero-padded oracle by the tests above.
	// So agreement here extends that byte-level pin to the production drivers,
	// which run the seeded transform with one extra early layer skipped.
	fn test_drivers_agree_helper<P: PackedField>(
		log_dim: usize,
		log_inv_rate: usize,
		log_batch_size: usize,
	) where
		P::Scalar: BinaryField,
	{
		use crate::ntt::{NeighborsLastMultiThread, NeighborsLastSingleThread};

		let mut rng = StdRng::seed_from_u64(0);

		let rs_code = ReedSolomonCode::<P::Scalar>::new(log_dim, log_inv_rate);
		let subspace = rs_code.subspace().clone();
		let domain_context = GenericPreExpanded::<P::Scalar>::generate_from_subspace(&subspace);

		let message = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);

		let reference = NeighborsLastReference {
			domain_context: &domain_context,
		};
		let single = NeighborsLastSingleThread::new(&domain_context);
		let multi = NeighborsLastMultiThread::new(&domain_context, 2);

		let encoded_ref = rs_code.encode_batch(&reference, message.to_ref(), log_batch_size);
		let encoded_single = rs_code.encode_batch(&single, message.to_ref(), log_batch_size);
		let encoded_multi = rs_code.encode_batch(&multi, message.to_ref(), log_batch_size);

		assert_eq!(
			encoded_single.as_ref(),
			encoded_ref.as_ref(),
			"single-thread driver mismatch at ({log_dim}, {log_inv_rate}, {log_batch_size})"
		);
		assert_eq!(
			encoded_multi.as_ref(),
			encoded_ref.as_ref(),
			"multi-thread driver mismatch at ({log_dim}, {log_inv_rate}, {log_batch_size})"
		);
	}

	#[test]
	fn test_encode_batch_agrees_across_ntt_drivers() {
		// Shapes chosen so the multithread driver exercises both shared and depth-first layers.
		test_drivers_agree_helper::<PackedBinaryGhash1x128b>(6, 1, 0);
		test_drivers_agree_helper::<PackedBinaryGhash1x128b>(8, 2, 1);
		test_drivers_agree_helper::<PackedBinaryGhash1x128b>(10, 1, 2);
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

		// Both codes are derived from a single shared domain context covering the larger code, so
		// the smaller code's subspace is the prefix the shared NTT twiddles expect.
		let subspace = BinarySubspace::<P::Scalar>::with_dim(log_dim_large + log_inv_rate);
		let domain_context = GenericPreExpanded::<P::Scalar>::generate_from_subspace(&subspace);
		let ntt = NeighborsLastReference {
			domain_context: &domain_context,
		};

		let rs_small = ReedSolomonCode::with_domain_context_subspace(
			&domain_context,
			log_dim_small,
			log_inv_rate,
		);
		let rs_large = ReedSolomonCode::with_domain_context_subspace(
			&domain_context,
			log_dim_large,
			log_inv_rate,
		);

		// Random message for the small code.
		let msg_small = random_field_buffer::<P>(&mut rng, log_dim_small);

		// ZeroPadMSB lift: the small message occupies the low `2^log_dim_small` hypercube values,
		// the high coordinates are zero.
		let mut msg_large = FieldBuffer::<P>::zeros(log_dim_large);
		for (i, val) in msg_small.iter_scalars().enumerate() {
			msg_large.set(i, val);
		}

		let enc_small = rs_small.encode_batch(&ntt, msg_small.to_ref(), 0);
		let enc_large = rs_large.encode_batch(&ntt, msg_large.to_ref(), 0);

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
}
