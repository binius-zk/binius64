// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! # NTT Lookup Table Module
//!
//! This module provides a precomputed lookup table implementation for fast Number Theoretic
//! Transform (NTT) operations on 64-bit binary field elements. The implementation is specifically
//! optimized for the Binius64 protocol's constraint system.
//!
//! ## Overview
//!
//! The NTT lookup table achieves significant performance improvements by precomputing all possible
//! NTT evaluations for 8-bit input chunks. This allows the full 64-bit NTT to be computed by:
//!
//! 1. Splitting the 64 input bits into eight 8-bit chunks
//! 2. Looking up precomputed NTT values for each chunk
//! 3. Adding the results together (exploiting the linearity of the NTT)
//!
//! ## Algorithm
//!
//! The transformation is really a *low-degree extension* (LDE) over a binary subspace, not a plain
//! NTT. It maps the 64 input bits — viewed as evaluations of a polynomial over the input domain
//! (the lower half of the subspace) — to 64 evaluations of that same polynomial over the output
//! domain (the upper half, a coset shift of the input domain). The LDE is the composition of an
//! inverse NTT over the input domain (recovering the polynomial's coefficients) with a forward NTT
//! over the coset-shifted output domain; see `LowDegreeExtension::transform`.
//!
//! Because the LDE is linear, the LDE of a 64-bit input is the sum of the LDEs of its eight bytes:
//!
//! - **Input**: 64 1-bit coefficients (evaluations over the input domain)
//! - **Output**: 64 field elements (evaluations over the output domain)
//! - **Optimization**: precompute all 256 LDE images for each 8-bit position, so the full LDE is 8
//!   table lookups plus 7 packed additions.
//!
//! ## Compressed storage
//!
//! Storing an independent 256-entry table for each of the eight byte positions would take
//! `8 * 256 * 64` field elements. We instead store tables for only two byte positions and
//! reconstruct the rest by permuting the packed evaluations.
//!
//! This exploits a *translation-invariance* property of the LDE matrix observed in the Flock paper
//! (<https://github.com/succinctlabs/flock/blob/main/paper/flock-paper.pdf>, Section 4.2): because
//! the output domain is a coset shift of the input domain over a binary subspace, the LDE images of
//! the unit inputs at different byte positions are fixed permutations of one another. The LDE image
//! for byte position `b` is therefore recovered from the stored table for parity `b % 2` by
//! permuting its packed evaluations according to `b / 2` (see [`NTTLookup::ntt`]). This shrinks the
//! table footprint by 4x — to `2 * 256 * 64` field elements — adding only a cheap in-register
//! permute to each lookup.

use std::{array, marker::PhantomData};

use binius_core::Word;
use binius_field::{
	AESTowerField8b as B8, BinaryField, BinaryField1b as B1, Divisible,
	PackedAESBinaryField64x8b as Packed64xB8, PackedField, WithUnderlier, arch::M128,
	util::expand_subset_sums_array,
};
use binius_math::{
	BinarySubspace, FieldBuffer,
	ntt::{AdditiveNTT, NeighborsLastReference, domain_context::GenericOnTheFly},
};
use binius_verifier::protocols::bitand::{ROWS_PER_HYPERCUBE_VERTEX, SKIPPED_VARS};

/// A precomputed lookup table for fast LDE operations on 64-bit binary field elements.
///
/// This structure stores precomputed LDE evaluations for all possible 8-bit input combinations,
/// enabling fast computation of the full 64-bit LDE through table lookups and additions. See the
/// module-level documentation for the LDE definition and the Flock compression it uses.
///
/// ## Structure
///
/// The internal data structure is a boxed array `Box<[[Packed64xB8; 256]; 2]>` where:
/// - **First dimension**: the byte-position parity `b % 2`. Only these two tables are stored; the
///   remaining six byte positions are recovered by permutation (see the module-level "Compressed
///   storage" notes).
/// - **Second dimension**: the 8-bit value (0-255) for that byte.
///
/// Each entry holds the `ROWS_PER_HYPERCUBE_VERTEX` LDE evaluations of that byte's coefficients,
/// packed into a single [`Packed64xB8`].
#[derive(Debug, Clone)]
pub struct NTTLookup(Box<[[Packed64xB8; 256]; 2]>);

impl NTTLookup {
	/// Creates a new NTT lookup table by precomputing all possible NTT evaluations
	/// for 8-bit input chunks across all byte positions in a 64-bit word.
	///
	/// ## Parameters
	///
	/// - `subspace`: Binary subspace of dimension `SKIPPED_VARS + 1`. Its lower half defines the
	///   NTT input domain and its upper half the output domain at which evaluations are
	///   precomputed.
	///
	/// ## Constraints
	///
	/// - Subspace dimension must equal `SKIPPED_VARS + 1`
	pub fn new(subspace: &BinarySubspace<B8>) -> Self {
		assert_eq!(subspace.dim(), SKIPPED_VARS + 1);

		let lde = LowDegreeExtension::<Packed64xB8>::new(subspace);
		let lde_mat = array::from_fn::<_, 2, _>(|b| {
			array::from_fn::<_, 8, _>(|i| {
				let output = lde.transform(1 << (8 * b + i));
				assert_eq!(output.log_len(), SKIPPED_VARS + 1);
				// Pull out the second element, corresponding to the output domain
				output.as_ref()[1]
			})
		});

		let lookup = lde_mat.map(expand_subset_sums_array::<_, 8, 256>);
		NTTLookup(Box::new(lookup))
	}

	/// Computes the LDE of 64 1-bit coefficients using the precomputed lookup tables.
	///
	/// The 64-bit `input` is split into eight bytes B₀, B₁, ..., B₇. By linearity the LDE of the
	/// input is the sum of the per-byte LDEs: `LDE(input) = LDE(B₀) + LDE(B₁) + ... + LDE(B₇)`.
	///
	/// Only two byte-position tables are stored, so the LDE of byte position `b` is reconstructed
	/// from the table for parity `b % 2` by permuting its packed evaluations according to `b / 2`.
	/// This permutation is the translation of the output domain exploited by the Flock compression
	/// (see the module-level "Compressed storage" notes); in the packed representation it is a
	/// permutation of the four 128-bit lanes, `lane[i] <- lane[i ^ (b / 2)]`. The loop is unrolled
	/// over `b`, so the parity select and lane index are compile-time constants.
	///
	/// Used directly only in tests; `univariate_round_message_extension_domain` accesses the tables
	/// inline to compute three LDE evaluations at once, which is more efficient.
	///
	/// ## Returns
	///
	/// A [`Packed64xB8`] holding the `ROWS_PER_HYPERCUBE_VERTEX` LDE evaluations over the output
	/// domain.
	#[inline]
	pub fn ntt(&self, input: Word) -> Packed64xB8 {
		let input_bytes = input.as_u64().to_le_bytes();

		let mut out = Packed64xB8::default();
		// This will get unrolled, so indexing arithmetic washes away.
		for b in 0..8 {
			let packed = &self.0[b % 2][input_bytes[b] as usize];
			let bitvec = packed.to_underlier_ref();
			let dst_bitvec = Divisible::<M128>::from_iter(
				(0..4).map(|i| Divisible::<M128>::get(bitvec, i ^ (b / 2))),
			);
			out += Packed64xB8::from_underlier(dst_bitvec);
		}
		out
	}
}

struct LowDegreeExtension<P: PackedField> {
	interpolation: NeighborsLastReference<GenericOnTheFly<P::Scalar>>,
	extrapolation: NeighborsLastReference<GenericOnTheFly<P::Scalar>>,
	_marker: PhantomData<P>,
}

impl<F, P> LowDegreeExtension<P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	fn new(subspace: &BinarySubspace<F>) -> Self {
		assert_eq!(subspace.dim(), SKIPPED_VARS + 1);

		let input_subspace = subspace.reduce_dim(SKIPPED_VARS);
		let input_domain_context = GenericOnTheFly::generate_from_subspace(&input_subspace);
		let output_domain_context = GenericOnTheFly::generate_from_subspace(subspace);

		Self {
			interpolation: NeighborsLastReference {
				domain_context: input_domain_context,
			},
			extrapolation: NeighborsLastReference {
				domain_context: output_domain_context,
			},
			_marker: PhantomData,
		}
	}

	/// Computes the low-degree extension of a 64-bit input.
	///
	/// The LDE is the composition of two additive NTTs over the binary subspace:
	///
	/// 1. an **inverse NTT** over the input domain (the lower half of the subspace), which
	///    interprets the 64 input bits as evaluations over that domain and recovers the
	///    polynomial's coefficients;
	/// 2. a **forward NTT** over the full subspace, which re-evaluates that polynomial over the
	///    output domain (the upper half) — a coset shift of the input domain.
	///
	/// The returned buffer holds both halves; the output-domain evaluations are the upper half.
	fn transform(&self, input: u64) -> FieldBuffer<P> {
		let mut values = FieldBuffer::<P>::zeros(SKIPPED_VARS + 1);

		// Inverse NTT the inputs in the first half of the buffer.
		{
			let mut values_split = values.split_half_mut();
			let (mut input_elems, _) = values_split.halves();

			for i in 0..ROWS_PER_HYPERCUBE_VERTEX {
				input_elems.set(i, F::from(B1::from((input >> i) & 1 == 1)));
			}
			self.interpolation.inverse_transform(input_elems, 0, 0);
		}

		// Forward NTT the zero-padded coefficients.
		self.extrapolation.forward_transform(values.to_mut(), 0, 0);

		values
	}
}

#[cfg(test)]
mod test {
	use binius_field::Divisible;
	use binius_math::BinarySubspace;
	use rand::prelude::*;

	use super::*;

	#[test]
	fn test_against_ntt() {
		let subspace = BinarySubspace::with_dim(SKIPPED_VARS + 1);
		let lde = LowDegreeExtension::<B8>::new(&subspace);
		let ntt_lookup = NTTLookup::new(&subspace);

		// Repeat for 10 random values
		let mut rng = StdRng::seed_from_u64(0);
		for _ in 0..10 {
			let input = rng.random::<u64>();

			let lde_result = lde.transform(input);
			let ntt_lookup_result = ntt_lookup.ntt(Word(input));
			for i in 0..ROWS_PER_HYPERCUBE_VERTEX {
				let lookup_result = ntt_lookup_result.get(i);
				assert_eq!(lookup_result, lde_result.get(i + ROWS_PER_HYPERCUBE_VERTEX));
			}
		}
	}
}
