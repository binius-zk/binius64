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
//! The implementation uses additive NTT over binary fields, which is a linear transformation that
//! converts between coefficient and evaluation representations of polynomials. The specific
//! approach:
//!
//! - **Input**: 64 1-bit coefficients representing a polynomial in the Lagrange basis
//! - **Output**: 64 evaluations of the polynomial at specified domain points
//! - **Optimization**: Precomputes all 256 possible evaluations for each 8-bit position
//!
//! ## Performance
//!
//! By precomputing the lookup tables, the NTT operation is reduced to:
//! - 8 table lookups (one per byte)
//! - 7 packed field additions
//!
//! This trades memory (storing 8 * 256 * 64 field elements) for significant computation savings
//! compared to computing the NTT from scratch.

use std::{array, marker::PhantomData};

use binius_core::Word;
use binius_field::{
	AESTowerField8b as B8, BinaryField, BinaryField1b as B1, FieldOps,
	PackedAESBinaryField64x8b as Packed64xB8, PackedField, util::expand_subset_sums_array,
};
use binius_math::{
	BinarySubspace, FieldBuffer,
	ntt::{AdditiveNTT, NeighborsLastReference, domain_context::GenericOnTheFly},
};
use binius_verifier::protocols::bitand::{ROWS_PER_HYPERCUBE_VERTEX, SKIPPED_VARS};

/// A precomputed lookup table for fast NTT operations on 64-bit binary field elements.
///
/// This structure stores precomputed NTT evaluations for all possible 8-bit input combinations,
/// enabling fast computation of the full 64-bit NTT through table lookups and additions.
///
/// ## Structure
///
/// The internal data structure is a boxed 2-dimensional array `Box<[[Packed64xB8; 256]; 8]>` where:
/// - **First dimension**: Index of the 8-bit chunk within the 64-bit input (0-7)
/// - **Second dimension**: The 8-bit value (0-255) representing coefficient combinations
///
/// Each entry holds the `ROWS_PER_HYPERCUBE_VERTEX` NTT evaluations of that byte's coefficients at
/// the corresponding byte position, packed into a single [`Packed64xB8`].
#[derive(Debug, Clone)]
pub struct NTTLookup(Box<[[Packed64xB8; 256]; 8]>);

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
		let lde_mat = array::from_fn::<_, { ROWS_PER_HYPERCUBE_VERTEX / 8 }, _>(|b| {
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

	/// Computes the NTT of 64 1-bit coefficients using precomputed lookup tables.
	///
	/// Takes 64 1-bit coefficients provided as eight 8-bit chunks and computes their
	/// NTT by looking up precomputed values and adding them together, exploiting
	/// the linearity of the NTT operation.
	///
	/// Mathematically, if the input coefficients are c₀, c₁, ..., c₆₃, grouped into
	/// bytes B₀, B₁, ..., B₇, then NTT(c) = NTT(B₀) + NTT(B₁) + ... + NTT(B₇)
	/// where each NTT(Bᵢ) is retrieved from the precomputed lookup table.
	///
	/// Currently this method is used only for testing or reference purposes.
	/// In `univariate_round_message_extension_domain` we are accessing the lookup tables directly
	/// calculating 3 ntt evaluations at the same time as it appears to be more efficient.
	///
	/// ## Parameters
	///
	/// - `coeffs_in_byte_chunks`: Iterator yielding exactly 8 bytes, where each byte represents 8
	///   consecutive 1-bit coefficients from the 64-bit input.
	///
	/// ## Returns
	///
	/// Array of `ROWS_PER_HYPERCUBE_VERTEX / 16` packed field elements containing
	/// the NTT evaluations at all points in the output domain.
	#[cfg(test)]
	#[inline]
	pub fn ntt(&self, input: Word) -> Packed64xB8 {
		let input_bytes = input.as_u64().to_le_bytes();
		input_bytes
			.into_iter()
			.enumerate()
			.map(|(b, i)| self.0[b][i as usize])
			.sum()
	}

	/// Computes the NTTs of `N` 64-bit inputs simultaneously using the precomputed lookup tables.
	///
	/// Each input is split into its eight constituent bytes (LSB to MSB), and the NTT is computed
	/// by looking up the precomputed values for each byte position and summing them, exploiting the
	/// linearity of the NTT. Processing all `N` inputs together within each byte position keeps the
	/// independent accumulators in flight, which the compiler turns into instruction-level
	/// parallelism.
	///
	/// ## Parameters
	///
	/// - `inputs`: An array of `N` values, each divisible into bytes. The words' `u64`s can be
	///   passed directly.
	///
	/// ## Returns
	///
	/// An array of `N` packed field elements containing the NTT evaluations of each input.
	#[inline]
	pub fn multi_ntt_array<const N: usize>(&self, inputs: [Word; N]) -> [Packed64xB8; N] {
		let inputs_bytes = inputs.map(|input| input.as_u64().to_le_bytes());

		let mut results = [Packed64xB8::zero(); N];
		for (byte_index, lookup_byte) in self.0.iter().enumerate() {
			for (result, input_bytes) in std::iter::zip(&mut results, &inputs_bytes) {
				let byte = input_bytes[byte_index];
				*result += lookup_byte[byte as usize];
			}
		}
		results
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
