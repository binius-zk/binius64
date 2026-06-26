// Copyright 2025 Irreducible Inc.
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

use std::vec;

use binius_field::{
	BinaryField, BinaryField1b, Divisible, Field, PackedBinaryField8x1b, PackedField,
	field::FieldOps,
};
use binius_math::{BinarySubspace, univariate::lagrange_evals_scalars};
use binius_verifier::protocols::bitand::{ROWS_PER_HYPERCUBE_VERTEX, SKIPPED_VARS};

/// A precomputed lookup table for fast NTT operations on 64-bit binary field elements.
///
/// This structure stores precomputed NTT evaluations for all possible 8-bit input combinations,
/// enabling fast computation of the full 64-bit NTT through table lookups and additions.
///
/// ## Structure
///
/// The internal data structure is a boxed 3-dimensional array `Box<[[[P; 4]; 256]; 8]>` where:
/// - **First dimension**: Index of the 8-bit chunk within the 64-bit input (0-7)
/// - **Second dimension**: The 8-bit value (0-255) representing coefficient combinations
/// - **Third dimension**: Packed field element index (0-3)
///
/// ## Memory Layout
///
/// Packed field indices are placed on the innermost axis so that the 4 packed evaluations
/// for a given (byte position, byte value) are contiguous in memory. This is the access
/// pattern of the hot inner loop in `univariate_round_message_extension_domain`.
///
/// ## Type Parameters
///
/// - `P`: The packed field type used for storing precomputed values. Must implement `PackedField`
///   with a scalar type that is a binary field.
#[derive(Clone)]
pub struct NTTLookup<P>(Box<[[P; 256]; 8]>);

impl<F, PNTTDomain> NTTLookup<PNTTDomain>
where
	F: BinaryField,
	PNTTDomain: PackedField<Scalar = F>,
{
	/// Creates a new NTT lookup table by precomputing all possible NTT evaluations
	/// for 8-bit input chunks across all byte positions in a 64-bit word.
	///
	/// ## Parameters
	///
	/// - `ntt_input_domain`: Binary subspace defining the input domain for the NTT. Must have
	///   dimension `SKIPPED_VARS` (6 bits).
	/// - `ntt_output_domain`: Array of field elements where the NTT will be evaluated. Must have
	///   length `ROWS_PER_HYPERCUBE_VERTEX` (64 elements).
	///
	/// ## Constraints
	///
	/// - `PNTTDomain::WIDTH` must equal 16 (packed field constraint)
	/// - Input domain dimension must equal `SKIPPED_VARS` (6)
	/// - Output domain length must equal `ROWS_PER_HYPERCUBE_VERTEX` (64)
	pub fn new(
		ntt_input_domain: &BinarySubspace<PNTTDomain::Scalar>,
		ntt_output_domain: &[PNTTDomain::Scalar],
	) -> Self {
		assert_eq!(PNTTDomain::WIDTH, 64);
		assert_eq!(ntt_output_domain.len(), ROWS_PER_HYPERCUBE_VERTEX);
		assert_eq!(ntt_input_domain.dim(), SKIPPED_VARS);

		let mut lookup = Box::new([[PNTTDomain::zero(); 256]; 8]);

		let mut eval_point_lagrange_evals =
			vec![
				vec![PNTTDomain::Scalar::ZERO; ROWS_PER_HYPERCUBE_VERTEX];
				ntt_output_domain.len()
			];
		for (eval_point_idx, eval_point) in ntt_output_domain.iter().enumerate() {
			eval_point_lagrange_evals[eval_point_idx] =
				lagrange_evals_scalars(ntt_input_domain, *eval_point);
		}

		for eight_bit_chunk_idx in 0..ROWS_PER_HYPERCUBE_VERTEX / 8 {
			for log_coefficient_as_bit_string in 0..8 {
				let coefficient_as_bit_string: u8 = 1 << log_coefficient_as_bit_string;
				let mut nonzero = PackedBinaryField8x1b::zero();
				nonzero.set(log_coefficient_as_bit_string, BinaryField1b::ONE);
				let nonzero_lagrange_basis_coeffs: Vec<_> = nonzero.iter().collect();
				let mut lagrange_basis_coeffs = [BinaryField1b::ZERO; ROWS_PER_HYPERCUBE_VERTEX];

				for (i, nonzero_lagrange_basis_coeff) in
					nonzero_lagrange_basis_coeffs.into_iter().enumerate()
				{
					lagrange_basis_coeffs[eight_bit_chunk_idx * 8 + i] =
						nonzero_lagrange_basis_coeff;
				}

				#[allow(clippy::needless_range_loop)]
				for eval_point_idx in 0..ROWS_PER_HYPERCUBE_VERTEX {
					let mut result = PNTTDomain::Scalar::ZERO;
					for basis_point_idx in 0..1 << ntt_input_domain.dim() {
						result += eval_point_lagrange_evals[eval_point_idx][basis_point_idx]
							* lagrange_basis_coeffs[basis_point_idx];
					}

					lookup[eight_bit_chunk_idx][coefficient_as_bit_string as usize]
						.set(eval_point_idx, result);
				}
			}
		}

		// Build combined coefficient lookup table
		for byte_idx in 0..8 {
			for coefficient_as_bit_string in 0..1 << 8 {
				let mut result = PNTTDomain::zero();
				for bit_in_string in 0..8 {
					let this_one_hot = coefficient_as_bit_string & 1 << bit_in_string;
					result += lookup[byte_idx][this_one_hot];
				}
				lookup[byte_idx][coefficient_as_bit_string] = result;
			}
		}
		NTTLookup(lookup)
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
	pub fn ntt<T: Divisible<u8>>(&self, input: T) -> PNTTDomain {
		Divisible::value_iter(input)
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
	pub fn multi_ntt_array<T: Divisible<u8>, const N: usize>(
		&self,
		inputs: [T; N],
	) -> [PNTTDomain; N] {
		let mut results = [PNTTDomain::zero(); N];
		for (byte_index, lookup_byte) in self.0.iter().enumerate() {
			for (result, input) in std::iter::zip(&mut results, &inputs) {
				let byte = Divisible::<u8>::get(input, byte_index);
				*result += lookup_byte[byte as usize];
			}
		}
		results
	}
}

#[cfg(test)]
mod test {
	use binius_field::{AESTowerField8b, BinaryField1b as B1, PackedAESBinaryField64x8b};
	use binius_math::{
		BinarySubspace, FieldBuffer,
		ntt::{AdditiveNTT, NeighborsLastReference, domain_context::GenericOnTheFly},
	};
	use rand::prelude::*;

	use super::*;

	#[test]
	fn test_against_ntt() {
		let subspace = BinarySubspace::with_dim(SKIPPED_VARS + 1);
		let input_subspace = subspace.reduce_dim(SKIPPED_VARS);

		let output_domain = subspace
			.iter()
			.skip(ROWS_PER_HYPERCUBE_VERTEX)
			.collect::<Vec<_>>();
		let ntt_lookup =
			NTTLookup::<PackedAESBinaryField64x8b>::new(&input_subspace, &output_domain);

		let input_domain_context = GenericOnTheFly::generate_from_subspace(&input_subspace);
		let input_ntt = NeighborsLastReference {
			domain_context: input_domain_context,
		};

		let output_domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let output_ntt = NeighborsLastReference {
			domain_context: output_domain_context,
		};

		// Repeat for 10 random values
		let mut rng = StdRng::seed_from_u64(0);
		for _ in 0..10 {
			let input = rng.random::<u64>();

			let ntt_lookup_result = ntt_lookup.ntt(input);
			let mut values_b8s = FieldBuffer::<AESTowerField8b>::zeros(subspace.dim());

			// iNTT the inputs in the first half of the buffer.
			{
				let mut values_b8s_split = values_b8s.split_half_mut();
				let (mut inputs_as_b8s, _) = values_b8s_split.halves();

				for i in 0..ROWS_PER_HYPERCUBE_VERTEX {
					inputs_as_b8s.set(i, AESTowerField8b::from(B1::from((input >> i) & 1 == 1)));
				}
				input_ntt.inverse_transform(inputs_as_b8s, 0, 0);
			}

			output_ntt.forward_transform(values_b8s.to_mut(), 0, 0);

			for i in 0..ROWS_PER_HYPERCUBE_VERTEX {
				let lookup_result = ntt_lookup_result.get(i);
				assert_eq!(lookup_result, values_b8s.get(i + ROWS_PER_HYPERCUBE_VERTEX));
			}
		}
	}
}
