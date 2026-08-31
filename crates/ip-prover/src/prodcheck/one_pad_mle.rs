// Copyright 2026 The Binius Developers

//! MLE-check prover for one layer of a one-padded product check.
//!
//! One-padding lifts a product tree of depth $k$ to depth $n \ge k$ by filling the extra leaves
//! with ones, which leaves the tree's product unchanged. Batching product checks of unequal depths
//! pads each shallow tree up to the deepest one, so the batch's layer loop runs a single uniform
//! schedule and the verifier never learns the individual depths. See the *Batched Product Checks of
//! Unequal Depths* appendix of the Binius64 whitepaper for the protocol and the derivation behind
//! the round polynomials.
//!
//! No padded layer is ever materialized.
//! The shared wrapper corrects the unpadded layer's own messages, at $O(1)$ per round.

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::FieldVec;

use crate::sumcheck::{
	bivariate_product_mle,
	common::MleCheckProver,
	pad_mle::{Fill, PadMleCheckProver, PadShape, mul_linear, pad_eq_prefixes},
};

/// The composition one product layer reduces.
///
/// The layer carries one claim, the product of the two halves of the child layer on its highest
/// variable.
pub struct BivariateProduct;

impl<F: Field> PadShape<F, 2> for BivariateProduct {
	const CHILDREN: [Fill; 2] = [Fill::One, Fill::One];
	const CLAIMS: &'static [Fill] = &[Fill::One];

	#[inline(always)]
	fn compose([g_0, g_1]: [[F; 2]; 2]) -> Vec<RoundCoeffs<F>> {
		vec![mul_linear(g_0, g_1)]
	}
}

/// MLE-check prover for one layer of a product check over a one-padded witness.
pub type OnePadMleCheckProver<F, Inner> = PadMleCheckProver<F, Inner, BivariateProduct, 2>;

/// Creates the prover for one padded product-check layer.
///
/// # Arguments
///
/// * `layer` - The unpadded child layer, whose low and high halves on its highest variable are the
///   two multilinears whose product this layer reduces.
/// * `pad_len` - Length of `eval_point`'s padding segment. Zero leaves the inner reduction
///   uncorrected.
/// * `eval_point` - The padded layer's claim point, `[padding | real]`.
/// * `claim` - The padded layer's claimed evaluation at `eval_point`.
///
/// # Preconditions
///
/// * `layer.log_len() >= 1`
/// * `eval_point.len() == layer.log_len() - 1 + pad_len`
///
/// # Panics
///
/// Panics if the padding segment's equality weight $q$ is zero, which requires one of its
/// coordinates — all verifier challenges — to equal one, and so happens with probability at most
/// $\nu / |K|$.
pub fn new<'alloc, A, F, P>(
	alloc: &'alloc A,
	layer: FieldVec<P, A>,
	pad_len: usize,
	eval_point: Vec<F>,
	claim: F,
) -> OnePadMleCheckProver<F, impl MleCheckProver<F> + 'alloc>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert!(layer.log_len() >= 1); // precondition
	let n_real_rounds = layer.log_len() - 1;
	assert_eq!(eval_point.len(), pad_len + n_real_rounds); // precondition

	// The round polynomials' equality weights and the claim's are all lookups into one table.
	let prefixes = pad_eq_prefixes(&eval_point[..pad_len]);
	let pad_eq = prefixes[pad_len];
	assert!(pad_eq != F::ZERO, "a padding coordinate of the claim point equals one");

	// The padded claim runs from one at padding weight zero to the unpadded claim at weight one, so
	// the inner prover starts from the preimage.
	let inner_claim = Fill::One.at(claim, pad_eq.invert_or_zero());
	let inner = bivariate_product_mle::new_split_half(
		alloc,
		layer,
		eval_point[pad_len..].to_vec(),
		inner_claim,
	);

	OnePadMleCheckProver::new(prefixes, eval_point, inner)
}

// The prover is checked against the padded layer it stands in for: the same reduction run by an
// ordinary bivariate-product MLE-check over an explicitly materialized one-padded layer must
// produce the same round polynomials and the same child evaluations.
#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Random, field::FieldOps};
	use binius_math::{
		FieldBuffer,
		multilinear::evaluate::evaluate,
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use rand::prelude::*;

	use super::*;

	type P = Packed128b;
	type F = <P as FieldOps>::Scalar;

	/// Materializes `OnePad_{pad_len}` of a layer, whose variables are `[real | split]`.
	///
	/// The padding variables land below the real ones, matching the claim-point layout the wrapper
	/// expects.
	fn one_pad_layer(layer: &FieldBuffer<P>, pad_len: usize) -> FieldBuffer<P> {
		let values = (0..1 << (layer.log_len() + pad_len))
			.map(|index| {
				let padding = index & ((1 << pad_len) - 1);
				if padding == 0 {
					layer.get(index >> pad_len)
				} else {
					F::ONE
				}
			})
			.collect::<Vec<_>>();
		FieldBuffer::from_values(&values)
	}

	/// The bivariate-product MLE-check claim on a buffer's two halves at `eval_point`.
	fn split_half_claim(buffer: &FieldBuffer<P>, eval_point: &[F]) -> F {
		let (low, high) = buffer.split_half();
		let products = (0..low.len())
			.map(|i| low.get(i) * high.get(i))
			.collect::<Vec<_>>();
		evaluate(&FieldBuffer::<P>::from_values(&products), eval_point)
	}

	/// Runs the padded prover and the reference prover over the materialized padded layer in
	/// lockstep.
	fn assert_matches_padded_reference(layer: FieldBuffer<P>, pad_len: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		let padded_layer = one_pad_layer(&layer, pad_len);
		let n_vars = padded_layer.log_len() - 1;
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let claim = split_half_claim(&padded_layer, &eval_point);

		let mut reference =
			bivariate_product_mle::new_split_half(&alloc, padded_layer, eval_point.clone(), claim);
		let mut prover = new(&alloc, layer, pad_len, eval_point, claim);

		for round in 0..n_vars {
			assert_eq!(prover.n_vars(), n_vars - round);
			assert_eq!(prover.eval_point(), reference.eval_point());
			assert_eq!(prover.execute(), reference.execute(), "round {round}");

			let challenge = F::random(&mut rng);
			prover.fold(challenge);
			reference.fold(challenge);
		}

		assert_eq!(prover.finish(), reference.finish());
	}

	#[test]
	fn matches_padded_reference() {
		let mut rng = StdRng::seed_from_u64(1);
		// A layer of `n_real_rounds + 1` variables splits into two halves of `n_real_rounds`, so
		// zero real rounds is a layer of one pair. The packing width is 4 scalars, so the range
		// straddles it in both directions, and 3 and 5 padding rows are not powers of two.
		for n_real_rounds in [0, 1, 2, 3, 5] {
			for pad_len in [0, 1, 2, 3, 5] {
				let layer = random_field_buffer::<P>(&mut rng, n_real_rounds + 1);
				assert_matches_padded_reference(layer, pad_len);
			}
		}
	}

	// The layers a shallow tree spends while the batch is still above it are one-paddings of its
	// product, whose high child is identically one. That degenerate shape is what
	// `batch_prove_unequal_depths` feeds in for those layers.
	#[test]
	fn matches_padded_reference_with_constant_one_child() {
		let mut rng = StdRng::seed_from_u64(2);
		for pad_len in [1, 2, 3, 4, 5] {
			let product = F::random(&mut rng);
			let layer = FieldBuffer::<P>::from_values(&[product, F::ONE]);
			assert_matches_padded_reference(layer, pad_len);
		}
	}
}
