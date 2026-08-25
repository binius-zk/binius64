// Copyright 2026 The Binius Developers

//! Zero-fraction padding for batching fractional-addition trees of unequal depths.
//!
//! A batch runs one uniform layer schedule, so every tree in it must be of the same depth.
//! Padding lifts a tree of depth $m$ to the batch's depth $n \ge m$.
//! The $n - m$ extra leaf positions hold the zero fraction $0/1$, the additive identity.
//! So the tree's fractional sum is unchanged and the verifier never learns the individual depths.
//!
//! The padding variables are the lowest ones.
//! Padding a witness $(N, D)$ over $\nu = n - m$ of them gives
//!
//! $$
//! N'(X_\text{pad}, X_\text{real}) = N(X_\text{real}) \cdot \text{eq}(0^\nu; X_\text{pad}),
//! \qquad
//! D'(X_\text{pad}, X_\text{real}) = 1 + \bigl( D(X_\text{real}) - 1 \bigr) \cdot
//! \text{eq}(0^\nu; X_\text{pad}),
//! $$
//!
//! so the numerators are zero-padded and the denominators one-padded.
//!
//! The prover never materializes a padded witness.
//! [`PaddedBatch`] holds the trees and how deep each one sits.
//! Every layer it pops hands out one [`PaddedLayerProver`] per tree.
//!
//! Each of those wraps the tree's own layer prover in a [`ZeroPadMleCheckProver`].
//! That wrapper corrects the unpadded layer's messages at a cost of $O(1)$ per round.
//!
//! A tree the batch has not reached yet has no layer to wrap.
//! It contributes a [`ConstantFraction`] instead.
//!
//! [`unpad_leaf_claim`] inverts the identity above on the claims the batch outputs.

use std::iter;

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::{fracaddcheck::FracAddEvalClaim, sumcheck::RoundCoeffs};
use binius_math::{batch_invert::BatchInversion, multilinear::hypercube::Hypercube};
use either::Either;
use itertools::izip;

use super::{
	FracAddCircuit, LayerProver,
	fraction::Fraction,
	zero_pad_mle::{self, ConstantFraction, ZeroPadMleCheckProver},
};
use crate::sumcheck::common::MleCheckProver;

/// The layer one tree contributes to the batch's current depth.
///
/// Once the batch has reached the tree, that is a real layer of it.
///
/// Until then it is the layer standing in for one.
/// That stand-in is the tree's own fractional sum beside the zero fraction $0/1$.
type TreeLayer<'a, A, F, P> = Either<LayerProver<'a, A, F, P>, ConstantFraction<F>>;

/// One tree's contribution to a batched layer, lifted to the batch's depth.
///
/// Either kind of [`TreeLayer`] is a layer of the padded tree.
///
/// So either one needs its messages corrected.
/// One [`ZeroPadMleCheckProver`] does that for both.
pub(super) struct PaddedLayerProver<'a, A: Allocator, F: Field, P: PackedField<Scalar = F>>(
	ZeroPadMleCheckProver<F, TreeLayer<'a, A, F, P>>,
);

impl<A, F, P> MleCheckProver<F> for PaddedLayerProver<'_, A, F, P>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	fn n_vars(&self) -> usize {
		self.0.n_vars()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		self.0.execute()
	}

	fn fold(&mut self, challenge: F) {
		self.0.fold(challenge);
	}

	fn finish(self) -> Vec<F> {
		self.0.finish()
	}

	fn eval_point(&self) -> &[F] {
		self.0.eval_point()
	}
}

/// A batch's trees, lifted to one uniform depth by zero-fraction padding.
///
/// The batch runs a single layer schedule, so every tree in it must have the same depth.
/// Padding lifts each tree to the depth of the deepest one.
///
/// A padded tree sits out the layers above its own root.
/// It contributes a stand-in layer for each of them, and only then starts spending its real ones.
///
/// No padded layer is ever materialized.
/// The whole padding is these two vectors, plus the $O(1)$ per-round correction each layer carries.
pub(super) struct PaddedBatch<'a, A: Allocator, P: PackedField> {
	/// The trees, in input order, each spending its layers deepest-first.
	trees: Vec<FracAddCircuit<'a, A, P>>,
	/// How much depth each tree is padded by, in tree order.
	pad_lens: Vec<usize>,
	/// The depth every tree is padded to, which is how many layers the schedule runs.
	n_layers: usize,
}

impl<'a, A, F, P> PaddedBatch<'a, A, P>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	/// Pads `trees` up to the depth of the deepest one.
	///
	/// # Preconditions
	/// * `trees` is non-empty
	/// * at least one tree has a layer
	pub(super) fn new(trees: Vec<FracAddCircuit<'a, A, P>>) -> Self {
		let n_layers = trees
			.iter()
			.map(FracAddCircuit::n_layers)
			.max()
			.expect("precondition: trees is non-empty");
		assert!(n_layers >= 1); // precondition

		let pad_lens = trees
			.iter()
			.map(|tree| n_layers - tree.n_layers())
			.collect();

		Self {
			trees,
			pad_lens,
			n_layers,
		}
	}

	/// How many layers the batch's schedule runs.
	pub(super) const fn n_layers(&self) -> usize {
		self.n_layers
	}

	/// How many trees the batch holds.
	pub(super) const fn n_trees(&self) -> usize {
		self.trees.len()
	}

	/// The allocator every tree's layer buffers are drawn from.
	pub(super) fn alloc(&self) -> &'a A {
		self.trees[0].alloc
	}

	/// Consumes the spent batch.
	///
	/// A depth-zero tree is all padding, so it is passed through every layer and never popped.
	/// Every tree that had layers has spent them all.
	pub(super) fn finish(self) {
		debug_assert!(
			self.trees.iter().all(|tree| tree.n_layers() == 0),
			"every tree with layers is exhausted after n_layers reductions"
		);
	}

	/// Pops one padded layer prover per tree, for the layer claimed at `node_point`.
	///
	/// The provers come back in tree order, so they stay aligned with `claims`.
	///
	/// A tree the batch has not reached yet keeps every layer it has.
	pub(super) fn pop_layer(
		&mut self,
		claims: &[Fraction<F>],
		node_point: &[F],
	) -> Vec<PaddedLayerProver<'a, A, F, P>> {
		let node_len = node_point.len();

		// Every tree's padding segment is a prefix of this one node point, so a single table of
		// prefix products serves the whole batch.
		let pad_eq_prefixes = iter::once(F::ONE)
			.chain(node_point.iter().scan(F::ONE, |acc, &coord| {
				*acc *= Hypercube::One.eq_one_var(F::ZERO, coord);
				Some(*acc)
			}))
			.collect::<Vec<_>>();

		// De-padding a claim divides by the padding segment's equality weight, so the batch pays
		// one inversion rather than one per tree.
		let mut pad_eq_invs = self
			.pad_lens
			.iter()
			.map(|&pad_len| pad_eq_prefixes[pad_len.min(node_len)])
			.collect::<Vec<_>>();
		assert!(
			pad_eq_invs.iter().all(|&pad_eq| pad_eq != F::ZERO),
			"a padding coordinate of the claim point equals one"
		);
		BatchInversion::<F>::new(pad_eq_invs.len()).invert_nonzero(&mut pad_eq_invs);

		izip!(&mut self.trees, &self.pad_lens, claims, &pad_eq_invs)
			.map(|(tree, &tree_pad_len, &Fraction { num, den }, &pad_eq_inv)| {
				let pad_len = tree_pad_len.min(node_len);
				let point = node_point[pad_len..].to_vec();
				let [num_claim, den_claim] = zero_pad_mle::unpad_claims(pad_eq_inv, [num, den]);

				let inner = if node_len < tree_pad_len {
					// The batch is still above this tree, so every variable of its layer is a
					// padding variable and the de-padded claim is the tree's own fractional sum.
					// The layer is that fraction beside the zero fraction 0/1, and the tree keeps
					// all of its layers.
					Either::Right(ConstantFraction::new(num_claim, den_claim))
				} else {
					Either::Left(tree.pop_layer(FracAddEvalClaim {
						num_eval: num_claim,
						den_eval: den_claim,
						point,
					}))
				};

				PaddedLayerProver(zero_pad_mle::new(
					pad_eq_prefixes[..=pad_len].to_vec(),
					node_point.to_vec(),
					inner,
				))
			})
			.collect()
	}
}

/// Reduces a leaf claim on a zero-fraction-padded witness to the claim on the witness itself.
///
/// A batched fractional-addition check over trees of unequal depths pads each shallow tree.
/// [`binius_ip::fracaddcheck::verify`] is oblivious to that padding.
/// So the claims it outputs for such a tree are claims on the padded witness $(N', D')$.
/// Its padding variables are the lowest `n_pad_vars` coordinates of `point`.
///
/// This divides out the padding variables' equality weight and drops them from the point.
/// What remains are the claims on $N$ and $D$.
///
/// # Arguments
///
/// * `fraction` - The claimed numerator and denominator evaluations of the padded witness.
/// * `point` - The reduced evaluation point, with the batch's selector coordinates already
///   stripped.
/// * `n_pad_vars` - How much depth this tree was padded by: the batch's layer count less the tree's
///   own.
///
/// # Preconditions
/// * `point.len() >= n_pad_vars`
///
/// # Panics
///
/// Panics if the padding coordinates' equality weight is zero, which requires one of them to equal
/// one. They are the verifier's own challenges, so no prover can induce this; it happens with
/// probability at most $\nu / |K|$.
pub fn unpad_leaf_claim<F: Field>(
	fraction: Fraction<F>,
	point: &[F],
	n_pad_vars: usize,
) -> FracAddEvalClaim<F> {
	assert!(point.len() >= n_pad_vars); // precondition

	let pad_eq = point[..n_pad_vars]
		.iter()
		.map(|&coord| Hypercube::One.eq_one_var(F::ZERO, coord))
		.product::<F>();
	assert!(pad_eq != F::ZERO, "a padding coordinate equals one");
	let pad_eq_inv = pad_eq.invert_or_zero();

	let Fraction {
		num: num_eval,
		den: den_eval,
	} = fraction;
	FracAddEvalClaim {
		num_eval: num_eval * pad_eq_inv,
		den_eval: F::ONE + (den_eval - F::ONE) * pad_eq_inv,
		point: point[n_pad_vars..].to_vec(),
	}
}

#[cfg(test)]
mod tests {
	use binius_field::FieldOps;
	use binius_ip::fracaddcheck;
	use binius_math::test_utils::{Packed128b, random_scalars};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	type F = <Packed128b as FieldOps>::Scalar;

	proptest! {
		// Invariant: `unpad_leaf_claim` is the exact inverse of `pad_leaf_fraction`.
		//
		// The verifier pads a transparent leaf fraction, the prover unpads the claim it gets back.
		// Only an end-to-end proof failure notices if either map drifts from the other.
		#[test]
		fn unpad_leaf_claim_inverts_pad_leaf_fraction(
			seed in any::<u64>(),
			n_pad_vars in 0usize..=5,
			n_real_vars in 0usize..=5,
		) {
			let mut rng = StdRng::seed_from_u64(seed);

			// Splitting the point's length in two keeps `n_pad_vars <= point.len()` by construction.
			let point = random_scalars::<F>(&mut rng, n_pad_vars + n_real_vars);
			let halves = random_scalars::<F>(&mut rng, 2);
			let fraction = Fraction::new(halves[0], halves[1]);

			let pad_eq = point[..n_pad_vars]
				.iter()
				.map(|&coord| Hypercube::One.eq_one_var(F::ZERO, coord))
				.product::<F>();
			// Unpadding asserts on a zero weight, which needs a padding coordinate equal to one.
			// Random 128-bit coordinates never are, so this rejects nothing.
			prop_assume!(pad_eq != F::ZERO);

			let padded = fracaddcheck::pad_leaf_fraction(fraction.into(), pad_eq);
			let claim = unpad_leaf_claim(padded.into(), &point, n_pad_vars);

			prop_assert_eq!(claim.num_eval, fraction.num);
			prop_assert_eq!(claim.den_eval, fraction.den);
			// The padding variables are the lowest ones, so unpadding strips them off the point.
			prop_assert_eq!(claim.point, point[n_pad_vars..].to_vec());
		}
	}
}
