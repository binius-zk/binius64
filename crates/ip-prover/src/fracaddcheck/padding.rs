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
//! `layer_provers` wraps each tree's own layer prover in a [`ZeroPadMleCheckProver`].
//! That wrapper corrects the unpadded layer's messages at a cost of $O(1)$ per round.
//! A tree the batch has not reached yet has no layer to wrap.
//! It contributes a [`ConstantFraction`] instead.
//! [`unpad_leaf_claim`] inverts the identity above on the claims the batch outputs.

use std::iter;

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::fracaddcheck::FracAddEvalClaim;
use binius_math::{batch_invert::BatchInversion, multilinear::eq::eq_one_var};
use either::Either;
use itertools::izip;

use super::{
	FracAddCheckProver, LayerProver,
	fraction::Fraction,
	zero_pad_mle::{self, ConstantFraction, ZeroPadMleCheckProver},
};

/// The per-tree layer prover: either a real layer of the tree, or the padding layer it contributes
/// while the batch is still above it.
pub(super) type PaddedLayerProver<'a, A, F, P> =
	ZeroPadMleCheckProver<F, Either<LayerProver<'a, A, F, P>, ConstantFraction<F>>>;

/// Builds one padded layer prover per tree, for the layer claimed at `node_point`.
///
/// The provers come back in input order, one per tree, so they stay aligned with `pad_lens` and
/// `claims`. A tree the batch has not reached yet keeps every layer it has.
pub(super) fn layer_provers<'a, A, F, P>(
	provers: &mut [FracAddCheckProver<'a, A, P>],
	pad_lens: &[usize],
	claims: &[Fraction<F>],
	node_point: &[F],
) -> Vec<PaddedLayerProver<'a, A, F, P>>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	let node_len = node_point.len();

	// Every tree's padding segment is a prefix of this one node point, so a single table of prefix
	// products serves the whole batch.
	let pad_eq_prefixes = iter::once(F::ONE)
		.chain(node_point.iter().scan(F::ONE, |acc, &coord| {
			*acc *= eq_one_var(F::ZERO, coord);
			Some(*acc)
		}))
		.collect::<Vec<_>>();

	// De-padding a claim divides by the padding segment's equality weight, so the batch pays one
	// inversion rather than one per tree.
	let mut pad_eq_invs = pad_lens
		.iter()
		.map(|&pad_len| pad_eq_prefixes[pad_len.min(node_len)])
		.collect::<Vec<_>>();
	assert!(
		pad_eq_invs.iter().all(|&pad_eq| pad_eq != F::ZERO),
		"a padding coordinate of the claim point equals one"
	);
	BatchInversion::<F>::new(pad_eq_invs.len()).invert_nonzero(&mut pad_eq_invs);

	izip!(provers, pad_lens, claims, &pad_eq_invs)
		.map(|(prover, &tree_pad_len, &Fraction { num, den }, &pad_eq_inv)| {
			let pad_len = tree_pad_len.min(node_len);
			let point = node_point[pad_len..].to_vec();
			let [num_claim, den_claim] = zero_pad_mle::unpad_claims(pad_eq_inv, [num, den]);

			let inner = if node_len < tree_pad_len {
				// The batch is still above this tree, so every variable of its layer is a padding
				// variable and the de-padded claim is the tree's own fractional sum. The layer is
				// that fraction beside the zero fraction 0/1, and the tree keeps all of its layers.
				Either::Right(ConstantFraction::new(num_claim, den_claim))
			} else {
				Either::Left(prover.pop_layer(FracAddEvalClaim {
					num_eval: num_claim,
					den_eval: den_claim,
					point,
				}))
			};

			zero_pad_mle::new(pad_eq_prefixes[..=pad_len].to_vec(), node_point.to_vec(), inner)
		})
		.collect()
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
		.map(|&coord| eq_one_var(F::ZERO, coord))
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
