// Copyright 2026 The Binius Developers

//! MLE-check prover for one layer of a zero-fraction-padded fractional-addition check.
//!
//! Zero-fraction padding lifts a fractional-addition tree of depth $k$ to depth $n \ge k$ by
//! filling the extra leaves with the zero fraction $0/1$, which leaves the tree's fractional sum
//! unchanged. The numerators are therefore zero-padded and the denominators one-padded. Batching
//! fracadd checks of unequal depths pads each shallow tree up to the deepest one, so the batch's
//! layer loop runs a single uniform schedule and the verifier never learns the individual depths.
//!
//! The *Batched Product Checks of Unequal Depths* appendix of the Binius64 whitepaper derives the
//! multiplicative case, whose one-padding is exactly the padding these denominators carry.
//!
//! No padded layer is ever materialized.
//! The shared wrapper corrects the unpadded layer's own messages, at $O(1)$ per round.

use binius_field::Field;
use binius_ip::sumcheck::RoundCoeffs;

use crate::sumcheck::{
	common::MleCheckProver,
	pad_mle::{Fill, PadMleCheckProver, PadShape, mul_linear},
};

/// The composition one fractional-addition layer reduces.
///
/// Two sibling fractions add by $(a_0 b_1 + a_1 b_0) / (b_0 b_1)$, so the layer carries a numerator
/// claim and a denominator claim over the four halves $[a_0, a_1, b_0, b_1]$.
pub struct FractionAdd;

impl<F: Field> PadShape<F, 4> for FractionAdd {
	const CHILDREN: [Fill; 4] = [Fill::Zero, Fill::Zero, Fill::One, Fill::One];
	const CLAIMS: &'static [Fill] = &[Fill::Zero, Fill::One];

	#[inline(always)]
	fn compose([a_0, a_1, b_0, b_1]: [[F; 2]; 4]) -> Vec<RoundCoeffs<F>> {
		vec![
			mul_linear(a_0, b_1) + &mul_linear(a_1, b_0),
			mul_linear(b_0, b_1),
		]
	}
}

/// MLE-check prover for one layer of a fractional-addition check over a zero-fraction-padded
/// witness.
pub type ZeroPadMleCheckProver<F, Inner> = PadMleCheckProver<F, Inner, FractionAdd, 4>;

/// Divides the padding back out of a padded layer's claims.
///
/// The padded layer's numerator is the unpadded one scaled by $q$ and its denominator the unpadded
/// one pushed through the padding selector at $q$, so recovering the unpadded pair is a scale and a
/// selector at $q^{-1}$. Callers seed the inner prover with the result; for a layer that is *all*
/// padding it is the tree's own fractional sum.
///
/// # Arguments
///
/// * `pad_eq_inv` - The inverse of the padding segment's equality weight $q$.
/// * `claims` - The padded layer's numerator and denominator claims.
pub fn unpad_claims<F: Field>(pad_eq_inv: F, claims: [F; 2]) -> [F; 2] {
	let [num, den] = claims;
	[
		Fill::Zero.at(num, pad_eq_inv),
		Fill::One.at(den, pad_eq_inv),
	]
}

/// Creates the prover for one padded fractional-addition layer.
///
/// The inner MLE-check is seeded at the real segment of the claim point, with the de-padded claims.
/// The prefix table's length fixes the padding segment at one less, and every layer of a batch
/// shares one such table.
pub fn new<F, Inner>(
	pad_eq_prefixes: Vec<F>,
	eval_point: Vec<F>,
	inner: Inner,
) -> ZeroPadMleCheckProver<F, Inner>
where
	F: Field,
	Inner: MleCheckProver<F>,
{
	ZeroPadMleCheckProver::new(pad_eq_prefixes, eval_point, inner)
}

/// The layer a tree contributes while the batch is still above it: one fraction beside the zero
/// fraction $0/1$.
///
/// Such a layer is a padding of the tree's own fractional sum, so its low child is that sum and its
/// high child is identically $0/1$. Every one of its variables is a padding variable, so there is
/// nothing to reduce: the wrapper goes straight to its padding rounds and only ever asks for these
/// four child evaluations.
pub struct ConstantFraction<F> {
	/// The child evaluations `[num_0, num_1, den_0, den_1]`.
	children: [F; 4],
}

impl<F: Field> ConstantFraction<F> {
	/// The layer whose low child is the fraction `(num, den)` and whose high child is $0/1$.
	pub const fn new(num: F, den: F) -> Self {
		Self {
			children: [num, F::ZERO, den, F::ONE],
		}
	}
}

impl<F: Field> MleCheckProver<F> for ConstantFraction<F> {
	fn n_vars(&self) -> usize {
		0
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		panic!("a constant-fraction layer has no variables to reduce")
	}

	fn fold(&mut self, _challenge: F) {
		panic!("a constant-fraction layer has no variables to bind")
	}

	fn finish(self) -> Vec<F> {
		self.children.to_vec()
	}

	fn eval_point(&self) -> &[F] {
		&[]
	}
}

// The prover is checked against the padded layer it stands in for: the same reduction run by an
// ordinary fractional-addition MLE-check over an explicitly materialized padded layer must produce
// the same round polynomials and the same child evaluations.
#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_field::{Random, arithmetic_traits::InvertOrZero, field::FieldOps};
	use binius_math::{
		FieldBuffer,
		multilinear::{eq::eq_one_var, evaluate::evaluate},
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::frac_add_mle;

	type P = Packed128b;
	type F = <P as FieldOps>::Scalar;

	/// Materializes the `pad_len`-fold padding of a layer buffer, whose variables are
	/// `[real | split]`, filling the extra positions with `fill`.
	///
	/// The padding variables land below the real ones, matching the claim-point layout the wrapper
	/// expects.
	fn pad_layer(layer: &FieldBuffer<P>, pad_len: usize, fill: F) -> FieldBuffer<P> {
		let values = (0..1 << (layer.log_len() + pad_len))
			.map(|index| {
				let padding = index & ((1 << pad_len) - 1);
				if padding == 0 {
					layer.get(index >> pad_len)
				} else {
					fill
				}
			})
			.collect::<Vec<_>>();
		FieldBuffer::from_values(&values)
	}

	/// The fractional-addition MLE-check claims on the two buffers' halves at `eval_point`.
	fn split_half_claims(num: &FieldBuffer<P>, den: &FieldBuffer<P>, eval_point: &[F]) -> [F; 2] {
		let (num_0, num_1) = num.split_half();
		let (den_0, den_1) = den.split_half();
		let composite = |compose: fn(F, F, F, F) -> F| {
			let values = (0..num_0.len())
				.map(|i| compose(num_0.get(i), num_1.get(i), den_0.get(i), den_1.get(i)))
				.collect::<Vec<_>>();
			evaluate(&FieldBuffer::<P>::from_values(&values), eval_point)
		};
		[
			composite(|num_0, num_1, den_0, den_1| num_0 * den_1 + num_1 * den_0),
			composite(|_, _, den_0, den_1| den_0 * den_1),
		]
	}

	/// Runs `prover` and an ordinary fracadd MLE-check over the materialized padded layer in
	/// lockstep, requiring the same round polynomials and the same child evaluations.
	fn assert_matches_padded_reference(
		rng: &mut impl Rng,
		padded_num: FieldBuffer<P>,
		padded_den: FieldBuffer<P>,
		eval_point: Vec<F>,
		claims: [F; 2],
		mut prover: impl MleCheckProver<F>,
	) {
		let alloc = GlobalAllocator;
		let n_vars = eval_point.len();
		let mut reference =
			frac_add_mle::new_split_half(&alloc, padded_num, padded_den, eval_point, claims);

		for round in 0..n_vars {
			assert_eq!(prover.n_vars(), n_vars - round);
			assert_eq!(prover.eval_point(), reference.eval_point());
			assert_eq!(prover.execute(), reference.execute(), "round {round}");

			let challenge = F::random(&mut *rng);
			prover.fold(challenge);
			reference.fold(challenge);
		}

		assert_eq!(prover.finish(), reference.finish());
	}

	/// Prefix products over the first `pad_len` coordinates of `eval_point`.
	fn pad_eq_prefixes(eval_point: &[F], pad_len: usize) -> Vec<F> {
		iter::once(F::ONE)
			.chain(eval_point[..pad_len].iter().scan(F::ONE, |acc, &coord| {
				*acc *= eq_one_var(F::ZERO, coord);
				Some(*acc)
			}))
			.collect()
	}

	/// The padded layer of an ordinary tree layer, and the claims on it.
	fn padded_layer(
		rng: &mut impl Rng,
		num: &FieldBuffer<P>,
		den: &FieldBuffer<P>,
		pad_len: usize,
	) -> (FieldBuffer<P>, FieldBuffer<P>, Vec<F>, [F; 2]) {
		// The zero fraction 0/1 fills the padding positions.
		let padded_num = pad_layer(num, pad_len, F::ZERO);
		let padded_den = pad_layer(den, pad_len, F::ONE);
		let eval_point = random_scalars::<F>(rng, padded_num.log_len() - 1);
		let claims = split_half_claims(&padded_num, &padded_den, &eval_point);
		(padded_num, padded_den, eval_point, claims)
	}

	#[test]
	fn matches_padded_reference() {
		let mut rng = StdRng::seed_from_u64(1);
		let alloc = GlobalAllocator;

		// A layer of `n_real_rounds + 1` variables splits into two halves of `n_real_rounds`, so
		// zero real rounds is a layer of one fraction pair. The packing width is 4 scalars, so the
		// range straddles it in both directions, and 3 and 5 padding rows are not powers of two.
		for n_real_rounds in [0, 1, 2, 3, 5] {
			for pad_len in [0, 1, 2, 3, 5] {
				let num = random_field_buffer::<P>(&mut rng, n_real_rounds + 1);
				let den = random_field_buffer::<P>(&mut rng, n_real_rounds + 1);
				let (padded_num, padded_den, eval_point, claims) =
					padded_layer(&mut rng, &num, &den, pad_len);

				let prefixes = pad_eq_prefixes(&eval_point, pad_len);
				let pad_eq_inv = prefixes[pad_len].invert_or_zero();
				let inner = frac_add_mle::new_split_half(
					&alloc,
					num,
					den,
					eval_point[pad_len..].to_vec(),
					unpad_claims(pad_eq_inv, claims),
				);
				let prover = new(prefixes, eval_point.clone(), inner);

				assert_matches_padded_reference(
					&mut rng, padded_num, padded_den, eval_point, claims, prover,
				);
			}
		}
	}

	// While the batch is still above a tree, that tree's layer is a padding of its own fractional
	// sum, whose high child is the zero fraction 0/1. `ConstantFraction` stands in for it, so it
	// must drive the padding rounds exactly as the materialized layer does.
	#[test]
	fn constant_fraction_matches_padded_reference() {
		let mut rng = StdRng::seed_from_u64(2);

		for pad_len in [1, 2, 3, 4, 5] {
			let root_num = F::random(&mut rng);
			let root_den = F::random(&mut rng);
			let num = FieldBuffer::<P>::from_values(&[root_num, F::ZERO]);
			let den = FieldBuffer::<P>::from_values(&[root_den, F::ONE]);
			let (padded_num, padded_den, eval_point, claims) =
				padded_layer(&mut rng, &num, &den, pad_len);

			// De-padding an all-padding layer's claims recovers the fraction it stands for, which
			// is how the driver reaches `ConstantFraction` without carrying the root separately.
			let prefixes = pad_eq_prefixes(&eval_point, pad_len);
			let pad_eq_inv = prefixes[pad_len].invert_or_zero();
			assert_eq!(unpad_claims(pad_eq_inv, claims), [root_num, root_den]);
			let prover =
				new(prefixes, eval_point.clone(), ConstantFraction::new(root_num, root_den));

			assert_matches_padded_reference(
				&mut rng, padded_num, padded_den, eval_point, claims, prover,
			);
		}
	}
}
