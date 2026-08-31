// Copyright 2026 The Binius Developers

//! MLE-check prover for one layer of a padded tree check.
//!
//! Padding fills a shallow tree's extra leaves with the value its fold ignores.
//! Batching unequal depths pads each tree up to the deepest, so one schedule runs the batch.
//!
//! No padded layer is materialized.
//! Every padded quantity is one interpolation: the fill at weight zero, the value at weight one.

use std::{array, iter, marker::PhantomData, mem};

use binius_field::Field;
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::multilinear::eq::eq_one_var;

use crate::sumcheck::common::MleCheckProver;

/// The value a padded quantity takes where the padding fills the layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
	/// Zero, which is what a padded fractional numerator holds.
	Zero,
	/// One, which is what a padded denominator or product factor holds.
	One,
}

impl Fill {
	/// The fill itself.
	#[inline(always)]
	const fn value<F: Field>(self) -> F {
		match self {
			Self::Zero => F::ZERO,
			Self::One => F::ONE,
		}
	}

	/// The padded quantity at padding weight `s`: the fill at zero, `v` at one.
	pub fn at<F: Field>(self, v: F, s: F) -> F {
		let fill = self.value::<F>();
		fill + (v - fill) * s
	}

	/// The padded quantity as a linear polynomial, for a padding weight given in monomial
	/// coefficients.
	#[inline(always)]
	fn linear<F: Field>(self, v: F, [s_0, s_1]: [F; 2]) -> [F; 2] {
		let fill = self.value::<F>();
		let slope = v - fill;
		[fill + slope * s_0, slope * s_1]
	}

	/// Scales a round polynomial to `weight`, with the fill carrying the residual weight.
	#[inline(always)]
	fn correct<F: Field>(self, coeffs: &mut RoundCoeffs<F>, weight: F) {
		*coeffs *= weight;
		// Off the all-zeros padding slab the composition is constant at the fill, so only a
		// fill of one contributes.
		if self == Self::One {
			coeffs.0[0] += F::ONE - weight;
		}
	}
}

/// The composition one padded layer reduces.
pub trait PadShape<F: Field, const N: usize> {
	/// The fill of each child the unpadded layer reduces.
	const CHILDREN: [Fill; N];
	/// The fill of each claim's composition, one entry per emitted polynomial.
	const CLAIMS: &'static [Fill];

	/// The claims' round polynomials, from the padded children as linear polynomials in this
	/// round's variable.
	fn compose(children: [[F; 2]; N]) -> Vec<RoundCoeffs<F>>;
}

/// The product of two linear polynomials, in monomial coefficients.
#[inline(always)]
pub fn mul_linear<F: Field>([p_0, p_1]: [F; 2], [q_0, q_1]: [F; 2]) -> RoundCoeffs<F> {
	RoundCoeffs(vec![p_0 * q_0, p_0 * q_1 + p_1 * q_0, p_1 * q_1])
}

/// Prefix products of the equality weights of `coords` against zero.
///
/// Entry `i` is the product over the first `i` coordinates, so the last entry weights them all.
/// Every layer of a batch shares one claim point, so one table serves all of its trees.
pub fn pad_eq_prefixes<F: Field>(coords: &[F]) -> Vec<F> {
	iter::once(F::ONE)
		.chain(coords.iter().scan(F::ONE, |acc, &coord| {
			*acc *= eq_one_var(F::ZERO, coord);
			Some(*acc)
		}))
		.collect()
}

/// MLE-check prover for one layer of a tree check over a padded witness.
///
/// The claim point is node coordinates only, padding lowest:
///
/// ```text
///     [ padding (nu) | real (m) ]
/// ```
///
/// Variables bind highest first, so the real rounds run before the padding ones.
/// Finishing returns the padded layer's child evaluations.
pub struct PadMleCheckProver<F: Field, Inner, S, const N: usize> {
	/// The padded claim point `[padding | real]`, low variables first.
	eval_point: Vec<F>,
	/// Length of the point's padding segment.
	pad_len: usize,
	/// Number of folds performed so far.
	round: usize,
	/// Equality weights of the claim point's padding segment: entry `i` is
	/// $\prod_{c < i} \textsf{eq}(0, \rho_{\text{pa}, c})$, so the last entry is $q$.
	pad_eq_prefixes: Vec<F>,
	/// Which segment of rounds the prover is in.
	phase: Phase<F, Inner, N>,
	/// The composition being reduced, carried by the type alone.
	shape: PhantomData<S>,
}

/// The segment of rounds the prover is in.
enum Phase<F, Inner, const N: usize> {
	/// Reducing the unpadded layer's real node variables.
	Real(Inner),
	/// Every real variable is bound, leaving a closed form in these scalars.
	Padding {
		/// The unpadded layer's child evaluations.
		children: [F; N],
		/// $\prod \textsf{eq}(0, r)$ over the padding challenges bound so far, which is the
		/// constant factor of $E$.
		bound_eq: F,
	},
}

impl<F: Field, Inner: MleCheckProver<F>, S, const N: usize> PadMleCheckProver<F, Inner, S, N> {
	/// Creates the prover for one padded layer.
	///
	/// # Arguments
	///
	/// * `pad_eq_prefixes` - Equality weights of the padding segment, entry `i` being the product
	///   over the first `i` coordinates. Its length fixes that segment at one less.
	/// * `eval_point` - The padded layer's claim point, `[padding | real]`.
	/// * `inner` - The unpadded layer's MLE-check, seeded at the real segment of the claim point
	///   with the de-padded claims.
	///
	/// # Preconditions
	///
	/// * `pad_eq_prefixes` is non-empty
	/// * `eval_point.len() + 1 >= pad_eq_prefixes.len()`
	/// * `inner.n_vars() == eval_point.len() + 1 - pad_eq_prefixes.len()`
	///
	/// # Panics
	///
	/// Panics if the padding segment's equality weight is zero, which needs a verifier challenge
	/// to land on one.
	pub fn new(pad_eq_prefixes: Vec<F>, eval_point: Vec<F>, inner: Inner) -> Self {
		let pad_len = pad_eq_prefixes
			.len()
			.checked_sub(1)
			.expect("precondition: non-empty");
		assert!(eval_point.len() >= pad_len); // precondition
		assert!(
			pad_eq_prefixes[pad_len] != F::ZERO,
			"a padding coordinate of the claim point equals one"
		);
		assert_eq!(inner.n_vars(), eval_point.len() - pad_len); // precondition

		let mut prover = Self {
			eval_point,
			pad_len,
			round: 0,
			pad_eq_prefixes,
			phase: Phase::Real(inner),
			shape: PhantomData,
		};
		// A layer with no real variables starts in the padding phase.
		prover.advance();
		prover
	}

	/// The number of rounds that reduce the unpadded layer's real variables.
	const fn n_real_rounds(&self) -> usize {
		self.eval_point.len() - self.pad_len
	}

	/// Finishes the inner prover once its last real variable is bound, fixing the child evaluations
	/// the padding rounds close over.
	fn advance(&mut self) {
		if self.round != self.n_real_rounds() || !matches!(self.phase, Phase::Real(_)) {
			return;
		}
		// The guard above pins the phase, so this placeholder is overwritten before it is read.
		let placeholder = Phase::Padding {
			children: [F::ONE; N],
			bound_eq: F::ONE,
		};
		let Phase::Real(inner) = mem::replace(&mut self.phase, placeholder) else {
			unreachable!("the guard checked the phase");
		};
		self.phase = Phase::Padding {
			children: inner
				.finish()
				.try_into()
				.expect("the layer prover reduces one multilinear per child"),
			bound_eq: F::ONE,
		};
	}
}

impl<F: Field, Inner: MleCheckProver<F>, S: PadShape<F, N>, const N: usize> MleCheckProver<F>
	for PadMleCheckProver<F, Inner, S, N>
{
	fn n_vars(&self) -> usize {
		self.eval_point.len() - self.round
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		// Destructured so a padding round can read the prefix table while the phase is borrowed.
		let Self {
			eval_point,
			pad_len,
			round,
			pad_eq_prefixes,
			phase,
			..
		} = self;
		let n_vars = eval_point.len() - *round;

		// Both phases produce the polynomials of an unpadded composition, and the weight the
		// padding still owes them.
		let (mut coeffs, weight) = match phase {
			Phase::Real(inner) => {
				let coeffs = inner.execute();
				assert_eq!(
					coeffs.len(),
					S::CLAIMS.len(),
					"the layer prover carries one claim per composition"
				);
				(coeffs, pad_eq_prefixes[*pad_len])
			}
			Phase::Padding { children, bound_eq } => {
				// E(X) = bound_eq * eq(0, X) in monomial coefficients.
				let big_e = [*bound_eq, -*bound_eq];
				let padded = array::from_fn(|i| S::CHILDREN[i].linear(children[i], big_e));
				// The equality weight of the padding coordinates still unbound below this round's.
				(S::compose(padded), pad_eq_prefixes[n_vars - 1])
			}
		};

		// Corrected in place, so a round allocates only the vector it returns.
		for (coeffs, &fill) in iter::zip(&mut coeffs, S::CLAIMS) {
			fill.correct(coeffs, weight);
		}
		coeffs
	}

	fn fold(&mut self, challenge: F) {
		match &mut self.phase {
			Phase::Real(inner) => inner.fold(challenge),
			Phase::Padding { bound_eq, .. } => {
				*bound_eq *= eq_one_var(F::ZERO, challenge);
			}
		}
		self.round += 1;
		self.advance();
	}

	fn finish(self) -> Vec<F> {
		match self.phase {
			Phase::Padding { children, bound_eq } => iter::zip(children, S::CHILDREN)
				.map(|(child, fill)| fill.at(child, bound_eq))
				.collect(),
			Phase::Real(_) => panic!("finish requires every variable to be bound"),
		}
	}

	fn eval_point(&self) -> &[F] {
		&self.eval_point[..self.n_vars()]
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b;

	use super::*;
	use crate::{
		fracaddcheck::zero_pad_mle::FractionAdd, prodcheck::one_pad_mle::BivariateProduct,
	};

	type F = Ghash128b;

	// Invariant: at padding weight zero every child sits at its fill, so the composition does too.
	// The round polynomials carry the residual weight as a constant, which needs exactly that.
	fn assert_claim_fills_are_the_composed_child_fills<S, const N: usize>()
	where
		S: PadShape<F, N>,
	{
		// A witness value that is neither fill, so a shape that reads the child rather than its
		// fill cannot pass by coincidence.
		let witness = F::new(3);
		// Weight zero in monomial coefficients: the padded child collapses to its fill.
		let padded = S::CHILDREN.map(|fill| fill.linear(witness, [F::ZERO, F::ZERO]));

		let composed = S::compose(padded);
		assert_eq!(composed.len(), S::CLAIMS.len());
		for (coeffs, &fill) in iter::zip(&composed, S::CLAIMS) {
			assert_eq!(coeffs.0[0], fill.value::<F>());
			assert!(coeffs.0[1..].iter().all(|&coeff| coeff == F::ZERO));
		}
	}

	#[test]
	fn claim_fills_are_the_composed_child_fills() {
		assert_claim_fills_are_the_composed_child_fills::<FractionAdd, 4>();
		assert_claim_fills_are_the_composed_child_fills::<BivariateProduct, 2>();
	}
}
