// Copyright 2026 The Binius Developers

//! MLE-check prover for one layer of a zero-fraction-padded fractional-addition check.
//!
//! Zero-fraction padding lifts a fractional-addition tree of depth $k$ to depth $n \ge k$ by
//! filling the extra leaves with the zero fraction $0/1$, which leaves the tree's fractional sum
//! unchanged. The numerators are therefore zero-padded and the denominators one-padded. Batching
//! fracadd checks of unequal depths pads each shallow tree up to the deepest one, so the batch's
//! layer loop runs a single uniform schedule and the verifier never learns the individual depths.
//!
//! This is the fractional-addition analog of the padding
//! [`crate::prodcheck::batch_prove`] applies to shallow product trees; the *Batched Product Checks
//! of Unequal Depths* appendix of the Binius64 whitepaper derives the multiplicative case, whose
//! one-padding is exactly the padding these denominators carry.
//!
//! The point of this module is that the prover never materializes a padded layer:
//! [`ZeroPadMleCheckProver`] wraps the unpadded layer's own MLE-check and corrects its messages at
//! a cost of $O(1)$ per round.

use std::{iter, mem};

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldVec, multilinear::eq::eq_one_var};

use crate::sumcheck::{
	common::MleCheckProver,
	frac_add_mle::{self, LayerProver},
};

/// The one-padding selector $\textsf{sel}(s, v) = 1 + (v - 1) s$.
///
/// It interpolates between the constant one at $s = 0$ and $v$ at $s = 1$, which is how a padded
/// leaf position holds the zero fraction's denominator while a real one holds the witness value.
fn select<F: Field>(s: F, v: F) -> F {
	F::ONE + (v - F::ONE) * s
}

/// The product of two linear polynomials, in monomial coefficients.
fn mul_linear<F: Field>([p_0, p_1]: [F; 2], [q_0, q_1]: [F; 2]) -> RoundCoeffs<F> {
	RoundCoeffs(vec![p_0 * q_0, p_0 * q_1 + p_1 * q_0, p_1 * q_1])
}

/// MLE-check prover for one layer of a fractional-addition check over a zero-fraction-padded
/// witness.
///
/// The tree's fractional sum is a scalar, so the layer's claim point is node coordinates only,
/// split into two segments with the padding ones lowest:
///
/// ```text
///     [ padding (nu) | real (m) ]
/// ```
///
/// The padded layer is the unpadded one with its numerators scaled by
/// $\textsf{eq}(0^\nu, Z')$ and its denominators wrapped in the one-padding
/// $\textsf{sel}(\textsf{eq}(0^\nu, Z'), \cdot)$ over the padding variables. MLE-check binds
/// variables from the highest index down, so the real rounds come first and the padding rounds
/// last:
///
/// - **Real rounds.** Delegate to `inner`, the ordinary MLE-check over the unpadded layer, and
///   correct its two round polynomials, where $q$ is the equality weight $\textsf{eq}(0^\nu,
///   \rho_\text{pa})$ of the claim point's padding segment. Off the all-zeros padding slab every
///   leaf is the zero fraction, whose numerator composition vanishes and whose denominator
///   composition is one, so the numerator's polynomial is scaled to $q \cdot R(X)$ and the
///   denominator's shifted to $1 + q \cdot (R(X) - 1)$.
/// - **Padding rounds.** No multilinear is touched. Every real variable is bound by now, so
///   `inner`'s four child evaluations are scalars and both round polynomials are closed forms in
///   them, quadratic through $E(X)$, the equality weight of the padding coordinates already bound
///   together with this round's.
///
/// [`MleCheckProver::finish`] returns the *padded* layer's child evaluations, which is what the
/// batch's selector rounds consume.
pub struct ZeroPadMleCheckProver<F: Field, Inner> {
	/// The padded claim point `[padding | real]`, low variables first.
	eval_point: Vec<F>,
	/// Length of the point's padding segment.
	pad_len: usize,
	/// Number of folds performed so far.
	round: usize,
	/// Equality weights of the claim point's padding segment: entry `i` is
	/// $\prod_{c < i} \textsf{eq}(0, \rho_{\text{pa}, c})$, so the last entry is $q$.
	pad_eq_prefixes: Vec<F>,
	phase: Phase<F, Inner>,
}

/// The segment of rounds the prover is in. See [`ZeroPadMleCheckProver`].
enum Phase<F, Inner> {
	/// Reducing the unpadded layer's real node variables.
	Real(Inner),
	/// Every real variable is bound, leaving a closed form in these scalars.
	Padding {
		/// The unpadded layer's child evaluations `[num_0, num_1, den_0, den_1]`.
		children: [F; 4],
		/// $\prod \textsf{eq}(0, r)$ over the padding challenges bound so far, which is the
		/// constant factor of $E$.
		bound_eq: F,
	},
}

/// Creates the prover for one padded fractional-addition layer.
///
/// # Arguments
///
/// * `num`, `den` - The unpadded child layer's numerator and denominator buffers, whose low and
///   high halves on their highest variable are the multilinears this layer adds.
/// * `pad_len` - Length of `eval_point`'s padding segment. Zero leaves the inner reduction
///   uncorrected.
/// * `eval_point` - The padded layer's claim point, `[padding | real]`.
/// * `claims` - The padded layer's claimed numerator and denominator evaluations at `eval_point`.
///
/// # Preconditions
///
/// * `num.log_len() == den.log_len() >= 1`
/// * `eval_point.len() == num.log_len() - 1 + pad_len`
///
/// # Panics
///
/// Panics if the padding segment's equality weight $q$ is zero, which requires one of its
/// coordinates — all verifier challenges — to equal one, and so happens with probability at most
/// $\nu / |K|$.
pub fn new<'alloc, A, F, P>(
	alloc: &'alloc A,
	num: FieldVec<P, A>,
	den: FieldVec<P, A>,
	pad_len: usize,
	eval_point: Vec<F>,
	claims: [F; 2],
) -> ZeroPadMleCheckProver<F, LayerProver<'alloc, A, F, P>>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert!(num.log_len() >= 1); // precondition
	let n_real_rounds = num.log_len() - 1;
	assert_eq!(eval_point.len(), pad_len + n_real_rounds); // precondition

	// Prefix products over the padding segment, so both the round polynomials' `e` factors and the
	// claims' `q` are lookups.
	let pad_eq_prefixes = iter::once(F::ONE)
		.chain(eval_point[..pad_len].iter().scan(F::ONE, |acc, &coord| {
			*acc *= eq_one_var(F::ZERO, coord);
			Some(*acc)
		}))
		.collect::<Vec<_>>();
	let pad_eq = pad_eq_prefixes[pad_len];
	assert!(pad_eq != F::ZERO, "a padding coordinate of the claim point equals one");

	// The padded claims are the images of the unpadded ones under the padding, so the inner prover
	// starts from the preimages.
	let pad_eq_inv = pad_eq.invert_or_zero();
	let [num_claim, den_claim] = claims;
	let inner_claims = [num_claim * pad_eq_inv, select(pad_eq_inv, den_claim)];
	let (inner, _cols) =
		frac_add_mle::new_split_half(alloc, num, den, eval_point[pad_len..].to_vec(), inner_claims);

	let mut prover = ZeroPadMleCheckProver {
		eval_point,
		pad_len,
		round: 0,
		pad_eq_prefixes,
		phase: Phase::Real(inner),
	};
	// A layer with no real variables starts in the padding phase.
	prover.advance();
	prover
}

impl<F: Field, Inner: MleCheckProver<F>> ZeroPadMleCheckProver<F, Inner> {
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
			children: [F::ONE; 4],
			bound_eq: F::ONE,
		};
		let Phase::Real(inner) = mem::replace(&mut self.phase, placeholder) else {
			unreachable!("the guard checked the phase");
		};
		self.phase = Phase::Padding {
			children: inner
				.finish()
				.try_into()
				.expect("the layer prover reduces four multilinears"),
			bound_eq: F::ONE,
		};
	}
}

impl<F: Field, Inner: MleCheckProver<F>> MleCheckProver<F> for ZeroPadMleCheckProver<F, Inner> {
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
		} = self;
		let n_vars = eval_point.len() - *round;

		match phase {
			Phase::Real(inner) => {
				let mut round_coeffs = inner.execute();
				assert_eq!(round_coeffs.len(), 2, "the layer prover carries two claims");
				let mut den = round_coeffs.pop().expect("the vector holds two elements");
				let mut num = round_coeffs.pop().expect("the vector holds two elements");
				// Off the all-zeros padding slab every leaf is the zero fraction: its numerator
				// composition vanishes and its denominator composition is one, the latter picking
				// up the residual weight 1 - q.
				let pad_eq = pad_eq_prefixes[*pad_len];
				num *= pad_eq;
				den *= pad_eq;
				den.0[0] += F::ONE - pad_eq;
				vec![num, den]
			}
			Phase::Padding { children, bound_eq } => {
				// The equality weight of the padding coordinates still unbound below this round's.
				let unbound_eq = pad_eq_prefixes[n_vars - 1];
				// E(X) = bound_eq * eq(0, X) in monomial coefficients.
				let big_e = [*bound_eq, -*bound_eq];
				let [num_0, num_1, den_0, den_1] = *children;
				// The padded children, linear in X: a numerator is scaled by E(X), a denominator
				// pushed through the one-padding selector at E(X).
				let [a_0, a_1] = [num_0, num_1].map(|num| [num * big_e[0], num * big_e[1]]);
				let [b_0, b_1] =
					[den_0, den_1].map(|den| [select(big_e[0], den), (den - F::ONE) * big_e[1]]);

				// R_num(X) = e * (A_0 B_1 + A_1 B_0) and R_den(X) = (1 - e) + e * B_0 B_1: off the
				// all-zeros slab of the still-unbound padding coordinates the numerators vanish and
				// both denominators are one.
				let num_coeffs = (mul_linear(a_0, b_1) + &mul_linear(a_1, b_0)) * unbound_eq;
				let mut den_coeffs = mul_linear(b_0, b_1) * unbound_eq;
				den_coeffs.0[0] += F::ONE - unbound_eq;
				vec![num_coeffs, den_coeffs]
			}
		}
	}

	fn fold(&mut self, challenge: F) {
		match &mut self.phase {
			Phase::Real(inner) => inner.fold(challenge),
			Phase::Padding { bound_eq, .. } => *bound_eq *= eq_one_var(F::ZERO, challenge),
		}
		self.round += 1;
		self.advance();
	}

	fn finish(self) -> Vec<F> {
		match self.phase {
			Phase::Padding { children, bound_eq } => {
				let [num_0, num_1, den_0, den_1] = children;
				vec![
					num_0 * bound_eq,
					num_1 * bound_eq,
					select(bound_eq, den_0),
					select(bound_eq, den_1),
				]
			}
			Phase::Real(_) => panic!("finish requires every variable to be bound"),
		}
	}

	fn eval_point(&self) -> &[F] {
		&self.eval_point[..self.n_vars()]
	}
}

// The prover is checked against the padded layer it stands in for: the same reduction run by an
// ordinary fractional-addition MLE-check over an explicitly materialized padded layer must produce
// the same round polynomials and the same child evaluations.
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

	/// Materializes the `pad_len`-fold padding of a layer buffer, whose variables are
	/// `[real | split]`, filling the extra positions with `fill`.
	///
	/// The padding variables land below the real ones, matching the claim-point layout [`new`]
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
		let (num_0, num_1) = num.split_half_ref();
		let (den_0, den_1) = den.split_half_ref();
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

	/// Runs the padded prover and the reference prover over the materialized padded layer in
	/// lockstep.
	fn assert_matches_padded_reference(num: FieldBuffer<P>, den: FieldBuffer<P>, pad_len: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// The zero fraction 0/1 fills the padding positions.
		let padded_num = pad_layer(&num, pad_len, F::ZERO);
		let padded_den = pad_layer(&den, pad_len, F::ONE);
		let n_vars = padded_num.log_len() - 1;
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let claims = split_half_claims(&padded_num, &padded_den, &eval_point);

		let (mut reference, _cols) = frac_add_mle::new_split_half(
			&alloc,
			padded_num,
			padded_den,
			eval_point.clone(),
			claims,
		);
		let mut prover = new(&alloc, num, den, pad_len, eval_point, claims);

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
		for n_real_rounds in [0, 1, 3] {
			for pad_len in [0, 1, 3] {
				let num = random_field_buffer::<P>(&mut rng, n_real_rounds + 1);
				let den = random_field_buffer::<P>(&mut rng, n_real_rounds + 1);
				assert_matches_padded_reference(num, den, pad_len);
			}
		}
	}

	// The layers a shallow tree spends while the batch is still above it are paddings of its
	// fractional sum, whose high child is the zero fraction 0/1. That degenerate shape is what
	// `batch_prove_unequal_depths` feeds in for those layers.
	#[test]
	fn matches_padded_reference_with_zero_fraction_child() {
		let mut rng = StdRng::seed_from_u64(2);
		for pad_len in [1, 2, 4] {
			let root_num = F::random(&mut rng);
			let root_den = F::random(&mut rng);
			let num = FieldBuffer::<P>::from_values(&[root_num, F::ZERO]);
			let den = FieldBuffer::<P>::from_values(&[root_den, F::ONE]);
			assert_matches_padded_reference(num, den, pad_len);
		}
	}
}
