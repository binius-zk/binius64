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

use std::mem;

use binius_field::Field;
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::multilinear::eq::eq_one_var;

use crate::sumcheck::common::MleCheckProver;

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
	[num * pad_eq_inv, select(pad_eq_inv, den)]
}

/// Creates the prover for one padded fractional-addition layer.
///
/// # Arguments
///
/// * `pad_eq_prefixes` - Equality weights of the claim point's padding segment: entry `i` is
///   $\prod_{c < i} \textsf{eq}(0, \rho_{\text{pa}, c})$. Its length fixes the padding segment at
///   one less, so a single-entry table leaves the inner reduction uncorrected. Every layer of a
///   batch shares one point, so one table of prefix products serves them all.
/// * `eval_point` - The padded layer's claim point, `[padding | real]`.
/// * `inner` - The unpadded layer's MLE-check, seeded at the real segment of the claim point with
///   the claims [`unpad_claims`] returns.
///
/// # Preconditions
///
/// * `pad_eq_prefixes` is non-empty and its last entry — the padding segment's equality weight — is
///   non-zero
/// * `eval_point.len() + 1 >= pad_eq_prefixes.len()`
/// * `inner.n_vars() == eval_point.len() + 1 - pad_eq_prefixes.len()`
pub fn new<F, Inner>(
	pad_eq_prefixes: Vec<F>,
	eval_point: Vec<F>,
	inner: Inner,
) -> ZeroPadMleCheckProver<F, Inner>
where
	F: Field,
	Inner: MleCheckProver<F>,
{
	let pad_len = pad_eq_prefixes
		.len()
		.checked_sub(1)
		.expect("precondition: non-empty");
	assert!(eval_point.len() >= pad_len); // precondition
	assert_ne!(pad_eq_prefixes[pad_len], F::ZERO); // precondition
	assert_eq!(inner.n_vars(), eval_point.len() - pad_len); // precondition

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

/// The layer a tree contributes while the batch is still above it: one fraction beside the zero
/// fraction $0/1$.
///
/// Such a layer is a padding of the tree's own fractional sum, so its low child is that sum and its
/// high child is identically $0/1$. Every one of its variables is a padding variable, so there is
/// nothing to reduce — [`ZeroPadMleCheckProver`] goes straight to its padding rounds and only ever
/// asks for these four child evaluations.
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
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_field::{Random, arithmetic_traits::InvertOrZero, field::FieldOps};
	use binius_math::{
		FieldBuffer,
		multilinear::evaluate::evaluate,
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
		let (mut reference, _cols) =
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

		for n_real_rounds in [0, 1, 3] {
			for pad_len in [0, 1, 3] {
				let num = random_field_buffer::<P>(&mut rng, n_real_rounds + 1);
				let den = random_field_buffer::<P>(&mut rng, n_real_rounds + 1);
				let (padded_num, padded_den, eval_point, claims) =
					padded_layer(&mut rng, &num, &den, pad_len);

				let prefixes = pad_eq_prefixes(&eval_point, pad_len);
				let pad_eq_inv = prefixes[pad_len].invert_or_zero();
				let (inner, _cols) = frac_add_mle::new_split_half(
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

		for pad_len in [1, 2, 4] {
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
