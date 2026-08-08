// Copyright 2026 The Binius Developers

//! MLE-check prover for one layer of a one-padded product check.
//!
//! One-padding lifts a product tree of depth $k$ to depth $n \ge k$ by filling the extra leaves
//! with ones, which leaves the tree's product unchanged. Batching product checks of unequal depths
//! pads each shallow tree up to the deepest one, so the batch's layer loop runs a single uniform
//! schedule and the verifier never learns the individual depths. See the *Batched Product Checks of
//! Unequal Depths* appendix of the Binius64 whitepaper for the protocol and the derivation behind
//! the round polynomials below.
//!
//! The point of this module is that the prover never materializes a padded layer:
//! [`OnePadMleCheckProver`] wraps the unpadded layer's own MLE-check and corrects its messages at a
//! cost of $O(1)$ per round.

use std::{iter, mem};

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldVec, multilinear::eq::eq_one_var};

use crate::sumcheck::{
	bivariate_product_mle,
	common::{MleCheckProver, SumcheckProver},
};

/// The one-padding selector $\textsf{sel}(s, v) = 1 + (v - 1) s$.
///
/// It interpolates between the constant one at $s = 0$ and $v$ at $s = 1$, which is how a padded
/// leaf position holds a one while a real one holds the witness value.
fn select<F: Field>(s: F, v: F) -> F {
	F::ONE + (v - F::ONE) * s
}

/// MLE-check prover for one layer of a product check over a one-padded witness.
///
/// The tree's product is a scalar, so the layer's claim point is node coordinates only, split into
/// two segments with the padding ones lowest:
///
/// ```text
///     [ padding (nu) | real (m) ]
/// ```
///
/// The padded layer is the unpadded one wrapped in the one-padding
/// $\textsf{sel}(\textsf{eq}(0^\nu, Z'), \cdot)$ over the padding variables. MLE-check binds
/// variables from the highest index down, so the real rounds come first and the padding rounds
/// last:
///
/// - **Real rounds.** Delegate to `inner`, the ordinary MLE-check over the unpadded layer, and
///   correct each round polynomial by the affine map $R'(X) = 1 + q \cdot (R(X) - 1)$, where $q$ is
///   the equality weight $\textsf{eq}(0^\nu, \rho_\text{pa})$ of the claim point's padding segment.
///   Off the all-zeros padding slab both children are one, which is where the residual weight $1 -
///   q$ comes from.
/// - **Padding rounds.** No multilinear is touched. Every real variable is bound by now, so
///   `inner`'s two child evaluations $g_0, g_1$ are scalars and the round polynomial is the closed
///   form $R'(X) = (1 - e) + e \cdot \textsf{sel}(E(X), g_0) \cdot \textsf{sel}(E(X), g_1)$, where
///   $e$ is the equality weight of the padding coordinates still unbound and $E(X)$ that of the
///   ones already bound.
///
/// [`SumcheckProver::finish`] returns the *padded* layer's child evaluations
/// $\textsf{sel}(e^*, g_b)$, which is what the batch's selector rounds consume.
pub struct OnePadMleCheckProver<F: Field, Inner> {
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

/// The segment of rounds the prover is in. See [`OnePadMleCheckProver`].
enum Phase<F, Inner> {
	/// Reducing the unpadded layer's real node variables.
	Real(Inner),
	/// Every real variable is bound, leaving a closed form in these scalars.
	Padding {
		/// The unpadded layer's two child evaluations.
		children: [F; 2],
		/// $\prod \textsf{eq}(0, r)$ over the padding challenges bound so far, which is the
		/// constant factor of $E$.
		bound_eq: F,
	},
}

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

	// Prefix products over the padding segment, so both the round polynomials' `e` factors and the
	// claim's `q` are lookups.
	let pad_eq_prefixes = iter::once(F::ONE)
		.chain(eval_point[..pad_len].iter().scan(F::ONE, |acc, &coord| {
			*acc *= eq_one_var(F::ZERO, coord);
			Some(*acc)
		}))
		.collect::<Vec<_>>();
	let pad_eq = pad_eq_prefixes[pad_len];
	assert!(pad_eq != F::ZERO, "a padding coordinate of the claim point equals one");

	// The padded claim is `sel(q, z)` for the unpadded claim `z`, so the inner prover starts from
	// the preimage.
	let inner_claim = F::ONE + (claim - F::ONE) * pad_eq.invert_or_zero();
	let inner = bivariate_product_mle::new_split_half(
		alloc,
		layer,
		eval_point[pad_len..].to_vec(),
		inner_claim,
	);

	let mut prover = OnePadMleCheckProver {
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

impl<F: Field, Inner: MleCheckProver<F>> OnePadMleCheckProver<F, Inner> {
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
			children: [F::ONE; 2],
			bound_eq: F::ONE,
		};
		let Phase::Real(inner) = mem::replace(&mut self.phase, placeholder) else {
			unreachable!("the guard checked the phase");
		};
		self.phase = Phase::Padding {
			children: inner
				.finish()
				.try_into()
				.expect("the layer prover reduces two multilinears"),
			bound_eq: F::ONE,
		};
	}
}

impl<F: Field, Inner: MleCheckProver<F>> SumcheckProver<F> for OnePadMleCheckProver<F, Inner> {
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

		let coeffs = match phase {
			Phase::Real(inner) => {
				let mut round_coeffs = inner.execute();
				assert_eq!(round_coeffs.len(), 1, "the layer prover carries one claim");
				let mut coeffs = round_coeffs.pop().expect("the vector holds one element");
				// R'(X) = 1 + q * (R(X) - 1): on the all-zeros padding slab the sum is the unpadded
				// one, weighted by q; off it both children are one and the weights sum to 1 - q.
				let pad_eq = pad_eq_prefixes[*pad_len];
				coeffs *= pad_eq;
				coeffs.0[0] += F::ONE - pad_eq;
				coeffs
			}
			Phase::Padding { children, bound_eq } => {
				// The equality weight of the padding coordinates still unbound below this round's.
				let unbound_eq = pad_eq_prefixes[n_vars - 1];
				// E(X) = bound_eq * eq(0, X) in monomial coefficients, then each child pushed
				// through the one-padding selector at E(X).
				let big_e = [*bound_eq, -*bound_eq];
				let [[a_0, a_1], [b_0, b_1]] =
					children.map(|child| [select(big_e[0], child), (child - F::ONE) * big_e[1]]);
				// R'(X) = (1 - e) + e * sel(E(X), g_0) * sel(E(X), g_1): off the all-zeros slab of
				// those unbound coordinates both children are one.
				RoundCoeffs(vec![
					F::ONE - unbound_eq + unbound_eq * a_0 * b_0,
					unbound_eq * (a_0 * b_1 + a_1 * b_0),
					unbound_eq * a_1 * b_1,
				])
			}
		};
		vec![coeffs]
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
				children.map(|child| select(bound_eq, child)).to_vec()
			}
			Phase::Real(_) => panic!("finish requires every variable to be bound"),
		}
	}
}

impl<F: Field, Inner: MleCheckProver<F>> MleCheckProver<F> for OnePadMleCheckProver<F, Inner> {
	fn eval_point(&self) -> &[F] {
		&self.eval_point[..self.n_vars()]
	}
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
	/// The padding variables land below the real ones, matching the claim-point layout [`new`]
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
		let (low, high) = buffer.split_half_ref();
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
		for n_real_rounds in [0, 1, 3] {
			for pad_len in [0, 1, 3] {
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
		for pad_len in [1, 2, 4] {
			let product = F::random(&mut rng);
			let layer = FieldBuffer::<P>::from_values(&[product, F::ONE]);
			assert_matches_padded_reference(layer, pad_len);
		}
	}
}
