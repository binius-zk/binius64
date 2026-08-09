// Copyright 2023-2025 Irreducible Inc.

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_math::FieldVec;

use super::{
	mle_store::{ColId, MleStore},
	quadratic_mle_evaluator::{QuadraticMleEvaluator, quadratic_mlecheck_prover},
	round_evaluator::{MleCheckRoundEvaluator, SharedMleCheckProver},
};
use crate::sumcheck::common::MleCheckProver;

/// The prover [`new_split_half`] and [`new_one_padded`] return.
///
/// The evaluator is boxed so that the two constructors — whose compositions differ — produce one
/// type, which is what lets a caller switch from one to the other partway through a reduction.
pub type LayerProver<'a, A, F, P> =
	SharedMleCheckProver<'a, A, F, P, Box<dyn MleCheckRoundEvaluator<F, P> + 'a>>;

/// Creates an [`MleCheckProver`] that reduces an evaluation claim on a multilinear extension
/// of the product of two multilinears to evaluation claims on said multilinears.
///
/// ## Mathematical Definition
/// * $n \in N$ - number of variables in multilinear polynomials
/// * $A, B \in F\[x\], x = \(x_1, \ldots, x_n\)$ - multilinears being multiplied
/// * $(\widetilde{AB})\[x\] = y$ - evaluation claim on the product MLE
///
/// The claim is equivalent to $P(x) = \sum_{v \in B} \widetilde{eq}(v, x) A(v) B(v) = y$, and the
/// reduction can be achieved by sumchecking the latter degree-3 composition. The paper [Gruen24],
/// however, describes a way to partition the $\widetilde{eq}(v, x)$ into three parts in round $j
/// \in 1, \ldots, n$ during specialization of variable $v_{n-j+1}$, with $j-1$ challenges
/// $\alpha_i$ already sampled:
///
/// $$ \widetilde{eq}(x_{n-j+2}, \ldots, x_n; \alpha_{j-1}, \ldots, \alpha_{1}) \tag{1} $$
/// $$ \widetilde{eq}(x_{n-j+1}; v_{n-j+1}) \tag{2} $$
/// $$ \widetilde{eq}(x_1, \ldots, x_{n-j}; v_1, \ldots, v_{n-j}) \tag{3} $$
///
/// The following holds:
/// * (1) is a constant that can be incrementally updated in O(1) time,
/// * (2) is a linear polynomial that is easy to compute in monomial form specialized to either
///   variable
/// * (3) is a an equality indicator over the claim point suffix
///
/// These observations allow us to instead sumcheck:
/// $$
/// P'(x) = \sum_{v \in B} \widetilde{eq}(x_1, \ldots, x_{n-j}; v_1, \ldots, v_{n-j}) A(v) B(v)
/// $$
///
/// Which is simpler because:
/// * $P'(x)$ is degree-2 in $j$-th variable, requiring one less evaluation point
/// * Equality indicator expansion does not depend on $j$-th variable and thus doesn't need to be
///   interpolated
///
/// After computing the round polynomial for $P'(x)$ in monomial form, one can simply multiply by
/// (2) and (1) in polynomial form. For more details, see the
/// [equality trackers](crate::sumcheck::eq_tracker) and [Gruen24] Section 3.2.
///
/// Note 1: as evident from the definition, this prover binds variables in high-to-low index order.
///
/// Note 2: evaluation points are 0 (implicit), 1 and Karatsuba infinity.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub fn new<'alloc, A, F, P>(
	alloc: &'alloc A,
	multilinears: [FieldVec<P, A>; 2],
	eval_point: Vec<F>,
	eval_claim: F,
) -> impl MleCheckProver<F> + 'alloc
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	// The product is symmetric, so the infinity composition (highest-degree terms) equals the full
	// composition.
	quadratic_mlecheck_prover(
		alloc,
		multilinears,
		|[a, b]| a * b,
		|[a, b]| a * b,
		eval_point,
		eval_claim,
	)
}

/// Reduces the product of the two halves of a single buffer, sharing one allocation.
///
/// `buffer` has one more variable than `eval_point`:
/// - its low half fixes the highest variable to 0,
/// - its high half fixes it to 1.
///
/// The reduction proves the product of those two halves.
/// Both halves live inside the one buffer, so separating them costs no copy.
/// This is the zero-copy path the product-check layer reduction uses on its large witness layers.
///
/// # Returns
///
/// A prover whose reduction emits the low half's evaluation, then the high half's, and the store
/// column ids `[low, high]` of those two halves.
pub fn new_split_half<'alloc, A, F, P>(
	alloc: &'alloc A,
	buffer: FieldVec<P, A>,
	eval_point: Vec<F>,
	eval_claim: F,
) -> (LayerProver<'alloc, A, F, P>, [ColId; 2])
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	let mut store = MleStore::new(eval_point.len(), alloc);
	// The store checks that the buffer has exactly one more variable than itself.
	let cols = store.push_split_half(buffer);
	let evaluator: Box<dyn MleCheckRoundEvaluator<F, P> + 'alloc> =
		Box::new(QuadraticMleEvaluator::new(cols, |[a, b]: [P; 2]| a * b, |[a, b]: [P; 2]| a * b));
	(SharedMleCheckProver::new(store, [(eval_claim, evaluator)], eval_point), cols)
}

/// Creates the [`LayerProver`] for the product of the *one-paddings* of two multilinears.
///
/// The one-padding selector $\textsf{sel}(s, v) = 1 + (v - 1) s$ interpolates between the constant
/// one at $s = 0$ and $v$ at $s = 1$. This proves a claim on
///
/// $$
/// \textsf{sel}(s, A) \cdot \textsf{sel}(s, B)
/// $$
///
/// for the fixed selector value `pad_eq`, which is the padding coordinates' equality weight in a
/// batched product check over trees of unequal depths — see [`crate::prodcheck::batch_prove`]. The
/// selector rides in the composition rather than in the columns, so the reduction still emits the
/// evaluations of $A$ and $B$ themselves.
///
/// Passing `pad_eq` of one recovers [`new`].
///
/// # Returns
///
/// A prover whose reduction emits the two multilinears' own evaluations, in the order given.
pub fn new_one_padded<'alloc, A, F, P>(
	alloc: &'alloc A,
	multilinears: [FieldVec<P, A>; 2],
	pad_eq: F,
	eval_point: Vec<F>,
	eval_claim: F,
) -> LayerProver<'alloc, A, F, P>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	// sel(pad_eq, v) = v * pad_eq + (1 - pad_eq), an affine map applied to each multilinear.
	let scale = P::broadcast(pad_eq);
	let shift = P::broadcast(F::ONE - pad_eq);
	let select = move |v: P| v * scale + shift;

	let mut store = MleStore::new(eval_point.len(), alloc);
	let cols = multilinears.map(|col| store.push_owned(col));
	let evaluator: Box<dyn MleCheckRoundEvaluator<F, P> + 'alloc> =
		Box::new(QuadraticMleEvaluator::new(
			cols,
			move |[a, b]: [P; 2]| select(a) * select(b),
			// The selector is affine, so the composition's quadratic term is the scaled product.
			move |[a, b]: [P; 2]| (a * scale) * (b * scale),
		));
	SharedMleCheckProver::new(store, [(eval_claim, evaluator)], eval_point)
}

#[cfg(test)]
mod tests {
	use binius_field::arch::{OptimalB128, OptimalPackedB128};
	use binius_ip::{mlecheck, sumcheck::verify};
	use binius_math::{
		FieldBuffer,
		multilinear::{eq::eq_ind, evaluate::evaluate},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use binius_compute::GlobalAllocator;
	use itertools::{self, Itertools};
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::{MleToSumCheckDecorator, prove::prove_single, prove_single_mlecheck};

	fn test_mlecheck_prove_verify<F, P>(
		prover: impl MleCheckProver<F>,
		eval_claim: F,
		eval_point: &[F],
		multilinear_a: &FieldBuffer<P>,
		multilinear_b: &FieldBuffer<P>,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut prover_transcript);

		// Write the multilinear evaluations to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Convert to verifier transcript and run verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = mlecheck::verify(
			eval_point,
			2, // degree 2 for bivariate product
			eval_claim,
			&mut verifier_transcript,
		)
		.unwrap();

		let mut reduced_eval_point = sumcheck_output.challenges.clone();
		reduced_eval_point.reverse();

		// Read the multilinear evaluations from the transcript
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();

		// Check that the product of the evaluations equals the reduced evaluation
		assert_eq!(
			multilinear_evals[0] * multilinear_evals[1],
			sumcheck_output.eval,
			"Product of multilinear evaluations should equal the reduced evaluation"
		);

		// Check that the original multilinears evaluate to the claimed values at the challenge
		// point The prover binds variables from high to low, but evaluate expects them from low
		// to high
		let eval_a = evaluate(multilinear_a, &reduced_eval_point);
		let eval_b = evaluate(multilinear_b, &reduced_eval_point);

		assert_eq!(
			eval_a, multilinear_evals[0],
			"Multilinear A should evaluate to the first claimed evaluation"
		);
		assert_eq!(
			eval_b, multilinear_evals[1],
			"Multilinear B should evaluate to the second claimed evaluation"
		);

		// Also verify the challenges match what the prover saw
		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"Prover and verifier challenges should match"
		);
	}

	fn test_wrapped_sumcheck_prove_verify<F, P>(
		mlecheck_prover: impl MleCheckProver<F>,
		eval_claim: F,
		eval_point: &[F],
		multilinear_a: &FieldBuffer<P>,
		multilinear_b: &FieldBuffer<P>,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let n_vars = mlecheck_prover.n_vars();
		let prover = MleToSumCheckDecorator::new(mlecheck_prover);

		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(prover, &mut prover_transcript);

		// Write the multilinear evaluations to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Convert to verifier transcript and run verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = verify(
			n_vars,
			3, // degree 3 for trivariate product (bivariate by equality indicator)
			eval_claim,
			&mut verifier_transcript,
		)
		.unwrap();

		// The prover binds variables from high to low, but evaluate expects them from low
		// to high
		let mut reduced_eval_point = sumcheck_output.challenges.clone();
		reduced_eval_point.reverse();

		// Read the multilinear evaluations from the transcript
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();

		// Evaluate the equality indicator
		let eq_ind_eval = eq_ind(eval_point, &reduced_eval_point);

		// Check that the product of the evaluations equals the reduced evaluation
		assert_eq!(
			multilinear_evals[0] * multilinear_evals[1] * eq_ind_eval,
			sumcheck_output.eval,
			"Product of multilinear evaluations should equal the reduced evaluation"
		);

		// Check that the original multilinears evaluate to the claimed values at the challenge
		// point
		let eval_a = evaluate(multilinear_a, &reduced_eval_point);
		let eval_b = evaluate(multilinear_b, &reduced_eval_point);

		assert_eq!(
			eval_a, multilinear_evals[0],
			"Multilinear A should evaluate to the first claimed evaluation"
		);
		assert_eq!(
			eval_b, multilinear_evals[1],
			"Multilinear B should evaluate to the second claimed evaluation"
		);

		// Also verify the challenges match what the prover saw
		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"Prover and verifier challenges should match"
		);
	}

	#[test]
	fn test_bivariate_product_mlecheck() {
		type F = OptimalB128;
		type P = OptimalPackedB128;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// Generate two random multilinear polynomials
		let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
		let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);

		// Compute product multilinear
		let product = itertools::zip_eq(multilinear_a.as_ref(), multilinear_b.as_ref())
			.map(|(&l, &r)| l * r)
			.collect_vec();
		let product_buffer = FieldBuffer::new(n_vars, product);

		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&product_buffer, &eval_point);

		// Create the prover
		let mlecheck_prover = new(
			&alloc,
			[multilinear_a.clone(), multilinear_b.clone()],
			eval_point.clone(),
			eval_claim,
		);

		test_mlecheck_prove_verify(
			mlecheck_prover,
			eval_claim,
			&eval_point,
			&multilinear_a,
			&multilinear_b,
		);

		// Create another prover for the wrapped test
		let mlecheck_prover = new(
			&alloc,
			[multilinear_a.clone(), multilinear_b.clone()],
			eval_point.clone(),
			eval_claim,
		);

		test_wrapped_sumcheck_prove_verify(
			mlecheck_prover,
			eval_claim,
			&eval_point,
			&multilinear_a,
			&multilinear_b,
		);
	}
}
