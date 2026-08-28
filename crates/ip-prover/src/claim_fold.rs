// Copyright 2026 The Binius Developers

//! Proving a fold of evaluation claims on one sparse tensor.

use binius_field::{Field, PackedField};
use binius_ip::MultilinearEvalClaim;
use binius_math::multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars};

use crate::{
	channel::IPProverChannel,
	sumcheck::{
		batch::batch_prove, factored_multilinear::FactoredMultilinear,
		sparse_dense_product::SparseMultiDenseProductSumcheckProver,
	},
};

/// An evaluation claim whose point is cut into one run per tensor axis.
///
/// The verifier reduces claims over a flat point, since the axes mean nothing to it.
/// A prover needs the cut, because it builds one weight factor per axis.
///
/// Axes run lowest first: the first run owns the lowest index bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisClaim<F> {
	/// The point, one coordinate run per axis, lowest axis first.
	pub point: Vec<Vec<F>>,
	/// The value the extension is claimed to take there.
	pub value: F,
}

impl<F: Clone> AxisClaim<F> {
	/// The width of each axis, in variables, lowest axis first.
	pub fn axes(&self) -> Vec<usize> {
		self.point.iter().map(Vec::len).collect()
	}

	/// The same claim over a flat point, as the verifier reduces it.
	pub fn flatten(&self) -> MultilinearEvalClaim<F> {
		MultilinearEvalClaim {
			eval: self.value.clone(),
			point: self.point.concat(),
		}
	}
}

/// Cuts a flat point into one run per axis, lowest axis first.
///
/// This is how a reduced claim regains the structure its inputs had, so a caller may reduce again.
pub fn split_axes<F: Clone>(point: &[F], axes: &[usize]) -> Vec<Vec<F>> {
	let mut runs = Vec::with_capacity(axes.len());
	let mut rest = point;
	for &width in axes {
		let (run, tail) = rest.split_at(width);
		runs.push(run.to_vec());
		rest = tail;
	}
	runs
}

/// One nonzero of a sparse tensor: a flat index over every axis, and the value carried there.
///
/// Axes run lowest first, so the first axis owns the lowest index bits.
/// Entries at a repeated index add, so a caller need not deduplicate them.
pub type TensorEntry<F> = (usize, F);

/// Proves a fold of evaluation claims on one tensor, returning the claim they folded to.
///
/// The claims all name the same tensor at different points.
/// One sumcheck reduces them to one point.
///
/// What comes out has the same shape as what went in, so a caller can fold again.
///
/// # Cost
///
/// Per round, one pass over the entries per claim.
///
/// Nothing is materialized over the product of the axes.
/// So the axes may span a space far larger than the entry list.
///
/// # Panics
///
/// Panics if no claim is given, or if the claims disagree about the axis widths.
pub fn prove<F, P>(
	entries: &[TensorEntry<F>],
	claims: &[AxisClaim<F>],
	channel: &mut impl IPProverChannel<F>,
) -> AxisClaim<F>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	let first = claims
		.first()
		.expect("precondition: a fold needs at least one claim");
	let axes = first.axes();
	assert!(
		claims.iter().all(|claim| claim.axes() == axes),
		"precondition: every claim must span the same axes"
	);

	// One weight per claim, all riding a single copy of the tensor's entries.
	//
	// The weight is an equality indicator over every axis at once, which factorizes across them.
	// Holding it as one factor per axis is what keeps it off the product of their lengths.
	let weights = claims
		.iter()
		.map(|claim| {
			FactoredMultilinear::new(claim.point.iter().map(|run| eq_ind_partial_eval::<P>(run)))
		})
		.collect::<Vec<_>>();
	let sums = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();

	// One prover over all of them, so the entry list is stored once and folded once.
	// A prover per claim would hold a copy each, and fold every copy every round.
	let prover = SparseMultiDenseProductSumcheckProver::new(entries.to_vec(), weights, &sums);
	let output = batch_prove(vec![prover], channel);

	// The evaluations lead with the tensor's, shared by every claim, then one weight's per claim.
	let tensor_eval = output.multilinear_evals[0][0];

	// The verifier derives its own weight evaluations, so the tensor's is the one thing it needs.
	channel.send_one(tensor_eval);

	// The rounds bind the highest variable first, so the point reads back in reverse.
	let mut point = output.challenges;
	point.reverse();

	// Cut the point into one run per axis, matching the shape the claims came in with.
	let mut runs = Vec::with_capacity(axes.len());
	let mut rest = point.as_slice();
	for width in axes {
		let (run, tail) = rest.split_at(width);
		runs.push(run.to_vec());
		rest = tail;
	}

	AxisClaim {
		point: runs,
		value: tensor_eval,
	}
}

/// The tensor's multilinear extension at one point, evaluated directly from its entries.
///
/// This is the reference a folded claim is settled against.
/// It reads the entries and the point, and nothing from the run that raised the claim.
pub fn evaluate<F: Field>(entries: &[TensorEntry<F>], point: &[Vec<F>]) -> F {
	let flat = point.concat();
	let indicator = eq_ind_partial_eval_scalars(&flat);
	entries
		.iter()
		.map(|&(index, value)| value * indicator[index])
		.sum()
}

/// Builds a claim asserting the tensor's extension takes the value it actually takes.
///
/// The value comes from the entries, so the claim is true by construction.
pub fn claim_at<F: Field>(entries: &[TensorEntry<F>], point: Vec<Vec<F>>) -> AxisClaim<F> {
	let value = evaluate(entries, &point);
	AxisClaim { point, value }
}

#[cfg(test)]
mod tests {
	use binius_field::{
		Random,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_ip::batch_eval;
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::{SeedableRng, prelude::*};

	use super::*;

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	/// Three axes over 2, 1 and 2 variables, so five in total.
	const AXES: [usize; 3] = [2, 1, 2];

	fn random_entries(rng: &mut StdRng, n_entries: usize) -> Vec<TensorEntry<F>> {
		let n_vars: usize = AXES.iter().sum();
		(0..n_entries)
			.map(|_| (rng.random_range(0..1usize << n_vars), F::random(&mut *rng)))
			.collect()
	}

	fn random_point(rng: &mut StdRng) -> Vec<Vec<F>> {
		AXES.iter()
			.map(|&width| (0..width).map(|_| F::random(&mut *rng)).collect())
			.collect()
	}

	/// Folds the claims and returns what the verifier made of it.
	///
	/// The verifier reduces over flat points.
	/// So the only fold-specific work here is cutting the reduced point back into axis runs.
	fn fold(
		entries: &[TensorEntry<F>],
		claims: &[AxisClaim<F>],
	) -> Result<AxisClaim<F>, binius_ip::sumcheck::Error> {
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		prove::<F, P>(entries, claims, &mut transcript);

		let mut verifier = transcript.into_verifier();
		let reduced =
			batch_eval::verify::<F, _>(claims.iter().map(AxisClaim::flatten), &mut verifier)?;
		verifier
			.finalize()
			.expect("the tape must be fully consumed");

		Ok(AxisClaim {
			point: split_axes(&reduced.point, &claims[0].axes()),
			value: reduced.eval,
		})
	}

	#[test]
	fn folding_true_claims_yields_a_true_claim() {
		// Invariant: a fold preserves truth, and preserves shape.
		//
		// Fixture state: three true claims about one tensor, at three independent points.
		//
		//     claim_1 at p_1  -.
		//     claim_2 at p_2   >-- fold -->  one claim at one new point
		//     claim_3 at p_3  -'
		//
		// The claim that comes out must hold against the tensor read directly.
		// It must also span the same axes, or it could not be folded again at the next level.
		let mut rng = StdRng::seed_from_u64(1);
		let entries = random_entries(&mut rng, 20);
		let claims = (0..3)
			.map(|_| claim_at(&entries, random_point(&mut rng)))
			.collect::<Vec<_>>();

		// A vacuous fixture would let a broken fold pass, so the claims must say something.
		assert!(claims.iter().any(|claim| claim.value != F::ZERO));

		let folded = fold(&entries, &claims).expect("a fold of true claims must verify");

		assert_eq!(folded.axes(), AXES.to_vec(), "the shape must survive the fold");
		assert_eq!(
			folded.value,
			evaluate(&entries, &folded.point),
			"the folded claim must hold against the tensor"
		);
	}

	#[test]
	fn one_false_claim_makes_the_folded_claim_false() {
		// Invariant: this is the accumulation property, and the whole reason a fold is sound.
		//
		// A fold does not check its inputs.
		// It produces a claim that is false whenever any input was.
		//
		// So settling the *output* once, at the root, catches an error anywhere below.
		//
		// Fixture state: three true claims, then each one corrupted in turn.
		//
		//     before:  claim_i holds
		//     after:   claim_i's value moved by one, everything else identical
		//
		// Either the fold's own assertion rejects, or it returns a claim that does not hold.
		// Both are correct.
		//
		// What must never happen is a true output from a false input.
		let mut rng = StdRng::seed_from_u64(2);
		let entries = random_entries(&mut rng, 20);
		let honest = (0..3)
			.map(|_| claim_at(&entries, random_point(&mut rng)))
			.collect::<Vec<_>>();

		for index in 0..honest.len() {
			let mut claims = honest.clone();
			claims[index].value += F::ONE;

			match fold(&entries, &claims) {
				// The assertion caught it inside the fold.
				Err(_) => {}
				// Or it came through, and the claim it produced must be false.
				Ok(folded) => assert_ne!(
					folded.value,
					evaluate(&entries, &folded.point),
					"corrupting claim {index} must not fold to a true claim"
				),
			}
		}
	}

	#[test]
	fn a_single_claim_folds_to_a_claim_about_the_same_tensor() {
		// Invariant: one claim is a valid fold, and the reduction still moves the point.
		//
		// A tree's leaf level may hand over a single claim.
		// So the degenerate arity has to work, rather than being a case a caller must avoid.
		let mut rng = StdRng::seed_from_u64(3);
		let entries = random_entries(&mut rng, 12);
		let claim = claim_at(&entries, random_point(&mut rng));

		let folded = fold(&entries, std::slice::from_ref(&claim)).expect("one claim must fold");

		assert_eq!(folded.value, evaluate(&entries, &folded.point));
		assert_ne!(folded.point, claim.point, "the fold lands on a fresh point");
	}
}
