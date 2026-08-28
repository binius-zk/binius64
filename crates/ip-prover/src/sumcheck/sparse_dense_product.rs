// Copyright 2026 The Binius Developers

//! Sumcheck prover for the product of a sparse and a dense multilinear.

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};

use super::{
	common::SumcheckProver, factored_multilinear::FactoredMultilinear, round_evals::RoundEvals,
	round_state::RoundState,
};

/// One entry of a sparse multilinear: a hypercube index and the value carried there.
///
/// The multilinear is the sum of its entries, so several entries may share an index.
pub type SparseEntry<F> = (usize, F);

/// Proves the hypercube sum of the product of a sparse and a dense multilinear.
///
/// The sparse multilinear $A$ is given as a list of (index, value) entries, defined as
///
/// $$
/// A(v) = \sum_{(i, c) \in \text{entries}, i = v} c
/// $$
///
/// so entries at a repeated index add up and need not be deduplicated. The dense multilinear $B$
/// is a full buffer over the same $n$ variables. The prover argues the claim
///
/// $$
/// s = \sum_{v \in B_n} A(v) B(v)
/// $$
///
/// which is the plain, non-eq-weighted sumcheck of a degree-2 composition. Its rounds cost one
/// pass over the entry list plus two dense lookups per entry, so the work per round is set by the
/// number of entries rather than by the size of the hypercube.
///
/// Variables bind from the highest index down, as in the dense provers of this module. Round $j$
/// therefore splits the index space at `half`, the bit the round binds: an entry below it lies in
/// the half where the bound variable is 0, one at or above it in the half where it is 1.
///
/// ## Round polynomial
///
/// With $A_0, A_1$ the two halves of $A$ on the bound variable (likewise $B$), the round
/// polynomial is sampled at 1 and at infinity, and its value at 0 recovered from the round claim:
///
/// $$
/// R(1) = \sum_v A_1(v) B_1(v) \qquad R(\infty) = \sum_v (A_0 + A_1)(v) (B_0 + B_1)(v)
/// $$
///
/// Both are linear in $A$, so each entry contributes to them on its own: pairing an entry with
/// the one facing it across the split is never needed. An entry $(i, c)$ with $v = i \bmod
/// \text{half}$ adds $c B(i)$ to $R(1)$ when it lies in the upper half, and $c (B_0 + B_1)(v)$ to
/// $R(\infty)$ wherever it lies.
///
/// ## Folding
///
/// Folding is linear in $A$ for the same reason: an entry keeps its identity across the fold,
/// scaled by the challenge weight of the half it sits in and moved down into the lower half.
/// Entries thus stay a flat list of the same length for the whole protocol, and collapse to the
/// evaluation of $A$ at the challenge point only in [`SumcheckProver::finish`], which emits the
/// sparse multilinear's evaluation before the dense one's.
pub struct SparseDenseProductSumcheckProver<P: PackedField> {
	/// The sparse multilinear, folded in place.
	///
	/// Indices are always below `1 << self.n_vars()`.
	sparse: Vec<SparseEntry<P::Scalar>>,
	/// The dense multilinear, folded in place. Its length tracks the free variables.
	///
	/// Held as a product of factors rather than one table.
	/// So a weight whose table is too large to materialize can drive this prover too.
	///
	/// A weight that really is one table is one factor, so nothing is given up by taking this.
	dense: FactoredMultilinear<P>,
	/// This round's sum claim, or the round polynomial awaiting the challenge that reduces it.
	state: RoundState<RoundCoeffs<P::Scalar>, P::Scalar>,
}

impl<P: PackedField> SparseDenseProductSumcheckProver<P> {
	/// Creates a prover for the claim that the sparse-dense product sums to `sum`.
	///
	/// # Arguments
	///
	/// * `sparse` - the entries of the sparse multilinear, in any order, indices repeatable
	/// * `dense` - the dense multilinear, whose length fixes the number of variables
	/// * `sum` - the claimed sum of the product over the hypercube
	///
	/// # Panics
	///
	/// Panics if any entry index is out of range for `dense`.
	pub fn new(
		sparse: Vec<SparseEntry<P::Scalar>>,
		dense: FactoredMultilinear<P>,
		sum: P::Scalar,
	) -> Self {
		assert!(
			sparse.iter().all(|&(index, _)| index < 1 << dense.n_vars()),
			"precondition: every sparse index must be within the dense multilinear"
		);

		Self {
			sparse,
			dense,
			state: RoundState::Claim(sum),
		}
	}

	/// The index-space bit this round binds, separating the two halves of the hypercube.
	///
	/// # Panics
	///
	/// Panics if no variables are left to bind.
	fn half(&self) -> usize {
		let n_vars = self.dense.n_vars();
		assert!(n_vars > 0, "no variables remain to bind");
		1 << (n_vars - 1)
	}
}

impl<F: Field, P: PackedField<Scalar = F>> SumcheckProver<F>
	for SparseDenseProductSumcheckProver<P>
{
	fn n_vars(&self) -> usize {
		self.dense.n_vars()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let claim = *self.state.claim();
		let half = self.half();

		// Each entry contributes on its own, by linearity of both sampled evaluations in the
		// sparse multilinear. `dense.get(index)` is the entry's own half, `index ^ half` faces it
		// across the split.
		let (y_1, y_inf) = self
			.sparse
			.par_iter()
			.with_min_task(WorkPerItem::FieldMuls)
			.map(|&(index, value)| {
				let own = self.dense.get(index);
				let facing = self.dense.get(index ^ half);

				// The infinity evaluation reads B(0) + B(1), which is the same sum from either
				// half, so the entry's own half only decides the evaluation at 1.
				let y_1 = if index & half == 0 {
					F::ZERO
				} else {
					value * own
				};
				(y_1, value * (own + facing))
			})
			.reduce(
				|| (F::ZERO, F::ZERO),
				|(lhs_1, lhs_inf), (rhs_1, rhs_inf)| (lhs_1 + rhs_1, lhs_inf + rhs_inf),
			);

		let coeffs = RoundEvals([y_1, y_inf]).interpolate(claim);
		self.state = RoundState::Coeffs(coeffs.clone());
		vec![coeffs]
	}

	fn fold(&mut self, challenge: F) {
		let claim = self.state.coeffs().evaluate(&challenge);
		let half = self.half();

		// Scale each entry by the challenge weight of the half it sits in, then move it down into
		// the folded index space.
		let lower_weight = F::ONE - challenge;
		self.sparse
			.par_iter_mut()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|(index, value)| {
				if *index & half == 0 {
					*value *= lower_weight;
				} else {
					*value *= challenge;
					*index ^= half;
				}
			});

		self.dense.fold_highest_var(challenge);
		self.state = RoundState::Claim(claim);
	}

	fn finish(self) -> Vec<F> {
		assert_eq!(self.n_vars(), 0, "finish called before the last fold");

		// Every entry has folded down onto the single remaining index, so the sparse evaluation is
		// what is left of their sum.
		let sparse_eval = self.sparse.iter().map(|&(_, value)| value).sum();
		vec![sparse_eval, self.dense.get(0)]
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		Random,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_ip::sumcheck::verify;
	use binius_math::{
		FieldBuffer, multilinear::evaluate::evaluate, test_utils::random_field_buffer,
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use proptest::prelude::*;
	use rand::{SeedableRng, prelude::*};

	use super::*;
	use crate::sumcheck::{factored_multilinear::FactoredMultilinear, prove::prove_single};

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	/// Materializes a sparse multilinear as a dense buffer, adding up repeated indices.
	fn densify(sparse: &[SparseEntry<F>], n_vars: usize) -> FieldBuffer<P> {
		let mut buffer = FieldBuffer::<P>::zeros(n_vars);
		for &(index, value) in sparse {
			buffer.set(index, buffer.get(index) + value);
		}
		buffer
	}

	/// Proves the sparse-dense product sum, verifies it, and checks the reduced claim against the
	/// two multilinears evaluated at the challenge point.
	fn prove_verify(sparse: Vec<SparseEntry<F>>, dense: &FieldBuffer<P>) {
		let n_vars = dense.log_len();
		let sparse_dense = densify(&sparse, n_vars);
		let sum = sparse
			.iter()
			.map(|&(index, value)| value * dense.get(index))
			.sum::<F>();

		// One factor over every variable is exactly the table it holds.
		let weight = FactoredMultilinear::new([dense.clone()]);
		let prover = SparseDenseProductSumcheckProver::new(sparse, weight, sum);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(prover, &mut prover_transcript);
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = verify(n_vars, 2, sum, &mut verifier_transcript).unwrap();
		let multilinear_evals = verifier_transcript.message().read_vec::<F>(2).unwrap();

		assert_eq!(
			multilinear_evals[0] * multilinear_evals[1],
			sumcheck_output.eval,
			"product of the multilinear evaluations should equal the reduced evaluation"
		);

		// The prover binds variables high-to-low; `evaluate` expects them low-to-high.
		let mut eval_point = sumcheck_output.challenges.clone();
		eval_point.reverse();
		assert_eq!(evaluate(&sparse_dense, &eval_point), multilinear_evals[0]);
		assert_eq!(evaluate(dense, &eval_point), multilinear_evals[1]);
		assert_eq!(output.challenges, sumcheck_output.challenges);
	}

	#[test]
	fn a_factored_weight_proves_the_same_sum_as_the_table_it_stands_for() {
		// Invariant: the dense side's storage is invisible to the protocol.
		//
		// A weight held as factors and the table it stands for are the same multilinear.
		// So the same claim must prove, verify, and reduce to the same evaluations.
		//
		// This is the whole point of the generalization.
		// A weight too large to materialize can then drive this prover.
		//
		// Fixture state: three factors over 2, 1 and 2 variables, so five in total.
		//
		//     index bits:  [ f0 : 2 | f1 : 1 | f2 : 2 ]
		//                    low                  high
		let mut rng = StdRng::seed_from_u64(7);

		let factors = [2usize, 1, 2]
			.iter()
			.map(|&log_len| random_field_buffer::<P>(&mut rng, log_len))
			.collect::<Vec<_>>();
		let n_vars = 5;

		// The same weight, twice: once as factors, once as the table they multiply out to.
		let factored = FactoredMultilinear::new(factors);
		let table = FieldBuffer::<P>::from_values(
			&(0..1usize << n_vars)
				.map(|index| factored.get(index))
				.collect::<Vec<_>>(),
		);

		let sparse = random_sparse(&mut rng, n_vars, 12);
		let sum = sparse
			.iter()
			.map(|&(index, value)| value * table.get(index))
			.sum::<F>();
		// A vacuous sum would let a broken weight pass unnoticed.
		assert_ne!(sum, F::ZERO);

		// Prove the same claim twice, once over each storage form.
		let transcripts = [
			{
				let prover = SparseDenseProductSumcheckProver::new(sparse.clone(), factored, sum);
				let mut transcript = ProverTranscript::new(StdChallenger::default());
				let output = prove_single(prover, &mut transcript);
				(output.multilinear_evals, transcript.finalize())
			},
			{
				let prover = SparseDenseProductSumcheckProver::new(
					sparse.clone(),
					FactoredMultilinear::new([table.clone()]),
					sum,
				);
				let mut transcript = ProverTranscript::new(StdChallenger::default());
				let output = prove_single(prover, &mut transcript);
				(output.multilinear_evals, transcript.finalize())
			},
		];

		// Identical transcripts mean identical round polynomials, every round.
		assert_eq!(transcripts[0].1, transcripts[1].1, "the two forms must prove the same rounds");
		assert_eq!(transcripts[0].0, transcripts[1].0, "and reduce to the same evaluations");

		// And the shared proof verifies, so neither form is consistently wrong.
		prove_verify(sparse, &table);
	}

	/// Draws `n_entries` entries at uniformly random indices, so repeats arise on their own.
	fn random_sparse(
		mut rng: impl rand::Rng,
		n_vars: usize,
		n_entries: usize,
	) -> Vec<SparseEntry<F>> {
		(0..n_entries)
			.map(|_| (rng.random_range(0..1 << n_vars), F::random(&mut rng)))
			.collect()
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(32))]

		// Invariant: the sumcheck verifier accepts every round, and the evaluation claims the
		// prover reduces to are those of the two multilinears at the challenge point.
		//
		// Indices are drawn uniformly over a hypercube that may be narrower than one packed
		// element, so repeated indices, unused vertices, entry lists longer than the hypercube and
		// dead packed lanes all arise on their own.
		#[test]
		fn prove_verify_matches_the_materialized_multilinears(
			n_vars in 1usize..=8,
			n_entries in 0usize..=100,
			seed in any::<u64>(),
		) {
			let mut rng = StdRng::seed_from_u64(seed);

			let sparse = random_sparse(&mut rng, n_vars, n_entries);
			let dense = random_field_buffer::<P>(&mut rng, n_vars);
			prove_verify(sparse, &dense);
		}
	}

	// Entries are not deduplicated, so a multilinear whose every entry shares one index has to
	// prove out the same as its materialized form. Uniform indices rarely pile up like this.
	#[test]
	fn test_sparse_dense_product_sumcheck_repeated_indices() {
		let n_vars = 6;
		let mut rng = StdRng::seed_from_u64(0);

		let sparse = (0..10).map(|_| (23, F::random(&mut rng))).collect();
		let dense = random_field_buffer::<P>(&mut rng, n_vars);
		prove_verify(sparse, &dense);
	}
}
