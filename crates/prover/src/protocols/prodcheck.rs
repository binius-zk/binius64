// Copyright 2025 The Binius Developers

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, line::extrapolate_line_packed};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::rayon::prelude::*;
use binius_verifier::protocols::prodcheck::MultilinearEvalClaim;

use crate::protocols::sumcheck::{
	Error as SumcheckError, ProveSingleOutput, bivariate_product_mle, common::MleCheckProver,
	prove_single_mlecheck,
};

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
}

/// Prover for the product check protocol.
///
/// This prover reduces the claim that a multilinear polynomial evaluates to a product over a
/// Boolean hypercube to a single multilinear evaluation claim.
#[derive(Clone)]
pub struct ProdcheckProver<P: PackedField> {
	/// Product layers from largest (original witness) to second-smallest.
	/// `layers[0]` is the original witness. The final products layer is returned
	/// separately from the constructor.
	layers: Vec<FieldBuffer<P>>,
}

impl<F, P> ProdcheckProver<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	/// Creates a new [`ProdcheckProver`].
	///
	/// Returns `(prover, products)` where `products` is the final layer containing the
	/// products over all `k` variables.
	///
	/// # Arguments
	/// * `k` - The number of variables over which the product is taken. Each reduction step reduces
	///   one variable by computing pairwise products.
	/// * `witness` - The witness polynomial
	///
	/// # Preconditions
	/// * `witness.log_len() >= k`
	pub fn new(k: usize, witness: FieldBuffer<P>) -> (Self, FieldBuffer<P>) {
		assert!(witness.log_len() >= k); // precondition

		let mut layers = Vec::with_capacity(k + 1);
		layers.push(witness);

		for _ in 0..k {
			let prev_layer = layers.last().expect("layers is non-empty");
			let (half_0, half_1) = prev_layer
				.split_half_ref()
				.expect("layer has at least one variable");

			let next_layer_evals = (half_0.as_ref(), half_1.as_ref())
				.into_par_iter()
				.map(|(v0, v1)| *v0 * *v1)
				.collect();
			let next_layer = FieldBuffer::new(prev_layer.log_len() - 1, next_layer_evals)
				.expect("half of previous layer length");

			layers.push(next_layer);
		}

		let products = layers.pop().expect("layers has k+1 elements");
		(Self { layers }, products)
	}

	/// Returns the number of remaining layers to prove.
	pub fn n_layers(&self) -> usize {
		self.layers.len()
	}

	/// Pops the last layer and returns an MLE-check prover for it.
	///
	/// Returns `(layer_prover, remaining)` where:
	/// - `layer_prover` is an MLE-check prover for the popped layer
	/// - `remaining` is `Some(self)` if there are more layers, `None` otherwise
	pub fn layer_prover(
		mut self,
		claim: MultilinearEvalClaim<F>,
	) -> Result<(impl MleCheckProver<F>, Option<Self>), Error> {
		let layer = self.layers.pop().expect("layers is non-empty");
		let split = layer.split_half().expect("layer has at least one variable");

		let remaining = if self.layers.is_empty() {
			None
		} else {
			Some(self)
		};

		let prover = bivariate_product_mle::new(split, claim.point, claim.eval)?;

		Ok((prover, remaining))
	}

	/// Runs the product check protocol and returns the final evaluation claim.
	///
	/// This consumes the prover and runs sumcheck reductions from the smallest layer back to
	/// the largest.
	///
	/// # Arguments
	/// * `claim` - The initial multilinear evaluation claim
	/// * `transcript` - The prover transcript
	///
	/// # Preconditions
	/// * `claim.point.len() == witness.log_len() - k` (where k is the number of reduction layers)
	pub fn prove<Challenger_>(
		self,
		claim: MultilinearEvalClaim<F>,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<MultilinearEvalClaim<F>, Error>
	where
		Challenger_: Challenger,
	{
		let mut prover_opt = Some(self);
		let mut claim = claim;

		while let Some(prover) = prover_opt {
			let (mle_prover, remaining) = prover.layer_prover(claim.clone())?;
			prover_opt = remaining;

			let ProveSingleOutput {
				multilinear_evals,
				challenges,
			} = prove_single_mlecheck(mle_prover, transcript)?;

			let [eval_0, eval_1] = multilinear_evals
				.try_into()
				.expect("prover has two multilinears");

			transcript.message().write(&[eval_0, eval_1]);

			let r = transcript.sample();
			let next_eval = extrapolate_line_packed(eval_0, eval_1, r);

			let mut next_point = challenges;
			next_point.reverse();
			next_point.push(r);

			claim = MultilinearEvalClaim {
				eval: next_eval,
				point: next_point,
			};
		}

		Ok(claim)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::PackedField;
	use binius_math::{
		multilinear::evaluate::evaluate,
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use binius_transcript::ProverTranscript;
	use binius_verifier::{config::StdChallenger, protocols::prodcheck};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	fn test_prodcheck_prove_verify_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		// 1. Create random witness with log_len = n + k
		let witness = random_field_buffer::<P>(&mut rng, n + k);

		// 2. Create prover (computes product layers)
		let (prover, products) = ProdcheckProver::new(k, witness.clone());

		// 3. Generate random n-dimensional challenge point
		let eval_point = random_scalars::<P::Scalar>(&mut rng, n);

		// 4. Evaluate products layer at challenge point to create claim
		let products_eval = evaluate(&products, &eval_point).unwrap();
		let claim = MultilinearEvalClaim {
			eval: products_eval,
			point: eval_point,
		};

		// 5. Run prover
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_output = prover.prove(claim.clone(), &mut prover_transcript).unwrap();

		// 6. Run verifier
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output = prodcheck::verify(k, claim, &mut verifier_transcript).unwrap();

		// 7. Check outputs match
		assert_eq!(prover_output, verifier_output);

		// 8. Verify multilinear evaluation of original witness
		let expected_eval = evaluate(&witness, &verifier_output.point).unwrap();
		assert_eq!(verifier_output.eval, expected_eval);
	}

	#[test]
	fn test_prodcheck_prove_verify() {
		test_prodcheck_prove_verify_helper::<Packed128b>(4, 3);
	}

	#[test]
	fn test_prodcheck_full_prove_verify() {
		test_prodcheck_prove_verify_helper::<Packed128b>(0, 4);
	}

	fn test_prodcheck_layer_computation_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		// Create random witness with log_len = n + k
		let witness = random_field_buffer::<P>(&mut rng, n + k);

		// Create prover (computes product layers)
		let (_prover, products) = ProdcheckProver::new(k, witness.clone());

		// For each index i in the products layer, verify it equals the product of witness values
		// at indices i + z * 2^n for z in 0..2^k (strided access, not contiguous)
		let stride = 1 << n;
		let num_terms = 1 << k;
		for i in 0..(1 << n) {
			let mut expected_product = P::Scalar::ONE;
			for z in 0..num_terms {
				expected_product *= witness.get_checked(i + z * stride).unwrap();
			}
			let actual = products.get_checked(i).unwrap();
			assert_eq!(actual, expected_product, "Product mismatch at index {i}");
		}
	}

	#[test]
	fn test_prodcheck_layer_computation() {
		test_prodcheck_layer_computation_helper::<Packed128b>(4, 3);
	}
}
