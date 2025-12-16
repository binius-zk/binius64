// Copyright 2025 The Binius Developers

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, FieldSlice, line::extrapolate_line_packed};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::rayon::prelude::*;
use binius_verifier::protocols::prodcheck::MultilinearEvalClaim;

use crate::protocols::sumcheck::{
	Error as SumcheckError, ProveSingleOutput, bivariate_product_mle, prove_single_mlecheck,
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
pub struct ProdcheckProver<P: PackedField> {
	claim: MultilinearEvalClaim<P::Scalar>,
	/// Product layers from largest (original witness) to smallest (final products).
	/// `layers[0]` is the original witness, `layers[k]` is the final product layer.
	layers: Vec<FieldBuffer<P>>,
}

impl<F, P> ProdcheckProver<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	/// Creates a new [`ProdcheckProver`].
	///
	/// # Arguments
	/// * `k` - The number of variables over which the product is taken. Each reduction step
	///   reduces one variable by computing pairwise products.
	/// * `claim` - The initial multilinear evaluation claim
	/// * `witness` - The witness polynomial
	///
	/// # Preconditions
	/// * `witness.log_len() == k + claim.point.len()`
	pub fn new(k: usize, claim: MultilinearEvalClaim<F>, witness: FieldBuffer<P>) -> Self {
		assert_eq!(witness.log_len(), k + claim.point.len()); // precondition

		let mut layers = Vec::with_capacity(k + 1);
		layers.push(witness);

		for _ in 0..k {
			let prev_layer = layers.last().expect("layers is non-empty");
			let (half_0, half_1) = prev_layer
				.split_half()
				.expect("layer has at least one variable");

			let next_layer_evals = (half_0.as_ref(), half_1.as_ref())
				.into_par_iter()
				.map(|(v0, v1)| *v0 * *v1)
				.collect();
			let next_layer = FieldBuffer::new(prev_layer.log_len() - 1, next_layer_evals)
				.expect("half of previous layer length");

			layers.push(next_layer);
		}

		Self { claim, layers }
	}

	/// Returns the final product layer as a [`FieldSlice`].
	///
	/// This is the smallest computed layer containing the products over all `k` variables.
	pub fn products(&self) -> FieldSlice<'_, P> {
		self.layers.last().expect("layers is non-empty").to_ref()
	}

	/// Runs the product check protocol and returns the final evaluation claim.
	///
	/// This consumes the prover and runs sumcheck reductions from the smallest layer back to
	/// the largest.
	pub fn prove<Challenger_>(
		self,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<MultilinearEvalClaim<F>, Error>
	where
		Challenger_: Challenger,
	{
		let Self {
			claim,
			mut layers,
		} = self;
		let k = layers.len() - 1;

		if k == 0 {
			return Ok(claim);
		}

		let mut claim = claim;

		// Iterate from the smallest layer back to the largest.
		// Sumchecks run on layers[k-1], layers[k-2], ..., layers[0].
		for i in (0..k).rev() {
			let layer = &mut layers[i];
			let mut split = layer
				.split_half_mut()
				.expect("layer has at least one variable");
			let (half_0, half_1) = split.halves();

			let prover = bivariate_product_mle::new([half_0, half_1], &claim.point, claim.eval)?;
			let ProveSingleOutput {
				multilinear_evals,
				challenges,
			} = prove_single_mlecheck(prover, transcript)?;

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
