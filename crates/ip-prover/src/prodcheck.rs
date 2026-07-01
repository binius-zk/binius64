// Copyright 2025-2026 The Binius Developers

use std::iter;

use binius_field::{Field, PackedField};
use binius_ip::{mlecheck, prodcheck::MultilinearEvalClaim};
use binius_math::{
	FieldBuffer, line::extrapolate_line_packed, multilinear::eq::eq_ind_partial_eval,
};
use binius_utils::rayon::prelude::*;
use itertools::izip;

use crate::{
	channel::IPProverChannel,
	sumcheck::{
		Error as SumcheckError, ProveSingleOutput, bivariate_product_mle,
		common::{MleCheckProver, SumcheckProver},
		prove_single_mlecheck,
	},
};

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
}

/// Witness-based prover for the product check protocol.
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
			let (half_0, half_1) = prev_layer.split_half_ref();

			let next_layer_evals = (half_0.as_ref(), half_1.as_ref())
				.into_par_iter()
				.map(|(v0, v1)| *v0 * *v1)
				.collect();
			let next_layer = FieldBuffer::new(prev_layer.log_len() - 1, next_layer_evals);

			layers.push(next_layer);
		}

		let products = layers.pop().expect("layers has k+1 elements");
		(Self { layers }, products)
	}

	/// Returns the number of remaining layers to prove.
	pub const fn n_layers(&self) -> usize {
		self.layers.len()
	}

	/// Consumes the prover and returns its single remaining (widest) layer.
	///
	/// # Preconditions
	/// * `self.n_layers() == 1`
	pub fn into_final_layer(mut self) -> FieldBuffer<P> {
		assert_eq!(self.layers.len(), 1, "precondition: exactly one remaining layer");
		self.layers.pop().expect("layers has exactly one element")
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
		let split = layer.split_half();

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
	/// * `channel` - The channel for sending prover messages and sampling challenges
	///
	/// # Preconditions
	/// * `claim.point.len() == witness.log_len() - k` (where k is the number of reduction layers)
	pub fn prove(
		self,
		claim: MultilinearEvalClaim<F>,
		channel: &mut impl IPProverChannel<F>,
	) -> Result<MultilinearEvalClaim<F>, Error> {
		let mut prover_opt = Some(self);
		let mut claim = claim;

		while let Some(prover) = prover_opt {
			let (mle_prover, remaining) = prover.layer_prover(claim.clone())?;
			prover_opt = remaining;

			let ProveSingleOutput {
				multilinear_evals,
				challenges,
			} = prove_single_mlecheck(mle_prover, channel)?;

			let [eval_0, eval_1] = multilinear_evals
				.try_into()
				.expect("prover has two multilinears");

			channel.send_many(&[eval_0, eval_1]);

			let r = channel.sample();
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

/// Output of [`batch_prove`].
///
/// After running `n_layers - 1` reduction layers, each prover retains its final (widest) layer.
/// `provers` pairs each remaining prover with its reduced evaluation at `eval_point`.
pub struct BatchProveOutput<P: PackedField> {
	/// The reduced evaluation point shared by all remaining provers.
	pub eval_point: Vec<P::Scalar>,
	/// Each remaining prover (with its final layer) paired with its reduced eval at `eval_point`.
	pub provers: Vec<(P::Scalar, ProdcheckProver<P>)>,
}

/// Runs a batched product check protocol for multiple independent prodcheck provers.
///
/// This combines n provers, each for an $m$-variate multilinear, using multilinear interpolation
/// over k selector variables (where $n \le 2^k$). The combined claim is the multilinear
/// extrapolation of the individual claimed products (padded with zeros to $2^k$) evaluated at the
/// given point.
///
/// The claimed products may themselves be evaluations of the $m$-variate product multilinears at a
/// shared `content_point` (of length equal to each prover's reduced product dimension). When the
/// products are scalars (each prover reduces over all of its variables), `content_point` is empty.
///
/// # Arguments
/// * `provers` - Vec of n prodcheck provers. All must have the same `n_layers()`, which is $m$.
/// * `claimed_products` - Vec of n claimed product values, one per prover. Each is the
///   corresponding prover's product multilinear evaluated at `content_point`.
/// * `selector_point` - Evaluation point for the selector variables. Length is $k$.
/// * `content_point` - Shared evaluation point at which the claimed products are taken. Length is
///   the product-multilinear dimension (i.e. `witness.log_len() - n_layers`). Empty for scalar
///   products.
/// * `channel` - The channel for sending prover messages and sampling challenges.
///
/// # Preconditions
/// * `provers` must be non-empty.
/// * All provers must have the same `n_layers()` value.
/// * `2^selector_point.len() >= provers.len()`.
/// * `claimed_products.len() == provers.len()`.
/// * `content_point.len() == witness.log_len() - n_layers` for each prover.
///
/// This runs `n_layers - 1` of the per-layer reductions, stopping one layer short so that each
/// prover retains its final (widest) layer. The remaining provers are returned (each paired with
/// its reduced eval at the shared reduced `eval_point`) rather than combined into a single claim,
/// so the caller can finish the final layer itself.
///
/// # Returns
/// A [`BatchProveOutput`] with the reduced `eval_point` and the remaining (single-layer) provers,
/// each paired with its reduced eval at that point.
///
/// # Mathematical Description
///
/// Let $f_i \in K[X_0, \ldots, X_{m-1}]$ be multilinear for all $i \in \{0, \ldots, n - 1\}$. The
/// $i$'th prover is a prodcheck prover for $f_i$. Let $p_i \in K$ be the claimed hypercube product
/// of $f_i$.
///
/// Let $y \in K^k$ be the evaluation point. The prover is proving a claim that
///
/// $$
/// \sum_{i \in B_k} \textsf{eq}(i; y) \prod_{j \in B_m} f_i(j) = \sum_{i \in B_k} \textsf{eq}(i; y)
/// p_i, $$
///
/// reducing to an evaluation of the interpolated multilinear
///
/// $$
/// \hat{f}(Y_0, \ldots, Y_{k-1}, X_0, \ldots, X_{m-1}) = \sum_{i \in B_k} \textsf{eq}(i; Y) f_i(X).
/// $$
pub fn batch_prove<F: Field, P: PackedField<Scalar = F>>(
	provers: Vec<ProdcheckProver<P>>,
	claimed_products: Vec<F>,
	selector_point: Vec<F>,
	content_point: Vec<F>,
	channel: &mut impl IPProverChannel<F>,
) -> Result<BatchProveOutput<P>, Error> {
	assert!(!provers.is_empty()); // precondition
	assert_eq!(claimed_products.len(), provers.len()); // precondition

	let k = selector_point.len();
	assert!(provers.len() <= (1 << k)); // precondition

	let n_layers = provers[0].n_layers();
	assert!(n_layers >= 1); // precondition
	assert!(provers.iter().all(|p| p.n_layers() == n_layers)); // precondition

	// Thread the content point as the initial inner (content) coordinates of the evaluation point.
	// `batch_prove_layer` splits `eval_point.split_at(k)` into (selector, content); on the first
	// layer this seeds each layer prover with `claim{point: content_point, eval: claimed_product}`.
	let eval_point = [selector_point, content_point].concat();

	// Run `n_layers - 1` reductions, stopping one layer short so each prover retains its final
	// (widest) layer for the caller to finish.
	let (provers, claimed_products, eval_point) = (0..n_layers - 1).try_fold(
		(provers, claimed_products, eval_point),
		|(provers, claimed_products, eval_point), _| {
			batch_prove_layer(provers, claimed_products, eval_point, k, channel)
		},
	)?;

	// Pair each remaining (single-layer) prover with its reduced eval at `eval_point`.
	let provers = iter::zip(claimed_products, provers).collect();

	Ok(BatchProveOutput {
		eval_point,
		provers,
	})
}

#[allow(clippy::type_complexity)]
fn batch_prove_layer<F: Field, P: PackedField<Scalar = F>>(
	provers: Vec<ProdcheckProver<P>>,
	claimed_products: Vec<F>,
	eval_point: Vec<F>,
	k: usize,
	channel: &mut impl IPProverChannel<F>,
) -> Result<(Vec<ProdcheckProver<P>>, Vec<F>, Vec<F>), Error> {
	// Split eval_point into outer (selector) and inner (content) coordinates.
	let (outer_coords, inner_coords) = eval_point.split_at(k);

	let (mut layer_provers, next_provers): (Vec<_>, Vec<_>) = iter::zip(provers, claimed_products)
		.map(|(prover, prod)| {
			prover
				.layer_prover(MultilinearEvalClaim {
					eval: prod,
					point: inner_coords.to_vec(),
				})
				.unwrap()
		})
		.unzip();

	// Compute eq weights for batching: eq(i, outer_coords) for all i in B_k.
	let eq_weights = eq_ind_partial_eval::<F>(outer_coords);

	// Content rounds: individual provers operate independently.
	let mut challenges = Vec::with_capacity(eval_point.len());

	for _round in 0..inner_coords.len() {
		// Execute each prover and compute weighted sum of round coefficients.
		let coeffss = layer_provers
			.iter_mut()
			.map(|prover| {
				let mut round_coeffs_vec = prover.execute()?;
				Ok(round_coeffs_vec
					.pop()
					.expect("prodcheck layer provers have round_coeffs_vec.len() == 1"))
			})
			.collect::<Result<Vec<_>, SumcheckError>>()?;

		let coeffs = iter::zip(coeffss, eq_weights.iter_scalars())
			.map(|(coeffs, weight)| coeffs * weight)
			.sum();

		// Send truncated round proof to channel.
		channel.send_many(mlecheck::RoundProof::truncate(coeffs).coeffs());

		// Sample challenge and fold all provers.
		let challenge = channel.sample();
		challenges.push(challenge);

		for prover in layer_provers.iter_mut() {
			prover.fold(challenge)?;
		}
	}

	// Finish inner provers to get [eval_0, eval_1] pairs.
	let (mut vals_0, mut vals_1): (Vec<F>, Vec<F>) = layer_provers
		.into_iter()
		.map(|prover| {
			let evals = prover.finish()?;
			let [e0, e1]: [F; 2] = evals
				.try_into()
				.expect("bivariate product prover has two multilinears");
			Ok((e0, e1))
		})
		.collect::<Result<Vec<_>, SumcheckError>>()?
		.into_iter()
		.unzip();

	// Pad vals_0 and vals_1 to 2^k with zeros for FieldBuffer::from_values.
	vals_0.resize(1 << k, F::ZERO);
	vals_1.resize(1 << k, F::ZERO);

	// Compute eval from buffers: sum_v eq(v, outer_coords) * vals_0[v] * vals_1[v].
	let eval = izip!(&vals_0, &vals_1, eq_weights.as_ref())
		.map(|(&v0, &v1, &eq_i)| v0 * v1 * eq_i)
		.sum();

	// Selector rounds: pack eval pairs into FieldBuffers and use a single prover.
	let buffer_0 = FieldBuffer::<P>::from_values(&vals_0);
	let buffer_1 = FieldBuffer::<P>::from_values(&vals_1);

	let outer_prover =
		bivariate_product_mle::new([buffer_0, buffer_1], outer_coords.to_vec(), eval)?;

	let ProveSingleOutput {
		multilinear_evals: outer_evals,
		challenges: outer_challenges,
	} = prove_single_mlecheck(outer_prover, channel)?;

	challenges.extend(outer_challenges);

	let [merged_eval_0, merged_eval_1]: [F; 2] =
		outer_evals.try_into().expect("prover has two multilinears");

	// Finalize layer: send evals, sample r, compute next claim.
	channel.send_many(&[merged_eval_0, merged_eval_1]);

	let r = channel.sample();

	let mut next_point = challenges;
	next_point.reverse();
	next_point.push(r);

	// Update claimed products for next iteration.
	let next_claimed_products = iter::zip(&vals_0, &vals_1)
		.map(|(e0, e1)| extrapolate_line_packed(*e0, *e1, r))
		.collect();

	let next_provers = next_provers.into_iter().flatten().collect();

	Ok((next_provers, next_claimed_products, next_point))
}
#[cfg(test)]
mod tests {
	use binius_field::PackedField;
	use binius_ip::prodcheck;
	use binius_math::{
		inner_product::inner_product,
		multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use binius_utils::checked_arithmetics::log2_ceil_usize;

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use rand::prelude::*;

	use super::*;

	/// Finishes a [`batch_prove`] output by running the final retained layer and combining the
	/// per-prover reduced evals into a single [`MultilinearEvalClaim`], matching the full-reduction
	/// claim the verifier produces.
	fn finish_batch_prove<F: Field, P: PackedField<Scalar = F>>(
		output: BatchProveOutput<P>,
		k: usize,
		channel: &mut impl IPProverChannel<F>,
	) -> MultilinearEvalClaim<F> {
		let BatchProveOutput {
			eval_point,
			provers,
		} = output;
		let (claimed_products, provers): (Vec<_>, Vec<_>) = provers.into_iter().unzip();

		let (_provers, claimed_products, eval_point) =
			batch_prove_layer(provers, claimed_products, eval_point, k, channel).unwrap();

		let eq_weights = eq_ind_partial_eval::<F>(&eval_point[..k]);
		let final_eval = inner_product(claimed_products.iter().copied(), eq_weights.iter_scalars());

		MultilinearEvalClaim {
			eval: final_eval,
			point: eval_point,
		}
	}

	fn test_prodcheck_prove_verify_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		// 1. Create random witness with log_len = n + k
		let witness = random_field_buffer::<P>(&mut rng, n + k);

		// 2. Create prover (computes product layers)
		let (prover, products) = ProdcheckProver::new(k, witness.clone());

		// 3. Generate random n-dimensional challenge point
		let eval_point = random_scalars::<P::Scalar>(&mut rng, n);

		// 4. Evaluate products layer at challenge point to create claim
		let products_eval = evaluate(&products, &eval_point);
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
		let expected_eval = evaluate(&witness, &verifier_output.point);
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
				expected_product *= witness.get(i + z * stride);
			}
			let actual = products.get(i);
			assert_eq!(actual, expected_product, "Product mismatch at index {i}");
		}
	}

	#[test]
	fn test_prodcheck_layer_computation() {
		test_prodcheck_layer_computation_helper::<Packed128b>(4, 3);
	}

	// ==================== batch_prove tests ====================

	/// Helper function for testing batch_prove with ProdcheckProvers.
	///
	/// Each witness has exactly `n_layers` variables so that the products are scalars (0-variate).
	///
	/// # Arguments
	/// * `n_layers` - Number of product reduction layers (= variables per witness)
	/// * `n_provers` - Number of provers to batch
	fn test_batch_prove_verify_helper<P: PackedField>(n_layers: usize, n_provers: usize) {
		let mut rng = StdRng::seed_from_u64(42);

		let log_n_provers = log2_ceil_usize(n_provers);

		// Each witness has exactly n_layers variables; products are scalars
		let witnesses: Vec<FieldBuffer<P>> = (0..n_provers)
			.map(|_| random_field_buffer::<P>(&mut rng, n_layers))
			.collect();

		// Create ProdcheckProver for each
		let provers_and_products: Vec<(ProdcheckProver<P>, FieldBuffer<P>)> = witnesses
			.iter()
			.map(|witness| ProdcheckProver::new(n_layers, witness.clone()))
			.collect();

		let (provers, individual_products): (Vec<_>, Vec<_>) =
			provers_and_products.into_iter().unzip();

		// Products are 0-variate (scalars): just get the single value
		let claimed_products: Vec<P::Scalar> = individual_products
			.iter()
			.map(|products| {
				assert_eq!(products.log_len(), 0);
				products.get(0)
			})
			.collect();

		// Generate random selector challenge point (length = log_n_provers)
		let selector_challenge = random_scalars::<P::Scalar>(&mut rng, log_n_provers);

		// Compute combined claim using eq weights
		let eq_weights = eq_ind_partial_eval::<P>(&selector_challenge);
		let combined_eval = inner_product(
			claimed_products.iter().copied(),
			(0..n_provers).map(|i| eq_weights.get(i)),
		);

		// The verifier claim has point = selector_challenge (length log_n_provers)
		let claim = MultilinearEvalClaim {
			eval: combined_eval,
			point: selector_challenge.clone(),
		};

		// Run batch_prove (scalar products: empty content point), then finish the final layer.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let batch_output = batch_prove(
			provers,
			claimed_products,
			selector_challenge,
			Vec::new(),
			&mut prover_transcript,
		)
		.unwrap();
		// Each remaining prover has exactly one layer.
		assert!(batch_output.provers.iter().all(|(_, p)| p.n_layers() == 1));
		let prover_output = finish_batch_prove(batch_output, log_n_provers, &mut prover_transcript);

		// Run verifier with n_layers layers
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output = prodcheck::verify(n_layers, claim, &mut verifier_transcript).unwrap();

		// Check prover and verifier outputs match
		assert_eq!(prover_output, verifier_output);

		// Verify final evaluation against multilinear extrapolation of input witnesses
		let final_point = &verifier_output.point;
		assert_eq!(final_point.len(), log_n_provers + n_layers);

		let selector_challenges = &final_point[..log_n_provers];
		let content_challenges = &final_point[log_n_provers..];

		let selector_weights = eq_ind_partial_eval::<P>(selector_challenges);

		let expected_eval: P::Scalar = inner_product(
			(0..n_provers).map(|i| evaluate(&witnesses[i], content_challenges)),
			(0..n_provers).map(|i| selector_weights.get(i)),
		);

		assert_eq!(
			verifier_output.eval, expected_eval,
			"Final evaluation should match batch witness interpolation"
		);
	}

	#[test]
	fn test_batch_prove_power_of_two_provers() {
		// 4 provers, 3 layers
		test_batch_prove_verify_helper::<Packed128b>(3, 4);
	}

	#[test]
	fn test_batch_prove_non_power_of_two_provers() {
		// 3 provers (non-power of 2, requires padding), 4 layers
		test_batch_prove_verify_helper::<Packed128b>(4, 3);
	}

	#[test]
	fn test_batch_prove_single_prover() {
		// 1 prover (edge case), 5 layers
		test_batch_prove_verify_helper::<Packed128b>(5, 1);
	}

	#[test]
	fn test_batch_prove_single_layer() {
		// n_layers=1 edge case (the minimum): batch_prove runs 0 reductions and retains the single
		// (final) layer, which the test then finishes.
		test_batch_prove_verify_helper::<Packed128b>(1, 4);
	}

	/// Helper for testing batch_prove where the claimed products are non-scalar: each prover's
	/// product multilinear is `content_len`-variate, claimed at a shared random content point.
	///
	/// # Arguments
	/// * `n_layers` - Number of product reduction layers (= selector-reduced variables per witness)
	/// * `n_provers` - Number of provers to batch
	/// * `content_len` - Number of variables of the product multilinear (witness has log_len =
	///   content_len + n_layers)
	fn test_batch_prove_with_content_helper<P: PackedField>(
		n_layers: usize,
		n_provers: usize,
		content_len: usize,
	) {
		let mut rng = StdRng::seed_from_u64(7);

		let log_n_provers = log2_ceil_usize(n_provers);

		// Each witness has log_len = content_len + n_layers; products are content_len-variate.
		let witnesses: Vec<FieldBuffer<P>> = (0..n_provers)
			.map(|_| random_field_buffer::<P>(&mut rng, content_len + n_layers))
			.collect();

		let provers_and_products: Vec<(ProdcheckProver<P>, FieldBuffer<P>)> = witnesses
			.iter()
			.map(|witness| ProdcheckProver::new(n_layers, witness.clone()))
			.collect();

		let (provers, individual_products): (Vec<_>, Vec<_>) =
			provers_and_products.into_iter().unzip();

		// Shared content point; each claimed product is its multilinear evaluated there.
		let content_point = random_scalars::<P::Scalar>(&mut rng, content_len);
		let claimed_products: Vec<P::Scalar> = individual_products
			.iter()
			.map(|products| {
				assert_eq!(products.log_len(), content_len);
				evaluate(products, &content_point)
			})
			.collect();

		// Combined verifier claim: eq(selector)-weighted sum of the claimed products, at point
		// selector ++ content.
		let selector_challenge = random_scalars::<P::Scalar>(&mut rng, log_n_provers);
		let eq_weights = eq_ind_partial_eval::<P>(&selector_challenge);
		let combined_eval = inner_product(
			claimed_products.iter().copied(),
			(0..n_provers).map(|i| eq_weights.get(i)),
		);

		let claim = MultilinearEvalClaim {
			eval: combined_eval,
			point: [selector_challenge.clone(), content_point.clone()].concat(),
		};

		// Run batch_prove with non-empty content point, then finish the final layer.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let batch_output = batch_prove(
			provers,
			claimed_products,
			selector_challenge,
			content_point,
			&mut prover_transcript,
		)
		.unwrap();
		assert!(batch_output.provers.iter().all(|(_, p)| p.n_layers() == 1));
		let prover_output = finish_batch_prove(batch_output, log_n_provers, &mut prover_transcript);

		// Run verifier with n_layers layers.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output = prodcheck::verify(n_layers, claim, &mut verifier_transcript).unwrap();

		assert_eq!(prover_output, verifier_output);

		// The reduced point is [selector (log_n_provers), witness vars (content_len + n_layers)],
		// where the witness-var coordinates are in each witness's own variable order — so
		// `evaluate(witness_i, witness_challenges)` is the per-prover content evaluation that the
		// selector-weighted sum recombines.
		let final_point = &verifier_output.point;
		assert_eq!(final_point.len(), log_n_provers + n_layers + content_len);

		let selector_challenges = &final_point[..log_n_provers];
		let witness_challenges = &final_point[log_n_provers..];

		let selector_weights = eq_ind_partial_eval::<P>(selector_challenges);

		let expected_eval: P::Scalar = inner_product(
			(0..n_provers).map(|i| evaluate(&witnesses[i], witness_challenges)),
			(0..n_provers).map(|i| selector_weights.get(i)),
		);

		assert_eq!(
			verifier_output.eval, expected_eval,
			"Final evaluation should match batch witness interpolation"
		);
	}

	#[test]
	fn test_batch_prove_with_content() {
		// 3 provers (non power of 2), 4 layers, content_len = 2.
		test_batch_prove_with_content_helper::<Packed128b>(4, 3, 2);
	}
}
