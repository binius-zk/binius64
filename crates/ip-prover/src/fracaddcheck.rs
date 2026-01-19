// Copyright 2025-2026 The Binius Developers

use itertools::Itertools;
use std::{array, iter::zip};

use crate::sumcheck::{
	Error as SumcheckError,
	batch::batch_prove_mle_and_write_evals,
	common::MleCheckProver,
	frac_add_last_layer_mle::{SharedFracAddInput, SharedFracAddLastLayerProver},
	frac_add_mle::{self, FractionalBuffer},
};
use binius_field::{Field, PackedField};
use binius_ip::fracaddcheck::FracAddEvalClaim;
use binius_math::{FieldBuffer, line::extrapolate_line_packed};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Prover for the fractional addition protocol.
///
/// Each layer stores paired numerator/denominator evaluations for all leaves at that depth.
/// Moving to the next layer combines siblings using:
/// $$\frac{a_0}{b_0} + \frac{a_1}{b_1} = \frac{a_0b_1 + a_1b_0}{b_0b_1}.$$
#[derive(Debug)]
pub struct FracAddCheckProver<P: PackedField> {
	layers: Vec<(FieldBuffer<P>, FieldBuffer<P>)>,
}

/// Batched prover for multiple fractional-addition trees sharing the same depth.
pub struct BatchFracAddCheckProver<P: PackedField, const N: usize> {
	provers: [FracAddCheckProver<P>; N],
	/// Optional sharing mode for the final layer.
	last_layer_sharing: Option<LastLayerSharing>,
	/// Shared last-layer witness, when available.
	shared_last_layer: Option<SharedLastLayer<P, N>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LastLayerSharing {
	CommonNumerator,
	CommonDenominator,
}

#[derive(Debug)]
pub enum SharedLastLayer<P: PackedField, const N: usize> {
	CommonNumerator {
		/// Shared numerator buffer.
		num: FieldBuffer<P>,
		/// Per-instance denominators.
		den: [FieldBuffer<P>; N],
	},
	CommonDenominator {
		/// Shared denominator buffer.
		den: FieldBuffer<P>,
		/// Per-instance numerators.
		num: [FieldBuffer<P>; N],
	},
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error(
		"mismatched numerator/denominator lengths: numerator log_len {num_log_len}, denominator log_len {den_log_len}"
	)]
	MismatchedWitnessLengths {
		num_log_len: usize,
		den_log_len: usize,
	},
	#[error("batch claims must share the evaluation point")]
	BatchPointMismatch,
	#[error("batch claim point length mismatch: expected {expected}, got {actual}")]
	BatchPointLengthMismatch { expected: usize, actual: usize },
	#[error("batch layer count mismatch")]
	BatchLayerCountMismatch,
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
}

impl<F, P> FracAddCheckProver<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	/// Creates a new [`FracAddCheckProver`].
	///
	/// Returns `(prover, sums)` where `sums` is the final layer containing the
	/// fractional additions over all `k` variables.
	///
	/// This builds `k` reduction layers by repeatedly combining sibling fractions.
	///
	/// # Arguments
	/// * `k` - The number of variables over which the reduction is taken. Each reduction step
	///   reduces one variable by computing fractional additions of sibling terms.
	/// * `witness` - The witness numerator/denominator layers
	///
	/// # Preconditions
	/// * `witness.0.log_len() >= k`
	pub fn new(k: usize, witness: FractionalBuffer<P>) -> (Self, FractionalBuffer<P>) {
		let (witness_num, witness_den) = witness;
		assert!(witness_num.log_len() == witness_den.log_len());
		assert!(witness_num.log_len() >= k);

		let mut layers = Vec::with_capacity(k + 1);
		layers.push((witness_num, witness_den));

		for _ in 0..k {
			let prev_layer = layers.last().expect("layers is non-empty");

			let (num, den) = prev_layer;
			let (num_0, num_1) = num.split_half_ref();
			let (den_0, den_1) = den.split_half_ref();

			// Combine sibling fractions in parallel for the next layer.
			let (next_layer_num, next_layer_den) =
				(num_0.as_ref(), den_0.as_ref(), num_1.as_ref(), den_1.as_ref())
					.into_par_iter()
					.map(|(&a_0, &b_0, &a_1, &b_1)| (a_0 * b_1 + a_1 * b_0, b_0 * b_1))
					.collect::<(Vec<_>, Vec<_>)>();

			let next_layer = (
				FieldBuffer::new(num.log_len() - 1, next_layer_num.into_boxed_slice()),
				FieldBuffer::new(den.log_len() - 1, next_layer_den.into_boxed_slice()),
			);

			layers.push(next_layer);
		}

		let sums = layers.pop().expect("layers has k+1 elements");
		(Self { layers }, sums)
	}

	/// Returns the number of remaining layers to prove.
	pub fn n_layers(&self) -> usize {
		self.layers.len()
	}

	fn pop_layer(mut self) -> ((FieldBuffer<P>, FieldBuffer<P>), Option<Self>) {
		let layer = self.layers.pop().expect("layers is non-empty");
		let remaining = if self.layers.is_empty() {
			None
		} else {
			Some(self)
		};
		(layer, remaining)
	}

	/// Pops the last layer and returns a sumcheck prover for it.
	///
	/// Returns `(layer_prover, remaining)` where:
	/// - `layer_prover` is a sumcheck prover for the popped layer
	/// - `remaining` is `Some(self)` if there are more layers, `None` otherwise
	pub fn layer_prover(
		self,
		claim: FracAddEvalClaim<F>,
	) -> (impl MleCheckProver<F>, Option<Self>) {
		let FracAddEvalClaim {
			num_eval,
			den_eval,
			point,
		} = claim;

		let (layer, remaining) = self.pop_layer();
		let (num, den) = layer;
		let (num_0, num_1) = num.split_half_ref();
		let (den_0, den_1) = den.split_half_ref();
		let num_0 = FieldBuffer::new(num_0.log_len(), num_0.as_ref().into());
		let num_1 = FieldBuffer::new(num_1.log_len(), num_1.as_ref().into());
		let den_0 = FieldBuffer::new(den_0.log_len(), den_0.as_ref().into());
		let den_1 = FieldBuffer::new(den_1.log_len(), den_1.as_ref().into());
		// Build a single-layer MLE-check prover for this fractional-addition step.
		let prover =
			frac_add_mle::new([num_0, num_1, den_0, den_1], point.clone(), [num_eval, den_eval])
				.expect(
					"The splits will go through as long as the layers are constructed correctly.",
				);

		(prover, remaining)
	}

	/// Runs the fractional addition check protocol and returns the final evaluation claims.
	///
	/// This consumes the prover and runs sumcheck reductions from the smallest layer back to
	/// the largest.
	///
	/// # Arguments
	/// * `claim` - The initial numerator/denominator evaluation claim.
	/// * `transcript` - The prover transcript
	///
	/// # Preconditions
	/// * `claim.point.len() == witness.log_len() - k` (where k is the number of reduction layers)
	pub fn prove<Challenger_>(
		self,
		claim: FracAddEvalClaim<F>,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<FracAddEvalClaim<F>, Error>
	where
		Challenger_: Challenger,
	{
		let mut prover_opt = Some(self);
		let mut claim = claim;

		while let Some(prover) = prover_opt {
			let (sumcheck_prover, remaining) = prover.layer_prover(claim);
			prover_opt = remaining;

			let output = batch_prove_mle_and_write_evals(vec![sumcheck_prover], transcript)?;

			let mut multilinear_evals = output.multilinear_evals;
			let evals = multilinear_evals.pop().expect("batch contains one prover");

			let [num_0, num_1, den_0, den_1] = evals
				.try_into()
				.expect("prover evaluates four multilinears");

			let r = transcript.sample();

			let next_num = extrapolate_line_packed(num_0, num_1, r);
			let next_den = extrapolate_line_packed(den_0, den_1, r);

			let mut next_point = output.challenges;
			next_point.push(r);

			claim = FracAddEvalClaim {
				num_eval: next_num,
				den_eval: next_den,
				point: next_point,
			};
		}

		Ok(claim)
	}
}

impl<F, P, const N: usize> BatchFracAddCheckProver<P, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	pub const BATCH_SIZE: usize = N;

	fn convert_evals_to_claims(
		multilinear_evals: Vec<Vec<F>>,
		next_point: Vec<F>,
		r: F,
	) -> Result<[FracAddEvalClaim<F>; N], Error> {
		let mut iter = multilinear_evals.into_iter();
		let claims = array::from_fn(|_| {
			// Each prover emits [num_0, num_1, den_0, den_1] at the current round.
			let evals = iter.next().expect("batch contains N provers");
			let [num_0, num_1, den_0, den_1] = evals
				.try_into()
				.expect("prover evaluates four multilinears");
			let next_num = extrapolate_line_packed(num_0, num_1, r);
			let next_den = extrapolate_line_packed(den_0, den_1, r);
			FracAddEvalClaim {
				num_eval: next_num,
				den_eval: next_den,
				point: next_point.clone(),
			}
		});
		debug_assert!(iter.next().is_none());
		Ok(claims)
	}

	fn convert_shared_evals_to_claims(
		multilinear_evals: Vec<Vec<F>>,
		next_point: Vec<F>,
		r: F,
	) -> Result<[FracAddEvalClaim<F>; N], Error> {
		let mut iter = multilinear_evals.into_iter();
		let evals = iter.next().expect("batch contains one prover");
		debug_assert!(iter.next().is_none());
		assert_eq!(evals.len(), 4 * N, "shared prover emits 4*N evals");

		let claims = array::from_fn(|i| {
			let offset = 4 * i;
			let num_0 = evals[offset];
			let num_1 = evals[offset + 1];
			let den_0 = evals[offset + 2];
			let den_1 = evals[offset + 3];
			// Interpolate halves with the shared challenge for the next claim.
			let next_num = extrapolate_line_packed(num_0, num_1, r);
			let next_den = extrapolate_line_packed(den_0, den_1, r);
			FracAddEvalClaim {
				num_eval: next_num,
				den_eval: next_den,
				point: next_point.clone(),
			}
		});

		Ok(claims)
	}

	/// Creates a batched prover from multiple witnesses, returning final layer sums for each.
	pub fn new(k: usize, witnesses: [FractionalBuffer<P>; N]) -> (Self, [FractionalBuffer<P>; N]) {
		let (provers, sums) = witnesses
			.into_iter()
			.map(|witness| FracAddCheckProver::new(k, witness))
			.collect::<(Vec<_>, Vec<_>)>();

		let provers = provers
			.try_into()
			.expect("witness count matches batch size");
		let sums = sums.try_into().expect("witness count matches batch size");

		(
			Self {
				provers,
				last_layer_sharing: None,
				shared_last_layer: None,
			},
			sums,
		)
	}

	/// Creates a batched prover with last-layer sharing, returning final layer sums for each.
	pub fn new_with_last_layer_sharing(
		k: usize,
		sharing: SharedLastLayer<P, N>,
	) -> (Self, [FractionalBuffer<P>; N]) {
		// Pre-compute one reduction step using the shared witness so the remaining
		// provers all have depth k-1.
		let pruned_witnesses: [(FieldBuffer<P>, FieldBuffer<P>); N] = match &sharing {
			SharedLastLayer::CommonNumerator { num, den } => {
				let (num_0, num_1) = num.split_half_ref();

				den.iter()
					.map(|den_i| {
						let (den_0, den_1) = den_i.split_half_ref();
						let (next_layer_num, next_layer_den) =
							(num_0.as_ref(), den_0.as_ref(), num_1.as_ref(), den_1.as_ref())
								.into_par_iter()
								.map(|(&a_0, &b_0, &a_1, &b_1)| (a_0 * b_1 + a_1 * b_0, b_0 * b_1))
								.collect::<(Vec<_>, Vec<_>)>();
						(
							FieldBuffer::new(num.log_len() - 1, next_layer_num.into_boxed_slice()),
							FieldBuffer::new(
								den_i.log_len() - 1,
								next_layer_den.into_boxed_slice(),
							),
						)
					})
					.collect_array()
					.expect("den is of length N.")
			}
			SharedLastLayer::CommonDenominator { den, num } => {
				let (den_0, den_1) = den.split_half_ref();

				num.iter()
					.map(|num_i| {
						let (num_0, num_1) = num_i.split_half_ref();
						let (next_layer_num, next_layer_den) =
							(num_0.as_ref(), den_0.as_ref(), num_1.as_ref(), den_1.as_ref())
								.into_par_iter()
								.map(|(&a_0, &b_0, &a_1, &b_1)| (a_0 * b_1 + a_1 * b_0, b_0 * b_1))
								.collect::<(Vec<_>, Vec<_>)>();
						(
							FieldBuffer::new(
								num_i.log_len() - 1,
								next_layer_num.into_boxed_slice(),
							),
							FieldBuffer::new(den.log_len() - 1, next_layer_den.into_boxed_slice()),
						)
					})
					.collect_array()
					.expect("num is of length N.")
			}
		};

		let (mut prover, sums) = Self::new(k.saturating_sub(1), pruned_witnesses);
		prover.last_layer_sharing = match &sharing {
			SharedLastLayer::CommonNumerator { .. } => Some(LastLayerSharing::CommonNumerator),
			SharedLastLayer::CommonDenominator { .. } => Some(LastLayerSharing::CommonDenominator),
		};
		prover.shared_last_layer = Some(sharing);

		(prover, sums)
	}

	/// Pops the last layer from each prover and returns sumcheck provers for the batch.
	///
	/// Returns `(layer_provers, remaining)` where:
	/// - `layer_provers` are the sumcheck provers for the popped layer
	/// - `remaining` is `Some(provers)` if there are more layers, `None` otherwise
	pub fn layer_provers(
		self,
		claims: [FracAddEvalClaim<F>; N],
	) -> (Vec<impl MleCheckProver<F>>, Option<BatchFracAddCheckProver<P, N>>) {
		if N == 0 {
			return (Vec::new(), None);
		};
		let n_layers = self.provers[0].n_layers();
		assert!(
			self.provers
				.iter()
				.any(|prover| prover.n_layers() == n_layers)
		);

		let expected_len = self
			.provers
			.get(0)
			.and_then(|prover| prover.layers.last())
			.map(|(num, _)| num.log_len().saturating_sub(1))
			.unwrap_or(0);

		// All batched claims must target the same layer dimension.
		assert!(claims.iter().all(|x| x.point.len() == expected_len));
		let mut remaining: [Option<FracAddCheckProver<P>>; N] = array::from_fn(|_| None);

		let sumcheck_provers = zip(self.provers.into_iter(), claims)
			.enumerate()
			.map(|(idx, (prover, claim))| {
				let (sumcheck_prover, next) = prover.layer_prover(claim);
				remaining[idx] = next;
				sumcheck_prover
			})
			.collect();

		assert!(
			remaining.iter().map(|opt| opt.is_some()).all_equal(),
			"batch layer count mismatch"
		);

		let next_provers = match remaining[0] {
			Some(_) => Some(BatchFracAddCheckProver {
				provers: remaining.map(|opt| opt.expect("remaining prover present")),
				last_layer_sharing: self.last_layer_sharing,
				shared_last_layer: self.shared_last_layer,
			}),
			None => match self.shared_last_layer {
				Some(_) => Some(BatchFracAddCheckProver {
					provers: array::from_fn(|_| FracAddCheckProver { layers: Vec::new() }),
					last_layer_sharing: self.last_layer_sharing,
					shared_last_layer: self.shared_last_layer,
				}),
				None => None,
			},
		};

		(sumcheck_provers, next_provers)
	}

	/// Builds a shared last-layer prover for a batched fractional-addition check.
	///
	/// This validates that all claims have the expected point length, extracts the evaluation
	/// point and per-claim numerator/denominator evaluations, and then constructs a
	/// `SharedFracAddInput` in one of two ways:
	/// - If `self.shared_last_layer` is present, it verifies that the requested `sharing` mode
	///   matches the shared layer variant and splits the shared buffer plus per-claim buffers into
	///   halves.
	/// - Otherwise, it pops the final layer from each prover, ensures there are no remaining
	///   layers, and then builds the shared input from those per-claim buffers, again splitting
	///   into halves.
	///
	/// The resulting input is used to initialize a `SharedFracAddLastLayerProver`.
	fn shared_last_layer_prover(
		self,
		claims: [FracAddEvalClaim<F>; N],
		sharing: LastLayerSharing,
	) -> Result<SharedFracAddLastLayerProver<P, N>, Error> {
		let expected_len = self
			.shared_last_layer
			.as_ref()
			.map(|shared| match shared {
				SharedLastLayer::CommonNumerator { num, .. } => num.log_len().saturating_sub(1),
				SharedLastLayer::CommonDenominator { den, .. } => den.log_len().saturating_sub(1),
			})
			.unwrap_or_else(|| {
				self.provers
					.get(0)
					.and_then(|prover| prover.layers.last())
					.map(|(num, _)| num.log_len().saturating_sub(1))
					.unwrap_or(0)
			});
		// Claims must align with the last-layer dimension of the witness buffers.
		assert!(claims.iter().all(|x| x.point.len() == expected_len));

		let eval_point = claims[0].point.clone();
		let num_evals = array::from_fn(|i| claims[i].num_eval);
		let den_evals = array::from_fn(|i| claims[i].den_eval);

		let BatchFracAddCheckProver {
			provers,
			shared_last_layer,
			..
		} = self;

		let input = match shared_last_layer {
			Some(shared_last_layer) => match (shared_last_layer, sharing) {
				(
					SharedLastLayer::CommonNumerator { num, den },
					LastLayerSharing::CommonNumerator,
				) => {
					let (num_0, num_1) = split_half(num);
					let (den_0, den_1) = split_all(
						den,
						"den_0 length matches batch size",
						"den_1 length matches batch size",
					);
					Ok(SharedFracAddInput::CommonNumerator {
						num_0,
						num_1,
						den_0,
						den_1,
					})
				}
				(
					SharedLastLayer::CommonDenominator { den, num },
					LastLayerSharing::CommonDenominator,
				) => {
					let (den_0, den_1) = split_half(den);
					let (num_0, num_1) = split_all(
						num,
						"num_0 length matches batch size",
						"num_1 length matches batch size",
					);
					Ok(SharedFracAddInput::CommonDenominator {
						den_0,
						den_1,
						num_0,
						num_1,
					})
				}
				_ => Err(Error::BatchLayerCountMismatch),
			},
			None => {
				let layers = provers
					.into_iter()
					.map(|prover| {
						let (layer, remaining) = prover.pop_layer();
						// All provers must be at their final layer for shared reduction.
						remaining
							.is_none()
							.then_some(layer)
							.ok_or(Error::BatchLayerCountMismatch)
					})
					.collect::<Result<Vec<_>, _>>()?;

				let (nums, dens): (Vec<_>, Vec<_>) = layers.into_iter().unzip();

				match sharing {
					LastLayerSharing::CommonNumerator => {
						let shared_num = nums.into_iter().next().expect("batch size > 0");
						let (num_0, num_1) = split_half(shared_num);
						let (den_0, den_1) = split_all(
							dens,
							"den_0 length matches batch size",
							"den_1 length matches batch size",
						);
						Ok(SharedFracAddInput::CommonNumerator {
							num_0,
							num_1,
							den_0,
							den_1,
						})
					}
					LastLayerSharing::CommonDenominator => {
						let shared_den = dens.into_iter().next().expect("batch size > 0");
						let (den_0, den_1) = split_half(shared_den);
						let (num_0, num_1) = split_all(
							nums,
							"num_0 length matches batch size",
							"num_1 length matches batch size",
						);
						Ok(SharedFracAddInput::CommonDenominator {
							den_0,
							den_1,
							num_0,
							num_1,
						})
					}
				}
			}
		}?;

		Ok(SharedFracAddLastLayerProver::new(input, eval_point, num_evals, den_evals)?)
	}

	/// Runs the fractional addition check protocol over a batch of claims.
	///
	/// Iteratively proves each layer using batched sumcheck, updating the evaluation point
	/// and claims at each step via transcript challenges. When the final layer is reached,
	/// it may switch to a shared last-layer prover (if configured) to combine work across
	/// the batch; otherwise it returns the current claims.
	pub fn prove<Challenger_>(
		self,
		claims: [FracAddEvalClaim<F>; N],
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<[FracAddEvalClaim<F>; N], Error>
	where
		Challenger_: Challenger,
	{
		if N == 0 {
			return Ok(claims);
		}
		let mut prover_opt = Some(self);
		let mut claims = claims;

		while let Some(prover) = prover_opt {
			let n_layers = prover.provers[0].n_layers();
			if n_layers == 0 {
				// All per-prover layers consumed; only a shared last-layer remains (if configured).
				if let Some(sharing) = prover.last_layer_sharing {
					if prover.shared_last_layer.is_some() {
						let shared_prover = prover.shared_last_layer_prover(claims, sharing)?;
						let output =
							batch_prove_mle_and_write_evals(vec![shared_prover], transcript)?;

						let r = transcript.sample();
						let mut next_point = output.challenges;
						next_point.push(r);

						let next_claims = Self::convert_shared_evals_to_claims(
							output.multilinear_evals,
							next_point,
							r,
						)?;

						return Ok(next_claims);
					}
				}
				return Ok(claims);
			}

			let (sumcheck_provers, remaining) = prover.layer_provers(claims);
			prover_opt = remaining;

			let output = batch_prove_mle_and_write_evals(sumcheck_provers, transcript)?;

			let r = transcript.sample();
			let mut next_point = output.challenges;
			next_point.push(r);

			let next_claims =
				Self::convert_evals_to_claims(output.multilinear_evals, next_point, r)?;

			claims = next_claims;
		}

		Ok(claims)
	}
}

fn split_half<P: PackedField>(buffer: FieldBuffer<P>) -> (FieldBuffer<P>, FieldBuffer<P>) {
	let (half_0, half_1) = buffer.split_half_ref();
	(
		FieldBuffer::new(half_0.log_len(), half_0.as_ref().into()),
		FieldBuffer::new(half_1.log_len(), half_1.as_ref().into()),
	)
}

fn split_all<P: PackedField, I, const N: usize>(
	buffers: I,
	left_msg: &'static str,
	right_msg: &'static str,
) -> ([FieldBuffer<P>; N], [FieldBuffer<P>; N])
where
	I: IntoIterator<Item = FieldBuffer<P>>,
{
	// Split every buffer into halves and collect left/right halves into fixed arrays.
	let (left, right): (Vec<_>, Vec<_>) = buffers.into_iter().map(split_half).unzip();
	(left.try_into().expect(left_msg), right.try_into().expect(right_msg))
}
#[cfg(test)]
mod tests {
	use std::array;

	use binius_field::PackedField;
	use binius_ip::fracaddcheck;
	use binius_math::{
		multilinear::evaluate::evaluate,
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	fn test_frac_add_check_prove_verify_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		// 1. Create random witness with log_len = n + k
		let witness_num = random_field_buffer::<P>(&mut rng, n + k);
		let witness_den = random_field_buffer::<P>(&mut rng, n + k);

		// 2. Create prover (computes fractional-add layers)
		let (prover, sums) = FracAddCheckProver::new(k, (witness_num.clone(), witness_den.clone()));

		// 3. Generate random n-dimensional challenge point
		let eval_point = random_scalars::<P::Scalar>(&mut rng, n);

		// 4. Evaluate sums at challenge point to create claims
		let sum_num_eval = evaluate(&sums.0, &eval_point);
		let sum_den_eval = evaluate(&sums.1, &eval_point);
		let prover_claim = fracaddcheck::FracAddEvalClaim {
			num_eval: sum_num_eval,
			den_eval: sum_den_eval,
			point: eval_point,
		};
		let verifier_claim = prover_claim.clone();

		// 5. Run prover
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_output = prover
			.prove(prover_claim.clone(), &mut prover_transcript)
			.unwrap();

		// 6. Run verifier
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output =
			fracaddcheck::verify(k, verifier_claim, &mut verifier_transcript).unwrap();

		// 7. Check outputs match
		assert_eq!(prover_output.point, verifier_output.point);
		assert_eq!(prover_output.num_eval, verifier_output.num_eval);
		assert_eq!(prover_output.den_eval, verifier_output.den_eval);

		// 8. Verify multilinear evaluation of original witness
		let expected_num = evaluate(&witness_num, &verifier_output.point);
		let expected_den = evaluate(&witness_den, &verifier_output.point);
		assert_eq!(verifier_output.num_eval, expected_num);
		assert_eq!(verifier_output.den_eval, expected_den);
	}

	#[test]
	fn test_frac_add_check_prove_verify() {
		test_frac_add_check_prove_verify_helper::<Packed128b>(4, 3);
	}

	#[test]
	fn test_frac_add_check_full_prove_verify() {
		test_frac_add_check_prove_verify_helper::<Packed128b>(0, 4);
	}

	fn test_frac_add_check_layer_computation_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		// Create random witness with log_len = n + k
		let witness_num = random_field_buffer::<P>(&mut rng, n + k);
		let witness_den = random_field_buffer::<P>(&mut rng, n + k);

		// Create prover (computes fractional-add layers)
		let (_prover, sums) =
			FracAddCheckProver::new(k, (witness_num.clone(), witness_den.clone()));

		// For each index i in the sums layer, verify it equals the fractional sum of witness values
		// at indices i + z * 2^n for z in 0..2^k (strided access, not contiguous)
		let stride = 1 << n;
		let num_terms = 1 << k;
		for i in 0..(1 << n) {
			let mut expected_num = witness_num.get(i);
			let mut expected_den = witness_den.get(i);
			for z in 1..num_terms {
				let idx = i + z * stride;
				let num_z = witness_num.get(idx);
				let den_z = witness_den.get(idx);
				expected_num = expected_num * den_z + num_z * expected_den;
				expected_den *= den_z;
			}
			let actual_num = sums.0.get(i);
			let actual_den = sums.1.get(i);
			assert_eq!(actual_num, expected_num, "Numerator mismatch at index {i}");
			assert_eq!(actual_den, expected_den, "Denominator mismatch at index {i}");
		}
	}

	#[test]
	fn test_frac_add_check_batch_prove_verify() {
		type P = Packed128b;
		type F = <P as PackedField>::Scalar;
		const N: usize = 3;

		let mut rng = StdRng::seed_from_u64(0);
		let n = 2;
		let k = 3;

		let witnesses: [FractionalBuffer<P>; N] = array::from_fn(|_| {
			let num = random_field_buffer::<P>(&mut rng, n + k);
			let den = random_field_buffer::<P>(&mut rng, n + k);
			(num, den)
		});
		let witnesses_clone = witnesses.clone();

		let (batch_prover, sums) = BatchFracAddCheckProver::<P, N>::new(k, witnesses);
		let eval_point = random_scalars::<F>(&mut rng, n);
		let claims: [fracaddcheck::FracAddEvalClaim<F>; N] = array::from_fn(|i| {
			let (num, den) = &sums[i];
			fracaddcheck::FracAddEvalClaim {
				num_eval: evaluate(num, &eval_point),
				den_eval: evaluate(den, &eval_point),
				point: eval_point.clone(),
			}
		});

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_output = batch_prover
			.prove(claims.clone(), &mut prover_transcript)
			.unwrap();

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output =
			fracaddcheck::verify_batch(k, Vec::from(claims.clone()), &mut verifier_transcript)
				.unwrap();

		assert_eq!(prover_output.into_iter().collect::<Vec<_>>(), verifier_output);

		for (output, (num, den)) in verifier_output.iter().zip(witnesses_clone.iter()) {
			let expected_num = evaluate(num, &output.point);
			let expected_den = evaluate(den, &output.point);
			assert_eq!(output.num_eval, expected_num);
			assert_eq!(output.den_eval, expected_den);
		}
	}

	#[test]
	fn test_frac_add_check_batch_layer_mismatch() {
		type P = Packed128b;
		type F = <P as PackedField>::Scalar;
		let mut rng = StdRng::seed_from_u64(0);
		let n = 1;

		let (prover_a, sums_a) = FracAddCheckProver::new(
			1,
			(random_field_buffer::<P>(&mut rng, n + 1), random_field_buffer::<P>(&mut rng, n + 1)),
		);
		let (prover_b, sums_b) = FracAddCheckProver::new(
			2,
			(random_field_buffer::<P>(&mut rng, n + 2), random_field_buffer::<P>(&mut rng, n + 2)),
		);

		let batch_prover = BatchFracAddCheckProver::<P, 2> {
			provers: [prover_a, prover_b],
			last_layer_sharing: None,
			shared_last_layer: None,
		};

		let eval_point = random_scalars::<F>(&mut rng, n);
		let claims = [
			fracaddcheck::FracAddEvalClaim {
				num_eval: evaluate(&sums_a.0, &eval_point),
				den_eval: evaluate(&sums_a.1, &eval_point),
				point: eval_point.clone(),
			},
			fracaddcheck::FracAddEvalClaim {
				num_eval: evaluate(&sums_b.0, &eval_point),
				den_eval: evaluate(&sums_b.1, &eval_point),
				point: eval_point,
			},
		];

		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let err = batch_prover.prove(claims, &mut transcript).unwrap_err();
		assert!(matches!(err, Error::BatchLayerCountMismatch));
	}

	#[test]
	fn test_frac_add_check_shared_last_layer_common_numerator() {
		type P = Packed128b;
		type F = <P as PackedField>::Scalar;
		const N: usize = 2;

		let mut rng = StdRng::seed_from_u64(0);
		let n = 2;
		let k = 1;

		let shared_num = random_field_buffer::<P>(&mut rng, n + k);
		let dens: [FieldBuffer<P>; N] =
			array::from_fn(|_| random_field_buffer::<P>(&mut rng, n + k));

		let (batch_prover, sums) = BatchFracAddCheckProver::<P, N>::new_with_last_layer_sharing(
			k,
			SharedLastLayer::CommonNumerator {
				num: shared_num.clone(),
				den: dens.clone(),
			},
		);
		let eval_point = random_scalars::<F>(&mut rng, n);
		let claims: [fracaddcheck::FracAddEvalClaim<F>; N] = array::from_fn(|i| {
			let (num, den) = &sums[i];
			fracaddcheck::FracAddEvalClaim {
				num_eval: evaluate(num, &eval_point),
				den_eval: evaluate(den, &eval_point),
				point: eval_point.clone(),
			}
		});

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_output = batch_prover
			.prove(claims.clone(), &mut prover_transcript)
			.unwrap();

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output =
			fracaddcheck::verify_batch(k, Vec::from(claims.clone()), &mut verifier_transcript)
				.unwrap();

		assert_eq!(prover_output.into_iter().collect::<Vec<_>>(), verifier_output);

		for (output, den) in verifier_output.iter().zip(dens) {
			let expected_num = evaluate(&shared_num, &output.point);
			let expected_den = evaluate(&den, &output.point);
			assert_eq!(output.num_eval, expected_num);
			assert_eq!(output.den_eval, expected_den);
		}
	}

	#[test]
	fn test_frac_add_check_shared_last_layer_common_denominator() {
		type P = Packed128b;
		type F = <P as PackedField>::Scalar;
		const N: usize = 3;

		let mut rng = StdRng::seed_from_u64(0);
		let n = 2;
		let k = 1;

		let shared_den = random_field_buffer::<P>(&mut rng, n + k);
		let nums: [FieldBuffer<P>; N] =
			array::from_fn(|_| random_field_buffer::<P>(&mut rng, n + k));

		let (batch_prover, sums) = BatchFracAddCheckProver::<P, N>::new_with_last_layer_sharing(
			k,
			SharedLastLayer::CommonDenominator {
				den: shared_den.clone(),
				num: nums.clone(),
			},
		);
		let eval_point = random_scalars::<F>(&mut rng, n);
		let claims: [fracaddcheck::FracAddEvalClaim<F>; N] = array::from_fn(|i| {
			let (num, den) = &sums[i];
			fracaddcheck::FracAddEvalClaim {
				num_eval: evaluate(num, &eval_point),
				den_eval: evaluate(den, &eval_point),
				point: eval_point.clone(),
			}
		});

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_output = batch_prover
			.prove(claims.clone(), &mut prover_transcript)
			.unwrap();

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output =
			fracaddcheck::verify_batch(k, Vec::from(claims.clone()), &mut verifier_transcript)
				.unwrap();

		assert_eq!(prover_output.into_iter().collect::<Vec<_>>(), verifier_output);

		for (output, num) in verifier_output.iter().zip(nums) {
			let expected_den = evaluate(&shared_den, &output.point);
			let expected_num = evaluate(&num, &output.point);

			assert_eq!(output.num_eval, expected_num);
			assert_eq!(output.den_eval, expected_den);
		}
	}

	#[test]
	fn test_frac_add_check_layer_computation() {
		test_frac_add_check_layer_computation_helper::<Packed128b>(4, 3);
	}
}
