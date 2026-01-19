// Copyright 2025-2026 The Binius Developers

use std::array;

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
/// Each layer is a double of the numerator and denominator values of fractional terms. Each layer
/// represents the addition of siblings with respect to the fractional addition rule:
/// $$\frac{a_0}{b_0} + \frac{a_1}{b_1} = \frac{a_0b_1 + a_1b_0}{b_0b_1}$
#[derive(Debug)]
pub struct FracAddCheckProver<P: PackedField> {
	layers: Vec<(FieldBuffer<P>, FieldBuffer<P>)>,
}

/// Batched prover for multiple fractional-addition trees sharing the same depth.
pub struct BatchFracAddCheckProver<P: PackedField, const N: usize> {
	provers: [FracAddCheckProver<P>; N],
	last_layer_sharing: Option<LastLayerSharing>,
	shared_last_layer: Option<SharedLastLayer<P, N>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LastLayerSharing {
	CommonNumerator,
	CommonDenominator,
}

#[derive(Debug)]
enum SharedLastLayer<P: PackedField, const N: usize> {
	CommonNumerator {
		num: FieldBuffer<P>,
		den: [FieldBuffer<P>; N],
	},
	CommonDenominator {
		den: FieldBuffer<P>,
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

	fn take_first_layer(mut self) -> ((FieldBuffer<P>, FieldBuffer<P>), Option<Self>) {
		let layer = self.layers.remove(0);
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
	) -> Result<(impl MleCheckProver<F>, Option<Self>), Error> {
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
		let prover =
			frac_add_mle::new([num_0, num_1, den_0, den_1], point.clone(), [num_eval, den_eval])?;

		Ok((prover, remaining))
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
			let (sumcheck_prover, remaining) = prover.layer_prover(claim)?;
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

	fn validate_claim_points(
		claims: &[FracAddEvalClaim<F>; N],
		expected_len: usize,
	) -> Result<(), Error> {
		if N == 0 {
			return Ok(());
		}
		let point = &claims[0].point;
		if point.len() != expected_len {
			return Err(Error::BatchPointLengthMismatch {
				expected: expected_len,
				actual: point.len(),
			});
		}
		if !claims.iter().all(|claim| claim.point == *point) {
			return Err(Error::BatchPointMismatch);
		}
		Ok(())
	}

	fn convert_evals_to_claims(
		multilinear_evals: Vec<Vec<F>>,
		next_point: Vec<F>,
		r: F,
	) -> Result<[FracAddEvalClaim<F>; N], Error> {
		let mut iter = multilinear_evals.into_iter();
		let claims = array::from_fn(|_| {
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
		let mut provers = Vec::with_capacity(N);
		let mut sums = Vec::with_capacity(N);
		for witness in witnesses {
			let (prover, sum) = FracAddCheckProver::new(k, witness);
			provers.push(prover);
			sums.push(sum);
		}

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
		witnesses: [FractionalBuffer<P>; N],
		sharing: LastLayerSharing,
	) -> (Self, [FractionalBuffer<P>; N]) {
		let (prover, sums) = Self::new(k, witnesses);
		(prover.with_last_layer_sharing(sharing), sums)
	}

	pub fn with_last_layer_sharing(mut self, sharing: LastLayerSharing) -> Self {
		self.last_layer_sharing = Some(sharing);
		if N == 0 || self.shared_last_layer.is_some() {
			return self;
		}
		if self.provers[0].n_layers() == 0 {
			return self;
		}

		let mut remaining: [Option<FracAddCheckProver<P>>; N] = array::from_fn(|_| None);
		match sharing {
			LastLayerSharing::CommonNumerator => {
				let mut shared_num = None;
				let mut den_layers = Vec::with_capacity(N);
				for (idx, prover) in self.provers.into_iter().enumerate() {
					let (layer, next) = prover.take_first_layer();
					let (num, den) = layer;
					if idx == 0 {
						shared_num = Some(num);
					}
					den_layers.push(den);
					remaining[idx] = next;
				}

				let shared_num = shared_num.expect("batch size > 0");
				let den = den_layers
					.try_into()
					.expect("den length matches batch size");
				self.shared_last_layer = Some(SharedLastLayer::CommonNumerator {
					num: shared_num,
					den,
				});
			}
			LastLayerSharing::CommonDenominator => {
				let mut shared_den = None;
				let mut num_layers = Vec::with_capacity(N);
				for (idx, prover) in self.provers.into_iter().enumerate() {
					let (layer, next) = prover.take_first_layer();
					let (num, den) = layer;
					if idx == 0 {
						shared_den = Some(den);
					}
					num_layers.push(num);
					remaining[idx] = next;
				}

				let shared_den = shared_den.expect("batch size > 0");
				let num = num_layers
					.try_into()
					.expect("num length matches batch size");
				self.shared_last_layer = Some(SharedLastLayer::CommonDenominator {
					den: shared_den,
					num,
				});
			}
		}

		let any_some = remaining.iter().any(|opt| opt.is_some());
		let any_none = remaining.iter().any(|opt| opt.is_none());
		assert!(!(any_some && any_none), "batch layer count mismatch");

		self.provers = if any_some {
			remaining.map(|opt| opt.expect("remaining prover present"))
		} else {
			array::from_fn(|_| FracAddCheckProver { layers: Vec::new() })
		};
		self
	}

	/// Pops the last layer from each prover and returns sumcheck provers for the batch.
	///
	/// Returns `(layer_provers, remaining)` where:
	/// - `layer_provers` are the sumcheck provers for the popped layer
	/// - `remaining` is `Some(provers)` if there are more layers, `None` otherwise
	pub fn layer_provers(
		self,
		claims: [FracAddEvalClaim<F>; N],
	) -> Result<(Vec<impl MleCheckProver<F>>, Option<BatchFracAddCheckProver<P, N>>), Error> {
		if N == 0 {
			return Ok((Vec::new(), None));
		};
		let n_layers = self.provers[0].n_layers();
		if self
			.provers
			.iter()
			.any(|prover| prover.n_layers() != n_layers)
		{
			return Err(Error::BatchLayerCountMismatch);
		}

		let expected_len = self
			.provers
			.get(0)
			.and_then(|prover| prover.layers.last())
			.map(|(num, _)| num.log_len().saturating_sub(1))
			.unwrap_or(0);
		Self::validate_claim_points(&claims, expected_len)?;

		let mut remaining: [Option<FracAddCheckProver<P>>; N] = array::from_fn(|_| None);
		let mut sumcheck_provers = Vec::with_capacity(N);
		for (idx, (prover, claim)) in self.provers.into_iter().zip(claims).enumerate() {
			let (sumcheck_prover, next) = prover.layer_prover(claim)?;
			sumcheck_provers.push(sumcheck_prover);
			remaining[idx] = next;
		}

		let any_some = remaining.iter().any(|opt| opt.is_some());
		let any_none = remaining.iter().any(|opt| opt.is_none());
		if any_some && any_none {
			return Err(Error::BatchLayerCountMismatch);
		}

		let next_provers = if any_some {
			Some(BatchFracAddCheckProver {
				provers: remaining.map(|opt| opt.expect("remaining prover present")),
				last_layer_sharing: self.last_layer_sharing,
				shared_last_layer: self.shared_last_layer,
			})
		} else if self.shared_last_layer.is_some() {
			Some(BatchFracAddCheckProver {
				provers: array::from_fn(|_| FracAddCheckProver { layers: Vec::new() }),
				last_layer_sharing: self.last_layer_sharing,
				shared_last_layer: self.shared_last_layer,
			})
		} else {
			None
		};

		Ok((sumcheck_provers, next_provers))
	}

	fn shared_last_layer_prover(
		self,
		claims: [FracAddEvalClaim<F>; N],
		sharing: LastLayerSharing,
	) -> Result<SharedFracAddLastLayerProver<P, N>, Error> {
		let expected_len = match &self.shared_last_layer {
			Some(SharedLastLayer::CommonNumerator { num, .. }) => num.log_len().saturating_sub(1),
			Some(SharedLastLayer::CommonDenominator { den, .. }) => den.log_len().saturating_sub(1),
			None => self
				.provers
				.get(0)
				.and_then(|prover| prover.layers.last())
				.map(|(num, _)| num.log_len().saturating_sub(1))
				.unwrap_or(0),
		};
		Self::validate_claim_points(&claims, expected_len)?;

		let eval_point = claims[0].point.clone();
		let num_evals = array::from_fn(|i| claims[i].num_eval);
		let den_evals = array::from_fn(|i| claims[i].den_eval);

		if let Some(shared_last_layer) = self.shared_last_layer {
			match (shared_last_layer, sharing) {
				(
					SharedLastLayer::CommonNumerator { num, den },
					LastLayerSharing::CommonNumerator,
				) => {
					let (num_0_half, num_1_half) = num.split_half_ref();
					let num_0 = FieldBuffer::new(num_0_half.log_len(), num_0_half.as_ref().into());
					let num_1 = FieldBuffer::new(num_1_half.log_len(), num_1_half.as_ref().into());

					let mut den_0 = Vec::with_capacity(N);
					let mut den_1 = Vec::with_capacity(N);
					for den in den.into_iter() {
						let (den_0_half, den_1_half) = den.split_half_ref();
						den_0.push(FieldBuffer::new(
							den_0_half.log_len(),
							den_0_half.as_ref().into(),
						));
						den_1.push(FieldBuffer::new(
							den_1_half.log_len(),
							den_1_half.as_ref().into(),
						));
					}
					let den_0 = den_0.try_into().expect("den_0 length matches batch size");
					let den_1 = den_1.try_into().expect("den_1 length matches batch size");

					let input = SharedFracAddInput::CommonNumerator {
						num_0,
						num_1,
						den_0,
						den_1,
					};

					return Ok(SharedFracAddLastLayerProver::new(
						input, eval_point, num_evals, den_evals,
					)?);
				}
				(
					SharedLastLayer::CommonDenominator { den, num },
					LastLayerSharing::CommonDenominator,
				) => {
					let (den_0_half, den_1_half) = den.split_half_ref();
					let den_0 = FieldBuffer::new(den_0_half.log_len(), den_0_half.as_ref().into());
					let den_1 = FieldBuffer::new(den_1_half.log_len(), den_1_half.as_ref().into());

					let mut num_0 = Vec::with_capacity(N);
					let mut num_1 = Vec::with_capacity(N);
					for num in num.into_iter() {
						let (num_0_half, num_1_half) = num.split_half_ref();
						num_0.push(FieldBuffer::new(
							num_0_half.log_len(),
							num_0_half.as_ref().into(),
						));
						num_1.push(FieldBuffer::new(
							num_1_half.log_len(),
							num_1_half.as_ref().into(),
						));
					}
					let num_0 = num_0.try_into().expect("num_0 length matches batch size");
					let num_1 = num_1.try_into().expect("num_1 length matches batch size");

					let input = SharedFracAddInput::CommonDenominator {
						den_0,
						den_1,
						num_0,
						num_1,
					};

					return Ok(SharedFracAddLastLayerProver::new(
						input, eval_point, num_evals, den_evals,
					)?);
				}
				(_, _) => return Err(Error::BatchLayerCountMismatch),
			}
		}

		match sharing {
			LastLayerSharing::CommonNumerator => {
				let mut shared_num = None;
				let mut den_0 = Vec::with_capacity(N);
				let mut den_1 = Vec::with_capacity(N);

				for (idx, prover) in self.provers.into_iter().enumerate() {
					let (layer, remaining) = prover.pop_layer();
					if remaining.is_some() {
						return Err(Error::BatchLayerCountMismatch);
					}
					let (num, den) = layer;
					if idx == 0 {
						shared_num = Some(num);
					}

					let (den_0_half, den_1_half) = den.split_half_ref();
					den_0.push(FieldBuffer::new(den_0_half.log_len(), den_0_half.as_ref().into()));
					den_1.push(FieldBuffer::new(den_1_half.log_len(), den_1_half.as_ref().into()));
				}

				let shared_num = shared_num.expect("batch size > 0");
				let (num_0_half, num_1_half) = shared_num.split_half_ref();
				let num_0 = FieldBuffer::new(num_0_half.log_len(), num_0_half.as_ref().into());
				let num_1 = FieldBuffer::new(num_1_half.log_len(), num_1_half.as_ref().into());
				let den_0 = den_0.try_into().expect("den_0 length matches batch size");
				let den_1 = den_1.try_into().expect("den_1 length matches batch size");

				let input = SharedFracAddInput::CommonNumerator {
					num_0,
					num_1,
					den_0,
					den_1,
				};

				Ok(SharedFracAddLastLayerProver::new(input, eval_point, num_evals, den_evals)?)
			}
			LastLayerSharing::CommonDenominator => {
				let mut shared_den = None;
				let mut num_0 = Vec::with_capacity(N);
				let mut num_1 = Vec::with_capacity(N);

				for (idx, prover) in self.provers.into_iter().enumerate() {
					let (layer, remaining) = prover.pop_layer();
					if remaining.is_some() {
						return Err(Error::BatchLayerCountMismatch);
					}
					let (num, den) = layer;
					if idx == 0 {
						shared_den = Some(den);
					}

					let (num_0_half, num_1_half) = num.split_half_ref();
					num_0.push(FieldBuffer::new(num_0_half.log_len(), num_0_half.as_ref().into()));
					num_1.push(FieldBuffer::new(num_1_half.log_len(), num_1_half.as_ref().into()));
				}

				let shared_den = shared_den.expect("batch size > 0");
				let (den_0_half, den_1_half) = shared_den.split_half_ref();
				let den_0 = FieldBuffer::new(den_0_half.log_len(), den_0_half.as_ref().into());
				let den_1 = FieldBuffer::new(den_1_half.log_len(), den_1_half.as_ref().into());
				let num_0 = num_0.try_into().expect("num_0 length matches batch size");
				let num_1 = num_1.try_into().expect("num_1 length matches batch size");

				let input = SharedFracAddInput::CommonDenominator {
					den_0,
					den_1,
					num_0,
					num_1,
				};

				Ok(SharedFracAddLastLayerProver::new(input, eval_point, num_evals, den_evals)?)
			}
		}
	}

	/// Runs the fractional addition check protocol over a batch of claims.
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

			if n_layers == 1 {
				if let Some(sharing) = prover.last_layer_sharing {
					if prover.shared_last_layer.is_none() {
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
			}

			let (sumcheck_provers, remaining) = prover.layer_provers(claims)?;
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

		let (batch_prover, sums) = BatchFracAddCheckProver::<P, N>::new_with_last_layer_sharing(
			k,
			witnesses,
			LastLayerSharing::CommonNumerator,
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
		let witnesses: [FractionalBuffer<P>; N] = array::from_fn(|_| {
			let den = random_field_buffer::<P>(&mut rng, n + k);
			(shared_num.clone(), den)
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
	fn test_frac_add_check_shared_last_layer_common_denominator() {
		type P = Packed128b;
		type F = <P as PackedField>::Scalar;
		const N: usize = 3;

		let mut rng = StdRng::seed_from_u64(0);
		let n = 2;
		let k = 1;

		let shared_den = random_field_buffer::<P>(&mut rng, n + k);
		let witnesses: [FractionalBuffer<P>; N] = array::from_fn(|_| {
			let num = random_field_buffer::<P>(&mut rng, n + k);
			(num, shared_den.clone())
		});
		let witnesses_clone = witnesses.clone();

		let (batch_prover, sums) = BatchFracAddCheckProver::<P, N>::new_with_last_layer_sharing(
			k,
			witnesses,
			LastLayerSharing::CommonDenominator,
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

		for (output, (num, den)) in verifier_output.iter().zip(witnesses_clone.iter()) {
			let expected_num = evaluate(num, &output.point);
			let expected_den = evaluate(den, &output.point);
			assert_eq!(output.num_eval, expected_num);
			assert_eq!(output.den_eval, expected_den);
		}
	}

	#[test]
	fn test_frac_add_check_layer_computation() {
		test_frac_add_check_layer_computation_helper::<Packed128b>(4, 3);
	}
}
