use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, line::extrapolate_line_packed, multilinear::eq};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_verifier::protocols::prodcheck::MultilinearEvalClaim;
use itertools::Itertools;
use std::{array, iter::chain};

use crate::protocols::fracaddcheck::FracAddCheckProver;
use crate::protocols::sumcheck::{
	Error as SumcheckError, batch::BatchSumcheckOutput,
	batch_quadratic::BatchQuadraticSumcheckProver,
};
use crate::protocols::{
	logup::helper::{generate_index_fingerprints, generate_pushforward},
	sumcheck::batch::{batch_prove_and_write_evals, batch_prove_mle_and_write_evals},
};

/// This struct enscapsulates logic required by the prover for the LogUp* indexed lookup arguement.
/// It operates in the batch mode by default. Supports N_LOOKUPS into N_TABLES.
pub struct LogUp<P: PackedField, const N_TABLES: usize, const N_LOOKUPS: usize> {
	fingerprinted_indexes: [FieldBuffer<P>; N_LOOKUPS],
	table_ids: [usize; N_LOOKUPS],
	push_forwards: [FieldBuffer<P>; N_LOOKUPS],
	tables: [FieldBuffer<P>; N_TABLES],
	eval_point: Vec<P::Scalar>,
	eq_kernel: FieldBuffer<P>,
	lookup_evals: [P::Scalar; N_LOOKUPS],
	fingerprint_scalar: P::Scalar,
}

#[derive(Debug, Clone)]
pub struct PushforwardEvalClaims<F: Field> {
	pub challenges: Vec<F>,
	pub pushforward_evals: Vec<F>,
	pub table_evals: Vec<F>,
}

/// We assume the bits for each index has been committed as a separate MLE over the base field.
impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
	pub fn new<Challenger_: Challenger>(
		indexes: [&[usize]; N_LOOKUPS],
		table_ids: [usize; N_LOOKUPS],
		eval_point: &[P::Scalar],
		lookup_evals: [F; N_LOOKUPS],
		tables: [FieldBuffer<P>; N_TABLES],
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Self {
		assert!(N_TABLES > 0 && N_LOOKUPS > 0);
		let eq_kernel = eq::eq_ind_partial_eval::<P>(eval_point);
		let push_forwards = build_pushforwards(&indexes, &table_ids, &eq_kernel, &tables);
		let max_log_len = tables
			.iter()
			.map(|table| table.log_len())
			.max()
			.expect("There will be atleast 1 table");
		let fingerprint_scalar = transcript.sample();
		let indexes = generate_index_fingerprints(indexes, fingerprint_scalar, max_log_len);

		LogUp {
			fingerprinted_indexes: indexes,
			table_ids,
			push_forwards,
			tables,
			eval_point: eval_point.to_vec(),
			eq_kernel,
			fingerprint_scalar,
			lookup_evals,
		}
	}

	/// Proves the outer instance, which reduces the evaluation claim on the lookup values, to that on the pushforward.
	pub fn prove_pushforward<
		Challenger_: Challenger,
		// N_MLES is the total number of MLEs involved, this is precisely N_LOOKUPS + N_TABLES.
		const N_MLES: usize,
	>(
		&self,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<PushforwardEvalClaims<F>, SumcheckError> {
		// TODO: Remove implicit assumption of equal table size.
		assert_eq!(N_TABLES + N_LOOKUPS, N_MLES);
		let prover = make_pushforward_sumcheck_prover::<P, F, N_TABLES, N_LOOKUPS, N_MLES>(
			&self.table_ids,
			&self.tables,
			&self.push_forwards,
			self.lookup_evals,
		)?;

		let BatchSumcheckOutput {
			challenges,
			multilinear_evals,
		} = batch_prove_and_write_evals(vec![prover], transcript)?;

		let (pushforward_evals, table_evals) = multilinear_evals[0].split_at(N_LOOKUPS);

		Ok(PushforwardEvalClaims {
			challenges,
			pushforward_evals: pushforward_evals.to_vec(),
			table_evals: table_evals.to_vec(),
		})
	}

	/// Proves the inner instance which is reminiscient of logup gkr, using a binary tree of fractional additions.
	pub fn prove_log_sum<Challenger_: Challenger>(
		&self,
		transcript: &mut ProverTranscript<Challenger_>,
	) {
		let shift: F = transcript.sample();

		let eq_log_len = self.eq_kernel.log_len();
		let mut eq_provers = Vec::with_capacity(N_LOOKUPS);
		let mut eq_claims = Vec::with_capacity(N_LOOKUPS);
		let base_point = Vec::new();

		for i in 0..N_LOOKUPS {
			assert_eq!(
				self.fingerprinted_indexes[i].log_len(),
				eq_log_len,
				"fingerprinted index length must match eq kernel length"
			);

			let den_values = self.fingerprinted_indexes[i]
				.iter_scalars()
				.map(|value| value - shift)
				.collect::<Vec<_>>();
			let denom = FieldBuffer::from_values(&den_values);

			let (prover, sums) =
				FracAddCheckProver::<P>::new(eq_log_len, (self.eq_kernel.clone(), denom));
			assert_eq!(
				sums.0.log_len(),
				0,
				"fractional-add reduction should fully collapse the layer"
			);

			let num_eval = sums.0.get(0);
			let den_eval = sums.1.get(0);
			let point = base_point.clone();
			eq_claims.push((
				MultilinearEvalClaim {
					eval: num_eval,
					point: point.clone(),
				},
				MultilinearEvalClaim {
					eval: den_eval,
					point,
				},
			));
			eq_provers.push(prover);
		}

		prove_frac_add_batch(eq_provers, eq_claims, transcript);

		let max_log_len = self
			.tables
			.iter()
			.map(|table| table.log_len())
			.max()
			.expect("there is at least one table");
		let pushforward_log_len = self.push_forwards[0].log_len();
		for pushforward in &self.push_forwards[1..] {
			assert_eq!(
				pushforward.log_len(),
				pushforward_log_len,
				"pushforward lengths must match across the batch"
			);
		}

		let index_count = self.push_forwards[0].len();
		let index_range = (0..index_count).collect::<Vec<_>>();
		let [common_denominator] = generate_index_fingerprints::<P, F, 1>(
			[index_range.as_slice()],
			self.fingerprint_scalar,
			max_log_len,
		);

		let mut push_provers = Vec::with_capacity(N_LOOKUPS);
		let mut push_claims = Vec::with_capacity(N_LOOKUPS);

		for i in 0..N_LOOKUPS {
			let (prover, sums) = FracAddCheckProver::<P>::new(
				pushforward_log_len,
				(self.push_forwards[i].clone(), common_denominator.clone()),
			);
			assert_eq!(
				sums.0.log_len(),
				0,
				"fractional-add reduction should fully collapse the layer"
			);

			let num_eval = sums.0.get(0);
			let den_eval = sums.1.get(0);
			let point = base_point.clone();
			push_claims.push((
				MultilinearEvalClaim {
					eval: num_eval,
					point: point.clone(),
				},
				MultilinearEvalClaim {
					eval: den_eval,
					point,
				},
			));
			push_provers.push(prover);
		}

		prove_frac_add_batch(push_provers, push_claims, transcript);
	}
}

fn prove_frac_add_batch<P, F, Challenger_>(
	mut provers: Vec<FracAddCheckProver<P>>,
	mut claims: Vec<(MultilinearEvalClaim<F>, MultilinearEvalClaim<F>)>,
	transcript: &mut ProverTranscript<Challenger_>,
) where
	P: PackedField<Scalar = F>,
	F: Field,
	Challenger_: Challenger,
{
	if provers.is_empty() {
		return;
	}
	assert_eq!(
		provers.len(),
		claims.len(),
		"fractional-add prover/claim count mismatch"
	);

	let n_layers = provers
		.first()
		.expect("non-empty provers")
		.n_layers();
	assert!(
		provers.iter().all(|prover| prover.n_layers() == n_layers),
		"all fractional-add provers must have the same number of layers"
	);

	for round in 0..n_layers {
		let mut sumcheck_provers = Vec::with_capacity(provers.len());
		let mut next_provers = Vec::with_capacity(provers.len());

		for (prover, claim) in provers.into_iter().zip(claims.into_iter()) {
			let (layer_prover, remaining) = prover
				.layer_prover(claim)
				.expect("fractional-add layer prover construction should succeed");
			sumcheck_provers.push(layer_prover);

			if round + 1 < n_layers {
				next_provers.push(
					remaining.expect("fractional-add prover should have remaining layers"),
				);
			} else {
				assert!(
					remaining.is_none(),
					"fractional-add prover should be exhausted after the last layer"
				);
			}
		}

		let output = batch_prove_mle_and_write_evals(sumcheck_provers, transcript)
			.expect("batched fractional-add sumcheck should succeed");

		let r = transcript.sample();
		let mut next_point = output.challenges;
		next_point.push(r);

		let mut next_claims = Vec::with_capacity(output.multilinear_evals.len());
		for evals in output.multilinear_evals {
			let [num_0, num_1, den_0, den_1] = evals
				.try_into()
				.expect("fractional-add prover evaluates four multilinears");
			let next_num = extrapolate_line_packed(num_0, num_1, r);
			let next_den = extrapolate_line_packed(den_0, den_1, r);

			next_claims.push((
				MultilinearEvalClaim {
					eval: next_num,
					point: next_point.clone(),
				},
				MultilinearEvalClaim {
					eval: next_den,
					point: next_point.clone(),
				},
			));
		}

		provers = next_provers;
		claims = next_claims;
	}
}

fn build_pushforwards<P: PackedField, const N_TABLES: usize, const N_LOOKUPS: usize>(
	indexes: &[&[usize]; N_LOOKUPS],
	table_ids: &[usize; N_LOOKUPS],
	eq_kernel: &FieldBuffer<P>,
	tables: &[FieldBuffer<P>; N_TABLES],
) -> [FieldBuffer<P>; N_LOOKUPS] {
	array::from_fn(|i| {
		let (indices, table_id) = (indexes[i], table_ids[i]);
		generate_pushforward(indices, eq_kernel, tables[table_id].len())
	})
}

fn make_pushforward_sumcheck_prover<
	P: PackedField<Scalar = F>,
	F: Field,
	const N_TABLES: usize,
	const N_LOOKUPS: usize,
	const N_MLES: usize,
>(
	table_ids: &[usize; N_LOOKUPS],
	tables: &[FieldBuffer<P>; N_TABLES],
	push_forwards: &[FieldBuffer<P>; N_LOOKUPS],
	lookup_evals: [F; N_LOOKUPS],
) -> Result<
	BatchQuadraticSumcheckProver<
		P,
		impl Fn([P; N_MLES], &mut [P; N_LOOKUPS]),
		impl Fn([P; N_MLES], &mut [P; N_LOOKUPS]),
		N_MLES,
		N_LOOKUPS,
	>,
	SumcheckError,
> {
	assert!(N_TABLES + N_LOOKUPS == N_MLES);
	let mles: [FieldBuffer<P>; N_MLES] =
		chain(push_forwards.iter().cloned(), tables.iter().cloned())
			.collect_array()
			.expect("N_TABLES + N_LOOKUPS == N_MLES");

	let pushforward_composition = |mle_evals: [P; N_MLES], comp_evals: &mut [P; N_LOOKUPS]| {
		let (pushforwards, tables) = mle_evals.split_at(N_LOOKUPS);
		for i in 0..N_LOOKUPS {
			comp_evals[i] = pushforwards[i] * tables[table_ids[i]]
		}
	};
	BatchQuadraticSumcheckProver::new(
		mles,
		pushforward_composition,
		pushforward_composition,
		lookup_evals,
	)
}

#[cfg(test)]
mod tests {
	use super::*;
	use binius_math::{
		inner_product::inner_product_buffers,
		multilinear::evaluate::evaluate,
		test_utils::{Packed128b, random_field_buffer, random_scalars},
		univariate::evaluate_univariate,
	};
	use binius_transcript::{
		ProverTranscript, VerifierTranscript,
		fiat_shamir::CanSample,
	};
	use binius_verifier::{config::StdChallenger, protocols::sumcheck::batch_verify_mle};
	use rand::{Rng, SeedableRng, rngs::StdRng};

	fn verify_frac_add_batch<F: Field, Challenger_: Challenger>(
		n_layers: usize,
		mut claims: Vec<(MultilinearEvalClaim<F>, MultilinearEvalClaim<F>)>,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Vec<(MultilinearEvalClaim<F>, MultilinearEvalClaim<F>)> {
		if n_layers == 0 {
			return claims;
		}
		assert!(!claims.is_empty(), "batch claims must be non-empty");

		for _ in 0..n_layers {
			let eval_point = claims[0].0.point.clone();
			for (num_claim, den_claim) in &claims {
				assert_eq!(num_claim.point, eval_point);
				assert_eq!(den_claim.point, eval_point);
			}

			let evals = claims
				.iter()
				.flat_map(|(num_claim, den_claim)| [num_claim.eval, den_claim.eval])
				.collect::<Vec<_>>();

			let output = batch_verify_mle(&eval_point, 2, &evals, transcript)
				.expect("batch verify should succeed");

			let mut composed_evals = Vec::with_capacity(claims.len() * 2);
			let mut layer_evals = Vec::with_capacity(claims.len());

			for _ in 0..claims.len() {
				let [num_0, num_1, den_0, den_1] = transcript
					.message()
					.read()
					.expect("transcript should contain fractional evals");
				let numerator_eval = num_0 * den_1 + num_1 * den_0;
				let denominator_eval = den_0 * den_1;
				composed_evals.push(numerator_eval);
				composed_evals.push(denominator_eval);
				layer_evals.push((num_0, num_1, den_0, den_1));
			}

			let expected_eval = evaluate_univariate(&composed_evals, output.batch_coeff);
			assert_eq!(
				expected_eval, output.eval,
				"batched eval should match verifier reduction"
			);

			let r = transcript.sample();
			let mut reduced_eval_point = output.challenges;
			reduced_eval_point.reverse();
			reduced_eval_point.push(r);

			let mut next_claims = Vec::with_capacity(claims.len());
			for (num_0, num_1, den_0, den_1) in layer_evals {
				let next_num = extrapolate_line_packed(num_0, num_1, r);
				let next_den = extrapolate_line_packed(den_0, den_1, r);
				next_claims.push((
					MultilinearEvalClaim {
						eval: next_num,
						point: reduced_eval_point.clone(),
					},
					MultilinearEvalClaim {
						eval: next_den,
						point: reduced_eval_point.clone(),
					},
				));
			}

			claims = next_claims;
		}

		claims
	}

	#[test]
	fn test_logup_prove_log_sum_batches() {
		type P = Packed128b;
		type F = <P as PackedField>::Scalar;

		const N_TABLES: usize = 1;
		const N_LOOKUPS: usize = 2;

		let mut rng = StdRng::seed_from_u64(0);
		let table_log_len = 3;
		let eval_point_len = 2;

		let tables = [random_field_buffer::<P>(&mut rng, table_log_len)];
		let eval_point = random_scalars::<F>(&mut rng, eval_point_len);
		let eq_kernel = eq::eq_ind_partial_eval::<P>(&eval_point);

		let table_len = 1 << table_log_len;
		let index_count = 1 << eval_point_len;
		let indices_a = (0..index_count)
			.map(|_| rng.random_range(0..table_len))
			.collect::<Vec<_>>();
		let indices_b = (0..index_count)
			.map(|_| rng.random_range(0..table_len))
			.collect::<Vec<_>>();
		let indexes = [&indices_a[..], &indices_b[..]];
		let table_ids = [0usize; N_LOOKUPS];

		let lookup_evals = std::array::from_fn(|i| {
			let lookup_values = super::super::helper::generate_lookup_values::<P, F>(
				indexes[i],
				&tables[table_ids[i]],
			);
			inner_product_buffers::<F, P, _, _>(&eq_kernel, &lookup_values)
		});

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let logup = LogUp::<P, N_TABLES, N_LOOKUPS>::new(
			indexes,
			table_ids,
			&eval_point,
			lookup_evals,
			tables,
			&mut prover_transcript,
		);

		logup.prove_log_sum(&mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let fingerprint_scalar: F = verifier_transcript.sample();
		assert_eq!(fingerprint_scalar, logup.fingerprint_scalar);
		let shift: F = verifier_transcript.sample();

		let eq_log_len = logup.eq_kernel.log_len();
		let mut eq_claims = Vec::with_capacity(N_LOOKUPS);
		let mut eq_denoms = Vec::with_capacity(N_LOOKUPS);
		for i in 0..N_LOOKUPS {
			let den_values = logup.fingerprinted_indexes[i]
				.iter_scalars()
				.map(|value| value - shift)
				.collect::<Vec<_>>();
			let denom = FieldBuffer::from_values(&den_values);
			let (_prover, sums) =
				FracAddCheckProver::<P>::new(eq_log_len, (logup.eq_kernel.clone(), denom.clone()));

			eq_claims.push((
				MultilinearEvalClaim {
					eval: sums.0.get(0),
					point: Vec::new(),
				},
				MultilinearEvalClaim {
					eval: sums.1.get(0),
					point: Vec::new(),
				},
			));
			eq_denoms.push(denom);
		}

		let eq_final_claims = verify_frac_add_batch(eq_log_len, eq_claims, &mut verifier_transcript);
		let eq_final_point = eq_final_claims[0].0.point.clone();
		for ((num_claim, den_claim), denom) in eq_final_claims.iter().zip(eq_denoms.iter()) {
			assert_eq!(num_claim.point, eq_final_point);
			assert_eq!(den_claim.point, eq_final_point);
			let expected_num = evaluate(&logup.eq_kernel, &eq_final_point);
			let expected_den = evaluate(denom, &eq_final_point);
			assert_eq!(num_claim.eval, expected_num);
			assert_eq!(den_claim.eval, expected_den);
		}

		let max_log_len = logup
			.tables
			.iter()
			.map(|table| table.log_len())
			.max()
			.expect("there is at least one table");
		let pushforward_log_len = logup.push_forwards[0].log_len();
		let index_count = logup.push_forwards[0].len();
		let index_range = (0..index_count).collect::<Vec<_>>();
		let [common_denominator] = generate_index_fingerprints::<P, F, 1>(
			[index_range.as_slice()],
			logup.fingerprint_scalar,
			max_log_len,
		);

		let mut push_claims = Vec::with_capacity(N_LOOKUPS);
		for i in 0..N_LOOKUPS {
			let (_prover, sums) = FracAddCheckProver::<P>::new(
				pushforward_log_len,
				(logup.push_forwards[i].clone(), common_denominator.clone()),
			);
			push_claims.push((
				MultilinearEvalClaim {
					eval: sums.0.get(0),
					point: Vec::new(),
				},
				MultilinearEvalClaim {
					eval: sums.1.get(0),
					point: Vec::new(),
				},
			));
		}

		let push_final_claims =
			verify_frac_add_batch(pushforward_log_len, push_claims, &mut verifier_transcript);
		let push_final_point = push_final_claims[0].0.point.clone();
		for ((num_claim, den_claim), numerator) in
			push_final_claims.iter().zip(logup.push_forwards.iter())
		{
			assert_eq!(num_claim.point, push_final_point);
			assert_eq!(den_claim.point, push_final_point);
			let expected_num = evaluate(numerator, &push_final_point);
			let expected_den = evaluate(&common_denominator, &push_final_point);
			assert_eq!(num_claim.eval, expected_num);
			assert_eq!(den_claim.eval, expected_den);
		}

		verifier_transcript
			.finalize()
			.expect("verifier transcript should be fully consumed");
	}
}
