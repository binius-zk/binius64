use crate::protocols::fracaddcheck::BatchFracAddCheckProver;
use crate::protocols::logup::helper::generate_index_fingerprints;
use crate::protocols::logup::prover::LogUp;
use binius_field::{Field, PackedField};
use binius_math::FieldBuffer;
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_verifier::protocols::fracaddcheck::FracAddEvalClaim;

impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
	/// Proves the log-sum instance using batched fractional-addition trees.
	///
	/// Two batches are produced:
	/// 1. `eq_kernel / (fingerprinted_index - shift)` for each lookup.
	/// 2. `push_forward / common_denominator`, where the denominator is the
	///    fingerprint of indices `0..len(push_forward)`.
	pub fn prove_log_sum<Challenger_: Challenger>(
		&self,
		transcript: &mut ProverTranscript<Challenger_>,
	) {
		let shift: F = transcript.sample();

		let eq_log_len = self.eq_kernel.log_len();
		let mut eq_witnesses = Vec::with_capacity(N_LOOKUPS);
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

			eq_witnesses.push((self.eq_kernel.clone(), denom));
		}

		let (eq_prover, eq_sums) = BatchFracAddCheckProver::<P>::new(eq_log_len, eq_witnesses);

		let eq_claims = eq_sums
			.into_iter()
			.map(|(num, den)| FracAddEvalClaim {
				num_eval: num.get(0),
				den_eval: den.get(0),
				point: base_point.clone(),
			})
			.collect();
		eq_prover
			.prove(eq_claims, transcript)
			.expect("batched fractional-add prover should succeed");

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

		let mut push_witnesses = Vec::with_capacity(N_LOOKUPS);

		for i in 0..N_LOOKUPS {
			push_witnesses.push((self.push_forwards[i].clone(), common_denominator.clone()));
		}

		let (push_prover, push_sums) =
			BatchFracAddCheckProver::<P>::new(pushforward_log_len, push_witnesses);

		let push_claims = push_sums
			.into_iter()
			.map(|(num, den)| FracAddEvalClaim {
				num_eval: num.get(0),
				den_eval: den.get(0),
				point: base_point.clone(),
			})
			.collect();
		push_prover
			.prove(push_claims, transcript)
			.expect("batched fractional-add prover should succeed");
	}
}
