use crate::protocols::logup::helper::generate_index_fingerprints;
use crate::protocols::{fracaddcheck::BatchFracAddCheckProver, logup::LogUp};
use binius_field::{Field, PackedField};
use binius_math::FieldBuffer;
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_verifier::protocols::fracaddcheck::FracAddEvalClaim;
use itertools::Itertools;

impl<P: PackedField<Scalar = F>, F: Field, const N_TABLES: usize, const N_LOOKUPS: usize>
	LogUp<P, N_TABLES, N_LOOKUPS>
{
	/// Converts the top layer of each frac-add tree into evaluation claims.
	fn tree_sums_to_claims(
		sums: Vec<(FieldBuffer<P>, FieldBuffer<P>)>,
	) -> Vec<FracAddEvalClaim<F>> {
		sums.into_iter()
			.map(|(num, den)| FracAddEvalClaim {
				num_eval: num.get(0),
				den_eval: den.get(0),
				point: Vec::new(),
			})
			.collect()
	}

	fn common_denominator(
		log_len: usize,
		index_count: usize,
		fingerprint_scalar: F,
		shift_scalar: F,
	) -> FieldBuffer<P> {
		// Build a fingerprinted table for indices 0..index_count-1.
		let index_range = (0..index_count).collect::<Vec<_>>();
		let [common_denominator] = generate_index_fingerprints::<P, F, 1>(
			[index_range.as_slice()],
			fingerprint_scalar,
			shift_scalar,
			log_len,
		);
		common_denominator
	}

	/// Proves the log-sum instance using batched fractional-addition trees.
	///
	/// Two batches are produced:
	/// 1. `eq_kernel / (fingerprinted_index - shift)` for each lookup.
	/// 2. `push_forward / common_denominator`, where the denominator is the
	///    fingerprint of indices `0..len(push_forward)`.
	///
	/// Returns the top-layer fractional-sum claims for verifier consumption.
	pub fn prove_log_sum<Challenger_: Challenger>(
		&self,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<
		(Vec<FracAddEvalClaim<F>>, Vec<FracAddEvalClaim<F>>),
		crate::protocols::fracaddcheck::Error,
	> {
		let eq_log_len = self.eq_kernel.log_len();

		assert!(eq_log_len == self.fingerprinted_indexes[0].log_len());

		assert!(
			self.fingerprinted_indexes
				.iter()
				.map(FieldBuffer::log_len)
				.all_equal()
		);

		let eq_witnesses = self
			.fingerprinted_indexes
			.iter()
			.map(|idx| (self.eq_kernel.clone(), idx.clone()))
			.collect::<Vec<_>>();

		let (eq_prover, eq_sums) = BatchFracAddCheckProver::<P>::new(eq_log_len, eq_witnesses);
		let eq_claims = Self::tree_sums_to_claims(eq_sums);
		eq_prover.prove(eq_claims.clone(), transcript)?;

		let common_denominator = Self::common_denominator(
			eq_log_len,
			self.push_forwards[0].len(),
			self.fingerprint_scalar,
			self.shift_scalar,
		);

		let push_witnesses = self
			.push_forwards
			.iter()
			.cloned()
			.map(|push_forward| (push_forward, common_denominator.clone()))
			.collect();

		let (push_prover, push_sums) =
			BatchFracAddCheckProver::<P>::new(eq_log_len, push_witnesses);
		let push_claims = Self::tree_sums_to_claims(push_sums);

		push_prover.prove(push_claims.clone(), transcript)?;

		Ok((eq_claims, push_claims))
	}
}
