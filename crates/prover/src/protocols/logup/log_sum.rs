// Copyright 2025-2026 The Binius Developers
use std::{array, iter::zip};

use binius_field::{Field, PackedField};
use binius_iop_prover::channel::IOPProverChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::FieldBuffer;
use binius_verifier::protocols::fracaddcheck::FracAddEvalClaim;
use itertools::Itertools;

use crate::protocols::{
	fracaddcheck::{BatchFracAddCheckProver, Error, SharedLastLayer},
	logup::{LogUp, helper::generate_enumeration_fingerprint},
};
type BatchLogSumClaims<F, const N: usize> = [FracAddEvalClaim<F>; N];

impl<P: PackedField<Scalar = F>, Channel: IOPProverChannel<P>, F: Field, const N_TABLES: usize>
	LogUp<P, Channel, N_TABLES>
{
	/// Converts the top layer of each frac-add tree into evaluation claims.
	///
	/// The top layer is a single scalar per numerator/denominator, so the
	/// evaluation point is empty and the claim values are taken directly.
	fn tree_sums_to_claims<const N: usize>(
		sums: [(FieldBuffer<P>, FieldBuffer<P>); N],
	) -> [FracAddEvalClaim<F>; N] {
		array::from_fn(|i| {
			let (num, den) = &sums[i];
			FracAddEvalClaim {
				num_eval: num.get(0),
				den_eval: den.get(0),
				point: Vec::new(),
			}
		})
	}

	/// Proves the log-sum instance using batched fractional-addition trees.
	///
	/// Two batches are produced:
	/// 1. `eq_kernel / (fingerprinted_index - shift)` for each lookup.
	/// 2. `push_forward / common_denominator`, where the denominator is the fingerprint of indices
	///    `0..len(push_forward)`.
	///
	/// Returns the top-layer fractional-sum claims for verifier consumption.
	pub fn prove_log_sum(
		&self,
		channel: &mut Channel,
	) -> Result<(BatchLogSumClaims<F, N_TABLES>, BatchLogSumClaims<F, N_TABLES>), Error> {
		let eq_log_len = self.eq_kernel.log_len();

		// Every per-table fingerprint vector must be aligned with the eq-kernel domain.
		assert!(eq_log_len == self.fingerprinted_indexes[0].log_len());

		assert!(
			self.fingerprinted_indexes
				.iter()
				.map(FieldBuffer::log_len)
				.all_equal()
		);

		// Batch 1 witness: same eq-kernel numerator for all tables, table-specific denominators.
		let eq_witness = SharedLastLayer::CommonNumerator {
			den: self.fingerprinted_indexes.clone(),
			num: self.eq_kernel.clone(),
		};

		// Batch 2 denominator fingerprints the canonical index sequence 0..2^eq_log_len.
		let common_denominator = generate_enumeration_fingerprint(
			eq_log_len,
			self.fingerprint_scalar,
			self.shift_scalar,
		);

		// Batch 2 witness: shared denominator, table-specific pushforward numerators.
		let push_witnesses = SharedLastLayer::CommonDenominator {
			den: common_denominator,
			num: self.push_forwards.clone(),
		};

		// eq-kernel numerator is shared; denominators are per-table fingerprints of indexes.
		let (eq_prover, eq_sums) =
			BatchFracAddCheckProver::<P, N_TABLES>::new_with_last_layer_sharing(
				eq_log_len, eq_witness,
			);
		// Pushforward denominators are shared; numerators are per-table pushforwards.
		let (push_prover, push_sums) =
			BatchFracAddCheckProver::<P, N_TABLES>::new_with_last_layer_sharing(
				eq_log_len,
				push_witnesses,
			);

		let index_log_sum_claims = Self::tree_sums_to_claims(eq_sums);
		let pushforward_log_sum_claims = Self::tree_sums_to_claims(push_sums);

		// Transcript order is part of the protocol: (eq_num, eq_den, push_num, push_den) per table.
		zip(index_log_sum_claims.iter(), pushforward_log_sum_claims.iter()).for_each(
			|(index_claim, pushforward_claim)| {
				channel.send_many(&[
					index_claim.num_eval,
					index_claim.den_eval,
					pushforward_claim.num_eval,
					pushforward_claim.den_eval,
				])
			},
		);

		// Prove each batch against the just-emitted top-layer claims.
		let index_log_sum_claims = eq_prover.prove(index_log_sum_claims.clone(), channel)?;
		let pushforward_log_sum_claims =
			push_prover.prove(pushforward_log_sum_claims.clone(), channel)?;

		Ok((index_log_sum_claims, pushforward_log_sum_claims))
	}
}
