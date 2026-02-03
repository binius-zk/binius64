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
	logup::{LogUp, helper::generate_index_fingerprints},
};
type BatchLogSumClaims<F, const N: usize> = [FracAddEvalClaim<F>; N];

impl<
	P: PackedField<Scalar = F>,
	Channel: IOPProverChannel<P>,
	F: Field,
	const N_TABLES: usize,
	const N_LOOKUPS: usize,
> LogUp<P, Channel, N_TABLES, N_LOOKUPS>
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

	fn common_denominator(
		log_len: usize,
		index_count: usize,
		fingerprint_scalar: F,
		shift_scalar: F,
	) -> FieldBuffer<P> {
		// Build a fingerprinted table for indices 0..index_count-1.
		// This is the shared denominator for all pushforward fractions.
		let index_range = (0..index_count).collect::<Vec<_>>();
		let [common_denominator] = generate_index_fingerprints::<P, F, 1, 1>(
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
	/// 2. `push_forward / common_denominator`, where the denominator is the fingerprint of indices
	///    `0..len(push_forward)`.
	///
	/// Returns the top-layer fractional-sum claims for verifier consumption.
	pub fn prove_log_sum(
		&self,
		channel: &mut Channel,
	) -> Result<(BatchLogSumClaims<F, N_TABLES>, BatchLogSumClaims<F, N_TABLES>), Error> {
		let eq_log_len = self.eq_kernel.log_len();

		assert!(eq_log_len == self.fingerprinted_indexes[0].log_len());

		assert!(
			self.fingerprinted_indexes
				.iter()
				.map(FieldBuffer::log_len)
				.all_equal()
		);

		let eq_witness = SharedLastLayer::CommonNumerator {
			den: self.fingerprinted_indexes.clone(),
			num: self.eq_kernel.clone(),
		};
		// eq-kernel numerator is shared; denominators are per-lookup fingerprints.
		let (eq_prover, eq_sums) =
			BatchFracAddCheckProver::<P, N_TABLES>::new_with_last_layer_sharing(
				eq_log_len, eq_witness,
			);
		let eq_claims = Self::tree_sums_to_claims(eq_sums);

		let common_denominator = Self::common_denominator(
			eq_log_len,
			self.push_forwards[0].len(),
			self.fingerprint_scalar,
			self.shift_scalar,
		);

		let push_witnesses = SharedLastLayer::CommonDenominator {
			den: common_denominator,
			num: self.push_forwards.clone(),
		};
		// Pushforward denominators are shared; numerators are per-lookup pushforwards.

		let (push_prover, push_sums) =
			BatchFracAddCheckProver::<P, N_TABLES>::new_with_last_layer_sharing(
				eq_log_len,
				push_witnesses,
			);
		let push_claims = Self::tree_sums_to_claims(push_sums);

		// Combine eq_claims and push_claims into LogSumClaim objects and write to transcript.
		zip(eq_claims.iter(), push_claims.iter()).for_each(|(eq, push)| {
			channel.send_many(&[eq.num_eval, eq.den_eval, push.num_eval, push.den_eval])
		});

		let eq_claims = eq_prover.prove(eq_claims.clone(), channel)?;
		let push_claims = push_prover.prove(push_claims.clone(), channel)?;

		Ok((eq_claims, push_claims))
	}
}
