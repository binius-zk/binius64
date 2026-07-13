//! W2 — merged NON-ZK batched BaseFold opening of the two discharge oracles
//! (M_VK, `n_d + 2` message vars; M_D, `n_d` vars) in ONE combined FRI.
//!
//! ## Upstream channel port
//!
//! This is now a THIN adapter over upstream's native combined opener
//! (`binius_iop_prover::basefold::prove_mlecheck_basefold` /
//! `binius_iop::basefold::verify_mlecheck_basefold`, #1500/#1586), replacing the
//! ~330-LoC hand-vendored FRI opener the pinned rev needed when
//! `verify_mlecheck_basefold_zk_batch` still hard-asserted `max_log_batch_size == 1`
//! (finding (iii), now GRANTED). The vendored `InteriorPaddedSumcheckProver`,
//! `build_combined_witness`, the pre-#1611 first-fold challenge ordering, and the
//! pre-#1693 raw-transcript FRI loop are all DELETED.
//!
//! The reduction + combine mirror upstream's `prove_batch_zk_basefold` /
//! `verify_batch_zk_basefold` (`crates/iop{,-prover}/src/basefold_channel.rs`),
//! specialized to our case: both oracles are NON-ZK (no masking, `batch_challenge =
//! None`), there are exactly two of them, and each opening claim is a point-evaluation
//! (transparent = `eq(·, point)`). Crucially, the codeword COMMITMENTS are passed in
//! PINNED (M_VK's is the audited `vk_digest`; M_D's is the pre-`tau`-observed
//! `digest_D`) rather than being read fresh through the channel — the verifier binds the
//! FRI queries to exactly those digests. This is what preserves the STEP-2 vkey
//! semantics (the M_VK table is fixed before the Phase-C challenges) that a naive
//! `recv_oracle`-at-opening-time channel flow would break.
//!
//! Construction:
//! * Reduction: one batched degree-2 sumcheck over `𝐧 = n_d + 2` vars with two claims —
//!   `[M_VK · eq(vk_point)]` (native `𝐧` vars) and `[M_D · eq(sigma)]` top-padded by
//!   `𝐧 − n_d = 2` `eq(0, ·)` rounds (`PaddedSumcheckDecorator`). Reduces both to a
//!   shared point `r` with per-oracle evals `α_vk = M̃_VK(r)`, `α_d = M̃_D(r[..n_d])`.
//! * Combine: sample the `log2(2) = 1` outer challenge `r'`; build the piecewise
//!   combined multilinear `𝛑 = Σ_i e[i]·π_i^↑` (each oracle placed at the low `2^{n_i}`
//!   of every `2^{n_i+log_lift}` lift block and repeated across the high dims, exactly as
//!   the FRI lift/repeat structure), and the target `s' = Σ_i e[i]·α_i·eq0(r[n_i..][..lift])`.
//! * One degree-1 MLE-check on `𝛑` at `r` interleaved with the combined FRI over BOTH
//!   pinned codewords (`prove_mlecheck_basefold` / `verify_mlecheck_basefold`).

use anyhow::ensure;
use binius_field::{Field, PackedField};
use binius_iop::{
	basefold::{mlecheck_fri_consistency, verify_mlecheck_basefold},
	fri::FRIParams,
	merkle_channel::MerkleIPVerifierChannel,
};
use binius_ip::sumcheck as ip_sumcheck;
use binius_ip_prover::sumcheck::{
	self, PaddedSumcheckDecorator, bivariate_product::BivariateProductSumcheckProver,
};
use binius_iop_prover::{
	basefold::prove_mlecheck_basefold, fri::FRIFoldProver, merkle_channel::MerkleIPProverChannel,
};
use binius_math::{
	FieldBuffer, FieldSlice, FieldSliceMut,
	multilinear::eq::{
		eq_ind, eq_ind_partial_eval, eq_ind_partial_eval_scalars, eq_ind_zero,
	},
	ntt::AdditiveNTT,
	univariate::evaluate_univariate,
};
use binius_utils::rayon::prelude::*;
use binius_verifier::config::B128;

/// One point-evaluation opening claim against a committed oracle: `M̃(point) = eval`.
pub struct OpenClaim<'a, P: PackedField<Scalar = B128>> {
	/// The committed multilinear (message buffer), low-coordinate-first.
	pub message: FieldBuffer<P>,
	/// The evaluation point (low-first).
	pub point: &'a [B128],
	/// The claimed evaluation.
	pub eval: B128,
}

/// Accumulate `dst += scalar · src` over a lift/repeat block (mirrors the private
/// `accumulate_scaled_buffer` of `iop-prover::basefold_channel`).
fn accumulate_scaled_buffer<P: PackedField>(
	mut dst: FieldSliceMut<P>,
	src: FieldSlice<P>,
	scalar_broadcast: P,
) {
	if src.log_len() >= P::LOG_WIDTH {
		let src = src.as_ref();
		dst.as_mut()
			.par_iter_mut()
			.zip(src.as_ref())
			.for_each(|(dst_i, src_i)| {
				*dst_i += scalar_broadcast * *src_i;
			});
	} else {
		let src = P::from_scalars(src.iter_scalars());
		dst.as_mut()[0] += scalar_broadcast * src;
	}
}

/// Prover: merged opening of the two oracles [M_VK (index 0), M_D (index 1)] under one
/// batched `FRIParams`. `committed_codewords` are in `params.input_oracles()` order and
/// carry the commitment handles produced by `channel.send_merkle_commitment` (M_VK's root
/// == the audited vk_digest; M_D's == the pre-`tau` digest_D).
///
/// ## Preconditions
/// * `params.input_oracles().len() == 2`, both NON-ZK.
/// * `claims[0]` is the larger oracle (message vars == `n_d + 2`), `claims[1]` the smaller.
pub fn prove_merged_openings<P, NTT, Channel>(
	params: &FRIParams<B128>,
	ntt: &NTT,
	channel: &mut Channel,
	claims: [OpenClaim<'_, P>; 2],
	committed_codewords: Vec<(FieldBuffer<P>, Channel::Commitment)>,
) -> anyhow::Result<()>
where
	P: PackedField<Scalar = B128>,
	NTT: AdditiveNTT<Field = B128> + Sync,
	Channel: MerkleIPProverChannel<B128>,
{
	let input_oracles = params.input_oracles();
	ensure!(input_oracles.len() == 2, "expected exactly two input oracles");
	ensure!(committed_codewords.len() == 2, "expected exactly two committed codewords");
	let max_n = claims.iter().map(|c| c.message.log_len()).max().expect("two claims");
	ensure!(claims[0].message.log_len() == max_n, "claims[0] must be the larger oracle");

	// ---- Reduction: one batched degree-2 sumcheck over max_n vars (no masking). ----
	let mut witness_primes: [Option<FieldBuffer<P>>; 2] = [None, None];
	let mut provers = Vec::with_capacity(2);
	for (i, claim) in claims.iter().enumerate() {
		let n_i = claim.message.log_len();
		ensure!(claim.point.len() == n_i, "claim {i}: point arity mismatch");
		let transparent = eq_ind_partial_eval::<P>(claim.point);
		let inner = BivariateProductSumcheckProver::new([claim.message.clone(), transparent], claim.eval);
		provers.push(PaddedSumcheckDecorator::new(inner, max_n - n_i));
		witness_primes[i] = Some(claim.message.clone());
	}
	let output = sumcheck::batch_prove(provers, channel);
	// batch_prove returns challenges already reversed to low-to-high.
	let point = output.challenges.clone();
	ensure!(point.len() == max_n, "reduction round count");
	let alphas = [output.multilinear_evals[0][0], output.multilinear_evals[1][0]];
	channel.send_many(&alphas);

	// ---- Combine: one outer challenge, piecewise combined witness + target. ----
	let outer_challenges = channel.sample_many(1); // log2(2 oracles) = 1
	let eq_tensor = eq_ind_partial_eval_scalars::<B128>(&outer_challenges);
	let mut combined = FieldBuffer::<P>::zeros(max_n);
	let mut s_prime = B128::ZERO;
	for (fri_oracle, wp, &eq_i, &alpha_i) in itertools::izip!(
		input_oracles,
		witness_primes,
		eq_tensor.iter(),
		alphas.iter(),
	) {
		let wp = wp.expect("each oracle carries a claim");
		let n_i = wp.log_len();
		let log_lift = fri_oracle.log_lift;
		// Repeat placement: add eq_i·π_i into the low 2^{n_i} of each 2^{n_i+log_lift} block.
		assert!(n_i + log_lift >= P::LOG_WIDTH, "repeat placement requires whole-packed lift blocks");
		let scalar_broadcast = P::broadcast(eq_i);
		let chunk_packed = 1usize << (n_i + log_lift - P::LOG_WIDTH);
		combined
			.as_mut()
			.par_chunks_mut(chunk_packed)
			.for_each(|chunk| {
				let chunk_buf = FieldSliceMut::from_slice(n_i + log_lift, chunk);
				accumulate_scaled_buffer(chunk_buf, wp.to_ref(), scalar_broadcast);
			});
		s_prime += eq_i * alpha_i * eq_ind_zero(&point[n_i..][..log_lift]);
	}

	// ---- One MLE-check on the combined witness at r, interleaved with the combined FRI. ----
	let fri_folder = FRIFoldProver::new_batch(params, ntt, committed_codewords);
	prove_mlecheck_basefold(combined, &point, s_prime, None, &outer_challenges, fri_folder, channel);
	Ok(())
}

/// Verifier: merged opening of the two oracles against `params` and the two PINNED
/// commitments (`commitments[0]` == M_VK bound to the audited vk_digest; `commitments[1]`
/// == M_D bound to the pre-`tau` digest_D). `claims[i].point`/`.eval` must already be
/// Fiat-Shamir-bound by the caller.
pub fn verify_merged_openings<Channel>(
	params: &FRIParams<B128>,
	channel: &mut Channel,
	commitments: &[Channel::Commitment; 2],
	claims: [(&[B128], B128); 2],
) -> anyhow::Result<()>
where
	Channel: MerkleIPVerifierChannel<B128, Elem = B128>,
{
	let input_oracles = params.input_oracles();
	ensure!(input_oracles.len() == 2, "expected exactly two input oracles");
	let max_n = claims.iter().map(|(p, _)| p.len()).max().expect("two claims");
	ensure!(claims[0].0.len() == max_n, "claims[0] must be the larger oracle");

	// ---- Reduction. ----
	let sums = [claims[0].1, claims[1].1];
	let out = ip_sumcheck::batch_verify::<B128, _>(max_n, 2, &sums, channel)
		.map_err(|e| anyhow::anyhow!("reduction batch_verify: {e}"))?;
	let mu = out.batch_coeff;
	let reduced = out.eval;
	let mut point = out.challenges;
	point.reverse(); // low-first

	let alphas = channel
		.recv_array::<2>()
		.map_err(|e| anyhow::anyhow!("reduction alphas: {e}"))?;

	// Recombination: each oracle contributes α_i · eq(point_i, r[..n_i]) · eq(0, r[n_i..]).
	let contributions: Vec<B128> = claims
		.iter()
		.zip(alphas.iter())
		.map(|((pt, _), &alpha)| {
			let n_i = pt.len();
			alpha * eq_ind(pt, &point[..n_i]) * eq_ind_zero(&point[n_i..])
		})
		.collect();
	let expected = evaluate_univariate(&contributions, mu);
	ensure!(
		reduced == expected,
		"reduction recombination failed (the point-evaluation claims are inconsistent)"
	);

	// ---- Combine target. ----
	let outer_challenges = channel.sample_many(1);
	let eq_tensor = eq_ind_partial_eval_scalars::<B128>(&outer_challenges);
	let s_prime: B128 = itertools::izip!(input_oracles, claims.iter(), eq_tensor.iter(), alphas.iter())
		.map(|(fri_oracle, (pt, _), &eq_i, &alpha)| {
			let n_i = pt.len();
			eq_i * alpha * eq_ind_zero(&point[n_i..][..fri_oracle.log_lift])
		})
		.sum();

	// ---- Combined MLE-check + FRI over both pinned codewords. ----
	let reduced_out = verify_mlecheck_basefold(
		params,
		commitments,
		s_prime,
		&point,
		None,
		&outer_challenges,
		channel,
	)
	.map_err(|e| anyhow::anyhow!("FRI query/Merkle verification: {e}"))?;
	ensure!(
		mlecheck_fri_consistency(reduced_out.final_fri_value, reduced_out.final_sumcheck_value),
		"FRI/MLE-check inconsistency"
	);
	Ok(())
}
