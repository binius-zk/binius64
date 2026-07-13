//! W2 — merged non-ZK batched BaseFold opening of the two discharge oracles
//! (M_VK, n_vk = n_d + 2 message vars; M_D, n_d vars) in ONE combined FRI, replacing
//! the two separate 232-query openings.
//!
//! Construction (probed against the pinned rev's batch machinery — see
//! bin/probe_batch.rs; the "MidPad" lift is the one consistent with
//! `BatchBrakedownFolder`'s entry-duplication lift and uniform interleave batches):
//!
//! * Both oracles are committed exactly as before (same per-oracle Reed-Solomon
//!   dimension, same interleave batch b = 6, same subspace family → same digests), but
//!   under ONE `FRIParams::optimal_for_batch` parameter set with rs dim
//!   𝐧 = n_vk − b and lift = n_vk − n_d for M_D.
//! * Reduction: ONE batched degree-2 sumcheck over N = n_vk variables with two claims:
//!   [M_VK · eq(q)] (native N vars) and [M_D · eq(σ)] wrapped in an
//!   [`InteriorPaddedSumcheckProver`]: M_D's interleave (top-b) variables stay
//!   TOP-ALIGNED with M_VK's, and the `lift` missing dimension variables are padded
//!   with eq(0, ·) rounds just below them. Reduces both claims to a shared point ρ with
//!   per-oracle evals α_vk = M̃_VK(ρ), α_d = M̃_D(ρ_perm),
//!   ρ_perm = ρ[..n_d−b] ++ ρ[N−b..].
//! * Combine: one outer challenge r' (log2(2 oracles)); combined witness
//!   W = e0·M_VK + e1·midpad(M_D); target s' = e0·α_vk + e1·α_d·eq0(ρ[n_d−b..N−b]).
//! * One degree-1 MLE-check on W at ρ interleaved with the combined FRI
//!   (`FRIFoldProver::new_batch`): fold challenges = [c_0..c_{b-1}] ++ [r'] ++
//!   [c_b..c_{N-1}], first commitment at round b + 1, queries via
//!   `FRIQueryVerifier::new_batch` — one 232-query pass covering BOTH oracles.

use anyhow::ensure;
use binius_field::{Field, PackedField};
use binius_iop::{
	basefold::mlecheck_fri_consistency,
	fri::{FRIFoldVerifier, FRIParams, verify::FRIQueryVerifier},
	merkle_tree::MerkleTreeScheme,
};
use binius_ip::{
	channel::IPVerifierChannel,
	mlecheck,
	sumcheck::{self as ip_sumcheck, RoundCoeffs},
};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		batch::batch_prove,
		bivariate_product::BivariateProductSumcheckProver,
		common::SumcheckProver,
		multilinear_eval::MultilinearEvalProver,
		Error as SumcheckError,
	},
};
use binius_iop_prover::{
	fri::{FRIFoldProver, FoldRoundOutput},
	merkle_tree::MerkleTreeProver,
};
use binius_math::{
	FieldBuffer,
	multilinear::eq::{eq_ind, eq_ind_partial_eval, eq_ind_zero},
};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::DeserializeBytes;
use binius_verifier::config::B128;

use crate::packed::build_buffer_par;

/// Builds the combined witness `W = e[0]·big + e[1]·midpad(small)` from materialized
/// messages (n_big vars; `b` = uniform interleave batch). The mid-pad keeps the small
/// oracle's top-`b` (interleave) variables top-aligned and zero-pads the missing
/// dimension variables just below them.
pub fn build_combined_witness<P: PackedField<Scalar = B128>>(
	big: &FieldBuffer<P>,
	small: &FieldBuffer<P>,
	b: usize,
	e: [B128; 2],
) -> FieldBuffer<P> {
	let n_big = big.log_len();
	let n_small = small.log_len();
	let lift = n_big - n_small;
	let dim_small = n_small - b;
	let pad_mask = (1usize << lift) - 1;
	let low_mask = (1usize << dim_small) - 1;
	build_buffer_par::<P, _>(n_big, |idx| {
		let mut v = e[0] * big.get(idx);
		let pad = (idx >> dim_small) & pad_mask;
		if pad == 0 {
			let top = idx >> (n_big - b);
			let low = idx & low_mask;
			v += e[1] * small.get(low | (top << dim_small));
		}
		v
	})
}

/// Sumcheck decorator padding an inner prover with `n_extra` eq(0, ·) variables placed
/// BELOW the inner's top `head` variables (an interior zero-pad). Mirrors the pinned
/// rev's `PaddedSumcheckDecorator` (ip-prover/src/sumcheck/padded.rs), which pads at
/// the top only. Round order (high→low): `head` real rounds, `n_extra` padding rounds,
/// then the inner's remaining rounds. `finish` returns the RAW inner evals (at the
/// challenge point with the padding coordinates removed).
#[derive(Debug)]
pub struct InteriorPaddedSumcheckProver<F: Field, Inner> {
	inner: Inner,
	head: usize,
	n_extra: usize,
	round: usize,
	/// prod over the padding challenges of eq(0, r) = 1 - r.
	eq_prefix: F,
}

impl<F: Field, Inner: SumcheckProver<F>> InteriorPaddedSumcheckProver<F, Inner> {
	pub fn new(inner: Inner, head: usize, n_extra: usize) -> Self {
		assert!(inner.n_vars() >= head);
		Self {
			inner,
			head,
			n_extra,
			round: 0,
			eq_prefix: F::ONE,
		}
	}

	fn pads_done(&self) -> usize {
		self.round.saturating_sub(self.head).min(self.n_extra)
	}

	fn in_padding_phase(&self) -> bool {
		self.round >= self.head && self.round < self.head + self.n_extra
	}
}

impl<F: Field, Inner: SumcheckProver<F>> SumcheckProver<F>
	for InteriorPaddedSumcheckProver<F, Inner>
{
	fn n_vars(&self) -> usize {
		self.inner.n_vars() + self.n_extra - self.pads_done()
	}

	fn n_claims(&self) -> usize {
		self.inner.n_claims()
	}

	fn round_claim(&self) -> Vec<F> {
		self.inner
			.round_claim()
			.into_iter()
			.map(|claim| claim * self.eq_prefix)
			.collect()
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, SumcheckError> {
		if self.in_padding_phase() {
			// R(X) = claim * eq(0, X) = claim * (1 - X): degree-1, inner untouched.
			Ok(self
				.round_claim()
				.into_iter()
				.map(|claim| RoundCoeffs(vec![claim, -claim]))
				.collect())
		} else {
			let coeffs = self.inner.execute()?;
			if self.eq_prefix == F::ONE {
				Ok(coeffs)
			} else {
				Ok(coeffs
					.into_iter()
					.map(|c| c * self.eq_prefix)
					.collect())
			}
		}
	}

	fn fold(&mut self, challenge: F) -> Result<(), SumcheckError> {
		if self.in_padding_phase() {
			self.eq_prefix *= F::ONE - challenge;
		} else {
			self.inner.fold(challenge)?;
		}
		self.round += 1;
		Ok(())
	}

	fn finish(self) -> Result<Vec<F>, SumcheckError> {
		self.inner.finish()
	}
}

/// One point-evaluation claim against a committed oracle. The message buffer is MOVED
/// into the reduction sumcheck (it is folded in place); the combined witness for the
/// FRI phase is built afresh by the caller's `witness_builder` (which may regenerate
/// the data instead of cloning — the 16 GB memory path).
pub struct MergedClaim<'a, P: PackedField<Scalar = B128>> {
	/// The committed multilinear (message), n_i vars, low-coordinate-first layout.
	pub message: FieldBuffer<P>,
	/// The evaluation point (n_i coords, low-first).
	pub point: &'a [B128],
	/// The claimed evaluation.
	pub eval: B128,
}

/// Prover: merged opening of TWO oracles [large (index 0), small (index 1)] under one
/// batched `FRIParams`. `codewords` are in `params.input_oracles()` order.
///
/// `witness_builder(e)` must return the combined witness
/// `W = e[0]·M_big + e[1]·midpad(M_small)` (n_big vars) where
/// `midpad(M)(x) = M(x[..dim_small] ++ x[n_big-b..]) · [x[dim_small..n_big-b] == 0]`.
///
/// ## Preconditions
/// * `params.input_oracles().len() == 2`, uniform per-oracle `log_batch_size`.
/// * `claim_big` is the larger oracle (message vars == rs dim + batch).
#[allow(clippy::too_many_arguments)]
pub fn prove_merged_openings<'a, P, NTT, MerkleScheme, MerkleProver_, Challenger_>(
	params: &FRIParams<B128>,
	ntt: &'a NTT,
	merkle_prover: &'a MerkleProver_,
	claim_big: MergedClaim<'_, P>,
	claim_small: MergedClaim<'_, P>,
	witness_builder: impl FnOnce([B128; 2]) -> FieldBuffer<P>,
	codewords: Vec<(FieldBuffer<P>, &'a MerkleProver_::Committed)>,
	transcript: &mut ProverTranscript<Challenger_>,
) -> anyhow::Result<()>
where
	P: PackedField<Scalar = B128>,
	NTT: binius_math::ntt::AdditiveNTT<Field = B128> + Sync,
	MerkleScheme: MerkleTreeScheme<B128, Digest: binius_utils::SerializeBytes>,
	MerkleProver_: MerkleTreeProver<B128, Scheme = MerkleScheme>,
	Challenger_: Challenger,
{
	let specs = params.input_oracles();
	ensure!(specs.len() == 2, "expected exactly two input oracles");
	let b = specs[0].log_batch_size;
	ensure!(specs[1].log_batch_size == b, "uniform interleave batch required");
	let n_big = specs[0].log_msg_len;
	let n_small = specs[1].log_msg_len;
	let lift = n_big - n_small;
	ensure!(claim_big.message.log_len() == n_big && claim_big.point.len() == n_big);
	ensure!(claim_small.message.log_len() == n_small && claim_small.point.len() == n_small);
	ensure!(n_big == params.rs_code().log_dim() + b, "big oracle must span the rs dimension");
	let dim_small = n_small - b;

	// ---- Reduction: one batched degree-2 sumcheck over n_big vars. ----
	// Messages are moved in and folded in place (no clones).
	let provers_big = BivariateProductSumcheckProver::new(
		[claim_big.message, eq_ind_partial_eval::<P>(claim_big.point)],
		claim_big.eval,
	)
	.map_err(|e| anyhow::anyhow!("reduction prover (big): {e}"))?;
	let provers_small = InteriorPaddedSumcheckProver::new(
		BivariateProductSumcheckProver::new(
			[claim_small.message, eq_ind_partial_eval::<P>(claim_small.point)],
			claim_small.eval,
		)
		.map_err(|e| anyhow::anyhow!("reduction prover (small): {e}"))?,
		b,
		lift,
	);

	enum Either<A, B> {
		A(A),
		B(B),
	}
	impl<F: Field, A: SumcheckProver<F>, B: SumcheckProver<F>> SumcheckProver<F> for Either<A, B> {
		fn n_vars(&self) -> usize {
			match self {
				Either::A(a) => a.n_vars(),
				Either::B(b) => b.n_vars(),
			}
		}
		fn n_claims(&self) -> usize {
			match self {
				Either::A(a) => a.n_claims(),
				Either::B(b) => b.n_claims(),
			}
		}
		fn round_claim(&self) -> Vec<F> {
			match self {
				Either::A(a) => a.round_claim(),
				Either::B(b) => b.round_claim(),
			}
		}
		fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, SumcheckError> {
			match self {
				Either::A(a) => a.execute(),
				Either::B(b) => b.execute(),
			}
		}
		fn fold(&mut self, challenge: F) -> Result<(), SumcheckError> {
			match self {
				Either::A(a) => a.fold(challenge),
				Either::B(b) => b.fold(challenge),
			}
		}
		fn finish(self) -> Result<Vec<F>, SumcheckError> {
			match self {
				Either::A(a) => a.finish(),
				Either::B(b) => b.finish(),
			}
		}
	}

	let output = batch_prove(
		vec![Either::A(provers_big), Either::B(provers_small)],
		transcript,
	)
	.map_err(|e| anyhow::anyhow!("reduction batch_prove: {e}"))?;
	// batch_prove returns challenges reversed to low-first.
	let rho = output.challenges;
	ensure!(rho.len() == n_big, "reduction round count");
	let alpha_big = output.multilinear_evals[0][0];
	let alpha_small = output.multilinear_evals[1][0];
	IPProverChannel::<B128>::send_many(transcript, &[alpha_big, alpha_small]);

	// ---- Combine: outer challenge, combined witness (caller-built), target. ----
	let r_outer: B128 = IPProverChannel::<B128>::sample(transcript);
	let e = {
		let t = eq_ind_partial_eval::<B128>(&[r_outer]);
		[t.get(0), t.get(1)]
	};
	let w = witness_builder(e);
	ensure!(w.log_len() == n_big, "witness_builder must return an n_big-var buffer");
	let s_prime = e[0] * alpha_big
		+ e[1] * alpha_small * eq_ind_zero(&rho[dim_small..n_big - b]);

	// ---- One MLE-check on W at rho interleaved with the combined FRI. ----
	let mut mle = MultilinearEvalProver::new(w, &rho, s_prime)
		.map_err(|e| anyhow::anyhow!("mle prover: {e}"))?;
	let mut folder = FRIFoldProver::new_batch(params, ntt, merkle_prover, codewords);
	ensure!(folder.n_rounds() == params.n_fold_rounds());
	for round in 0..n_big {
		if round == b {
			folder.receive_challenge(r_outer);
		}
		let mut coeffs_vec = mle.execute().map_err(|e| anyhow::anyhow!("mle exec: {e}"))?;
		let coeffs = coeffs_vec.pop().expect("one claim");
		let commitment = folder.execute_fold_round();
		transcript
			.message()
			.write_scalar_slice(mlecheck::RoundProof::truncate(coeffs).coeffs());
		if let FoldRoundOutput::Commitment(c) = commitment {
			transcript.message().write(&c);
		}
		let c: B128 = binius_transcript::fiat_shamir::CanSample::sample(transcript);
		mle.fold(c).map_err(|e| anyhow::anyhow!("mle fold: {e}"))?;
		folder.receive_challenge(c);
	}
	let commitment = folder.execute_fold_round();
	if let FoldRoundOutput::Commitment(c) = commitment {
		transcript.message().write(&c);
	}
	folder.finish_proof(transcript);
	Ok(())
}

/// Verifier: merged opening of TWO oracles against `params` and the two commitments
/// (in `params.input_oracles()` order). `point_big`/`point_small` and the claimed
/// evals must already be Fiat-Shamir-bound by the caller.
#[allow(clippy::too_many_arguments)]
pub fn verify_merged_openings<MerkleScheme, Challenger_>(
	params: &FRIParams<B128>,
	merkle_scheme: &MerkleScheme,
	commitments: &[MerkleScheme::Digest; 2],
	point_big: &[B128],
	eval_big: B128,
	point_small: &[B128],
	eval_small: B128,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> anyhow::Result<()>
where
	MerkleScheme: MerkleTreeScheme<B128, Digest: DeserializeBytes + Clone>,
	Challenger_: Challenger,
{
	let specs = params.input_oracles();
	ensure!(specs.len() == 2, "expected exactly two input oracles");
	let b = specs[0].log_batch_size;
	ensure!(specs[1].log_batch_size == b, "uniform interleave batch required");
	let n_big = specs[0].log_msg_len;
	let n_small = specs[1].log_msg_len;
	ensure!(point_big.len() == n_big && point_small.len() == n_small);
	let dim_small = n_small - b;

	// ---- Reduction. ----
	let out = ip_sumcheck::batch_verify::<B128, _>(
		n_big,
		2,
		&[eval_big, eval_small],
		transcript,
	)
	.map_err(|e| anyhow::anyhow!("reduction batch_verify: {e}"))?;
	let mu = out.batch_coeff;
	let reduced = out.eval;
	let mut rho = out.challenges;
	rho.reverse(); // low-first

	let alphas: Vec<B128> = IPVerifierChannel::<B128>::recv_many(transcript, 2)
		.map_err(|e| anyhow::anyhow!("alphas: {e}"))?;
	let (alpha_big, alpha_small) = (alphas[0], alphas[1]);
	// rho_perm for the small oracle: its real coords are the low dim_small plus the top b.
	let rho_perm: Vec<B128> = rho[..dim_small]
		.iter()
		.chain(&rho[n_big - b..])
		.copied()
		.collect();
	let pad_eq = eq_ind_zero(&rho[dim_small..n_big - b]);
	let contributions = [
		alpha_big * eq_ind(point_big, &rho),
		alpha_small * eq_ind(point_small, &rho_perm) * pad_eq,
	];
	let expected = binius_math::univariate::evaluate_univariate(&contributions, mu);
	ensure!(expected == reduced, "reduction recombination failed");

	// ---- Combine + MLE-check + FRI. ----
	let r_outer: B128 = IPVerifierChannel::<B128>::sample(transcript);
	let e = {
		let t = binius_math::multilinear::eq::eq_ind_partial_eval_scalars(&[r_outer]);
		[t[0], t[1]]
	};
	let s_prime = e[0] * alpha_big + e[1] * alpha_small * pad_eq;

	let mut fri_fold = FRIFoldVerifier::<B128, _>::new(params);
	let mut challenges: Vec<B128> = Vec::with_capacity(params.n_fold_rounds());
	let mut sum = s_prime;
	for round in 0..n_big {
		if round == b {
			fri_fold
				.process_round(&mut transcript.message())
				.map_err(|e| anyhow::anyhow!("fri outer round: {e}"))?;
			challenges.push(r_outer);
		}
		let round_proof = mlecheck::RoundProof(RoundCoeffs(
			transcript
				.message()
				.read_vec(1)
				.map_err(|e| anyhow::anyhow!("mle round proof: {e}"))?,
		));
		fri_fold
			.process_round(&mut transcript.message())
			.map_err(|e| anyhow::anyhow!("fri round: {e}"))?;
		let alpha = rho[n_big - 1 - round];
		let round_coeffs = round_proof.recover(sum, alpha);
		let c: B128 = IPVerifierChannel::<B128>::sample(transcript);
		sum = round_coeffs.evaluate(c);
		challenges.push(c);
	}
	fri_fold
		.process_round(&mut transcript.message())
		.map_err(|e| anyhow::anyhow!("fri final round: {e}"))?;
	let round_commitments = fri_fold.finalize();

	let fri_verifier = FRIQueryVerifier::new_batch(
		params,
		merkle_scheme,
		commitments,
		&round_commitments,
		&challenges,
	);
	let final_fri_value = fri_verifier
		.verify(transcript)
		.map_err(|e| anyhow::anyhow!("merged FRI queries: {e}"))?;
	ensure!(
		mlecheck_fri_consistency(final_fri_value, sum),
		"merged opening: FRI/MLE-check inconsistency"
	);
	Ok(())
}

#[cfg(test)]
mod tests {
	use binius_field::Random;
	use binius_hash::StdHashSuite;
	use binius_iop::{fri::PartialOracleSpec, merkle_tree::BinaryMerkleTreeScheme};
	use binius_iop_prover::{fri::commit_interleaved, merkle_tree::prover::BinaryMerkleTreeProver};
	use binius_math::{
		BinarySubspace,
		multilinear::evaluate::evaluate,
		ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
	};
	use binius_transcript::ProverTranscript;
	use binius_verifier::config::StdChallenger;
	use rand::prelude::*;

	use super::*;

	type Scheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

	/// E2E of the merged opening on small random oracles (mirrors probe_batch), plus
	/// tamper negatives on both claims.
	#[test]
	fn test_merged_openings_roundtrip_and_tampers() {
		const N1: usize = 9;
		const N2: usize = 7;
		const B: usize = 2;
		let mut rng = StdRng::seed_from_u64(23);
		let m1: FieldBuffer<B128> = {
			let v: Vec<B128> = (0..1 << N1).map(|_| B128::random(&mut rng)).collect();
			FieldBuffer::from_values(&v)
		};
		let m2: FieldBuffer<B128> = {
			let v: Vec<B128> = (0..1 << N2).map(|_| B128::random(&mut rng)).collect();
			FieldBuffer::from_values(&v)
		};
		let q1: Vec<B128> = (0..N1).map(|_| B128::random(&mut rng)).collect();
		let q2: Vec<B128> = (0..N2).map(|_| B128::random(&mut rng)).collect();
		let c1 = evaluate(&m1, &q1);
		let c2 = evaluate(&m2, &q2);

		let merkle_prover = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();
		let scheme: &Scheme = merkle_prover.scheme();
		let subspace: BinarySubspace<B128> = BinarySubspace::with_dim(N1 - B + 1);
		let dc = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(dc);
		let (params, _) = FRIParams::<B128>::optimal_for_batch(
			&ntt.domain_context,
			scheme,
			&[
				PartialOracleSpec {
					log_msg_len: N1,
					log_batch_size: Some(B),
				},
				PartialOracleSpec {
					log_msg_len: N2,
					log_batch_size: Some(B),
				},
			],
			1,
			8,
		);
		let co1 = commit_interleaved(&params, 0, &ntt, &merkle_prover, m1.to_ref());
		let co2 = commit_interleaved(&params, 1, &ntt, &merkle_prover, m2.to_ref());

		let run = |c1v: B128, c2v: B128| -> anyhow::Result<()> {
			let mut pt = ProverTranscript::new(StdChallenger::default());
			prove_merged_openings(
				&params,
				&ntt,
				&merkle_prover,
				MergedClaim {
					message: m1.clone(),
					point: &q1,
					eval: c1v,
				},
				MergedClaim {
					message: m2.clone(),
					point: &q2,
					eval: c2v,
				},
				|e| build_combined_witness(&m1, &m2, B, e),
				vec![(co1.codeword.clone(), &co1.committed), (co2.codeword.clone(), &co2.committed)],
				&mut pt,
			)?;
			let mut vt = pt.into_verifier();
			verify_merged_openings(
				&params,
				scheme,
				&[co1.commitment.clone(), co2.commitment.clone()],
				&q1,
				c1v,
				&q2,
				c2v,
				&mut vt,
			)?;
			vt.finalize()?;
			Ok(())
		};

		run(c1, c2).expect("honest merged opening must verify");
		assert!(run(c1 + B128::ONE, c2).is_err(), "tampered big claim must fail");
		assert!(run(c1, c2 + B128::ONE).is_err(), "tampered small claim must fail");
	}

	/// The interior-padded prover must agree round-by-round with a naively materialized
	/// mid-padded composite (value identity of the reduction).
	#[test]
	fn test_interior_padded_matches_materialized() {
		const N2: usize = 6;
		const HEAD: usize = 2;
		const EXTRA: usize = 3;
		let n_total = N2 + EXTRA;
		let mut rng = StdRng::seed_from_u64(5);
		let m: Vec<B128> = (0..1 << N2).map(|_| B128::random(&mut rng)).collect();
		let t: Vec<B128> = (0..1 << N2).map(|_| B128::random(&mut rng)).collect();
		let sum: B128 = m.iter().zip(&t).map(|(&a, &b)| a * b).sum();

		// Materialized mid-pad: vars [0..N2-HEAD) real-low, [N2-HEAD..N2-HEAD+EXTRA) pad,
		// top HEAD real.
		let dim_low = N2 - HEAD;
		let mk_padded = |src: &[B128]| -> FieldBuffer<B128> {
			let mut v = vec![B128::ZERO; 1 << n_total];
			for (idx, slot) in v.iter_mut().enumerate() {
				let pad = (idx >> dim_low) & ((1 << EXTRA) - 1);
				if pad == 0 {
					let top = idx >> (n_total - HEAD);
					let low = idx & ((1 << dim_low) - 1);
					*slot = src[low | (top << dim_low)];
				}
			}
			FieldBuffer::from_values(&v)
		};
		// Composite reference: midpad(m) * constant-extend(t). The message carries the
		// single eq0 pad indicator; the transparent is CONSTANT along the pad vars, so
		// the composite = m·t·eq0(pad) exactly (one indicator, degree-1 pad rounds) —
		// the same composite the InteriorPaddedSumcheckProver proves.
		let mk_lifted = |src: &[B128]| -> FieldBuffer<B128> {
			let mut v = vec![B128::ZERO; 1 << n_total];
			for (idx, slot) in v.iter_mut().enumerate() {
				let top = idx >> (n_total - HEAD);
				let low = idx & ((1 << dim_low) - 1);
				*slot = src[low | (top << dim_low)];
			}
			FieldBuffer::from_values(&v)
		};
		let m_padded = mk_padded(&m);
		let t_lifted = mk_lifted(&t);

		let mut naive =
			BivariateProductSumcheckProver::new([m_padded, t_lifted], sum).unwrap();
		let mut ours = InteriorPaddedSumcheckProver::new(
			BivariateProductSumcheckProver::new(
				[
					FieldBuffer::<B128>::from_values(&m),
					FieldBuffer::<B128>::from_values(&t),
				],
				sum,
			)
			.unwrap(),
			HEAD,
			EXTRA,
		);

		let mut rng2 = StdRng::seed_from_u64(9);
		for round in 0..n_total {
			let a = naive.execute().unwrap().pop().unwrap();
			let b = ours.execute().unwrap().pop().unwrap();
			// Compare as polynomials evaluated at points (degree may differ: the pad
			// rounds emit degree-1; the naive emits degree-2 with zero top coeff).
			for x in [B128::ZERO, B128::ONE, B128::new(7), B128::new(1 << 20)] {
				assert_eq!(a.evaluate(x), b.evaluate(x), "round {round} poly mismatch at {x:?}");
			}
			let c = B128::random(&mut rng2);
			naive.fold(c).unwrap();
			ours.fold(c).unwrap();
		}
		let ea = naive.finish().unwrap();
		let eb = ours.finish().unwrap();
		// naive evals are the PADDED multilinears' evals at the full point; ours are the
		// inner evals at the permuted sub-point. Their relation: padded_eval =
		// inner_eval * eq0(pad coords) — checked via the round-claim path already; here
		// just sanity-check lengths.
		assert_eq!(ea.len(), 2);
		assert_eq!(eb.len(), 2);
	}
}
