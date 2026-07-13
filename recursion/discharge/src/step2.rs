//! STEP-2 discharge: committed M_VK + per-batch committed M_D + Phase C
//! (weighted fracaddcheck over the (n_d + 2)-var union domain) + PCS final check
//! (two BaseFold openings). The verifier NEVER touches the ConstraintSystem — it takes
//! (VKM, statement, transcript) only (P0.2 checks the statement's cs_digest against
//! the VKM's).
//!
//! Transcript layout (one FS transcript; spec section 3 challenge order):
//!   observe(VKM) | observe(statement) | Phase A [mu; rounds; 3K finish evals] |
//!   8K d values | digest_D | tau | d_root | fracaddcheck layers (num/den to point pi) |
//!   [v00 v10 v01 m_pi] | phi | Phase B rounds | m | BaseFold(M_D at sigma, claim m) |
//!   rho_c | BaseFold(M_VK at [pi_lo, rho_c], corner-combined claim).
//!
//! Union-domain layout (spec 1.1, selector-high): low n_l := n_d coords = t (blocks
//! 00/10/01 = X/Y/U rows) or M_D's own (a, c) index (block 11); blk at the TOP two
//! coords. num = [eq(t,rho_ext) | eq | eq | M_D]; den = [tau+X | tau+Y | tau+U |
//! tau+emb], emb(w) = B128(w) under the aligned tag basis (vk.rs). The total
//! fractional sum is identically 0 for the honest histograms (char-2 pole
//! cancellation); coset-disjoint tags make partial fractions force M_D = the
//! rho-weighted histograms, pole family by pole family.

use std::time::Instant;

use anyhow::ensure;
use binius_core::constraint_system::ConstraintSystem;
use binius_field::Field;
use binius_ip::{
	channel::IPVerifierChannel,
	fracaddcheck::{self, FracAddEvalClaim},
	prodcheck::MultilinearEvalClaim,
};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		batch::batch_prove_and_write_evals, bivariate_product::BivariateProductSumcheckProver,
		prove_single,
	},
};
use binius_iop_prover::{
	fri::commit_interleaved,
};
use binius_math::{
	FieldBuffer,
	multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
	univariate::evaluate_univariate,
};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::Challenger};
use binius_verifier::{config::B128, protocols::shift::SHIFT_VARIANT_COUNT};

use crate::{
	cubic::CubicProductSumcheckProver,
	discharge::{DischargeStatement, axpy_point_tensor, claim_contexts_from_dims},
	fracadd::FastFracAddProver,
	packed::{
		PB, assemble_m_d_packed, axpy_dense_par, build_m_vk_packed, build_phase_c_leaf_halves,
		gather_column, vk_corner_values_packed,
	},
	table::{TermTable, build_histograms, evaluate_d_g, eq_point, extract_table, m_d_points},
	vk::{
		Digest, DischargeMerkleProver, DischargeVkm, beta, build_ntt, build_pcs,
		commit_m_vk, serialize_digest,
	},
};

/// Adversarial knobs for the STEP-2 prover (tests only; `None` = honest). The tampered
/// prover follows the honest machinery on tampered data, which is the strongest
/// adaptive cheat the copied-prover STEP-1 adversarial suite exercised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step2Tamper {
	None,
	/// +1 at one slot of M_D's (1,1) selector block BEFORE the M_D commit. Invisible
	/// to Phases A and B (W_eq vanishes there and m/m_pi are consistently tampered);
	/// in STEP 1 only the native rebuild caught this — here Phase C MUST.
	MdBlock3,
	/// Flip one byte of digest_D at the observation point (the tree and all openings
	/// remain honest): the FS stream is consistent on both sides, so every phase
	/// passes until the M_D opening's Merkle verification rejects.
	DigestD,
}

/// Timing breakdown of a STEP-2 discharge proving run (seconds).
#[derive(Debug, Default, Clone)]
pub struct ProveTimings2 {
	pub phase_a_s: f64,
	pub histograms_s: f64,
	pub commit_d_s: f64,
	pub phase_c_s: f64,
	pub phase_b_s: f64,
	pub recommit_vk_s: f64,
	/// W2: ONE merged batched opening of [M_VK, M_D] (replaces the two openings).
	pub open_merged_s: f64,
	pub total_s: f64,
}

/// Timing breakdown of a STEP-2 discharge verification run (seconds).
#[derive(Debug, Default, Clone)]
pub struct VerifyTimings2 {
	/// Phases A+B + Phase C transcript verification (everything before the opening).
	pub transcript_s: f64,
	/// W2: the merged batched opening (reduction + MLE-check/FRI + one query pass).
	pub open_merged_s: f64,
	pub total_s: f64,
}

/// Shared structural preconditions (P0.1/P0.2 metadata coherence, statement side).
fn check_statement_against_vkm(vkm: &DischargeVkm, stmt: &DischargeStatement) -> anyhow::Result<()> {
	ensure!(
		stmt.cs_digest == vkm.cs_digest,
		"P0.2 cs_digest mismatch: statement {} vs VKM {}",
		hex(&stmt.cs_digest),
		hex(&vkm.cs_digest)
	);
	ensure!(
		stmt.n_terms == vkm.dims.n_terms
			&& stmt.n_pad == vkm.dims.n_pad
			&& stmt.parity == vkm.dims.parity,
		"P0.1 table metadata mismatch: statement (N={}, N_pad={}, parity={}) vs VKM (N={}, N_pad={}, parity={})",
		stmt.n_terms,
		stmt.n_pad,
		stmt.parity,
		vkm.dims.n_terms,
		vkm.dims.n_pad,
		vkm.dims.parity,
	);
	ensure!(!stmt.claims.is_empty(), "empty claim batch");
	Ok(())
}

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The Phase-B claim list shared by prover and verifier: per claim l the 10 M_D points
/// with evals [a_l, b_l, d_{l,0..8}], plus the Phase-C point (pi_lo, m_pi) LAST.
fn phase_b_claims(
	dims: &crate::table::ShapeDims,
	ctxs: &[crate::discharge::ClaimCtx],
	finish_evals: &[B128],
	d_vals: &[B128],
	pi_lo: &[B128],
	m_pi: B128,
) -> (Vec<Vec<B128>>, Vec<B128>) {
	let k = ctxs.len();
	let mut points = Vec::with_capacity(10 * k + 1);
	let mut evals = Vec::with_capacity(10 * k + 1);
	for (l, ctx) in ctxs.iter().enumerate() {
		points.extend(m_d_points(dims, &ctx.tr));
		evals.push(finish_evals[3 * l]);
		evals.push(finish_evals[3 * l + 1]);
		evals.extend_from_slice(&d_vals[l * SHIFT_VARIANT_COUNT..(l + 1) * SHIFT_VARIANT_COUNT]);
	}
	let mut pi_point = pi_lo.to_vec();
	// The pi point addresses M_D's full n_d-var domain directly (address + its own
	// selector pair); it is already n_d coordinates.
	debug_assert_eq!(pi_point.len(), dims.n_d);
	points.push(std::mem::take(&mut pi_point));
	evals.push(m_pi);
	(points, evals)
}

/// STEP-2 prover. The prover holds the CS (it is the table's source) and the VKM.
pub fn discharge_prove_step2<C: Challenger>(
	cs: &ConstraintSystem,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
) -> anyhow::Result<ProveTimings2> {
	discharge_prove_step2_impl(cs, vkm, stmt, transcript, Step2Tamper::None)
}

/// Test-only entry with adversarial knobs. Follows the honest machinery on tampered
/// data (see [`Step2Tamper`]).
#[doc(hidden)]
pub fn discharge_prove_step2_tampered<C: Challenger>(
	cs: &ConstraintSystem,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
	tamper: Step2Tamper,
) -> anyhow::Result<ProveTimings2> {
	discharge_prove_step2_impl(cs, vkm, stmt, transcript, tamper)
}

fn discharge_prove_step2_impl<C: Challenger>(
	cs: &ConstraintSystem,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
	tamper: Step2Tamper,
) -> anyhow::Result<ProveTimings2> {
	let table = extract_table(cs)?;
	ensure!(
		table.cs_digest == vkm.cs_digest && table.dims == vkm.dims,
		"prover CS does not regenerate the VKM's shape (T1 hygiene)"
	);
	discharge_prove_step2_on_table(&table, &table, vkm, stmt, transcript, tamper)
}

/// STEP-2 prover over a CALLER-SUPPLIED term table (no CS extraction, no T1 hygiene
/// check against the table contents — the table is trusted as given). Intended for
/// adversarial test harnesses that prove over a deliberately tampered table (the
/// "consistent lie" that Phases A/B cannot see and only the committed-table binding
/// rejects). `table` drives Phases A/B/C data; `vk_table` drives the M_VK re-commit
/// and opening (a realistic adversary keeps it honest so the re-committed digest
/// matches the pinned vk_digest, and is then caught by the opening's false claim).
/// Honest callers use [`discharge_prove_step2`], which passes the same table for both.
#[doc(hidden)]
pub fn discharge_prove_step2_on_table<C: Challenger>(
	table: &TermTable,
	vk_table: &TermTable,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
	tamper: Step2Tamper,
) -> anyhow::Result<ProveTimings2> {
	let t_total = Instant::now();
	check_statement_against_vkm(vkm, stmt)?;
	let mut ctxs = claim_contexts_from_dims(&table.dims, stmt.parity, stmt, false)?;
	let dims = &table.dims;
	let n_l = dims.n_d;
	let k = ctxs.len();

	let pcs = build_pcs(vkm)?;
	let ntt = build_ntt(pcs.log_domain);
	let merkle_prover = DischargeMerkleProver::new();

	// Phase 0: VKM + statement absorption (P0.1) — before mu is sampled.
	IPProverChannel::<B128>::observe_many(transcript, &vkm.to_elems());
	IPProverChannel::<B128>::observe_many(transcript, &stmt.to_elems());

	// ---- Phase A: K cubic sumchecks via the upstream batch driver. ----
	// Columns are packed parallel gathers (W1); the cubic prover itself runs the
	// upstream packed round idiom (rayon + WideMul).
	let t = Instant::now();
	let mut provers = Vec::with_capacity(k);
	for ctx in &ctxs {
		provers.push(
			CubicProductSumcheckProver::new(
				[
					gather_column::<PB, _>(&table.terms, &ctx.tr.x_tensor, dims.n_t, |t| {
						t.x as usize
					}),
					gather_column::<PB, _>(&table.terms, &ctx.tr.y_tensor, dims.n_t, |t| {
						t.y as usize
					}),
					gather_column::<PB, _>(&table.terms, &ctx.tr.g_tab, dims.n_t, |t| {
						t.u as usize
					}),
				],
				ctx.sum,
			),
		);
	}
	let output = batch_prove_and_write_evals(provers, transcript);
	let rho = output.challenges.clone(); // driver returns low-first
	ensure!(rho.len() == dims.n_t, "phase A round count");
	let phase_a_s = t.elapsed().as_secs_f64();

	// ---- Histograms at rho + the 8K d values. ----
	let t = Instant::now();
	let hist = build_histograms(table, &rho);
	let mut d_vals = Vec::with_capacity(k * SHIFT_VARIANT_COUNT);
	for ctx in &ctxs {
		let points = m_d_points(dims, &ctx.tr);
		for point in &points[2..] {
			d_vals.push(evaluate_d_g(&hist.d_g, &point[..crate::table::N_U]));
		}
	}
	IPProverChannel::<B128>::send_many(transcript, &d_vals);
	// Free the per-claim eq tensors before the memory-heavy Phase C (only O(arity)
	// transparents — parsed points, h_ops, m_coords — are needed from here on).
	for ctx in &mut ctxs {
		ctx.tr.x_tensor = Vec::new();
		ctx.tr.y_tensor = Vec::new();
		ctx.tr.g_tab = Vec::new();
	}
	let ctxs = ctxs; // immutable from here
	let histograms_s = t.elapsed().as_secs_f64();

	// ---- Commit M_D (second non-ZK oracle); digest observed BEFORE tau (P0.1). ----
	let t = Instant::now();
	let mut m_d = assemble_m_d_packed::<PB>(dims, &hist);
	if tamper == Step2Tamper::MdBlock3 {
		let idx = (3usize << dims.n_a) + (12345 % (1usize << dims.n_a));
		let cur = m_d.get(idx);
		m_d.set(idx, cur + B128::ONE);
	}
	// M_D is ORACLE 1 of the batched params (W2 merged opening).
	let commit_d = commit_interleaved(&pcs.params, 1, &ntt, &merkle_prover, m_d.to_ref());
	if tamper == Step2Tamper::DigestD {
		let mut bytes = serialize_digest(&commit_d.commitment);
		bytes[0] ^= 1;
		use binius_utils::DeserializeBytes;
		let flipped = Digest::deserialize(&bytes[..]).expect("32-byte digest roundtrip");
		transcript.message().write(&flipped);
	} else {
		transcript.message().write(&commit_d.commitment);
	}
	let commit_d_s = t.elapsed().as_secs_f64();

	// ---- Phase C: weighted fracaddcheck over the (n_l + 2)-var union domain. ----
	let t = Instant::now();
	let timing_detail = std::env::var_os("DISCHARGE_TIMING").is_some();
	let mut lap = Instant::now();
	let sub = |label: &str, lap: &mut Instant| {
		if timing_detail {
			eprintln!("    [phase C] {label:12} {:.2}s", lap.elapsed().as_secs_f64());
		}
		*lap = Instant::now();
	};
	let tau: B128 = IPProverChannel::<B128>::sample(transcript);
	let leaf = {
		let eq_rho = eq_ind_partial_eval::<PB>(&rho);
		build_phase_c_leaf_halves(table, tau, &eq_rho, &m_d)?
		// eq_rho dropped here, before the layered tree doubles the footprint
	};
	sub("leaf build", &mut lap);
	let (frac_prover, (num_root, den_root)) = FastFracAddProver::new(leaf);
	sub("tree build", &mut lap);
	if tamper != Step2Tamper::MdBlock3 {
		ensure!(num_root == B128::ZERO, "phase C tree root numerator nonzero (table/M_D mismatch)");
	}
	ensure!(den_root != B128::ZERO, "phase C tree root denominator zero");
	IPProverChannel::<B128>::send_one(transcript, den_root);
	let root_claim = (
		MultilinearEvalClaim {
			eval: B128::ZERO,
			point: Vec::new(),
		},
		MultilinearEvalClaim {
			eval: den_root,
			point: Vec::new(),
		},
	);
	let (num_leaf_claim, _den_leaf_claim) = frac_prover.prove(root_claim, transcript);
	sub("layer prove", &mut lap);
	let pi = num_leaf_claim.point;
	ensure!(pi.len() == n_l + 2, "phase C leaf point arity");
	let pi_lo = &pi[..n_l];

	// Corner values + m_pi (recv'd by the verifier; validated at the PCS step /
	// Phase B respectively).
	let (v_corner, m_pi) = {
		let eq_pi = eq_ind_partial_eval::<PB>(pi_lo);
		(vk_corner_values_packed(table, &eq_pi), evaluate(&m_d, pi_lo))
	};
	sub("corners+m_pi", &mut lap);
	IPProverChannel::<B128>::send_many(
		transcript,
		&[v_corner[0], v_corner[1], v_corner[2], m_pi],
	);
	let phase_c_s = t.elapsed().as_secs_f64();

	// ---- Phase B: one bivariate sumcheck over [W_eq, M_D], 10K + 1 claims. ----
	let t = Instant::now();
	let phi: B128 = IPProverChannel::<B128>::sample(transcript);
	let finish_evals: Vec<B128> = output
		.multilinear_evals
		.iter()
		.flat_map(|evals| {
			assert_eq!(evals.len(), 3, "cubic finish arity");
			evals.iter().copied()
		})
		.collect();
	let (points, evals) = phase_b_claims(dims, &ctxs, &finish_evals, &d_vals, pi_lo, m_pi);
	let combined = evaluate_univariate(&evals, phi);

	let mut w_eq = vec![B128::ZERO; 1 << dims.n_d];
	let mut phi_pow = B128::ONE;
	for p in points.iter().take(10 * k) {
		axpy_point_tensor(&mut w_eq, dims.n_a, p, phi_pow);
		phi_pow *= phi;
	}
	{
		// The Phase-C point is dense (no zero tail / 0-1 selectors): full tensor,
		// expanded packed-parallel and added packed-parallel (W1).
		let pi_tensor = eq_ind_partial_eval::<B128>(pi_lo);
		axpy_dense_par(&mut w_eq, pi_tensor.as_ref(), phi_pow);
	}
	let prover_b = BivariateProductSumcheckProver::new(
		[FieldBuffer::<PB>::from_values(&w_eq), m_d.clone()],
		combined,
	);
	drop(w_eq);
	let out_b = prove_single(prover_b, transcript);
	let mut sigma = out_b.challenges.clone();
	sigma.reverse(); // low-first
	ensure!(sigma.len() == dims.n_d, "phase B round count");
	let m = out_b.multilinear_evals[1];
	IPProverChannel::<B128>::send_one(transcript, m);
	let phase_b_s = t.elapsed().as_secs_f64();

	// ---- Final check (W2): ONE merged batched opening of [M_VK, M_D]. ----
	// rho_c is sampled AFTER m (the M_D claim) and the corner values are FS-bound, so
	// both point-evaluation claims are fixed before the opening begins.
	let rho_c: Vec<B128> = IPProverChannel::<B128>::sample_many(transcript, 2);
	let t = Instant::now();
	// Deterministic re-commit (spec section 4): rebuild M_VK from the CS-derived table
	// and assert the digest matches the VKM (T1 regeneration in-process).
	let m_vk = build_m_vk_packed::<PB>(vk_table);
	let commit_vk = commit_m_vk(&pcs, &ntt, &merkle_prover, &m_vk);
	ensure!(
		serialize_digest(&commit_vk.commitment) == vkm.vk_digest,
		"re-committed M_VK digest differs from the VKM's vk_digest (T1 violation)"
	);
	let recommit_vk_s = t.elapsed().as_secs_f64();

	let t = Instant::now();
	let claim_vk = eq_point(&[B128::ZERO, B128::ZERO], &rho_c) * v_corner[0]
		+ eq_point(&[B128::ONE, B128::ZERO], &rho_c) * v_corner[1]
		+ eq_point(&[B128::ZERO, B128::ONE], &rho_c) * v_corner[2];
	let mut vk_point = pi_lo.to_vec();
	vk_point.extend_from_slice(&rho_c);
	{
		let b = vkm.fri_batch.log_batch_size;
		// The combined witness is REGENERATED from the table + M_D at combine time (no
		// clone of the 2^{n_d+2} M_VK buffer is held through the reduction).
		let table_for_w = vk_table;
		let m_d_for_w = m_d.clone();
		let n_big = dims.n_d + 2;
		let dim_small = dims.n_d - b;
		crate::merged::prove_merged_openings(
			&pcs.params,
			&ntt,
			&merkle_prover,
			crate::merged::MergedClaim {
				message: m_vk,
				point: &vk_point,
				eval: claim_vk,
			},
			crate::merged::MergedClaim {
				message: m_d,
				point: &sigma,
				eval: m,
			},
			|e| {
				// W = e0*M_VK + e1*midpad(M_D), lanes generated (M_VK never re-buffered).
				let n = table_for_w.terms.len();
				let terms = &table_for_w.terms;
				let kappa = crate::vk::tags(dims.n_a);
				let n_l = dims.n_d;
				let mask_l = (1usize << n_l) - 1;
				let pad_mask = (1usize << (n_big - dims.n_d)) - 1;
				let low_mask = (1usize << dim_small) - 1;
				crate::packed::build_buffer_par::<PB, _>(n_big, |w| {
					// M_VK lane (build_m_vk_packed generator, inlined).
					let blk = w >> n_l;
					let tt = w & mask_l;
					let vk_lane = match blk {
						0 | 1 | 2 => {
							if tt < n {
								let term = &terms[tt];
								let addr = match blk {
									0 => term.x as u64,
									1 => term.y as u64,
									_ => term.u as u64,
								};
								crate::vk::vk_entry(blk, addr, dims.n_a)
							} else {
								kappa[blk]
							}
						}
						_ => B128::ZERO,
					};
					let mut v = e[0] * vk_lane;
					let pad = (w >> dim_small) & pad_mask;
					if pad == 0 {
						let top = w >> (n_big - b);
						let low = w & low_mask;
						v += e[1] * m_d_for_w.get(low | (top << dim_small));
					}
					v
				})
			},
			vec![
				(commit_vk.codeword, &commit_vk.committed),
				(commit_d.codeword, &commit_d.committed),
			],
			transcript,
		)?;
	}
	let open_merged_s = t.elapsed().as_secs_f64();

	Ok(ProveTimings2 {
		phase_a_s,
		histograms_s,
		commit_d_s,
		phase_c_s,
		phase_b_s,
		recommit_vk_s,
		open_merged_s,
		total_s: t_total.elapsed().as_secs_f64(),
	})
}

/// STEP-2 verifier: (VKM, statement, transcript) — NO ConstraintSystem anywhere.
pub fn discharge_verify_step2<C: Challenger>(
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut VerifierTranscript<C>,
) -> anyhow::Result<VerifyTimings2> {
	let t_total = Instant::now();
	check_statement_against_vkm(vkm, stmt)?;
	let dims = &vkm.dims;
	let n_l = dims.n_d;
	// LIGHT contexts: the verifier's per-claim transparents are O(arity) mults — no
	// eq-tensor expansion, no O(2^n_x) term anywhere in this function.
	let ctxs = claim_contexts_from_dims(dims, stmt.parity, stmt, true)?;
	let k = ctxs.len();
	let pcs = build_pcs(vkm)?;

	let t = Instant::now();
	// Phase 0 (P0.1): observe the full VKM blob, then the statement, before mu.
	IPVerifierChannel::<B128>::observe_many(transcript, &vkm.to_elems());
	IPVerifierChannel::<B128>::observe_many(transcript, &stmt.to_elems());

	// ---- Phase A. ----
	let sums: Vec<B128> = ctxs.iter().map(|c| c.sum).collect();
	let out_a = binius_ip::sumcheck::batch_verify::<B128, _>(dims.n_t, 3, &sums, transcript)
		.map_err(|e| anyhow::anyhow!("phase A batch_verify: {e}"))?;
	let mu = out_a.batch_coeff;
	let e_a = out_a.eval;
	let mut rho = out_a.challenges.clone();
	rho.reverse(); // low-first

	let finish_evals: Vec<B128> = IPVerifierChannel::<B128>::recv_many(transcript, 3 * k)
		.map_err(|e| anyhow::anyhow!("phase A finish evals: {e}"))?;
	let prods: Vec<B128> = finish_evals
		.chunks_exact(3)
		.map(|abg| abg[0] * abg[1] * abg[2])
		.collect();
	ensure!(
		evaluate_univariate(&prods, mu) == e_a,
		"phase A recombination failed (Horner check): claimed sums are inconsistent"
	);

	let d_vals: Vec<B128> =
		IPVerifierChannel::<B128>::recv_many(transcript, SHIFT_VARIANT_COUNT * k)
			.map_err(|e| anyhow::anyhow!("phase A d values: {e}"))?;
	for (l, ctx) in ctxs.iter().enumerate() {
		let g_l = finish_evals[3 * l + 2];
		let gammas = ctx.tr.gammas();
		let mut rhs = B128::ZERO;
		for o in 0..SHIFT_VARIANT_COUNT {
			rhs += gammas[o] * d_vals[l * SHIFT_VARIANT_COUNT + o];
		}
		ensure!(g_l == rhs, "phase A (G) recombination failed for claim {l}");
	}

	// ---- digest_D (bound by FS before tau), then Phase C. ----
	let digest_d: Digest = transcript
		.message()
		.read()
		.map_err(|e| anyhow::anyhow!("digest_D read: {e}"))?;
	let tau: B128 = IPVerifierChannel::<B128>::sample(transcript);
	ensure!(
		tau.val() >= (1u128 << (dims.n_a + 2)).into(),
		"tau hit the committed-value range (honest abort)"
	);
	let d_root: B128 = IPVerifierChannel::<B128>::recv_one(transcript)
		.map_err(|e| anyhow::anyhow!("phase C d_root: {e}"))?;
	ensure!(d_root != B128::ZERO, "phase C root denominator is zero");
	let leaf_claim = fracaddcheck::verify::<B128, _>(
		n_l + 2,
		FracAddEvalClaim {
			num_eval: B128::ZERO,
			den_eval: d_root,
			point: Vec::new(),
		},
		transcript,
	)
	.map_err(|e| anyhow::anyhow!("phase C fracaddcheck: {e}"))?;
	let pi = leaf_claim.point;
	ensure!(pi.len() == n_l + 2, "phase C leaf point arity");
	let (pi_lo, pi_hi) = pi.split_at(n_l);

	let vvals: Vec<B128> = IPVerifierChannel::<B128>::recv_many(transcript, 4)
		.map_err(|e| anyhow::anyhow!("phase C corner values: {e}"))?;
	let (v00, v10, v01, m_pi) = (vvals[0], vvals[1], vvals[2], vvals[3]);

	// Transparent leaf-claim decompositions (spec section 2, Phase C).
	let zero = B128::ZERO;
	let one = B128::ONE;
	let eq_blk = [
		eq_point(&[zero, zero], pi_hi),
		eq_point(&[one, zero], pi_hi),
		eq_point(&[zero, one], pi_hi),
		eq_point(&[one, one], pi_hi),
	];
	let mut rho_ext = rho.clone();
	rho_ext.resize(n_l, zero);
	let num_expected =
		(eq_blk[0] + eq_blk[1] + eq_blk[2]) * eq_point(pi_lo, &rho_ext) + eq_blk[3] * m_pi;
	ensure!(
		num_expected == leaf_claim.num_eval,
		"phase C numerator transparent mismatch at pi"
	);
	// col_11 = iota~(pi[..n_a]) + iota~'(pi[n_a..n_l]): the MLE of a linear map is the
	// map itself ([REV S3], plainly linear under the aligned basis). The X/Y/U columns
	// carry their kappa tags INSIDE the committed values, so col_00/10/01 are exactly
	// the prover-sent corner values.
	let col_emb: B128 = pi_lo
		.iter()
		.enumerate()
		.map(|(k_bit, &p)| p * beta(k_bit))
		.sum();
	let den_expected = eq_blk[0] * (tau + v00)
		+ eq_blk[1] * (tau + v10)
		+ eq_blk[2] * (tau + v01)
		+ eq_blk[3] * (tau + col_emb);
	ensure!(
		den_expected == leaf_claim.den_eval,
		"phase C denominator transparent mismatch at pi"
	);

	// ---- Phase B (10K + 1 claims). ----
	let phi: B128 = IPVerifierChannel::<B128>::sample(transcript);
	let (points, evals) = phase_b_claims(dims, &ctxs, &finish_evals, &d_vals, pi_lo, m_pi);
	let combined = evaluate_univariate(&evals, phi);
	let out_b = binius_ip::sumcheck::verify::<B128, _>(dims.n_d, 2, combined, transcript)
		.map_err(|e| anyhow::anyhow!("phase B verify: {e}"))?;
	let e_b = out_b.eval;
	let mut sigma = out_b.challenges.clone();
	sigma.reverse(); // low-first
	let m: B128 = IPVerifierChannel::<B128>::recv_one(transcript)
		.map_err(|e| anyhow::anyhow!("phase B m: {e}"))?;
	let mut w_eq_sigma = B128::ZERO;
	let mut phi_pow = B128::ONE;
	for p in &points {
		w_eq_sigma += phi_pow * eq_point(p, &sigma);
		phi_pow *= phi;
	}
	ensure!(m * w_eq_sigma == e_b, "phase B eq_ind final check failed");
	let transcript_s = t.elapsed().as_secs_f64();

	// ---- Final check (W2): ONE merged batched opening of [M_VK, M_D]. ----
	// rho_c after m + corner values (both claims FS-bound before the opening);
	// vk_digest is FS-bound by P0.1, digest_D was observed before tau.
	let t = Instant::now();
	let rho_c: Vec<B128> = IPVerifierChannel::<B128>::sample_many(transcript, 2);
	let claim_vk = eq_point(&[zero, zero], &rho_c) * v00
		+ eq_point(&[one, zero], &rho_c) * v10
		+ eq_point(&[zero, one], &rho_c) * v01;
	let mut vk_point = pi_lo.to_vec();
	vk_point.extend_from_slice(&rho_c);
	let vk_digest = vkm.vk_digest_typed()?;
	crate::merged::verify_merged_openings(
		&pcs.params,
		&pcs.scheme,
		&[vk_digest, digest_d],
		&vk_point,
		claim_vk,
		&sigma,
		m,
		transcript,
	)
	.map_err(|e| anyhow::anyhow!("merged [M_VK, M_D] opening: {e}"))?;
	let open_merged_s = t.elapsed().as_secs_f64();

	Ok(VerifyTimings2 {
		transcript_s,
		open_merged_s,
		total_s: t_total.elapsed().as_secs_f64(),
	})
}

/// Which final-check mode a discharge instance runs (task item 6: STEP 1 stays
/// available as a regression/native mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DischargeMode {
	/// STEP 1: no commitments; the verifier rebuilds M_D from the CS natively.
	Step1Native,
	/// STEP 2: committed M_VK/M_D; Phase C + two BaseFold openings; CS-free verifier.
	Step2Committed,
}

/// Verifier-side key material for [`discharge_verify_any`].
pub enum VerifierKey<'a> {
	Native(&'a ConstraintSystem),
	Committed(&'a DischargeVkm),
}

/// Mode-dispatched verification (returns total seconds).
pub fn discharge_verify_any<C: Challenger>(
	key: VerifierKey<'_>,
	stmt: &DischargeStatement,
	transcript: &mut VerifierTranscript<C>,
) -> anyhow::Result<f64> {
	match key {
		VerifierKey::Native(cs) => {
			let t = crate::discharge::discharge_verify(cs, stmt, transcript)?;
			Ok(t.total_s)
		}
		VerifierKey::Committed(vkm) => {
			let t = discharge_verify_step2(vkm, stmt, transcript)?;
			Ok(t.total_s)
		}
	}
}
