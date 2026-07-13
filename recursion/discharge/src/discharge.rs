//! STEP-1 discharge orchestration: Phase 0 (statement absorption + structural asserts),
//! Phase A (batched degree-3 sumcheck, K claims -> rho), Phase B (bivariate sumcheck
//! [W_eq, M_D] -> (sigma, m)), and the native final check (one-pass M_D rebuild).
//!
//! Standalone claim intake per spec P0.4: the (c_l, v_l) pairs ARE the statement and
//! are observed explicitly before mu is sampled.

use std::time::Instant;

use anyhow::{Context, bail, ensure};
use binius_core::constraint_system::ConstraintSystem;
use binius_field::Field;
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{batch::batch_prove_and_write_evals, bivariate_product::BivariateProductSumcheckProver, prove_single},
};
use binius_math::{
	FieldBuffer,
	multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
	univariate::evaluate_univariate,
};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::Challenger};
use binius_verifier::{config::B128, protocols::shift::SHIFT_VARIANT_COUNT};

use crate::{
	cubic::CubicProductSumcheckProver,
	table::{
		Claim, ClaimTransparents, Histograms, TermTable, assemble_m_d, build_histograms,
		claim_context, evaluate_d_g, eq_point, extract_table, m_d_points,
		native_term_sum,
	},
};

/// Discharge statement (STEP-1 VKM flavor, spec P0.1):
/// {cs_digest, N, N_pad, parity, K} plus the K (c_l, v_l) claims.
#[derive(Debug, Clone)]
pub struct DischargeStatement {
	pub cs_digest: [u8; 32],
	pub n_terms: usize,
	pub n_pad: usize,
	pub parity: bool,
	pub claims: Vec<Claim>,
}

impl DischargeStatement {
	/// Builds the statement from the CS metadata and captured claims, enforcing the
	/// P0.4 coverage rule: the statement claims are exactly the sink's records, in
	/// sink order (`claims` here must be the full capture sink of the batch).
	pub fn new(table: &TermTable, claims: Vec<Claim>) -> anyhow::Result<Self> {
		ensure!(!claims.is_empty(), "empty claim batch");
		for (i, c) in claims.iter().enumerate() {
			ensure!(
				c.point.len() == table.dims.arity,
				"claim {i} arity {} != shape arity {} (P0.4)",
				c.point.len(),
				table.dims.arity
			);
		}
		Ok(Self {
			cs_digest: table.cs_digest,
			n_terms: table.dims.n_terms,
			n_pad: table.dims.n_pad,
			parity: table.dims.parity,
			claims,
		})
	}

	/// Canonical statement encoding, observed into the transcript before mu (P0.1).
	pub(crate) fn to_elems(&self) -> Vec<B128> {
		let mut elems = Vec::with_capacity(6 + self.claims.len() * 64);
		let lo = u128::from_le_bytes(self.cs_digest[..16].try_into().expect("16 bytes"));
		let hi = u128::from_le_bytes(self.cs_digest[16..].try_into().expect("16 bytes"));
		elems.push(B128::new(lo));
		elems.push(B128::new(hi));
		elems.push(B128::new(self.n_terms as u128));
		elems.push(B128::new(self.n_pad as u128));
		elems.push(B128::new(self.parity as u128));
		elems.push(B128::new(self.claims.len() as u128));
		for c in &self.claims {
			elems.extend_from_slice(&c.point);
			elems.push(c.value);
		}
		elems
	}
}

/// P0.4 coverage assert for the standalone path: the discharge statement must contain
/// exactly the claims captured for the batch, in sink order. `k_expected` is the
/// number of leaf verifications performed; a truncated (or padded) claim list is
/// rejected — an undischarged v_l would leave that leaf's check_eval vacuous.
pub fn statement_from_capture(
	table: &TermTable,
	captured: Vec<Claim>,
	k_expected: usize,
) -> anyhow::Result<DischargeStatement> {
	ensure!(
		captured.len() == k_expected,
		"P0.4 coverage violated: capture sink holds {} claims but the batch performed {} leaf verifications",
		captured.len(),
		k_expected
	);
	DischargeStatement::new(table, captured)
}

/// Per-claim context: transparents + parity-corrected Phase-A sum (spec 1.3).
pub(crate) struct ClaimCtx {
	pub(crate) tr: ClaimTransparents,
	pub(crate) sum: B128,
}

/// Builds the per-claim contexts from the STATEMENT dims/parity (never the CS): the
/// STEP-2 verifier calls this with VKM-derived dims only. `light` skips the eq-tensor
/// expansion (verifier side: no O(2^n_x) work; prover side needs the tensors for the
/// Phase-A columns).
pub(crate) fn claim_contexts_from_dims(
	dims: &crate::table::ShapeDims,
	parity: bool,
	stmt: &DischargeStatement,
	light: bool,
) -> anyhow::Result<Vec<ClaimCtx>> {
	stmt.claims
		.iter()
		.enumerate()
		.map(|(i, claim)| {
			let tr = if light {
				crate::table::ClaimTransparents::new_light(dims, claim)
					.with_context(|| format!("claim {i}"))?
			} else {
				claim_context(dims, claim).with_context(|| format!("claim {i}"))?
			};
			// sums[l] := v_l + parity * w_d(c_l) — derived from THE claim value element.
			let mut sum = claim.value;
			if parity {
				sum += tr.dummy_weight;
			}
			Ok(ClaimCtx { tr, sum })
		})
		.collect()
}

fn claim_contexts(
	table: &TermTable,
	stmt: &DischargeStatement,
) -> anyhow::Result<Vec<ClaimCtx>> {
	claim_contexts_from_dims(&table.dims, stmt.parity, stmt, false)
}

/// Verifier-side contexts: no eq tensors (STEP-1 verify uses none of them; the final
/// native pass builds its own rho tensor).
fn claim_contexts_light(
	table: &TermTable,
	stmt: &DischargeStatement,
) -> anyhow::Result<Vec<ClaimCtx>> {
	claim_contexts_from_dims(&table.dims, stmt.parity, stmt, true)
}

/// Structural preconditions shared by prover and verifier (P0.1/P0.2/P0.3):
/// the statement metadata must equal what the CS in hand derives.
fn check_statement_against_cs(
	table: &TermTable,
	stmt: &DischargeStatement,
) -> anyhow::Result<()> {
	ensure!(
		stmt.cs_digest == table.cs_digest,
		"P0.2 cs_digest mismatch: statement {} vs CS {}",
		hex(&stmt.cs_digest),
		hex(&table.cs_digest)
	);
	ensure!(
		stmt.n_terms == table.dims.n_terms
			&& stmt.n_pad == table.dims.n_pad
			&& stmt.parity == table.dims.parity,
		"P0.1 table metadata mismatch: statement (N={}, N_pad={}, parity={}) vs CS (N={}, N_pad={}, parity={})",
		stmt.n_terms,
		stmt.n_pad,
		stmt.parity,
		table.dims.n_terms,
		table.dims.n_pad,
		table.dims.parity,
	);
	ensure!(!stmt.claims.is_empty(), "empty claim batch");
	Ok(())
}

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Adds `scale * eq(w, point)` into `w_eq` for all w, exploiting that the point's
/// address coordinates beyond some prefix are ZERO (making the tensor vanish outside
/// the prefix range) and that the two selector coordinates are exact 0/1 bits
/// (making the tensor vanish outside one block). Exact for the m_d_points family.
/// The tensor expansion and axpy are parallel above a small-size cutoff (W1).
pub(crate) fn axpy_point_tensor(w_eq: &mut [B128], n_a: usize, point: &[B128], scale: B128) {
	debug_assert_eq!(point.len(), n_a + 2);
	let c0 = point[n_a];
	let c1 = point[n_a + 1];
	debug_assert!(c0 == B128::ZERO || c0 == B128::ONE);
	debug_assert!(c1 == B128::ZERO || c1 == B128::ONE);
	let block = usize::from(c0 == B128::ONE) + 2 * usize::from(c1 == B128::ONE);
	let base = block << n_a;
	let mut j = n_a;
	while j > 0 && point[j - 1] == B128::ZERO {
		j -= 1;
	}
	if j >= 15 {
		// Parallel path: packed tensor expansion + parallel axpy.
		let prefix_tensor = binius_math::multilinear::eq::eq_ind_partial_eval::<B128>(&point[..j]);
		crate::packed::axpy_dense_par(&mut w_eq[base..base + prefix_tensor.as_ref().len()], prefix_tensor.as_ref(), scale);
	} else {
		let prefix_tensor = eq_ind_partial_eval_scalars(&point[..j]);
		for (slot, t_val) in w_eq[base..base + prefix_tensor.len()]
			.iter_mut()
			.zip(prefix_tensor.iter())
		{
			*slot += scale * *t_val;
		}
	}
}

/// Timing breakdown of a discharge proving run.
#[derive(Debug, Default, Clone)]
pub struct ProveTimings {
	pub build_columns_s: f64,
	pub phase_a_s: f64,
	pub histograms_s: f64,
	pub phase_b_s: f64,
	pub total_s: f64,
}

/// Proves the discharge of `stmt.claims` against the CS's term table.
///
/// Transcript layout (one FS transcript, order fixed):
/// observe(statement) | Phase A: [mu; rounds; 3K finish evals] | 8K d values |
/// phi | Phase B rounds | m.
pub fn discharge_prove<C: Challenger>(
	cs: &ConstraintSystem,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
) -> anyhow::Result<ProveTimings> {
	let t_total = Instant::now();
	let table = extract_table(cs)?;
	check_statement_against_cs(&table, stmt)?;
	let ctxs = claim_contexts(&table, stmt)?;
	let dims = &table.dims;
	let k = ctxs.len();

	// Phase 0: statement absorption (P0.1) — before mu is sampled.
	IPProverChannel::<B128>::observe_many(transcript, &stmt.to_elems());

	// Phase A: build the three virtual columns per claim (packed parallel gathers, W1)
	// and run the batched cubic sumcheck via the upstream driver.
	let t = Instant::now();
	let mut provers = Vec::with_capacity(k);
	for ctx in &ctxs {
		let prover = CubicProductSumcheckProver::new(
			[
				crate::packed::gather_column::<crate::packed::PB, _>(
					&table.terms,
					&ctx.tr.x_tensor,
					dims.n_t,
					|t| t.x as usize,
				),
				crate::packed::gather_column::<crate::packed::PB, _>(
					&table.terms,
					&ctx.tr.y_tensor,
					dims.n_t,
					|t| t.y as usize,
				),
				crate::packed::gather_column::<crate::packed::PB, _>(
					&table.terms,
					&ctx.tr.g_tab,
					dims.n_t,
					|t| t.u as usize,
				),
			],
			ctx.sum,
		);
		provers.push(prover);
	}
	let build_columns_s = t.elapsed().as_secs_f64();

	let t = Instant::now();
	let output = batch_prove_and_write_evals(provers, transcript);
	// output.challenges are already reversed by the driver -> low-first rho.
	let rho = output.challenges.clone();
	ensure!(rho.len() == dims.n_t, "phase A round count");
	let phase_a_s = t.elapsed().as_secs_f64();

	// Histograms at rho (one table pass), then the 8K d values.
	let t = Instant::now();
	let hist = build_histograms(&table, &rho);
	let mut d_vals = Vec::with_capacity(k * SHIFT_VARIANT_COUNT);
	for ctx in &ctxs {
		let points = m_d_points(dims, &ctx.tr);
		for point in &points[2..] {
			// g-point: first 11 coordinates address D_g within its block.
			d_vals.push(evaluate_d_g(&hist.d_g, &point[..crate::table::N_U]));
		}
	}
	IPProverChannel::<B128>::send_many(transcript, &d_vals);
	let histograms_s = t.elapsed().as_secs_f64();

	// Phase B: one bivariate sumcheck over [W_eq, M_D].
	let t = Instant::now();
	let phi: B128 = IPProverChannel::<B128>::sample(transcript);

	// Canonical claim-eval order: per claim l: a_l, b_l, d_{l,0..8}.
	let mut points = Vec::with_capacity(10 * k);
	let mut evals = Vec::with_capacity(10 * k);
	for (l, ctx) in ctxs.iter().enumerate() {
		let claim_points = m_d_points(dims, &ctx.tr);
		let finish = &output.multilinear_evals[l];
		ensure!(finish.len() == 3, "cubic finish arity");
		evals.push(finish[0]);
		evals.push(finish[1]);
		evals.extend_from_slice(&d_vals[l * SHIFT_VARIANT_COUNT..(l + 1) * SHIFT_VARIANT_COUNT]);
		points.extend(claim_points);
	}
	let combined = evaluate_univariate(&evals, phi);

	// W_eq[w] = sum_i phi^i eq(w, p_i). Each claim point has, by construction, a
	// zero tail up to the selector coordinates and a 0/1 selector pair, so its eq
	// tensor is supported on one block prefix (spec Phase B note: g-points cost only
	// 2^11 each). The trailing-zero scan below is exact for any point.
	let mut w_eq = vec![B128::ZERO; 1 << dims.n_d];
	let mut phi_pow = B128::ONE;
	for p in &points {
		axpy_point_tensor(&mut w_eq, dims.n_a, p, phi_pow);
		phi_pow *= phi;
	}
	let m_d = crate::packed::assemble_m_d_packed::<crate::packed::PB>(dims, &hist);
	let prover_b = BivariateProductSumcheckProver::new(
		[FieldBuffer::<crate::packed::PB>::from_values(&w_eq), m_d],
		combined,
	);
	let out_b = prove_single(prover_b, transcript);
	// prove_single challenges are in round order (high-to-low binding).
	let mut sigma = out_b.challenges.clone();
	sigma.reverse();
	ensure!(sigma.len() == dims.n_d, "phase B round count");
	// Send m = M~_D(sigma) — the verifier computes W~_eq(sigma) itself.
	let m = out_b.multilinear_evals[1];
	IPProverChannel::<B128>::send_one(transcript, m);
	let phase_b_s = t.elapsed().as_secs_f64();

	Ok(ProveTimings {
		build_columns_s,
		phase_a_s,
		histograms_s,
		phase_b_s,
		total_s: t_total.elapsed().as_secs_f64(),
	})
}

/// Timing breakdown of a discharge verification run.
#[derive(Debug, Default, Clone)]
pub struct VerifyTimings {
	pub transcript_s: f64,
	pub final_native_s: f64,
	pub total_s: f64,
}

/// Verifies a STEP-1 discharge. The verifier holds the CS (it rebuilds M_D natively
/// for the final check), the statement, and the transcript.
pub fn discharge_verify<C: Challenger>(
	cs: &ConstraintSystem,
	stmt: &DischargeStatement,
	transcript: &mut VerifierTranscript<C>,
) -> anyhow::Result<VerifyTimings> {
	let t_total = Instant::now();
	let table = extract_table(cs)?;
	check_statement_against_cs(&table, stmt)?;
	let ctxs = claim_contexts_light(&table, stmt)?;
	let dims = &table.dims;
	let k = ctxs.len();

	let t = Instant::now();
	// Phase 0: statement absorption (P0.1).
	IPVerifierChannel::<B128>::observe_many(transcript, &stmt.to_elems());

	// Phase A verification.
	let sums: Vec<B128> = ctxs.iter().map(|c| c.sum).collect();
	let out_a = binius_ip::sumcheck::batch_verify::<B128, _>(dims.n_t, 3, &sums, transcript)
		.map_err(|e| anyhow::anyhow!("phase A batch_verify: {e}"))?;
	let mu = out_a.batch_coeff;
	let e_a = out_a.eval;
	let mut rho = out_a.challenges.clone();
	rho.reverse(); // low-first

	// (i) recv the 3K finish evals in driver order; (ii) Horner check.
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

	// (iii) recv the 8K d values; per-claim (G) recombination: g_l == sum_o gamma_{l,o} d_{l,o}.
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

	// Phase B verification.
	let phi: B128 = IPVerifierChannel::<B128>::sample(transcript);
	let mut points = Vec::with_capacity(10 * k);
	let mut evals = Vec::with_capacity(10 * k);
	for (l, ctx) in ctxs.iter().enumerate() {
		let claim_points = m_d_points(dims, &ctx.tr);
		evals.push(finish_evals[3 * l]);
		evals.push(finish_evals[3 * l + 1]);
		evals.extend_from_slice(&d_vals[l * SHIFT_VARIANT_COUNT..(l + 1) * SHIFT_VARIANT_COUNT]);
		points.extend(claim_points);
	}
	let combined = evaluate_univariate(&evals, phi);

	let out_b = binius_ip::sumcheck::verify::<B128, _>(dims.n_d, 2, combined, transcript)
		.map_err(|e| anyhow::anyhow!("phase B verify: {e}"))?;
	let e_b = out_b.eval;
	let mut sigma = out_b.challenges.clone();
	sigma.reverse(); // low-first

	let m: B128 = IPVerifierChannel::<B128>::recv_one(transcript)
		.map_err(|e| anyhow::anyhow!("phase B m: {e}"))?;

	// W~_eq(sigma) computed by the verifier itself: the K*O(|point|) eq work.
	let mut w_eq_sigma = B128::ZERO;
	let mut phi_pow = B128::ONE;
	for p in &points {
		w_eq_sigma += phi_pow * eq_point(p, &sigma);
		phi_pow *= phi;
	}
	ensure!(m * w_eq_sigma == e_b, "phase B eq_ind final check failed");
	let transcript_s = t.elapsed().as_secs_f64();

	// STEP-1 final check: native one-pass rebuild of M_D from the CS at rho,
	// then M~_D(sigma) must equal m.
	let t = Instant::now();
	let hist: Histograms = build_histograms(&table, &rho);
	let m_d = assemble_m_d(dims, &hist);
	let m_native = evaluate(&m_d, &sigma);
	ensure!(
		m_native == m,
		"final native M_D check failed: prover's m does not match the CS-derived table"
	);
	let final_native_s = t.elapsed().as_secs_f64();

	Ok(VerifyTimings {
		transcript_s,
		final_native_s,
		total_s: t_total.elapsed().as_secs_f64(),
	})
}

/// Cross-validation gate (spec/TASK item 2): natively evaluate the term-list sum at a
/// captured claim point and compare with the captured v_l. Run this BEFORE any
/// sumcheck work — it is the make-or-break correctness gate for table extraction.
pub fn cross_validate_claim(table: &TermTable, claim: &Claim) -> anyhow::Result<()> {
	let tr = claim_context(&table.dims, claim)?;
	let native = native_term_sum(table, &tr);
	if native != claim.value {
		bail!(
			"table-extraction cross-validation FAILED: term sum {native:?} != captured monster value {:?}",
			claim.value
		);
	}
	Ok(())
}
