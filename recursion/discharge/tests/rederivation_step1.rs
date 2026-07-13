//! Wall-A validation on upstream/main (segmented value vector, #1554/#1583/#1724/#1585).
//!
//! This is the MAKE-OR-BREAK gate for the re-derived segmented Y-block: it captures the deferred
//! monster claim from the REAL upstream `Verifier::verify` (through the recorder's two-site
//! selection) and checks that the discharge's natively re-derived term-sum reproduces that
//! captured `monster_eval` BIT-FOR-BIT. The old flat `eq(y, r_y)` interpretation fails this; only
//! the segmented `R[y]` (public prefix scaled by `(1+r_seg)`, hidden suffix scaled by `r_seg`,
//! hidden words re-indexed to the cube's high half) reproduces it.
//!
//! It then runs the STEP-1 discharge (native final check, no PCS) end-to-end and the STEP-1
//! negatives. STEP-2 (committed PCS) is gated off while its FRI/commit plumbing is ported to the
//! upstream channel-oriented BaseFold API.

use binius_recursion_discharge::{
	discharge::{DischargeStatement, cross_validate_claim, discharge_prove, discharge_verify},
	leaf::LeafPipeline,
	recorder::verify_and_capture,
	synth::{synth_cs, synth_witness},
	table::{Claim, extract_table, parse_claim},
};
use binius_field::Field;
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::{B128, StdChallenger};

const K: usize = 2;

/// Captures K real deferred monster claims of `synth_cs(variant)` from the upstream verifier.
fn capture_claims(variant: u8) -> anyhow::Result<(binius_core::constraint_system::ConstraintSystem, Vec<Claim>)> {
	let pipeline = LeafPipeline::setup(synth_cs(variant))?;
	let cs = pipeline.verifier.constraint_system().clone();
	let table = extract_table(&cs)?;
	let mut claims = Vec::with_capacity(K);
	for i in 0..K {
		let wit = synth_witness(&cs, 4000 + i as u64)?;
		let public = LeafPipeline::public(&wit);
		let proof = pipeline.prove(&wit)?;
		claims.push(verify_and_capture(&pipeline.verifier, &public, proof, table.dims.arity)?);
	}
	Ok((cs, claims))
}

/// GATE (a) — the make-or-break cross-validate: the re-derived segmented term-sum equals the REAL
/// captured monster value, on genuine upstream leaf proofs.
#[test]
fn crux_cross_validate_segmented_matches_real_capture() {
	let (cs, claims) = capture_claims(0).expect("capture");
	let table = extract_table(&cs).expect("table");
	eprintln!(
		"[crux] shape: N={} N_pad=2^{} parity={} n_x={} n_y(=lw+1)={} lp={} n_pub={} n_a={} n_d={} arity={}",
		table.dims.n_terms, table.dims.n_t, table.dims.parity, table.dims.n_x, table.dims.n_y,
		table.dims.lp, table.dims.n_pub, table.dims.n_a, table.dims.n_d, table.dims.arity,
	);
	assert_eq!(claims.len(), K);
	assert_ne!(claims[0].point, claims[1].point, "distinct claims");
	for (i, c) in claims.iter().enumerate() {
		// The captured claim point ends with r_segment; sanity-check the parse.
		let parsed = parse_claim(&table.dims, c).expect("parse");
		assert_eq!(parsed.r_y.len(), table.dims.n_y - 1, "r_y length = lw");
		cross_validate_claim(&table, c)
			.unwrap_or_else(|e| panic!("CROSS-VALIDATE FAILED for claim {i} (re-derivation wrong): {e}"));
		eprintln!("[crux] claim {i}: native segmented term-sum == captured monster_eval  (v={:?})", c.value);
	}
	eprintln!("[crux] GATE (a) PASS: re-derived segmented Y-block reproduces the real monster_eval bit-for-bit");
}

/// Discriminating control: `r_segment` is load-bearing. Corrupting only the claim point's final
/// (r_segment) coordinate — leaving the captured value intact — must make the native term-sum
/// diverge, so cross-validate rejects. This proves the segmentation is not vacuous.
#[test]
fn crux_r_segment_is_load_bearing() {
	let (cs, claims) = capture_claims(0).expect("capture");
	let table = extract_table(&cs).expect("table");
	let mut corrupted = claims[0].clone();
	let last = corrupted.point.len() - 1; // r_segment is the final coordinate
	corrupted.point[last] += B128::ONE;
	assert!(
		cross_validate_claim(&table, &corrupted).is_err(),
		"flipping r_segment must break cross-validation (segmentation is load-bearing)"
	);
	eprintln!("[crux] control PASS: corrupting r_segment breaks the term-sum (segmentation load-bearing)");
}

/// GATE (b, STEP-1) — the full STEP-1 discharge (Phase 0/A/B + native M_D rebuild) verifies K real
/// captured claims of an odd-parity shape (the w_d parity correction, now carrying the (1+r_seg)
/// factor, is load-bearing here).
#[test]
fn step1_e2e_on_real_captured_claims() {
	let (cs, claims) = capture_claims(0).expect("capture");
	let table = extract_table(&cs).expect("table");
	assert!(table.dims.parity, "synthetic shape must be odd-parity");
	let stmt = DischargeStatement::new(&table, claims).expect("statement");

	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove(&cs, &stmt, &mut pt).expect("step1 prove");
	let bytes = pt.finalize();
	let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes.clone());
	discharge_verify(&cs, &stmt, &mut vt).expect("step1 verify");
	vt.finalize().expect("finalize");
	eprintln!("[crux] GATE (b STEP-1) PASS: STEP-1 discharge E2E on {K} real captured claims ({} bytes)", bytes.len());
}

/// GATE (c, STEP-1) — negatives must reject.
#[test]
fn step1_negatives_reject() {
	let (cs, claims) = capture_claims(0).expect("capture");
	let table = extract_table(&cs).expect("table");
	let stmt = DischargeStatement::new(&table, claims.clone()).expect("statement");
	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove(&cs, &stmt, &mut pt).expect("prove");
	let bytes = pt.finalize();

	// N1 (table-swap): verify against a DIFFERENT CS shape -> P0.2 cs_digest mismatch.
	let (foreign_cs, _) = capture_claims(1).expect("foreign capture");
	assert_ne!(
		extract_table(&foreign_cs).expect("f").cs_digest,
		table.cs_digest,
		"foreign CS must differ"
	);
	let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes.clone());
	let err = discharge_verify(&foreign_cs, &stmt, &mut vt).expect_err("table-swap MUST reject");
	assert!(format!("{err:#}").contains("P0.2"), "expected P0.2 cs_digest check: {err:#}");

	// N2 (tampered claim value): the re-derived term-sum no longer matches -> cross-validate rejects.
	let mut tampered = claims[0].clone();
	tampered.value += B128::ONE;
	assert!(
		cross_validate_claim(&table, &tampered).is_err(),
		"a tampered monster value MUST fail cross-validation"
	);

	// N3 (Phase-A soundness): a statement with a tampered sum is rejected by the discharge verifier
	// (the honest CS histograms bind a_l·b_l·g_l to the TRUE value, contradicting the tampered sum).
	let mut bad_claims = claims;
	bad_claims[0].value += B128::ONE;
	let bad_stmt = DischargeStatement::new(&table, bad_claims).expect("stmt");
	let mut pt2 = ProverTranscript::new(StdChallenger::default());
	discharge_prove(&cs, &bad_stmt, &mut pt2).expect("prover proves whatever sum it is given");
	let bad_bytes = pt2.finalize();
	let mut vt2 = VerifierTranscript::new(StdChallenger::default(), bad_bytes);
	assert!(
		discharge_verify(&cs, &bad_stmt, &mut vt2).is_err(),
		"a tampered Phase-A sum MUST be rejected by the discharge verifier"
	);
	eprintln!("[crux] GATE (c STEP-1) PASS: table-swap, tampered-value, and tampered-sum all reject");
}
