//! STEP-2 tests on the tiny synthetic odd-parity AND-only shape (spec 1.3 w_d path,
//! previously unexercised) — the reduced-domain E2E plus every STEP-2 negative:
//!
//! (e) odd-parity positive: STEP-1 AND STEP-2 E2E on real captured claims of a CS with
//!     N_pad - N = 1 (parity correction is load-bearing in Phase A);
//! (T1) VKGEN regeneration audit: byte-identical vk_digest across runs;
//! (b) TamperMdBlock3 port: the (1,1)-block lie in the COMMITTED M_D must now be caught
//!     by Phase C (there is no native rebuild);
//! (c) table-swap: statement discharged against a DIFFERENT CS's vk_digest is rejected
//!     — (c1) honest full VKM of the foreign CS dies at the P0.2 cs_digest structural
//!     check; (c2) a LYING hybrid VKM (main cs_digest, foreign vk_digest) dies because
//!     vk_digest is FS-observed before mu (P0.1): the transcript diverges at Phase A;
//! (d) digest_D mutated at the observation point (tree and openings honest): every
//!     transcript phase passes, the M_D opening's Merkle/FRI verification rejects.

use std::sync::OnceLock;

use binius_core::constraint_system::ConstraintSystem;
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::StdChallenger;
use binius_recursion_discharge::{
	discharge::{
		DischargeStatement, cross_validate_claim, discharge_prove, discharge_verify,
		statement_from_capture,
	},
	leaf::LeafPipeline,
	recorder::verify_and_capture,
	step2::{Step2Tamper, discharge_prove_step2, discharge_prove_step2_tampered, discharge_verify_step2},
	synth::{synth_cs, synth_witness},
	table::{Claim, TermTable, extract_table},
	vk::{DischargeVkm, vkgen},
};

const K: usize = 2;

struct Ctx {
	cs: ConstraintSystem,
	table: TermTable,
	claims: Vec<Claim>,
	vkm: DischargeVkm,
	/// Same-dims foreign shape: different term table, same arity/N/N_pad/parity.
	foreign_cs_digest: [u8; 32],
	foreign_vkm: DischargeVkm,
}

static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> &'static Ctx {
	CTX.get_or_init(|| build_ctx().expect("shared synthetic test context"))
}

fn build_ctx() -> anyhow::Result<Ctx> {
	// Main shape (variant 0).
	let pipeline = LeafPipeline::setup(synth_cs(0))?;
	let cs = pipeline.verifier.constraint_system().clone();
	let table = extract_table(&cs)?;
	eprintln!(
		"[synth] N={} N_pad=2^{} parity={} dims: n_x={} n_y={} n_a={} n_d={} arity={}",
		table.dims.n_terms,
		table.dims.n_t,
		table.dims.parity,
		table.dims.n_x,
		table.dims.n_y,
		table.dims.n_a,
		table.dims.n_d,
		table.dims.arity,
	);
	anyhow::ensure!(table.dims.parity, "synthetic shape must have ODD N_pad - N");

	let mut claims = Vec::with_capacity(K);
	for i in 0..K {
		let wit = synth_witness(&cs, 1000 + i as u64)?;
		let public = LeafPipeline::public(&wit);
		let proof = pipeline.prove(&wit)?;
		claims.push(verify_and_capture(
			&pipeline.verifier,
			&public,
			proof,
			table.dims.arity,
		)?);
	}
	anyhow::ensure!(claims[0].point != claims[1].point, "claims should be distinct");

	let (vkm, vkgen_s) = vkgen(&table)?;
	eprintln!("[synth] vkgen in {vkgen_s:.3}s; vk_digest={}", hex(&vkm.vk_digest));

	// Foreign shape (variant 1): same dims, different table.
	let f_pipeline = LeafPipeline::setup(synth_cs(1))?;
	let f_cs = f_pipeline.verifier.constraint_system().clone();
	let f_table = extract_table(&f_cs)?;
	anyhow::ensure!(f_table.dims == table.dims, "foreign shape must share all dims");
	anyhow::ensure!(f_table.cs_digest != table.cs_digest, "foreign CS must differ");
	anyhow::ensure!(f_table.terms != table.terms, "foreign term table must differ");
	let (foreign_vkm, _) = vkgen(&f_table)?;
	anyhow::ensure!(foreign_vkm.vk_digest != vkm.vk_digest, "foreign vk_digest must differ");

	Ok(Ctx {
		cs,
		table,
		claims,
		vkm,
		foreign_cs_digest: f_table.cs_digest,
		foreign_vkm,
	})
}

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn prove2(
	cs: &ConstraintSystem,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	tamper: Step2Tamper,
) -> anyhow::Result<Vec<u8>> {
	let mut pt = ProverTranscript::new(StdChallenger::default());
	if tamper == Step2Tamper::None {
		discharge_prove_step2(cs, vkm, stmt, &mut pt)?;
	} else {
		discharge_prove_step2_tampered(cs, vkm, stmt, &mut pt, tamper)?;
	}
	Ok(pt.finalize())
}

fn verify2(vkm: &DischargeVkm, stmt: &DischargeStatement, bytes: Vec<u8>) -> anyhow::Result<()> {
	let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes);
	discharge_verify_step2(vkm, stmt, &mut vt)?;
	vt.finalize().map_err(|e| anyhow::anyhow!("transcript finalize: {e}"))?;
	Ok(())
}

/// (e) odd-parity positive, both modes. The cross-validation gate proves the term
/// table reproduces monster_eval bit-for-bit on this shape too.
#[test]
fn odd_parity_positive_step1_and_step2() {
	let ctx = ctx();
	for (i, c) in ctx.claims.iter().enumerate() {
		cross_validate_claim(&ctx.table, c).unwrap_or_else(|e| panic!("claim {i}: {e}"));
	}
	let stmt = statement_from_capture(&ctx.table, ctx.claims.clone(), K).expect("coverage");

	// STEP 1 (native mode regression; exercises the w_d parity correction end-to-end).
	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove(&ctx.cs, &stmt, &mut pt).expect("step1 prove");
	let bytes1 = pt.finalize();
	let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes1.clone());
	discharge_verify(&ctx.cs, &stmt, &mut vt).expect("step1 verify");
	vt.finalize().expect("step1 finalize");
	eprintln!("[odd-parity] STEP-1 E2E OK ({} proof bytes)", bytes1.len());

	// STEP 2 (committed mode; CS-free verifier).
	let bytes2 = prove2(&ctx.cs, &ctx.vkm, &stmt, Step2Tamper::None).expect("step2 prove");
	let mut vt2 = VerifierTranscript::new(StdChallenger::default(), bytes2.clone());
	let vtimings = discharge_verify_step2(&ctx.vkm, &stmt, &mut vt2).expect("step2 verify");
	vt2.finalize().expect("step2 finalize");
	eprintln!(
		"[odd-parity] STEP-2 E2E OK ({} proof bytes; STEP-1 was {}); verify {:.3}s (transcript+C {:.3} | openMerged {:.3})",
		bytes2.len(),
		bytes1.len(),
		vtimings.total_s,
		vtimings.transcript_s,
		vtimings.open_merged_s,
	);
}

/// (T1) VKGEN regeneration audit: vkgen is a pure function of the canonical CS
/// serialization — a second run yields a byte-identical digest (and identical VKM).
#[test]
fn t1_vkgen_regeneration_deterministic() {
	let ctx = ctx();
	let (vkm2, _) = vkgen(&ctx.table).expect("vkgen regeneration");
	assert_eq!(vkm2.vk_digest, ctx.vkm.vk_digest, "vk_digest must be byte-identical");
	assert_eq!(vkm2, ctx.vkm, "full VKM must be identical");
	// And regeneration from a freshly re-extracted table (new prover process model).
	let table2 = extract_table(&ctx.cs).expect("re-extract");
	let (vkm3, _) = vkgen(&table2).expect("vkgen from re-extracted table");
	assert_eq!(vkm3.vk_digest, ctx.vkm.vk_digest);
	eprintln!("[T1] regeneration audit OK: {}", hex(&ctx.vkm.vk_digest));
}

/// (b) TamperMdBlock3 ported to STEP 2: the (1,1)-block lie is invisible to Phases A/B
/// AND to both openings (they are self-consistent with the tampered commitment) — the
/// weighted fracaddcheck is the ONLY thing that catches it now.
#[test]
fn negative_tamper_md_block3_caught_by_phase_c() {
	let ctx = ctx();
	let stmt = DischargeStatement::new(&ctx.table, ctx.claims.clone()).expect("statement");
	let bytes = prove2(&ctx.cs, &ctx.vkm, &stmt, Step2Tamper::MdBlock3)
		.expect("tampered prover must still produce a transcript");
	let err = verify2(&ctx.vkm, &stmt, bytes).expect_err("M_D (1,1)-block lie MUST be rejected");
	let msg = format!("{err:#}");
	eprintln!("[TamperMdBlock3] rejected: {msg}");
	assert!(
		msg.contains("phase C"),
		"MdBlock3 must be caught by Phase C (not the openings): {msg}"
	);
}

/// (c) table-swap. c1: the foreign CS's full VKM fails the P0.2 structural check.
/// c2: a hybrid VKM lying only about vk_digest fails because vk_digest is FS-observed
/// BEFORE mu (P0.1) — the FS streams diverge and Phase A's first round check fires.
#[test]
fn negative_table_swap_rejected() {
	let ctx = ctx();
	let stmt = DischargeStatement::new(&ctx.table, ctx.claims.clone()).expect("statement");
	let bytes = prove2(&ctx.cs, &ctx.vkm, &stmt, Step2Tamper::None).expect("honest prove");

	// c1: full foreign VKM — the statement's cs_digest no longer matches (P0.2).
	let err = verify2(&ctx.foreign_vkm, &stmt, bytes.clone())
		.expect_err("foreign VKM MUST be rejected");
	let msg = format!("{err:#}");
	eprintln!("[table-swap c1] rejected: {msg}");
	assert!(msg.contains("P0.2"), "expected the P0.2 cs_digest check: {msg}");
	assert_eq!(ctx.foreign_vkm.cs_digest, ctx.foreign_cs_digest, "test wiring");

	// c2: hybrid VKM — same cs_digest (P0.2 passes), foreign vk_digest. The lie is
	// caught by the FS binding: mu diverges, so the phase A round-sum check fails.
	let mut hybrid = ctx.vkm.clone();
	hybrid.vk_digest = ctx.foreign_vkm.vk_digest.clone();
	let err = verify2(&hybrid, &stmt, bytes).expect_err("hybrid vk_digest MUST be rejected");
	let msg = format!("{err:#}");
	eprintln!("[table-swap c2] rejected: {msg}");
	assert!(
		msg.contains("phase A"),
		"expected the FS-divergence failure at phase A (vk_digest binding): {msg}"
	);
}

/// (d) digest_D mutated at the observation point: the FS streams stay consistent
/// (the prover absorbed the flipped digest too), every transcript phase passes, and
/// the failure is EXACTLY the M_D opening's Merkle/FRI verification.
#[test]
fn negative_tampered_digest_d_rejected_by_opening() {
	let ctx = ctx();
	let stmt = DischargeStatement::new(&ctx.table, ctx.claims.clone()).expect("statement");
	let bytes = prove2(&ctx.cs, &ctx.vkm, &stmt, Step2Tamper::DigestD)
		.expect("prover with flipped digest observation still produces a transcript");
	let err = verify2(&ctx.vkm, &stmt, bytes).expect_err("flipped digest_D MUST be rejected");
	let msg = format!("{err:#}");
	eprintln!("[DigestD] rejected: {msg}");
	assert!(
		msg.contains("opening") && msg.contains("Merkle"),
		"expected the merged opening's FRI/Merkle rejection, got: {msg}"
	);
	assert!(
		!msg.contains("phase"),
		"failure must NOT be a transcript-phase failure: {msg}"
	);
}
