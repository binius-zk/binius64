//! INDEPENDENT-VERIFIER attacks (written by the reviewing session, NOT the builder).
//! Each targets the S1 single-source / positional-binding / discharge-binding claims
//! from an angle the builder's own suite does not cover.
//!
//! Run: cargo test --release --test independent_attacks -- --nocapture --test-threads=1
//!
//! Shape: K=3 tiny synthetic AND-only leaves (fast). Distinct witnesses -> distinct
//! claim points -> (whp) distinct monster values, which is what makes the positional
//! attacks meaningful (asserted at runtime).

use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use binius_recursion_wrap::integrated::{IntegratedProof, IntegratedProver, IntegratedVerifier};
use binius_recursion_discharge::{step2::discharge_prove_step2, synth::{synth_cs, synth_witness}};
use rand::{SeedableRng, rngs::StdRng};

const K: usize = 3;

fn build_proof(seed_base: u64, rng_seed: u64) -> anyhow::Result<(IntegratedVerifier, Vec<Vec<binius_core::word::Word>>, IntegratedProof)> {
	let cs_raw = synth_cs(0);
	let verifier = IntegratedVerifier::setup(cs_raw.clone(), K, 1)?;
	let cs = verifier.leaf_constraint_system().clone();
	let witnesses: Vec<_> = (0..K)
		.map(|i| synth_witness(&cs_raw, seed_base + i as u64))
		.collect::<Result<_, _>>()?;
	let _ = cs;
	let publics: Vec<Vec<binius_core::word::Word>> =
		witnesses.iter().map(|w| w.public().to_vec()).collect();
	let prover = IntegratedProver::setup(&verifier)?;
	let (proof, _pt) = prover.prove(witnesses, StdRng::seed_from_u64(rng_seed))?;
	Ok((verifier, publics, proof))
}

/// Standalone-measured discharge-segment byte length (shape-deterministic; equals the
/// shared-transcript discharge tail). Used to locate the wrap/discharge boundary.
fn discharge_len(verifier: &IntegratedVerifier, publics: &[Vec<binius_core::word::Word>], proof: &IntegratedProof) -> anyhow::Result<usize> {
	let stmt = verifier.capture_statement(publics, proof)?;
	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove_step2(verifier.leaf_constraint_system(), verifier.vkm(), &stmt, &mut pt)?;
	Ok(pt.finalize().len())
}

/// ATTACK A — PERMUTED SIDECAR. Swap monster_values[0] <-> [1]. Each leaf's certified
/// value is now bound to the WRONG leaf. If S1 positional binding holds, verification
/// must reject. (A verifier that only checked "each supplied value is *some* real
/// monster value" would accept this; the swap keeps the multiset identical.)
#[test]
fn attack_a_permuted_sidecar() -> anyhow::Result<()> {
	let (verifier, publics, proof) = build_proof(3000, 11)?;
	// Sanity: the honest artifact verifies.
	verifier.verify(&publics, &proof)?;

	assert_ne!(
		proof.monster_values[0], proof.monster_values[1],
		"precondition: leaf 0 and leaf 1 must have distinct monster values for the swap to bite"
	);
	eprintln!(
		"[attack A] honest monster values: v0={:?} v1={:?} v2={:?}",
		proof.monster_values[0].val(),
		proof.monster_values[1].val(),
		proof.monster_values[2].val()
	);

	let mut permuted = proof.clone();
	permuted.monster_values.swap(0, 1);
	let err = verifier
		.verify(&publics, &permuted)
		.expect_err("permuted sidecar (multiset-preserving) MUST be rejected");
	eprintln!("[attack A] permuted sidecar rejected: {err:#}");
	Ok(())
}

/// ATTACK C — LEGITIMATE-VALUE SUBSTITUTION. Replace leaf 1's value with leaf 0's
/// (a genuine, table-consistent monster value — just of the WRONG point). This is a
/// sharper single-leaf forgery than v+1: the injected value passes any "is this a
/// plausible monster value" check; only binding value-to-its-own-point can catch it.
#[test]
fn attack_c_legitimate_value_wrong_point() -> anyhow::Result<()> {
	let (verifier, publics, proof) = build_proof(4000, 13)?;
	verifier.verify(&publics, &proof)?;
	assert_ne!(proof.monster_values[0], proof.monster_values[1]);

	let mut forged = proof.clone();
	forged.monster_values[1] = proof.monster_values[0]; // real value, wrong leaf
	let err = verifier
		.verify(&publics, &forged)
		.expect_err("legitimate-but-misbound value MUST be rejected");
	eprintln!("[attack C] legitimate-value-wrong-point rejected: {err:#}");
	Ok(())
}

/// ATTACK B — FOREIGN DISCHARGE SPLICE. Take proof P2's honest wrap segment and append
/// proof P1's honest discharge segment. The verifier reconstructs the discharge STATEMENT
/// from its own capture sink (P2's claims) and observes it into FS; P1's discharge bytes
/// were produced against P1's claims. If the discharge is cryptographically bound to the
/// claims captured in ITS OWN wrap (the structural core of S1 — the statement is NEVER
/// carried in the artifact, only the sidecar values are), the spliced proof must reject.
///
/// Step 1 (replaying P2's honest wrap) is expected to PASS, so the rejection isolates
/// cleanly at the discharge — proving the discharge cannot be transplanted.
#[test]
fn attack_b_foreign_discharge_splice() -> anyhow::Result<()> {
	// Two honest artifacts, same CS shape / K, different witnesses => different claims,
	// different monster values, different discharge (different M_D commitment).
	let (v1, pub1, p1) = build_proof(5000, 17)?;
	let (v2, pub2, p2) = build_proof(6000, 19)?;
	v1.verify(&pub1, &p1)?;
	v2.verify(&pub2, &p2)?;

	// Shape-deterministic boundary (identical for both proofs).
	let dlen1 = discharge_len(&v1, &pub1, &p1)?;
	let dlen2 = discharge_len(&v2, &pub2, &p2)?;
	assert_eq!(dlen1, dlen2, "same shape => identical discharge segment length");
	assert_eq!(p1.transcript.len(), p2.transcript.len(), "same shape => identical total length");
	let wrap_end = p2.transcript.len() - dlen2;

	// The two discharge tails must actually differ (else the splice is a no-op).
	assert_ne!(
		&p1.transcript[wrap_end..],
		&p2.transcript[wrap_end..],
		"the two honest discharge segments must differ"
	);

	// Splice: P2 wrap head || P1 discharge tail.
	let mut spliced_bytes = p2.transcript[..wrap_end].to_vec();
	spliced_bytes.extend_from_slice(&p1.transcript[wrap_end..]);
	assert_eq!(spliced_bytes.len(), p2.transcript.len());
	let spliced = IntegratedProof {
		transcript: spliced_bytes,
		monster_values: p2.monster_values.clone(),
	};

	let err = v2
		.verify(&pub2, &spliced)
		.expect_err("foreign discharge segment MUST NOT verify against P2's captured claims");
	let msg = format!("{err:#}");
	eprintln!("[attack B] foreign discharge splice rejected: {msg}");
	// It must die in the discharge, not the wrap (the wrap head is P2's own honest bytes).
	assert!(
		msg.contains("discharge") || msg.contains("phase") || msg.contains("M_VK") || msg.contains("M_D") || msg.contains("fracaddcheck"),
		"expected a discharge-stage rejection, got: {msg}"
	);
	assert!(
		!msg.contains("outer verify"),
		"the wrap head is P2's honest bytes; it must not fail at the outer stage: {msg}"
	);
	Ok(())
}
