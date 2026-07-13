#![cfg(feature = "step2")]
//! Standalone (no real leaf proof) synthetic scaling test: drives the discharge on a
//! SIZED synthetic AND-only table with claims synthesized natively (spec §P0.4 standalone
//! path). Reproduces, at CI-friendly sizes, the note's shapes:
//!   - STEP 1 verify carries one O(N) native table pass (grows with N);
//!   - STEP 2 verify is the flat FRI/opening endgame (CS-free, ~independent of N).
//! Both must close green at every size, and the K claims must be distinct.
//!
//! Run: cargo test --release -p binius-recursion-discharge --test synth_scaling -- --nocapture

use std::time::Instant;

use binius_recursion_discharge::{
	discharge::{DischargeStatement, cross_validate_claim, discharge_prove, discharge_verify},
	step2::{discharge_prove_step2, discharge_verify_step2},
	synth::{prepared_synth_table, synth_claims},
	vk::vkgen,
};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::StdChallenger;

const K: usize = 3;
const N_Y_LOG: usize = 12;

#[test]
fn synth_scaling_step1_and_step2_green() -> anyhow::Result<()> {
	// Two sizes spanning a 16x range in N; both must close.
	for &lc in &[8usize, 12] {
		let (cs, table) = prepared_synth_table(lc, N_Y_LOG, 0)?;
		let dims = table.dims.clone();
		assert!(!dims.parity, "sized shape has even N_pad-N (3*2^L padded to 2^(L+2))");
		let claims = synth_claims(&dims, &table, 7_000, K)?;
		assert_ne!(claims[0].point, claims[1].point, "claims must be distinct");
		assert_ne!(claims[0].value, claims[1].value, "distinct points => distinct values (whp)");

		// The make-or-break gate: each synthesized value IS the native monster term-sum.
		for (i, c) in claims.iter().enumerate() {
			cross_validate_claim(&table, c).unwrap_or_else(|e| panic!("claim {i}: {e}"));
		}

		let stmt = DischargeStatement::new(&table, claims)?;

		// STEP 1.
		let mut pt1 = ProverTranscript::new(StdChallenger::default());
		discharge_prove(&cs, &stmt, &mut pt1)?;
		let bytes1 = pt1.finalize();
		let mut vt1 = VerifierTranscript::new(StdChallenger::default(), bytes1.clone());
		let t = Instant::now();
		discharge_verify(&cs, &stmt, &mut vt1)?;
		let st1_v = t.elapsed().as_secs_f64();
		vt1.finalize().map_err(|e| anyhow::anyhow!("st1 finalize: {e}"))?;

		// STEP 2.
		let (vkm, _) = vkgen(&table)?;
		let mut pt2 = ProverTranscript::new(StdChallenger::default());
		discharge_prove_step2(&cs, &vkm, &stmt, &mut pt2)?;
		let bytes2 = pt2.finalize();
		let mut vt2 = VerifierTranscript::new(StdChallenger::default(), bytes2.clone());
		let t = Instant::now();
		let vt = discharge_verify_step2(&vkm, &stmt, &mut vt2)?;
		let st2_v = t.elapsed().as_secs_f64();
		vt2.finalize().map_err(|e| anyhow::anyhow!("st2 finalize: {e}"))?;

		eprintln!(
			"[scaling] N={:>7} N_pad=2^{:<2} n_d={:<2} | ST1 {} B, verify {:.1} ms | ST2 {} B, verify {:.1} ms (transcript {:.1} + open {:.1})",
			dims.n_terms,
			dims.n_t,
			dims.n_d,
			bytes1.len(),
			st1_v * 1e3,
			bytes2.len(),
			st2_v * 1e3,
			vt.transcript_s * 1e3,
			vt.open_merged_s * 1e3,
		);
	}
	Ok(())
}

/// Negative: a single corrupted claim value must make the discharge reject (proving the
/// standalone path is a genuine check on v = monster_eval(c), not a formality).
#[test]
fn synth_corrupted_value_rejected() -> anyhow::Result<()> {
	use binius_recursion_discharge::B128;
	let (cs, table) = prepared_synth_table(8, N_Y_LOG, 0)?;
	let dims = table.dims.clone();
	let mut claims = synth_claims(&dims, &table, 9_000, K)?;
	claims[1].value += B128::new(1); // corrupt one deferred value
	let stmt = DischargeStatement::new(&table, claims)?;
	let mut pt = ProverTranscript::new(StdChallenger::default());
	// The honest prover builds columns whose sum is the TRUE value, so Phase A cannot
	// close against the corrupted statement sum.
	let prove_res = discharge_prove(&cs, &stmt, &mut pt);
	if prove_res.is_ok() {
		let bytes = pt.finalize();
		let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes);
		let verify_res = discharge_verify(&cs, &stmt, &mut vt).and_then(|_| {
			vt.finalize().map_err(|e| anyhow::anyhow!("finalize: {e}"))
		});
		assert!(verify_res.is_err(), "corrupted claim value MUST be rejected");
	}
	// Either the prover fails to close or the verifier rejects — both are correct.
	Ok(())
}
