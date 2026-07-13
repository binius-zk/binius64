//! Synthetic reproduction of the discharge's headline scaling claims — no application
//! circuits, no real leaf proofs.
//!
//! The note reports, from real leaf proofs, that the STEP-2 discharge verify is flat in
//! the term count N (its dominant FRI/opening endgame is FRI-log-depth only) and
//! K-independent by construction, while the STEP-1 verify carries one O(N) native table
//! pass. This bin reproduces exactly those shapes on a SIZED synthetic AND-only table
//! with STANDALONE-synthesized claims (spec §P0.4 standalone path: the (c, v) pairs ARE
//! the statement; the discharge machinery is byte-identical to the real-proof path).
//!
//! It sweeps N across a wide range and prints, per size, the STEP-1 verify (grows with
//! N — the native pass) vs the STEP-2 verify (flat — FRI log-depth only).
//!
//! Run: cargo run --release -p binius-recursion-discharge --bin scaling_demo

use std::time::Instant;

use binius_recursion_discharge::{
	discharge::{DischargeStatement, discharge_prove, discharge_verify},
	step2::{discharge_prove_step2, discharge_verify_step2},
	synth::{prepared_synth_table, synth_claims},
	vk::vkgen,
};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::StdChallenger;

const K: usize = 3;
/// Witness-vector width exponent (n_y). Kept fixed so only N varies across the sweep.
const N_Y_LOG: usize = 12;

fn median_ms(mut xs: Vec<f64>) -> f64 {
	xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
	xs[xs.len() / 2] * 1e3
}

fn main() -> anyhow::Result<()> {
	// Wide N sweep. Each step is +2 bits in log2(constraints) => +2 bits in N.
	// Largest here is N_pad = 2^20 (~1M terms), comfortably within a 16 GB laptop; the
	// note's real-proof point is N = 24.5M (N_pad = 2^25) — cited, not reproduced here.
	let log2_constraints = [6usize, 10, 14, 18];

	println!("== synthetic discharge scaling (K={K}, standalone claims, N_y=2^{N_Y_LOG}) ==");
	println!(
		"{:>10} {:>8} {:>6} | {:>12} {:>14} | {:>12} {:>14}",
		"N", "N_pad", "n_d", "ST1 prove(s)", "ST1 verify(ms)", "ST2 prove(s)", "ST2 verify(ms)"
	);

	for &lc in &log2_constraints {
		let (cs, table) = prepared_synth_table(lc, N_Y_LOG, 0)?;
		let dims = table.dims.clone();
		let claims = synth_claims(&dims, &table, 1_000, K)?;
		let stmt = DischargeStatement::new(&table, claims)?;

		// STEP 1 (native final check; verifier holds the CS).
		let t = Instant::now();
		let mut pt1 = ProverTranscript::new(StdChallenger::default());
		discharge_prove(&cs, &stmt, &mut pt1)?;
		let bytes1 = pt1.finalize();
		let st1_prove = t.elapsed().as_secs_f64();
		let mut st1_verify = Vec::new();
		for _ in 0..3 {
			let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes1.clone());
			let t = Instant::now();
			discharge_verify(&cs, &stmt, &mut vt)?;
			vt.finalize().map_err(|e| anyhow::anyhow!("st1 finalize: {e}"))?;
			st1_verify.push(t.elapsed().as_secs_f64());
		}

		// STEP 2 (committed table; CS-free verifier — the flat endgame).
		let (vkm, _vkgen_s) = vkgen(&table)?;
		let t = Instant::now();
		let mut pt2 = ProverTranscript::new(StdChallenger::default());
		discharge_prove_step2(&cs, &vkm, &stmt, &mut pt2)?;
		let bytes2 = pt2.finalize();
		let st2_prove = t.elapsed().as_secs_f64();
		let mut st2_verify = Vec::new();
		for _ in 0..3 {
			let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes2.clone());
			let t = Instant::now();
			discharge_verify_step2(&vkm, &stmt, &mut vt)?;
			vt.finalize().map_err(|e| anyhow::anyhow!("st2 finalize: {e}"))?;
			st2_verify.push(t.elapsed().as_secs_f64());
		}

		println!(
			"{:>10} {:>8} {:>6} | {:>12.3} {:>14.2} | {:>12.3} {:>14.2}",
			dims.n_terms,
			dims.n_pad,
			dims.n_d,
			st1_prove,
			median_ms(st1_verify),
			st2_prove,
			median_ms(st2_verify),
		);
	}

	println!(
		"\nExpected shape: ST1 verify GROWS with N (one O(N) native table pass); ST2 verify \
		 stays ~FLAT (FRI log-depth only) — the note's N-independence claim, reproduced \
		 synthetically. Absolute ms differ from the note (different N range + machine)."
	);
	Ok(())
}
