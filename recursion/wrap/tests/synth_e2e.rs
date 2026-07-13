//! Fast integrated-wrap E2E + negative suite on K = 3 tiny synthetic AND-only leaves
//! (monster-discharge synth shape: N = 15, N_pad = 16, odd parity, n_d = 13).
//!
//! One test fn amortizes the setup/prove; the negatives all tamper the SAME honest
//! artifact (verification is cheap).
//!
//! Run: cargo test --release --test synth_e2e -- --nocapture --test-threads=1

use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::StdChallenger;
use binius_recursion_wrap::integrated::{IntegratedProver, IntegratedVerifier};
use binius_recursion_discharge::{
	discharge::DischargeStatement,
	step2::{Step2Tamper, discharge_prove_step2, discharge_prove_step2_on_table, discharge_verify_step2},
	synth::{synth_cs, synth_witness},
	table::{claim_context, extract_table, native_term_sum},
};
use rand::{SeedableRng, rngs::StdRng};

const K: usize = 3;

#[test]
fn synth_integrated_e2e_and_negatives() -> anyhow::Result<()> {
	// ---- setup + honest prove (shared by all checks below). ----
	let cs_raw = synth_cs(0);
	let verifier = IntegratedVerifier::setup(cs_raw.clone(), K, 1)?;
	let cs = verifier.leaf_constraint_system().clone();
	let witnesses: Vec<_> = (0..K)
		.map(|i| synth_witness(&cs_raw, 1000 + i as u64))
		.collect::<Result<_, _>>()?;
	let publics: Vec<Vec<binius_core::word::Word>> =
		witnesses.iter().map(|w| w.public().to_vec()).collect();

	let prover = IntegratedProver::setup(&verifier)?;
	let (proof, pt) = prover.prove(witnesses.clone(), StdRng::seed_from_u64(7))?;
	eprintln!(
		"[synth] prove: leaves {:?} wrap_finish {:.2}s discharge {:.2}s total {:.2}s; artifact {} B transcript + {} B sidecar",
		pt.leaf_proves_s,
		pt.wrap_finish_s,
		pt.discharge.total_s,
		pt.total_s,
		proof.transcript.len(),
		proof.monster_values.len() * 16,
	);

	// ---- POSITIVE: integrated verify (substituted; no O(N) monster anywhere). ----
	let vt = verifier.verify(&publics, &proof)?;
	eprintln!(
		"[synth] verify OK: leaf replays {:?} outer+opening {:.3}s discharge {:.3}s total {:.3}s",
		vt.leaf_replays_s, vt.outer_and_opening_s, vt.discharge.total_s, vt.total_s,
	);

	// ---- POSITIVE: Phase-1b baseline mode on the same artifact (native monster). ----
	// Also asserts native monster values == the artifact's substituted values.
	let bt = verifier.verify_baseline_native(&publics, &proof)?;
	eprintln!(
		"[synth] baseline OK: leaf replays {:?} outer+opening {:.3}s total {:.3}s",
		bt.leaf_replays_s, bt.outer_and_opening_s, bt.total_s,
	);

	// ---- (a)+(d) forged substituted value: sidecar v+1 at leaf 1, everything else
	// honest (the discharge segment certifies the honest values — this IS the S1
	// "check_eval consumes an uncertified copy" attempt: in this design the forged
	// copy and the certified copy cannot diverge without SOME check failing, because
	// the verifier wires the SAME consumed element into publics AND statement). ----
	{
		let mut forged = proof.clone();
		forged.monster_values[1] += binius_recursion_wrap::B128::new(1);
		let err = verifier
			.verify(&publics, &forged)
			.expect_err("forged monster value must be rejected");
		let msg = format!("{err:#}");
		assert!(
			msg.contains("outer verify"),
			"expected the outer stage to reject the uncompensated forgery, got: {msg}"
		);
		eprintln!("[synth] (a/d) forged sidecar value rejected: {err:#}");
	}

	// ---- (c) coverage: one leaf's value omitted from the artifact -> P0.4. ----
	{
		let mut truncated = proof.clone();
		truncated.monster_values.pop();
		let err = verifier
			.verify(&publics, &truncated)
			.expect_err("truncated sidecar must be rejected");
		let msg = format!("{err:#}");
		assert!(msg.contains("P0.4"), "expected P0.4 coverage error, got: {msg}");
		eprintln!("[synth] (c) coverage violation rejected: {err:#}");
	}

	// ---- (e) tampered combined-opening byte -> FRI/Merkle reject (regression). ----
	// Locate the wrap segment end by measuring the discharge segment on a standalone
	// transcript (its byte length is shape-deterministic), then flip a byte in the
	// tail of the wrap segment = inside the combined ZK BaseFold opening.
	{
		let stmt = verifier.capture_statement(&publics, &proof)?;
		let discharge_len = {
			let mut pt2 = ProverTranscript::new(StdChallenger::default());
			discharge_prove_step2(&cs, verifier.vkm(), &stmt, &mut pt2)?;
			pt2.finalize().len()
		};
		let wrap_end = proof.transcript.len() - discharge_len;
		eprintln!(
			"[synth] segment split: wrap {} B + discharge {} B (standalone-measured)",
			wrap_end, discharge_len
		);
		let mut tampered = proof.clone();
		tampered.transcript[wrap_end - 64] ^= 0x01;
		let err = verifier
			.verify(&publics, &tampered)
			.expect_err("tampered combined-opening byte must be rejected");
		let msg = format!("{err:#}");
		assert!(
			msg.contains("Merkle") || msg.contains("FRI"),
			"expected an FRI/Merkle rejection, got: {msg}"
		);
		eprintln!("[synth] (e) tampered combined-opening byte rejected: {err:#}");

		// Same regression aimed at the discharge's own openings (tail of transcript).
		let mut tampered2 = proof.clone();
		let len = tampered2.transcript.len();
		tampered2.transcript[len - 64] ^= 0x01;
		let err2 = verifier
			.verify(&publics, &tampered2)
			.expect_err("tampered discharge-opening byte must be rejected");
		eprintln!("[synth] (e') tampered discharge-opening byte rejected: {err2:#}");

		// ---- (b) consistent lie vs the COMMITTED table: tamper the term table (flip
		// term 0's shift-amount bit), recompute all K claim values against the tampered
		// table (so Phases A and B are self-consistent), prove the discharge over the
		// tampered table with honest machinery. Only the committed-VK binding (the
		// M_VK opening against vk_digest) can reject — and must. Standalone-path
		// discharge (statement explicit, spec P0.4 standalone), same shape/VKM as the
		// integrated flow. ----
		{
			let honest_table = extract_table(&cs)?;
			let mut tampered_table = honest_table.clone();
			tampered_table.terms[0].u ^= 1;
			let forged_claims: Vec<_> = stmt
				.claims
				.iter()
				.map(|c| {
					let tr = claim_context(&tampered_table.dims, c)?;
					Ok(binius_recursion_discharge::table::Claim {
						point: c.point.clone(),
						value: native_term_sum(&tampered_table, &tr),
					})
				})
				.collect::<anyhow::Result<_>>()?;
			// The lie must actually change at least one value.
			assert!(
				forged_claims.iter().zip(&stmt.claims).any(|(f, h)| f.value != h.value),
				"table tamper did not change any claim value"
			);
			let stmt_forged = DischargeStatement {
				claims: forged_claims,
				..stmt.clone()
			};
			// The adversary keeps the M_VK re-commit honest (else its own T1 digest
			// check would already refuse); the false corner values then make the
			// verifier-side M_VK opening claim wrong, and the opening rejects.
			let mut pt3 = ProverTranscript::new(StdChallenger::default());
			discharge_prove_step2_on_table(
				&tampered_table,
				&honest_table,
				verifier.vkm(),
				&stmt_forged,
				&mut pt3,
				Step2Tamper::None,
			)?;
			let bytes3 = pt3.finalize();
			let mut vt3 = VerifierTranscript::new(StdChallenger::default(), bytes3);
			let err3 = discharge_verify_step2(verifier.vkm(), &stmt_forged, &mut vt3)
				.expect_err("consistent-lie discharge must be rejected by the committed-VK binding");
			let msg3 = format!("{err3:#}");
			assert!(
				msg3.contains("merged") || msg3.contains("phase C"),
				"expected committed-table binding rejection (merged opening / phase C), got: {msg3}"
			);
			assert!(
				!msg3.contains("phase A") && !msg3.contains("phase B"),
				"lie was supposed to be invisible to phases A/B, got: {msg3}"
			);
			eprintln!("[synth] (b) consistent lie rejected by committed-table binding: {err3:#}");
		}
	}

	Ok(())
}

/// The "validate wall": a prover that forges a monster value at the replay (witness
/// fill) without compensating witness_eval cannot even PRODUCE an artifact — the
/// outer witness violates the in-circuit `witness_eval * monster_eval == eval`
/// constraint and `ConstraintSystemPadded::validate` panics inside the upstream
/// `ZKWrappedProverChannel::finish`. This is why the "forged value reaches the
/// discharge" adversary requires a compensated leaf prover (spec section 0, S1
/// discussion), whose forged value then flows into the verifier's sink and is
/// rejected by discharge Phase A.
#[test]
#[should_panic]
fn synth_forged_replay_value_cannot_prove() {
	use binius_recursion_wrap::substituting::ValueSource;
	let cs_raw = synth_cs(0);
	let verifier = IntegratedVerifier::setup(cs_raw.clone(), K, 1).unwrap();
	let witnesses: Vec<_> = (0..K)
		.map(|i| synth_witness(&cs_raw, 2000 + i as u64).unwrap())
		.collect();
	let prover = IntegratedProver::setup(&verifier).unwrap();
	let _ = prover.prove_with_source(
		witnesses,
		StdRng::seed_from_u64(9),
		ValueSource::ComputeTampered {
			delta: binius_recursion_wrap::B128::new(1),
			at: 1,
		},
	);
}
