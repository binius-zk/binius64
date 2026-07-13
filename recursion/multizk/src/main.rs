//! Synthetic demo: aggregate K distinct Binius64 leaves into ONE recursion proof.
//!
//! Builds K inner constraint systems entirely from in-tree gadgets (a toy AND+MUL
//! circuit and SHA-256 single-block preimages) — no application data of any kind — and
//! aggregates them with `MultiZKVerifier`/`MultiZKProver`: one outer IronSpartan circuit,
//! one shared transcript, one combined BaseFold opening. Positive path must verify;
//! bit-flips and a wrong public input must all be rejected.
//!
//! Run: cargo run --release -p binius-recursion-multizk

use std::time::Instant;

use binius_circuits::sha256::{Compress, State};
use binius_core::{
	constraint_system::{ConstraintSystem, ValueVec},
	word::Word,
};
use binius_frontend::{CircuitBuilder, Wire};
use binius_recursion_multizk::{MultiZKProver, MultiZKVerifier};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::StdChallenger;
use rand::{SeedableRng, rngs::StdRng};

const LOG_INV_RATE: usize = 1;

/// Toy inner circuit: witnesses x, y; public c_and = x & y and m_lo = low64(x * y).
fn toy_leaf(xv: u64, yv: u64) -> (ConstraintSystem, ValueVec) {
	let circuit = CircuitBuilder::new();
	let x = circuit.add_witness();
	let y = circuit.add_witness();
	let c_and = circuit.add_inout();
	let m_lo = circuit.add_inout();
	let and_xy = circuit.band(x, y);
	circuit.assert_eq("and", and_xy, c_and);
	let (_hi, lo) = circuit.imul(x, y);
	circuit.assert_eq("mul_lo", lo, m_lo);
	let circuit = circuit.build();
	let mut w = circuit.new_witness_filler();
	w[x] = Word(xv);
	w[y] = Word(yv);
	w[c_and] = Word(xv & yv);
	w[m_lo] = Word(xv.wrapping_mul(yv));
	circuit.populate_wire_witness(&mut w).unwrap();
	(circuit.constraint_system().clone(), w.into_value_vec())
}

/// SHA-256 single-block preimage leaf for the message "abc" (known digest), a distinct
/// shape from the toy leaves.
fn sha256_abc_leaf() -> (ConstraintSystem, ValueVec) {
	let mut preimage: [u8; 64] = [0; 64];
	preimage[0..3].copy_from_slice(b"abc");
	preimage[3] = 0x80;
	preimage[63] = 0x18;
	#[rustfmt::skip]
	let expected_state: [u32; 8] = [
		0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223,
		0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad,
	];

	let circuit = CircuitBuilder::new();
	let state = State::iv(&circuit);
	let input: [Wire; 16] = std::array::from_fn(|_| circuit.add_witness());
	let output: [Wire; 8] = std::array::from_fn(|_| circuit.add_inout());
	let compress = Compress::new(&circuit, state, input);
	let mask32 = circuit.add_constant(Word::MASK_32);
	for (actual_x, expected_x) in compress.state_out.0.iter().zip(output) {
		circuit.assert_eq("eq", circuit.band(*actual_x, mask32), expected_x);
	}
	let circuit = circuit.build();
	let mut w = circuit.new_witness_filler();
	compress.populate_m(&mut w, preimage);
	for (i, &o) in output.iter().enumerate() {
		w[o] = Word(expected_state[i] as u64);
	}
	circuit.populate_wire_witness(&mut w).unwrap();
	(circuit.constraint_system().clone(), w.into_value_vec())
}

fn main() -> anyhow::Result<()> {
	tracing_subscriber::fmt()
		.with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
		.init();

	println!("== multizk demo: aggregate K=3 synthetic Binius64 leaves (mixed shapes) into ONE proof ==");
	let leaves = vec![
		toy_leaf(0xdead_beef_cafe_f00d, 0x0123_4567_89ab_cdef),
		toy_leaf(0xa5a5_a5a5_5a5a_5a5a, 0xffff_0000_ffff_0000),
		sha256_abc_leaf(),
	];
	let (css, witnesses): (Vec<ConstraintSystem>, Vec<ValueVec>) = leaves.into_iter().unzip();
	for (i, cs) in css.iter().enumerate() {
		println!(
			"[leaf {i}] and={} mul={} public_words={}",
			cs.and_constraints.len(),
			cs.mul_constraints.len(),
			witnesses[i].public().len(),
		);
	}
	let publics: Vec<Vec<Word>> = witnesses.iter().map(|w| w.public().to_vec()).collect();

	let t = Instant::now();
	let mzk_verifier = MultiZKVerifier::setup(css, LOG_INV_RATE)?;
	println!("[setup] MultiZKVerifier (K symbolic verifier circuits): {:?}", t.elapsed());
	let t = Instant::now();
	let mzk_prover = MultiZKProver::setup(&mzk_verifier)?;
	println!("[setup] MultiZKProver: {:?}", t.elapsed());

	let t = Instant::now();
	let mut pt = ProverTranscript::new(StdChallenger::default());
	mzk_prover.prove(witnesses.clone(), StdRng::seed_from_u64(0), &mut pt)?;
	let proof_bytes = pt.finalize();
	println!(
		"[prove] ONE combined proof for {} leaves: {} bytes in {:?}",
		publics.len(),
		proof_bytes.len(),
		t.elapsed()
	);

	let t = Instant::now();
	{
		let mut vt = VerifierTranscript::new(StdChallenger::default(), proof_bytes.clone());
		mzk_verifier.verify(&publics, &mut vt)?;
		vt.finalize()?;
	}
	println!("[verify] combined proof VERIFIED OK in {:?}", t.elapsed());

	// Negatives: bit-flips at three offsets + a wrong public on leaf 1.
	let mut rejected = 0usize;
	for off in [proof_bytes.len() / 4, proof_bytes.len() / 2, proof_bytes.len() - 100] {
		let mut tampered = proof_bytes.clone();
		tampered[off] ^= 0x01;
		let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			let mut vt = VerifierTranscript::new(StdChallenger::default(), tampered);
			mzk_verifier.verify(&publics, &mut vt).and_then(|_| Ok(vt.finalize()?))
		}));
		match res {
			Ok(Ok(())) => anyhow::bail!("SOUNDNESS FAILURE: bit-flip at {off} accepted"),
			_ => {
				rejected += 1;
				println!("[tamper] flip@{off}: rejected");
			}
		}
	}
	let mut bad_publics = publics.clone();
	let mid = bad_publics[1].len() / 2;
	bad_publics[1][mid] = Word(bad_publics[1][mid].0 ^ 1);
	let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		let mut vt = VerifierTranscript::new(StdChallenger::default(), proof_bytes.clone());
		mzk_verifier.verify(&bad_publics, &mut vt).and_then(|_| Ok(vt.finalize()?))
	}));
	match res {
		Ok(Ok(())) => anyhow::bail!("SOUNDNESS FAILURE: wrong public (leaf 1) accepted"),
		_ => {
			rejected += 1;
			println!("[tamper] wrong-public leaf1: rejected");
		}
	}

	println!("\nPASS: {} leaves (mixed shapes) aggregated into ONE verified proof ({rejected}/4 tampers rejected)", publics.len());
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Aggregate K=3 mixed-shape leaves into one proof: positive verify + a bit-flip and
	/// a wrong-public rejection. Keeps `cargo test -p binius-recursion-multizk` meaningful.
	#[test]
	fn multizk_aggregate_verify_and_tamper() -> anyhow::Result<()> {
		let leaves = vec![
			toy_leaf(0x1111_2222_3333_4444, 0x5555_6666_7777_8888),
			toy_leaf(0x0f0f_0f0f_0f0f_0f0f, 0x00ff_00ff_00ff_00ff),
			sha256_abc_leaf(),
		];
		let (css, witnesses): (Vec<ConstraintSystem>, Vec<ValueVec>) = leaves.into_iter().unzip();
		let publics: Vec<Vec<Word>> = witnesses.iter().map(|w| w.public().to_vec()).collect();

		let verifier = MultiZKVerifier::setup(css, LOG_INV_RATE)?;
		let prover = MultiZKProver::setup(&verifier)?;
		let mut pt = ProverTranscript::new(StdChallenger::default());
		prover.prove(witnesses, StdRng::seed_from_u64(0), &mut pt)?;
		let proof = pt.finalize();

		// Positive.
		{
			let mut vt = VerifierTranscript::new(StdChallenger::default(), proof.clone());
			verifier.verify(&publics, &mut vt)?;
			vt.finalize()?;
		}
		// Bit-flip -> reject.
		{
			let mut bad = proof.clone();
			bad[proof.len() / 2] ^= 0x01;
			let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
				let mut vt = VerifierTranscript::new(StdChallenger::default(), bad);
				verifier.verify(&publics, &mut vt).and_then(|_| Ok(vt.finalize()?))
			}));
			assert!(!matches!(res, Ok(Ok(()))), "bit-flip must be rejected");
		}
		// Wrong public on leaf 0 -> reject.
		{
			let mut bad_pub = publics.clone();
			bad_pub[0][0] = Word(bad_pub[0][0].0 ^ 1);
			let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
				let mut vt = VerifierTranscript::new(StdChallenger::default(), proof.clone());
				verifier.verify(&bad_pub, &mut vt).and_then(|_| Ok(vt.finalize()?))
			}));
			assert!(!matches!(res, Ok(Ok(()))), "wrong public must be rejected");
		}
		Ok(())
	}
}
