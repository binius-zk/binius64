//! Phase-0 gate: smallest end-to-end native Binius64 recursion.
//!
//! Uses the upstream ZK-wrap machinery (binius-zk/binius64 @ HEAD):
//!   - inner proof P: a Binius64 word-level constraint system, proven by binius-prover
//!   - outer circuit: the Binius64 IOP verifier symbolically executed into an IronSpartan
//!     constraint system (IronSpartanBuilderChannel, built inside ZKVerifier::setup)
//!   - R: one combined transcript = encrypted inner messages + outer Spartan proof + one
//!     combined BaseFold opening over inner+outer oracles
//!   - the O(N) "monster" evaluation is DEFERRED: carried as a single public (inout) wire of
//!     the outer circuit (compute_public_value), checked natively at the next layer
//!
//! Positive path: R must verify. Negative paths: bit-flipped R and tampered public input
//! must both fail.

use std::time::Instant;

use binius_circuits::sha256::{Compress, State};
use binius_core::{
	constraint_system::{ConstraintSystem, ValueVec},
	word::Word,
};
use binius_field::arch::OptimalPackedB128;
use binius_frontend::{CircuitBuilder, Wire};
use binius_hash::StdHashSuite;
use binius_prover::zk_config::ZKProver;
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::{config::StdChallenger, zk_config::ZKVerifier};
use rand::{SeedableRng, rngs::StdRng};

const LOG_INV_RATE: usize = 1;

type GateZKVerifier = ZKVerifier<StdHashSuite>;
type GateZKProver = ZKProver<OptimalPackedB128, StdHashSuite>;

fn setup_zk(cs: ConstraintSystem) -> anyhow::Result<(GateZKVerifier, GateZKProver)> {
	let verifier = GateZKVerifier::setup(cs, LOG_INV_RATE)?;
	let prover = GateZKProver::setup(verifier.clone())?;
	Ok((verifier, prover))
}

fn create_proof_zk(prover: &GateZKProver, witness: ValueVec) -> anyhow::Result<Vec<u8>> {
	let mut transcript = ProverTranscript::new(StdChallenger::default());
	let mut rng = StdRng::seed_from_u64(0);
	prover.prove(witness, &mut rng, &mut transcript)?;
	Ok(transcript.finalize())
}

fn check_proof_zk(
	verifier: &GateZKVerifier,
	public: &[Word],
	proof_bytes: Vec<u8>,
) -> anyhow::Result<()> {
	let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof_bytes);
	verifier.verify(public, &mut transcript)?;
	transcript.finalize()?;
	Ok(())
}

/// Toy inner circuit: witnesses x, y; public c_and = x & y and m_lo = low64(x * y).
/// Exercises both constraint lanes (AND + intmul) with a handful of constraints.
fn toy_circuit() -> (ConstraintSystem, ValueVec) {
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
	let xv: u64 = 0xdead_beef_cafe_f00d;
	let yv: u64 = 0x0123_4567_89ab_cdef;
	w[x] = Word(xv);
	w[y] = Word(yv);
	w[c_and] = Word(xv & yv);
	w[m_lo] = Word(xv.wrapping_mul(yv));
	circuit.populate_wire_witness(&mut w).unwrap();

	(circuit.constraint_system().clone(), w.into_value_vec())
}

/// Medium inner circuit: SHA-256 single-block preimage (same as upstream's E2E test).
fn sha256_preimage_circuit() -> (ConstraintSystem, ValueVec) {
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
	for (i, &output) in output.iter().enumerate() {
		w[output] = Word(expected_state[i] as u64);
	}
	circuit.populate_wire_witness(&mut w).unwrap();

	(circuit.constraint_system().clone(), w.into_value_vec())
}

fn run_case(name: &str, cs: ConstraintSystem, witness: ValueVec) -> anyhow::Result<()> {
	println!("\n================ CASE: {name} ================");
	println!(
		"[inner CS] and_constraints={} mul_constraints={} value_vec_len={} public_words={}",
		cs.and_constraints.len(),
		cs.mul_constraints.len(),
		witness.combined_witness().len(),
		witness.public().len(),
	);

	// setup_zk symbolically executes the Binius64 verifier for this CS into the outer
	// IronSpartan circuit (stats surface on the tracing debug line "ZK wrapper circuit stats").
	let t = Instant::now();
	let (zk_verifier, zk_prover) = setup_zk(cs)?;
	println!("[setup] zk setup (incl. symbolic verifier execution + compile): {:?}", t.elapsed());
	println!(
		"[outer CS] padded mul_constraints={} constants={}",
		zk_verifier.outer_iop_verifier().constraint_system().mul_constraints().len(),
		zk_verifier.outer_iop_verifier().constraint_system().constants().len(),
	);

	// Prove: inner Binius64 proof + outer IronSpartan proof, one combined transcript R.
	let t = Instant::now();
	let proof_bytes = create_proof_zk(&zk_prover, witness.clone())?;
	let prove_ms = t.elapsed();
	println!("[prove] combined recursion proof R: {} bytes in {:?}", proof_bytes.len(), prove_ms);

	// GATE positive: R must verify.
	let t = Instant::now();
	check_proof_zk(&zk_verifier, witness.public(), proof_bytes.clone())?;
	println!("[verify] R VERIFIED OK in {:?}", t.elapsed());

	// GATE negative 1: bit-flip R at several offsets -> must fail.
	let offsets = [
		proof_bytes.len() / 7,
		proof_bytes.len() / 3,
		proof_bytes.len() / 2,
		proof_bytes.len().saturating_sub(64),
	];
	for off in offsets {
		let mut tampered = proof_bytes.clone();
		tampered[off] ^= 0x01;
		let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			check_proof_zk(&zk_verifier, witness.public(), tampered)
		}));
		match res {
			Ok(Ok(())) => {
				println!("[tamper] !!! SOUNDNESS FAILURE: bit-flip at {off} ACCEPTED !!!");
				anyhow::bail!("tampered proof accepted at offset {off}");
			}
			Ok(Err(e)) => println!("[tamper] flip@{off}: rejected ({e:.60})"),
			Err(_) => println!("[tamper] flip@{off}: rejected (panic during verify)"),
		}
	}

	// GATE negative 2: correct proof, tampered public input -> must fail.
	let mut bad_public: Vec<Word> = witness.public().to_vec();
	let last = bad_public.len() - 1;
	bad_public[last] = Word(bad_public[last].0 ^ 1);
	let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		check_proof_zk(&zk_verifier, &bad_public, proof_bytes.clone())
	}));
	match res {
		Ok(Ok(())) => {
			println!("[tamper] !!! SOUNDNESS FAILURE: wrong public input ACCEPTED !!!");
			anyhow::bail!("wrong public input accepted");
		}
		Ok(Err(e)) => println!("[tamper] wrong-public: rejected ({e:.60})"),
		Err(_) => println!("[tamper] wrong-public: rejected (panic during verify)"),
	}

	println!("[gate] case {name}: PASS (verified + 5/5 tamper rejections)");
	Ok(())
}

fn main() -> anyhow::Result<()> {
	tracing_subscriber::fmt()
		.with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
		.init();

	let (cs, witness) = toy_circuit();
	run_case("toy (1 AND + 1 MUL lane)", cs, witness)?;

	let (cs, witness) = sha256_preimage_circuit();
	run_case("sha256-preimage (1 compression)", cs, witness)?;

	println!("\nALL GATE CASES PASSED");
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The recursion gate E2E (positive verify + all tamper rejections) for both the toy
	/// AND+MUL leaf and the SHA-256 leaf.
	#[test]
	fn gate_toy_and_sha256_recursion() {
		let (cs, w) = toy_circuit();
		run_case("toy (1 AND + 1 MUL lane)", cs, w).expect("toy gate");
		let (cs, w) = sha256_preimage_circuit();
		run_case("sha256-preimage (1 compression)", cs, w).expect("sha256 gate");
	}
}
