// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::collections::HashSet;

use binius_circuits::{
	hash_based_sig::{
		MESSAGE_LEN, Message,
		aggregate::{MultiSigWires, circuit_xmss_multisig},
		xmss::{XmssPublicKey, XmssSignature, generate_signature},
	},
	sha256::{State, populate_message_block, sha256_compress},
};
use binius_core::{
	constraint_system::{
		AndConstraint, BmulConstraint, ConstraintSystem, ImulConstraint, InoutSegment,
		ValueSegment, ValueVec,
	},
	word::Word,
};
use binius_field::{BinaryField128bGhash, Field, Random, arch::OptimalPackedB128};
use binius_frontend::{CircuitBuilder, Options, Wire};
use binius_hash::StdHashSuite;
use binius_prover::{Prover, zk_config::ZKProver};
use binius_transcript::ProverTranscript;
use binius_utils::{DeserializeBytes, SerializeBytes};
use binius_verifier::{Verifier, config::StdChallenger, zk_config::ZKVerifier};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn prove_verify(cs: ConstraintSystem, witness: &ValueVec) {
	const LOG_INV_RATE: usize = 1;

	let verifier = Verifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();

	let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(verifier.clone()).unwrap();

	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	prover.prove(witness, &mut prover_transcript).unwrap();

	let mut verifier_transcript = prover_transcript.into_verifier();
	verifier
		.verify(witness.inout(), &mut verifier_transcript)
		.unwrap();
	verifier_transcript.finalize().unwrap();
}

fn prove_verify_zk(cs: ConstraintSystem, witness: &ValueVec) {
	const LOG_INV_RATE: usize = 1;

	let zk_verifier = ZKVerifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();

	let zk_prover = ZKProver::<OptimalPackedB128, StdHashSuite>::setup(&zk_verifier).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	zk_prover
		.prove(witness, &mut rng, &mut prover_transcript)
		.unwrap();

	let mut verifier_transcript = prover_transcript.into_verifier();
	zk_verifier
		.verify(witness.inout(), &mut verifier_transcript)
		.unwrap();
	verifier_transcript.finalize().unwrap();
}

fn prove_verify_zk_serialized(cs: ConstraintSystem, witness: &ValueVec) {
	const LOG_INV_RATE: usize = 1;

	let zk_verifier = ZKVerifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();
	let zk_prover = ZKProver::<OptimalPackedB128, StdHashSuite>::setup(&zk_verifier).unwrap();

	// Round-trip both through serialization, mimicking save-to-disk / reload-in-a-fresh-process.
	// The reloaded prover (which reuses the deserialized KeyCollection and recomputes the cheaper
	// derived state) must produce a proof the reloaded verifier accepts.
	let mut prover_bytes = Vec::new();
	zk_prover.serialize(&mut prover_bytes).unwrap();
	let zk_prover =
		ZKProver::<OptimalPackedB128, StdHashSuite>::deserialize(prover_bytes.as_slice()).unwrap();

	let mut verifier_bytes = Vec::new();
	zk_verifier.serialize(&mut verifier_bytes).unwrap();
	let zk_verifier = ZKVerifier::<StdHashSuite>::deserialize(verifier_bytes.as_slice()).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	zk_prover
		.prove(witness, &mut rng, &mut prover_transcript)
		.unwrap();

	let mut verifier_transcript = prover_transcript.into_verifier();
	zk_verifier
		.verify(witness.inout(), &mut verifier_transcript)
		.unwrap();
	verifier_transcript.finalize().unwrap();
}

fn sha256_preimage_circuit() -> (ConstraintSystem, ValueVec) {
	// Use the test-vector for SHA256 single block message: "abc".
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
	let state_out = sha256_compress(&circuit, state, input);

	// Mask to only low 32-bit.
	let mask32 = circuit.add_constant(Word::MASK_32);
	for (actual_x, expected_x) in state_out.0.iter().zip(output) {
		circuit.assert_eq("eq", circuit.band(*actual_x, mask32), expected_x);
	}

	let circuit = circuit.build();
	let mut w = circuit.new_witness_filler();

	// Populate the input message for the compression function.
	populate_message_block(&mut w, &input, preimage);

	for (i, &output) in output.iter().enumerate() {
		w[output] = Word(expected_state[i] as u64);
	}
	circuit.populate_wire_witness(&mut w).unwrap();

	(circuit.constraint_system().clone(), w.into_value_vec())
}

#[test]
fn test_prove_verify_sha256_preimage() {
	let (cs, witness) = sha256_preimage_circuit();
	prove_verify(cs, &witness);
}

/// Builds a circuit that computes the 7th power of an input GHASH-field element `x` using four
/// `bmul` gates, and constrains the result to a public output element. This exercises the BinMul
/// reduction end-to-end in both the IOP prover and verifier.
fn binmul_seventh_power_circuit() -> (ConstraintSystem, ValueVec) {
	let circuit = CircuitBuilder::new();
	// Input element x = (x_lo, x_hi), a private witness carried by a (lo, hi) word pair.
	let x_lo = circuit.add_witness();
	let x_hi = circuit.add_witness();
	// Public output element y = (y_lo, y_hi).
	let y_lo = circuit.add_inout();
	let y_hi = circuit.add_inout();

	// Compute x^7 with four GHASH-field multiplications: x^2, x^3, x^6, x^7.
	let (x2_lo, x2_hi) = circuit.bmul(x_lo, x_hi, x_lo, x_hi);
	let (x3_lo, x3_hi) = circuit.bmul(x2_lo, x2_hi, x_lo, x_hi);
	let (x6_lo, x6_hi) = circuit.bmul(x3_lo, x3_hi, x3_lo, x3_hi);
	let (x7_lo, x7_hi) = circuit.bmul(x6_lo, x6_hi, x_lo, x_hi);

	circuit.assert_eq("x7_lo", x7_lo, y_lo);
	circuit.assert_eq("x7_hi", x7_hi, y_hi);

	let circuit = circuit.build();
	let mut w = circuit.new_witness_filler();

	// A random input element and its 7th power, computed independently via GHASH-field arithmetic.
	let mut rng = StdRng::seed_from_u64(0);
	let x = BinaryField128bGhash::random(&mut rng);
	let x7 = x.pow([7u64]);

	let x_val = u128::from(x);
	let y_val = u128::from(x7);
	w[x_lo] = Word(x_val as u64);
	w[x_hi] = Word((x_val >> 64) as u64);
	w[y_lo] = Word(y_val as u64);
	w[y_hi] = Word((y_val >> 64) as u64);

	circuit.populate_wire_witness(&mut w).unwrap();

	(circuit.constraint_system().clone(), w.into_value_vec())
}

/// The 7th-power circuit uses BMUL constraints, exercising the BinMul reduction that the SHA-256
/// tests (AND-only) never reach.
#[test]
fn test_prove_verify_binmul_seventh_power() {
	let (cs, witness) = binmul_seventh_power_circuit();
	assert!(cs.n_bmul_constraints() > 0, "circuit should have BMUL constraints");
	prove_verify(cs.clone(), &witness);
	prove_verify_zk(cs, &witness);
}

/// Builds a circuit whose AND, IMUL and BMUL constraint counts are all non-powers of two, so every
/// reduction runs over a constraint axis wider than the operand columns it is handed.
///
/// Each `band` gate contributes one AND constraint, each `imul` gate one IMUL constraint, and each
/// `bmul` gate one BMUL constraint. The caller asserts the counts are not powers of two rather
/// than relying on that silently, since the frontend decides how a gate lowers.
fn non_power_of_two_constraint_circuit() -> (ConstraintSystem, ValueVec) {
	const N_AND_GATES: usize = 3;
	const N_IMUL_GATES: usize = 3;
	const N_BMUL_GATES: usize = 3;

	let circuit = CircuitBuilder::new();

	// `z == x & y`, with all three words private.
	let and_wires = (0..N_AND_GATES)
		.map(|i| {
			let x = circuit.add_witness();
			let y = circuit.add_witness();
			let z = circuit.add_witness();
			circuit.assert_eq(format!("and_{i}"), circuit.band(x, y), z);
			(x, y, z)
		})
		.collect::<Vec<_>>();

	// `(hi, lo) = a * b`. `force_commit` makes both product words hidden, so the IMUL operands read
	// them from the witness rather than folding them away.
	let imul_wires = (0..N_IMUL_GATES)
		.map(|_| {
			let a = circuit.add_witness();
			let b = circuit.add_witness();
			let (hi, lo) = circuit.imul(a, b);
			circuit.force_commit(hi);
			circuit.force_commit(lo);
			(a, b)
		})
		.collect::<Vec<_>>();

	// GHASH-field products `(c_lo, c_hi) = (a_lo, a_hi) * (b_lo, b_hi)`, likewise committed.
	let bmul_wires = (0..N_BMUL_GATES)
		.map(|_| {
			let a_lo = circuit.add_witness();
			let a_hi = circuit.add_witness();
			let b_lo = circuit.add_witness();
			let b_hi = circuit.add_witness();
			let (c_lo, c_hi) = circuit.bmul(a_lo, a_hi, b_lo, b_hi);
			circuit.force_commit(c_lo);
			circuit.force_commit(c_hi);
			[a_lo, a_hi, b_lo, b_hi]
		})
		.collect::<Vec<_>>();

	let circuit = circuit.build();
	let mut w = circuit.new_witness_filler();

	for (i, &(x, y, z)) in and_wires.iter().enumerate() {
		let x_val = Word(0x0123_4567_89AB_CDEF ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
		let y_val = Word(0xFEDC_BA98_7654_3210 ^ (i as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
		w[x] = x_val;
		w[y] = y_val;
		w[z] = x_val & y_val;
	}

	for (i, &(a, b)) in imul_wires.iter().enumerate() {
		w[a] = Word(0xDEAD_BEEF_CAFE_BABE ^ (i as u64).wrapping_mul(0x1234_5678_9ABC_DEF0));
		w[b] = Word(0x0F0F_0F0F_0F0F_0F0F ^ (i as u64).wrapping_mul(0xA5A5_A5A5_A5A5_A5A5));
	}

	let mut rng = StdRng::seed_from_u64(0);
	for &[a_lo, a_hi, b_lo, b_hi] in &bmul_wires {
		let a = u128::from(BinaryField128bGhash::random(&mut rng));
		let b = u128::from(BinaryField128bGhash::random(&mut rng));
		w[a_lo] = Word(a as u64);
		w[a_hi] = Word((a >> 64) as u64);
		w[b_lo] = Word(b as u64);
		w[b_hi] = Word((b >> 64) as u64);
	}

	// The gates derive the AND, product and GHASH-product outputs, so the filled witness satisfies
	// every constraint.
	circuit.populate_wire_witness(&mut w).unwrap();

	(circuit.constraint_system().clone(), w.into_value_vec())
}

/// A circuit whose AND, IMUL and BMUL constraint counts are all non-powers of two proves and
/// verifies. The constraint system keeps those true counts, and the prover's operand columns stop
/// at the last constraint; each reduction rounds its own constraint axis up to a power of two and
/// reads the rows past a column's end as zero, and both sides derive their sumcheck variable
/// counts by rounding the same true count up.
#[test]
fn test_prove_verify_non_power_of_two_constraint_counts() {
	let (cs, witness) = non_power_of_two_constraint_circuit();
	for (name, n_constraints) in [
		("AND", cs.n_and_constraints()),
		("IMUL", cs.n_imul_constraints()),
		("BMUL", cs.n_bmul_constraints()),
	] {
		assert!(
			n_constraints > 0 && !n_constraints.is_power_of_two(),
			"{name} constraint count is {n_constraints}, which must be non-zero and not a power \
			 of two for this test to exercise padding"
		);
	}
	prove_verify(cs.clone(), &witness);
	prove_verify_zk(cs, &witness);
}

/// Dropping the operand columns' padding rows moves nothing on the wire.
///
/// Every constraint type's `Default` has empty operands, so its row is identically zero — exactly
/// the padding row the prover used to materialize. Padding each constraint list up to a power of
/// two therefore reconstructs the old column shape without touching the prover, and the two must
/// produce the same proof byte for byte. The reduction runs over the same axis either way, since
/// `log2_ceil` of a count and of that count rounded up agree.
#[test]
fn test_padded_constraint_lists_produce_an_identical_proof() {
	let (cs, witness) = non_power_of_two_constraint_circuit();

	let mut padded_cs = cs.clone();
	padded_cs
		.and_constraints
		.resize(cs.n_and_constraints().next_power_of_two(), AndConstraint::default());
	padded_cs
		.imul_constraints
		.resize(cs.n_imul_constraints().next_power_of_two(), ImulConstraint::default());
	padded_cs
		.bmul_constraints
		.resize(cs.n_bmul_constraints().next_power_of_two(), BmulConstraint::default());
	padded_cs.validate().unwrap();

	assert_eq!(prove_to_bytes(cs, &witness), prove_to_bytes(padded_cs, &witness));
}

/// Proves `witness` against `cs` and returns the proof bytes.
fn prove_to_bytes(cs: ConstraintSystem, witness: &ValueVec) -> Vec<u8> {
	const LOG_INV_RATE: usize = 1;

	let verifier = Verifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();
	let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(verifier).unwrap();

	let mut transcript = ProverTranscript::new(StdChallenger::default());
	prover.prove(witness, &mut transcript).unwrap();
	transcript.finalize().to_vec()
}

/// A SHA-256 circuit uses only AND constraints, so its constraint system has zero IMUL
/// constraints. This locks in that the prover and verifier skip the IntMul reduction entirely
/// (rather than padding up to a dummy IMUL constraint); see `IOPProver::prove` /
/// `IOPVerifier::verify`.
#[test]
fn test_prove_verify_zero_imul_constraints() {
	let (cs, witness) = sha256_preimage_circuit();
	assert_eq!(cs.n_imul_constraints(), 0, "SHA-256 circuit should have no IMUL constraints");
	prove_verify(cs.clone(), &witness);
	prove_verify_zk(cs, &witness);
}

/// Builds a circuit whose linear constraints lower to ZERO constraints: `n_xor` from `bxor` gates,
/// `n_rotr` from `rotr` gates (whose ZERO constraints carry a *shifted* operand), and `n_and` AND
/// constraints from `band` gates.
///
/// Gate fusion is off: it inlines a linear definition into the gate that consumes it, which would
/// leave no linear constraint to lower. The remaining passes are off so the counts are exactly what
/// the gates emit.
fn zero_constraint_circuit(
	n_xor: usize,
	n_rotr: usize,
	n_and: usize,
) -> (ConstraintSystem, ValueVec) {
	let builder = CircuitBuilder::with_opts(Options {
		enable_gate_fusion: false,
		enable_common_subexpression_elimination: false,
		enable_dead_code_elimination: false,
		enable_algebraic_folding: false,
		..Options::default()
	});
	let a = builder.add_witness();
	let b = builder.add_witness();

	// Each `bxor` emits one linear constraint, which the option lowers to a ZERO constraint.
	let mut acc = a;
	for _ in 0..n_xor {
		acc = builder.bxor(acc, b);
		builder.force_commit(acc);
	}

	// Each `rotr` likewise emits one linear constraint, whose ZERO constraint names a shifted
	// value rather than a plain one.
	for i in 0..n_rotr {
		let rotated = builder.rotr(acc, 5 + i as u32);
		builder.force_commit(rotated);
	}

	// Each `band` emits one AND constraint.
	for _ in 0..n_and {
		let and_out = builder.band(a, b);
		builder.force_commit(and_out);
	}

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	w[a] = Word(0x0123_4567_89AB_CDEF);
	w[b] = Word(0xFEDC_BA98_7654_3210);
	circuit.populate_wire_witness(&mut w).unwrap();

	let cs = circuit.constraint_system().clone();
	assert_eq!(cs.n_zero_constraints(), n_xor + n_rotr);
	assert_eq!(cs.n_and_constraints(), n_and);
	assert!(
		cs.zero_constraints
			.iter()
			.any(|c| c.val().iter().any(|svi| !svi.is_unshifted())),
		"the fixture must emit a ZERO constraint with a shifted operand"
	);
	let witness = w.into_value_vec();
	cs.verify(&witness).unwrap();
	(cs, witness)
}

/// The Zero reduction discharges ZERO constraints end to end. The reduction sends nothing itself;
/// its claim rides along in the shift reduction's batch, so this exercises the whole path from the
/// constraint system through the key collection to the final evaluation check.
///
/// The ZERO set is the larger of the two, so the reduction's constraint point runs past the BitAnd
/// output point and draws a fresh challenge. Neither count is a power of two.
#[test]
fn test_prove_verify_zero_constraints() {
	let (cs, witness) = zero_constraint_circuit(3, 2, 3);
	assert_eq!(cs.log_zero_constraints(), Some(3));
	assert_eq!(cs.log_and_constraints(), Some(2));
	prove_verify(cs.clone(), &witness);
	prove_verify_zk(cs, &witness);
}

/// The ZERO set smaller than the AND set, so the reduction's constraint point is a strict prefix
/// of the BitAnd output point and draws nothing of its own.
#[test]
fn test_prove_verify_fewer_zero_than_and_constraints() {
	let (cs, witness) = zero_constraint_circuit(1, 1, 5);
	assert_eq!(cs.log_zero_constraints(), Some(1));
	assert_eq!(cs.log_and_constraints(), Some(3));
	prove_verify(cs, &witness);
}

/// Builds a circuit whose public segment is wider than its hidden one: `n_inout` input/output
/// values against a handful of private ones.
///
/// Each inout value is asserted equal to the same AND output, which lowers to a ZERO constraint
/// and commits nothing further, so the hidden segment stays at three words however many inout
/// values the circuit declares. The optimization passes are off so that the repeated assertions
/// survive as distinct constraints.
fn public_heavy_circuit(n_inout: usize) -> (ConstraintSystem, ValueVec) {
	let builder = CircuitBuilder::with_opts(Options {
		enable_gate_fusion: false,
		enable_common_subexpression_elimination: false,
		enable_dead_code_elimination: false,
		enable_algebraic_folding: false,
		..Options::default()
	});
	let a = builder.add_witness();
	let b = builder.add_witness();
	let and_out = builder.band(a, b);

	let inout = (0..n_inout)
		.map(|_| builder.add_inout())
		.collect::<Vec<_>>();
	for &wire in &inout {
		builder.assert_eq("inout_is_and_output", wire, and_out);
	}

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	let (a_val, b_val) = (Word(0x0123_4567_89AB_CDEF), Word(0xFEDC_BA98_7654_3210));
	w[a] = a_val;
	w[b] = b_val;
	for &wire in &inout {
		w[wire] = a_val & b_val;
	}
	circuit.populate_wire_witness(&mut w).unwrap();

	let cs = circuit.constraint_system().clone();
	let witness = w.into_value_vec();
	cs.verify(&witness).unwrap();
	(cs, witness)
}

/// A public segment wider than the hidden one proves and verifies.
///
/// Neither segment is padded to the other's length, so the shift reduction's word-index space
/// spans the *wider* of the two — here the public one. Both sides size that space from
/// `log_segment_words`, and the hidden half zero-extends up to it, so the sumcheck draws word-index
/// challenges the hidden segment alone would not have called for. Every other circuit in this file
/// has the hidden segment wider, which is the case the padding used to guarantee.
#[test]
fn test_prove_verify_public_wider_than_hidden() {
	let (cs, witness) = public_heavy_circuit(300);
	assert!(
		cs.log_public_words(InoutSegment::Public) > cs.log_witness_words(InoutSegment::Public),
		"public segment ({} words) must be wider than the hidden one ({} words) for this test to \
		 exercise the extra word-index challenges",
		cs.n_public_words(InoutSegment::Public),
		cs.n_hidden_words(InoutSegment::Public),
	);
	assert_eq!(
		cs.log_segment_words(InoutSegment::Public),
		cs.log_public_words(InoutSegment::Public)
	);
	prove_verify(cs.clone(), &witness);
	prove_verify_zk(cs, &witness);
}

/// A witness violating one ZERO constraint is rejected. The prover has nothing to send for the
/// Zero reduction, so it claims the constant zero regardless; the shift reduction, running against
/// the committed witness, is what catches the discrepancy.
#[test]
fn test_prove_verify_rejects_violated_zero_constraint() {
	const LOG_INV_RATE: usize = 1;

	let (cs, witness) = zero_constraint_circuit(3, 2, 3);

	// Corrupt a word that only the ZERO constraints read, so a ZERO constraint is the only thing
	// standing between this witness and a valid proof.
	let and_words = cs
		.and_constraints
		.iter()
		.flat_map(|c| &c.0)
		.flatten()
		.map(|svi| svi.value_index)
		.collect::<HashSet<_>>();
	let victim = cs
		.zero_constraints
		.iter()
		.flat_map(|c| &c.0)
		.flatten()
		.map(|svi| svi.value_index)
		// The rebuild below sources the constants from the system, so tampering with one would be
		// undone; the victim has to be a word the caller supplies.
		.find(|index| !and_words.contains(index) && index.segment() != ValueSegment::Constant)
		.expect("some ZERO constraint reads a non-constant word no AND constraint does");

	let mut words = witness.combined_witness().to_vec();
	let victim_word = cs.word_offset(victim);
	words[victim_word] = words[victim_word] ^ Word::ONE;
	let corrupted = cs.value_vec_from_data(
		&words[cs.n_const()..cs.n_public_values()],
		&words[cs.n_public_values()..],
	);
	assert!(cs.verify(&corrupted).is_err());

	let verifier = Verifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();
	let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(verifier.clone()).unwrap();

	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	prover.prove(&corrupted, &mut prover_transcript).unwrap();

	let mut verifier_transcript = prover_transcript.into_verifier();
	assert!(
		verifier
			.verify(corrupted.public(), &mut verifier_transcript)
			.is_err(),
		"a violated ZERO constraint must not verify"
	);
}

#[test]
fn test_zk_prove_verify_sha256_preimage() {
	let (cs, witness) = sha256_preimage_circuit();
	prove_verify_zk(cs, &witness);
}

#[test]
fn test_zk_prove_verify_serialized() {
	let (cs, witness) = sha256_preimage_circuit();
	prove_verify_zk_serialized(cs, &witness);
}

/// Produces a ZK signature-of-knowledge proof over `sign_message`, then verifies it against
/// `verify_message`. Returns whether verification (including transcript finalization) succeeded.
///
/// Signatures of knowledge are only supported by the ZK prover/verifier.
fn sign_verify(
	cs: ConstraintSystem,
	witness: &ValueVec,
	sign_message: Option<&[u8]>,
	verify_message: Option<&[u8]>,
) -> bool {
	const LOG_INV_RATE: usize = 1;

	let zk_verifier = ZKVerifier::<StdHashSuite>::setup(cs, LOG_INV_RATE).unwrap();
	let zk_prover = ZKProver::<OptimalPackedB128, StdHashSuite>::setup(&zk_verifier).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	match sign_message {
		Some(message) => zk_prover
			.prove_sig(witness, message, &mut rng, &mut prover_transcript)
			.unwrap(),
		None => zk_prover
			.prove(witness, &mut rng, &mut prover_transcript)
			.unwrap(),
	}

	let mut verifier_transcript = prover_transcript.into_verifier();
	let verify_ok = match verify_message {
		Some(message) => zk_verifier
			.verify_sig(witness.inout(), message, &mut verifier_transcript)
			.is_ok(),
		None => zk_verifier
			.verify(witness.inout(), &mut verifier_transcript)
			.is_ok(),
	};
	verify_ok && verifier_transcript.finalize().is_ok()
}

#[test]
fn test_signature_of_knowledge_roundtrip() {
	let (cs, witness) = sha256_preimage_circuit();
	// Signing and verifying with the same message succeeds.
	assert!(sign_verify(cs, &witness, Some(b"hello world"), Some(b"hello world")));
}

#[test]
fn test_signature_of_knowledge_wrong_message_fails() {
	let (cs, witness) = sha256_preimage_circuit();
	// A proof signed over one message must not verify against a different message.
	assert!(!sign_verify(cs, &witness, Some(b"hello world"), Some(b"goodbye world")));
}

#[test]
fn test_signature_of_knowledge_missing_message_fails() {
	let (cs, witness) = sha256_preimage_circuit();
	// A signature of knowledge must not verify as a plain proof of knowledge (no message).
	assert!(!sign_verify(cs, &witness, Some(b"hello world"), None));
}

#[test]
fn test_plain_proof_rejects_message() {
	let (cs, witness) = sha256_preimage_circuit();
	// A plain proof of knowledge must not verify when a message is supplied.
	assert!(!sign_verify(cs, &witness, None, Some(b"hello world")));
}

/// An aggregate XMSS verification, with `n_pad` extra public words pinned to zero.
///
/// The padding is inert: every padded word is a constant-zero assertion, and nothing else about
/// the circuit depends on `n_pad`. It exists only to move the public segment's width.
fn xmss_aggregate_circuit(num_signers: usize, n_pad: usize) -> (ConstraintSystem, ValueVec) {
	const EPOCH: u32 = 42;

	let mut rng = StdRng::seed_from_u64(1);
	let mut message: Message = [0u8; MESSAGE_LEN];
	rng.fill_bytes(&mut message);
	let signatures: Vec<(XmssPublicKey, XmssSignature)> = (0..num_signers)
		.map(|_| generate_signature(&mut rng, &message, EPOCH))
		.collect();

	let builder = CircuitBuilder::new();
	let wires = MultiSigWires::new(&builder, num_signers);
	circuit_xmss_multisig(&builder, &wires);

	let zero = builder.add_constant(Word::ZERO);
	let pad = (0..n_pad).map(|_| builder.add_inout()).collect::<Vec<_>>();
	for &wire in &pad {
		builder.assert_eq("pad_is_zero", wire, zero);
	}

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	wires.populate(&mut w, &message, EPOCH, &signatures);
	for &wire in &pad {
		w[wire] = Word::ZERO;
	}
	circuit.populate_wire_witness(&mut w).unwrap();

	let cs = circuit.constraint_system().clone();
	let witness = w.into_value_vec();
	cs.verify(&witness).unwrap();
	(cs, witness)
}

/// A public segment of exactly 2^9 words proves and verifies, plain and ZK.
#[test]
fn test_zk_prove_verify_aggregate_public_segment_at_power_of_two() {
	let (cs, witness) = xmss_aggregate_circuit(1, 151);
	assert_eq!(cs.n_public_words(InoutSegment::Public), 512);
	prove_verify(cs.clone(), &witness);
	prove_verify_zk(cs, &witness);
}

/// One public word more, and the ZK wrapper rejects an honest proof.
///
/// The plain prover accepts the same circuit and witness, so this is the ZK composition alone:
/// verification fails in the outer Spartan check with `IPChannel(InvalidAssert)`. The only
/// difference from the test above is one inert padding word, which takes the public segment from
/// 512 words to 513.
#[test]
fn test_zk_prove_verify_aggregate_public_segment_over_power_of_two() {
	let (cs, witness) = xmss_aggregate_circuit(1, 152);
	assert_eq!(cs.n_public_words(InoutSegment::Public), 513);
	prove_verify(cs.clone(), &witness);
	prove_verify_zk(cs, &witness);
}
