// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_circuits::sha256::Sha256;
use binius_core::{
	constraint_system::{AndConstraint, ConstraintSystem, MulConstraint, ValueVec},
	verify::verify_constraints,
	word::Word,
};
use binius_field::{AESTowerField8b, BinaryField};
use binius_frontend::{CircuitBuilder, Wire};
use binius_math::{
	BinarySubspace,
	inner_product::{inner_product, inner_product_buffers},
	multilinear::eq::eq_ind_partial_eval,
	univariate::lagrange_evals,
};
use binius_prover::{
	fold_word::fold_words,
	protocols::shift::{OperatorData, build_key_collection, prove, prove_batch},
};
use binius_transcript::ProverTranscript;
use binius_utils::checked_arithmetics::{log2_ceil_usize, strict_log_2};
use binius_verifier::{
	config::{LOG_WORD_SIZE_BITS, StdChallenger},
	protocols::shift::{OperatorData as VerifierOperatorData, check_eval, verify},
};
use itertools::Itertools;
use rand::{SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256 as Sha256Hasher};

pub fn create_sha256_cs_with_witness() -> (ConstraintSystem, ValueVec) {
	let builder = CircuitBuilder::new();
	let max_len: usize = 64; // Maximum message length in bytes

	// Create wires for the SHA256 circuit
	let len = builder.add_witness(); // Actual message length
	let digest = [
		builder.add_inout(), // Expected digest as 4x64-bit words
		builder.add_inout(),
		builder.add_inout(),
		builder.add_inout(),
	];
	let message: Vec<Wire> = (0..max_len.div_ceil(8))
		.map(|_| builder.add_witness())
		.collect();

	// Create the SHA256 circuit
	let sha256 = Sha256::new(&builder, len, digest, message);

	let circuit = builder.build();
	let mut witness_filler = circuit.new_witness_filler();

	// Populate with concrete message: "abc"
	let message_bytes = b"abc";
	sha256.populate_len_bytes(&mut witness_filler, message_bytes.len());
	sha256.populate_message(&mut witness_filler, message_bytes);

	// Calculate SHA256 digest of the message dynamically
	let hash = Sha256Hasher::digest(message_bytes);
	let expected_digest: [u8; 32] = hash.into();
	sha256.populate_digest(&mut witness_filler, expected_digest);

	// Get the witness vector
	circuit.populate_wire_witness(&mut witness_filler).unwrap();

	(circuit.constraint_system().clone(), witness_filler.into_value_vec())
}

pub fn create_concat_cs_with_witness() -> (ConstraintSystem, ValueVec) {
	use binius_circuits::{concat::concat, fixed_byte_vec::ByteVec};

	let builder = CircuitBuilder::new();

	// Create terms: "Hello" + " " + "World!"
	let terms: Vec<ByteVec> = (0..3)
		.map(|_| ByteVec::new(vec![builder.add_witness()], builder.add_witness()))
		.collect();

	let _joined = concat(&builder, &terms);

	let circuit = builder.build();
	let mut witness_filler = circuit.new_witness_filler();

	let term_data: [&[u8]; 3] = [b"Hello", b" ", b"World!"];
	for (term, data) in terms.iter().zip(term_data.iter()) {
		term.populate_len_bytes(&mut witness_filler, data.len());
		term.populate_data(&mut witness_filler, data);
	}

	circuit.populate_wire_witness(&mut witness_filler).unwrap();

	(circuit.constraint_system().clone(), witness_filler.into_value_vec())
}

pub fn create_slice_cs_with_witness() -> (ConstraintSystem, ValueVec) {
	use binius_circuits::slice::{assert_slice_eq, slice};
	use binius_frontend::util::pack_bytes_into_wires_le;

	let builder = CircuitBuilder::new();

	// Create wires for slice circuit
	let len_input = builder.add_witness();
	let len_slice = builder.add_witness();
	let input: Vec<Wire> = (0..4).map(|_| builder.add_witness()).collect();
	let expected: Vec<Wire> = (0..2).map(|_| builder.add_witness()).collect();
	let offset = builder.add_witness();

	// Extract the slice and assert it matches `expected` in the first `len_slice` bytes.
	let actual = slice(&builder, len_input, len_slice, &input, offset, expected.len());
	assert_slice_eq(&builder, "slice_eq", len_slice, &actual, &expected);

	let circuit = builder.build();
	let mut witness_filler = circuit.new_witness_filler();

	// Test slicing "Hello World!" from offset 6 with length 5 to get "World"
	let input_data = b"Hello World!";
	let slice_data = b"World";
	let offset_val = 6u64;

	witness_filler[len_input] = Word(input_data.len() as u64);
	witness_filler[len_slice] = Word(slice_data.len() as u64);
	pack_bytes_into_wires_le(&mut witness_filler, &input, input_data);
	pack_bytes_into_wires_le(&mut witness_filler, &expected, slice_data);
	witness_filler[offset] = Word(offset_val);

	// Get the witness vector
	circuit.populate_wire_witness(&mut witness_filler).unwrap();

	(circuit.constraint_system().clone(), witness_filler.into_value_vec())
}

// Compute the image of the witness applied to the AND constraints
pub fn compute_bitand_images(constraints: &[AndConstraint], witness: &ValueVec) -> [Vec<Word>; 3] {
	let (a_image, b_image, c_image) = constraints
		.iter()
		.map(|constraint| {
			let a = witness.eval_operand(&constraint.a);
			let b = witness.eval_operand(&constraint.b);
			let c = witness.eval_operand(&constraint.c);
			(a, b, c)
		})
		.multiunzip();
	[a_image, b_image, c_image]
}

// Compute the image of the witness applied to the MUL constraints
fn compute_intmul_images(constraints: &[MulConstraint], witness: &ValueVec) -> [Vec<Word>; 4] {
	let (a_image, b_image, hi_image, lo_image) = constraints
		.iter()
		.map(|constraint| {
			let a = witness.eval_operand(&constraint.a);
			let b = witness.eval_operand(&constraint.b);
			let hi = witness.eval_operand(&constraint.hi);
			let lo = witness.eval_operand(&constraint.lo);
			(a, b, hi, lo)
		})
		.multiunzip();
	[a_image, b_image, hi_image, lo_image]
}

// Evaluate the image of the witness applied to the AND or MUL constraints
// Univariate point is `r_zhat_prime`, multilinear point tensor-expanded is `r_x_prime_tensor`
fn evaluate_image<F: BinaryField>(
	subspace: &BinarySubspace<F>,
	image: &[Word],
	r_zhat_prime: F,
	r_x_prime_tensor: &[F],
) -> F {
	let l_tilde = lagrange_evals(subspace, r_zhat_prime);
	let univariate = image
		.iter()
		.map(|&word| {
			(0..64)
				.filter(|&i| (word >> i) & Word::ONE == Word::ONE)
				.map(|i| l_tilde[i as usize])
				.sum()
		})
		.collect::<Vec<_>>();
	inner_product(r_x_prime_tensor.iter().copied(), univariate.iter().copied())
}

/// Compute inner product of tensor with all bits from words
pub fn evaluate_witness<F: BinaryField>(words: &[Word], r_j: &[F], r_y: &[F]) -> F {
	let r_j_tensor = eq_ind_partial_eval::<F>(r_j);
	let r_y_tensor = eq_ind_partial_eval::<F>(r_y);

	let r_j_witness = fold_words::<_, F>(words, r_j_tensor.as_ref());

	inner_product_buffers(&r_j_witness, &r_y_tensor)
}

#[test]
fn test_shift_prove_and_verify() {
	use binius_field::{BinaryField128bGhash, Field, PackedBinaryGhash2x128b, Random};
	type F = BinaryField128bGhash;
	type P = PackedBinaryGhash2x128b;
	let mut rng = StdRng::seed_from_u64(0);

	let mut constraint_systems_to_test = vec![
		create_sha256_cs_with_witness(),
		create_slice_cs_with_witness(),
		create_concat_cs_with_witness(),
	];
	for (constraint_system, _) in constraint_systems_to_test.iter_mut() {
		constraint_system.validate_and_prepare().unwrap();
	}

	for (cs, value_vec) in constraint_systems_to_test.into_iter() {
		// Validate constraints using frontend verifier first
		if let Err(e) = verify_constraints(&cs, &value_vec) {
			panic!("Circuit failed constraint validation: {e}");
		}

		// Sample multilinear challenge point
		let r_x_prime_bitand = {
			let log_bitand_constraint_count = strict_log_2(cs.and_constraints.len()).unwrap();
			(0..log_bitand_constraint_count as u128)
				.map(F::new)
				.collect::<Vec<_>>()
		};
		// A constraint system may have zero MUL constraints (e.g. a pure-AND circuit like SHA-256).
		// The IntMul operator is then empty — an empty challenge point and a zero claim — mirroring
		// the prover/verifier skip of the IntMul reduction in `binius_prover` / `binius_verifier`.
		let intmul_is_empty = cs.mul_constraints.is_empty();
		let r_x_prime_intmul = if intmul_is_empty {
			Vec::new()
		} else {
			let log_intmul_constraint_count = strict_log_2(cs.mul_constraints.len()).unwrap();
			(0..log_intmul_constraint_count as u128)
				.map(F::new)
				.collect::<Vec<_>>()
		};

		// Sample univariate eval point — the bitand and intmul operators share
		// `r_zhat_prime` so the verifier can compute `h_op_evals` once for both.
		let r_zhat_prime = F::random(&mut rng);

		let subspace = BinarySubspace::<AESTowerField8b>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();

		let bitand_evals = compute_bitand_images(&cs.and_constraints, &value_vec).map(|image| {
			evaluate_image(
				&subspace,
				&image,
				r_zhat_prime,
				eq_ind_partial_eval(&r_x_prime_bitand).as_ref(),
			)
		});

		let intmul_evals: [F; 4] = if intmul_is_empty {
			[F::ZERO; 4]
		} else {
			compute_intmul_images(&cs.mul_constraints, &value_vec).map(|image| {
				evaluate_image(
					&subspace,
					&image,
					r_zhat_prime,
					eq_ind_partial_eval(&r_x_prime_intmul).as_ref(),
				)
			})
		};

		// Build prover's constraint system
		let key_collection = build_key_collection(&cs);

		// Create prover transcript and call the prover
		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();

		let prover_bitand_data = OperatorData {
			evals: bitand_evals.to_vec(),
			r_zhat_prime,
			r_x_prime: r_x_prime_bitand.clone(),
		};
		let prover_intmul_data = OperatorData {
			evals: intmul_evals.to_vec(),
			r_zhat_prime,
			r_x_prime: r_x_prime_intmul.clone(),
		};

		let prover_output = prove::<F, P, _>(
			&key_collection,
			value_vec.combined_witness(),
			prover_bitand_data.clone(),
			prover_intmul_data.clone(),
			&subspace,
			&mut prover_transcript,
		);

		// Create verifier transcript and call the verifier
		let mut verifier_transcript = prover_transcript.into_verifier();

		let verifier_bitand_data = VerifierOperatorData::new(r_x_prime_bitand, bitand_evals);
		let verifier_intmul_data = VerifierOperatorData::new(r_x_prime_intmul, intmul_evals);

		let verifier_output =
			verify(&cs, &verifier_bitand_data, &verifier_intmul_data, &mut verifier_transcript)
				.unwrap();

		// Check consistency with verifier output
		check_eval(
			&cs,
			value_vec.public(),
			&verifier_bitand_data,
			&verifier_intmul_data,
			&subspace,
			r_zhat_prime,
			&verifier_output,
			&mut verifier_transcript,
		)
		.unwrap();
		verifier_transcript.finalize().unwrap();

		// Check the claimed witness eval matches the direct evaluation of the non-public words.
		// The witness segment is zero-padded from the folded length up to the segment length,
		// contributing the `(1 - r)` factors.
		let r_y = verifier_output.r_y();
		let non_public = value_vec.non_public();
		let log_folded = log2_ceil_usize(non_public.len());
		let expected_eval = r_y[log_folded..].iter().fold(
			evaluate_witness(non_public, verifier_output.r_j(), &r_y[..log_folded]),
			|acc, &r_y_i| acc * (F::ONE - r_y_i),
		);
		assert_eq!(expected_eval, verifier_output.witness_eval);

		// Check consistency of prover and verifier outputs
		let eval_point = [
			verifier_output.r_j(),
			r_y,
			std::slice::from_ref(&verifier_output.r_segment),
		]
		.concat();
		assert_eq!(prover_output.challenges, eval_point);
		assert_eq!(prover_output.eval, verifier_output.witness_eval);
	}
}

// Builds `K = 2^log_instances` value vectors of one circuit with shifted AND operands.
// Each instance is populated with distinct inputs.
// Returns the prepared constraint system and the per-instance value vectors.
//
// The circuit asserts, over public words x, y, z0, z1:
//     shl(x, 5) & rotr(y, 13) == z0     (Sll amount 5, Rotr amount 13)
//     shr(x, 7) & sar(y, 3)   == z1     (Slr amount 7, Sar amount 3)
// Four shift variants at non-zero amounts flow through the batched g-build and monster.
fn build_shifted_batch(log_instances: usize) -> (ConstraintSystem, Vec<ValueVec>) {
	let builder = CircuitBuilder::new();
	let x = builder.add_inout();
	let y = builder.add_inout();
	let z0 = builder.add_inout();
	let z1 = builder.add_inout();
	let and0 = builder.band(builder.shl(x, 5), builder.rotr(y, 13));
	builder.assert_eq("z0", and0, z0);
	let and1 = builder.band(builder.shr(x, 7), builder.sar(y, 3));
	builder.assert_eq("z1", and1, z1);
	let circuit = builder.build();

	let value_vecs = (0..(1usize << log_instances))
		.map(|i| {
			// Distinct inputs per instance, so the public and hidden segments both differ.
			let xv = 0x0123_4567_89ab_cdefu64.wrapping_mul(i as u64 + 1) ^ 0xdead_beef;
			let yv = 0xfedc_ba98_7654_3210u64.wrapping_add(i as u64 * 0x9e37) ^ 0x1234;
			let mut filler = circuit.new_witness_filler();
			filler[x] = Word(xv);
			filler[y] = Word(yv);
			filler[z0] = Word((xv << 5) & yv.rotate_right(13));
			filler[z1] = Word((xv >> 7) & (((yv as i64) >> 3) as u64));
			circuit.populate_wire_witness(&mut filler).unwrap();
			filler.into_value_vec()
		})
		.collect();

	let mut cs = circuit.constraint_system().clone();
	cs.validate_and_prepare().unwrap();
	(cs, value_vecs)
}

#[test]
fn test_batch_shift_prove_with_unmodified_verifier() {
	use binius_field::{BinaryField128bGhash, Field, PackedBinaryGhash2x128b, Random};
	use binius_math::multilinear::eq::eq_ind_partial_eval_scalars;
	type F = BinaryField128bGhash;
	type P = PackedBinaryGhash2x128b;
	let mut rng = StdRng::seed_from_u64(0);

	// Fixture state: K = 4 instances of one circuit, distinct inputs, four shift variants.
	let log_instances = 2;
	let (cs, value_vecs) = build_shifted_batch(log_instances);
	for vv in &value_vecs {
		verify_constraints(&cs, vv).unwrap();
	}
	let instances = value_vecs
		.iter()
		.map(|v| v.combined_witness())
		.collect::<Vec<_>>();

	// Challenges: the local constraint point, the univariate bit point, and the instance point.
	let r_x_prime_bitand = (0..strict_log_2(cs.and_constraints.len()).unwrap())
		.map(|_| F::random(&mut rng))
		.collect::<Vec<_>>();
	let r_zhat_prime = F::random(&mut rng);
	let r_kappa = (0..log_instances)
		.map(|_| F::random(&mut rng))
		.collect::<Vec<_>>();
	let subspace = BinarySubspace::<AESTowerField8b>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();

	// The batched operand claim: the eq(r_kappa, .)-weighted sum of the per-instance operand evals.
	// This is exactly what the folded witness produces, so the reduction chains.
	let eps = eq_ind_partial_eval_scalars(&r_kappa);
	let r_x_tensor = eq_ind_partial_eval(&r_x_prime_bitand);
	let mut bitand_evals = [F::ZERO; 3];
	for (kappa, vv) in value_vecs.iter().enumerate() {
		let images = compute_bitand_images(&cs.and_constraints, vv);
		for (eval, image) in bitand_evals.iter_mut().zip(&images) {
			*eval +=
				eps[kappa] * evaluate_image(&subspace, image, r_zhat_prime, r_x_tensor.as_ref());
		}
	}

	// This circuit is pure-AND, so the IntMul operator is empty (mirrors the single-instance skip).
	let intmul_evals = [F::ZERO; 4];
	let r_x_prime_intmul: Vec<F> = Vec::new();

	let key_collection = build_key_collection(&cs);
	let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
	let bitand_data = OperatorData {
		evals: bitand_evals.to_vec(),
		r_zhat_prime,
		r_x_prime: r_x_prime_bitand.clone(),
	};
	let intmul_data = OperatorData {
		evals: intmul_evals.to_vec(),
		r_zhat_prime,
		r_x_prime: r_x_prime_intmul.clone(),
	};
	let prover_output = prove_batch::<F, P, _>(
		&key_collection,
		&instances,
		&r_kappa,
		bitand_data,
		intmul_data,
		&subspace,
		&mut prover_transcript,
	);

	// Verify with the UNMODIFIED single-instance verifier and its evaluation check.
	// The public words are the shared reconstruction set (instance 0's public segment).
	let mut verifier_transcript = prover_transcript.into_verifier();
	let verifier_bitand_data = VerifierOperatorData::new(r_x_prime_bitand, bitand_evals);
	let verifier_intmul_data = VerifierOperatorData::new(r_x_prime_intmul, intmul_evals);
	let verifier_output =
		verify(&cs, &verifier_bitand_data, &verifier_intmul_data, &mut verifier_transcript)
			.unwrap();
	check_eval(
		&cs,
		value_vecs[0].public(),
		&verifier_bitand_data,
		&verifier_intmul_data,
		&subspace,
		r_zhat_prime,
		&verifier_output,
		&mut verifier_transcript,
	)
	.unwrap();
	verifier_transcript.finalize().unwrap();

	// The prover and verifier agree on the reduced claim.
	let eval_point = [
		verifier_output.r_j(),
		verifier_output.r_y(),
		std::slice::from_ref(&verifier_output.r_segment),
	]
	.concat();
	assert_eq!(prover_output.challenges, eval_point);
	assert_eq!(prover_output.eval, verifier_output.witness_eval);
}
