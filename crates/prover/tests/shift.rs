// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::array;

use binius_circuits::{fixed_byte_vec::ByteVec, sha256::sha256_varlen};
use binius_compute::GlobalAllocator;
use binius_core::{
	constraint_system::{
		AndConstraint, BmulConstraint, Composition, ConstraintSystem, ImulConstraint, InoutSegment,
		Shift, ShiftedValueIndex, ValueIndex, ValueVec, ZeroConstraint,
	},
	word::Word,
};
use binius_field::{AESTowerField8b, BinaryField, Field, Random};
use binius_frontend::{CircuitBuilder, Wire};
use binius_math::{
	BinarySubspace,
	inner_product::{inner_product, inner_product_buffers},
	multilinear::eq::eq_ind_partial_eval,
	univariate::lagrange_evals,
};
use binius_prover::{
	fold_word::fold_words,
	protocols::shift::{OperatorClaims, OperatorData, build_key_collection, prove},
};
use binius_transcript::ProverTranscript;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{
	config::StdChallenger,
	protocols::shift::{
		LOG_SHIFT_VARIANT_COUNT, OperatorData as VerifierOperatorData, VerifyOutput, check_eval,
		has_double_shift, verify,
	},
};
use itertools::Itertools;
use rand::{RngExt, SeedableRng, rngs::StdRng};
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
	let data: Vec<Wire> = (0..max_len.div_ceil(8))
		.map(|_| builder.add_witness())
		.collect();

	// Create the SHA256 circuit
	let message = ByteVec::new(data, len);
	let computed = sha256_varlen(&builder, &message);
	for i in 0..4 {
		builder.assert_eq(format!("digest[{i}]"), computed[i], digest[i]);
	}

	let circuit = builder.build();
	let mut witness_filler = circuit.new_witness_filler();

	// Populate with concrete message: "abc"
	let message_bytes = b"abc";
	message.populate_len_bytes(&mut witness_filler, message_bytes.len());
	message.populate_data(&mut witness_filler, message_bytes);

	// Calculate SHA256 digest of the message dynamically
	let hash = Sha256Hasher::digest(message_bytes);
	let expected_digest: [u8; 32] = hash.into();
	for (i, chunk) in expected_digest.chunks(8).enumerate() {
		witness_filler[digest[i]] = Word(u64::from_be_bytes(chunk.try_into().unwrap()));
	}

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
	witness_filler.pack_bytes_le(&input, input_data);
	witness_filler.pack_bytes_le(&expected, slice_data);
	witness_filler[offset] = Word(offset_val);

	// Get the witness vector
	circuit.populate_wire_witness(&mut witness_filler).unwrap();

	(circuit.constraint_system().clone(), witness_filler.into_value_vec())
}

// Compute the image of the witness applied to the AND constraints
//
// Each image is zero-padded to a power-of-two length, matching the operand columns the prover
// materializes.
pub fn compute_bitand_images(constraints: &[AndConstraint], witness: &ValueVec) -> [Vec<Word>; 3] {
	let (a_image, b_image, c_image) = constraints
		.iter()
		.map(|constraint| {
			let a = witness.eval_operand(constraint.a());
			let b = witness.eval_operand(constraint.b());
			let c = witness.eval_operand(constraint.c());
			(a, b, c)
		})
		.multiunzip::<(Vec<_>, Vec<_>, Vec<_>)>();
	[a_image, b_image, c_image].map(|image| pad_image(image, constraints.len()))
}

// Zero-pad a per-constraint image up to the power-of-two row count the reductions run over.
fn pad_image(mut image: Vec<Word>, n_constraints: usize) -> Vec<Word> {
	image.resize(n_constraints.next_power_of_two(), Word::ZERO);
	image
}

// Compute the image of the witness applied to the IMUL constraints
//
// Each image is zero-padded to a power-of-two length, matching the operand columns the prover
// materializes.
fn compute_intmul_images(constraints: &[ImulConstraint], witness: &ValueVec) -> [Vec<Word>; 4] {
	let (a_image, b_image, lo_image, hi_image) = constraints
		.iter()
		.map(|constraint| {
			let a = witness.eval_operand(constraint.a());
			let b = witness.eval_operand(constraint.b());
			let lo = witness.eval_operand(constraint.lo());
			let hi = witness.eval_operand(constraint.hi());
			(a, b, lo, hi)
		})
		.multiunzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<_>)>();
	[a_image, b_image, lo_image, hi_image].map(|image| pad_image(image, constraints.len()))
}

// Compute the image of the witness applied to the BMUL constraints
//
// Each image is zero-padded to a power-of-two length, matching the operand columns the prover
// materializes.
fn compute_binmul_images(constraints: &[BmulConstraint], witness: &ValueVec) -> [Vec<Word>; 6] {
	array::from_fn(|op_idx| {
		let image = constraints
			.iter()
			.map(|constraint| witness.eval_operand(&constraint.as_ref()[op_idx]))
			.collect();
		pad_image(image, constraints.len())
	})
}

// Evaluate the image of the witness applied to the AND or IMUL constraints
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

	let r_j_witness = fold_words::<_, F, _>(&GlobalAllocator, words, r_j_tensor.as_ref());

	inner_product_buffers(&r_j_witness, &r_y_tensor)
}

#[test]
fn test_shift_prove_and_verify() {
	use binius_field::{BinaryField128bGhash, Field, PackedBinaryGhash2x128b, Random};
	type F = BinaryField128bGhash;
	type P = PackedBinaryGhash2x128b;
	let mut rng = StdRng::seed_from_u64(0);

	let constraint_systems_to_test = vec![
		create_sha256_cs_with_witness(),
		create_slice_cs_with_witness(),
		create_concat_cs_with_witness(),
	];
	for (constraint_system, _) in constraint_systems_to_test.iter() {
		constraint_system.validate().unwrap();
	}

	for (cs, value_vec) in constraint_systems_to_test.into_iter() {
		// Validate constraints using frontend verifier first
		if let Err(e) = cs.verify(&value_vec) {
			panic!("Circuit failed constraint validation: {e}");
		}

		// Sample multilinear challenge point
		let r_x_prime_bitand = {
			// The BitAnd reduction always runs; an empty AND set reduces over its single all-zero
			// padding row, i.e. an empty point.
			let log_bitand_constraint_count = cs.log_and_constraints().unwrap_or(0);
			(0..log_bitand_constraint_count as u128)
				.map(F::new)
				.collect::<Vec<_>>()
		};
		// A constraint system may have zero IMUL constraints (e.g. a pure-AND circuit like
		// SHA-256). The IntMul operator is then empty — an empty challenge point and a zero claim
		// — mirroring the prover/verifier skip of the IntMul reduction in `binius_prover` /
		// `binius_verifier`.
		let intmul_is_empty = cs.imul_constraints.is_empty();
		let r_x_prime_intmul = if let Some(log_intmul_constraint_count) = cs.log_imul_constraints()
		{
			(0..log_intmul_constraint_count as u128)
				.map(F::new)
				.collect::<Vec<_>>()
		} else {
			Vec::new()
		};

		// A constraint system may equally have zero BMUL constraints, and the BinMul operator is
		// then empty for the same reason.
		let binmul_is_empty = cs.bmul_constraints.is_empty();
		let r_x_prime_binmul = if let Some(log_binmul_constraint_count) = cs.log_bmul_constraints()
		{
			(0..log_binmul_constraint_count as u128)
				.map(F::new)
				.collect::<Vec<_>>()
		} else {
			Vec::new()
		};

		// Sample univariate eval point — the bitand and intmul operators share
		// `r_zhat_prime` so the verifier can compute `h_op_evals` once for both.
		let r_zhat_prime = F::random(&mut rng);

		let subspace = BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();

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
			compute_intmul_images(&cs.imul_constraints, &value_vec).map(|image| {
				evaluate_image(
					&subspace,
					&image,
					r_zhat_prime,
					eq_ind_partial_eval(&r_x_prime_intmul).as_ref(),
				)
			})
		};

		let binmul_evals: [F; 6] = if binmul_is_empty {
			[F::ZERO; 6]
		} else {
			compute_binmul_images(&cs.bmul_constraints, &value_vec).map(|image| {
				evaluate_image(
					&subspace,
					&image,
					r_zhat_prime,
					eq_ind_partial_eval(&r_x_prime_binmul).as_ref(),
				)
			})
		};

		// Build prover's constraint system
		let key_collection = build_key_collection(&cs, InoutSegment::Public);

		// Create prover transcript and call the prover
		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();

		let prover_bitand_data = OperatorData {
			evals: bitand_evals,
			r_zhat_prime,
			r_x_prime: r_x_prime_bitand.clone(),
		};
		let prover_intmul_data = OperatorData {
			evals: intmul_evals,
			r_zhat_prime,
			r_x_prime: r_x_prime_intmul.clone(),
		};
		// The Zero claim closes at its own constraint point, as wide as the ZERO set. Its value is
		// zero at any point: a satisfied ZERO constraint array vanishes identically, so its
		// multilinear extension is the zero polynomial.
		let r_x_prime_zero = (0..cs.log_zero_constraints().unwrap_or(0) as u128)
			.map(F::new)
			.collect::<Vec<_>>();
		let prover_zero_data = OperatorData {
			evals: [F::ZERO],
			r_zhat_prime,
			r_x_prime: r_x_prime_zero.clone(),
		};
		let prover_binmul_data = OperatorData {
			evals: binmul_evals,
			r_zhat_prime,
			r_x_prime: r_x_prime_binmul.clone(),
		};

		let prover_output = prove::<F, P, _, _>(
			&key_collection,
			value_vec.public(),
			value_vec.non_public(),
			OperatorClaims {
				zero: prover_zero_data.clone(),
				bitand: prover_bitand_data.clone(),
				intmul: prover_intmul_data.clone(),
				binmul: prover_binmul_data.clone(),
			},
			&subspace,
			&mut prover_transcript,
			&GlobalAllocator,
		);

		// Create verifier transcript and call the verifier
		let mut verifier_transcript = prover_transcript.into_verifier();

		let verifier_zero_data = VerifierOperatorData::new(r_x_prime_zero, [F::ZERO]);
		let verifier_bitand_data = VerifierOperatorData::new(r_x_prime_bitand, bitand_evals);
		let verifier_intmul_data = VerifierOperatorData::new(r_x_prime_intmul, intmul_evals);
		let verifier_binmul_data = VerifierOperatorData::new(r_x_prime_binmul, binmul_evals);

		let verifier_output = verify(
			&cs,
			InoutSegment::Public,
			&verifier_zero_data,
			&verifier_bitand_data,
			&verifier_intmul_data,
			&verifier_binmul_data,
			&mut verifier_transcript,
		)
		.unwrap();

		// Check consistency with verifier output
		check_eval(
			&cs,
			InoutSegment::Public,
			value_vec.public(),
			&verifier_zero_data,
			&verifier_bitand_data,
			&verifier_intmul_data,
			&verifier_binmul_data,
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

/// The field the double-shift reduction tests run over.
type DoubleShiftF = binius_field::BinaryField128bGhash;
/// The packed field those tests instantiate the prover with.
type DoubleShiftP = binius_field::PackedBinaryGhash2x128b;

/// Every shift-variant pair whose composition genuinely needs two slots.
///
/// A pair that collapses would be rejected by validation, so the fixtures below draw from this
/// list. Amounts are chosen to keep each pair irreducible for its variant combination.
/// Dropping bits one way and moving them back the other is the shape no single shift has.
fn irreducible_pairs() -> Vec<[Shift; 2]> {
	let candidates = [
		[Shift::srl(3), Shift::sll(3)],
		[Shift::srl(5), Shift::sll(9)],
		[Shift::sll(8), Shift::srl(2)],
		[Shift::sll(17), Shift::srl(11)],
		[Shift::sar(7), Shift::sll(4)],
		[Shift::rotr(1), Shift::sll(6)],
		[Shift::rotr(13), Shift::srl(5)],
		[Shift::srl32(4), Shift::sll32(11)],
		[Shift::sll32(6), Shift::srl32(3)],
		[Shift::sra32(9), Shift::sll32(3)],
		[Shift::rotr32(5), Shift::srl32(7)],
		[Shift::rotr32(2), Shift::sll32(8)],
		// Crossing the two families: a half-word shift after a full-width one.
		[Shift::srl(33), Shift::sll32(4)],
		[Shift::sll(35), Shift::srl32(6)],
	];
	// Keep only the pairs that really need both slots, so the fixture cannot silently degenerate.
	let pairs = candidates
		.into_iter()
		.filter(|&[inner, outer]| Shift::compose(inner, outer) == Composition::Pair)
		.collect::<Vec<_>>();
	assert!(pairs.len() >= 12, "the fixture needs a spread of genuine pairs, got {}", pairs.len());
	pairs
}

/// A constraint system whose operands carry genuine shift pairs, with a satisfying value vector.
///
/// The frontend does not emit pairs yet, so this is built by hand.
/// The layout is
///
/// ```text
/// private[0 .. n_sources]   random source words
/// private[n_sources]        the AND constraint's C operand, set to A & B
/// private[n_sources + 1]    the ZERO constraint's second term, set to cancel the first
/// ```
///
/// so both constraints are satisfied by construction, whatever the pairs move.
fn create_double_shift_cs_with_witness() -> (ConstraintSystem, ValueVec) {
	let pairs = irreducible_pairs();
	let n_sources = pairs.len();
	let n_private = n_sources + 2;

	// Spread the pairs over the source words: the first third builds `A`, the next `B`, and one
	// more drives the ZERO constraint.
	let split = n_sources / 3;
	let term =
		|index: usize| ShiftedValueIndex::new(ValueIndex::private(index as u32), pairs[index]);

	let a_terms = (0..split).map(term).collect::<Vec<_>>();
	let b_terms = (split..2 * split).map(term).collect::<Vec<_>>();
	let zero_term = term(2 * split);

	let c_index = ValueIndex::private(n_sources as u32);
	let zero_cancel_index = ValueIndex::private(n_sources as u32 + 1);

	let cs = ConstraintSystem {
		constants: Vec::new(),
		n_inout: 0,
		n_private,
		// `val` must vanish: the pair-shifted word XORed with the word holding that same value.
		zero_constraints: vec![ZeroConstraint::new(vec![
			zero_term,
			ShiftedValueIndex::plain(zero_cancel_index),
		])],
		// `A & B ^ C = 0`, with `C` the plain word holding the conjunction.
		and_constraints: vec![AndConstraint([
			a_terms,
			b_terms,
			vec![ShiftedValueIndex::plain(c_index)],
		])],
		imul_constraints: Vec::new(),
		bmul_constraints: Vec::new(),
	};

	// Random source words, then the two derived words that make the system satisfied.
	let mut rng = StdRng::seed_from_u64(7);
	let mut private = (0..n_private)
		.map(|_| Word(rng.random::<u64>()))
		.collect::<Vec<_>>();
	private[n_sources] = Word::ZERO;
	private[n_sources + 1] = Word::ZERO;

	let mut value_vec = ValueVec::new_from_data(0, &[], &private);
	let [a_operand, b_operand, _] = cs.and_constraints[0].0.each_ref();
	let conjunction = value_vec.eval_operand(a_operand) & value_vec.eval_operand(b_operand);
	value_vec[c_index] = conjunction;
	let shifted = value_vec.eval_operand(&[zero_term]);
	value_vec[zero_cancel_index] = shifted;

	cs.validate().unwrap();
	cs.verify(&value_vec).unwrap();
	(cs, value_vec)
}

/// Runs the reduction over one constraint system and returns the verifier's output.
///
/// The claims are computed the way the single-shift round trip computes them.
/// So this exercises the real path rather than a stand-in.
fn prove_and_verify_reduction(
	cs: &ConstraintSystem,
	value_vec: &ValueVec,
	tamper: bool,
) -> Result<VerifyOutput<DoubleShiftF>, binius_verifier::protocols::shift::Error> {
	type F = DoubleShiftF;
	type P = DoubleShiftP;

	let mut rng = StdRng::seed_from_u64(11);
	let subspace = BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
	let r_zhat_prime = F::random(&mut rng);

	let point_of = |log_count: Option<usize>| {
		(0..log_count.unwrap_or(0) as u128)
			.map(F::new)
			.collect::<Vec<_>>()
	};
	let r_x_prime_zero = point_of(cs.log_zero_constraints());
	let r_x_prime_bitand = point_of(cs.log_and_constraints());
	// An unused operation reduces over an empty point, as both sides synthesize for it.
	// A used one needs its real point, or a key naming its constraints indexes past the tensor.
	let r_x_prime_intmul = point_of(cs.log_imul_constraints());
	let r_x_prime_binmul = point_of(cs.log_bmul_constraints());

	// The operand claims, from the images the witness actually produces.
	let evaluate_at = |image: Vec<Word>, r_x_prime: &[F]| {
		evaluate_image(&subspace, &image, r_zhat_prime, eq_ind_partial_eval(r_x_prime).as_ref())
	};
	let bitand_evals = compute_bitand_images(&cs.and_constraints, value_vec)
		.map(|image| evaluate_at(image, &r_x_prime_bitand));
	let intmul_evals: [F; 4] = if cs.imul_constraints.is_empty() {
		[F::ZERO; 4]
	} else {
		compute_intmul_images(&cs.imul_constraints, value_vec)
			.map(|image| evaluate_at(image, &r_x_prime_intmul))
	};
	let binmul_evals: [F; 6] = if cs.bmul_constraints.is_empty() {
		[F::ZERO; 6]
	} else {
		compute_binmul_images(&cs.bmul_constraints, value_vec)
			.map(|image| evaluate_at(image, &r_x_prime_binmul))
	};

	// A tampered run claims something the witness does not satisfy.
	let bitand_evals = if tamper {
		let mut evals = bitand_evals;
		evals[0] += F::ONE;
		evals
	} else {
		bitand_evals
	};

	let zero_data = OperatorData {
		evals: [F::ZERO],
		r_zhat_prime,
		r_x_prime: r_x_prime_zero.clone(),
	};
	let bitand_data = OperatorData {
		evals: bitand_evals,
		r_zhat_prime,
		r_x_prime: r_x_prime_bitand.clone(),
	};
	let intmul_data = OperatorData {
		evals: intmul_evals,
		r_zhat_prime,
		r_x_prime: r_x_prime_intmul.clone(),
	};
	let binmul_data = OperatorData {
		evals: binmul_evals,
		r_zhat_prime,
		r_x_prime: r_x_prime_binmul.clone(),
	};

	let key_collection = build_key_collection(cs, InoutSegment::Public);
	let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
	prove::<F, P, _, _>(
		&key_collection,
		value_vec.public(),
		value_vec.non_public(),
		OperatorClaims {
			zero: zero_data,
			bitand: bitand_data,
			intmul: intmul_data,
			binmul: binmul_data,
		},
		&subspace,
		&mut prover_transcript,
		&GlobalAllocator,
	);

	let mut verifier_transcript = prover_transcript.into_verifier();
	let verifier_zero = VerifierOperatorData::new(r_x_prime_zero, [F::ZERO]);
	let verifier_bitand = VerifierOperatorData::new(r_x_prime_bitand, bitand_evals);
	let verifier_intmul = VerifierOperatorData::new(r_x_prime_intmul, intmul_evals);
	let verifier_binmul = VerifierOperatorData::new(r_x_prime_binmul, binmul_evals);

	let output = verify(
		cs,
		InoutSegment::Public,
		&verifier_zero,
		&verifier_bitand,
		&verifier_intmul,
		&verifier_binmul,
		&mut verifier_transcript,
	)?;
	check_eval(
		cs,
		InoutSegment::Public,
		value_vec.public(),
		&verifier_zero,
		&verifier_bitand,
		&verifier_intmul,
		&verifier_binmul,
		&subspace,
		r_zhat_prime,
		&output,
		&mut verifier_transcript,
	)?;
	Ok(output)
}

#[test]
fn double_shifted_terms_prove_and_verify() {
	// Invariant: a system built on genuine shift pairs verifies.
	//
	// Terms whose two shifts do not collapse are reduced through the outer phase against the
	// intermediate words, then the inner phase against the witness.
	let (cs, value_vec) = create_double_shift_cs_with_witness();
	let output = prove_and_verify_reduction(&cs, &value_vec, false).unwrap();

	// The outer phase ran, which is what distinguishes this from the two-phase path.
	let outer = output
		.outer
		.as_ref()
		.expect("a system with genuine shift pairs runs the outer phase");
	assert_eq!(outer.r_k.len(), Word::LOG_BITS);
	assert_eq!(outer.r_s.len(), Word::LOG_BITS);
	assert_eq!(outer.r_v.len(), LOG_SHIFT_VARIANT_COUNT);
}

#[test]
fn double_shifted_terms_reject_an_unsatisfied_claim() {
	// Invariant: a claim the witness does not satisfy must not verify, pairs or not.
	//
	// Perturbing the claim is the same thing from the reduction's side as perturbing the witness:
	// the sum it is handed stops matching the one the witness produces, and that difference
	// travels to the final evaluation check.
	let (cs, value_vec) = create_double_shift_cs_with_witness();
	let result = prove_and_verify_reduction(&cs, &value_vec, true);
	assert!(result.is_err(), "a claim the witness does not satisfy must not verify");
}

#[test]
fn lone_shifts_skip_the_outer_phase() {
	// Invariant: a system of lone shifts skips the outer phase, and both sides agree it does.
	//
	// The phase structure is derived from the constraint system, so no message says which is in
	// use. That is what leaves a single-shift circuit's transcript unchanged.
	let (cs, value_vec) = create_slice_cs_with_witness();
	assert!(
		!has_double_shift(&cs),
		"the frontend emits lone shifts, so this system needs no outer phase"
	);

	let output = prove_and_verify_reduction(&cs, &value_vec, false).unwrap();
	assert!(output.outer.is_none(), "a system of lone shifts must not run the outer phase");
}

#[test]
fn each_irreducible_pair_proves_and_verifies_alone() {
	// Invariant: every irreducible variant combination survives the reduction on its own.
	//
	// Run together, one pair's contribution could mask another's.
	// One constraint system per pair keeps a mis-handled variant from hiding.
	for shift_seq in irreducible_pairs() {
		// `A = pair(w_0)`, `B = w_1`, `C = A & B`, on three private words.
		let a_term = ShiftedValueIndex::new(ValueIndex::private(0), shift_seq);
		let cs = ConstraintSystem {
			constants: Vec::new(),
			n_inout: 0,
			n_private: 3,
			zero_constraints: Vec::new(),
			and_constraints: vec![AndConstraint([
				vec![a_term],
				vec![ShiftedValueIndex::plain(ValueIndex::private(1))],
				vec![ShiftedValueIndex::plain(ValueIndex::private(2))],
			])],
			imul_constraints: Vec::new(),
			bmul_constraints: Vec::new(),
		};
		cs.validate().unwrap();

		let mut rng = StdRng::seed_from_u64(3);
		let private = vec![
			Word(rng.random::<u64>()),
			Word(rng.random::<u64>()),
			Word::ZERO,
		];
		let mut value_vec = ValueVec::new_from_data(0, &[], &private);
		let conjunction = value_vec.eval_operand(&[a_term]) & private[1];
		value_vec[ValueIndex::private(2)] = conjunction;
		cs.verify(&value_vec).unwrap();

		let output = prove_and_verify_reduction(&cs, &value_vec, false)
			.unwrap_or_else(|error| panic!("{shift_seq:?} failed to verify: {error}"));
		assert!(output.outer.is_some(), "{shift_seq:?} must run the outer phase");
	}
}
