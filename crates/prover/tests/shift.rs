// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::array;

use binius_circuits::{fixed_byte_vec::ByteVec, sha256::sha256_varlen};
use binius_compute::GlobalAllocator;
use binius_core::{
	constraint_system::{
		AndConstraint, BmulConstraint, ConstraintSystem, ImulConstraint, InoutSegment, ValueVec,
	},
	word::Word,
};
use binius_field::{AESTowerField8b, BinaryField};
use binius_frontend::{CircuitBuilder, Wire};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace,
	inner_product::{inner_product, inner_product_buffers},
	multilinear::eq::eq_ind_partial_eval,
	univariate::subspace_lagrange_evals,
};
use binius_prover::{
	fold_word::fold_words,
	protocols::shift::{KeyCollection, OperatorClaims, OperatorData, ShiftProver},
};
use binius_transcript::ProverTranscript;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{
	config::StdChallenger,
	protocols::shift::{
		OperatorData as VerifierOperatorData, check_eval, evaluate_words_mle, verify,
	},
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

/// A system whose operands carry genuinely doubly shifted value indices, with a witness.
///
/// The frontend still lowers every term to a single shift, so nothing a circuit builds reaches
/// this path. It is built by hand instead, which is what lets the reduction be exercised over two
/// shift slots ahead of the compiler emitting them.
///
/// Each pair is one AND constraint `a & b ^ c = 0`, with `a` doubly shifted, `b` a second private
/// word and `c` the word the witness sets to their AND. One operand also mixes an unshifted, a
/// singly shifted and a doubly shifted term, since that is what a compiler emitting pairs would
/// produce.
pub fn create_double_shift_cs_with_witness() -> (ConstraintSystem, ValueVec) {
	use binius_core::constraint_system::{Composition, Shift, ShiftedValueIndex, ValueIndex};
	use rand::RngExt;

	/// The word a shift sequence carries its operand to, inner slot first.
	fn apply(sequence: [Shift; 2], word: Word) -> Word {
		let [inner, outer] = sequence;
		outer.apply(inner.apply(word))
	}

	// Sequences that genuinely need both slots: a sign extension of a sub-word field, a shift under
	// a rotate, a byte masked off one end and back, and the half-word family's own sign extension.
	// `sar` is the case worth covering in either slot, being the one shift whose vacated positions
	// all read a single input bit. Two shifts of one variant chain into one, so no pair here shares
	// a variant — the assertion below is what holds that to the composition rule rather than to
	// this comment.
	let sequences = [
		[Shift::sll(40), Shift::sar(40)],
		[Shift::rotr(1), Shift::sll(9)],
		[Shift::sll(8), Shift::srl(8)],
		[Shift::sll32(11), Shift::sra32(11)],
		[Shift::sar(7), Shift::sll(3)],
	];
	for [inner, outer] in sequences {
		assert_eq!(
			Shift::compose(inner, outer),
			Composition::Pair,
			"a collapsible sequence is not a doubly shifted term: {inner:?} then {outer:?}"
		);
	}

	// Three private words per sequence: the shifted operand, the mask, and their AND.
	let mut rng = StdRng::seed_from_u64(7);
	let mut private = (0..3 * sequences.len())
		.map(|_| Word::from_u64(rng.random()))
		.collect::<Vec<_>>();

	let mut and_constraints = Vec::new();
	for (i, sequence) in sequences.into_iter().enumerate() {
		let base = 3 * i as u32;
		// The witness makes the constraint hold: the product word is what `a & b` comes to.
		private[base as usize + 2] =
			apply(sequence, private[base as usize]) & private[base as usize + 1];
		and_constraints.push(AndConstraint([
			vec![ShiftedValueIndex::new(ValueIndex::private(base), sequence)],
			vec![ShiftedValueIndex::plain(ValueIndex::private(base + 1))],
			vec![ShiftedValueIndex::plain(ValueIndex::private(base + 2))],
		]));
	}

	// One more constraint whose first operand mixes all three term classes, since that is what a
	// compiler emitting pairs would produce. Its terms XOR to one word, which the other two
	// operands are then set to.
	{
		let sign_extend = [Shift::sll(40), Shift::sar(40)];
		let terms = vec![
			ShiftedValueIndex::plain(ValueIndex::private(0)),
			ShiftedValueIndex::srl(ValueIndex::private(3), 17),
			ShiftedValueIndex::new(ValueIndex::private(1), sign_extend),
		];
		let mixed = private[0] ^ (private[3] >> 17) ^ apply(sign_extend, private[1]);
		let base = private.len() as u32;
		private.push(mixed);
		private.push(mixed);
		and_constraints.push(AndConstraint([
			terms,
			vec![ShiftedValueIndex::plain(ValueIndex::private(base))],
			vec![ShiftedValueIndex::plain(ValueIndex::private(base + 1))],
		]));
	}

	let cs = ConstraintSystem {
		constants: vec![Word::ZERO; 4],
		n_inout: 0,
		n_private: private.len(),
		zero_constraints: Vec::new(),
		and_constraints,
		imul_constraints: Vec::new(),
		bmul_constraints: Vec::new(),
	};
	let value_vec = ValueVec::new_from_data(4, &[Word::ZERO; 4], &private);
	(cs, value_vec)
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
	let l_tilde = subspace_lagrange_evals(subspace, r_zhat_prime);
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
	use binius_field::{Field, Ghash128b, PackedBinaryGhash2x128b, Random};
	type F = Ghash128b;
	type P = PackedBinaryGhash2x128b;
	let mut rng = StdRng::seed_from_u64(0);

	let constraint_systems_to_test = vec![
		create_sha256_cs_with_witness(),
		create_slice_cs_with_witness(),
		create_concat_cs_with_witness(),
		create_double_shift_cs_with_witness(),
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
		let key_collection = KeyCollection::build(&cs, InoutSegment::Public);

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

		let prover_output = ShiftProver::<_, P, _>::new(&mut prover_transcript, &GlobalAllocator)
			.prove(
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
			);

		// The full reduction sends this after the public segment's evaluation claim; driving the
		// shift alone, it follows the reduction directly.
		prover_transcript.send_public_claim(prover_output.wiring_eval);

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

		// The public segment over the shift's whole index space. The full reduction reads this
		// from the prover and ties it to the public words with a ring-switch; driving the shift
		// alone, evaluate it here.
		let public_eval = evaluate_words_mle::<F, F>(
			value_vec.public(),
			verifier_output.r_j(),
			verifier_output.r_y(),
		);

		// Check consistency with verifier output
		let wiring_claim = check_eval(
			&cs,
			InoutSegment::Public,
			public_eval,
			&verifier_zero_data,
			&verifier_bitand_data,
			&verifier_intmul_data,
			&verifier_binmul_data,
			&subspace,
			&r_zhat_prime,
			&verifier_output,
			&mut verifier_transcript,
		)
		.unwrap();

		// Discharge the wiring claim the way the full reduction's caller does.
		wiring_claim.check_native().unwrap();
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
		assert_eq!(prover_output.sumcheck.challenges, eval_point);
		assert_eq!(prover_output.sumcheck.eval, verifier_output.witness_eval);
	}
}
