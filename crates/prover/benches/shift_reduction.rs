// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_circuits::sha256::sha256_fixed;
use binius_core::{ValueVec, constraint_system::ConstraintSystem, word::Word};
use binius_field::{BinaryField128bGhash, Field, Random, arch::OptimalPackedB128};
use binius_frontend::{CircuitBuilder, Wire};
use binius_ip::sumcheck::SumcheckOutput;
use binius_math::multilinear::eq::eq_ind_partial_eval;
use binius_prover::{
	fold_word::fold_words,
	protocols::shift::{
		OperatorData, PreparedOperatorData, build_key_collection,
		monster::{build_h_parts, build_monster_segments},
		phase_1::{build_g_parts, run_phase_1_sumcheck},
		phase_2::run_sumcheck,
		prove,
	},
};
use binius_transcript::ProverTranscript;
use binius_utils::checked_arithmetics::strict_log_2;
use binius_verifier::{
	config::{LOG_WORD_SIZE_BITS, StdChallenger},
	protocols::shift::{OperatorData as VerifierOperatorData, verify},
};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use sha2::{Digest, Sha256 as Sha256Hasher};

pub fn create_sha256_cs_with_witness(
	log_message_len_bytes: usize,
	rng: &mut impl rand::Rng,
) -> (ConstraintSystem, ValueVec) {
	let builder = CircuitBuilder::new();
	let message_len_bytes: usize = 1 << log_message_len_bytes; // 2^log_message_len

	// Message wires: one 32-bit word per 4 message bytes (`message_len_bytes` is a power of two,
	// so this divides evenly).
	let n_message_words = message_len_bytes.div_ceil(4);
	let message: Vec<Wire> = (0..n_message_words)
		.map(|_| builder.add_witness())
		.collect();

	// Expected digest as 8 big-endian 32-bit words.
	let expected_digest: [Wire; 8] = std::array::from_fn(|_| builder.add_inout());

	// Compute the SHA256 digest of the fixed-length message and constrain it to the expected wires.
	let computed_digest = sha256_fixed(&builder, &message, message_len_bytes);
	for i in 0..8 {
		builder.assert_eq(format!("digest[{i}]"), computed_digest[i], expected_digest[i]);
	}

	let circuit = builder.build();
	let mut witness_filler = circuit.new_witness_filler();

	// Generate random message bytes of specified length and pack them into big-endian 32-bit words.
	let mut message_bytes = vec![0u8; message_len_bytes];
	rng.fill_bytes(&mut message_bytes);
	for (word_idx, wire) in message.iter().enumerate() {
		let mut packed = 0u32;
		for i in 0..4 {
			packed |= (message_bytes[word_idx * 4 + i] as u32) << (24 - i * 8);
		}
		witness_filler[*wire] = Word(packed as u64);
	}

	// Calculate SHA256 digest of the message dynamically and populate the expected digest wires.
	let hash = Sha256Hasher::digest(&message_bytes);
	let expected_bytes: [u8; 32] = hash.into();
	for (i, wire) in expected_digest.iter().enumerate() {
		let mut word = 0u32;
		for j in 0..4 {
			word |= (expected_bytes[i * 4 + j] as u32) << (24 - j * 8);
		}
		witness_filler[*wire] = Word(word as u64);
	}

	// Get the witness vector
	circuit.populate_wire_witness(&mut witness_filler).unwrap();

	(circuit.constraint_system().clone(), witness_filler.into_value_vec())
}

fn bench_prove_and_verify(c: &mut Criterion) {
	type F = BinaryField128bGhash;
	type P = OptimalPackedB128;
	let mut rng = rand::rng();

	// Configurable log message lengths to benchmark (actual lengths will be 2^log_len)
	let log_message_lengths_bytes = [8, 12, 16]; // Actual lengths: 256, 4096, 65536 bytes

	for &log_message_len_bytes in &log_message_lengths_bytes {
		let message_len_bytes = 1 << log_message_len_bytes;
		let (mut cs, value_vec) = create_sha256_cs_with_witness(log_message_len_bytes, &mut rng);
		cs.validate_and_prepare().unwrap();

		// Sample multilinear eval points
		let r_x_prime_bitand = {
			let log_bitand_constraint_count = strict_log_2(cs.and_constraints.len()).unwrap();
			(0..log_bitand_constraint_count as u128)
				.map(F::new)
				.collect::<Vec<_>>()
		};
		// SHA256 has no MUL constraints, so the IntMul operator is the zero claim (four zero evals
		// at an empty point), exactly as the real prover/verifier synthesize it (`prove.rs` /
		// `verify.rs` `None` branch). Its `r_x_prime` is therefore empty.
		let r_x_prime_intmul: Vec<F> = Vec::new();

		// Sample univariate eval point — shared across bitand and intmul operators.
		let r_zhat_prime = F::random(&mut rng);

		let bitand_evals = [F::random(&mut rng); 3];
		let intmul_evals = [F::ZERO; 4];
		let key_collection = build_key_collection(&cs);

		let mut group = c.benchmark_group(format!(
			"shift_reduction_log2_{log_message_len_bytes}_bytes_{message_len_bytes}"
		));
		group.sample_size(10);

		group.bench_function("prove", |b| {
			b.iter(|| {
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

				let mut prover_transcript = ProverTranscript::<StdChallenger>::default();

				prove::<F, P, _>(
					&key_collection,
					value_vec.combined_witness(),
					prover_bitand_data,
					prover_intmul_data,
					&mut prover_transcript,
				)
			})
		});

		// Pre-run the prover to get the transcript for verifier benchmarking
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

		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();

		prove::<F, P, _>(
			&key_collection,
			value_vec.combined_witness(),
			prover_bitand_data,
			prover_intmul_data,
			&mut prover_transcript,
		);

		let setup_verifier_transcript = prover_transcript.into_verifier();

		group.bench_function("verify", |b| {
			b.iter(|| {
				let mut verifier_transcript = setup_verifier_transcript.clone();

				let verifier_bitand_data =
					VerifierOperatorData::new(r_x_prime_bitand.clone(), bitand_evals);
				let verifier_intmul_data =
					VerifierOperatorData::new(r_x_prime_intmul.clone(), intmul_evals);

				verify(&cs, &verifier_bitand_data, &verifier_intmul_data, &mut verifier_transcript)
					.unwrap();
			})
		});
	}
}

/// Fine-grained benchmarks for the individual phases of the shift-reduction prover, mirroring the
/// `intmul/phases` breakdown. Each of the five phase functions is timed on its own, sharing one
/// expensive circuit / witness / key-collection setup.
fn bench_shift_phases(c: &mut Criterion) {
	type F = BinaryField128bGhash;
	type P = OptimalPackedB128;
	let mut rng = rand::rng();

	// A single fixed size (16384-byte SHA256 message), rather than a sweep, so the per-phase
	// benches share one setup and stay quick.
	const LOG_MESSAGE_LEN_BYTES: usize = 14;

	let (mut cs, value_vec) = create_sha256_cs_with_witness(LOG_MESSAGE_LEN_BYTES, &mut rng);
	cs.validate_and_prepare().unwrap();

	let r_x_prime_bitand = (0..strict_log_2(cs.and_constraints.len()).unwrap() as u128)
		.map(F::new)
		.collect::<Vec<_>>();
	// SHA256 has no MUL constraints, so the IntMul operator is the zero claim at an empty point,
	// matching the real prover (`prove.rs` `None` branch).
	let r_x_prime_intmul: Vec<F> = Vec::new();
	// `r_zhat_prime` is shared across the bitand and intmul operators.
	let r_zhat_prime = F::random(&mut rng);
	let bitand_evals = [F::random(&mut rng); 3];
	let intmul_evals = [F::ZERO; 4];

	let key_collection = build_key_collection(&cs);
	let words = value_vec.combined_witness();

	// Prepare the operator data. Lambda sampling is cheap and not part of any benched phase, so a
	// random lambda (rather than one drawn from a transcript) yields realistic-magnitude data.
	let prepared_bitand = PreparedOperatorData::new(
		OperatorData {
			evals: bitand_evals.to_vec(),
			r_zhat_prime,
			r_x_prime: r_x_prime_bitand,
		},
		F::random(&mut rng),
	);
	let prepared_intmul = PreparedOperatorData::new(
		OperatorData {
			evals: intmul_evals.to_vec(),
			r_zhat_prime,
			r_x_prime: r_x_prime_intmul,
		},
		F::random(&mut rng),
	);

	// The phases are sequential and stateful: each consumes the previous one's outputs. Rather than
	// re-deriving predecessors inside each phase's per-iteration setup, advance the protocol once
	// here (with a throwaway transcript) to capture each phase's inputs; the setup closures below
	// then only clone what a phase consumes by value. The specific transcript challenges do not
	// change the work a phase performs.
	let g_parts = build_g_parts::<F, P>(words, &key_collection, &prepared_bitand, &prepared_intmul);
	let h_parts = build_h_parts::<F, P>(prepared_bitand.r_zhat_prime);
	let SumcheckOutput {
		challenges: mut r_jr_s,
		eval: gamma,
	} = {
		let mut transcript = ProverTranscript::<StdChallenger>::default();
		run_phase_1_sumcheck::<F, P, _>(g_parts.clone(), h_parts.clone(), &mut transcript)
	};
	// Split phase-1 challenges into `r_j` (low) and `r_s` (high) halves.
	let r_s = r_jr_s.split_off(LOG_WORD_SIZE_BITS);
	let r_j = r_jr_s;
	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);
	let (public_words, hidden_words) = words.split_at(key_collection.public.n_words());
	let public_folded = fold_words::<F, P>(public_words, r_j_tensor.as_ref());
	let hidden_folded = fold_words::<F, P>(hidden_words, r_j_tensor.as_ref());
	let (public_monster, hidden_monster) = build_monster_segments::<F, P>(
		&key_collection,
		&prepared_bitand,
		&prepared_intmul,
		&r_j,
		&r_s,
	);

	let mut group = c.benchmark_group("shift_reduction_phases");
	group.sample_size(10);

	// Phase 1. `build_g_parts` / `build_h_parts` take their inputs by reference, so no
	// per-iteration clone is needed; `run_phase_1_sumcheck` consumes `g_parts`/`h_parts` by value.
	group.bench_function("phase1_build_g_parts", |b| {
		b.iter(|| {
			build_g_parts::<F, P>(words, &key_collection, &prepared_bitand, &prepared_intmul)
		});
	});
	group.bench_function("phase1_build_h_parts", |b| {
		b.iter(|| build_h_parts::<F, P>(prepared_bitand.r_zhat_prime));
	});
	group.bench_function("phase1_run_sumcheck", |b| {
		b.iter_batched(
			|| (g_parts.clone(), h_parts.clone()),
			|(g, h)| {
				let mut transcript = ProverTranscript::<StdChallenger>::default();
				run_phase_1_sumcheck::<F, P, _>(g, h, &mut transcript)
			},
			BatchSize::SmallInput,
		);
	});

	// Phase 2. `build_monster_segments` takes its inputs by reference; `run_sumcheck` consumes
	// its buffers and `r_j` by value.
	group.bench_function("phase2_build_monster_segments", |b| {
		b.iter(|| {
			build_monster_segments::<F, P>(
				&key_collection,
				&prepared_bitand,
				&prepared_intmul,
				&r_j,
				&r_s,
			)
		});
	});
	group.bench_function("phase2_run_sumcheck", |b| {
		b.iter_batched(
			|| {
				(
					public_folded.clone(),
					hidden_folded.clone(),
					public_monster.clone(),
					hidden_monster.clone(),
					r_j.clone(),
				)
			},
			|(public_folded, hidden_folded, public_monster, hidden_monster, r_j)| {
				let mut transcript = ProverTranscript::<StdChallenger>::default();
				run_sumcheck::<F, P, _>(
					public_folded,
					hidden_folded,
					public_monster,
					hidden_monster,
					public_words,
					r_j,
					gamma,
					&mut transcript,
				)
			},
			BatchSize::SmallInput,
		);
	});

	group.finish();
}

criterion_group!(benches, bench_prove_and_verify, bench_shift_phases);
criterion_main!(benches);
