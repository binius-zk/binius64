// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_circuits::sha256::sha256_fixed;
use binius_compute::{BufferPool, GlobalAllocator};
use binius_core::{
	ValueVec,
	constraint_system::{ConstraintSystem, InoutSegment},
	word::Word,
};
use binius_field::{AESTowerField8b, BinaryField128bGhash, Field, Random, arch::OptimalPackedB128};
use binius_frontend::{CircuitBuilder, Wire};
use binius_math::{
	BinarySubspace, multilinear::eq::eq_ind_partial_eval, univariate::subspace_lagrange_evals,
};
use binius_prover::{
	fold_word::fold_words,
	protocols::shift::{
		KeyCollection, OperatorClaims, OperatorData, ShiftProver,
		monster::shift_operator_table,
		phase_1::{Phase1Output, SparseShiftRows},
		phase_2::run_sumcheck,
	},
};
use binius_transcript::ProverTranscript;
use binius_verifier::{
	config::StdChallenger,
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
		let (cs, value_vec) = create_sha256_cs_with_witness(log_message_len_bytes, &mut rng);
		cs.validate().unwrap();

		// Sample multilinear eval points
		let r_x_prime_bitand = {
			// The BitAnd reduction always runs; `None` is its single all-zero padding row.
			let log_bitand_constraint_count = cs.log_and_constraints().unwrap_or(0);
			(0..log_bitand_constraint_count as u128)
				.map(F::new)
				.collect::<Vec<_>>()
		};
		// The circuit's digest assertions lower to ZERO constraints, so the Zero reduction runs
		// over its own constraint point, as wide as the ZERO set.
		let r_x_prime_zero = (0..cs.log_zero_constraints().unwrap_or(0) as u128)
			.map(F::new)
			.collect::<Vec<_>>();
		// SHA256 has no IMUL constraints, so the IntMul operator is the zero claim (four zero evals
		// at an empty point), exactly as the real prover/verifier synthesize it (`prove.rs` /
		// `verify.rs` `None` branch). Its `r_x_prime` is therefore empty.
		let r_x_prime_intmul: Vec<F> = Vec::new();

		// Sample univariate eval point — shared across bitand and intmul operators.
		let r_zhat_prime = F::random(&mut rng);

		let zero_evals = [F::random(&mut rng)];
		let bitand_evals = [F::random(&mut rng); 3];
		let intmul_evals = [F::ZERO; 4];
		let key_collection = KeyCollection::build(&cs, InoutSegment::Public);
		let subspace = BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();

		let mut group = c.benchmark_group(format!(
			"shift_reduction_log2_{log_message_len_bytes}_bytes_{message_len_bytes}"
		));
		group.sample_size(10);

		group.bench_function("prove", |b| {
			let pool = BufferPool::new();
			let alloc = &pool;
			b.iter(|| {
				let prover_zero_data = OperatorData {
					evals: zero_evals,
					r_zhat_prime,
					r_x_prime: r_x_prime_zero.clone(),
				};
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

				let mut prover_transcript = ProverTranscript::<StdChallenger>::default();

				ShiftProver::<_, P, _>::new(&mut prover_transcript, &alloc).prove(
					&key_collection,
					value_vec.public(),
					value_vec.non_public(),
					OperatorClaims {
						zero: prover_zero_data,
						bitand: prover_bitand_data,
						intmul: prover_intmul_data,
						binmul: OperatorData::zero_claim(r_zhat_prime),
					},
					&subspace,
				)
			})
		});

		// Pre-run the prover to get the transcript for verifier benchmarking
		let prover_zero_data = OperatorData {
			evals: zero_evals,
			r_zhat_prime,
			r_x_prime: r_x_prime_zero.clone(),
		};
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

		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();

		ShiftProver::<_, P, _>::new(&mut prover_transcript, &&BufferPool::new()).prove(
			&key_collection,
			value_vec.public(),
			value_vec.non_public(),
			OperatorClaims {
				zero: prover_zero_data,
				bitand: prover_bitand_data,
				intmul: prover_intmul_data,
				binmul: OperatorData::zero_claim(r_zhat_prime),
			},
			&subspace,
		);

		let setup_verifier_transcript = prover_transcript.into_verifier();

		group.bench_function("verify", |b| {
			b.iter(|| {
				let mut verifier_transcript = setup_verifier_transcript.clone();

				let verifier_bitand_data =
					VerifierOperatorData::new(r_x_prime_bitand.clone(), bitand_evals);
				let verifier_intmul_data =
					VerifierOperatorData::new(r_x_prime_intmul.clone(), intmul_evals);
				let verifier_binmul_data = VerifierOperatorData::new(Vec::new(), [F::ZERO; 6]);

				verify(
					&cs,
					InoutSegment::Public,
					&VerifierOperatorData::new(r_x_prime_zero.clone(), zero_evals),
					&verifier_bitand_data,
					&verifier_intmul_data,
					&verifier_binmul_data,
					&mut verifier_transcript,
				)
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

	let (cs, value_vec) = create_sha256_cs_with_witness(LOG_MESSAGE_LEN_BYTES, &mut rng);
	cs.validate().unwrap();

	// The BitAnd reduction always runs; `None` is its single all-zero padding row.
	let r_x_prime_bitand = (0..cs.log_and_constraints().unwrap_or(0) as u128)
		.map(F::new)
		.collect::<Vec<_>>();
	// The circuit's digest assertions lower to ZERO constraints, so the Zero reduction runs over
	// its own constraint point, as wide as the ZERO set.
	let r_x_prime_zero = (0..cs.log_zero_constraints().unwrap_or(0) as u128)
		.map(F::new)
		.collect::<Vec<_>>();
	// SHA256 has no IMUL constraints, so the IntMul operator is the zero claim at an empty point,
	// matching the real prover (`prove.rs` `None` branch).
	let r_x_prime_intmul: Vec<F> = Vec::new();
	// `r_zhat_prime` is shared across the bitand and intmul operators.
	let r_zhat_prime = F::random(&mut rng);
	let zero_evals = [F::random(&mut rng)];
	let bitand_evals = [F::random(&mut rng); 3];
	let intmul_evals = [F::ZERO; 4];

	let key_collection = KeyCollection::build(&cs, InoutSegment::Public);
	// The phase functions take each segment as the circuit declares it: `build_g` zips the
	// words with their key ranges and `fold_words` pads each fold to `log2_ceil(len)` variables.
	let public_words = value_vec.public();
	let hidden_words = value_vec.non_public();
	let subspace = BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();

	// Prepare the operator data. Lambda sampling is cheap and not part of any benched phase, so a
	// random lambda (rather than one drawn from a transcript) yields realistic-magnitude data.
	// SHA256 has no BMUL constraints, so that one is a zero claim at an empty point, matching the
	// real prover (`prove.rs` `None` branch).
	let prepared = OperatorClaims {
		zero: OperatorData {
			evals: zero_evals,
			r_zhat_prime,
			r_x_prime: r_x_prime_zero,
		},
		bitand: OperatorData {
			evals: bitand_evals,
			r_zhat_prime,
			r_x_prime: r_x_prime_bitand,
		},
		intmul: OperatorData {
			evals: intmul_evals,
			r_zhat_prime,
			r_x_prime: r_x_prime_intmul,
		},
		binmul: OperatorData::zero_claim(r_zhat_prime),
	}
	.prepare(|| F::random(&mut rng));

	// The phases are sequential and stateful: each one consumes the previous phase's outputs.
	//
	// Rather than re-deriving predecessors inside each phase's own per-iteration setup, the
	// protocol runs once here, ahead of time, with a throwaway transcript, to capture each
	// phase's inputs.
	// The benchmark closures below then only clone what a phase consumes by value.
	// The specific transcript challenges do not change the work a phase performs, so reusing
	// them here does not bias any of the timings below.
	//
	// The witness-and-batching multilinear is built once per key segment; the combined
	// multilinear is the two segments' rows concatenated.
	let build_combined_g = || {
		let public = key_collection
			.public
			.build_g::<F, P>(public_words, &prepared);
		let hidden = key_collection
			.hidden
			.build_g::<F, P>(hidden_words, &prepared);
		SparseShiftRows::from_segments([
			(&public, &key_collection.public.dense_shift_enc),
			(&hidden, &key_collection.hidden.dense_shift_enc),
		])
	};

	let g = build_combined_g();
	let oblong_weights = subspace_lagrange_evals(&subspace, prepared.bitand.r_zhat_prime);
	let Phase1Output {
		r_j,
		inner: inner_shift,
		outer: outer_shift,
		psi: _,
		gamma,
		g_eval: _,
	} = {
		let mut transcript = ProverTranscript::<StdChallenger>::default();
		g.clone().run_phase_1_sumcheck(
			oblong_weights.as_ref(),
			prepared.batched_eval(),
			&mut transcript,
			&GlobalAllocator,
		)
	};
	// The bit-index phases' rounds are not benchmarked; the last phase only needs the scalar they
	// reduce their factors to, so a stand-in value serves.
	let shift_ind_eval = F::random(&mut rng);
	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);
	let public_folded = fold_words::<F, P, _>(&GlobalAllocator, public_words, r_j_tensor.as_ref());
	let hidden_folded = fold_words::<F, P, _>(&GlobalAllocator, hidden_words, r_j_tensor.as_ref());
	let (public_monster, hidden_monster) = key_collection.build_monster_segments::<F, P, _>(
		&GlobalAllocator,
		&prepared,
		shift_ind_eval,
		&inner_shift,
		&outer_shift,
	);

	let mut group = c.benchmark_group("shift_reduction_phases");
	group.sample_size(10);

	// Phase 1.
	//
	// The row-building and weight-table steps take their inputs by reference.
	// So neither needs a per-iteration clone.
	// The sumcheck step consumes the row list by value, so its benchmark clones it once per
	// iteration instead.
	group.bench_function("phase1_build_g_parts", |b| {
		b.iter(&build_combined_g);
	});
	group.bench_function("phase1_build_h_parts", |b| {
		b.iter(|| shift_operator_table::<F, P, _>(&GlobalAllocator, oblong_weights.as_ref()));
	});
	group.bench_function("phase1_run_sumcheck", |b| {
		b.iter_batched(
			|| g.clone(),
			|g| {
				let mut transcript = ProverTranscript::<StdChallenger>::default();
				g.run_phase_1_sumcheck(
					oblong_weights.as_ref(),
					prepared.batched_eval(),
					&mut transcript,
					&GlobalAllocator,
				)
			},
			BatchSize::SmallInput,
		);
	});

	// Phase 2 builds its monster segments from its inputs by reference; the sumcheck that
	// follows consumes its buffers and challenge point by value.
	group.bench_function("phase2_build_monster_segments", |b| {
		b.iter(|| {
			key_collection.build_monster_segments::<F, P, _>(
				&GlobalAllocator,
				&prepared,
				shift_ind_eval,
				&inner_shift,
				&outer_shift,
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
				run_sumcheck::<F, P, _, _>(
					&public_folded,
					hidden_folded,
					&public_monster,
					hidden_monster,
					shift_ind_eval,
					public_words,
					r_j,
					gamma,
					&mut transcript,
					&GlobalAllocator,
				)
			},
			BatchSize::SmallInput,
		);
	});

	group.finish();
}

criterion_group!(benches, bench_prove_and_verify, bench_shift_phases);
criterion_main!(benches);
