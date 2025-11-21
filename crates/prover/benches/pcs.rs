// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField, arch::OptimalPackedB128};
use binius_math::{
	BinarySubspace,
	multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate_inplace},
	ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
	test_utils::{random_field_buffer, random_scalars},
};
use binius_prover::{
	fri::CommitOutput, hash::parallel_compression::ParallelCompressionAdaptor,
	merkle_tree::prover::BinaryMerkleTreeProver, pcs::OneBitPCSProver, ring_switch,
};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::{
	config::{B1, B128, StdChallenger},
	fri::FRIParams,
	hash::{StdCompression, StdDigest},
	pcs,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn bench_pcs(c: &mut Criterion) {
	let mut group = c.benchmark_group("pcs");

	type P = OptimalPackedB128;

	for log_len in [12, 16, 20] {
		const LOG_INV_RATE: usize = 1;
		const SECURITY_BITS: usize = 96;
		const ARITY: usize = 4;

		// Calculate throughput based on the input message size in bytes
		let message_bytes = (1 << log_len) * B128::N_BITS / 8;
		group.throughput(Throughput::Bytes(message_bytes as u64));

		let mut rng = rand::rng();
		let packed_multilin = random_field_buffer::<P>(&mut rng, log_len);

		let compression = ParallelCompressionAdaptor::new(StdCompression::default());
		let merkle_prover = BinaryMerkleTreeProver::<B128, StdDigest, _>::new(compression);

		let subspace =
			BinarySubspace::<B128>::with_dim(log_len).expect("Failed to create subspace");
		let domain_context = GenericPreExpanded::generate_from_subspace(&subspace);
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		let fri_params = FRIParams::choose_with_constant_fold_arity(
			&ntt,
			log_len,
			SECURITY_BITS,
			LOG_INV_RATE,
			ARITY,
		)
		.expect("Failed to create FRI params");

		let pcs_prover = OneBitPCSProver::new(&ntt, &merkle_prover, &fri_params);

		group.bench_function(format!("commit/log_len={log_len}"), |b| {
			b.iter(|| pcs_prover.commit(packed_multilin.clone()).unwrap());
		});

		// Commit the packed multilinear
		let CommitOutput {
			commitment: codeword_commitment,
			committed: codeword_committed,
			codeword,
		} = pcs_prover
			.commit(packed_multilin.clone())
			.expect("Failed to commit");

		let mut transcript = ProverTranscript::new(StdChallenger::default());
		transcript.message().write(&codeword_commitment);

		// Generate random evaluation point
		let eval_point =
			random_scalars(&mut rng, log_len + <B128 as ExtensionField<B1>>::LOG_DEGREE);

		let (prefix, suffix) = eval_point.split_at(<B128 as ExtensionField<B1>>::LOG_DEGREE);
		let prefix_tensor = eq_ind_partial_eval(prefix);
		let partial = ring_switch::fold_b128_elems_inplace(packed_multilin.clone(), &prefix_tensor);
		let eval = evaluate_inplace(partial, suffix).unwrap();
		transcript.message().write_scalar(eval);

		group.bench_function(format!("prove/log_len={log_len}"), |b| {
			b.iter(|| {
				let mut transcript = transcript.clone();
				pcs_prover
					.prove(
						codeword.as_ref(),
						&codeword_committed,
						packed_multilin.clone(),
						eval_point.clone(),
						&mut transcript,
					)
					.unwrap()
			});
		});

		pcs_prover
			.prove(
				codeword.as_ref(),
				&codeword_committed,
				packed_multilin.clone(),
				eval_point.clone(),
				&mut transcript,
			)
			.unwrap();

		let proof = transcript.finalize();
		println!("Proof size {} B", proof.len());

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let commitment = verifier_transcript.message().read().unwrap();
		let eval = verifier_transcript.message().read_scalar().unwrap();

		group.bench_function(format!("verify/log_len={log_len}"), |b| {
			b.iter(|| {
				let mut transcript = verifier_transcript.clone();
				pcs::verify(
					&mut transcript,
					eval,
					&eval_point,
					commitment,
					&fri_params,
					merkle_prover.scheme(),
				)
				.unwrap()
			});
		});
	}

	group.finish();
}

criterion_group!(pcs, bench_pcs);
criterion_main!(pcs);
