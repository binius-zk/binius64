// Copyright 2026 The Binius Developers

use binius_field::{BinaryField128bGhash as B128, Random, arch::OptimalPackedB128};
use binius_hash::{ParallelCompressionAdaptor, StdCompression, StdDigest};
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator},
	circuits::powers,
	compiler::compile,
};
use binius_spartan_prover::Prover;
use binius_spartan_verifier::{Verifier, config::StdChallenger};
use binius_transcript::ProverTranscript;
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

fn power_n_circuit<Builder: CircuitBuilder>(
	builder: &mut Builder,
	x_wire: Builder::Wire,
	y_wire: Builder::Wire,
	n: usize,
) {
	let powers_vec = powers(builder, x_wire, n);
	builder.assert_eq(powers_vec[n - 1], y_wire);
}

fn compute_power(x: B128, n: usize) -> B128 {
	(1..n).fold(x, |acc, _| acc * x)
}

fn bench_spartan_prove(c: &mut Criterion) {
	let mut group = c.benchmark_group("spartan_prove");
	group.sample_size(10);

	for n_muls in [1024, 4096, 16384] {
		group.bench_function(format!("{n_muls}_muls"), |b| {
			let mut constraint_builder = ConstraintBuilder::new();
			let x_wire = constraint_builder.alloc_inout();
			let y_wire = constraint_builder.alloc_inout();
			power_n_circuit(&mut constraint_builder, x_wire, y_wire, n_muls);
			let (cs, layout) = compile(constraint_builder);

			let log_inv_rate = 1;
			let compression = StdCompression::default();
			let verifier =
				Verifier::<_, StdDigest, _>::setup(cs, log_inv_rate, compression.clone())
					.expect("verifier setup failed");
			let prover = Prover::<OptimalPackedB128, _, StdDigest>::setup(
				verifier.clone(),
				ParallelCompressionAdaptor::new(compression),
			)
			.expect("prover setup failed");

			let cs = verifier.constraint_system();
			let layout = layout.with_blinding(cs.blinding_info().clone());

			let mut rng = StdRng::seed_from_u64(0);
			let x_val = B128::random(&mut rng);
			let y_val = compute_power(x_val, n_muls);

			b.iter(|| {
				let mut witness_gen = WitnessGenerator::new(&layout);
				let x_assigned = witness_gen.write_inout(x_wire, x_val);
				let y_assigned = witness_gen.write_inout(y_wire, y_val);
				power_n_circuit(&mut witness_gen, x_assigned, y_assigned, n_muls);
				let witness = witness_gen.build().expect("failed to build witness");

				let mut rng = StdRng::seed_from_u64(42);
				let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
				prover
					.prove(&witness, &mut rng, &mut prover_transcript)
					.expect("prove failed");
			});
		});
	}

	group.finish();
}

criterion_group!(benches, bench_spartan_prove);
criterion_main!(benches);
