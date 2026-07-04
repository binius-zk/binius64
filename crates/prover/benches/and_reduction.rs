// Copyright 2025 Irreducible Inc.
use std::{iter, iter::repeat_with};

use binius_core::word::Word;
use binius_field::{
	Field, Random,
	linear_transformation::{
		BytewiseLookupTransformationFactory, LinearTransformationFactory,
		OutputWrappingTransformationFactory,
	},
};
use binius_ip_prover::sumcheck::{common::SumcheckProver, quadratic_mle::QuadraticMleCheckProver};
use binius_math::{
	BinarySubspace,
	univariate::{extrapolate_over_subspace, lagrange_evals_scalars},
};
use binius_prover::{
	OptimalPackedB128,
	and_reduction::{
		NTTLookup, sumcheck_round_messages::univariate_round_message_extension_domain,
	},
	fold_word::fold_words_with_transform,
};
use binius_verifier::{
	config::{B128, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES},
	protocols::bitand::{ROWS_PER_HYPERCUBE_VERTEX, SKIPPED_VARS},
};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

fn bench(c: &mut Criterion) {
	let mut rng = rand::rng();

	let log_words = 21;
	let big_field_zerocheck_challenges =
		vec![B128::random(&mut rng); log_words - PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len()];

	let a_words: Vec<Word> = repeat_with(|| Word(rng.random()))
		.take(1 << log_words)
		.collect();
	let b_words: Vec<Word> = repeat_with(|| Word(rng.random()))
		.take(1 << log_words)
		.collect();
	let c_words: Vec<Word> = iter::zip(&a_words, &b_words)
		.map(|(&a, &b)| a & b)
		.collect();

	let prover_message_domain = BinarySubspace::with_dim(SKIPPED_VARS + 1);

	let univariate_domain: BinarySubspace<B128> = prover_message_domain
		.reduce_dim(prover_message_domain.dim() - 1)
		.isomorphic();

	let mut group = c.benchmark_group("evaluate");
	group.bench_function("NTT lookup precompute", |bench| {
		bench.iter(|| NTTLookup::new(&prover_message_domain));
	});

	group.throughput(Throughput::Elements(1 << log_words));
	group.bench_function(format!("univariate_round_message 2^{log_words}"), |bench| {
		bench.iter(|| {
			univariate_round_message_extension_domain::<B128>(
				log_words,
				&a_words,
				&b_words,
				&c_words,
				&big_field_zerocheck_challenges,
				&prover_message_domain,
			)
		});
	});

	let urm = univariate_round_message_extension_domain::<B128>(
		log_words,
		&a_words,
		&b_words,
		&c_words,
		&big_field_zerocheck_challenges,
		&prover_message_domain,
	);
	let univariate_challenge = B128::random(&mut rng);

	group.bench_function(format!("univariate fold 2^{log_words}"), |bench| {
		bench.iter(|| {
			let lagrange_evals = lagrange_evals_scalars(&univariate_domain, univariate_challenge);
			let transform =
				OutputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory)
					.create(&lagrange_evals);

			[&a_words, &b_words, &c_words]
				.map(|mlv| fold_words_with_transform::<_, OptimalPackedB128, _>(&transform, mlv))
		});
	});

	let lagrange_evals = lagrange_evals_scalars(&univariate_domain, univariate_challenge);
	let transform = OutputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory)
		.create(&lagrange_evals);
	let proving_polys = [&a_words, &b_words, &c_words]
		.map(|mlv| fold_words_with_transform::<_, OptimalPackedB128, _>(&transform, mlv));

	let mut univariate_message_coeffs = vec![B128::ZERO; 2 * ROWS_PER_HYPERCUBE_VERTEX];
	univariate_message_coeffs[ROWS_PER_HYPERCUBE_VERTEX..2 * ROWS_PER_HYPERCUBE_VERTEX]
		.copy_from_slice(&urm);

	let next_round_claim = extrapolate_over_subspace(
		&prover_message_domain.clone().isomorphic::<B128>(),
		&univariate_message_coeffs,
		univariate_challenge,
	);

	group.bench_function(format!("remaining zerocheck 2^{log_words}"), |bench| {
		bench.iter_batched(
			|| proving_polys.clone(),
			|proving_polys| {
				let multilinear_zerocheck_challenges: Vec<_> =
					PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES
						.into_iter()
						.map(B128::from)
						.chain(big_field_zerocheck_challenges.iter().copied())
						.collect();

				let mut prover = QuadraticMleCheckProver::new(
					proving_polys,
					|[a, b, c]| a * b - c,
					|[a, b, _]| a * b,
					multilinear_zerocheck_challenges,
					next_round_claim,
				);

				for _ in 0..log_words {
					let _ = prover.execute();
					prover.fold(B128::random(&mut rng));
				}

				prover.finish()
			},
			BatchSize::SmallInput,
		);
	});
}

criterion_group!(univariate_round, bench);
criterion_main!(univariate_round);
