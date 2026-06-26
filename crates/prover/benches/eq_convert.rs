// Copyright 2025 Irreducible Inc.
//! Head-to-head benchmark for the AND-reduction univariate round message.
//!
//! Compares the unoptimized equality weighting, which pays one large-field multiply per hypercube
//! vertex, against the convert-table path, which tabulates the innermost equality coordinates and
//! replaces those multiplies with lookups and XORs.

use std::{iter, iter::repeat_with};

use binius_core::word::Word;
use binius_field::{AESTowerField8b, PackedAESBinaryField64x8b, Random};
use binius_math::{BinarySubspace, multilinear::eq::eq_ind_partial_eval};
use binius_prover::and_reduction::{
	NTTLookup, eq_convert::univariate_round_message,
	sumcheck_round_messages::univariate_round_message_extension_domain,
};
use binius_verifier::{config::B128, protocols::bitand::SKIPPED_VARS};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

fn bench(c: &mut Criterion) {
	// Problem size: 2^24 total variables, so 2^18 words per column.
	let log_num_rows = 24;
	let n_words = 1usize << (log_num_rows - SKIPPED_VARS);

	let mut rng = StdRng::seed_from_u64(0);

	// Random a, b columns and c = a & b, a satisfying AND-constraint witness.
	let first: Vec<Word> = repeat_with(|| Word(rng.random())).take(n_words).collect();
	let second: Vec<Word> = repeat_with(|| Word(rng.random())).take(n_words).collect();
	let third: Vec<Word> = iter::zip(&first, &second).map(|(&a, &b)| a & b).collect();

	// The three small coordinates handled in F_{2^8}, shared by both paths.
	let small_challenges = [
		AESTowerField8b::new(0x2),
		AESTowerField8b::new(0x4),
		AESTowerField8b::new(0x10),
	];

	// Random large-field coordinates.
	let big: Vec<B128> = repeat_with(|| B128::random(&mut rng))
		.take(log_num_rows - SKIPPED_VARS - small_challenges.len())
		.collect();

	// The lookup-based low-degree-extension table for the skipped bit-index variables.
	let prover_message_domain = BinarySubspace::<AESTowerField8b>::with_dim(SKIPPED_VARS + 1);
	let ntt_lookup = NTTLookup::<PackedAESBinaryField64x8b>::new(&prover_message_domain);

	let mut group = c.benchmark_group("eq_convert");
	group.throughput(Throughput::Elements(n_words as u64));

	// Reference: expand the full large-field equality buffer, one large-field multiply per vertex.
	group.bench_function(format!("reference 2^{log_num_rows}"), |bench| {
		bench.iter(|| {
			let eq_big = eq_ind_partial_eval::<B128>(&big);
			let urm: [B128; _] = univariate_round_message_extension_domain(
				&first,
				&second,
				&third,
				&eq_big,
				&ntt_lookup,
				&small_challenges,
			);
			urm
		});
	});

	// Optimized: tabulate the innermost coordinates, lookups and XORs in place of multiplies.
	group.bench_function(format!("eq_convert 2^{log_num_rows}"), |bench| {
		bench.iter(|| {
			let urm: [B128; _] = univariate_round_message::<B128, PackedAESBinaryField64x8b>(
				&first,
				&second,
				&third,
				&big,
				&ntt_lookup,
				&small_challenges,
			);
			urm
		});
	});
}

criterion_group!(eq_convert, bench);
criterion_main!(eq_convert);
