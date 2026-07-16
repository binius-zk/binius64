// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_field::{BinaryField128bGhash, Random, arch::OptimalPackedB128};
use binius_prover::protocols::binmul::{BinMulWitness, prove};
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, prelude::StdRng};

type F = BinaryField128bGhash;
type P = OptimalPackedB128;

/// The six word columns `(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi)` of a BinMul witness.
type WordColumns = (Vec<Word>, Vec<Word>, Vec<Word>, Vec<Word>, Vec<Word>, Vec<Word>);

/// Split a GHASH-field element into its `(lo, hi)` 64-bit word pair.
fn to_word_pair(elem: F) -> (Word, Word) {
	let value = u128::from(elem);
	(Word::from_u64(value as u64), Word::from_u64((value >> 64) as u64))
}

/// Build a valid BinMul witness with `1 << log_n` random constraints `a * b = c`.
fn random_witness(rng: &mut impl rand::Rng, log_n: usize) -> WordColumns {
	let n = 1 << log_n;
	let mut a_lo = Vec::with_capacity(n);
	let mut a_hi = Vec::with_capacity(n);
	let mut b_lo = Vec::with_capacity(n);
	let mut b_hi = Vec::with_capacity(n);
	let mut c_lo = Vec::with_capacity(n);
	let mut c_hi = Vec::with_capacity(n);

	for _ in 0..n {
		let a = F::random(&mut *rng);
		let b = F::random(&mut *rng);
		let c = a * b;

		let (a_lo_i, a_hi_i) = to_word_pair(a);
		let (b_lo_i, b_hi_i) = to_word_pair(b);
		let (c_lo_i, c_hi_i) = to_word_pair(c);

		a_lo.push(a_lo_i);
		a_hi.push(a_hi_i);
		b_lo.push(b_lo_i);
		b_hi.push(b_hi_i);
		c_lo.push(c_lo_i);
		c_hi.push(c_hi_i);
	}

	(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi)
}

fn bench_binmul_prove(c: &mut Criterion) {
	let mut group = c.benchmark_group("binmul/prove");

	let mut rng = StdRng::seed_from_u64(0);

	for log_n in [12, 16, 20] {
		group.throughput(Throughput::Elements(1 << log_n));
		group.bench_function(format!("n_vars={log_n}"), |b| {
			let (a_lo, a_hi, b_lo, b_hi, c_lo, c_hi) = random_witness(&mut rng, log_n);

			b.iter_batched_ref(
				|| ProverTranscript::new(StdChallenger::default()),
				|transcript| {
					let witness = BinMulWitness {
						a_lo: &a_lo,
						a_hi: &a_hi,
						b_lo: &b_lo,
						b_hi: &b_hi,
						c_lo: &c_lo,
						c_hi: &c_hi,
					};
					prove::<F, P, _>(&witness, transcript)
				},
				BatchSize::SmallInput,
			);
		});
	}

	group.finish();
}

criterion_group!(binmul, bench_binmul_prove);
criterion_main!(binmul);
