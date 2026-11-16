// Copyright 2026 The Binius Developers

use std::ops::Mul;

use binius_core::word::Word;
use binius_field::{Field, FieldOps, Ghash128b as B128, arch::OptimalPackedB128};
use binius_ip::sumcheck::verify;
use binius_math::{
	FieldBuffer,
	multilinear::{eq::eq_ind, evaluate::evaluate},
	test_utils::random_scalars,
};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
	Composition, Compute, ExecStats, FakeDeviceCompute, ReferenceCompute, SimdCompute, scalar,
	sumcheck::{self, RoundKernel},
};

type Simd = SimdCompute<OptimalPackedB128>;

fn rng() -> StdRng {
	StdRng::seed_from_u64(0)
}

fn download<C: Compute<B128>>(c: &C, v: C::FieldVec) -> Vec<B128> {
	c.exec(c.to_host(vec![v])).remove(0)
}

fn random_words(rng: &mut StdRng, n: usize) -> Vec<Word> {
	(0..n).map(|_| Word::from_u64(rng.next_u64())).collect()
}

/// `x0 * x1`.
struct Product;

impl Composition<B128> for Product {
	const N_IN: usize = 2;
	const DEGREE: usize = 2;

	fn eval<T: FieldOps<Scalar = B128> + Mul<B128, Output = T>>(&self, x: &[T]) -> T {
		x[0].clone() * &x[1]
	}
}

/// `eq * (a * b - c)`.
struct MleCheck;

impl Composition<B128> for MleCheck {
	const N_IN: usize = 4;
	const DEGREE: usize = 3;

	fn eval<T: FieldOps<Scalar = B128> + Mul<B128, Output = T>>(&self, x: &[T]) -> T {
		x[0].clone() * (x[1].clone() * &x[2] - &x[3])
	}
}

/// `eq * (a * b - c)` where `eq` is broadcast (constant along each line): degree 2.
struct EqMleCheck;

impl Composition<B128> for EqMleCheck {
	const N_IN: usize = 4;
	const DEGREE: usize = 2;

	fn eval<T: FieldOps<Scalar = B128> + Mul<B128, Output = T>>(&self, x: &[T]) -> T {
		x[0].clone() * (x[1].clone() * &x[2] - &x[3])
	}
}

fn check_tensor_expand<C: Compute<B128>>(c: &C) {
	let point = random_scalars::<B128>(&mut rng(), 7);
	let got = download(c, c.exec(c.tensor_expand(&point)));
	assert_eq!(got, binius_math::multilinear::eq::eq_ind_partial_eval_scalars(&point));
}

fn check_inner_product_and_evaluate<C: Compute<B128>>(c: &C) {
	let mut rng = rng();
	// Include lengths below one packed element.
	for log_len in [0, 1, 2, 7] {
		let a = random_scalars::<B128>(&mut rng, 1 << log_len);
		let b = random_scalars::<B128>(&mut rng, 1 << log_len);
		let point = random_scalars::<B128>(&mut rng, log_len);
		let (a_dev, b_dev) = (c.copy_field_from_host(&a), c.copy_field_from_host(&b));
		let (a_s, b_s) = (c.as_slice(&a_dev), c.as_slice(&b_dev));
		let (ip, ev) = c.exec(c.join(c.inner_product(a_s, b_s), c.evaluate(a_s, &point)));
		assert_eq!(ip, a.iter().zip(&b).map(|(&x, &y)| x * y).sum::<B128>());
		assert_eq!(ev, evaluate(&FieldBuffer::<B128>::from_values(&a), &point));
	}
}

fn check_fold_words<C: Compute<B128>>(c: &C) {
	let mut rng = rng();
	let weights: [B128; 64] = random_scalars::<B128>(&mut rng, 64).try_into().unwrap();
	for n_words in [1, 5, 64, 100] {
		let words = random_words(&mut rng, n_words);
		let dev = c.copy_words_from_host(&words);
		let got = download(c, c.exec(c.fold_words(c.as_word_slice(&dev), &weights)));
		assert_eq!(got, scalar::fold_words(&words, &weights));
	}
}

fn check_fold_highest_var<C: Compute<B128>>(c: &C) {
	let mut rng = rng();
	let r = random_scalars::<B128>(&mut rng, 1)[0];
	let cols: Vec<Vec<B128>> = (0..2).map(|_| random_scalars(&mut rng, 1 << 6)).collect();
	let dev = cols.iter().map(|col| c.copy_field_from_host(col)).collect();
	let got = c.exec(c.then(c.fold_highest_var(dev, r), |cols| c.to_host(cols)));
	for (col, got) in cols.into_iter().zip(got) {
		let mut want = col;
		scalar::fold_highest_var(&mut want, r);
		assert_eq!(got, want);
	}
}

fn check_pair_map_reduce<C: Compute<B128>>(c: &C) {
	let mut rng = rng();
	for log_len in [1, 2, 6] {
		let cols: Vec<Vec<B128>> = (0..4)
			.map(|_| random_scalars(&mut rng, 1 << log_len))
			.collect();
		let dev = cols.iter().map(|col| c.copy_field_from_host(col)).collect();
		let kernel = RoundKernel(&MleCheck);
		let (_, got) = c.exec(c.pair_map_reduce(dev, &kernel));
		let views: Vec<&[B128]> = cols.iter().map(Vec::as_slice).collect();
		assert_eq!(got, scalar::pair_map_reduce(&views, &kernel));
	}
}

fn run_ops<C: Compute<B128>>(c: &C) {
	check_tensor_expand(c);
	check_inner_product_and_evaluate(c);
	check_fold_words(c);
	check_fold_highest_var(c);
	check_pair_map_reduce(c);
}

#[test]
fn ops_reference() {
	run_ops(&ReferenceCompute);
}

#[test]
fn ops_simd() {
	run_ops(&Simd::new());
}

#[test]
fn ops_fake_device() {
	run_ops(&FakeDeviceCompute::<B128>::default());
}

type Transcript = ProverTranscript<HasherChallenger<sha2::Sha256>>;

/// Verifies the proof with the real verifier; returns the evaluation point (lowest variable
/// first), the final column evaluations, and the verifier's reduced evaluation.
fn verify_proof(
	mut transcript: Transcript,
	evals: &[B128],
	challenges: &[B128],
	degree: usize,
	claim: B128,
) -> (Vec<B128>, B128) {
	transcript.message().write_scalar_slice(evals);
	let mut verifier = transcript.into_verifier();
	let output = verify(challenges.len(), degree, claim, &mut verifier).unwrap();
	assert_eq!(output.challenges, challenges);
	let mut point = output.challenges;
	point.reverse();
	(point, output.eval)
}

fn bivariate_product<C: Compute<B128>>(c: &C) {
	let n = 10;
	let mut rng = rng();
	let a = random_scalars::<B128>(&mut rng, 1 << n);
	let b = random_scalars::<B128>(&mut rng, 1 << n);
	let claim = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
	let cols = vec![c.copy_field_from_host(&a), c.copy_field_from_host(&b)];

	let mut transcript = Transcript::new(Default::default());
	let (challenges, evals) = sumcheck::prove(c, &Product, n, cols, claim, &mut transcript);
	let (point, eval) = verify_proof(transcript, &evals, &challenges, 2, claim);

	assert_eq!(eval, evals[0] * evals[1]);
	assert_eq!(evals[0], evaluate(&FieldBuffer::<B128>::from_values(&a), &point));
	assert_eq!(evals[1], evaluate(&FieldBuffer::<B128>::from_values(&b), &point));
}

fn mlecheck<C: Compute<B128>>(c: &C) {
	let n = 10;
	let mut rng = rng();
	let point = random_scalars::<B128>(&mut rng, n);
	let weights: [[B128; 64]; 2] =
		[0, 1].map(|_| random_scalars::<B128>(&mut rng, 64).try_into().unwrap());
	let words: Vec<Vec<Word>> = (0..2).map(|_| random_words(&mut rng, 1 << n)).collect();
	let word_devs: Vec<C::WordVec> = words.iter().map(|w| c.copy_words_from_host(w)).collect();

	// One exec builds the eq tensor and both folded columns.
	let (eq, (a, b)) = c.exec(c.join(
		c.tensor_expand(&point),
		c.join(
			c.fold_words(c.as_word_slice(&word_devs[0]), &weights[0]),
			c.fold_words(c.as_word_slice(&word_devs[1]), &weights[1]),
		),
	));
	let ab = c.exec(c.to_host(vec![a, b]));
	let prod: Vec<B128> = ab[0].iter().zip(&ab[1]).map(|(&x, &y)| x * y).collect();
	// c = a * b is computed on the host and uploaded; a and b are re-uploaded for ownership.
	let cols = vec![
		eq,
		c.copy_field_from_host(&ab[0]),
		c.copy_field_from_host(&ab[1]),
		c.copy_field_from_host(&prod),
	];

	let claim = B128::ZERO;
	let mut transcript = Transcript::new(Default::default());
	let (challenges, evals) = sumcheck::prove(c, &MleCheck, n, cols, claim, &mut transcript);
	let (rho, eval) = verify_proof(transcript, &evals, &challenges, 3, claim);

	assert_eq!(eval, evals[0] * (evals[1] * evals[2] - evals[3]));
	assert_eq!(evals[0], eq_ind(&point, &rho));
	assert_eq!(evals[1], evaluate(&FieldBuffer::<B128>::from_values(&ab[0]), &rho));
	assert_eq!(evals[2], evaluate(&FieldBuffer::<B128>::from_values(&ab[1]), &rho));
}

/// Gruen-factored MLE-check of `sum_x eq(point, x) (a b - c)` with a random, unsatisfied `c`.
fn gruen_mlecheck<C: Compute<B128>>(c: &C) {
	let n = 10;
	let mut rng = rng();
	let point = random_scalars::<B128>(&mut rng, n);
	let [a, b, cc] = [0, 1, 2].map(|_| random_scalars::<B128>(&mut rng, 1 << n));
	let eq = binius_math::multilinear::eq::eq_ind_partial_eval_scalars(&point);
	let claim: B128 = (0..1 << n).map(|i| eq[i] * (a[i] * b[i] - cc[i])).sum();
	let cols = [&a, &b, &cc].map(|col| c.copy_field_from_host(col)).into();

	let mut transcript = Transcript::new(Default::default());
	let (challenges, evals) =
		sumcheck::prove_mlecheck(c, &EqMleCheck, &point, cols, claim, &mut transcript);
	transcript.message().write_scalar_slice(&evals);
	let mut verifier = transcript.into_verifier();
	let output = binius_ip::mlecheck::verify(&point, 2, claim, &mut verifier).unwrap();
	assert_eq!(output.challenges, challenges);
	let mut rho = challenges;
	rho.reverse();

	// The verifier's eval has the eq factor stripped.
	assert_eq!(output.eval, evals[0] * evals[1] - evals[2]);
	for (col, eval) in [&a, &b, &cc].into_iter().zip(&evals) {
		assert_eq!(*eval, evaluate(&FieldBuffer::<B128>::from_values(col), &rho));
	}
}

#[test]
fn sumcheck_reference() {
	bivariate_product(&ReferenceCompute);
	mlecheck(&ReferenceCompute);
	gruen_mlecheck(&ReferenceCompute);
}

#[test]
fn sumcheck_simd() {
	bivariate_product(&Simd::new());
	mlecheck(&Simd::new());
	gruen_mlecheck(&Simd::new());
}

/// Asserts one sync per round, no hidden syncs, and `launches` launches in every round exec
/// after the first (`first` in the first), and in the final fold exec (1).
fn assert_rounds(per_exec: &[ExecStats], first: usize, rest: usize) {
	let (last, rounds) = per_exec.split_last().unwrap();
	assert!(per_exec.iter().all(|e| e.hidden_syncs == 0), "{per_exec:?}");
	assert_eq!(rounds[0].launches, first);
	assert!(rounds[1..].iter().all(|e| e.launches == rest), "{per_exec:?}");
	assert_eq!(last.launches, 1);
}

#[test]
fn sumcheck_fake_device_counts() {
	let n = 10;
	let device = FakeDeviceCompute::<B128>::default();
	bivariate_product(&device);
	let s = device.stats();
	// Two uploads of 2^n; each round and the final read is one exec; no bulk copies after setup.
	assert_eq!((s.h2d_calls, s.h2d_elems), (2, 2 << n));
	assert_eq!(s.execs, n + 1);
	assert_eq!(s.launches, 2 * n);
	assert_eq!((s.d2h_calls, s.d2h_elems), (n + 1, n * 2 + 2));
	assert_rounds(&s.per_exec, 1, 2);
	println!("bivariate product: {s:?}");

	let device = FakeDeviceCompute::<B128>::default();
	mlecheck(&device);
	let s = device.stats();
	// Setup: 2 word uploads + 3 column uploads, and a download of a and b. Then n + 1 execs of
	// O(degree) scalars each, plus the 4 final evaluations.
	assert_eq!((s.h2d_calls, s.h2d_elems), (5, (2 << n) + (3 << n)));
	assert_eq!(s.execs, 2 + n + 1);
	assert_eq!(s.d2h_elems, (2 << n) + n * 3 + 4);
	// Execs 0 (setup) and 1 (download) precede the sumcheck.
	assert_eq!(s.per_exec[0].launches, 3);
	assert_rounds(&s.per_exec[2..], 1, 2);
	println!("mlecheck: {s:?}");

	let device = FakeDeviceCompute::<B128>::default();
	gruen_mlecheck(&device);
	let s = device.stats();
	// Per round: tensor_expand + fold + pair_map_reduce (the first round has no fold).
	assert_eq!(s.execs, n + 1);
	assert_eq!((s.h2d_calls, s.h2d_elems), (3, 3 << n));
	assert_eq!(s.d2h_elems, n * 2 + 3);
	assert_rounds(&s.per_exec, 2, 3);
	println!("gruen mlecheck: {s:?}");
}

#[test]
fn fake_device_counts_hidden_sync() {
	let device = FakeDeviceCompute::<B128>::default();
	let col = device.copy_field_from_host(&[B128::ONE; 4]);
	let ip = device.inner_product(device.as_slice(&col), device.as_slice(&col));
	// The closure runs while the inner product's scalar is pending on the device.
	device.exec(device.then(ip, |_| device.to_host(vec![])));
	let s = device.stats();
	assert_eq!(s.hidden_syncs, 1);
	assert_eq!(
		s.per_exec[0],
		ExecStats {
			launches: 1,
			hidden_syncs: 1
		}
	);
}
