// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_field::{Field, Ghash128b as B128, arch::OptimalPackedB128};
use binius_ip::sumcheck::verify;
use binius_math::{
	FieldBuffer,
	multilinear::{eq::eq_ind, evaluate::evaluate},
	test_utils::random_scalars,
};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
	Composition, Compute, Elem, ExecStats, FakeDeviceCompute, PairKernel, ReferenceCompute,
	SimdCompute, scalar,
	sumcheck::{self, RoundKernel},
};

type Simd = SimdCompute<OptimalPackedB128>;

fn rng() -> StdRng {
	StdRng::seed_from_u64(0)
}

fn download<C: Compute<B128>>(c: &C, v: &C::FieldVec) -> Vec<B128> {
	c.exec(c.to_host(c.as_slice(v)))
}

fn random_words(rng: &mut StdRng, n: usize) -> Vec<Word> {
	(0..n).map(|_| Word::from_u64(rng.next_u64())).collect()
}

/// `x0 * x1`.
struct Product;

impl Composition<B128> for Product {
	const N_IN: usize = 2;
	const DEGREE: usize = 2;

	fn eval<T: Elem<B128>>(&self, x: &[T]) -> T {
		x[0].clone() * x[1].clone()
	}
	fn eval_infinity<T: Elem<B128>>(&self, x: &[T]) -> T {
		self.eval(x)
	}
}

/// `eq * (a * b - c)` with eq as an ordinary column.
struct MleCheck;

impl Composition<B128> for MleCheck {
	const N_IN: usize = 4;
	const DEGREE: usize = 3;

	fn eval<T: Elem<B128>>(&self, x: &[T]) -> T {
		x[0].clone() * (x[1].clone() * x[2].clone() - x[3].clone())
	}
	fn eval_infinity<T: Elem<B128>>(&self, x: &[T]) -> T {
		x[0].clone() * x[1].clone() * x[2].clone()
	}
}

/// `a * b - c`.
struct AndC;

impl Composition<B128> for AndC {
	const N_IN: usize = 3;
	const DEGREE: usize = 2;

	fn eval<T: Elem<B128>>(&self, x: &[T]) -> T {
		x[0].clone() * x[1].clone() - x[2].clone()
	}
	fn eval_infinity<T: Elem<B128>>(&self, x: &[T]) -> T {
		x[0].clone() * x[1].clone()
	}
}

/// Outputs `lo[j]` and `hi[j]` of column `j`: their sums are the two halves' sums.
struct HalfSums(usize);

impl PairKernel<B128> for HalfSums {
	fn n_in(&self) -> usize {
		3
	}
	fn n_out(&self) -> usize {
		2
	}
	fn scratch(&self) -> usize {
		0
	}
	fn map<T: Elem<B128>>(&self, lo: &[T], hi: &[T], _: &mut [T], out: &mut [T]) {
		out[0] = lo[self.0].clone();
		out[1] = hi[self.0].clone();
	}
}

fn check_tensor_expand<C: Compute<B128>>(c: &C) {
	let point = random_scalars::<B128>(&mut rng(), 7);
	let got = download(c, &c.exec(c.tensor_expand(&point)));
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
		let got = download(c, &c.exec(c.fold_words(c.as_word_slice(&dev), &weights)));
		assert_eq!(got, scalar::fold_words(&words, &weights));
	}
}

fn check_fold_highest_var<C: Compute<B128>>(c: &C) {
	let mut rng = rng();
	let r = random_scalars::<B128>(&mut rng, 1)[0];
	let cols: Vec<Vec<B128>> = (0..2).map(|_| random_scalars(&mut rng, 1 << 6)).collect();
	let mut dev: Vec<C::FieldVec> = cols.iter().map(|col| c.copy_field_from_host(col)).collect();
	c.exec(c.fold_highest_var(&mut dev, r));
	for (col, dev) in cols.into_iter().zip(&dev) {
		let got = download(c, dev);
		let mut want = col;
		scalar::fold_highest_var(&mut want, r);
		assert_eq!(got, want);
	}
}

/// `fold_pair_map_reduce` of `[a, b, c]` against a direct sum over the hypercube, for post-fold
/// sizes `n` in `log_ns`, with and without a fold and with and without eq.
fn check_fold_pair_map_reduce<C: Compute<B128>>(c: &C, log_ns: std::ops::RangeInclusive<usize>) {
	let mut rng = rng();
	let r = random_scalars::<B128>(&mut rng, 1)[0];
	for n in log_ns {
		for (fold, weighted) in [(false, false), (false, true), (true, false), (true, true)] {
			let len = 1 << (n + usize::from(fold));
			let [a, b, cc] = [0, 1, 2].map(|_| random_scalars::<B128>(&mut rng, len));
			let point = random_scalars::<B128>(&mut rng, n - 1);
			let mut dev: Vec<C::FieldVec> =
				[&a, &b, &cc].map(|col| c.copy_field_from_host(col)).into();
			let got = c.exec(c.fold_pair_map_reduce(
				&mut dev,
				fold.then_some(r),
				weighted.then_some(point.as_slice()),
				&RoundKernel(&AndC),
			));

			let folded = |col: &Vec<B128>| -> Vec<B128> {
				if !fold {
					return col.clone();
				}
				let half = col.len() / 2;
				(0..half)
					.map(|i| col[i] + r * (col[i + half] - col[i]))
					.collect()
			};
			let [a, b, cc] = [a, b, cc].map(|col| folded(&col));
			let half = a.len() / 2;
			let eq = binius_math::multilinear::eq::eq_ind_partial_eval_scalars(&point);
			let (mut y1, mut y_inf) = (B128::ZERO, B128::ZERO);
			for i in 0..half {
				let w = if weighted { eq[i] } else { B128::ONE };
				y1 += w * (a[i + half] * b[i + half] - cc[i + half]);
				y_inf += w * (a[i] + a[i + half]) * (b[i] + b[i + half]);
			}
			assert_eq!(got, [y1, y_inf], "n = {n}, fold = {fold}, weighted = {weighted}");
			if fold {
				assert_eq!(download(c, &dev[0]), a);
			}
		}
	}
}

/// Two kernels (a pair, and a `Vec`) in one pass give the concatenated single-kernel outputs.
fn check_multi_kernel<C: Compute<B128>>(c: &C) {
	let mut rng = rng();
	let n = 8;
	let cols: Vec<Vec<B128>> = (0..3).map(|_| random_scalars(&mut rng, 1 << n)).collect();
	let point = random_scalars::<B128>(&mut rng, n - 1);
	let mut dev: Vec<C::FieldVec> = cols.iter().map(|col| c.copy_field_from_host(col)).collect();
	let kernel = (RoundKernel(&AndC), (0..3).map(HalfSums).collect::<Vec<_>>());
	for eq in [None, Some(point.as_slice())] {
		let got = c.exec(c.fold_pair_map_reduce(&mut dev, None, eq, &kernel));
		assert_eq!(got.len(), 2 + 6);
		let eq_vals = eq.map(binius_math::multilinear::eq::eq_ind_partial_eval_scalars);
		let w = |i: usize| eq_vals.as_ref().map_or(B128::ONE, |e| e[i]);
		for j in 0..3 {
			let half = 1 << (n - 1);
			let (lo, hi): (B128, B128) = (
				(0..half).map(|i| w(i) * cols[j][i]).sum(),
				(0..half).map(|i| w(i) * cols[j][i + half]).sum(),
			);
			assert_eq!(got[2 + 2 * j..4 + 2 * j], [lo, hi]);
		}
	}
}

/// A column of the wrong length must panic, not be treated as broadcast.
fn wrong_length_column<C: Compute<B128>>(c: &C) {
	let mut cols = [8, 2].map(|len| c.copy_field_from_host(&vec![B128::ONE; len]));
	c.exec(c.fold_pair_map_reduce(&mut cols, None, None, &RoundKernel(&Product)));
}

/// Evaluates `Comp` along a line at `DEGREE + 2` points and checks the degree is exactly `DEGREE`
/// and the leading coefficient is `eval_infinity` at the direction `lo + hi`.
fn check_degree<Comp: Composition<B128>>(comp: &Comp) {
	let mut rng = rng();
	let lo = random_scalars::<B128>(&mut rng, Comp::N_IN);
	let hi = random_scalars::<B128>(&mut rng, Comp::N_IN);
	let xs: Vec<B128> = [B128::ZERO, B128::ONE]
		.into_iter()
		.chain(sumcheck::extra_points(Comp::DEGREE + 2))
		.collect();
	let ys: Vec<B128> = xs
		.iter()
		.map(|&x| {
			let line: Vec<B128> = (0..Comp::N_IN)
				.map(|i| lo[i] + x * (lo[i] + hi[i]))
				.collect();
			comp.eval(&line)
		})
		.collect();
	let coeffs = sumcheck::interpolate(&xs, &ys);
	assert_eq!(coeffs[Comp::DEGREE + 1], B128::ZERO, "degree exceeds DEGREE");
	let dir: Vec<B128> = (0..Comp::N_IN).map(|i| lo[i] + hi[i]).collect();
	assert_ne!(coeffs[Comp::DEGREE], B128::ZERO, "degree is below DEGREE");
	assert_eq!(coeffs[Comp::DEGREE], comp.eval_infinity(&dir));
}

#[test]
fn composition_degrees() {
	check_degree(&Product);
	check_degree(&MleCheck);
	check_degree(&AndC);
}

fn run_ops<C: Compute<B128>>(c: &C) {
	check_tensor_expand(c);
	check_inner_product_and_evaluate(c);
	check_fold_words(c);
	check_fold_highest_var(c);
	check_fold_pair_map_reduce(c, 1..=6);
	check_multi_kernel(c);
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
fn simd_log_width_boundary() {
	let width = OptimalPackedB128::LOG_WIDTH;
	check_fold_pair_map_reduce(&Simd::new(), 1..=width + 3);
	// Past the chunk size, so the eq suffix is folded by the reduce tree.
	check_fold_pair_map_reduce(&Simd::new(), 13..=15);
}

#[test]
#[should_panic(expected = "log_len")]
fn wrong_length_reference() {
	wrong_length_column(&ReferenceCompute);
}

#[test]
#[should_panic(expected = "log_len")]
fn wrong_length_simd() {
	wrong_length_column(&Simd::new());
}

#[test]
#[should_panic(expected = "log_len")]
fn wrong_length_fake_device() {
	wrong_length_column(&FakeDeviceCompute::<B128>::default());
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
	let ab = c.exec(c.join_all(vec![c.to_host(c.as_slice(&a)), c.to_host(c.as_slice(&b))]));
	let prod: Vec<B128> = ab[0].iter().zip(&ab[1]).map(|(&x, &y)| x * y).collect();
	// c = a * b is computed on the host and uploaded.
	let cols = vec![eq, a, b, c.copy_field_from_host(&prod)];

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
		sumcheck::prove_mlecheck(c, &AndC, &point, cols, claim, &mut transcript);
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

/// Asserts one exec per round with no hidden syncs: `first` launches in the first round exec,
/// `rest` in the others, and 1 in the final fold-and-read exec.
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
	assert_eq!(s.launches, n + 1);
	assert_eq!((s.d2h_calls, s.d2h_elems), (n + 2, n * 2 + 2));
	assert_rounds(&s.per_exec, 1, 1);
	println!("bivariate product: {s:?}");

	let device = FakeDeviceCompute::<B128>::default();
	mlecheck(&device);
	let s = device.stats();
	// Setup: 2 word uploads + the c upload, and a download of a and b. Then n + 1 execs of
	// O(degree) scalars each, plus the 4 final evaluations.
	assert_eq!((s.h2d_calls, s.h2d_elems), (3, 3 << n));
	assert_eq!(s.execs, 2 + n + 1);
	assert_eq!(s.d2h_elems, (2 << n) + n * 3 + 4);
	// Execs 0 (setup) and 1 (download) precede the sumcheck.
	assert_eq!(s.per_exec[0].launches, 3);
	assert_rounds(&s.per_exec[2..], 1, 1);
	println!("mlecheck: {s:?}");

	let device = FakeDeviceCompute::<B128>::default();
	gruen_mlecheck(&device);
	let s = device.stats();
	// Eq lives inside the op: one launch per round, like the plain sumcheck.
	assert_eq!(s.execs, n + 1);
	assert_eq!((s.h2d_calls, s.h2d_elems), (3, 3 << n));
	assert_eq!(s.d2h_elems, n * 2 + 3);
	assert_rounds(&s.per_exec, 1, 1);
	println!("gruen mlecheck: {s:?}");
}

#[test]
fn fake_device_counts_hidden_sync() {
	let device = FakeDeviceCompute::<B128>::default();
	let col = device.copy_field_from_host(&[B128::ONE; 4]);
	let ip = device.inner_product(device.as_slice(&col), device.as_slice(&col));
	// The closure runs while the inner product's scalar is pending on the device.
	device.exec(device.then(ip, |_| device.to_host(device.as_slice(&col))));
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
