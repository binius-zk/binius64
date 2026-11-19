// Copyright 2026 The Binius Developers

//! `ComputeMleCheckProver` against the real `SharedMleCheckProver`: same inputs, same transcript.

use binius_compute::GlobalAllocator;
use binius_compute_hal::{
	Column, Composition, Compute, Elem, FakeDeviceCompute, ReferenceCompute, SimdCompute,
	mlecheck::{ComputeMleCheckProver, HiKernel, Pick, multilinear_eval, quadratic},
};
use binius_field::{Ghash128b as F, arch::OptimalPackedB128 as P};
use binius_ip_prover::sumcheck::{
	MleToSumCheckDecorator, PaddedSumcheckDecorator,
	batch::{batch_prove_and_write_evals, batch_prove_mle},
	common::MleCheckProver,
	mle_store::MleStore,
	multilinear_eval::MultilinearEvalEvaluator,
	prove_single_mlecheck,
	quadratic_mle_evaluator::QuadraticMleEvaluator,
	round_evaluator::{MleCheckRoundEvaluator, SharedMleCheckProver},
};
use binius_math::{
	FieldBuffer, multilinear::eq::eq_ind_partial_eval_scalars, test_utils::random_scalars,
};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use either::Either;
use rand::{SeedableRng, rngs::StdRng};

type Transcript = ProverTranscript<HasherChallenger<sha2::Sha256>>;
type Simd = SimdCompute<P>;
/// Transcript bytes, final evals and challenges.
type Out = (Vec<u8>, Vec<F>, Vec<F>);

struct AndC;

impl Composition<F> for AndC {
	const N_IN: usize = 3;
	const DEGREE: usize = 2;

	fn eval<T: Elem<F>>(&self, x: &[T]) -> T {
		x[0].clone() * x[1].clone() - x[2].clone()
	}
	fn eval_infinity<T: Elem<F>>(&self, x: &[T]) -> T {
		x[0].clone() * x[1].clone()
	}
}

/// Three columns `a, b, c` over `n` variables, a point, and the honest claims: the eq-weighted
/// sum of `a * b - c`, then of each column.
struct Inst {
	n: usize,
	point: Vec<F>,
	cols: Vec<Vec<F>>,
	quad: F,
	evals: Vec<F>,
}

fn inst(n: usize, seed: u64) -> Inst {
	let mut rng = StdRng::seed_from_u64(seed);
	let point = random_scalars::<F>(&mut rng, n);
	let cols: Vec<Vec<F>> = (0..3).map(|_| random_scalars(&mut rng, 1 << n)).collect();
	let eq = eq_ind_partial_eval_scalars(&point);
	let weighted = |f: &dyn Fn(usize) -> F| (0..1 << n).map(|i| eq[i] * f(i)).sum();
	Inst {
		n,
		quad: weighted(&|i| cols[0][i] * cols[1][i] - cols[2][i]),
		evals: cols.iter().map(|c| weighted(&|i| c[i])).collect(),
		point,
		cols,
	}
}

/// One claim goes through `prove_single_mlecheck`, several through `batch_prove_mle`.
fn run(prover: impl MleCheckProver<F>, single: bool) -> Out {
	let mut transcript = Transcript::new(Default::default());
	let (evals, challenges) = if single {
		let out = prove_single_mlecheck(prover, &mut transcript);
		(out.multilinear_evals, out.challenges)
	} else {
		let out = batch_prove_mle(vec![prover], &mut transcript);
		out.send_evals(&mut transcript);
		(out.multilinear_evals.concat(), out.challenges)
	};
	(transcript.finalize(), evals, challenges)
}

/// The real prover: the quadratic claim and, if `evals`, one `MultilinearEval` claim per column,
/// all on one store. The last column is borrowed if `lent`.
fn real<'a>(
	alloc: &'a GlobalAllocator,
	bufs: &'a [FieldBuffer<P>],
	i: &Inst,
	evals: bool,
	lent: bool,
) -> impl MleCheckProver<F> + 'a {
	type Eval<'a> = Box<dyn MleCheckRoundEvaluator<F, P> + 'a>;
	let mut store = MleStore::new(i.n, alloc);
	let ids: Vec<_> = bufs
		.iter()
		.enumerate()
		.map(|(k, b)| {
			if lent && k == 2 {
				store.push(b.as_view())
			} else {
				store.push_owned(b.clone())
			}
		})
		.collect();
	let quad = QuadraticMleEvaluator::new(
		[ids[0], ids[1], ids[2]],
		|[a, b, c]: [P; 3]| a * b - c,
		|[a, b, _]: [P; 3]| a * b,
	);
	let mut claims: Vec<(F, Eval<'_>)> = vec![(i.quad, Box::new(quad))];
	if evals {
		let each = |(&id, &claim)| (claim, Box::new(MultilinearEvalEvaluator::new(id)) as Eval<'_>);
		claims.extend(ids.iter().zip(&i.evals).map(each));
	}
	SharedMleCheckProver::new(store, claims, i.point.clone())
}

fn real_evals<'a>(
	alloc: &'a GlobalAllocator,
	bufs: &'a [FieldBuffer<P>],
	i: &Inst,
) -> impl MleCheckProver<F> + 'a {
	let mut store = MleStore::new(i.n, alloc);
	let claims: Vec<_> = (bufs.iter().zip(&i.evals))
		.map(|(b, &claim)| (claim, MultilinearEvalEvaluator::new(store.push_owned(b.clone()))))
		.collect();
	SharedMleCheckProver::new(store, claims, i.point.clone())
}

fn bufs(i: &Inst) -> Vec<FieldBuffer<P>> {
	i.cols.iter().map(|c| FieldBuffer::from_values(c)).collect()
}

/// The HAL prover of `real`; `lent` is the backend copy of the last column. Its kernel is a
/// tuple (quadratic, evals), the evals empty if `!evals`.
fn hal<'a, C: Compute<F>>(
	c: &'a C,
	lent: Option<&'a C::FieldVec>,
	i: &Inst,
	evals: bool,
) -> impl MleCheckProver<F> + 'a {
	let cols = i.cols.iter().enumerate().map(|(k, v)| match (k, lent) {
		(2, Some(l)) => Column::Borrowed(c.as_slice(l)),
		_ => Column::Owned(c.copy_field_from_host(v)),
	});
	let kernel = (
		quadratic(&AndC, 3, vec![0, 1, 2]),
		(0..3 * usize::from(evals))
			.map(|k| multilinear_eval(3, k))
			.collect::<Vec<_>>(),
	);
	let mut claims = vec![(i.quad, 2)];
	claims.extend(i.evals.iter().take(3 * usize::from(evals)).map(|&e| (e, 1)));
	ComputeMleCheckProver::new(c, cols.collect(), kernel, claims, i.point.clone())
}

fn hal_evals<'a, C: Compute<F>>(c: &'a C, i: &Inst) -> impl MleCheckProver<F> + 'a {
	let cols = i
		.cols
		.iter()
		.map(|v| Column::Owned(c.copy_field_from_host(v)));
	let kernel: Vec<Pick<HiKernel>> = (0..3).map(|k| multilinear_eval(3, k)).collect();
	let claims = i.evals.iter().map(|&e| (e, 1)).collect();
	ComputeMleCheckProver::new(c, cols.collect(), kernel, claims, i.point.clone())
}

/// (a) and (b) on one backend.
fn check_single<C: Compute<F>>(c: &C) {
	let alloc = GlobalAllocator;
	// Includes sizes below the packing width, and a single variable with a lent column.
	for (n, evals, lent) in [
		(1, true, true),
		(2, false, false),
		(3, true, true),
		(8, false, false),
		(8, true, true),
	]
	.into_iter()
	.chain([
		(10, false, false),
		(10, true, true),
		(10, true, false),
		(14, true, true),
		(14, false, false),
	]) {
		let i = inst(n, n as u64);
		let b = bufs(&i);
		let dev = c.copy_field_from_host(&i.cols[2]);
		let want = run(real(&alloc, &b, &i, evals, lent), !evals);
		let got = run(hal(c, lent.then_some(&dev), &i, evals), !evals);
		assert_eq!(got, want, "n = {n}, evals = {evals}, lent = {lent}");
		// A lent column is read, never consumed or overwritten.
		assert_eq!(c.exec(c.to_host(c.as_slice(&dev))), i.cols[2]);
	}
}

#[test]
fn single_reference() {
	check_single(&ReferenceCompute);
}

#[test]
fn single_simd() {
	check_single(&Simd::new());
}

#[test]
fn single_fake_device() {
	check_single(&FakeDeviceCompute::<F>::default());
}

/// (c): the rerand shape through the decorators, `Either<quadratic, evals>`, the evals padded.
fn batched(left: impl MleCheckProver<F>, right: impl MleCheckProver<F>, l: &Inst, r: &Inst) -> Out {
	let provers = vec![
		PaddedSumcheckDecorator::new(
			MleToSumCheckDecorator::new(Either::Left(left)),
			0,
			vec![l.quad],
			3,
		),
		PaddedSumcheckDecorator::new(
			MleToSumCheckDecorator::new(Either::Right(right)),
			l.n - r.n,
			r.evals.clone(),
			3,
		),
	];
	let mut transcript = Transcript::new(Default::default());
	let out = batch_prove_and_write_evals(provers, &mut transcript);
	(transcript.finalize(), out.multilinear_evals.concat(), out.challenges)
}

fn check_batched<C: Compute<F>>(c: &C) {
	let alloc = GlobalAllocator;
	let (l, r) = (inst(9, 1), inst(7, 2));
	let (lb, rb) = (bufs(&l), bufs(&r));
	let want = batched(real(&alloc, &lb, &l, false, false), real_evals(&alloc, &rb, &r), &l, &r);
	assert_eq!(batched(hal(c, None, &l, false), hal_evals(c, &r), &l, &r), want, "hal + hal");
	assert_eq!(batched(hal(c, None, &l, false), real_evals(&alloc, &rb, &r), &l, &r), want);
	assert_eq!(batched(real(&alloc, &lb, &l, false, false), hal_evals(c, &r), &l, &r), want);
}

#[test]
fn batched_reference() {
	check_batched(&ReferenceCompute);
}

#[test]
fn batched_simd() {
	check_batched(&Simd::new());
}

#[test]
fn batched_fake_device() {
	check_batched(&FakeDeviceCompute::<F>::default());
}

/// One exec per round plus the final read; nothing but the round's slots (and the one-element
/// columns of `finish`) comes back; the backend never re-uploads.
#[test]
fn fake_device_counts() {
	let (n, slots, cols) = (10, 2 + 3, 3);
	let device = FakeDeviceCompute::<F>::default();
	let i = inst(n, 0);
	let lent = device.copy_field_from_host(&i.cols[2]);
	run(hal(&device, Some(&lent), &i, true), false);
	let s = device.stats();
	assert_eq!(s.execs, n + 1);
	assert_eq!(s.launches, n + 1);
	assert_eq!(s.hidden_syncs, 0);
	assert!(
		s.per_exec
			.iter()
			.all(|e| e.launches == 1 && e.hidden_syncs == 0),
		"{s:?}"
	);
	// Uploads: the lent column and the two owned ones, once.
	assert_eq!(s.h2d_elems, 3 << n);
	assert_eq!(s.d2h_elems, n * slots + cols, "{s:?}");
	println!("{s:?}");
}
