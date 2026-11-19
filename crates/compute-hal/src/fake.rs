// Copyright 2026 The Binius Developers

//! A simulated device: buffers live in a backend-owned arena and host code sees them only through
//! explicit downloads. Kernels are traced into expression graphs and interpreted per element.

// Every op holds the arena lock for its whole body by design.
#![allow(clippy::significant_drop_tightening)]

use std::{
	marker::PhantomData,
	ops::{Add, Mul, Sub},
	sync::{Arc, Mutex},
};

use binius_core::word::Word;
use binius_field::{BinaryField, Field};
use binius_math::{inner_product::inner_product, multilinear::eq::eq_ind_partial_eval_scalars};

use crate::{Column, Compute, Elem, Lazy, PairKernel, scalar};

/// Host/device traffic and dispatch counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
	/// Host-to-device copy calls.
	pub h2d_calls: usize,
	/// Elements (fields or words) copied host-to-device.
	pub h2d_elems: usize,
	/// Device-to-host copy calls, including reduction results.
	pub d2h_calls: usize,
	/// Elements copied device-to-host, including reduction results.
	pub d2h_elems: usize,
	/// Sync points (`exec` calls).
	pub execs: usize,
	/// Kernel launches.
	pub launches: usize,
	/// `then` closures that ran while device-computed scalars were pending: each would force a
	/// device-to-host sync in the middle of a pipeline.
	pub hidden_syncs: usize,
	/// One entry per `exec`, in order.
	pub per_exec: Vec<ExecStats>,
}

/// What one `exec` did. Each `exec` is itself one sync, at its return.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecStats {
	/// Kernel launches (op nodes executed).
	pub launches: usize,
	/// Hidden syncs inside the pipeline.
	pub hidden_syncs: usize,
}

struct Arena<F> {
	fields: Vec<Vec<F>>,
	words: Vec<Vec<Word>>,
	stats: Stats,
	cur: ExecStats,
	/// Host-typed results of device data produced since the last sync.
	pending_scalars: usize,
}

/// An opaque device buffer of `2^log_len` field elements.
#[derive(Debug)]
pub struct FakeVec {
	id: usize,
	log_len: usize,
}

/// A borrowed view of a [`FakeVec`].
#[derive(Clone, Copy, Debug)]
pub struct FakeSlice<'a> {
	id: usize,
	_borrow: PhantomData<&'a FakeVec>,
}

/// A borrowed view of a device word buffer.
#[derive(Clone, Copy, Debug)]
pub struct FakeWords<'a> {
	id: usize,
	_borrow: PhantomData<&'a ()>,
}

/// The simulated device. Cloning shares the arena.
pub struct FakeDeviceCompute<F> {
	arena: Arc<Mutex<Arena<F>>>,
}

impl<F> Default for FakeDeviceCompute<F> {
	fn default() -> Self {
		let arena = Arena {
			fields: vec![],
			words: vec![],
			stats: Stats::default(),
			cur: ExecStats::default(),
			pending_scalars: 0,
		};
		Self {
			arena: Arc::new(Mutex::new(arena)),
		}
	}
}

impl<F: Field> FakeDeviceCompute<F> {
	/// The counters so far.
	pub fn stats(&self) -> Stats {
		self.arena.lock().unwrap().stats.clone()
	}

	fn upload(&self, values: Vec<F>) -> FakeVec {
		let mut arena = self.arena.lock().unwrap();
		arena.stats.h2d_calls += 1;
		arena.stats.h2d_elems += values.len();
		push(&mut arena, values)
	}
}

fn launch<F: Field>(arena: &Mutex<Arena<F>>) -> std::sync::MutexGuard<'_, Arena<F>> {
	let mut guard = arena.lock().unwrap();
	guard.stats.launches += 1;
	guard.cur.launches += 1;
	guard
}

const fn id(col: &Column<FakeVec, FakeSlice<'_>>) -> usize {
	match col {
		Column::Owned(v) => v.id,
		Column::Borrowed(s) => s.id,
	}
}

/// Binds the highest variable of `col`; a borrowed column becomes a fresh owned device buffer.
fn fold_col<F: Field>(arena: &mut Arena<F>, col: &mut Column<FakeVec, FakeSlice<'_>>, r: F) {
	match col {
		Column::Owned(v) => {
			scalar::fold_highest_var(&mut arena.fields[v.id], r);
			v.log_len -= 1;
		}
		Column::Borrowed(s) => {
			let mut values = arena.fields[s.id].clone();
			scalar::fold_highest_var(&mut values, r);
			*col = Column::Owned(push(arena, values));
		}
	}
}

/// Stores `values` as a new device buffer (a device-side result, no transfer).
fn push<F: Field>(arena: &mut Arena<F>, values: Vec<F>) -> FakeVec {
	arena.fields.push(values);
	let id = arena.fields.len() - 1;
	FakeVec {
		id,
		log_len: arena.fields[id].len().ilog2() as usize,
	}
}

impl<F: BinaryField> Compute<F> for FakeDeviceCompute<F> {
	type FieldVec = FakeVec;
	type FieldSlice<'a> = FakeSlice<'a>;
	type WordVec = usize;
	type WordSlice<'a> = FakeWords<'a>;
	type Future<'a, T: Send + 'a> = Lazy<'a, T>;

	fn copy_field_from_host(&self, values: &[F]) -> FakeVec {
		assert!(values.len().is_power_of_two());
		self.upload(values.to_vec())
	}

	fn copy_words_from_host(&self, words: &[Word]) -> usize {
		let mut arena = self.arena.lock().unwrap();
		arena.stats.h2d_calls += 1;
		arena.stats.h2d_elems += words.len();
		arena.words.push(words.to_vec());
		arena.words.len() - 1
	}

	fn as_slice<'a>(&self, v: &'a FakeVec) -> FakeSlice<'a> {
		FakeSlice {
			id: v.id,
			_borrow: PhantomData,
		}
	}

	fn as_word_slice<'a>(&self, v: &'a usize) -> FakeWords<'a> {
		FakeWords {
			id: *v,
			_borrow: PhantomData,
		}
	}

	fn to_host<'a>(&self, v: FakeSlice<'a>) -> Lazy<'a, Vec<F>>
	where
		Self: 'a,
	{
		let arena = self.arena.clone();
		Box::new(move || {
			let mut arena = arena.lock().unwrap();
			let out = arena.fields[v.id].clone();
			arena.stats.d2h_calls += 1;
			arena.stats.d2h_elems += out.len();
			arena.pending_scalars += 1;
			out
		})
	}

	fn tensor_expand<'a>(&self, point: &'a [F]) -> Lazy<'a, FakeVec> {
		let arena = self.arena.clone();
		Box::new(move || {
			let mut arena = launch(&arena);
			push(&mut arena, eq_ind_partial_eval_scalars(point))
		})
	}

	fn inner_product<'a>(&self, a: FakeSlice<'a>, b: FakeSlice<'a>) -> Lazy<'a, F>
	where
		Self: 'a,
	{
		let arena = self.arena.clone();
		Box::new(move || {
			let mut arena = launch(&arena);
			arena.stats.d2h_calls += 1;
			arena.stats.d2h_elems += 1;
			arena.pending_scalars += 1;
			inner_product(arena.fields[a.id].iter().copied(), arena.fields[b.id].iter().copied())
		})
	}

	fn evaluate<'a>(&self, mle: FakeSlice<'a>, point: &'a [F]) -> Lazy<'a, F> {
		let arena = self.arena.clone();
		Box::new(move || {
			let mut arena = launch(&arena);
			arena.stats.d2h_calls += 1;
			arena.stats.d2h_elems += 1;
			arena.pending_scalars += 1;
			let eq = eq_ind_partial_eval_scalars(point);
			inner_product(arena.fields[mle.id].iter().copied(), eq)
		})
	}

	fn fold_words<'a>(&self, words: FakeWords<'a>, weights: &[F; 64]) -> Lazy<'a, FakeVec>
	where
		Self: 'a,
	{
		let arena = self.arena.clone();
		let weights = *weights;
		Box::new(move || {
			let mut arena = launch(&arena);
			let out = scalar::fold_words(&arena.words[words.id], &weights);
			push(&mut arena, out)
		})
	}

	fn fold_highest_var<'a, 's: 'a>(
		&self,
		cols: &'a mut [Column<FakeVec, FakeSlice<'s>>],
		r: F,
	) -> Lazy<'a, &'a mut [Column<FakeVec, FakeSlice<'s>>]>
	where
		Self: 'a,
	{
		let arena = self.arena.clone();
		Box::new(move || {
			let mut arena = launch(&arena);
			cols.iter_mut().for_each(|col| fold_col(&mut arena, col, r));
			cols
		})
	}

	fn fold_pair_map_reduce<'a, 's: 'a, K: PairKernel<F> + 'a>(
		&self,
		cols: &'a mut [Column<FakeVec, FakeSlice<'s>>],
		r: Option<F>,
		eq_point: Option<&'a [F]>,
		kernel: &'a K,
	) -> Lazy<'a, Vec<F>>
	where
		Self: 'a,
	{
		let arena = self.arena.clone();
		Box::new(move || {
			let mut arena = launch(&arena);
			assert_eq!(cols.len(), kernel.n_in());
			let log_lens: Vec<usize> = cols
				.iter()
				.map(|c| arena.fields[id(c)].len().ilog2() as usize)
				.collect();
			let log_n = scalar::round_log_len(&log_lens, r.is_some());
			if let Some(r) = r {
				cols.iter_mut().for_each(|col| fold_col(&mut arena, col, r));
			}
			// Trace the kernel once, then interpret the graph for every pair.
			let lo: Vec<Sym<F>> = (0..kernel.n_in()).map(|i| Sym::new(Node::Lo(i))).collect();
			let hi: Vec<Sym<F>> = (0..kernel.n_in()).map(|i| Sym::new(Node::Hi(i))).collect();
			let mut graph = vec![Sym::default(); kernel.n_out()];
			let mut scratch = vec![Sym::default(); kernel.scratch()];
			kernel.map(&lo, &hi, &mut scratch, &mut graph);

			let n = 1 << log_n;
			// The fake materializes eq; only SIMD avoids the full tensor.
			let eq = eq_point.map(eq_ind_partial_eval_scalars);
			let (mut lo, mut hi) = (vec![F::ZERO; kernel.n_in()], vec![F::ZERO; kernel.n_in()]);
			let mut sums = vec![F::ZERO; kernel.n_out()];
			for i in 0..n / 2 {
				for (c, col) in cols.iter().enumerate() {
					let col = &arena.fields[id(col)];
					(lo[c], hi[c]) = (col[i], col[i + n / 2]);
				}
				for (sum, g) in sums.iter_mut().zip(&graph) {
					*sum += g.eval(&lo, &hi) * eq.as_ref().map_or(F::ONE, |eq| eq[i]);
				}
			}
			arena.stats.d2h_calls += 1;
			arena.stats.d2h_elems += kernel.n_out();
			arena.pending_scalars += 1;
			sums
		})
	}

	fn join<'a, A: Send + 'a, B: Send + 'a>(
		&self,
		a: Lazy<'a, A>,
		b: Lazy<'a, B>,
	) -> Lazy<'a, (A, B)> {
		Box::new(move || (a(), b()))
	}

	fn join_all<'a, T: Send + 'a>(&self, futures: Vec<Lazy<'a, T>>) -> Lazy<'a, Vec<T>> {
		Box::new(move || futures.into_iter().map(|f| f()).collect())
	}

	fn then<'a, A: Send + 'a, B: Send + 'a>(
		&self,
		a: Lazy<'a, A>,
		f: impl FnOnce(A) -> Lazy<'a, B> + Send + 'a,
	) -> Lazy<'a, B> {
		let arena = self.arena.clone();
		Box::new(move || {
			let a = a();
			{
				let mut arena = arena.lock().unwrap();
				if arena.pending_scalars > 0 {
					arena.pending_scalars = 0;
					arena.stats.hidden_syncs += 1;
					arena.cur.hidden_syncs += 1;
				}
			}
			f(a)()
		})
	}

	fn exec<'a, T: Send + 'a>(&self, f: Lazy<'a, T>) -> T {
		self.arena.lock().unwrap().cur = ExecStats::default();
		let out = f();
		let mut arena = self.arena.lock().unwrap();
		arena.stats.execs += 1;
		arena.pending_scalars = 0;
		let cur = arena.cur;
		arena.stats.per_exec.push(cur);
		out
	}
}

enum Node<F> {
	Zero,
	Lo(usize),
	Hi(usize),
	Add(Sym<F>, Sym<F>),
	Mul(Sym<F>, Sym<F>),
	Scale(Sym<F>, F),
}

/// A traced field element: a node of an expression graph over a pair's `lo` and `hi` entries.
struct Sym<F>(Arc<Node<F>>);

impl<F> Clone for Sym<F> {
	fn clone(&self) -> Self {
		Self(self.0.clone())
	}
}

impl<F: Field> Sym<F> {
	fn new(node: Node<F>) -> Self {
		Self(Arc::new(node))
	}

	fn eval(&self, lo: &[F], hi: &[F]) -> F {
		match &*self.0 {
			Node::Zero => F::ZERO,
			Node::Lo(i) => lo[*i],
			Node::Hi(i) => hi[*i],
			Node::Add(a, b) => a.eval(lo, hi) + b.eval(lo, hi),
			Node::Mul(a, b) => a.eval(lo, hi) * b.eval(lo, hi),
			Node::Scale(a, c) => a.eval(lo, hi) * *c,
		}
	}
}

impl<F: Field> Add for Sym<F> {
	type Output = Self;
	fn add(self, rhs: Self) -> Self {
		Sym::new(Node::Add(self, rhs))
	}
}

impl<F: Field> Sub for Sym<F> {
	type Output = Self;
	// Subtraction is addition in characteristic 2.
	#[allow(clippy::suspicious_arithmetic_impl)]
	fn sub(self, rhs: Self) -> Self {
		Sym::new(Node::Add(self, rhs))
	}
}

impl<F: Field> Mul for Sym<F> {
	type Output = Self;
	fn mul(self, rhs: Self) -> Self {
		Sym::new(Node::Mul(self, rhs))
	}
}

impl<F: Field> Mul<F> for Sym<F> {
	type Output = Self;
	fn mul(self, rhs: F) -> Self {
		Sym::new(Node::Scale(self, rhs))
	}
}

impl<F: Field> Default for Sym<F> {
	fn default() -> Self {
		Sym::new(Node::Zero)
	}
}

impl<F: Field> Elem<F> for Sym<F> {}
