// Copyright 2026 The Binius Developers

//! A simulated device: buffers live in a backend-owned arena and host code sees them only through
//! explicit downloads. Kernels are traced into expression graphs and interpreted per element.

// Every op holds the arena lock for its whole body by design.
#![allow(clippy::significant_drop_tightening)]

use std::{
	iter::{Product, Sum},
	marker::PhantomData,
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
	sync::{Arc, Mutex},
};

use binius_core::word::Word;
use binius_field::{
	BinaryField, Field, FieldOps,
	arithmetic_traits::{InvertOrZero, Square},
};
use binius_math::{inner_product::inner_product, multilinear::eq::eq_ind_partial_eval_scalars};

use crate::{Compute, Lazy, PairKernel, scalar};

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

	fn to_host<'a>(&self, cols: Vec<FakeVec>) -> Lazy<'a, Vec<Vec<F>>> {
		let arena = self.arena.clone();
		Box::new(move || {
			let mut arena = arena.lock().unwrap();
			let out: Vec<Vec<F>> = cols.iter().map(|c| arena.fields[c.id].clone()).collect();
			arena.stats.d2h_calls += 1;
			arena.stats.d2h_elems += out.iter().map(Vec::len).sum::<usize>();
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

	fn fold_highest_var<'a>(&self, cols: Vec<FakeVec>, r: F) -> Lazy<'a, Vec<FakeVec>> {
		let arena = self.arena.clone();
		Box::new(move || {
			let mut arena = launch(&arena);
			cols.into_iter()
				.map(|col| {
					scalar::fold_highest_var(&mut arena.fields[col.id], r);
					FakeVec {
						id: col.id,
						log_len: col.log_len - 1,
					}
				})
				.collect()
		})
	}

	fn pair_map_reduce<'a, K: PairKernel<F> + 'a>(
		&self,
		cols: Vec<FakeVec>,
		kernel: &'a K,
	) -> Lazy<'a, (Vec<FakeVec>, Vec<F>)> {
		let arena = self.arena.clone();
		Box::new(move || {
			let mut arena = launch(&arena);
			// Trace the kernel once, then interpret the graph for every pair.
			let lo: Vec<Sym<F>> = (0..K::N_IN).map(|i| Sym::new(Node::Lo(i))).collect();
			let hi: Vec<Sym<F>> = (0..K::N_IN).map(|i| Sym::new(Node::Hi(i))).collect();
			let mut graph = vec![Sym::zero(); K::N_OUT];
			let mut scratch = vec![Sym::zero(); K::SCRATCH];
			kernel.map(&lo, &hi, &mut scratch, &mut graph);

			let n = cols.iter().map(|c| arena.fields[c.id].len()).max().unwrap();
			let (mut lo, mut hi) = (vec![F::ZERO; K::N_IN], vec![F::ZERO; K::N_IN]);
			let mut sums = vec![F::ZERO; K::N_OUT];
			for i in 0..n / 2 {
				for (c, col) in cols.iter().enumerate() {
					let col = &arena.fields[col.id];
					(lo[c], hi[c]) = if col.len() == n {
						(col[i], col[i + n / 2])
					} else {
						(col[i], col[i])
					};
				}
				for (sum, g) in sums.iter_mut().zip(&graph) {
					*sum += g.eval(&lo, &hi);
				}
			}
			arena.stats.d2h_calls += 1;
			arena.stats.d2h_elems += K::N_OUT;
			arena.pending_scalars += 1;
			(cols, sums)
		})
	}

	fn join<'a, A: Send + 'a, B: Send + 'a>(
		&self,
		a: Lazy<'a, A>,
		b: Lazy<'a, B>,
	) -> Lazy<'a, (A, B)> {
		Box::new(move || (a(), b()))
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
	One,
	Lo(usize),
	Hi(usize),
	Add(Sym<F>, Sym<F>),
	Mul(Sym<F>, Sym<F>),
	Scale(Sym<F>, F),
	Square(Sym<F>),
	Inv(Sym<F>),
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
			Node::One => F::ONE,
			Node::Lo(i) => lo[*i],
			Node::Hi(i) => hi[*i],
			Node::Add(a, b) => a.eval(lo, hi) + b.eval(lo, hi),
			Node::Mul(a, b) => a.eval(lo, hi) * b.eval(lo, hi),
			Node::Scale(a, c) => a.eval(lo, hi) * *c,
			Node::Square(a) => Square::square(a.eval(lo, hi)),
			Node::Inv(a) => a.eval(lo, hi).invert_or_zero(),
		}
	}
}

macro_rules! sym_binop {
	($trait:ident, $method:ident, $assign_trait:ident, $assign:ident, $node:ident) => {
		impl<F: Field> $trait for Sym<F> {
			type Output = Self;
			fn $method(self, rhs: Self) -> Self {
				Sym::new(Node::$node(self, rhs))
			}
		}
		impl<F: Field> $trait<&Sym<F>> for Sym<F> {
			type Output = Self;
			fn $method(self, rhs: &Self) -> Self {
				Sym::new(Node::$node(self, rhs.clone()))
			}
		}
		impl<F: Field> $assign_trait for Sym<F> {
			fn $assign(&mut self, rhs: Self) {
				*self = Sym::new(Node::$node(self.clone(), rhs));
			}
		}
		impl<F: Field> $assign_trait<&Sym<F>> for Sym<F> {
			fn $assign(&mut self, rhs: &Self) {
				*self = Sym::new(Node::$node(self.clone(), rhs.clone()));
			}
		}
	};
}

sym_binop!(Add, add, AddAssign, add_assign, Add);
// Subtraction is addition in characteristic 2.
#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
const _: () = {
	sym_binop!(Sub, sub, SubAssign, sub_assign, Add);
};
sym_binop!(Mul, mul, MulAssign, mul_assign, Mul);

impl<F: Field> Mul<F> for Sym<F> {
	type Output = Self;
	fn mul(self, rhs: F) -> Self {
		Sym::new(Node::Scale(self, rhs))
	}
}

impl<F: Field> Neg for Sym<F> {
	type Output = Self;
	fn neg(self) -> Self {
		self
	}
}

impl<F: Field> Sum for Sym<F> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::zero(), |a, b| a + b)
	}
}

impl<'a, F: Field> Sum<&'a Sym<F>> for Sym<F> {
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::zero(), |a, b| a + b)
	}
}

impl<F: Field> Product for Sym<F> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::one(), |a, b| a * b)
	}
}

impl<'a, F: Field> Product<&'a Sym<F>> for Sym<F> {
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::one(), |a, b| a * b)
	}
}

impl<F: Field> Square for Sym<F> {
	fn square(self) -> Self {
		Sym::new(Node::Square(self))
	}
}

impl<F: Field> InvertOrZero for Sym<F> {
	fn invert_or_zero(self) -> Self {
		Sym::new(Node::Inv(self))
	}
}

impl<F: Field> FieldOps for Sym<F> {
	type Scalar = F;

	fn zero() -> Self {
		Sym::new(Node::Zero)
	}

	fn one() -> Self {
		Sym::new(Node::One)
	}

	fn square_transpose<FSub: Field>(_elems: &mut [Self])
	where
		F: binius_field::ExtensionField<FSub>,
	{
		unimplemented!("a traced element has no subfield structure to transpose")
	}
}
