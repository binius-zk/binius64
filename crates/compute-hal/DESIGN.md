# Design B: lazy future combinators

Ops return `Self::Future<'a, T>` (a description of work). `join`, `join_all` and `then` compose them and
`exec` runs the pipeline. Every backend uses `Lazy<'a, T> = Box<dyn FnOnce() -> T + Send + 'a>`; SIMD's
`join`/`join_all` use rayon. FakeDevice runs ops inside `exec` and counts them; its "op log" is counters
only, not a replayable command buffer (a `then` closure needs the real result to continue).

Columns are owned by the host and borrowed by futures: read-only ops take `FieldSlice<'a>`, and column
changing ops take `&'a mut [Column<..>]` (owned or lent). No owned-column threading. The sumcheck round is ONE op,
`fold_pair_map_reduce(cols, Option<r>, Option<eq_point>, kernel)`, mirroring the real prover's `map_reduce_with_fold`.

## Size (`wc -l`, incl. docs)

| file | LOC |
|---|---|
| `lib.rs` (trait, `Column`, `Elem`, `Composition`, `PairKernel` + tuple/`Vec` impls, scalar helpers) | 343 |
| `mlecheck.rs` (`ComputeMleCheckProver`, `Pick`, `quadratic`, `multilinear_eval`) | 194 |
| `sumcheck.rs` (`prove`, `RoundKernel`, `interpolate`) | 164 |
| `reference.rs` / `simd.rs` / `fake.rs` | 148 / 283 / 432 |
| `tests.rs` (12 tests) / `tests/mlecheck.rs` (7 tests) | 495 / 279 |
| `benches/mlecheck.rs` | 127 |

## API (phase 2)

```rust
pub trait Elem<F>: Clone + Default + Send + Sync + Add<Output = Self> + Sub<Output = Self>
	+ Mul<Output = Self> + Mul<F, Output = Self> {}
impl<P: PackedField> Elem<P::Scalar> for P {}   // covers Field (width-1 packed); Sym<F> impls it too

pub trait Composition<F>: Sync {
	const N_IN: usize; const DEGREE: usize;
	fn eval<T: Elem<F>>(&self, x: &[T]) -> T;
	fn eval_infinity<T: Elem<F>>(&self, x: &[T]) -> T;  // degree-DEGREE homogeneous part
}
pub trait PairKernel<F>: Sync {
	fn n_in(&self) -> usize; fn n_out(&self) -> usize; fn scratch(&self) -> usize;
	fn map<T: Elem<F>>(&self, lo: &[T], hi: &[T], scratch: &mut [T], out: &mut [T]);
}
pub enum Column<V, S> { Owned(V), Borrowed(S) }   // Borrowed: read in place, folded into a fresh Owned
fn fold_pair_map_reduce<'a, 's: 'a, K: PairKernel<F> + 'a>(&self, cols: &'a mut [Column<FieldVec, FieldSlice<'s>>],
	r: Option<F>, eq_point: Option<&'a [F]>, kernel: &'a K) -> Future<'a, Vec<F>>;
fn fold_highest_var<'a, 's: 'a>(&self, cols: &'a mut [Column<..>], r: F) -> Future<'a, &'a mut [Column<..>]>;
```

`out[j] = sum_i eq(eq_point, i) * kernel(lo_i, hi_i)[j]` over the halved hypercube (bit `k` of `i` pairs with
`eq_point[k]`, like `eq_ind_partial_eval`); no eq when `None`. All columns must have equal `log_len`.

- **Eq behind the backend.** Reference/Fake materialize the tensor. SIMD copies `SharedMleCheckProver::execute`:
  eq over the low `chunk_vars` (12) coordinates only, a rayon bisection tree over chunks, and a level-aware
  combine `lo + z (hi - lo)` with `z = eq_point[level]`. The driver passes `point[..n - 1 - round]`.
- **Wide accumulation** (SIMD, eq path): per chunk `P::wide_mul(kernel_out, eq)` summed in `<P as WideMul>::Output`,
  one `P::reduce` per chunk. `PairKernel` stays in plain `T`; the unweighted path sums plain `P`s.
- **Point at infinity.** `RoundKernel` outputs `R(1), R(inf), R(b_1), ..., R(b_{d-2})`: `R(1) = comp(hi)`,
  `R(inf) = eval_infinity(lo + hi)`. The host subtracts `R(inf) X^d` and Lagrange-interpolates the rest, so any
  degree works (`RoundEvals` handles `D <= 2` only, so the local `interpolate` stays). `DEGREE >= 2` is asserted.
- **Multiple kernels, one pass.** `PairKernel` is implemented for `(A, B)` and `Vec<K>`: outputs concatenated. Every kernel sees ALL columns and picks its own by index (`Pick`). Static, no dyn. An empty `Vec`
  constrains nothing (`n_in` 0), so `(quad, vec![])` is a valid kernel; the per-pair `n_in` assert was dropped.

## The drivers protocol authors write

Plain sumcheck round: one op, one sync, nothing to thread.

```rust
let round = compute.fold_pair_map_reduce(&mut cols, challenges.last().copied(), None, &kernel);
let evals = compute.exec(round);
```

Gruen-factored MLE-check round (inside `ComputeMleCheckProver::execute`, verified by `binius_ip::mlecheck::verify`):
the same single op with `eq_point`, so the round is one launch and one sync, like the plain one.

```rust
let eq_point = &point[..n_vars - 1 - round];
let evals = compute.exec(compute.fold_pair_map_reduce(&mut cols, last_r, Some(eq_point), &kernel));
// R'(0) = (claim - alpha R'(1)) / (1 - alpha); interpolate per claim; the driver sends and samples r
```

## The port: `mlecheck::ComputeMleCheckProver`

Implements `binius_ip_prover::sumcheck::common::MleCheckProver<F>` exactly, so `prove_single_mlecheck`,
`batch_prove_mle`, `MleToSumCheckDecorator`, `PaddedSumcheckDecorator`, `Either` and
`batch_prove_and_write_evals` take it unchanged (no bound rejected it). `SharedMleCheckProver`'s pieces map as:

| real | here |
|---|---|
| `MleStore` columns | `Vec<Column<FieldVec, FieldSlice<'a>>>`, owned or lent |
| `EvaluatorGroup` + evaluators | one `PairKernel` (tuple/`Vec` composition) + `claims: Vec<(F, degree)>` |
| `QuadraticMleEvaluator` | `quadratic(&comp, n_in, cols)` = `Pick<RoundKernel>`, `C::DEGREE` slots |
| `MultilinearEvalEvaluator` | `multilinear_eval(n_in, col)` = `Pick<HiKernel>`, 1 slot |
| `RoundState`, `buffered_challenge` | the same `RoundState`; `challenge: Option<F>` applied by the next fused op |
| `interpolate_eq` | local: degree 1 is `[y0, y1 - y0]`; degree >= 2 reuses `round_coeffs` (same as `RoundEvals` at 2) |

`execute` = one `exec`; `fold` = bookkeeping only; `finish` = one `exec` (fold + read of the one-element columns).

### Borrowed columns: the API change

`Column<V, S>` is passed to `fold_pair_map_reduce` and `fold_highest_var`. A `Borrowed` column is read in place by
the kernel; its first fold writes a fresh half-size `Owned` buffer into the same slot (the real `Column::Borrowed`).
Round 1 has no fold, so a lent column is never copied before round 2. The slice lifetime `'s: 'a` is separate from
the op's borrow of the column slice: a GAT slice behind `&'a mut` is invariant, so `'s = 'a` would freeze the
columns for the whole proof.

Rejected: a read-only first-round op (`pair_map_reduce(&[FieldSlice])`) plus an out-of-place `fold_into` op. The
round-2 op needs the fresh buffers as `&mut` input to the same pass; chaining via `then` makes the closure own the
buffers while the next future borrows them (no lifetime works), and the other way costs a second `exec`.
`prove` still takes `Vec<FieldVec>` and wraps it.

## Transcript equality vs the real prover (`tests/mlecheck.rs`, all three backends)

Same inputs, `HasherChallenger<Sha256>`; asserts equal `finalize()` bytes, final evals and challenges.
- (a) single quadratic `a*b - c`, owned columns, n = 2, 8, 10 (below and above the SIMD width): equal.
- (b) quadratic + 3 `MultilinearEval` claims, last column lent (n = 1, 3, 8, 10, and n = 10 owned): equal;
  the lent column is unchanged afterwards.
- (c) `Either<quadratic, evals>` through `MleToSumCheckDecorator` + `PaddedSumcheckDecorator` (2 extra vars) +
  `batch_prove_and_write_evals`: equal for real/real, hal/hal, hal/real and real/hal.
- `prove_single_mlecheck` accepts one claim only; multi-claim (b) runs through `batch_prove_mle`.

## FakeDevice counts (n = 10)

A sync is an `exec` return; a hidden sync is a `then` closure run while a device-computed scalar is
pending (`inner_product`, `evaluate`, reduction sums, `to_host`). Hidden syncs were 0 in every exec.

| sumcheck | execs | launches per round exec | final exec | hidden | d2h elems | h2d elems |
|---|---|---|---|---|---|---|
| bivariate product (deg 2) | 11 | 1 (every round) | 1 | 0 | 22 | 2048 |
| eq as a column (deg 3) | 13 (2 setup) | 1 | 1 | 0 | 2082 | 3072 |
| Gruen mle-check (deg 2) | 11 | 1 (every round) | 1 | 0 | 23 | 3072 |
| port: quadratic + 3 evals, one lent column | 11 | 1 (every round, incl. the borrowed fold) | 1 | 0 | 53 | 3072 |

Per round the host receives `DEGREE` scalars (`n_out` for the port); the eq-column run's 2082 is the setup
round trip of `c = a*b`. `fake_device_counts_hidden_sync` shows the hidden-sync counter firing.

## What laziness buys when Fiat-Shamir forces a sync per round: little

Syncs are `n + 1` regardless. `then` cannot fuse (its closure is opaque until its input exists), so fusion needs an
explicit op, `fold_pair_map_reduce`. Futures still pay for `join`/`join_all` (setup, final reads), and `then` routes
handles for free on a device (over a device scalar it is a hidden sync, counted).

## Benchmark: `cargo bench -p binius-compute-hal --bench mlecheck`

`a * b - c` Gruen MLE-check, `OptimalPackedB128`, bench profile (thin LTO), no extra RUSTFLAGS (as the existing
benches). `ip_prover` is `SharedMleCheckProver` + `QuadraticMleEvaluator` (BufferPool); `hal_simd` is
`ComputeMleCheckProver` (`quadratic` kernel) on `SimdCompute`, both through `prove_single_mlecheck`.
Median of criterion's estimate, same transcript type.

| n_vars | ip_prover | hal_simd | ratio |
|---|---|---|---|
| 16 | 10.5 ms | 11.1 ms | 1.06x |
| 20 | 63.8 ms | 66.5 ms | 1.04x |
| 22 | 219.5 ms | 226.7 ms | 1.03x |

Before fusing the fold into the map: 1.54x / 1.28x / 1.12x (16.4 / 80.8 / 246.7 ms).

The port costs about nothing over the old driver (15.8 / 79.8 / 242.5 ms before): `Pick` copies the selected pair
into scratch per element, below noise.

The fold is fused into the map's read as in `MleStore::map_reduce_with_fold` (`fold_pair_map_reduce_fused`): each
column is a deferred `[lo, hi]` pair of segments split alongside the chunk tree, folded in place (owned, then truncated)
or into a fresh buffer (borrowed, which becomes owned). Chunking matches the real prover (`MAX_CHUNK_VARS = 12`,
`chunk_vars = (n - 1).min(12.max(LOG_WIDTH))`, rayon bisection, one accumulator Vec per chunk). It fell back to
fold-then-map below `LOG_WIDTH`. That one change bought the whole gain. The residual 3-4% is likely the per-element
`Pick` copy and kernel gather, not profiled.

## Pain points

1. **GAT early/late lifetime binding (E0195):** `where Self: 'a` on every op taking a `FieldSlice<'a>`.
2. **`evaluate` stays a required op:** the `then` closure owns the eq tensor, so no slice of it outlives it.
3. `then` is a host closure ("does not read data" is a convention); closures must be `Send`, so kernels are `Sync`.

## Deviations from the spec

- `prove_mlecheck` is gone, superseded by the port.
- Dropped as unused: `alloc_*`, `copy_words_to_host`, `WordSliceMut`, `FieldSliceMut`, `ready`, `map`; `pair_map_reduce`
  is `fold_pair_map_reduce` with `r = None`.
- `Compute<F>` has `F` as a type parameter; SIMD calls `BitAxisFolder` (`binius-prover` dependency).
- `binius-ip-prover` is a regular dependency now (the port implements its trait).

## Known gaps before migration

- `Column::SplitHalf` is not ported: a third `Column` variant and a half-view in every backend, over 30 lines.
- `into_shared_sumcheck`, `add_evaluator`, `store_mut` (growing the store after construction) are not ported.
- Per-pair gather in `Pick`; eq chunk rebuilt per round. The fused fold needs `eq_point`; without it fold and map are two passes.
- Multi-kernel is static; `SharedSumcheckProver`'s heterogeneous `Vec<Box<dyn ..>>` needs nesting. `InfCube` is not covered.
- `SimdCompute<P, A: Allocator>` should use `BufferPool`; buffers are plain `Vec`s now.
- Traced constants are baked into the graph (per-proof batching coefficients would recompile CUDA kernels).
- Bespoke ops (shift, univariate skip, intmul switchover, sparse-dense) and Merkle/FRI memory are not modelled.
