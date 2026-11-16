# Design B: lazy future combinators

Ops return `Self::Future<'a, T>` (a description of work). `join`, `join_all` and `then` compose them and
`exec` runs the pipeline. Every backend uses `Lazy<'a, T> = Box<dyn FnOnce() -> T + Send + 'a>`; SIMD's
`join`/`join_all` use rayon. FakeDevice runs ops inside `exec` and counts them; its "op log" is counters
only, not a replayable command buffer (a `then` closure needs the real result to continue).

Columns are owned by the host and borrowed by futures: read-only ops take `FieldSlice<'a>`, and column
changing ops take `&'a mut [FieldVec]`. No owned-column threading. The sumcheck round is ONE op,
`fold_pair_map_reduce(cols, Option<r>, Option<eq_point>, kernel)`, mirroring the real prover's `map_reduce_with_fold`.

## Size (`wc -l`, rustfmt'd, incl. docs)

| file | LOC |
|---|---|
| `lib.rs` (trait, `Elem`, `Composition`, `PairKernel` + tuple/`Vec` impls, scalar helpers) | 317 |
| `sumcheck.rs` (`prove`, `prove_mlecheck`, `RoundKernel`, `interpolate`) | 214 |
| `reference.rs` / `simd.rs` / `fake.rs` | 127 / 269 / 409 |
| `tests.rs` (12 tests) | 477 |
| `benches/mlecheck.rs` | 114 |

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
fn fold_pair_map_reduce<'a, K: PairKernel<F> + 'a>(&self, cols: &'a mut [FieldVec], r: Option<F>,
	eq_point: Option<&'a [F]>, kernel: &'a K) -> Future<'a, Vec<F>>;
```

`out[j] = sum_i eq(eq_point, i) * kernel(lo_i, hi_i)[j]` over the halved hypercube (bit `k` of `i` pairs with
`eq_point[k]`, like `eq_ind_partial_eval`); no eq when `None`. All columns must have equal `log_len`.

- **Eq behind the backend.** Reference/Fake materialize the tensor. SIMD copies `SharedMleCheckProver::execute`:
  eq over the low `chunk_vars` (12) coordinates only, a rayon bisection tree over chunks, and a level-aware
  combine `lo + z (hi - lo)` with `z = eq_point[level]`. The driver passes `point[..n - 1 - round]` and no eq
  column. The broadcast-column semantic and the per-round `tensor_expand` are gone (the op stays for setup).
- **Wide accumulation** (SIMD, eq path): per chunk `P::wide_mul(kernel_out, eq)` summed in `<P as WideMul>::Output`,
  one `P::reduce` per chunk. `PairKernel` stays in plain `T`. The unweighted path sums plain `P`s (no eq multiply to
  widen, so nothing to gain).
- **Point at infinity.** `RoundKernel` outputs `R(1), R(inf), R(b_1), ..., R(b_{d-2})` (the order of
  `round_evals`): `R(1) = comp(hi)`, `R(inf) = eval_infinity(lo + hi)` (char 2: `hi - lo = hi + lo`). The host
  subtracts `R(inf) X^d` and Lagrange-interpolates the rest, so any degree works. `RoundEvals` is
  public but handles `D <= 2` only, so the local `interpolate` stays. `DEGREE >= 2` is asserted.
- **Multiple kernels, one pass.** `PairKernel` is implemented for `(A, B)` and `Vec<K>`: outputs concatenated,
  scratch split. Every kernel sees ALL columns and picks its own by index (same `n_in`, asserted). Static, no dyn.
  The port runs one `RoundKernel` plus N multilinear-eval kernels as `(RoundKernel<..>, Vec<EvalKernel>)`.
- `PairKernel`'s consts became methods, because `Vec<K>` has a runtime count.

## The drivers protocol authors write

Plain sumcheck round: one op, one sync, nothing to thread.

```rust
let round = compute.fold_pair_map_reduce(&mut cols, challenges.last().copied(), &kernel);
let evals = compute.exec(round);
```

Gruen-factored MLE-check round (`prove_mlecheck`, verified by `binius_ip::mlecheck::verify`): the same single
op with `eq_point`, so the round is one launch and one sync, like the plain one.

```rust
let eq_point = &point[..n_vars - 1 - round];
let evals = compute.exec(compute.fold_pair_map_reduce(&mut cols, last_r, Some(eq_point), &kernel));
// R'(0) = (sum - alpha R'(1)) / (1 - alpha); round_coeffs, send, sample r, sum = R'(r)
```

Final exec of both: `then(fold_highest_var(cols, r), |cols| join_all(cols.map(to_host)))`.

## FakeDevice counts (n = 10)

A sync is an `exec` return; a hidden sync is a `then` closure run while a device-computed scalar is
pending (`inner_product`, `evaluate`, reduction sums, `to_host`). Hidden syncs were 0 in every exec.

| sumcheck | execs | launches per round exec | final exec | hidden | d2h elems | h2d elems |
|---|---|---|---|---|---|---|
| bivariate product (deg 2) | 11 | 1 (every round) | 1 | 0 | 22 | 2048 |
| eq as a column (deg 3) | 13 (2 setup) | 1 | 1 | 0 | 2082 | 3072 |
| Gruen mle-check (deg 2) | 11 | 1 (every round) | 1 | 0 | 23 | 3072 |

Per round the host receives `DEGREE` scalars. Gruen's second launch (the eq `tensor_expand`) is gone. The 2082 of
the eq-column run is the `c = a*b` host round trip in setup. `fake_device_counts_hidden_sync` shows the
counter firing on `then(inner_product, ..)`.

## What laziness buys when Fiat-Shamir forces a sync per round: little

- Syncs per sumcheck are `n + 1` regardless. An eager API with one blocking read per round has the same.
- Fusion is NOT something `then` provides: a `then` closure is opaque until its input exists, so a backend
  can enqueue back-to-back but not fuse across it. Fusion needs an explicit op, which is why
  `fold_pair_map_reduce` exists. With it, a round is one launch.
- What futures still buy: independent work in one pipeline (`join`/`join_all`: setup, final reads).
- `then` stays for routing handles (`tensor_expand` -> the eq slot, `fold` -> reads). Over handles it is
  free on a device; over a device scalar it is a hidden sync (counted).

## Benchmark: `cargo bench -p binius-compute-hal --bench mlecheck`

`a * b - c` Gruen MLE-check, `OptimalPackedB128`, bench profile (thin LTO), no extra RUSTFLAGS (as the existing
benches). `ip_prover` is `SharedMleCheckProver` + `QuadraticMleEvaluator` (BufferPool); `hal_simd` is
`prove_mlecheck` on `SimdCompute`. Median of criterion's estimate, same transcript type.

| n_vars | ip_prover | hal_simd | ratio |
|---|---|---|---|
| 16 | 10.4 ms | 15.8 ms | 1.52x |
| 20 | 63.0 ms | 79.8 ms | 1.27x |
| 22 | 222.8 ms | 242.5 ms | 1.09x |

Not profiled (within 1.3x at 20/22). Likely causes of the gap, by size: HAL folds in a separate pass
(`fold_highest_var_inplace`, a second read/write of every column per round) where the real prover fuses the
fold into the map read (`map_reduce_with_fold`); at n = 16 the fixed per-round cost (kernel buffers per chunk,
`Vec` outputs, `then`/`exec` boxes, `eq_ind_partial_eval`) dominates; and the pair gather is per element
(`get(c, i)` over pre-bound slices, vs the real evaluator's fixed-size stack arrays).

## Pain points

1. **GAT early/late lifetime binding (E0195).** `Self::FieldSlice<'a>` params make `'a` early-bound in the
   trait and late-bound in impls using `&'a [F]`. Fix: `where Self: 'a` on `inner_product`, `fold_words`,
   `to_host`, `fold_highest_var`, `fold_pair_map_reduce` in the trait and every impl.
2. **`evaluate` could not become a default method.** `then(tensor_expand, |eq| inner_product(.., eq))`
   needs a slice of `eq`, which the closure owns, so the future cannot outlive it. It stays a required op.
3. **Futures borrowing `&mut` columns work** when returned through the future
   (`fold_highest_var` returns `&'a mut [FieldVec]` so a `then` can read them). No HRTB errors.
4. `then` is a host closure; "does not read data" is a convention (only counted in the fake).
5. Closures must be `Send`, forcing `PairKernel: Sync`, `Composition: Sync`, backends `Sync`.

## Deviations from the spec

- `interpolate` is local: `RoundEvals` covers only `D <= 2`.
- Dropped as unused: `alloc_*`, `copy_words_to_host`, `WordSliceMut`, `FieldSliceMut`, `ready`, `map`.
  `pair_map_reduce` is now `fold_pair_map_reduce` with `r = None`.
- `Compute<F: BinaryField>` has `F` as a type parameter so `SimdCompute<P>` implements `Compute<P::Scalar>`.
- SIMD calls `BitAxisFolder` (adds a `binius-prover` dependency) and re-pads to a power of two.

## Known gaps before migration

- Fold is a separate pass from the map (not fused like `map_reduce_with_fold`); per-pair gather, no chunk kernels.
- Multi-kernel is static (`(A, B)`, `Vec<K>`); `SharedSumcheckProver`'s heterogeneous `Vec<Box<dyn ..>>` needs nesting.
- Eq is rebuilt per round (chunk part only on SIMD); no `EqTracker` truncation.
- `InfCube` folds and the {0, infinity} sumcheck are not covered.
- `SimdCompute<P, A: Allocator>` should use `BufferPool`; buffers are plain `Vec`s now.
- Traced constants are baked into the graph: per-proof batching coefficients would recompile CUDA kernels;
  lift them into a uniform table.
- Bespoke ops for shift, univariate skip, intmul switchover and sparse-dense are not in the op list.
- Merkle/FRI digest memory and indexed readback are not modelled.
- The small-n tail runs on the host inside the backend (SIMD falls back to scalars below one packed word).
