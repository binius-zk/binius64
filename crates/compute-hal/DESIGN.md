# Design B: lazy future combinators

Ops return `Self::Future<'a, T>` (a description of work). `join`, `join_all` and `then` compose them and
`exec` runs the pipeline. Every backend uses `Lazy<'a, T> = Box<dyn FnOnce() -> T + Send + 'a>`; SIMD's
`join`/`join_all` use rayon. FakeDevice runs ops inside `exec` and counts them; its "op log" is counters
only, not a replayable command buffer (a `then` closure needs the real result to continue).

Columns are owned by the host and borrowed by futures: read-only ops take `FieldSlice<'a>`, and column
changing ops take `&'a mut [FieldVec]`. No owned-column threading. The sumcheck round is ONE op,
`fold_pair_map_reduce(cols, Option<r>, kernel)`, mirroring the real prover's `map_reduce_with_fold`.

## Size (`wc -l`, rustfmt'd, incl. docs)

| file | LOC |
|---|---|
| `lib.rs` (trait ~170, `Composition`, `PairKernel`, shared scalar kernels) | 268 |
| `sumcheck.rs` (`prove`, `prove_mlecheck`, `RoundKernel`, `interpolate`) | 202 |
| `reference.rs` / `simd.rs` / `fake.rs` | 126 / 202 / 486 (fake: ~150 traced `Sym`, ~70 counters) |
| `tests.rs` (12 tests: ops, boundaries, panics, degrees, 3 sumchecks x 3 backends, counts) | 431 |

## The drivers protocol authors write

Plain sumcheck round: one op, one sync, nothing to thread.

```rust
let round = compute.fold_pair_map_reduce(&mut cols, challenges.last().copied(), &kernel);
let evals = compute.exec(round);
```

Gruen-factored MLE-check round (`prove_mlecheck`, verified by `binius_ip::mlecheck::verify`). Slot 0 of
`all` is the eq tensor; it is broadcast (one variable shorter) and never folded by the op:

```rust
let evals = match challenges.last().copied() {
	None => compute.exec(compute.fold_pair_map_reduce(&mut all, None, &kernel)),
	Some(r) => {
		let next_eq = compute.tensor_expand(&point[..n_vars - 1 - round]);
		let (all, kernel) = (&mut all, &kernel);
		compute.exec(compute.then(next_eq, move |eq| {
			all[0] = eq;                      // routes a handle, reads no data
			compute.fold_pair_map_reduce(all, Some(r), kernel)
		}))
	}
};
// R'(0) = (sum - alpha R'(1)) / (1 - alpha); interpolate, send, sample r, sum = R'(r)
```

Final exec of both: `then(fold_highest_var(cols, r), |cols| join_all(cols.map(to_host)))`.

## FakeDevice counts (n = 10)

A sync is an `exec` return; a hidden sync is a `then` closure run while a device-computed scalar is
pending (`inner_product`, `evaluate`, reduction sums, `to_host`). Hidden syncs were 0 in every exec.

| sumcheck | execs | launches per round exec | final exec | hidden | d2h elems | h2d elems |
|---|---|---|---|---|---|---|
| bivariate product (deg 2) | 11 | 1 (every round) | 1 | 0 | 22 | 2048 |
| eq as a column (deg 3) | 13 (2 setup) | 1 | 1 | 0 | 2082 | 3072 |
| Gruen mle-check (deg 2) | 12 (1 setup) | 1, then 2 | 1 | 0 | 23 | 3072 |

Per round the host receives `DEGREE` scalars. Gruen's second launch is the eq `tensor_expand`. The 2082 of
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

## Does PairKernel survive the Gruen hot path? Yes

One semantic extension of `fold_pair_map_reduce`: a column one variable shorter than the rest is broadcast
(`lo = hi = col[i]`) and never folded. The kernel and `RoundKernel` did not change for it; eq is input 0 and
`Composition::DEGREE` is the degree along a line, so eq adds none. Eq is re-expanded per round from a prefix
of the point (geometric total). Misuse panics: after folding, every column must be `n` or `n - 1`.

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

- Round polynomial points are `X = 1` and `basis(1..d)`, not `X = infinity`: the leading coefficient is
  `comp(hi + lo)` only for homogeneous compositions. `interpolate` is local (no helper in math/ip).
- Dropped as unused: `alloc_*`, `copy_words_to_host`, `WordSliceMut`, `FieldSliceMut`, `ready`, `map`.
  `pair_map_reduce` is now `fold_pair_map_reduce` with `r = None`.
- `Compute<F: BinaryField>` has `F` as a type parameter so `SimdCompute<P>` implements `Compute<P::Scalar>`.
- SIMD calls `BitAxisFolder` (adds a `binius-prover` dependency) and re-pads to a power of two.

## Known gaps before migration

- WideMul unreduced accumulation needs a `PairKernel` signature change (accumulator type per output).
- Multi-evaluator single pass: `PairKernel::map<T>` is not object-safe, but `SharedSumcheckProver` runs
  `Vec<Box<dyn ..>>`.
- Per-pair gather vs chunk-level kernels: the pair loop is unbenchmarked.
- Split-eq / `EqTracker` truncation: eq is re-expanded each round here.
- `InfCube` folds and the {0, infinity} sumcheck are not covered.
- `SimdCompute<P, A: Allocator>` should use `BufferPool`; buffers are plain `Vec`s now.
- Traced constants are baked into the graph: per-proof batching coefficients would recompile CUDA kernels;
  lift them into a uniform table.
- The `FieldOps` bound on `T` is too wide (`square_transpose` is `unimplemented!` on the traced type);
  use a minimal element trait.
- Bespoke ops for shift, univariate skip, intmul switchover and sparse-dense are not in the op list.
- Merkle/FRI digest memory and indexed readback are not modelled.
- The small-n tail runs on the host inside the backend (SIMD falls back to scalars below one packed word).
