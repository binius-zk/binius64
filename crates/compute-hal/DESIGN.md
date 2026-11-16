# Design B: lazy future combinators

Ops return `Self::Future<'a, T>` (a description of work). `join` and `then` compose them and `exec` runs
the pipeline. Every backend uses `Lazy<'a, T> = Box<dyn FnOnce() -> T + Send + 'a>`; SIMD's `join` is
`rayon::join`. FakeDevice runs ops inside `exec` and counts them; it does not record and replay an op log,
because reduction results (`Vec<F>`) have to exist at record time for `then` closures to use them.

## Size (`wc -l`, rustfmt'd, incl. docs)

| file | LOC |
|---|---|
| `lib.rs` trait + `Composition`/`PairKernel` (lines 1-145) | ~145 |
| `sumcheck.rs` mini-prover (`prove`, `RoundKernel`, `interpolate`) | 120 |
| `reference.rs` / `simd.rs` / `fake.rs` | 105 / 177 / 417 (fake: ~150 is the traced `Sym` type) |
| `tests.rs` (ops x3 backends, 2 sumchecks x3 backends, count asserts) | 250 |

## The driver protocol authors write

```rust
for _ in 0..n_vars {
	let round = match challenges.last() {
		None => compute.pair_map_reduce(cols, &kernel),
		Some(&r) => compute.then(compute.fold_highest_var(cols, r), |cols| {
			compute.pair_map_reduce(cols, &kernel)
		}),
	};
	let (next_cols, evals) = compute.exec(round);   // the one sync point per round
	cols = next_cols;
	// interpolate R from evals + the claim, write to transcript, sample r, sum = R(r)
}
// final: exec(then(fold_highest_var(cols, last), |cols| to_host(cols)))
```

## FakeDevice counts (n = 10 variables)

| run | execs | launches | d2h calls / elems | h2d calls / elems |
|---|---|---|---|---|
| bivariate product (degree 2) | 11 | 20 | 11 / 22 | 2 / 2048 |
| mle-check (degree 3, incl. setup) | 13 | 23 | 12 / 2082 | 5 / 5120 |

Sumcheck: `n + 1` execs and `2n` launches. Per round the host receives `DEGREE` scalars in one call.
The 2082 for mle-check is 2 x 2^10 for the `c = a*b` host round-trip in setup, plus `3n + 4`.
Setup is one `exec` that `join`s `tensor_expand` with both `fold_words`.

## What laziness buys when Fiat-Shamir forces a sync per round: little

- Each round is one `exec` holding two launches (fold with the previous challenge, then evaluate).
  A device sees both before running, so it can fuse them or submit one command buffer. That is the best
  case, and it is a real but small win: it halves submissions, not syncs.
- Syncs per sumcheck are `n + 1` either way. An eager API (design A) with an explicit
  `fold` + `pair_map_reduce` and one blocking read per round has the same sync count; it only loses the
  chance to batch the two launches. The same fusion is available eagerly with a stream/queue and one
  blocking read.
- Where lazy wins more: independent work in one round (several sumchecks' kernels, `tensor_expand` next to
  `fold_words`, final eval reads). Only the setup `join` shows this in the prototype.
- Cost: the API is shaped by ownership threading and by `then` (below). Protocol code reads worse than
  eager calls.

## Pain points (what I hit)

1. **Ownership instead of `&mut` slices.** A future holding `&mut cols` for the fold cannot coexist with a
   later future that reads `cols`, and fold-then-evaluate in one pipeline needs exactly that. So
   `fold_highest_var` and `pair_map_reduce` take `Vec<FieldVec>` and return it. `FieldSliceMut` is gone
   (unused) and `FieldSlice` serves only the read-only ops. A CPU backend cannot express the borrowed form
   at all; only a handle-based device could.
2. **`then` is a host closure.** Needed to chain (`fold` result -> `pair_map_reduce` input). It only builds
   the next future from opaque handles, but on a real device that closure runs at submit time, so it must
   not read data. This is a convention, not enforced. I dropped `map` and `ready` (host closures on data,
   unused). A cleaner variant is ops that accept `Future<Vec<FieldVec>>` inputs directly.
3. **GAT lifetime early/late binding (E0195).** `inner_product<'a>(a: Self::FieldSlice<'a>, ..)` has `'a`
   only inside projections, so it is early-bound in the trait and late-bound in impls that use `&'a [F]`.
   Fix: `where Self: 'a` on those methods in the trait and every impl. No HRTB errors otherwise.
4. `evaluate` is a required op, not default `tensor_expand` + `inner_product`: the temporary tensor can't be
   borrowed by a future that outlives the building closure.
5. Futures capture only cheap clones (`Arc` arena); `exec` must be callable from `&self` so every backend
   is `Sync`. Closures must be `Send`, which forces `PairKernel: Sync`, `Composition: Sync`.

## Deviations from the spec

- Round polynomial points: `X = 1` and `d - 1` further points (`basis(1..d)`), not `X = infinity`. The leading
  coefficient is `comp(hi + lo)` only for homogeneous compositions, and `eq * (a*b - c)` is not. Interpolation
  is a local 15-line Lagrange (`interpolate`); nothing reusable in `binius-math`.
- Dropped as unused: `alloc_field`, `alloc_words`, `copy_words_to_host`, `WordSliceMut`, `FieldSliceMut`.
  `copy_field_to_host` became `to_host(Vec<FieldVec>) -> Future<Vec<Vec<F>>>` (batched, owned, lazy).
- `pair_map_reduce` and `fold_highest_var` take and return owned columns (pain point 1).
- `Compute<F: BinaryField>` has `F` as a type parameter so `SimdCompute<P>` implements `Compute<P::Scalar>`.
- SIMD calls `BitAxisFolder` (adds a `binius-prover` dependency) and re-pads to a power of two.
- Fake uploads (`copy_*_from_host`) are eager and counted; reductions' scalar results count as d2h.
- Gaps: SIMD compositions can't use `WideMul` unreduced accumulation; `RoundKernel::map` allocates `Vec`s
  per pair (untuned); the fake arena never frees buffers.
