# Design B: lazy future combinators

Ops return `Self::Future<'a, T>` (a description of work). `join` and `then` compose them and `exec` runs
the pipeline. Every backend uses `Lazy<'a, T> = Box<dyn FnOnce() -> T + Send + 'a>`; SIMD's `join` is
`rayon::join`. FakeDevice runs ops inside `exec` and counts them; it does not record and replay an op log,
because reduction results (`Vec<F>`) have to exist at record time for `then` closures to use them.

## Size (`wc -l`, rustfmt'd, incl. docs)

| file | LOC |
|---|---|
| `lib.rs` (trait, `Composition`, `PairKernel`, shared scalar kernels) | 234 (trait itself ~150) |
| `sumcheck.rs` (`prove`, `prove_mlecheck`, `RoundKernel`, `interpolate`) | 192 |
| `reference.rs` / `simd.rs` / `fake.rs` | 105 / 181 / 467 (fake: ~150 traced `Sym`, ~60 counters) |
| `tests.rs` (ops x3 backends, 3 sumchecks x3 backends, counts) | 333 |

## The drivers protocol authors write

Plain sumcheck round (the eq-as-column check is the same with 4 columns, degree 3):

```rust
let round = match challenges.last() {
	None => compute.pair_map_reduce(cols, &kernel),
	Some(&r) => compute.then(compute.fold_highest_var(cols, r), |cols| {
		compute.pair_map_reduce(cols, &kernel)
	}),
};
let (next_cols, evals) = compute.exec(round);   // the one sync point per round
```

Gruen-factored MLE-check round (`prove_mlecheck`; verified by `binius_ip::mlecheck::verify`):

```rust
for (round, &alpha) in point.iter().rev().enumerate() {
	let eq = compute.tensor_expand(&point[..n_vars - 1 - round]);
	let with_eq = |eq, cols: Vec<C::FieldVec>| {
		compute.pair_map_reduce(iter::once(eq).chain(cols).collect(), &kernel)
	};
	let pipeline = match challenges.last() {
		None => compute.then(eq, |eq| with_eq(eq, cols)),
		Some(&r) => compute.then(compute.join(eq, compute.fold_highest_var(cols, r)),
			|(eq, cols)| with_eq(eq, cols)),
	};
	let (mut all, evals) = compute.exec(pipeline);
	all.remove(0); cols = all;
	// R'(0) = (sum - alpha R'(1)) / (1 - alpha); interpolate, send, sample r, sum = R'(r)
}
```

## FakeDevice counts (n = 10 variables)

Syncs = execs (each exec is one sync at its return). Hidden syncs are `then` closures run while a
device-computed scalar is pending (`inner_product`, `evaluate`, `pair_map_reduce` sums, `to_host`).

| sumcheck | execs | launches per round exec | final exec | hidden syncs | d2h elems | h2d elems |
|---|---|---|---|---|---|---|
| bivariate product (deg 2) | 11 | 1, then 2 | 1 | 0 | 22 | 2048 |
| eq as a column (deg 3) | 13 incl. 2 setup | 1, then 2 | 1 | 0 | 2082 | 5120 |
| Gruen mle-check (deg 2 prime) | 11 | 2, then 3 | 1 | 0 | 23 | 3072 |

Round = one sync, zero hidden syncs, per-round host traffic is `DEGREE` scalars. A round exec holds fold
and evaluate (and, for Gruen, the eq `tensor_expand`) as 2-3 launches. `fake_device_counts_hidden_sync`
shows the counter firing on `then(inner_product, ..)`. Setup of the eq-as-column run is one `exec`
joining `tensor_expand` with both `fold_words`; its 2082 d2h includes the `c = a*b` host round trip.

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

## Does PairKernel survive the Gruen hot path? Yes: no signature change for the Gruen shape

The only change for Gruen was a semantic extension of `pair_map_reduce`: a column one variable shorter than
the longest is broadcast (`lo = hi = col[i]`). It carries the eq tensor, so the kernel and `RoundKernel`
are unchanged and the composition just lists eq as input 0. `Composition::DEGREE` is now "degree along a
line", so eq adds none (prime degree 2). Eq is re-expanded each round with `tensor_expand` on a prefix of
the point (geometric total, no truncate op), and the driver, not the backend, keeps it out of the fold.
Counting changes from the first prototype: `Compute` trait: 0 new ops, 1 doc/semantic change
(`pair_map_reduce` broadcast); `PairKernel`: 1 signature change, from round 2 item 3 (`SCRATCH` +
`scratch` arg); new driver code (~70 lines). Not covered: split-eq (Gruen's `eq` over the low half
only), packed WideMul accumulation, and the dedicated BitAnd evaluator's three-columns-from-two trick.

Scratch: `reduce_pairs` allocates `lo`, `hi`, `scratch`, `out`, `acc` once per call; the SIMD backend calls it
once per 256-pair chunk under rayon, the reference once per round. `RoundKernel::map` is allocation-free.
No interpolation helper exists in `binius-math` or `binius-ip` (only domain `lagrange_evals`), so
`interpolate` stays.

## Pain points (what I hit)

1. **Ownership instead of `&mut` slices.** A future holding `&mut cols` for the fold cannot coexist with a
   later future that reads `cols`, and fold-then-evaluate in one pipeline needs exactly that. So
   `fold_highest_var` and `pair_map_reduce` take `Vec<FieldVec>` and return it. `FieldSliceMut` is gone
   (unused) and `FieldSlice` serves only the read-only ops. A CPU backend cannot express the borrowed form
   at all; only a handle-based device could.
2. **`then` is a host closure.** Needed to chain (`fold` result -> `pair_map_reduce` input). Over handles
   it is free on a device; over a device-computed scalar it is a hidden sync. That distinction is a
   convention the types do not enforce; `FakeDevice` only counts it (conservatively: any pending host-typed
   result counts). `map` is gone; `ready` was never needed.
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
- Gaps: SIMD compositions can't use `WideMul` unreduced accumulation; the fake arena never frees buffers.
- The fake runs ops at `exec`, so its "op log" is counters only, not a replayable command buffer.
