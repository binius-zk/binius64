# Binius64 recursion: a committed-table discharge of the deferred monster evaluation

This directory adds four self-contained Cargo workspace members that build **native
Binius64 recursion** on top of primitives already in this repo — nothing here is a new
PCS, IOP, or field. The centrepiece is a protocol that makes the Binius64 IOP verifier
**succinct under recursion**: it batches the K deferred "monster" claims that
`compute_public_value` hoists out of the recursion circuit
(`crates/ip/src/channel.rs`, the `IronSpartanBuilderChannel` ZK-wrap seam) and discharges
all K with one argument whose dominant verifier cost is a fixed FRI/opening endgame,
independent of both the constraint-system size N and the batch size K. It is built
entirely from `sumcheck::batch_verify`, `sumcheck::verify`, `fracaddcheck::verify`,
`basefold::verify`, `fri::commit_interleaved`, and the existing ZK-wrap /
`compute_public_value` deferral hook.

**Read [`NOTE-for-jimpo-binius64-recursion.md`](./NOTE-for-jimpo-binius64-recursion.md)
first** — it is the design doc / protocol note (with `path:line` citations verified
against this tree), and it also lists several independent findings in `binius64`
(x86 GHASH cfg-gating with no runtime dispatch, an aarch64 `OptimalPackedB128`
regression, the `batch_prove`/`batch_verify` challenge-reversal TODO, and the expanded
`check_eval` eq-tensor). [`SPEC-monster-discharge.md`](./SPEC-monster-discharge.md) is the
detailed protocol spec (v2, soundness/implementability revisions marked).

## Branch base rev

This branch is based on **`5818a33`** — the rev the note's line-number citations are
verified against — so Jim's `path:line` references line up exactly. The recursion crates
were originally developed against an earlier rev of the same fork (`c799aa10`); they
compile and test green in-tree at `5818a33` with two small forward-ports: the SHA-256
gadget is now `sha256::Compress::new(..).state_out` + `populate_m` (was
`sha256_compress` + `populate_message_block`), used only by the demo/gate. No existing
`binius64` crate is modified; the only change outside this directory is adding
`"recursion/*"` to the root `Cargo.toml` `members`.

## Layout

| crate (dir) | package | what it is |
|---|---|---|
| `discharge/` | `binius-recursion-discharge` | The committed-table monster discharge: STEP 1 (K→1 batching, native final check) and STEP 2 (committed M_VK/M_D + weighted `fracaddcheck` Phase C + merged BaseFold opening; **CS-free verifier**). Vendored prover glue: cubic-product sumcheck, `W_eq` builder, native histogram evaluator, deterministic VKGen, chunked Phase-C tree, non-ZK merged batched opening. |
| `multizk/` | `binius-recursion-multizk` | The general **K-inner** generalization of the Binius64 ZK wrap (`multi_zk.rs`): one outer IronSpartan circuit symbolically executes the IOP verifier for K possibly-distinct leaves, one transcript, one combined BaseFold opening. |
| `wrap/` | `binius-recursion-wrap` | The **integrated flat wrap**: K same-shape leaf IOPs through a *substituting channel* (the §5.1 capture/substitution seam) + one outer proof + the STEP-2 discharge certifying exactly the K substituted values, all on one Fiat-Shamir transcript. |
| `gate/` | `binius-recursion-gate` | The smallest end-to-end native recursion "gate": a toy AND+MUL leaf and a SHA-256 leaf, each wrapped by the IOP verifier with the monster deferred via `compute_public_value`. Positive verify + tamper rejections. |

Everything is MIT/Apache-clean and separable; the general contribution is the discharge
(`discharge/`) plus the K-inner wrap (`multizk/multi_zk.rs`) and the substituting-channel
seam (`wrap/substituting.rs`).

## Reproduce (one `cargo test`)

From the repo root. The heavy synthetic discharge wants `--release`; single-threaded
keeps peak RAM modest.

```sh
# Everything, green:
cargo test --release -p binius-recursion-discharge -p binius-recursion-wrap \
                     -p binius-recursion-gate -- --test-threads=1

# Or per crate, with the interesting output:
cargo test --release -p binius-recursion-discharge -- --nocapture --test-threads=1
cargo test --release -p binius-recursion-wrap      -- --nocapture --test-threads=1
cargo test --release -p binius-recursion-gate      -- --nocapture --test-threads=1

# The synthetic scaling demo (the flat-verify headline; see numbers below):
cargo run  --release -p binius-recursion-discharge --bin scaling_demo

# The multi-inner aggregation demo (K=3 mixed-shape leaves -> one proof):
cargo run  --release -p binius-recursion-multizk
```

## What reproduces here vs. what is cited from real proofs

The discharge takes K `(claim point c_ℓ, value v_ℓ = monster_eval(c_ℓ))` pairs. Whether
those pairs come from a real leaf proof (captured through the recorder) or are synthesized
directly from the term table is **irrelevant to the discharge machinery** — this is the
spec's §P0.4 *standalone path* ("the (c, v) pairs ARE the statement"). So we reproduce the
**shape** of the note's headline numbers here on a synthetic AND-only table with
standalone claims, at CI/laptop-friendly sizes:

`scaling_demo` output (M2, K=3, this run):

```
         N    N_pad    n_d | ST1 verify(ms) | ST2 verify(ms)
       192      256     14 |          0.40  |          0.66
      3072     4096     14 |          0.45  |          0.57
     49152    65536     16 |          2.95  |          0.70
    786432  1048576     20 |         46.81  |          1.02
```

- **STEP-2 discharge verify is flat in N** (0.66 → 1.02 ms across a 4096× range in N —
  FRI log-depth only) and K-independent by construction. This is the synthetic reproduction
  of the note's *"discharge verify (STEP 2): dominant FRI/opening endgame N-independent
  (measured N=15 → 1 ms; N=24.5M → 4 ms)"*.
- **STEP-1 verify grows with N** (0.40 → 46.81 ms) because it does one O(N) native table
  pass — matching the note's STEP-1 line.

**Cited from real proofs in the note, NOT reproduced here** (they need real Binius64 leaf
proofs of a fixed 24.5M-term shape, which are out of scope for this share):

- the absolute `4 ms` / `18 ms` / `361 ms native` figures and the `N = 24,470,148`,
  `N_pad = 2^25` point;
- the integrated wrap's `18 ms vs 361 ms` (20×) real-leaf comparison and per-leaf replay
  timings;
- the prove-side figures (STEP-2 prove 22.7 s, VKGen 3.6 s, peak 7.5 GB) and proof-delta
  numbers.

The `wrap/` integrated tests reproduce the *integrated* flow end-to-end (substituting
channel → outer IronSpartan proof → STEP-2 discharge on one transcript) on tiny synthetic
leaves, including the adversarial suite (permuted sidecar, legitimate-value-wrong-point,
foreign-discharge splice, consistent-lie-vs-committed-table, forged-replay-cannot-prove).

## Notes / caveats reproduced faithfully

- The discharge instantiation built here targets an **all-BitAnd** CS shape (`T_mul = ∅`),
  enforced by an `ANDONLY` admission check; the general two-lane extension is sketched in
  the note §3.6 / spec §8.4 and is not built.
- STEP 2 is **one flat aggregation level** (K same-shape leaves → one combined proof);
  multi-level trees are out of scope (note §3.6, spec §8.11).
- Prover is O(K·N) in streaming mults (inherent); all committed data / proof size / final
  verifier are K-independent.
