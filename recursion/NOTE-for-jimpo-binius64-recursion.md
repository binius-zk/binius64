# Making the Binius64 verifier succinct under recursion: a committed-table discharge of the deferred monster evaluation

*A protocol note for Jim Posen (Irreducible / The Binius Developers). Citations are `path:line` in binius64. The body below was written against the working tree at HEAD `5818a33` (base `7e0e5df`).*

> **Re-verified against upstream/main `922df33f` (2026-07-12).** The inline `path:line` citations below are as-of `5818a33`. Between `5818a33` and `922df33f` upstream landed 227 commits, and a cluster of them **substantively changed the exact verifier code this note's argument rests on** — most importantly the deferred-monster claim structure (`shift/verify.rs`), the monster evaluator (`monster.rs`, rewritten by #1728), and the `compute_public_value` hook (`ip/channel.rs`, now `FieldFn`). See the **Addendum: upstream/main drift** at the end for a per-citation moved-vs-changed table; treat the addendum as authoritative over the inline line numbers when reading against current main.

---

## 1. Summary / contribution

The Binius64 IOP verifier is O(circuit size): essentially all of its per-proof arithmetic is the "monster" evaluation `check_eval` performs (`crates/verifier/src/protocols/shift/verify.rs:207-319`) — on real proofs we measured a **median 51.7M GF(2^128) mults per verify** (26 proofs, range 12.5M–101M), of which **~99.93% is the monster**. You already ship the escape hatch: `compute_public_value` (`crates/ip/src/channel.rs:87-111`) hoists the monster to a single inout wire in `IronSpartanBuilderChannel`, so the O(N) work disappears from the recursion circuit and is instead recomputed *natively by whoever verifies the outer proof*. That makes the **circuit** succinct but not the **verifier**: for K aggregated leaves the final verifier still pays K native monster passes.

This note describes a protocol that closes that last gap. Fix one CS shape S. Commit the constraint-term table **once** as a per-shape verification key. Then, for a batch of K same-shape proofs, discharge all K deferred monster claims with **one** batched argument. It collapses the K native monster passes (K·O(N)) the final verifier would otherwise pay into a single O(N) table pass — or, in the fully-succinct variant, into a fixed FRI/opening endgame independent of both N and K — leaving in either case only a sub-dominant K·polylog eq/h_op residual. Everything is built from primitives you already have — `sumcheck::batch_verify`, `sumcheck::verify` over a bivariate product, `fracaddcheck::verify`, `basefold::verify` — plus your ZK-wrap and your `compute_public_value` deferral hook. There is no new PCS, no new IOP, and no new field machinery. We built it, ran it on real proofs, and had it independently re-audited; the succinct discharge (STEP 2) verifies in **4 ms** at K=3 — its dominant endgame flat in N (N=15 → 1 ms, N=24.5M → 4 ms, FRI log-depth only) and K-independent by construction, with only the sub-dominant K·polylog residual — and the integrated wrap (K leaves + discharge on one Fiat-Shamir stream) verifies K=3 real leaves in **18 ms** versus **361 ms** for native re-verification.

We note up front: you flagged `compute_public_value` as `HACK … This feature should be killed and handled more elegantly` (`channel.rs:105-106`). Our experience is the opposite of "kill it" — it is exactly the right seam for recursion, and this note is in part an argument that the deferral hook deserves a first-class, sound discharge rather than removal. The instantiation we built and measured targets an all-BitAnd CS shape (`T_mul = ∅`); the general two-lane extension is mechanical and sketched in §3.6.

---

## 2. Background: why the verifier is O(N), and the deferral hook

**The monster.** `check_eval` closes the shift reduction by asserting `witness_eval · monster_eval == eval` (`shift/verify.rs:315-316`), where `monster_eval` is your batched multilinear evaluation of the constraint matrices. Re-associating `evaluate_monster_multilinear_for_operation` (`monster.rs:123-149`) and `evaluate_matrices` (`monster.rs:165-219`), for an all-BitAnd shape it is exactly

```
monster_eval(c) = Σ_{t < N}  λ_and^{m_t+1} · h_{op_t} · eq6(s_t, r_s) · eq(x_t, r_x') · eq(y_t, r_y)
```

- `t` indexes the `N = Σ_x (|a|+|b|+|c|)` `ShiftedValueIndex` occurrences (AND arity 3);
- `m_t ∈ {0,1,2}` is the operand slot → the weight `λ_and^{m+1}` is your `lambda_powers = powers(λ).skip(1)` (`monster.rs:137-140`);
- `op_t ∈ [8]` is the `ShiftVariant` → `h_{op}` is your `evaluate_h_op` output (`monster.rs:27`);
- `s_t` is the 6-bit shift amount → `eq6(s_t, r_s)` is the `evaluate_inplace_scalars(·, r_s)` fold (`monster.rs:144`);
- `x_t` is the constraint index → `eq(x_t, r_x')` is a read into `r_x_prime_tensor` (built at `monster.rs:135`, read in the scatter at `:179`/`:197`);
- `y_t = value_index` → `eq(y_t, r_y)` is a read into `r_y_tensor` (`monster.rs:198`, tensor built at `shift/verify.rs:269`).

The cost is one GF(2^128) mult **per term** (the scatter `evals[shift_id][amount] += constraint_eval · r_y_tensor[value_index]`, `monster.rs:197-198`) plus the three eq-tensor materializations `2^a, 2^m, 2^w` (`eq_ind_partial_eval_scalars`, `crates/math/src/multilinear/eq.rs:225-241`) plus `evaluate_h_op` / `l_tilde` (~7K). So `monster_eval ≈ T_and + T_mul + 2^a + 2^m + 2^w + ~7K`. Everything else in `IOPVerifier::verify` (`crates/verifier/src/verify.rs:105-265`) — intmul prodcheck/GKR, bitand, shift sumchecks, ring-switch, BaseFold/FRI — is polylog, except the O(public) pubcheck `evaluate_inplace_scalars` (`verify.rs:242`). We modeled the residual after removing the monster as **~36K mults + ~12K SHA-256 compressions** per child; the hash count matches the measured 12.2K almost exactly, confirming the monster is the whole story.

**The hook.** The monster is wrapped in `channel.compute_public_value(&inputs, closure)` (`shift/verify.rs:254-307`). Its inputs are the 61-scalar claim point `c = [r_zhat', λ_and, λ_int, r_x'_and(23), r_x'_mul(0), r_j(6), r_s(6), r_y(23)]` (`shift/verify.rs:244-252`), all public-coin. In `IronSpartanBuilderChannel` the closure is **skipped** and the result becomes one inout wire (`builder_channel.rs:224-243`, via `combine_varlen` with an all-public input path); the contract (`channel.rs:87-111`) states this replaces "a sub-circuit's worth of constraints." Symbolic execution of the whole leaf verifier through this channel is exactly your ZK-wrap (landed #1448, present at our pin); we repurposed it as recursion.

**The gap.** Deferral removes ~99.93% of the mults *from the circuit*, but the deferred value `v = monster_eval(c)` still has to be produced and certified somewhere. In the one-level wrap that "somewhere" is the final verifier's native recompute — O(N) per leaf, paid K times for K leaves. That is the non-succinctness this note removes.

---

## 3. The discharge protocol

Fix a shape S with term table T = the flattened list of N tuples `τ_t = (x_t, y_t, s_t, op_t, m_t)`, one per `ShiftedValueIndex` occurrence (source: `crates/core/src/constraint_system.rs:134-143, 325, 332`). The deferred claim of leaf ℓ is `v_ℓ = monster_eval(c_ℓ)` above. Given K same-shape claims `{(c_ℓ, v_ℓ)}`, prove all K with one argument whose dominant verifier cost is a fixed FRI/opening endgame — independent of N and K — above a sub-dominant K·polylog eq/h_op residual.

### 3.1 Algebraic decomposition

Define three per-row **virtual** columns for claim ℓ (never committed):
`E_x^ℓ[t] = eq(x_t, r_x'^ℓ)`, `E_y^ℓ[t] = eq(y_t, r_y^ℓ)`, and `E_g^ℓ[t] = λ_and^{m_t+1} · h_{op_t}^ℓ · eq6(s_t, r_s^ℓ)` — the last folds the whole "meta" weight (slot × shift-variant × amount) into an 11-bit index `u_t = (s_t‖op_t‖m_t)`. Then

```
v_ℓ = Σ_t E_x^ℓ[t] · E_y^ℓ[t] · E_g^ℓ[t]
```

— a **product of three multilinears** over `t ∈ B_25`, i.e. a degree-3 sumcheck claim. This is `evaluate_matrices` + `evaluate_monster_multilinear_for_operation` re-associated, nothing more.

Two identities carry the reduction:

- **(D) Marginalization.** For the ρ-weighted histograms `D_x(a;ρ) = Σ_t eq(t,ρ)·⟦x_t=a⟧` (and `D_y, D_g`), the reduced column evals *are* the histogram MLEs at the claim's own point: `Ẽ_x^ℓ(ρ) = D̃_x(r_x'^ℓ)`, etc.
- **(G) Eight-point split of H.** In char 2, `λ^{m0} = (λ+1)·eq(m0, (λ+1)^{-1}+1)` and `λ^{2m1} = (λ+1)²·eq(m1, ((λ+1)²)^{-1}+1)` (needs `λ_and ∉ {0,1}`, FS-random, abort ≤ 2/2^128 per claim); with `H^ℓ(op) = Σ_{o∈[8]} h_o^ℓ · eq3(op, bits(o))`, `E_g^ℓ` reduces to **8** evaluations of `D̃_g` at fixed shift-variant points. So each claim's three column-evals become `a_ℓ = D̃_x(·)`, `b_ℓ = D̃_y(·)`, and 8 values `d_{ℓ,o} = D̃_g(·)` — all evaluations of a single histogram oracle `M_D` (blocks `D_x | D_y | D_g`) at points determined by `c_ℓ`.

### 3.2 The committed table (= the vkey)

Per shape (STEP 2), commit three scalar **address** columns over `t ∈ B_25` — `X[t] = ι(x_t)+κ_x`, `Y[t] = ι(y_t)+κ_y`, `U[t] = ι(u_t)+κ_u` — laid out as one 27-var oracle `M_VK = [X|Y|U|0]` via `fri::commit_interleaved` (`crates/iop-prover/src/fri/commit.rs:39-91`, transcript-free). Its digest **is the verification key**. `ι` embeds address bit `k` to basis element `β_k`, `k ∈ [0,23)`; `V := ι(B_23)`. The block tags are `κ_c = ι'(c)` mapping into the **next two** basis elements `β_23, β_24` (κ_x=0, κ_y=β_23, κ_u=β_24, κ_pad=β_23+β_24). Nothing here is secret — the discharge uses **no ZK masking anywhere**, which sidesteps both the mask-reuse hazards and your `log_batch_size==1` batched-BaseFold restriction (see §6).

### 3.3 The K→1 reduction (four phases, one transcript)

**Phase 0 — absorb + structural preconditions.** Observe the full VK metadata blob (`vk_digest, cs_digest, N, N_pad, parity, dims, FRIParams, hash-suite id, row-order version`) into the transcript **before** sampling any challenge. Then assert, natively, structural facts that are probability-free: `cs_digest == H(SerializeBytes(leaf_cs))` (canonical versioned serialization exists at `constraint_system.rs:607-616`); all-BitAnd admission (every `MulConstraint` is the empty-operand padding default, `constraint_system.rs:560-574`, so `intmul_part ≡ 0` and `monster_eval` = the T-sum exactly); single-source and coverage (§4).

**Phase A — one batched degree-3 sumcheck over `t`.** `sumcheck::batch_verify(n_vars=25, degree=3, sums=[v_ℓ]_1^K, channel)` (`crates/ip/src/sumcheck/batch.rs:37-60`) → shared point ρ. The prover is a vendored cubic-product sumcheck (a ~120-line mirror of your `BivariateProductSumcheckProver`, `crates/ip-prover/src/sumcheck/bivariate_product.rs:19-27`, driven by `batch_prove_and_write_evals`, `.../batch.rs:126-141`). At `finish` it writes the three multilinear evals per claim `(a_ℓ, b_ℓ, g_ℓ)`; the verifier checks `Horner(μ; a_ℓ·b_ℓ·g_ℓ) == e_A` (your intmul `verify_phase_3` multi-point pattern, `crates/verifier/src/protocols/intmul/verify.rs:111-183`), then receives the 8K values `d_{ℓ,o}` and checks `g_ℓ == Σ_o γ_{ℓ,o}·d_{ℓ,o}` (γ from `evaluate_h_op`, transparent). By (D)/(G) this is now `10K` MLE-eval claims on `M_D`.

**Phase B — 10K(+1) claims → one point σ.** Your `BivariateProductSumcheckProver::new([W_eq, M_D], Σφ^i e_i)` with `W_eq = Σ_i φ^i·eq(·, p_i)`; verifier `sumcheck::verify(25, 2, sum, channel)` (`crates/ip/src/sumcheck/verify.rs:39-63`) → σ, then `m·W̃_eq(σ) == e_B` with `W̃_eq(σ) = Σ_i φ^i·eq_ind(p_i, σ)` computed via your `eq_ind` (`eq.rs:198-203`) — this is the only K·O(|point|) work in the verifier (the sub-dominant residual noted throughout).

**Phase C (STEP 2) — well-formedness of `M_D` vs `M_VK` via one weighted fracaddcheck.** The prover commits `M_D`; the verifier samples τ and runs `fracaddcheck::verify(k=27, {num_eval:0, den_eval:d_root, point:[]}, channel)` (`crates/ip/src/fracaddcheck.rs:26-84`) on the claimed rational identity: `Σ_t eq(t,ρ)/(τ+X[t]) + …/(τ+Y[t]) + …/(τ+U[t]) + Σ_{a,c} M_D(a,c)/(τ+emb(a,c)) ≡ 0`. With the block tags **coset-disjoint** (§3.5), partial-fraction uniqueness forces, pole by pole, `D_x(a) = Σ_{t: X[t]=ι(a)+κ_x} eq(t,ρ)` — the histogram definition. This is a **weighted** (coefficient-matching) use of fractional sums: numerators are field eq-weights, not integer multiplicities, so the char-2 even-multiplicity cancellation attack on LogUp-style counting does not apply. (This is our one "please sanity-check against the literature" item, §7.) The prover layer split is on the top variable (`crates/ip-prover/src/fracaddcheck.rs:69-70`, via `split_half_ref`), so a blocks-high layout is prover-natural.

**Final check.** STEP 1: the verifier natively rebuilds `M_D` from the CS (whose digest it checked) in one O(N) pass and asserts `M̃_D(σ) == m` — this collapses `K × (51.7M + rebuild)` to **one** table pass. STEP 2: two `basefold::verify` openings — `M_D` at σ, and `M_VK` via a corner trick (the prover already sent the three block corner values; sample `ρ_c ∈ F²` and open the combined `M̃_VK([σ, ρ_c])` against `vk_digest`).

### 3.4 Soundness — S1 single-source (the headline)

The object the discharge certifies is not "a value `v_ℓ`" in the abstract — it is **the specific wire leaf ℓ's `check_eval` multiplies.** `check_eval` asserts `witness_eval · monster_eval == eval` (`shift/verify.rs:315-316`). In your native protocol the *other two* factors are already fixed: `witness_eval` is `recv_one`'d at `shift/verify.rs:174` and then PCS-bound — it is fed as the evaluation claim into `ring_switch::verify(shift_output.witness_eval(), …)` (`verify.rs:234`), whose reduced claim becomes the `claim` of `verify_oracle_relations` on the committed `trace_oracle` (`verify.rs:256-260`; `trace_oracle = recv_oracle()` at `:130`); and `eval` is the shift-sumcheck output (`shift/verify.rs:167-170`), determined by that reduction. (Your own note at `shift/verify.rs:310-314` is a completeness remark — you *could* compute `witness_eval = eval/monster_eval` but that needs inverting a random element, so you read it instead; it is not a statement that `witness_eval` is unbound.) So natively the equation has force *precisely because `monster_eval` is also fixed* — computed in-line as the true public value — which is what pins the committed witness to the reduced claim.

Under deferral this last anchor is removed: the closure at `:254` is skipped and `monster_eval` becomes a free inout wire, while `:316` still asserts in-circuit. With a **free** third factor, `witness_eval · monster_eval == eval` is satisfiable for *any* committed `witness_eval` and *any* reduced `eval` — a cheating prover just sets the wire to `eval · witness_eval^{-1}`. The equation imposes nothing; the bindings of `witness_eval` and `eval` go vacuous. The **only** thing that restores force is pinning this wire to its true value `monster_true(c)`. That is what the discharge is for — and it must pin the **same** wire `:315` multiplies. If the discharge instead certifies a *decoupled* second value (e.g. a second read of `v_ℓ`), the multiplied wire stays free and `:316` stays vacuous: a clean forgery, independent of the discharge's own soundness. In our wrap this is structural — the value is materialized exactly once, at the `compute_public_value` site (`:254`), and consumed only at `:315-316`; Phase A's target is that same materialized wire.

### 3.5 Soundness — Fiat-Shamir binding of the table, and char-2 subtleties

- **The commitment must be observed before the challenge.** `basefold::verify` takes `codeword_commitment` as an **argument** and never absorbs it (`crates/iop/src/basefold.rs:57-106`), and FRI query indices are `transcript.sample_bits(index_bits)` (`crates/iop/src/fri/verify.rs:200`). So a `vk_digest` (or `digest_D`) that is not observed into the transcript is **not bound by the queries**, and admits a table-swap: a prover picks the table after seeing ρ. Phase 0 observation is a hard soundness step, not hygiene. (In your own non-recursive flow the trace commitment *is* absorbed, via `recv_oracle` at `verify.rs:130` — so this is a requirement on *our* discharge, not a defect in yours.)
- **Coset-disjoint tags.** For Phase C to isolate poles per block, the tag *differences* must lie outside `V`. "Field-distinct" tags are insufficient — cosets of the same F2-subspace collide unless `κ_c ⊕ κ_c' ∉ V`. Constructing `κ_c = ι'(c)` over `β_23, β_24` gives `κ_c ⊕ κ_c' ∈ span(β_23,β_24)\{0}`, disjoint from `V` by construction (asserted at VKGen). Bonus: `emb(a,c) = ι(a)+ι'(c)` is then plainly linear, so the Phase-C den column is transparent.
- **Parity.** We pad T to `N_pad` with copies of one fixed dummy tuple. In char 2 an *even* number of copies cancels identically in every sum; since N has no parity guarantee, we don't require evenness — instead `parity := (N_pad−N) mod 2` is frozen into the VK, and Phase A runs on `v_ℓ + parity·w_d(c_ℓ)`, where `w_d` is the single unpaired dummy row's weight (~60 transparent mults, reusing `eq(0,r)=1+r`, `eq.rs:212-215`). Exact, and it preserves S1 (the correction is public-derived arithmetic on the one recv'd `v_ℓ`).

**Reduction chain.** S1 + coverage (structural) ⇒ the certified element *is* each leaf's `check_eval` multiplicand, for every leaf; PCS binding ⇒ `M_D` fixed; Phase C ⇒ `M_D` = ρ-histograms of the committed columns; Phase B ⇒ the 10K+1 eval claims true; (D),(G)+parity (exact algebra) ⇒ `a_ℓ,b_ℓ,d_{ℓ,o}` are true marginals; Phase A ⇒ every `v_ℓ = monster_eval(c_ℓ)` whp. Error budget (|F|=2^128, 96-bit target): Phase A `(K−1)/2^128 + 75/2^128`; Phase B `10K/2^128 + 50/2^128`; Phase C rational-identity SZ `≤ 2^-100`; total `≪ 2^-96` for `K ≤ 2^20`. The one trust root (STEP 2 only): that the committed columns are the canonical table of the digest's CS — discharged by deterministic regeneration audit, the same trust class as the CS itself. STEP 1 has no such root (it rebuilds `M_D` from the CS directly).

### 3.6 Scope

This is **one flat aggregation level**: K same-shape leaves → one combined proof. We deliberately do *not* claim to discharge the outer IronSpartan wiring evals — those use a different seam (`spartan-verifier/src/lib.rs:197-218` public closure via `compute_public_value`, the call at `:207`; `:226-253` `verify_oracle_relations` + `TransparentEvalFn`) and a different kernel (`eq·eq·λ^m`, `wiring.rs:41-99`). A strict-simplification "Variant B" table discharges those too (one derived point per claim, no 8× blowup), but it is unbuilt and out of scope here. The general (non-AND-only) leaf needs a second lane/table for the `MulConstraint` slots `(a,b,lo,hi) ↦ λ_int^{1..4}` (`shift/verify.rs:289-306`); mechanical, same shape.

---

## 4. Integration with the ZK-wrap

The discharge runs as one more inner segment of the flat multi-inner wrap, **after** the K leaf IOP verifications, on the same transcript. Because Phases A/B/C use only `IPVerifierChannel` operations (`batch_verify`, `sumcheck::verify`, `fracaddcheck::verify`, recv/sample/assert), they symbolically execute through `IronSpartanBuilderChannel` into outer constraints exactly like everything else — measured in-circuit cost ≈ 150–250K B128 mults at K=64 (grows with K, the residual term), versus K×51.7M removed. Only the PCS openings (STEP 2) / the native pass (STEP 1) escape the channel abstraction (Merkle queries do; precedent `crates/iop/src/basefold_zk_channel.rs`) and join the final verifier's native endgame alongside the wrap's own combined opening.

**The seam.** We interpose a substituting channel during each leaf's verify. At the sole leaf-verifier `compute_public_value` call site (`shift/verify.rs:254` — we grep-confirmed it is the *only* one in the leaf verifier; the other in the tree, `spartan-verifier/src/lib.rs:207`, belongs to the outer verifier and is never wrapped), it (1) asserts `|inputs|` equals the shape arity — **our** check — and relies on your public-tag enforcement: `compute_public_value` forwards to `combine_varlen`, whose non-public branch hard-`panic!`s (`builder_channel.rs:234-238`), so a `recv`-derived input can never silently slip through (the trait doc at `channel.rs:99` calls this a "debug-assert," but the `IronSpartanBuilderChannel` impl is an unconditional panic); (2) supplies the prover-claimed monster value as the single materialized wire — **dropping the closure un-invoked**, exactly as your symbolic builder does (`channel.rs:101-103`); (3) records `(c_ℓ, v_ℓ)` to a sink; (4) returns that wire, which `:315-316` then multiplies. Coverage assert: after K leaves, the sink holds exactly K claims and Phase A's targets are exactly the sink's values (parity-corrected). An undischarged `v_ℓ` leaves that leaf's `check_eval` vacuous (§3.4), so coverage is a soundness step.

> **Correction to our own spec worth flagging:** an earlier draft made the seam a second `recv_one` of `v_ℓ`. That is unsound *and* unimplementable in your wrap — unsound because a second read is exactly the decoupled wire of §3.4 (the multiplied wire stays free), and unimplementable because wrap recvs are OTP-encrypted (`builder_channel.rs:175-182`: `recv_one` returns `inout − key`, the precommit-segment pad — the native verifier never learns the plaintext). The correct and only sound seam is `compute_public_value` substitution, with that materialization as the single S1 binding site. This is what we built and audited.

---

## 5. Measurements (real proofs, M2 laptop unless noted)

Fixture: K=3 real, distinct same-shape leaf proofs; table `N = 24,470,148` terms, `N_pad = 2^25`.

| metric | value |
|---|---|
| **discharge verify (STEP 2)** | **4 ms** at K=3; dominant FRI/opening endgame N-independent (measured `N=15 → 1 ms`; `N=24.5M → 4 ms` — FRI log-depth only) and K-independent by construction, above a sub-dominant K·polylog eq/h_op residual (not separately K-swept) |
| discharge verify (STEP 1) | 1.37 s incl. one O(N) native table pass done **once** (0.23 s, K-independent) vs `0.112 s × K` native monster passes |
| **integrated final verify** (K=3, one FS stream) | **18 ms** vs **361 ms** native re-verify (20×; gap grows with N·K via the native K·O(N) baseline); per-leaf transcript replay 0.9–1.3 ms vs 113–149 ms native monster |
| discharge prove | STEP 1: 6.96 s, **+2,512 B** to the proof. STEP 2: 22.7 s (Phase C 9.1 s), VKGen once 3.6 s, peak 7.5 GB |
| proof delta (STEP 2) | +1.394 MB/batch (two 232-query openings); merged non-ZK opening → 1.048 MB (−25%); amortized at K=64 ≈ **22 KB/leaf** |

Independently re-audited end to end (from-scratch re-prove byte-identical; three adaptive attacks — permuted sidecar / right-value-wrong-point / foreign-discharge splice — all rejected at the correct layer). Decisive negative test: an adaptive lie planted in `M_D`'s unused selector block, provably invisible to both sumcheck phases, is caught by the STEP-1 native rebuild (and, in STEP 2, by Phase C) — exactly where the soundness argument says it must be.

**Native fast-path (independent of the discharge).** For the native monster itself, `evaluate_matrices` materializes the full `r_y_tensor` of `2^w` field elements (`= value_vec_layout.committed_total_len`, i.e. `2^w · 16` B — hundreds of MB to ~1 GB at the `w ≈ 25–26` of our larger shapes) and does **one random read per term** into it (`monster.rs:197-198`, `r_y_tensor[value_index.0]`) — a cache-missing scatter that dominates per-term cost. Separately, `check_eval` rebuilds `3·n_and` operand-reference entries on **every** call via the `multiunzip` (`shift/verify.rs:274-278`). We replaced the tensor with a **factored split** `eq(y, r_y) = hi[y>>12] · lo[y & 4095]` (a coordinate-partition of `r_y` into two small tensors `2^{w−12}` and `2^{12}`, multiplied on the fly — exact), which removes the large materialization and turns each per-term read into two small-table reads; this is a drop-in for the native evaluator and accounts for most of the STEP-1 native-pass speedup.

---

## 6. Findings in binius64, and what we had to vendor

Neutral peer feedback, precise:

**(i) x86 GHASH is compile-time cfg-gated with no runtime dispatch → silent software fallback.** `crates/field/src/arch/x86_64/packed_ghash_{128,256,512}.rs` gate on `#[cfg(target_feature = "pclmulqdq")]` (128, e.g. `:22,25,54,81,106`), `"vpclmulqdq"` (256, `:25,28,63,…`), and `"vpclmulqdq"+"avx512f"` (512, e.g. `:23,26,61`). There is **no `is_x86_feature_detected!` anywhere in `crates/field`** — we grepped. Default `rustc` for `x86_64-unknown-linux-gnu` targets the 2003 baseline, which predates pclmul, so **every default-flags x86 build runs the field multiply in software.** Measured on an Ice Lake r6i.8xlarge (real proof): native verify **9.218 s → 0.784 s (11.8×)** with `-C target-cpu=native`, constant across thread counts (804 `(v)pclmul` + 60 GFNI in the native disassembly); prove **5.1–5.5×**. Apple Silicon is unaffected (aarch64 enables clmul by default), which is why the penalty is invisible on dev machines. Because selection is compile-time with no fallback, a binary built with a feature the host lacks **SIGILLs** — so a portable binary is stuck in software today. Suggestion: `is_x86_feature_detected!` runtime dispatch (the standard approach for CLMUL/AVX-512 crypto kernels), or at minimum a build warning when the pclmul path is not compiled in.

**(i-bis) aarch64 `OptimalPackedB128` regression.** Separately, at our pinned rev `PackedBinaryGhash1x128b` multiplies **~7× slower than scalar B128 on aarch64** (12.5 vs 1.8 ns/mult) — the packed wrapper's strategy dispatch doesn't inline; a naive port to it regressed our prover 22.6→36.9 s until we forced `PB=B128` on aarch64. This likely taxes the main `Prover<OptimalPackedB128>` on Apple Silicon at this rev. Worth a look.

**(ii) `batch_prove` / `batch_verify` challenge-reversal asymmetry.** `crates/ip-prover/src/sumcheck/batch.rs:96-98` (and `:209-211`) reverse the challenge vector with an in-repo `// TODO: this differs from prove_single, which doesn't reverse.`, while the verifier side does not — the note at `crates/ip/src/sumcheck/batch.rs:27` documents the high-to-low convention. We hit this as a footgun wiring the batched driver; it cost us a debugging session. A comment cross-link at the verifier, or resolving the TODO, would help downstream users.

**(iii) `log_batch_size == 1` assertion in the batched BaseFold verifier.** `verify_mlecheck_basefold_zk_batch` hard-asserts `max_log_batch_size == 1` (`crates/iop/src/basefold.rs:158`) — it is shaped for the ZK interleaved `(π‖ω)` mask. To merge our two discharge openings (`M_D` at `2^25`, `M_VK` at `2^27`) into one 232-query opening that binds both Merkle trees, we had to **vendor a ~330-line non-ZK batched verifier**. Mixed-size lift had to be interior zero-pad; the ZK-style low-block placement provably fails FRI/MLE consistency. Question: would a non-ZK batched-opening path be wanted upstream? It seems generally useful, not just to us.

**(iv) Expanded eq-tensor + per-verify `Vec` allocations in `check_eval`.** As in §5: the full `r_y_tensor` of `2^w` field elements (`2^w · 16` B — hundreds of MB to ~1 GB at the witness sizes we run) read once per term at `monster.rs:197-198`, plus `3·n_and` operand-reference entries rebuilt per verify via the `multiunzip` at `shift/verify.rs:274-278`. The factored split-eq-tensor (`eq(y,r_y) = hi[y>>12]·lo[y&4095]`, exact) removes the large materialization and the cache-missing per-term read; the per-verify allocation is straightforward to hoist. Happy to contribute the split evaluator if useful.

**Vendored (prover-side/glue; zero verifier-protocol novelty):** the cubic-product sumcheck prover (mirror of `bivariate_product.rs`), the substituting channel, the `W_eq` builder, the native histogram evaluator, deterministic VKGen (tags/parity/ANDONLY/cs_digest), the chunked Phase-C tree builder, and the non-ZK batched verifier of (iii).

---

## 7. Open questions / requests for review

1. **Is the S1 substitution sound as argued (§3.4, §4)?** Our claim: under deferral the monster factor is a *free prover input* to `check_eval`, so `witness_eval · monster == eval` (`:315-316`) constrains *only that wire* — with `witness_eval` PCS-bound via ring-switch (`verify.rs:234` → `verify_oracle_relations` on `trace_oracle`, `:256-260`) and `eval` fixed by the shift reduction, the equation is vacuous unless the deferred wire is pinned to `monster_true(c)`. Hence the discharge must bind the *same* wire `:315` multiplies, materialized once at `:254`; a decoupled second read is the forgery surface, which we close structurally. We would value your read on whether that is the complete argument in your channel model.

2. **Is there a cleaner "committed-table-as-vkey" form in your framework?** We commit `[X|Y|U|0]` as one interleaved oracle and reduce well-formedness with a *weighted* fracaddcheck (coefficient-matching partial fractions, char 2). Is the weighted variant something you'd trust over a tag-packed prodcheck multiset argument (our fallback), and does it have a canonical treatment we should cite (LogUp/Habock-style) before we freeze it? This is the one step we'd most like external eyes on.

3. **Would upstream want any of:** the monster discharge itself; the non-ZK batched opening (finding iii); or the factored split-eq-tensor native evaluator (finding iv)? All are MIT/Apache-clean and separable.

4. **The field-arithmetic fixes:** runtime CPU dispatch (or a build warning) for x86 GHASH (finding i), the aarch64 `OptimalPackedB128` regression (i-bis), and the `batch_prove`/`batch_verify` challenge-order TODO (finding ii). These are independent of the recursion work and we're glad to file issues / PRs with the measurements above if that's the useful path.

*Everything above is against the tree at HEAD `5818a33`; the verifier algorithm is identical across `7e0e5df..HEAD` (only the fixed-size-serialization commit `6adfac7` touches `crates/{verifier,ip,iop}`, and it moves no cited line). All measurements are on real proofs and were independently reproduced.*

---

## Addendum: upstream/main drift (re-verified against `922df33f`, 2026-07-12)

The 227 commits `5818a33..922df33f` moved almost every cited line and **substantively changed the core of the note's argument.** Two upstream work-streams matter:

- **Value-vector public/hidden segment refactor** (#1554 don't-require-pow2-padding, #1583 align-ValueVec-words, #1724 store-hidden-segment-length, plus #1585 generalize `compute_public_value`). The deferred monster claim is no longer `monster_eval(c)` over a flat `2^w` value vector: the word index is now `log_witness_words()+1` coordinates — an `r_y` of length `log_witness_words()` **plus a separate top `r_segment`** selecting the public (low) vs hidden (high) half — and the value-vec MLE tensor is built **segmented and scaled** (`scaled_eq_ind_partial_eval_scalars`, `public_scale = eq_one_var(r_segment,0)·eq_ind_zero(r_y_high)`), so `E_y[t] ≠ eq(y_t, r_y)`. There are now **two** `compute_public_value` sites in the leaf verifier (`shift/verify.rs:316` monster **and** `:328` `PublicWordsEvalFn`), so the note's §4 claim "the sole leaf-verifier `compute_public_value` call site" is **no longer true**.
- **Monster rewrite** (#1728): `evaluate_monster_multilinear_for_operation` has a new signature (a single pre-tensored `shift_scalars: &[E; SHIFT_VARIANT_COUNT·Word::BITS]` in place of `(r_s, r_y_tensor, h_op_evals)`), returns `E` directly, and the standalone `evaluate_matrices` scatter is **deleted**.

**Impact on the discharge (headline):** the recursion crates compile-port mechanically, but the discharge's Y-column histogram identity (§3.1 `E_y[t]=eq(y_t,r_y)`, §3.3 Phase A `b_ℓ=D̃_y(r_y)`), its claim-point parsing (arity, §2 hook), and the recorder's "exactly one monster claim per leaf" invariant (§4) are all invalidated by the segment split. Re-deriving the Y-block for the segmented value vector is protocol work, not an API rename.

### Per-citation status (M = moved only; **C = code changed** — read current behavior)

| note citation (`@5818a33`) | status | current location / what changed (`@922df33f`) |
|---|---|---|
| `shift/verify.rs:207-319` (check_eval/monster) | **C** | `check_eval` at `:257`; monster deferral at `:289-317`; final assert at `:341-342` — now `trace_eval·monster_eval`, `trace_eval` reconstructed from public+hidden segments via `extrapolate_line` (`:334`) |
| `shift/verify.rs:254` / `:254-307` (sole `compute_public_value`) | **C** | now **two** sites: `:316` (`MonsterEvalFn`) and `:328` (`PublicWordsEvalFn`); the monster closure is a `FieldFn` impl (`:491-501`) |
| `shift/verify.rs:244-252` (claim point `c`) | **C** | `:296-305`; **extra trailing `r_segment`** (`:304`) → arity +1; `r_y` now `log_witness_words()` long |
| `shift/verify.rs:315-316` (`witness_eval·monster==eval`) | **C** | `:341-342`; multiplicand is reconstructed `trace_eval`, not the raw `witness_eval` (`witness_eval` is now only the hidden-segment half, `:215`, `:334`) |
| `shift/verify.rs:167-170,174,269,274-278,289-306,310-314` | **C** | all inside the rewritten `check_eval`/`MonsterEvalFn` (`:190-500`); `r_y_tensor` now segmented (`:435-445`); operand multiunzip at `:460-465` / `:469-482` |
| `monster.rs:27` (`evaluate_h_op`) | M | `:22` (signature unchanged) |
| `monster.rs:123-149` (`evaluate_monster_multilinear_for_operation`) | **C** | `:117-154`; **rewritten (#1728)** — new `shift_scalars` arg, returns `E` (no `Result`) |
| `monster.rs:135,137-140,144,165-219,197-198` (`evaluate_matrices` + scatter) | **C** | `evaluate_matrices` **deleted**; per-term work now at `:140-151` (`operand_shift_scalars[index]·r_y_tensor[value_index]`) |
| `crates/ip/src/channel.rs:87-111` (`compute_public_value`) | **C** | trait at `:87-110`; signature now `f: impl FieldFn<F>` (was `impl FnOnce(&[F])->F`); non-wrapper impl `f.call_native(inputs)` (`:157`); HACK note `:108-109` |
| `crates/verifier/src/verify.rs:105-265,130,234,242,256-260` | **C** (:130) / M (rest) | `trace_oracle = channel.recv_oracle(self.log_witness_elems(), true)` at `:141` — `recv_oracle` now takes `(log_msg_len, is_witness_dependent)` |
| `crates/ip/src/sumcheck/batch.rs:37-60` (`batch_verify`) | M | `:37` (prover-side `batch_prove*` now **infallible** — returns output directly, no `Result`) |
| `crates/ip/src/sumcheck/verify.rs:39-63` | M | `:39` |
| `crates/ip/src/fracaddcheck.rs:26-84` | M | `:26` (verifier); prover `fracaddcheck::Error` type removed (infallible) |
| `crates/iop/src/basefold.rs:57-106` (basefold verify) | **C** | `verify_mlecheck_basefold` at `:59`, now **batched** (`codeword_commitments: &[Channel::Commitment]`) and driven through a `MerkleIPVerifierChannel` (#1693) |
| `crates/iop/src/basefold.rs:158` (`log_batch_size==1` assert, finding iii) | **C** | **assert gone**; upstream #1500/#1586 added combined-FRI / mixed ZK-non-ZK batched openings — the note's vendored non-ZK batched verifier (finding iii) may now be replaceable by the upstream batched API |
| `crates/iop/src/fri/verify.rs:200` (query `sample_bits`) | M | `:194`, now `channel.sample_bits(index_bits())` via the Merkle IP channel (#1693) |
| `crates/iop/src/basefold_zk_channel.rs` (§4 precedent) | **C** | file **renamed** to `crates/iop/src/basefold_channel.rs` (#1693/#1696); FRI/BaseFold now take `MerkleIP{Prover,Verifier}Channel` wrappers, not raw transcripts |
| `crates/math/src/multilinear/eq.rs:198-203,212-215,225-241` | M | `eq_ind` `:113`, `eq_ind_partial_eval_scalars` `:136` (eq.rs reorganized) |
| `crates/verifier/src/protocols/intmul/verify.rs:111-183` | M | `verify_phase_3` `:75` |
| `crates/core/src/constraint_system.rs:134-143,325,332,560-574,607-616` | **C** (path) | `constraint_system.rs` split into a **directory**: `ShiftedValueIndex`→`constraint_system/shift.rs` (`amount` is now `u8`, not `usize`), constraints/serialize→`system.rs`, layout→`layout.rs` |
| `spartan-verifier/src/lib.rs:197-218,207,226-253` (outer wrap) | M | outer `compute_public_value` at `:219` (a `FieldFn`), `verify_oracle_relations` `:242`, `TransparentEvalFn` `:393` |
| `builder_channel.rs:175-182,224-243,234-238` (IronSpartan wrap) | **C** (path) | moved to `crates/spartan-verifier/src/wrapper/builder_channel.rs`; `compute_public_value` at `:108` now materializes via `hint_varsize` with `f.call_native` (`:121`) |
| `wiring.rs:41-99` (outer eq·eq·λ^m kernel) | M | `crates/spartan-verifier/src/wiring.rs`; `PublicWiringEvalFn` (a `FieldFn`) `:98-118`, split into `evaluate_segment_wiring_mle`/`evaluate_wiring_mle_public` |

### Transcript / Fiat-Shamir (the `#1611` question)

`#1611` ("rework FRI & BaseFold challenge order") **did** change the FRI first-fold challenge order from `[early ++ later ++ outer]` to **`[early ++ outer ++ later]`** (only the oracle-combine block moved to the middle; total count and `max_log_msg_len` unchanged). This is **not** a change to what the transcript *absorbs* (roots stay observed messages, opening advice stays unobserved decommitment — preserved verbatim by the new `MerkleIP{Prover,Verifier}TranscriptChannel` wrappers from #1693). But it **does** change the order challenges are consumed, and the discharge's *vendored* non-ZK batched opener (finding iii, `recursion/discharge/src/merged.rs`) hand-rolls the **pre-#1611** order (outer challenge placed after the `b` batch-fold rounds) plus the pre-#1693 raw-transcript FRI API. Under `922df33f`, `FRIQueryVerifier::new_batch` re-slices the supplied challenge vector as `[early ++ outer ++ later]` (`fri/verify.rs:113,134`), so `merged.rs` must be re-derived to the new order (or, better, replaced by the upstream batched BaseFold opening now that the `log_batch_size==1` restriction is gone). This is the STEP-2 half of the port blocker.