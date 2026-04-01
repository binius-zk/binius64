# Engineering Roadmap: Symbolic Verifier Refactor

## Purpose

This document turns the symbolic-verifier architecture direction into a concrete
engineering roadmap for the `RingSwitch + BaseFold / FRI` verifier slice.

The target end state is:

- one semantic verifier core for `RingSwitch -> BaseFold / FRI -> final consistency`
- one native streaming interpreter for performance
- one replay interpreter for extraction and testing
- one symbolic interpreter suitable for circuit compilation / recursion
- clean extraction to Lean with both `hax` and `aeneas`

This roadmap is intentionally incremental. The plan should be updated as new
information arrives, especially around performance and symbolic-data-model
constraints.

## Design Constraints

- Do not regress native verifier performance.
- Do not force the native path to materialize an owned replay trace.
- Do not keep `verify_oracle_relations` as the long-term semantic boundary.
- Do not keep `TransparentEvalFn = Box<dyn Fn(...)>` as the long-term relation boundary.
- Preserve the good existing property that the algebraic reductions are already
  mostly generic over `IPVerifierChannel`.

## Current Boundary Problems

The current verifier spine in `crates/verifier/src/verify.rs` already computes:

- `IntMul`
- `AndCheck`
- `Shift`
- `RingSwitch`

But the PCS portion is still hidden behind:

- `crates/iop/src/channel.rs`
- `crates/iop/src/basefold_channel.rs`
- `IOPVerifierChannel::verify_oracle_relations`

That boundary currently bundles together:

- BaseFold opening
- FRI query verification
- challenge-point reconstruction
- transparent relation evaluation
- final consistency checking

That is the wrong abstraction for:

- modular verifier composition
- symbolic execution
- recursion / circuit compilation
- Lean extraction

## Target Architecture

### Layer 1: Verifier Plan

This layer contains static protocol metadata and planning decisions.

Proposed modules:

- `crates/verifier/src/pcs/plan.rs`
- `crates/iop/src/fri/plan.rs`

Primary responsibilities:

- oracle specs
- FRI arity schedule
- opening mode (`Plain`, `Zk`)
- query count
- layer-depth schedule
- domain / twiddle planning handles

Candidate types:

```rust
pub enum OpeningMode {
    Plain,
    Zk,
}

pub struct OraclePlan {
    pub log_msg_len: usize,
}

pub struct FriPlan<F> {
    pub log_batch_size: usize,
    pub fold_arities: Vec<usize>,
    pub index_bits: usize,
    pub log_inv_rate: usize,
    pub n_final_challenges: usize,
    pub n_test_queries: usize,
    pub opening_mode: OpeningMode,
    pub _marker: std::marker::PhantomData<F>,
}

pub struct PcsPlan<F> {
    pub oracle: OraclePlan,
    pub fri: FriPlan<F>,
}
```

### Layer 2: Pure Interactive Core

This layer remains the semantic core for the algebraic reductions.

Primary modules that should remain conceptually here:

- `crates/verifier/src/and_reduction/verifier.rs`
- `crates/verifier/src/protocols/intmul/verify.rs`
- `crates/verifier/src/protocols/shift/verify.rs`
- `crates/verifier/src/ring_switch.rs`

No transcript or Merkle types should leak inward here.

The existing `IPVerifierChannel` from `crates/ip/src/channel.rs` is the right
seed abstraction for this layer.

### Layer 3: PCS Semantic Core

This is the missing layer today.

Proposed modules:

- `crates/verifier/src/pcs/mod.rs`
- `crates/verifier/src/pcs/relation.rs`
- `crates/verifier/src/pcs/opening.rs`
- `crates/verifier/src/pcs/consistency.rs`

This layer should explicitly model:

- what RingSwitch hands to the PCS
- what BaseFold / FRI returns
- how the final transparent relation is checked

Candidate types:

```rust
pub struct QueryPoint<F> {
    pub coords_low_to_high: Vec<F>,
}

pub struct FoldChallenges<F> {
    pub coords_high_to_low: Vec<F>,
}

pub struct ZkFoldChallenges<F> {
    pub batch_challenge: F,
    pub fold_coords_high_to_low: Vec<F>,
}

pub enum OpeningChallenges<F> {
    Plain(FoldChallenges<F>),
    Zk(ZkFoldChallenges<F>),
}

pub struct BaseFoldOpeningOutput<F> {
    pub final_fri_value: F,
    pub final_sumcheck_value: F,
    pub query_point: QueryPoint<F>,
    pub challenges: OpeningChallenges<F>,
}

pub struct RingSwitchClaim<F> {
    pub sumcheck_claim: F,
    pub relation: TransparentRelation<F>,
}

pub struct PcsOpeningOutput<F> {
    pub ring_switch_claim: RingSwitchClaim<F>,
    pub opening: BaseFoldOpeningOutput<F>,
    pub transparent_eval: F,
}
```

### Layer 4: PCS Interpreters

This layer implements the same semantics in different ways.

Proposed modules:

- `crates/verifier/src/pcs/native.rs`
- `crates/verifier/src/pcs/replay.rs`
- `crates/verifier/src/pcs/symbolic.rs`
- `crates/verifier/src/pcs/size_tracking.rs`

The important design rule is:

- native streaming interpreter may use borrowed decoded data and scratch buffers
- replay interpreter may use owned plain-data objects
- symbolic interpreter may use symbolic values / circuit builder handles

But all three should feed the same PCS semantic core.

### Layer 5: Transport / Fiat-Shamir / Merkle

This layer owns proof transport and authentication.

Proposed modules:

- `crates/iop/src/fri/proof.rs`
- `crates/iop/src/fri/parser.rs`
- `crates/iop/src/fri/native_transport.rs`
- `crates/iop/src/merkle_tree/native_verify.rs`

This layer should be the only place that directly knows about:

- `VerifierTranscript`
- `TranscriptReader<B>`
- `Buf`
- byte-level proof parsing
- Merkle proof verification details

## New Core Traits

### 1. Keep a Minimal Public-Coin Core

The existing shape of `IPVerifierChannel` is good and should remain the base
public-coin protocol abstraction.

### 2. Replace `IOPVerifierChannel::verify_oracle_relations`

Current problem:

- one call hides too many semantic stages
- boxed closures erase relation identity
- challenge-point semantics are interpreter-specific

Proposed replacement:

```rust
pub trait OracleCommitmentReceiver<F> {
    type Oracle;

    fn recv_oracle(&mut self, spec: &OraclePlan) -> Result<Self::Oracle, Error>;
}

pub trait PcsOpeningVerifier<F> {
    type Oracle;
    type ProofView<'a>
    where
        Self: 'a;

    fn verify_opening<'a>(
        &mut self,
        plan: &PcsPlan<F>,
        oracle: &Self::Oracle,
        claim: F,
        proof: Self::ProofView<'a>,
    ) -> Result<BaseFoldOpeningOutput<F>, Error>;
}
```

This keeps the native interpreter free to stream/decode/authenticate however it
wants, while exposing a typed semantic result to the rest of the verifier.

### 3. Replace `TransparentEvalFn`

Current issue:

- `Box<dyn Fn(&[Elem]) -> Elem>` is opaque to planners, symbolic builders, and extractors

Proposed replacement:

```rust
pub enum TransparentRelation<F> {
    RingSwitchEq {
        eval_point_high: Vec<F>,
        eq_r_double_prime: Vec<F>,
    },
    SpartanMask {
        // future extension
    },
}

impl<F: BinaryField> TransparentRelation<F> {
    pub fn eval(&self, point: &QueryPoint<F>) -> F {
        match self {
            TransparentRelation::RingSwitchEq {
                eval_point_high,
                eq_r_double_prime,
            } => crate::ring_switch::eval_rs_eq(
                eval_point_high,
                &point.coords_low_to_high,
                eq_r_double_prime,
            ),
            TransparentRelation::SpartanMask { .. } => todo!(),
        }
    }
}
```

This gives:

- inspectable relation identity
- typed arity/domain meaning
- explicit extensibility

### 4. Introduce Typed Proof Views

The council strongly converged on the parse seam around FRI opening
verification.

`verify_coset_opening` in `crates/iop/src/fri/verify.rs` is the clean cut
between:

- commitment authentication
- arithmetic fold verification

So the medium-term target should be typed proof views like:

```rust
pub struct VerifiedLayer<D> {
    pub root: D,
    pub layer_depth: usize,
    pub digests: Vec<D>,
}

pub struct VerifiedCosetOpening<F, D> {
    pub coset_index: usize,
    pub values: Vec<F>,
    pub tree_depth: usize,
    pub layer_depth: usize,
    pub layer: VerifiedLayer<D>,
}

pub struct BaseFoldProofView<F, D> {
    pub round_coeffs: Vec<[F; 2]>,
    pub round_commitments: Vec<D>,
    pub terminal_codeword: Vec<F>,
    pub layers: Vec<VerifiedLayer<D>>,
    pub queries: Vec<Vec<VerifiedCosetOpening<F, D>>>,
}
```

The native interpreter does not need to store these permanently. It can decode
and verify them on the fly, then feed borrowed or short-lived typed values into
the semantic core.

## Proposed Module Layout

### New modules to add

- `crates/verifier/src/pcs/mod.rs`
- `crates/verifier/src/pcs/plan.rs`
- `crates/verifier/src/pcs/relation.rs`
- `crates/verifier/src/pcs/opening.rs`
- `crates/verifier/src/pcs/consistency.rs`
- `crates/verifier/src/pcs/native.rs`
- `crates/verifier/src/pcs/replay.rs`
- `crates/verifier/src/pcs/symbolic.rs`
- `crates/iop/src/fri/proof.rs`
- `crates/iop/src/fri/parser.rs`

### Existing files that should shrink over time

- `crates/verifier/src/verify.rs`
- `crates/iop/src/channel.rs`
- `crates/iop/src/basefold_channel.rs`
- `crates/iop/src/basefold_zk_channel.rs`
- `crates/iop/src/basefold.rs`
- `crates/iop/src/fri/verify.rs`

### Existing files that should remain semantic anchors

- `crates/verifier/src/ring_switch.rs`
- `crates/verifier/src/protocols/*/verify.rs`
- `crates/ip/src/channel.rs`

## Phase-by-Phase Cutover

### Phase 0: Freeze the Current Extraction Seam and Add Missing Gates

Goal:

- keep current extraction/replay success
- add missing parity and perf guardrails before larger changes

Files to add / modify:

- `crates/prover/tests/pcs_extract_parity.rs`
- new direct BaseFold parity test, likely `crates/iop/tests/basefold_extract_parity.rs`
- new microbench, likely under `crates/iop/benches/`

Acceptance:

- current `pcs_extract_parity` still passes
- new direct BaseFold parity passes
- malformed-opening / malformed-fold cases are covered
- microbench baseline exists for FRI verifier path

### Phase 1: Make the PCS Slice Explicit at the Top Level

Goal:

- remove the semantic opacity of `verify_oracle_relations` without changing the
  transport implementation yet

Files to add:

- `crates/verifier/src/pcs/mod.rs`
- `crates/verifier/src/pcs/opening.rs`
- `crates/verifier/src/pcs/consistency.rs`

Files to modify:

- `crates/verifier/src/verify.rs`
- `crates/verifier/src/lib.rs`

Concrete cut:

- introduce `pcs::verify_opening`
- make `verify.rs` call it explicitly after `ring_switch::verify`
- keep `pcs::verify_opening` internally delegating to current BaseFold/native path

Acceptance:

- no behavior change
- `verify.rs` explicitly composes RingSwitch, BaseFold opening, and final consistency
- no more boxed closure construction at the top-level verifier call site

### Phase 2: Type the Relation and Challenge Surface

Goal:

- remove closure-based and raw-`Vec` semantics from the PCS interface

Files to add:

- `crates/verifier/src/pcs/relation.rs`

Files to modify:

- `crates/verifier/src/ring_switch.rs`
- `crates/verifier/src/verify.rs`
- `crates/iop/src/basefold.rs`
- `crates/iop/src/basefold_channel.rs`
- `crates/iop/src/basefold_zk_channel.rs`

Concrete cut:

- `ring_switch::verify` returns `sumcheck_claim + TransparentRelation`
- BaseFold returns typed challenge/query-point output instead of raw `Vec<F>`
- eliminate manual `reverse()` / `challenges[1..]` semantics outside one canonical constructor

Acceptance:

- all challenge-order conversions happen in one typed place
- no `TransparentEvalFn` at the PCS seam
- replay and native paths produce the same typed query-point object

### Phase 3: Split `IOPVerifierChannel`

Goal:

- remove `verify_oracle_relations` as a trait-level semantic operation

Files to add:

- `crates/iop/src/opening.rs` or `crates/verifier/src/pcs/native.rs`

Files to modify:

- `crates/iop/src/channel.rs`
- `crates/iop/src/basefold_channel.rs`
- `crates/iop/src/basefold_zk_channel.rs`
- `crates/iop/src/size_tracking_channel.rs`

Concrete cut:

- keep `recv_oracle`
- add typed opening verification interface
- remove top-level relation checking from channel implementations

Acceptance:

- native, size-tracking, and symbolic/replay backends all implement the new split interface
- no channel backend is responsible for evaluating the transparent relation itself

### Phase 4: Split BaseFold / FRI Parser from Arithmetic Semantics

Goal:

- move transcript/Merkle transport out of the semantic verifier

Files to add:

- `crates/iop/src/fri/proof.rs`
- `crates/iop/src/fri/parser.rs`
- possibly `crates/iop/src/basefold/proof.rs`

Files to modify:

- `crates/iop/src/basefold.rs`
- `crates/iop/src/fri/verify.rs`
- `crates/iop/src/merkle_tree/*`

Concrete cut:

- parser/auth layer yields typed proof sections / openings
- arithmetic verifier consumes typed openings and borrowed decoded slices
- `verify_coset_opening` becomes parser/auth-side

Acceptance:

- arithmetic FRI verification can run without direct access to `VerifierTranscript` or `Buf`
- native interpreter still streams and reuses scratch buffers
- no meaningful verifier regression in the microbench

### Phase 5: Collapse Replay Extraction onto Shared Semantics

Goal:

- stop duplicating verifier semantics in extraction-only modules

Files to modify:

- `crates/iop/src/basefold_extract.rs`
- `crates/verifier/src/ring_switch_extract.rs`
- `crates/verifier/src/pcs_extract.rs`

Concrete cut:

- replay modules become interpreter adapters over the shared semantic core
- keep plain-data proof objects, but remove duplicated arithmetic logic

Acceptance:

- replay parity still passes
- `hax` and `aeneas` still succeed
- extraction modules become thin wrappers rather than semantic forks

### Phase 6: Add the Symbolic Interpreter

Goal:

- produce a real symbolic verifier backend for circuit compilation / recursion

Files to add:

- `crates/verifier/src/pcs/symbolic.rs`
- possibly a new support crate if symbolic values become large enough to deserve one

Concrete cut:

- implement the PCS semantic interface over symbolic scalar/value handles
- keep Merkle / FS outside this layer
- support the same typed relation and opening-result surface as native/replay

Acceptance:

- symbolic backend can run the PCS semantic core
- no changes needed in top-level semantic composition
- extraction and symbolic backends share the same typed relation / challenge model

## Performance Guardrails

The symbolic refactor should not force the native path into a replay-style shape.

Hard rules:

- keep `fold_chunk` and `fold_interleaved_chunk` native and optimized
- keep transcript decoding and Merkle authentication in native wrappers
- use borrowed slices plus caller-owned scratch in the native interpreter
- do not redesign the core around owned `Vec` openings

Bench gates to add and maintain:

- crate-local microbench for `FRIQueryVerifier::verify` or `verify_query`
- end-to-end verifier benchmark using the existing example bench framework
- peak-memory reporting for verifier-heavy workloads

## Testing Strategy

### Required parity boundaries

- direct `ring_switch` generic vs replay parity
- direct `basefold` live transcript vs replay parity
- composed `pcs` live transcript vs replay parity

### Required negative tests

- malformed Merkle vector
- malformed Merkle opening
- malformed Merkle layer
- incorrect fold value
- incorrect final consistency

### Required extraction gates

- `./scripts/check_extraction.sh hax`
- `./scripts/check_extraction.sh aeneas`

These should remain green across every phase after Phase 0.

## Recommended First Concrete Step

The first high-value prototype is:

1. add `crates/verifier/src/pcs/mod.rs`
2. move the current replay-shape composition from `crates/verifier/src/pcs_extract.rs`
   into a shared semantic `pcs::verify_opening`
3. make `crates/verifier/src/verify.rs` call that explicit PCS layer
4. keep the implementation backed by the current native BaseFold path at first

If that lands cleanly, the rest of the roadmap becomes much lower risk.

## Non-Goals for the First Iteration

- generic symbolic fields across the whole repo
- full refactor of every transcript user
- rearchitecting all Merkle code before the PCS seam is fixed
- immediate deletion of the replay extraction path

The first iterations should be about getting the boundary right, not finishing
the whole refactor in one leap.
