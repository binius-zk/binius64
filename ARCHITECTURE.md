# Architecture

This document describes the high-level architecture of the Binius64 repository.

## Proof Systems

This repository implements two separate proof systems that share common dependencies but prove different constraint system relations:

| System | Crates | Zero-Knowledge | Constraint Type |
|--------|--------|----------------|-----------------|
| Binius64 | binius-prover, binius-verifier | No | AND/MUL on 64-bit words |
| Iron Spartan | binius-spartan-prover, binius-spartan-verifier | Yes | Multiplication over GF(2^128) |

### Binius64

Binius64 proves satisfiability of constraint systems over 64-bit words. The witness is an array of 64-bit words, and constraints reference these words using *shifted value indices*—tuples combining a word index, a shift operation (logical left, logical right, or arithmetic right), and a shift amount.

**AND constraints** assert bitwise AND relations:
```
(w[i] << s₁ ⊕ w[j] >> s₂ ⊕ ...) & (w[k] ⊕ ...) = (w[m] ~>> s₃ ⊕ ...)
```

**MUL constraints** assert 64-bit unsigned integer multiplication:
```
p * q = 2^64 * hi + lo
```
where `p`, `q`, `hi`, and `lo` are each XOR-sums of shifted witness values.

This design achieves a 64-fold reduction in constraint complexity compared to bit-level approaches. XOR provides free linear combinations, while AND and MUL are the non-linear operations.

For the detailed protocol specification, see the [Blueprint](https://www.binius.xyz/blueprint) documentation.

### Iron Spartan

Iron Spartan is a zero-knowledge proof system using multiplication constraints over the GHASH binary field GF(2^128). This is analogous to R1CS but adapted for binary fields:

- Constraints have the form `A * B = C`
- Operands A, B, C are XOR-sums of witness field elements
- The sparse constraint matrices have only 0 and 1 entries (no arbitrary field coefficients)

The use of binary field arithmetic and XOR-based linear combinations makes this well-suited for binary computations while maintaining zero-knowledge properties.

## Crate Organization

### Shared Crates

| Crate | Purpose |
|-------|---------|
| binius-field | Binary field arithmetic with architecture-specific optimizations |
| binius-math | Mathematical primitives (multilinear polynomials, FFT, Reed-Solomon) |
| binius-hash | Hash functions (SHA-256, Vision) and compression functions |
| binius-transcript | Fiat-Shamir transcript handling for non-interactive proofs |
| binius-utils | Common utilities (serialization, bitwise operations, arrays) |

### Frontend Crates

| Crate | Purpose |
|-------|---------|
| binius-core | Constraint system data structures shared by prover and verifier |
| binius-frontend | Circuit construction API for Binius64 (CircuitBuilder, wires, witness) |
| binius-circuits | Standard library of circuit gadgets (SHA256, ECDSA, base64, etc.) |
| binius-spartan-frontend | Constraint system builder for Iron Spartan |

### Verifier Crates

| Crate | Purpose |
|-------|---------|
| binius-ip | Interactive polynomial protocol structures (sumcheck, prodcheck) |
| binius-iop | Interactive oracle protocol structures (BaseFold, FRI, Merkle trees) |
| binius-verifier | High-level Binius64 proof verification API |
| binius-spartan-verifier | Iron Spartan proof verification |

### Prover Crates

| Crate | Purpose |
|-------|---------|
| binius-ip-prover | IP prover implementations |
| binius-iop-prover | IOP prover implementations |
| binius-prover | High-level Binius64 proof generation API |
| binius-spartan-prover | Iron Spartan proof generation |

### Examples and Benchmarks

| Crate | Purpose |
|-------|---------|
| binius-examples | Example circuits with CLI framework |
| binius-arith-bench | Arithmetic microbenchmarks |

## Design Principles

### Dependency Direction

**Prover crates depend on verifier crates, not vice versa.** This allows:
- Verifiers to be deployed independently
- Provers to reuse verifier data structures for consistency
- Clear separation of security-critical code

**Frontend crates must not depend on backend (prover/verifier) crates.** Circuit definitions should be independent of the proving system.

### Verifier vs Prover Priorities

**Verifier crates** prioritize security, auditability, and clarity:
- Use only scalar fields (not packed fields)
- Avoid parallelization
- Prefer simple data structures
- Optimize for readability over performance

**Prover crates** prioritize performance:
- Use packed fields and SIMD optimizations
- Leverage parallelization (Rayon)
- May sacrifice code clarity for efficiency

### Test Location

Most tests live in prover crates rather than verifier crates. This is a natural consequence of the dependency order—prover crates can test the full prove-then-verify flow, while verifier crates cannot generate proofs to test against.

## Further Reading

- [binius.xyz](https://www.binius.xyz) - Documentation website
- [docs.binius.xyz](https://docs.binius.xyz) - Rust API documentation
- Individual crate documentation via `cargo doc`
