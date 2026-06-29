# Contributing to Binius64

## Copyright

All source files should include a copyright header. New files should start with `// Copyright YYYY The Binius Developers`, where YYYY is the current year. When modifying an existing file, add the copyright line if one referencing "The Binius Developers" is not already present.

## Style Guide & Conventions

Many code formatting and style rules are enforced using
[rustfmt](https://doc.rust-lang.org/book/appendix-04-useful-development-tools.html#automatic-formatting-with-rustfmt)
and [Clippy](https://doc.rust-lang.org/clippy/). The remaining sections document conventions that cannot be enforced
with automated tooling.

### Running automated checks

The codebase is formatted with a nightly version of `cargo fmt` because stable doesn't support all of the rustfmt
options we use. You can run the formatter and linter with

```bash
$ cargo +nightly-2026-01-01 fmt  # see prek.toml for the exact nightly version checked by CI
$ cargo clippy --all --all-features --tests --benches --examples -- -D warnings
```

[prek](https://prek.j178.dev/) hooks are configured to run `rustfmt`. You can also invoke it via prek:

```bash
$ prek run rustfmt --all-files
```

### Cross-compilation

`binius-field` and `binius-arith-bench` contain architecture-specific optimizations: CLMUL/SIMD
implementations of `GF(2^128)` (and related) arithmetic, selected at compile time with
`#[cfg(target_arch = ...)]` and `#[cfg(target_feature = ...)]`. Code on an *inactive* arch/feature
path is never type-checked by your native build, so it is easy to break the `aarch64` paths from an
`x86_64` host (or vice versa) and not notice until CI fails — CI builds `x86_64` (both portable and
`-Ctarget-cpu=native`), `aarch64`, and `wasm32`.

When you touch these crates, cross-compile them for the target(s) you are not running natively.
You do **not** need an emulator — compiling is enough to type-check the inactive paths.

> **The optimized paths are gated behind target features that are off in the baseline target**
> (e.g. `aes`/PMULL on `aarch64`, `pclmulqdq` on `x86_64`). A cross-build with *default* features
> compiles only the portable fallback, which gives false confidence. Enable the features (via the
> `RUSTFLAGS` below) to actually type-check the optimized code. On your native arch,
> `-Ctarget-cpu=native` does the same thing.

One-time setup (targets are added to the pinned toolchain in `rust-toolchain.toml`):

```bash
rustup target add aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu wasm32-wasip1 wasm32-unknown-unknown

# Only needed to *link* aarch64 test/bench binaries (`--all-targets`) or crates with C build
# dependencies. `cargo check` and a plain library `cargo build` do not link, so they don't need it.
sudo apt-get install -y gcc-aarch64-linux-gnu
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
```

Compile the architecture-specific crates for the non-native arch and for wasm:

```bash
# aarch64 — +neon,+aes enables the PMULL/CLMUL GHASH paths (not just the portable fallback)
RUSTFLAGS="-C target-feature=+neon,+aes" \
  cargo check --target aarch64-unknown-linux-gnu -p binius-field -p binius-arith-bench

# x86_64 — a consistent SIMD+CLMUL set (avx2 is required for the 256-bit vpclmulqdq paths;
# add +avx512f for the 512-bit path). On an x86_64 host, `-C target-cpu=native` is simpler.
RUSTFLAGS="-C target-feature=+sse2,+avx2,+pclmulqdq,+vpclmulqdq" \
  cargo check --target x86_64-unknown-linux-gnu -p binius-field -p binius-arith-bench

# wasm32 — matches CI (binius-field on wasm32-unknown-unknown; the wider crate set on wasm32-wasip1)
cargo build -p binius-field --target wasm32-unknown-unknown
cargo build -p binius-field --target wasm32-wasip1
```

(All four commands above are verified to compile cleanly. The 512-bit AVX-512 path —
`+sse2,+avx2,+avx512f,+pclmulqdq,+vpclmulqdq` — also compiles cleanly, including
`cargo build --all-targets` and a full `--workspace` build, even on a host without AVX-512:
its `std::arch::x86_64::_mm512_*` intrinsics are stable on the pinned toolchain. Older Rust,
where those intrinsics were unstable, rejects this build.)

`cargo check` is the fast type-check of the library paths. To lint tests and benches the way CI
does, swap in `cargo clippy --target <triple> -p binius-field -p binius-arith-bench --all-targets
-- -D warnings` (this links, so it needs the cross C toolchain above).

### Documentation

We follow guidance from the [rustdoc book](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html). The
["Documenting components"](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html#documenting-components)
section is quite prescriptive. To copy verbatim:

> It is recommended that each item's documentation follows this basic structure:
>
> ```
> [short sentence explaining what it is]
>
> [more detailed explanation]
>
> [at least one code example that users can copy/paste to try it]
>
> [even more advanced explanations if necessary]
> ```

Documentation and commit messages should be written in the present tense. For example,

```
❌ This function will return the right answer
✅ This function returns the right answer

❌ Fixed the bug in the gizmo
✅ Fix the bug in the gizmo
```

#### Module-level documentation

Every crate must have module-level documentation in `lib.rs` using `//!` comments. This documentation should include:

1. **One sentence summary**: What this crate does
2. **When to use**: In what situations should someone reach for this crate
3. **Key types**: Brief list of the main types/traits with one-line descriptions
4. **Usage example**: A minimal working example (where practical)
5. **Related crates**: How this crate relates to others in the workspace

All crates should enable `#![warn(missing_crate_level_docs)]` to ensure crate-level documentation exists.

Example crates with good module documentation: `binius-field`, `binius-frontend`, `binius-spartan-frontend`.

#### When to add examples

Include code examples for:
- Public functions that are part of the main API
- Types that users will construct directly
- Non-obvious behavior or edge cases

Examples in documentation are tested by `cargo test --doc`, so they also serve as regression tests.

### Naming philosophy & conventions

This codebase biases towards longer, more descriptive names for identifiers. This extends to the names of generic type
parameters.

#### Generic parameter names

In idiomatic Rust code, generic parameters are often identified by single letters or short, capitalized abbreviations.
We tend to prefer more descriptive, CamelCase identifiers for type parameters, especially for methods that have more
than one or two type parameters. There are some exceptions for common type parameters that have single-letter
abbreviations. They are:

* `F` indicates a `Field` parameter
* `P` indicates a `PackedField` parameter

If a function or struct is generic over multiple types implementing those traits, the type names should start with the
single-letter abbreviation. For example, a function that is parameterized by multiple fields may name them
`F`, `FSub`, `FDomain`, `FEncode`, etc., where `FSub` is a subfield of `F`, `FDomain` is a field used as an evaluation
domain, and `FEncode` is used as the field of an encoding matrix.

#### Use namespacing

If an identifier is defined in a module and is unambiguous in the context of that module, it does _not_ need to
duplicate the module name into the identifier. For example, we have many protocols defined in `binius_core::protocols`
that expose a `prove` and `verify` method. Because they are namespaced within the protocol modules, for example the
`sumcheck` module, these identifiers do not need to be named `sumcheck_verify` and `sumcheck_prove`. The caller has the
option of referring to these functions as `sumcheck::prove` / `sumcheck::verify` or renaming the imported symbol, like
`use sumcheck::prove as sumcheck_prove`.

### Functional programming style

Prefer functional style over imperative style with mutable variables. Use iterator combinators (`map`, `filter`, `fold`,
etc.) instead of loops with mutable state. Exceptions are allowed when algorithms have substantial mutable state that
would be awkward to express functionally.

Good examples:
```rust
// Use fold instead of mutable accumulator
let result = items.iter().fold(initial, |acc, &item| {
    compute_next(acc, item)
});

// Use iter::zip instead of .iter().zip()
let pairs: Vec<_> = iter::zip(&vec_a, &vec_b)
    .map(|(&a, &b)| process(a, b))
    .collect();

// Use successors for generating sequences
let powers = iter::successors(Some(x), |&prev| Some(prev * x))
    .take(n)
    .collect();
```

Poor examples:
```rust
// Avoid mutable state when fold is clearer
let mut result = initial;
for &item in items.iter() {
    result = compute_next(result, item);
}

// Avoid .iter().zip() - use iter::zip() instead
let pairs: Vec<_> = vec_a.iter()
    .zip(&vec_b)
    .map(|(&a, &b)| process(a, b))
    .collect();

// Avoid imperative loops for sequence generation
let mut powers = Vec::new();
let mut current = x;
for _ in 0..n {
    powers.push(current);
    current = current * x;
}
```

### Unwrap

Don't call `unwrap` in library code. Either throw or propagate an `Err` or call `expect`, leaving an explanation of why
the code will not panic. Unwrap is fine in test code.

Example from the [Substrate style guide](https://github.com/paritytech/substrate/blob/master/docs/STYLE_GUIDE.md#style):

```rust
let mut target_path =
	self.path().expect(
		"self is instance of DiskDirectory;\
		DiskDirectory always returns path;\
		qed"
	);
```

### Turbofish over type annotations

Prefer turbofish (`::<...>`) to resolve type ambiguities rather than adding type annotations on local variables. This
keeps the type information at the call site where it's needed.

```rust
// Good
let vals = elems.iter().map(|e| e.value()).collect::<Vec<_>>();

// Bad
let vals: Vec<_> = elems.iter().map(|e| e.value()).collect();
```

### Visibility scope

Use `pub(crate)` and `pub(super)` sparingly; instead, prefer to control visibility at the ancestor module level. Most of
the time, an item that should not be part of the public API can simply be `pub` within a non-`pub` module: keeping the
module (or one of its ancestors) private already prevents external reachability, and you avoid annotating every item.
Only reach for `pub(crate)` and `pub(super)` when ancestor-level visibility cannot express what you need:

* A `pub mod` that exposes some items publicly but has helpers that other crate modules must call. Here the module
  cannot be made private, so `pub(crate)` is the right tool for the internal helpers.
* A field of a `pub` struct that crate code must access but external code must not. Fields have no ancestor-module
  escape hatch, so `pub(crate)` (or `pub(super)`) is the only option.

```rust
// Good: a private module gates external reachability; no per-item modifiers needed.
mod internal {
    pub fn helper() {}
}

// Good: the modifier is necessary because the module is public.
pub mod codec {
    pub fn encode() {}          // public API
    pub(crate) fn scratch() {}  // crate-internal helper
}

// Good: fields have no module-level alternative.
pub struct Session {
    pub id: SessionId,
    pub(crate) refcount: usize,
}
```

## Prover-verifier separation

Verifier code is optimized for simplicity, security, and readability, whereas prover code is optimized for performance.
This naturally means there are different conventions and standards for verifier and prover code. Some notable
differences are

* Prover code often uses packed fields; verifier code should only use scalar fields.
* Prover code often uses subfields; verifier code should primarily use a single field.
* Prover code often uses Rayon for multithreaded parallelization; verifier code should not use Rayon.
* Prover code can use complex data structures like hash maps; verifier code prefer direct-mapped indexes.
* Prover code can use more experimental dependencies; verifier code should be conservative with dependencies.

## Error Handling

The codebase uses different error-handling strategies depending on the abstraction level and trust boundary.

### Precondition Contracts

Most internal code should use **assertions and documented precondition contracts** rather than returning `Result` types.
This simplifies internal APIs and makes code easier to reason about. Functions should document their preconditions and
use `assert!`, `debug_assert!`, or `expect()` to enforce them.

```rust
/// Evaluates the polynomial at the given point.
///
/// # Preconditions
/// - `point.len()` must equal the number of variables in the polynomial
fn evaluate(&self, point: &[F]) -> F {
    assert_eq!(point.len(), self.n_vars());
    // ...
}
```

### When to Return Errors

Errors should be returned when **unchecked external input could cause a panic**. The key distinction is:

- **Verifier**: The high-level input is the proof. The verifier cannot trust the proof, so it must return
  `VerificationError` for invalid proofs rather than panicking. This is the boundary where untrusted data enters.

- **Prover**: The high-level input is the witness. The prover **may assume the witness is satisfying**. If the witness
  is invalid, the prover code may panic. This is acceptable because the caller is responsible for providing a valid
  witness.

### Trust Boundaries

Error types should only be used at high-level interfaces:

- `binius_prover::Prover` - returns errors only for system-level failures (not invalid witnesses)
- `binius_verifier::Verifier` - returns `VerificationError` for invalid proofs
- Similar interfaces in spartan modules

Below these interfaces, code should use precondition contracts. This keeps internal APIs simple and pushes validation
to the boundaries where untrusted data enters the system.

## Dependencies

We use plenty of useful crates from the Rust ecosystem. When including a crate as a dependency, be sure to assess:

* Is it widely used? You can see when it was published and total downloads on `crates.io`.
* Is it maintained? If the documentation has an explicit deprecation notice or has not been updated in a long time, try
  to find an alternative.
* Is it developed by one person or an organization?
