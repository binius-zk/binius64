# Contributing to Binius64

## Style Guide & Conventions

Many code formatting and style rules are enforced using
[rustfmt](https://doc.rust-lang.org/book/appendix-04-useful-development-tools.html#automatic-formatting-with-rustfmt)
and [Clippy](https://doc.rust-lang.org/clippy/). The remaining sections document conventions that cannot be enforced
with automated tooling.

### Running automated checks

You can run the formatter and linter with

```bash
$ cargo fmt
$ cargo clippy --all --all-features --tests --benches --examples -- -D warnings
```

[Pre-commit](https://pre-commit.com/) hooks are configured to run `rustfmt`. The codebase is formatted with a nightly
version of `cargo fmt` because stable doesn't support all of the rustfmt options we use. To run it, you can use:

```bash
$ pre-commit run rustfmt --all-files
$ cargo +nightly-2026-01-01 fmt  # see .pre-commit-config.yaml for the exact nightly version checked by CI
```

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

## First-time contributions

The project welcomes first time contributions from developers who want to learn more about Binius64 and make an impact
on the open source cryptography community.

If you are new to the project and don't know where to start, you can look for [open issues labeled
`good first issue`](https://github.com/IrreducibleOSS/binius/issues?q=is%3Aissue+state%3Aopen+label%3A%22good+first+issue%22) or add
test coverage for existing code. Adding unit tests is a great way to learn how to interact with the codebase, make a
meaningful contribution, and maybe even find bugs!

On the other hand, _we do not accept typo fix PRs from first-time contributors_. These are not significant enough to
justify the additional work for maintainers nor any potential benefits, tangible or intangible, one might get from
being listed as a contributor to the repo.
