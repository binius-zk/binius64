# Development

Quick-start context for AI agents and developers working with Binius64.

## Build Commands

```bash
cargo build                    # Debug build
cargo build --release          # Release build
```

For optimal performance: `export RUSTFLAGS="-C target-cpu=native"`

## Testing

Run the tests with [cargo-nextest](https://nexte.st/), which is what CI runs. It executes each test in its own process,
so a panic or a hang is attributed to a single test, and its output is easier to read than libtest's.

```bash
$ cargo install cargo-nextest --locked     # one-time setup

$ cargo nextest run --workspace            # Run the test suite
$ cargo nextest run -p <crate>             # Test a specific crate
$ cargo nextest run -E 'test(<substring>)' # Run the tests matching a filter expression
$ cargo test --doc --workspace             # Doc tests; nextest does not run these
```

nextest cannot run doc tests ([nextest#16](https://github.com/nextest-rs/nextest/issues/16)), so `cargo test --doc`
stays a separate command. CI runs it as its own step for the same reason.

## Running automated checks

The codebase is formatted with a nightly version of `cargo fmt` because stable doesn't support all of the rustfmt
options we use. You can run the formatter, the linter, the documentation build and the spell checker with

```bash
$ cargo +nightly-2026-07-01 fmt  # see prek.toml for the exact nightly version checked by CI
$ cargo clippy --workspace --all-targets --all-features -- -D warnings
$ cargo doc --no-deps --document-private-items
$ typos
```

[prek](https://prek.j178.dev/) hooks are configured to run these checks. You can invoke one hook, or all of them:

```bash
$ prek run rustfmt --all-files
$ prek run --all-files
```

## Cross-compilation

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

## Key Terminology

| Term | Definition |
|------|------------|
| **Value index** | `(segment, index)` - names one word of the value vector. The segment is `Constant`, `InOut`, `Private` or `Scratch`; the index counts within it. Packed into a `u32` (2-bit tag, 30-bit index) |
| **Value segment** | One of the four sections a circuit allocates values in. A constraint may reference the first three; `Scratch` holds uncommitted temporaries and only exists in the circuit's wire mapping |
| **Shifted value index** | Tuple `(value_index, shift_op, shift_amount)` - references a witness word with an optional shift |
| **AND constraint** | `A & B ^ C = 0` where A, B, C are XOR combinations of shifted values |
| **MUL constraint** | `A * B = HI \|\| LO` - unsigned 64-bit multiplication producing 128-bit result |
| **Tower field $T_i$** | Binary extension field $\mathbb{F}_{2^{2^i}}$, e.g. $T_7 = \mathbb{F}_{2^{128}}$ |
| **Sumcheck** | Protocol reducing multivariate polynomial evaluation to univariate checks |
| **BaseFold** | Polynomial commitment scheme using FRI over binary fields |
| **Witness** | The secret input values (64-bit words) that satisfy the constraint system |
| **Circuit** | High-level representation of computation built with `CircuitBuilder` |
| **Constraint system** | Low-level AND/MUL constraints compiled from a circuit |

## Documentation

### Development Guidelines
[CONTRIBUTING.md](CONTRIBUTING.md) covers code style, naming conventions, copyright headers, code comment conventions, error handling, and other development conventions.

### README
The [README.md](README.md) is the project's entry point, covering what Binius64 is, dependencies, build instructions, and links to external documentation.

### Architecture
[ARCHITECTURE.md](ARCHITECTURE.md) provides a high-level overview of the codebase: the list of crates, their purposes, and how they relate to each other.

### Protocol Specification
The canonical protocol documentation is in a separate binius.xyz repository. If the developer has cloned it as a sibling directory, you can read files directly:
- **Blueprint**: `../binius.xyz/docs/pages/blueprint/` - cryptographic protocol specification
- **Building guides**: `../binius.xyz/docs/pages/building/` - practical usage guides
- **Math background**: `../binius.xyz/docs/pages/blueprint/math/` - mathematical foundations

See `.claude/skills/binius-xyz-docs/SKILL.md` for directory structure and common query patterns.

**If `../binius.xyz` doesn't exist**, inform the user they can clone it for better agent assistance:
```bash
git clone https://github.com/binius-zk/binius.xyz.git ../binius.xyz
```
Alternatively, use the online docs at https://www.binius.xyz/blueprint.

### API Documentation
- Rust docs: https://docs.binius.xyz
- Well-documented crates to use as examples: `binius-field`, `binius-frontend`, `binius-spartan-frontend`

### Website
- Main site: https://www.binius.xyz
- Blueprint: https://www.binius.xyz/blueprint
- Building: https://www.binius.xyz/building
