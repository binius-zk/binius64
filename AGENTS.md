# AGENTS.md

Quick-start context for AI agents and developers working with Binius64.

## Build Commands

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test                     # Run tests
cargo test -p <crate>          # Test specific crate
cargo fmt                      # Format code
cargo clippy --all --all-features --tests --benches --examples -- -D warnings
typos                          # Check for typos
pre-commit run --all-files     # Run all checks
```

For optimal performance: `export RUSTFLAGS="-C target-cpu=native"`

## Key Terminology

| Term | Definition |
|------|------------|
| **Shifted value index** | Tuple `(value_id, shift_op, shift_amount)` - references a witness word with an optional shift |
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
[CONTRIBUTING.md](CONTRIBUTING.md) covers code style, naming conventions, copyright headers, error handling, and other development conventions.

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
