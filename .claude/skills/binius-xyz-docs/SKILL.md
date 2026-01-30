---
name: binius-xyz-docs
description: Access protocol documentation from binius.xyz repository
allowed-tools: Read, Glob, Grep
---

# Binius.xyz Documentation Access

The canonical Binius64 protocol documentation is in a sibling repository at `../binius.xyz/docs/pages/`.

## Prerequisites

Before reading documentation, verify the repo exists by attempting to read a file (e.g., `../binius.xyz/docs/pages/blueprint/overview.mdx`). If the read fails, the repo is not available - see AGENTS.md for setup instructions or use the online docs at https://www.binius.xyz/blueprint.

## Directory Structure

```
../binius.xyz/docs/pages/
├── basics/              # High-level introduction and concepts
│   ├── concepts.mdx     # SNARK concepts, why Binius differs
│   └── resources/       # Posts, talks, external resources
├── building/            # Practical developer guides
│   ├── tutorials.mdx    # Getting started tutorials
│   ├── circuits/        # Circuit construction guide
│   │   ├── introduction.mdx
│   │   ├── gadgets.mdx
│   │   ├── wires.mdx
│   │   ├── control-flow.mdx
│   │   └── performance.mdx
│   └── end-to-end/      # Complete example (SHA-256 commitment)
├── blueprint/           # Technical protocol specification
│   ├── overview.mdx     # Protocol phases overview
│   ├── commitment.mdx   # Polynomial commitment scheme
│   ├── math/            # Mathematical foundations
│   │   ├── fields.mdx       # Binary tower fields
│   │   ├── multilinears.mdx # Multilinear polynomials
│   │   ├── sumcheck.mdx     # Sumcheck protocol
│   │   └── oblong.mdx       # Oblong-multilinearization
│   ├── constraints/     # Constraint system specification
│   │   ├── introduction.mdx
│   │   ├── indices.mdx      # Shifted value indices (key concept!)
│   │   ├── arrays.mdx
│   │   ├── ands.mdx         # AND constraint definition
│   │   └── muls.mdx         # MUL constraint definition
│   └── backend/         # Protocol reductions
│       ├── shifts/      # Shift indicator polynomials
│       ├── ands/        # AND reduction (Rijndael zerocheck)
│       ├── muls/        # MUL reduction (GKR protocol)
│       └── reduction/   # Shift reduction (sumcheck)
└── benchmarks/          # Performance data
```

## Common Queries

**Understanding the constraint system:**
```
Read ../binius.xyz/docs/pages/blueprint/constraints/introduction.mdx
Read ../binius.xyz/docs/pages/blueprint/constraints/indices.mdx
```

**Mathematical background:**
```
Read ../binius.xyz/docs/pages/blueprint/math/fields.mdx
Read ../binius.xyz/docs/pages/blueprint/math/sumcheck.mdx
```

**Building circuits:**
```
Read ../binius.xyz/docs/pages/building/circuits/introduction.mdx
Read ../binius.xyz/docs/pages/building/circuits/gadgets.mdx
```

**Protocol overview:**
```
Read ../binius.xyz/docs/pages/blueprint/overview.mdx
```

## Searching Documentation

Find files by topic:
```
Glob ../binius.xyz/docs/pages/**/*.mdx
```

Search for specific terms:
```
Grep "shifted value" ../binius.xyz/docs/pages/
Grep "sumcheck" ../binius.xyz/docs/pages/blueprint/
```

## File Format Notes

- Files use `.mdx` format (Markdown with JSX components)
- Math is written in LaTeX: `$inline$` and `$$display$$`
- Some files import React components - ignore import statements and focus on markdown content
