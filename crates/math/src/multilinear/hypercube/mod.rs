// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Multilinear coefficients indexed by a hypercube, and the expansions built over one.
//!
//! An `n`-variate multilinear is stored as `2^n` coefficients.
//! The basis those coefficients are taken against factors as a tensor product over the variables.
//! Every variable contributes the same two-element basis `(b_0, b_1)` of linear polynomials.
//! That single choice fixes the cube, and with it what each coefficient means:
//!
//! ```text
//! basis (1 - X, X)    vertices {0, 1}      coefficients are evaluations
//! basis (1, X)        vertices {0, inf}    coefficients are monomial coefficients
//! ```
//!
//! The object built over a cube again and again is the equality indicator.
//! Written `eq(X, Y)`, it extends the predicate `X == Y` multilinearly over the cube.
//! Fixing one operand to a point leaves `2^n` coefficients, called the expansion of that point.
//!
//! Every operation derived from the basis is generic over the cube and lives in its own module:
//!
//! ```text
//! cube         the two bases, and everything a cube derives from its own
//! expansion    the expansion of a point, chosen seed by seed and store by store
//! ```

mod cube;
mod expansion;

pub use cube::Hypercube;
pub use expansion::Expansion;
