// Copyright 2026 The Binius Developers

//! The Spark compiler for sparse multilinear polynomials.
//!
//! Spark lets a verifier obtain the evaluation of a sparse multilinear polynomial with work
//! sublinear in the number of nonzero terms, backed by a one-time preprocessing commitment and an
//! offline memory-checking argument. See the Spartan paper, §7.

pub mod timestamps;
