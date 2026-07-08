// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub mod batch;
pub mod batch_quadratic_mle;
pub mod bivariate_product;
pub mod bivariate_product_mle;
pub mod bivariate_product_multi_mle;
pub mod common;
pub mod gruen32;
mod mle_to_sumcheck;
pub mod multilinear_eval;
mod padded;
mod prove;
pub mod quadratic_mle;
pub mod rerand_mle;
// `round_evals` is internal implementation, exposed (via `#[doc(hidden)]` `pub mod`) only so
// `binius-prover` can compute the shift reduction's sparse first sumcheck round with the exact
// interpolation the in-crate provers use. Not a stable API.
#[doc(hidden)]
pub mod round_evals;
mod round_state;
pub mod selector_mle;
mod switchover;
pub use mle_to_sumcheck::*;
pub use padded::*;
pub use prove::*;
pub mod frac_add_mle;
pub mod zk_mlecheck;

pub use batch::batch_prove;
