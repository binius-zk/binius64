//! binius-recursion-wrap — integrated flat aggregation, final integration:
//! K leaf IOPs through SUBSTITUTING wrapped channels (prover-supplied monster values,
//! zero O(leaf) monster work at final verification) + one outer IronSpartan proof +
//! one combined ZK BaseFold opening + the STEP-2 batched monster discharge certifying
//! exactly the K substituted values — all on ONE Fiat-Shamir transcript.

pub mod integrated;
pub mod substituting;

pub use binius_verifier::config::B128;
