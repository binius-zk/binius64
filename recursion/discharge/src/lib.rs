//! monster-discharge — STEP 1 of SPEC v2 "MONSTER DISCHARGE".
//!
//! Batch-discharges K deferred Binius64 "monster" claims (the per-leaf O(N) monster
//! multilinear evaluation of shift/verify.rs check_eval) against ONE shared table pass:
//!
//! Phase 0 statement absorption + structural asserts (P0.1..P0.4)
//! Phase A one batched degree-3 sumcheck over the term domain (K claims -> shared rho)
//! Phase C (STEP 2) weighted fracaddcheck: committed M_D == rho-weighted histograms of
//!         the committed VK columns (union-domain partial fractions, coset-disjoint tags)
//! Phase B one bivariate sumcheck [W_eq, M_D] (10K(+1) MLE claims -> one (sigma, m))
//! Final  STEP 1: native one-pass rebuild of M_D from the CS at rho + M~_D(sigma) == m
//!        STEP 2: BaseFold opening of M_D at sigma + corner-trick M_VK opening — the
//!                verifier never touches the CS (takes VKM + statement + transcript).
//!
//! STEP 1 (native mode) remains available for regression: see `step2::DischargeMode`.

pub mod cubic;
pub mod discharge;
pub mod fracadd;
pub mod leaf;
pub mod merged;
pub mod packed;
pub mod recorder;
pub mod step2;
pub mod synth;
pub mod table;
pub mod vk;

pub use binius_verifier::config::B128;
