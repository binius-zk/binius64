//! binius-recursion-multizk — the multi-inner generalization of the Binius64 ZK wrap.
//!
//! ONE outer IronSpartan circuit symbolically executes the Binius64 IOP verifier for K
//! (possibly distinct) inner constraint systems, in order. All K inner proofs and the
//! outer proof share a single Fiat-Shamir transcript and ONE combined BaseFold opening.
//! This is the general K-inner wrapper the integrated discharge (see
//! `binius-recursion-wrap`) is layered on top of.

pub mod multi_zk;

pub use multi_zk::{MultiZKProver, MultiZKVerifier};
