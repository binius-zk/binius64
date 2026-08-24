// Copyright 2025-2026 The Binius Developers

//! Fractional-addition check: proving a claim about a sum of fractions.
//!
//! A witness is a numerator column beside a denominator column, one fraction $N_i/D_i$ per leaf.
//! The check reduces a claim on the sum of those fractions to a claim on the witness itself.
//! Two sibling fractions add by
//!
//! $$\frac{a_0}{b_0} + \frac{a_1}{b_1} = \frac{a_0b_1 + a_1b_0}{b_0b_1}$$
//!
//! so a circuit is a tree of layers, each half the width of the one below it.
//! One layer costs an MLE-check over the four column halves, then a line-fold.
//!
//! A batch runs one uniform layer schedule, so every tree in it must be of the same depth.
//! Zero-fraction padding lifts a shallower tree to the batch's depth.
//! The extra leaf positions hold $0/1$, the additive identity, so the tree's own sum is unchanged.
//! The verifier is oblivious to that padding and never learns the individual depths.
//! No padded witness is materialized: a layer's messages are corrected in $O(1)$ per round.
//!
//! One file per concern:
//!
//! - `fraction.rs` — the numerator/denominator pair that every layer and every claim carries.
//! - `circuit.rs` — the materialized layers of one tree, and the loop that proves them.
//! - `driver.rs` — the batched layer schedule, in four steps per layer.
//! - `padding.rs` — the unequal-depth policy: pad lengths, equality weights, and unpadding.
//! - `zero_pad_mle.rs` — the wrapper that corrects one padded layer's MLE-check messages.
//!
//! One layout decision is worth stating, because the obvious alternative is slower.
//! Each instance of a batched layer keeps its own column store and its own round pass.
//! Folding them into one store reads as the natural use of that type and costs 1.5-2x:
//! it trades a fat parallel region per instance for chunk parallelism over a far larger set.

mod circuit;
mod driver;
pub mod fraction;
pub mod padding;
pub mod zero_pad_mle;

pub use circuit::FracAddCircuit;
pub use driver::{BatchProveOutput, batch_prove_unequal_depths};
pub use padding::unpad_leaf_claim;

pub use crate::sumcheck::frac_add_mle::LayerProver;
