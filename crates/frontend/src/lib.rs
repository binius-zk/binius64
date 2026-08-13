// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Circuit construction frontend for Binius64.
//!
//! This crate provides the [`CircuitBuilder`] API for constructing arithmetic circuits
//! that compile to Binius64 constraint systems. You describe your computation as a graph
//! of operations on 64-bit words, and the frontend compiles it to AND/IMUL/BMUL constraints.
//!
//! # Usage Flow
//!
//! Use [`CircuitBuilder`] to construct your circuit. Call methods like `add_witness()`
//! and `add_inout()` to create [`Wire`]s - handles to 64-bit values that will exist during
//! proof generation. Use operations like `band()`, `bxor()`, and `iadd_32()` to transform
//! these wires, building up your computation graph.
//!
//! When you call `build()`, the builder compiles your graph into a [`Circuit`]. This circuit
//! contains the optimized constraint system and everything needed for proof generation.
//!
//! To generate a witness, create a [`WitnessFiller`] from the circuit. Assign concrete values
//! to your input wires, then call `populate_wire_witness()` to compute all intermediate values
//! through circuit evaluation.
//!
//! Use [`CircuitStat`] to inspect metrics like constraint counts and wire usage, helpful for
//! optimization and debugging.
//!
//! # Layout
//!
//! - `builder` — the vocabulary a circuit author writes.
//! - `ir` — the gate graph those calls build, held as dense index maps.
//! - `gates` — the shape, the constraints and the instruction of each gate kind.
//! - `pass` — transformations over the graph, and the pipeline that runs them.
//! - `lower` — the constraints a graph emits, over wires rather than value indices.
//! - `eval_form` — the bytecode that fills a witness, and the interpreters that run it.
//! - `artifact` — what a build hands back: the circuit, its witnesses, its statistics.
//!
//! Every module reads `ir`, and nothing reads `builder`.

#![warn(rustdoc::missing_crate_level_docs)]

mod artifact;
mod builder;
mod eval_form;
mod gates;
mod ir;
mod lower;
mod pass;

pub use artifact::{
	chip::{self, ChipRef, CircuitM4, CircuitM4Error, EmbeddedCircuit, PopulateM4Error},
	circuit::Circuit,
	stat::{self, CircuitStat},
	witness::{AssertionFailure, BatchWitnessFiller, PopulateError, WitnessFiller},
};
pub use builder::{CircuitBuilder, Options};
pub use eval_form::{BatchPopulateError, MAX_ASSERTION_FAILURES};
pub use ir::{
	Wire,
	hints::{self, Hint},
};
