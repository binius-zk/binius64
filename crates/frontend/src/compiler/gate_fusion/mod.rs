// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Gate fusion optimization.
//!
//! The main cost of our system is coming from the number of AND constraints. The less we have the
//! cheaper it is.
//!
//! Our AND constraints are powerful construct construct. They can handle a single AND of two XOR
//! combinations where each of the values could be shifted.
//!
//! `ConstraintBuilder` which this pass operates consists of AND, IMUL and linear constraints.
//! Linear constraints are basically are constraints that define a single wire using a XOR
//! combination and/or shifts. Since our system does not suppose standalone linear combinations a
//! wire whose definition survives has to be committed, and its definition costs a constraint of
//! its own — an AND against the all-ones wire, or a Zero constraint under
//! `enable_zero_constraints`.
//!
//! BUT we have a chance of avoiding that if we manage to inline that wire into every consumer
//! constraint which means we don't have to commit that value and thus we don't spend a constraint
//! on it at all!
//!
//! A definition this pass does have to commit stays a linear constraint, carrying the inlined
//! cone as its right-hand side. Which of the two lowerings it gets is
//! [`ConstraintBuilder::build`]'s call, not this pass's.

use cranelift_entity::EntitySet;
use legraph::LeGraph;

use crate::compiler::{Wire, constraint_builder::ConstraintBuilder};

mod commit_set;
mod legraph;
mod patch;

mod stat;
#[cfg(test)]
mod tests;

use stat::Stat;

pub fn run_pass(cb: &mut ConstraintBuilder, pinned_wires: &EntitySet<Wire>) {
	let mut stat = Stat::new(cb);

	let mut leg = LeGraph::new(cb);
	commit_set::run_decide_commit_set(&mut leg, &mut stat);
	// Pin force-committed wires that are linear definitions so their values survive as committed
	// definitions. Pinned wires that are not linear definitions (e.g. AND or IMUL outputs) are
	// already committed by their own constraints, so they must be excluded here: `patch::build`
	// treats every wire in the commit set as a linear definition and would otherwise panic.
	let pinned_lin_defs = pinned_wires
		.iter()
		.filter(|&wire| leg.is_lin_def(wire))
		.collect::<Vec<_>>();
	leg.lin_committed.extend(pinned_lin_defs);
	let patches = patch::build(cb, &leg);
	patch::apply_patches(cb, patches);
}
