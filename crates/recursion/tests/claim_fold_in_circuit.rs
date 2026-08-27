// Copyright 2026 The Binius Developers

//! The verifier side of a claim fold, run as a circuit.
//!
//! The fold is what keeps an aggregation node's carried state from growing.
//! For that to help, its verifier has to run *inside* the node.
//!
//! So this drives it over the builder channel.
//! What comes out must be constraints rather than a verdict.

use binius_field::Ghash128b as B128;
use binius_frontend::CircuitStat;
use binius_ip::{MultilinearEvalClaim, batch_eval, channel::IPVerifierChannel};
use binius_recursion::Binius64BuilderChannel;

/// Three axes over 2, 1 and 2 variables, as a node folding two children would carry.
const AXES: [usize; 3] = [2, 1, 2];

#[test]
fn the_fold_verifier_records_constraints_rather_than_checking_values() {
	// Invariant: the reduction's verifier names no concrete field element.
	//
	// It reads claims off a channel and asserts one relation.
	// Over a channel carrying wires, that assertion becomes a constraint.
	//
	// So the same code that settles a fold natively also expresses it in a circuit.
	// Nothing here is fold-specific: it is what the channel abstraction buys.
	//
	//     claims as wires -> the reduction -> constraints, and a point of the same width
	let mut channel = Binius64BuilderChannel::new();

	// Two claims, read off the tape exactly as a node reads its children's statements.
	let n_vars: usize = AXES.iter().sum();
	let points = (0..2)
		.map(|_| {
			(0..n_vars)
				.map(|_| IPVerifierChannel::<B128>::recv_one(&mut channel).unwrap())
				.collect::<Vec<_>>()
		})
		.collect::<Vec<_>>();
	let evals = (0..2)
		.map(|_| IPVerifierChannel::<B128>::recv_one(&mut channel).unwrap())
		.collect::<Vec<_>>();

	let claims = points
		.into_iter()
		.zip(evals)
		.map(|(point, eval)| MultilinearEvalClaim { eval, point })
		.collect::<Vec<_>>();

	let reduced = batch_eval::verify::<B128, _>(claims, &mut channel)
		.expect("recording cannot fail: nothing is compared");

	// The reduced claim spans the same variables, so a node above could reduce it again.
	assert_eq!(reduced.point.len(), n_vars);

	// Every symbolic value must be dropped before the circuit is built.
	drop(reduced);
	let recorded = channel.build();

	let stat = CircuitStat::collect(&recorded.circuit);
	println!(
		"fold verifier in-circuit: {} AND, {} BMUL, {} ZERO, {} recorded inputs",
		stat.n_and_constraints,
		stat.n_bmul_constraints,
		stat.n_zero_constraints,
		recorded.inputs.len(),
	);

	// The weight evaluations are field arithmetic, so they land in the multiplication column.
	assert!(stat.n_bmul_constraints > 0, "the weight evaluations must be recorded");
	// And the reduction's one relation is asserted, not compared.
	assert!(stat.n_zero_constraints > 0, "the reduction's relation must become a constraint");
	// The round polynomials and the tensor evaluation are proof data, so a replay supplies them.
	assert!(!recorded.inputs.is_empty(), "the tape's values must be recorded for a replay");
}
