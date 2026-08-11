// Copyright 2026 The Binius Developers

//! A smoke test that the channel records arithmetic into a circuit that compiles.
//!
//! This exercises the paths the skeleton implements — receiving elements, multiplying them, and
//! asserting — and deliberately avoids the ones it does not, which panic with `todo!()`.

use binius_field::BinaryField128bGhash as B128;
use binius_frontend::CircuitStat;
use binius_ip::channel::IPVerifierChannel;
use binius_recursion::Binius64BuilderChannel;

#[test]
fn records_received_arithmetic_as_constraints() {
	let mut channel = Binius64BuilderChannel::new(0);

	// Read `a`, `b` and `c` off the proof and record that `a * b == c`.
	let a = channel.recv_one().unwrap();
	let b = channel.recv_one().unwrap();
	let c = channel.recv_one().unwrap();
	channel.assert_zero(a * b - c).unwrap();

	let recorded = channel.build();

	// Three elements at two wires each, in the order the verifier read them.
	assert_eq!(recorded.inputs.len(), 6);

	let stat = CircuitStat::collect(&recorded.circuit);

	// One GHASH multiplication is one BMUL constraint.
	assert_eq!(stat.n_bmul_constraints, 1);
	// The assertion lands in the ZERO column, one per word of the element.
	assert_eq!(stat.n_zero_constraints, 2);
}

#[test]
fn folds_constants_without_constraints() {
	let mut channel = Binius64BuilderChannel::new(0);

	// A product of build-time constants is decided while building, so it records nothing, and
	// asserting a zero constant succeeds without a constraint.
	let two = binius_recursion::SymbolicElem::Constant(B128::new(2));
	let three = binius_recursion::SymbolicElem::Constant(B128::new(3));
	let product = two * &three;
	channel
		.assert_zero(product.clone() - product)
		.expect("a constant zero satisfies the assertion");

	let stat = CircuitStat::collect(&channel.build().circuit);
	assert_eq!(stat.n_bmul_constraints, 0);
	assert_eq!(stat.n_zero_constraints, 0);
}
