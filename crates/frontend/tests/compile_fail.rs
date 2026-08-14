// Copyright 2026 The Binius Developers

//! Pins the ownership contract on building a circuit.
//!
//! A builder is consumed by the call.
//! So the compiler rejects reuse at compile time, instead of a panic surfacing at run time.

#[test]
fn build_consumes_the_builder() {
	let t = trybuild::TestCases::new();
	t.compile_fail("tests/compile-fail/*.rs");
}
