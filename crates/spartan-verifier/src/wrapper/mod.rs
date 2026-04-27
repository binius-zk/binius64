// Copyright 2026 The Binius Developers

//! Spartan wrapper for symbolically executing IOP verifiers to build constraint systems.
//!
//! This crate provides [`IronSpartanBuilderChannel`], an implementation of [`IPVerifierChannel`]
//! that symbolically executes a verifier and records the computation as an IronSpartan constraint
//! system via [`ConstraintBuilder`].
//!
//! [`IPVerifierChannel`]: binius_ip::channel::IPVerifierChannel
//! [`ConstraintBuilder`]: binius_spartan_frontend::circuit_builder::ConstraintBuilder

mod channel;
pub mod circuit_elem;
pub mod gadgets;
mod zk_wrapped_channel;

pub use channel::{IronSpartanBuilderChannel, ReplayChannel};
pub use zk_wrapped_channel::ZKWrappedVerifierChannel;

#[cfg(test)]
mod tests {
	use std::rc::Rc;

	use binius_field::{
		BinaryField1b as B1, BinaryField128bGhash as B128, ExtensionField, Field, Random,
		arithmetic_traits::InvertOrZero, field::FieldOps,
	};
	use binius_ip::channel::IPVerifierChannel;
	use binius_spartan_frontend::circuit_builder::{ConstraintBuilder, WitnessGenerator};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::wrapper::circuit_elem::{CircuitElem, CircuitWire};

	type BuildElem = CircuitElem<ConstraintBuilder<B128>>;
	type BuildWire = CircuitWire<ConstraintBuilder<B128>>;

	/// Helper to create a BuildWire from a ConstraintBuilder Rc for tests.
	fn alloc_inout_wire(rc: &Rc<std::cell::RefCell<ConstraintBuilder<B128>>>) -> BuildElem {
		let wire = rc.borrow_mut().alloc_inout();
		BuildElem::Wire {
			wire: BuildWire::new(rc, wire),
			public: false,
		}
	}

	#[test]
	fn test_constant_arithmetic() {
		let a = BuildElem::Constant(B128::new(3));
		let b = BuildElem::Constant(B128::new(5));

		// Addition in char 2: 3 + 5 = 3 ^ 5 = 6
		let sum = a.clone() + b.clone();
		assert!(matches!(sum, BuildElem::Constant(c) if c == B128::new(3) + B128::new(5)));

		// Multiplication
		let product = a.clone() * b.clone();
		assert!(matches!(product, BuildElem::Constant(c) if c == B128::new(3) * B128::new(5)));

		// Subtraction equals addition in char 2
		let diff = a - b;
		assert!(matches!(diff, BuildElem::Constant(c) if c == B128::new(3) + B128::new(5)));
	}

	#[test]
	fn test_constant_identity_shortcuts() {
		let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::<B128>::new()));
		let elem = alloc_inout_wire(&rc);

		// Adding zero returns the wire unchanged.
		let result = elem.clone() + BuildElem::Constant(B128::ZERO);
		assert!(matches!(result, BuildElem::Wire { .. }));

		// Multiplying by one returns the wire unchanged.
		let result = elem.clone() * BuildElem::Constant(B128::ONE);
		assert!(matches!(result, BuildElem::Wire { .. }));

		// Multiplying by zero returns constant zero.
		let result = elem * BuildElem::Constant(B128::ZERO);
		assert!(matches!(result, BuildElem::Constant(c) if c == B128::ZERO));
	}

	#[test]
	fn test_wire_addition_creates_constraint() {
		let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::<B128>::new()));
		let a = alloc_inout_wire(&rc);
		let b = alloc_inout_wire(&rc);

		let _sum = a + b;
		// The addition should produce constraints. After finalization, the zero constraint
		// from addition becomes a mul constraint (multiplied by one).
		let (cs, _layout) = Rc::try_unwrap(rc).unwrap().into_inner().build().finalize();
		assert!(!cs.mul_constraints().is_empty());
	}

	#[test]
	fn test_wire_multiplication_creates_constraint() {
		let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::<B128>::new()));
		let a = alloc_inout_wire(&rc);
		let b = alloc_inout_wire(&rc);

		let _product = a * b;
		let (cs, _layout) = Rc::try_unwrap(rc).unwrap().into_inner().build().finalize();
		// At least 1 mul constraint from a*b.
		assert!(!cs.mul_constraints().is_empty());
	}

	#[test]
	fn test_invert_or_zero_creates_constraints() {
		let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::<B128>::new()));
		let elem = alloc_inout_wire(&rc);

		let _inv = elem.invert_or_zero();
		let (cs, _layout) = Rc::try_unwrap(rc).unwrap().into_inner().build().finalize();
		// InvertOrZero creates: a mul constraint (wire * inv) and a zero constraint
		// (product ^ one), both of which become mul constraints after finalization.
		assert!(cs.mul_constraints().len() >= 2);
	}

	#[test]
	fn test_channel_recv_and_sample() {
		let mut channel = IronSpartanBuilderChannel::new(ConstraintBuilder::<B128>::new());

		let a = channel.recv_one().unwrap();
		let b = channel.sample();
		let c = channel.recv_array::<3>().unwrap();

		// All should be Wire variants.
		assert!(matches!(a, BuildElem::Wire { .. }));
		assert!(matches!(b, BuildElem::Wire { .. }));
		for elem in &c {
			assert!(matches!(elem, BuildElem::Wire { .. }));
		}
	}

	#[test]
	fn test_channel_assert_zero() {
		let mut channel = IronSpartanBuilderChannel::new(ConstraintBuilder::<B128>::new());

		// Assert zero on a constant zero should succeed.
		assert!(channel.assert_zero(BuildElem::Constant(B128::ZERO)).is_ok());

		// Assert zero on a nonzero constant should fail.
		assert!(channel.assert_zero(BuildElem::Constant(B128::ONE)).is_err());

		// Assert zero on a wire should succeed (records constraint).
		let wire_elem = channel.recv_one().unwrap();
		assert!(channel.assert_zero(wire_elem).is_ok());
	}

	#[test]
	fn test_verify_iop_builds_constraint_system() {
		use binius_spartan_frontend::{
			circuit_builder::CircuitBuilder, circuits::powers, compiler::compile,
		};

		use crate::{
			IOPVerifier,
			constraint_system::{BlindingInfo, ConstraintSystemPadded},
		};

		// Build a power7 circuit: assert that x^7 = y.
		fn power7_circuit<Builder: CircuitBuilder>(
			builder: &mut Builder,
			x_wire: Builder::Wire,
			y_wire: Builder::Wire,
		) {
			let powers_vec = powers(builder, x_wire, 7);
			let x7 = powers_vec[6];
			builder.assert_eq(x7, y_wire);
		}

		let mut constraint_builder = ConstraintBuilder::<B128>::new();
		let x_wire = constraint_builder.alloc_inout();
		let y_wire = constraint_builder.alloc_inout();
		power7_circuit(&mut constraint_builder, x_wire, y_wire);
		let (cs, _layout) = compile(constraint_builder);

		// Build IOPVerifier directly from the constraint system.
		let blinding_info = BlindingInfo {
			n_dummy_wires: 10,
			n_dummy_constraints: 2,
		};
		let cs = ConstraintSystemPadded::new(cs, blinding_info);
		let iop_verifier = IOPVerifier::new(cs);
		let cs = iop_verifier.constraint_system();
		let public_size = 1 << cs.log_public();

		// Create the builder channel and run IOPVerifier::verify symbolically.
		let mut channel = IronSpartanBuilderChannel::new(ConstraintBuilder::<B128>::new());

		// Use zero-filled public inputs of the correct length.
		let public = vec![B128::ZERO; public_size];
		let public_elems = channel.observe_many(&public);
		// IronSpartanBuilderChannel::Oracle = () and recv_oracle is a no-op, so pass () directly.
		iop_verifier
			.verify((), public_elems, &mut channel)
			.expect("symbolic verify failed");

		let builder = channel.finish();

		// Extract the constraint system built by the symbolic execution.
		let (wrapper_cs, _layout) = builder.build().finalize();

		// The symbolic execution should have produced a nontrivial constraint system.
		assert!(wrapper_cs.n_inout() > 0);
		assert!(!wrapper_cs.mul_constraints().is_empty());
	}

	#[test]
	fn test_square_transpose_constants() {
		type FSub = B1;
		let degree = <B128 as ExtensionField<FSub>>::DEGREE;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random elements and transpose them both natively and via CircuitElem.
		let values = (0..degree)
			.map(|_| B128::random(&mut rng))
			.collect::<Vec<_>>();

		let mut expected = values.clone();
		<B128 as ExtensionField<FSub>>::square_transpose(&mut expected);

		let mut elems = values
			.iter()
			.map(|&v| BuildElem::Constant(v))
			.collect::<Vec<_>>();
		<BuildElem as FieldOps>::square_transpose::<FSub>(&mut elems);

		for (i, (elem, &exp)) in elems.iter().zip(&expected).enumerate() {
			match elem {
				CircuitElem::Constant(c) => assert_eq!(*c, exp, "mismatch at index {i}"),
				CircuitElem::Wire { .. } => panic!("expected constant after all-constants transpose"),
			}
		}
	}

	#[test]
	fn test_square_transpose_wires() {
		// Test that square_transpose on wire elements builds a valid constraint system,
		// and that a WitnessGenerator with correct values satisfies all constraints.
		type FSub = B1;
		let degree = <B128 as ExtensionField<FSub>>::DEGREE;

		// Phase 1: Build the constraint system symbolically.
		let mut constraint_builder = ConstraintBuilder::<B128>::new();
		let inout_wires: Vec<_> = (0..degree)
			.map(|_| constraint_builder.alloc_inout())
			.collect();

		// Build CircuitElem wires via a shared Rc.
		let rc = Rc::new(std::cell::RefCell::new(constraint_builder));
		let mut elems: Vec<BuildElem> = inout_wires
			.iter()
			.map(|&w| BuildElem::Wire {
				wire: BuildWire::new(&rc, w),
				public: false,
			})
			.collect();

		<BuildElem as FieldOps>::square_transpose::<FSub>(&mut elems);

		// The transposed outputs are wires; drop them so we can extract the builder.
		drop(elems);
		let constraint_builder = Rc::try_unwrap(rc).unwrap().into_inner();
		let (cs, layout) = constraint_builder.build().finalize();

		// The constraint system should have multiplication constraints from
		// Frobenius checks, reconstruction, and transposed output.
		assert!(!cs.mul_constraints().is_empty());

		// Phase 2: Generate a witness with concrete values and verify all constraints.
		let test_values: Vec<B128> = (0..degree)
			.map(<B128 as ExtensionField<FSub>>::basis)
			.collect();

		let mut witness_gen = WitnessGenerator::new(&layout);
		let witness_wires: Vec<_> = inout_wires
			.iter()
			.zip(&test_values)
			.map(|(&w, &val)| witness_gen.write_inout(w, val))
			.collect();

		type WitnessElem<'a> = CircuitElem<WitnessGenerator<'a, B128>>;
		let witness_rc = Rc::new(std::cell::RefCell::new(witness_gen));
		let mut witness_elems: Vec<WitnessElem> = witness_wires
			.iter()
			.map(|&w| WitnessElem::Wire {
				wire: crate::wrapper::circuit_elem::CircuitWire::new(&witness_rc, w),
				public: false,
			})
			.collect();

		<WitnessElem as FieldOps>::square_transpose::<FSub>(&mut witness_elems);

		drop(witness_elems);
		let witness_gen = Rc::try_unwrap(witness_rc).unwrap().into_inner();
		let witness = witness_gen
			.build()
			.expect("witness generation should succeed (all constraints satisfied)");

		cs.validate(&witness);
	}

	#[test]
	fn test_channel_integration_simple_circuit() {
		// Build a simple circuit: recv two values, multiply them, assert_zero on the
		// difference with a third received value (ie. a * b == c).
		let mut channel = IronSpartanBuilderChannel::new(ConstraintBuilder::<B128>::new());
		let a = channel.recv_one().unwrap();
		let b = channel.recv_one().unwrap();
		let c = channel.recv_one().unwrap();
		let product = a * b;
		// In char 2, product - c == product + c.
		let diff = product + c;
		channel.assert_zero(diff).unwrap();

		// Finish and extract the constraint system.
		let builder = channel.finish();
		let (cs, _layout) = builder.build().finalize();

		// Should have inout wires from recv + private wires from mul and add.
		assert!(cs.n_inout() >= 3);
		// Should have at least 1 mul constraint from a*b and 1 from the finalized zero
		// constraint.
		assert!(!cs.mul_constraints().is_empty());
	}
}
