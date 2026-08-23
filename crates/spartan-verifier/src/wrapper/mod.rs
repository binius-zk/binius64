// Copyright 2026 The Binius Developers

//! Spartan wrapper for symbolically executing IOP verifiers to build constraint systems.
//!
//! This crate provides [`IronSpartanBuilderChannel`], an implementation of [`IPVerifierChannel`]
//! that symbolically executes a verifier and records the computation as an IronSpartan constraint
//! system via [`ConstraintBuilder`].
//!
//! [`IPVerifierChannel`]: binius_ip::channel::IPVerifierChannel
//! [`ConstraintBuilder`]: binius_spartan_frontend::circuit_builder::ConstraintBuilder

pub mod builder_channel;
pub mod circuit_elem;
pub mod gadgets;
pub mod zk_wrapped_channel;

pub use builder_channel::IronSpartanBuilderChannel;
pub use zk_wrapped_channel::ZKWrappedVerifierChannel;

#[cfg(test)]
mod tests {
	use std::rc::Rc;

	use binius_core::word::Word;
	use binius_field::{
		BinaryField1b as B1, ExtensionField, Field, Ghash128b as B128, Random,
		arithmetic_traits::InvertOrZero, field::FieldOps,
	};
	use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
	use binius_spartan_frontend::circuit_builder::ConstraintBuilder;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::wrapper::circuit_elem::CircuitElem;

	type BuildElem = CircuitElem<B128, ConstraintBuilder<B128>>;

	/// Helper to create a private-backed `BuildElem` wire from a ConstraintBuilder Rc for tests.
	///
	/// Uses a precommit wire (the real source of a private wire in wrapper usage — the OTP key) so
	/// that arithmetic on it stays private and records constraints. An inout-backed wire would
	/// instead be elided into a derived wire with no constraint.
	fn alloc_private_wire(rc: &Rc<std::cell::RefCell<ConstraintBuilder<B128>>>) -> BuildElem {
		let wire = rc.borrow_mut().alloc_precommit();
		BuildElem::wire(rc, wire)
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
		let elem = alloc_private_wire(&rc);

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
		let a = alloc_private_wire(&rc);
		let b = alloc_private_wire(&rc);

		let _sum = a + b;
		// The addition should produce constraints. After finalization, the zero constraint
		// from addition becomes a mul constraint (multiplied by one).
		let (cs, _layout) = Rc::try_unwrap(rc).unwrap().into_inner().build().finalize();
		assert!(!cs.mul_constraints().is_empty());
	}

	#[test]
	fn test_wire_multiplication_creates_constraint() {
		let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::<B128>::new()));
		let a = alloc_private_wire(&rc);
		let b = alloc_private_wire(&rc);

		let _product = a * b;
		let (cs, _layout) = Rc::try_unwrap(rc).unwrap().into_inner().build().finalize();
		// At least 1 mul constraint from a*b.
		assert!(!cs.mul_constraints().is_empty());
	}

	#[test]
	fn test_invert_creates_constraints() {
		let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::<B128>::new()));
		let elem = alloc_private_wire(&rc);

		// SAFETY: nothing constrains the wire, so the contract is vacuous here; the test only
		// counts the constraints the inverse emits.
		let _inv = unsafe { elem.invert() };
		let (cs, _layout) = Rc::try_unwrap(rc).unwrap().into_inner().build().finalize();
		// The inverse creates: a mul constraint (wire * inv) and a zero constraint
		// (product ^ one), both of which become mul constraints after finalization.
		assert!(cs.mul_constraints().len() >= 2);
	}

	#[test]
	#[should_panic(expected = "the wrapper inverts only values argued non-zero")]
	fn test_invert_or_zero_is_unimplemented() {
		let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::<B128>::new()));
		let elem = alloc_private_wire(&rc);

		let _ = elem.invert_or_zero();
	}

	#[test]
	fn test_channel_recv_and_sample() {
		let mut channel = IronSpartanBuilderChannel::<B128>::new();

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

	/// The packed statement enters the circuit as inout wires, not as constants.
	///
	/// The wrapper circuit is built once, against whatever statement the symbolic run was handed,
	/// and reused for every other one. Packing the words into constants would fix that first
	/// statement into the circuit, so the words become wires the concrete channels fill in.
	#[test]
	fn test_pack_words_allocates_inout_wires() {
		let mut channel = IronSpartanBuilderChannel::<B128>::new();

		// Two words to a `B128`, so three words span two elements. The zero word is deliberate:
		// nothing about the packing may turn on the values.
		let words = [Word::from_u64(7), Word::ZERO, Word::from_u64(9)];
		let elems = channel.pack_words(&words);

		assert_eq!(elems.len(), 2);
		assert!(
			elems
				.iter()
				.all(|elem| matches!(elem, BuildElem::Wire { .. }))
		);
	}

	#[test]
	fn test_channel_assert_zero() {
		let mut channel = IronSpartanBuilderChannel::<B128>::new();

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
		let mut channel = IronSpartanBuilderChannel::<B128>::new();

		// Use zero-filled public inputs of the correct length.
		let public = vec![B128::ZERO; public_size];
		let public_elems = channel.observe_many(&public);
		// IronSpartanBuilderChannel::Oracle = () and recv_oracle is a no-op, so pass () directly.
		iop_verifier
			.verify((), &public_elems, &mut channel)
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
				CircuitElem::Wire { .. } => {
					panic!("expected constant after all-constants transpose")
				}
			}
		}
	}

	#[test]
	fn test_channel_integration_simple_circuit() {
		// Build a simple circuit: recv two values, multiply them, assert_zero on the
		// difference with a third received value (ie. a * b == c).
		let mut channel = IronSpartanBuilderChannel::<B128>::new();
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
