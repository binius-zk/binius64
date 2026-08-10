// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::{Composition, Shift};

use super::legraph::LeGraph;
use crate::compiler::{
	Wire,
	constraint_builder::{
		ConstraintBuilder, ShiftedWire, WireAndConstraint, WireBmulConstraint, WireImulConstraint,
		WireLinearConstraint, WireOperand,
	},
	gate_fusion::legraph::ConstraintRef,
};

/// A patch is a description of a change to the constraint system.
///
/// It specifies a list of constraints that are going to be removed and a list of constraints that
/// are going to be added.
///
/// The lhs and rhs MUST be equivalent to preserve soundness.
pub struct Patch {
	/// The constraint set that is going to be replaced with this one.
	subsumes: Vec<ConstraintRef>,
	/// The new constraints that is going to be added to the graph.
	added: AddedConstraint,
}

enum AddedConstraint {
	And(WireAndConstraint),
	Imul(WireImulConstraint),
	Bmul(WireBmulConstraint),
	/// A committed linear definition, kept linear so that
	/// [`ConstraintBuilder::build`](crate::compiler::constraint_builder::ConstraintBuilder::build)
	/// picks its lowering.
	Linear(WireLinearConstraint),
}

/// Apply the given patches to the constraint builder given.
pub fn apply_patches(cb: &mut ConstraintBuilder, patches: Vec<Patch>) {
	// One flag per existing constraint, per kind.
	// A patch names constraints by position, so subsumption is answered by indexing, not hashing.
	let mut subsumed_and = vec![false; cb.and_constraints.len()];
	let mut subsumed_imul = vec![false; cb.imul_constraints.len()];
	let mut subsumed_bmul = vec![false; cb.bmul_constraints.len()];
	let mut subsumed_linear = vec![false; cb.linear_constraints.len()];

	let mut new_and_constraints = Vec::new();
	let mut new_imul_constraints = Vec::new();
	let mut new_bmul_constraints = Vec::new();
	let mut new_linear_constraints = Vec::new();

	// Collect all subsumed constraints and new constraints to add.
	// Patches may name the same constraint more than once, which sets the same flag twice.
	for patch in patches {
		for subsumed in patch.subsumes {
			match subsumed {
				ConstraintRef::And { index } => subsumed_and[index] = true,
				ConstraintRef::Imul { index } => subsumed_imul[index] = true,
				ConstraintRef::Bmul { index } => subsumed_bmul[index] = true,
				ConstraintRef::Linear { index } => subsumed_linear[index] = true,
			}
		}
		match patch.added {
			AddedConstraint::And(and_constraint) => new_and_constraints.push(and_constraint),
			AddedConstraint::Imul(imul_constraint) => new_imul_constraints.push(imul_constraint),
			AddedConstraint::Bmul(bmul_constraint) => new_bmul_constraints.push(bmul_constraint),
			AddedConstraint::Linear(linear_constraint) => {
				new_linear_constraints.push(linear_constraint)
			}
		}
	}

	// Drop the subsumed constraints in place, keeping the survivors in their original order.
	retain_unsubsumed(&mut cb.and_constraints, &subsumed_and);
	retain_unsubsumed(&mut cb.imul_constraints, &subsumed_imul);
	retain_unsubsumed(&mut cb.bmul_constraints, &subsumed_bmul);
	retain_unsubsumed(&mut cb.linear_constraints, &subsumed_linear);

	// Add the new constraints
	cb.and_constraints.extend(new_and_constraints);
	cb.imul_constraints.extend(new_imul_constraints);
	cb.bmul_constraints.extend(new_bmul_constraints);
	cb.linear_constraints.extend(new_linear_constraints);
}

/// Removes the constraints flagged as subsumed, preserving the order of the rest.
fn retain_unsubsumed<T>(constraints: &mut Vec<T>, subsumed: &[bool]) {
	let mut position = 0;
	constraints.retain(|_| {
		let keep = !subsumed[position];
		position += 1;
		keep
	});
}

/// Builds a list of patches that would remove the inlined linear definitions and potentially
/// AND constraints.
///
/// NB: patches may have overlapping subsumes.
pub fn build(cb: &ConstraintBuilder, leg: &LeGraph) -> Vec<Patch> {
	let mut patches = vec![];
	build_non_linear_patches(cb, leg, &mut patches);
	for committed in leg.commit_set().iter() {
		let patch = build_committed_lin_def_patch(cb, leg, committed);
		patches.push(patch);
	}
	patches
}

/// Collect patches for the non-linear constraints that inline linear definitions.
fn build_non_linear_patches(cb: &ConstraintBuilder, leg: &LeGraph, patches: &mut Vec<Patch>) {
	// Collect *distinct* constraint references for each root constraint.
	let mut constraints = Vec::with_capacity(leg.roots.len());
	for root in leg.roots.iter() {
		constraints.push(leg.root_constraint_ref(*root));
	}
	constraints.sort_unstable();
	constraints.dedup();

	// Create a patch for each distinct constraint.
	for constraint_ref in constraints {
		let patch = build_non_lin_patch(cb, leg, constraint_ref);
		patches.push(patch);
	}
}

fn build_non_lin_patch(
	cb: &ConstraintBuilder,
	leg: &LeGraph,
	constraint_ref: ConstraintRef,
) -> Patch {
	let mut subsumes = vec![constraint_ref];

	let new_constraint = match constraint_ref {
		ConstraintRef::And { index } => {
			let a = process_operand(cb, leg, &mut subsumes, &cb.and_constraints[index].a);
			let b = process_operand(cb, leg, &mut subsumes, &cb.and_constraints[index].b);
			let c = process_operand(cb, leg, &mut subsumes, &cb.and_constraints[index].c);
			AddedConstraint::And(WireAndConstraint { a, b, c })
		}
		ConstraintRef::Imul { index } => {
			let a = process_operand(cb, leg, &mut subsumes, &cb.imul_constraints[index].a);
			let b = process_operand(cb, leg, &mut subsumes, &cb.imul_constraints[index].b);
			let lo = process_operand(cb, leg, &mut subsumes, &cb.imul_constraints[index].lo);
			let hi = process_operand(cb, leg, &mut subsumes, &cb.imul_constraints[index].hi);
			AddedConstraint::Imul(WireImulConstraint { a, b, lo, hi })
		}
		ConstraintRef::Bmul { index } => {
			let bmul = &cb.bmul_constraints[index];
			let a_lo = process_operand(cb, leg, &mut subsumes, &bmul.a_lo);
			let a_hi = process_operand(cb, leg, &mut subsumes, &bmul.a_hi);
			let b_lo = process_operand(cb, leg, &mut subsumes, &bmul.b_lo);
			let b_hi = process_operand(cb, leg, &mut subsumes, &bmul.b_hi);
			let c_lo = process_operand(cb, leg, &mut subsumes, &bmul.c_lo);
			let c_hi = process_operand(cb, leg, &mut subsumes, &bmul.c_hi);
			AddedConstraint::Bmul(WireBmulConstraint {
				a_lo,
				a_hi,
				b_lo,
				b_hi,
				c_lo,
				c_hi,
			})
		}
		ConstraintRef::Linear { .. } => unreachable!(),
	};

	subsumes.sort_unstable();
	subsumes.dedup();

	Patch {
		subsumes,
		added: new_constraint,
	}
}

/// Build a patch for a committed linear definition.
///
/// Given the wire that defines a linear definition build a patch that replaces the original linear
/// definition and all definitions that could be inlined into it. Therefore, the returned
/// patch will replace the given linear definition and the cone of linear definitions it used.
///
/// The patch stays linear: the wire has to be committed, and its defining equation is enforced by
/// the Zero constraint
/// [`ConstraintBuilder::build`](crate::compiler::constraint_builder::ConstraintBuilder::build)
/// lowers it to.
fn build_committed_lin_def_patch(cb: &ConstraintBuilder, leg: &LeGraph, root: Wire) -> Patch {
	// `subsumes` is a list of constraints that become redundant with application of this patch.
	// The first redundant constraint is the linear definition that's being committed.
	let mut subsumes = vec![leg.lin_def_constraint_ref(root)];

	let old_operand = leg.lin_def_operand(cb, root);
	let new_operand = process_operand(cb, leg, &mut subsumes, old_operand);

	// Enforce: root = new_operand, over the inlined cone rather than the original definition.
	Patch {
		subsumes,
		added: AddedConstraint::Linear(WireLinearConstraint {
			rhs: new_operand,
			dst: root,
		}),
	}
}

fn process_operand(
	cb: &ConstraintBuilder,
	leg: &LeGraph,
	subsumes: &mut Vec<ConstraintRef>,
	old_operand: &WireOperand,
) -> WireOperand {
	let mut new_operand = WireOperand::new();
	for term in old_operand {
		process_term(cb, leg, &mut new_operand, subsumes, term.wire, term.shift);
	}
	new_operand
}

/// Recursively process a term, inlining non-committed linear definitions.
fn process_term(
	cb: &ConstraintBuilder,
	leg: &LeGraph,
	new_operand: &mut WireOperand,
	subsumes: &mut Vec<ConstraintRef>,
	wire: Wire,
	shift: Shift,
) {
	// Check if this wire is committed or not a linear def (i.e., opaque)
	if leg.commit_set().contains(wire) || !leg.is_lin_def(wire) {
		// This is a terminal or committed wire - add it to the result with the accumulated shift
		new_operand.push(ShiftedWire { wire, shift });
	} else {
		// This is a non-committed linear def - we need to inline it!
		let inner_operand = leg.lin_def_operand(cb, wire);
		let constraint_ref = leg.lin_def_constraint_ref(wire);
		subsumes.push(constraint_ref);

		// Distribute the current shift over all terms in the inner operand
		// This is crucial for correctness: shift(a ^ b) = shift(a) ^ shift(b)
		for inner_term in inner_operand {
			// Compose shifts: we're applying 'shift' to 'inner_term'
			// So we need Shift::compose(inner_term.shift, shift)
			match Shift::compose(inner_term.shift, shift) {
				Composition::Single(composed_shift) => {
					// Recursively process this term with the composed shift
					process_term(cb, leg, new_operand, subsumes, inner_term.wire, composed_shift);
				}
				// The two shifts clear the word, so the term is identically zero. An operand is
				// an XOR, so a zero term contributes nothing and the inlining simply drops it.
				Composition::Zero => {}
				// Needing both slots means the commit set inlined a definition it should have
				// committed: the two passes disagree about what is inlinable.
				Composition::Pair => {
					panic!(
						"Incompatible shifts during inlining: {:?} followed by {:?} for wire {:?}",
						inner_term.shift, shift, inner_term.wire
					);
				}
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use std::collections::BTreeMap;

	use super::*;
	use crate::compiler::{Wire, constraint_builder::expr, gate_fusion::Stat};

	/// Test helper to create a Wire with a given ID
	fn w(id: u32) -> Wire {
		Wire::from_u32(id)
	}

	/// Concise test helper that builds a circuit and verifies both commit decisions and expressions
	fn test_inlining(
		build_constraints: impl FnOnce(&mut ConstraintBuilder),
		expected_committed: &[Wire],
		expected_expressions: &[(Wire, Vec<ShiftedWire>)],
	) {
		let mut cb = ConstraintBuilder::new();
		build_constraints(&mut cb);

		let mut stat = Stat::default();
		let mut leg = LeGraph::new(&cb);
		crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

		// Verify commit set
		for &wire in expected_committed {
			assert!(
				leg.commit_set().contains(wire),
				"Wire {:?} should be committed but wasn't. Commit set: {:?}",
				wire,
				leg.commit_set()
			);
		}

		// Verify expressions expand correctly
		for &(wire, ref expected_expansion) in expected_expressions {
			let actual = expand_expression(&cb, &leg, wire);

			// Convert to BTreeMap for easier comparison (order-independent)
			let expected_map: BTreeMap<(Wire, Shift), usize> =
				expected_expansion
					.iter()
					.fold(BTreeMap::new(), |mut map, term| {
						*map.entry((term.wire, term.shift)).or_insert(0) += 1;
						map
					});

			let actual_map: BTreeMap<(Wire, Shift), usize> =
				actual.iter().fold(BTreeMap::new(), |mut map, term| {
					*map.entry((term.wire, term.shift)).or_insert(0) += 1;
					map
				});

			assert_eq!(
				actual_map, expected_map,
				"Wire {:?} expanded incorrectly.\nExpected: {:?}\nActual: {:?}",
				wire, expected_expansion, actual
			);
		}
	}

	#[test]
	fn test_rotr_identity_nested() {
		// y = a ^ b
		// z = rotr(y, 20)
		// t = rotr(z, 44)  // total 64 -> identity
		// Expect t expands to: a ^ b (no shifts)
		test_inlining(
			|cb| {
				cb.linear(expr::xor2(w(0), w(1)), w(2)); // y
				cb.linear(expr::rotr(w(2), 20), w(3)); // z
				cb.linear(expr::rotr(w(3), 44), w(4)); // t
				cb.and(w(4), w(5), w(6));
			},
			&[],
			&[(
				w(4),
				vec![
					ShiftedWire {
						wire: w(0),
						shift: Shift::IDENTITY,
					},
					ShiftedWire {
						wire: w(1),
						shift: Shift::IDENTITY,
					},
				],
			)],
		);
	}

	#[test]
	fn test_committed_lin_def_patch_stays_linear() {
		// Directly exercise build_committed_lin_def_patch: a committed definition keeps its linear
		// shape, leaving the AND-vs-Zero lowering to `ConstraintBuilder::build`.
		fn w(id: u32) -> Wire {
			Wire::from_u32(id)
		}

		let mut cb = ConstraintBuilder::new();
		cb.linear(expr::xor2(w(0), w(2)), w(3));

		let leg = LeGraph::new(&cb);
		let patch = super::build_committed_lin_def_patch(&cb, &leg, w(3));
		match patch.added {
			AddedConstraint::Linear(ref linc) => {
				assert!(!linc.rhs.is_empty());
				assert_eq!(linc.dst, w(3));
			}
			_ => panic!("expected linear constraint in committed patch"),
		}
	}

	#[test]
	fn test_mul_distinct_linears_all_fields() {
		// a, b, hi, lo each reference distinct linear defs; all inlinable (no shifts)
		fn w(id: u32) -> Wire {
			Wire::from_u32(id)
		}

		let mut cb = ConstraintBuilder::new();
		cb.linear(expr::xor2(w(0), w(1)), w(10)); // a_src
		cb.linear(expr::xor2(w(2), w(3)), w(11)); // b_src
		cb.linear(expr::xor2(w(4), w(5)), w(12)); // hi_src
		cb.linear(expr::xor2(w(6), w(7)), w(13)); // lo_src

		cb.imul(w(10), w(11), w(12), w(13));

		let mut stat = Stat::default();
		let mut leg = LeGraph::new(&cb);
		crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

		let patches = super::build(&cb, &leg);
		let mut cb2 = cb;
		super::apply_patches(&mut cb2, patches);

		assert_eq!(cb2.imul_constraints.len(), 1);
		let m = &cb2.imul_constraints[0];
		assert_eq!(m.a.len(), 2);
		assert_eq!(m.b.len(), 2);
		assert_eq!(m.hi.len(), 2);
		assert_eq!(m.lo.len(), 2);
	}

	#[test]
	fn test_bmul_distinct_linears_all_fields() {
		// a_lo, a_hi, b_lo, b_hi, c_lo, c_hi each reference distinct linear defs; all inlinable
		// (no shifts). This exercises the BMUL path through the legraph use-def harvest and the
		// patch builder.
		fn w(id: u32) -> Wire {
			Wire::from_u32(id)
		}

		let mut cb = ConstraintBuilder::new();
		cb.linear(expr::xor2(w(0), w(1)), w(20)); // a_lo_src
		cb.linear(expr::xor2(w(2), w(3)), w(21)); // a_hi_src
		cb.linear(expr::xor2(w(4), w(5)), w(22)); // b_lo_src
		cb.linear(expr::xor2(w(6), w(7)), w(23)); // b_hi_src
		cb.linear(expr::xor2(w(8), w(9)), w(24)); // c_lo_src
		cb.linear(expr::xor2(w(10), w(11)), w(25)); // c_hi_src

		cb.bmul(w(20), w(21), w(22), w(23), w(24), w(25));

		let mut stat = Stat::default();
		let mut leg = LeGraph::new(&cb);
		crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

		let patches = super::build(&cb, &leg);
		let mut cb2 = cb;
		super::apply_patches(&mut cb2, patches);

		// Every linear def is used exactly once (in the BMUL operand), so all six are inlined into
		// the single BMUL constraint rather than committed as AND constraints. Each operand expands
		// to its two XOR terms.
		assert_eq!(cb2.bmul_constraints.len(), 1);
		assert_eq!(cb2.and_constraints.len(), 0);
		let m = &cb2.bmul_constraints[0];
		assert_eq!(m.a_lo.len(), 2);
		assert_eq!(m.a_hi.len(), 2);
		assert_eq!(m.b_lo.len(), 2);
		assert_eq!(m.b_hi.len(), 2);
		assert_eq!(m.c_lo.len(), 2);
		assert_eq!(m.c_hi.len(), 2);
	}

	#[test]
	fn test_stress_shift_combinations_no_panic() {
		// Iterate over a small set of shift pairs; ensure commit_set + build/apply don't panic
		fn w(id: u32) -> Wire {
			Wire::from_u32(id)
		}

		use binius_core::constraint_system::ShiftVariant;

		use crate::compiler::constraint_builder::WireExprTerm;

		/// The expression term applying one shift to a wire.
		///
		/// The identity has no constructor of its own, so it reads the wire plainly.
		fn shifted_expr(wire: Wire, shift: Shift) -> WireExprTerm {
			let amount = shift.amount as u32;
			if shift.is_identity() {
				return wire.into();
			}
			match shift.variant {
				ShiftVariant::Sll => expr::sll(wire, amount),
				ShiftVariant::Slr => expr::srl(wire, amount),
				ShiftVariant::Sar => expr::sar(wire, amount),
				ShiftVariant::Rotr => expr::rotr(wire, amount),
				ShiftVariant::Sll32 => expr::sll32(wire, amount),
				ShiftVariant::Srl32 => expr::srl32(wire, amount),
				ShiftVariant::Sra32 => expr::sra32(wire, amount),
				ShiftVariant::Rotr32 => expr::rotr32(wire, amount),
			}
		}

		let shifts = [
			Shift::IDENTITY,
			Shift::sll(5),
			Shift::sll32(5),
			Shift::srl(5),
			Shift::srl32(5),
			Shift::sar(5),
			Shift::sra32(5),
			Shift::rotr(13),
			Shift::rotr32(13),
		];

		for (i, s1) in shifts.iter().enumerate() {
			for (j, s2) in shifts.iter().enumerate() {
				let mut cb = ConstraintBuilder::new();
				// y = shift1(x)
				cb.linear(shifted_expr(w(0), *s1), w(2));
				// z = shift2(y)
				cb.linear(shifted_expr(w(2), *s2), w(4));
				cb.and(w(4), w(5), w(6));

				let mut stat = Stat::default();
				let mut leg = LeGraph::new(&cb);
				crate::compiler::gate_fusion::commit_set::run_decide_commit_set(
					&mut leg, &mut stat,
				);
				let patches = super::build(&cb, &leg);
				let mut cb2 = cb;
				super::apply_patches(&mut cb2, patches);

				// Basic sanity: we should have at least one AND constraint after patches
				assert!(!cb2.and_constraints.is_empty(), "empty AND set for pair ({},{})", i, j);
			}
		}
	}

	/// Helper to expand an expression fully (for testing)
	fn expand_expression(cb: &ConstraintBuilder, leg: &LeGraph, wire: Wire) -> Vec<ShiftedWire> {
		let mut result = Vec::new();

		if !leg.is_lin_def(wire) {
			// Not a linear def - return as is
			result.push(ShiftedWire {
				wire,
				shift: Shift::IDENTITY,
			});
			return result;
		}

		let operand = leg.lin_def_operand(cb, wire);
		for term in operand {
			expand_term_recursive(cb, leg, &mut result, term.wire, term.shift);
		}
		result
	}

	fn expand_term_recursive(
		cb: &ConstraintBuilder,
		leg: &LeGraph,
		result: &mut Vec<ShiftedWire>,
		wire: Wire,
		shift: Shift,
	) {
		// Check if this wire is committed OR not a linear def (terminal)
		if !leg.is_lin_def(wire) || leg.commit_set().contains(wire) {
			// Terminal or committed - add it as is
			result.push(ShiftedWire { wire, shift });
		} else {
			// This is a non-committed linear def - expand recursively
			let inner = leg.lin_def_operand(cb, wire);
			for term in inner {
				match Shift::compose(term.shift, shift) {
					Composition::Single(composed) => {
						expand_term_recursive(cb, leg, result, term.wire, composed)
					}
					// A term the two shifts clear contributes nothing to the XOR.
					Composition::Zero => {}
					Composition::Pair => {
						panic!("the commit set only inlines definitions whose shifts compose")
					}
				}
			}
		}
	}

	/// Build a frequency map for an operand for order-independent comparison.
	fn operand_count_map(ops: &[ShiftedWire]) -> std::collections::BTreeMap<(Wire, Shift), usize> {
		let mut map = std::collections::BTreeMap::new();
		for t in ops {
			*map.entry((t.wire, t.shift)).or_insert(0) += 1;
		}
		map
	}

	fn assert_operand_eq(actual: &WireOperand, expected: &[ShiftedWire], ctx: &str) {
		let am = operand_count_map(actual.as_slice());
		let em = operand_count_map(expected);
		assert_eq!(
			am, em,
			"operand mismatch for {}\nexpected: {:?}\nactual:   {:?}",
			ctx, expected, actual
		);
	}

	// Now let's copy some test cases from plan_tests.rs and verify both commits and expressions

	#[test]
	fn test_simple_xor_inlining() {
		// Test: y = x ^ a, z = y ^ b, then z is used in AND
		// Both y and z should be inlinable into the AND constraint
		test_inlining(
			|cb| {
				// y = x ^ a
				cb.linear(expr::xor2(w(0), w(1)), w(2));
				// z = y ^ b
				cb.linear(expr::xor2(w(2), w(3)), w(4));
				// Use z in AND constraint (creates the root)
				cb.and(w(4), w(5), w(6));
			},
			&[],
			&[
				// z should expand to: x ^ a ^ b
				(
					w(4),
					vec![
						ShiftedWire {
							wire: w(0),
							shift: Shift::IDENTITY,
						},
						ShiftedWire {
							wire: w(1),
							shift: Shift::IDENTITY,
						},
						ShiftedWire {
							wire: w(3),
							shift: Shift::IDENTITY,
						},
					],
				),
				// y should expand to: x ^ a
				(
					w(2),
					vec![
						ShiftedWire {
							wire: w(0),
							shift: Shift::IDENTITY,
						},
						ShiftedWire {
							wire: w(1),
							shift: Shift::IDENTITY,
						},
					],
				),
			],
		);
	}

	#[test]
	fn test_shift_composition_same_type() {
		// Test: y = x << 10, z = y << 20
		// Shifts should compose: z = x << 30
		test_inlining(
			|cb| {
				// y = x << 10
				cb.linear(expr::sll(w(0), 10), w(1));
				// z = y << 20
				cb.linear(expr::sll(w(1), 20), w(2));
				// Use z in an AND constraint so it becomes a root
				cb.and(w(2), w(3), w(4));
			},
			&[],
			&[
				// z should expand to: x << 30
				(
					w(2),
					vec![ShiftedWire {
						wire: w(0),
						shift: Shift::sll(30),
					}],
				),
				// y should expand to: x << 10
				(
					w(1),
					vec![ShiftedWire {
						wire: w(0),
						shift: Shift::sll(10),
					}],
				),
			],
		);
	}

	#[test]
	fn test_rotr_distributes_over_xor() {
		// Test: y = a ^ b, z = rotr(y, 5)
		// Should distribute: z = rotr(a, 5) ^ rotr(b, 5)
		test_inlining(
			|cb| {
				// y = a ^ b
				cb.linear(expr::xor2(w(0), w(1)), w(2));
				// z = rotr(y, 5)
				cb.linear(expr::rotr(w(2), 5), w(3));
				// Use z in an AND constraint
				cb.and(w(3), w(4), w(5));
			},
			&[],
			&[
				// z should expand to: rotr(a, 5) ^ rotr(b, 5)
				(
					w(3),
					vec![
						ShiftedWire {
							wire: w(0),
							shift: Shift::rotr(5),
						},
						ShiftedWire {
							wire: w(1),
							shift: Shift::rotr(5),
						},
					],
				),
			],
		);
	}

	#[test]
	fn test_shift_composition_different_types() {
		// Test: y = x << 10, z = y >> 20
		// Different shift types cannot compose, y must be committed
		test_inlining(
			|cb| {
				// y = x << 10
				cb.linear(expr::sll(w(0), 10), w(1));
				// z = y >> 20
				cb.linear(expr::srl(w(1), 20), w(2));
				// Use z in an AND constraint
				cb.and(w(2), w(3), w(4));
			},
			&[w(1)], // y must be committed (incompatible shifts)
			&[
				// z should expand to: y >> 20 (y is committed, not inlined)
				(
					w(2),
					vec![ShiftedWire {
						wire: w(1),
						shift: Shift::srl(20),
					}],
				),
			],
		);
	}

	#[test]
	fn test_complex_xor_chain() {
		// Test: y = a ^ b ^ c, z = y ^ d ^ e
		// Both should be inlinable
		test_inlining(
			|cb| {
				// y = a ^ b ^ c
				cb.linear(expr::xor3(w(0), w(1), w(2)), w(3));
				// z = y ^ d ^ e
				cb.linear(expr::xor3(w(3), w(4), w(5)), w(6));
				// Use z in AND constraint
				cb.and(w(6), w(7), w(8));
			},
			&[],
			&[
				// z should expand to: a ^ b ^ c ^ d ^ e
				(
					w(6),
					vec![
						ShiftedWire {
							wire: w(0),
							shift: Shift::IDENTITY,
						},
						ShiftedWire {
							wire: w(1),
							shift: Shift::IDENTITY,
						},
						ShiftedWire {
							wire: w(2),
							shift: Shift::IDENTITY,
						},
						ShiftedWire {
							wire: w(4),
							shift: Shift::IDENTITY,
						},
						ShiftedWire {
							wire: w(5),
							shift: Shift::IDENTITY,
						},
					],
				),
			],
		);
	}

	#[test]
	fn test_apply_patches() {
		use super::ConstraintRef;

		// Create a constraint builder with various constraints
		let mut cb = ConstraintBuilder::new();

		// Add some AND constraints
		cb.and(w(0), w(1), w(2)); // index 0
		cb.and(w(3), w(4), w(5)); // index 1
		cb.and(w(6), w(7), w(8)); // index 2

		// Add some IMUL constraints
		cb.imul(w(9), w(10), w(12), w(11)); // index 0
		cb.imul(w(13), w(14), w(16), w(15)); // index 1

		// Add some LINEAR constraints
		cb.linear(expr::xor2(w(17), w(18)), w(19)); // index 0
		cb.linear(expr::xor2(w(20), w(21)), w(22)); // index 1

		// Create patches that:
		// 1. Replace AND constraint at index 1 with a new one
		// 2. Replace LINEAR constraint at index 0 with an AND constraint
		// 3. Replace IMUL constraint at index 0 with a new one
		let patches = vec![
			Patch {
				subsumes: vec![ConstraintRef::And { index: 1 }],
				added: AddedConstraint::And(WireAndConstraint {
					a: vec![ShiftedWire {
						wire: w(30),
						shift: Shift::IDENTITY,
					}]
					.into(),
					b: vec![ShiftedWire {
						wire: w(31),
						shift: Shift::IDENTITY,
					}]
					.into(),
					c: vec![ShiftedWire {
						wire: w(32),
						shift: Shift::IDENTITY,
					}]
					.into(),
				}),
			},
			Patch {
				subsumes: vec![ConstraintRef::Linear { index: 0 }],
				added: AddedConstraint::And(WireAndConstraint {
					a: vec![ShiftedWire {
						wire: w(33),
						shift: Shift::IDENTITY,
					}]
					.into(),
					b: vec![ShiftedWire {
						wire: w(34),
						shift: Shift::IDENTITY,
					}]
					.into(),
					c: vec![ShiftedWire {
						wire: w(35),
						shift: Shift::IDENTITY,
					}]
					.into(),
				}),
			},
			Patch {
				subsumes: vec![ConstraintRef::Imul { index: 0 }],
				added: AddedConstraint::Imul(WireImulConstraint {
					a: vec![ShiftedWire {
						wire: w(36),
						shift: Shift::IDENTITY,
					}]
					.into(),
					b: vec![ShiftedWire {
						wire: w(37),
						shift: Shift::IDENTITY,
					}]
					.into(),
					lo: vec![ShiftedWire {
						wire: w(38),
						shift: Shift::IDENTITY,
					}]
					.into(),
					hi: vec![ShiftedWire {
						wire: w(39),
						shift: Shift::IDENTITY,
					}]
					.into(),
				}),
			},
		];

		// Apply patches
		apply_patches(&mut cb, patches);

		// Verify results
		// AND constraints: originally 3, removed index 1, added 2 new ones = 4 total
		assert_eq!(cb.and_constraints.len(), 4);
		// Check that original constraints at indices 0 and 2 are preserved
		assert_eq!(cb.and_constraints[0].a[0].wire, w(0));
		assert_eq!(cb.and_constraints[1].a[0].wire, w(6)); // was index 2, now index 1
		// Check new constraints are added at the end
		assert_eq!(cb.and_constraints[2].a[0].wire, w(30));
		assert_eq!(cb.and_constraints[3].a[0].wire, w(33));

		// IMUL constraints: originally 2, removed index 0, added 1 new one = 2 total
		assert_eq!(cb.imul_constraints.len(), 2);
		// Check that original constraint at index 1 is preserved (now at index 0)
		assert_eq!(cb.imul_constraints[0].a[0].wire, w(13));
		// Check new constraint is added at the end
		assert_eq!(cb.imul_constraints[1].a[0].wire, w(36));

		// LINEAR constraints: originally 2, removed index 0 = 1 total
		assert_eq!(cb.linear_constraints.len(), 1);
		// Check that original constraint at index 1 is preserved (now at index 0)
		assert_eq!(cb.linear_constraints[0].rhs[0].wire, w(20));
	}

	#[test]
	fn test_patch_overlap_committed_and_non_linear() {
		// Build a scenario where a committed-linear patch and a non-linear patch both subsume
		// the same inner linear (overlap), and ensure apply_patches handles it correctly.
		//
		// t = a ^ b                    // linear (index 0)
		// y = srl(t, 10)               // linear (index 1), will be committed due to incompatible
		// use z = sll(y, 5)                // linear (index 2)
		// AND1 uses z (root)           // non-linear patch inlines z, subsumes z
		// AND2 uses t (root)           // non-linear patch inlines t, subsumes t
		// y committed patch subsumes y and also t (overlap with AND2 patch)
		let mut cb = ConstraintBuilder::new();

		// Inputs
		fn w(id: u32) -> Wire {
			Wire::from_u32(id)
		}
		// t = a ^ b
		cb.linear(expr::xor2(w(0), w(1)), w(2));
		// y = srl(t, 10)
		cb.linear(expr::srl(w(2), 10), w(3));
		// z = sll(y, 5)
		cb.linear(expr::sll(w(3), 5), w(4));

		// AND1: use z
		cb.and(w(4), w(5), w(6));
		// AND2: use t
		cb.and(w(2), w(7), w(8));

		let mut stat = Stat::default();
		let mut leg = LeGraph::new(&cb);
		crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

		// Sanity: y should be committed; t and z should not be
		assert!(leg.commit_set().contains(w(3)), "y should be committed");
		assert!(!leg.commit_set().contains(w(2)), "t should not be committed");
		assert!(!leg.commit_set().contains(w(4)), "z should not be committed");

		let patches = super::build(&cb, &leg);
		let mut cb2 = cb; // clone-by-move and apply patches
		super::apply_patches(&mut cb2, patches);

		// Expectations:
		// - AND constraints: start 2, both subsumed and replaced => 2
		assert_eq!(cb2.and_constraints.len(), 2);
		// - Linear constraints: t, y, z all subsumed, and committed y re-added as a linear
		//   definition over the inlined cone => 1 remaining
		assert_eq!(cb2.linear_constraints.len(), 1);
		assert_eq!(cb2.linear_constraints[0].dst, w(3));
	}

	#[test]
	fn test_mul_operand_duplicate_inlining() {
		// Build IMUL where the same linear def appears twice in a single operand:
		// y = x ^ c
		// IMUL.a = y ^ y ^ z  (duplicate y)
		// After inlining, expect: a = x ^ c ^ x ^ c ^ z (5 terms, preserving duplicates)
		fn w(id: u32) -> Wire {
			Wire::from_u32(id)
		}

		let mut cb = ConstraintBuilder::new();
		// y = x ^ c
		cb.linear(expr::xor2(w(0), w(1)), w(2));
		// IMUL: a = y ^ y ^ z; b = u; hi, lo are outputs
		cb.imul(
			crate::compiler::constraint_builder::expr::xor3(w(2), w(2), w(3)),
			w(4),
			w(5),
			w(6),
		);

		let mut stat = Stat::default();
		let mut leg = LeGraph::new(&cb);
		crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

		let patches = super::build(&cb, &leg);
		let mut cb2 = cb;
		super::apply_patches(&mut cb2, patches);

		// Verify results: one IMUL remains, and its operand a equals x ^ c ^ x ^ c ^ z
		assert_eq!(cb2.imul_constraints.len(), 1);
		let a = &cb2.imul_constraints[0].a;
		assert_operand_eq(
			a,
			&[
				ShiftedWire {
					wire: w(0),
					shift: Shift::IDENTITY,
				},
				ShiftedWire {
					wire: w(1),
					shift: Shift::IDENTITY,
				},
				ShiftedWire {
					wire: w(0),
					shift: Shift::IDENTITY,
				},
				ShiftedWire {
					wire: w(1),
					shift: Shift::IDENTITY,
				},
				ShiftedWire {
					wire: w(3),
					shift: Shift::IDENTITY,
				},
			],
			"mul.a",
		);
	}

	#[test]
	fn test_mul_mixed_inlinable_and_committed() {
		// Scenario: committed linear feeds one IMUL operand; inlinable linear feeds the other.
		// t_committed = sll(a, 40)
		// y = sll(t_committed, 30)  // uses committed
		// u = x ^ c                  // inlinable
		// IMUL: a=y, b=u
		fn w(id: u32) -> Wire {
			Wire::from_u32(id)
		}

		let mut cb = ConstraintBuilder::new();
		cb.linear(expr::sll(w(0), 40), w(1)); // t_committed
		cb.linear(expr::sll(w(1), 30), w(2)); // y
		cb.linear(expr::xor2(w(3), w(4)), w(5)); // u
		cb.imul(w(2), w(5), w(6), w(7));

		let mut stat = Stat::default();
		let mut leg = LeGraph::new(&cb);
		crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

		assert!(leg.commit_set().contains(w(1)), "t_committed should be committed");
		let patches = super::build(&cb, &leg);
		let mut cb2 = cb;
		super::apply_patches(&mut cb2, patches);

		assert_eq!(cb2.imul_constraints.len(), 1);
		let m = &cb2.imul_constraints[0];
		assert_operand_eq(
			&m.a,
			&[ShiftedWire {
				wire: w(1),
				shift: Shift::sll(30),
			}],
			"mul.a (committed)",
		);
		assert_operand_eq(
			&m.b,
			&[
				ShiftedWire {
					wire: w(3),
					shift: Shift::IDENTITY,
				},
				ShiftedWire {
					wire: w(4),
					shift: Shift::IDENTITY,
				},
			],
			"mul.b (inlinable)",
		);
	}

	#[test]
	fn test_mul_hi_lo_mixed_inlinable_committed() {
		// hi comes from a committed path; lo from an inlinable XOR
		fn w(id: u32) -> Wire {
			Wire::from_u32(id)
		}

		let mut cb = ConstraintBuilder::new();
		// committed producer
		cb.linear(expr::sll(w(0), 48), w(1)); // t
		cb.linear(expr::sll(w(1), 20), w(2)); // hi_src (should commit t)
		// inlinable lo_src = x ^ c
		cb.linear(expr::xor2(w(3), w(4)), w(5));
		// build IMUL: a,b plain; hi=hi_src; lo=lo_src
		cb.imul(w(6), w(7), w(2), w(5));

		let mut stat = Stat::default();
		let mut leg = LeGraph::new(&cb);
		crate::compiler::gate_fusion::commit_set::run_decide_commit_set(&mut leg, &mut stat);

		assert!(leg.commit_set().contains(w(1)), "inner t should be committed");
		let patches = super::build(&cb, &leg);
		let mut cb2 = cb;
		super::apply_patches(&mut cb2, patches);

		assert_eq!(cb2.imul_constraints.len(), 1);
		let m = &cb2.imul_constraints[0];
		assert_operand_eq(
			&m.hi,
			&[ShiftedWire {
				wire: w(1),
				shift: Shift::sll(20),
			}],
			"mul.hi (committed)",
		);
		assert_operand_eq(
			&m.lo,
			&[
				ShiftedWire {
					wire: w(3),
					shift: Shift::IDENTITY,
				},
				ShiftedWire {
					wire: w(4),
					shift: Shift::IDENTITY,
				},
			],
			"mul.lo (inlinable)",
		);
	}
}
