// Copyright 2026 The Binius Developers

use std::iter;

use super::{ValueVec, constraint::Operand};
use crate::{
	ConstraintSystem, Word,
	error::{ConstraintSystemError, VerificationM4Error},
};

/// A constraint system that represents a single chip in a [`ConstraintSystemM4`].
///
/// The [`crate::constraint_system::ShiftedValueIndex`] values in the constraints and in the chip
/// calls name words of the embedded constraint system's value vector, each index counting within
/// its own segment. See [`ConstraintSystem`] for the exact layout.
///
/// ## Validity criteria
/// * every operand of every chip call in `chip_calls` references a value of this chip
///
/// The `chip_id` of a chip call names a chip of the enclosing system, which this type does not
/// know; [`ConstraintSystemM4::validate`] is what range-checks it.
#[derive(Debug, Clone)]
pub struct EmbeddedConstraintSystem {
	/// The constraints one instance of the chip must satisfy.
	pub cs: ConstraintSystem,
	/// The chips this one delegates subrelations to, one entry per call per instance.
	pub chip_calls: Vec<ChipCall>,
}

impl EmbeddedConstraintSystem {
	/// Checks the constraint system and the chip-call operands over it.
	pub fn validate(&self) -> Result<(), ConstraintSystemError> {
		self.cs.validate()?;

		for (call_index, chip_call) in self.chip_calls.iter().enumerate() {
			for (operand_index, operand) in chip_call.inout.iter().enumerate() {
				if let Some(source) = self.cs.operand_fault(operand) {
					return Err(ConstraintSystemError::ChipCallOperand {
						call_index,
						operand_index,
						source,
					});
				}
			}
		}

		Ok(())
	}
}

/// One invocation of a chip by the constraint system holding this call.
#[derive(Debug, Clone)]
pub struct ChipCall {
	/// The ID of the chip being called: its index in [`ConstraintSystemM4::chips`].
	pub chip_id: usize,
	/// The words passed as the callee's inout values, positionally, as operands over the caller's
	/// value vector.
	///
	/// There is at most one operand per inout value of the callee; the values past them are
	/// constrained to zero.
	pub inout: Vec<Operand>,
}

/// An M4 constraint system.
///
/// An M4 constraint system is essentially defined by the composition of chips, each of which
/// validates a relation on its local inout values. Chips can delegate subrelation constraints to
/// other chips via chip calls.
///
/// Validity invariants:
/// - all embedded constraint systems in `chips` must have a chip_id value that indexes into `chips`
///
/// [`Self::validate`] accepts any acyclic call graph, but call positions are defined by
/// enumerating the chips in ID order, and witness generation walks them the same way — so only a
/// system whose calls run to higher IDs is provable. The frontend's
/// `CircuitM4::validate` is what enforces that ordering.
pub struct ConstraintSystemM4 {
	/// The entry point of the system. It calls chips, but no chip ID names it, so nothing calls
	/// it.
	pub main: EmbeddedConstraintSystem,
	/// The chips, indexed by chip ID, each paired with its number of active instances.
	///
	/// A chip runs once per call that reaches it, and those instances are the active ones: only
	/// they have their own chip calls enforced. The instances past them pad the count and are
	/// matched by no call.
	pub chips: Vec<(EmbeddedConstraintSystem, usize)>,
}

impl ConstraintSystemM4 {
	/// Checks every chip and every chip call, and rejects call-graph cycles.
	pub fn validate(&self) -> Result<(), ConstraintSystemError> {
		let n_chips = self.chips.len();

		self.main.validate()?;
		self.validate_calls(None, &self.main.chip_calls)?;

		// The chips each chip calls, indexed by the caller's chip ID.
		let mut callees = vec![Vec::new(); n_chips];
		for (chip_index, (chip, _)) in self.chips.iter().enumerate() {
			chip.validate()?;
			self.validate_calls(Some(chip_index), &chip.chip_calls)?;
			callees[chip_index].extend(chip.chip_calls.iter().map(|call| call.chip_id));
		}

		if !is_acyclic(&callees) {
			return Err(ConstraintSystemError::CyclicChipCalls);
		}

		Ok(())
	}

	/// Checks that one caller's calls name existing chips and pass no more operands than the
	/// callee has inout values.
	///
	/// A call may pass fewer: the inout values past its operands are constrained to zero.
	fn validate_calls(
		&self,
		chip_index: Option<usize>,
		calls: &[ChipCall],
	) -> Result<(), ConstraintSystemError> {
		let n_chips = self.chips.len();
		for (call_index, call) in calls.iter().enumerate() {
			if call.chip_id >= n_chips {
				return Err(ConstraintSystemError::OutOfRangeChipId {
					chip_index,
					chip_id: call.chip_id,
					n_chips,
				});
			}
			let n_inout = self.chips[call.chip_id].0.cs.n_inout;
			if call.inout.len() > n_inout {
				return Err(ConstraintSystemError::WrongCallArity {
					chip_index,
					call_index,
					chip_id: call.chip_id,
					arity: call.inout.len(),
					n_inout,
				});
			}
		}
		Ok(())
	}

	/// Checks that a witness satisfies this system.
	///
	/// The witness is the main chip's value vector and, per chip, one value vector per instance.
	/// It must satisfy:
	///
	/// - the main chip's constraints, on `main`;
	/// - each chip's constraints, on every one of its instances — the instances past the active
	///   ones included, since every instance is committed;
	/// - every chip call, by the instance at the call's position: counting main's calls first and
	///   then the calls of each chip's active instances in ID, instance and call order, invocation
	///   `i` of a chip passes exactly the inout words its instance `i` holds. An inout value the
	///   call has no operand for must hold zero.
	///
	/// This is the reference the proving protocol's argument is checked against, in the manner of
	/// [`ConstraintSystem::verify`].
	///
	/// Malformed systems are [`Self::validate`]'s to reject; verifying one may panic or
	/// misreport.
	///
	/// # Errors
	///
	/// Reports the first failure found, in the order listed above.
	pub fn verify(
		&self,
		main: &ValueVec,
		chip_instances: &[Vec<ValueVec>],
	) -> Result<(), VerificationM4Error> {
		if chip_instances.len() != self.chips.len() {
			return Err(VerificationM4Error::WrongChipCount {
				n_witness_chips: chip_instances.len(),
				n_chips: self.chips.len(),
			});
		}

		self.main.cs.verify(main)?;
		for (chip_id, ((chip, n_active), instances)) in
			iter::zip(&self.chips, chip_instances).enumerate()
		{
			if instances.len() < *n_active {
				return Err(VerificationM4Error::MissingInstances {
					chip_id,
					n_instances: instances.len(),
					n_active: *n_active,
				});
			}
			for (instance, values) in instances.iter().enumerate() {
				chip.cs
					.verify(values)
					.map_err(|source| VerificationM4Error::ChipInstance {
						chip_id,
						instance,
						source,
					})?;
			}
		}

		// The next instance each chip's calls are served by, advanced call by call so that
		// invocation `i` lands on instance `i`.
		let mut cursor = vec![0usize; self.chips.len()];
		self.check_caller(None, &self.main.chip_calls, main, &mut cursor, chip_instances)?;
		for (caller_chip, (chip, n_active)) in self.chips.iter().enumerate() {
			for caller_instance in 0..*n_active {
				self.check_caller(
					Some((caller_chip, caller_instance)),
					&chip.chip_calls,
					&chip_instances[caller_chip][caller_instance],
					&mut cursor,
					chip_instances,
				)?;
			}
		}

		for (chip_id, ((_, n_active), &n_invocations)) in
			iter::zip(&self.chips, &cursor).enumerate()
		{
			if n_invocations != *n_active {
				return Err(VerificationM4Error::WrongInvocationCount {
					chip_id,
					n_invocations,
					n_active: *n_active,
				});
			}
		}

		Ok(())
	}

	/// Checks one caller's calls against the instances at their positions, advancing the cursors.
	///
	/// `caller` names the caller for diagnostics: a chip instance, or `None` for the main chip.
	/// `values` is that caller's value vector, which the call operands are evaluated on. A call
	/// past its chip's active instances is counted but compared to nothing; the caller reports the
	/// overshoot when the cursors are read back.
	fn check_caller(
		&self,
		caller: Option<(usize, usize)>,
		calls: &[ChipCall],
		values: &ValueVec,
		cursor: &mut [usize],
		chip_instances: &[Vec<ValueVec>],
	) -> Result<(), VerificationM4Error> {
		let (caller_chip, caller_instance) = match caller {
			Some((chip, instance)) => (Some(chip), instance),
			None => (None, 0),
		};
		for (call_index, call) in calls.iter().enumerate() {
			let chip_id = call.chip_id;
			let (chip, n_active) = &self.chips[chip_id];
			let row = cursor[chip_id];
			cursor[chip_id] += 1;
			if row >= *n_active {
				continue;
			}

			let n_inout = chip.cs.n_inout;
			let served = chip_instances[chip_id][row].inout();
			for word in 0..n_inout {
				let passed = call
					.inout
					.get(word)
					.map(|operand| values.eval_operand(operand))
					.unwrap_or(Word::ZERO);
				if served[word] != passed {
					return Err(VerificationM4Error::CallMismatch {
						chip_id,
						row,
						caller_chip,
						caller_instance,
						call_index,
						word,
						passed: passed.as_u64(),
						served: served[word].as_u64(),
					});
				}
			}
		}
		Ok(())
	}
}

/// Checks that the call graph given by the callee list of each chip is acyclic.
///
/// This is Kahn's algorithm: chips with no remaining callers are removed one at a time, and any
/// chip still left when none remain lies on a cycle. A chip that calls itself is a cycle.
fn is_acyclic(callees: &[Vec<usize>]) -> bool {
	let mut n_callers = vec![0usize; callees.len()];
	for &callee in callees.iter().flatten() {
		n_callers[callee] += 1;
	}

	let mut uncalled = (0..callees.len())
		.filter(|&chip| n_callers[chip] == 0)
		.collect::<Vec<_>>();

	let mut n_removed = 0;
	while let Some(chip) = uncalled.pop() {
		n_removed += 1;
		for &callee in &callees[chip] {
			n_callers[callee] -= 1;
			if n_callers[callee] == 0 {
				uncalled.push(callee);
			}
		}
	}

	n_removed == callees.len()
}

#[cfg(test)]
mod tests {
	use super::{
		super::{ShiftedValueIndex, ValueIndex, ValueSegment, ValueVecLayout},
		*,
	};
	use crate::error::OperandFault;

	/// A chip that calls each of `callees` once, over a value vector of 8 inout and 8 private
	/// words.
	fn chip(callees: &[usize]) -> EmbeddedConstraintSystem {
		let cs = ValueVecLayout {
			n_const: 0,
			n_inout: 8,
			n_witness: 8,
			n_internal: 0,
			n_scratch: 0,
		}
		.constraint_system_shape(vec![]);
		let chip_calls = callees
			.iter()
			.map(|&chip_id| ChipCall {
				chip_id,
				inout: vec![],
			})
			.collect();
		EmbeddedConstraintSystem { cs, chip_calls }
	}

	/// A system whose main chip calls each of `main_callees` once, over the given chips.
	///
	/// The active-instance counts are not what `validate` checks, so every chip declares one.
	fn system(main_callees: &[usize], chips: Vec<EmbeddedConstraintSystem>) -> ConstraintSystemM4 {
		ConstraintSystemM4 {
			main: chip(main_callees),
			chips: chips.into_iter().map(|chip| (chip, 1)).collect(),
		}
	}

	#[test]
	fn validate_rejects_a_chip_call_operand_past_its_own_segment() {
		// Inout index 8 is the word after the chip's eight inout values. Its position in the value
		// vector is one a private value occupies, so only a per-segment check catches it.
		let mut cs = system(&[0], vec![chip(&[])]);
		cs.main.chip_calls[0].inout = vec![vec![ShiftedValueIndex::plain(ValueIndex::inout(8))]];
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::ChipCallOperand {
				call_index: 0,
				operand_index: 0,
				source: OperandFault::OutOfRangeValueIndex {
					segment: ValueSegment::InOut,
					value_index: 8,
					segment_len: 8,
				},
			})
		));
	}

	#[test]
	fn validate_rejects_a_call_passing_more_operands_than_the_callee_takes() {
		// The chips have eight inout values; nine operands leave one with no value to land in.
		let mut cs = system(&[0], vec![chip(&[])]);
		cs.main.chip_calls[0].inout = vec![vec![]; 9];
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::WrongCallArity {
				chip_index: None,
				call_index: 0,
				chip_id: 0,
				arity: 9,
				n_inout: 8,
			})
		));
	}

	#[test]
	fn validate_accepts_an_acyclic_call_graph() {
		// Main calls 0, which calls 1 and 2; 1 calls 2, and 2 is a leaf.
		let cs = system(&[0], vec![chip(&[1, 2]), chip(&[2]), chip(&[])]);
		cs.validate().unwrap();
	}

	#[test]
	fn validate_rejects_a_chip_that_calls_itself() {
		let cs = system(&[0], vec![chip(&[0])]);
		assert!(matches!(cs.validate(), Err(ConstraintSystemError::CyclicChipCalls)));
	}

	#[test]
	fn validate_rejects_a_cycle_reachable_from_an_uncalled_chip() {
		// Chip 0 is called by nobody, so removing it leaves the cycle 1 -> 2 -> 1.
		let cs = system(&[0], vec![chip(&[1]), chip(&[2]), chip(&[1])]);
		assert!(matches!(cs.validate(), Err(ConstraintSystemError::CyclicChipCalls)));
	}

	#[test]
	fn validate_rejects_a_call_to_a_chip_that_does_not_exist() {
		let cs = system(&[0], vec![chip(&[0, 7])]);
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::OutOfRangeChipId {
				chip_index: Some(0),
				chip_id: 7,
				n_chips: 1,
			})
		));
	}

	#[test]
	fn validate_rejects_a_main_call_to_a_chip_that_does_not_exist() {
		let cs = system(&[7], vec![chip(&[])]);
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::OutOfRangeChipId {
				chip_index: None,
				chip_id: 7,
				n_chips: 1,
			})
		));
	}
}
