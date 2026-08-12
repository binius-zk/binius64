// Copyright 2026 The Binius Developers

use std::{borrow::Cow, iter};

use super::{ValueVec, constraint::Operand};
use crate::{
	ConstraintSystem, Word,
	error::{ConstraintSystemError, VerificationM4Error},
};

mod witness;

pub use witness::WitnessM4;

/// The chip instances of an M4 witness, addressed by chip ID and row.
///
/// [`ConstraintSystemM4::verify`] reads one instance at a time and never holds two, so this is all
/// it needs of a witness. A witness that stores its instances packed — as the prover's tables do,
/// one column per instance — can therefore serve them one at a time rather than expanding the whole
/// witness into value vectors first, which for a system of any size is most of its memory.
///
/// Instances are served, not lent, because a packed witness has to build one to hand it over. One
/// that already holds value vectors lends them and pays nothing.
pub trait ChipInstances {
	/// The number of chips the witness covers, which must be the number the system has.
	fn n_chips(&self) -> usize;

	/// The number of instances of the given chip, which must be at least its active-instance count.
	fn n_instances(&self, chip_id: usize) -> usize;

	/// The value vector of one instance of one chip.
	fn instance(&self, chip_id: usize, row: usize) -> Cow<'_, ValueVec>;
}

impl ChipInstances for [Vec<ValueVec>] {
	fn n_chips(&self) -> usize {
		self.len()
	}

	fn n_instances(&self, chip_id: usize) -> usize {
		self[chip_id].len()
	}

	fn instance(&self, chip_id: usize, row: usize) -> Cow<'_, ValueVec> {
		Cow::Borrowed(&self[chip_id][row])
	}
}

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
	/// The instance of the callee that the caller's first instance invokes.
	///
	/// A call site runs once per instance of its caller, so it claims the callee's instances
	/// `first_instance..first_instance + n_active`, where `n_active` is the *caller's* own
	/// active-instance count: the caller's instance `i` invokes instance `first_instance + i`.
	/// Main runs once, so each of its calls claims a single instance.
	///
	/// Like the active-instance counts, this is denormalized — the call graph already says it, and
	/// [`ConstraintSystemM4::validate`] holds the two to each other.
	pub first_instance: usize,
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
/// - the chip calls must claim each chip's active instances exactly once between them
///
/// The chips are in topological order: a chip calls only chips with a higher ID, and main, which
/// is not one of them, calls any. Enumerating the chips in ID order therefore reaches every caller
/// of a chip before the chip itself, which is what lets one pass assign the instances and one pass
/// generate the witness. [`Self::validate`] requires the ordering, which makes the call graph
/// acyclic as a consequence.
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
	/// Checks every chip and every chip call, requires the chips to be in topological order, and
	/// holds the instances the calls name against the ones the call graph gives them.
	pub fn validate(&self) -> Result<(), ConstraintSystemError> {
		self.main.validate()?;
		self.validate_calls(None, &self.main.chip_calls)?;

		for (chip_index, (chip, _)) in self.chips.iter().enumerate() {
			chip.validate()?;
			self.validate_calls(Some(chip_index), &chip.chip_calls)?;
		}

		self.validate_instances()
	}

	/// Checks that one caller's calls name existing chips, run only to later chips, and pass no
	/// more operands than the callee has inout values.
	///
	/// A call may pass fewer: the inout values past its operands are constrained to zero.
	///
	/// Main is not one of the numbered chips and runs before all of them, so it may call any.
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
			if let Some(caller) = chip_index
				&& call.chip_id <= caller
			{
				return Err(ConstraintSystemError::CallOutOfOrder {
					chip_index: caller,
					callee: call.chip_id,
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

	/// Checks that the calls claim each chip's active instances exactly once between them.
	///
	/// A call site claims one instance of its callee per instance of its caller, and the claims are
	/// handed out in the order the chips are populated: main's calls first, then the calls of each
	/// chip in ID order. Only main and lower-numbered chips call a chip, so one pass in ID order
	/// settles a chip's own claims before reading them.
	///
	/// This runs after [`Self::validate_calls`] has range-checked every `chip_id` and rejected the
	/// calls that run backwards, both of which it relies on.
	fn validate_instances(&self) -> Result<(), ConstraintSystemError> {
		let mut n_claimed = vec![0usize; self.chips.len()];
		let callers = iter::once((None, &self.main, 1)).chain(
			self.chips
				.iter()
				.enumerate()
				.map(|(chip_index, (chip, n_active))| (Some(chip_index), chip, *n_active)),
		);
		for (chip_index, caller, n_active) in callers {
			for (call_index, call) in caller.chip_calls.iter().enumerate() {
				let claimed = &mut n_claimed[call.chip_id];
				if call.first_instance != *claimed {
					return Err(ConstraintSystemError::WrongCallInstance {
						chip_index,
						call_index,
						first_instance: call.first_instance,
						expected: *claimed,
					});
				}

				// The claims multiply down the call graph — a chain whose every chip calls the
				// next twice reaches `2^depth` — so a system of a few dozen chips can outgrow a
				// `usize`. Counting it out unchecked would wrap to a total that a declared count
				// could then agree with.
				*claimed = claimed.checked_add(n_active).ok_or(
					ConstraintSystemError::TooManyInstances {
						chip_id: call.chip_id,
					},
				)?;
			}
		}

		for (chip_id, ((_, n_active), &claimed)) in iter::zip(&self.chips, &n_claimed).enumerate() {
			if claimed != *n_active {
				return Err(ConstraintSystemError::WrongActiveInstanceCount {
					chip_id,
					declared: *n_active,
					actual: claimed,
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
	/// - every chip call, by the instance it names: the caller's instance `i` invokes the callee's
	///   instance [`first_instance`](ChipCall::first_instance) plus `i`, and passes exactly the
	///   inout words that instance holds. An inout value the call has no operand for must hold
	///   zero.
	///
	/// This is the reference the proving protocol's argument is checked against, in the manner of
	/// [`ConstraintSystem::verify`].
	///
	/// Instances are read through [`ChipInstances`] one at a time, so a witness that stores them
	/// packed is never expanded whole. The price is that an instance serving a call is built again
	/// when the call is checked, having already been built for its own constraints.
	///
	/// Malformed systems are [`Self::validate`]'s to reject; verifying one may panic, misreport, or
	/// pass. In particular a call passing more operands than the callee has inout values — which
	/// [`Self::validate`] rejects — has the operands past them ignored here.
	///
	/// # Errors
	///
	/// Reports the first failure found, in the order listed above.
	pub fn verify<I: ChipInstances + ?Sized>(
		&self,
		main: &ValueVec,
		chip_instances: &I,
	) -> Result<(), VerificationM4Error> {
		if chip_instances.n_chips() != self.chips.len() {
			return Err(VerificationM4Error::WrongChipCount {
				n_witness_chips: chip_instances.n_chips(),
				n_chips: self.chips.len(),
			});
		}

		self.main.cs.verify(main)?;
		for (chip_id, (chip, n_active)) in self.chips.iter().enumerate() {
			let n_instances = chip_instances.n_instances(chip_id);
			if n_instances < *n_active {
				return Err(VerificationM4Error::MissingInstances {
					chip_id,
					n_instances,
					n_active: *n_active,
				});
			}
			for instance in 0..n_instances {
				chip.cs
					.verify(&chip_instances.instance(chip_id, instance))
					.map_err(|source| VerificationM4Error::ChipInstance {
						chip_id,
						instance,
						source,
					})?;
			}
		}

		self.check_caller(None, &self.main.chip_calls, main, chip_instances)?;
		for (caller_chip, (chip, n_active)) in self.chips.iter().enumerate() {
			for caller_instance in 0..*n_active {
				let values = chip_instances.instance(caller_chip, caller_instance);
				self.check_caller(
					Some((caller_chip, caller_instance)),
					&chip.chip_calls,
					&values,
					chip_instances,
				)?;
			}
		}

		Ok(())
	}

	/// Checks one caller's calls against the instances they name.
	///
	/// `caller` names the caller for diagnostics: a chip instance, or `None` for the main chip.
	/// `values` is that caller's value vector, which the call operands are evaluated on.
	///
	/// The instance a call reaches is its own, offset by which instance of the caller is making it.
	/// That every one of them exists is [`Self::validate`]'s to establish.
	fn check_caller<I: ChipInstances + ?Sized>(
		&self,
		caller: Option<(usize, usize)>,
		calls: &[ChipCall],
		values: &ValueVec,
		chip_instances: &I,
	) -> Result<(), VerificationM4Error> {
		// Main runs once, so its calls are the ones its single instance makes.
		let caller_instance = caller.map_or(0, |(_, instance)| instance);
		for (call_index, call) in calls.iter().enumerate() {
			let chip_id = call.chip_id;
			let chip = &self.chips[chip_id].0;
			let row = call.first_instance + caller_instance;

			// Only the callee's own inout values are compared. An operand past them is one
			// `validate` rejects, and nothing here would have a word to compare it to.
			let n_inout = chip.cs.n_inout;
			let instance = chip_instances.instance(chip_id, row);
			let served = instance.inout();
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
						caller,
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

#[cfg(test)]
mod tests {
	use super::{
		super::{ShiftedValueIndex, ValueIndex, ValueSegment, ValueVecLayout, ZeroConstraint},
		*,
	};
	use crate::error::{OperandFault, VerificationError};

	/// The value-vector layout every chip in these tests is shaped by.
	const LAYOUT: ValueVecLayout = ValueVecLayout {
		n_const: 0,
		n_inout: 8,
		n_witness: 8,
		n_internal: 0,
		n_scratch: 0,
	};

	/// A chip that calls each of `callees` once, over a value vector of 8 inout and 8 private
	/// words.
	///
	/// The calls name instance 0 until [`system`] gives them the ones the call graph does.
	fn chip(callees: &[usize]) -> EmbeddedConstraintSystem {
		let cs = LAYOUT.constraint_system_shape(vec![]);
		let chip_calls = callees
			.iter()
			.map(|&chip_id| ChipCall {
				chip_id,
				first_instance: 0,
				inout: vec![],
			})
			.collect();
		EmbeddedConstraintSystem { cs, chip_calls }
	}

	/// An all-zero instance of a chip, which satisfies one with no constraints of its own.
	fn instance() -> ValueVec {
		ValueVec::new(&LAYOUT)
	}

	/// A system whose main chip calls each of `main_callees` once, over the given chips, with the
	/// instances and active-instance counts the call graph gives them.
	///
	/// This is the assignment `CircuitM4::recompute_instances` writes in the frontend, redone here
	/// because these systems are built by hand rather than lowered from a circuit. Like it, this
	/// panics on a call to a chip the system does not have, so a test wanting one corrupts a
	/// system built without it.
	fn system(main_callees: &[usize], chips: Vec<EmbeddedConstraintSystem>) -> ConstraintSystemM4 {
		let mut cs = ConstraintSystemM4 {
			main: chip(main_callees),
			chips: chips.into_iter().map(|chip| (chip, 0)).collect(),
		};

		let mut n_calls = vec![0usize; cs.chips.len()];
		for call in &mut cs.main.chip_calls {
			call.first_instance = n_calls[call.chip_id];
			n_calls[call.chip_id] += 1;
		}
		for chip_index in 0..cs.chips.len() {
			let n_active = n_calls[chip_index];
			cs.chips[chip_index].1 = n_active;
			for call in &mut cs.chips[chip_index].0.chip_calls {
				call.first_instance = n_calls[call.chip_id];
				n_calls[call.chip_id] = n_calls[call.chip_id].saturating_add(n_active);
			}
		}
		cs
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
	fn validate_accepts_calls_running_only_to_later_chips() {
		// Main calls 0, which calls 1 and 2; 1 calls 2, and 2 is a leaf. Main is not one of the
		// numbered chips, so its call to 0 is not a backward one.
		let cs = system(&[0], vec![chip(&[1, 2]), chip(&[2]), chip(&[])]);
		cs.validate().unwrap();
	}

	#[test]
	fn validate_rejects_a_chip_that_calls_itself() {
		// A chip is not later than itself, so self-recursion is out of order rather than a
		// separate case.
		let cs = system(&[0], vec![chip(&[0])]);
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::CallOutOfOrder {
				chip_index: 0,
				callee: 0,
			})
		));
	}

	#[test]
	fn validate_rejects_a_chip_that_calls_an_earlier_chip() {
		// Chip 2 calls back to chip 1, which is populated before it. The graph as a whole need not
		// be cyclic for this to break the one pass over the chips.
		let cs = system(&[0], vec![chip(&[1]), chip(&[2]), chip(&[1])]);
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::CallOutOfOrder {
				chip_index: 2,
				callee: 1,
			})
		));
	}

	#[test]
	fn validate_rejects_a_call_to_a_chip_that_does_not_exist() {
		// Chip 0's first call is to a later chip, so the out-of-range second one is the only
		// fault and the range check is what has to catch it.
		let mut cs = system(&[0], vec![chip(&[1]), chip(&[])]);
		cs.chips[0].0.chip_calls.push(ChipCall {
			chip_id: 7,
			first_instance: 0,
			inout: vec![],
		});
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::OutOfRangeChipId {
				chip_index: Some(0),
				chip_id: 7,
				n_chips: 2,
			})
		));
	}

	#[test]
	fn validate_rejects_a_main_call_to_a_chip_that_does_not_exist() {
		let mut cs = system(&[0], vec![chip(&[])]);
		cs.main.chip_calls[0].chip_id = 7;
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::OutOfRangeChipId {
				chip_index: None,
				chip_id: 7,
				n_chips: 1,
			})
		));
	}

	// A call site claims one instance of its callee per instance of its caller, so the instance a
	// call names is the one the calls before it leave free.
	#[test]
	fn validate_rejects_a_call_naming_the_wrong_instance() {
		let mut cs = system(&[0, 0], vec![chip(&[])]);
		cs.main.chip_calls[1].first_instance = 0;
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::WrongCallInstance {
				chip_index: None,
				call_index: 1,
				first_instance: 0,
				expected: 1,
			})
		));
	}

	// The call-to-instance map is a bijection, not just an injection: a chip claimed by more
	// invocations than it has active instances has calls that no row answers for.
	#[test]
	fn validate_rejects_more_invocations_than_a_chip_has_active_instances() {
		// Main calls chip 0 twice; declaring one active instance leaves the second call unanswered.
		let mut cs = system(&[0, 0], vec![chip(&[])]);
		cs.chips[0].1 = 1;
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::WrongActiveInstanceCount {
				chip_id: 0,
				declared: 1,
				actual: 2,
			})
		));
	}

	// Instance counts multiply down the call graph, so a chain of a few dozen chips outgrows a
	// `usize`. Counting it out unchecked would wrap to a plausible-looking total.
	#[test]
	fn validate_rejects_an_instance_count_that_outgrows_a_usize() {
		// A chain of 70 chips, each calling the next twice, so chip `i` is reached 2^i times.
		const DEPTH: usize = 70;
		let chips = (0..DEPTH)
			.map(|i| {
				let callees = if i + 1 < DEPTH {
					vec![i + 1, i + 1]
				} else {
					vec![]
				};
				chip(&callees)
			})
			.collect();

		// Chip 63 is reached 2^63 times and calls chip 64 twice, which is where the count leaves
		// the range.
		let cs = system(&[0], chips);
		assert!(matches!(
			cs.validate(),
			Err(ConstraintSystemError::TooManyInstances { chip_id: 64 })
		));
	}

	// A call passing fewer operands than the callee has inout values constrains the rest to zero,
	// so an all-zero instance serves a call that passes nothing at all.
	#[test]
	fn verify_constrains_the_inout_values_a_call_passes_no_operand_for() {
		let cs = system(&[0], vec![chip(&[])]);
		let main = instance();
		cs.verify(&main, &[vec![instance()]][..]).unwrap();

		// Give the instance a nonzero word where the call passes nothing, and the zero-fill is what
		// it stops matching.
		let mut served = instance();
		served[ValueIndex::inout(3)] = Word::from_u64(0xdead);
		let err = cs.verify(&main, &[vec![served]][..]).unwrap_err();
		assert!(
			matches!(
				err,
				VerificationM4Error::CallMismatch {
					chip_id: 0,
					row: 0,
					caller: None,
					word: 3,
					passed: 0,
					served: 0xdead,
					..
				}
			),
			"{err}"
		);
	}

	#[test]
	fn verify_rejects_a_witness_covering_the_wrong_number_of_chips() {
		let cs = system(&[0], vec![chip(&[])]);
		let err = cs.verify(&instance(), &[][..]).unwrap_err();
		assert!(
			matches!(
				err,
				VerificationM4Error::WrongChipCount {
					n_witness_chips: 0,
					n_chips: 1,
				}
			),
			"{err}"
		);
	}

	#[test]
	fn verify_rejects_a_chip_with_fewer_instances_than_active_ones() {
		let cs = system(&[0], vec![chip(&[])]);
		let err = cs.verify(&instance(), &[vec![]][..]).unwrap_err();
		assert!(
			matches!(
				err,
				VerificationM4Error::MissingInstances {
					chip_id: 0,
					n_instances: 0,
					n_active: 1,
				}
			),
			"{err}"
		);
	}

	#[test]
	fn verify_reports_the_instance_whose_own_constraints_fail() {
		let mut cs = system(&[0], vec![chip(&[])]);
		// The chip requires its first inout value to vanish, and the padding instance holds a word
		// that does not. Every instance is committed, so padding is checked like the rest.
		cs.chips[0]
			.0
			.cs
			.zero_constraints
			.push(ZeroConstraint::plain([ValueIndex::inout(0)]));

		let mut padding = instance();
		padding[ValueIndex::inout(0)] = Word::ONE;
		let err = cs
			.verify(&instance(), &[vec![instance(), padding]][..])
			.unwrap_err();
		assert!(
			matches!(
				err,
				VerificationM4Error::ChipInstance {
					chip_id: 0,
					instance: 1,
					source: VerificationError::Unsatisfied { .. },
				}
			),
			"{err}"
		);
	}
}
