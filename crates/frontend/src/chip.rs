// Copyright 2026 The Binius Developers

//! Circuits composed out of chips, each generating the witness of one M4 chip.

use binius_core::m4::{ChipCall, ConstraintSystemM4, EmbeddedConstraintSystem};

use crate::Circuit;

/// One chip of a [`CircuitM4`], as the circuit that generates its witness.
///
/// The chip's interface is its circuit's inout segment: a call site supplies those words
/// positionally, and every value the chip holds beyond them is derived from them. An inout wire the
/// call does not reach is filled with zero.
///
/// A wire promoted with [`CircuitBuilder::mark_inout`](crate::CircuitBuilder::mark_inout) serves
/// as well as one declared with [`CircuitBuilder::add_inout`](crate::CircuitBuilder::add_inout).
/// Witness generation assigns every inout wire from the call data, then evaluation recomputes the
/// promoted ones over it. Nothing checks that the two agree, and nothing has to: where they
/// disagree the row stops matching the call site, which is what the chip call itself enforces.
///
/// That is what lets a caller pass a chip's output. The caller is populated whole before its calls
/// are read, so it holds the output already — generally from a hint, whose correctness the chip
/// call is what constrains.
pub struct EmbeddedCircuit {
	/// The circuit generating one instance of the chip.
	pub circuit: Circuit,
	/// The chips this one delegates subrelations to, one entry per call per instance.
	pub chip_calls: Vec<ChipCall>,
}

/// A circuit composed of chips, as the circuits that generate their witnesses.
///
/// `main` is the entry point: it calls chips, but no chip ID names it, so nothing calls it. The
/// chips have an ID equal to their index in `chips`.
pub struct CircuitM4 {
	/// The entry point, which runs once.
	pub main: EmbeddedCircuit,
	/// The chips, indexed by chip ID, each paired with its number of active instances.
	///
	/// A chip runs once per call that reaches it, and those instances are the active ones: only
	/// they have their own chip calls enforced. The instances past them pad the count up to a
	/// power of two.
	///
	/// The count is denormalized — it says what the call graph already says, and
	/// [`Self::recompute_n_active`] derives it. [`Self::validate`] holds the two to each other.
	pub chips: Vec<(EmbeddedCircuit, usize)>,
}

impl From<Circuit> for CircuitM4 {
	/// Makes a circuit the whole system, as a main that calls no chips.
	fn from(circuit: Circuit) -> Self {
		Self {
			main: EmbeddedCircuit {
				circuit,
				chip_calls: Vec::new(),
			},
			chips: Vec::new(),
		}
	}
}

impl CircuitM4 {
	/// Checks that this system can be populated in one pass over the chips, in ID order.
	///
	/// Specifically checks that:
	///
	/// - every chip call names a chip of this system;
	/// - the chips are in topological order, each calling only chips with a higher ID, so every
	///   caller of a chip is populated before the chip itself;
	/// - each chip's declared active-instance count is the number of invocations that reach it, and
	///   no chip is left uncalled.
	pub fn validate(&self) -> Result<(), CircuitM4Error> {
		let n_chips = self.chips.len();

		for call in &self.main.chip_calls {
			if call.chip_id >= n_chips {
				return Err(CircuitM4Error::OutOfRangeChipId {
					chip_index: None,
					chip_id: call.chip_id,
					n_chips,
				});
			}
		}
		for (chip_index, (chip, _)) in self.chips.iter().enumerate() {
			for call in &chip.chip_calls {
				if call.chip_id >= n_chips {
					return Err(CircuitM4Error::OutOfRangeChipId {
						chip_index: Some(chip_index),
						chip_id: call.chip_id,
						n_chips,
					});
				}
				if call.chip_id <= chip_index {
					return Err(CircuitM4Error::CallOutOfOrder {
						chip_index,
						callee: call.chip_id,
					});
				}
			}
		}

		// The invocations reaching each chip, counted the way witness generation gathers them. Only
		// main and lower-numbered chips call a chip, so a single pass in ID order sees every caller
		// of chip `i` before it reads chip `i`'s own total.
		let mut n_calls = vec![0; n_chips];
		for call in &self.main.chip_calls {
			n_calls[call.chip_id] += 1;
		}
		for (chip_index, (chip, n_active)) in self.chips.iter().enumerate() {
			if n_calls[chip_index] != *n_active {
				return Err(CircuitM4Error::WrongActiveInstanceCount {
					chip_index,
					declared: *n_active,
					actual: n_calls[chip_index],
				});
			}
			if *n_active == 0 {
				return Err(CircuitM4Error::NeverCalled { chip_index });
			}

			// Only the active instances of this chip have their calls enforced, so only they
			// demand an instance of the callee.
			for call in &chip.chip_calls {
				n_calls[call.chip_id] += n_active;
			}
		}

		Ok(())
	}

	/// Sets each chip's active-instance count to the number of invocations that reach it.
	///
	/// A chip is invoked once per call site naming it, per active instance of the caller. Main runs
	/// once, so each of its call sites counts for one.
	///
	/// This is what [`Self::validate`] checks the declared counts against, so a system whose call
	/// sites have just been written or rewritten passes it here rather than counting by hand.
	///
	/// # Panics
	///
	/// Panics if a chip call names a chip this system does not have. Chips out of topological
	/// order are not detected here: a call to a lower ID counts against a total already written
	/// back, and [`Self::validate`] is what rejects the result.
	pub fn recompute_n_active(&mut self) {
		let mut n_calls = vec![0; self.chips.len()];
		for call in &self.main.chip_calls {
			n_calls[call.chip_id] += 1;
		}
		// Only main and lower-numbered chips call a chip, so one pass in ID order settles chip
		// `i`'s own total before reading it.
		for chip_index in 0..self.chips.len() {
			let n_active = n_calls[chip_index];
			self.chips[chip_index].1 = n_active;

			// Only the active instances of this chip have their calls enforced, so only they
			// demand an instance of the callee.
			for call in &self.chips[chip_index].0.chip_calls {
				n_calls[call.chip_id] += n_active;
			}
		}
	}

	/// Lowers this system to the constraint-system form the proving protocol consumes.
	///
	/// Each circuit contributes its compiled constraint system; the chip calls and the
	/// active-instance counts carry over unchanged.
	pub fn to_constraint_system(&self) -> ConstraintSystemM4 {
		let lower = |chip: &EmbeddedCircuit| EmbeddedConstraintSystem {
			cs: chip.circuit.constraint_system().clone(),
			chip_calls: chip.chip_calls.clone(),
		};
		ConstraintSystemM4 {
			main: lower(&self.main),
			chips: self
				.chips
				.iter()
				.map(|(chip, n_active)| (lower(chip), *n_active))
				.collect(),
		}
	}
}

/// A chip of the system a [`CircuitBuilder`](crate::CircuitBuilder) is building.
///
/// [`CircuitBuilder::add_chip`](crate::CircuitBuilder::add_chip) returns one for each chip it
/// registers, and a call site names its callee by it. Registering further chips never moves a chip
/// already registered, so a reference stays good for the rest of the build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChipRef(usize);

impl ChipRef {
	/// Names the chip at the given index of [`CircuitM4::chips`].
	pub(crate) const fn new(chip_id: usize) -> Self {
		Self(chip_id)
	}

	/// Returns the chip's index in [`CircuitM4::chips`], which is what a [`ChipCall`] names it by.
	///
	/// Reading the index out is the only direction: a reference cannot be made from one, so every
	/// reference names a chip that was registered.
	pub const fn chip_id(self) -> usize {
		self.0
	}
}

/// Names the circuit of an M4 system that a diagnostic is about.
///
/// The main circuit is not one of the numbered chips, so it has no index.
fn circuit_name(chip_index: Option<usize>) -> String {
	match chip_index {
		Some(chip_index) => format!("chip #{chip_index}"),
		None => "the main circuit".to_string(),
	}
}

/// Reason an M4 circuit cannot be populated as it stands.
#[allow(missing_docs)] // errors are self-documenting
#[derive(Debug, thiserror::Error)]
pub enum CircuitM4Error {
	#[error("{} calls chip {chip_id}, but the system has {n_chips} chips", circuit_name(*chip_index))]
	OutOfRangeChipId {
		chip_index: Option<usize>,
		chip_id: usize,
		n_chips: usize,
	},
	#[error("chip #{chip_index} calls chip {callee}, which is not a later chip")]
	CallOutOfOrder { chip_index: usize, callee: usize },
	#[error("chip #{chip_index} declares {declared} active instances, but {actual} calls reach it")]
	WrongActiveInstanceCount {
		chip_index: usize,
		declared: usize,
		actual: usize,
	},
	#[error("chip #{chip_index} is never called")]
	NeverCalled { chip_index: usize },
}
