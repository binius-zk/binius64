// Copyright 2026 The Binius Developers

//! The witness for an M4 constraint system: the main chip's values and one table per chip.

use std::borrow::Cow;

use super::{ChipInstances, ConstraintSystemM4};
use crate::{ValueTable, ValueVec, error::VerificationM4Error};

/// A full M4 witness: the main chip's values and one [`ValueTable`] per chip of a
/// [`ConstraintSystemM4`].
///
/// The tables are indexed by chip ID, so `tables[i]` holds every instance of chip `i`. One row of a
/// table is one invocation of that chip: the chip's local constraints must hold on the row, and the
/// row's inout values must be matched by exactly one chip call elsewhere in the system.
///
/// Generating one is the circuit frontend's business, since it takes circuits to evaluate — see
/// `CircuitM4::generate_witness`. This crate is where one is checked.
#[derive(Debug)]
pub struct WitnessM4 {
	/// The values of the main chip, which runs once.
	pub main: ValueVec,
	/// The instances of each chip, indexed by chip ID.
	pub tables: Vec<ValueTable>,
}

impl WitnessM4 {
	/// Checks that this witness satisfies an M4 constraint system.
	///
	/// [`ConstraintSystemM4::verify`] checks the local constraints of every instance and matches
	/// every chip call against the instance serving it. It reads the instances one at a time, so a
	/// table is never expanded into value vectors whole: the tables are the witness, and they stay
	/// the only copy of it.
	pub fn verify(&self, cs: &ConstraintSystemM4) -> Result<(), VerificationM4Error> {
		cs.verify(
			&self.main,
			&TableInstances {
				tables: &self.tables,
				cs,
			},
		)
	}
}

/// A witness's tables read as chip instances, each built when it is asked for.
///
/// The constants are the one part of an instance a table does not store, so the system is held
/// alongside to supply them.
struct TableInstances<'a> {
	tables: &'a [ValueTable],
	cs: &'a ConstraintSystemM4,
}

impl ChipInstances for TableInstances<'_> {
	fn n_chips(&self) -> usize {
		self.tables.len()
	}

	fn n_instances(&self, chip_id: usize) -> usize {
		self.tables[chip_id].n_instances()
	}

	fn instance(&self, chip_id: usize, row: usize) -> Cow<'_, ValueVec> {
		let constants = &self.cs.chips[chip_id].0.cs.constants;
		Cow::Owned(self.tables[chip_id].instance_value_vec(row, constants))
	}
}
