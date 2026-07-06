// Copyright 2026 The Binius Developers

//! The witness for an M4 circuit: the main circuit's values and one table per chip.

use std::iter;

use binius_core::{
	ValueVec, VerificationM4Error, Word,
	m4::{ChipCall, ConstraintSystemM4},
};
use binius_frontend::{BatchPopulateError, CircuitM4, PopulateError, WitnessFiller};
use binius_utils::checked_arithmetics::log2_ceil_usize;

use crate::value_table::ValueTable;

/// A full M4 witness: the main circuit's values and one [`ValueTable`] per chip of a
/// [`CircuitM4`].
///
/// The tables are indexed by chip ID, so `tables[i]` holds every instance of chip `i`. One row of a
/// table is one invocation of that chip: the chip's local constraints must hold on the row, and the
/// row's inout values must be matched by exactly one chip call elsewhere in the system.
#[derive(Debug)]
pub struct WitnessM4 {
	/// The values of the main circuit, which runs once.
	pub main: ValueVec,
	/// The instances of each chip, indexed by chip ID.
	pub tables: Vec<ValueTable>,
}

impl WitnessM4 {
	/// Generates the witness for a whole system from the main circuit's inputs.
	///
	/// `fill_main` assigns the witness inputs of the main circuit; every other value in the system
	/// is derived from them. Each chip's table holds one instance per invocation that reaches it,
	/// ordered by caller: main's calls first, then the calls of each chip in ID order, and within a
	/// chip by instance. The instance count is rounded up to a power of two by repeating the last
	/// invocation, which satisfies the chip because the invocation it copies does.
	///
	/// # Panics
	///
	/// Panics if the system does not pass [`CircuitM4::validate`].
	pub fn generate<F>(circuit: &CircuitM4, fill_main: F) -> Result<Self, PopulateM4Error>
	where
		F: FnOnce(&mut WitnessFiller<'_>),
	{
		let mut main_witness_filler = circuit.main.circuit.new_witness_filler();
		fill_main(&mut main_witness_filler);
		circuit
			.main
			.circuit
			.populate_wire_witness(&mut main_witness_filler)?;

		let main_values = main_witness_filler.into_value_vec();

		let mut tables = Vec::<ValueTable>::with_capacity(circuit.chips.len());
		for (chip_id, (chip, n_active)) in circuit.chips.iter().enumerate() {
			let mut call_data = Vec::with_capacity(*n_active);

			// Push main's calls.
			for call in &circuit.main.chip_calls {
				if call.chip_id == chip_id {
					call_data.push(eval_call(&main_values, call));
				}
			}

			// Push the calls of the chips populated so far. Calls only run to higher IDs, so
			// zipping against the tables built up to here yields exactly this chip's callers.
			for ((caller, caller_active), table) in iter::zip(&circuit.chips, &tables) {
				if !caller.chip_calls.iter().any(|call| call.chip_id == chip_id) {
					continue;
				}
				let constants = &caller.circuit.constraint_system().constants;
				for instance in 0..*caller_active {
					let values = table.instance_value_vec(instance, constants);
					for call in &caller.chip_calls {
						if call.chip_id == chip_id {
							call_data.push(eval_call(&values, call));
						}
					}
				}
			}

			// Invariants checked in `CircuitM4::validate()`
			assert_eq!(call_data.len(), *n_active);
			assert!(*n_active > 0, "chip {chip_id} is never called");

			let log_instances = log2_ceil_usize(call_data.len());
			let table =
				ValueTable::populate_parallel(&chip.circuit, log_instances, |instance, filler| {
					// Instances past the last invocation repeat it.
					let inout = &call_data[instance.min(call_data.len() - 1)];
					for (i, &wire) in chip.circuit.inout().iter().enumerate() {
						filler[wire] = inout.get(i).copied().unwrap_or(Word::ZERO);
					}
				})
				.map_err(|source| PopulateM4Error::Chip { chip_id, source })?;

			tables.push(table);
		}

		Ok(Self {
			main: main_values,
			tables,
		})
	}

	/// Checks that this witness satisfies an M4 constraint system.
	///
	/// Each table's instances are materialized as value vectors and handed to
	/// [`ConstraintSystemM4::verify`], which checks the local constraints of every instance and
	/// matches every chip call against the instance serving it.
	pub fn verify(&self, cs: &ConstraintSystemM4) -> Result<(), VerificationM4Error> {
		if self.tables.len() != cs.chips.len() {
			return Err(VerificationM4Error::WrongChipCount {
				n_witness_chips: self.tables.len(),
				n_chips: cs.chips.len(),
			});
		}
		let chip_instances = iter::zip(&cs.chips, &self.tables)
			.map(|((chip, _), table)| {
				(0..table.n_instances())
					.map(|instance| table.instance_value_vec(instance, &chip.cs.constants))
					.collect()
			})
			.collect::<Vec<_>>();
		cs.verify(&self.main, &chip_instances)
	}
}

/// Evaluates the inout operands of one chip call against the caller's values.
fn eval_call(values: &ValueVec, call: &ChipCall) -> Vec<Word> {
	call.inout
		.iter()
		.map(|operand| values.eval_operand(operand))
		.collect()
}

/// Reason the witness of an M4 circuit could not be generated.
#[allow(missing_docs)] // errors are self-documenting
#[derive(Debug, thiserror::Error)]
pub enum PopulateM4Error {
	#[error("the main circuit is not satisfied: {0}")]
	Main(#[from] PopulateError),
	#[error("chip #{chip_id} is not satisfied: {source}")]
	Chip {
		chip_id: usize,
		#[source]
		source: BatchPopulateError,
	},
}

#[cfg(test)]
mod tests {
	use binius_core::ShiftedValueIndex;
	use binius_frontend::{Circuit, CircuitBuilder, CircuitM4Error, EmbeddedCircuit, Wire};

	use super::*;

	/// A chip whose inout words are `(a, b, c)`, constrained by `c == a & b`.
	///
	/// It calls chip `callee` once per instance, forwarding `(c, c)`.
	fn and_chip(callee: usize) -> EmbeddedCircuit {
		let builder = CircuitBuilder::new();
		let (a, b, c) = (builder.add_inout(), builder.add_inout(), builder.add_inout());
		builder.assert_eq("and", builder.band(a, b), c);
		let circuit = builder.build();

		let forward_c = operand(&circuit, c);
		EmbeddedCircuit {
			chip_calls: vec![ChipCall {
				chip_id: callee,
				inout: vec![forward_c.clone(), forward_c],
			}],
			circuit,
		}
	}

	/// A leaf chip whose inout words are `(a, b, a & b)`, the conjunction promoted rather than
	/// declared and asserted against.
	fn promoting_and_chip() -> EmbeddedCircuit {
		let builder = CircuitBuilder::new();
		let (a, b) = (builder.add_inout(), builder.add_inout());
		builder.mark_inout(builder.band(a, b));
		EmbeddedCircuit {
			circuit: builder.build(),
			chip_calls: vec![],
		}
	}

	/// A leaf chip whose two inout words must be equal.
	fn eq_chip() -> EmbeddedCircuit {
		let builder = CircuitBuilder::new();
		let (x, y) = (builder.add_inout(), builder.add_inout());
		builder.assert_eq("eq", x, y);
		EmbeddedCircuit {
			circuit: builder.build(),
			chip_calls: vec![],
		}
	}

	/// The operand reading a single wire of a circuit's value vector.
	fn operand(circuit: &Circuit, wire: Wire) -> Vec<ShiftedValueIndex> {
		vec![ShiftedValueIndex::plain(circuit.witness_index(wire))]
	}

	/// A main circuit passing `n_calls` triples of its own witness wires to chip 0.
	///
	/// Triple `i` is `(a_i, b_i, a_i & b_i)`, so every call satisfies the chip it reaches.
	fn main_circuit(n_calls: usize) -> (EmbeddedCircuit, Vec<(Wire, Wire)>) {
		let builder = CircuitBuilder::new();
		let inputs = (0..n_calls)
			.map(|_| (builder.add_witness(), builder.add_witness()))
			.collect::<Vec<_>>();
		let conjunctions = inputs
			.iter()
			.map(|&(a, b)| {
				let and = builder.band(a, b);
				builder.mark_inout(and);
				and
			})
			.collect::<Vec<_>>();
		let circuit = builder.build();

		let chip_calls = iter::zip(&inputs, &conjunctions)
			.map(|(&(a, b), &and)| ChipCall {
				chip_id: 0,
				inout: vec![
					operand(&circuit, a),
					operand(&circuit, b),
					operand(&circuit, and),
				],
			})
			.collect();

		let main = EmbeddedCircuit {
			circuit,
			chip_calls,
		};
		(main, inputs)
	}

	/// A system whose main circuit calls chip 0 `n_calls` times, and whose chip 0 calls chip 1
	/// once per instance.
	fn system(n_calls: usize) -> (CircuitM4, Vec<(Wire, Wire)>) {
		let (main, inputs) = main_circuit(n_calls);
		let mut circuit = CircuitM4 {
			main,
			chips: vec![(and_chip(1), 0), (eq_chip(), 0)],
		};
		circuit.recompute_n_active();
		(circuit, inputs)
	}

	/// The inout words of one instance of a chip, read back off its table.
	fn instance_inout(chip: &EmbeddedCircuit, table: &ValueTable, instance: usize) -> Vec<u64> {
		let constants = &chip.circuit.constraint_system().constants;
		let values = table.instance_value_vec(instance, constants);
		chip.circuit
			.inout()
			.iter()
			.map(|&wire| values[chip.circuit.witness_index(wire)].as_u64())
			.collect()
	}

	// The whole path with nothing hand-assembled: a chip built by one builder, registered and
	// called by another, and a witness generated off the result.
	#[test]
	fn generate_serves_the_calls_a_builder_emitted() {
		// The chip constrains its third inout word to be the conjunction of the first two.
		let chip = CircuitBuilder::new();
		let (x, y, z) = (chip.add_inout(), chip.add_inout(), chip.add_inout());
		chip.assert_eq("and", chip.band(x, y), z);

		// Main delegates two conjunctions to it, passing each result alongside its operands.
		let builder = CircuitBuilder::new();
		let chip_ref = builder.add_chip(CircuitM4::from(chip.build()));
		let inputs = (0..2)
			.map(|_| (builder.add_witness(), builder.add_witness()))
			.collect::<Vec<_>>();
		for &(a, b) in &inputs {
			builder.call_chip(chip_ref, &[a, b, builder.band(a, b)]);
		}
		let circuit = builder.build_m4();

		assert_eq!(circuit.chips[0].1, 2);
		circuit.validate().unwrap();

		let words = [(0b1100u64, 0b1010u64), (0xff00, 0x0ff0)];
		let witness = WitnessM4::generate(&circuit, |filler| {
			for (&(a, b), &(a_word, b_word)) in iter::zip(&inputs, &words) {
				filler[a] = Word(a_word);
				filler[b] = Word(b_word);
			}
		})
		.unwrap();

		assert_eq!(witness.tables[0].n_instances(), 2);
		for (instance, &(a, b)) in words.iter().enumerate() {
			assert_eq!(
				instance_inout(&circuit.chips[0].0, &witness.tables[0], instance),
				vec![a, b, a & b]
			);
		}
	}

	#[test]
	fn generate_fills_one_instance_per_call() {
		let (circuit, inputs) = system(2);
		circuit.validate().unwrap();

		let words = [(0b1100u64, 0b1010u64), (0xff00, 0x0ff0)];
		let witness = WitnessM4::generate(&circuit, |filler| {
			for (&(a, b), &(a_word, b_word)) in iter::zip(&inputs, &words) {
				filler[a] = Word(a_word);
				filler[b] = Word(b_word);
			}
		})
		.unwrap();

		// Chip 0 serves main's two calls, in call order.
		let (chip_0, chip_1) = (&circuit.chips[0].0, &circuit.chips[1].0);
		assert_eq!(witness.tables[0].n_instances(), 2);
		for (instance, &(a, b)) in words.iter().enumerate() {
			assert_eq!(instance_inout(chip_0, &witness.tables[0], instance), vec![a, b, a & b]);
		}

		// Chip 1 serves chip 0's call from each of those instances, which forwards `(c, c)`.
		assert_eq!(witness.tables[1].n_instances(), 2);
		for (instance, &(a, b)) in words.iter().enumerate() {
			assert_eq!(instance_inout(chip_1, &witness.tables[1], instance), vec![a & b, a & b]);
		}

		witness.verify(&circuit.to_constraint_system()).unwrap();
	}

	// A chip may promote an inout word instead of declaring it, which is what lets a chip return a
	// result. Generation assigns every inout wire from the call data and evaluation then recomputes
	// the promoted ones, so a row holds the chip's own word whatever the call passed. Nothing here
	// checks the two agree; where they do not, the row stops matching the call site.
	#[test]
	fn generate_recomputes_a_promoted_inout_word() {
		let (main, inputs) = main_circuit(1);
		let mut circuit = CircuitM4 {
			main,
			chips: vec![(promoting_and_chip(), 1)],
		};
		circuit.validate().unwrap();

		let (a, b) = inputs[0];
		let fill = |filler: &mut WitnessFiller<'_>| {
			filler[a] = Word(0b1100);
			filler[b] = Word(0b1010);
		};
		let row = |circuit: &CircuitM4, witness: &WitnessM4| {
			instance_inout(&circuit.chips[0].0, &witness.tables[0], 0)
		};

		let witness = WitnessM4::generate(&circuit, fill).unwrap();
		assert_eq!(row(&circuit, &witness), vec![0b1100, 0b1010, 0b1000]);
		witness.verify(&circuit.to_constraint_system()).unwrap();

		// Pass `a` as the third word, which the chip's conjunction disagrees with. Generation still
		// succeeds, and the row still holds the conjunction rather than what the call passed — so
		// the call no longer matches its instance, which is exactly what verification rejects.
		circuit.main.chip_calls[0].inout[2] = operand(&circuit.main.circuit, a);
		let witness = WitnessM4::generate(&circuit, fill).unwrap();
		assert_eq!(row(&circuit, &witness), vec![0b1100, 0b1010, 0b1000]);
		let err = witness.verify(&circuit.to_constraint_system()).unwrap_err();
		assert!(
			matches!(
				err,
				VerificationM4Error::CallMismatch {
					chip_id: 0,
					row: 0,
					caller_chip: None,
					word: 2,
					..
				}
			),
			"{err}"
		);
	}

	#[test]
	fn generate_pads_the_instance_count_by_repeating_the_last_call() {
		let (circuit, inputs) = system(3);
		circuit.validate().unwrap();

		let words = [(0b1100u64, 0b1010u64), (0xff00, 0x0ff0), (0xabcd, 0xdcba)];
		let witness = WitnessM4::generate(&circuit, |filler| {
			for (&(a, b), &(a_word, b_word)) in iter::zip(&inputs, &words) {
				filler[a] = Word(a_word);
				filler[b] = Word(b_word);
			}
		})
		.unwrap();

		// Three calls round up to four instances, the fourth repeating the third.
		let chip_0 = &circuit.chips[0].0;
		assert_eq!(witness.tables[0].n_instances(), 4);
		let (a, b) = words[2];
		assert_eq!(instance_inout(chip_0, &witness.tables[0], 3), vec![a, b, a & b]);

		// The padding instances pass verification too: their local constraints hold, and no call
		// claims them.
		witness.verify(&circuit.to_constraint_system()).unwrap();
	}

	#[test]
	fn generate_reports_the_chip_whose_calls_do_not_satisfy_it() {
		let (mut circuit, inputs) = system(1);

		// Pass `a` where chip 0 expects `a & b`, so no witness can serve the call.
		let (a, b) = inputs[0];
		circuit.main.chip_calls[0].inout[2] = operand(&circuit.main.circuit, a);

		let err = WitnessM4::generate(&circuit, |filler| {
			filler[a] = Word(0b1100);
			filler[b] = Word(0b1010);
		})
		.unwrap_err();
		assert!(matches!(err, PopulateM4Error::Chip { chip_id: 0, .. }), "{err}");
	}

	#[test]
	fn validate_rejects_a_call_to_a_chip_that_does_not_exist() {
		let (mut circuit, _) = system(1);
		circuit.main.chip_calls[0].chip_id = 2;
		assert!(matches!(
			circuit.validate(),
			Err(CircuitM4Error::OutOfRangeChipId {
				chip_index: None,
				chip_id: 2,
				n_chips: 2,
			})
		));
	}

	#[test]
	fn validate_rejects_chips_out_of_topological_order() {
		let (mut circuit, _) = system(1);
		// Chip 1 is a leaf; make it call chip 0, which is populated before it.
		circuit.chips[1].0.chip_calls.push(ChipCall {
			chip_id: 0,
			inout: vec![],
		});
		assert!(matches!(
			circuit.validate(),
			Err(CircuitM4Error::CallOutOfOrder {
				chip_index: 1,
				callee: 0,
			})
		));
	}

	#[test]
	fn validate_rejects_a_wrong_active_instance_count() {
		let (mut circuit, _) = system(2);
		circuit.chips[1].1 = 1;
		assert!(matches!(
			circuit.validate(),
			Err(CircuitM4Error::WrongActiveInstanceCount {
				chip_index: 1,
				declared: 1,
				actual: 2,
			})
		));
	}

	#[test]
	fn validate_rejects_a_chip_nothing_calls() {
		let (mut circuit, _) = system(1);
		circuit.main.chip_calls.clear();
		circuit.recompute_n_active();
		assert!(matches!(circuit.validate(), Err(CircuitM4Error::NeverCalled { chip_index: 0 })));
	}
}
