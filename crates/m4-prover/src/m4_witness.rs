// Copyright 2026 The Binius Developers

//! The witness for an M4 circuit: the main circuit's values and one table per chip.

use std::{borrow::Cow, mem};

use binius_core::{
	ValueVec, VerificationM4Error, Word,
	m4::{ChipCall, ChipInstances, ConstraintSystemM4},
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
	/// Panics if the system does not pass [`CircuitM4::validate`], which covers both the ordering
	/// this walks the chips in and the well-formedness of the operands it evaluates.
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

		// The invocations awaiting each chip, in the order the calls run: main's first, then the
		// calls of each chip in ID order, instance-major and call-minor. Calls only run to higher
		// IDs, so a chip's list is complete by the time the pass below reaches it.
		let mut pending = vec![Vec::new(); circuit.chips.len()];
		for call in &circuit.main.chip_calls {
			pending[call.chip_id].push(eval_call(&main_values, call));
		}

		let mut tables = Vec::<ValueTable>::with_capacity(circuit.chips.len());
		for (chip_id, (chip, n_active)) in circuit.chips.iter().enumerate() {
			let call_data = mem::take(&mut pending[chip_id]);

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

			// Read this chip's own calls off each active instance, for the chips after it to
			// serve. Building the instance is what costs here — a value vector allocated and
			// gathered word by word — so each is built once and every call it makes is read off
			// it, rather than once per callee.
			if !chip.chip_calls.is_empty() {
				let constants = &chip.circuit.constraint_system().constants;
				for instance in 0..*n_active {
					let values = table.instance_value_vec(instance, constants);
					for call in &chip.chip_calls {
						pending[call.chip_id].push(eval_call(&values, call));
					}
				}
			}

			tables.push(table);
		}

		Ok(Self {
			main: main_values,
			tables,
		})
	}

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
	use std::iter;

	use binius_core::{ShiftedValueIndex, ValueIndex, error::OperandFault};
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

	/// A chip whose inout words are `(a, b, c)`, constrained by `c == a & b`.
	///
	/// It calls chip `callee` twice per instance, forwarding `(a, a)` and then `(b, b)`.
	fn twice_calling_and_chip(callee: usize) -> EmbeddedCircuit {
		let builder = CircuitBuilder::new();
		let (a, b, c) = (builder.add_inout(), builder.add_inout(), builder.add_inout());
		builder.assert_eq("and", builder.band(a, b), c);
		let circuit = builder.build();

		let call = |wire| ChipCall {
			chip_id: callee,
			inout: vec![operand(&circuit, wire), operand(&circuit, wire)],
		};
		let chip_calls = vec![call(a), call(b)];
		EmbeddedCircuit {
			circuit,
			chip_calls,
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
					caller: None,
					word: 2,
					..
				}
			),
			"{err}"
		);
	}

	// A caller with several instances making several calls to one callee is what pins the order
	// generation and verification both walk the calls in: instance-major, call-minor. With one
	// call per callee the two orders coincide and neither side would notice the other drifting.
	#[test]
	fn generate_interleaves_a_callers_instances_before_its_calls() {
		let (main, inputs) = main_circuit(2);
		let mut circuit = CircuitM4 {
			main,
			chips: vec![(twice_calling_and_chip(1), 0), (eq_chip(), 0)],
		};
		circuit.recompute_n_active();
		circuit.validate().unwrap();

		// Chip 0 serves main's two calls, and each of its instances calls chip 1 twice.
		assert_eq!(circuit.chips[0].1, 2);
		assert_eq!(circuit.chips[1].1, 4);

		let words = [(0b1100u64, 0b1010u64), (0xff00, 0x0ff0)];
		let witness = WitnessM4::generate(&circuit, |filler| {
			for (&(a, b), &(a_word, b_word)) in iter::zip(&inputs, &words) {
				filler[a] = Word(a_word);
				filler[b] = Word(b_word);
			}
		})
		.unwrap();

		// Instance 0's two calls come before instance 1's, each forwarding `(a, a)` then `(b, b)`.
		let expected = [words[0].0, words[0].1, words[1].0, words[1].1];
		assert_eq!(witness.tables[1].n_instances(), 4);
		for (instance, &word) in expected.iter().enumerate() {
			assert_eq!(
				instance_inout(&circuit.chips[1].0, &witness.tables[1], instance),
				vec![word, word]
			);
		}

		witness.verify(&circuit.to_constraint_system()).unwrap();
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

	// An operand past the callee's interface has nowhere to land: generation drops it and
	// verification never looks at it, so nothing downstream would report it.
	#[test]
	fn validate_rejects_a_call_passing_more_operands_than_the_callee_takes() {
		let (mut circuit, inputs) = system(1);
		let extra = operand(&circuit.main.circuit, inputs[0].0);
		circuit.main.chip_calls[0].inout.push(extra);
		assert!(matches!(
			circuit.validate(),
			Err(CircuitM4Error::WrongCallArity {
				chip_index: None,
				call_index: 0,
				chip_id: 0,
				arity: 4,
				n_inout: 3,
			})
		));
	}

	// Scratch words are uncommitted temporaries, so a call reading one names a word no instance
	// holds. `call_chip` pins its wires out of scratch, but `ChipCall` is built by hand too.
	#[test]
	fn validate_rejects_a_call_operand_naming_a_scratch_value() {
		let (mut circuit, _) = system(1);
		circuit.main.chip_calls[0].inout[2] =
			vec![ShiftedValueIndex::plain(ValueIndex::scratch(0))];
		assert!(matches!(
			circuit.validate(),
			Err(CircuitM4Error::CallOperand {
				chip_index: None,
				call_index: 0,
				operand_index: 2,
				source: OperandFault::ScratchValueIndex,
			})
		));
	}

	// Instance counts multiply down the call graph, so a chain of a few dozen chips outgrows a
	// `usize`. Counting it out unchecked would wrap to a plausible-looking total.
	#[test]
	fn validate_rejects_an_instance_count_that_outgrows_a_usize() {
		// A chain of 70 chips, each calling the next twice, so chip `i` is reached 2^i times.
		let chip = |callee: Option<usize>| {
			let builder = CircuitBuilder::new();
			builder.add_inout();
			EmbeddedCircuit {
				circuit: builder.build(),
				chip_calls: callee
					.into_iter()
					.flat_map(|chip_id| {
						iter::repeat_with(move || ChipCall {
							chip_id,
							inout: vec![],
						})
						.take(2)
					})
					.collect(),
			}
		};

		const DEPTH: usize = 70;
		let mut circuit = CircuitM4 {
			main: EmbeddedCircuit {
				circuit: CircuitBuilder::new().build(),
				chip_calls: vec![ChipCall {
					chip_id: 0,
					inout: vec![],
				}],
			},
			chips: (0..DEPTH)
				.map(|i| (chip((i + 1 < DEPTH).then_some(i + 1)), 0))
				.collect(),
		};
		circuit.recompute_n_active();

		// Chip 63 is reached 2^63 times and calls chip 64 twice, which is where the count leaves
		// the range.
		assert!(matches!(
			circuit.validate(),
			Err(CircuitM4Error::TooManyInstances { chip_index: 64 })
		));
	}
}
