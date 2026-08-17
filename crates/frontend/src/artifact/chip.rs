// Copyright 2026 The Binius Developers

//! Circuits composed out of chips, each generating the witness of one M4 chip.
//!
//! The witness itself is [`binius_core`]'s [`WitnessM4`], which is also where one is checked.

use std::mem;

use binius_compute::GlobalAllocator;
use binius_core::{
	ValueTable, Word, WordSource,
	error::{ChipName, OperandFault},
	eval_operand,
	m4::{ChipCall, ConstraintSystemM4, EmbeddedConstraintSystem, WitnessM4},
};
use binius_utils::checked_arithmetics::log2_ceil_usize;

use crate::{
	Circuit, CircuitBuilder,
	artifact::witness::{PopulateError, WitnessFiller},
	eval_form::BatchPopulateError,
	ir::{Wire, hints::Hint},
};

/// One chip of a [`CircuitM4`], as the circuit that generates its witness.
///
/// The chip's interface is its circuit's inout segment: a call site supplies those words
/// positionally, and every value the chip holds beyond them is derived from them. An inout wire the
/// call does not reach is filled with zero.
///
/// A wire promoted with [`CircuitBuilder::mark_inout`] serves
/// as well as one declared with [`CircuitBuilder::add_inout`].
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
	/// [`Self::recompute_instances`] derives it, along with the instance each call claims.
	/// [`Self::validate`] holds the two to each other.
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
	/// - every chip call names a chip of this system, passes no more operands than that chip has
	///   inout values, and reads only committed words of its own caller;
	/// - the chips are in topological order, each calling only chips with a higher ID, so every
	///   caller of a chip is populated before the chip itself;
	/// - each chip's declared active-instance count is the number of invocations that reach it, no
	///   chip is left uncalled, and no count outgrows a `usize`;
	/// - each call names the callee instance the call graph gives it.
	///
	/// A system that passes here lowers to one that passes
	/// [`ConstraintSystemM4::validate`](binius_core::m4::ConstraintSystemM4::validate), which
	/// requires the same ordering and additionally validates each chip's compiled constraint
	/// system — the one thing here the compiler rather than this check keeps well-formed. Nothing
	/// downstream therefore has to run the lowered check to know its preconditions hold.
	pub fn validate(&self) -> Result<(), CircuitM4Error> {
		let n_chips = self.chips.len();

		self.validate_calls(None, &self.main)?;
		for (chip_index, (chip, _)) in self.chips.iter().enumerate() {
			self.validate_calls(Some(chip_index), chip)?;
			for call in &chip.chip_calls {
				if call.chip_id <= chip_index {
					return Err(CircuitM4Error::CallOutOfOrder {
						chip_index,
						callee: call.chip_id,
					});
				}
			}
		}

		// The invocations reaching each chip, counted the way [`Self::recompute_instances`] hands
		// them out. Only main and lower-numbered chips call a chip, so a single pass in ID order
		// sees every caller of chip `i` before it reads chip `i`'s own total.
		let mut n_calls = vec![0usize; n_chips];
		for (call_index, call) in self.main.chip_calls.iter().enumerate() {
			Self::check_call_instance(None, call_index, call, &n_calls)?;
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
			//
			// The counts multiply down the call graph — a chain whose every chip calls the next
			// twice reaches `2^depth` — so a system of a few dozen chips can outgrow a `usize`.
			// Counting it out unchecked would wrap to a small total that then agrees with a
			// `recompute_instances` that wrapped the same way, and the system would go on to be
			// populated against a count that is not the number of calls.
			for (call_index, call) in chip.chip_calls.iter().enumerate() {
				Self::check_call_instance(Some(chip_index), call_index, call, &n_calls)?;
				n_calls[call.chip_id] = n_calls[call.chip_id].checked_add(*n_active).ok_or(
					CircuitM4Error::TooManyInstances {
						chip_index: call.chip_id,
					},
				)?;
			}
		}

		Ok(())
	}

	/// Checks that one call names the callee instance the invocations counted so far leave it.
	fn check_call_instance(
		chip_index: Option<usize>,
		call_index: usize,
		call: &ChipCall,
		n_calls: &[usize],
	) -> Result<(), CircuitM4Error> {
		let expected = n_calls[call.chip_id];
		if call.first_instance != expected {
			return Err(CircuitM4Error::WrongCallInstance {
				chip_index,
				call_index,
				first_instance: call.first_instance,
				expected,
			});
		}
		Ok(())
	}

	/// Checks one caller's calls: that each names a chip of this system, passes no more operands
	/// than that chip has inout values, and reads only committed words of the caller's own value
	/// vector.
	///
	/// A call may pass fewer operands than the callee takes: the inout values past them are
	/// constrained to zero.
	fn validate_calls(
		&self,
		chip_index: Option<usize>,
		caller: &EmbeddedCircuit,
	) -> Result<(), CircuitM4Error> {
		let n_chips = self.chips.len();
		let cs = caller.circuit.constraint_system();
		for (call_index, call) in caller.chip_calls.iter().enumerate() {
			if call.chip_id >= n_chips {
				return Err(CircuitM4Error::OutOfRangeChipId {
					chip_index,
					chip_id: call.chip_id,
					n_chips,
				});
			}
			let n_inout = self.chips[call.chip_id]
				.0
				.circuit
				.constraint_system()
				.n_inout;
			if call.inout.len() > n_inout {
				return Err(CircuitM4Error::WrongCallArity {
					chip_index,
					call_index,
					chip_id: call.chip_id,
					arity: call.inout.len(),
					n_inout,
				});
			}
			for (operand_index, operand) in call.inout.iter().enumerate() {
				if let Some(source) = cs.operand_fault(operand) {
					return Err(CircuitM4Error::CallOperand {
						chip_index,
						call_index,
						operand_index,
						source,
					});
				}
			}
		}
		Ok(())
	}

	/// Gives each call the callee instances it invokes, and each chip the count of the ones it is
	/// left with.
	///
	/// A chip is invoked once per call site naming it, per active instance of the caller, and a
	/// call site's invocations are consecutive instances of the callee: the caller's instance `i`
	/// invokes the call's [`first_instance`](ChipCall::first_instance) plus `i`. Main runs once, so
	/// each of its call sites claims a single instance.
	///
	/// This is what [`Self::validate`] checks the declared counts and instances against, so a
	/// system whose call sites have just been written or rewritten passes it here rather than
	/// counting by hand.
	///
	/// A count that outgrows a `usize` saturates rather than wrapping, so it stays too large for
	/// the invocations that reach the chip and [`Self::validate`] reports it.
	///
	/// # Panics
	///
	/// Panics if a chip call names a chip this system does not have. Chips out of topological
	/// order are not detected here: a call to a lower ID counts against a total already written
	/// back, and [`Self::validate`] is what rejects the result.
	pub fn recompute_instances(&mut self) {
		let mut n_calls = vec![0usize; self.chips.len()];
		for call in &mut self.main.chip_calls {
			call.first_instance = n_calls[call.chip_id];
			n_calls[call.chip_id] += 1;
		}
		// Only main and lower-numbered chips call a chip, so one pass in ID order settles chip
		// `i`'s own total before reading it.
		for chip_index in 0..self.chips.len() {
			let n_active = n_calls[chip_index];
			self.chips[chip_index].1 = n_active;

			// Only the active instances of this chip have their calls enforced, so only they
			// demand an instance of the callee.
			for call in &mut self.chips[chip_index].0.chip_calls {
				call.first_instance = n_calls[call.chip_id];
				n_calls[call.chip_id] = n_calls[call.chip_id].saturating_add(n_active);
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

	/// Generates the witness for a whole system from the main circuit's inputs.
	///
	/// `fill_main` assigns the witness inputs of the main circuit; every other value in the system
	/// is derived from them. Each chip's table holds one instance per invocation that reaches it,
	/// each at the row the invoking call names. The instance count is rounded up to a power of two
	/// by repeating the last invocation, which satisfies the chip because the invocation it copies
	/// does.
	///
	/// # Panics
	///
	/// Panics if the system does not pass [`Self::validate`], which covers both the ordering this
	/// walks the chips in and the well-formedness of the operands it evaluates.
	pub fn generate_witness<F>(&self, fill_main: F) -> Result<WitnessM4, PopulateM4Error>
	where
		F: FnOnce(&mut WitnessFiller<'_>),
	{
		let mut main_witness_filler = self.main.circuit.new_witness_filler();
		fill_main(&mut main_witness_filler);
		self.main
			.circuit
			.populate_wire_witness(&mut main_witness_filler)?;

		let main_values = main_witness_filler.into_value_vec();

		// The instances of each chip, each written by the call that invokes it. Calls only run to
		// higher IDs, so a chip's instances are all written by the time the pass below reaches it.
		let mut pending = self
			.chips
			.iter()
			.map(|(_, n_active)| vec![Vec::new(); *n_active])
			.collect::<Vec<_>>();
		for call in &self.main.chip_calls {
			pending[call.chip_id][call.first_instance] = eval_call(&main_values, call);
		}

		let mut tables = Vec::<ValueTable>::with_capacity(self.chips.len());
		for (chip_id, (chip, n_active)) in self.chips.iter().enumerate() {
			let call_data = mem::take(&mut pending[chip_id]);

			// Invariant checked in `CircuitM4::validate()`
			assert!(*n_active > 0, "chip {chip_id} is never called");

			let log_instances = log2_ceil_usize(call_data.len());
			let table = chip
				.circuit
				.populate_batch_parallel(&GlobalAllocator, log_instances, |instance, filler| {
					// Instances past the last invocation repeat it.
					let inout = &call_data[instance.min(call_data.len() - 1)];
					for (i, &wire) in chip.circuit.inout().iter().enumerate() {
						filler[wire] = inout.get(i).copied().unwrap_or(Word::ZERO);
					}
				})
				.map_err(|source| PopulateM4Error::Chip { chip_id, source })?;

			// Read this chip's own calls off each active instance, for the chips after it to serve.
			// `instance_words` reads each call's operands straight off the table's strided rows.
			if !chip.chip_calls.is_empty() {
				let constants = &chip.circuit.constraint_system().constants;
				for instance in 0..*n_active {
					let values = table.instance_words(instance, constants);
					for call in &chip.chip_calls {
						pending[call.chip_id][call.first_instance + instance] =
							eval_call(&values, call);
					}
				}
			}

			tables.push(table);
		}

		Ok(WitnessM4 {
			main: main_values,
			tables,
		})
	}
}

/// Evaluates the inout operands of one chip call against the caller's values.
fn eval_call(values: &impl WordSource, call: &ChipCall) -> Vec<Word> {
	call.inout
		.iter()
		.map(|operand| eval_operand(values, operand))
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

/// A chip of the system a [`CircuitBuilder`] is building.
///
/// [`CircuitBuilder::add_chip`] returns one for each chip it
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

/// A gadget the builder emits either as inline gates or as a call to a chip.
///
/// The gadget is written once, in [`build`](Self::build), and where it lands is the building
/// circuit's to decide: [`CircuitBuilder::build_gadget`]
/// emits the gates unless
/// [`CircuitBuilder::register_chip`] has made the gadget a
/// chip, in which case it emits a hint and a call constraining it.
///
/// The [`Hint`] half is what the chip path needs of a gadget beyond its gates. Its `NAME` and
/// `dimensions` name which gadget a registered chip serves; [`shape`](Hint::shape) gives the arity
/// of the gates and of the chip's interface alike; and [`execute`](Hint::execute) computes the
/// outputs a call passes alongside its inputs.
///
/// So a `Hint::execute` and a `build` of the same gadget must agree on every input the circuit can
/// reach them with. Where they disagree, the chip instance recomputes a word the call did not name,
/// and only [`WitnessM4::verify`](binius_core::m4::WitnessM4::verify) reports it.
///
/// ```
/// use binius_core::word::Word;
/// use binius_frontend::{ChipGadget, CircuitBuilder, Hint, Wire};
///
/// /// The bitwise conjunction of two words.
/// struct And;
///
/// impl Hint for And {
///     const NAME: &'static str = "doc.and";
///
///     fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
///         (2, 1)
///     }
///
///     fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
///         outputs[0] = Word(inputs[0].as_u64() & inputs[1].as_u64());
///     }
/// }
///
/// impl ChipGadget for And {
///     fn build(&self, builder: &CircuitBuilder, _dims: &[usize], inputs: &[Wire]) -> Vec<Wire> {
///         vec![builder.band(inputs[0], inputs[1])]
///     }
/// }
///
/// // Without a chip the gadget is its gates, and the circuit builds as any other.
/// let builder = CircuitBuilder::new();
/// let (a, b) = (builder.add_inout(), builder.add_inout());
/// builder.build_gadget(And, &[], &[a, b]);
/// builder.build();
///
/// // Registering the gadget is the whole of the opt-in: the same call is now a chip call.
/// let builder = CircuitBuilder::new();
/// builder.register_chip(And, &[]);
/// let (a, b) = (builder.add_inout(), builder.add_inout());
/// builder.build_gadget(And, &[], &[a, b]);
/// builder.build_m4().validate().unwrap();
/// ```
pub trait ChipGadget: Hint {
	/// Emits the gadget's gates, returning the outputs its
	/// [`shape`](Hint::shape) declares.
	///
	/// Each returned wire must be gate-created: a chip promotes them with
	/// [`CircuitBuilder::mark_inout`], which takes no other
	/// kind. Returning an input or a constant unchanged is what that rules out.
	fn build(&self, builder: &CircuitBuilder, dimensions: &[usize], inputs: &[Wire]) -> Vec<Wire>;
}

/// Reason an M4 circuit cannot be populated as it stands.
#[allow(missing_docs)] // errors are self-documenting
#[derive(Debug, thiserror::Error)]
pub enum CircuitM4Error {
	#[error("{} calls chip {chip_id}, but the system has {n_chips} chips", ChipName(*chip_index))]
	OutOfRangeChipId {
		chip_index: Option<usize>,
		chip_id: usize,
		n_chips: usize,
	},
	#[error(
		"{}'s call #{call_index} passes {arity} operands to chip {chip_id}, which has {n_inout} inout values",
		ChipName(*chip_index)
	)]
	WrongCallArity {
		chip_index: Option<usize>,
		call_index: usize,
		chip_id: usize,
		arity: usize,
		n_inout: usize,
	},
	#[error(
		"{}'s call #{call_index} has a malformed operand #{operand_index}: {source}",
		ChipName(*chip_index)
	)]
	CallOperand {
		chip_index: Option<usize>,
		call_index: usize,
		operand_index: usize,
		#[source]
		source: OperandFault,
	},
	#[error("chip #{chip_index} calls chip {callee}, which is not a later chip")]
	CallOutOfOrder { chip_index: usize, callee: usize },
	#[error(
		"{}'s call #{call_index} names instance {first_instance}, but the call graph gives it {expected}",
		ChipName(*chip_index)
	)]
	WrongCallInstance {
		chip_index: Option<usize>,
		call_index: usize,
		first_instance: usize,
		expected: usize,
	},
	#[error("chip #{chip_index} declares {declared} active instances, but {actual} calls reach it")]
	WrongActiveInstanceCount {
		chip_index: usize,
		declared: usize,
		actual: usize,
	},
	#[error("chip #{chip_index} is never called")]
	NeverCalled { chip_index: usize },
	#[error("more invocations reach chip #{chip_index} than a usize can count")]
	TooManyInstances { chip_index: usize },
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_core::{ShiftedValueIndex, ValueIndex, VerificationM4Error, error::OperandFault};

	use super::*;
	use crate::{Circuit, CircuitBuilder, CircuitM4Error, EmbeddedCircuit, Wire};

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
				first_instance: 0,
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
			first_instance: 0,
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
				first_instance: 0,
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
		circuit.recompute_instances();
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
		let witness = circuit
			.generate_witness(|filler| {
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
		let witness = circuit
			.generate_witness(|filler| {
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

		let witness = circuit.generate_witness(fill).unwrap();
		assert_eq!(row(&circuit, &witness), vec![0b1100, 0b1010, 0b1000]);
		witness.verify(&circuit.to_constraint_system()).unwrap();

		// Pass `a` as the third word, which the chip's conjunction disagrees with. Generation still
		// succeeds, and the row still holds the conjunction rather than what the call passed — so
		// the call no longer matches its instance, which is exactly what verification rejects.
		circuit.main.chip_calls[0].inout[2] = operand(&circuit.main.circuit, a);
		let witness = circuit.generate_witness(fill).unwrap();
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

	// A caller with several instances making several calls to one callee is what tells the two
	// apart: each call site claims a contiguous block of the callee's instances, one per instance
	// of the caller, rather than the caller's instances taking consecutive rows. With one call per
	// callee the two coincide and nothing would notice generation and verification disagreeing.
	#[test]
	fn generate_gives_each_call_site_a_contiguous_block_of_instances() {
		let (main, inputs) = main_circuit(2);
		let mut circuit = CircuitM4 {
			main,
			chips: vec![(twice_calling_and_chip(1), 0), (eq_chip(), 0)],
		};
		circuit.recompute_instances();
		circuit.validate().unwrap();

		// Chip 0 serves main's two calls, and each of its instances calls chip 1 twice.
		assert_eq!(circuit.chips[0].1, 2);
		assert_eq!(circuit.chips[1].1, 4);

		let words = [(0b1100u64, 0b1010u64), (0xff00, 0x0ff0)];
		let witness = circuit
			.generate_witness(|filler| {
				for (&(a, b), &(a_word, b_word)) in iter::zip(&inputs, &words) {
					filler[a] = Word(a_word);
					filler[b] = Word(b_word);
				}
			})
			.unwrap();

		// The first call site forwards `(a, a)` from each of chip 0's two instances, and the second
		// forwards `(b, b)` from each, so the two blocks are `a`s then `b`s.
		let expected = [words[0].0, words[1].0, words[0].1, words[1].1];
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
		let witness = circuit
			.generate_witness(|filler| {
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

		let err = circuit
			.generate_witness(|filler| {
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
			first_instance: 0,
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

	// A call site claims one instance of its callee per instance of its caller, so the instance it
	// names is the one the calls before it leave free. A graph edited without recomputing the
	// instances leaves a call naming a row another one already claims.
	#[test]
	fn validate_rejects_a_call_naming_the_wrong_instance() {
		let (mut circuit, _) = system(2);
		circuit.main.chip_calls[1].first_instance = 0;
		assert!(matches!(
			circuit.validate(),
			Err(CircuitM4Error::WrongCallInstance {
				chip_index: None,
				call_index: 1,
				first_instance: 0,
				expected: 1,
			})
		));
	}

	#[test]
	fn validate_rejects_a_chip_nothing_calls() {
		let (mut circuit, _) = system(1);
		circuit.main.chip_calls.clear();
		circuit.recompute_instances();
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
							first_instance: 0,
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
					first_instance: 0,
					inout: vec![],
				}],
			},
			chips: (0..DEPTH)
				.map(|i| (chip((i + 1 < DEPTH).then_some(i + 1)), 0))
				.collect(),
		};
		circuit.recompute_instances();

		// Chip 63 is reached 2^63 times and calls chip 64 twice, which is where the count leaves
		// the range.
		assert!(matches!(
			circuit.validate(),
			Err(CircuitM4Error::TooManyInstances { chip_index: 64 })
		));
	}
}
