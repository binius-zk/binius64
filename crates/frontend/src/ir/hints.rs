// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Hint system.
//!
//! Hints are deterministic computations that happen on the prover side.
//!
//! They can be used for operations that require many constraints to compute but few constraints
//! to verify.

use std::{
	any::TypeId,
	collections::{HashMap, hash_map::Entry},
};

use binius_core::Word;

/// Names one hint by the slot it holds in a registry.
///
/// Slots are handed out from zero in registration order.
/// So an identifier only means something against the registry that issued it.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HintId(u32);

impl HintId {
	/// The identifier as the bytecode encodes it.
	pub(crate) const fn as_u32(self) -> u32 {
		self.0
	}

	/// Reads an identifier back from its bytecode encoding.
	pub(crate) const fn from_u32(raw: u32) -> Self {
		Self(raw)
	}
}

/// Hint handler trait for extensible operations.
///
/// A registry holds one slot per implementing type.
///
/// # The `dimensions` parameter
///
/// Both [`shape`](Hint::shape) and [`execute`](Hint::execute) take a `dimensions: &[usize]`
/// slice. This is hint-defined parameterization for a single gate — the values the caller
/// passes when invoking the hint via
/// [`CircuitBuilder::call_hint`](crate::builder::CircuitBuilder::call_hint). The same slice
/// is then handed back to `execute` at witness-generation time.
///
/// `dimensions` controls input/output arity: `shape(dimensions) -> (n_in, n_out)` tells the
/// builder how many wires the gate consumes and produces, and `execute` is later called with
/// `inputs.len() == n_in` and `outputs.len() == n_out`. A hint whose arity is fixed
/// (e.g. always 4 inputs / 6 outputs) takes an empty slice and ignores it. A hint that is
/// parameterized over, say, big-integer limb counts takes those counts as `dimensions`.
///
/// Two arity modes illustrate the contract:
/// - A parameterized hint reads limb counts from `dimensions` and derives its arity from them.
/// - A fixed-arity hint ignores `dimensions` (an empty slice) and returns a constant shape.
pub trait Hint: Send + Sync + 'static {
	/// A label for this hint, used in diagnostics.
	const NAME: &'static str;

	/// Compute the gate's input/output arity as a function of `dimensions`.
	///
	/// Called once when the gate is emitted by `call_hint` to allocate output wires. The
	/// returned `(n_in, n_out)` is the contract for the matching [`execute`](Hint::execute)
	/// call: the builder will provide `n_in` input wires and expect `n_out` outputs.
	///
	/// Implementations must be a pure function of `dimensions` and must agree with
	/// [`execute`](Hint::execute) on the same `dimensions`.
	fn shape(&self, dimensions: &[usize]) -> (usize, usize);

	/// Compute the hint's outputs from its inputs at witness-generation time.
	///
	/// Receives the same `dimensions` slice that was passed to [`shape`](Hint::shape) when the
	/// gate was emitted. `inputs.len() == n_in` and `outputs.len() == n_out` where
	/// `(n_in, n_out) == self.shape(dimensions)`. Implementations write all `n_out` output
	/// slots — including zero-padding when the natural result has fewer significant words.
	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]);
}

/// Object-safe adapter so the registry can store hints behind `Box<dyn _>`.
///
/// `Hint` itself is not dyn-compatible because it carries an associated `const NAME`.
/// A blanket impl adapts any `Hint` to this trait.
trait ErasedHint: Send + Sync {
	fn shape(&self, dimensions: &[usize]) -> (usize, usize);
	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]);
}

impl<T: Hint> ErasedHint for T {
	fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
		<T as Hint>::shape(self, dimensions)
	}

	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		<T as Hint>::execute(self, dimensions, inputs, outputs)
	}
}

/// Holds the hints a circuit calls, one slot per hint type.
pub struct HintRegistry {
	/// The registered hints, in registration order.
	///
	/// An identifier indexes this vector directly.
	handlers: Vec<Box<dyn ErasedHint>>,
	/// The slot each hint type holds.
	///
	/// The concrete type is a hint's identity, which no two distinct hints can share.
	slots: HashMap<TypeId, HintId>,
}

impl HintRegistry {
	pub fn new() -> Self {
		Self {
			handlers: Vec::new(),
			slots: HashMap::new(),
		}
	}

	/// Registers a hint and returns its identifier.
	///
	/// A hint already registered keeps its identifier.
	/// The handler passed here is then dropped.
	pub fn register<T: Hint>(&mut self, handler: T) -> HintId {
		// Borrowing the two fields apart lets the vacant arm append while the map entry is held.
		let Self { handlers, slots } = self;
		match slots.entry(TypeId::of::<T>()) {
			Entry::Occupied(slot) => *slot.get(),
			Entry::Vacant(slot) => {
				let id = HintId::from_u32(
					u32::try_from(handlers.len()).expect("a circuit calls fewer than 2^32 hints"),
				);
				handlers.push(Box::new(handler));
				*slot.insert(id)
			}
		}
	}

	/// Computes the input and output arity of one hint, for the given dimensions.
	pub fn shape(&self, hint_id: HintId, dimensions: &[usize]) -> (usize, usize) {
		self.handler(hint_id).shape(dimensions)
	}

	/// Runs one hint over its inputs, filling every output slot.
	pub fn execute(
		&self,
		hint_id: HintId,
		dimensions: &[usize],
		inputs: &[Word],
		outputs: &mut [Word],
	) {
		self.handler(hint_id).execute(dimensions, inputs, outputs);
	}

	/// The hint one identifier names.
	///
	/// # Panics
	///
	/// Panics unless this registry issued the identifier.
	fn handler(&self, hint_id: HintId) -> &dyn ErasedHint {
		self.handlers
			.get(hint_id.as_u32() as usize)
			.map(Box::as_ref)
			.unwrap_or_else(|| {
				panic!("no hint is registered under id {}", hint_id.as_u32());
			})
	}
}

impl Default for HintRegistry {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	// Doubles its input.
	struct Doubler;

	impl Hint for Doubler {
		// This name and "hint_50152" both fold to 407677619 under a 32-bit hash of the name.
		// So a name cannot serve as an identity.
		const NAME: &'static str = "hint_33476";

		fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
			(1, 1)
		}

		fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
			outputs[0] = Word::from_u64(inputs[0].as_u64().wrapping_mul(2));
		}
	}

	// Splits its input into low and high halves, for an arity the doubler does not share.
	struct SplitHalves;

	impl Hint for SplitHalves {
		const NAME: &'static str = "hint_50152";

		fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
			(1, 2)
		}

		fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
			outputs[0] = Word::from_u64(inputs[0].as_u64() & 0xffff_ffff);
			outputs[1] = Word::from_u64(inputs[0].as_u64() >> 32);
		}
	}

	#[test]
	fn hints_whose_names_share_a_hash_keep_their_own_slots() {
		// Invariant: a hint's identity is its concrete type, not its name.
		//
		// Both names fold to one 32-bit hash.
		// Keying on that hash would give the two hints a single slot.
		// The second registration would then resolve to the first hint's handler.
		let mut registry = HintRegistry::new();
		let doubler = registry.register(Doubler);
		let split = registry.register(SplitHalves);

		//     slot 0 <- Doubler
		//     slot 1 <- SplitHalves
		assert_ne!(doubler, split);

		// Each identifier reports the arity of its own hint.
		assert_eq!(registry.shape(doubler, &[]), (1, 1));
		assert_eq!(registry.shape(split, &[]), (1, 2));

		// 0x0000_000b_0000_0007 doubles to 0x0000_0016_0000_000e, and splits to (7, 11).
		let input = [Word::from_u64(0x0000_000b_0000_0007)];

		let mut doubled = [Word::ZERO];
		registry.execute(doubler, &[], &input, &mut doubled);
		assert_eq!(doubled, [Word::from_u64(0x0000_0016_0000_000e)]);

		let mut halves = [Word::ZERO; 2];
		registry.execute(split, &[], &input, &mut halves);
		assert_eq!(halves, [Word::from_u64(7), Word::from_u64(11)]);
	}

	#[test]
	fn registering_one_hint_twice_reuses_its_slot() {
		// Invariant: registration is idempotent, so a hint called from many gates costs one slot.
		let mut registry = HintRegistry::new();

		let first = registry.register(Doubler);
		let second = registry.register(Doubler);

		assert_eq!(first, second);
		assert_eq!(registry.handlers.len(), 1);
	}

	#[test]
	fn slots_are_handed_out_from_zero_in_registration_order() {
		// Invariant: an identifier is a position.
		// So resolving one is an index, not a probe.
		let mut registry = HintRegistry::new();

		assert_eq!(registry.register(Doubler), HintId::from_u32(0));
		assert_eq!(registry.register(SplitHalves), HintId::from_u32(1));
	}

	#[test]
	#[should_panic(expected = "no hint is registered under id 7")]
	fn an_identifier_this_registry_never_issued_names_the_id_it_rejects() {
		// A registry holding one hint has only slot 0, so slot 7 cannot resolve.
		let mut registry = HintRegistry::new();
		registry.register(Doubler);

		registry.shape(HintId::from_u32(7), &[]);
	}
}
