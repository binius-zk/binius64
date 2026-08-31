// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Hint system.
//!
//! Hints are deterministic computations that happen on the prover side.
//!
//! They can be used for operations that require many constraints to compute but few constraints
//! to verify.

use std::{
	collections::{HashMap, hash_map::Entry},
	hash::{DefaultHasher, Hash, Hasher},
};

use binius_core::Word;

/// Registry key for one prover-side computation.
///
/// Derived from the declared name rather than assigned in order, so it is stable across runs.
pub type HintId = u32;

/// Hint handler trait for extensible operations.
///
/// Each implementor declares a globally unique name, and the registry keys on the hash of that
/// name alone.
///
/// Every gate using the same hint type therefore shares one handler entry.
///
/// A hint's fields are not part of its identity.
/// Only the first value registered under a name is kept; later values are dropped.
/// Gates that differ only in those fields fold together under deduplication.
/// Parameterize a hint through its dimensions, never through its fields.
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
	/// Globally unique name for this hint. Used to derive a stable [`HintId`].
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

/// Derive a [`HintId`] from a hint's name.
///
/// Hashes the name and folds the resulting 64-bit value down to 32 bits by XORing its two halves.
///
/// The hash algorithm is unspecified, so an id is stable only within one process.
/// Never persist an id or compare one across builds.
pub fn hint_id_of(name: &str) -> HintId {
	let mut hasher = DefaultHasher::new();
	name.hash(&mut hasher);
	let h = hasher.finish();
	(h as u32) ^ ((h >> 32) as u32)
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
		<T as Hint>::execute(self, dimensions, inputs, outputs);
	}
}

/// Registry for hint handlers keyed by [`HintId`].
///
/// Each entry keeps the name it was registered under.
/// An id shared by two names is then caught, instead of running one hint as the other.
pub struct HintRegistry {
	handlers: HashMap<HintId, (&'static str, Box<dyn ErasedHint>)>,
}

impl HintRegistry {
	/// An empty registry.
	pub fn new() -> Self {
		Self {
			handlers: HashMap::new(),
		}
	}

	/// Register a hint, returning its [`HintId`].
	///
	/// Registering a name already present is a no-op, the handler's fields included.
	///
	/// # Panics
	///
	/// Panics if a different name already holds this id.
	pub fn register<T: Hint>(&mut self, handler: T) -> HintId {
		let id = hint_id_of(T::NAME);
		match self.handlers.entry(id) {
			Entry::Vacant(entry) => {
				entry.insert((T::NAME, Box::new(handler)));
			}
			Entry::Occupied(entry) => assert_eq!(
				entry.get().0,
				T::NAME,
				"hint id collision: {} and {}",
				entry.get().0,
				T::NAME,
			),
		}
		id
	}

	/// Compute the `(n_in, n_out)` arity of the hint identified by `hint_id`.
	pub fn shape(&self, hint_id: HintId, dimensions: &[usize]) -> (usize, usize) {
		self.handlers[&hint_id].1.shape(dimensions)
	}

	/// Run the handler under `hint_id`, writing its results into `outputs`.
	///
	/// # Panics
	///
	/// Panics if nothing is registered under `hint_id`.
	pub fn execute(
		&self,
		hint_id: HintId,
		dimensions: &[usize],
		inputs: &[Word],
		outputs: &mut [Word],
	) {
		let handler = &self.handlers[&hint_id].1;
		handler.execute(dimensions, inputs, outputs);
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

	struct AddK {
		k: u64,
	}

	impl Hint for AddK {
		const NAME: &'static str = "test::add_k";

		fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
			(1, 1)
		}

		fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
			outputs[0] = Word(inputs[0].0.wrapping_add(self.k));
		}
	}

	fn run(registry: &HintRegistry, id: HintId, input: u64) -> u64 {
		let mut outputs = [Word::ZERO];
		registry.execute(id, &[], &[Word(input)], &mut outputs);
		outputs[0].0
	}

	#[test]
	fn re_registering_one_hint_type_keeps_a_single_entry() {
		let mut registry = HintRegistry::new();
		let first = registry.register(AddK { k: 7 });
		let second = registry.register(AddK { k: 7 });
		assert_eq!(first, second);
		assert_eq!(registry.handlers.len(), 1);
	}

	// Pins the known limitation the trait doc warns about.
	#[test]
	fn fields_of_a_second_registration_are_ignored() {
		let mut registry = HintRegistry::new();
		let id = registry.register(AddK { k: 7 });
		registry.register(AddK { k: 1000 });
		assert_eq!(run(&registry, id, 0), 7);
	}

	// A 32-bit collision is out of reach of a search over static names, so it is planted.
	#[test]
	#[should_panic(expected = "hint id collision: test::squatter and test::add_k")]
	fn colliding_names_panic() {
		let mut registry = HintRegistry::new();
		registry
			.handlers
			.insert(hint_id_of(AddK::NAME), ("test::squatter", Box::new(AddK { k: 7 })));
		registry.register(AddK { k: 7 });
	}
}
