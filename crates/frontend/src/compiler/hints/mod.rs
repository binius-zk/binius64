// Copyright 2025 Irreducible Inc.
//! Hint system.
//!
//! Hints are deterministic computations that happen on the prover side.
//!
//! They can be used for operations that require many constraints to compute but few constraints
//! to verify.

use std::{
	collections::HashMap,
	hash::{DefaultHasher, Hash, Hasher},
};

use binius_core::Word;

mod big_uint_divide;
mod big_uint_mod_pow;
mod mod_inverse;
mod secp256k1_endosplit;

pub use big_uint_divide::BigUintDivideHint;
pub use big_uint_mod_pow::BigUintModPowHint;
pub use mod_inverse::ModInverseHint;
pub use secp256k1_endosplit::Secp256k1EndosplitHint;

pub type HintId = u32;

/// Hint handler trait for extensible operations.
///
/// Each implementor declares a globally unique `NAME`. The registry identifies hints by the
/// hash of this name (see [`hint_id_of`]), so registering the same hint twice is a no-op
/// and every gate using the same hint type shares a single handler entry.
///
/// # The `dimensions` parameter
///
/// Both [`shape`](Hint::shape) and [`execute`](Hint::execute) take a `dimensions: &[usize]`
/// slice. This is hint-defined parameterization for a single gate — the values the caller
/// passes when invoking the hint via
/// [`CircuitBuilder::call_hint`](crate::compiler::CircuitBuilder::call_hint). The same slice
/// is then handed back to `execute` at witness-generation time.
///
/// `dimensions` controls input/output arity: `shape(dimensions) -> (n_in, n_out)` tells the
/// builder how many wires the gate consumes and produces, and `execute` is later called with
/// `inputs.len() == n_in` and `outputs.len() == n_out`. A hint whose arity is fixed
/// (e.g. always 4 inputs / 6 outputs) takes an empty slice and ignores it. A hint that is
/// parameterized over, say, big-integer limb counts takes those counts as `dimensions`.
///
/// Example: `BigUintDivideHint` uses `dimensions = [dividend_limbs, divisor_limbs]` and
/// computes `(n_in, n_out) = (dividend_limbs + divisor_limbs, dividend_limbs + divisor_limbs)`.
/// `Secp256k1EndosplitHint` uses an empty `dimensions` and returns `(4, 6)` unconditionally.
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
/// Hashes the name with `std::hash::DefaultHasher` (fixed seed, deterministic across runs)
/// and folds the resulting 64-bit value down to 32 bits by XORing its two halves.
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
		<T as Hint>::execute(self, dimensions, inputs, outputs)
	}
}

/// Registry for hint handlers keyed by [`HintId`].
///
/// Registration is idempotent: the same hint type always hashes to the same id, so a second
/// call to [`HintRegistry::register`] with the same concrete type is a no-op.
pub struct HintRegistry {
	handlers: HashMap<HintId, Box<dyn ErasedHint>>,
}

impl HintRegistry {
	pub fn new() -> Self {
		Self {
			handlers: HashMap::new(),
		}
	}

	/// Register a hint, returning its stable [`HintId`]. No-op if the same hint is already
	/// registered.
	pub fn register<T: Hint>(&mut self, handler: T) -> HintId {
		let id = hint_id_of(T::NAME);
		self.handlers.entry(id).or_insert_with(|| Box::new(handler));
		id
	}

	/// Compute the `(n_in, n_out)` arity of the hint identified by `hint_id`.
	pub fn shape(&self, hint_id: HintId, dimensions: &[usize]) -> (usize, usize) {
		self.handlers[&hint_id].shape(dimensions)
	}

	pub fn execute(
		&self,
		hint_id: HintId,
		dimensions: &[usize],
		inputs: &[Word],
		outputs: &mut [Word],
	) {
		self.handlers[&hint_id].execute(dimensions, inputs, outputs);
	}
}

impl Default for HintRegistry {
	fn default() -> Self {
		Self::new()
	}
}
