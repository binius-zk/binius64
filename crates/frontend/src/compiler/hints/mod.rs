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
/// FNV-1a hash of this name (see [`hint_id_of`]), so registering the same hint twice is a
/// no-op and every gate using the same hint type shares a single handler entry.
pub trait Hint: Send + Sync + 'static {
	/// Globally unique name for this hint. Used to derive a stable [`HintId`].
	const NAME: &'static str;

	/// Execute the hint with given inputs, writing outputs
	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]);

	/// Get the shape of this hint (n_inputs, n_outputs)
	fn shape(&self, dimensions: &[usize]) -> (usize, usize);
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
	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]);
}

impl<T: Hint> ErasedHint for T {
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
