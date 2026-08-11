// Copyright 2026 The Binius Developers

//! The 64-bit word the wrapper channels carry.

use std::{cell::RefCell, iter, ops::Shr, rc::Rc};

use binius_core::word::Word;
use binius_field::{BinaryField1b as B1, ExtensionField, Field};
use binius_ip::channel::{select_word, subset_sum_word};
use binius_spartan_frontend::circuit_builder::CircuitBuilder;

use super::circuit_elem::CircuitElem;

/// A 64-bit word that is either fixed while the circuit is built or carried by wires.
///
/// This is the `Word` associated type of the wrapper channels. The statement a wrapped verifier
/// checks arrives this way, and it has to: the wrapper's circuit is built once, before any
/// statement exists, so an operation reading a concrete word's bits would settle its answer from
/// whatever stood in for the statement at build time, and the run on the real one would not fit
/// the circuit that produced.
///
/// A word the protocol fixes — a constraint system constant — is the same in every run, so it
/// stays a `Constant` and folds for free.
pub enum CircuitWord<F: Field, B: CircuitBuilder<Field = F>> {
	Constant(Word),
	/// The word's bits, low first, each holding zero or one.
	Bits(Rc<Vec<CircuitElem<F, B>>>),
}

// Manual `Clone` that does not require `B: Clone`, as on [`CircuitElem`].
impl<F: Field, B: CircuitBuilder<Field = F>> Clone for CircuitWord<F, B> {
	fn clone(&self) -> Self {
		match self {
			Self::Constant(word) => Self::Constant(*word),
			Self::Bits(bits) => Self::Bits(Rc::clone(bits)),
		}
	}
}

impl<F, B> CircuitWord<F, B>
where
	F: Field + ExtensionField<B1>,
	B: CircuitBuilder<Field = F>,
{
	/// The field element a word is carried as: its bits as the `B1` coefficients.
	pub fn embed(word: Word) -> F {
		F::from_bases((0..Word::BITS).map(|bit| {
			if word.extract_bit(bit) {
				B1::ONE
			} else {
				B1::ZERO
			}
		}))
	}

	/// Reads the word `wire` carries as its bits.
	///
	/// The bits are hinted rather than constrained. A hint over public inputs is not a value a
	/// prover chooses: the wire is public, so the outer verifier recomputes the bits for itself,
	/// and they cost no constraint for the same reason.
	pub fn from_wire(builder: &Rc<RefCell<B>>, wire: B::Wire) -> Self {
		let bit_wires =
			builder
				.borrow_mut()
				.hint_varsize(std::slice::from_ref(&wire), Word::BITS, |vals| {
					(0..Word::BITS)
						.map(|bit| F::from(vals[0].get_base(bit)))
						.collect()
				});
		Self::Bits(Rc::new(
			bit_wires
				.into_iter()
				.map(|bit| CircuitElem::wire(builder, bit))
				.collect(),
		))
	}
}

impl<F, B> CircuitWord<F, B>
where
	F: Field,
	B: CircuitBuilder<Field = F>,
{
	/// The sum of the `elems` selected by this word's low bits, low bit first.
	///
	/// This is [`WordIPVerifierChannel::subset_sum`](binius_ip::channel::WordIPVerifierChannel::subset_sum)
	/// over one word. Every term is a product of public-derivable values, so the sub-circuit it
	/// builds emits no constraint; the outer verifier evaluates it.
	///
	/// ## Preconditions
	///
	/// * `elems.len()` must be at most 64.
	pub fn subset_sum(&self, elems: &[CircuitElem<F, B>]) -> CircuitElem<F, B> {
		assert!(elems.len() <= Word::BITS); // precondition

		match self {
			Self::Constant(word) => subset_sum_word(elems, *word),
			// A bit is zero or one, so multiplying by it keeps or drops its element.
			Self::Bits(bits) => iter::zip(elems, bits.iter())
				.map(|(elem, bit)| elem.clone() * bit)
				.sum(),
		}
	}

	/// The element of `elems` at the index in this word's low bits.
	///
	/// ## Preconditions
	///
	/// * `elems` must be non-empty and its length must be a power of two.
	pub fn select(&self, elems: &[CircuitElem<F, B>]) -> CircuitElem<F, B> {
		assert!(!elems.is_empty() && elems.len().is_power_of_two()); // precondition

		match self {
			Self::Constant(word) => select_word(elems, *word),
			Self::Bits(bits) => {
				// Halve the layer at each index bit, low bit first: `lo + b * (lo + hi)`, which is
				// `(1 - b) * lo + b * hi` in characteristic two.
				let mut layer = elems.to_vec();
				for bit in bits.iter().take(elems.len().ilog2() as usize) {
					layer = layer
						.chunks_exact(2)
						.map(|pair| {
							let (lo, hi) = (&pair[0], &pair[1]);
							lo.clone() + bit.clone() * (lo.clone() + hi)
						})
						.collect();
				}
				layer
					.pop()
					.expect("halving a power-of-two layer ends at one element")
			}
		}
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> From<Word> for CircuitWord<F, B> {
	fn from(word: Word) -> Self {
		Self::Constant(word)
	}
}

impl<F: Field, B: CircuitBuilder<Field = F>> Shr<u32> for CircuitWord<F, B> {
	type Output = Self;

	fn shr(self, rhs: u32) -> Self {
		match self {
			Self::Constant(word) => Self::Constant(word >> rhs),
			Self::Bits(bits) => Self::Bits(Rc::new(
				bits.iter()
					.skip(rhs as usize)
					.cloned()
					.chain(iter::repeat_with(|| CircuitElem::Constant(F::ZERO)))
					.take(Word::BITS)
					.collect(),
			)),
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField128bGhash as B128;
	use binius_spartan_frontend::circuit_builder::ConstraintBuilder;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;

	type TestWord = CircuitWord<B128, ConstraintBuilder<B128>>;
	type TestElem = CircuitElem<B128, ConstraintBuilder<B128>>;

	/// The bits of `word` as constant elements.
	///
	/// A `Bits` word normally holds wires, but the arithmetic over them does not care which, so
	/// constants exercise it while staying readable: every result folds to a value.
	fn bits_of(word: Word) -> TestWord {
		CircuitWord::Bits(Rc::new(
			(0..Word::BITS)
				.map(|bit| {
					CircuitElem::Constant(if word.extract_bit(bit) {
						B128::ONE
					} else {
						B128::ZERO
					})
				})
				.collect(),
		))
	}

	fn value(elem: &TestElem) -> B128 {
		match elem {
			CircuitElem::Constant(value) => *value,
			CircuitElem::Wire { .. } => panic!("constant inputs should fold to a constant"),
		}
	}

	fn random_elems(rng: &mut StdRng, n: usize) -> Vec<TestElem> {
		(0..n)
			.map(|_| CircuitElem::Constant(B128::new(rng.random::<u128>())))
			.collect()
	}

	#[test]
	fn test_subset_sum_over_bits_matches_the_concrete_word() {
		let mut rng = StdRng::seed_from_u64(0);
		let elems = random_elems(&mut rng, 40);

		for _ in 0..8 {
			let word = Word::from_u64(rng.random::<u64>());
			assert_eq!(
				value(&bits_of(word).subset_sum(&elems)),
				value(&TestWord::Constant(word).subset_sum(&elems)),
			);
		}
	}

	#[test]
	fn test_select_over_bits_matches_the_concrete_word() {
		let mut rng = StdRng::seed_from_u64(0);
		let elems = random_elems(&mut rng, 16);

		for _ in 0..8 {
			let word = Word::from_u64(rng.random::<u64>());
			assert_eq!(
				value(&bits_of(word).select(&elems)),
				value(&TestWord::Constant(word).select(&elems)),
			);
		}
	}

	#[test]
	fn test_shift_over_bits_matches_the_concrete_word() {
		let mut rng = StdRng::seed_from_u64(0);
		let elems = random_elems(&mut rng, 40);

		for shift in [0, 1, 7, 63] {
			let word = Word::from_u64(rng.random::<u64>());
			assert_eq!(
				value(&(bits_of(word) >> shift).subset_sum(&elems)),
				value(&TestWord::Constant(word >> shift).subset_sum(&elems)),
			);
		}
	}

	/// The bits a statement wire is read as are the bits of the word it was written from.
	#[test]
	fn test_embed_round_trips_through_the_bases() {
		let mut rng = StdRng::seed_from_u64(0);

		for _ in 0..8 {
			let word = Word::from_u64(rng.random::<u64>());
			let embedded = TestWord::embed(word);
			for bit in 0..Word::BITS {
				let base = <B128 as ExtensionField<B1>>::get_base(&embedded, bit);
				assert_eq!(base == B1::ONE, word.extract_bit(bit));
			}
		}
	}
}
