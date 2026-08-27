// Copyright 2026 The Binius Developers

//! An [`IOPVerifierChannel`] dry run that records the [`OracleSpec`] sequence an IOP uses.

use std::{
	iter::{Product, Sum},
	marker::PhantomData,
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_core::word::Word;
use binius_field::{
	BinaryField, ExtensionField, Field, FieldOps,
	arithmetic_traits::{InvertOrZero, Square},
};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};

use crate::channel::{Error, IOPVerifierChannel, OracleSchedule, OracleSpec, TransparentEvalFn};

/// A dummy field element for [`OracleSetupChannel`], generic over the field `F` it stands in for.
///
/// The setup channel performs no real verification, so the field values flowing through it are
/// never inspected. `DummyElem<F>` is a zero-sized stand-in whose arithmetic is all no-ops; the
/// `PhantomData<F>` lets it satisfy `FieldOps<Scalar = F>` without doing (pointless) real field
/// arithmetic during the structural dry run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DummyElem<F>(PhantomData<F>);

macro_rules! dummy_binop {
	($trait:ident, $method:ident) => {
		impl<F> $trait for DummyElem<F> {
			type Output = Self;
			fn $method(self, _rhs: Self) -> Self {
				self
			}
		}
		impl<F> $trait<&DummyElem<F>> for DummyElem<F> {
			type Output = Self;
			fn $method(self, _rhs: &Self) -> Self {
				self
			}
		}
	};
}
dummy_binop!(Add, add);
dummy_binop!(Sub, sub);
dummy_binop!(Mul, mul);

macro_rules! dummy_assign {
	($trait:ident, $method:ident) => {
		impl<F> $trait for DummyElem<F> {
			fn $method(&mut self, _rhs: Self) {}
		}
		impl<F> $trait<&DummyElem<F>> for DummyElem<F> {
			fn $method(&mut self, _rhs: &Self) {}
		}
	};
}
dummy_assign!(AddAssign, add_assign);
dummy_assign!(SubAssign, sub_assign);
dummy_assign!(MulAssign, mul_assign);

impl<F> Neg for DummyElem<F> {
	type Output = Self;
	fn neg(self) -> Self {
		self
	}
}

impl<F> Sum for DummyElem<F> {
	fn sum<I: Iterator<Item = Self>>(_iter: I) -> Self {
		Self(PhantomData)
	}
}
impl<'a, F> Sum<&'a DummyElem<F>> for DummyElem<F> {
	fn sum<I: Iterator<Item = &'a Self>>(_iter: I) -> Self {
		Self(PhantomData)
	}
}
impl<F> Product for DummyElem<F> {
	fn product<I: Iterator<Item = Self>>(_iter: I) -> Self {
		Self(PhantomData)
	}
}
impl<'a, F> Product<&'a DummyElem<F>> for DummyElem<F> {
	fn product<I: Iterator<Item = &'a Self>>(_iter: I) -> Self {
		Self(PhantomData)
	}
}

impl<F> Square for DummyElem<F> {
	fn square(self) -> Self {
		self
	}
}
impl<F> InvertOrZero for DummyElem<F> {
	fn invert_or_zero(self) -> Self {
		self
	}
}

impl<F> From<F> for DummyElem<F> {
	fn from(_value: F) -> Self {
		Self(PhantomData)
	}
}

impl<F: Field> FieldOps for DummyElem<F> {
	type Scalar = F;

	fn zero() -> Self {
		Self(PhantomData)
	}

	fn one() -> Self {
		Self(PhantomData)
	}

	fn square_transpose<FSub: Field>(_elems: &mut [Self])
	where
		F: ExtensionField<FSub>,
	{
	}
}

/// An [`IOPVerifierChannel`] that records the [`OracleSpec`] of each received oracle.
///
/// This performs no verification.
///
/// Every receive returns a dummy value, and sampling, observing, and asserting are no-ops.
///
/// Drive an IOP verifier with it, then read back what it recorded.
///
/// The channel carries one protocol-level zero-knowledge flag.
///
/// An oracle is zero-knowledge only if that flag is set and the oracle is witness-dependent.
///
/// # What is recorded
///
/// Two views of the same dry run, differing only in whether round boundaries survive.
///
/// - The flat sequence of oracles, in arrival order.
/// - That same sequence, grouped into the rounds the oracles are committed in.
///
/// A round closes wherever a challenge could be drawn.
///
/// That is where a decorator committing a round as one oracle stops taking members.
#[derive(Debug, Default, Clone)]
pub struct OracleSetupChannel {
	is_zk: bool,
	schedule: OracleSchedule,
}

impl OracleSetupChannel {
	/// Creates a new setup channel with the given protocol-level zero-knowledge flag.
	pub const fn new(is_zk: bool) -> Self {
		Self {
			is_zk,
			schedule: OracleSchedule::new(),
		}
	}

	/// Returns the oracle specs recorded so far.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		self.schedule.specs()
	}

	/// Consumes the channel and returns the recorded oracle specs, in the order received.
	pub fn into_oracle_specs(self) -> Vec<OracleSpec> {
		self.schedule.into_specs()
	}

	/// Consumes the channel and returns the recorded oracles, grouped by commit round.
	///
	/// The round left open at the end of the dry run is closed here.
	///
	/// So every recorded oracle belongs to some round.
	pub fn into_oracle_schedule(mut self) -> OracleSchedule {
		self.schedule.end_round();
		self.schedule
	}
}

impl<F: Field> IPVerifierChannel<F> for OracleSetupChannel {
	type Elem = DummyElem<F>;

	fn recv_one(&mut self) -> Result<DummyElem<F>, binius_ip::channel::Error> {
		Ok(DummyElem(PhantomData))
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<DummyElem<F>>, binius_ip::channel::Error> {
		Ok(vec![DummyElem(PhantomData); n])
	}

	fn recv_array<const N: usize>(
		&mut self,
	) -> Result<[DummyElem<F>; N], binius_ip::channel::Error> {
		Ok([DummyElem(PhantomData); N])
	}

	fn sample(&mut self) -> DummyElem<F> {
		// A challenge follows the commitments it is derived from, so it ends their round.
		self.schedule.end_round();
		DummyElem(PhantomData)
	}

	fn observe_one(&mut self, _val: F) -> DummyElem<F> {
		DummyElem(PhantomData)
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<DummyElem<F>> {
		vec![DummyElem(PhantomData); vals.len()]
	}

	fn assert_zero(&mut self, _val: DummyElem<F>) -> Result<(), binius_ip::channel::Error> {
		Ok(())
	}
}

impl<F: BinaryField> WordIPVerifierChannel<F> for OracleSetupChannel {
	type Word = Word;

	// The dry run records oracle shapes only, so nothing reaches a Fiat-Shamir state.
	fn observe_words(&mut self, words: &[Word]) -> Vec<Word> {
		words.to_vec()
	}

	fn subset_sum(&mut self, _elems: &[DummyElem<F>], _word: &Word) -> DummyElem<F> {
		DummyElem(PhantomData)
	}

	fn select(&mut self, _elems: &[DummyElem<F>], _word: &Word) -> DummyElem<F> {
		DummyElem(PhantomData)
	}

	// The recorded oracle shapes do not depend on which leaves a protocol would query.
	fn sample_bits(&mut self, _bits: usize) -> Word {
		// A sampled word is a challenge like any other, so it ends the open round too.
		self.schedule.end_round();
		Word::ZERO
	}

	// Only the element count matters here, and it follows from the word count alone.
	fn pack_words(&mut self, words: &[Word]) -> Vec<DummyElem<F>> {
		let words_per_elem = F::N_BITS / Word::BITS;
		vec![DummyElem(PhantomData); words.len().div_ceil(words_per_elem)]
	}
}

impl<F: Field> IOPVerifierChannel<F> for OracleSetupChannel {
	type Oracle = ();

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		// A setup channel has no pre-supplied specs; it records them as they are received.
		&[]
	}

	fn recv_oracle(
		&mut self,
		log_msg_len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error> {
		// A non-witness-dependent oracle is never masked, whatever the protocol-level flag says.
		self.schedule.push(OracleSpec {
			log_msg_len,
			is_zk: self.is_zk && is_witness_dependent,
		});
		Ok(())
	}

	fn verify_oracle_relation(
		&mut self,
		_oracle: Self::Oracle,
		_transparent: TransparentEvalFn<Self::Elem>,
		_claim: Self::Elem,
	) -> Result<(), Error> {
		// Opening an oracle needs every commitment in place, so nothing may still be queued.
		self.schedule.end_round();
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b;

	use super::*;

	type F = Ghash128b;

	/// Draws a challenge, which is what closes a round.
	fn sample(channel: &mut OracleSetupChannel) {
		IPVerifierChannel::<F>::sample(channel);
	}

	/// Receives one witness-carrying oracle of the given length.
	fn recv(channel: &mut OracleSetupChannel, log_msg_len: usize) {
		IOPVerifierChannel::<F>::recv_oracle(channel, log_msg_len, true)
			.expect("the setup channel never fails");
	}

	#[test]
	fn a_challenge_ends_the_open_round() {
		let mut channel = OracleSetupChannel::new(false);

		// Two oracles, a challenge, then one more.
		//
		//     recv 3, recv 3, sample, recv 4, sample
		//     \____________/          \____/
		//        round 0              round 1
		recv(&mut channel, 3);
		recv(&mut channel, 3);
		sample(&mut channel);
		recv(&mut channel, 4);
		sample(&mut channel);

		let schedule = channel.into_oracle_schedule();
		assert_eq!(schedule.n_rounds(), 2);
		assert_eq!(
			schedule.rounds().collect::<Vec<_>>(),
			vec![
				[OracleSpec::new(3), OracleSpec::new(3)].as_slice(),
				[OracleSpec::new(4)].as_slice(),
			]
		);
	}

	#[test]
	fn the_final_round_closes_without_a_challenge() {
		let mut channel = OracleSetupChannel::new(false);

		// The run ends with an oracle, not a challenge.
		// That trailing oracle is still a round of its own.
		recv(&mut channel, 2);
		sample(&mut channel);
		recv(&mut channel, 5);

		let schedule = channel.into_oracle_schedule();
		assert_eq!(schedule.n_rounds(), 2);
		assert_eq!(schedule.rounds().last(), Some([OracleSpec::new(5)].as_slice()));
	}

	#[test]
	fn a_sampled_word_ends_the_open_round_too() {
		let mut channel = OracleSetupChannel::new(false);

		// Query indices are drawn as words rather than as field elements.
		// That is still a challenge, so it still closes the round before it.
		recv(&mut channel, 3);
		WordIPVerifierChannel::<F>::sample_bits(&mut channel, 4);
		recv(&mut channel, 2);

		assert_eq!(channel.into_oracle_schedule().n_rounds(), 2);
	}

	#[test]
	fn back_to_back_challenges_make_no_empty_round() {
		let mut channel = OracleSetupChannel::new(false);

		// A protocol may draw several challenges in a row with nothing committed between them.
		// Only the first of those closes anything.
		recv(&mut channel, 3);
		sample(&mut channel);
		sample(&mut channel);
		sample(&mut channel);

		assert_eq!(channel.into_oracle_schedule().n_rounds(), 1);
	}

	#[test]
	fn a_run_with_no_oracles_has_no_rounds() {
		let mut channel = OracleSetupChannel::new(false);

		// A challenge alone commits nothing, so it closes nothing.
		sample(&mut channel);

		let schedule = channel.into_oracle_schedule();
		assert_eq!(schedule.n_rounds(), 0);
		assert!(schedule.specs().is_empty());
	}

	#[test]
	fn dropping_the_boundaries_leaves_the_flat_sequence() {
		let mut channel = OracleSetupChannel::new(false);

		// Grouping is the only thing the schedule adds.
		// The oracles themselves, and their order, must survive it untouched.
		recv(&mut channel, 3);
		sample(&mut channel);
		recv(&mut channel, 1);
		recv(&mut channel, 4);

		let flat = channel.clone().into_oracle_specs();
		assert_eq!(channel.into_oracle_schedule().into_specs(), flat);
	}

	#[test]
	fn a_round_is_masked_if_any_of_its_oracles_is() {
		// Protocol-level zero-knowledge is on, but not every oracle carries witness data.
		//
		//     round 0: [structural 2^3, witness 2^3] -> 2^4, masked because one member is
		//     round 1: [structural 2^2]              -> 2^2, nothing to hide
		let mut channel = OracleSetupChannel::new(true);
		IOPVerifierChannel::<F>::recv_oracle(&mut channel, 3, false).unwrap();
		IOPVerifierChannel::<F>::recv_oracle(&mut channel, 3, true).unwrap();
		sample(&mut channel);
		IOPVerifierChannel::<F>::recv_oracle(&mut channel, 2, false).unwrap();

		let merged = channel.into_oracle_schedule().merged_specs();
		assert_eq!(merged, vec![OracleSpec::new_zk(4), OracleSpec::new(2)]);
	}
}
