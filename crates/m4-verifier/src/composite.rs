// Copyright 2026 The Binius Developers

//! Verifying a whole [`ConstraintSystemM4`]: the main chip and every numbered chip.

use std::marker::PhantomData;

use binius_core::{
	Word,
	constraint_system::{InoutSegment, m4::ConstraintSystemM4},
};
use binius_field::FieldOps;
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop::{
	basefold::compiler::BaseFoldVerifierCompiler,
	channel::{IOPVerifierChannel, OracleSpec},
	fri::{MinProofSizeStrategy, calculate_n_test_queries},
};
use binius_ip::channel::WordIPVerifierChannel;
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, checked_arithmetics::log2_ceil_usize};
use binius_verifier::{Error, SECURITY_BITS, config::B128};
use digest::Output;

use crate::{
	commit::BatchCommitLayout,
	verify::{IOPVerifier, Scheme},
};

/// IOP verifier for a composite M4 system.
///
/// The main chip runs once, so it is verified by the Binius64 verifier: its inout values are the
/// statement, known to the verifier and not committed. Each numbered chip runs once per call that
/// reaches it, so it is verified by this crate's data-parallel [`IOPVerifier`] over a batch of
/// instances, with its inout values committed alongside the private ones.
///
/// The sub-proofs share one channel, and so one Fiat-Shamir transcript and one combined FRI
/// opening. They are otherwise independent: no claim crosses between them.
///
/// ## What a composite proof does not prove
///
/// Chip calls are unconstrained. A proof therefore establishes that main's constraints hold and
/// that each chip's constraints hold on its own committed instances — it does *not* establish that
/// the instances are the ones main's calls asked for, nor that they were called at all.
/// [`ConstraintSystemM4::verify`] checks strictly more than this argument does. Constraining chip
/// calls is future work.
#[derive(Debug, Clone)]
pub struct IOPVerifierM4 {
	/// The verifier for the main chip, which runs once.
	main: binius_verifier::IOPVerifier,
	/// The verifier for each numbered chip's batch of instances, indexed by chip ID.
	chips: Vec<IOPVerifier>,
}

impl IOPVerifierM4 {
	/// Builds the sub-verifiers for a validated composite system.
	///
	/// A chip's instance count is not carried by the system: it is derived from the chip's active
	/// count, rounded up to a power of two. Witness generation derives it the same way, which is
	/// what holds the committed batch and the oracle to one size.
	pub fn new(cs: &ConstraintSystemM4) -> Self {
		// The main chip's inout values are the statement, so its public segment spans the
		// constants and the inout values both.
		let main = binius_verifier::IOPVerifier::new(
			cs.main.cs.clone(),
			cs.main.cs.log_public_words(InoutSegment::Public),
		);

		let chips = cs
			.chips
			.iter()
			.map(|(chip, n_active)| {
				let log_instances = log2_ceil_usize(*n_active);
				let layout = BatchCommitLayout::for_constraint_system(&chip.cs, log_instances);
				IOPVerifier::new(chip.cs.clone(), layout)
			})
			.collect();

		Self { main, chips }
	}

	/// The verifier for the main chip.
	pub const fn main(&self) -> &binius_verifier::IOPVerifier {
		&self.main
	}

	/// The verifiers for the numbered chips, indexed by chip ID.
	pub fn chips(&self) -> &[IOPVerifier] {
		&self.chips
	}

	/// Returns the oracle specs the prover commits to, over the whole system.
	///
	/// Each sub-verifier derives its own specs by replaying its verification against an
	/// [`OracleSetupChannel`](binius_iop::channel::oracle_setup::OracleSetupChannel), so no list is
	/// hand-maintained. Concatenating them in the order [`Self::verify`] runs the sub-proofs gives
	/// the sequence one composite proof commits.
	///
	/// The specs come from the sub-verifiers rather than from one replay of [`Self::verify`]
	/// because a replay stops at the first sub-proof that errors, and the Binius64 verifier's own
	/// replay tolerates such an error rather than reporting it. Composing the replays would
	/// therefore silently truncate the list, and the FRI parameters derived from it would be sized
	/// for only part of the system.
	///
	/// A composite proof is transparent, so no oracle is masked.
	pub fn oracle_specs(&self) -> Vec<OracleSpec> {
		self.main
			.oracle_specs(false)
			.into_iter()
			.chain(self.chips.iter().flat_map(IOPVerifier::oracle_specs))
			.collect()
	}

	/// Verifies one composite proof using an IOP channel.
	///
	/// The sub-proofs run on one channel in one fixed order — the main chip, then the chips in ID
	/// order — which the prover must match exactly, since one channel is one Fiat-Shamir
	/// transcript. It is the order [`ConstraintSystemM4::verify`] and witness generation walk the
	/// system in.
	///
	/// `inout` is the main chip's statement as the channel carries it, which is what
	/// [`WordIPVerifierChannel::observe_words`] returns. The chips have no statement: their inout
	/// values are committed.
	///
	/// # Errors
	///
	/// Returns an error if any sub-proof fails.
	pub fn verify<Channel>(
		&self,
		inout: &[Channel::Word],
		channel: &mut Channel,
	) -> Result<(), Error>
	where
		Channel: IOPVerifierChannel<B128> + WordIPVerifierChannel<B128>,
		Channel::Elem: FieldOps<Scalar = B128> + From<B128>,
	{
		// Each sub-proof leaves a wiring claim against its own constraint system, discharged here
		// as the sub-proof it belongs to finishes.
		self.main.verify(inout, channel)?.check(channel)?;
		for chip in &self.chips {
			chip.verify_chip(channel)?.check(channel)?;
		}
		Ok(())
	}
}

/// Verifies a proof of a composite M4 system.
///
/// One-time setup fixes the sub-verifiers and the FRI parameters; a later verification checks one
/// proof against them. The prover is built from this verifier, so both sides share one set of FRI
/// parameters.
///
/// See [`IOPVerifierM4`] for what a composite proof does and does not establish.
///
/// `H` is the hash suite the Merkle commitments and the transcript channel use. One composite
/// proof commits every sub-system on one channel, so one suite covers the whole system.
pub struct VerifierM4<H: HashSuite> {
	/// The IOP verifier, holding the sub-verifier for every sub-system.
	iop_verifier: IOPVerifierM4,
	/// The precomputed BaseFold verifier, holding the FRI parameters.
	iop_compiler: BaseFoldVerifierCompiler<B128>,
	/// The verifier creates its Merkle transcript channels with the hash suite `H`.
	_hash_marker: PhantomData<H>,
}

impl<H> VerifierM4<H>
where
	H: HashSuite,
	Output<H::LeafHash>: DeserializeBytes,
{
	/// Builds the verifier for a composite system at the given code rate.
	///
	/// # Arguments
	///
	/// - `cs`: the composite constraint system.
	/// - `log_inv_rate`: base-2 logarithm of the inverse Reed-Solomon rate.
	///
	/// # Errors
	///
	/// Returns an error if the constraint system is invalid.
	pub fn setup(cs: &ConstraintSystemM4, log_inv_rate: usize) -> Result<Self, Error> {
		cs.validate()?;

		let iop_verifier = IOPVerifierM4::new(cs);

		// One channel commits every sub-system's oracles into one codeword, and sizing that
		// codeword is the compiler's job: it batches the specs over all of them. So nothing here
		// needs to know which sub-system commits the longest oracle.
		let oracle_specs = iop_verifier.oracle_specs();

		// The query count is fixed by the rate and the soundness target.
		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, log_inv_rate);

		let iop_compiler = BaseFoldVerifierCompiler::new(
			&Scheme::<H>::new(),
			oracle_specs,
			log_inv_rate,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		Ok(Self {
			iop_verifier,
			iop_compiler,
			_hash_marker: PhantomData,
		})
	}

	/// Returns a reference to the IOP verifier.
	///
	/// The prover clones this to build its matching `IOPProverM4`.
	pub const fn iop_verifier(&self) -> &IOPVerifierM4 {
		&self.iop_verifier
	}

	/// The precomputed BaseFold verifier compiler.
	///
	/// The prover reuses it so both sides share one set of FRI parameters.
	pub const fn iop_compiler(&self) -> &BaseFoldVerifierCompiler<B128> {
		&self.iop_compiler
	}

	/// Verifies one composite proof against the main chip's statement.
	///
	/// Creates the IOP channel from the transcript, observes the statement, delegates to
	/// [`IOPVerifierM4::verify`], then finishes the channel with the combined FRI opening.
	///
	/// The observation is this wrapper's step because it is the one place the protocol handles
	/// concrete words; what it returns is what the IOP verifies against.
	///
	/// # Errors
	///
	/// Returns an error if any sub-proof or the combined opening fails.
	pub fn verify<Challenger_>(
		&self,
		inout: &[Word],
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error>
	where
		Challenger_: Challenger,
	{
		let mut channel = self
			.iop_compiler
			.create_channel_from_transcript::<H, Challenger_, _>(transcript);
		let inout = channel.observe_words(inout);
		self.iop_verifier.verify(&inout, &mut channel)?;
		channel.finish()?;

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_frontend::{CircuitBuilder, CircuitM4, Wire, hints::Hint};
	use binius_hash::StdHashSuite;

	use super::*;

	/// The number of rounds the chip's mixing loop runs, which is what makes it much larger than
	/// the main chip that calls it.
	const N_ROUNDS: u32 = 64;

	/// Mixes two words into one, in the same rounds the chip's gates run.
	fn mix(a: u64, b: u64) -> u64 {
		(0..N_ROUNDS).fold(a, |acc, i| acc.rotate_left(1) ^ (b >> i))
	}

	/// The mixed word, which main takes on trust and the chip call constrains.
	struct Mix;

	impl Hint for Mix {
		const NAME: &'static str = "mix";

		fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
			(2, 1)
		}

		fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
			outputs[0] = Word(mix(inputs[0].as_u64(), inputs[1].as_u64()));
		}
	}

	/// A chip mixing two inout words into a third, promoted one.
	fn mix_chip() -> CircuitM4 {
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		let mut acc = a;
		for i in 0..N_ROUNDS {
			acc = builder.bxor(builder.rotl(acc, 1), builder.shr(b, i));
		}
		builder.mark_inout(acc);
		builder.build_m4()
	}

	/// A composite system whose main chip is far smaller than the chip it calls.
	///
	/// Main computes nothing: it takes each mixed word from a hint and calls the chip to constrain
	/// it. That asymmetry is the point — the two sub-systems commit oracles of different sizes, so
	/// a setup that sized the code off either one alone would be visibly wrong.
	fn composite_system(n_calls: usize) -> ConstraintSystemM4 {
		let builder = CircuitBuilder::new();
		let chip = builder.add_chip(mix_chip());

		let inputs: Vec<Wire> = (0..n_calls).map(|_| builder.add_inout()).collect();
		let mut acc = builder.add_constant_64(1);
		for &word in &inputs {
			let next = builder.call_hint(Mix, &[], &[acc, word])[0];
			builder.call_chip(chip, &[acc, word, next]);
			acc = next;
		}
		builder.mark_inout(acc);

		builder.build_m4().to_constraint_system()
	}

	#[test]
	fn setup_covers_every_sub_system() {
		let cs = composite_system(4);
		let verifier = VerifierM4::<StdHashSuite>::setup(&cs, 1).unwrap();
		let iop_verifier = verifier.iop_verifier();

		// One sub-verifier per chip, alongside main's.
		assert_eq!(iop_verifier.chips().len(), cs.chips.len());

		// Four calls reach the one chip, so its batch holds four instances.
		assert_eq!(iop_verifier.chips()[0].layout().log_instances, 2);

		// The composite list is main's specs followed by each chip's, so nothing is dropped or
		// duplicated.
		let expected = iop_verifier.main().oracle_specs(false).len()
			+ iop_verifier
				.chips()
				.iter()
				.map(|chip| chip.oracle_specs().len())
				.sum::<usize>();
		assert_eq!(iop_verifier.oracle_specs().len(), expected);
	}

	#[test]
	fn the_code_holds_every_sub_system() {
		let log_inv_rate = 1;
		let cs = composite_system(4);
		let verifier = VerifierM4::<StdHashSuite>::setup(&cs, log_inv_rate).unwrap();

		// The codeword encodes every oracle, so it is at least the encoding of the longest one. A
		// setup that had handed the compiler one sub-system's specs would fall short of this.
		let specs = verifier.iop_verifier().oracle_specs();
		let log_max_msg_len = specs.iter().map(|spec| spec.log_msg_len).max().unwrap();
		assert!(verifier.iop_compiler().fri_params().log_len() >= log_max_msg_len + log_inv_rate);

		// The fixture is only worth having if its sub-systems really do differ in size; otherwise
		// the assertion above would hold however the code was sized.
		let main_msg_len = verifier.iop_verifier().main().oracle_specs(false)[0].log_msg_len;
		let chip_msg_len = verifier.iop_verifier().chips()[0].oracle_specs()[0].log_msg_len;
		assert_ne!(main_msg_len, chip_msg_len);
	}

	#[test]
	fn setup_rejects_an_invalid_system() {
		let mut cs = composite_system(4);

		// Mutation: drop the chip main calls, leaving every call naming a chip that is not there.
		cs.chips.clear();

		assert!(VerifierM4::<StdHashSuite>::setup(&cs, 1).is_err());
	}
}
