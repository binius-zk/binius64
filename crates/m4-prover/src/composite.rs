// Copyright 2026 The Binius Developers

//! Proving a whole [`ConstraintSystemM4`](binius_core::constraint_system::m4::ConstraintSystemM4):
//! the main chip and every numbered chip.

use std::{iter, marker::PhantomData};

use binius_compute::{Allocator, BufferPool};
use binius_core::{
	Word,
	constraint_system::{InoutSegment, m4::WitnessM4},
};
use binius_field::PackedField;
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop_prover::{basefold::compiler::BaseFoldProverCompiler, channel::IOPProverChannel};
use binius_ip_prover::channel::WordIPProverChannel;
use binius_m4_verifier::{IOPVerifierM4, VerifierM4};
use binius_math::ntt::{NeighborsLastMultiThread, domain_context::GaoMateerPreExpanded};
use binius_prover::{Error, protocols::shift::build_key_collection};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::SerializeBytes;
use binius_verifier::config::B128;
use digest::Output;

use crate::prove::{IOPProver, ProverNtt};

/// IOP prover for a composite M4 system.
///
/// It mirrors [`IOPVerifierM4`] sub-system for sub-system: the Binius64 prover for the main chip,
/// which runs once, and this crate's data-parallel [`IOPProver`] for each numbered chip's batch of
/// instances. The sub-proofs share one channel, and so one Fiat-Shamir transcript and one combined
/// FRI opening.
///
/// ## What a composite proof does not prove
///
/// Chip calls are unconstrained; see [`IOPVerifierM4`] for what that leaves out.
pub struct IOPProverM4 {
	/// The prover for the main chip, which runs once.
	main: binius_prover::IOPProver,
	/// The prover for each numbered chip's batch of instances, indexed by chip ID.
	chips: Vec<IOPProver>,
}

impl IOPProverM4 {
	/// Builds the sub-provers matching an IOP verifier, deriving each one's shift keys.
	///
	/// The main chip's keys are built over the public segment, since its inout values are the
	/// statement; every chip's are built over the hidden segment, since its inout values are
	/// committed with the private ones. The two are not interchangeable.
	pub fn new(iop_verifier: &IOPVerifierM4) -> Self {
		let main_cs = iop_verifier.main().constraint_system();
		let main = binius_prover::IOPProver::new(
			iop_verifier.main().clone(),
			build_key_collection(main_cs, InoutSegment::Public),
		);

		let chips = iop_verifier
			.chips()
			.iter()
			.map(|chip| {
				let key_collection =
					build_key_collection(chip.constraint_system(), InoutSegment::Hidden);
				IOPProver::new(chip.clone(), key_collection)
			})
			.collect();

		Self { main, chips }
	}

	/// The prover for the main chip.
	pub const fn main(&self) -> &binius_prover::IOPProver {
		&self.main
	}

	/// The provers for the numbered chips, indexed by chip ID.
	pub fn chips(&self) -> &[IOPProver] {
		&self.chips
	}

	/// Proves that a witness satisfies the composite system, using an IOP channel.
	///
	/// The sub-proofs run on one channel in the order
	/// [`IOPVerifierM4::verify`](binius_m4_verifier::IOPVerifierM4::verify) reads them — the main
	/// chip, then the chips in ID order. One channel is one Fiat-Shamir transcript, so the two
	/// orders must agree exactly.
	///
	/// The main chip's prover observes its own inout values from the witness, so the statement is
	/// not a separate argument here as it is on the verifier.
	///
	/// ## Preconditions
	/// * the witness holds one table per chip of the system this prover was built for
	///
	/// # Errors
	///
	/// Returns an error if the main chip's witness does not fit its committed shape.
	pub fn prove<P, Channel, A>(
		&self,
		witness: &WitnessM4,
		channel: &mut Channel,
		alloc: &A,
	) -> Result<(), Error>
	where
		P: PackedField<Scalar = B128>,
		Channel: IOPProverChannel<P, A> + WordIPProverChannel<B128, Word = Word>,
		A: Allocator,
	{
		// A short witness would leave the trailing chips' oracles unsent while the verifier still
		// reads them, so hold the two to the same length rather than letting `zip` truncate.
		assert_eq!(
			witness.tables.len(),
			self.chips.len(),
			"the witness must hold one table per chip of the system this prover was built for"
		);

		self.main.prove::<_, P, _>(&witness.main, channel, alloc)?;
		for (prover, table) in iter::zip(&self.chips, &witness.tables) {
			prover.prove_chip::<P, _, _>(table, channel, alloc);
		}
		Ok(())
	}
}

/// Proves a composite M4 system.
///
/// Built from a [`VerifierM4`], so both sides share one set of FRI parameters.
///
/// See [`IOPVerifierM4`] for what a composite proof does and does not establish.
pub struct ProverM4<P, H>
where
	P: PackedField<Scalar = B128>,
	H: HashSuite,
{
	iop_prover: IOPProverM4,
	/// The precomputed BaseFold prover, holding the NTT and the FRI parameters.
	basefold_compiler: BaseFoldProverCompiler<P, ProverNtt>,
	/// The pool that recycles this prover's working buffers. It lives for the prover's lifetime,
	/// so blocks freed by one proof are reused by the next.
	pool: BufferPool,
	/// The prover creates its Merkle transcript channels with the hash suite `H`, matching the
	/// verifier it was built from.
	_hash_marker: PhantomData<H>,
}

impl<P, H> ProverM4<P, H>
where
	P: PackedField<Scalar = B128>,
	H: HashSuite,
	Output<H::LeafHash>: SerializeBytes,
{
	/// Builds the prover from a verifier, inheriting its sub-systems and FRI parameters.
	///
	/// The prover encodes the codeword with the multithreaded NTT, spread across the cores.
	/// Reusing the verifier's compiler keeps both sides on one set of FRI parameters.
	pub fn setup(verifier: &VerifierM4<H>) -> Self {
		// Reuse the verifier's evaluation domain so both sides agree on the code: its compiler
		// fixed that domain as the Gao-Mateer basis of this dimension.
		let domain_context =
			GaoMateerPreExpanded::generate(verifier.iop_compiler().max_log_domain_size());

		// Spread the NTT across the available cores.
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		// Inherit the verifier's oracle specs and FRI parameters verbatim.
		let basefold_compiler =
			BaseFoldProverCompiler::from_verifier_compiler(verifier.iop_compiler(), ntt);

		Self {
			iop_prover: IOPProverM4::new(verifier.iop_verifier()),
			basefold_compiler,
			pool: BufferPool::new(),
			_hash_marker: PhantomData,
		}
	}

	/// Returns a reference to the IOP prover.
	pub const fn iop_prover(&self) -> &IOPProverM4 {
		&self.iop_prover
	}

	/// Proves that a witness satisfies the composite system.
	///
	/// Creates the IOP channel from the transcript, delegates to [`IOPProverM4::prove`], then
	/// finishes the channel with the combined FRI opening.
	///
	/// # Errors
	///
	/// Returns an error if the main chip's witness does not fit its committed shape.
	pub fn prove<Challenger_>(
		&self,
		witness: &WitnessM4,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error>
	where
		Challenger_: Challenger,
	{
		// Working buffers for this proof are drawn from the prover's pool, recycling blocks freed
		// by earlier proofs. The channel commits its Merkle trees out of the same pool.
		let alloc = &self.pool;
		// A composite proof is transparent: every oracle is committed without a mask, so the
		// channel draws no randomness.
		let mut channel = self
			.basefold_compiler
			.create_channel_without_zk_from_transcript::<H, Challenger_, _, _>(transcript, alloc);
		self.iop_prover
			.prove::<P, _, _>(witness, &mut channel, &alloc)?;

		let _scope = tracing::debug_span!("PCS opening").entered();
		channel.finish(&alloc);

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_core::{ValueTable, constraint_system::m4::ConstraintSystemM4};
	use binius_field::PackedBinaryGhash1x128b;
	use binius_hash::StdHashSuite;
	use binius_transcript::VerifierTranscript;
	use binius_verifier::config::StdChallenger;

	use super::*;
	use crate::test_utils::{crc64_chip_circuit, crc64_iso_reference};

	type P = PackedBinaryGhash1x128b;

	/// The rate every test proves at.
	const LOG_INV_RATE: usize = 1;

	/// The message the fixture absorbs, one chip call per word.
	const MESSAGE: [u64; 4] = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 0, u64::MAX];

	/// Builds the composite fixture and a witness over [`MESSAGE`].
	fn fixture() -> (ConstraintSystemM4, WitnessM4) {
		let c = crc64_chip_circuit(MESSAGE.len());
		let cs = c.circuit.to_constraint_system();
		let witness = c
			.circuit
			.generate_witness(|filler| {
				for (&wire, &word) in iter::zip(&c.input, &MESSAGE) {
					filler[wire] = Word(word);
				}
			})
			.unwrap();

		// The witness computes the right thing and satisfies the system, so a proof of it failing
		// to verify below is the protocol's fault and not the fixture's.
		let output = c.circuit.main.circuit.witness_index(c.output);
		assert_eq!(witness.main[output], Word(crc64_iso_reference(&MESSAGE)));
		witness.verify(&cs).unwrap();

		(cs, witness)
	}

	/// Proves `witness` against `cs` and returns the proof bytes.
	fn prove(cs: &ConstraintSystemM4, witness: &WitnessM4) -> Vec<u8> {
		let verifier = VerifierM4::setup(cs, LOG_INV_RATE).unwrap();
		let prover = ProverM4::<P, StdHashSuite>::setup(&verifier);

		let mut transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(witness, &mut transcript).unwrap();
		transcript.finalize()
	}

	#[test]
	fn a_composite_proof_verifies() {
		let (cs, witness) = fixture();
		let proof = prove(&cs, &witness);

		let verifier = VerifierM4::<StdHashSuite>::setup(&cs, LOG_INV_RATE).unwrap();
		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		verifier
			.verify(witness.main.inout(), &mut transcript)
			.expect("a faithful composite proof verifies");
		transcript.finalize().expect("no trailing proof data");
	}

	#[test]
	fn the_prover_mirrors_the_verifier_sub_system_for_sub_system() {
		let (cs, _) = fixture();
		let verifier = VerifierM4::<StdHashSuite>::setup(&cs, LOG_INV_RATE).unwrap();
		let prover = ProverM4::<P, StdHashSuite>::setup(&verifier);

		assert_eq!(prover.iop_prover().chips().len(), cs.chips.len());
	}

	#[test]
	fn a_proof_is_rejected_against_another_statement() {
		let (cs, witness) = fixture();
		let proof = prove(&cs, &witness);

		// Mutation: one inout word of the statement differs from the one proved.
		let mut inout = witness.main.inout().to_vec();
		inout[0] = Word(inout[0].as_u64() ^ 1);

		let verifier = VerifierM4::<StdHashSuite>::setup(&cs, LOG_INV_RATE).unwrap();
		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		assert!(
			verifier.verify(&inout, &mut transcript).is_err(),
			"a proof of one statement must not verify against another"
		);
	}

	#[test]
	fn a_chip_instance_that_breaks_its_constraints_is_rejected() {
		let (cs, mut witness) = fixture();

		// Mutation: flip a bit of the chip's committed table, so one instance no longer satisfies
		// the chip's constraints. Nothing about main changes, so only the chip's sub-proof can
		// catch this — which is the point of the test.
		let table = &witness.tables[0];
		let mut words = table.as_words().to_vec();
		words[0] = Word(words[0].as_u64() ^ 1);
		witness.tables[0] =
			ValueTable::from_hidden_words(table.layout().clone(), table.log_instances(), words);

		let proof = prove(&cs, &witness);

		let verifier = VerifierM4::<StdHashSuite>::setup(&cs, LOG_INV_RATE).unwrap();
		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		assert!(
			verifier
				.verify(witness.main.inout(), &mut transcript)
				.is_err(),
			"a chip instance that breaks the chip's constraints must not verify"
		);
	}
}
