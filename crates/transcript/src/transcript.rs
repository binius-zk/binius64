// Copyright 2024-2025 Irreducible Inc.

use std::{fs::File, io::Write, iter::repeat_with, slice};

use binius_field::Field;
use binius_utils::{DeserializeBytes, SerializeBytes};
use bytes::{Buf, BufMut, Bytes, BytesMut};

use super::{
	error::Error,
	fiat_shamir::{Challenger, FiatShamirBuf},
};
use crate::fiat_shamir::{CanSample, CanSampleBits, sample_bits_reader};

/// Largest proof-of-work difficulty the transcript accepts, in bits.
///
/// The sampler masks what it returns to 32 bits.
/// A larger difficulty could not be expressed, and would silently weaken to this one.
/// A 32-bit grind already costs about four billion hashes, so the cap never binds in practice.
pub const MAX_GRINDING_BITS: usize = 32;

/// Configuration options for transcript behavior
#[derive(Debug, Clone, Copy)]
pub struct Options {
	/// Whether to enable debug assertions
	pub debug_assertions: bool,
}

impl Default for Options {
	fn default() -> Self {
		Self {
			debug_assertions: cfg!(debug_assertions),
		}
	}
}

/// Verifier transcript over some Challenger that reads from the internal tape and `CanSample<F:
/// Field>`
///
/// You must manually call the destructor with `finalize()` to check anything that's written is
/// fully read out
#[derive(Debug, Clone)]
pub struct VerifierTranscript<Challenger> {
	combined: FiatShamirBuf<Bytes, Challenger>,
	options: Options,
}

impl<Challenger_: Challenger> VerifierTranscript<Challenger_> {
	pub fn new(challenger: Challenger_, vec: Vec<u8>) -> Self {
		Self::with_opts(challenger, vec, Options::default())
	}

	pub fn with_opts(challenger: Challenger_, vec: Vec<u8>, options: Options) -> Self {
		Self {
			combined: FiatShamirBuf {
				buffer: Bytes::from(vec),
				challenger,
			},
			options,
		}
	}

	pub fn finalize(self) -> Result<(), Error> {
		if self.combined.buffer.has_remaining() {
			return Err(Error::TranscriptNotEmpty {
				remaining: self.combined.buffer.remaining(),
			});
		}
		Ok(())
	}

	/// Returns a writable buffer that only observes the data written, without reading it from the
	/// proof tape.
	///
	/// This method should be used to observe the input statement.
	pub fn observe<'a, 'b>(&'a mut self) -> TranscriptWriter<'b, impl BufMut + 'b>
	where
		'a: 'b,
	{
		TranscriptWriter {
			buffer: self.combined.challenger.observer(),
			options: self.options,
		}
	}

	/// Returns a readable buffer that only reads the data from the proof tape, without observing
	/// it.
	///
	/// This method should only be used to read advice that was previously written to the transcript
	/// as an observed message.
	pub fn decommitment(&mut self) -> TranscriptReader<'_, impl Buf + '_> {
		TranscriptReader {
			buffer: &mut self.combined.buffer,
			options: self.options,
		}
	}

	/// Returns a readable buffer that observes the data read.
	///
	/// This method should be used by default to read verifier messages in an interactive protocol.
	pub fn message<'a, 'b>(&'a mut self) -> TranscriptReader<'b, impl Buf>
	where
		'a: 'b,
	{
		TranscriptReader {
			buffer: &mut self.combined,
			options: self.options,
		}
	}

	/// Checks the proof of work that [`ProverTranscript::grind`] wrote, returning its nonce.
	///
	/// Reads the nonce as a message, so the challenger observes the same bytes the prover fed it.
	/// Then samples `bits` and requires every one of them to be zero.
	/// Both transcripts consume the same challenger state either way.
	/// So a caller may keep going after an error without the two sides drifting apart.
	///
	/// ## Errors
	///
	/// Returns [`Error::InsufficientWork`] when the sampled bits are not all zero.
	/// Returns a deserialization error when the tape holds no nonce.
	///
	/// ## Preconditions
	///
	/// * `bits` must be at most 32, which is the width the sampler masks to.
	pub fn verify_grind(&mut self, bits: usize) -> Result<u64, Error> {
		assert!(
			bits <= MAX_GRINDING_BITS,
			"precondition: bits must be at most {MAX_GRINDING_BITS}"
		);

		let nonce = self.message().read::<u64>()?;
		let sampled = CanSampleBits::<u32>::sample_bits(self, bits);
		if sampled != 0 {
			return Err(Error::InsufficientWork { bits, sampled });
		}
		Ok(nonce)
	}
}

// Useful warnings to see if we are neglecting to read any advice or transcript entirely
impl<Challenger> Drop for VerifierTranscript<Challenger> {
	fn drop(&mut self) {
		if self.combined.buffer.has_remaining() {
			tracing::warn!(
				"Transcript reader is not fully read out: {:?} bytes left",
				self.combined.buffer.remaining()
			);
		}
	}
}

impl<F, Challenger_> CanSample<F> for VerifierTranscript<Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	fn sample(&mut self) -> F {
		DeserializeBytes::deserialize(self.combined.challenger.sampler())
			.expect("challenger has infinite buffer")
	}
}

impl<Challenger_> CanSampleBits<u32> for VerifierTranscript<Challenger_>
where
	Challenger_: Challenger,
{
	fn sample_bits(&mut self, bits: usize) -> u32 {
		sample_bits_reader(self.combined.challenger.sampler(), bits)
	}
}

pub struct TranscriptReader<'a, B: Buf> {
	buffer: &'a mut B,
	options: Options,
}

impl<B: Buf> TranscriptReader<'_, B> {
	pub const fn buffer(&mut self) -> &mut B {
		self.buffer
	}

	pub fn read<T: DeserializeBytes>(&mut self) -> Result<T, Error> {
		T::deserialize(self.buffer()).map_err(Into::into)
	}

	pub fn read_vec<T: DeserializeBytes>(&mut self, n: usize) -> Result<Vec<T>, Error> {
		let mut buffer = self.buffer();
		repeat_with(move || T::deserialize(&mut buffer).map_err(Into::into))
			.take(n)
			.collect()
	}

	pub fn read_bytes(&mut self, buf: &mut [u8]) -> Result<(), Error> {
		let buffer = self.buffer();
		if buffer.remaining() < buf.len() {
			return Err(Error::NotEnoughBytes);
		}
		buffer.copy_to_slice(buf);
		Ok(())
	}

	pub fn read_scalar<F: Field>(&mut self) -> Result<F, Error> {
		let mut out = F::default();
		self.read_scalar_slice_into(slice::from_mut(&mut out))?;
		Ok(out)
	}

	pub fn read_scalar_slice_into<F: Field>(&mut self, buf: &mut [F]) -> Result<(), Error> {
		let mut buffer = self.buffer();
		for elem in buf {
			*elem = DeserializeBytes::deserialize(&mut buffer)?;
		}
		Ok(())
	}

	pub fn read_scalar_slice<F: Field>(&mut self, len: usize) -> Result<Vec<F>, Error> {
		let mut elems = vec![F::default(); len];
		self.read_scalar_slice_into(&mut elems)?;
		Ok(elems)
	}

	pub fn read_debug(&mut self, msg: &str) {
		if self.options.debug_assertions {
			let msg_bytes = msg.as_bytes();
			let mut buffer = vec![0; msg_bytes.len()];
			assert!(self.read_bytes(&mut buffer).is_ok());
			assert_eq!(msg_bytes, buffer);
		}
	}
}

/// Prover transcript over some Challenger that writes to the internal tape and `CanSample<F:
/// Field>`
///
/// A Transcript is an abstraction over Fiat-Shamir so the prover and verifier can send and receive
/// data.
#[derive(Debug, Clone)]
pub struct ProverTranscript<Challenger> {
	combined: FiatShamirBuf<BytesMut, Challenger>,
	options: Options,
}

impl<Challenger_: Challenger> ProverTranscript<Challenger_> {
	/// Creates a new prover transcript.
	///
	/// By default debug assertions are set to the feature flag `debug_assertions`.
	pub fn new(challenger: Challenger_) -> Self {
		Self::with_opts(challenger, Options::default())
	}

	pub fn with_opts(challenger: Challenger_, options: Options) -> Self {
		Self {
			combined: FiatShamirBuf {
				buffer: BytesMut::default(),
				challenger,
			},
			options,
		}
	}

	pub fn finalize(self) -> Vec<u8> {
		let transcript = self.combined.buffer.to_vec();

		// Emit proof size as a tracing event
		let proof_size_bytes = transcript.len();
		tracing::event!(
			name: "proof_size",
			tracing::Level::INFO,
			category = "metrics",
			proof_size_bytes = proof_size_bytes,
		);

		// Dumps the transcript to the path set in the BINIUS_DUMP_PROOF env variable.
		if let Ok(path) = std::env::var("BINIUS_DUMP_PROOF") {
			let path = if cfg!(test) {
				// Because tests may run simultaneously, each test includes its name in the file
				// name to avoid collisions.
				let current_thread = std::thread::current();
				let test_name = current_thread.name().unwrap_or("unknown");
				// Adjust "./" to "../../" to ensure files are saved in the project root rather than
				// the package root.
				let rebased = path.strip_prefix("./").map(|s| format!("../../{s}"));
				let path = rebased.unwrap_or(path);
				std::fs::create_dir_all(&path)
					.unwrap_or_else(|_| panic!("Failed to create directories for path: {path}",));
				format!("{path}/{test_name}.bin")
			} else {
				path
			};

			let mut file = File::create(&path)
				.unwrap_or_else(|_| panic!("Failed to create proof dump file: {path}"));
			file.write_all(&transcript)
				.expect("Failed to write proof to dump file");
		}
		transcript
	}

	/// Returns a writeable buffer that only observes the data written, without writing it to the
	/// proof tape.
	///
	/// This method should be used to observe the input statement.
	pub fn observe<'a, 'b>(&'a mut self) -> TranscriptWriter<'b, impl BufMut + 'b>
	where
		'a: 'b,
	{
		TranscriptWriter {
			buffer: self.combined.challenger.observer(),
			options: self.options,
		}
	}

	/// Returns a writeable buffer that only writes the data to the proof tape, without observing
	/// it.
	///
	/// This method should only be used to write openings of commitments that were already written
	/// to the transcript as an observed message. For example, in the FRI protocol, the prover sends
	/// a Merkle tree root as a commitment, and later sends leaf openings. The leaf openings should
	/// be written using [`Self::decommitment`] because they are verified with respect to the
	/// previously sent Merkle root.
	pub fn decommitment(&mut self) -> TranscriptWriter<'_, impl BufMut> {
		TranscriptWriter {
			buffer: &mut self.combined.buffer,
			options: self.options,
		}
	}

	/// Returns a writeable buffer that observes the data written and writes it to the proof tape.
	///
	/// This method should be used by default to write prover messages in an interactive protocol.
	pub fn message<'a, 'b>(&'a mut self) -> TranscriptWriter<'b, impl BufMut>
	where
		'a: 'b,
	{
		TranscriptWriter {
			buffer: &mut self.combined,
			options: self.options,
		}
	}

	/// Grinds a proof of work into the transcript, returning the nonce it found.
	///
	/// Searches for a nonce whose observation drives the next `bits` sampled bits to zero.
	/// The nonce goes out as a message, and those bits are then sampled so the challenger advances.
	/// [`VerifierTranscript::verify_grind`] is the matching check.
	///
	/// A prover that wants to re-roll whatever challenge comes next must redo this search first.
	/// Every term derived from that challenge therefore gains `bits` of soundness.
	/// That is how a protocol buys back a term no query count can touch.
	/// `binius_iop::soundness::Grinding` is where that credit enters a security budget.
	///
	/// The search is the obvious one: try nonces in order until one lands.
	/// A trial is one challenger observation plus one sample, and lands with probability `2^-bits`.
	/// The expected cost is therefore `2^bits` trials.
	/// The search is serial, so a caller pays that in wall clock.
	///
	/// ## Preconditions
	///
	/// * `bits` must be at most 32, which is the width the sampler masks to.
	pub fn grind(&mut self, bits: usize) -> u64
	where
		Challenger_: Clone,
	{
		assert!(
			bits <= MAX_GRINDING_BITS,
			"precondition: bits must be at most {MAX_GRINDING_BITS}"
		);

		// Trial on a clone so a failed nonce leaves the real challenger untouched. Observing the
		// nonce's little-endian bytes is what `write::<u64>` feeds the challenger, so the two
		// agree.
		let mut nonce = 0u64;
		loop {
			let mut trial = self.combined.challenger.clone();
			trial.observer().put_slice(&nonce.to_le_bytes());
			if sample_bits_reader(trial.sampler(), bits) == 0 {
				break;
			}
			nonce += 1;
		}

		self.message().write(&nonce);
		let sampled = CanSampleBits::<u32>::sample_bits(self, bits);
		debug_assert_eq!(sampled, 0, "the nonce search only exits on a landing nonce");
		nonce
	}
}

impl<Challenger_: Default + Challenger> ProverTranscript<Challenger_> {
	pub fn into_verifier(self) -> VerifierTranscript<Challenger_> {
		let options = self.options;
		let transcript = self.finalize();

		VerifierTranscript::with_opts(Challenger_::default(), transcript, options)
	}
}

impl<Challenger_: Default + Challenger> Default for ProverTranscript<Challenger_> {
	fn default() -> Self {
		Self::new(Challenger_::default())
	}
}

/// Writes data to a transcript buffer, tracking proof size via tracing events.
///
/// Transcript buffers are always growable (`BytesMut` or equivalent), so serialization
/// writes are infallible in practice. The write methods use `expect` rather than returning
/// `Result` because the underlying buffers dynamically resize and cannot run out of space.
pub struct TranscriptWriter<'a, B: BufMut> {
	buffer: &'a mut B,
	options: Options,
}

impl<B: BufMut> TranscriptWriter<'_, B> {
	pub const fn buffer(&mut self) -> &mut B {
		self.buffer
	}

	/// Serializes and writes a value to the transcript buffer.
	///
	/// # Panics
	///
	/// Panics if serialization fails. Transcript buffers are growable, so this cannot fail
	/// due to insufficient space.
	pub fn write<T: SerializeBytes>(&mut self, value: &T) {
		self.proof_size_event_wrapper(move |buffer| {
			value
				.serialize(buffer)
				.expect("serialization to a growable transcript buffer is infallible");
		});
	}

	/// Serializes and writes a slice of values to the transcript buffer.
	///
	/// # Panics
	///
	/// Panics if serialization fails. Transcript buffers are growable, so this cannot fail
	/// due to insufficient space.
	pub fn write_slice<T: SerializeBytes>(&mut self, values: &[T]) {
		self.proof_size_event_wrapper(move |buffer| {
			T::serialize_slice(values, buffer)
				.expect("serialization to a growable transcript buffer is infallible");
		});
	}

	pub fn write_bytes(&mut self, data: &[u8]) {
		self.proof_size_event_wrapper(|buffer| {
			buffer.put_slice(data);
		});
	}

	pub fn write_scalar<F: Field>(&mut self, f: F) {
		self.write_scalar_slice(slice::from_ref(&f));
	}

	/// Serializes and writes an iterator of field elements to the transcript buffer.
	///
	/// # Panics
	///
	/// Panics if serialization fails. Transcript buffers are growable, so this cannot fail
	/// due to insufficient space.
	pub fn write_scalar_iter<F: Field>(&mut self, it: impl IntoIterator<Item = F>) {
		self.proof_size_event_wrapper(move |buffer| {
			for elem in it {
				SerializeBytes::serialize(&elem, &mut *buffer)
					.expect("serialization to a growable transcript buffer is infallible");
			}
		});
	}

	pub fn write_scalar_slice<F: Field>(&mut self, elems: &[F]) {
		self.proof_size_event_wrapper(move |buffer| {
			F::serialize_slice(elems, buffer)
				.expect("serialization to a growable transcript buffer is infallible");
		});
	}

	pub fn write_debug(&mut self, msg: &str) {
		if self.options.debug_assertions {
			self.write_bytes(msg.as_bytes());
		}
	}

	fn proof_size_event_wrapper<F: FnOnce(&mut B)>(&mut self, f: F) {
		let buffer = self.buffer();
		let start_bytes = buffer.remaining_mut();
		f(buffer);
		let end_bytes = buffer.remaining_mut();
		tracing::event!(
			name: "incremental_proof_size",
			tracing::Level::TRACE,
			counter=true,
			incremental=true,
			value=start_bytes - end_bytes,
		);
	}
}

impl<F, Challenger_> CanSample<F> for ProverTranscript<Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	fn sample(&mut self) -> F {
		DeserializeBytes::deserialize(self.combined.challenger.sampler())
			.expect("challenger has infinite buffer")
	}
}

impl<Challenger_> CanSampleBits<u32> for ProverTranscript<Challenger_>
where
	Challenger_: Challenger,
{
	fn sample_bits(&mut self, bits: usize) -> u32 {
		sample_bits_reader(self.combined.challenger.sampler(), bits)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b as B128;
	use sha2::Sha256;

	use super::*;
	use crate::fiat_shamir::{CanSample, HasherChallenger};

	#[test]
	fn test_transcript_interactions() {
		let mut prover_transcript = ProverTranscript::new(HasherChallenger::<Sha256>::default());

		// Write messages using message()
		prover_transcript
			.message()
			.write_scalar(B128::new(0x11111111222222223333333344444444));
		prover_transcript
			.message()
			.write_scalar(B128::new(0xAAAAAAAABBBBBBBBCCCCCCCCDDDDDDDD));

		// Write decommitment (not observed)
		prover_transcript
			.decommitment()
			.write_scalar(B128::new(0x5555555566666666777777778888888));

		// Write observed data
		prover_transcript
			.observe()
			.write_scalar(B128::new(0xFFFFFFFFEEEEEEEEDDDDDDDDCCCCCCCC));

		// Sample a challenge
		let sampled_challenge: B128 = prover_transcript.sample();

		// Convert to verifier transcript
		let mut verifier_transcript = prover_transcript.into_verifier();

		// Read messages
		let msg1: B128 = verifier_transcript.message().read_scalar().unwrap();
		let msg2: B128 = verifier_transcript.message().read_scalar().unwrap();
		assert_eq!(msg1, B128::new(0x11111111222222223333333344444444));
		assert_eq!(msg2, B128::new(0xAAAAAAAABBBBBBBBCCCCCCCCDDDDDDDD));

		// Read decommitment
		let decommit: B128 = verifier_transcript.decommitment().read_scalar().unwrap();
		assert_eq!(decommit, B128::new(0x5555555566666666777777778888888));

		// Observe the same data (doesn't read from tape)
		verifier_transcript
			.observe()
			.write_scalar(B128::new(0xFFFFFFFFEEEEEEEEDDDDDDDDCCCCCCCC));

		// Sample should produce the same challenge
		let verifier_challenge: B128 = verifier_transcript.sample();
		assert_eq!(verifier_challenge, sampled_challenge);

		// Check that transcript is empty
		verifier_transcript.finalize().unwrap();
	}

	#[test]
	fn test_transcript_debug() {
		let options = Options {
			debug_assertions: true,
		};
		let mut transcript =
			ProverTranscript::with_opts(HasherChallenger::<Sha256>::default(), options);

		transcript.message().write_debug("test_transcript_debug");
		transcript
			.into_verifier()
			.message()
			.read_debug("test_transcript_debug");
	}

	#[test]
	#[should_panic]
	fn test_transcript_debug_fail() {
		let options = Options {
			debug_assertions: true,
		};
		let mut transcript =
			ProverTranscript::with_opts(HasherChallenger::<Sha256>::default(), options);

		transcript.message().write_debug("test_transcript_debug");
		transcript
			.into_verifier()
			.message()
			.read_debug("test_transcript_debug_should_fail");
	}
	#[test]
	fn grinding_round_trips_and_keeps_both_challengers_in_step() {
		const BITS: usize = 12;

		let mut prover = ProverTranscript::new(HasherChallenger::<Sha256>::default());
		prover.message().write_scalar(B128::new(7));
		let nonce = prover.grind(BITS);
		// A difficulty this high is not met by the first nonce tried, so the search really ran.
		assert!(nonce > 0);
		let prover_challenge: B128 = prover.sample();

		let mut verifier = prover.into_verifier();
		let echoed: B128 = verifier.message().read_scalar().unwrap();
		assert_eq!(echoed, B128::new(7));
		assert_eq!(verifier.verify_grind(BITS).unwrap(), nonce);

		// The nonce and its sample advance both challengers identically, so what follows agrees.
		let verifier_challenge: B128 = verifier.sample();
		assert_eq!(verifier_challenge, prover_challenge);
		verifier.finalize().unwrap();
	}

	#[test]
	fn zero_difficulty_is_met_by_the_first_nonce() {
		let mut prover = ProverTranscript::new(HasherChallenger::<Sha256>::default());
		// Every nonce satisfies an empty condition, so the search stops immediately.
		assert_eq!(prover.grind(0), 0);

		// It still consumes a sample, which is what keeps the verifier in step.
		let mut verifier = prover.into_verifier();
		assert_eq!(verifier.verify_grind(0).unwrap(), 0);
		verifier.finalize().unwrap();
	}

	#[test]
	fn a_tampered_nonce_is_rejected() {
		const BITS: usize = 12;

		let mut prover = ProverTranscript::new(HasherChallenger::<Sha256>::default());
		prover.grind(BITS);
		let mut tape = prover.finalize();

		// Flip a bit of the written nonce. It is the first thing on the tape, and the challenger
		// observes it, so the sampled bits move off zero.
		tape[0] ^= 1;

		let mut verifier =
			VerifierTranscript::new(HasherChallenger::<Sha256>::default(), tape.clone());
		let err = verifier.verify_grind(BITS).unwrap_err();
		let Error::InsufficientWork { bits, sampled } = err else {
			panic!("expected InsufficientWork, got {err:?}");
		};
		assert_eq!(bits, BITS);
		assert_ne!(sampled, 0);
		// The masked sample never exceeds the difficulty it was masked to.
		assert!(sampled < 1 << BITS);
		verifier.finalize().unwrap();
	}

	#[test]
	fn an_empty_tape_has_no_nonce_to_read() {
		let mut verifier = VerifierTranscript::new(HasherChallenger::<Sha256>::default(), vec![]);
		let err = verifier.verify_grind(8).unwrap_err();
		let Error::Serialization(inner) = err else {
			panic!("expected Serialization, got {err:?}");
		};
		assert!(matches!(inner, binius_utils::SerializationError::NotEnoughBytes));
		verifier.finalize().unwrap();
	}

	#[test]
	#[should_panic(expected = "bits must be at most 32")]
	fn a_difficulty_past_the_sampler_width_is_rejected() {
		ProverTranscript::new(HasherChallenger::<Sha256>::default()).grind(MAX_GRINDING_BITS + 1);
	}
}
