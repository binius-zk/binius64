// Copyright 2026 The Binius Developers

//! The circuit that verifies a Binius64 proof, built once per inner shape.
//!
//! ```text
//!   inner shape      ->  build    ->  circuit
//!   circuit + proof  ->  witness  ->  values that satisfy it
//! ```
//!
//! A shape is everything the protocol fixes before any proof exists: the constraint system, the
//! FRI parameters, the oracle specs. No length in the protocol depends on a received value, so
//! one circuit verifies every proof of that shape.
//!
//! [`RecursiveCircuit::build`] therefore reads a [`Verifier`] and never a proof.
//! A proof is needed only to fill a witness.
//!
//! # Going deeper
//!
//! What `build` returns is an ordinary circuit, so it has an ordinary [`Verifier`].
//! Building again over that verifier is one level deeper.
//!
//! ```text
//!   Verifier(inner) -> build -> circuit_1 -> Verifier(circuit_1) -> build -> circuit_2
//! ```
//!
//! Every level pays for the level below it in full, because the wiring claim is discharged
//! in-circuit rather than deferred. Measured over a CRC-64 inner circuit, a level costs about
//! fifteen times the constraint rows of the level it verifies, so the tower diverges instead of
//! closing. Deferring that claim is what makes a fixed point possible; see `RECURSION_PLAN.md`.

use std::iter;

use binius_core::{constraint_system::ValueVec, word::Word};
use binius_frontend::{Circuit, PopulateError, Wire, WitnessFiller};
use binius_hash::StdHashSuite;
use binius_ip::channel::WordIPVerifierChannel;
use binius_transcript::VerifierTranscript;
use binius_verifier::{Pcs, Verifier, config::StdChallenger};

use crate::{Binius64BuilderChannel, Recorded, WitnessFillerChannel};

/// Why a recursive circuit could not be built, or its witness could not be filled.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// The inner trace is opened with a scheme the in-circuit gadgets do not express.
	#[error("the inner trace is opened with {pcs:?}, and only FRI is expressed in-circuit")]
	UnsupportedPcs {
		/// The scheme the inner verifier was set up with.
		pcs: Pcs,
	},

	/// The statement handed to [`RecursiveCircuit::witness`] is the wrong length.
	#[error("the inner statement holds {expected} words, and {actual} were supplied")]
	StatementLength {
		/// The inner constraint system's inout count.
		expected: usize,
		/// What the caller supplied.
		actual: usize,
	},

	/// Running the verifier over a recursion channel failed.
	#[error("verifier error: {0}")]
	Verifier(#[from] binius_verifier::Error),

	/// The oracle layer rejected the run.
	#[error("IOP channel error: {0}")]
	IOPChannel(#[from] binius_iop::channel::Error),

	/// The replayed values leave the recorded circuit unsatisfied.
	#[error("the recorded circuit is not satisfied by the replayed witness")]
	Unsatisfied(PopulateError),
}

/// A Binius64 circuit that verifies Binius64 proofs of one inner shape.
///
/// The inner statement is the circuit's public interface. Whoever checks an outer proof reads
/// which statement was verified, rather than trusting whoever filled the witness.
///
/// The shape is owned rather than borrowed, so a witness cannot be filled against a verifier
/// that differs from the one the circuit was recorded from.
pub struct RecursiveCircuit {
	/// The inner shape this circuit verifies.
	///
	/// Held because filling a witness replays the same verifier that built the circuit.
	/// A different one would visit different operations and desync the wire cursor.
	verifier: Verifier<StdHashSuite>,

	/// The compiled circuit, and the wires a witness must supply.
	recorded: Recorded,

	/// One public wire per inner statement word.
	///
	/// Each is constrained to equal the word the verifier read, so the two cannot disagree.
	statement: Vec<Wire>,
}

impl RecursiveCircuit {
	/// Records `verifier`'s run as a circuit that verifies any proof of its shape.
	///
	/// The whole inner statement is bound to public inputs. A caller wanting to expose only part
	/// of it drives [`Binius64BuilderChannel`] directly and chooses what to
	/// [`bind_public`](Binius64BuilderChannel::bind_public).
	///
	/// [`Verifier`] is [`Clone`], so a caller still needing the shape elsewhere clones it here
	/// rather than paying for a clone this constructor does not need.
	///
	/// # Errors
	///
	/// Returns [`Error::UnsupportedPcs`] when the inner trace is not opened with FRI.
	///
	/// Returns [`Error::Verifier`] when the shape itself is unsatisfiable, which a build-time
	/// constant assertion decides before any proof exists.
	pub fn build(verifier: Verifier<StdHashSuite>) -> Result<Self, Error> {
		let Some(compiler) = verifier.iop_compiler().as_basefold() else {
			return Err(Error::UnsupportedPcs {
				pcs: verifier.pcs(),
			});
		};
		let mut channel = compiler.create_channel(Binius64BuilderChannel::new());

		// Invariant: on this channel `observe_words` reads the length and never the values.
		//
		// It allocates one witness wire per word and absorbs those wires, so the statement
		// enters as circuit input rather than as a constant. That is what keeps one circuit good
		// for every proof of this shape, and it is why a placeholder is sound here.
		let n_inout = verifier.constraint_system().n_inout;
		let statement = channel.observe_words(&vec![Word::ZERO; n_inout]);

		// The verifier's own arithmetic becomes constraints, down to every assertion along the
		// way. Discharging the wiring claim is part of that, and it is the level's largest cost.
		verifier
			.iop_verifier()
			.verify(&statement, &mut channel)?
			.check_symbolic(&mut channel)
			.map_err(binius_verifier::Error::from)?;

		let mut builder_channel = channel.finish()?;
		let statement = builder_channel.bind_public(statement);

		Ok(Self {
			verifier,
			recorded: builder_channel.build(),
			statement,
		})
	}

	/// The compiled circuit.
	pub const fn circuit(&self) -> &Circuit {
		&self.recorded.circuit
	}

	/// The inner shape this circuit verifies.
	pub const fn inner(&self) -> &Verifier<StdHashSuite> {
		&self.verifier
	}

	/// The public wires carrying the inner statement, in inout order.
	pub fn statement(&self) -> &[Wire] {
		&self.statement
	}

	/// The compiled circuit, and the inventory of wires a witness supplies.
	///
	/// The inventory names the operation that allocated each wire, so it reads as a breakdown of
	/// what the proof had to carry.
	pub const fn recorded(&self) -> &Recorded {
		&self.recorded
	}

	/// Fills the witness that satisfies this circuit for one inner proof.
	///
	/// `inout` lands twice: on the public wires an outer proof exposes, and on the wires the
	/// replay fills. The binding [`build`](Self::build) emitted constrains the two equal, so a
	/// caller cannot claim one statement while verifying a proof of another.
	///
	/// # Errors
	///
	/// Returns [`Error::StatementLength`] when `inout` is not the inner statement's length.
	///
	/// Returns [`Error::Verifier`] when the transcript does not carry a well-formed proof.
	///
	/// Returns [`Error::Unsatisfied`] when the replayed values leave the circuit unsatisfied,
	/// which is where tampered proof data lands.
	pub fn witness(&self, inout: &[Word], proof: Vec<u8>) -> Result<ValueVec, Error> {
		let mut filler = self.fill(inout, proof)?;

		self.recorded
			.circuit
			.populate_wire_witness(&mut filler)
			.map_err(Error::Unsatisfied)?;

		Ok(filler.into_value_vec())
	}

	/// Replays the proof into a filler, and hands it back before the circuit checks it.
	///
	/// Ordinary use wants [`witness`](Self::witness), which checks. This is the seam for a caller
	/// that must read or perturb what the replay wrote first — a test pinning which constraint
	/// rejects a tampered proof, say.
	///
	/// What comes back is filled, not checked. A caller owes it
	/// [`populate_wire_witness`](Circuit::populate_wire_witness) before the values mean anything:
	/// that call is what derives every internal wire and decides whether the circuit holds.
	///
	/// # Errors
	///
	/// As [`witness`](Self::witness), minus [`Error::Unsatisfied`], which only the check reaches.
	pub fn fill(&self, inout: &[Word], proof: Vec<u8>) -> Result<WitnessFiller<'_>, Error> {
		if inout.len() != self.statement.len() {
			return Err(Error::StatementLength {
				expected: self.statement.len(),
				actual: inout.len(),
			});
		}

		let mut filler = self.recorded.circuit.new_witness_filler();

		// The public half of every binding is supplied by whoever checks the outer proof.
		for (&wire, &word) in iter::zip(&self.statement, inout) {
			filler[wire] = word;
		}

		self.replay(inout, proof, &mut filler)?;

		Ok(filler)
	}

	/// Runs the verifier over the real transcript, writing what it reads into the recorded wires.
	///
	/// The build and this replay reach the same operations in the same order, so one cursor pairs
	/// what the replay saw with the wire the build allocated for it.
	fn replay(
		&self,
		inout: &[Word],
		proof: Vec<u8>,
		filler: &mut WitnessFiller<'_>,
	) -> Result<(), Error> {
		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let filler_channel = WitnessFillerChannel::<_, StdChallenger, StdHashSuite>::new(
			&mut transcript,
			filler,
			self.recorded.inputs.clone(),
		);

		// `build` rejected any other scheme, so this shape opens a FRI trace.
		let mut channel = self
			.verifier
			.iop_compiler()
			.as_basefold()
			.expect("build accepts only a FRI-opened shape")
			.create_channel(filler_channel);

		let statement = channel.observe_words(inout);
		self.verifier
			.iop_verifier()
			.verify(&statement, &mut channel)?
			.check_symbolic(&mut channel)
			.map_err(binius_verifier::Error::from)?;

		// Checks the replay consumed exactly the wires the build recorded, no more and no fewer.
		channel.finish()?.finish();

		Ok(())
	}
}
