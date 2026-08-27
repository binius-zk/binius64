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
//! How much a level costs depends on what it does with the inner wiring claim, which is what
//! [`Discharge`] selects.
//!
//! - [`Discharge::InCircuit`] evaluates the claim as constraints. That walks every inner
//!   constraint, so a level costs about fifteen times the rows of the level it verifies, and the
//!   tower diverges.
//! - [`Discharge::Deferred`] puts the claim on public wires instead and leaves it unchecked. The
//!   claim is logarithmic in the inner system, so the cost proportional to it disappears.
//!
//! Deferring moves work rather than removing it. Whoever holds the outer proof owes the check,
//! and [`RecursiveCircuit::check_deferred`] is it. Nothing anywhere verifies the claim unless
//! that runs.

use std::iter;

use binius_core::{constraint_system::ValueVec, word::Word};
use binius_field::Ghash128b as B128;
use binius_frontend::{Circuit, PopulateError, Wire, WitnessFiller};
use binius_hash::StdHashSuite;
use binius_ip::channel::WordIPVerifierChannel;
use binius_transcript::VerifierTranscript;
use binius_verifier::{
	Pcs, Verifier,
	config::StdChallenger,
	protocols::shift::{DeferredWiringClaim, WiringEvalShape},
};

use crate::{Binius64BuilderChannel, Recorded, WitnessFillerChannel, merkle::element_words};

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

	/// [`RecursiveCircuit::check_deferred`] was called on a circuit that checks in-circuit.
	#[error("this circuit discharges the wiring claim in-circuit, so nothing is left to check")]
	NothingDeferred,
}

/// What a recursive circuit does with the inner proof's wiring claim.
///
/// The claim says the wiring multilinear evaluates to a stated value. Discharging it means
/// evaluating that multilinear, which walks every constraint of the inner system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Discharge {
	/// Evaluate the claim as constraints, inside the circuit.
	///
	/// Self-contained: an outer proof of this circuit needs nothing checked beside it. The cost
	/// is proportional to the inner system, so a tower of these cannot close.
	#[default]
	InCircuit,

	/// Put the claim on public wires and leave it unchecked.
	///
	/// The claim is logarithmic in the inner system, so this is the cheap half of the trade. It
	/// is only sound if someone settles the claim: see [`RecursiveCircuit::check_deferred`].
	Deferred,
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

	/// The wiring claim this circuit exported, when it exported one.
	///
	/// `None` under [`Discharge::InCircuit`], where the claim was checked instead.
	deferred: Option<Deferred>,
}

/// A wiring claim a recursive circuit exported rather than checked.
struct Deferred {
	/// How the claim's flat input splits back into its sections.
	///
	/// Fixed by the inner system, so it is build-time data and travels with the circuit rather
	/// than with each proof.
	shape: WiringEvalShape,

	/// The inout wires carrying the claim: two per element, `(lo, hi)`.
	///
	/// The claim's inputs come first, in the order the wiring multilinear reads them, and the
	/// claimed evaluation is the last element.
	wires: Vec<Wire>,
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
		Self::build_with(verifier, Discharge::InCircuit)
	}

	/// Records `verifier`'s run, settling the inner wiring claim the way `discharge` says.
	///
	/// See [`Discharge`] for what the choice costs, and what it obliges.
	///
	/// # Errors
	///
	/// As [`build`](Self::build).
	pub fn build_with(
		verifier: Verifier<StdHashSuite>,
		discharge: Discharge,
	) -> Result<Self, Error> {
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

		// The verifier's own arithmetic becomes constraints, down to every assertion along the way.
		let claim = verifier.iop_verifier().verify(&statement, &mut channel)?;

		// The wiring claim is the one piece whose cost tracks the inner system, so it is the one
		// piece worth a choice.
		let exported = match discharge {
			Discharge::InCircuit => {
				claim
					.check_symbolic(&mut channel)
					.map_err(binius_verifier::Error::from)?;
				None
			}
			Discharge::Deferred => Some(claim.defer()),
		};

		let mut builder_channel = channel.finish()?;

		// The inner statement is bound first, so it occupies the front of the outer statement and
		// an exported claim follows it. `check_deferred` reads that layout back.
		let statement = builder_channel.bind_public(statement);
		let deferred = exported.map(|claim| {
			let DeferredWiringClaim {
				shape,
				inputs,
				claimed,
			} = claim;
			let elems = inputs
				.into_iter()
				.chain(iter::once(claimed))
				.collect::<Vec<_>>();
			Deferred {
				shape,
				wires: builder_channel.bind_public_elems(&elems),
			}
		});

		Ok(Self {
			verifier,
			recorded: builder_channel.build(),
			statement,
			deferred,
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

	/// The inout wires carrying the exported wiring claim, or `None` if it was checked in-circuit.
	///
	/// Two wires per element, `(lo, hi)`, inputs first and the claimed evaluation last.
	pub fn deferred_wires(&self) -> Option<&[Wire]> {
		self.deferred.as_ref().map(|d| d.wires.as_slice())
	}

	/// Discharges the wiring claim this circuit exported, reading it off the outer statement.
	///
	/// This is the check [`Discharge::Deferred`] skipped. **Without it the outer proof attests to
	/// strictly less than a verification**: the inner proof's wiring constraint is simply not
	/// checked, anywhere, by anyone.
	///
	/// `outer_inout` is the outer proof's public values, which is what
	/// [`ValueVec::inout`](binius_core::constraint_system::ValueVec::inout) returns for a witness
	/// this circuit produced. The inner statement occupies its front and the claim follows.
	///
	/// The cost is one native evaluation of the inner wiring multilinear. That is ordinary work
	/// outside a circuit, and it is paid once no matter how many levels deferred to here.
	///
	/// # Errors
	///
	/// Returns [`Error::NothingDeferred`] when the circuit checks the claim in-circuit.
	///
	/// Returns [`Error::StatementLength`] when `outer_inout` is not this circuit's statement.
	///
	/// Returns [`Error::Verifier`] when the claim does not hold.
	pub fn check_deferred(&self, outer_inout: &[Word]) -> Result<(), Error> {
		let Some(deferred) = &self.deferred else {
			return Err(Error::NothingDeferred);
		};

		// The claim sits behind the inner statement, so the two lengths together fix the layout.
		// A mismatch means the caller handed over some other circuit's statement.
		let expected = self.statement.len() + deferred.wires.len();
		if outer_inout.len() != expected {
			return Err(Error::StatementLength {
				expected,
				actual: outer_inout.len(),
			});
		}

		// Each element is the `(lo, hi)` pair the binding placed, in the order the wiring
		// multilinear reads its input. The claimed evaluation is the last of them.
		let elems = outer_inout[self.statement.len()..]
			.chunks_exact(2)
			.map(|pair| {
				B128::new((u128::from(pair[1].as_u64()) << 64) | u128::from(pair[0].as_u64()))
			})
			.collect::<Vec<_>>();
		let (inputs, claimed) = elems
			.split_last()
			.map(|(claimed, inputs)| (inputs.to_vec(), *claimed))
			.expect("a bound claim holds at least its claimed evaluation");

		let claim = DeferredWiringClaim {
			shape: deferred.shape,
			inputs,
			claimed,
		};
		claim
			.check(self.verifier.constraint_system())
			.map_err(binius_verifier::Error::from)?;
		Ok(())
	}

	/// Verifies an outer proof of this circuit, settling everything it defers.
	///
	/// This is the safe way to check an outer proof, and the reason it exists is that the unsafe
	/// way is otherwise shorter. Verifying the proof alone attests to a *verification minus its
	/// wiring constraint* whenever the claim was deferred, and nothing about the proof says so.
	///
	/// ```text
	///   verify the outer proof  +  check_deferred  =  the inner statement really was proved
	/// ```
	///
	/// `outer` must be a verifier for [`circuit`](Self::circuit); a different system would check a
	/// different proof.
	///
	/// # Errors
	///
	/// Returns [`Error::Verifier`] when the outer proof does not verify, or when the deferred
	/// claim does not hold.
	pub fn verify_outer(
		&self,
		outer: &Verifier<StdHashSuite>,
		inout: &[Word],
		proof: Vec<u8>,
	) -> Result<(), Error> {
		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		outer.verify(inout, &mut transcript)?;
		transcript
			.finalize()
			.map_err(binius_verifier::Error::from)?;

		// An in-circuit discharge already checked the claim as constraints, so there is nothing
		// left to settle and nothing to complain about.
		if self.deferred.is_some() {
			self.check_deferred(inout)?;
		}
		Ok(())
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
		let claim = self
			.verifier
			.iop_verifier()
			.verify(&statement, &mut channel)?;

		// This channel carries values, so the same claim comes back concrete here. Under
		// `Discharge::Deferred` those values are what the bound public wires need; the build's
		// equality then pins them against the wires the circuit derived.
		//
		// `assert_zero` is a no-op on this channel, so checking in-circuit costs nothing to mirror.
		let exported = match self.deferred {
			None => {
				claim
					.check_symbolic(&mut channel)
					.map_err(binius_verifier::Error::from)?;
				None
			}
			Some(_) => Some(claim.defer()),
		};

		// Checks the replay consumed exactly the wires the build recorded, no more and no fewer.
		// This also ends the channel's borrow of the filler, so the claim can be written below.
		channel.finish()?.finish();

		if let (Some(deferred), Some(claim)) = (&self.deferred, exported) {
			let words = claim
				.inputs
				.iter()
				.chain(iter::once(&claim.claimed))
				.flat_map(|elem| element_words(u128::from(*elem)));
			for (&wire, word) in iter::zip(&deferred.wires, words) {
				filler[wire] = Word(word);
			}
		}

		Ok(())
	}
}
