// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

use binius_core::{
	constraint_system::{ConstraintSystem, InoutSegment},
	word::Word,
};
use binius_field::{BinaryField, ExtensionField, FieldOps, Rijndael8b as B8};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop::channel::{IOPVerifierChannel, OracleSpec, oracle_setup::OracleSetupChannel};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_math::BinarySubspace;
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::DeserializeBytes;
use digest::Output;
use itertools::chain;

use super::error::Error;
use crate::{
	config::{B1, B128, LOG_WORDS_PER_ELEM, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES},
	fri::FRIParams,
	merkle_tree::BinaryMerkleTreeScheme,
	pcs::Pcs,
	pcs_compiler::PcsVerifierCompiler,
	protocols::{
		bitand::{AndCheckOutput, verify_with_channel},
		shift::WiringEvalClaim,
	},
	reduction::{Instances, reduce_constraints},
	ring_switch,
};

pub const SECURITY_BITS: usize = 96;

/// IOP verifier for a particular constraint system.
///
/// This struct encapsulates the constraint system, providing the core verification logic
/// independent of the specific IOP compilation strategy. Most users should use [`Verifier`]
/// instead, which wraps this with a BaseFold compiler.
#[derive(Debug, Clone)]
pub struct IOPVerifier {
	constraint_system: ConstraintSystem,
	log_public_words: usize,
}

impl IOPVerifier {
	/// Constructs an IOP verifier for a constraint system.
	///
	/// The constraint system must already be validated via [`ConstraintSystem::validate`].
	pub const fn new(constraint_system: ConstraintSystem, log_public_words: usize) -> Self {
		Self {
			constraint_system,
			log_public_words,
		}
	}

	/// Returns the constraint system.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		&self.constraint_system
	}

	/// Consumes the IOP verifier and returns the inner constraint system.
	pub fn into_constraint_system(self) -> ConstraintSystem {
		self.constraint_system
	}

	/// Returns log2 of the number of public constants and input/output words.
	pub const fn log_public_words(&self) -> usize {
		self.log_public_words
	}

	/// Returns log2 of the number of field elements in the packed trace.
	///
	/// The trace oracle commits only the witness's hidden segment; the public segment is a
	/// verifier-known polynomial. The shift reduction claims that segment over the shared
	/// word-index space, which spans the wider of the two segments, so the oracle covers
	/// `log_segment_words` words. That is the hidden segment's own length for every system
	/// whose private values outnumber its public ones, and the committed words above the
	/// hidden segment are zero.
	///
	/// ## Preconditions
	/// * the wider segment spans at least one field element's worth of words, which every system
	///   with more than one public or private value satisfies
	pub const fn log_witness_elems(&self) -> usize {
		let log_segment_words = self
			.constraint_system
			.log_segment_words(InoutSegment::Public);
		assert!(
			log_segment_words >= LOG_WORDS_PER_ELEM,
			"the committed trace is a whole number of field elements, so the wider value \
			 segment spans at least that many words"
		);
		log_segment_words - LOG_WORDS_PER_ELEM
	}

	/// Returns log2 of the number of words in the committed trace.
	pub const fn log_witness_words(&self) -> usize {
		self.log_witness_elems() + LOG_WORDS_PER_ELEM
	}

	/// Returns the oracle specs for the IOP channel.
	///
	/// These describe the oracles (the witness) that the prover commits to.
	///
	/// `is_zk` is the protocol-level zero-knowledge flag: in a ZK proof the witness oracle is
	/// masked, in a transparent proof it is not. The flag is taken per call so that a non-ZK
	/// oracle can still participate in a ZK protocol (e.g. indexed relation openings).
	///
	/// The specs are derived by running [`Self::verify`] against an [`OracleSetupChannel`], which
	/// records each oracle received (via `recv_oracle`) without performing any real verification.
	/// This keeps the spec sequence automatically in lockstep with the `recv_oracle` calls in
	/// `verify`, rather than duplicating it here.
	pub fn oracle_specs(&self, is_zk: bool) -> Vec<OracleSpec> {
		let mut channel = OracleSetupChannel::new(is_zk);
		// The setup channel carries concrete words, so the statement it verifies against is just
		// words — handing them through `observe_words` to get them back buys nothing.
		let inout = vec![Word::ZERO; self.constraint_system.n_inout];
		// The result is discarded: the setup channel performs no real verification (all `recv_*`
		// return zero, `assert_zero` is a no-op), so we only read back the recorded oracle specs.
		// Its wiring claim goes with it, since discharging one records no oracle.
		let _ = self.verify(&inout, &mut channel);
		channel.into_oracle_specs()
	}

	/// Verifies a proof using an IOP channel.
	///
	/// This is the core verification logic, independent of the specific IOP compilation strategy.
	/// For most users, [`Verifier::verify`] is the simpler interface.
	///
	/// `inout` is the statement as the channel carries it, which is what
	/// [`WordIPVerifierChannel::observe_words`] returns: the caller observes the concrete words and
	/// passes on what comes back. A channel that carries words as wires introduces them there, so
	/// the verification below reads the statement symbolically rather than as fixed data.
	///
	/// The reduction's wiring evaluation comes back as a [`WiringEvalClaim`] rather than being
	/// checked here, so the caller chooses how the constraint system is read: by computing the
	/// evaluation, or by any other argument that opens the claim.
	pub fn verify<Channel>(
		&self,
		inout: &[Channel::Word],
		channel: &mut Channel,
	) -> Result<WiringEvalClaim<'_, Channel::Elem>, Error>
	where
		Channel: IOPVerifierChannel<B128> + WordIPVerifierChannel<B128>,
		Channel::Elem: FieldOps<Scalar = B128> + From<B128>,
	{
		// The caller passes only the inout values. The constants are already part of the
		// constraint system, so restating them would be redundant — and a caller that restated
		// them wrongly would be describing a different system.
		if inout.len() != self.constraint_system.n_inout {
			return Err(Error::IncorrectPublicInputLength {
				expected: self.constraint_system.n_inout,
				actual: inout.len(),
			});
		}

		// The shift reduction reads the whole public segment, which is the constants followed by
		// the inout values — the order the value vector places them in. The constants are fixed by
		// the constraint system, so they lift into the channel's word type as themselves.
		let public = self
			.constraint_system
			.constants
			.iter()
			.map(|&word| Channel::Word::from(word))
			.chain(inout.iter().cloned())
			.collect::<Vec<_>>();

		let _verify_guard =
			tracing::info_span!("Verify", operation = "verify", perfetto_category = "operation")
				.entered();

		// Receive the trace oracle commitment via channel. The trace is the witness, so it is
		// witness-dependent (masked in a ZK proof).
		let trace_oracle = channel.recv_oracle(self.log_witness_elems(), true)?;

		// Reduce every constraint to one claim on the committed trace.
		let reduction = reduce_constraints(
			self.constraint_system(),
			Instances::Single,
			InoutSegment::Public,
			&public,
			channel,
		)?;

		// [phase] Ring-Switching + Verify PCS Opening
		let pcs_guard = tracing::info_span!(
			"[phase] Verify PCS Opening",
			phase = "verify_pcs_opening",
			perfetto_category = "phase"
		)
		.entered();

		// Ring-switching verification of the witness claim.
		let eval_point = reduction.trace_point();
		let ring_switch::RingSwitchVerifyOutput {
			eq_r_double_prime,
			sumcheck_claim,
		} = ring_switch::verify(reduction.shift.witness_eval().clone(), &eval_point, channel)?;

		let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
		let eval_point_high = eval_point[log_packing..].to_vec();

		let transparent = Box::new(move |point: &[Channel::Elem]| {
			ring_switch::eval_rs_eq(&eval_point_high, point, eq_r_double_prime.as_ref())
		});

		// Verify oracle relations (runs BaseFold internally and verifies the product check). The
		// intmul pushforward relation, when the IntMul reduction ran, was already queued inside
		// phase 5.
		channel.verify_oracle_relation(trace_oracle, transparent, sumcheck_claim)?;

		drop(pcs_guard);

		Ok(reduction.wiring)
	}

	/// Observes the statement, verifies against it, and discharges the wiring claim in the field.
	///
	/// This is the whole verification for a channel that carries field values rather than wires.
	/// Observing is done here rather than by the caller.
	/// Such a channel has no other use for the concrete words.
	pub fn verify_statement<Channel>(
		&self,
		inout: &[Word],
		channel: &mut Channel,
	) -> Result<(), Error>
	where
		Channel: IOPVerifierChannel<B128, Elem = B128> + WordIPVerifierChannel<B128>,
	{
		let inout = channel.observe_words(inout);
		self.verify(&inout, channel)?.check_native()?;
		Ok(())
	}
}

/// Struct for verifying instances of a particular constraint system.
///
/// The [`Self::setup`] constructor determines public parameters for proving instances of the given
/// constraint system. Then [`Self::verify`] is called one or more times with individual instances.
#[derive(Clone)]
pub struct Verifier<H: HashSuite> {
	/// The constraint system and the reduction that opens it, independent of the commitment.
	iop_verifier: IOPVerifier,
	/// The commitment scheme's parameters, chosen once at setup.
	iop_compiler: PcsVerifierCompiler<B128>,
	/// The verifier creates its Merkle transcript channels with the hash suite `H`.
	_hash_marker: PhantomData<H>,
}

impl<H> Verifier<H>
where
	H: HashSuite,
	Output<H::LeafHash>: DeserializeBytes,
{
	/// Constructs a verifier for a constraint system, opening the trace with the default scheme.
	///
	/// See [`Verifier`] struct documentation for details.
	pub fn setup(constraint_system: ConstraintSystem, log_inv_rate: usize) -> Result<Self, Error> {
		Self::setup_with_pcs(constraint_system, log_inv_rate, Pcs::default())
	}

	/// Constructs a verifier that opens the trace with the named commitment scheme.
	///
	/// Both schemes read the rate the same way.
	/// So two verifiers set up at one rate encode the same message into the same first codeword.
	/// That is what makes their proof sizes and their prover times comparable.
	///
	/// # Errors
	///
	/// Returns an error when the constraint system is invalid.
	/// Also when the chosen scheme reaches the security target under no parameters at all.
	pub fn setup_with_pcs(
		constraint_system: ConstraintSystem,
		log_inv_rate: usize,
		scheme: Pcs,
	) -> Result<Self, Error> {
		constraint_system.validate()?;

		let log_public_words = constraint_system.log_public_words(InoutSegment::Public);

		let iop_verifier = IOPVerifier::new(constraint_system, log_public_words);

		// A plain `Verifier` produces a transparent (non-ZK) proof, so the witness oracle is not
		// masked.
		let oracle_specs = iop_verifier.oracle_specs(false);

		let merkle_scheme = BinaryMerkleTreeScheme::<B128, H>::new();
		let iop_compiler = PcsVerifierCompiler::new(
			scheme,
			&merkle_scheme,
			oracle_specs,
			log_inv_rate,
			SECURITY_BITS,
		)?;

		Ok(Self {
			iop_verifier,
			iop_compiler,
			_hash_marker: PhantomData,
		})
	}

	/// Returns a reference to the IOP verifier.
	pub const fn iop_verifier(&self) -> &IOPVerifier {
		&self.iop_verifier
	}

	/// Consumes the verifier and returns the inner IOP verifier.
	pub fn into_iop_verifier(self) -> IOPVerifier {
		self.iop_verifier
	}

	/// Returns log2 of the number of words in the witness.
	pub const fn log_witness_words(&self) -> usize {
		self.iop_verifier.log_witness_words()
	}

	/// Returns log2 of the number of field elements in the packed trace.
	pub const fn log_witness_elems(&self) -> usize {
		self.iop_verifier.log_witness_elems()
	}

	/// Returns the constraint system.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		self.iop_verifier.constraint_system()
	}

	/// Returns the chosen FRI parameters, when the trace is opened with FRI.
	pub const fn fri_params(&self) -> Option<&FRIParams<B128>> {
		match self.iop_compiler.as_basefold() {
			Some(compiler) => Some(compiler.fri_params()),
			None => None,
		}
	}

	/// Returns which commitment scheme opens the trace.
	pub const fn pcs(&self) -> Pcs {
		self.iop_compiler.scheme()
	}

	/// Returns log2 of the number of public constants and input/output words.
	pub const fn log_public_words(&self) -> usize {
		self.iop_verifier.log_public_words()
	}

	/// Returns the IOP compiler for creating verifier channels.
	pub const fn iop_compiler(&self) -> &PcsVerifierCompiler<B128> {
		&self.iop_compiler
	}

	pub fn verify<Challenger_: Challenger>(
		&self,
		inout: &[Word],
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		let cs = self.iop_verifier.constraint_system();

		let _verify_scope = tracing::info_span!(
			"Verify",
			n_hidden_words = cs.n_hidden_words(InoutSegment::Public),
			n_bitand = cs.and_constraints.len(),
			n_intmul = cs.imul_constraints.len(),
		)
		.entered();

		// The two schemes lay the transcript out differently, so each arm builds its own channel
		// and closes it with its own opening. What runs in between is the same verification.
		match &self.iop_compiler {
			PcsVerifierCompiler::BaseFold(compiler) => {
				let mut channel =
					compiler.create_channel_from_transcript::<H, Challenger_, _>(transcript);
				self.iop_verifier.verify_statement(inout, &mut channel)?;
				channel.finish()?;
			}
			PcsVerifierCompiler::Ligerito(compiler) => {
				let mut channel =
					compiler.create_channel_from_transcript::<H, Challenger_, _>(transcript);
				self.iop_verifier.verify_statement(inout, &mut channel)?;
				channel.finish()?;
			}
		}
		Ok(())
	}
}

/// Verifies the batched BitAnd check: `A & B == C` on every row.
///
/// This is the univariate-skip zerocheck of `A(Z, X) * B(Z, X) - C(Z, X) == 0` for all rows
/// `(Z, X)`, where `Z` is the bit index within a 64-bit word and `X` is the row index.
///
/// # Arguments
///
/// - `log_constraint_count`: base-2 logarithm of the row count — the operand column length, which
///   the prover zero-pads up to a power of two. The single-instance verifier passes `ceil(log2(n))`
///   for `n` AND constraints; the batched M4 verifier adds its log instance count, since one row
///   there is an (instance, constraint) pair.
/// - `eval_domain`: the univariate-skip domain, one dimension above the 64-bit word, already lifted
///   to `F`. The caller passes it so it matches the shift reduction's domain by construction.
/// - `channel`: the verifier channel that reads messages and redraws Fiat-Shamir challenges.
///
/// # Errors
///
/// Returns an error if any sumcheck round message or the final consistency check fails.
pub fn verify_bitand_reduction<F, C>(
	log_constraint_count: usize,
	eval_domain: &BinarySubspace<F>,
	channel: &mut C,
) -> Result<AndCheckOutput<C::Elem>, Error>
where
	F: BinaryField + From<B8>,
	C: IPVerifierChannel<F>,
	// Used to make deterministic basis challenges symbolic
	C::Elem: From<F>,
{
	let small_field_zerocheck_challenges = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES
		.into_iter()
		.take(log_constraint_count)
		.map(|b8_val| C::Elem::from(F::from(b8_val)))
		.collect::<Vec<_>>();

	let big_field_zerocheck_challenges =
		channel.sample_many(log_constraint_count - small_field_zerocheck_challenges.len());

	let zerocheck_challenges =
		chain!(small_field_zerocheck_challenges, big_field_zerocheck_challenges)
			.collect::<Vec<_>>();
	verify_with_channel(&zerocheck_challenges, channel, eval_domain)
}
