//! Generic (circuit-agnostic) plain prove/capture pipeline for one leaf constraint
//! system.
//!
//! It has NO knowledge of any application circuit. Given a prepared
//! `ConstraintSystem` it runs the plain (non-ZK) pinned pipeline
//! (`Verifier::setup(cs, 1)` + `Prover::setup` + `prove`), which is exactly what the
//! synthetic tests use to produce genuine leaf proofs whose deferred monster claim is
//! then captured by [`crate::recorder::verify_and_capture`].

use binius_core::constraint_system::{ConstraintSystem, ValueVec};
use binius_core::word::Word;
use binius_field::arch::OptimalPackedB128;
use binius_hash::StdHashSuite;
use binius_prover::Prover;
use binius_transcript::ProverTranscript;
use binius_verifier::{Verifier, config::StdChallenger};

/// log2 inverse Reed-Solomon rate for the plain leaf pipeline.
pub const LOG_INV_RATE: usize = 1;

/// Plain (non-ZK) pipeline for one leaf CS: `Verifier::setup(cs, 1)` + `Prover::setup`.
pub struct LeafPipeline {
	pub verifier: Verifier<StdHashSuite>,
	pub prover: Prover<OptimalPackedB128, StdHashSuite>,
}

impl LeafPipeline {
	pub fn setup(cs: ConstraintSystem) -> anyhow::Result<Self> {
		let verifier = Verifier::<StdHashSuite>::setup(cs, LOG_INV_RATE)
			.map_err(|e| anyhow::anyhow!("verifier setup: {e}"))?;
		let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(verifier.clone())
			.map_err(|e| anyhow::anyhow!("prover setup: {e}"))?;
		Ok(Self { verifier, prover })
	}

	/// Generates one real proof of the leaf.
	pub fn prove(&self, witness: &ValueVec) -> anyhow::Result<Vec<u8>> {
		let mut pt = ProverTranscript::new(StdChallenger::default());
		self.prover
			.prove(witness.clone(), &mut pt)
			.map_err(|e| anyhow::anyhow!("leaf prove: {e}"))?;
		Ok(pt.finalize())
	}

	pub fn public(witness: &ValueVec) -> Vec<Word> {
		witness.public().to_vec()
	}
}
