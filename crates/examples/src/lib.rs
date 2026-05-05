// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub mod circuits;
pub mod cli;
pub mod snapshot;

use anyhow::Result;
use binius_core::constraint_system::{ConstraintSystem, ValueVec};
use binius_frontend::{CircuitBuilder, WitnessFiller};
use binius_hash::{
	binary_merkle_tree::HashSuite, sha256::Sha256HashSuite, vision::VisionHashSuite,
};
use binius_prover::{KeyCollection, OptimalPackedB128, Prover, zk_config::ZKProver};
use binius_utils::SerializeBytes;
use binius_verifier::{
	Verifier,
	config::StdChallenger,
	transcript::{ProverTranscript, VerifierTranscript},
	zk_config::ZKVerifier,
};
use clap::ValueEnum;
pub use cli::Cli;
use digest::Output;

#[derive(Debug, Clone, ValueEnum)]
pub enum CompressionType {
	/// SHA-256 compression function
	Sha256,
	/// Vision compression function (Vision-6 leaves, Vision-4 inner-node compression)
	Vision,
}

/// Standard verifier using SHA256 compression
pub type StdVerifier = Verifier<Sha256HashSuite>;
/// Standard prover using SHA256 compression
pub type StdProver = Prover<OptimalPackedB128, Sha256HashSuite>;
/// Vision verifier (Vision-6 leaves + Vision-4 compression)
pub type VisionVerifier = Verifier<VisionHashSuite>;
/// Vision prover (Vision-6 leaves + Vision-4 compression)
pub type VisionProver = Prover<OptimalPackedB128, VisionHashSuite>;
/// Standard ZK verifier using SHA256 compression
pub type StdZKVerifier = ZKVerifier<Sha256HashSuite>;
/// Standard ZK prover using SHA256 compression
pub type StdZKProver = ZKProver<OptimalPackedB128, Sha256HashSuite>;
/// Vision ZK verifier (Vision-6 leaves + Vision-4 compression)
pub type VisionZKVerifier = ZKVerifier<VisionHashSuite>;
/// Vision ZK prover (Vision-6 leaves + Vision-4 compression)
pub type VisionZKProver = ZKProver<OptimalPackedB128, VisionHashSuite>;

/// Setup the prover and verifier and use SHA256 for Merkle tree compression.
/// Providing the `key_collection` skips expensive key collection building.
pub fn setup_sha256(
	cs: ConstraintSystem,
	log_inv_rate: usize,
	key_collection: Option<KeyCollection>,
) -> Result<(StdVerifier, StdProver)> {
	let _setup_guard = tracing::info_span!("Setup", log_inv_rate).entered();
	let verifier = Verifier::setup(cs, log_inv_rate)?;
	let prover = if let Some(key_collection) = key_collection {
		Prover::setup_with_key_collection(verifier.clone(), key_collection)?
	} else {
		Prover::setup(verifier.clone())?
	};
	Ok((verifier, prover))
}

/// Setup the prover and verifier and use the ZK-friendly Vision suite for Merkle tree hashing.
/// Providing the `key_collection` skips expensive key collection building.
pub fn setup_vision(
	cs: ConstraintSystem,
	log_inv_rate: usize,
	key_collection: Option<KeyCollection>,
) -> Result<(VisionVerifier, VisionProver)> {
	let _setup_guard = tracing::info_span!("Setup", log_inv_rate).entered();
	let verifier = Verifier::setup(cs, log_inv_rate)?;
	let prover = if let Some(key_collection) = key_collection {
		Prover::setup_with_key_collection(verifier.clone(), key_collection)?
	} else {
		Prover::setup(verifier.clone())?
	};
	Ok((verifier, prover))
}

/// Setup the ZK prover and verifier using SHA256 for Merkle tree compression.
pub fn setup_zk_sha256(
	cs: ConstraintSystem,
	log_inv_rate: usize,
) -> Result<(StdZKVerifier, StdZKProver)> {
	let _setup_guard = tracing::info_span!("ZK setup", log_inv_rate).entered();
	let verifier = ZKVerifier::setup(cs, log_inv_rate)?;
	let prover = ZKProver::setup(verifier.clone())?;
	Ok((verifier, prover))
}

/// Setup the ZK prover and verifier using the ZK-friendly Vision suite for Merkle tree hashing.
pub fn setup_zk_vision(
	cs: ConstraintSystem,
	log_inv_rate: usize,
) -> Result<(VisionZKVerifier, VisionZKProver)> {
	let _setup_guard = tracing::info_span!("ZK setup", log_inv_rate).entered();
	let verifier = ZKVerifier::setup(cs, log_inv_rate)?;
	let prover = ZKProver::setup(verifier.clone())?;
	Ok((verifier, prover))
}

pub fn prove_verify<H>(
	verifier: &Verifier<H>,
	prover: &Prover<OptimalPackedB128, H>,
	witness: ValueVec,
) -> Result<()>
where
	H: HashSuite,
	Output<H::LeafHash>: SerializeBytes + binius_utils::DeserializeBytes,
{
	let challenger = StdChallenger::default();

	let mut prover_transcript = ProverTranscript::new(challenger.clone());
	prover.prove(witness.clone(), &mut prover_transcript)?;

	let proof = prover_transcript.finalize();
	tracing::info!("Proof size: {} KiB", proof.len() / 1024);

	let mut verifier_transcript = VerifierTranscript::new(challenger, proof);
	verifier.verify(witness.public(), &mut verifier_transcript)?;
	verifier_transcript.finalize()?;

	Ok(())
}

pub fn prove_verify_zk<H>(
	verifier: &ZKVerifier<H>,
	prover: &ZKProver<OptimalPackedB128, H>,
	witness: ValueVec,
) -> Result<()>
where
	H: HashSuite,
	Output<H::LeafHash>: SerializeBytes + binius_utils::DeserializeBytes,
{
	let challenger = StdChallenger::default();

	let proof = {
		let _scope = tracing::info_span!("Prove").entered();
		let mut prover_transcript = ProverTranscript::new(challenger.clone());
		let mut rng = rand::rng();
		prover.prove(witness.clone(), &mut rng, &mut prover_transcript)?;
		prover_transcript.finalize()
	};

	tracing::info!("Proof size: {} KiB", proof.len() / 1024);

	let _scope = tracing::info_span!("Verify").entered();
	let mut verifier_transcript = VerifierTranscript::new(challenger, proof);
	verifier.verify(witness.public(), &mut verifier_transcript)?;
	verifier_transcript.finalize()?;

	Ok(())
}

/// Trait for standardizing circuit examples in the Binius framework.
///
/// This trait provides a common pattern for implementing circuit examples by separating:
/// - **Circuit parameters** (`Params`): compile-time configuration that affects circuit structure
/// - **Instance data** (`Instance`): runtime data used to populate the witness
/// - **Circuit building**: logic to construct the circuit based on parameters
/// - **Witness population**: logic to fill in witness values based on instance data
///
/// # Example Implementation
///
/// ```rust,ignore
/// struct MyExample {
///     params: MyParams,
///     // Store any gadgets or wire references needed for witness population
/// }
///
/// #[derive(clap::Args)]
/// struct MyParams {
///     #[arg(long)]
///     max_size: usize,
/// }
///
/// #[derive(clap::Args)]
/// struct MyInstance {
///     #[arg(long)]
///     input_value: Option<String>,
/// }
///
/// impl ExampleCircuit for MyExample {
///     type Params = MyParams;
///     type Instance = MyInstance;
///
///     fn build(params: MyParams, builder: &mut CircuitBuilder) -> Result<Self> {
///         // Construct circuit based on parameters
///         Ok(Self { params })
///     }
///
///     fn populate_witness(&self, instance: MyInstance, filler: &mut WitnessFiller) -> Result<()> {
///         // Fill witness values based on instance data
///         Ok(())
///     }
/// }
/// ```
///
/// # Lifecycle
///
/// 1. Parse CLI arguments to get `Params` and `Instance`
/// 2. Call `build()` with parameters to construct the circuit
/// 3. Build the constraint system
/// 4. Set up prover and verifier
/// 5. Call `populate_witness()` to fill witness values
/// 6. Generate and verify proof
pub trait ExampleCircuit: Sized {
	/// Circuit parameters that affect the structure of the circuit.
	/// These are typically compile-time constants or bounds.
	type Params: clap::Args;

	/// Instance data used to populate the witness.
	/// This represents the actual input values for a specific proof.
	type Instance: clap::Args;

	/// Build the circuit with the given parameters.
	///
	/// This method should:
	/// - Add witnesses, constants, and constraints to the builder
	/// - Store any wire references needed for witness population
	/// - Return a Self instance that can later populate witness values
	fn build(params: Self::Params, builder: &mut CircuitBuilder) -> Result<Self>;

	/// Populate witness values for a specific instance.
	///
	/// This method should:
	/// - Process the instance data (e.g., parse inputs, compute hashes)
	/// - Fill all witness values using the provided filler
	/// - Validate that instance data is compatible with circuit parameters
	fn populate_witness(&self, instance: Self::Instance, filler: &mut WitnessFiller) -> Result<()>;

	/// Generate a concise parameter summary for perfetto trace filenames.
	///
	/// This method should return a short string (5-10 chars max) that captures
	/// the most important parameters for this circuit configuration.
	/// Used to differentiate traces with different parameter settings.
	///
	/// Format suggestions:
	/// - Bytes: "2048b", "4096b"
	/// - Counts: "10p" (permutations), "5s" (signatures)
	///
	/// Returns None if no meaningful parameters to include in filename.
	fn param_summary(params: &Self::Params) -> Option<String> {
		let _ = params;
		None
	}
}
