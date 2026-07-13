//! Multi-inner generalization of upstream's ZK wrap (vendored/adapted from
//! binius64 `crates/verifier/src/zk_config.rs` + `crates/prover/src/zk_config.rs`
//! at rev c799aa10, Apache-2.0/MIT). Concrete over OptimalPackedB128 + StdHashSuite.
//!
//! ONE outer IronSpartan circuit symbolically executes the Binius64 IOP verifier for K
//! (possibly different) inner constraint systems, in order. All K inner proofs and the
//! outer proof share a single transcript and ONE combined BaseFold opening:
//! oracle layout = [outer_precommit, inner_1 oracles, ..., inner_K oracles, outer suffix].

use binius_core::{
	constraint_system::{ConstraintSystem, ValueVec},
	word::Word,
};
use binius_field::{BinaryField128bGhash as B128, arch::OptimalPackedB128};
use binius_hash::StdHashSuite;
use binius_iop::{
	basefold_compiler::BaseFoldZKVerifierCompiler,
	channel::OracleSpec,
	fri::{self, MinProofSizeStrategy},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_iop_prover::basefold_compiler::BaseFoldZKProverCompiler;
use binius_math::ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded};
use binius_prover::{
	IOPProver, merkle_tree::prover::BinaryMerkleTreeProver,
	protocols::shift::build_key_collection,
};
use binius_spartan_frontend::{
	compiler::compile,
	constraint_system::{BlindingInfo, WitnessLayout},
};
use binius_spartan_prover::wrapper::{ReplayChannel, ZKWrappedProverChannel};
use binius_spartan_verifier::{
	IOPVerifier as IronSpartanIOPVerifier,
	constraint_system::ConstraintSystemPadded,
	wrapper::{IronSpartanBuilderChannel, ZKWrappedVerifierChannel},
};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{IOPVerifier, SECURITY_BITS, config::LOG_WORDS_PER_ELEM};
use rand::CryptoRng;

type ProverNTT = NeighborsLastMultiThread<GenericPreExpanded<B128>>;

/// Multi-inner ZK verifier: K Binius64 leaves verified by ONE outer IronSpartan circuit.
#[derive(Clone)]
pub struct MultiZKVerifier {
	inner_iop_verifiers: Vec<IOPVerifier>,
	outer_iop_verifier: IronSpartanIOPVerifier<B128>,
	basefold_compiler: BaseFoldZKVerifierCompiler<B128, BinaryMerkleTreeScheme<B128, StdHashSuite>>,
}

impl MultiZKVerifier {
	pub fn setup(
		constraint_systems: Vec<ConstraintSystem>,
		log_inv_rate: usize,
	) -> anyhow::Result<Self> {
		anyhow::ensure!(!constraint_systems.is_empty(), "need at least one inner CS");

		let mut inner_iop_verifiers = Vec::with_capacity(constraint_systems.len());
		for mut constraint_system in constraint_systems {
			constraint_system.validate_and_prepare()?;
			let n_public = constraint_system.value_vec_layout.offset_witness;
			let log_public_words = log2_ceil_usize(n_public);
			anyhow::ensure!(n_public.is_power_of_two());
			anyhow::ensure!(log_public_words >= LOG_WORDS_PER_ELEM);
			inner_iop_verifiers.push(IOPVerifier::new(constraint_system, log_public_words));
		}

		// Symbolically execute ALL K inner verifiers, in order, into one outer circuit.
		let outer_builder = {
			let mut builder_channel = IronSpartanBuilderChannel::new();
			for inner in &inner_iop_verifiers {
				let dummy_public_words = vec![Word::from_u64(0); 1 << inner.log_public_words()];
				inner
					.verify(&dummy_public_words, &mut builder_channel)
					.expect("symbolic verify should not fail");
			}
			builder_channel.finish()
		};
		let (outer_cs, _) = compile(outer_builder);

		tracing::info!(
			n_public = outer_cs.n_public(),
			n_precommit = outer_cs.n_precommit(),
			n_private = outer_cs.n_private(),
			n_mul_constraints = outer_cs.mul_constraints().len(),
			"multi-ZK wrapper circuit stats"
		);

		let n_test_queries = fri::calculate_n_test_queries(SECURITY_BITS, log_inv_rate);
		let blinding_info = BlindingInfo {
			n_dummy_wires: n_test_queries,
			n_dummy_constraints: 2,
		};
		let outer_cs = ConstraintSystemPadded::new(outer_cs, blinding_info);
		let outer_iop_verifier = IronSpartanIOPVerifier::new(outer_cs);

		// Transcript layout: outer precommit first, then all K inners' oracles, then the
		// outer suffix (private, mask).
		let outer_oracle_specs = outer_iop_verifier.oracle_specs();
		let mut oracle_specs: Vec<OracleSpec> = vec![outer_oracle_specs[0]];
		for inner in &inner_iop_verifiers {
			oracle_specs.extend(inner.oracle_specs());
		}
		oracle_specs.extend(outer_oracle_specs[1..].to_vec());

		let merkle_scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		let basefold_compiler = BaseFoldZKVerifierCompiler::new(
			merkle_scheme,
			oracle_specs,
			log_inv_rate,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		Ok(Self {
			inner_iop_verifiers,
			outer_iop_verifier,
			basefold_compiler,
		})
	}

	pub fn inner_iop_verifiers(&self) -> &[IOPVerifier] {
		&self.inner_iop_verifiers
	}

	pub fn outer_iop_verifier(&self) -> &IronSpartanIOPVerifier<B128> {
		&self.outer_iop_verifier
	}

	pub fn basefold_compiler(
		&self,
	) -> &BaseFoldZKVerifierCompiler<B128, BinaryMerkleTreeScheme<B128, StdHashSuite>> {
		&self.basefold_compiler
	}

	/// Verifies the combined proof: K inner Binius64 verifications through the wrapped
	/// channel (publics per inner, in setup order), then the outer Spartan verification.
	pub fn verify<Challenger_: Challenger>(
		&self,
		publics: &[Vec<Word>],
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> anyhow::Result<()> {
		anyhow::ensure!(
			publics.len() == self.inner_iop_verifiers.len(),
			"publics len mismatch"
		);

		let channel = self.basefold_compiler.create_channel(transcript);
		let mut wrapped_channel = ZKWrappedVerifierChannel::new(channel, &self.outer_iop_verifier)?;

		for (inner, public) in self.inner_iop_verifiers.iter().zip(publics) {
			inner.verify(public, &mut wrapped_channel)?;
		}

		wrapped_channel.finish()?;
		Ok(())
	}
}

/// Multi-inner ZK prover: proves K inner witnesses + the outer wrapper in one transcript.
pub struct MultiZKProver {
	inner_iop_provers: Vec<IOPProver>,
	inner_iop_verifiers: Vec<IOPVerifier>,
	outer_iop_prover: binius_spartan_prover::IOPProver<B128>,
	outer_layout: WitnessLayout<B128>,
	basefold_compiler:
		BaseFoldZKProverCompiler<OptimalPackedB128, ProverNTT, BinaryMerkleTreeProver<B128, StdHashSuite>>,
}

impl MultiZKProver {
	pub fn setup(mzk_verifier: &MultiZKVerifier) -> anyhow::Result<Self> {
		let inner_iop_verifiers = mzk_verifier.inner_iop_verifiers().to_vec();
		let inner_iop_provers = inner_iop_verifiers
			.iter()
			.map(|v| {
				let key_collection = build_key_collection(v.constraint_system());
				IOPProver::new(v.clone(), key_collection)
			})
			.collect::<Vec<_>>();

		// Re-derive the outer constraint system and layout via the same symbolic execution
		// (must match MultiZKVerifier::setup's order exactly).
		let outer_builder = {
			let mut builder_channel = IronSpartanBuilderChannel::new();
			for inner in &inner_iop_verifiers {
				let dummy_public_words = vec![Word::from_u64(0); 1 << inner.log_public_words()];
				inner
					.verify(&dummy_public_words, &mut builder_channel)
					.expect("symbolic verify should not fail");
			}
			builder_channel.finish()
		};
		let (outer_cs, outer_layout) = compile(outer_builder);

		let outer_cs = ConstraintSystemPadded::new(
			outer_cs,
			mzk_verifier
				.outer_iop_verifier()
				.constraint_system()
				.blinding_info()
				.clone(),
		);
		let outer_layout = outer_layout.with_blinding(outer_cs.blinding_info().clone());
		let outer_iop_prover = binius_spartan_prover::IOPProver::new(outer_cs);

		let subspace = mzk_verifier.basefold_compiler().max_subspace();
		let domain_context = GenericPreExpanded::generate_from_subspace(subspace);
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);
		let merkle_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::new();
		let basefold_compiler = BaseFoldZKProverCompiler::from_verifier_compiler(
			mzk_verifier.basefold_compiler(),
			ntt,
			merkle_prover,
		);

		Ok(Self {
			inner_iop_provers,
			inner_iop_verifiers,
			outer_iop_prover,
			outer_layout,
			basefold_compiler,
		})
	}

	/// Proves K inner witnesses (setup order) + the outer wrapper in one transcript.
	pub fn prove<Challenger_: Challenger>(
		&self,
		witnesses: Vec<ValueVec>,
		mut rng: impl CryptoRng,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> anyhow::Result<()> {
		anyhow::ensure!(
			witnesses.len() == self.inner_iop_provers.len(),
			"witnesses len mismatch"
		);

		let public_words: Vec<Vec<Word>> = witnesses.iter().map(|w| w.public().to_vec()).collect();

		let basefold_channel = self.basefold_compiler.create_channel(transcript, &mut rng);
		let mut wrapped_channel = ZKWrappedProverChannel::new(
			basefold_channel,
			&self.outer_iop_prover,
			&self.outer_layout,
			&mut rng,
			{
				let inner_iop_verifiers = &self.inner_iop_verifiers;
				let public_words = public_words.clone();
				move |replay_channel: &mut ReplayChannel<'_, B128>| {
					for (inner, public) in inner_iop_verifiers.iter().zip(&public_words) {
						inner
							.verify(public, replay_channel)
							.expect("replay verification should not fail");
					}
				}
			},
		);

		for (prover, witness) in self.inner_iop_provers.iter().zip(witnesses) {
			prover.prove::<OptimalPackedB128, _>(witness, &mut wrapped_channel)?;
		}

		wrapped_channel.finish(rng)?;
		Ok(())
	}
}
