//! Integrated wrap: K same-shape Binius64 leaf IOPs through SUBSTITUTING wrapped
//! channels + one outer IronSpartan proof + one combined ZK BaseFold opening + the
//! STEP-2 monster discharge — ALL on ONE Fiat-Shamir transcript.
//!
//! Vendored/generalized from recursion-window/src/multi_zk.rs (itself vendored from
//! upstream zk_config.rs at pinned rev c799aa10) with the spec §5.1/§5.2 seam added.
//!
//! Transcript layout (single FS stream, positions identical to Phase 1b through the
//! wrap segment — the substitution performs no transcript operations):
//!
//!   [outer precommit oracle]
//!   [leaf 1 IOP segment] ... [leaf K IOP segment]     <- substituting channel active;
//!                                                        each leaf's monster site
//!                                                        consumes one artifact value
//!   [outer IronSpartan proof segment]
//!   [combined ZK BaseFold opening (all K leaf oracles + outer oracles)]
//!   [DISCHARGE segment: observe(VKM) | observe(statement from the verifier's OWN
//!    capture sink) | Phase A | 8K d values | digest_D | Phase C | Phase B |
//!    BaseFold(M_D) | BaseFold(M_VK corner)]           <- discharge challenges are all
//!                                                        sampled after every claim is
//!                                                        bound (P0.1 before mu)
//!
//! The proof artifact = (transcript bytes, K monster values). The K values are consumed
//! exactly once each, at the K substitution sites; the verifier re-observes them into FS
//! via its self-built statement at the discharge head. Single-source invariant S1: the
//! discharge statement is constructed EXCLUSIVELY from the verifier's own capture sink —
//! the artifact has no second field that could carry "discharge claims", and the sink
//! record is created atomically with the Elem that check_eval multiplies.

use std::{cell::RefCell, rc::Rc, time::Instant};

use anyhow::{Context, ensure};
use binius_core::{
	constraint_system::{ConstraintSystem, ValueVec},
	word::Word,
};
use binius_field::arch::OptimalPackedB128;
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
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{
	IOPVerifier, SECURITY_BITS,
	config::{B128, LOG_WORDS_PER_ELEM, StdChallenger},
};
use binius_recursion_discharge::{
	discharge::DischargeStatement,
	step2::{ProveTimings2, VerifyTimings2, discharge_prove_step2, discharge_verify_step2},
	table::{Claim, cs_digest_bytes, extract_table},
	vk::{DischargeVkm, vkgen},
};
use rand::CryptoRng;

use crate::substituting::{ClaimRecord, ClaimSink, SubstitutingChannel, ValueSource};

type ProverNTT = NeighborsLastMultiThread<GenericPreExpanded<B128>>;

/// The combined proof artifact.
#[derive(Clone)]
pub struct IntegratedProof {
	/// The single shared FS transcript (wrap segment + discharge segment).
	pub transcript: Vec<u8>,
	/// The K prover-supplied monster values, consumed in leaf order by the K
	/// substitution sites. FS-bound by the discharge statement observation.
	pub monster_values: Vec<B128>,
}

impl IntegratedProof {
	pub fn total_bytes(&self) -> usize {
		self.transcript.len() + self.monster_values.len() * 16
	}
}

#[derive(Debug, Default, Clone)]
pub struct IntegratedProveTimings {
	pub leaf_proves_s: Vec<f64>,
	/// Replay (incl. K native monster evals — prover side) + outer prove + combined opening.
	pub wrap_finish_s: f64,
	pub discharge: ProveTimings2,
	pub total_s: f64,
}

#[derive(Debug, Default, Clone)]
pub struct IntegratedVerifyTimings {
	/// Per-leaf transcript replay through the substituting wrapped channel
	/// (NO native monster work — the ~36K-mult residual).
	pub leaf_replays_s: Vec<f64>,
	/// Outer IronSpartan verify + combined ZK BaseFold opening.
	pub outer_and_opening_s: f64,
	pub discharge: VerifyTimings2,
	pub total_s: f64,
}

#[derive(Debug, Default, Clone)]
pub struct BaselineVerifyTimings {
	/// Per-leaf transcript replay INCLUDING the native O(N) monster evaluation
	/// (Phase-1b behavior).
	pub leaf_replays_s: Vec<f64>,
	pub outer_and_opening_s: f64,
	pub total_s: f64,
}

/// Integrated verifier: K same-shape leaves + discharge VKM. Setup is one-time and
/// heavy (outer circuit compile + VKGEN); per-proof verification never touches the
/// leaf CS content (the CS is held only to define the circuit identity; the P0.2
/// digest check runs once at setup).
pub struct IntegratedVerifier {
	inner_iop_verifiers: Vec<IOPVerifier>,
	outer_iop_verifier: IronSpartanIOPVerifier<B128>,
	basefold_compiler: BaseFoldZKVerifierCompiler<B128, BinaryMerkleTreeScheme<B128, StdHashSuite>>,
	vkm: DischargeVkm,
}

impl IntegratedVerifier {
	/// `cs` is the ONE leaf shape; the wrap aggregates `k` instances of it (same-shape
	/// batch per spec limitation 2 — one VK per shape).
	pub fn setup(mut cs: ConstraintSystem, k: usize, log_inv_rate: usize) -> anyhow::Result<Self> {
		ensure!(k >= 1, "need at least one leaf");
		cs.validate_and_prepare()
			.map_err(|e| anyhow::anyhow!("validate_and_prepare: {e}"))?;

		// Discharge VK first (its commit transient is large; freed before the wrap build).
		let vkm = {
			let table = extract_table(&cs)?;
			let (vkm, vkgen_s) = vkgen(&table)?;
			tracing::info!(vkgen_s, n_terms = table.dims.n_terms, n_d = table.dims.n_d, "VKGEN done");
			vkm
		};
		// P0.2: the VKM's cs_digest IS the digest of the leaf CS this wrap is built
		// against. Checked once here; per-proof statements then carry vkm.cs_digest.
		ensure!(
			vkm.cs_digest == cs_digest_bytes(&cs),
			"P0.2: VKM cs_digest does not match the wrap's leaf CS"
		);

		let n_public = cs.value_vec_layout.offset_witness;
		let log_public_words = log2_ceil_usize(n_public);
		ensure!(n_public.is_power_of_two());
		ensure!(log_public_words >= LOG_WORDS_PER_ELEM);
		let inner_iop_verifiers: Vec<IOPVerifier> = (0..k)
			.map(|_| IOPVerifier::new(cs.clone(), log_public_words))
			.collect();

		// Symbolically execute ALL K inner verifiers, in order, into one outer circuit.
		// NO interposer here: the builder channel never invokes compute_public_value's
		// closure (builder_channel.rs is_inout branch), so the substituted and native
		// paths produce THE SAME outer constraint system by construction.
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
			"integrated wrapper circuit stats"
		);

		let n_test_queries = fri::calculate_n_test_queries(SECURITY_BITS, log_inv_rate);
		let blinding_info = BlindingInfo {
			n_dummy_wires: n_test_queries,
			n_dummy_constraints: 2,
		};
		let outer_cs = ConstraintSystemPadded::new(outer_cs, blinding_info);
		let outer_iop_verifier = IronSpartanIOPVerifier::new(outer_cs);

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
			vkm,
		})
	}

	pub fn k(&self) -> usize {
		self.inner_iop_verifiers.len()
	}

	pub fn vkm(&self) -> &DischargeVkm {
		&self.vkm
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

	/// The prepared leaf constraint system (shape identity of the wrap).
	pub fn leaf_constraint_system(&self) -> &ConstraintSystem {
		self.inner_iop_verifiers[0].constraint_system()
	}

	/// Runs the K leaf verifications through the wrapped channel with the given value
	/// source, then the outer verify + combined opening. Returns the capture sink and
	/// per-stage timings. Shared by the integrated and baseline verify paths.
	fn run_wrap_segment<'t>(
		&self,
		publics: &[Vec<Word>],
		transcript: &'t mut VerifierTranscript<StdChallenger>,
		source: ValueSource<B128>,
	) -> anyhow::Result<(Vec<ClaimRecord<B128>>, Vec<f64>, f64)> {
		ensure!(
			publics.len() == self.inner_iop_verifiers.len(),
			"publics len mismatch: {} vs K={}",
			publics.len(),
			self.inner_iop_verifiers.len()
		);
		let sink: ClaimSink<B128> = Rc::new(RefCell::new(Vec::new()));
		let mut leaf_replays_s = Vec::with_capacity(self.k());
		let outer_s;
		{
			let channel = self.basefold_compiler.create_channel(transcript);
			let mut wrapped_channel =
				ZKWrappedVerifierChannel::new(channel, &self.outer_iop_verifier)
					.map_err(|e| anyhow::anyhow!("wrapped channel new: {e}"))?;

			for (i, (inner, public)) in
				self.inner_iop_verifiers.iter().zip(publics).enumerate()
			{
				let t = Instant::now();
				let mut sub = SubstitutingChannel::new(
					&mut wrapped_channel,
					source.clone(),
					Rc::clone(&sink),
				);
				inner
					.verify(public, &mut sub)
					.map_err(|e| anyhow::anyhow!("leaf {i} verify: {e}"))?;
				leaf_replays_s.push(t.elapsed().as_secs_f64());
			}

			// P0.4 coverage (before the outer segment): exactly K claims captured, one
			// per leaf, each with the shape's arity; a substitution feed must be exactly
			// consumed.
			{
				let records = sink.borrow();
				ensure!(
					records.len() == self.k(),
					"P0.4 coverage: captured {} claims for K={} leaf verifications",
					records.len(),
					self.k()
				);
				for (i, r) in records.iter().enumerate() {
					ensure!(
						r.inputs.len() == self.vkm.dims.arity,
						"P0.4 arity: claim {i} has {} inputs, shape arity {}",
						r.inputs.len(),
						self.vkm.dims.arity
					);
				}
			}
			if let Some(remaining) = source.remaining() {
				ensure!(
					remaining == 0,
					"P0.4 coverage: {remaining} supplied monster values were not consumed"
				);
			}

			let t = Instant::now();
			wrapped_channel
				.finish()
				.map_err(|e| anyhow::anyhow!("outer verify + combined opening: {e}"))?;
			outer_s = t.elapsed().as_secs_f64();
		}
		let records = Rc::try_unwrap(sink)
			.map_err(|_| anyhow::anyhow!("claim sink still shared"))?
			.into_inner();
		Ok((records, leaf_replays_s, outer_s))
	}

	/// Verifies the integrated proof: K substituted leaf replays + outer + combined
	/// opening + STEP-2 discharge, all on the artifact's single transcript. Performs
	/// NO O(leaf) monster work anywhere.
	pub fn verify(
		&self,
		publics: &[Vec<Word>],
		proof: &IntegratedProof,
	) -> anyhow::Result<IntegratedVerifyTimings> {
		let t_total = Instant::now();
		ensure!(
			proof.monster_values.len() == self.k(),
			"P0.4 coverage: artifact supplies {} monster values for K={} leaves",
			proof.monster_values.len(),
			self.k()
		);
		let mut transcript =
			VerifierTranscript::new(StdChallenger::default(), proof.transcript.clone());
		let source = ValueSource::substitute(proof.monster_values.clone());
		let (records, leaf_replays_s, outer_and_opening_s) =
			self.run_wrap_segment(publics, &mut transcript, source)?;

		// Discharge statement built EXCLUSIVELY from the verifier's own capture sink
		// (S1 single source): these are the exact elements each leaf's check_eval
		// multiplied, at the exact claim points its transcript replay produced.
		let claims: Vec<Claim> = records
			.into_iter()
			.map(|r| Claim {
				point: r.inputs,
				value: r.output,
			})
			.collect();
		let stmt = DischargeStatement {
			cs_digest: self.vkm.cs_digest,
			n_terms: self.vkm.dims.n_terms,
			n_pad: self.vkm.dims.n_pad,
			parity: self.vkm.dims.parity,
			claims,
		};

		let discharge = discharge_verify_step2(&self.vkm, &stmt, &mut transcript)
			.context("discharge verify")?;
		transcript
			.finalize()
			.map_err(|e| anyhow::anyhow!("transcript finalize: {e}"))?;

		Ok(IntegratedVerifyTimings {
			leaf_replays_s,
			outer_and_opening_s,
			discharge,
			total_s: t_total.elapsed().as_secs_f64(),
		})
	}

	/// TEST HELPER: runs only the wrap segment of an artifact (substituted) and returns
	/// the discharge statement the verifier would build from its capture sink. Used by
	/// tests to reconstruct the statement for standalone discharge measurements. Does
	/// not finalize the transcript (the discharge segment bytes are left unread).
	#[doc(hidden)]
	pub fn capture_statement(
		&self,
		publics: &[Vec<Word>],
		proof: &IntegratedProof,
	) -> anyhow::Result<DischargeStatement> {
		ensure!(proof.monster_values.len() == self.k(), "P0.4: sidecar len");
		let mut transcript =
			VerifierTranscript::new(StdChallenger::default(), proof.transcript.clone());
		let source = ValueSource::substitute(proof.monster_values.clone());
		let (records, _, _) = self.run_wrap_segment(publics, &mut transcript, source)?;
		Ok(DischargeStatement {
			cs_digest: self.vkm.cs_digest,
			n_terms: self.vkm.dims.n_terms,
			n_pad: self.vkm.dims.n_pad,
			parity: self.vkm.dims.parity,
			claims: records
				.into_iter()
				.map(|r| Claim {
					point: r.inputs,
					value: r.output,
				})
				.collect(),
		})
	}

	/// Phase-1b BASELINE verify on the same artifact: the K leaf replays invoke the
	/// native O(N) monster closure (no substitution), then outer verify + combined
	/// opening. The trailing discharge segment is ignored (no finalize). Additionally
	/// cross-checks that the natively computed monster values equal the artifact's
	/// supplied values (honest-prover diagnostic).
	pub fn verify_baseline_native(
		&self,
		publics: &[Vec<Word>],
		proof: &IntegratedProof,
	) -> anyhow::Result<BaselineVerifyTimings> {
		let t_total = Instant::now();
		let mut transcript =
			VerifierTranscript::new(StdChallenger::default(), proof.transcript.clone());
		let (records, leaf_replays_s, outer_and_opening_s) =
			self.run_wrap_segment(publics, &mut transcript, ValueSource::Compute)?;
		for (i, r) in records.iter().enumerate() {
			ensure!(
				r.output == proof.monster_values[i],
				"native monster value of leaf {i} differs from the artifact's supplied value"
			);
		}
		Ok(BaselineVerifyTimings {
			leaf_replays_s,
			outer_and_opening_s,
			total_s: t_total.elapsed().as_secs_f64(),
		})
	}
}

/// Integrated prover. `prove` consumes self so the leaf/outer prover state can be
/// dropped before the discharge phase (16 GB laptop sequencing).
pub struct IntegratedProver {
	inner_iop_provers: Vec<IOPProver>,
	inner_iop_verifiers: Vec<IOPVerifier>,
	outer_iop_prover: binius_spartan_prover::IOPProver<B128>,
	outer_layout: WitnessLayout<B128>,
	basefold_compiler:
		BaseFoldZKProverCompiler<OptimalPackedB128, ProverNTT, BinaryMerkleTreeProver<B128, StdHashSuite>>,
	leaf_cs: ConstraintSystem,
	vkm: DischargeVkm,
}

impl IntegratedProver {
	pub fn setup(verifier: &IntegratedVerifier) -> anyhow::Result<Self> {
		let inner_iop_verifiers = verifier.inner_iop_verifiers().to_vec();
		let inner_iop_provers = inner_iop_verifiers
			.iter()
			.map(|v| {
				let key_collection = build_key_collection(v.constraint_system());
				IOPProver::new(v.clone(), key_collection)
			})
			.collect::<Vec<_>>();

		// Re-derive the outer constraint system and layout via the same symbolic
		// execution (must match IntegratedVerifier::setup's order exactly).
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
			verifier
				.outer_iop_verifier()
				.constraint_system()
				.blinding_info()
				.clone(),
		);
		let outer_layout = outer_layout.with_blinding(outer_cs.blinding_info().clone());
		let outer_iop_prover = binius_spartan_prover::IOPProver::new(outer_cs);

		let subspace = verifier.basefold_compiler().max_subspace();
		let domain_context = GenericPreExpanded::generate_from_subspace(subspace);
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);
		let merkle_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::new();
		let basefold_compiler = BaseFoldZKProverCompiler::from_verifier_compiler(
			verifier.basefold_compiler(),
			ntt,
			merkle_prover,
		);

		Ok(Self {
			inner_iop_provers,
			inner_iop_verifiers,
			outer_iop_prover,
			outer_layout,
			basefold_compiler,
			leaf_cs: verifier.leaf_constraint_system().clone(),
			vkm: verifier.vkm().clone(),
		})
	}

	/// Proves K leaf witnesses + outer wrapper + combined opening + STEP-2 discharge on
	/// one transcript. The replay closure captures each leaf's monster claim (computing
	/// the value natively — prover-side O(N) is allowed); the captured values become the
	/// artifact's `monster_values` AND the discharge statement — one source.
	///
	/// `replay_tamper`: test-only adversarial knob threaded into the replay channel's
	/// value source (`ValueSource::Compute` for honest proving).
	pub fn prove(
		self,
		witnesses: Vec<ValueVec>,
		rng: impl CryptoRng,
	) -> anyhow::Result<(IntegratedProof, IntegratedProveTimings)> {
		self.prove_with_source(witnesses, rng, ValueSource::Compute)
	}

	#[doc(hidden)]
	pub fn prove_with_source(
		self,
		witnesses: Vec<ValueVec>,
		mut rng: impl CryptoRng,
		replay_source: ValueSource<B128>,
	) -> anyhow::Result<(IntegratedProof, IntegratedProveTimings)> {
		let t_total = Instant::now();
		let Self {
			inner_iop_provers,
			inner_iop_verifiers,
			outer_iop_prover,
			outer_layout,
			basefold_compiler,
			leaf_cs,
			vkm,
		} = self;
		let k = inner_iop_provers.len();
		ensure!(witnesses.len() == k, "witnesses len mismatch");
		let public_words: Vec<Vec<Word>> =
			witnesses.iter().map(|w| w.public().to_vec()).collect();

		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let sink: ClaimSink<B128> = Rc::new(RefCell::new(Vec::new()));
		let mut leaf_proves_s = Vec::with_capacity(k);
		let wrap_finish_s;
		{
			let basefold_channel = basefold_compiler.create_channel(&mut transcript, &mut rng);
			let mut wrapped_channel = ZKWrappedProverChannel::new(
				basefold_channel,
				&outer_iop_prover,
				&outer_layout,
				&mut rng,
				{
					let inner_iop_verifiers = &inner_iop_verifiers;
					let public_words = public_words.clone();
					let sink = Rc::clone(&sink);
					let replay_source = replay_source.clone();
					move |replay_channel: &mut ReplayChannel<'_, B128>| {
						for (inner, public) in inner_iop_verifiers.iter().zip(&public_words) {
							let mut sub = SubstitutingChannel::new(
								replay_channel,
								replay_source.clone(),
								Rc::clone(&sink),
							);
							inner
								.verify(public, &mut sub)
								.expect("replay verification should not fail");
						}
					}
				},
			);

			for (prover, witness) in inner_iop_provers.iter().zip(witnesses) {
				let t = Instant::now();
				prover
					.prove::<OptimalPackedB128, _>(witness, &mut wrapped_channel)
					.map_err(|e| anyhow::anyhow!("leaf prove: {e}"))?;
				leaf_proves_s.push(t.elapsed().as_secs_f64());
			}

			let t = Instant::now();
			wrapped_channel
				.finish(rng)
				.map_err(|e| anyhow::anyhow!("wrap finish (replay + outer prove + opening): {e}"))?;
			wrap_finish_s = t.elapsed().as_secs_f64();
		}
		// Free the wrap-phase state before the discharge phase (RAM sequencing).
		drop(inner_iop_provers);
		drop(inner_iop_verifiers);
		drop(outer_iop_prover);
		drop(outer_layout);
		drop(basefold_compiler);

		// P0.4 coverage on the prover's own capture (replay ran K leaf verifies).
		let records = Rc::try_unwrap(sink)
			.map_err(|_| anyhow::anyhow!("claim sink still shared"))?
			.into_inner();
		ensure!(
			records.len() == k,
			"P0.4 coverage (prove): captured {} claims for K={k} leaves",
			records.len()
		);
		for (i, r) in records.iter().enumerate() {
			ensure!(
				r.inputs.len() == vkm.dims.arity,
				"P0.4 arity (prove): claim {i} has {} inputs, shape arity {}",
				r.inputs.len(),
				vkm.dims.arity
			);
		}
		let claims: Vec<Claim> = records
			.into_iter()
			.map(|r| Claim {
				point: r.inputs,
				value: r.output,
			})
			.collect();
		let monster_values: Vec<B128> = claims.iter().map(|c| c.value).collect();
		let stmt = DischargeStatement {
			cs_digest: vkm.cs_digest,
			n_terms: vkm.dims.n_terms,
			n_pad: vkm.dims.n_pad,
			parity: vkm.dims.parity,
			claims,
		};

		let discharge = discharge_prove_step2(&leaf_cs, &vkm, &stmt, &mut transcript)
			.context("discharge prove")?;

		let bytes = transcript.finalize();
		Ok((
			IntegratedProof {
				transcript: bytes,
				monster_values,
			},
			IntegratedProveTimings {
				leaf_proves_s,
				wrap_finish_s,
				discharge,
				total_s: t_total.elapsed().as_secs_f64(),
			},
		))
	}
}
