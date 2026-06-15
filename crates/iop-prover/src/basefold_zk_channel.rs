// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP prover channel.
//!
//! This module provides [`BaseFoldZKProverChannel`], which implements [`IOPProverChannel`]
//! using FRI commitment and ZK BaseFold opening protocols. Unlike [`super::basefold_channel`],
//! this channel always applies zero-knowledge blinding to all oracles by generating masks
//! internally.

use std::iter;

use binius_field::{BinaryField, PackedField};
use binius_iop::{channel::OracleSpec, fri::FRIParams, merkle_tree::MerkleTreeScheme};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{self, PaddedSumcheckDecorator, bivariate_product::BivariateProductSumcheckProver},
};
use binius_math::{
	FieldBuffer, FieldSlice,
	inner_product::inner_product_par,
	line::{extrapolate_line, extrapolate_line_packed},
	multilinear::eq::{eq_ind_partial_eval_scalars, eq_ind_zero},
	ntt::AdditiveNTT,
};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::{SerializeBytes, checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use itertools::izip;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
	basefold::prove_mlecheck_basefold_zk_batch,
	basefold_compiler::BaseFoldZKProverCompiler,
	channel::IOPProverChannel,
	fri::{self, CommitMaskedOutput, FRIFoldProver},
	merkle_tree::MerkleTreeProver,
};

/// Oracle handle returned by [`BaseFoldZKProverChannel::send_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldZKOracle {
	index: usize,
}

/// Committed oracle data stored internally.
struct CommittedOracleData<P: PackedField, Committed> {
	/// The mask buffer generated during [`fri::commit_masked`]. Held by the channel because it is
	/// the only party that knows it.
	mask: FieldBuffer<P>,
	/// RS-encoded codeword.
	codeword: FieldBuffer<P>,
	/// Merkle commitment data for query proofs.
	committed: Committed,
}

/// A prover channel that uses ZK BaseFold for all oracle commitments and openings.
///
/// This channel owns an [`StdRng`] and generates random masks internally during
/// [`send_oracle`](IOPProverChannel::send_oracle). The caller provides only the raw witness
/// buffer (not doubled). The channel handles:
/// - Generating a random mask of equal length
/// - Interleaving witness and mask for FRI commitment
/// - Running ZK BaseFold proofs in `prove_oracle_relations`
///
/// # Type Parameters
///
/// - `F`: The binary field type
/// - `P`: The packed field type with `Scalar = F`
/// - `NTT`: The additive NTT for Reed-Solomon encoding
/// - `MerkleProver_`: The Merkle tree prover for commitments
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct BaseFoldZKProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleProver_: MerkleTreeProver<F>,
	Challenger_: Challenger,
{
	transcript: &'a mut ProverTranscript<Challenger_>,
	ntt: &'a NTT,
	merkle_prover: &'a MerkleProver_,
	oracle_specs: Vec<OracleSpec>,
	/// The combined FRI parameters over all committed oracles.
	fri_params: FRIParams<F>,
	committed_oracles: Vec<CommittedOracleData<P, MerkleProver_::Committed>>,
	/// Oracle relations queued by [`Self::prove_oracle_relations`], opened together in
	/// [`Self::finish`]. Each entry is `(oracle_index, message π_i, transparent t_i, claim s_i)`.
	queue: Vec<(usize, FieldBuffer<P>, FieldBuffer<P>, F)>,
	next_oracle_index: usize,
	rng: StdRng,
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver_, Challenger_>
	BaseFoldZKProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
	Challenger_: Challenger,
{
	/// Creates a new BaseFold ZK prover channel from a compiler with precomputed FRI parameters.
	///
	/// The `rng` is used to seed an internal `StdRng` for mask generation.
	pub fn from_compiler(
		compiler: &'a BaseFoldZKProverCompiler<P, NTT, MerkleProver_>,
		transcript: &'a mut ProverTranscript<Challenger_>,
		mut rng: impl Rng,
	) -> Self {
		Self {
			transcript,
			ntt: compiler.ntt(),
			merkle_prover: compiler.merkle_prover(),
			oracle_specs: compiler.oracle_specs().to_vec(),
			fri_params: compiler.fri_params().clone(),
			committed_oracles: Vec::new(),
			queue: Vec::new(),
			next_oracle_index: 0,
			rng: StdRng::from_rng(&mut rng),
		}
	}

	/// Returns a reference to the underlying transcript.
	pub fn transcript(&self) -> &ProverTranscript<Challenger_> {
		self.transcript
	}

	/// Consumes the channel and proves the single combined opening over **all** committed oracles.
	///
	/// All oracle relations queued by
	/// [`prove_oracle_relations`](IOPProverChannel::prove_oracle_relations) across every call are
	/// processed here in one batch: masking, one batched sumcheck reducing the masked claims to a
	/// shared point `r`, then one combined FRI opening over every committed oracle
	/// (in oracle-index order). Mirrors [`BaseFoldZKVerifierChannel::finish`].
	///
	/// [`BaseFoldZKVerifierChannel::finish`]: binius_iop::basefold_zk_channel::BaseFoldZKVerifierChannel::finish
	pub fn finish(self) {
		let Self {
			transcript,
			ntt,
			merkle_prover,
			oracle_specs,
			fri_params,
			committed_oracles,
			queue,
			next_oracle_index,
			rng: _,
		} = self;

		let n_remaining = oracle_specs.len() - next_oracle_index;
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining",);

		if queue.is_empty() {
			return;
		}

		prove_batch_zk_basefold(
			transcript,
			ntt,
			merkle_prover,
			&oracle_specs,
			&fri_params,
			committed_oracles,
			queue,
		);
	}
}

/// Proves the combined ZK BaseFold opening over all committed oracles.
///
/// This drives `channel` — the [`ProverTranscript`] taken from the destructured
/// [`BaseFoldZKProverChannel`] — through its [`IPProverChannel`] interface: it sends the masked
/// inner products σ_i, runs one batched sumcheck reducing the masked claims to a shared point `r`,
/// then opens each committed oracle with its own FRI parameters. Mirrors
/// [`binius_iop::basefold_zk_channel::BaseFoldZKVerifierChannel::finish`].
///
/// The masking inner products and the batched sumcheck process the `relations` in arrival order (so
/// each reduced eval lines up with its batched-claim coefficient), while the per-oracle evaluations
/// α_i and the FRI openings are emitted in oracle-index order. Each relation carries its oracle's
/// index, so the two orders are reconciled by indexing rather than by sorting the relations; the
/// per-oracle data (`oracle_specs`, `fri_params`, `committed_oracles`) is all indexed by oracle
/// index.
///
/// `channel` is the concrete [`ProverTranscript`] rather than an arbitrary [`IPProverChannel`]
/// because the Phase-B FRI openings write Merkle query proofs, which fall outside that interface.
#[allow(clippy::too_many_arguments)]
fn prove_batch_zk_basefold<F, P, NTT, MerkleScheme, MerkleProver_, Challenger_>(
	channel: &mut ProverTranscript<Challenger_>,
	ntt: &NTT,
	merkle_prover: &MerkleProver_,
	oracle_specs: &[OracleSpec],
	fri_params: &FRIParams<F>,
	committed_oracles: Vec<CommittedOracleData<P, MerkleProver_::Committed>>,
	relations: Vec<(usize, FieldBuffer<P>, FieldBuffer<P>, F)>,
) where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
	Challenger_: Challenger,
{
	let n_committed = committed_oracles.len();
	assert_eq!(oracle_specs.len(), n_committed);

	// TODO: Remove this limitation, it shouldn't be necessary. It is currently because of how the
	// sumcheck reduces to the multilinear evaluations (alphas).
	assert_eq!(relations.len(), n_committed, "expects exactly one relation per committed oracle",);

	// `𝐧 = max_i n_i`, the dimension of the combined codeword.
	let max_n = fri_params.rs_code().log_dim();

	// === Masking step (whitepaper 7.2) ===
	// Send the masked inner products σ_i = ⟨ω_i, t_i⟩, then sample a single masking challenge γ.
	let sigmas = relations
		.iter()
		.map(|(index, _, transparent, _)| {
			inner_product_par(&committed_oracles[*index].mask, transparent)
		})
		.collect::<Vec<_>>();
	channel.send_many(&sigmas);

	let gamma = IPProverChannel::<F>::sample(channel);
	let gamma_broadcast = P::broadcast(gamma);

	// === Phase A: batched sumcheck on the masked claims ⟨π_i', t_i⟩ = s_i' ===
	// Register provers in arrival order; form π_i' = (1-γ)π_i + γω_i, storing each clone for Phase
	// B keyed by oracle index, and pad each prover to `max_n`. `prover_oracle_indices` records the
	// oracle index behind each (arrival-order) prover so the reduced evals can be scattered back
	// into oracle-index order.
	let mut witness_primes = vec![None; n_committed];
	let mut prover_oracle_indices = Vec::with_capacity(n_committed);
	let mut provers = Vec::with_capacity(n_committed);
	for ((index, mut message, transparent, claim), sigma) in iter::zip(relations, sigmas) {
		let n_i = oracle_specs[index].log_msg_len;
		assert_eq!(message.log_len(), n_i); // pre-condition
		assert_eq!(transparent.log_len(), n_i); // pre-condition

		let mask = &committed_oracles[index].mask;

		// Modify the message π_i to be the blinded message π_i'.
		(message.as_mut(), mask.as_ref())
			.into_par_iter()
			.for_each(|(message_i, &mask_i)| {
				*message_i = extrapolate_line_packed(*message_i, mask_i, gamma_broadcast);
			});
		witness_primes[index] = Some(message.clone());
		prover_oracle_indices.push(index);

		let sum_prime = extrapolate_line(claim, sigma, gamma);
		let inner = BivariateProductSumcheckProver::new([message, transparent], sum_prime)
			.expect("π_i' and t_i have equal length");
		provers.push(PaddedSumcheckDecorator::new(inner, max_n - n_i));
	}

	let output =
		sumcheck::batch_prove(provers, channel).expect("batched sumcheck proving should succeed");

	// Reduced oracle evaluations α_i = π_i'(ρ_i) come out in arrival order; scatter them into
	// oracle-index order to match how the verifier indexes them. `output.challenges` is already
	// reversed to low-to-high (variable-indexed) order, so ρ_i is its first n_i coords.
	//
	// TODO: This will fail if one oracle isn't opened. For robustness, we should do a regular
	// multilinear evaluation in that case.
	let mut alphas = vec![F::ZERO; n_committed];
	for (eval_pos, &index) in prover_oracle_indices.iter().enumerate() {
		alphas[index] = output.multilinear_evals[eval_pos][0];
	}
	channel.send_many(&alphas);

	// === Phase B: single combined-FRI MLE-check over the piecewise-concatenated oracle ===
	// Collapse the oracle-index variables up front at sampled batching challenges `r'`: build the
	// combined multilinear 𝛑(X) = Σ_i e[i]·π_i^↑(X) with e = eq(·, r') into one 2^𝐧 buffer, and the
	// combined target s' = 𝛑(r) = Σ_i e[i]·α_i·∏_{j≥n_i}(1 - r_j).
	let point = &output.challenges;
	let log_n_oracles = log2_ceil_usize(n_committed);
	let outer_challenges = channel.sample_many(log_n_oracles);
	let eq_tensor = eq_ind_partial_eval_scalars(&outer_challenges);

	let mut combined = FieldBuffer::<P>::zeros(max_n);
	let mut s_prime = F::ZERO;
	for (witness_prime, scalar, alpha_i) in izip!(witness_primes, eq_tensor, alphas) {
		let witness_prime =
			witness_prime.expect("every committed oracle carries exactly one queued relation");
		let n_i = witness_prime.log_len();
		// Add scalar · π_i^↑ into the low 2^{n_i} block of the combined buffer (the high block is
		// the ZeroPadMSB lift, left as the zeros already in `combined`).
		if n_i >= P::LOG_WIDTH {
			let scalar_p = P::broadcast(scalar);
			let src = witness_prime.as_ref();
			combined.as_mut()[..src.len()]
				.par_iter_mut()
				.zip(src)
				.for_each(|(dst, &w)| {
					*dst += scalar_p * w;
				});
		} else {
			let src = P::from_scalars(witness_prime.iter_scalars());
			combined.as_mut()[0] += src;
		}
		s_prime += scalar * alpha_i * eq_ind_zero(&point[n_i..]);
	}

	// Codeword commitments in oracle-index order, matching `open_fri_params.input_oracles()`.
	let (committed_codewords, committeds) = committed_oracles
		.into_iter()
		.map(|committed| (committed.codeword, committed.committed))
		.unzip::<_, _, Vec<_>, Vec<_>>();
	// TODO: Annoying that we need to pass references for committeds. Maybe we can change the
	// FRIFoldProver constructor.
	let committed_codewords = iter::zip(committed_codewords, &committeds).collect();

	let fri_folder = FRIFoldProver::new_batch(fri_params, ntt, merkle_prover, committed_codewords);
	prove_mlecheck_basefold_zk_batch(
		combined,
		point,
		s_prime,
		gamma,
		&outer_challenges,
		fri_folder,
		channel,
	)
	.expect("combined MLE-check BaseFold proof should succeed");
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver_, Challenger_> IPProverChannel<F>
	for BaseFoldZKProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
	Challenger_: Challenger,
{
	fn send_one(&mut self, elem: F) {
		self.transcript.message().write_scalar(elem);
	}

	fn send_many(&mut self, elems: &[F]) {
		self.transcript.message().write_scalar_slice(elems);
	}

	fn observe_one(&mut self, val: F) {
		self.transcript.observe().write_scalar(val);
	}

	fn observe_many(&mut self, vals: &[F]) {
		self.transcript.observe().write_scalar_slice(vals);
	}

	fn sample(&mut self) -> F {
		CanSample::sample(&mut self.transcript)
	}
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver_, Challenger_> IOPProverChannel<P>
	for BaseFoldZKProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
	Challenger_: Challenger,
{
	type Oracle = BaseFoldZKOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn send_oracle(&mut self, buffer: FieldSlice<P>) -> Self::Oracle {
		let remaining = self.remaining_oracle_specs();
		assert!(!remaining.is_empty(), "send_oracle called but no remaining oracle specs");

		let index = self.next_oracle_index;
		let spec = &remaining[0];

		// ZK channel expects raw witness buffer (NOT doubled).
		assert_eq!(
			buffer.log_len(),
			spec.log_msg_len,
			"oracle buffer log_len mismatch: expected {}, got {}",
			spec.log_msg_len,
			buffer.log_len()
		);

		// Generate mask, interleave, and commit oracle `index` of the combined FRI parameters via
		// commit_masked.
		let CommitMaskedOutput {
			commitment,
			committed,
			codeword,
			mask,
		} = fri::commit_masked(
			&self.fri_params,
			index,
			self.ntt,
			self.merkle_prover,
			buffer.to_ref(),
			&mut self.rng,
		);

		// Send commitment via transcript.
		self.transcript.message().write(&commitment);

		self.committed_oracles.push(CommittedOracleData {
			mask,
			codeword,
			committed,
		});

		self.next_oracle_index += 1;

		BaseFoldZKOracle { index }
	}

	fn prove_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<
			Item = (Self::Oracle, FieldBuffer<P>, FieldBuffer<P>, P::Scalar),
		>,
	) {
		// Queue the relations; the actual opening (masking + sumcheck + combined FRI) happens once,
		// over all committed oracles, in [`Self::finish`].
		for (oracle, message, transparent, claim) in oracle_relations {
			assert!(
				oracle.index < self.committed_oracles.len(),
				"oracle index {} out of bounds, expected < {}",
				oracle.index,
				self.committed_oracles.len()
			);
			self.queue.push((oracle.index, message, transparent, claim));
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		BinaryField, BinaryField128bGhash, Field, PackedBinaryGhash1x128b, PackedField,
	};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		basefold_compiler::BaseFoldZKVerifierCompiler,
		channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
		fri::MinProofSizeStrategy,
	};
	use binius_math::{
		BinarySubspace, FieldBuffer,
		inner_product::inner_product_buffers,
		multilinear::eq::eq_ind_partial_eval,
		ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::IOPProverChannel;
	use crate::{
		basefold_compiler::BaseFoldZKProverCompiler, merkle_tree::prover::BinaryMerkleTreeProver,
	};

	type StdChallenger = HasherChallenger<StdDigest>;

	const LOG_INV_RATE: usize = 1;
	const SECURITY_BITS: usize = 32;

	fn calculate_n_test_queries(security_bits: usize, log_inv_rate: usize) -> usize {
		security_bits.div_ceil(log_inv_rate)
	}

	fn make_ntt(
		subspace: &BinarySubspace<BinaryField128bGhash>,
	) -> NeighborsLastSingleThread<GenericOnTheFly<BinaryField128bGhash>> {
		let domain_context = GenericOnTheFly::generate_from_subspace(subspace);
		NeighborsLastSingleThread::new(domain_context)
	}

	fn make_merkle_prover() -> BinaryMerkleTreeProver<BinaryField128bGhash, StdHashSuite> {
		BinaryMerkleTreeProver::new()
	}

	fn generate_zk_oracle_data<F, P, R: Rng>(
		rng: &mut R,
		n_vars: usize,
	) -> (FieldBuffer<P>, FieldBuffer<P>, F)
	where
		F: BinaryField,
		P: PackedField<Scalar = F>,
	{
		let buffer = random_field_buffer::<P>(&mut *rng, n_vars);
		let evaluation_point = random_scalars::<F>(&mut *rng, n_vars);
		let transparent_poly = eq_ind_partial_eval::<P>(&evaluation_point);
		let evaluation_claim = inner_product_buffers(&buffer, &transparent_poly);
		(buffer, transparent_poly, evaluation_claim)
	}

	#[test]
	fn test_basefold_zk_channel_single_oracle() {
		type F = BinaryField128bGhash;
		type P = PackedBinaryGhash1x128b;

		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 8;

		let (buffer, transparent_poly, eval_claim) =
			generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

		let oracle_specs = vec![OracleSpec::new_zk(n_vars)];

		let merkle_prover = make_merkle_prover();
		let verifier_compiler = BaseFoldZKVerifierCompiler::new(
			merkle_prover.scheme().clone(),
			oracle_specs,
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// === PROVER SIDE ===
		let ntt = make_ntt(verifier_compiler.max_subspace());
		let prover_compiler = BaseFoldZKProverCompiler::<P, _, _>::from_verifier_compiler(
			&verifier_compiler,
			ntt,
			merkle_prover,
		);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_rng = StdRng::seed_from_u64(1);
		let mut prover_channel = prover_compiler.create_channel(&mut prover_transcript, prover_rng);

		let oracle = prover_channel.send_oracle(buffer.to_ref());
		assert_eq!(oracle.index, 0);

		prover_channel.prove_oracle_relations([(
			oracle,
			buffer,
			transparent_poly.clone(),
			eval_claim,
		)]);
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler.create_channel(&mut verifier_transcript);

		let v_oracle = verifier_channel.recv_oracle().unwrap();

		verifier_channel
			.verify_oracle_relations([OracleLinearRelation {
				oracle: v_oracle,
				transparent: Box::new(move |point: &[F]| {
					let eq = eq_ind_partial_eval::<P>(point);
					inner_product_buffers(&transparent_poly, &eq)
				}),
				claim: eval_claim,
			}])
			.unwrap();
		verifier_channel.finish().unwrap();
	}

	#[test]
	fn test_basefold_zk_channel_two_oracles() {
		type F = BinaryField128bGhash;
		type P = PackedBinaryGhash1x128b;

		let mut rng = StdRng::seed_from_u64(0);
		let n_vars_1 = 6;
		let n_vars_2 = 8;

		let (buffer_1, transparent_poly_1, eval_claim_1) =
			generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars_1);
		let (buffer_2, transparent_poly_2, eval_claim_2) =
			generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars_2);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

		let oracle_specs = vec![OracleSpec::new_zk(n_vars_1), OracleSpec::new_zk(n_vars_2)];

		let merkle_prover = make_merkle_prover();
		let verifier_compiler = BaseFoldZKVerifierCompiler::new(
			merkle_prover.scheme().clone(),
			oracle_specs,
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// === PROVER SIDE ===
		let ntt = make_ntt(verifier_compiler.max_subspace());
		let prover_compiler = BaseFoldZKProverCompiler::<P, _, _>::from_verifier_compiler(
			&verifier_compiler,
			ntt,
			merkle_prover,
		);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_rng = StdRng::seed_from_u64(1);
		let mut prover_channel = prover_compiler.create_channel(&mut prover_transcript, prover_rng);

		let oracle_1 = prover_channel.send_oracle(buffer_1.to_ref());
		let oracle_2 = prover_channel.send_oracle(buffer_2.to_ref());

		prover_channel.prove_oracle_relations([
			(oracle_1, buffer_1, transparent_poly_1.clone(), eval_claim_1),
			(oracle_2, buffer_2, transparent_poly_2.clone(), eval_claim_2),
		]);
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler.create_channel(&mut verifier_transcript);

		let v_oracle_1 = verifier_channel.recv_oracle().unwrap();
		let v_oracle_2 = verifier_channel.recv_oracle().unwrap();

		let tp1 = transparent_poly_1.clone();
		let tp2 = transparent_poly_2.clone();

		verifier_channel
			.verify_oracle_relations([
				OracleLinearRelation {
					oracle: v_oracle_1,
					transparent: Box::new(move |point: &[F]| {
						let eq = eq_ind_partial_eval::<P>(point);
						inner_product_buffers(&tp1, &eq)
					}),
					claim: eval_claim_1,
				},
				OracleLinearRelation {
					oracle: v_oracle_2,
					transparent: Box::new(move |point: &[F]| {
						let eq = eq_ind_partial_eval::<P>(point);
						inner_product_buffers(&tp2, &eq)
					}),
					claim: eval_claim_2,
				},
			])
			.unwrap();
		verifier_channel.finish().unwrap();
	}

	/// Runs a full prove/verify cycle of the Batched ZK BaseFold channel over oracles of the given
	/// sizes. If `tamper`, the verifier's claim on the first oracle is corrupted; verification must
	/// then fail. Returns whether verification accepted.
	fn run_zk_channel(n_vars_list: &[usize], tamper: bool) -> bool {
		type F = BinaryField128bGhash;
		type P = PackedBinaryGhash1x128b;

		let mut rng = StdRng::seed_from_u64(0);
		let data: Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> = n_vars_list
			.iter()
			.map(|&n| generate_zk_oracle_data::<F, P, _>(&mut rng, n))
			.collect();

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
		let oracle_specs: Vec<OracleSpec> =
			n_vars_list.iter().map(|&n| OracleSpec::new_zk(n)).collect();

		let merkle_prover = make_merkle_prover();
		let verifier_compiler = BaseFoldZKVerifierCompiler::new(
			merkle_prover.scheme().clone(),
			oracle_specs,
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// === PROVER SIDE ===
		let ntt = make_ntt(verifier_compiler.max_subspace());
		let prover_compiler = BaseFoldZKProverCompiler::<P, _, _>::from_verifier_compiler(
			&verifier_compiler,
			ntt,
			merkle_prover,
		);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_rng = StdRng::seed_from_u64(1);
		let mut prover_channel = prover_compiler.create_channel(&mut prover_transcript, prover_rng);

		let oracles: Vec<_> = data
			.iter()
			.map(|(buffer, _, _)| prover_channel.send_oracle(buffer.to_ref()))
			.collect();
		let prover_relations: Vec<_> = oracles
			.into_iter()
			.zip(&data)
			.map(|(oracle, (buffer, transparent, claim))| {
				(oracle, buffer.clone(), transparent.clone(), *claim)
			})
			.collect();
		prover_channel.prove_oracle_relations(prover_relations);
		prover_channel.finish();

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler.create_channel(&mut verifier_transcript);

		let v_oracles: Vec<_> = (0..n_vars_list.len())
			.map(|_| verifier_channel.recv_oracle().unwrap())
			.collect();
		let relations: Vec<_> = v_oracles
			.into_iter()
			.zip(&data)
			.enumerate()
			.map(|(i, (oracle, (_, transparent, claim)))| {
				let transparent = transparent.clone();
				let claim = if tamper && i == 0 {
					*claim + F::ONE
				} else {
					*claim
				};
				OracleLinearRelation {
					oracle,
					transparent: Box::new(move |point: &[F]| {
						let eq = eq_ind_partial_eval::<P>(point);
						inner_product_buffers(&transparent, &eq)
					}),
					claim,
				}
			})
			.collect();
		verifier_channel
			.verify_oracle_relations(relations)
			.expect("verify_oracle_relations only queues");
		verifier_channel.finish().is_ok()
	}

	#[test]
	fn test_basefold_zk_channel_three_oracles_non_power_of_two() {
		// 3 oracles (not a power of two) of unequal sizes: exercises oracle padding (Lifted FRI)
		// and the `⌈log 3⌉ = 2` outer oracle-combine rounds.
		assert!(run_zk_channel(&[5, 6, 8], false));
	}

	#[test]
	fn test_basefold_zk_channel_invalid_proof() {
		assert!(!run_zk_channel(&[6, 8], true));
	}
}
