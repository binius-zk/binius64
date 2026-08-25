// Copyright 2026 The Binius Developers

//! One committed Ligerito level, on the prover side.
//!
//! The counterpart of [`binius_iop::ligerito::LevelVerifier`], which describes the protocol.
//! Two facts recorded there drive everything here.
//! The MLE-check challenges are the lane fold.
//! And the residual is bound before the queries are drawn.

use binius_compute::{Allocator, GlobalAllocator};
use binius_field::{BinaryField, PackedField};
use binius_iop::ligerito::LigeritoLevel;
use binius_ip::mlecheck;
use binius_ip_prover::sumcheck::{
	common::MleCheckProver, multilinear_eval::multilinear_eval_prover,
};
use binius_math::{
	FieldBuffer, FieldSlice, multilinear::MultilinearMut, ntt::AdditiveNTT,
	reed_solomon::ReedSolomonCode,
};

use crate::{
	fri::{BrakedownOracleProver, ProxTestOracleProver},
	merkle_channel::MerkleIPProverChannel,
};

/// Proves one committed Ligerito level whose residual is sent in the clear.
///
/// Holds the level's shape alongside the oracle that answers queries against its codeword.
pub struct LevelProver<P: PackedField, C> {
	/// The level's shape: lane count, column count, rate, and query count.
	level: LigeritoLevel,
	/// The committed interleaved codeword, and the handle that opens it.
	oracle: BrakedownOracleProver<P, C>,
}

impl<F, P, C> LevelProver<P, C>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	/// Binds a prover to a level shape and the oracle over its committed codeword.
	///
	/// [`Self::commit`] is the way to build both consistently.
	pub const fn new(level: LigeritoLevel, oracle: BrakedownOracleProver<P, C>) -> Self {
		Self { level, oracle }
	}

	/// Encodes `message` lane by lane, Merkle-commits the result, and sends the commitment.
	///
	/// `message` is the level's multilinear in variable order, so its high `log_lanes` variables
	/// carry the lane index and its low `log_msg_cols` variables carry the column.
	/// That is already the layout [`ReedSolomonCode::encode_batch`] reads, which takes lane
	/// `lane`'s columns from
	///
	/// ```text
	///     message[(reverse_bits(lane, log_lanes) << log_msg_cols) | j]
	/// ```
	///
	/// as [`binius_math::ntt::subspace_polys`] pins.
	/// The bit reversal is what makes an opened coset fold by the MLE-check challenges in sampling
	/// order, which [`binius_iop::ligerito::LevelVerifier`] spells out.
	/// So no reshaping happens here, and a caller must not do any either.
	///
	/// One leaf is one codeword position across every lane, so a leaf is `2^log_lanes` scalars.
	///
	/// The codeword is allocated globally rather than from an arena.
	/// A level's codeword outlives the call that builds it, and nothing here is on a hot path yet.
	///
	/// ## Preconditions
	///
	/// * `message` has `level.log_msg_len()` variables.
	/// * `ntt` is defined over the level's codeword domain.
	pub fn commit<NTT, Channel>(
		level: LigeritoLevel,
		ntt: &NTT,
		message: FieldSlice<'_, P>,
		channel: &mut Channel,
	) -> Self
	where
		NTT: AdditiveNTT<Field = F> + Sync,
		Channel: MerkleIPProverChannel<F, Commitment = C>,
	{
		assert_eq!(
			message.log_len(),
			level.log_msg_len(),
			"precondition: message must have one variable per message variable of the level"
		);

		let code = ReedSolomonCode::<F>::new(level.log_msg_cols, level.log_inv_rate);
		let codeword = code.encode_batch(ntt, message, level.log_lanes, &GlobalAllocator);
		let commitment = channel.send_merkle_commitment(codeword.as_view(), 1 << level.log_lanes);

		Self::new(level, BrakedownOracleProver::new(codeword, commitment, 0))
	}

	/// Proves the opening of `<message, eq(eval_point)> = eval_claim`.
	///
	/// Runs `log_lanes` MLE-check rounds, folds the message by their challenges into the residual,
	/// commits and sends the residual, and only then answers the queries the channel draws.
	///
	/// The trailing sample matches the challenge the verifier batches the opened rows with.
	/// The prover has nothing to do with that value beyond keeping the two transcripts in step.
	///
	/// ## Preconditions
	///
	/// * `message` is the multilinear [`Self::commit`] committed.
	/// * `eval_point` has `level.log_msg_len()` coordinates, in low-to-high variable order.
	pub fn prove<A, Channel>(
		&self,
		message: FieldSlice<'_, P>,
		eval_point: &[F],
		eval_claim: F,
		alloc: &A,
		channel: &mut Channel,
	) where
		A: Allocator,
		Channel: MerkleIPProverChannel<F, Commitment = C>,
	{
		let n_vars = self.level.log_msg_len();
		assert_eq!(
			message.log_len(),
			n_vars,
			"precondition: message must have one variable per message variable of the level"
		);
		assert_eq!(
			eval_point.len(),
			n_vars,
			"precondition: eval_point must have one coordinate per message variable"
		);

		let mut sumcheck = multilinear_eval_prover(
			alloc,
			FieldBuffer::from_view_in(alloc, message),
			eval_point,
			eval_claim,
		);

		// Each round binds the message's highest remaining variable, so once the lane variables
		// are gone what is left of the message is the residual.
		let mut residual = FieldBuffer::from_view_in(alloc, message);
		for _ in 0..self.level.log_lanes {
			let round_coeffs = sumcheck
				.execute()
				.pop()
				.expect("the multilinear-evaluation prover proves exactly one claim");
			channel.send_many(mlecheck::RoundProof::truncate(round_coeffs).coeffs());

			let challenge = channel.sample();
			sumcheck.fold(challenge);
			residual.fold_highest_var(challenge);
		}

		// The residual is bound here, before a single query position exists.
		let residual_commitment = channel.send_merkle_commitment(residual.as_view(), 1);
		channel.send_committed_vector(&residual_commitment, residual.as_view());

		let indices = (0..self.level.n_queries)
			.map(|_| channel.sample_bits(self.level.log_codeword_len()))
			.collect::<Vec<_>>();
		self.oracle.open_queries(&indices, channel);

		// The verifier's row-batching challenge, drawn only to keep the two transcripts in step.
		let _batching_challenge: F = channel.sample();
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Ghash128b as B128, Random};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		ligerito::{Error, LevelVerifier, LigeritoParams},
		merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
		merkle_tree::BinaryMerkleTreeScheme,
		soundness::SoundnessRegime,
	};
	use binius_math::{
		multilinear::Multilinear,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::random_field_buffer,
	};
	use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::merkle_channel::ProverMerkleTranscriptChannel;

	type StdChallenger = HasherChallenger<StdDigest>;

	/// Commits `committed`, then proves and verifies `<opened, eq(z)> = opened(z) + claim_offset`.
	///
	/// Splitting the committed message from the opened one is how a dishonest prover is expressed
	/// here: every value it sends in the clear is consistent with `opened`, while the codeword the
	/// queries land in encodes `committed`.
	/// An honest run passes the same buffer twice and a zero offset.
	///
	/// Returns the finished proof's length in bytes, so a byte count is only ever taken from a
	/// transcript that convinced the verifier.
	fn run(
		level: LigeritoLevel,
		committed: &FieldBuffer<B128>,
		opened: &FieldBuffer<B128>,
		claim_offset: B128,
	) -> Result<usize, Error> {
		let mut rng = StdRng::seed_from_u64(level.log_msg_len() as u64);
		let eval_point = (0..level.log_msg_len())
			.map(|_| B128::random(&mut rng))
			.collect::<Vec<_>>();
		let eval_claim = opened.evaluate(&eval_point) + claim_offset;

		let ntt =
			NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(level.log_codeword_len()));
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel =
			ProverMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::new(
				&mut prover_transcript,
			);
		let prover = LevelProver::commit(level, &ntt, committed.as_view(), &mut prover_channel);
		prover.prove(
			opened.as_view(),
			&eval_point,
			eval_claim,
			&GlobalAllocator,
			&mut prover_channel,
		);
		prover_channel.into_transcript();

		let proof = prover_transcript.finalize();
		let proof_size = proof.len();

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let mut verifier_channel =
			VerifierMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::new(
				&mut verifier_transcript,
			);
		let commitment = verifier_channel
			.recv_merkle_commitment(1 << level.log_lanes, level.log_codeword_len())?;
		LevelVerifier::new(level, commitment).verify(
			&eval_point,
			eval_claim,
			&mut verifier_channel,
		)?;

		Ok(proof_size)
	}

	/// A random message of the level's shape.
	fn message(level: LigeritoLevel, seed: u64) -> FieldBuffer<B128> {
		random_field_buffer(&mut StdRng::seed_from_u64(seed), level.log_msg_len())
	}

	/// Every shape of level an honest prover can produce must verify.
	///
	/// The shapes vary all three dimensions independently, including `log_lanes = 0`, where the
	/// MLE-check runs no rounds at all and the residual is the whole message.
	#[test]
	fn an_honest_opening_verifies() {
		for log_msg_cols in 1..5 {
			for log_lanes in 0..4 {
				for log_inv_rate in 1..3 {
					let level = LigeritoLevel {
						log_msg_cols,
						log_lanes,
						log_inv_rate,
						n_queries: 5,
					};
					let msg = message(level, 0);
					run(level, &msg, &msg, B128::ZERO)
						.unwrap_or_else(|err| panic!("{level:?}: {err}"));
				}
			}
		}
	}

	/// A residual that does not fold out of the committed codeword must be caught.
	///
	/// This is the check the query round exists for.
	/// The prover's MLE-check, its residual, and its claim are all consistent with each other; only
	/// the codeword the queries open disagrees.
	#[test]
	fn a_residual_unrelated_to_the_codeword_is_rejected() {
		let level = LigeritoLevel {
			log_msg_cols: 3,
			log_lanes: 2,
			log_inv_rate: 2,
			n_queries: 5,
		};
		// Everything sent in the clear is consistent with the proved message, so the sumcheck half
		// passes. Only the codeword the queries land in disagrees, which is the query half alone.
		let err = run(level, &message(level, 0), &message(level, 1), B128::ZERO)
			.expect_err("the opened rows encode a different message than the residual");
		assert!(matches!(err, Error::IPChannel(binius_ip::channel::Error::InvalidAssert)));
	}

	/// A claim the message does not satisfy must be caught.
	///
	/// This is the check the MLE-check exists for: the reduced sum no longer matches the residual's
	/// own evaluation, even though every value the prover sent is internally consistent.
	#[test]
	fn a_claim_the_message_does_not_satisfy_is_rejected() {
		let level = LigeritoLevel {
			log_msg_cols: 3,
			log_lanes: 2,
			log_inv_rate: 2,
			n_queries: 5,
		};
		// The round proofs still fix the same challenges, so the fold and the query half agree.
		// Only the reduced sum is wrong, which is the sumcheck half alone.
		let msg = message(level, 0);
		let err =
			run(level, &msg, &msg, B128::ONE).expect_err("the evaluation claim is off by one");
		assert!(matches!(err, Error::IPChannel(binius_ip::channel::Error::InvalidAssert)));
	}
	/// The proof-size estimate must equal the transcript, not merely approximate it.
	///
	/// [`LigeritoParams::proof_size`] calls itself exact, and the ladder search minimizes it.
	/// An estimate that undercounts therefore picks the shape of a proof nobody produces.
	///
	/// The sweep moves all four dimensions of a level independently.
	/// `n_queries` is swept because it alone picks the Merkle layer the prover decommits at, and
	/// the estimate has to pick the same one.
	#[test]
	fn the_estimate_equals_the_proof_the_prover_writes() {
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();

		for log_msg_cols in 2..6 {
			for log_lanes in 0..4 {
				for log_inv_rate in 1..3 {
					for n_queries in [1, 3, 5, 12, 31] {
						let level = LigeritoLevel {
							log_msg_cols,
							log_lanes,
							log_inv_rate,
							n_queries,
						};
						// A level never opens more rows than its codeword has positions.
						if !level.is_feasible() {
							continue;
						}

						// The query count is pinned on the level, so the security target only has
						// to be one the constructor's feasibility check accepts.
						let params =
							LigeritoParams::new(vec![level], SoundnessRegime::UniqueDecoding, 8);
						let msg = message(level, 0);
						let written = run(level, &msg, &msg, B128::ZERO).unwrap_or_else(|err| {
							panic!("{level:?}: honest proof rejected: {err}")
						});

						assert_eq!(params.proof_size(&scheme), written, "{level:?}");
					}
				}
			}
		}
	}
}
