// Copyright 2026 The Binius Developers

//! The Ligerito opening protocol on the prover side.
//!
//! The counterpart of [`binius_iop::ligerito::LigeritoVerifier`], which describes the protocol.
//! Three facts recorded there drive everything here.
//! The sumcheck challenges of a level are its lane fold.
//! Whatever a level folds to is committed before that level's queries are drawn.
//! And only level 0 may use the MLE-check shortcut.
//!
//! A fourth fact is this side's alone.
//! The proof of work is paid at exactly the two points the verifier checks it.
//! That module's `Where the proof of work stands` section draws them.

use std::iter::zip;

use binius_compute::{Allocator, GlobalAllocator};
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_iop::ligerito::{LigeritoLevel, LigeritoParams};
use binius_ip::mlecheck;
use binius_ip_prover::sumcheck::{
	bivariate_product_evaluator::bivariate_product_prover,
	common::{MleCheckProver, SumcheckProver},
	multilinear_eval::multilinear_eval_prover,
};
use binius_math::{
	FieldBuffer, FieldSlice, FieldVec,
	inner_product::inner_product_packed,
	multilinear::{MultilinearMut, hypercube::Hypercube},
	ntt::AdditiveNTT,
	reed_solomon::ReedSolomonCode,
};

use super::induced_weight::InducedWeight;
use crate::{
	channel::grinding::GrindingProverChannel,
	fri::{BatchBrakedownOracleProver, BrakedownOracleProver, ProxTestOracleProver},
	merkle_channel::MerkleIPProverChannel,
};

/// Encodes `message` lane by lane, Merkle-commits the result, and sends the commitment.
///
/// `message` is the level's multilinear in variable order, so its high `log_lanes` variables carry
/// the lane index and its low `log_msg_cols` variables carry the column.
/// That is already the layout [`ReedSolomonCode::encode_batch`] reads, which takes lane `lane`'s
/// columns from
///
/// ```text
///     message[(reverse_bits(lane, log_lanes) << log_msg_cols) | j]
/// ```
///
/// as [`binius_math::ntt::subspace_polys`] pins.
/// The bit reversal is what makes an opened coset fold by the sumcheck challenges in sampling
/// order, which [`binius_iop::ligerito::LigeritoVerifier`] spells out.
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
/// * `ntt`'s domain covers the level's codeword domain.
pub(crate) fn commit_level<F, P, NTT, Channel>(
	level: &LigeritoLevel,
	ntt: &NTT,
	message: FieldSlice<'_, P>,
	channel: &mut Channel,
) -> BrakedownOracleProver<P, Channel::Commitment>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
{
	assert_eq!(
		message.log_len(),
		level.log_msg_len(),
		"precondition: message must have one variable per message variable of the level"
	);

	let code = ReedSolomonCode::<F>::new(level.log_msg_cols, level.log_inv_rate);
	// Named per level shape, so a profile attributes the ladder's encoding level by level.
	let codeword = tracing::debug_span!(
		"Encode level",
		log_msg_cols = level.log_msg_cols,
		log_lanes = level.log_lanes,
		log_inv_rate = level.log_inv_rate
	)
	.in_scope(|| code.encode_batch(ntt, message, level.log_lanes, &GlobalAllocator));
	let commitment = tracing::debug_span!("Merkle commit level")
		.in_scope(|| channel.send_merkle_commitment(codeword.as_view(), 1 << level.log_lanes));

	BrakedownOracleProver::new(codeword, commitment, 0)
}

/// Proves a Ligerito opening against a committed ladder of Reed-Solomon codewords.
///
/// Holds the ladder's shape, the transform its levels encode over, and level 0's oracles.
/// Deeper levels only exist once the folds above them have run, so [`Self::prove`] commits those.
pub struct LigeritoProver<'a, P: PackedField, C, NTT> {
	/// The ladder's shape, one [`LigeritoLevel`] per committed level.
	params: &'a LigeritoParams,
	/// The transform every level encodes over, sized for the largest of them.
	ntt: &'a NTT,
	/// Level 0's committed interleaved codewords, in the order their openings are written.
	oracles: BatchBrakedownOracleProver<P, C>,
}

impl<'a, F, P, C, NTT> LigeritoProver<'a, P, C, NTT>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
{
	/// Encodes and commits level 0, sending its Merkle root.
	///
	/// A single `ntt` serves the whole ladder.
	/// The levels encode over nested Gao-Mateer domains, so the largest of them covers them all.
	/// That is normally level 0, whose codeword is the longest, but a level that drops the rate
	/// without folding any lanes has a longer one still.
	///
	/// ## Preconditions
	///
	/// * `message` has `params.log_msg_len()` variables.
	/// * `ntt`'s domain covers every level's codeword domain.
	pub fn commit<Channel>(
		params: &'a LigeritoParams,
		ntt: &'a NTT,
		message: FieldSlice<'_, P>,
		channel: &mut Channel,
	) -> Self
	where
		Channel: MerkleIPProverChannel<F, Commitment = C>,
	{
		let level = &params.levels()[0];
		let oracle = commit_level(level, ntt, message, channel);
		Self::new(params, ntt, BatchBrakedownOracleProver::new(vec![oracle]))
	}

	/// Binds a prover to a ladder and the level-0 codewords its opening answers queries against.
	///
	/// The codewords must be listed in the order their commitments were sent.
	/// That is the order the verifier reads their openings in.
	///
	/// ## Preconditions
	///
	/// * Every codeword spans level 0's codeword length, so one query position addresses all.
	pub const fn new(
		params: &'a LigeritoParams,
		ntt: &'a NTT,
		oracles: BatchBrakedownOracleProver<P, C>,
	) -> Self {
		Self {
			params,
			ntt,
			oracles,
		}
	}

	/// The ladder's shape, one level per committed level.
	pub const fn params(&self) -> &LigeritoParams {
		self.params
	}

	/// Proves the opening of `<message, eq(eval_point)> = eval_claim`.
	///
	/// Per level: runs the fold rounds, commits what they folded to, and only then answers the
	/// queries the channel draws against that level's codeword.
	/// An intermediate level's query claim is glued into the running sumcheck at a fresh challenge;
	/// the last level's is left for the verifier to check against the cleartext residual.
	///
	/// The proof of work the parameters fix is ground before every fold challenge.
	/// One more grind stands before each level's queries.
	/// The verifier checks each of them at the point it was paid.
	///
	/// ## Preconditions
	///
	/// * `message` is the multilinear [`Self::commit`] committed.
	/// * `eval_point` has `params.log_msg_len()` coordinates, in low-to-high variable order.
	pub fn prove<A, Channel>(
		&self,
		message: FieldSlice<'_, P>,
		eval_point: &[F],
		eval_claim: F,
		alloc: &A,
		channel: &mut Channel,
	) where
		A: Allocator,
		Channel: MerkleIPProverChannel<F, Commitment = C, Word = Word> + GrindingProverChannel,
	{
		let levels = self.params.levels();
		let grinding = self.params.grinding();
		assert_eq!(
			message.log_len(),
			self.params.log_msg_len(),
			"precondition: message must have one variable per message variable of the ladder"
		);
		assert_eq!(
			eval_point.len(),
			self.params.log_msg_len(),
			"precondition: eval_point must have one coordinate per message variable"
		);

		// The message the current level commits, folded down the ladder into the residual.
		let mut current = FieldBuffer::from_view_in(alloc, message);
		// The running sumcheck weight, absent while the MLE-check carries it implicitly.
		let mut weight: Option<FieldVec<P, A>> = None;
		let mut sum = eval_claim;
		// Level 0's oracle lives in `self`; every deeper level's is built in this loop.
		let mut deeper: Option<BrakedownOracleProver<P, C>> = None;

		for (i, level) in levels.iter().enumerate() {
			let n_vars = level.log_msg_len();
			let _level_guard =
				tracing::debug_span!("Ligerito level", level = i, log_msg_len = n_vars).entered();
			let witness = FieldBuffer::from_view_in(alloc, current.as_view());

			let fold_guard =
				tracing::debug_span!("Fold rounds", n_rounds = level.log_lanes).entered();
			match weight.take() {
				None => {
					// Level 0's weight is the equality indicator, which the MLE-check factors out.
					// So it interpolates a degree-1 polynomial and truncates its constant term.
					let mut prover =
						multilinear_eval_prover(alloc, witness, &eval_point[..n_vars], sum);
					for _ in 0..level.log_lanes {
						let coeffs = prover.execute().pop().expect("the prover has one claim");
						channel.send_many(mlecheck::RoundProof::truncate(coeffs.clone()).coeffs());
						channel.grind(grinding.challenge_bits());
						let challenge = channel.sample();
						prover.fold(challenge);
						// The full round polynomial at the challenge is the reduced claim, which
						// is what the verifier recovers from the truncated one.
						sum = coeffs.evaluate(&challenge);
						current.fold_highest_var(challenge);
					}
				}
				Some(mut running) => {
					// A glued weight is not an equality indicator, so the round polynomial is the
					// full degree-2 product and the truncated coefficient is the leading one.
					let mut prover = bivariate_product_prover(
						alloc,
						[witness, FieldBuffer::from_view_in(alloc, running.as_view())],
						sum,
					);
					for _ in 0..level.log_lanes {
						let coeffs = prover.execute().pop().expect("the prover has one claim");
						channel.send_many(coeffs.clone().truncate().coeffs());
						channel.grind(grinding.challenge_bits());
						let challenge = channel.sample();
						prover.fold(challenge);
						sum = coeffs.evaluate(&challenge);
						current.fold_highest_var(challenge);
						running.fold_highest_var(challenge);
					}
					weight = Some(running);
				}
			}
			drop(fold_guard);

			// Whatever this level folded to is bound here, before a query position exists.
			let next = match levels.get(i + 1) {
				Some(next_level) => {
					Some(commit_level(next_level, self.ntt, current.as_view(), channel))
				}
				None => {
					let _residual_guard = tracing::debug_span!("Send residual").entered();
					let commitment = channel.send_merkle_commitment(current.as_view(), 1);
					channel.send_committed_vector(&commitment, current.as_view());
					None
				}
			};

			// The commitment above is fixed, and no position exists yet, which is where the
			// verifier checks this grind too.
			channel.grind(grinding.query_bits());

			let indices = (0..level.n_queries)
				.map(|_| channel.sample_bits(level.log_codeword_len()))
				.collect::<Vec<_>>();
			let query_guard =
				tracing::debug_span!("Open queries", n_queries = level.n_queries).entered();
			match deeper.as_ref() {
				// Level 0 writes every committed codeword's openings, one after another.
				None => self.oracles.open_queries(&indices, channel),
				// A deeper level has one codeword, committed by the level above it.
				Some(oracle) => oracle.open_queries(&indices, channel),
			}
			drop(query_guard);

			// The row-batching challenge.
			// The last level's is drawn only to keep the two transcripts in step, since its rows
			// meet the residual in the clear rather than a weight this prover has to build.
			let alpha: F = channel.sample();

			if next.is_some() {
				let _glue_guard = tracing::debug_span!("Glue induced weight").entered();
				let beta: F = channel.sample();
				let mut glued = InducedWeight::new(level, self.ntt, &indices, alpha).build(alloc);

				// The claim the opened rows make is the induced weight paired with the message
				// they folded to, which is what the verifier reads off as `enforced_sum`.
				sum += beta
					* inner_product_packed::<F, P>(
						level.log_msg_cols,
						glued.as_ref().iter().copied(),
						current.as_ref().iter().copied(),
					);

				// Level 0 leaves no running weight, having carried its equality indicator inside
				// the MLE-check, so it is materialized here at the unbound coordinates.
				let running = weight.take().unwrap_or_else(|| {
					Hypercube::One
						.expand(&eval_point[..level.log_msg_cols])
						.build_in(alloc)
				});
				let beta = P::broadcast(beta);
				for (entry, &carried) in zip(glued.as_mut(), running.as_ref()) {
					*entry = carried + beta * *entry;
				}
				weight = Some(glued);
			}
			deeper = next;
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{
		cell::Cell,
		iter::{Product, Sum},
		ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
	};

	use binius_field::{
		ExtensionField, Field, Ghash128b as B128, Random,
		arithmetic_traits::{InvertOrZero, Square},
		field::FieldOps,
	};
	use binius_hash::{
		CompressionFunction, ParallelCompressionAdaptor, StdDigest, StdHashSuite,
		binary_merkle_tree::HashSuite,
	};
	use binius_iop::{
		channel::grinding::GrindingVerifierChannel,
		ligerito::{Error, LigeritoVerifier, VerifierCost},
		merkle_channel::{self, MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
		merkle_tree::{self, BinaryMerkleTreeScheme},
		soundness::{Grinding, SoundnessRegime},
	};
	use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
	use binius_math::{
		multilinear::Multilinear,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::random_field_buffer,
	};
	use binius_transcript::{
		Error as TranscriptError, ProverTranscript, VerifierTranscript,
		fiat_shamir::HasherChallenger,
	};
	use digest::Output;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::merkle_channel::ProverMerkleTranscriptChannel;

	type StdChallenger = HasherChallenger<StdDigest>;
	type StdLeafHash = <StdHashSuite as HashSuite>::LeafHash;

	/// A ladder whose level `i` commits at inverse rate `2^(i + 1)` and opens `n_queries` rows.
	///
	/// `lanes[i]` is level `i`'s fold amount, and `log_msg_cols` is level 0's column count.
	/// Level `i + 1` takes what level `i` folds to, so its columns are `cols_i - lanes_{i+1}`.
	fn ladder(log_msg_cols: usize, lanes: &[usize], n_queries: usize) -> LigeritoParams {
		let mut log_msg_cols = log_msg_cols;
		let levels = lanes
			.iter()
			.enumerate()
			.map(|(i, &log_lanes)| {
				// Level 0 keeps the column count it was given; every deeper one loses its lanes.
				if i > 0 {
					log_msg_cols -= log_lanes;
				}
				LigeritoLevel {
					log_msg_cols,
					log_lanes,
					log_inv_rate: i + 1,
					n_queries,
				}
			})
			.collect();
		LigeritoParams::new(levels, SoundnessRegime::default(), 32)
	}

	/// Commits `committed`, then proves `<opened, eq(z)> = opened(z) + claim_offset`.
	///
	/// Splitting the committed message from the opened one is how a dishonest prover is expressed
	/// here: every value it sends in the clear is consistent with `opened`, while the codeword
	/// level 0's queries land in encodes `committed`.
	/// An honest run passes the same buffer twice and a zero offset.
	///
	/// Returns the finished transcript alongside the point and claim it opens at.
	/// A verifier can then be pointed at it through whichever channel a test wants to watch.
	fn write_proof(
		params: &LigeritoParams,
		committed: &FieldBuffer<B128>,
		opened: &FieldBuffer<B128>,
		claim_offset: B128,
	) -> (Vec<u8>, Vec<B128>, B128) {
		let n_vars = params.log_msg_len();
		let mut rng = StdRng::seed_from_u64(n_vars as u64);
		let eval_point = (0..n_vars)
			.map(|_| B128::random(&mut rng))
			.collect::<Vec<_>>();
		let eval_claim = opened.evaluate(&eval_point) + claim_offset;

		// One transform for the whole ladder, sized for its longest codeword.
		let ntt = NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(
			params.max_log_codeword_len(),
		));

		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let mut channel =
			ProverMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::new(
				&mut transcript,
			);
		let prover = LigeritoProver::commit(params, &ntt, committed.as_view(), &mut channel);
		prover.prove(opened.as_view(), &eval_point, eval_claim, &GlobalAllocator, &mut channel);
		channel.into_transcript();

		(transcript.finalize(), eval_point, eval_claim)
	}

	/// Verifies a written proof against `params`, returning its length in bytes.
	///
	/// The ladder is taken from `params` rather than from the transcript, so a verifier can be
	/// pointed at a proof written under different parameters.
	fn verify_proof(
		params: &LigeritoParams,
		proof: Vec<u8>,
		eval_point: &[B128],
		eval_claim: B128,
	) -> Result<usize, Error> {
		let proof_size = proof.len();

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let mut verifier_channel =
			VerifierMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::new(
				&mut verifier_transcript,
			);
		let level = &params.levels()[0];
		let commitment = verifier_channel
			.recv_merkle_commitment(1 << level.log_lanes, level.log_codeword_len())?;
		LigeritoVerifier::new(params, commitment).verify(
			eval_point,
			eval_claim,
			&mut verifier_channel,
		)?;

		Ok(proof_size)
	}

	/// Proves and verifies one opening, returning the finished proof's length in bytes.
	///
	/// The byte count is only ever taken from a transcript that convinced the verifier.
	fn run(
		params: &LigeritoParams,
		committed: &FieldBuffer<B128>,
		opened: &FieldBuffer<B128>,
		claim_offset: B128,
	) -> Result<usize, Error> {
		let (proof, eval_point, eval_claim) = write_proof(params, committed, opened, claim_offset);
		verify_proof(params, proof, &eval_point, eval_claim)
	}

	/// Proves an honest opening under one ladder and verifies it under another.
	///
	/// The two ladders are the same shape and differ only in the proof of work they pay, which is
	/// how a grind checked in the wrong place is expressed here.
	fn run_across(
		prover_params: &LigeritoParams,
		verifier_params: &LigeritoParams,
	) -> Result<usize, Error> {
		let msg = message(prover_params, 0);
		let (proof, eval_point, eval_claim) = write_proof(prover_params, &msg, &msg, B128::ZERO);
		verify_proof(verifier_params, proof, &eval_point, eval_claim)
	}

	/// A random message of the ladder's shape.
	fn message(params: &LigeritoParams, seed: u64) -> FieldBuffer<B128> {
		random_field_buffer(&mut StdRng::seed_from_u64(seed), params.log_msg_len())
	}

	/// Every ladder an honest prover can produce must verify.
	///
	/// The shapes vary the depth, the fold amounts and the column count independently.
	/// One level is the terminal case, where no claim is ever glued and the whole protocol is the
	/// MLE-check plus one query round.
	/// A level folding no lanes is the degenerate case, where a level recommits at a lower rate
	/// without reducing anything.
	#[test]
	fn an_honest_opening_verifies() {
		let shapes: &[(usize, &[usize])] = &[
			(3, &[1]),
			(4, &[2]),
			(4, &[1, 1]),
			(5, &[2, 1]),
			(6, &[2, 2]),
			(6, &[1, 1, 1]),
			(6, &[2, 2, 2]),
			(7, &[3, 1, 1, 1]),
			(4, &[2, 0, 1]),
		];
		for &(log_msg_cols, lanes) in shapes {
			let params = ladder(log_msg_cols, lanes, 5);
			let msg = message(&params, 0);
			run(&params, &msg, &msg, B128::ZERO)
				.unwrap_or_else(|err| panic!("{log_msg_cols} {lanes:?}: {err}"));
		}
	}

	/// A codeword that does not encode the message the prover reduced must be caught.
	///
	/// This is the check the query round exists for, at one level.
	/// The MLE-check, the residual and the claim are all consistent with each other; only the
	/// codeword the queries open disagrees, and the terminal check pairs it against the residual.
	#[test]
	fn a_codeword_unrelated_to_the_residual_is_rejected() {
		let params = ladder(3, &[2], 5);
		let err = run(&params, &message(&params, 0), &message(&params, 1), B128::ZERO)
			.expect_err("the opened rows encode a different message than the residual");
		assert!(matches!(err, Error::IPChannel(binius_ip::channel::Error::InvalidAssert)));
	}

	/// The same mismatch, deep enough that the glue is what has to catch it.
	///
	/// Level 0's rows no longer meet a message in the clear.
	/// Their claim is folded into the running sumcheck at a challenge the prover cannot predict,
	/// and only the final pairing against the residual sees it.
	#[test]
	fn a_glued_row_claim_that_does_not_hold_is_rejected() {
		let params = ladder(6, &[2, 2, 2], 5);
		let err = run(&params, &message(&params, 0), &message(&params, 1), B128::ZERO)
			.expect_err("level 0's rows encode a different message than the ladder reduced");
		assert!(matches!(err, Error::IPChannel(binius_ip::channel::Error::InvalidAssert)));
	}
	/// A claim the message does not satisfy must be caught, at every depth.
	///
	/// This is the check the sumcheck exists for: the reduced sum no longer matches the residual
	/// paired with the accumulated weight, even though every value the prover sent is internally
	/// consistent.
	#[test]
	fn a_claim_the_message_does_not_satisfy_is_rejected() {
		for lanes in [&[2usize] as &[usize], &[2, 2], &[2, 2, 2]] {
			let params = ladder(6, lanes, 5);
			let msg = message(&params, 0);
			let err = run(&params, &msg, &msg, B128::ONE)
				.expect_err("the evaluation claim is off by one");
			assert!(matches!(err, Error::IPChannel(binius_ip::channel::Error::InvalidAssert)));
		}
	}

	// Proof of work

	/// A ladder that grinds must still verify, at every depth and on either side of the split.
	#[test]
	fn an_honest_opening_with_a_proof_of_work_verifies() {
		// Fixture state: the two grinds are exercised alone and together, and zero is included so
		// the grinding path and the ungrinding one are held to the same ladder.
		let grinds = [
			Grinding::NONE,
			Grinding::new(0, 4),
			Grinding::new(4, 0),
			Grinding::new(3, 5),
		];
		for grinding in grinds {
			for lanes in [&[2usize] as &[usize], &[2, 2], &[1, 2, 1]] {
				let params = ladder(6, lanes, 5).with_grinding(grinding);
				let msg = message(&params, 0);
				run(&params, &msg, &msg, B128::ZERO)
					.unwrap_or_else(|err| panic!("{lanes:?} {grinding:?}: {err}"));
			}
		}
	}

	/// Grinding lengthens the proof by one nonce per grind, and by nothing else.
	#[test]
	fn a_grind_costs_one_nonce_and_the_estimate_counts_it() {
		// Invariant: the two call sites are the only thing grinding adds to a transcript. One
		// nonce stands before each fold challenge, one more before each level's queries, and a
		// difficulty of zero writes nothing at all.
		//
		// Fixture state: ladders of three shapes, priced bare and then at each of the three
		// grinding profiles.
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		for lanes in [&[2usize] as &[usize], &[2, 2], &[1, 2, 1]] {
			let bare = ladder(6, lanes, 5);
			let msg = message(&bare, 0);
			let bare_size = run(&bare, &msg, &msg, B128::ZERO).expect("honest proof rejected");
			assert_eq!(bare.proof_size(&scheme), bare_size);

			for grinding in [
				Grinding::new(0, 4),
				Grinding::new(4, 0),
				Grinding::new(3, 5),
			] {
				let ground = bare.clone().with_grinding(grinding);
				let ground_size =
					run(&ground, &msg, &msg, B128::ZERO).expect("honest proof rejected");

				// One `u64` per nonce, and the ladder says how many nonces it writes.
				let nonces = ground
					.levels()
					.iter()
					.map(|level| level.n_grind_nonces(grinding))
					.sum::<usize>();
				assert_eq!(
					ground_size - bare_size,
					nonces * size_of::<u64>(),
					"{lanes:?} {grinding:?}"
				);
				// Which is also what the estimate says, so the ladder search prices a real proof.
				assert_eq!(ground.proof_size(&scheme), ground_size, "{lanes:?} {grinding:?}");
			}

			// A query grind writes one nonce per level; a challenge grind writes one per fold
			// round. So the two profiles differ in length whenever a level folds more than one
			// lane, which is what keeps them from being confused for one another by size alone.
			let query_only = bare
				.levels()
				.iter()
				.map(|level| level.n_grind_nonces(Grinding::new(0, 4)))
				.sum::<usize>();
			let challenge_only = bare
				.levels()
				.iter()
				.map(|level| level.n_grind_nonces(Grinding::new(4, 0)))
				.sum::<usize>();
			assert_eq!(query_only, bare.n_levels());
			assert_eq!(challenge_only, bare.n_fold_rounds());
		}
	}

	/// A verifier expecting a different amount of work must reject the proof.
	#[test]
	fn a_proof_ground_at_one_difficulty_is_rejected_at_another() {
		// Invariant: the difficulty is part of the statement, not advice the prover carries. The
		// sampler reads four bytes whatever the difficulty, so a mismatch of nonzero difficulties
		// leaves both transcripts in step and shows up only as unmet work.
		//
		// Fixture state: four bits are ground and twenty-four are demanded. The gap is what makes
		// the rejection certain rather than a coin flip: a nonce that happens to satisfy twenty
		// more bits than it was searched for turns up once in a million.
		let base = ladder(6, &[2, 2], 5);
		let ground = base.clone().with_grinding(Grinding::new(4, 0));
		let demanded = base.clone().with_grinding(Grinding::new(24, 0));
		let err = run_across(&ground, &demanded)
			.expect_err("four bits of work cannot satisfy a twenty-four bit demand");
		let Error::ProofOfWork(TranscriptError::InsufficientWork { bits, sampled }) = err else {
			panic!("expected unmet proof of work, got {err}")
		};
		assert_eq!(bits, 24);
		assert_ne!(sampled, 0);

		// The same on the query side, where the nonce stands after the commitment instead.
		let ground = base.clone().with_grinding(Grinding::new(0, 4));
		let demanded = base.clone().with_grinding(Grinding::new(0, 24));
		let err = run_across(&ground, &demanded)
			.expect_err("four bits of work cannot satisfy a twenty-four bit demand");
		let Error::ProofOfWork(TranscriptError::InsufficientWork { bits, sampled }) = err else {
			panic!("expected unmet proof of work, got {err}")
		};
		assert_eq!(bits, 24);
		assert_ne!(sampled, 0);

		// And the other way round: work the verifier never asks for is eight stray bytes on the
		// tape. The two sides then read different things at every offset below, and the residual
		// is the first value checked against a commitment, so that is where it breaks.
		let bare = base.clone();
		let ground = base.with_grinding(Grinding::new(4, 4));
		let err = run_across(&ground, &bare)
			.expect_err("a nonce the verifier never reads desynchronizes the transcript");
		let Error::Channel(merkle_channel::Error::MerkleTree(merkle_tree::Error::Verification(
			merkle_tree::VerificationError::InvalidProof,
		))) = err
		else {
			panic!("expected a Merkle verification failure, got {err}")
		};
	}

	/// Work paid at one call site must not settle the debt at the other.
	#[test]
	fn a_challenge_grind_is_not_accepted_as_a_query_grind() {
		// Invariant: the two grinds buy different terms of the security budget, so the transcript
		// has to tell them apart by *where* the nonce stands rather than by how the tape looks.
		//
		// The two ladders below swap the difficulties between the sites. Both write one nonce
		// before the fold challenge and one before the queries, both of the same width, so the
		// two transcripts are laid out byte for byte alike and stay in step to the very end.
		// Nothing but the position separates them. A grind checked at the wrong site would
		// therefore be checked against the *other* site's work and pass.
		//
		// Fixture state: one level folding one lane, so each site carries exactly one nonce. The
		// difficulties differ by sixteen bits, which is what makes the rejection certain rather
		// than a coin flip.
		let base = ladder(6, &[1], 5);
		let heavy_fold = base.clone().with_grinding(Grinding::new(18, 2));
		let heavy_query = base.with_grinding(Grinding::new(2, 18));
		assert_eq!(
			heavy_fold.levels()[0].n_grind_nonces(heavy_fold.grinding()),
			heavy_query.levels()[0].n_grind_nonces(heavy_query.grinding()),
		);

		// Eighteen bits before the fold challenge do not pay for the query positions.
		let err = run_across(&heavy_fold, &heavy_query)
			.expect_err("work before the fold challenge does not pay for the query positions");
		let Error::ProofOfWork(TranscriptError::InsufficientWork { bits, sampled }) = err else {
			panic!("expected unmet proof of work, got {err}")
		};
		assert_eq!(bits, 18);
		assert_ne!(sampled, 0);

		// And eighteen bits before the query positions do not pay for the fold challenge.
		let err = run_across(&heavy_query, &heavy_fold)
			.expect_err("work before the query positions does not pay for the fold challenge");
		let Error::ProofOfWork(TranscriptError::InsufficientWork { bits, sampled }) = err else {
			panic!("expected unmet proof of work, got {err}")
		};
		assert_eq!(bits, 18);
		assert_ne!(sampled, 0);
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
						let msg = message(&params, 0);
						let written = run(&params, &msg, &msg, B128::ZERO).unwrap_or_else(|err| {
							panic!("{level:?}: honest proof rejected: {err}")
						});

						assert_eq!(params.proof_size(&scheme), written, "{level:?}");
					}
				}
			}
		}
	}
	/// The estimate must stay exact once the ladder recurses.
	///
	/// Level 0 discounts its round message and no deeper level does.
	/// A one-level sweep cannot tell those two prices apart, so this walks real ladders.
	/// It is the only thing that measures [`PRODUCT_DEGREE`] rather than arguing for it.
	#[test]
	fn the_estimate_stays_exact_down_the_ladder() {
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();

		for lanes in [&[2usize] as &[usize], &[2, 2], &[1, 2, 1], &[2, 2, 2]] {
			for n_queries in [1, 5, 12] {
				let params = ladder(6, lanes, n_queries);
				let msg = message(&params, 0);
				let written = run(&params, &msg, &msg, B128::ZERO)
					.unwrap_or_else(|err| panic!("{lanes:?}: honest proof rejected: {err}"));

				assert_eq!(params.proof_size(&scheme), written, "{lanes:?} n_queries={n_queries}");
			}
		}
	}

	// Counting what the verifier spends

	thread_local! {
		/// Field multiplications the verifier on this thread has performed.
		static FIELD_MULS: Cell<usize> = const { Cell::new(0) };
		/// Two-to-one Merkle compressions the verifier on this thread has performed.
		static COMPRESSIONS: Cell<usize> = const { Cell::new(0) };
	}

	/// A field element that counts its own multiplications.
	///
	/// Every arithmetic helper the verifier reaches for is generic over the field operations.
	/// So substituting this for the concrete field counts multiplications wherever they happen.
	///
	/// - Inside the induced basis, where the closed form is evaluated.
	/// - Inside the sumcheck round recovery.
	/// - Inside the residual's evaluation and pairing.
	///
	/// Inversion is refused rather than implemented.
	/// A verifier compiled into a circuit cannot invert a value that depends on the proof.
	/// A test that silently allowed one would not notice the day it appeared.
	///
	/// This is not a packed field, but that buys nothing the ordinary build does not already give.
	/// `verify` is generic over an element type bounded by `FieldOps + From<F>`, and `to_dense`
	/// requires `PackedField`, so a call to it fails to compile at `verify`'s own definition.
	/// The succinct route is enforced every time the crate builds, instantiation or not.
	#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
	struct Counted(B128);

	impl Counted {
		/// Runs the verification with the counter zeroed, and reports its value afterwards.
		fn measure<T>(f: impl FnOnce() -> T) -> (T, usize) {
			FIELD_MULS.with(|muls| muls.set(0));
			let out = f();
			(out, FIELD_MULS.with(Cell::get))
		}

		/// The counter's value right now, for a mark taken part way through a run.
		fn so_far() -> usize {
			FIELD_MULS.with(Cell::get)
		}

		/// Multiplies, charging one multiplication.
		fn multiply(self, other: Self) -> Self {
			FIELD_MULS.with(|muls| muls.set(muls.get() + 1));
			Self(self.0 * other.0)
		}
	}

	impl From<B128> for Counted {
		fn from(value: B128) -> Self {
			Self(value)
		}
	}

	impl Neg for Counted {
		type Output = Self;

		fn neg(self) -> Self {
			Self(-self.0)
		}
	}

	impl Add for Counted {
		type Output = Self;

		fn add(self, other: Self) -> Self {
			Self(self.0 + other.0)
		}
	}

	impl Sub for Counted {
		type Output = Self;

		fn sub(self, other: Self) -> Self {
			Self(self.0 - other.0)
		}
	}

	impl Mul for Counted {
		type Output = Self;

		fn mul(self, other: Self) -> Self {
			self.multiply(other)
		}
	}

	impl Add<&Self> for Counted {
		type Output = Self;

		fn add(self, other: &Self) -> Self {
			self + *other
		}
	}

	impl Sub<&Self> for Counted {
		type Output = Self;

		fn sub(self, other: &Self) -> Self {
			self - *other
		}
	}

	impl Mul<&Self> for Counted {
		type Output = Self;

		fn mul(self, other: &Self) -> Self {
			self.multiply(*other)
		}
	}

	impl AddAssign for Counted {
		fn add_assign(&mut self, other: Self) {
			*self = *self + other;
		}
	}

	impl SubAssign for Counted {
		fn sub_assign(&mut self, other: Self) {
			*self = *self - other;
		}
	}

	impl MulAssign for Counted {
		fn mul_assign(&mut self, other: Self) {
			*self = self.multiply(other);
		}
	}

	impl AddAssign<&Self> for Counted {
		fn add_assign(&mut self, other: &Self) {
			*self = *self + *other;
		}
	}

	impl SubAssign<&Self> for Counted {
		fn sub_assign(&mut self, other: &Self) {
			*self = *self - *other;
		}
	}

	impl MulAssign<&Self> for Counted {
		fn mul_assign(&mut self, other: &Self) {
			*self = self.multiply(*other);
		}
	}

	impl Sum for Counted {
		fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
			// Addition is a bitwise exclusive or in characteristic two, so a sum charges nothing.
			iter.fold(Self(B128::ZERO), |acc, item| acc + item)
		}
	}

	impl<'a> Sum<&'a Self> for Counted {
		fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
			iter.copied().sum()
		}
	}

	impl Product for Counted {
		fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
			// Folding through the counting multiply is what charges `n - 1` for `n` factors.
			iter.fold(Self(B128::ONE), Self::multiply)
		}
	}

	impl<'a> Product<&'a Self> for Counted {
		fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
			iter.copied().product()
		}
	}

	impl Square for Counted {
		fn square(self) -> Self {
			self.multiply(self)
		}
	}

	impl InvertOrZero for Counted {
		fn invert_or_zero(self) -> Self {
			panic!("the verifier must never invert a value that depends on the proof")
		}
	}

	impl FieldOps for Counted {
		type Scalar = B128;

		fn zero() -> Self {
			Self(B128::ZERO)
		}

		fn one() -> Self {
			Self(B128::ONE)
		}

		fn square_transpose<FSub: Field>(elems: &mut [Self])
		where
			B128: ExtensionField<FSub>,
		{
			let mut raw = elems.iter().map(|elem| elem.0).collect::<Vec<_>>();
			<B128 as FieldOps>::square_transpose::<FSub>(&mut raw);
			for (elem, value) in zip(elems, raw) {
				*elem = Self(value);
			}
		}
	}

	/// A two-to-one compression that counts its calls before delegating.
	#[derive(Debug, Clone, Default)]
	struct CountingCompression(<StdHashSuite as HashSuite>::Compression);

	impl CompressionFunction<Output<StdLeafHash>, 2> for CountingCompression {
		fn compress(&self, input: [Output<StdLeafHash>; 2]) -> Output<StdLeafHash> {
			COMPRESSIONS.with(|count| count.set(count.get() + 1));
			self.0.compress(input)
		}
	}

	/// The shipped hash suite with its compression function counted.
	///
	/// The leaf hash is left alone.
	/// So a transcript written under the shipped suite verifies under this one unchanged.
	#[derive(Debug)]
	struct CountingHashSuite;

	impl HashSuite for CountingHashSuite {
		type LeafHash = StdLeafHash;
		type Compression = CountingCompression;
		type ParLeafHash = <StdHashSuite as HashSuite>::ParLeafHash;
		type ParCompression = ParallelCompressionAdaptor<CountingCompression>;
	}

	/// Wraps a verifier channel, carrying counting elements and tallying what it is asked for.
	///
	/// The wrapper sees what a circuit-building channel would have to emit gates for.
	///
	/// - The bit decompositions of a query index.
	/// - The leaves whose digests have to be rebuilt.
	struct CountingChannel<C> {
		inner: C,
		/// Bit-decomposition sums performed over fixed constants.
		subset_sums: usize,
		/// Merkle leaves the verifier handed back to the scheme to re-hash.
		leaves_rebuilt: usize,
		/// The multiplication count at the moment each level's rows were opened.
		marks: Vec<usize>,
	}

	impl<C> CountingChannel<C> {
		fn new(inner: C) -> Self {
			Self {
				inner,
				subset_sums: 0,
				leaves_rebuilt: 0,
				marks: Vec::new(),
			}
		}
	}

	impl<C> IPVerifierChannel<B128> for CountingChannel<C>
	where
		C: IPVerifierChannel<B128, Elem = B128>,
	{
		type Elem = Counted;

		fn recv_one(&mut self) -> Result<Counted, binius_ip::channel::Error> {
			self.inner.recv_one().map(Counted)
		}

		fn sample(&mut self) -> Counted {
			Counted(self.inner.sample())
		}

		fn observe_one(&mut self, val: B128) -> Counted {
			Counted(self.inner.observe_one(val))
		}

		fn assert_zero(&mut self, val: Counted) -> Result<(), binius_ip::channel::Error> {
			self.inner.assert_zero(val.0)
		}
	}

	impl<C> WordIPVerifierChannel<B128> for CountingChannel<C>
	where
		C: WordIPVerifierChannel<B128, Elem = B128>,
	{
		type Word = C::Word;

		fn observe_words(&mut self, words: &[Word]) -> Vec<Self::Word> {
			self.inner.observe_words(words)
		}

		fn subset_sum(&mut self, elems: &[Counted], word: &Self::Word) -> Counted {
			// One call is one bit decomposition, whatever the constants behind it.
			self.subset_sums += 1;
			let raw = elems.iter().map(|elem| elem.0).collect::<Vec<_>>();
			Counted(self.inner.subset_sum(&raw, word))
		}

		fn select(&mut self, elems: &[Counted], word: &Self::Word) -> Counted {
			let raw = elems.iter().map(|elem| elem.0).collect::<Vec<_>>();
			Counted(self.inner.select(&raw, word))
		}

		fn sample_bits(&mut self, bits: usize) -> Self::Word {
			self.inner.sample_bits(bits)
		}

		fn pack_words(&mut self, words: &[Self::Word]) -> Vec<Counted> {
			self.inner
				.pack_words(words)
				.into_iter()
				.map(Counted)
				.collect()
		}
	}

	impl<C: GrindingVerifierChannel> GrindingVerifierChannel for CountingChannel<C> {
		fn verify_grind(&mut self, bits: usize) -> Result<(), binius_transcript::Error> {
			self.inner.verify_grind(bits)
		}
	}

	impl<C> MerkleIPVerifierChannel<B128> for CountingChannel<C>
	where
		C: MerkleIPVerifierChannel<B128, Elem = B128>,
	{
		type Commitment = C::Commitment;

		fn recv_merkle_commitment(
			&mut self,
			leaf_size: usize,
			depth: usize,
		) -> Result<Self::Commitment, binius_iop::merkle_channel::Error> {
			self.inner.recv_merkle_commitment(leaf_size, depth)
		}

		fn recv_openings(
			&mut self,
			commitment: &Self::Commitment,
			indices: &[Self::Word],
		) -> Result<Vec<Counted>, binius_iop::merkle_channel::Error> {
			// Exactly one call per level, which makes this the boundary the cost table is cut at.
			self.marks.push(Counted::so_far());
			// Every opened row is one leaf digest recomputed from the revealed values.
			self.leaves_rebuilt += indices.len();
			Ok(self
				.inner
				.recv_openings(commitment, indices)?
				.into_iter()
				.map(Counted)
				.collect())
		}

		fn recv_committed_vector(
			&mut self,
			commitment: &Self::Commitment,
		) -> Result<Vec<Counted>, binius_iop::merkle_channel::Error> {
			let values = self.inner.recv_committed_vector(commitment)?;
			// The residual is committed one element to a leaf, so its whole tree is rebuilt.
			self.leaves_rebuilt += values.len();
			Ok(values.into_iter().map(Counted).collect())
		}
	}

	/// What the verifier spent, as counted rather than as predicted by the cost model.
	struct Measured {
		/// Field multiplications over the whole verification.
		field_muls: usize,
		/// The multiplication count at each level's query round, in ladder order.
		marks: Vec<usize>,
		/// Merkle leaves rebuilt.
		leaf_hashes: usize,
		/// Merkle compressions performed.
		compressions: usize,
		/// Bit decompositions of a query index.
		subset_sums: usize,
	}

	/// Verifies an honest opening through the counting element type and the counting hash suite.
	///
	/// The prover runs first, under the shipped suite, so nothing it does reaches an instrument.
	fn measure(params: &LigeritoParams) -> Result<Measured, Error> {
		let msg = message(params, 0);
		let (proof, eval_point, eval_claim) = write_proof(params, &msg, &msg, B128::ZERO);

		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let inner =
			VerifierMerkleTranscriptChannel::<_, StdChallenger, B128, CountingHashSuite>::new(
				&mut transcript,
			);
		let mut channel = CountingChannel::new(inner);

		let level = &params.levels()[0];
		let commitment =
			channel.recv_merkle_commitment(1 << level.log_lanes, level.log_codeword_len())?;
		let point = eval_point.into_iter().map(Counted).collect::<Vec<_>>();
		let verifier = LigeritoVerifier::new(params, commitment);

		COMPRESSIONS.with(|count| count.set(0));
		let (verified, field_muls) = Counted::measure(|| {
			verifier.verify::<B128, _>(&point, Counted(eval_claim), &mut channel)
		});
		verified?;

		Ok(Measured {
			field_muls,
			marks: channel.marks.clone(),
			leaf_hashes: channel.leaves_rebuilt,
			compressions: COMPRESSIONS.with(Cell::get),
			subset_sums: channel.subset_sums,
		})
	}

	/// The multiplications charged between one level's query round and the next one's.
	///
	/// The tail entry runs from the last level's query round to the end of the verification.
	/// So it carries the closing checks against the cleartext residual.
	fn segments(measured: &Measured) -> Vec<usize> {
		measured
			.marks
			.iter()
			.skip(1)
			.chain(std::iter::once(&measured.field_muls))
			.zip(&measured.marks)
			.map(|(end, start)| end - start)
			.collect()
	}

	/// The model prices a ladder without running it, so a real run has to agree with it.
	#[test]
	fn the_measured_costs_are_the_ones_the_model_predicts() {
		// Invariant: the cost table prices a ladder without running it, so what it says has to be
		// what a real verification does. All three of its columns are observable: the leaves
		// handed back for re-hashing, the compressions the scheme performs, and the bit
		// decompositions a query index drives.
		//
		// Fixture state: ladders moving depth, fold amounts, column count and query count
		// independently, since the table's layer-depth term turns on the query count alone.
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		let shapes: &[(usize, &[usize], usize)] = &[
			(4, &[2], 1),
			(4, &[2], 5),
			(6, &[2, 2], 8),
			(6, &[1, 1, 1], 3),
			(8, &[2, 2, 2], 12),
			(7, &[3, 1, 1, 1], 5),
		];

		for &(log_msg_cols, lanes, n_queries) in shapes {
			let params = ladder(log_msg_cols, lanes, n_queries);
			let measured = measure(&params).unwrap_or_else(|err| {
				panic!("{log_msg_cols} {lanes:?}: honest proof rejected: {err}")
			});

			assert_eq!(
				VerifierCost {
					leaf_hashes: measured.leaf_hashes,
					compressions: measured.compressions,
					subset_sums: measured.subset_sums,
				},
				VerifierCost::total(&params.verifier_cost(&scheme)),
				"{log_msg_cols} {lanes:?} n_queries={n_queries}"
			);
		}
	}

	/// A message sixteen times longer costs the verifier a little more, not sixteen times more.
	#[test]
	fn a_longer_message_costs_a_logarithm_rather_than_a_factor() {
		// Invariant: the closed-form basis keeps the verifier's work logarithmic in the message.
		// Whether an opening verifies says nothing about which route produced it, so the growth
		// rate is the only thing that separates the two.
		//
		// Fixture state: two ladders reduced to the same 2^4 residual at the same query count.
		//
		//     small: cols_0 = 8,  three levels -> a dense weight vector of 256 entries
		//     large: cols_0 = 12, five levels  -> 4096 entries, sixteen times more
		let small = measure(&ladder(8, &[2, 2, 2], 8)).expect("honest proof rejected");
		let large = measure(&ladder(12, &[2, 2, 2, 2, 2], 8)).expect("honest proof rejected");

		assert!(
			large.field_muls < 3 * small.field_muls,
			"multiplications went from {} to {} while the message grew sixteenfold",
			small.field_muls,
			large.field_muls
		);
		assert!(
			large.subset_sums < 3 * small.subset_sums,
			"decompositions went from {} to {} while the message grew sixteenfold",
			small.subset_sums,
			large.subset_sums
		);
	}

	/// The per-level figures a recursive circuit pays, printed and held to their protocol shape.
	#[test]
	fn the_verifier_cost_table() {
		// Invariant: expanding a level's induced weight vector costs at least 2^cols
		// multiplications, one per entry. So a level charged far fewer than that provably did not
		// expand it, which is the succinctness claim stated as a number rather than a comment.
		//
		// Fixture state: a four-level ladder over 2^18 message elements, 30 queries a level,
		// reduced to a 2^10 residual.
		let params = ladder(16, &[2, 2, 2, 2], 30);
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		let rows = params.verifier_cost(&scheme);
		let measured = measure(&params).expect("honest proof rejected");
		let segments = segments(&measured);

		println!();
		println!(
			"Ligerito verifier cost, message 2^{}, residual 2^{}",
			params.log_msg_len(),
			params.log_residual_dim()
		);
		println!();
		println!(
			"{:>5}  {:>5}  {:>7}  {:>7}  {:>9}  {:>8}  {:>11}  {:>12}",
			"level",
			"cols",
			"queries",
			"leaves",
			"compress",
			"subsets",
			"field muls",
			"if expanded"
		);
		for (i, (level, row)) in zip(params.levels(), &rows).enumerate() {
			println!(
				"{i:>5}  {:>5}  {:>7}  {:>7}  {:>9}  {:>8}  {:>11}  {:>12}",
				level.log_msg_cols,
				level.n_queries,
				row.leaf_hashes,
				row.compressions,
				row.subset_sums,
				segments[i],
				1usize << level.log_msg_cols,
			);
		}
		let residual = rows
			.last()
			.expect("the table always ends with the residual");
		println!(
			"{:>5}  {:>5}  {:>7}  {:>7}  {:>9}  {:>8}  {:>11}  {:>12}",
			"resid",
			params.log_residual_dim(),
			"-",
			residual.leaf_hashes,
			residual.compressions,
			residual.subset_sums,
			"-",
			"-",
		);
		let total = VerifierCost::total(&rows);
		println!(
			"{:>5}  {:>5}  {:>7}  {:>7}  {:>9}  {:>8}  {:>11}  {:>12}",
			"total",
			"",
			"",
			total.leaf_hashes,
			total.compressions,
			total.subset_sums,
			measured.field_muls,
			"",
		);
		println!();
		println!(
			"the last level's multiplications carry the closing check, which pairs every glued \
			 basis against the 2^{} residual in the clear",
			params.log_residual_dim()
		);

		// Every level but the last is charged far fewer multiplications than its own weight vector
		// has entries, and expanding that vector costs at least one multiplication per entry.
		for (i, (level, spent)) in zip(params.levels(), &segments)
			.enumerate()
			.take(params.n_levels() - 1)
		{
			assert!(
				*spent < 1 << level.log_msg_cols,
				"level {i} spent {spent} multiplications against a dense vector's {}",
				1usize << level.log_msg_cols
			);
		}

		// The last segment is the closing check, which pairs every glued basis against the
		// residual in the clear. That is the one place the verifier works at a level's full width,
		// and the width is the residual's rather than the message's.
		let pairing =
			(params.n_levels() * params.levels()[0].n_queries) << params.log_residual_dim();
		let closing = segments.last().expect("a ladder has at least one level");
		assert!(
			*closing < 2 * pairing,
			"closing check spent {closing} against a bound of {pairing}"
		);

		// The residual is the only row that grows with a power of two, so a recursive circuit's
		// hash budget turns on the residual dimension rather than on the message length.
		assert!(residual.hash_calls() > total.hash_calls() / 2);
	}
}
