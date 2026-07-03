// Copyright 2026 The Binius Developers

//! logUp* proving with the pushforward committed as an oracle.
//!
//! The bare reduction returns a claimed evaluation of the pushforward `Y = I_* eq_r`.
//! It commits nothing, so that claim cannot be checked on its own.
//! This layer commits `Y` over the IOP channel and returns the relation that opens it.
//!
//! The commit precedes the reduction, so the logUp challenge binds the committed `Y`.
//! The table `T` and index `I` stay the caller's oracles, so their claims are returned unchanged.
//! `Y` is the only oracle this protocol introduces, per [Soukhanov25, Section 3].
//!
//! [Soukhanov25]: <https://eprint.iacr.org/2025/946>

use binius_field::{BinaryField, Divisible, PackedField};
pub use binius_ip_prover::logup_star::Looker;
use binius_ip_prover::logup_star::{self as reduction, witness};
use binius_math::{
	FieldBuffer, multilinear::eq::eq_ind_partial_eval, univariate::evaluate_univariate,
};

use crate::channel::IOPProverChannel;

/// The reduced claims of a committed logUp* proof.
///
/// The table and index claims are left for the caller to open against its own commitments.
/// The pushforward claim is opened against the commitment sent here, through the channel.
pub struct LogupProof<F> {
	/// The `m`-coordinate point shared by the table and pushforward claims.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluation of the table `T` at the point.
	pub table_eval_claim: F,
	/// The `n`-coordinate point shared by the index claims.
	pub index_eval_point: Vec<F>,
	/// The claimed evaluations of the per-looker embedded index columns at the point.
	pub index_eval_claims: Vec<F>,
}

/// Prove a logUp* reduction whose pushforward is committed as an oracle.
///
/// This wraps [`binius_ip_prover::logup_star::prove`] with the pushforward commitment.
/// It builds the pushforward `Y` once, commits it, then runs the reduction over that same buffer.
/// Committing before the reduction binds `Y` into the logUp challenge.
///
/// The relation `<Y, eq_r'> = Y(r')` at the reduced table point `r'` is opened through the
/// channel, which may defer the actual opening to `finish()`.
///
/// # Arguments
///
/// * `table` - The table multilinear `T` over `m` variables (`2^m` entries).
/// * `lookers` - The looker columns and claims; every evaluation point must have the same length
///   `n` and every index column `2^n` entries.
/// * `channel` - The IOP prover channel, whose next oracle has message length `2^m`.
///
/// # Preconditions
///
/// - The table has at least one variable.
/// - Every index entry is less than `2^m`.
pub fn prove<F, P, Channel>(
	table: &FieldBuffer<P>,
	lookers: &[Looker<'_, F>],
	channel: &mut Channel,
) -> LogupProof<F>
where
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
	Channel: IOPProverChannel<P>,
{
	let m = table.log_len();

	// Sample the looker batching challenge, then build the two witnesses that do not depend on
	// the logUp challenge c.
	//
	//     gamma^j * eq_{r_j} = the per-looker scaled numerators
	//     Y = sum_j gamma^j * (I_j)_* eq_{r_j}     the combined pushforward
	let gamma = channel.sample();
	let (numerators, pushforward) = witness::combined_lookers::<F, P>(lookers, gamma, m);

	// Commit Y before the reduction, so the logUp challenge binds the commitment.
	let oracle = channel.send_oracle(pushforward.to_ref());

	// The product check binds <T, Y> to the gamma-combination of the looker claims.
	let claims = lookers
		.iter()
		.map(|looker| looker.eval_claim)
		.collect::<Vec<_>>();
	let combined_eval_claim = evaluate_univariate(&claims, gamma);

	// Run the reduction over the committed Y and the numerators, viewing the channel as IP.
	let output = reduction::prove_reduction(
		table,
		lookers,
		combined_eval_claim,
		numerators,
		&pushforward,
		channel,
	);

	// Open the pushforward relation through the channel; a deferring channel (e.g. BaseFold)
	// batches it with every other queued relation in `finish()`.
	//
	//     <Y, eq_r'> = Y(r') = pushforward_eval_claim
	let transparent = eq_ind_partial_eval::<P>(&output.table_eval_point);
	channel.prove_oracle_relations([(
		oracle,
		pushforward,
		transparent,
		output.pushforward_eval_claim,
	)]);

	LogupProof {
		table_eval_point: output.table_eval_point,
		table_eval_claim: output.table_eval_claim,
		index_eval_point: output.index_eval_point,
		index_eval_claims: output.index_eval_claims,
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		BinaryField1b, ExtensionField, Field,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_iop::{
		channel::OracleSpec, logup_star as verify_logup, naive_channel::NaiveVerifierChannel,
	};
	use binius_ip::logup_star::LookerClaim;
	use binius_math::{
		FieldBuffer,
		multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::naive_channel::NaiveProverChannel;

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	// Embed a table position j into the field through the GF(2)-linear basis, as the protocol does.
	//
	//     iota(j) = sum_{t : bit t of j is set} basis(t)
	fn iota<E: Field + ExtensionField<BinaryField1b>>(j: usize, m: usize) -> E {
		(0..m)
			.filter(|t| (j >> t) & 1 == 1)
			.map(<E as ExtensionField<BinaryField1b>>::basis)
			.fold(E::ZERO, |acc, b| acc + b)
	}

	// Build a random instance and return (table, index, eval_point, eq_r scalars, true eval claim).
	fn random_instance<E, Q>(
		rng: &mut StdRng,
		n: usize,
		m: usize,
	) -> (FieldBuffer<Q>, Vec<usize>, Vec<E>, Vec<E>, E)
	where
		E: Field,
		Q: PackedField<Scalar = E>,
	{
		let table = random_field_buffer::<Q>(&mut *rng, m);
		let index = (0..(1usize << n))
			.map(|_| rng.random_range(0..(1usize << m)))
			.collect::<Vec<_>>();
		let eval_point = random_scalars::<E>(&mut *rng, n);

		// The looked-up evaluation: e = (I^* T)(r) = sum_i eq_r(i) * T[index[i]].
		let eq_r = eq_ind_partial_eval_scalars::<E>(&eval_point);
		let eval_claim = index
			.iter()
			.zip(&eq_r)
			.map(|(&j, &eq)| eq * table.get(j))
			.fold(E::ZERO, |acc, t| acc + t);

		(table, index, eval_point, eq_r, eval_claim)
	}

	fn check_prove_verify(n: usize, m: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let (table, index, eval_point, eq_r, eval_claim) = random_instance::<F, P>(&mut rng, n, m);

		// The one oracle is the pushforward Y, of message length 2^m.
		let specs = vec![OracleSpec::new(m)];

		// Prove: commit Y, run the reduction, then open Y as the caller would.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, specs.clone());
		let looker = reduction::Looker {
			index: &index,
			eval_point: &eval_point,
			eval_claim,
		};
		let prover_proof = prove::<F, P, _>(&table, &[looker], &mut prover_channel);

		prover_channel.finish();

		// Verify: receive Y, run the reduction, then open the pushforward relation.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &specs);
		let looker_claim = LookerClaim {
			eval_point: &eval_point,
			eval_claim,
		};
		let verifier_proof = verify_logup::verify(m, &[looker_claim], &mut verifier_channel)
			.expect("verification succeeds");
		verifier_channel.finish();

		// The prover and verifier must derive identical reduced claims from the same transcript.
		assert_eq!(prover_proof.table_eval_point, verifier_proof.table_eval_point, "table point");
		assert_eq!(prover_proof.table_eval_claim, verifier_proof.table_eval_claim, "table claim");
		assert_eq!(prover_proof.index_eval_point, verifier_proof.index_eval_point, "index point");
		assert_eq!(
			prover_proof.index_eval_claims, verifier_proof.index_eval_claims,
			"index claims"
		);

		// The reduced table claim is the honest evaluation of T at the reduced point.
		assert_eq!(
			prover_proof.table_eval_claim,
			evaluate(&table, &prover_proof.table_eval_point),
			"table claim wrong (n={n}, m={m})"
		);

		// The pushforward opening is checked inside the channel; nothing further to assert here.
		let _ = eq_r;

		// The index claim is the honest evaluation of the embedded index column.
		let index_embedded = index.iter().map(|&j| iota::<F>(j, m)).collect::<Vec<_>>();
		let index_embedded = FieldBuffer::<P>::from_values(&index_embedded);
		assert_eq!(
			prover_proof.index_eval_claims,
			vec![evaluate(&index_embedded, &prover_proof.index_eval_point)],
			"index claim wrong (n={n}, m={m})"
		);
	}

	#[test]
	fn test_prove_verify_round_trip() {
		// A spread of shapes: m << n (the target regime), m == n, and a wide table.
		for (n, m) in [(6, 2), (5, 3), (4, 4), (3, 5), (7, 1)] {
			check_prove_verify(n, m, 0);
		}
	}

	#[test]
	fn test_multi_looker_committed_round_trip() {
		let mut rng = StdRng::seed_from_u64(13);
		let (n, m) = (5, 3);

		let (table, index_0, eval_point_0, eq_r_0, eval_claim_0) =
			random_instance::<F, P>(&mut rng, n, m);
		let (_, index_1, eval_point_1, eq_r_1, _) = random_instance::<F, P>(&mut rng, n, m);
		// The second looker's claim reads the shared table.
		let eval_claim_1 = index_1
			.iter()
			.zip(&eq_r_1)
			.map(|(&j, &eq)| eq * table.get(j))
			.fold(F::ZERO, |acc, t| acc + t);

		let specs = vec![OracleSpec::new(m)];

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, specs.clone());
		let lookers = [
			reduction::Looker {
				index: &index_0,
				eval_point: &eval_point_0,
				eval_claim: eval_claim_0,
			},
			reduction::Looker {
				index: &index_1,
				eval_point: &eval_point_1,
				eval_claim: eval_claim_1,
			},
		];
		let prover_proof = prove::<F, P, _>(&table, &lookers, &mut prover_channel);
		prover_channel.finish();

		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &specs);
		let looker_claims = [
			LookerClaim {
				eval_point: &eval_point_0,
				eval_claim: eval_claim_0,
			},
			LookerClaim {
				eval_point: &eval_point_1,
				eval_claim: eval_claim_1,
			},
		];
		let verifier_proof = verify_logup::verify(m, &looker_claims, &mut verifier_channel)
			.expect("verification succeeds");
		verifier_channel.finish();

		assert_eq!(prover_proof.table_eval_point, verifier_proof.table_eval_point);
		assert_eq!(prover_proof.table_eval_claim, verifier_proof.table_eval_claim);
		assert_eq!(prover_proof.index_eval_claims, verifier_proof.index_eval_claims);
		assert_eq!(prover_proof.table_eval_claim, evaluate(&table, &prover_proof.table_eval_point),);
		// The eq_r tensors of both lookers went into the shared pushforward; nothing further to
		// check here beyond the openings above.
		let _ = (eq_r_0, eq_r_1);
	}

	#[test]
	fn test_prove_verify_single_table_variable() {
		// m = 1 exercises the batched final layer with an empty layer-1 point.
		check_prove_verify(4, 1, 1);
	}

	#[test]
	fn test_verifier_rejects_wrong_eval_claim() {
		let mut rng = StdRng::seed_from_u64(3);
		let (table, index, eval_point, _eq_r, eval_claim) = random_instance::<F, P>(&mut rng, 5, 3);
		let specs = vec![OracleSpec::new(3)];

		// Prove a false statement by perturbing the looked-up evaluation.
		let wrong_claim = eval_claim + F::ONE;
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, specs.clone());
		let looker = reduction::Looker {
			index: &index,
			eval_point: &eval_point,
			eval_claim: wrong_claim,
		};
		let _prover_proof = prove::<F, P, _>(&table, &[looker], &mut prover_channel);
		prover_channel.finish();

		// The reduction's product check must surface the inconsistency as a verification failure.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &specs);
		let looker_claim = LookerClaim {
			eval_point: &eval_point,
			eval_claim: wrong_claim,
		};
		let result = verify_logup::verify(3, &[looker_claim], &mut verifier_channel);
		assert!(result.is_err(), "verifier must reject a wrong eval claim");
	}

	#[test]
	fn test_basefold_round_trip() {
		use binius_field::PackedBinaryGhash1x128b;
		use binius_hash::{StdDigest, StdHashSuite};
		use binius_iop::{
			basefold_compiler::BaseFoldVerifierCompiler, fri::MinProofSizeStrategy,
			merkle_tree::BinaryMerkleTreeScheme,
		};
		use binius_math::ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly};

		use crate::basefold_compiler::BaseFoldProverCompiler;

		// The commitment field is the GHASH 128-bit field; use a single-lane packing for BaseFold.
		type BP = PackedBinaryGhash1x128b;
		type Chal = HasherChallenger<StdDigest>;

		let (n, m) = (6, 2);
		let mut rng = StdRng::seed_from_u64(7);
		let (table, index, eval_point, eq_r, eval_claim) = random_instance::<F, BP>(&mut rng, n, m);

		// One witness-dependent (ZK) oracle: the pushforward Y, with 2^m entries.
		const LOG_INV_RATE: usize = 1;
		const SECURITY_BITS: usize = 32;
		let n_test_queries = SECURITY_BITS.div_ceil(LOG_INV_RATE);
		let oracle_specs = vec![OracleSpec::new_zk(m)];

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			BinaryMerkleTreeScheme::<F, StdHashSuite>::new(),
			oracle_specs,
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// Prove: commit Y with real FRI, run the reduction, open the pushforward.
		let domain_context =
			GenericOnTheFly::generate_from_subspace(verifier_compiler.max_subspace());
		let ntt = NeighborsLastSingleThread::new(domain_context);
		let prover_compiler =
			BaseFoldProverCompiler::<BP, _>::from_verifier_compiler(&verifier_compiler, ntt);

		let mut prover_transcript = ProverTranscript::new(Chal::default());
		let prover_channel_rng = StdRng::seed_from_u64(8);
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, Chal, _>(
				&mut prover_transcript,
				prover_channel_rng,
			);

		let looker = reduction::Looker {
			index: &index,
			eval_point: &eval_point,
			eval_claim,
		};
		let prover_proof = prove::<F, BP, _>(&table, &[looker], &mut prover_channel);
		prover_channel.finish();

		// Verify: receive Y, run the reduction, open the pushforward through the real FRI check.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, Chal, _>(&mut verifier_transcript);
		let looker_claim = LookerClaim {
			eval_point: &eval_point,
			eval_claim,
		};
		let verifier_proof = verify_logup::verify(m, &[looker_claim], &mut verifier_channel)
			.expect("verification succeeds");
		verifier_channel
			.finish()
			.expect("the batched FRI opening verifies");

		// The FRI opening already bound Y to its claim.
		// Cross-check the table and index claims against honest values.
		assert_eq!(prover_proof.table_eval_point, verifier_proof.table_eval_point);
		assert_eq!(prover_proof.table_eval_claim, verifier_proof.table_eval_claim);
		assert_eq!(prover_proof.index_eval_claims, verifier_proof.index_eval_claims);
		assert_eq!(
			prover_proof.table_eval_claim,
			evaluate(&table, &prover_proof.table_eval_point),
			"table claim must be the honest table evaluation"
		);

		// The FRI opening already bound the pushforward claim inside the channel.
		let _ = eq_r;
	}
}
