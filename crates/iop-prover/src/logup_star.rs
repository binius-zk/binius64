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

use binius_field::{BinaryField1b, ExtensionField, Field, PackedField};
use binius_ip_prover::logup_star::{self as reduction, witness};
use binius_math::{FieldBuffer, multilinear::eq::eq_ind_partial_eval};

use crate::channel::IOPProverChannel;

/// The pushforward oracle relation produced by the prover.
///
/// The four fields match the oracle-relation tuple the prover channel opens.
/// The caller batches this with the table and index openings into one channel open.
pub struct PushforwardRelation<P: PackedField, Oracle> {
	/// The handle of the committed pushforward oracle.
	pub oracle: Oracle,
	/// The committed message, the pushforward `Y` over the `m`-variable cube.
	pub message: FieldBuffer<P>,
	/// The transparent polynomial, the equality indicator at the reduced table point.
	pub transparent: FieldBuffer<P>,
	/// The inner-product claim, the pushforward evaluation `Y(table_eval_point)`.
	pub claim: P::Scalar,
}

/// The reduced claims of a committed logUp* proof.
///
/// The table and index claims are left for the caller to open against its own commitments.
/// The pushforward claim is packaged as an oracle relation against the commitment sent here.
pub struct LogupProof<P: PackedField, Oracle> {
	/// The `m`-coordinate point shared by the table and pushforward claims.
	pub table_eval_point: Vec<P::Scalar>,
	/// The claimed evaluation of the table `T` at the point.
	pub table_eval_claim: P::Scalar,
	/// The `n`-coordinate point of the index claim.
	pub index_eval_point: Vec<P::Scalar>,
	/// The claimed evaluation of the embedded index column at its point.
	pub index_eval_claim: P::Scalar,
	/// The oracle relation that opens the pushforward at the reduced point.
	pub pushforward: PushforwardRelation<P, Oracle>,
}

/// Prove a logUp* reduction whose pushforward is committed as an oracle.
///
/// This wraps [`binius_ip_prover::logup_star::prove`] with the pushforward commitment.
/// It builds the pushforward `Y` once, commits it, then runs the reduction over that same buffer.
/// Committing before the reduction binds `Y` into the logUp challenge.
///
/// The returned relation asserts `<Y, eq_r'> = Y(r')` at the reduced table point `r'`.
/// The caller batches this relation with the table and index openings into one channel open.
///
/// # Arguments
///
/// * `table` - The table multilinear `T` over `m` variables (`2^m` entries).
/// * `index` - The index column, one table position per looker row, of length `2^n`.
/// * `eval_point` - The `n`-coordinate evaluation point `r`.
/// * `eval_claim` - The claimed evaluation `e = (I^* T)(eval_point)`.
/// * `channel` - The IOP prover channel, whose next oracle has message length `2^m`.
///
/// # Preconditions
///
/// - The table has at least one variable.
/// - Every index entry is less than `2^m`.
pub fn prove<F, P, Channel>(
	table: &FieldBuffer<P>,
	index: &[usize],
	eval_point: &[F],
	eval_claim: F,
	channel: &mut Channel,
) -> LogupProof<P, Channel::Oracle>
where
	F: Field + ExtensionField<BinaryField1b>,
	P: PackedField<Scalar = F>,
	Channel: IOPProverChannel<P>,
{
	let m = table.log_len();

	// Build the two witnesses that do not depend on the logUp challenge c.
	//
	//     eq_r = eq(eval_point, .)     the looker numerator
	//     Y    = I_* eq_r              the pushforward, scattered onto table positions
	let eq_r = witness::equality_indicator::<F, P>(eval_point);
	let pushforward = witness::pushforward::<F, P>(&eq_r, index, m);

	// Commit Y before the reduction, so the logUp challenge binds the commitment.
	let oracle = channel.send_oracle(pushforward.to_ref());

	// Run the reduction over the committed Y and the shared eq_r, viewing the channel as IP.
	let output = reduction::prove_reduction(table, index, eval_claim, eq_r, &pushforward, channel);

	// The pushforward relation opens Y at the reduced point.
	//
	//     <Y, eq_r'> = Y(r') = pushforward_eval_claim
	let transparent = eq_ind_partial_eval::<P>(&output.table_eval_point);

	LogupProof {
		table_eval_point: output.table_eval_point,
		table_eval_claim: output.table_eval_claim,
		index_eval_point: output.index_eval_point,
		index_eval_claim: output.index_eval_claim,
		pushforward: PushforwardRelation {
			oracle,
			message: pushforward,
			transparent,
			claim: output.pushforward_eval_claim,
		},
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		BinaryField1b, ExtensionField, Field,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_iop::{
		channel::{IOPVerifierChannel, OracleSpec},
		logup_star as verify_logup,
		naive_channel::NaiveVerifierChannel,
	};
	use binius_math::{
		FieldBuffer,
		multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::{channel::IOPProverChannel, naive_channel::NaiveProverChannel};

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
		let prover_proof =
			prove::<F, P, _>(&table, &index, &eval_point, eval_claim, &mut prover_channel);

		let PushforwardRelation {
			oracle,
			message,
			transparent,
			claim: pushforward_claim,
		} = prover_proof.pushforward;
		prover_channel.prove_oracle_relations([(oracle, message, transparent, pushforward_claim)]);
		prover_channel.finish();

		// Verify: receive Y, run the reduction, then open the pushforward relation.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &specs);
		let verifier_proof =
			verify_logup::verify(m, eval_claim, &eval_point, &mut verifier_channel)
				.expect("verification succeeds");
		let verifier_pushforward_claim = verifier_proof.pushforward.claim;
		verifier_channel
			.verify_oracle_relations([verifier_proof.pushforward])
			.expect("the pushforward opening verifies");
		verifier_channel.finish();

		// The prover and verifier must derive identical reduced claims from the same transcript.
		assert_eq!(prover_proof.table_eval_point, verifier_proof.table_eval_point, "table point");
		assert_eq!(prover_proof.table_eval_claim, verifier_proof.table_eval_claim, "table claim");
		assert_eq!(prover_proof.index_eval_point, verifier_proof.index_eval_point, "index point");
		assert_eq!(prover_proof.index_eval_claim, verifier_proof.index_eval_claim, "index claim");
		assert_eq!(pushforward_claim, verifier_pushforward_claim, "pushforward claim");

		// The reduced table claim is the honest evaluation of T at the reduced point.
		assert_eq!(
			prover_proof.table_eval_claim,
			evaluate(&table, &prover_proof.table_eval_point),
			"table claim wrong (n={n}, m={m})"
		);

		// The pushforward claim is the honest evaluation of Y = I_* eq_r at the same point.
		let mut pushforward = vec![F::ZERO; 1usize << m];
		for (&j, &eq) in index.iter().zip(&eq_r) {
			pushforward[j] += eq;
		}
		let pushforward = FieldBuffer::<P>::from_values(&pushforward);
		assert_eq!(
			pushforward_claim,
			evaluate(&pushforward, &prover_proof.table_eval_point),
			"pushforward claim wrong (n={n}, m={m})"
		);

		// The index claim is the honest evaluation of the embedded index column.
		let index_embedded = index.iter().map(|&j| iota::<F>(j, m)).collect::<Vec<_>>();
		let index_embedded = FieldBuffer::<P>::from_values(&index_embedded);
		assert_eq!(
			prover_proof.index_eval_claim,
			evaluate(&index_embedded, &prover_proof.index_eval_point),
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
		let prover_proof =
			prove::<F, P, _>(&table, &index, &eval_point, wrong_claim, &mut prover_channel);
		let PushforwardRelation {
			oracle,
			message,
			transparent,
			claim,
		} = prover_proof.pushforward;
		prover_channel.prove_oracle_relations([(oracle, message, transparent, claim)]);
		prover_channel.finish();

		// The reduction's product check must surface the inconsistency as a verification failure.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &specs);
		let result = verify_logup::verify(3, wrong_claim, &eval_point, &mut verifier_channel);
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

		let prover_proof =
			prove::<F, BP, _>(&table, &index, &eval_point, eval_claim, &mut prover_channel);
		let PushforwardRelation {
			oracle,
			message,
			transparent,
			claim,
		} = prover_proof.pushforward;
		prover_channel.prove_oracle_relations([(oracle, message, transparent, claim)]);
		prover_channel.finish();

		// Verify: receive Y, run the reduction, open the pushforward through the real FRI check.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, Chal, _>(&mut verifier_transcript);
		let verifier_proof =
			verify_logup::verify(m, eval_claim, &eval_point, &mut verifier_channel)
				.expect("verification succeeds");
		verifier_channel
			.verify_oracle_relations([verifier_proof.pushforward])
			.expect("the FRI pushforward opening verifies");
		verifier_channel
			.finish()
			.expect("the batched FRI opening verifies");

		// The FRI opening already bound Y to its claim.
		// Cross-check the table and index claims against honest values.
		assert_eq!(prover_proof.table_eval_point, verifier_proof.table_eval_point);
		assert_eq!(prover_proof.table_eval_claim, verifier_proof.table_eval_claim);
		assert_eq!(prover_proof.index_eval_claim, verifier_proof.index_eval_claim);
		assert_eq!(
			prover_proof.table_eval_claim,
			evaluate(&table, &prover_proof.table_eval_point),
			"table claim must be the honest table evaluation"
		);

		// The pushforward claim is the honest evaluation of Y = I_* eq_r at the reduced point.
		let mut pushforward = vec![F::ZERO; 1usize << m];
		for (&j, &eq) in index.iter().zip(&eq_r) {
			pushforward[j] += eq;
		}
		let pushforward = FieldBuffer::<BP>::from_values(&pushforward);
		assert_eq!(claim, evaluate(&pushforward, &prover_proof.table_eval_point));
	}
}
