// Copyright 2025-2026 The Binius Developers
use std::array;

use binius_field::PackedField;
use binius_iop::{channel::OracleSpec, naive_channel::NaiveVerifierChannel};
use binius_iop_prover::naive_channel::NaiveProverChannel;
use binius_math::{
	FieldBuffer,
	multilinear::evaluate::evaluate,
	test_utils::{Packed128b, random_field_buffer, random_scalars},
};
use binius_transcript::ProverTranscript;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{config::StdChallenger, protocols::logup as logup_verifier};
use rand::{SeedableRng, rngs::StdRng};

use super::{LogUp, helper::batch_pushforwards};

#[test]
fn test_logup_prove_verify() {
	type P = Packed128b;
	type F = <P as PackedField>::Scalar;

	const N_TABLES: usize = 2;
	const N_LOOKUPS_PER_TABLE: usize = 1;
	const N_LOOKUPS: usize = N_TABLES * N_LOOKUPS_PER_TABLE;
	const N_MLES: usize = 2 * N_TABLES;

	let mut rng = StdRng::seed_from_u64(0);
	let eq_log_len = 3;
	let table_log_len = 3;
	let table_len = 1 << table_log_len;
	let lookup_len = 1 << eq_log_len;

	let tables: [FieldBuffer<P>; N_TABLES] =
		array::from_fn(|_| random_field_buffer::<P>(&mut rng, table_log_len));
	let table_ids: [usize; N_LOOKUPS] = array::from_fn(|i| i % N_TABLES);
	let indexes: [Vec<usize>; N_LOOKUPS] = array::from_fn(|lookup_idx| {
		if table_ids[lookup_idx] == 0 {
			(0..lookup_len)
				.map(|i| (i * 3 + lookup_idx) % table_len)
				.collect::<Vec<_>>()
		} else {
			(0..lookup_len)
				.map(|i| (table_len - 1 - i + lookup_idx) % table_len)
				.collect::<Vec<_>>()
		}
	});
	let indexes_ref: [&[usize]; N_LOOKUPS] = array::from_fn(|i| indexes[i].as_slice());

	let eval_point = random_scalars::<F>(&mut rng, eq_log_len);

	let lookup_evals: [F; N_LOOKUPS] = array::from_fn(|lookup_idx| {
		let table = &tables[table_ids[lookup_idx]];
		let values = indexes_ref[lookup_idx]
			.iter()
			.map(|&idx| table.get(idx))
			.collect::<Vec<_>>();
		let lookup_values = FieldBuffer::<P>::from_values(&values);
		evaluate(&lookup_values, &eval_point)
	});
	let lookup_evals_for_verify = lookup_evals;

	let batch_log_len = table_log_len + log2_ceil_usize(N_TABLES);
	let oracle_specs = vec![OracleSpec {
		log_msg_len: batch_log_len,
		is_zk: false,
	}];

	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	let mut prover_channel =
		NaiveProverChannel::<F, P, _>::new(&mut prover_transcript, oracle_specs.clone());

	let logup = LogUp::<P, _, N_TABLES>::new(
		&indexes_ref,
		&table_ids,
		&eval_point,
		&lookup_evals,
		tables,
		&mut prover_channel,
	);
	logup.prove_lookup::<N_MLES>(&mut prover_channel).unwrap();

	let fingerprinted_indexes = logup.fingerprinted_indexes.clone();
	let tables = logup.tables.clone();
	let push_forwards = logup.push_forwards.clone();
	drop(logup);
	drop(prover_channel);
	let mut verifier_transcript = prover_transcript.clone().into_verifier();
	let mut verifier_channel =
		NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &oracle_specs);
	let eval_claims = logup_verifier::verify_lookup::<F, _, N_TABLES>(
		eq_log_len,
		table_log_len,
		&eval_point,
		&lookup_evals_for_verify,
		&table_ids,
		&mut verifier_channel,
	)
	.unwrap();

	for (claim, index) in eval_claims
		.index_claims
		.iter()
		.zip(fingerprinted_indexes.iter())
	{
		assert_eq!(claim.eval, evaluate(index, &claim.point));
	}

	for (claim, table) in eval_claims.table_sumcheck_claims.iter().zip(tables.iter()) {
		assert_eq!(claim.eval, evaluate(table, &claim.point));
	}

	let batched_pushforward = batch_pushforwards::<P, F, N_TABLES>(&push_forwards);
	assert_eq!(
		eval_claims.pushforward_eval_claim.eval,
		evaluate(&batched_pushforward, &eval_claims.pushforward_eval_claim.point)
	);
}
