use std::array;

use binius_field::PackedField;
use binius_math::{
	FieldBuffer,
	multilinear::evaluate::evaluate,
	test_utils::{Packed128b, random_field_buffer, random_scalars},
};
use binius_transcript::ProverTranscript;
use binius_verifier::{config::StdChallenger, protocols::logup as logup_verifier};
use rand::{SeedableRng, rngs::StdRng};

use super::LogUp;

#[test]
fn test_logup_prove_verify() {
	type P = Packed128b;
	type F = <P as PackedField>::Scalar;

	const N_TABLES: usize = 2;
	const N_LOOKUPS: usize = 2;
	const N_MLES: usize = N_TABLES + N_LOOKUPS;

	let mut rng = StdRng::seed_from_u64(0);
	let eq_log_len = 3;
	let table_log_len = 3;
	let table_len = 1 << table_log_len;
	let lookup_len = 1 << eq_log_len;

	let tables = [
		random_field_buffer::<P>(&mut rng, table_log_len),
		random_field_buffer::<P>(&mut rng, table_log_len),
	];

	let indices_0 = (0..lookup_len)
		.map(|i| (i * 3) % table_len)
		.collect::<Vec<_>>();
	let indices_1 = (0..lookup_len)
		.map(|i| (table_len - 1 - i) % table_len)
		.collect::<Vec<_>>();
	let indexes = [indices_0.as_slice(), indices_1.as_slice()];
	let table_ids = [0usize, 1usize];

	let eval_point = random_scalars::<F>(&mut rng, eq_log_len);

	let lookup_evals = array::from_fn(|lookup_idx| {
		let table = &tables[table_ids[lookup_idx]];
		let values = indexes[lookup_idx]
			.iter()
			.map(|&idx| table.get(idx))
			.collect::<Vec<_>>();
		let lookup_values = FieldBuffer::<P>::from_values(&values);
		evaluate(&lookup_values, &eval_point)
	});
	let lookup_evals_for_verify = lookup_evals;

	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	let logup = LogUp::<P, N_TABLES, N_LOOKUPS>::new(
		[indices_0.as_slice(), indices_1.as_slice()],
		table_ids,
		&eval_point,
		lookup_evals,
		tables,
		&mut prover_transcript,
	);
	let claims = logup
		.prove_lookup::<StdChallenger, N_MLES>(&mut prover_transcript)
		.unwrap();

	let mut verifier_transcript = prover_transcript.into_verifier();
	logup_verifier::verify_lookup(
		eq_log_len,
		table_log_len,
		&lookup_evals_for_verify,
		&claims,
		&mut verifier_transcript,
	)
	.unwrap();
}
