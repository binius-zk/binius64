// Copyright 2026 The Binius Developers

//! logUp* proving with the pushforwards committed as oracles.
//!
//! The bare reduction returns a claimed evaluation of each table's pushforward `Y = I_* eq_r`.
//! It commits nothing, so those claims cannot be checked on their own.
//! This layer commits one `Y` per table over the IOP channel and returns the relations that open
//! them.
//!
//! The commits precede the reduction, so its logUp challenges bind the committed oracles.
//! The tables `T` and indexes `I` stay the caller's oracles, so their claims are returned
//! unchanged. The pushforwards are the only oracles this protocol introduces, per
//! [Soukhanov25, Section 3].
//!
//! [`prove_transparent`] is the variant for tables the verifier evaluates itself.
//! It opens each `Y` against both `eq_z` and the table, and so needs no pushforward sumcheck.
//!
//! [Soukhanov25]: <https://eprint.iacr.org/2025/946>

use binius_compute::Allocator;
use binius_field::{BinaryField, Divisible, PackedField};
use binius_ip_prover::logup_star::{self as reduction, LogupTableOutput, witness};
pub use binius_ip_prover::logup_star::{Looker, TableLookup};
use binius_math::{FieldBuffer, multilinear::eq::eq_ind_partial_eval_in};
use itertools::izip;

use crate::channel::IOPProverChannel;

/// The reduced claims of a committed logUp* proof.
///
/// The table and index claims are left for the caller to open against its own commitments.
/// The pushforward claims are opened against the commitments sent here, through the channel.
pub struct LogupProof<F> {
	/// The point the table claims are drawn from, spanning the widest table.
	///
	/// A table over `m` variables is claimed at its **first `m`** coordinates.
	pub table_eval_point: Vec<F>,
	/// The point the index claims are drawn from, spanning the deepest looker.
	///
	/// A looker over `n` variables is claimed at its **last `n`** coordinates.
	pub index_eval_point: Vec<F>,
	/// One entry per table, in the order the tables were given.
	pub tables: Vec<LogupTableOutput<F>>,
}

/// Prove a logUp* reduction whose pushforwards are committed as oracles.
///
/// This wraps [`binius_ip_prover::logup_star::prove_reduction`] with the pushforward commitments.
/// It builds each table's pushforward `Y` once, commits it, then runs the reduction over those same
/// buffers. Committing before the reduction binds every `Y` into the logUp challenges.
///
/// The relations `<Y, eq_r> = Y(r)` at each table's reduced point are opened through the channel,
/// which may defer the actual openings to `finish()`.
///
/// One oracle per table is the simple arrangement; the pushforwards could instead be concatenated
/// into a single oracle, which is left for later.
///
/// # Arguments
///
/// * `tables` - One [`binius_ip_prover::logup_star::TableLookup`] per table, by value: its
///   multilinear and the lookers that read it.
/// * `channel` - The IOP prover channel, whose next `tables.len()` oracles have message lengths
///   `2^m` in table order.
/// * `alloc` - The allocator the witnesses are drawn from.
///
/// # Preconditions
///
/// - `tables` is non-empty, each has at least one variable, and each has at least one looker.
/// - Every index entry is less than the size of the table its looker reads.
#[tracing::instrument(skip_all, level = "debug", name = "logup* (committed)")]
pub fn prove<'a, F, P, Channel, A>(
	tables: impl IntoIterator<Item = reduction::TableLookup<'a, P>>,
	channel: &mut Channel,
	alloc: &A,
) -> LogupProof<F>
where
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F> + 'a,
	Channel: IOPProverChannel<P, A>,
	A: Allocator,
{
	// The tables are walked several times — to build the witnesses, to run the reduction, and to
	// open the commitments — so the iterator is materialized once here.
	let tables = tables.into_iter().collect::<Vec<_>>();

	// Sample the looker batching challenge before the commitments: the prover needs gamma to build
	// the pushforwards it commits.
	//
	//     gamma^i * eq_{r_i} = a table's scaled numerators
	//     Y = the scatter of that table's numerators
	let gamma = channel.sample();
	let (numerators, pushforwards) = witness::combined_lookers::<A, F, P>(alloc, gamma, &tables);

	// Commit every Y before the reduction, so the logUp challenges bind the commitments.
	let oracles = tracing::debug_span!("Commit pushforwards").in_scope(|| {
		pushforwards
			.iter()
			.map(|pushforward| channel.send_oracle(pushforward.to_ref()))
			.collect::<Vec<_>>()
	});

	// Run the reduction over the committed pushforwards and the numerators, viewing the channel as
	// IP.
	let pushforward_slices = pushforwards
		.iter()
		.map(FieldBuffer::to_ref)
		.collect::<Vec<_>>();
	let output =
		reduction::prove_reduction(alloc, gamma, &tables, numerators, &pushforward_slices, channel);

	// Open each pushforward relation through the channel; a deferring channel (e.g. BaseFold)
	// batches them with every other queued relation in `finish()`.
	//
	//     <Y, eq_r> = Y(r) = that table's pushforward claim
	//
	// A table's own point is the first m coordinates of the shared reduced point.
	let _open_guard = tracing::debug_span!("Open pushforward relations").entered();
	for (oracle, pushforward, table, table_output) in
		izip!(oracles, pushforwards, &tables, &output.tables)
	{
		let m = table.table.log_len();
		let transparent = eq_ind_partial_eval_in::<A, P>(alloc, &output.table_eval_point[..m]);
		channel.prove_oracle_relation(oracle.clone(), transparent, table_output.pushforward_claim);
		channel.finalize_oracle(oracle, pushforward);
	}

	LogupProof {
		table_eval_point: output.table_eval_point,
		index_eval_point: output.index_eval_point,
		tables: output.tables,
	}
}

/// The reduced claims of a committed logUp* proof over transparent tables.
///
/// Nothing is left on the tables or the pushforwards: both of a table's claims are opened here,
/// against the one `Y` commitment. Only the index claims are the caller's to prove.
pub struct LogupTransparentProof<F> {
	/// The point the index claims are drawn from, spanning the deepest looker.
	///
	/// A looker over `n` variables is claimed at its **last `n`** coordinates.
	pub index_eval_point: Vec<F>,
	/// One entry per table, in the order the tables were given, holding that table's lookers'
	/// index claims in its own looker order.
	pub index_eval_claims: Vec<Vec<F>>,
}

/// Prove a logUp* reduction over transparent tables, with the pushforwards committed as oracles.
///
/// This wraps [`binius_ip_prover::logup_star::prove_reduction_transparent`] the way [`prove`] wraps
/// the committed-table reduction. The reduction leaves two claims on each pushforward instead of
/// one, and both are opened here through the channel:
///
/// ```text
///     <Y, eq_z> = Y(z)      the fractional-addition leaf claim
///     <Y, T>    = e         the product claim, against the transparent table itself
/// ```
///
/// The two are queued in that order, matching
/// [`binius_iop::logup_star::verify_transparent`](binius_iop::logup_star::verify_transparent).
///
/// # Arguments
///
/// The arguments of [`prove`]. The tables are transparent, so no oracle is committed for them;
/// their multilinears are still needed, both by the reduction and by the product relation.
///
/// # Preconditions
///
/// The preconditions of [`prove`].
#[tracing::instrument(skip_all, level = "debug", name = "logup* transparent (committed)")]
pub fn prove_transparent<'a, F, P, Channel, A>(
	tables: impl IntoIterator<Item = reduction::TableLookup<'a, P>>,
	channel: &mut Channel,
	alloc: &A,
) -> LogupTransparentProof<F>
where
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F> + 'a,
	Channel: IOPProverChannel<P, A>,
	A: Allocator,
{
	let tables = tables.into_iter().collect::<Vec<_>>();

	// Sample gamma, build the witnesses, then commit every Y before the reduction, exactly as
	// [`prove`] does.
	let gamma = channel.sample();
	let (numerators, pushforwards) = witness::combined_lookers::<A, F, P>(alloc, gamma, &tables);
	let oracles = tracing::debug_span!("Commit pushforwards").in_scope(|| {
		pushforwards
			.iter()
			.map(|pushforward| channel.send_oracle(pushforward.to_ref()))
			.collect::<Vec<_>>()
	});

	let pushforward_slices = pushforwards
		.iter()
		.map(FieldBuffer::to_ref)
		.collect::<Vec<_>>();
	let output = reduction::prove_reduction_transparent(
		alloc,
		gamma,
		&tables,
		numerators,
		&pushforward_slices,
		channel,
	);

	// Open both of a table's claims against its one pushforward commitment. The product relation
	// weighs Y against a copy of the table drawn from the caller's allocator, since the channel
	// owns its transparents until the opening runs.
	let _open_guard = tracing::debug_span!("Open pushforward relations").entered();
	for (oracle, pushforward, table, open) in izip!(oracles, pushforwards, &tables, &output.tables)
	{
		let leaf_eq = eq_ind_partial_eval_in::<A, P>(alloc, &open.pushforward_eval_point);
		channel.prove_oracle_relation(oracle.clone(), leaf_eq, open.pushforward_eval_claim);
		channel.prove_oracle_relation(
			oracle.clone(),
			FieldBuffer::clone_from_slice(alloc, table.table),
			open.product_claim,
		);
		channel.finalize_oracle(oracle, pushforward);
	}

	LogupTransparentProof {
		index_eval_point: output.index_eval_point,
		index_eval_claims: output
			.tables
			.into_iter()
			.map(|table| table.index_eval_claims)
			.collect(),
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_field::{
		BinaryField1b, ExtensionField, Field, PackedBinaryGhash1x128b,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		basefold::compiler::BaseFoldVerifierCompiler,
		channel::{OracleSpec, naive::NaiveVerifierChannel},
		fri::MinProofSizeStrategy,
		logup_star::{self as verify_logup, TransparentTableLookup},
		merkle_tree::BinaryMerkleTreeScheme,
	};
	use binius_ip::logup_star::LookerClaim;
	use binius_math::{
		FieldBuffer,
		multilinear::{
			eq::eq_ind_partial_eval_scalars,
			evaluate::{evaluate, evaluate_inplace_scalars},
		},
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::{basefold::compiler::BaseFoldProverCompiler, channel::naive::NaiveProverChannel};

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;
	/// The commitment field is the GHASH 128-bit field; BaseFold uses a single-lane packing of it.
	type BP = PackedBinaryGhash1x128b;
	type Chal = HasherChallenger<StdDigest>;

	// Embed a table position j into the field through the GF(2)-linear basis, as the protocol does.
	//
	//     iota(j) = sum_{t : bit t of j is set} basis(t)
	fn iota<E: Field + ExtensionField<BinaryField1b>>(j: usize, m: usize) -> E {
		(0..m)
			.filter(|t| (j >> t) & 1 == 1)
			.map(<E as ExtensionField<BinaryField1b>>::basis)
			.fold(E::ZERO, |acc, b| acc + b)
	}

	/// One looker of a test instance: its column and its honest claim against its table.
	struct TestLooker<E> {
		index: Vec<usize>,
		eval_point: Vec<E>,
		eval_claim: E,
	}

	/// One table of a test instance: its values and the lookers that read it.
	struct TestTable<E, Q: PackedField> {
		values: FieldBuffer<Q>,
		lookers: Vec<TestLooker<E>>,
	}

	// Draw a looker over `n` variables reading `table_values`, with its honest claim.
	fn random_looker<E, Q>(
		rng: &mut StdRng,
		n: usize,
		table_values: &FieldBuffer<Q>,
	) -> TestLooker<E>
	where
		E: Field,
		Q: PackedField<Scalar = E>,
	{
		let m = table_values.log_len();
		let index = (0..(1usize << n))
			.map(|_| rng.random_range(0..(1usize << m)))
			.collect::<Vec<_>>();
		let eval_point = random_scalars::<E>(&mut *rng, n);

		// The looked-up evaluation: e = (I^* T)(r) = sum_i eq_r(i) * T[index[i]].
		let eq_r = eq_ind_partial_eval_scalars::<E>(&eval_point);
		let eval_claim = index
			.iter()
			.zip(&eq_r)
			.map(|(&j, &eq)| eq * table_values.get(j))
			.fold(E::ZERO, |acc, t| acc + t);

		TestLooker {
			index,
			eval_point,
			eval_claim,
		}
	}

	// Build the instance named by `spec`: one entry per table, giving its variable count and the
	// variable counts of the lookers that read it.
	fn random_instance<E, Q>(spec: &[(usize, Vec<usize>)], seed: u64) -> Vec<TestTable<E, Q>>
	where
		E: Field,
		Q: PackedField<Scalar = E>,
	{
		let mut rng = StdRng::seed_from_u64(seed);
		spec.iter()
			.map(|(m, looker_n_vars)| {
				let values = random_field_buffer::<Q>(&mut rng, *m);
				let lookers = looker_n_vars
					.iter()
					.map(|&n| random_looker::<E, Q>(&mut rng, n, &values))
					.collect::<Vec<_>>();
				TestTable { values, lookers }
			})
			.collect()
	}

	// Assert both sides agree and that every reduced claim is the honest evaluation. The
	// pushforward claims are bound by the channel's own opening, so they need no assertion here.
	fn check_proofs<Q>(
		prover_proof: &LogupProof<F>,
		verifier_proof: &verify_logup::LogupProof<F>,
		tables: &[TestTable<F, Q>],
		shape: &str,
	) where
		Q: PackedField<Scalar = F>,
	{
		assert_eq!(
			prover_proof.table_eval_point, verifier_proof.table_eval_point,
			"table point ({shape})"
		);
		assert_eq!(
			prover_proof.index_eval_point, verifier_proof.index_eval_point,
			"index point ({shape})"
		);
		assert_eq!(prover_proof.tables, verifier_proof.tables, "per-table claims ({shape})");

		// A table's claim is at the first m coordinates of the shared table point.
		let table_point = &prover_proof.table_eval_point;
		for (table_index, table) in tables.iter().enumerate() {
			assert_eq!(
				prover_proof.tables[table_index].eval_claim,
				evaluate(&table.values, &table_point[..table.values.log_len()]),
				"table claim wrong for table {table_index} ({shape})"
			);
		}

		let claims_by_table = prover_proof
			.tables
			.iter()
			.map(|table| table.index_eval_claims.clone())
			.collect::<Vec<_>>();
		check_index_claims(&prover_proof.index_eval_point, &claims_by_table, tables, shape);
	}

	// Check every looker's index claim, given the claims grouped by table.
	//
	// A looker's claim is at its own suffix of the shared index point.
	fn check_index_claims<Q>(
		index_point: &[F],
		claims_by_table: &[Vec<F>],
		tables: &[TestTable<F, Q>],
		shape: &str,
	) where
		Q: PackedField<Scalar = F>,
	{
		for (table_index, (table, claims)) in iter::zip(tables, claims_by_table).enumerate() {
			let m = table.values.log_len();
			assert_eq!(claims.len(), table.lookers.len(), "claim count ({shape})");
			for (looker, claim) in iter::zip(&table.lookers, claims) {
				let embedded = looker
					.index
					.iter()
					.map(|&j| iota::<F>(j, m))
					.collect::<Vec<_>>();
				let embedded = FieldBuffer::<P>::from_values(&embedded);
				let own_point = &index_point[index_point.len() - looker.eval_point.len()..];
				assert_eq!(
					*claim,
					evaluate(&embedded, own_point),
					"index claim wrong for table {table_index}, n={} ({shape})",
					looker.eval_point.len()
				);
			}
		}
	}

	// Build the prover-side witnesses for an instance.
	fn prover_tables<'a, Q>(tables: &'a [TestTable<F, Q>]) -> Vec<TableLookup<'a, Q>>
	where
		Q: PackedField<Scalar = F>,
	{
		tables
			.iter()
			.map(|table| TableLookup {
				table: table.values.to_ref(),
				lookers: table
					.lookers
					.iter()
					.map(|looker| Looker {
						index: &looker.index,
						eval_point: &looker.eval_point,
						eval_claim: looker.eval_claim,
					})
					.collect(),
			})
			.collect()
	}

	// Build the verifier-side claims for an instance.
	fn verifier_tables<'a, Q>(
		tables: &'a [TestTable<F, Q>],
	) -> Vec<binius_ip::logup_star::TableLookup<'a, F>>
	where
		Q: PackedField<Scalar = F>,
	{
		tables
			.iter()
			.map(|table| binius_ip::logup_star::TableLookup {
				n_vars: table.values.log_len(),
				lookers: table
					.lookers
					.iter()
					.map(|looker| LookerClaim {
						eval_point: &looker.eval_point,
						eval_claim: looker.eval_claim,
					})
					.collect(),
			})
			.collect()
	}

	// Build the verifier-side claims for an instance whose tables are transparent.
	//
	// A test table is a random buffer, so its "succinct" MLE is just the evaluation of its values.
	fn transparent_verifier_tables<'a, Q>(
		tables: &'a [TestTable<F, Q>],
	) -> Vec<TransparentTableLookup<'a, F>>
	where
		Q: PackedField<Scalar = F>,
	{
		iter::zip(verifier_tables(tables), tables)
			.map(|(lookup, table)| {
				let values = table.values.iter_scalars().collect::<Vec<_>>();
				TransparentTableLookup {
					lookup,
					table_eval: Box::new(move |point: &[F]| {
						evaluate_inplace_scalars(values.clone(), point)
					}),
				}
			})
			.collect()
	}

	/// Round-trip a whole instance over the naive channel and check every reduced claim.
	fn check_prove_verify(spec: &[(usize, Vec<usize>)], seed: u64) {
		let tables = random_instance::<F, P>(spec, seed);
		let shape = format!("{spec:?}");

		// One oracle per table: pushforward Y, of message length 2^m, in table order.
		let specs = tables
			.iter()
			.map(|table| OracleSpec::new(table.values.log_len()))
			.collect::<Vec<_>>();

		// Prove: commit every Y, run the reduction, then open them as the caller would.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, specs.clone());
		let prover_proof =
			prove::<F, P, _, _>(prover_tables(&tables), &mut prover_channel, &GlobalAllocator);
		prover_channel.finish();

		// Verify: receive every Y, run the reduction, then open the pushforward relations.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &specs);
		let verifier_proof = verify_logup::verify(verifier_tables(&tables), &mut verifier_channel)
			.expect("verification succeeds");
		verifier_channel.finish();

		check_proofs(&prover_proof, &verifier_proof, &tables, &shape);
	}

	/// Round-trip a transparent-table instance over the naive channel.
	///
	/// The channel checks both relations on every pushforward, so a passing run already proves the
	/// two claims the reduction left open. Only the index claims need checking here.
	fn check_prove_verify_transparent(spec: &[(usize, Vec<usize>)], seed: u64) {
		let tables = random_instance::<F, P>(spec, seed);
		let shape = format!("{spec:?}");

		// One oracle per table still: the pushforward Y. A transparent table commits nothing.
		let specs = tables
			.iter()
			.map(|table| OracleSpec::new(table.values.log_len()))
			.collect::<Vec<_>>();

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, specs.clone());
		let prover_proof = prove_transparent::<F, P, _, _>(
			prover_tables(&tables),
			&mut prover_channel,
			&GlobalAllocator,
		);
		prover_channel.finish();

		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &specs);
		let verifier_proof = verify_logup::verify_transparent(
			transparent_verifier_tables(&tables),
			&mut verifier_channel,
		)
		.expect("verification succeeds");
		verifier_channel.finish();

		assert_eq!(
			prover_proof.index_eval_point, verifier_proof.index_eval_point,
			"index point ({shape})"
		);
		assert_eq!(
			prover_proof.index_eval_claims, verifier_proof.index_eval_claims,
			"index claims ({shape})"
		);
		check_index_claims(
			&prover_proof.index_eval_point,
			&prover_proof.index_eval_claims,
			&tables,
			&shape,
		);
	}

	#[test]
	fn test_prove_verify_round_trip() {
		// A spread of shapes: m << n (the target regime), m == n, and a wide table.
		for (n, m) in [(6, 2), (5, 3), (4, 4), (3, 5), (7, 1)] {
			check_prove_verify(&[(m, vec![n])], 0);
		}
	}

	#[test]
	fn test_multi_looker_committed_round_trip() {
		// Several lookers sharing one committed pushforward.
		check_prove_verify(&[(3, vec![5, 5])], 13);
	}

	#[test]
	fn test_prove_verify_single_table_variable() {
		// m = 1 exercises the batched final layer with an empty layer-1 point.
		check_prove_verify(&[(1, vec![4])], 1);
	}

	#[test]
	fn test_multi_table_committed_round_trip() {
		// One oracle per table, received in table order. The shapes mix table sizes, looker
		// lengths, and how many lookers each table has, and put the deepest instance on either
		// side.
		for spec in [
			vec![(3usize, vec![5usize, 3usize]), (2, vec![2, 6])],
			vec![(4, vec![1]), (2, vec![3]), (5, vec![2])],
			vec![(2, vec![4, 4, 2]), (3, vec![5])],
			vec![(1, vec![0]), (4, vec![3])],
		] {
			check_prove_verify(&spec, 23);
		}
	}

	#[test]
	fn test_prove_verify_transparent_round_trip() {
		// The transparent variant shares every step but the last, so one spread covers it: m << n,
		// m == n, a wide table, a one-variable table, several lookers on one pushforward, and
		// several tables with the deepest instance on either side.
		for spec in [
			vec![(2usize, vec![6usize])],
			vec![(4, vec![4])],
			vec![(5, vec![3])],
			vec![(1, vec![4])],
			vec![(3, vec![5, 5])],
			vec![(3, vec![5, 3]), (2, vec![2, 6])],
			vec![(4, vec![1]), (2, vec![3]), (5, vec![2])],
		] {
			check_prove_verify_transparent(&spec, 31);
		}
	}

	#[test]
	fn test_verifier_rejects_wrong_eval_claim() {
		let mut tables = random_instance::<F, P>(&[(3, vec![5])], 3);
		let specs = vec![OracleSpec::new(3)];

		// Prove a false statement by perturbing the looked-up evaluation.
		tables[0].lookers[0].eval_claim += F::ONE;

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, specs.clone());
		let _prover_proof =
			prove::<F, P, _, _>(prover_tables(&tables), &mut prover_channel, &GlobalAllocator);
		prover_channel.finish();

		// The reduction's product check must surface the inconsistency as a verification failure.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &specs);
		let result = verify_logup::verify(verifier_tables(&tables), &mut verifier_channel);
		assert!(result.is_err(), "verifier must reject a wrong eval claim");
	}

	// Run a transparent-table instance over the real BaseFold channel.
	//
	// One witness-dependent (ZK) oracle per table — the pushforward Y, with 2^m entries — each
	// carrying two relations that the channel folds together before its single FRI opening. The
	// verifier's `finish` is folded into the result, so a caller can tell a rejected opening from
	// an accepted one.
	fn run_basefold_transparent(
		tables: &[TestTable<F, BP>],
	) -> Result<
		(LogupTransparentProof<F>, verify_logup::LogupTransparentProof<F>),
		verify_logup::Error,
	> {
		const LOG_INV_RATE: usize = 1;
		const SECURITY_BITS: usize = 32;
		let n_test_queries = SECURITY_BITS.div_ceil(LOG_INV_RATE);
		let oracle_specs = tables
			.iter()
			.map(|table| OracleSpec::new_zk(table.values.log_len()))
			.collect::<Vec<_>>();

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&BinaryMerkleTreeScheme::<F, StdHashSuite>::new(),
			oracle_specs,
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// Prove: commit the pushforwards with real FRI, run the reduction, open both relations.
		let domain_context = GaoMateerOnTheFly::generate(verifier_compiler.max_log_domain_size());
		let ntt = NeighborsLastSingleThread::new(domain_context);
		let prover_compiler =
			BaseFoldProverCompiler::<BP, _>::from_verifier_compiler(&verifier_compiler, ntt);

		let mut prover_transcript = ProverTranscript::new(Chal::default());
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, Chal, _, _>(
				&mut prover_transcript,
				StdRng::seed_from_u64(8),
			);

		let alloc = GlobalAllocator;
		let prover_proof =
			prove_transparent::<F, BP, _, _>(prover_tables(tables), &mut prover_channel, &alloc);
		prover_channel.finish(&alloc);

		// Verify: receive the pushforwards, run the reduction, open both relations for real.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, Chal, _>(&mut verifier_transcript);
		let verifier_proof = verify_logup::verify_transparent(
			transparent_verifier_tables(tables),
			&mut verifier_channel,
		)?;
		verifier_channel.finish()?;

		Ok((prover_proof, verifier_proof))
	}

	#[test]
	fn test_basefold_round_trip() {
		// Two tables of different sizes, so the real FRI path carries one masked oracle per table
		// and each is opened at its own prefix of the reduced point.
		let spec = [(2usize, vec![6usize, 2usize]), (4, vec![3])];
		let tables = random_instance::<F, BP>(&spec, 7);

		const LOG_INV_RATE: usize = 1;
		const SECURITY_BITS: usize = 32;
		let n_test_queries = SECURITY_BITS.div_ceil(LOG_INV_RATE);
		let oracle_specs = tables
			.iter()
			.map(|table| OracleSpec::new_zk(table.values.log_len()))
			.collect::<Vec<_>>();

		let verifier_compiler = BaseFoldVerifierCompiler::new(
			&BinaryMerkleTreeScheme::<F, StdHashSuite>::new(),
			oracle_specs,
			LOG_INV_RATE,
			n_test_queries,
			&MinProofSizeStrategy,
		);

		// Prove: commit the pushforwards with real FRI, run the reduction, open them.
		let domain_context = GaoMateerOnTheFly::generate(verifier_compiler.max_log_domain_size());
		let ntt = NeighborsLastSingleThread::new(domain_context);
		let prover_compiler =
			BaseFoldProverCompiler::<BP, _>::from_verifier_compiler(&verifier_compiler, ntt);

		let mut prover_transcript = ProverTranscript::new(Chal::default());
		let prover_channel_rng = StdRng::seed_from_u64(8);
		let mut prover_channel = prover_compiler
			.create_channel_from_transcript::<StdHashSuite, Chal, _, _>(
				&mut prover_transcript,
				prover_channel_rng,
				GlobalAllocator,
			);

		let alloc = GlobalAllocator;
		let prover_proof =
			prove::<F, BP, _, _>(prover_tables(&tables), &mut prover_channel, &alloc);
		prover_channel.finish();

		// Verify: receive the pushforwards, run the reduction, open them through the real FRI
		// check.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel = verifier_compiler
			.create_channel_from_transcript::<StdHashSuite, Chal, _>(&mut verifier_transcript);
		let verifier_proof = verify_logup::verify(verifier_tables(&tables), &mut verifier_channel)
			.expect("verification succeeds");
		verifier_channel
			.finish()
			.expect("the batched FRI openings verify");

		// The FRI openings already bound every Y to its claim.
		// Cross-check the table and index claims against honest values.
		check_proofs(&prover_proof, &verifier_proof, &tables, "basefold");
	}

	#[test]
	fn test_basefold_transparent_round_trip() {
		// Two tables of different sizes, so each pushforward oracle carries its own pair of
		// relations into the one folded FRI opening.
		let tables = random_instance::<F, BP>(&[(2usize, vec![6usize, 2usize]), (4, vec![3])], 7);

		let (prover_proof, verifier_proof) =
			run_basefold_transparent(&tables).expect("the batched FRI openings verify");

		assert_eq!(prover_proof.index_eval_point, verifier_proof.index_eval_point, "index point");
		assert_eq!(
			prover_proof.index_eval_claims, verifier_proof.index_eval_claims,
			"index claims"
		);
		check_index_claims(
			&prover_proof.index_eval_point,
			&prover_proof.index_eval_claims,
			&tables,
			"basefold transparent",
		);
	}

	#[test]
	fn test_basefold_transparent_rejects_wrong_eval_claim() {
		// The transparent reduction hands the product claim back unchecked, so a perturbed
		// looked-up evaluation survives it. Opening <Y, T> = e is what must reject.
		let mut tables = random_instance::<F, BP>(&[(3usize, vec![5usize])], 3);
		tables[0].lookers[0].eval_claim += F::ONE;

		assert!(
			run_basefold_transparent(&tables).is_err(),
			"the opening must reject a wrong eval claim"
		);
	}
}
