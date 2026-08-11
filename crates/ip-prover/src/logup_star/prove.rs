// Copyright 2026 The Binius Developers

//! The top-level logUp* proving routine.

use std::iter;

use binius_compute::Allocator;
use binius_field::{BinaryField, Divisible, PackedField};
use binius_ip::{
	MultilinearEvalClaim,
	fracaddcheck::unpad_leaf_claim,
	logup_star::{LogupOutput, LogupTableOutput},
};
use binius_math::{FieldBuffer, FieldSlice, FieldVec, univariate::evaluate_univariate};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use itertools::izip;

use super::{
	pushforward::{PushforwardOutput, TableWitness, prove_pushforward},
	witness,
};
use crate::{
	channel::IPProverChannel,
	fracaddcheck::{self, FracAddCheckProver},
};

/// One looker's column and claim: `(I^* T)(eval_point) = eval_claim` against the table it reads.
#[derive(Debug, Clone, Copy)]
pub struct Looker<'a, F> {
	/// The index column, one table position per looker row (`2^n` entries).
	pub index: &'a [usize],
	/// The `n`-coordinate evaluation point of this looker's claim.
	pub eval_point: &'a [F],
	/// The claimed evaluation of this looker's looked-up vector at the point.
	pub eval_claim: F,
}

/// One table together with the lookers that read it.
pub struct TableLookup<'a, P: PackedField> {
	/// The table multilinear `T` over its `m` variables (`2^m` entries).
	pub table: FieldSlice<'a, P>,
	/// The lookers that read this table. Their columns may differ in length, both from each other
	/// and from the table.
	pub lookers: Vec<Looker<'a, P::Scalar>>,
}

/// Prove a logUp* indexed-lookup reduction.
///
/// This is the prover for [`binius_ip::logup_star::verify_reduction`].
/// It produces the transcript the verifier consumes and returns the same reduced claims.
///
/// The reduction proves the indexed lookups `(I_j^* T_{t(j)})(r_j) = e_j` for one or more lookers
/// reading one or more tables. The lookers batch by a random linear combination: a challenge
/// `gamma` scales looker `j`'s equality-indicator numerator by `gamma^j` across the whole batch,
/// and table `t`'s pushforward is the gamma-weighted sum of the pushforwards of the lookers that
/// read it, still with only `2^m_t` entries. The looked-up vectors are never committed. Every
/// fractional-addition circuit — looker `j`'s over `n_j` variables and table `t`'s over `m_t` — is
/// an instance of one GKR of `ceil(log2(#lookers + #tables)) + max(max_j n_j, max_t m_t)` layers,
/// with every shallower instance padded by zero fractions. Neither the lookers nor the tables need
/// agree on a length.
///
/// Each table is randomized by its own logUp challenge `c_t`, which is what makes the single root
/// check certify every table separately; see [`binius_ip::logup_star::verify_reduction`].
/// See [Soukhanov25] for the construction.
///
/// [Soukhanov25]: <https://eprint.iacr.org/2025/946>
///
/// # Arguments
///
/// * `tables` - The table multilinears `T_t`, each over its own `m_t` variables.
/// * `lookers` - The looker columns and claims; each names the table it reads, evaluation points
///   may differ in length, looker `j`'s index column must have `2^n_j` entries, and every index
///   entry must be less than the size of the table it reads.
/// * `channel` - The prover channel for sending messages and sampling challenges.
///
/// The logUp challenges are sampled against the committed `I_j`, `T_t`, and pushforwards `Y_t`.
/// So the caller must absorb those commitments into the transcript before calling this routine.
///
/// # Preconditions
///
/// - `tables` is non-empty and each has at least one variable, so every table-side GKR has a
///   variable to split on.
/// - Every `eval_claim` must equal `(I_j^* T_{t(j)})(r_j)`, or the proof will not verify.
///
/// # Returns
///
/// The reduced claims on the tables, the pushforwards, and the per-looker index multilinears. The
/// index claims are drawn from one point spanning the deepest looker; looker `j` is claimed at its
/// last `n_j` coordinates. The table claims are drawn from one point spanning the widest table;
/// table `t` is claimed at its first `m_t` coordinates.
/// The caller verifies those claims, which is out of scope here.
pub fn prove<'a, A, F, P>(
	alloc: &A,
	gamma: F,
	tables: impl IntoIterator<Item = TableLookup<'a, P>>,
	channel: &mut impl IPProverChannel<F>,
) -> LogupOutput<F>
where
	A: Allocator,
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F> + 'a,
{
	let tables = tables.into_iter().collect::<Vec<_>>();

	// Build the witnesses that do not depend on the logUp challenges. Each table's own gamma scales
	// its lookers:
	//
	//     gamma^i * eq_{r_i} = that table's scaled numerators
	//     Y = sum_i gamma^i * (I_i)_* eq_{r_i}     that table's pushforward
	let (numerators, pushforwards) = witness::combined_lookers::<A, F, P>(alloc, gamma, &tables);

	// The self-contained prover commits nothing.
	// It runs the reduction over the witnesses directly.
	let pushforward_slices = pushforwards
		.iter()
		.map(FieldBuffer::to_ref)
		.collect::<Vec<_>>();
	prove_reduction(alloc, gamma, &tables, numerators, &pushforward_slices, channel)
}

/// Run the logUp* reduction over the pre-built witnesses `numerators` and pushforwards `Y_t`.
///
/// This is the reduction core of [`prove`], split out so a caller can build the `Y_t` once and
/// commit them. The committing prover builds the numerators and the pushforwards, commits them,
/// then hands both here. That way each scatter-add runs only once.
///
/// # Arguments
///
/// * `tables` - One [`TableLookup`] per table, each carrying its batching challenge, its
///   multilinear, and its lookers.
/// * `numerators` - The gamma-scaled numerators `gamma^i * eq_{r_i}`, grouped per table in the same
///   order (see [`witness::combined_lookers`]).
/// * `pushforwards` - The per-table pushforwards `Y`, the scatter of that table's numerators.
/// * `channel` - The prover channel.
///
/// # Preconditions
///
/// - `tables` is non-empty, each has at least one variable, and each has at least one looker.
/// - `numerators` and `pushforwards` are grouped/ordered to match `tables`, and `Y` has the same
///   variable count as its table.
/// - A looker's index column has `2^n` entries for its own point length `n`, with every entry less
///   than the size of its table.
/// - Each `Y` equals the scatter of its table's numerators.
#[tracing::instrument(
	skip_all,
	level = "debug",
	name = "logup* reduction",
	fields(n_tables = tables.len())
)]
pub fn prove_reduction<A, F, P>(
	alloc: &A,
	gamma: F,
	tables: &[TableLookup<'_, P>],
	numerators: Vec<Vec<FieldVec<P, A>>>,
	pushforwards: &[FieldSlice<P>],
	channel: &mut impl IPProverChannel<F>,
) -> LogupOutput<F>
where
	A: Allocator,
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
{
	assert!(!tables.is_empty(), "at least one table is required");
	// Each table-side GKR circuit needs at least one variable to split on.
	assert!(
		tables.iter().all(|table| table.table.log_len() > 0),
		"every table must have at least one variable"
	);
	assert!(
		tables.iter().all(|table| !table.lookers.is_empty()),
		"every table must have at least one looker"
	);

	let n_lookers = tables
		.iter()
		.map(|table| table.lookers.len())
		.sum::<usize>();
	// One batch instance per looker, plus one per table.
	let k = log2_ceil_usize(n_lookers + tables.len());

	// Sample one logUp challenge per table, randomizing that table's logarithmic-derivative
	// denominators. This is the prover's first transcript action, mirroring the verifier.
	// A committing caller must absorb the I, T, and Y commitments into the transcript before this.
	let cs = channel.sample_many(tables.len());

	// Build the fractional-addition circuits, one per looker plus one per table. Constructing a
	// circuit computes every layer and returns its single root fraction.
	//
	//     looker j: gamma^j * eq_{r_j}(i) / (c_t - I_j(i))   over n_j variables
	//     table t:  Y_t(v)                / (c_t - v)        over m_t variables
	let circuits_guard = tracing::debug_span!("Build fracadd circuits").entered();
	// The circuits are independent, so they build in parallel across lookers. The looker instances
	// are laid out table by table, in the order the tables are given, matching the verifier.
	let flat_lookers = tables
		.iter()
		.zip(&cs)
		.flat_map(|(table, &c)| table.lookers.iter().map(move |looker| (looker, c)))
		.collect::<Vec<_>>();
	let flat_numerators = numerators.into_iter().flatten().collect::<Vec<_>>();
	let (mut provers, mut roots): (Vec<_>, Vec<_>) = (flat_lookers.as_slice(), flat_numerators)
		.into_par_iter()
		.map(|(&(looker, c), numerator)| {
			let den = witness::looker_denominator::<A, F, P>(alloc, c, looker.index);
			// Lookers may differ in length, so each circuit is built at its own depth.
			let (prover, root) =
				FracAddCheckProver::new(looker.eval_point.len(), alloc, (numerator, den));
			(prover, (root.0.get(0), root.1.get(0)))
		})
		.unzip();
	// A table's fraction enters the sum of every instance negated, which is what makes that sum's
	// numerator vanish. The negation rides on the denominator, which is built here anyway.
	for ((&c, table), pushforward) in iter::zip(iter::zip(&cs, tables), pushforwards) {
		let table = table.table;
		let table_den = witness::table_denominator::<A, F, P>(alloc, c, table.log_len());
		// The pushforward is borrowed — a committing caller keeps it for the oracle opening — so
		// the table circuit's leaf layer, which it folds in place, is a clone drawn from `alloc`.
		let (table_prover, table_root) = FracAddCheckProver::new(
			table.log_len(),
			alloc,
			(FieldBuffer::clone_from_slice(alloc, pushforward.to_ref()), table_den),
		);
		provers.push(table_prover);
		roots.push((table_root.0.get(0), table_root.1.get(0)));
	}

	// Top circuit: interpolate every instance's root fraction into a multilinear pair over the k
	// selector variables, padded with the zero fraction. Its own root is the fractional sum of all
	// of them, and the table denominators are already negated, so that sum is
	//
	//     sum_j num_j / den_j  -  sum_t num_t / den_t
	//
	// which is zero exactly when the logUp identities hold.
	let (top_prover, (top_root_num, top_root_den)) = FracAddCheckProver::new(k, alloc, {
		let (mut root_nums, mut root_dens): (Vec<_>, Vec<_>) = roots.iter().copied().unzip();
		root_nums.resize(1 << k, F::ZERO);
		root_dens.resize(1 << k, F::ONE);
		(
			FieldBuffer::<P, _>::from_values_in(alloc, &root_nums),
			FieldBuffer::<P, _>::from_values_in(alloc, &root_dens),
		)
	});
	let root_den = top_root_den.get(0);
	drop(circuits_guard);

	// A witness satisfying the lookup identity zeroes the root numerator, so it is not sent: the
	// verifier supplies the zero itself, and a prover whose lookups do not match cannot make the
	// rest of the GKR agree with it.
	debug_assert_eq!(
		top_root_num.get(0),
		F::ZERO,
		"the lookup identities must hold: each table's looker fractions must sum to its own"
	);
	channel.send_one(root_den);

	// One GKR over the whole thing: k layers of top circuit down to the per-instance roots, then
	// max(max_n, max_m) more over the instances themselves. Looker j's tree has depth n_j and table
	// t's tree depth m_t, so the batch pads every shallower instance up — the padding costs O(1)
	// per round and the layer count depends only on that maximum.
	let gkr_guard = tracing::debug_span!("Combined GKR").entered();
	let (top_num_claim, _top_den_claim) = top_prover.prove(
		(
			MultilinearEvalClaim {
				eval: F::ZERO,
				point: Vec::new(),
			},
			MultilinearEvalClaim {
				eval: root_den,
				point: Vec::new(),
			},
		),
		channel,
	);
	let selector_point = top_num_claim.point;
	let fracaddcheck::BatchProveOutput {
		eval_point,
		fractions,
	} = fracaddcheck::batch_prove_unequal_depths(provers, roots, selector_point, channel);
	drop(gkr_guard);

	// The leaf claims are on the padded witnesses; divide the padding back out to land on each
	// circuit's own leaves. The node coordinates are the point past the selector ones.
	let node_point = &eval_point[k..];
	let n_layers = node_point.len();
	let (looker_fractions, table_fractions) = fractions.split_at(n_lookers);

	// The per-looker leaf denominators are its table's c minus I(content), so the index claims are
	// their c-complements. The numerators are the transparent scaled equality indicators, which the
	// verifier evaluates itself. The claims stay grouped by table, as the output reports them.
	let mut remaining_fractions = looker_fractions;
	let index_eval_claims = iter::zip(tables, &cs)
		.map(|(table, &c)| {
			let (table_fractions, rest) = remaining_fractions.split_at(table.lookers.len());
			remaining_fractions = rest;
			iter::zip(table_fractions, &table.lookers)
				.map(|(&fraction, looker)| {
					let leaf =
						unpad_leaf_claim(fraction, node_point, n_layers - looker.eval_point.len());
					c - leaf.den_eval
				})
				.collect::<Vec<_>>()
		})
		.collect::<Vec<_>>();
	// Every looker's claim is a suffix of the deepest looker's point, so that one carries them all:
	// a looker's is its last n coordinates. The node point can run deeper still when a table is
	// the deepest instance, and those extra coordinates belong to the tables alone.
	let max_n = tables
		.iter()
		.flat_map(|table| &table.lookers)
		.map(|looker| looker.eval_point.len())
		.max()
		.expect("every table has at least one looker");
	let index_eval_point = node_point[n_layers - max_n..].to_vec();

	// A table's leaf claims Y at the point past its own padding; its denominator is the public
	// J - c, which the verifier checks itself.
	let table_leaves = iter::zip(table_fractions, tables)
		.map(|(&fraction, table)| {
			unpad_leaf_claim(fraction, node_point, n_layers - table.table.log_len())
		})
		.collect::<Vec<_>>();

	// Send the claims the verifier cannot derive: the index evaluations and the Y. Together with
	// the transparent halves they rebuild the batch's leaf claim. The index evaluations go out
	// table by table, flattened, which is the order the verifier reads them.
	let pushforward_evals = table_leaves
		.iter()
		.map(|leaf| leaf.num_eval)
		.collect::<Vec<_>>();
	for claims in &index_eval_claims {
		channel.send_many(claims);
	}
	channel.send_many(&pushforward_evals);

	// Reduce every table's leaf claim on Y_t and its product claim <T_t, Y_t> = e_t to one shared
	// evaluation point.
	let table_witnesses =
		izip!(tables, pushforwards, &table_leaves).map(|(table, &pushforward, leaf)| {
			// Its lookers are weighted gamma^0, gamma^1, ..., so the combination is the univariate
			// evaluation of their claims at gamma.
			let claims = table
				.lookers
				.iter()
				.map(|looker| looker.eval_claim)
				.collect::<Vec<_>>();
			let eval_claim = evaluate_univariate(&claims, &gamma);
			TableWitness {
				table: table.table,
				pushforward,
				eval_claim,
				pushforward_eval_claim: leaf.num_eval,
				pushforward_eval_point: &leaf.point,
			}
		});
	let PushforwardOutput {
		table_eval_point,
		table_eval_claims,
		pushforward_eval_claims,
	} = prove_pushforward(alloc, table_witnesses, channel);

	LogupOutput {
		table_eval_point,
		index_eval_point,
		tables: izip!(table_eval_claims, pushforward_eval_claims, index_eval_claims)
			.map(|(eval_claim, pushforward_claim, index_eval_claims)| LogupTableOutput {
				eval_claim,
				pushforward_claim,
				index_eval_claims,
			})
			.collect(),
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{
		BinaryField1b, ExtensionField, Field,
		arch::{OptimalB128, OptimalPackedB128},
		util::powers,
	};
	use binius_ip::{channel::IPVerifierChannel, logup_star};
	use binius_math::{
		FieldBuffer,
		multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	// Embed a table position j into the field through the GF(2)-linear basis, as the protocol does.
	//
	//     iota(j) = sum_{t : bit t of j is set} basis(t)
	fn iota(j: usize, m: usize) -> F {
		(0..m)
			.filter(|t| (j >> t) & 1 == 1)
			.map(<F as ExtensionField<BinaryField1b>>::basis)
			.fold(F::ZERO, |acc, b| acc + b)
	}

	/// One looker of a test instance: its column and its honest claim against its table.
	struct TestLooker {
		index: Vec<usize>,
		eval_point: Vec<F>,
		eq_r: Vec<F>,
		eval_claim: F,
	}

	/// One table of a test instance: its values and the lookers that read it.
	struct TestTable {
		values: FieldBuffer<P>,
		lookers: Vec<TestLooker>,
	}

	// Draw a looker over `n` variables reading `table_values`, with its honest claim.
	fn random_looker(rng: &mut StdRng, n: usize, table_values: &FieldBuffer<P>) -> TestLooker {
		let m = table_values.log_len();
		let index = (0..(1usize << n))
			.map(|_| rng.random_range(0..(1usize << m)))
			.collect::<Vec<_>>();
		let eval_point = random_scalars::<F>(&mut *rng, n);

		// The looked-up evaluation: e = (I^* T)(r) = sum_i eq_r(i) * T[index[i]].
		let eq_r = eq_ind_partial_eval_scalars::<F>(&eval_point);
		let eval_claim = index
			.iter()
			.zip(&eq_r)
			.map(|(&j, &eq)| eq * table_values.get(j))
			.fold(F::ZERO, |acc, t| acc + t);

		TestLooker {
			index,
			eval_point,
			eq_r,
			eval_claim,
		}
	}

	// Build the instance named by `spec`: one entry per table, giving its variable count and the
	// variable counts of the lookers that read it.
	fn random_instance(spec: &[(usize, Vec<usize>)], seed: u64) -> Vec<TestTable> {
		let mut rng = StdRng::seed_from_u64(seed);
		spec.iter()
			.map(|(m, looker_n_vars)| {
				let values = random_field_buffer::<P>(&mut rng, *m);
				let lookers = looker_n_vars
					.iter()
					.map(|&n| random_looker(&mut rng, n, &values))
					.collect::<Vec<_>>();
				TestTable { values, lookers }
			})
			.collect()
	}

	/// Round-trip a whole instance and check every reduced claim against the honest witness.
	///
	/// `spec` gives one `(table_n_vars, [looker_n_vars])` per table, so one helper covers the
	/// single-table, multi-looker, unequal-length and multi-table shapes alike.
	fn check_round_trip(spec: &[(usize, Vec<usize>)], seed: u64) {
		let alloc = GlobalAllocator;
		let tables = random_instance(spec, seed);
		let shape = format!("{spec:?}");

		// Prove, then replay the transcript through the verifier. Each table's batching challenge
		// is the caller's to sample, one per table, before the reduction runs.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let gamma = IPProverChannel::<F>::sample(&mut prover_transcript);
		let prover_tables = tables
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
			.collect::<Vec<_>>();
		let prover_out =
			prove::<GlobalAllocator, F, P>(&alloc, gamma, prover_tables, &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_gamma = IPVerifierChannel::<F>::sample(&mut verifier_transcript);
		assert_eq!(verifier_gamma, gamma, "both sides must draw the same challenge ({shape})");
		let verifier_tables = tables
			.iter()
			.map(|table| logup_star::TableLookup {
				n_vars: table.values.log_len(),
				lookers: table
					.lookers
					.iter()
					.map(|looker| logup_star::LookerClaim {
						eval_point: &looker.eval_point,
						eval_claim: looker.eval_claim,
					})
					.collect(),
			})
			.collect::<Vec<_>>();
		let verifier_out = logup_star::verify_reduction::<F, _>(
			&verifier_gamma,
			verifier_tables,
			&mut verifier_transcript,
		)
		.expect("verification succeeds");

		// The prover and verifier must derive identical reduced claims from the same transcript.
		assert_eq!(prover_out, verifier_out, "outputs disagree ({shape})");

		// The table point spans the widest table; table t's claims are at its first m_t
		// coordinates.
		let table_point = &prover_out.table_eval_point;
		for (table_index, table) in tables.iter().enumerate() {
			let m = table.values.log_len();
			let own_point = &table_point[..m];

			assert_eq!(
				prover_out.tables[table_index].eval_claim,
				evaluate(&table.values, own_point),
				"table claim wrong for table {table_index} ({shape})"
			);

			// The honest pushforward of this table: its own lookers' numerators, weighted by
			// gamma^i within the table — the same power series serves every table — scattered onto
			// its cube.
			let mut pushforward = vec![F::ZERO; 1usize << m];
			for (looker, power) in iter::zip(&table.lookers, powers(gamma)) {
				for (&j, &eq) in iter::zip(&looker.index, &looker.eq_r) {
					pushforward[j] += power * eq;
				}
			}
			let pushforward = FieldBuffer::<P>::from_values(&pushforward);
			assert_eq!(
				prover_out.tables[table_index].pushforward_claim,
				evaluate(&pushforward, own_point),
				"pushforward claim wrong for table {table_index} ({shape})"
			);
		}

		// The index point spans the deepest looker; a looker's claim is at its last n coordinates,
		// so a shorter looker reads a suffix of it. The claims come back grouped by table.
		let index_point = &prover_out.index_eval_point;
		for (table_index, table) in tables.iter().enumerate() {
			let m = table.values.log_len();
			let claims = &prover_out.tables[table_index].index_eval_claims;
			assert_eq!(claims.len(), table.lookers.len(), "claim count ({shape})");
			for (looker, claim) in iter::zip(&table.lookers, claims) {
				let embedded = looker.index.iter().map(|&j| iota(j, m)).collect::<Vec<_>>();
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

	#[test]
	fn test_prove_verify_round_trip() {
		// A spread of shapes: m << n (the target regime), m == n, and a wide table.
		for (n, m) in [(6, 2), (5, 3), (4, 4), (3, 5), (7, 1)] {
			check_round_trip(&[(m, vec![n])], 0);
		}
	}

	#[test]
	fn test_prove_verify_single_table_variable() {
		// m = 1 exercises the table side with a single GKR layer and a one-variable reduction.
		check_round_trip(&[(1, vec![4])], 1);
	}

	#[test]
	fn test_prove_verify_single_looker_row() {
		// n = 0 exercises the looker side with no GKR layers: the root is already the leaf claim.
		check_round_trip(&[(3, vec![0])], 2);
	}

	#[test]
	fn test_multi_looker_round_trip() {
		// Several lookers sharing one table, the shape the intmul limb columns use.
		check_round_trip(&[(3, vec![5, 5, 5])], 11);
	}

	#[test]
	fn test_lookers_of_unequal_length() {
		// A table's lookers need not agree on a column length: each is its own instance in the
		// batch, padded up to the deepest one. The spread puts the table both above and below the
		// deepest looker, and repeats a length so the shared-depth path is exercised too.
		for (looker_n_vars, m) in [
			(vec![5usize, 2, 4], 3usize),
			(vec![2, 6], 2),
			(vec![1, 1, 5], 4),
			(vec![3, 3], 5),
			(vec![0, 4], 3),
		] {
			check_round_trip(&[(m, looker_n_vars)], 17);
		}
	}

	#[test]
	fn test_multi_table_round_trip() {
		// Several tables of differing sizes, each with its own gamma and its own lookers. The
		// shapes vary how many lookers a table has, their lengths, and whether the deepest instance
		// is a table or a looker.
		for spec in [
			// Two tables, two lookers each, every column a different length.
			vec![(3usize, vec![5usize, 3usize]), (2, vec![2, 6])],
			// Three tables, one looker each, the deepest instance being a table.
			vec![(4, vec![1]), (2, vec![3]), (5, vec![2])],
			// A table read by several lookers beside one read by a single looker.
			vec![(2, vec![4, 4, 2]), (3, vec![5])],
			// Equal-size tables, the path where no table is padded at all.
			vec![(3, vec![4]), (3, vec![4]), (3, vec![4])],
			// A single-row looker against a one-variable table, both extremes at once.
			vec![(1, vec![0]), (4, vec![3])],
		] {
			check_round_trip(&spec, 23);
		}
	}

	#[test]
	fn test_tables_are_independent() {
		// Each table carries its own gamma and its own logUp challenge, so two tables holding the
		// same values but different lookers must still both verify.
		check_round_trip(&[(3, vec![4, 2]), (3, vec![5])], 29);
	}

	#[test]
	fn test_verifier_rejects_wrong_eval_claim() {
		let alloc = GlobalAllocator;
		let tables = random_instance(&[(3, vec![5])], 3);
		let table = &tables[0];
		let looker = &table.lookers[0];

		// Prove a false statement by perturbing the looked-up evaluation.
		let wrong_claim = looker.eval_claim + F::ONE;
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let gamma = IPProverChannel::<F>::sample(&mut prover_transcript);
		prove::<GlobalAllocator, F, P>(
			&alloc,
			gamma,
			[TableLookup {
				table: table.values.to_ref(),
				lookers: vec![Looker {
					index: &looker.index,
					eval_point: &looker.eval_point,
					eval_claim: wrong_claim,
				}],
			}],
			&mut prover_transcript,
		);

		// The product-check inconsistency must surface as a verification failure.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let gamma = IPVerifierChannel::<F>::sample(&mut verifier_transcript);
		let result = logup_star::verify_reduction::<F, _>(
			&gamma,
			[logup_star::TableLookup {
				n_vars: 3,
				lookers: vec![logup_star::LookerClaim {
					eval_point: &looker.eval_point,
					eval_claim: wrong_claim,
				}],
			}],
			&mut verifier_transcript,
		);
		assert!(result.is_err(), "verifier must reject a wrong eval claim");
	}

	#[test]
	fn test_verifier_rejects_lookup_against_the_wrong_table() {
		// The prover reads table 0's values but files the looker under table 1. Each table has its
		// own logUp challenge, so the two fractions cannot cancel in the root sum.
		let alloc = GlobalAllocator;
		let tables = random_instance(&[(3, vec![4]), (3, vec![4])], 31);
		let looker = &tables[0].lookers[0];

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let gamma = IPProverChannel::<F>::sample(&mut prover_transcript);
		// Table 1 gets table 0's honest looker; its own values differ, so the claim is false there.
		prove::<GlobalAllocator, F, P>(
			&alloc,
			gamma,
			[
				TableLookup {
					table: tables[0].values.to_ref(),
					lookers: vec![Looker {
						index: &tables[0].lookers[0].index,
						eval_point: &tables[0].lookers[0].eval_point,
						eval_claim: tables[0].lookers[0].eval_claim,
					}],
				},
				TableLookup {
					table: tables[1].values.to_ref(),
					lookers: vec![Looker {
						index: &looker.index,
						eval_point: &looker.eval_point,
						eval_claim: looker.eval_claim,
					}],
				},
			],
			&mut prover_transcript,
		);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let gamma = IPVerifierChannel::<F>::sample(&mut verifier_transcript);
		let result = logup_star::verify_reduction::<F, _>(
			&gamma,
			[
				logup_star::TableLookup {
					n_vars: 3,
					lookers: vec![logup_star::LookerClaim {
						eval_point: &tables[0].lookers[0].eval_point,
						eval_claim: tables[0].lookers[0].eval_claim,
					}],
				},
				logup_star::TableLookup {
					n_vars: 3,
					lookers: vec![logup_star::LookerClaim {
						eval_point: &looker.eval_point,
						eval_claim: looker.eval_claim,
					}],
				},
			],
			&mut verifier_transcript,
		);
		assert!(result.is_err(), "verifier must reject a lookup against the wrong table");
	}

	#[test]
	#[should_panic(expected = "every table must have at least one variable")]
	fn test_zero_variable_table_panics() {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// A zero-variable table has a single entry and no variable for the GKR to split on.
		let table = random_field_buffer::<P>(&mut rng, 0);
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let gamma = IPProverChannel::<F>::sample(&mut transcript);
		let _ = prove::<GlobalAllocator, F, P>(
			&alloc,
			gamma,
			[TableLookup {
				table: table.to_ref(),
				lookers: vec![Looker {
					index: &[0],
					eval_point: &[],
					eval_claim: F::ZERO,
				}],
			}],
			&mut transcript,
		);
	}

	#[test]
	#[should_panic(expected = "every table must have at least one looker")]
	fn test_table_without_lookers_panics() {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// A table nothing reads has no claim to prove, so it does not belong in the batch.
		let table = random_field_buffer::<P>(&mut rng, 3);
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let gamma = IPProverChannel::<F>::sample(&mut transcript);
		let _ = prove::<GlobalAllocator, F, P>(
			&alloc,
			gamma,
			[TableLookup {
				table: table.to_ref(),
				lookers: Vec::new(),
			}],
			&mut transcript,
		);
	}

	#[test]
	#[should_panic(expected = "index column has 3 entries but 16 were expected for 4 variables")]
	fn test_rejects_index_length_mismatch() {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;
		let table = random_field_buffer::<P>(&mut rng, 3);
		let eval_point = random_scalars::<F>(&mut rng, 4);
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let gamma = IPProverChannel::<F>::sample(&mut transcript);

		// eval_point has 4 coordinates, so the index column must have 2^4 = 16 entries, not 3.
		let _ = prove::<GlobalAllocator, F, P>(
			&alloc,
			gamma,
			[TableLookup {
				table: table.to_ref(),
				lookers: vec![Looker {
					index: &[0, 1, 2],
					eval_point: &eval_point,
					eval_claim: F::ZERO,
				}],
			}],
			&mut transcript,
		);
	}

	#[test]
	#[should_panic(expected = "every index entry must be less than the size of the table")]
	fn test_out_of_range_index_panics() {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;
		let table = random_field_buffer::<P>(&mut rng, 2);
		let eval_point = random_scalars::<F>(&mut rng, 1);
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let gamma = IPProverChannel::<F>::sample(&mut transcript);

		// The table has 2^2 = 4 positions, so index value 4 is out of range.
		// The range check is a debug_assert precondition, so this panics in debug builds.
		let _ = prove::<GlobalAllocator, F, P>(
			&alloc,
			gamma,
			[TableLookup {
				table: table.to_ref(),
				lookers: vec![Looker {
					index: &[0, 4],
					eval_point: &eval_point,
					eval_claim: F::ZERO,
				}],
			}],
			&mut transcript,
		);
	}
}
