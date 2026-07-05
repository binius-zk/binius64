// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

use binius_core::word::Word;
use binius_field::{BinaryField, BinaryField1b, Divisible, ExtensionField, FieldOps, PackedField};
use binius_iop_prover::{
	channel::IOPProverChannel,
	logup_star::{self, Looker},
};
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_ip_prover::{
	channel::IPProverChannel,
	prodcheck::{self, ProdcheckProver},
	sumcheck::{
		MleToSumCheckDecorator,
		batch::{BatchSumcheckOutput, batch_prove, batch_prove_and_write_evals},
		bivariate_product_mle,
		multilinear_eval::MultilinearEvalProver,
		quadratic_mle::QuadraticMleCheckProver,
		selector_mle::{Claim, SelectorMlecheckProver},
	},
};
use binius_math::{
	field_buffer::FieldBuffer,
	inner_product::inner_product_buffers,
	multilinear::{
		eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars},
		evaluate::{evaluate, evaluate_inplace},
	},
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use binius_verifier::{
	config::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS},
	protocols::intmul::common::{
		IntMulOutput, LIMB_BITS, LOG_N_LIMBS, N_LIMB_COLUMNS, N_LIMBS, Phase1Output, Phase2Output,
		Phase3Output, Phase4Output, frobenius_twist, limb_column_twists, twist_limb_claim,
	},
};
use either::Either;
use itertools::izip;

use super::witness::{Witness, limb_index, two_valued_field_buffer};
use crate::fold_word::{fold_across_words, fold_words};

/// A helper structure that encapsulates switchover settings and the prover channel for
/// the integer multiplication protocol.
pub struct IntMulProver<'a, P, Channel> {
	_p_marker: PhantomData<P>,

	switchover: usize,
	channel: &'a mut Channel,
}

impl<'a, P, Channel> IntMulProver<'a, P, Channel> {
	pub const fn new(switchover: usize, channel: &'a mut Channel) -> Self {
		Self {
			_p_marker: PhantomData,
			switchover,
			channel,
		}
	}
}

impl<F, P, Channel> IntMulProver<'_, P, Channel>
where
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
	Channel: IOPProverChannel<P>,
{
	/// Prove an integer multiplication statement.
	///
	/// This method consumes a `Witness` in order to reduce integer multiplication statement to
	/// evaluation claims on 1-bit multilinears. More formally:
	///  * `witness` contains po2-sized integer arrays  `a`, `b`, `c_lo` and `c_hi` that satisfy `a
	///    * b = c_lo | c_hi << WORD_SIZE_BITS`, as well as the layers of the constant- and
	///      variable-base GKR product check circuits
	///  * The proving consists of five phases:
	///    - Phase 1: GKR tree roots for B & C are evaluated at a sampled point, after which
	///      reductions are performed to obtain evaluation claims on $(b * (G^{a_i} - 1) + 1)^{2^i}$
	///    - Phase 2: Frobenius twist is applied to obtain claims on $b * (G^{a_i} - 1) + 1$
	///    - Phase 3: Two batched sumchecks:
	///      - Selector mlecheck to reduce claims on $b * (G^{a_i} - 1) + 1$ to claims on $G^{a_i}$
	///        and $b$, then recombine the $2^k$ per-bit `b` claims into one via a sampled $r_I^b$
	///      - First layer of GPA reduction for the `c_lo || c_hi` combined `c` tree
	///    - Phase 4: Batched product check over the three depth-`LOG_N_LIMBS` constant-base trees
	///      (`a`, `c_lo`, `c_hi`), reducing the roots to per-limb evaluation claims
	///    - Phase 5: The per-limb claims are Frobenius-twisted onto the shared power table `i ↦
	///      G^i` and read from it via a committed logup* lookup; a final batched sumcheck brings
	///      the reduced index claim, a single-claim rerandomization (MLE-eval) of the recombined
	///      `b` exponent claim from phase 3, and the overflow parity zerocheck to one shared point
	///
	/// The output of this protocol is a set of evaluation claims on the `b` selectors representing
	/// all of `a`, `b`, `c_lo` and `c_hi` as column-major bit matrices, at a common evaluation
	/// point. The logup* pushforward commitment is opened through the channel inside phase 5.
	pub fn prove(&mut self, witness: Witness<'_, P>) -> IntMulOutput<F> {
		let Witness {
			a_exponents,
			a_prodcheck,
			a_root,
			b_exponents,
			b_leaves,
			b_prodcheck,
			b_root,
			c_lo_exponents,
			c_lo_prodcheck,
			c_lo_root,
			c_hi_exponents,
			c_hi_prodcheck,
			c_hi_root,
			tables,
		} = witness;

		// `b_root` (the variable-base `b`-exponent tree root) equals the full product `c` root, so
		// it serves as the MLE root that opens the protocol.
		let n_vars = b_root.log_len();

		let initial_eval_point = self.channel.sample_many(n_vars);

		// `b_root` is not needed after this, so fold it in place rather than allocating a copy.
		let exp_eval = evaluate_inplace(b_root, &initial_eval_point);

		self.channel.send_one(exp_eval);

		// Phase 1: Prodcheck reduction on b_leaves
		let Phase1Output {
			eval_point: phase1_eval_point,
			b_leaves_evals,
		} = self.phase1(&initial_eval_point, b_prodcheck, &b_leaves, exp_eval);

		// Phase 2
		let Phase2Output {
			twisted_eval_points,
			twisted_evals,
		} = frobenius_twist(LOG_WORD_SIZE_BITS, &phase1_eval_point, &b_leaves_evals);

		// Phase 3
		let Phase3Output {
			eval_point: phase3_eval_point,
			r_ib,
			b_recomb,
			gpow_a_eval,
			gpow_c_lo_eval,
			gpow_c_hi_eval,
		} = self.phase3(
			&twisted_eval_points,
			&twisted_evals,
			a_root,
			b_exponents,
			[c_lo_root, c_hi_root],
			&initial_eval_point,
			exp_eval,
		);

		// Phase 4
		let phase_4_output = self.phase4(
			&phase3_eval_point,
			(gpow_a_eval, a_prodcheck),
			(gpow_c_lo_eval, c_lo_prodcheck),
			(gpow_c_hi_eval, c_hi_prodcheck),
			[a_exponents, c_lo_exponents, c_hi_exponents],
			&tables,
		);

		// Phase 5
		self.phase5(
			&phase_4_output,
			b_exponents,
			&phase3_eval_point,
			&r_ib,
			b_recomb,
			a_exponents,
			c_lo_exponents,
			c_hi_exponents,
			&tables[0],
		)
	}

	#[doc(hidden)] // exposed for benchmarking (`benches/intmul.rs`), not a stable API
	#[allow(clippy::too_many_arguments)]
	pub fn phase5(
		&mut self,
		phase_4_output: &Phase4Output<F>,
		b_exponents: &[Word],
		b_eval_point: &[F],
		r_ib: &[F],
		b_recomb: F,
		// The exponents supply the lookup indices, the overflow zerocheck bits (`a_0`, `c_lo_0`),
		// and the raw per-bit output evaluations.
		a_exponents: &[Word],
		c_lo_exponents: &[Word],
		c_hi_exponents: &[Word],
		table: &FieldBuffer<P>,
	) -> IntMulOutput<F> {
		let n_vars = b_eval_point.len();
		assert_eq!(phase_4_output.eval_point.len(), n_vars);

		// Twist each per-limb claim onto the shared table: column (t, l) is the Frobenius power
		// φ^{twist} of the looked-up column U_{t,l}(x) = T[e_{t,l}(x)], so its claim becomes a
		// claim on U_{t,l} at the twisted point.
		let twists = limb_column_twists();
		let exponents = [a_exponents, c_lo_exponents, c_hi_exponents];
		let limb_evals = [
			&phase_4_output.a_limb_evals,
			&phase_4_output.c_lo_limb_evals,
			&phase_4_output.c_hi_limb_evals,
		];

		let index_columns = (0..N_LIMB_COLUMNS)
			.map(|j| {
				let (tree, limb) = (j / N_LIMBS, j % N_LIMBS);
				exponents[tree]
					.iter()
					.map(|&word| limb_index(word, limb))
					.collect::<Vec<_>>()
			})
			.collect::<Vec<_>>();
		let twisted_claims = (0..N_LIMB_COLUMNS)
			.map(|j| {
				let (tree, limb) = (j / N_LIMBS, j % N_LIMBS);
				twist_limb_claim(twists[j], &phase_4_output.eval_point, limb_evals[tree][limb])
			})
			.collect::<Vec<_>>();

		// Read the N_LIMB_COLUMNS looked-up columns from the shared table via the committed multi-
		// looker logup* reduction. The pushforward oracle is committed inside; its opening relation
		// is returned to the caller. The reduction returns one index claim per column, all at the
		// shared content point.
		let lookers = izip!(&index_columns, &twisted_claims)
			.map(|(index, (twisted_point, twisted_eval))| Looker {
				index,
				eval_point: twisted_point,
				eval_claim: *twisted_eval,
			})
			.collect::<Vec<_>>();
		let log_cols = log2_ceil_usize(N_LIMB_COLUMNS);
		let logup_proof = logup_star::prove(table, &lookers, self.channel);

		// The index entries are the GF(2)-linear embeddings iota(e) = Σ_u basis(u) · bit_u(e),
		// materialized by a table of all 2^LIMB_BITS embeddings.
		let mut iota_table = Vec::with_capacity(1usize << LIMB_BITS);
		iota_table.push(F::ZERO);
		for row in 1..1usize << LIMB_BITS {
			let low_bit_basis =
				<F as ExtensionField<BinaryField1b>>::basis(row.trailing_zeros() as usize);
			iota_table.push(iota_table[row & (row - 1)] + low_bit_basis);
		}

		let index_content_point = logup_proof.index_eval_point.as_slice();
		let embedded_columns = index_columns
			.iter()
			.map(|rows| rows.iter().map(|&row| iota_table[row]).collect::<Vec<_>>())
			.collect::<Vec<_>>();

		// Collapse the per-column claims into a single claim on the eq(ρ)-folded column V by
		// sampling ρ, so the final unification runs over the content variables only.
		let rho = self.channel.sample_many(log_cols);
		let mut padded_column_evals = logup_proof.index_eval_claims.clone();
		padded_column_evals.resize(1 << log_cols, F::ZERO);
		let folded_index_claim =
			evaluate(&FieldBuffer::<P>::from_values(&padded_column_evals), &rho);
		let rho_tensor = eq_ind_partial_eval_scalars::<F>(&rho);
		let folded_column_scalars = (0..1usize << n_vars)
			.map(|i| {
				izip!(&embedded_columns, &rho_tensor)
					.map(|(column, &weight)| column[i] * weight)
					.sum::<F>()
			})
			.collect::<Vec<_>>();
		let folded_column = FieldBuffer::<P>::from_values(&folded_column_scalars);
		let index_prover = MleToSumCheckDecorator::new(MultilinearEvalProver::new(
			folded_column,
			index_content_point,
			folded_index_claim,
		));

		// Embed `a_0`, `b_0`, `c_lo_0` bits into field buffers for the overflow zerocheck.
		let binary_elements = [F::zero(), F::one()];

		// TODO: Use a special 1-bit-optimized MLE-check with switchover to save memory.
		let a_0: FieldBuffer<P> = two_valued_field_buffer(0, a_exponents, binary_elements);
		let b_0: FieldBuffer<P> = two_valued_field_buffer(0, b_exponents, binary_elements);
		let c_lo_0: FieldBuffer<P> = two_valued_field_buffer(0, c_lo_exponents, binary_elements);

		// The overflow parity check binds at the Phase-2 constraint point `b_eval_point` (r_2) —
		// reused for free from the `b` re-randomization.
		let overflow_prover =
			MleToSumCheckDecorator::new(QuadraticMleCheckProver::<P, _, _, 3>::new(
				[a_0, b_0, c_lo_0],
				|[a, b, c]| a * b - c,
				|[a, b, _c]| a * b,
				b_eval_point.to_vec(),
				F::ZERO,
			));

		// Fold the 2^k b bit-columns by the recombination tensor into a single field multilinear
		// B(x) = sum_i eq(r_I^b, i) * b(i, x), then re-randomize its claim B(r_2) = b_recomb from
		// `b_eval_point` (r_2) to the shared point via a single-claim MLE-eval check.
		assert_eq!(b_exponents.len(), 1 << n_vars);
		let b_tensor = eq_ind_partial_eval_scalars::<F>(r_ib);
		let b_folded = fold_words::<_, P>(b_exponents, &b_tensor);
		let b_sumcheck_prover = MleToSumCheckDecorator::new(MultilinearEvalProver::new(
			b_folded,
			b_eval_point,
			b_recomb,
		));

		let BatchSumcheckOutput {
			challenges,
			multilinear_evals: _,
		} = batch_prove(
			vec![
				Either::Left(index_prover),
				Either::Right(Either::Left(overflow_prover)),
				Either::Right(Either::Right(b_sumcheck_prover)),
			],
			self.channel,
		);

		// `challenges` (reversed) is the shared output point for all output claims.
		let r_out = challenges.as_slice();

		// Send the raw per-bit output evals at `r_out`, computed directly from the exponents. The
		// verifier binds the stacked-index claim via the GF(2)-linearity of the embedding, the `b`
		// evals via sum_i eq(r_I^b, i) * b(i, r_out) = B(r_out), and the parity bits directly.
		let per_bit_evals =
			|exponents: &[Word]| fold_across_words::<_, P>(exponents, r_out).to_vec();
		let a_evals = per_bit_evals(a_exponents);
		let c_lo_evals = per_bit_evals(c_lo_exponents);
		let c_hi_evals = per_bit_evals(c_hi_exponents);
		let b_evals = per_bit_evals(b_exponents);

		self.channel.send_many(&a_evals);
		self.channel.send_many(&c_lo_evals);
		self.channel.send_many(&c_hi_evals);
		self.channel.send_many(&b_evals);

		IntMulOutput {
			eval_point: r_out.to_vec(),
			a_evals,
			b_evals,
			c_lo_evals,
			c_hi_evals,
		}
	}
}

impl<F, P, Channel> IntMulProver<'_, P, Channel>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	#[doc(hidden)] // exposed for benchmarking (`benches/intmul.rs`), not a stable API
	pub fn phase1(
		&mut self,
		eval_point: &[F],
		b_prover: ProdcheckProver<P>,
		b_leaves: &FieldBuffer<P>,
		b_root_eval: F,
	) -> Phase1Output<F> {
		let n_vars = eval_point.len();

		// Create initial claim
		let claim = MultilinearEvalClaim {
			eval: b_root_eval,
			point: eval_point.to_vec(),
		};

		// Run prodcheck - reduces to claim on concatenated b_leaves
		let output_claim = b_prover.prove(claim, self.channel);

		// Split output point: first n are x-point, last k are z-challenges
		let (x_point, _z_suffix) = output_claim.point.split_at(n_vars);

		// Compute leaf evaluations at x_point
		let x_tensor = eq_ind_partial_eval(x_point);
		let b_leaves_evals = b_leaves
			.chunks_par(n_vars)
			.map(|b_leaf| inner_product_buffers(&b_leaf, &x_tensor))
			.collect::<Vec<_>>();

		// Write leaf evaluations to channel
		self.channel.send_many(&b_leaves_evals);

		Phase1Output {
			eval_point: x_point.to_vec(),
			b_leaves_evals,
		}
	}

	#[doc(hidden)] // exposed for benchmarking (`benches/intmul.rs`), not a stable API
	#[allow(clippy::too_many_arguments)]
	pub fn phase3(
		&mut self,
		twisted_eval_points: &[Vec<F>],
		twisted_evals: &[F],
		selector: FieldBuffer<P>,
		b_exponents: &[Word],
		c_lo_hi_roots: [FieldBuffer<P>; 2],
		c_eval_point: &[F],
		c_root_eval: F,
	) -> Phase3Output<F> {
		let n_vars = selector.log_len();
		assert!(
			twisted_eval_points
				.iter()
				.all(|point| point.len() == n_vars)
		);
		assert_eq!(b_exponents.len(), 1 << n_vars);

		let selector_claims = izip!(twisted_eval_points, twisted_evals)
			.map(|(point, &value)| Claim {
				point: point.clone(),
				value,
			})
			.collect();

		// Batch the 2^k Frobenius-twisted leaf claims with eq_k(γ, i): sample γ in K^k and pass the
		// eq_k(γ, ·) weights to the selector prover, which combines its 2^k per-claim round
		// polynomials into a single weighted one. This replaces a univariate-power batch over the
		// 2^k claims with a multilinear one; the verifier mirrors it by weighting the corresponding
		// terms by eq_k(γ, ·). γ is sampled before the batched sumcheck so the round polynomials
		// are fixed against it.
		let gamma = self.channel.sample_many(LOG_WORD_SIZE_BITS);
		let eq_weights = eq_ind_partial_eval_scalars::<F>(&gamma);
		// `SelectorMlecheckProver` reads the exponent bits through the `Bitwise` bitmask
		// abstraction, which is implemented for the primitive integer types. `Word` is
		// `repr(transparent)` over `u64`, so reinterpret the slice in place.
		let b_bitmasks: &[u64] = bytemuck::cast_slice(b_exponents);
		let selector_prover = SelectorMlecheckProver::new(
			selector,
			selector_claims,
			b_bitmasks,
			eq_weights,
			self.switchover,
		);

		let c_root_sumcheck_prover =
			bivariate_product_mle::new(c_lo_hi_roots, c_eval_point.to_vec(), c_root_eval);

		let c_root_prover = MleToSumCheckDecorator::new(c_root_sumcheck_prover);

		let provers = vec![Either::Left(selector_prover), Either::Right(c_root_prover)];
		let BatchSumcheckOutput {
			challenges,
			multilinear_evals,
		} = batch_prove_and_write_evals(provers, self.channel);

		let [mut selector_prover_evals, c_root_prover_evals] = multilinear_evals
			.try_into()
			.expect("batch_prove with two provers returns length-2 multilinear_evals");

		assert_eq!(selector_prover_evals.len(), 1 + WORD_SIZE_BITS);

		let gpow_a_eval = selector_prover_evals
			.pop()
			.expect("selector_prover_evals.len() > 0");
		let b_evals = selector_prover_evals;
		let [gpow_c_lo_eval, gpow_c_hi_eval] = c_root_prover_evals
			.try_into()
			.expect("c_root_prover with two multilinears returns two evals");

		// Recombine the 2^k per-bit b(i, r) claims into a single claim b(r_I^b, r) by sampling a
		// recombination point r_I^b in K^k, matching the verifier. This carries one exponent claim
		// (rather than 2^k) into Phases 4 and 5.
		let r_ib = self.channel.sample_many(LOG_WORD_SIZE_BITS);
		let b_recomb = evaluate(&FieldBuffer::<P>::from_values(&b_evals), &r_ib);

		Phase3Output {
			eval_point: challenges,
			r_ib,
			b_recomb,
			gpow_a_eval,
			gpow_c_lo_eval,
			gpow_c_hi_eval,
		}
	}

	#[doc(hidden)] // exposed for benchmarking (`benches/intmul.rs`), not a stable API
	#[allow(clippy::too_many_arguments)]
	pub fn phase4(
		&mut self,
		eval_point: &[F],
		(a_root_eval, a_prover): (F, ProdcheckProver<P>),
		(gpow_c_lo_eval, c_lo_prover): (F, ProdcheckProver<P>),
		(gpow_c_hi_eval, c_hi_prover): (F, ProdcheckProver<P>),
		exponents: [&[Word]; 3],
		tables: &[FieldBuffer<P>],
	) -> Phase4Output<F> {
		let n_vars = eval_point.len();

		// Each prover is over the full (widest) leaf layer of `N_LIMBS` limb columns.
		assert_eq!(a_prover.n_layers(), LOG_N_LIMBS);
		assert_eq!(c_lo_prover.n_layers(), LOG_N_LIMBS);
		assert_eq!(c_hi_prover.n_layers(), LOG_N_LIMBS);

		// Sample the selector challenges that batch the 3 trees (padded to 4).
		let selector = self.channel.sample_many(log2_ceil_usize(3));

		// Run the batched prodcheck over all LOG_N_LIMBS layers: content point is the Phase-3
		// evaluation point at which the three roots are claimed. The output pairs each tree with
		// its reduced leaf evaluation at the shared reduced point.
		let prodcheck::BatchProveOutput {
			eval_point: reduced_point,
			evals: _tree_evals,
		} = prodcheck::batch_prove(
			vec![a_prover, c_lo_prover, c_hi_prover],
			vec![a_root_eval, gpow_c_lo_eval, gpow_c_hi_eval],
			selector,
			eval_point.to_vec(),
			self.channel,
		);

		// The reduced point is [selector (2), r_content (n_vars), r_limb (LOG_N_LIMBS)]:
		// `r_content` is the shared point at which the limb columns are claimed; `r_limb`
		// collapses the limb dimension.
		let selector_len = log2_ceil_usize(3);
		let (r_content, _r_limb) = reduced_point[selector_len..].split_at(n_vars);

		// Send the per-limb evaluations at `r_content`, computed by re-gathering each limb column
		// from its twisted power table. The verifier recombines each tree's two leaf halves via
		// eq(r_limb) to bind them to the final-layer sumchecks.
		let twists = limb_column_twists();
		let x_tensor = eq_ind_partial_eval(r_content);
		let limb_evals = |tree: usize| {
			(0..N_LIMBS)
				.map(|limb| {
					let table = &tables[twists[tree * N_LIMBS + limb] / LIMB_BITS];
					let column_scalars = exponents[tree]
						.iter()
						.map(|&word| table.get(limb_index(word, limb)))
						.collect::<Vec<_>>();
					let column = FieldBuffer::<P>::from_values(&column_scalars);
					inner_product_buffers(&column, &x_tensor)
				})
				.collect::<Vec<_>>()
		};
		let a_limb_evals = limb_evals(0);
		let c_lo_limb_evals = limb_evals(1);
		let c_hi_limb_evals = limb_evals(2);
		self.channel.send_many(&a_limb_evals);
		self.channel.send_many(&c_lo_limb_evals);
		self.channel.send_many(&c_hi_limb_evals);

		Phase4Output {
			eval_point: r_content.to_vec(),
			a_limb_evals,
			c_lo_limb_evals,
			c_hi_limb_evals,
		}
	}
}
