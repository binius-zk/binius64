// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_core::word::Word;
use binius_field::{BinaryField, FieldOps, PackedField};
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_ip_prover::{
	channel::IPProverChannel,
	prodcheck::{self, ProdcheckProver},
	sumcheck::{
		MleToSumCheckDecorator, PaddedSumcheckDecorator,
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
use binius_verifier::protocols::intmul::common::{
	IntMulOutput, Phase1Output, Phase2Output, Phase3Output, frobenius_twist,
};
use either::Either;
use itertools::izip;

use super::witness::{Witness, two_valued_field_buffer};
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
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	/// Prove an integer multiplication statement.
	///
	/// This method consumes a `Witness` in order to reduce integer multiplication statement to
	/// evaluation claims on 1-bit multilinears. More formally:
	///  * `witness` contains po2-sized integer arrays  `a`, `b`, `c_lo` and `c_hi` that satisfy `a
	///    * b = c_lo | c_hi << (1 << log_bits)`, as well as the layers of the constant- and
	///      variable-base GKR product check circuits
	///  * The proving consists of five phases:
	///    - Phase 1: GKR tree roots for B & C are evaluated at a sampled point, after which
	///      reductions are performed to obtain evaluation claims on $(b * (G^{a_i} - 1) + 1)^{2^i}$
	///    - Phase 2: Frobenius twist is applied to obtain claims on $b * (G^{a_i} - 1) + 1$
	///    - Phase 3: Two batched sumchecks:
	///      - Selector mlecheck to reduce claims on $b * (G^{a_i} - 1) + 1$ to claims on $G^{a_i}$
	///        and $b$, then recombine the $2^k$ per-bit `b` claims into one via a sampled $r_I^b$
	///      - First layer of GPA reduction for the `c_lo || c_hi` combined `c` tree
	///    - Phase 4: Batching all but last layers and `a`, `c_lo` and `c_hi`
	///    - Phase 5: Proving the last (widest) layers of `a`, `c_lo` and `c_hi` batched with a
	///      single-claim rerandomization (MLE-eval) of the recombined `b` exponent claim from phase
	///      3
	///
	/// The output of this protocol is a set of evaluation claims on the `b` selectors representing
	/// all of `a`, `b`, `c_lo` and `c_hi` as column-major bit matrices, at a common evaluation
	/// point.
	pub fn prove(&mut self, witness: Witness<'_, P>) -> IntMulOutput<F> {
		let Witness {
			log_bits,
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
		} = witness;

		// `b_root` (the variable-base `b`-exponent tree root) equals the full product `c` root, so
		// it serves as the MLE root that opens the protocol.
		let n_vars = b_root.log_len();
		assert!(log_bits >= 1);

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
		} = frobenius_twist(log_bits, &phase1_eval_point, &b_leaves_evals);

		// Phase 3
		let Phase3Output {
			eval_point: phase3_eval_point,
			r_ib,
			b_recomb,
			gpow_a_eval,
			gpow_c_lo_eval,
			gpow_c_hi_eval,
		} = self.phase3(
			log_bits,
			&twisted_eval_points,
			&twisted_evals,
			a_root,
			b_exponents,
			[c_lo_root, c_hi_root],
			&initial_eval_point,
			exp_eval,
		);

		// Phase 4
		let ([a_claim, c_lo_claim, c_hi_claim], phase4_eval_point) = self.phase4(
			log_bits,
			&phase3_eval_point,
			(gpow_a_eval, a_prodcheck),
			(gpow_c_lo_eval, c_lo_prodcheck),
			(gpow_c_hi_eval, c_hi_prodcheck),
		);

		// Phase 5
		self.phase5(
			log_bits,
			&phase4_eval_point,
			a_claim,
			c_lo_claim,
			c_hi_claim,
			b_exponents,
			&phase3_eval_point,
			&r_ib,
			b_recomb,
			a_exponents,
			c_lo_exponents,
			c_hi_exponents,
		)
	}

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
		log_bits: usize,
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
		let gamma = self.channel.sample_many(log_bits);
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

		assert_eq!(selector_prover_evals.len(), 1 + (1 << log_bits));

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
		let r_ib = self.channel.sample_many(log_bits);
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
	#[allow(clippy::type_complexity)]
	pub fn phase4(
		&mut self,
		log_bits: usize,
		eval_point: &[F],
		(a_root_eval, a_prover): (F, ProdcheckProver<P>),
		(gpow_c_lo_eval, c_lo_prover): (F, ProdcheckProver<P>),
		(gpow_c_hi_eval, c_hi_prover): (F, ProdcheckProver<P>),
	) -> ([(F, ProdcheckProver<P>); 3], Vec<F>) {
		// Each prover is over the full (widest) leaf layer of `2^log_bits` node multilinears.
		assert_eq!(a_prover.n_layers(), log_bits);
		assert_eq!(c_lo_prover.n_layers(), log_bits);
		assert_eq!(c_hi_prover.n_layers(), log_bits);

		// Sample the selector challenges that batch the 3 trees (padded to 4).
		let selector = self.channel.sample_many(log2_ceil_usize(3));

		// Run the batched prodcheck: content point is the Phase-3 evaluation point at which the
		// three roots are claimed. This runs `log_bits - 1` reduction layers, reducing the three
		// trees down to (but not including) their final (widest) leaf layer, which it returns
		// inside the remaining provers. Each remaining prover is paired with its reduced eval at
		// the shared reduced point.
		let prodcheck::BatchProveOutput {
			eval_point: reduced_point,
			provers,
		} = prodcheck::batch_prove(
			vec![a_prover, c_lo_prover, c_hi_prover],
			vec![a_root_eval, gpow_c_lo_eval, gpow_c_hi_eval],
			selector,
			eval_point.to_vec(),
			self.channel,
		);

		// The reduced point is [selector (2), suffix (n_vars), bit_index (log_bits - 1)]. Drop the
		// selector coordinates: `[suffix, bit_index]` is the point at which each retained
		// all-but-last-layer node multilinear is now claimed, and is the content+node point the
		// Phase-5 final-layer sumchecks bind. Hand the three per-tree reduced claims (eval +
		// retained final layer) straight through to Phase 5 — no all-but-last-layer evals are
		// recomputed or sent here.
		let selector_len = log2_ceil_usize(3);
		let a_c_eval_point = reduced_point[selector_len..].to_vec();

		let claims: [(F, ProdcheckProver<P>); 3] = provers
			.try_into()
			.ok()
			.expect("batch_prove returns three provers");

		(claims, a_c_eval_point)
	}

	#[doc(hidden)] // exposed for benchmarking (`benches/intmul.rs`), not a stable API
	#[allow(clippy::too_many_arguments)]
	pub fn phase5(
		&mut self,
		log_bits: usize,
		a_c_eval_point: &[F],
		a_claim: (F, ProdcheckProver<P>),
		c_lo_claim: (F, ProdcheckProver<P>),
		c_hi_claim: (F, ProdcheckProver<P>),
		b_exponents: &[Word],
		b_eval_point: &[F],
		r_ib: &[F],
		b_recomb: F,
		// The exponents supply the overflow zerocheck bits (`a_0`, `c_lo_0`) and the raw per-bit
		// output evaluations.
		a_exponents: &[Word],
		c_lo_exponents: &[Word],
		c_hi_exponents: &[Word],
	) -> IntMulOutput<F> {
		assert!(log_bits >= 1);
		let n_vars = b_eval_point.len();
		// `a_c_eval_point = [suffix (n_vars), bit_index (log_bits - 1)]` — the content point plus
		// the all-but-last-layer node coordinates. The final-layer sumchecks bind all of it; the
		// overflow and `b` checks span only the `n_vars` content coordinates and are padded up.
		let n_extra = log_bits - 1;
		assert_eq!(a_c_eval_point.len(), n_vars + n_extra);

		// Send the three per-tree reduced claims. The verifier checks they combine, weighted by
		// eq(selector), to the batched prodcheck output claim, then uses each as the seed for that
		// tree's final-layer sumcheck.
		let (a_eval, a_prover) = a_claim;
		let (c_lo_eval, c_lo_prover) = c_lo_claim;
		let (c_hi_eval, c_hi_prover) = c_hi_claim;
		self.channel.send_many(&[a_eval, c_lo_eval, c_hi_eval]);

		// Prove the final (widest) GKR layer of each tree as a regular prodcheck-layer bivariate
		// product MLE-check, seeded by the tree's reduced claim, wrapped as a sumcheck.
		let make_tree_prover = |eval: F, prover: ProdcheckProver<P>| {
			let (mle_prover, remaining) = prover.layer_prover(MultilinearEvalClaim {
				eval,
				point: a_c_eval_point.to_vec(),
			});
			debug_assert!(remaining.is_none(), "one retained layer per tree");
			MleToSumCheckDecorator::new(mle_prover)
		};
		let a_tree = make_tree_prover(a_eval, a_prover);
		let c_lo_tree = make_tree_prover(c_lo_eval, c_lo_prover);
		let c_hi_tree = make_tree_prover(c_hi_eval, c_hi_prover);

		// Embed `a_0`, `b_0`, `c_lo_0` bits into field buffers for the overflow zerocheck.
		let binary_elements = [F::zero(), F::one()];

		// TODO: Use a special 1-bit-optimized MLE-check with switchover to save memory.
		let a_0: FieldBuffer<P> = two_valued_field_buffer(0, a_exponents, binary_elements);
		let b_0: FieldBuffer<P> = two_valued_field_buffer(0, b_exponents, binary_elements);
		let c_lo_0: FieldBuffer<P> = two_valued_field_buffer(0, c_lo_exponents, binary_elements);

		// The overflow parity check binds at the Phase-2 constraint point `b_eval_point` (r_2) —
		// reused for free from the `b` re-randomization. It spans only the `n_vars` content
		// coordinates, so pad it up by `n_extra` variables to batch with the final-layer sumchecks.
		let overflow_prover = PaddedSumcheckDecorator::new(
			MleToSumCheckDecorator::new(QuadraticMleCheckProver::<P, _, _, 3>::new(
				[a_0, b_0, c_lo_0],
				|[a, b, c]| a * b - c,
				|[a, b, _c]| a * b,
				b_eval_point.to_vec(),
				F::ZERO,
			)),
			n_extra,
		);

		// Fold the 2^k b bit-columns by the recombination tensor into a single field multilinear
		// B(x) = sum_i eq(r_I^b, i) * b(i, x), then re-randomize its claim B(r_2) = b_recomb from
		// `b_eval_point` (r_2) to the shared point via a single-claim MLE-eval check, padded up.
		assert_eq!(b_exponents.len(), 1 << n_vars);
		let b_tensor = eq_ind_partial_eval_scalars::<F>(r_ib);
		let b_folded = fold_words::<_, P>(b_exponents, &b_tensor);
		let b_sumcheck_prover = PaddedSumcheckDecorator::new(
			MleToSumCheckDecorator::new(MultilinearEvalProver::new(
				b_folded,
				b_eval_point,
				b_recomb,
			)),
			n_extra,
		);

		// Batch prove all five provers — the three final-layer sumchecks, the overflow zerocheck,
		// and the `b` re-randomization — all over `n_vars + n_extra` variables.
		let BatchSumcheckOutput {
			challenges,
			multilinear_evals,
		} = batch_prove(
			vec![
				Either::Left(a_tree),
				Either::Left(c_lo_tree),
				Either::Left(c_hi_tree),
				Either::Right(Either::Left(overflow_prover)),
				Either::Right(Either::Right(b_sumcheck_prover)),
			],
			self.channel,
		);

		// The reduced point is [r_content (n_vars), r_bit (n_extra)]: `r_content` is the shared
		// evaluation point for the output claims; `r_bit` collapses the node dimension (and is the
		// padding point for the overflow / `b` checks).
		let (r_content, _r_bit) = challenges.split_at(n_vars);

		// Send the raw per-bit output evals at `r_content`, computed directly from the exponents.
		// The verifier reconstructs the selected leaf values from these and recombines them via
		// eq(r_bit) to bind the final-layer sumchecks; it binds the `b` evals via
		// sum_i eq(r_I^b, i) * b(i, r_content) = B(r_content).
		let per_bit_evals =
			|exponents: &[Word]| fold_across_words::<_, P>(exponents, r_content).to_vec();
		let a_evals = per_bit_evals(a_exponents);
		let c_lo_evals = per_bit_evals(c_lo_exponents);
		let c_hi_evals = per_bit_evals(c_hi_exponents);
		let b_evals = per_bit_evals(b_exponents);

		self.channel.send_many(&a_evals);
		self.channel.send_many(&c_lo_evals);
		self.channel.send_many(&c_hi_evals);
		self.channel.send_many(&b_evals);

		// Sanity: the overflow zerocheck's finished per-bit evals and the `b` recombined eval match
		// the sent raw evals. The padded provers' `finish` returns the inner multilinear evals.
		let [
			_a_evals2,
			_c_lo_evals2,
			_c_hi_evals2,
			lsb_evals,
			b_recomb_evals,
		] = multilinear_evals
			.try_into()
			.expect("batch_prove with 5 provers returns 5 multilinear_evals vecs");
		let [a_0_eval, b_0_eval, c_lo_0_eval] = lsb_evals
			.try_into()
			.expect("overflow prover has 3 multilinears");
		debug_assert_eq!(a_0_eval, a_evals[0]);
		debug_assert_eq!(b_0_eval, b_evals[0]);
		debug_assert_eq!(c_lo_0_eval, c_lo_evals[0]);
		debug_assert_eq!(b_recomb_evals.len(), 1);
		debug_assert_eq!(
			b_recomb_evals[0],
			evaluate(&FieldBuffer::<P>::from_values(&b_evals), r_ib)
		);

		IntMulOutput {
			eval_point: r_content.to_vec(),
			a_evals,
			b_evals,
			c_lo_evals,
			c_hi_evals,
		}
	}
}
