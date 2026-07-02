// Copyright 2025 Irreducible Inc.

use std::{iter, marker::PhantomData};

use binius_core::word::Word;
use binius_field::{BinaryField, FieldOps, PackedField};
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_ip_prover::{
	channel::IPProverChannel,
	prodcheck::{self, ProdcheckProver},
	sumcheck::{
		MleToSumCheckDecorator,
		batch::{BatchSumcheckOutput, batch_prove, batch_prove_and_write_evals},
		bivariate_product_mle,
		bivariate_product_multi_mle::BivariateProductMultiMlecheckProver,
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
		evaluate::evaluate,
	},
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use binius_verifier::protocols::intmul::common::{
	IntMulOutput, Phase1Output, Phase2Output, Phase3Output, Phase4Output, frobenius_twist,
	normalize_a_c_exponent_evals,
};
use either::Either;
use itertools::{chain, izip};

use super::{
	error::Error,
	witness::{Witness, buffer_bivariate_product, two_valued_field_buffer},
};
use crate::fold_word::{fold_across_words, fold_words};

/// A helper structure that encapsulates switchover settings and the prover channel for
/// the integer multiplication protocol.
pub struct IntMulProver<'a, P, S, Channel> {
	_p_marker: PhantomData<P>,
	_s_marker: PhantomData<S>,

	switchover: usize,
	channel: &'a mut Channel,
}

impl<'a, P, S, Channel> IntMulProver<'a, P, S, Channel> {
	pub const fn new(switchover: usize, channel: &'a mut Channel) -> Self {
		Self {
			_p_marker: PhantomData,
			_s_marker: PhantomData,
			switchover,
			channel,
		}
	}
}

impl<'a, F, P, S, Channel> IntMulProver<'a, P, S, Channel>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	S: AsRef<[u64]> + Sync,
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
	pub fn prove(&mut self, witness: Witness<P, u64, S>) -> Result<IntMulOutput<F>, Error> {
		let Witness {
			log_bits,
			a_exponents,
			a_prodcheck,
			a_root,
			b_exponents,
			b_leaves,
			b_prodcheck,
			b_root: _,
			c_lo_exponents,
			c_lo_prodcheck,
			c_lo_root,
			c_hi_prodcheck,
			c_hi_root,
			c_root,
			_b_marker,
		} = witness;

		let n_vars = c_root.log_len();
		assert!(log_bits >= 1);

		let initial_eval_point = self.channel.sample_many(n_vars);

		let exp_eval = evaluate(&c_root, &initial_eval_point);

		self.channel.send_one(exp_eval);

		// Phase 1: Prodcheck reduction on b_leaves
		let Phase1Output {
			eval_point: phase1_eval_point,
			b_leaves_evals,
		} = self.phase1(&initial_eval_point, b_prodcheck, &b_leaves, exp_eval)?;

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
			b_exponents.as_ref(),
			[c_lo_root, c_hi_root],
			&initial_eval_point,
			exp_eval,
		)?;

		// Phase 4
		let (
			Phase4Output {
				eval_point: phase4_eval_point,
				a_evals,
				c_lo_evals,
				c_hi_evals,
			},
			[a_leaves, c_lo_leaves, c_hi_leaves],
		) = self.phase4(
			log_bits,
			&phase3_eval_point,
			(gpow_a_eval, a_prodcheck),
			(gpow_c_lo_eval, c_lo_prodcheck),
			(gpow_c_hi_eval, c_hi_prodcheck),
		)?;

		// Phase 5
		self.phase5(
			log_bits,
			&phase4_eval_point,
			(&a_evals, a_leaves),
			(&c_lo_evals, c_lo_leaves),
			(&c_hi_evals, c_hi_leaves),
			b_exponents.as_ref(),
			&phase3_eval_point,
			&r_ib,
			b_recomb,
			a_exponents.as_ref(),
			c_lo_exponents.as_ref(),
		)
	}

	fn phase1(
		&mut self,
		eval_point: &[F],
		b_prover: ProdcheckProver<P>,
		b_leaves: &FieldBuffer<P>,
		b_root_eval: F,
	) -> Result<Phase1Output<F>, Error> {
		let n_vars = eval_point.len();

		// Create initial claim
		let claim = MultilinearEvalClaim {
			eval: b_root_eval,
			point: eval_point.to_vec(),
		};

		// Run prodcheck - reduces to claim on concatenated b_leaves
		let output_claim = b_prover.prove(claim, self.channel)?;

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

		Ok(Phase1Output {
			eval_point: x_point.to_vec(),
			b_leaves_evals,
		})
	}

	#[allow(clippy::too_many_arguments)]
	fn phase3(
		&mut self,
		log_bits: usize,
		twisted_eval_points: &[Vec<F>],
		twisted_evals: &[F],
		selector: FieldBuffer<P>,
		b_exponents: &[u64],
		c_lo_hi_roots: [FieldBuffer<P>; 2],
		c_eval_point: &[F],
		c_root_eval: F,
	) -> Result<Phase3Output<F>, Error> {
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
		let selector_prover = SelectorMlecheckProver::new(
			selector,
			selector_claims,
			b_exponents,
			eq_weights,
			self.switchover,
		)?;

		let c_root_sumcheck_prover =
			bivariate_product_mle::new(c_lo_hi_roots, c_eval_point.to_vec(), c_root_eval)?;

		let c_root_prover = MleToSumCheckDecorator::new(c_root_sumcheck_prover);

		let provers = vec![Either::Left(selector_prover), Either::Right(c_root_prover)];
		let BatchSumcheckOutput {
			challenges,
			multilinear_evals,
		} = batch_prove_and_write_evals(provers, self.channel)?;

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

		Ok(Phase3Output {
			eval_point: challenges,
			r_ib,
			b_recomb,
			gpow_a_eval,
			gpow_c_lo_eval,
			gpow_c_hi_eval,
		})
	}

	#[allow(clippy::type_complexity)]
	fn phase4(
		&mut self,
		log_bits: usize,
		eval_point: &[F],
		(a_root_eval, a_prover): (F, ProdcheckProver<P>),
		(gpow_c_lo_eval, c_lo_prover): (F, ProdcheckProver<P>),
		(gpow_c_hi_eval, c_hi_prover): (F, ProdcheckProver<P>),
	) -> Result<(Phase4Output<F>, [Vec<FieldBuffer<P>>; 3]), Error> {
		let n_vars = eval_point.len();
		// Each prover is over the full (widest) leaf layer of `2^log_bits` node multilinears.
		assert_eq!(a_prover.n_layers(), log_bits);
		assert_eq!(c_lo_prover.n_layers(), log_bits);
		assert_eq!(c_hi_prover.n_layers(), log_bits);

		// Sample the selector challenges that batch the 3 trees (padded to 4).
		let selector = self.channel.sample_many(log2_ceil_usize(3));

		// Run the batched prodcheck: content point is the Phase-3 evaluation point at which the
		// three roots are claimed. This runs `log_bits - 1` reduction layers, reducing the three
		// trees down to (but not including) their final (widest) leaf layer, which it returns
		// inside the remaining provers.
		let prodcheck::BatchProveOutput {
			eval_point: reduced_point,
			provers,
		} = prodcheck::batch_prove(
			vec![a_prover, c_lo_prover, c_hi_prover],
			vec![a_root_eval, gpow_c_lo_eval, gpow_c_hi_eval],
			selector,
			eval_point.to_vec(),
			self.channel,
		)?;

		// The reduced point is [selector (2), suffix (n_vars), bit_index (log_bits - 1)]. The
		// suffix is the content point at which the all-but-last node multilinears are now
		// claimed.
		let selector_len = log2_ceil_usize(3);
		let suffix = reduced_point[selector_len..selector_len + n_vars].to_vec();

		// Extract each tree's retained leaf layer as `2^log_bits` per-node n_vars-variate buffers,
		// in the natural node order produced by `constant_base_leaves` (node `z` carries bit `z`).
		// The prodcheck reduces on the highest node bit, so the all-but-last-layer node `z` is the
		// product of leaves `z` and `z + half` (a strided pairing of bits `z` and `z + half`).
		let [a_leaves, c_lo_leaves, c_hi_leaves] = provers
			.into_iter()
			.map(|(_eval, prover)| split_leaf_layer(prover.into_final_layer(), n_vars))
			.collect::<Vec<_>>()
			.try_into()
			.expect("batch_prove returns three provers");

		// Compute the all-but-last-layer (`2^(log_bits - 1)` node) evals at `suffix` by folding the
		// pairwise leaf products. Node `z` = leaf[z] * leaf[z + half].
		// TODO: these leaf evals should later be pulled directly out of the sumcheck folding in the
		// last prodcheck layer rather than recomputed.
		let suffix_tensor = eq_ind_partial_eval(&suffix);
		let half = 1 << (log_bits - 1);
		let leaf_evals = |leaves: &[FieldBuffer<P>]| {
			(0..half)
				.map(|z| {
					let node = buffer_bivariate_product(&leaves[z], &leaves[z + half]);
					inner_product_buffers(&node, &suffix_tensor)
				})
				.collect::<Vec<_>>()
		};

		let a_evals = leaf_evals(&a_leaves);
		let c_lo_evals = leaf_evals(&c_lo_leaves);
		let c_hi_evals = leaf_evals(&c_hi_leaves);

		self.channel.send_many(&a_evals);
		self.channel.send_many(&c_lo_evals);
		self.channel.send_many(&c_hi_evals);

		Ok((
			Phase4Output {
				eval_point: suffix,
				a_evals,
				c_lo_evals,
				c_hi_evals,
			},
			[a_leaves, c_lo_leaves, c_hi_leaves],
		))
	}

	#[allow(clippy::too_many_arguments)]
	fn phase5(
		&mut self,
		log_bits: usize,
		a_c_eval_point: &[F],
		(a_evals, a_layer): (&[F], Vec<FieldBuffer<P>>),
		(c_lo_evals, c_lo_layer): (&[F], Vec<FieldBuffer<P>>),
		(c_hi_evals, c_hi_layer): (&[F], Vec<FieldBuffer<P>>),
		b_exponents: &[u64],
		b_eval_point: &[F],
		r_ib: &[F],
		b_recomb: F,
		// Needed for the zerocheck on `a_0 * b_0 = c_lo_0`.
		a_exponents: &[u64],
		c_lo_exponents: &[u64],
	) -> Result<IntMulOutput<F>, Error> {
		assert!(log_bits >= 1);
		assert_eq!(1 << log_bits, a_layer.len());
		assert_eq!(2 * a_evals.len(), a_layer.len());
		assert_eq!(2 * c_lo_evals.len(), c_lo_layer.len());
		assert_eq!(2 * c_hi_evals.len(), c_hi_layer.len());
		assert_eq!(b_eval_point.len(), a_layer.first().expect("log_bits >= 1").log_len());
		assert_eq!(a_c_eval_point.len(), b_eval_point.len());

		// Make the `BivariateProductMultiMlecheckProver` prover.
		// The prover proves an MLE eval claim on each pair of the retained leaf layer. The leaf
		// layer is in natural node order (node `z` carries bit `z`), so pairing node `z` with node
		// `z + half` reproduces the bivariate-product layer the verifier expects (with pair `z`
		// being the strided bits `z` and `z + half`).
		let pairs = chain!(split_pairs(a_layer), split_pairs(c_lo_layer), split_pairs(c_hi_layer))
			.collect::<Vec<_>>();
		let evals = [a_evals, c_lo_evals, c_hi_evals].concat();

		let bivariate_mle_prover =
			BivariateProductMultiMlecheckProver::new(pairs, a_c_eval_point, evals)?;
		let bivariate_sumcheck_prover = MleToSumCheckDecorator::new(bivariate_mle_prover);

		// Embed `a_0` and `b_0` bits into field buffers for `BivariateProductMultiMlecheckProver`.
		let binary_elements = [F::zero(), F::one()];

		// TODO: Use a special 1-bit-optimized MLE-check with switchover to save memory.
		let a_0: FieldBuffer<P> = two_valued_field_buffer(0, &a_exponents, binary_elements);
		let b_0: FieldBuffer<P> = two_valued_field_buffer(0, &b_exponents, binary_elements);
		let c_lo_0: FieldBuffer<P> = two_valued_field_buffer(0, &c_lo_exponents, binary_elements);

		// Make the sumcheck prover for the overflow parity check, binding it at the Phase-2
		// constraint point `b_eval_point` (r_2) per the spec (reused for free from the `b`
		// re-randomization) rather than the Phase-4 point.
		let overflow_prover =
			MleToSumCheckDecorator::new(QuadraticMleCheckProver::<P, _, _, 3>::new(
				[a_0, b_0, c_lo_0],
				|[a, b, c]| a * b - c,
				|[a, b, _c]| a * b,
				b_eval_point.to_vec(),
				F::ZERO,
			)?);

		// Fold the 2^k b bit-columns by the recombination tensor into a single field multilinear
		// B(x) = sum_i eq(r_I^b, i) * b(i, x), then re-randomize its claim B(r_2) = b_recomb from
		// `b_eval_point` (r_2) to the shared point via a single-claim MLE-eval check. This
		// replaces the 2^k separate b rerandomizations with the spec's single recombined claim.
		assert_eq!(b_exponents.len(), 1 << b_eval_point.len());

		let b_words = b_exponents
			.iter()
			.map(|&word| Word::from_u64(word))
			.collect::<Vec<_>>();

		let b_tensor = eq_ind_partial_eval_scalars::<F>(r_ib);
		let b_folded = fold_words::<_, P>(&b_words, &b_tensor);

		let b_eval_prover = MultilinearEvalProver::new(b_folded, b_eval_point, b_recomb)?;
		let b_sumcheck_prover = MleToSumCheckDecorator::new(b_eval_prover);

		// Batch prove all three provers.
		let BatchSumcheckOutput {
			challenges,
			multilinear_evals,
		} = batch_prove(
			vec![
				Either::Left(bivariate_sumcheck_prover),
				Either::Right(Either::Left(overflow_prover)),
				Either::Right(Either::Right(b_sumcheck_prover)),
			],
			self.channel,
		)?;

		// Pull out the evals of all three provers. The b prover is now a single-claim MLE-eval
		// check, so it yields one recombined eval B(r_x) rather than 2^k per-bit evals.
		let [mut bivariate_evals, lsb_evals, b_recomb_evals] = multilinear_evals
			.try_into()
			.expect("batch_prove with 3 provers returns 3 multilinear_evals vecs");

		assert_eq!(bivariate_evals.len(), 3 << log_bits);
		assert_eq!(lsb_evals.len(), 3);
		assert_eq!(b_recomb_evals.len(), 1);

		// The prover still sends the 2^k raw per-bit evals b(i, r_x) for Phase-5 leaf
		// reconstruction; the verifier binds them via sum_i eq(r_I^b, i) * b(i, r_x) = B(r_x).
		let b_evals = fold_across_words::<_, P>(&b_words, &challenges).to_vec();

		// Sanity: the single recombined rerandomization eval B(r_x) equals the recombination of
		// the raw per-bit evals.
		debug_assert_eq!(
			b_recomb_evals[0],
			evaluate(&FieldBuffer::<P>::from_values(&b_evals), r_ib)
		);

		// The bivariate prover flattens its `(leaf[z], leaf[z + half])` pairs pair-major, so each
		// tree's leaf evals come out interleaved as `[bit 0, bit half, bit 1, bit half+1, ...]`.
		// De-interleave them back into bit order for `normalize_a_c_exponent_evals`.
		let selected_c_hi_evals = deinterleave_pairs(bivariate_evals.split_off(2 << log_bits));
		let selected_c_lo_evals = deinterleave_pairs(bivariate_evals.split_off(1 << log_bits));
		let selected_a_evals = deinterleave_pairs(bivariate_evals);

		// Recover the raw per-bit evaluations from the leaf selectors and send those (spec Phase
		// 5). The verifier reconstructs the selectors forward rather than receiving them and
		// inverting.
		let [a_evals, c_lo_evals, c_hi_evals] = normalize_a_c_exponent_evals(
			log_bits,
			selected_a_evals,
			selected_c_lo_evals,
			selected_c_hi_evals,
		);

		self.channel.send_many(&a_evals);
		self.channel.send_many(&c_lo_evals);
		self.channel.send_many(&c_hi_evals);
		self.channel.send_many(&b_evals);

		let [a_0_eval, b_0_eval, c_lo_0_eval] =
			lsb_evals.try_into().expect("c_lo_prover_evals.len() == 3");

		debug_assert_eq!(a_0_eval, a_evals[0]);
		debug_assert_eq!(b_0_eval, b_evals[0]);
		debug_assert_eq!(c_lo_0_eval, c_lo_evals[0]);

		Ok(IntMulOutput {
			eval_point: challenges,
			a_evals,
			b_evals,
			c_lo_evals,
			c_hi_evals,
		})
	}
}

/// Splits a prodcheck prover's retained leaf layer — one `(n_vars + log_bits)`-variate buffer with
/// the node index in the high bits — into its `2^log_bits` per-node `n_vars`-variate buffers, in
/// node order.
fn split_leaf_layer<P: PackedField>(layer: FieldBuffer<P>, n_vars: usize) -> Vec<FieldBuffer<P>> {
	let scalars = layer.iter_scalars().collect::<Vec<_>>();
	let n_nodes = scalars.len() >> n_vars;
	(0..n_nodes)
		.map(|z| FieldBuffer::<P>::from_values(&scalars[z << n_vars..(z + 1) << n_vars]))
		.collect()
}

/// Pairs the leaf layer's node `z` with node `z + half` (the highest-bit split), reproducing the
/// bivariate-product pairing of the GKR tree's final layer. The layer is in natural node order, so
/// this pairs bits `z` and `z + half` (a strided pairing).
fn split_pairs<P: PackedField>(layer: Vec<FieldBuffer<P>>) -> Vec<[FieldBuffer<P>; 2]> {
	let half = layer.len() / 2;
	let (lo, hi) = layer.split_at(half);
	iter::zip(lo.to_vec(), hi.to_vec())
		.map(|(a, b)| [a, b])
		.collect()
}

/// De-interleave a tree's leaf evals from the bivariate prover's pair-major order back into bit
/// order.
///
/// The bivariate prover pairs leaf `z` with leaf `z + half` (the strided pairing of
/// [`split_pairs`]) and flattens the pairs, so its leaf evals come out interleaved as
/// `[bit 0, bit half, bit 1, bit half+1, ...]`: the even slots hold bits `0..half` and the odd
/// slots hold bits `half..2·half`. Splitting the evens from the odds restores bit order
/// `[bit 0, bit 1, ..., bit (2·half - 1)]`.
fn deinterleave_pairs<F: Clone>(evals: Vec<F>) -> Vec<F> {
	let evens = evals.iter().step_by(2);
	let odds = evals.iter().skip(1).step_by(2);
	evens.chain(odds).cloned().collect()
}
