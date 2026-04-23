// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{BinaryField, FieldOps, PackedField, WideningMul};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{batch::batch_prove, quadratic_mle::QuadraticMleCheckProver},
};
use binius_math::{
	field_buffer::FieldBuffer,
	inner_product::inner_product_wide_buffers,
	multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
};
use binius_utils::{bitwise::Bitwise, rayon::prelude::*};
use binius_verifier::protocols::{
	intmul::common::{
		IntMulOutput, Phase1Output, Phase2Output, Phase3Output, Phase4Output, frobenius_twist,
		normalize_a_c_exponent_evals,
	},
	prodcheck::MultilinearEvalClaim,
};
use either::Either;
use itertools::{Itertools, chain, izip};

use super::{
	error::Error,
	witness::{Witness, two_valued_field_buffer},
};
use crate::protocols::{
	prodcheck::ProdcheckProver,
	sumcheck::{
		MleToSumCheckDecorator,
		batch::{BatchSumcheckOutput, batch_prove_and_write_evals},
		bivariate_product_mle,
		bivariate_product_multi_mle::BivariateProductMultiMlecheckProver,
		rerand_mle::RerandMlecheckProver,
		selector_mle::{Claim, SelectorMlecheckProver},
	},
};

/// A helper structure that encapsulates switchover settings and the prover channel for
/// the integer multiplication protocol.
pub struct IntMulProver<'a, P, B, S, Channel> {
	_p_marker: PhantomData<P>,
	_b_marker: PhantomData<B>,
	_s_marker: PhantomData<S>,

	switchover: usize,
	channel: &'a mut Channel,
}

impl<'a, P, B, S, Channel> IntMulProver<'a, P, B, S, Channel> {
	pub fn new(switchover: usize, channel: &'a mut Channel) -> Self {
		Self {
			_p_marker: PhantomData,
			_b_marker: PhantomData,
			_s_marker: PhantomData,
			switchover,
			channel,
		}
	}
}

impl<'a, F, P, B, S, Channel> IntMulProver<'a, P, B, S, Channel>
where
	F: BinaryField,
	P: PackedField<Scalar = F> + WideningMul,
	B: Bitwise,
	S: AsRef<[B]> + Sync,
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
	///        and $b$
	///      - First layer of GPA reduction for the `c_lo || c_hi` combined `c` tree
	///    - Phase 4: Batching all but last layers and `a`, `c_lo` and `c_hi`
	///    - Phase 5: Proving the last (widest) layers of `a`, `c_lo` and `c_hi` batched with
	///      rerandomization degree-1 mlecheck on `b` evaluations from phase 3
	///
	/// The output of this protocol is a set of evaluation claims on the `b` selectors representing
	/// all of `a`, `b`, `c_lo` and `c_hi` as column-major bit matrices, at a common evaluation
	/// point.
	pub fn prove(&mut self, witness: Witness<P, B, S>) -> Result<IntMulOutput<F>, Error> {
		let Witness {
			a,
			b_exponents,
			b_leaves,
			b_prodcheck,
			b_root: _,
			c_lo,
			c_hi,
			c_root,
		} = witness;

		let (n_vars, log_bits) = (c_root.log_len(), a.log_bits());
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

		// Splitting
		let (a_exponents, a_root, mut a_layers) = a.split();
		let (c_lo_exponents, c_lo_root, mut c_lo_layers) = c_lo.split();
		let (_, c_hi_root, mut c_hi_layers) = c_hi.split();

		let a_last_layer = a_layers.pop().expect("log_bits >= 1");
		let c_lo_last_layer = c_lo_layers.pop().expect("log_bits >= 1");
		let c_hi_last_layer = c_hi_layers.pop().expect("log_bits >= 1");

		// Phase 3
		let Phase3Output {
			eval_point: phase3_eval_point,
			b_evals,
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
		let Phase4Output {
			eval_point: phase4_eval_point,
			a_evals,
			c_lo_evals,
			c_hi_evals,
		} = self.phase4(
			log_bits,
			&phase3_eval_point,
			(gpow_a_eval, a_layers.into_iter()),
			(gpow_c_lo_eval, c_lo_layers.into_iter()),
			(gpow_c_hi_eval, c_hi_layers.into_iter()),
		)?;

		// Phase 5
		self.phase5(
			log_bits,
			&phase4_eval_point,
			(&a_evals, a_last_layer),
			(&c_lo_evals, c_lo_last_layer),
			(&c_hi_evals, c_hi_last_layer),
			b_exponents.as_ref(),
			&phase3_eval_point,
			&b_evals,
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
			.map(|b_leaf| inner_product_wide_buffers(&b_leaf, &x_tensor))
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
		b_exponents: &[B],
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

		let selector_prover =
			SelectorMlecheckProver::new(selector, selector_claims, b_exponents, self.switchover)?;

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

		Ok(Phase3Output {
			eval_point: challenges,
			b_evals,
			gpow_a_eval,
			gpow_c_lo_eval,
			gpow_c_hi_eval,
		})
	}

	fn phase4(
		&mut self,
		log_bits: usize,
		eval_point: &[F],
		(a_root_eval, a_layers): (F, impl ExactSizeIterator<Item = Vec<FieldBuffer<P>>>),
		(gpow_c_lo_eval, c_lo_layers): (F, impl ExactSizeIterator<Item = Vec<FieldBuffer<P>>>),
		(gpow_c_hi_eval, c_hi_layers): (F, impl ExactSizeIterator<Item = Vec<FieldBuffer<P>>>),
	) -> Result<Phase4Output<F>, Error> {
		assert_eq!(a_layers.len(), log_bits - 1);
		assert_eq!(c_lo_layers.len(), log_bits - 1);
		assert_eq!(c_hi_layers.len(), log_bits - 1);

		let mut eval_point = eval_point.to_vec();
		let mut evals = vec![a_root_eval, gpow_c_lo_eval, gpow_c_hi_eval];

		for (depth, (a_l, c_lo_l, c_hi_l)) in izip!(a_layers, c_lo_layers, c_hi_layers).enumerate()
		{
			assert_eq!(a_l.len(), 2 << depth);
			assert_eq!(c_lo_l.len(), 2 << depth);
			assert_eq!(c_hi_l.len(), 2 << depth);
			assert_eq!(evals.len(), 3 << depth);

			let layer = a_l.into_iter().chain(c_lo_l).chain(c_hi_l);
			let sumcheck_prover = BivariateProductMultiMlecheckProver::new(
				make_pairs(layer),
				&eval_point,
				evals.clone(),
			)?;

			let prover = MleToSumCheckDecorator::new(sumcheck_prover);

			let BatchSumcheckOutput {
				challenges,
				mut multilinear_evals,
			} = batch_prove_and_write_evals(vec![prover], self.channel)?;

			assert_eq!(multilinear_evals.len(), 1);
			eval_point = challenges;
			evals = multilinear_evals
				.pop()
				.expect("multilinear_evals.len() == 1");
		}

		debug_assert_eq!(evals.len(), 3 << (log_bits - 1));
		let c_hi_evals = evals.split_off(2 << (log_bits - 1));
		let c_lo_evals = evals.split_off(1 << (log_bits - 1));
		let a_evals = evals;

		Ok(Phase4Output {
			eval_point,
			a_evals,
			c_lo_evals,
			c_hi_evals,
		})
	}

	#[allow(clippy::too_many_arguments)]
	fn phase5(
		&mut self,
		log_bits: usize,
		a_c_eval_point: &[F],
		(a_evals, a_layer): (&[F], Vec<FieldBuffer<P>>),
		(c_lo_evals, c_lo_layer): (&[F], Vec<FieldBuffer<P>>),
		(c_hi_evals, c_hi_layer): (&[F], Vec<FieldBuffer<P>>),
		b_exponents: &[B],
		b_eval_point: &[F],
		b_evals_pre: &[F],
		// Needed for the zerocheck on `a_0 * b_0 = c_lo_0`.
		a_exponents: &[B],
		c_lo_exponents: &[B],
	) -> Result<IntMulOutput<F>, Error> {
		assert!(log_bits >= 1);
		assert_eq!(1 << log_bits, a_layer.len());
		assert_eq!(2 * a_evals.len(), a_layer.len());
		assert_eq!(2 * c_lo_evals.len(), c_lo_layer.len());
		assert_eq!(2 * c_hi_evals.len(), c_hi_layer.len());
		assert_eq!(b_eval_point.len(), a_layer.first().expect("log_bits >= 1").log_len());
		assert_eq!(a_c_eval_point.len(), b_eval_point.len());

		// Make the `BivariateProductMultiMlecheckProver` prover.
		// The prover proves an MLE eval claim on each pair of adjacent multilinears
		// in the `multilinears` iterator below.
		let multilinears = chain!(a_layer, c_lo_layer, c_hi_layer);
		let evals = [a_evals, c_lo_evals, c_hi_evals].concat();

		let bivariate_mle_prover = BivariateProductMultiMlecheckProver::new(
			make_pairs(multilinears),
			a_c_eval_point,
			evals,
		)?;
		let bivariate_sumcheck_prover = MleToSumCheckDecorator::new(bivariate_mle_prover);

		// Embed `a_0` and `b_0` bits into field buffers for `BivariateProductMultiMlecheckProver`.
		let binary_elements = [F::zero(), F::one()];

		// TODO: Use a special 1-bit-optimized MLE-check with switchover to save memory.
		let a_0: FieldBuffer<P> = two_valued_field_buffer(0, &a_exponents, binary_elements);
		let b_0: FieldBuffer<P> = two_valued_field_buffer(0, &b_exponents, binary_elements);
		let c_lo_0: FieldBuffer<P> = two_valued_field_buffer(0, &c_lo_exponents, binary_elements);

		// Make the sumcheck prover for the overflow parity check.
		let overflow_prover =
			MleToSumCheckDecorator::new(QuadraticMleCheckProver::<P, _, _, 3>::new(
				[a_0, b_0, c_lo_0],
				|[a, b, c]| a * b - c,
				|[a, b, _c]| a * b,
				a_c_eval_point.to_vec(),
				F::ZERO,
			)?);

		// Make the `RerandMlecheckProver` for `b_exponents`.
		assert_eq!(b_exponents.len(), 1 << b_eval_point.len());
		assert_eq!(b_evals_pre.len(), 1 << log_bits);

		let b_rerand_prover = RerandMlecheckProver::<P, _>::new(
			b_eval_point,
			b_evals_pre,
			b_exponents,
			self.switchover,
		)?;
		let b_sumcheck_prover = MleToSumCheckDecorator::new(b_rerand_prover);

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

		// Pull out the evals of all three provers.
		let [mut bivariate_evals, lsb_evals, b_evals] = multilinear_evals
			.try_into()
			.expect("batch_prove with 3 provers returns 3 multilinear_evals vecs");

		assert_eq!(bivariate_evals.len(), 3 << log_bits);
		assert_eq!(lsb_evals.len(), 3);
		assert_eq!(b_evals.len(), 1 << log_bits);

		let selected_c_hi_evals = bivariate_evals.split_off(2 << log_bits);
		let selected_c_lo_evals = bivariate_evals.split_off(1 << log_bits);
		let selected_a_evals = bivariate_evals;

		self.channel.send_many(&selected_a_evals);
		self.channel.send_many(&selected_c_lo_evals);
		self.channel.send_many(&selected_c_hi_evals);
		self.channel.send_many(&b_evals);

		let [a_evals, c_lo_evals, c_hi_evals] = normalize_a_c_exponent_evals(
			log_bits,
			selected_a_evals,
			selected_c_lo_evals,
			selected_c_hi_evals,
		);

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

fn make_pairs<T>(layer: impl IntoIterator<Item = T>) -> Vec<[T; 2]> {
	layer
		.into_iter()
		.chunks(2)
		.into_iter()
		.map(|chunk| chunk.collect_array().expect("chunk.len() == 2"))
		.collect()
}
