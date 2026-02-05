// Copyright 2025-2026 The Binius Developers

use std::{array, cmp::max};

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldBuffer, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::prelude::*;

use crate::sumcheck::{
	Error,
	common::{MleCheckProver, SumcheckProver},
	gruen32::Gruen32,
	round_evals::RoundEvals2,
};

#[derive(Debug)]
pub enum SharedFracAddInput<P: PackedField, const N: usize> {
	CommonNumerator {
		num_0: FieldBuffer<P>,
		num_1: FieldBuffer<P>,
		den_0: [FieldBuffer<P>; N],
		den_1: [FieldBuffer<P>; N],
	},
	CommonDenominator {
		den_0: FieldBuffer<P>,
		den_1: FieldBuffer<P>,
		num_0: [FieldBuffer<P>; N],
		num_1: [FieldBuffer<P>; N],
	},
}

pub struct SharedFracAddLastLayerProver<P: PackedField, const N: usize> {
	input: SharedFracAddInput<P, N>,
	last: RoundCoeffsOrEvals<P::Scalar, N>,
	gruen32: Gruen32<P>,
}

impl<F, P, const N: usize> SharedFracAddLastLayerProver<P, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	pub fn new(
		input: SharedFracAddInput<P, N>,
		eval_point: Vec<F>,
		num_evals: [F; N],
		den_evals: [F; N],
	) -> Result<Self, Error> {
		let n_vars = eval_point.len();
		match &input {
			SharedFracAddInput::CommonNumerator {
				num_0,
				num_1,
				den_0,
				den_1,
			} => {
				if num_0.log_len() != n_vars
					|| num_1.log_len() != n_vars
					|| den_0.iter().any(|buf| buf.log_len() != n_vars)
					|| den_1.iter().any(|buf| buf.log_len() != n_vars)
				{
					return Err(Error::MultilinearSizeMismatch);
				}
			}
			SharedFracAddInput::CommonDenominator {
				den_0,
				den_1,
				num_0,
				num_1,
			} => {
				if den_0.log_len() != n_vars
					|| den_1.log_len() != n_vars
					|| num_0.iter().any(|buf| buf.log_len() != n_vars)
					|| num_1.iter().any(|buf| buf.log_len() != n_vars)
				{
					return Err(Error::MultilinearSizeMismatch);
				}
			}
		}

		Ok(Self {
			input,
			last: RoundCoeffsOrEvals::Evals {
				num: num_evals,
				den: den_evals,
			},
			gruen32: Gruen32::new(&eval_point),
		})
	}
}

impl<F, P, const N: usize> SumcheckProver<F> for SharedFracAddLastLayerProver<P, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	fn n_vars(&self) -> usize {
		self.gruen32.n_vars_remaining()
	}

	fn n_claims(&self) -> usize {
		2 * N
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
		let (last_num, last_den) = match &self.last {
			RoundCoeffsOrEvals::Evals { num, den } => (*num, *den),
			RoundCoeffsOrEvals::Coeffs { .. } => return Err(Error::ExpectedFold),
		};

		let n_vars_remaining = self.gruen32.n_vars_remaining();
		assert!(n_vars_remaining > 0);

		let eq_expansion = self.gruen32.eq_expansion();
		assert_eq!(eq_expansion.log_len(), n_vars_remaining - 1);

		const MAX_CHUNK_VARS: usize = 8;
		let chunk_vars = max(MAX_CHUNK_VARS, P::LOG_WIDTH).min(n_vars_remaining - 1);
		let chunk_count = 1 << (n_vars_remaining - 1 - chunk_vars);

		let (y_1_num, y_inf_num, y_1_den, y_inf_den) = match &self.input {
			SharedFracAddInput::CommonNumerator {
				num_0,
				num_1,
				den_0,
				den_1,
			} => {
				let (num_0_lo, num_0_hi) = num_0.split_half_ref();
				let (num_1_lo, num_1_hi) = num_1.split_half_ref();
				let den_0_splits = den_0.each_ref().map(FieldBuffer::split_half_ref);
				let den_1_splits = den_1.each_ref().map(FieldBuffer::split_half_ref);

				(0..chunk_count)
					.into_par_iter()
					.try_fold(
						|| {
							(
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
							)
						},
						|mut packed_prime_evals, chunk_index| -> Result<_, Error> {
							let eq_chunk = eq_expansion.chunk(chunk_vars, chunk_index);
							let num_0_lo_chunk = num_0_lo.chunk(chunk_vars, chunk_index);
							let num_0_hi_chunk = num_0_hi.chunk(chunk_vars, chunk_index);
							let num_1_lo_chunk = num_1_lo.chunk(chunk_vars, chunk_index);
							let num_1_hi_chunk = num_1_hi.chunk(chunk_vars, chunk_index);

							let den_0_lo_chunk: [FieldBuffer<
								P,
								binius_math::field_buffer::FieldSliceData<'_, P>,
							>; N] = array::from_fn(|i| {
								let (lo, _) = &den_0_splits[i];
								lo.chunk(chunk_vars, chunk_index)
							});
							let den_0_hi_chunk: [FieldBuffer<
								P,
								binius_math::field_buffer::FieldSliceData<'_, P>,
							>; N] = array::from_fn(|i| {
								let (_, hi) = &den_0_splits[i];
								hi.chunk(chunk_vars, chunk_index)
							});
							let den_1_lo_chunk: [FieldBuffer<
								P,
								binius_math::field_buffer::FieldSliceData<'_, P>,
							>; N] = array::from_fn(|i| {
								let (lo, _) = &den_1_splits[i];
								lo.chunk(chunk_vars, chunk_index)
							});
							let den_1_hi_chunk: [FieldBuffer<
								P,
								binius_math::field_buffer::FieldSliceData<'_, P>,
							>; N] = array::from_fn(|i| {
								let (_, hi) = &den_1_splits[i];
								hi.chunk(chunk_vars, chunk_index)
							});

							let (y_1_num, y_inf_num, y_1_den, y_inf_den) = &mut packed_prime_evals;
							for (idx, &eq_i) in eq_chunk.as_ref().iter().enumerate() {
								let num_0_1 = num_0_hi_chunk.as_ref()[idx];
								let num_1_1 = num_1_hi_chunk.as_ref()[idx];
								let num_0_inf = num_0_lo_chunk.as_ref()[idx] + num_0_1;
								let num_1_inf = num_1_lo_chunk.as_ref()[idx] + num_1_1;

								for i in 0..N {
									let den_0_1 = den_0_hi_chunk[i].as_ref()[idx];
									let den_1_1 = den_1_hi_chunk[i].as_ref()[idx];
									let den_0_inf = den_0_lo_chunk[i].as_ref()[idx] + den_0_1;
									let den_1_inf = den_1_lo_chunk[i].as_ref()[idx] + den_1_1;

									let numerator_y1 = num_0_1 * den_1_1 + num_1_1 * den_0_1;
									let denominator_y1 = den_0_1 * den_1_1;
									let numerator_yinf =
										num_0_inf * den_1_inf + num_1_inf * den_0_inf;
									let denominator_yinf = den_0_inf * den_1_inf;

									y_1_num[i] += numerator_y1 * eq_i;
									y_inf_num[i] += numerator_yinf * eq_i;
									y_1_den[i] += denominator_y1 * eq_i;
									y_inf_den[i] += denominator_yinf * eq_i;
								}
							}

							Ok(packed_prime_evals)
						},
					)
					.try_reduce(
						|| {
							(
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
							)
						},
						|lhs, rhs| {
							let mut out = (
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
							);
							for i in 0..N {
								out.0[i] = lhs.0[i] + rhs.0[i];
								out.1[i] = lhs.1[i] + rhs.1[i];
								out.2[i] = lhs.2[i] + rhs.2[i];
								out.3[i] = lhs.3[i] + rhs.3[i];
							}
							Ok(out)
						},
					)?
			}
			SharedFracAddInput::CommonDenominator {
				den_0,
				den_1,
				num_0,
				num_1,
			} => {
				let (den_0_lo, den_0_hi) = den_0.split_half_ref();
				let (den_1_lo, den_1_hi) = den_1.split_half_ref();
				let num_0_splits = num_0.each_ref().map(FieldBuffer::split_half_ref);
				let num_1_splits = num_1.each_ref().map(FieldBuffer::split_half_ref);

				(0..chunk_count)
					.into_par_iter()
					.try_fold(
						|| {
							(
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
							)
						},
						|mut packed_prime_evals, chunk_index| -> Result<_, Error> {
							let eq_chunk = eq_expansion.chunk(chunk_vars, chunk_index);
							let den_0_lo_chunk = den_0_lo.chunk(chunk_vars, chunk_index);
							let den_0_hi_chunk = den_0_hi.chunk(chunk_vars, chunk_index);
							let den_1_lo_chunk = den_1_lo.chunk(chunk_vars, chunk_index);
							let den_1_hi_chunk = den_1_hi.chunk(chunk_vars, chunk_index);

							let num_0_lo_chunk: [FieldBuffer<
								P,
								binius_math::field_buffer::FieldSliceData<'_, P>,
							>; N] = array::from_fn(|i| {
								let (lo, _) = &num_0_splits[i];
								lo.chunk(chunk_vars, chunk_index)
							});
							let num_0_hi_chunk: [FieldBuffer<
								P,
								binius_math::field_buffer::FieldSliceData<'_, P>,
							>; N] = array::from_fn(|i| {
								let (_, hi) = &num_0_splits[i];
								hi.chunk(chunk_vars, chunk_index)
							});
							let num_1_lo_chunk: [FieldBuffer<
								P,
								binius_math::field_buffer::FieldSliceData<'_, P>,
							>; N] = array::from_fn(|i| {
								let (lo, _) = &num_1_splits[i];
								lo.chunk(chunk_vars, chunk_index)
							});
							let num_1_hi_chunk: [FieldBuffer<
								P,
								binius_math::field_buffer::FieldSliceData<'_, P>,
							>; N] = array::from_fn(|i| {
								let (_, hi) = &num_1_splits[i];
								hi.chunk(chunk_vars, chunk_index)
							});

							let (y_1_num, y_inf_num, y_1_den, y_inf_den) = &mut packed_prime_evals;
							for (idx, &eq_i) in eq_chunk.as_ref().iter().enumerate() {
								let den_0_1 = den_0_hi_chunk.as_ref()[idx];
								let den_1_1 = den_1_hi_chunk.as_ref()[idx];
								let den_0_inf = den_0_lo_chunk.as_ref()[idx] + den_0_1;
								let den_1_inf = den_1_lo_chunk.as_ref()[idx] + den_1_1;

								for i in 0..N {
									let num_0_1 = num_0_hi_chunk[i].as_ref()[idx];
									let num_1_1 = num_1_hi_chunk[i].as_ref()[idx];
									let num_0_inf = num_0_lo_chunk[i].as_ref()[idx] + num_0_1;
									let num_1_inf = num_1_lo_chunk[i].as_ref()[idx] + num_1_1;

									let numerator_y1 = num_0_1 * den_1_1 + num_1_1 * den_0_1;
									let denominator_y1 = den_0_1 * den_1_1;
									let numerator_yinf =
										num_0_inf * den_1_inf + num_1_inf * den_0_inf;
									let denominator_yinf = den_0_inf * den_1_inf;

									y_1_num[i] += numerator_y1 * eq_i;
									y_inf_num[i] += numerator_yinf * eq_i;
									y_1_den[i] += denominator_y1 * eq_i;
									y_inf_den[i] += denominator_yinf * eq_i;
								}
							}

							Ok(packed_prime_evals)
						},
					)
					.try_reduce(
						|| {
							(
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
							)
						},
						|lhs, rhs| {
							let mut out = (
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
								[P::default(); N],
							);
							for i in 0..N {
								out.0[i] = lhs.0[i] + rhs.0[i];
								out.1[i] = lhs.1[i] + rhs.1[i];
								out.2[i] = lhs.2[i] + rhs.2[i];
								out.3[i] = lhs.3[i] + rhs.3[i];
							}
							Ok(out)
						},
					)?
			}
		};

		let alpha = self.gruen32.next_coordinate();
		let mut num_coeffs = array::from_fn(|_| RoundCoeffs::default());
		let mut den_coeffs = array::from_fn(|_| RoundCoeffs::default());

		for i in 0..N {
			let round_num = RoundEvals2 {
				y_1: y_1_num[i],
				y_inf: y_inf_num[i],
			}
			.sum_scalars(n_vars_remaining)
			.interpolate_eq(last_num[i], alpha);
			let round_den = RoundEvals2 {
				y_1: y_1_den[i],
				y_inf: y_inf_den[i],
			}
			.sum_scalars(n_vars_remaining)
			.interpolate_eq(last_den[i], alpha);
			num_coeffs[i] = round_num;
			den_coeffs[i] = round_den;
		}

		let mut out = Vec::with_capacity(2 * N);
		for i in 0..N {
			out.push(num_coeffs[i].clone());
			out.push(den_coeffs[i].clone());
		}

		self.last = RoundCoeffsOrEvals::Coeffs {
			num: num_coeffs,
			den: den_coeffs,
		};

		Ok(out)
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		let (num_coeffs, den_coeffs) = match &self.last {
			RoundCoeffsOrEvals::Coeffs { num, den } => (num, den),
			RoundCoeffsOrEvals::Evals { .. } => return Err(Error::ExpectedExecute),
		};

		assert!(
			self.n_vars() > 0,
			"n_vars is decremented in fold; \
			fold changes last to Eval variant; \
			fold only executes with Coeffs variant; \
			thus, n_vars should be > 0"
		);

		let num_evals = array::from_fn(|i| num_coeffs[i].evaluate(challenge));
		let den_evals = array::from_fn(|i| den_coeffs[i].evaluate(challenge));

		match &mut self.input {
			SharedFracAddInput::CommonNumerator {
				num_0,
				num_1,
				den_0,
				den_1,
			} => {
				fold_highest_var_inplace(num_0, challenge);
				fold_highest_var_inplace(num_1, challenge);
				for i in 0..N {
					fold_highest_var_inplace(&mut den_0[i], challenge);
					fold_highest_var_inplace(&mut den_1[i], challenge);
				}
			}
			SharedFracAddInput::CommonDenominator {
				den_0,
				den_1,
				num_0,
				num_1,
			} => {
				fold_highest_var_inplace(den_0, challenge);
				fold_highest_var_inplace(den_1, challenge);
				for i in 0..N {
					fold_highest_var_inplace(&mut num_0[i], challenge);
					fold_highest_var_inplace(&mut num_1[i], challenge);
				}
			}
		}

		self.gruen32.fold(challenge);
		self.last = RoundCoeffsOrEvals::Evals {
			num: num_evals,
			den: den_evals,
		};
		Ok(())
	}

	fn finish(self) -> Result<Vec<F>, Error> {
		if self.n_vars() > 0 {
			let error = match self.last {
				RoundCoeffsOrEvals::Coeffs { .. } => Error::ExpectedFold,
				RoundCoeffsOrEvals::Evals { .. } => Error::ExpectedExecute,
			};
			return Err(error);
		}

		let mut evals = Vec::with_capacity(4 * N);
		match self.input {
			SharedFracAddInput::CommonNumerator {
				num_0,
				num_1,
				den_0,
				den_1,
			} => {
				let num_0 = num_0.get(0);
				let num_1 = num_1.get(0);
				for i in 0..N {
					evals.push(num_0);
					evals.push(num_1);
					evals.push(den_0[i].get(0));
					evals.push(den_1[i].get(0));
				}
			}
			SharedFracAddInput::CommonDenominator {
				den_0,
				den_1,
				num_0,
				num_1,
			} => {
				let den_0 = den_0.get(0);
				let den_1 = den_1.get(0);
				for i in 0..N {
					evals.push(num_0[i].get(0));
					evals.push(num_1[i].get(0));
					evals.push(den_0);
					evals.push(den_1);
				}
			}
		}

		Ok(evals)
	}
}

impl<F, P, const N: usize> MleCheckProver<F> for SharedFracAddLastLayerProver<P, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	fn eval_point(&self) -> &[F] {
		&self.gruen32.eval_point()[..self.n_vars()]
	}
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrEvals<F: Field, const N: usize> {
	Coeffs {
		num: [RoundCoeffs<F>; N],
		den: [RoundCoeffs<F>; N],
	},
	Evals {
		num: [F; N],
		den: [F; N],
	},
}
