// Copyright 2026 The Binius Developers

//! The batched final layer of the table side.
//!
//! This batches the last fractional-addition GKR layer with the product sumcheck `<T, Y> = e`.
//! Both reductions end in an evaluation of the pushforward `Y`.
//! Running them as one `(m-1)`-variable sumcheck plus one shared line-fold gives one evaluation
//! point. That collapses the two `Y` evaluations into one.

use binius_field::{
	BinaryField1b, ExtensionField, Field, arithmetic_traits::Square, field::FieldOps,
};
use binius_math::{line::extrapolate_line, multilinear::eq::eq_ind};

use super::error::{Error, VerificationError};
use crate::{
	channel::IPVerifierChannel,
	sumcheck::{self, BatchSumcheckOutput},
};

/// The output of the batched final layer.
pub struct FinalLayer<F> {
	/// The `m`-coordinate point at which both `T` and `Y` are evaluated.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluation of `T` at the point.
	pub table_eval_claim: F,
	/// The claimed evaluation of `Y` at the point.
	pub pushforward_eval_claim: F,
}

/// Verify the batched final layer of the table side.
///
/// Batches three sum claims over the `m-1`-variable cube, then folds the highest variable once:
///
/// ```text
///     S_1 = sum_{x'} eq(x'; Z) * (Y_0 * D_1 + Y_1 * D_0)(x') = num_1(Z)
///     S_2 = sum_{x'} eq(x'; Z) * (D_0 * D_1)(x')             = den_1(Z)
///     S_3 = sum_{x'} (T_0 * Y_0 + T_1 * Y_1)(x')             = e
/// ```
///
/// The three claims are:
///
/// - `S_1` and `S_2`: the layer-1 numerator and denominator of the fractional-addition circuit.
/// - `S_3`: the product claim `<T, Y> = e`, split on the highest variable.
///
/// A shared challenge point and a shared line-fold collapse all three to one evaluation point.
/// That gives one evaluation of the pushforward `Y` and one evaluation of the table `T`.
///
/// The summand has degree 3 per variable.
/// It is the `eq` factor (degree 1) times a product of two halves (degree 2).
///
/// # Arguments
///
/// * `m` - The number of table variables.
/// * `c` - The logUp challenge, defining the table-side denominator `D = c - J`.
/// * `eval_claim` - The product claim `e = <T, Y>`.
/// * `layer1_num` - The layer-1 numerator claim `num_1(Z)`.
/// * `layer1_den` - The layer-1 denominator claim `den_1(Z)`.
/// * `layer1_point` - The layer-1 point `Z` of length `m-1`.
/// * `channel` - The verifier channel.
pub fn verify_final_layer<F, C>(
	m: usize,
	c: C::Elem,
	eval_claim: C::Elem,
	layer1_num: C::Elem,
	layer1_den: C::Elem,
	layer1_point: &[C::Elem],
	channel: &mut C,
) -> Result<FinalLayer<C::Elem>, Error>
where
	F: Field + ExtensionField<BinaryField1b>,
	C: IPVerifierChannel<F>,
	C::Elem: From<F>,
{
	// Batch the three sum claims with powers of a single coefficient and run the sumcheck.
	//
	//     sum = num_1(Z) + batch_coeff * den_1(Z) + batch_coeff^2 * e
	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		mut challenges,
	} = sumcheck::batch_verify::<F, C>(m - 1, 3, &[layer1_num, layer1_den, eval_claim], channel)?;

	// Read the leaf halves of the pushforward and the table at the sumcheck challenge point.
	//
	//     Y_0 = Y(rho, 0),  Y_1 = Y(rho, 1)   (split on the highest variable)
	//     T_0 = T(rho, 0),  T_1 = T(rho, 1)
	let [y_0, y_1, t_0, t_1] = channel
		.recv_array()
		.map_err(|_| VerificationError::TranscriptIsEmpty)?;

	// Sumcheck binds variables highest-to-lowest; reverse to align with the low-to-high point Z.
	challenges.reverse();
	let rho = challenges;

	// The eq factor of the layer-1 mle-check, evaluated by the verifier.
	let eq_eval = eq_ind::<C::Elem>(&rho, layer1_point);

	// Table-side denominator halves D_0, D_1 are public: D = c - J with J(x) = sum_t basis(t) *
	// x_t.
	let (d_0, d_1) = denominator_halves::<F, C::Elem>(&c, &rho, m);

	// Reconstruct the batched summand at the challenge point and check it equals the reduced eval.
	//
	//     num relation : eq * (Y_0 * D_1 + Y_1 * D_0)      (fractional-addition numerator)
	//     den relation : eq * (D_0 * D_1)                  (fractional-addition denominator)
	//     prod relation: T_0 * Y_0 + T_1 * Y_1             (leaf product of <T, Y>)
	let num_relation = y_0.clone() * d_1.clone() + y_1.clone() * d_0.clone();
	let den_relation = d_0 * d_1;
	let prod_relation = t_0.clone() * y_0.clone() + t_1.clone() * y_1.clone();
	// Both fractional-addition relations carry the same eq factor, so factor it out.
	//
	//     expected = eq * (num_relation + batch_coeff * den_relation)
	//              + batch_coeff^2 * prod_relation
	let batch_coeff_sq = batch_coeff.clone().square();
	let expected =
		eq_eval * (num_relation + batch_coeff * den_relation) + batch_coeff_sq * prod_relation;
	channel
		.assert_zero(eval - expected)
		.map_err(|_| VerificationError::FinalLayerMismatch)?;

	// Fold the highest variable once to collapse the halves into single evaluations.
	let r = channel.sample();
	let pushforward_eval_claim = extrapolate_line(y_0, y_1, r.clone());
	let table_eval_claim = extrapolate_line(t_0, t_1, r.clone());

	// The shared point places the folded variable as the highest (last) coordinate.
	let mut table_eval_point = rho;
	table_eval_point.push(r);

	Ok(FinalLayer {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
	})
}

/// Compute the table-side denominator at the two halves of the highest variable.
///
/// The denominator multilinear is `D(x) = c - J(x)`.
/// The index embedding is `J(x) = sum_{t=0}^{m-1} basis(t) * x_t`.
///
/// Splitting on the highest variable `x_{m-1}` and evaluating the low part at `rho`:
///
/// ```text
///     J_low = sum_{t=0}^{m-2} basis(t) * rho_t
///     D_0 = c - J_low                       (x_{m-1} = 0)
///     D_1 = c - J_low - basis(m-1)          (x_{m-1} = 1)
/// ```
///
/// In characteristic 2 subtraction is addition, but the field operations are written generically.
///
/// # Arguments
///
/// * `c` - The logUp challenge.
/// * `rho` - The `m-1` low coordinates, in low-to-high order.
/// * `m` - The number of table variables.
fn denominator_halves<F, E>(c: &E, rho: &[E], m: usize) -> (E, E)
where
	F: ExtensionField<BinaryField1b>,
	E: FieldOps + From<F>,
{
	// J_low = sum_{t=0}^{m-2} basis(t) * rho_t over the low m-1 coordinates.
	let j_low = rho
		.iter()
		.enumerate()
		.map(|(t, rho_t)| {
			let basis_t = E::from(<F as ExtensionField<BinaryField1b>>::basis(t));
			basis_t * rho_t.clone()
		})
		.fold(E::zero(), |acc, term| acc + term);

	// The highest variable contributes basis(m-1) only when it is set to 1.
	let basis_high = E::from(<F as ExtensionField<BinaryField1b>>::basis(m - 1));

	let d_0 = c.clone() - j_low.clone();
	let d_1 = c.clone() - j_low - basis_high;
	(d_0, d_1)
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField1b, ExtensionField, Random, arch::OptimalB128 as B128};
	use binius_math::{multilinear::eq::eq_ind_partial_eval_scalars, test_utils::random_scalars};
	use rand::prelude::*;

	use super::*;

	// Embed a table position j into the field through the GF(2)-linear basis.
	//
	//     iota(j) = sum_{t : bit t of j is set} basis(t)
	fn iota(j: usize, m: usize) -> B128 {
		(0..m)
			.filter(|t| (j >> t) & 1 == 1)
			.map(<B128 as ExtensionField<BinaryField1b>>::basis)
			.fold(B128::ZERO, |acc, b| acc + b)
	}

	// Evaluate the multilinear `values` at `point` as the inner product with the eq tensor.
	//
	//     mle(point) = sum_j values[j] * eq(j, point)
	fn evaluate_scalars(values: &[B128], point: &[B128]) -> B128 {
		let eq = eq_ind_partial_eval_scalars(point);
		values
			.iter()
			.zip(&eq)
			.map(|(v, e)| *v * *e)
			.fold(B128::ZERO, |acc, t| acc + t)
	}

	fn check_denominator_halves(m: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		// Random logUp challenge and a random low point of m-1 coordinates.
		let c = B128::random(&mut rng);
		let rho = random_scalars::<B128>(&mut rng, m - 1);

		// Build the full denominator multilinear D[j] = c - iota(j) over the table cube.
		let d_values = (0..(1usize << m))
			.map(|j| c - iota(j, m))
			.collect::<Vec<_>>();

		// The helper splits on the highest variable; build the matching low-to-high points.
		//
		//     point_0 = rho || 0   (highest variable = 0)
		//     point_1 = rho || 1   (highest variable = 1)
		let mut point_0 = rho.clone();
		point_0.push(B128::ZERO);
		let mut point_1 = rho.clone();
		point_1.push(B128::ONE);

		// Reference evaluations from the explicitly built multilinear.
		let expected_d0 = evaluate_scalars(&d_values, &point_0);
		let expected_d1 = evaluate_scalars(&d_values, &point_1);

		// The helper must reproduce both half-evaluations from the closed-form embedding.
		let (d_0, d_1) = denominator_halves::<B128, B128>(&c, &rho, m);
		assert_eq!(d_0, expected_d0, "D_0 mismatch for m = {m}");
		assert_eq!(d_1, expected_d1, "D_1 mismatch for m = {m}");
	}

	#[test]
	fn test_denominator_halves_matches_explicit_multilinear() {
		// m = 1 exercises the empty low point (rho has length 0).
		// Larger m exercise the basis sum over multiple coordinates.
		for m in 1..=6 {
			check_denominator_halves(m);
		}
	}
}
