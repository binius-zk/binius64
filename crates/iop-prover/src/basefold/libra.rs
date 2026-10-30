// Copyright 2026 The Binius Developers

//! The prover's Libra mask for the batch sumcheck over $\{0, \infty\}^\ell$.
//!
//! See [`binius_iop::basefold::libra`] for the mask and its coefficient table.

use binius_field::{Field, PackedField};
use binius_iop::basefold::libra::log_n_rows;
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::FieldBuffer;
use rand::Rng;

/// A Libra mask $h$ on `n_vars` variables, held as its coefficient table $h'$.
pub struct LibraMask<P: PackedField> {
	n_vars: usize,
	table: FieldBuffer<P>,
}

impl<F: Field, P: PackedField<Scalar = F>> LibraMask<P> {
	/// Samples a uniform table of `2^log_len` entries for a mask on `n_vars` variables.
	pub fn random(mut rng: impl Rng, n_vars: usize, log_len: usize) -> Self {
		let values = (0..1 << log_len)
			.map(|_| F::random(&mut rng))
			.collect::<Vec<_>>();
		Self {
			n_vars,
			table: FieldBuffer::from_values(&values),
		}
	}

	/// The coefficient table $h'$, the message of the Libra oracle.
	pub const fn table(&self) -> &FieldBuffer<P> {
		&self.table
	}

	/// The coefficient $h'(j, k)$ of $X_j^k$.
	fn coeff(&self, j: usize, k: usize) -> F {
		self.table.get(j | k << log_n_rows(self.n_vars))
	}

	/// $h_j(x) = h'(j, 0) + h'(j, 1) x + h'(j, 2) x^2$, so that $h(X) = \sum_j h_j(X_j)$.
	fn eval_row(&self, j: usize, x: F) -> F {
		self.coeff(j, 0) + x * (self.coeff(j, 1) + x * self.coeff(j, 2))
	}

	/// The sum of a row over $\{0, \infty\}$: the constant plus the leading coefficient.
	fn sum_row(&self, j: usize) -> F {
		self.coeff(j, 0) + self.coeff(j, 2)
	}

	/// $s_h = \sum_{x \in \{0, \infty\}^\ell} h(x)$.
	///
	/// A row is summed over its own coordinate, and every other coordinate contributes one.
	pub fn sum(&self) -> F {
		(0..self.n_vars).map(|j| self.sum_row(j)).sum()
	}

	/// $h(r)$ at the point `point`, low-to-high.
	pub fn evaluate(&self, point: &[F]) -> F {
		assert_eq!(point.len(), self.n_vars); // precondition
		point
			.iter()
			.enumerate()
			.map(|(j, &r_j)| self.eval_row(j, r_j))
			.sum()
	}

	/// The round polynomial after the top variables are bound to `challenges`, in binding order.
	///
	/// The round's variable is $X_c$, $c = \ell - 1 - i$ in round $i$. The rows above it are
	/// evaluated at their challenges, the rows below are summed over $\{0, \infty\}$, and row $c$
	/// is the round polynomial's non-constant part.
	pub fn round_coeffs(&self, challenges: &[F]) -> RoundCoeffs<F> {
		let c = self.n_vars - 1 - challenges.len();
		let bound = challenges
			.iter()
			.enumerate()
			.map(|(i, &r)| self.eval_row(self.n_vars - 1 - i, r))
			.sum::<F>();
		let free = (0..c).map(|j| self.sum_row(j)).sum::<F>();
		RoundCoeffs(vec![
			bound + free + self.coeff(c, 0),
			self.coeff(c, 1),
			self.coeff(c, 2),
		])
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		Random,
		arch::{OptimalB128 as B128, OptimalPackedB128},
	};
	use binius_iop::basefold::libra::{LOG_DEGREE, libra_eval_monomial, libra_eval_table};
	use binius_math::{
		multilinear::hypercube::{InfCube, eq_ind_partial_eval_scalars},
		test_utils::random_scalars,
	};
	use rand::prelude::*;

	use super::*;

	type P = OptimalPackedB128;

	#[test]
	fn the_table_evaluates_the_mask() {
		let mut rng = StdRng::seed_from_u64(0);
		for n_vars in [0, 1, 3, 4, 5] {
			let log_len = log_n_rows(n_vars) + LOG_DEGREE;
			let mask = LibraMask::<P>::random(&mut rng, n_vars, log_len);
			let point = random_scalars::<B128>(&mut rng, n_vars);

			let table = libra_eval_table(&point, log_len);
			let dot = mask
				.table()
				.iter_scalars()
				.zip(&table)
				.map(|(h, t)| h * *t)
				.sum::<B128>();
			assert_eq!(mask.evaluate(&point), dot, "n_vars={n_vars}");

			// The monomial evaluation of the table matches the table, padding included.
			let point_prime = random_scalars::<B128>(&mut rng, log_len + 1);
			let table = libra_eval_table(&point, log_len + 1);
			let monomials = eq_ind_partial_eval_scalars::<InfCube, B128>(&point_prime);
			let expected = table
				.iter()
				.zip(monomials)
				.map(|(t, m)| *t * m)
				.sum::<B128>();
			assert_eq!(libra_eval_monomial(&point, &point_prime), expected, "n_vars={n_vars}");
		}
	}

	#[test]
	fn round_polynomials_run_the_sumcheck() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 5;
		let mask = LibraMask::<P>::random(&mut rng, n_vars, log_n_rows(n_vars) + LOG_DEGREE);

		let mut claim = mask.sum();
		let mut challenges = Vec::new();
		for _ in 0..n_vars {
			let coeffs = mask.round_coeffs(&challenges);
			// R(0) + R(∞) is the constant plus the leading coefficient.
			assert_eq!(coeffs[0] + coeffs[2], claim);
			let challenge = B128::random(&mut rng);
			claim = coeffs.evaluate(&challenge);
			challenges.push(challenge);
		}
		challenges.reverse();
		assert_eq!(mask.evaluate(&challenges), claim);
	}
}
