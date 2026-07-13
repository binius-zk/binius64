//! Vendored `CubicProductSumcheckProver`: a [`SumcheckProver`] for a composite defined
//! as the product of THREE multilinears (degree-3 rounds). Mirrors the structure of
//! crates/ip-prover/src/sumcheck/bivariate_product.rs (Apache-2.0/MIT, Irreducible Inc /
//! The Binius Developers), extended from two to three factors, INCLUDING its packed
//! prover idioms at the pinned rev: generic packed buffers + rayon-parallel round
//! evaluation with packed-lane accumulators (W1 port; the round coefficients are
//! exactly the same field elements as the scalar loop — char-2 addition is XOR,
//! commutative and associative, so transcript bytes are unchanged).
//!
//! NOTE: the pinned rev c799aa10 predates upstream's `WideMul` unreduced-accumulation
//! API (landed later at HEAD); the port therefore uses plain packed multiplications,
//! which is exactly what the pinned rev's own provers do.
//!
//! Binds variables in high-to-low index order, like the upstream prover, so it is
//! driven by the same `batch_prove_and_write_evals` driver and verified by
//! `sumcheck::batch_verify`.
//!
//! Round polynomial computed in monomial form directly: with per-slot linear factors
//! A(X) = a0 + dA*X (dA = a0 + a1 in char 2), etc.:
//!   A*B*G = c0 + c1*X + c2*X^2 + c3*X^3,
//!   c0 = a0 b0 g0
//!   c1 = a0 b0 dG + (a0 dB + dA b0) g0
//!   c2 = (a0 dB + dA b0) dG + dA dB g0
//!   c3 = dA dB dG

use std::ops::Add;

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
// FWD-PORT: `SumcheckProver` is now INFALLIBLE (execute/fold/finish return values, not `Result`);
// the `Error` type was removed. Round math is unchanged (monomial `RoundCoeffs`).
use binius_ip_prover::sumcheck::common::SumcheckProver;
use binius_math::{FieldBuffer, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::prelude::*;

/// Packed-lane accumulator for the four cubic round coefficients (the `RoundEvals2`
/// pattern from the pinned rev's bivariate_product.rs, extended to degree 3).
#[derive(Clone, Copy)]
struct CubicEvals<P> {
	c: [P; 4],
}

impl<P: PackedField> Default for CubicEvals<P> {
	fn default() -> Self {
		Self {
			c: [P::zero(); 4],
		}
	}
}

impl<P: Add<Output = P> + Copy> Add for CubicEvals<P> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		let [a0, a1, a2, a3] = self.c;
		let [b0, b1, b2, b3] = rhs.c;
		Self {
			c: [a0 + b0, a1 + b1, a2 + b2, a3 + b3],
		}
	}
}

/// Sumcheck prover for the product of three multilinears over the same cube.
#[derive(Debug)]
pub struct CubicProductSumcheckProver<P: PackedField> {
	multilinears: [FieldBuffer<P>; 3],
	last_coeffs_or_sum: RoundCoeffsOrSum<P::Scalar>,
}

impl<F: Field, P: PackedField<Scalar = F>> CubicProductSumcheckProver<P> {
	/// Constructs a prover from the three multilinears and the claimed hypercube sum
	/// of their product. Infallible (asserts equal lengths), mirroring the upstream
	/// `BivariateProductSumcheckProver::new`.
	pub fn new(multilinears: [FieldBuffer<P>; 3], sum: F) -> Self {
		assert_eq!(
			multilinears[0].log_len(),
			multilinears[1].log_len(),
			"multilinears must have equal number of variables"
		);
		assert_eq!(
			multilinears[0].log_len(),
			multilinears[2].log_len(),
			"multilinears must have equal number of variables"
		);
		Self {
			multilinears,
			last_coeffs_or_sum: RoundCoeffsOrSum::Sum(sum),
		}
	}
}

impl<F: Field, P: PackedField<Scalar = F>> SumcheckProver<F> for CubicProductSumcheckProver<P> {
	fn n_vars(&self) -> usize {
		self.multilinears[0].log_len()
	}

	fn n_claims(&self) -> usize {
		1
	}

	fn round_claim(&self) -> Vec<F> {
		let claim = match &self.last_coeffs_or_sum {
			RoundCoeffsOrSum::Sum(sum) => *sum,
			RoundCoeffsOrSum::Coeffs(coeffs) => coeffs.sum_over_endpoints(),
		};
		vec![claim]
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		assert!(
			matches!(self.last_coeffs_or_sum, RoundCoeffsOrSum::Sum(_)),
			"execute called out of order; fold expected"
		);
		let n_vars_remaining = self.n_vars();
		assert!(n_vars_remaining > 0);

		let (a_lo, a_hi) = self.multilinears[0].split_half_ref();
		let (b_lo, b_hi) = self.multilinears[1].split_half_ref();
		let (g_lo, g_hi) = self.multilinears[2].split_half_ref();

		// Per-slot products accumulated in packed lanes under rayon; scalars beyond the
		// truncated length are zero by the FieldBuffer truncation invariant, so summing
		// all packed lanes at the end is exact.
		let evals = (
			a_lo.as_ref(),
			a_hi.as_ref(),
			b_lo.as_ref(),
			b_hi.as_ref(),
			g_lo.as_ref(),
			g_hi.as_ref(),
		)
			.into_par_iter()
			.map(|(&a0, &a1, &b0, &b1, &g0, &g1)| {
				let da = a0 + a1;
				let db = b0 + b1;
				let dg = g0 + g1;

				let p_ab = a0 * b0;
				let p_mid = a0 * db + da * b0;
				let p_dd = da * db;

				CubicEvals {
					c: [
						p_ab * g0,
						p_ab * dg + p_mid * g0,
						p_mid * dg + p_dd * g0,
						p_dd * dg,
					],
				}
			})
			.reduce(CubicEvals::<P>::default, CubicEvals::add);

		let c: Vec<F> = evals.c.into_iter().map(|p| p.iter().sum()).collect();

		let round_coeffs = RoundCoeffs(c);
		self.last_coeffs_or_sum = RoundCoeffsOrSum::Coeffs(round_coeffs.clone());
		vec![round_coeffs]
	}

	fn fold(&mut self, challenge: F) {
		let RoundCoeffsOrSum::Coeffs(last_coeffs) = self.last_coeffs_or_sum.clone() else {
			panic!("fold called out of order; execute expected");
		};
		for multilin in &mut self.multilinears {
			fold_highest_var_inplace(multilin, challenge);
		}
		let round_sum = last_coeffs.evaluate(challenge);
		self.last_coeffs_or_sum = RoundCoeffsOrSum::Sum(round_sum);
	}

	fn finish(self) -> Vec<F> {
		assert_eq!(self.n_vars(), 0, "finish called out of order; sumcheck rounds remain");
		self.multilinears.into_iter().map(|m| m.get(0)).collect()
	}
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrSum<F: Field> {
	Coeffs(RoundCoeffs<F>),
	Sum(F),
}

#[cfg(test)]
mod tests {
	use binius_field::Random;
	use binius_math::multilinear::evaluate::evaluate;
	use binius_transcript::ProverTranscript;
	use binius_verifier::config::{B128, StdChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::packed::PB;

	fn random_buffer<P: PackedField<Scalar = B128>>(rng: &mut StdRng, n_vars: usize) -> FieldBuffer<P> {
		let vals: Vec<B128> = (0..1usize << n_vars).map(|_| B128::random(&mut *rng)).collect();
		FieldBuffer::from_values(&vals)
	}

	fn roundtrip<P: PackedField<Scalar = B128>>() -> (Vec<B128>, Vec<u8>) {
		let n_vars = 9;
		let mut rng = StdRng::seed_from_u64(7);

		let a = random_buffer::<P>(&mut rng, n_vars);
		let b = random_buffer::<P>(&mut rng, n_vars);
		let g = random_buffer::<P>(&mut rng, n_vars);

		let expected_sum: B128 = (0..1usize << n_vars)
			.map(|i| a.get(i) * b.get(i) * g.get(i))
			.sum();

		let prover =
			CubicProductSumcheckProver::new([a.clone(), b.clone(), g.clone()], expected_sum);

		let mut pt = ProverTranscript::new(StdChallenger::default());
		let output =
			binius_ip_prover::sumcheck::batch::batch_prove_and_write_evals(vec![prover], &mut pt);

		let mut vt = pt.into_verifier();
		let out = binius_ip::sumcheck::batch_verify::<B128, _>(
			n_vars,
			3,
			&[expected_sum],
			&mut vt,
		)
		.unwrap();

		use binius_ip::channel::IPVerifierChannel as _;
		let evals: Vec<B128> = vt.recv_many(3).unwrap();
		assert_eq!(evals[0] * evals[1] * evals[2], out.eval, "finish evals vs reduced eval");

		// challenges from batch_verify are in round order (high-to-low); reverse for
		// the low-first evaluation point.
		let mut point = out.challenges.clone();
		point.reverse();
		assert_eq!(evaluate(&a, &point), evals[0]);
		assert_eq!(evaluate(&b, &point), evals[1]);
		assert_eq!(evaluate(&g, &point), evals[2]);

		// prover-side challenges are already reversed (low-first) by batch_prove
		assert_eq!(output.challenges, point);

		// return the transcript bytes for the packed-vs-scalar identity check
		let mut pt2 = ProverTranscript::new(StdChallenger::default());
		let prover2 = CubicProductSumcheckProver::new([a, b, g], expected_sum);
		binius_ip_prover::sumcheck::batch::batch_prove_and_write_evals(vec![prover2], &mut pt2);
		(point, pt2.finalize())
	}

	#[test]
	fn test_cubic_product_sumcheck_roundtrip() {
		let (point_scalar, bytes_scalar) = roundtrip::<B128>();
		let (point_packed, bytes_packed) = roundtrip::<PB>();
		// The packed prover must produce EXACTLY the scalar prover's transcript.
		assert_eq!(point_scalar, point_packed);
		assert_eq!(bytes_scalar, bytes_packed, "packed round coeffs must be byte-identical");
	}
}
