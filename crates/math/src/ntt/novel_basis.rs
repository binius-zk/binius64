// Copyright 2026 The Binius Developers

//! Evaluation of the normalized subspace polynomials of a novel polynomial basis.
//!
//! The additive NTT reads its input as novel-polynomial-basis coefficients, per [LCH14].
//! That basis factors bit by bit over the message index `j`:
//!
//! ```text
//!     Xhat_j(x) = prod_{k : bit_k(j) = 1} What_k(x)
//! ```
//!
//! `What_k` vanishes on the `k`-dimensional subspace `S_k`.
//! It is normalized so that `What_k(beta_k) = 1`.
//!
//! The factorization makes the whole basis at one point a tensor expansion.
//! Its factors are the `log_n` values `What_0(x), .., What_{log_n - 1}(x)`.
//! A row of the transform matrix is therefore a tensor product, not an opaque vector.
//! [`tensor_expand`] builds that row.
//!
//! The row pairs with [`AdditiveNTT::forward_transform`](super::AdditiveNTT::forward_transform)
//! when no layers are skipped.
//! `ReedSolomonCode::encode_batch` does skip its first `log_inv_rate` layers.
//! Its effective basis is not this one, so a caller must derive that mapping separately.
//!
//! [`DomainContext`] precomputes `What_i` on the *basis* elements only.
//! That is all the NTT and the FRI fold need.
//! This module adds two evaluations they do not provide.
//!
//! - At an arbitrary field point, via [`NovelBasis::evals_at`].
//! - At a domain index through `F2`-linearity, via [`NovelBasis::evals_at_domain_index`].
//!
//! [LCH14]: <https://arxiv.org/abs/1404.3458>

use binius_field::{BinaryField, Field};

use super::DomainContext;

/// The normalized subspace polynomials `What_0, .., What_{l-1}` of a novel polynomial basis.
///
/// Built from a [`DomainContext`], so these always match the basis its NTT transforms over.
/// The pairing is with a transform that skips no layers.
#[derive(Debug, Clone)]
pub struct NovelBasis<F> {
	/// `linear[k][j] = What_k(beta_{k + j})`, for `j` in `0..l - k`.
	///
	/// `What_k` is `F2`-linear and vanishes on `beta_0, .., beta_{k-1}`.
	/// A domain point's value is therefore the sum of the entries its set bits select.
	/// Row `k` starts at `beta_k` because every earlier basis element evaluates to zero.
	linear: Vec<Vec<F>>,
	/// `step_inv[i] = (d_i * (d_i + 1))^-1`, where `d_i = What_i(beta_{i+1})`.
	///
	/// This constant advances the arbitrary-point recurrence from `What_i` to `What_{i+1}`.
	/// Length is `l - 1`: one step per adjacent pair of polynomials.
	step_inv: Vec<F>,
	/// `beta_0^-1`, the normalizer of `What_0`.
	///
	/// `What_0` is the vanishing polynomial of the zero subspace, which is just `X`.
	/// Normalizing it is therefore a single multiplication.
	beta_0_inv: F,
}

impl<F: BinaryField> NovelBasis<F> {
	/// Builds the normalized subspace polynomials of `domain_context`'s basis.
	///
	/// ## Preconditions
	///
	/// * `domain_context.log_domain_size()` must be at least 1.
	pub fn new<DC: DomainContext<Field = F>>(domain_context: &DC) -> Self {
		let l = domain_context.log_domain_size();
		assert!(l >= 1, "precondition: log_domain_size must be at least 1");

		// `DomainContext::subspace(i)` holds the evaluations of `What_{l - i}`, so reading it at
		// `l - k` gives row `k`.
		let linear = (0..l)
			.map(|k| domain_context.subspace(l - k).basis().to_vec())
			.collect::<Vec<_>>();

		// Advancing the recurrence needs `d_k = What_k(beta_{k+1})`, which is row `k`'s second
		// entry. The last row has no successor, so it contributes no step.
		let step_inv = linear[..l - 1]
			.iter()
			.map(|row| {
				// `d * (d + 1)` is the normalizer of `What_{k+1}`, so it is non-zero by
				// construction of the novel basis. Assert it rather than assume it.
				let step = row[1] * (row[1] + F::ONE);
				assert_ne!(step, F::ZERO, "What_{{k+1}} normalizer must be non-zero");
				step.invert_or_zero()
			})
			.collect();

		// Row 0 holds `What_0(beta_j) = beta_j / beta_0`, which has already divided `beta_0` out,
		// so recover it from the full domain basis.
		let beta_0 = domain_context.subspace(l).basis()[0];
		assert_ne!(beta_0, F::ZERO, "beta_0 is a basis element, so it must be non-zero");
		let beta_0_inv = beta_0.invert_or_zero();

		Self {
			linear,
			step_inv,
			beta_0_inv,
		}
	}

	/// The number of polynomials, `l`, matching the domain context's `log_domain_size`.
	pub const fn log_domain_size(&self) -> usize {
		self.linear.len()
	}

	/// `[What_0(x), .., What_{l-1}(x)]` at an arbitrary field point.
	///
	/// Runs the normalized recurrence
	///
	/// ```text
	///     What_0(x)     = x * beta_0^-1
	///     What_{i+1}(x) = What_i(x) * (What_i(x) + 1) * step_inv[i]
	/// ```
	///
	/// It follows from `W_{i+1}(X) = W_i(X) * (W_i(X) + W_i(beta_i))`.
	/// Dividing that identity through by the normalizers leaves the form above.
	/// Costs `l` multiplications and no inversions.
	pub fn evals_at(&self, x: F) -> Vec<F> {
		let mut evals = Vec::with_capacity(self.log_domain_size());
		let mut w = x * self.beta_0_inv;
		evals.push(w);
		for &step_inv in &self.step_inv {
			w *= (w + F::ONE) * step_inv;
			evals.push(w);
		}
		evals
	}

	/// `[What_0(x), .., What_{l-1}(x)]` where `x` is the domain element selected by `index`.
	///
	/// Bit `i` of `index` selects whether `beta_i` is XORed into `x`.
	/// That matches [`BinarySubspace::get`](crate::BinarySubspace::get).
	///
	/// Each `What_k` is `F2`-linear, so its value is a subset sum of the `What_k(beta_i)`.
	/// The summands are fixed constants, so this needs no inversion and no multiplication.
	///
	/// This is the route a verifier takes for a sampled query index.
	/// Adding constants costs a recursive circuit what the FRI fold already pays for twiddles.
	///
	/// ## Preconditions
	///
	/// * `index` must be less than `2^l`.
	pub fn evals_at_domain_index(&self, index: usize) -> Vec<F> {
		assert!(
			index < 1 << self.log_domain_size(),
			"precondition: index must be less than 2^log_domain_size"
		);

		self.linear
			.iter()
			.enumerate()
			.map(|(k, row)| {
				// Row `k` is indexed from `beta_k`, so bit `k + j` of the index selects `row[j]`.
				row.iter()
					.enumerate()
					.filter(|(j, _)| (index >> (k + j)) & 1 == 1)
					.fold(F::ZERO, |acc, (_, &eval)| acc + eval)
			})
			.collect()
	}
}

/// Expands `evals` into the novel polynomial basis evaluated at the same point.
///
/// Returns `2^evals.len()` values, where entry `j` is `prod_{k : bit_k(j) = 1} evals[k]`.
/// Truncating [`NovelBasis::evals_at`] to `log_n` entries yields a transform-matrix row.
/// Pairing that row with a `log_n`-coefficient message reproduces the unskipped transform.
///
/// Entry `j` selects `evals[k]` exactly when bit `k` of `j` is set.
/// So `evals[0]` is the least significant factor, matching the NTT's doubling order.
///
/// The output is exponential in `evals.len()`, so this materializes a whole row.
/// A verifier should evaluate the row's multilinear extension in product form instead.
///
/// ## Preconditions
///
/// * `evals.len()` must be small enough that `2^evals.len()` elements fit in memory.
pub fn tensor_expand<F: Field>(evals: &[F]) -> Vec<F> {
	let mut tensor = Vec::with_capacity(1 << evals.len());
	tensor.push(F::ONE);
	for &w in evals {
		// Double the block, scaling the copy by `w`, so entry `j` gains the factor `w` exactly
		// when the new bit of `j` is set.
		let half = tensor.len();
		for j in 0..half {
			tensor.push(tensor[j] * w);
		}
	}
	tensor
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField128bGhash;
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		BinarySubspace,
		ntt::{
			AdditiveNTT, NeighborsLastSingleThread,
			domain_context::{GaoMateerPreExpanded, GenericPreExpanded},
		},
		test_utils::random_field_buffer,
	};

	type F = BinaryField128bGhash;

	/// A subspace whose first basis element is not 1, which is what makes `beta_0_inv` matter.
	///
	/// Scaling a basis by a non-zero constant is an `F2`-linear bijection.
	/// Independence is therefore preserved at every dimension.
	/// Perturbing each element independently is not safe.
	/// At `log_d = 6` the basis `5 + 17*i` has `beta_5 = beta_0 + beta_1 + beta_4`.
	fn scaled_subspace(log_d: usize) -> BinarySubspace<F> {
		let scale = F::new(5);
		let basis = BinarySubspace::<F>::with_dim(log_d)
			.basis()
			.iter()
			.map(|&b| b * scale)
			.collect::<Vec<_>>();
		BinarySubspace::new_unchecked(basis)
	}

	/// Runs `check` against each domain context shape a caller might build a `NovelBasis` from.
	///
	/// Gao-Mateer is the context [`ReedSolomonCode`](crate::reed_solomon::ReedSolomonCode) fixes.
	/// It is therefore the production-relevant case.
	fn for_each_context(
		log_d: usize,
		check: impl Fn(&dyn Fn(usize) -> BinarySubspace<F>, &NovelBasis<F>),
	) {
		let gao_mateer = GaoMateerPreExpanded::<F>::generate(log_d);
		check(&|i| gao_mateer.subspace(i), &NovelBasis::new(&gao_mateer));

		let standard = GenericPreExpanded::generate_from_subspace(&BinarySubspace::with_dim(log_d));
		check(&|i| standard.subspace(i), &NovelBasis::new(&standard));

		let scaled = GenericPreExpanded::generate_from_subspace(&scaled_subspace(log_d));
		check(&|i| scaled.subspace(i), &NovelBasis::new(&scaled));
	}

	#[test]
	fn evals_at_basis_elements_match_the_domain_context() {
		for log_d in 1..7 {
			for_each_context(log_d, |subspace, basis| {
				let domain = subspace(log_d);
				// Row `k` of the context's table is `What_k` on `beta_k..beta_{l-1}`, so the
				// arbitrary-point recurrence must reproduce it entry by entry.
				for k in 0..log_d {
					let row = subspace(log_d - k);
					for (j, &expected) in row.basis().iter().enumerate() {
						let got = basis.evals_at(domain.basis()[k + j])[k];
						assert_eq!(got, expected, "log_d={log_d} k={k} j={j}");
					}
				}
			});
		}
	}

	#[test]
	fn what_k_is_normalized_and_vanishes_below_its_subspace() {
		for log_d in 1..7 {
			for_each_context(log_d, |subspace, basis| {
				let domain = subspace(log_d);
				for k in 0..log_d {
					// `What_k` vanishes on the subspace it is built from.
					for j in 0..k {
						assert_eq!(basis.evals_at(domain.basis()[j])[k], F::ZERO);
					}
					// And it is normalized to 1 at the next basis element.
					assert_eq!(basis.evals_at(domain.basis()[k])[k], F::ONE);
				}
			});
		}
	}

	#[test]
	#[should_panic(expected = "normalizer must be non-zero")]
	fn new_rejects_a_dependent_basis() {
		// `beta_2 = beta_0 + beta_1` collapses the subspace, so `What_2(beta_2) = 0` and the
		// normalizer of `What_2` vanishes.
		let dependent = BinarySubspace::new_unchecked(vec![F::new(5), F::new(22), F::new(19)]);
		NovelBasis::new(&GenericPreExpanded::generate_from_subspace(&dependent));
	}

	#[test]
	fn what_k_vanishes_at_zero() {
		let dc = GaoMateerPreExpanded::<F>::generate(5);
		let basis = NovelBasis::new(&dc);
		// Every `What_k` is `F2`-linear, so it sends zero to zero.
		assert!(basis.evals_at(F::ZERO).iter().all(|&w| w == F::ZERO));
	}

	#[test]
	fn domain_index_route_matches_arbitrary_point_route() {
		for log_d in 1..7 {
			for_each_context(log_d, |subspace, basis| {
				let domain = subspace(log_d);
				// The `F2`-linear subset sum must agree with the recurrence on every domain point.
				for index in 0..1 << log_d {
					assert_eq!(
						basis.evals_at_domain_index(index),
						basis.evals_at(domain.get(index)),
						"log_d={log_d} index={index}"
					);
				}
			});
		}
	}

	#[test]
	fn tensor_expand_selects_the_product_over_set_bits() {
		let evals = [F::new(3), F::new(9), F::new(41)];
		let tensor = tensor_expand(&evals);
		assert_eq!(tensor.len(), 8);
		for (j, &entry) in tensor.iter().enumerate() {
			let expected = (0..evals.len())
				.filter(|k| (j >> k) & 1 == 1)
				.fold(F::ONE, |acc, k| acc * evals[k]);
			assert_eq!(entry, expected, "j={j}");
		}
	}

	#[test]
	fn tensor_expand_of_nothing_is_one() {
		assert_eq!(tensor_expand::<F>(&[]), vec![F::ONE]);
	}

	#[test]
	#[should_panic(expected = "index must be less than 2^log_domain_size")]
	fn evals_at_domain_index_rejects_an_out_of_range_index() {
		let dc = GaoMateerPreExpanded::<F>::generate(3);
		NovelBasis::new(&dc).evals_at_domain_index(8);
	}

	/// The headline identity, checked against the NTT itself.
	/// A coefficient vector paired with a tensor generator row reproduces the NTT's output.
	fn assert_tensor_row_matches_ntt<DC>(log_d: usize, dc: DC, seed: u64)
	where
		DC: DomainContext<Field = F>,
	{
		let basis = NovelBasis::new(&dc);
		let mut rng = StdRng::seed_from_u64(seed);
		let coeffs = random_field_buffer::<F>(&mut rng, log_d);

		let mut transformed = coeffs.clone();
		let ntt = NeighborsLastSingleThread::new(dc);
		ntt.forward_transform(transformed.to_mut(), 0, 0);

		for index in 0..1 << log_d {
			// Row `index` of the generator matrix is the tensor of `What_k` at that domain point.
			let row = tensor_expand(&basis.evals_at_domain_index(index));
			let dot = coeffs
				.as_ref()
				.iter()
				.zip(&row)
				.fold(F::ZERO, |acc, (&c, &r)| acc + c * r);
			assert_eq!(dot, transformed.as_ref()[index], "log_d={log_d} index={index}");
		}
	}

	#[test]
	fn tensor_row_matches_ntt_over_every_domain_context() {
		for log_d in 1..7 {
			assert_tensor_row_matches_ntt(log_d, GaoMateerPreExpanded::<F>::generate(log_d), 0);
			let standard = BinarySubspace::<F>::with_dim(log_d);
			assert_tensor_row_matches_ntt(
				log_d,
				GenericPreExpanded::generate_from_subspace(&standard),
				1,
			);
			assert_tensor_row_matches_ntt(
				log_d,
				GenericPreExpanded::generate_from_subspace(&scaled_subspace(log_d)),
				2,
			);
		}
	}

	proptest! {
		#[test]
		fn tensor_row_matches_ntt_on_random_coefficients(seed: u64) {
			const LOG_D: usize = 5;
			// Gao-Mateer is the basis a Reed-Solomon code actually encodes over.
			assert_tensor_row_matches_ntt(LOG_D, GaoMateerPreExpanded::<F>::generate(LOG_D), seed);
		}

		#[test]
		fn evals_at_is_f2_linear(a: u64, b: u64) {
			const LOG_D: usize = 6;
			let dc = GaoMateerPreExpanded::<F>::generate(LOG_D);
			let basis = NovelBasis::new(&dc);
			let (x, y) = (F::new(a as u128), F::new(b as u128));
			// Each `What_k` is a subspace vanishing polynomial, hence additive over `F2`.
			let sum = basis.evals_at(x + y);
			let parts = std::iter::zip(basis.evals_at(x), basis.evals_at(y));
			for (got, (wx, wy)) in std::iter::zip(sum, parts) {
				prop_assert_eq!(got, wx + wy);
			}
		}
	}
}
