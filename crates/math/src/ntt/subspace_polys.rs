// Copyright 2026 The Binius Developers

//! Evaluating the normalized subspace polynomials $\hat{W}_k$ away from the basis.
//!
//! The additive NTT reads its input as coefficients in the novel polynomial basis of [LCH14].
//! That basis factors bit by bit over the message index $j$:
//!
//! $$
//! \hat{X}_j(x) = \prod_{k \,:\, \mathrm{bit}_k(j) = 1} \hat{W}_k(x)
//! $$
//!
//! Each $\hat{W}_k$ vanishes on the $k$-dimensional subspace $S_k$.
//! It is normalized so that $\hat{W}_k(\beta_k) = 1$.
//!
//! [`DomainContext`] already tabulates $\hat{W}_k$ on the basis elements $\beta_j$.
//! That is everything the NTT and the FRI fold need.
//! This module adds the two evaluations they do not provide.
//!
//! - At an arbitrary field point, via [`NormalizedSubspacePolys::evals_at`].
//! - At a domain index, via [`evals_at_domain_index`], a subset sum over that same table.
//!
//! # A generator row is a tensor
//!
//! The factorization above makes one row of the transform matrix a tensor expansion.
//! Its factors are the $\ell$ numbers $\hat{W}_0(x), \ldots, \hat{W}_{\ell-1}(x)$.
//! With two variables:
//!
//! $$
//! m_0 + m_1 \hat{W}_0(x) + m_2 \hat{W}_1(x) + m_3 \hat{W}_1(x) \hat{W}_0(x)
//!   = \langle m, (1, \hat{W}_0(x)) \otimes (1, \hat{W}_1(x)) \rangle
//! $$
//!
//! Four row entries out of two numbers.
//! Out of $\ell$ numbers you get all $2^\ell$ entries.
//! That is what lets a verifier evaluate a row without materializing it.
//! `binius_field::util::expand_subset_products` performs the expansion.
//!
//! # Pairing with a Reed-Solomon codeword
//!
//! Two adjustments turn the identity above into a row of [`ReedSolomonCode::encode_batch`].
//!
//! First, build the polynomials on the *codeword* domain, then truncate to the message dimension.
//! Encoding zero-pads the message, so the extra dimensions contribute nothing.
//!
//! Second, reverse the evaluation order before expanding.
//! `encode_batch` encodes the bit-reversal permuted message, as `reed_solomon.rs` states.
//! Reversing the $\ell$ factors applies that same permutation to the tensor.
//!
//! Writing `evals` for the evaluations on the codeword domain:
//!
//! ```text
//!     encode_batch(msg, 0)[x] = <msg, expand_subset_products(rev(evals[..log_dim]))>
//! ```
//!
//! Interleaved lanes reduce to that case.
//! Lane `lane` of a `b`-lane encoding is the plain encoding of its own columns:
//!
//! ```text
//!     encode_batch(msg, b)[(x << b) | lane] = encode_batch(lane_msg, 0)[x]
//!     lane_msg[j] = msg[(reverse_bits(lane, b) << log_dim) | j]
//! ```
//!
//! Both identities are pinned by tests in this module.
//!
//! [LCH14]: <https://arxiv.org/abs/1404.3458>
//! [`ReedSolomonCode::encode_batch`]: crate::reed_solomon::ReedSolomonCode::encode_batch

use binius_field::BinaryField;

use super::DomainContext;

/// Constants for evaluating the normalized subspace polynomials at an arbitrary field point.
///
/// Built from a [`DomainContext`], so these match the basis its NTT transforms over.
///
/// Holds only what the recurrence needs.
/// The evaluations on the basis elements stay in the [`DomainContext`].
/// Keeping them there is what stops them drifting out of step with the NTT's own twiddles.
#[derive(Debug, Clone)]
pub struct NormalizedSubspacePolys<F> {
	/// `step_inv[i]` is $(d_i (d_i + 1))^{-1}$, where $d_i = \hat{W}_i(\beta_{i+1})$.
	///
	/// This constant advances the recurrence from $\hat{W}_i$ to $\hat{W}_{i+1}$.
	/// Length is $\ell - 1$: one step per adjacent pair of polynomials.
	step_inv: Vec<F>,
	/// $\beta_0^{-1}$, the normalizer of $\hat{W}_0$.
	///
	/// $\hat{W}_0$ vanishes on the zero subspace, so it is just $X$.
	/// Normalizing it is a single multiplication.
	beta_0_inv: F,
}

impl<F: BinaryField> NormalizedSubspacePolys<F> {
	/// Builds the constants for `domain_context`'s basis.
	///
	/// # Panics
	/// Panics if `domain_context.log_domain_size()` is zero.
	/// Panics if the basis is linearly dependent, which makes a normalizer vanish.
	pub fn new<DC: DomainContext<Field = F>>(domain_context: &DC) -> Self {
		let l = domain_context.log_domain_size();
		assert!(l >= 1, "precondition: log_domain_size must be at least 1");

		// Row k of the table is $\hat{W}_k$ on $\beta_k, \ldots$, so its second entry is $d_k$.
		// The last row has no successor and so contributes no step.
		let step_inv = (0..l - 1)
			.map(|k| {
				let d = domain_context.subspace(l - k).basis()[1];
				let step = d * (d + F::ONE);
				// Invariant: `step` normalizes $\hat{W}_{k+1}$, so a novel basis makes it nonzero.
				assert_ne!(step, F::ZERO, "W_hat_{{k+1}} normalizer must be non-zero");
				step.invert_or_zero()
			})
			.collect();

		// Row 0 has already divided $\beta_0$ out, so read it off the full domain basis instead.
		let beta_0 = domain_context.subspace(l).basis()[0];
		assert_ne!(beta_0, F::ZERO, "beta_0 is a basis element, so it must be non-zero");

		Self {
			step_inv,
			beta_0_inv: beta_0.invert_or_zero(),
		}
	}

	/// The number of polynomials, matching the domain context's `log_domain_size`.
	pub const fn log_domain_size(&self) -> usize {
		self.step_inv.len() + 1
	}

	/// Evaluates every $\hat{W}_k$ at an arbitrary field point, in increasing $k$.
	///
	/// Runs the normalized recurrence
	///
	/// ```text
	///     W_hat_0(x)     = x * beta_0^-1
	///     W_hat_{i+1}(x) = W_hat_i(x) * (W_hat_i(x) + 1) * step_inv[i]
	/// ```
	///
	/// It follows from $W_{i+1}(X) = W_i(X) (W_i(X) + W_i(\beta_i))$, divided by the normalizers.
	///
	/// Costs $\ell$ multiplications and no inversions.
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
}

/// Evaluates every $\hat{W}_k$ at the domain element `index` selects, in increasing $k$.
///
/// Bit `i` of `index` selects whether $\beta_i$ is XORed into the point.
/// That matches [`BinarySubspace::get`](crate::BinarySubspace::get).
///
/// Each $\hat{W}_k$ is $\mathbb{F}_2$-linear and vanishes on $\beta_0$ through $\beta_{k-1}$.
/// Its value is therefore a subset sum over row $k$ of the domain context's table.
/// That table *is* `domain_context.subspace(l - k)`, and the subset sum *is* its `get`.
/// So this route needs no precomputation of its own.
///
/// This is the route a verifier takes for a sampled query index.
/// Summing constants costs a recursive circuit what the FRI fold already pays for its twiddles.
///
/// # Panics
/// Panics if `index` is at least `2^log_domain_size`.
pub fn evals_at_domain_index<F, DC>(domain_context: &DC, index: usize) -> Vec<F>
where
	F: BinaryField,
	DC: DomainContext<Field = F>,
{
	let l = domain_context.log_domain_size();
	assert!(index < 1 << l, "precondition: index must be less than 2^log_domain_size");

	// Row k starts at $\beta_k$, so shifting the index past the first k bits aligns them.
	(0..l)
		.map(|k| domain_context.subspace(l - k).get(index >> k))
		.collect()
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, Ghash128b, util::expand_subset_products};
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		BinarySubspace, FieldBuffer,
		bit_reverse::reverse_bits,
		inner_product::inner_product_scalars,
		ntt::{
			AdditiveNTT, NeighborsLastSingleThread,
			domain_context::{GaoMateerOnTheFly, GaoMateerPreExpanded, GenericPreExpanded},
		},
		reed_solomon::ReedSolomonCode,
		test_utils::random_field_buffer,
	};

	type F = Ghash128b;

	/// A subspace whose first basis element is not 1, which is what makes `beta_0_inv` matter.
	///
	/// Scaling a basis by a nonzero constant is an $\mathbb{F}_2$-linear bijection.
	/// Independence therefore survives at every dimension.
	/// Perturbing each element on its own would not be safe.
	fn scaled_subspace(log_d: usize) -> BinarySubspace<F> {
		let scale = F::new(5);
		let basis = BinarySubspace::<F>::with_dim(log_d)
			.basis()
			.iter()
			.map(|&b| b * scale)
			.collect::<Vec<_>>();
		BinarySubspace::new_unchecked(basis)
	}

	/// Runs `check` against each domain context shape a caller might build from.
	///
	/// Gao-Mateer is the context [`ReedSolomonCode`] fixes, so it is the production-relevant case.
	fn for_each_context(log_d: usize, check: impl Fn(&dyn Fn(usize) -> BinarySubspace<F>)) {
		check(&|i| GaoMateerPreExpanded::<F>::generate(log_d).subspace(i));
		let standard = GenericPreExpanded::generate_from_subspace(&BinarySubspace::with_dim(log_d));
		check(&|i| standard.subspace(i));
		let scaled = GenericPreExpanded::generate_from_subspace(&scaled_subspace(log_d));
		check(&|i| scaled.subspace(i));
	}

	#[test]
	fn evals_at_basis_elements_match_the_domain_context() {
		for log_d in 1..7 {
			// Row k of the context's table is W_hat_k on beta_k..beta_{l-1}.
			// The arbitrary-point recurrence must reproduce it entry by entry.
			for_each_context(log_d, |subspace| {
				let dc = GenericPreExpanded::generate_from_subspace(&subspace(log_d));
				let polys = NormalizedSubspacePolys::new(&dc);
				let domain = subspace(log_d);
				for k in 0..log_d {
					for (j, &expected) in subspace(log_d - k).basis().iter().enumerate() {
						let got = polys.evals_at(domain.basis()[k + j])[k];
						assert_eq!(got, expected, "log_d={log_d} k={k} j={j}");
					}
				}
			});
		}
	}

	#[test]
	fn what_k_is_normalized_and_vanishes_below_its_subspace() {
		for log_d in 1..7 {
			for_each_context(log_d, |subspace| {
				let dc = GenericPreExpanded::generate_from_subspace(&subspace(log_d));
				let polys = NormalizedSubspacePolys::new(&dc);
				let domain = subspace(log_d);
				for k in 0..log_d {
					// W_hat_k vanishes on the subspace it is built from.
					for j in 0..k {
						assert_eq!(polys.evals_at(domain.basis()[j])[k], F::ZERO);
					}
					// And it is normalized to one at the next basis element.
					assert_eq!(polys.evals_at(domain.basis()[k])[k], F::ONE);
				}
			});
		}
	}

	#[test]
	#[should_panic(expected = "normalizer must be non-zero")]
	fn new_rejects_a_dependent_basis() {
		// beta_2 = beta_0 + beta_1 collapses the subspace, so W_hat_2's normalizer vanishes.
		let dependent = BinarySubspace::new_unchecked(vec![F::new(5), F::new(22), F::new(19)]);
		NormalizedSubspacePolys::new(&GenericPreExpanded::generate_from_subspace(&dependent));
	}

	#[test]
	fn what_k_vanishes_at_zero() {
		let dc = GaoMateerPreExpanded::<F>::generate(5);
		let polys = NormalizedSubspacePolys::new(&dc);
		// Every W_hat_k is F2-linear, so it sends zero to zero.
		assert!(polys.evals_at(F::ZERO).iter().all(|&w| w == F::ZERO));
	}

	#[test]
	fn domain_index_route_matches_arbitrary_point_route() {
		for log_d in 1..7 {
			for_each_context(log_d, |subspace| {
				let dc = GenericPreExpanded::generate_from_subspace(&subspace(log_d));
				let polys = NormalizedSubspacePolys::new(&dc);
				let domain = subspace(log_d);
				// The F2-linear subset sum must agree with the recurrence on every domain point.
				for index in 0..1 << log_d {
					assert_eq!(
						evals_at_domain_index(&dc, index),
						polys.evals_at(domain.get(index)),
						"log_d={log_d} index={index}"
					);
				}
			});
		}
	}

	#[test]
	#[should_panic(expected = "index must be less than 2^log_domain_size")]
	fn evals_at_domain_index_rejects_an_out_of_range_index() {
		let dc = GaoMateerPreExpanded::<F>::generate(3);
		evals_at_domain_index(&dc, 8);
	}

	/// The headline identity, checked against the NTT itself.
	fn assert_tensor_row_matches_ntt<DC>(log_d: usize, dc: DC, seed: u64)
	where
		DC: DomainContext<Field = F>,
	{
		let mut rng = StdRng::seed_from_u64(seed);
		let coeffs = random_field_buffer::<F>(&mut rng, log_d);

		let mut transformed = coeffs.clone();
		let ntt = NeighborsLastSingleThread::new(dc);
		ntt.forward_transform(transformed.as_mut_view(), 0, 0);

		for index in 0..1 << log_d {
			// Row `index` of the transform matrix is the tensor of W_hat_k at that domain point.
			let row = expand_subset_products(&evals_at_domain_index(ntt.domain_context(), index));
			let dot = inner_product_scalars(coeffs.as_ref().iter().copied(), row);
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

	/// Every row of `code`'s generator matrix, indexed by codeword position.
	///
	/// This is the module doc's recipe.
	/// Build on the codeword domain, truncate to the message dimension, then reverse.
	/// The domain context is built once, since generating one per row dominates the test.
	fn generator_rows(code: &ReedSolomonCode<F>) -> Vec<Vec<F>> {
		let dc = GaoMateerPreExpanded::<F>::generate(code.log_len());
		(0..1 << code.log_len())
			.map(|index| {
				let mut evals = evals_at_domain_index(&dc, index);
				evals.truncate(code.log_dim());
				evals.reverse();
				expand_subset_products(&evals)
			})
			.collect()
	}

	#[test]
	fn tensor_row_matches_reed_solomon_encoding() {
		for log_dim in 1..6 {
			for log_inv_rate in 1..4 {
				let code = ReedSolomonCode::<F>::new(log_dim, log_inv_rate);
				let ntt = NeighborsLastSingleThread::new(GaoMateerOnTheFly::<F>::generate(
					code.log_len(),
				));
				let mut rng = StdRng::seed_from_u64(7);
				let msg = random_field_buffer::<F>(&mut rng, log_dim);
				let codeword = code.encode_batch(&ntt, msg.as_view(), 0, &GlobalAllocator);

				for (index, row) in generator_rows(&code).into_iter().enumerate() {
					let dot = inner_product_scalars(msg.as_ref().iter().copied(), row);
					assert_eq!(
						dot,
						codeword.as_ref()[index],
						"log_dim={log_dim} log_inv_rate={log_inv_rate} index={index}"
					);
				}
			}
		}
	}

	#[test]
	fn interleaved_lanes_are_independent_codewords() {
		for log_dim in 1..5 {
			for log_inv_rate in 1..3 {
				for log_batch in 1..3 {
					let code = ReedSolomonCode::<F>::new(log_dim, log_inv_rate);
					let ntt = NeighborsLastSingleThread::new(GaoMateerOnTheFly::<F>::generate(
						code.log_len(),
					));
					let mut rng = StdRng::seed_from_u64(11);
					let msg = random_field_buffer::<F>(&mut rng, log_dim + log_batch);
					let codeword =
						code.encode_batch(&ntt, msg.as_view(), log_batch, &GlobalAllocator);
					let rows = generator_rows(&code);

					for lane in 0..1 << log_batch {
						// A lane's columns sit under the bit-reversed lane index in the message.
						let base = reverse_bits(lane, log_batch as u32) << log_dim;
						let lane_msg = (0..1 << log_dim)
							.map(|j| msg.as_ref()[base | j])
							.collect::<Vec<_>>();
						let lane_msg = FieldBuffer::<F, Vec<F>>::new(log_dim, lane_msg);

						// And that lane is the plain encoding of exactly those columns.
						for (index, row) in rows.iter().enumerate() {
							let dot = inner_product_scalars(
								lane_msg.as_ref().iter().copied(),
								row.iter().copied(),
							);
							assert_eq!(
								dot,
								codeword.as_ref()[(index << log_batch) | lane],
								"log_dim={log_dim} r={log_inv_rate} b={log_batch} lane={lane}"
							);
						}
					}
				}
			}
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
			let polys = NormalizedSubspacePolys::new(&dc);
			let (x, y) = (F::new(a as u128), F::new(b as u128));
			// Each W_hat_k is a subspace vanishing polynomial, hence additive over F2.
			let sum = polys.evals_at(x + y);
			let parts = std::iter::zip(polys.evals_at(x), polys.evals_at(y));
			for (got, (wx, wy)) in std::iter::zip(sum, parts) {
				prop_assert_eq!(got, wx + wy);
			}
		}
	}
}
