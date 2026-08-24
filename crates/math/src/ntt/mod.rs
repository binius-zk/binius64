// Copyright 2024-2025 Irreducible Inc.

//! Efficient implementations of the binary field additive NTT.
//!
//! See [LCH14] and [DP24] Section 2.3 for mathematical background.
//!
//! [LCH14]: <https://arxiv.org/abs/1404.3458>
//! [DP24]: <https://eprint.iacr.org/2024/504>

pub mod domain_context;
mod neighbors_last;
mod reference;
pub mod subspace_polys;
#[cfg(test)]
mod tests_evaluation;
#[cfg(test)]
pub mod tests_reference;

use binius_field::{BinaryField, PackedField};
pub use neighbors_last::{
	NeighborsLastBreadthFirst, NeighborsLastMultiThread, NeighborsLastSingleThread,
};
pub use reference::NeighborsLastReference;

use crate::{binary_subspace::BinarySubspace, field_buffer::FieldSliceMut};

/// The binary field additive NTT.
///
/// A number-theoretic transform (NTT) is a linear transformation on a finite field analogous to
/// the discrete fourier transform. The version of the additive NTT we use is originally described
/// in [LCH14]. In [DP24] Section 4.1, the authors present the LCH additive NTT algorithm in a way
/// that makes apparent its compatibility with the FRI proximity test. Throughout the
/// documentation, we will refer to the notation used in [DP24].
///
/// The additive NTT is parameterized by a binary field $K$ and $\mathbb{F}\_2$-linear subspace. We
/// write $\beta_0, \ldots, \beta_{\ell-1}$ for the ordered basis elements of the subspace. The
/// basis determines a novel polynomial basis and an evaluation domain. In the forward direction,
/// the additive NTT transforms a vector of polynomial coefficients, with respect to the novel
/// polynomial basis, into a vector of their evaluations over the evaluation domain. The inverse
/// transformation interpolates polynomial values over the domain into novel polynomial basis
/// coefficients.
///
/// An [`AdditiveNTT`] implementation with a maximum domain dimension of $\ell$ can be applied on
/// a sequence of $\ell + 1$ evaluation domains of sizes $2^0, \ldots, 2^\ell$. These are the
/// domains $S^{(\ell)}, S^{(\ell - 1)}, \ldots, S^{(0)}$ defined in [DP24] Section 4.
///
/// The methods [`Self::forward_transform`] and [`Self::inverse_transform`] take three parameters:
/// a `data` buffer with length `2^log_len`, `skip_early`, and `skip_late`. The number of total NTT
/// layers is considered to be `log_len`. "Early" layers at the beginning of the forward transform
/// or "late" layers at the end of the forward transform may be skipped. The NTT uses
/// $S^{(\ell - n)}$ for the evaluation domain for an $n$-layer NTT. (Remember, the novel polynomial
/// basis is itself parameterized by the evaluation domain.) Counterintuitively, the space
/// $S^{(n+1)}$ is not necessarily a subset of $S^{(n)}$**. We choose this behavior for the
/// [`AdditiveNTT`] trait because it facilitates compatibility with FRI when batching proximity
/// tests for codewords of different dimensions.
///
/// [LCH14]: <https://arxiv.org/abs/1404.3458>
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub trait AdditiveNTT {
	type Field: BinaryField;

	/// Forward transformation as defined in [DP24], Section 2.3.
	///
	/// Arguments:
	/// - `data` is the data on which the NTT is performed.
	/// - `skip_early` is the number of early layers that should be skipped
	/// - `skip_late` is the number of late layers that should be skipped
	///
	/// ## Preconditons
	///
	/// - `skip_early + skip_late <= data.log_len()`
	/// - `data.log_len() - skip_late <= self.log_domain_size()`
	///
	/// [DP24]: <https://eprint.iacr.org/2024/504>
	fn forward_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		data: FieldSliceMut<P>,
		skip_early: usize,
		skip_late: usize,
	);

	/// Inverse transformation of [`Self::forward_transform`].
	///
	/// Note that "early" layers here refer to "early" time in the forward transform, i.e. layers
	/// with low index in the forward transform.
	///
	/// ## Preconditions
	///
	/// - same as [`Self::forward_transform`]
	fn inverse_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		data: FieldSliceMut<P>,
		skip_early: usize,
		skip_late: usize,
	);

	/// Transpose of the linear map [`Self::forward_transform`] computes.
	///
	/// Write `M` for the matrix of the forward transform at the same skips.
	/// This applies `M^T`.
	/// The defining property is the adjoint identity
	///
	/// ```text
	///     <transpose_transform(a), m> = <a, forward_transform(m)>
	/// ```
	///
	/// which holds for every `a` and `m` and is what the tests pin.
	///
	/// This is not [`Self::inverse_transform`].
	/// The forward butterfly `[[1, t], [1, 1 + t]]` has determinant 1.
	/// Its inverse is therefore `[[1 + t, t], [1, 1]]`.
	/// Its transpose is `[[1, 1], [t, 1 + t]]`.
	/// The two agree only when `t` is zero.
	///
	/// A caller reaches for this when it needs `G^T a` for a generator matrix `G`.
	/// Building that column by column costs one generator row per nonzero of `a`.
	/// One transposed pass costs `O(2^log_len * log_len)` however many nonzeros there are.
	///
	/// The default body mirrors the reference forward transform and is correspondingly slow.
	/// An implementation with a fast forward transform should override it.
	///
	/// ## Preconditions
	///
	/// - same as [`Self::forward_transform`]
	fn transpose_transform<P: PackedField<Scalar = Self::Field>>(
		&self,
		mut data: FieldSliceMut<P>,
		skip_early: usize,
		skip_late: usize,
	) {
		let log_d = data.log_len();
		let domain_context = self.domain_context();
		reference::input_check(domain_context, log_d, skip_early, skip_late);

		// Transposing a composition reverses it, so the layers run backwards and each keeps the
		// twiddle it used going forward.
		for layer in (skip_early..(log_d - skip_late)).rev() {
			let num_blocks = 1 << layer;
			let block_size_half = 1 << (log_d - layer - 1);
			for block in 0..num_blocks {
				let twiddle = domain_context.twiddle(layer, block);
				let block_start = block << (log_d - layer);
				for idx0 in block_start..(block_start + block_size_half) {
					let idx1 = block_size_half | idx0;
					// The transposed butterfly, which is the forward one with its two steps
					// swapped: `u'' = u + v` and then `v'' = v + t * u''`.
					let mut u = data.get(idx0);
					let mut v = data.get(idx1);
					u += v;
					v += u * twiddle;
					data.set(idx0, u);
					data.set(idx1, v);
				}
			}
		}
	}

	/// The associated [`DomainContext`].
	fn domain_context(&self) -> &impl DomainContext<Field = Self::Field>;

	/// See [`DomainContext::log_domain_size`].
	fn log_domain_size(&self) -> usize {
		self.domain_context().log_domain_size()
	}

	/// See [`DomainContext::subspace`].
	fn subspace(&self, i: usize) -> BinarySubspace<Self::Field> {
		self.domain_context().subspace(i)
	}

	/// See [`DomainContext::twiddle`].
	fn twiddle(&self, i: usize, j: usize) -> Self::Field {
		self.domain_context().twiddle(i, j)
	}
}

/// Provides information about the domains $S^{(i)}$ and the associated twiddle factors.
///
/// Needed by the NTT and by FRI.
pub trait DomainContext {
	type Field: BinaryField;

	/// Base 2 logarithm of the size of $S^{(0)}$, i.e., $\ell$.
	///
	/// In other words: Index of the first layer that can _not_ be computed anymore.
	/// I.e., number of the latest layer that _can_ be computed, plus one.
	/// Layers are indexed starting from 0.
	///
	/// If you intend to call the NTT with `skip_late = 0`, then this should be equal to the base 2
	/// logarithm of the number of scalars in the input.
	fn log_domain_size(&self) -> usize;

	/// Returns the binary subspace with dimension $i$.
	///
	/// In [DP24], this subspace is referred to as $S^{(\ell - i)}$, where $\ell$ is the maximum
	/// domain size of the NTT, i.e., `self.log_domain_size()`. We choose to reverse the indexing
	/// order with respect to the paper because it is more natural in code that the $i$th subspace
	/// has dimension $i$.
	///
	/// ## Preconditions
	///
	/// - `i` must be less than or equal to `self.log_domain_size()`
	///
	/// [DP24]: <https://eprint.iacr.org/2024/504>
	fn subspace(&self, i: usize) -> BinarySubspace<Self::Field>;

	/// Returns the twiddle of a certain block in a certain layer.
	///
	/// The layer numbers start from 0, i.e., the earliest layer is layer 0.
	///
	/// Let $i$ be `layer`, and $j$ be `block`. This returns
	///
	/// $$
	/// S^{(\ell - i - 1)}_{2j} = \hat{W}_{\ell - i - 1}\left( \sum_{b = 0}^{i-1} j_b \beta_{\ell -
	/// i + b} \right) $$
	///
	/// The equality above is a consequence of Corollary 4.5 from [DP24].
	///
	/// ## Preconditions
	///
	/// - `layer < self.log_domain_size()`
	/// - `block < 2^layer`
	///
	/// [DP24]: <https://eprint.iacr.org/2024/504>
	fn twiddle(&self, layer: usize, block: usize) -> Self::Field;

	/// Returns an iterator over all twiddles in a layer.
	///
	/// For layer `i`, this iterates over `twiddle(i, block)` for `block` in `0..2^i`.
	///
	/// ## Preconditions
	///
	/// - `layer < self.log_domain_size()`
	fn iter_twiddles(
		&self,
		layer: usize,
		log_step_by: usize,
	) -> impl Iterator<Item = Self::Field> + '_ {
		(0..1 << (layer - log_step_by)).map(move |block| self.twiddle(layer, block << log_step_by))
	}
}

/// Make it so that references to a [`DomainContext` implement [`DomainContext`] themselves.
///
/// This is useful, for example, if you need two objects that each want to _own_ a
/// [`DomainContext`], but you don't want to clone the [`DomainContext`].
impl<T: DomainContext> DomainContext for &T {
	type Field = T::Field;

	fn log_domain_size(&self) -> usize {
		(*self).log_domain_size()
	}

	fn subspace(&self, i: usize) -> BinarySubspace<Self::Field> {
		(*self).subspace(i)
	}

	fn twiddle(&self, layer: usize, block: usize) -> Self::Field {
		(*self).twiddle(layer, block)
	}

	fn iter_twiddles(&self, layer: usize, log_step_by: usize) -> impl Iterator<Item = Self::Field> {
		(*self).iter_twiddles(layer, log_step_by)
	}
}

#[cfg(test)]
mod tests {
	//! The adjoint identity is the whole specification of the transpose, and it is basis free:
	//!
	//! ```text
	//!     <transpose_transform(a), m> = <a, forward_transform(m)>
	//! ```
	//!
	//! An error anywhere in the layer order, the butterfly, or the twiddle indexing breaks it.
	//! Checking against `forward_transform` is what makes a separate reference unnecessary.

	use binius_field::PackedField;
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::{AdditiveNTT, NeighborsLastReference, domain_context::GaoMateerPreExpanded};
	use crate::{
		inner_product::inner_product_scalars,
		test_utils::{B128, Packed128b, random_field_buffer},
	};

	/// Asserts the adjoint identity at one shape, over the field `P`.
	fn assert_adjoint<P: PackedField<Scalar = B128>>(
		log_n: usize,
		skip_early: usize,
		skip_late: usize,
		seed: u64,
	) {
		let domain_context = GaoMateerPreExpanded::generate(log_n);
		let ntt = NeighborsLastReference {
			domain_context: &domain_context,
		};
		let mut rng = StdRng::seed_from_u64(seed);

		let a = random_field_buffer::<P>(&mut rng, log_n);
		let m = random_field_buffer::<P>(&mut rng, log_n);

		let mut transposed = a.clone();
		ntt.transpose_transform(transposed.to_mut(), skip_early, skip_late);

		let mut forward = m.clone();
		ntt.forward_transform(forward.to_mut(), skip_early, skip_late);

		assert_eq!(
			inner_product_scalars(transposed.iter_scalars(), m.iter_scalars()),
			inner_product_scalars(a.iter_scalars(), forward.iter_scalars()),
			"log_n={log_n} skip_early={skip_early} skip_late={skip_late}"
		);
	}

	proptest! {
		#[test]
		fn transpose_is_the_adjoint_of_the_forward_transform(seed: u64, log_n in 1usize..8) {
			// Every split of the layer range, including the two that apply no layer at all.
			// A scalar and a packed field take different `get` and `set` paths, so both run.
			for skip_early in 0..=log_n {
				for skip_late in 0..=(log_n - skip_early) {
					assert_adjoint::<B128>(log_n, skip_early, skip_late, seed);
					assert_adjoint::<Packed128b>(log_n, skip_early, skip_late, seed);
				}
			}
		}
	}
}
