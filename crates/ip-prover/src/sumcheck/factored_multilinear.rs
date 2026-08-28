// Copyright 2026 The Binius Developers

//! A dense multilinear held as a product of factors over disjoint runs of variables.

use binius_compute::BufferData;
use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, multilinear::fold::fold_highest_var_inplace};

/// A multilinear that factorizes across disjoint runs of its variables.
///
/// Some weights are a product of small pieces rather than one table.
/// An equality indicator over several axes factorizes across them, for instance.
///
/// Storing such a weight whole costs the product of the pieces' lengths.
/// Storing the pieces costs their sum.
///
/// ```text
///     whole:   2^(n_1 + n_2 + n_3) entries
///     factors: 2^n_1 + 2^n_2 + 2^n_3 entries
/// ```
///
/// The saving is what lets a sumcheck range over a space far too large to materialize, so long as
/// nothing ever asks for the whole table at once.
///
/// # Variable order
///
/// Factors are held lowest run first.
///
/// A factor over `n` variables owns the next `n` bits of the index, above the bits the factors
/// before it own. So the last factor owns the highest variables, and is the one a fold consumes
/// first.
///
/// ```text
///     index = [ factor 0 bits | factor 1 bits | ... | last factor bits ]
///               lowest                                       highest
/// ```
///
/// # Examples
///
/// ```ignore
/// // A weight over an operand axis, a shift axis, and a value axis.
/// let mut weight = FactoredMultilinear::new(vec![operand, shift, value]);
/// let at_index = weight.get(index);
/// weight.fold_highest_var(challenge);
/// ```
/// Where the factors' packed words live.
///
/// Defaults to the heap, and the point of the parameter is that it need not be.
///
/// A caller proving out of an arena hands over arena-backed factors.
/// This holds them as they are, rather than forcing a copy onto the heap.
#[derive(Debug, Clone)]
pub struct FactoredMultilinear<P: PackedField, Data: BufferData<P> = Vec<P>> {
	/// The factors still holding variables, lowest run first.
	///
	/// A factor is dropped once folding has bound every one of its variables.
	factors: Vec<FieldBuffer<P, Data>>,

	/// The product of every factor already bound away.
	///
	/// A factor that runs out of variables holds one value, which multiplies in here.
	/// Keeping it separate is what lets the factor list shrink instead of carrying empty buffers.
	bound: P::Scalar,
}

impl<P: PackedField, Data: BufferData<P>> FactoredMultilinear<P, Data> {
	/// Builds a multilinear from its factors, lowest variable run first.
	///
	/// A factor with no variables is folded straight into the bound product rather than kept,
	/// since it holds a value and no axis.
	pub fn new(factors: impl IntoIterator<Item = FieldBuffer<P, Data>>) -> Self {
		let mut bound = P::Scalar::ONE;
		let factors = factors
			.into_iter()
			.filter(|factor| {
				if factor.log_len() == 0 {
					bound *= factor.get(0);
					false
				} else {
					true
				}
			})
			.collect();
		Self { factors, bound }
	}

	/// The number of variables still free.
	pub fn n_vars(&self) -> usize {
		self.factors.iter().map(FieldBuffer::log_len).sum()
	}

	/// The value at one vertex of the hypercube over the free variables.
	///
	/// Each factor reads the bits it owns, and the results multiply.
	/// So a lookup costs one multiplication per factor rather than one per variable.
	///
	/// # Panics
	///
	/// Panics if the index does not fit the free variables.
	pub fn get(&self, index: usize) -> P::Scalar {
		let n_vars = self.n_vars();
		assert!(
			n_vars >= usize::BITS as usize || index < 1 << n_vars,
			"precondition: index {index} must fit {n_vars} variables"
		);

		let mut value = self.bound;
		let mut rest = index;
		for factor in &self.factors {
			// The factor's own bits are the low ones of what is left.
			let mask = (1 << factor.log_len()) - 1;
			value *= factor.get(rest & mask);
			rest >>= factor.log_len();
		}
		value
	}

	/// Fixes the highest free variable to a value.
	///
	/// The highest variable belongs to the last factor, so that is the factor this folds.
	/// A factor whose variables are all bound holds one value, which moves into the bound product.
	///
	/// # Panics
	///
	/// Panics if no variable is free.
	pub fn fold_highest_var(&mut self, challenge: P::Scalar) {
		let factor = self
			.factors
			.last_mut()
			.expect("precondition: at least one variable must be free");
		fold_highest_var_inplace(factor, challenge);

		// A factor down to a single value is no longer an axis, so it leaves the list.
		if factor.log_len() == 0 {
			self.bound *= factor.get(0);
			self.factors.pop();
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Ghash128b, PackedBinaryGhash1x128b, Random, arch::OptimalPackedB128};
	use proptest::prelude::*;
	use rand::{SeedableRng, prelude::StdRng};

	use super::*;

	/// The product of the factors, materialized in full, as the reference to pin against.
	fn materialize<P: PackedField>(factors: &[FieldBuffer<P>]) -> FieldBuffer<P> {
		let n_vars: usize = factors.iter().map(FieldBuffer::log_len).sum();
		let values = (0..1usize << n_vars)
			.map(|index| {
				let mut rest = index;
				let mut value = P::Scalar::ONE;
				for factor in factors {
					let mask = (1 << factor.log_len()) - 1;
					value *= factor.get(rest & mask);
					rest >>= factor.log_len();
				}
				value
			})
			.collect::<Vec<_>>();
		FieldBuffer::from_values(&values)
	}

	fn random_factors<P: PackedField>(rng: &mut StdRng, shape: &[usize]) -> Vec<FieldBuffer<P>>
	where
		P::Scalar: Random,
	{
		shape
			.iter()
			.map(|&log_len| {
				let values = (0..1usize << log_len)
					.map(|_| P::Scalar::random(&mut *rng))
					.collect::<Vec<_>>();
				FieldBuffer::from_values(&values)
			})
			.collect()
	}

	#[test]
	fn a_factor_with_no_variables_folds_into_the_product() {
		// Invariant: a factor holding one value is a scalar, not an axis.
		//
		// Keeping it in the list would leave a buffer a fold could never consume, and the variable
		// count would then disagree with what folding can bind.
		type P = PackedBinaryGhash1x128b;
		let mut rng = StdRng::seed_from_u64(1);

		let factors = random_factors::<P>(&mut rng, &[0, 2, 0]);
		let expected = materialize(&factors);

		let factored = FactoredMultilinear::new(factors);

		// Only the two-variable factor remains an axis.
		assert_eq!(factored.n_vars(), 2);
		for index in 0..4 {
			assert_eq!(factored.get(index), expected.get(index));
		}
	}

	#[test]
	fn folding_consumes_the_factors_from_the_highest_variable_down() {
		// Invariant: the last factor owns the highest variables.
		//
		// Fixture state: three factors over 1, 2 and 1 variables, so four in total.
		//
		//     index bits:  [ f0 : 1 | f1 : 2 | f2 : 1 ]
		//                    low                 high
		//
		// Folding once must bind f2 away entirely, leaving three variables in two factors.
		type P = PackedBinaryGhash1x128b;
		let mut rng = StdRng::seed_from_u64(2);

		let factors = random_factors::<P>(&mut rng, &[1, 2, 1]);
		let mut factored = FactoredMultilinear::new(factors.clone());
		let mut reference = materialize(&factors);
		assert_eq!(factored.n_vars(), 4);

		let challenge = Ghash128b::random(&mut rng);
		factored.fold_highest_var(challenge);
		fold_highest_var_inplace(&mut reference, challenge);

		// The one-variable top factor is gone, so its value now rides the bound product.
		assert_eq!(factored.n_vars(), 3);
		for index in 0..8 {
			assert_eq!(factored.get(index), reference.get(index), "index {index}");
		}
	}

	#[test]
	fn factors_may_live_in_an_arena_rather_than_the_heap() {
		// Invariant: where a factor's words live is the caller's choice, not this type's.
		//
		// A prover working out of an arena builds its weights there.
		// Forcing them onto the heap would mean a copy per factor.
		//
		// The arena exists to avoid exactly that.
		//
		// Fixture state: the same two factors twice, one pair on the heap and one in an arena.
		//
		//     heap-backed  -> value at each vertex
		//     arena-backed -> the same value at each vertex
		//
		// Folding both and comparing at every step is what shows the storage is invisible.
		type P = PackedBinaryGhash1x128b;
		let mut rng = StdRng::seed_from_u64(23);
		let alloc = GlobalAllocator;

		let shape = [2usize, 1];
		let scalars = shape
			.iter()
			.map(|&log_len| {
				(0..1usize << log_len)
					.map(|_| Ghash128b::random(&mut rng))
					.collect::<Vec<_>>()
			})
			.collect::<Vec<_>>();

		let mut heap = FactoredMultilinear::<P>::new(
			scalars
				.iter()
				.map(|values| FieldBuffer::<P>::from_values(values)),
		);
		let mut arena = FactoredMultilinear::new(
			scalars
				.iter()
				.map(|values| FieldBuffer::<P>::from_values_in(&alloc, values)),
		);

		assert_eq!(arena.n_vars(), heap.n_vars());

		// Bind every variable, comparing the whole table after each one.
		while heap.n_vars() > 0 {
			for index in 0..1usize << heap.n_vars() {
				assert_eq!(arena.get(index), heap.get(index), "index {index}");
			}
			let challenge = Ghash128b::random(&mut rng);
			heap.fold_highest_var(challenge);
			arena.fold_highest_var(challenge);
		}
		assert_eq!(arena.get(0), heap.get(0));
	}

	proptest! {
		/// Folding a factored weight tracks folding the product it stands for, at every step.
		///
		/// This is the whole contract: a caller may use the factors in place of the table, and the
		/// two must agree after any sequence of challenges, not only at the start.
		#[test]
		fn folding_tracks_the_materialized_product(
			shape in prop::collection::vec(0usize..=3, 1..=4),
			seed: u64,
		) {
			type P = OptimalPackedB128;
			let mut rng = StdRng::seed_from_u64(seed);

			let factors = random_factors::<P>(&mut rng, &shape);
			let mut factored = FactoredMultilinear::new(factors.clone());
			let mut reference = materialize(&factors);

			// The materialized reference is the product, so the two must start equal.
			prop_assert_eq!(factored.n_vars(), reference.log_len());
			for index in 0..reference.len() {
				prop_assert_eq!(factored.get(index), reference.get(index));
			}

			// Bind every variable, comparing after each one.
			while factored.n_vars() > 0 {
				let challenge = Ghash128b::random(&mut rng);
				factored.fold_highest_var(challenge);
				fold_highest_var_inplace(&mut reference, challenge);

				prop_assert_eq!(factored.n_vars(), reference.log_len());
				for index in 0..reference.len() {
					prop_assert_eq!(factored.get(index), reference.get(index));
				}
			}

			// Fully bound, both are the single value the whole product folded to.
			prop_assert_eq!(factored.get(0), reference.get(0));
		}
	}
}
