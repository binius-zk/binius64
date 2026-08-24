// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The expansion of a point over a cube, as a plan you then materialize.
//!
//! Building one is a product of two independent choices:
//!
//! ```text
//! seed       one, or a constant every coefficient is scaled by
//! storage    a fresh store, one from an allocator, one the caller supplies, or plain scalars
//! ```
//!
//! Naming that product takes one method per cell, and leaves cells unnamed.
//! So the point and the seed are gathered first, and a terminal picks the storage.
//!
//! Appending the coordinate `r` doubles the length, turning every value `v` into a pair:
//!
//! ```text
//! before   [ v_0            v_1            ]
//! after    [ v_0 * b_0(r)   v_1 * b_0(r)   |   v_0 * b_1(r)   v_1 * b_1(r) ]
//! ```
//!
//! The two halves sit one after the other, so the appended variable is the highest indexed one.

use std::{iter, slice};

use binius_compute::Allocator;
use binius_field::{Field, PackedField, field::FieldOps};
use binius_utils::{
	buffer::VecLike,
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};

use super::Hypercube;
use crate::{FieldBuffer, FieldVec};

/// The tensor expansion of a point over one cube, before it is materialized.
///
/// For the point `r = (r_0, ..., r_{n-1})` the expansion holds the `2^n` coefficients
///
/// ```text
/// b(r_0) (x) ... (x) b(r_{n-1})
/// ```
///
/// which are the coefficients of the equality indicator `eq(X_0, ..., X_{n-1}, r)` over the cube.
/// A seed other than one scales every coefficient.
///
/// Only the recipe is held here: a borrow of the point, and the seed.
/// Nothing is computed until a terminal chooses where the coefficients land.
#[must_use = "an expansion computes nothing until it is built"]
pub struct Expansion<'a, F> {
	cube: Hypercube,
	point: &'a [F],
	scale: F,
}

impl<'a, F: FieldOps> Expansion<'a, F> {
	/// Begins an unscaled expansion of a point.
	pub(super) fn new(cube: Hypercube, point: &'a [F]) -> Self {
		// A seed of one leaves every coefficient as the basis product alone.
		Self {
			cube,
			point,
			scale: F::one(),
		}
	}

	/// Scales every coefficient of the expansion by a constant.
	///
	/// A scale of one is the identity, since the expansion is linear in its seed.
	pub fn scaled_by(self, scale: F) -> Self {
		Self { scale, ..self }
	}

	/// Builds the expansion as one scalar per cube vertex.
	///
	/// This is the scalar-only engine, which never touches a packed store.
	pub fn build_scalars(self) -> Vec<F> {
		// One coefficient per cube vertex, allocated once.
		let mut result = Vec::with_capacity(1 << self.point.len());
		// Seed with the scale, which every later multiplication carries through.
		result.push(self.scale);

		for r_i in self.point {
			// Each coordinate doubles the length.
			// The low half takes the constant basis factor, the appended high half the linear one.
			//
			//     read index j  ->  overwrite result[j], push its partner past the end
			//
			// Walking the low half front to back is safe, since pushing only appends past it.
			let len = result.len();
			for j in 0..len {
				let [lo, hi] = self.cube.expand_var(&result[j], r_i);
				result[j] = lo;
				result.push(hi);
			}
		}
		result
	}
}

impl<'a, F: Field> Expansion<'a, F> {
	/// Builds the expansion into a fresh store.
	pub fn build<P: PackedField<Scalar = F>>(self) -> FieldBuffer<P> {
		// Reserving the final packed length keeps the per-variable growth reallocation free.
		let packed_len = Self::packed_words::<P>(self.point.len());
		self.build_into(Vec::with_capacity(packed_len))
	}

	/// Builds the expansion into a store drawn from an allocator.
	///
	/// Backed by a pool, the result is a recyclable buffer rather than a fresh allocation.
	pub fn build_in<P: PackedField<Scalar = F>, A: Allocator>(self, alloc: &A) -> FieldVec<P, A> {
		// The allocator hands out the final packed length, which the expansion never outgrows.
		let packed_len = Self::packed_words::<P>(self.point.len());
		self.build_into(alloc.alloc::<P>(packed_len))
	}

	/// Builds the expansion into a store the caller supplies.
	///
	/// This is the allocation-hoisting form.
	/// The caller owns the store, so it can be drawn from a pool.
	/// It can equally be reserved on a different thread than the one that fills it.
	///
	/// # Preconditions
	///
	/// * The store's capacity must cover the packed length of the expansion.
	pub fn build_into<P: PackedField<Scalar = F>, Data: VecLike<P>>(
		self,
		mut store: Data,
	) -> FieldBuffer<P, Data> {
		assert!(
			store.capacity() >= Self::packed_words::<P>(self.point.len()),
			"precondition: store capacity must cover the packed expansion length"
		);

		// Seed a one-coefficient expansion with the scale.
		// Appending the coordinates multiplies it through, so every coefficient ends up scaled.
		store.clear();
		store.push(P::from_scalars(iter::once(self.scale)));

		self.append_onto(FieldBuffer::new(0, store))
	}

	/// Appends this expansion's variables onto an expansion already built.
	///
	/// Take `n` values and this expansion's `k` coordinates `r = (r_0, ..., r_{k-1})`.
	/// The result is the tensor product of those values with the basis at every coordinate:
	///
	/// ```text
	/// v (x) b(r_0) (x) ... (x) b(r_{k-1})
	/// ```
	///
	/// It holds `2^(n + k)` coefficients, one variable added per coordinate.
	///
	/// Read as polynomials, the input holds an `n`-variate multilinear `f`.
	/// The output then holds the `(n + k)`-variate multilinear
	///
	/// ```text
	/// g(X_0, ..., X_{n+k-1}) = f(X_0, ..., X_{n-1}) * eq(X_n, ..., X_{n+k-1}, r)
	/// ```
	///
	/// The values already present are the seed, so this expansion's own seed plays no part.
	pub fn append_to<P: PackedField<Scalar = F>>(
		self,
		values: FieldBuffer<P, Vec<P>>,
	) -> FieldBuffer<P, Vec<P>> {
		let start_log_len = values.log_len();
		let final_log_len = start_log_len + self.point.len();
		let mut data = values.into_inner();

		// Reserve the whole final capacity once, so no round reallocates.
		// Each round then writes its new coefficients straight into the reserved spare capacity,
		// instead of zero-initializing a region the expansion immediately overwrites.
		let final_packed_len = Self::packed_words::<P>(final_log_len);
		data.reserve_exact(final_packed_len.saturating_sub(data.len()));

		self.append_onto(FieldBuffer::new(start_log_len, data))
	}

	/// The number of packed words an expansion of that many variables occupies.
	///
	/// Below one packed word the count is one, since a single word backs any shorter length.
	const fn packed_words<P: PackedField>(log_len: usize) -> usize {
		1usize << log_len.saturating_sub(P::LOG_WIDTH)
	}

	/// Appends one variable per coordinate to a store that already has room for the result.
	///
	/// # Preconditions
	///
	/// * The store's capacity must cover the packed length of the final expansion.
	fn append_onto<P: PackedField<Scalar = F>, Data: VecLike<P>>(
		self,
		values: FieldBuffer<P, Data>,
	) -> FieldBuffer<P, Data> {
		let start_log_len = values.log_len();
		let final_log_len = start_log_len + self.point.len();
		let mut data = values.into_inner();

		// precondition
		debug_assert!(data.capacity() >= Self::packed_words::<P>(final_log_len));

		// The coordinates split cleanly in two at the packing width:
		//
		//     narrower than one word   the whole expansion lives in data[0]
		//     one word or wider        every round doubles the packed length
		let sub_width_count = self
			.point
			.len()
			.min(P::LOG_WIDTH.saturating_sub(start_log_len));
		let (sub_width_coords, packed_coords) = self.point.split_at(sub_width_count);

		// Sub-packing-width rounds: both halves of the result share the single word data[0].
		// Split that word into its two halves, expand them, and interleave them back together.
		// The backing store stays one element long throughout.
		for (i, &r_i) in sub_width_coords.iter().enumerate() {
			let log_len = start_log_len + i;
			let packed_r_i = P::broadcast(r_i);
			let (lo, _) = data[0].interleave(P::zero(), log_len);
			let [lo, hi] = self.cube.expand_var(&lo, &packed_r_i);
			data[0] = lo.interleave(hi, log_len).0;
		}

		// Packed rounds: the initialized words are exactly the low half of the result.
		//
		//     low half    the initialized prefix, expanded in place
		//     high half   reserved spare capacity, written once
		for &r_i in packed_coords {
			let packed_r_i = P::broadcast(r_i);
			let old_packed = data.len();

			// The safe two-slice split of a Vec into its initialized prefix and its spare capacity
			// is still unstable (rust-lang/rust#81944).
			// So the spare half comes from the safe accessor and the initialized half from a raw
			// part.
			let low_ptr = data.as_mut_ptr();
			let high = &mut data.spare_capacity_mut()[..old_packed];
			// SAFETY: `[0, old_packed)` is the initialized low half, disjoint from the spare `high`
			// half `[old_packed, 2 * old_packed)`; the two slices never overlap.
			let low = unsafe { slice::from_raw_parts_mut(low_ptr, old_packed) };
			// Each round doubles the expansion, starting from a single word.
			// So the first rounds are far too small to be worth splitting across threads.
			(low, high)
				.into_par_iter()
				.with_min_task(WorkPerItem::FieldMuls)
				.for_each(|(low_i, high_i)| {
					let [new_low, new_high] = self.cube.expand_var(low_i, &packed_r_i);
					*low_i = new_low;
					high_i.write(new_high);
				});
			// SAFETY: the loop above initialized every one of the `old_packed` spare words.
			unsafe { data.set_len(2 * old_packed) };
		}

		FieldBuffer::new(final_log_len, data)
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_utils::rayon::task_size::min_len_for_work;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{B128, Packed128b, index_to_hypercube_point, random_scalars};

	type P = Packed128b;
	type F = B128;

	/// Both bases, so a shared property is checked once against each of them.
	const CUBES: [Hypercube; 2] = [Hypercube::One, Hypercube::Inf];

	#[test]
	fn every_storage_terminal_holds_the_same_coefficients() {
		let mut rng = StdRng::seed_from_u64(0);

		// Invariant: the storage choice never changes what is computed.
		//
		//     fresh store | allocator | caller's store | plain scalars
		//
		// All four must agree coefficient for coefficient, at every size.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let reference = Hypercube::One.expand(&point).build::<P>();

			let pooled = Hypercube::One
				.expand(&point)
				.build_in::<P, _>(&GlobalAllocator);
			assert!(pooled.iter_scalars().eq(reference.iter_scalars()), "pool at log_n={log_n}");

			let capacity = 1 << log_n.saturating_sub(P::LOG_WIDTH);
			let supplied = Hypercube::One
				.expand(&point)
				.build_into::<P, _>(Vec::with_capacity(capacity));
			assert_eq!(supplied, reference, "supplied store at log_n={log_n}");

			let scalars = Hypercube::One.expand(&point).build_scalars();
			assert!(reference.iter_scalars().eq(scalars), "scalars at log_n={log_n}");
		}
	}

	#[test]
	fn the_seed_scales_every_terminal_alike() {
		let mut rng = StdRng::seed_from_u64(1);

		// Invariant: the seed axis is independent of the storage axis.
		// So scaling commutes with every terminal, including a scaled expansion in pooled memory.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];
			let unscaled = Hypercube::One.expand(&point).build::<P>();

			let scaled = Hypercube::One.expand(&point).scaled_by(scale).build::<P>();
			for (got, base) in scaled.iter_scalars().zip(unscaled.iter_scalars()) {
				assert_eq!(got, scale * base, "fresh store at log_n={log_n}");
			}

			let pooled = Hypercube::One
				.expand(&point)
				.scaled_by(scale)
				.build_in::<P, _>(&GlobalAllocator);
			assert!(pooled.iter_scalars().eq(scaled.iter_scalars()), "pool at log_n={log_n}");

			let scalars = Hypercube::One
				.expand(&point)
				.scaled_by(scale)
				.build_scalars();
			assert!(scaled.iter_scalars().eq(scalars), "scalars at log_n={log_n}");
		}
	}

	#[test]
	fn a_seed_of_one_is_the_identity() {
		let mut rng = StdRng::seed_from_u64(2);

		// Invariant: the expansion is linear in its seed, so a seed of one changes nothing.
		// Equality is checked packed word by packed word, not just coefficient by coefficient.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			assert_eq!(
				Hypercube::One.expand(&point).scaled_by(F::ONE).build::<P>(),
				Hypercube::One.expand(&point).build::<P>(),
				"mismatch at log_n={log_n}"
			);
		}
	}

	#[test]
	fn a_seed_of_zero_gives_all_zeros() {
		let mut rng = StdRng::seed_from_u64(3);

		// The other end of that linearity: a seed of zero yields the all-zero polynomial.
		for log_n in [0, 1, 2, 5] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scaled = Hypercube::One
				.expand(&point)
				.scaled_by(F::ZERO)
				.build::<P>();
			assert!(scaled.iter_scalars().all(|v| v == F::ZERO), "nonzero at log_n={log_n}");
		}
	}

	#[test]
	fn the_last_seed_wins() {
		let mut rng = StdRng::seed_from_u64(4);

		// The seed is one slot, not an accumulator.
		// So setting it twice keeps the second value, and never their product.
		let point = random_scalars::<F>(&mut rng, 4);
		let [first, second] = [(); 2].map(|_| random_scalars::<F>(&mut rng, 1)[0]);

		assert_eq!(
			Hypercube::One
				.expand(&point)
				.scaled_by(first)
				.scaled_by(second)
				.build::<P>(),
			Hypercube::One.expand(&point).scaled_by(second).build::<P>()
		);
	}

	#[test]
	fn a_caller_reserved_store_matches_the_allocating_form() {
		let mut rng = StdRng::seed_from_u64(5);

		// Invariant: filling a caller-reserved store reproduces the allocating variant exactly,
		// with the store reserved to the exact packed capacity the routine demands.
		for log_n in [0, 1, 2, 5, 8] {
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			let capacity = 1 << log_n.saturating_sub(P::LOG_WIDTH);
			let result = Hypercube::One
				.expand(&point)
				.scaled_by(scale)
				.build_into::<P, _>(Vec::with_capacity(capacity));

			assert_eq!(result.log_len(), log_n, "wrong length at log_n={log_n}");
			assert_eq!(
				result,
				Hypercube::One.expand(&point).scaled_by(scale).build::<P>(),
				"mismatch at log_n={log_n}"
			);
		}
	}

	#[test]
	fn appending_onto_a_one_coefficient_store_builds_from_scratch() {
		let mut rng = StdRng::seed_from_u64(6);

		// The values already present are the seed.
		// So appending a whole point onto the single coefficient one is the plain expansion.
		let point = random_scalars::<F>(&mut rng, 5);
		let seed = FieldBuffer::<P, _>::scalar_with_capacity(F::ONE, point.len());

		assert_eq!(
			Hypercube::One.expand(&point).append_to::<P>(seed),
			Hypercube::One.expand(&point).build::<P>()
		);
	}

	#[test]
	fn appending_in_batches_matches_one_full_expansion() {
		let mut rng = StdRng::seed_from_u64(7);

		// Append coordinates in batches of growing size, reusing one reserved backing store.
		//
		//     batch sizes 1, 2, 3, 4  ->  1 + 2 + 3 + 4 = 10 variables in total
		let batches = 4;
		let max_n_vars = batches * (batches + 1) / 2;
		let mut coords = Vec::with_capacity(max_n_vars);
		let mut eq_expansion = FieldBuffer::<P, _>::scalar_with_capacity(F::ONE, max_n_vars);

		for batch_len in 1..=batches {
			let extra = random_scalars(&mut rng, batch_len);

			eq_expansion = Hypercube::One.expand(&extra).append_to::<P>(eq_expansion);
			coords.extend(&extra);

			// Every batch must leave the indicator over all coordinates appended so far.
			assert_eq!(eq_expansion.log_len(), coords.len());
			for i in 0..eq_expansion.len() {
				let vertex = index_to_hypercube_point(coords.len(), i);
				assert_eq!(eq_expansion.get(i), Hypercube::One.eq_ind(&vertex, &coords));
			}
		}
	}

	#[test]
	fn prepending_via_bit_reverse_matches_one_full_expansion() {
		let mut rng = StdRng::seed_from_u64(8);

		// Appending is the only primitive, so prepending a variable is spelled as
		//
		//     bit reverse  ->  append  ->  bit reverse
		//
		// which is how the binary switchover prover adds one variable per round.
		// Iterating it over ten coordinates also covers the sub-packing-width early rounds.
		let n_vars = 10;
		let point = random_scalars::<F>(&mut rng, n_vars);

		let mut tensor = FieldBuffer::<P>::from_values(&[F::ONE]);
		for &r in point.iter().rev() {
			tensor.as_mut_view().bit_reverse();
			tensor = Hypercube::One.expand(&[r]).append_to::<P>(tensor);
			tensor.as_mut_view().bit_reverse();
		}

		assert_eq!(tensor, Hypercube::One.expand(&point).build::<P>());
	}

	#[test]
	fn growth_above_the_split_threshold_matches_the_inline_path() {
		let mut rng = StdRng::seed_from_u64(9);

		// Invariant: a round splits across threads only once it exceeds the minimum task size.
		// Below that it runs inline, leaving the parallel path unexercised.
		//
		//     words in the widest round = 2^(n_vars - 1) / scalars per word
		//     pick the smallest n_vars whose widest round reaches the minimum
		//
		// Every other test here is smaller, so this one covers the split.
		let min_len = min_len_for_work(WorkPerItem::FieldMuls);
		let n_vars = (2 * min_len * P::WIDTH).next_power_of_two().ilog2() as usize;
		let point = random_scalars::<F>(&mut rng, n_vars);

		let packed = Hypercube::One.expand(&point).build::<P>();
		let reference = Hypercube::One.expand(&point).build_scalars();
		assert!(packed.iter_scalars().eq(reference.iter().copied()));
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(16))]

		#[test]
		fn the_two_engines_agree(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			// Both bases, both the plain and the scaled form: the packed engine and the scalar
			// engine must land on the same coefficients.
			for cube in CUBES {
				prop_assert_eq!(
					cube.expand(&point).build::<P>().iter_scalars().collect::<Vec<_>>(),
					cube.expand(&point).build_scalars()
				);
				prop_assert_eq!(
					cube.expand(&point).scaled_by(scale).build::<P>()
						.iter_scalars()
						.collect::<Vec<_>>(),
					cube.expand(&point).scaled_by(scale).build_scalars()
				);

				// A scalar field is a packed field of one lane, so the packed engine also runs at
				// a packing width of one. That width is its own path through the growth loop:
				//
				//     4 lanes per word    two rounds live inside one word, then rounds double
				//     1 lane per word     no round fits inside a word, so every round doubles
				//
				// The one-lane store is exactly the scalars, so the two must agree there too.
				prop_assert_eq!(
					cube.expand(&point).build::<F>().into_inner(),
					cube.expand(&point).build_scalars()
				);
				prop_assert_eq!(
					cube.expand(&point).scaled_by(scale).build::<F>().into_inner(),
					cube.expand(&point).scaled_by(scale).build_scalars()
				);
			}
		}

		#[test]
		fn scaling_commutes_with_the_expansion(
			seed in any::<u64>(),
			log_n in 0usize..=8,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let point = random_scalars::<F>(&mut rng, log_n);
			let scale = random_scalars::<F>(&mut rng, 1)[0];

			// Scaling the seed scales every coefficient, for either basis.
			for cube in CUBES {
				let scaled = cube.expand(&point).scaled_by(scale).build::<P>();
				let reference = cube.expand(&point).build::<P>();
				for (got, base) in scaled.iter_scalars().zip(reference.iter_scalars()) {
					prop_assert_eq!(got, scale * base);
				}
			}
		}
	}
}
