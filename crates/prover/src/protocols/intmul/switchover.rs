// Copyright 2026 The Binius Developers

//! The partial evaluations of the 64 bit columns of a word list, as a sumcheck binds them.

use std::array;

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{
	BinaryField, Divisible, PackedField, U1, U2, U4, Underlier,
	linear_transformation::{
		BytewiseLookupTransformationFactory, LinearTransformationFactory,
		OutputWrappingTransformationFactory, Transformation,
	},
};
use binius_math::{
	FieldBuffer, FieldSlice, FieldSliceMut, FieldVec,
	bit_reverse::reverse_bits,
	multilinear::{eq::eq_ind_partial_eval_scalars, fold::fold_highest_var_inplace},
};
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};

use super::transpose_bits::transpose_bits;
use crate::fold_word::BitAxisFolder;

/// The 64 one-bit multilinears of a word list, folded high to low.
///
/// Selector `b` is the multilinear whose `i`-th value is bit `b` of word `i`. The sumcheck binds
/// its highest variable first.
///
/// For the first `h = min(n_vars, Word::LOG_BITS)` rounds the selectors stay transparent: their
/// values are read off the bit-transposed words by byte-table lookups against the tensor of the
/// challenges so far. Round `h` folds every word of [`transpose_bits`] into one buffer, which then
/// folds as usual. Its element `lo * 64 + b` is selector `b` at index `lo`, so binding the highest
/// variable binds it for every selector at once.
///
/// # Lookups
///
/// With `C = 2^(n_vars - h)`, after `r < h` rounds the remaining index is `hi_rest * C + lo`, with
/// `hi_rest` the low `h - r` bits of `hi`. The `2^r` bound rows of `hi_rest` sit at bits
/// `[g * 2^r, (g + 1) * 2^r)` of word `b` of block `lo`, with `g = bitrev_{h-r}(hi_rest)`, and the
/// bit at offset `p` there carries the tensor weight `p`.
pub struct BinarySwitchover<'alloc, P: PackedField, A: Allocator> {
	alloc: &'alloc A,
	n_vars: usize,
	/// The challenges bound so far.
	challenges: Vec<P::Scalar>,
	state: SwitchoverState<A::Vec<Word>, FieldVec<P, A>>,
}

/// The selectors before and after the switchover, owned or borrowed.
enum SwitchoverState<Blocks, Folded> {
	/// One block of `Word::BITS` words per column, from [`transpose_bits`].
	Pre { blocks: Blocks },
	/// The partial evaluations of all 64 selectors, element `lo * 64 + b`.
	Post(Folded),
}

/// A borrowed view of the selectors' current partial evaluations.
///
/// Unlike the [`BinarySwitchover`] it borrows from, it is shared across threads whatever the
/// allocator's buffers are.
pub struct Selectors<'a, P: PackedField> {
	n_vars: usize,
	challenges: &'a [P::Scalar],
	state: SwitchoverState<&'a [Word], FieldSlice<'a, P>>,
}

impl<'alloc, F, P, A> BinarySwitchover<'alloc, P, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	A: Allocator,
{
	/// Builds the selectors of `words`, read as `n_vars`-variate with missing words zero.
	///
	/// ## Preconditions
	///
	/// * `words.len() <= 1 << n_vars`
	pub fn new(alloc: &'alloc A, words: &[Word], n_vars: usize) -> Self {
		let state = if n_vars == 0 {
			// No variable to bind, so the selectors are the bits of the one word.
			let bits = array::from_fn::<_, { Word::BITS }, _>(|b| {
				if words.first().is_some_and(|word| word.extract_bit(b)) {
					F::ONE
				} else {
					F::ZERO
				}
			});
			SwitchoverState::Post(FieldBuffer::from_values_in(alloc, &bits))
		} else {
			SwitchoverState::Pre {
				blocks: transpose_bits(alloc, words, n_vars),
			}
		};
		Self {
			alloc,
			n_vars,
			challenges: Vec::new(),
			state,
		}
	}

	/// Borrows the selectors' current partial evaluations.
	pub fn selectors(&self) -> Selectors<'_, P> {
		let state = match &self.state {
			SwitchoverState::Pre { blocks } => SwitchoverState::Pre {
				blocks: &blocks[..],
			},
			SwitchoverState::Post(folded) => SwitchoverState::Post(folded.as_view()),
		};
		Selectors {
			n_vars: self.n_vars,
			challenges: &self.challenges,
			state,
		}
	}

	/// Binds the highest remaining variable of every selector to `challenge`.
	///
	/// Binding the last of the first `min(n_vars, Word::LOG_BITS)` variables runs the switchover:
	/// every word folds against the tensor of the challenges.
	pub fn fold(&mut self, challenge: F) {
		self.challenges.push(challenge);
		match &mut self.state {
			SwitchoverState::Post(folded) => fold_highest_var_inplace(folded, challenge),
			SwitchoverState::Pre { blocks }
				if self.challenges.len() == self.n_vars.min(Word::LOG_BITS) =>
			{
				let folder = BitAxisFolder::new(&padded_tensor(&self.challenges));
				self.state = SwitchoverState::Post(folder.fold(self.alloc, blocks));
			}
			SwitchoverState::Pre { .. } => {}
		}
	}

	/// The 64 selectors' values at the bound point.
	///
	/// ## Preconditions
	///
	/// * every variable has been folded
	pub fn finish(self) -> [F; Word::BITS] {
		assert_eq!(self.challenges.len(), self.n_vars, "every variable has been folded");
		let SwitchoverState::Post(folded) = self.state else {
			unreachable!("the switchover runs by the last fold");
		};
		array::from_fn(|selector| folded.get(selector))
	}
}

impl<F, P> Selectors<'_, P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	/// Writes the two halves of the `chunk_index`-th aligned chunk of `2^chunk_vars` values of
	/// selector `selector`'s current partial evaluation to `scratch`: its highest variable clear,
	/// then set.
	///
	/// ## Preconditions
	///
	/// * `chunk_vars` is below the remaining variable count
	/// * both scratch buffers hold `2^chunk_vars` values
	pub fn fill_halves(
		&self,
		selector: usize,
		chunk_vars: usize,
		chunk_index: usize,
		scratch: [FieldSliceMut<'_, P>; 2],
	) {
		let half_vars = self.n_vars - self.challenges.len() - 1;
		assert!(chunk_vars <= half_vars);
		assert!(scratch.iter().all(|half| half.log_len() == chunk_vars));

		let start = chunk_index << chunk_vars;
		match &self.state {
			// Round `r` reads `2^r`-bit groups of a word, so each round has its own underlier.
			SwitchoverState::Pre { blocks } => match self.challenges.len() {
				// A one-bit group needs no lookup: its value is the bit itself.
				0 => {
					self.fill_groups::<U1, _>(blocks, selector, half_vars, start, scratch, |bit| {
						if bit.val() == 1 { F::ONE } else { F::ZERO }
					});
				}
				1 => self.fill_lookups::<U2, u8>(blocks, selector, half_vars, start, scratch),
				2 => self.fill_lookups::<U4, u8>(blocks, selector, half_vars, start, scratch),
				3 => self.fill_lookups::<u8, u8>(blocks, selector, half_vars, start, scratch),
				4 => self.fill_lookups::<u16, u16>(blocks, selector, half_vars, start, scratch),
				5 => self.fill_lookups::<u32, u32>(blocks, selector, half_vars, start, scratch),
				_ => unreachable!("the switchover runs once Word::LOG_BITS variables are bound"),
			},
			SwitchoverState::Post(folded) => {
				fill(scratch, start, |index, half| {
					folded.get((half << half_vars | index) << Word::LOG_BITS | selector)
				});
			}
		}
	}

	/// [`Self::fill_groups`] through a lookup of each group against the tensor of the challenges.
	///
	/// `UIn` is the lookup's input, a byte at least, which a sub-byte group widens to.
	fn fill_lookups<UGroup, UIn>(
		&self,
		blocks: &[Word],
		selector: usize,
		half_vars: usize,
		start: usize,
		scratch: [FieldSliceMut<'_, P>; 2],
	) where
		u64: Divisible<UGroup>,
		UIn: Underlier + Divisible<u8> + From<UGroup>,
	{
		let mut weights = eq_ind_partial_eval_scalars(self.challenges);
		weights.resize(UIn::BITS, F::ZERO);
		let lookup = OutputWrappingTransformationFactory::<_, UIn, F>::new(
			BytewiseLookupTransformationFactory,
		)
		.create(&weights);
		self.fill_groups(blocks, selector, half_vars, start, scratch, |group| {
			lookup.transform(&UIn::from(group))
		});
	}

	/// Writes the halves from the groups of `UGroup` bits of the selector's words.
	///
	/// After `r` rounds a word holds `2^(6-r)` groups of `2^r` bits. The two halves are the
	/// adjacent groups `2g'` and `2g' + 1`, where `g'` is the bit-reversed index within the half.
	fn fill_groups<UGroup, Value>(
		&self,
		blocks: &[Word],
		selector: usize,
		half_vars: usize,
		start: usize,
		scratch: [FieldSliceMut<'_, P>; 2],
		value: Value,
	) where
		u64: Divisible<UGroup>,
		Value: Fn(UGroup) -> F + Sync,
	{
		let log_cols = self.n_vars.saturating_sub(Word::LOG_BITS);
		fill(scratch, start, |index, half| {
			let g = reverse_bits(index >> log_cols, (half_vars - log_cols) as u32);
			let word = blocks[(index & ((1 << log_cols) - 1)) << Word::LOG_BITS | selector];
			value(Divisible::<UGroup>::get(&word.0, 2 * g + half))
		});
	}
}

/// Writes `value(start + i, half)` to element `i` of `scratch[half]`.
///
/// A scratch buffer narrower than `P` repeats its values across the spare lanes.
fn fill<P: PackedField>(
	[mut scratch_0, mut scratch_1]: [FieldSliceMut<'_, P>; 2],
	start: usize,
	value: impl Fn(usize, usize) -> P::Scalar + Sync,
) {
	let mask = scratch_0.len() - 1;
	(scratch_0.as_mut(), scratch_1.as_mut())
		.into_par_iter()
		.enumerate()
		.with_min_task(WorkPerItem::FieldMuls)
		.for_each(|(i, (packed_0, packed_1))| {
			let index = |lane| start + ((i * P::WIDTH + lane) & mask);
			*packed_0 = P::from_fn(|lane| value(index(lane), 0));
			*packed_1 = P::from_fn(|lane| value(index(lane), 1));
		});
}

/// The tensor of `challenges` with one weight per bit position, zero past its `2^r` weights.
fn padded_tensor<F: BinaryField>(challenges: &[F]) -> Vec<F> {
	let mut tensor = eq_ind_partial_eval_scalars(challenges);
	tensor.resize(Word::BITS, F::ZERO);
	tensor
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::Field;
	use binius_math::{
		multilinear::evaluate::evaluate,
		test_utils::{B128, Packed128b, random_scalars},
	};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	type P = Packed128b;

	fn n_vars_len_rounds() -> impl Strategy<Value = (usize, usize, usize)> {
		(0..=12usize).prop_flat_map(|n_vars| (Just(n_vars), 0..=1usize << n_vars, 0..=n_vars))
	}

	proptest! {
		#[test]
		fn matches_folded_bit_columns((n_vars, n_words, n_rounds) in n_vars_len_rounds(), seed: u64) {
			let mut rng = StdRng::seed_from_u64(seed);
			let words = (0..n_words).map(|_| Word::from_u64(rng.random())).collect::<Vec<_>>();
			let challenges = random_scalars::<B128>(&mut rng, n_rounds);

			let mut columns = (0..Word::BITS)
				.map(|b| {
					(0..1 << n_vars)
						.map(|i| {
							let bit = words.get(i).is_some_and(|word| word.extract_bit(b));
							if bit { B128::ONE } else { B128::ZERO }
						})
						.collect::<FieldBuffer<P>>()
				})
				.collect::<Vec<_>>();
			let unfolded = columns.clone();

			let mut switchover = BinarySwitchover::<P, _>::new(&GlobalAllocator, &words, n_vars);
			for &challenge in &challenges {
				switchover.fold(challenge);
				for column in &mut columns {
					fold_highest_var_inplace(column, challenge);
				}
			}

			let remaining = n_vars - n_rounds;
			if remaining == 0 {
				let mut point = challenges;
				point.reverse();
				let evals = switchover.finish();
				for (b, column) in unfolded.iter().enumerate() {
					prop_assert_eq!(evals[b], evaluate(column, &point), "b={}", b);
				}
				return Ok(());
			}

			for chunk_vars in 0..remaining {
				let mut scratch = [FieldBuffer::zeros(chunk_vars), FieldBuffer::zeros(chunk_vars)];
				for (b, column) in columns.iter().enumerate() {
					let (half_0, half_1) = column.split_half();
					for chunk_index in 0..1 << (remaining - 1 - chunk_vars) {
						let [scratch_0, scratch_1] = &mut scratch;
						switchover.selectors().fill_halves(
							b,
							chunk_vars,
							chunk_index,
							[scratch_0.as_mut_view(), scratch_1.as_mut_view()],
						);
						prop_assert_eq!(scratch[0].as_view(), half_0.chunk(chunk_vars, chunk_index));
						prop_assert_eq!(scratch[1].as_view(), half_1.chunk(chunk_vars, chunk_index));
					}
				}
			}
		}
	}
}
