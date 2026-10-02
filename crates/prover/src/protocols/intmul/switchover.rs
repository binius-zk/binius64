// Copyright 2026 The Binius Developers

//! The partial evaluations of the 64 bit columns of a word list, as a sumcheck binds them.

use std::{array, iter};

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_math::{
	FieldBuffer, FieldSlice,
	bit_reverse::reverse_bits,
	multilinear::{eq::eq_ind_partial_eval_scalars, fold::fold_highest_var_inplace},
};
use binius_utils::rayon::{prelude::*, task_size::min_len_for_bytes};

use super::transpose_bits::transpose_bits;
use crate::fold_word::{BitAxisFolder, BitWeightTables};

/// The 64 one-bit multilinears of a word list, folded high to low.
///
/// Selector `b` is the multilinear whose `i`-th value is bit `b` of word `i`. The sumcheck binds
/// its highest variable first.
///
/// For the first `h = min(n_vars, Word::LOG_BITS)` rounds the selectors stay transparent: their
/// values are read off the bit-transposed words by byte-table lookups against the tensor of the
/// challenges so far. After round `h` they are folded into 64 buffers of `2^(n_vars - h)`
/// elements each, which then fold as usual.
///
/// # Layout
///
/// With `C = 2^(n_vars - h)`, the word list is a `2^h × C` matrix whose row `hi` holds the top `h`
/// index bits. Row `b` of [`transpose_bits`] is `C` contiguous words, and bit `bitrev_h(hi)` of
/// its word `lo` is bit `b` of word `hi * C + lo`.
///
/// After `r < h` rounds, the remaining index is `hi_rest * C + lo`, with `hi_rest` the low `h - r`
/// bits of `hi`. The `2^r` bound rows of `hi_rest` sit at bits `[g * 2^r, (g + 1) * 2^r)` with
/// `g = bitrev_{h-r}(hi_rest)`, and the bit at offset `p` there carries the tensor weight `p`.
/// The selector's value is that span's inner product with the tensor.
#[cfg_attr(
	not(test),
	expect(
		dead_code,
		reason = "the selector mlecheck switches over to it in BINIUS-679"
	)
)]
pub struct BinarySwitchover<'a, P: PackedField, A: Allocator> {
	alloc: &'a A,
	/// `Word::BITS` rows of `2^log_cols` words each, from [`transpose_bits`].
	rows: A::Vec<Word>,
	/// `h`, the number of rounds before the switchover.
	log_rows: usize,
	log_cols: usize,
	/// The challenges bound so far.
	challenges: Vec<P::Scalar>,
	/// This round's tables over the tensor of `challenges`, zero past its `2^r` weights.
	tables: BitWeightTables<P::Scalar>,
	/// The partial evaluations, once the switchover has run.
	folded: Option<Vec<FieldBuffer<P, A::Vec<P>>>>,
}

#[cfg_attr(
	not(test),
	expect(
		dead_code,
		reason = "the selector mlecheck switches over to it in BINIUS-679"
	)
)]
impl<'a, F, P, A> BinarySwitchover<'a, P, A>
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
	pub fn new(alloc: &'a A, words: &[Word], n_vars: usize) -> Self {
		let log_rows = n_vars.min(Word::LOG_BITS);
		let mut switchover = Self {
			alloc,
			rows: transpose_bits(alloc, words, n_vars),
			log_rows,
			log_cols: n_vars - log_rows,
			challenges: Vec::new(),
			tables: BitWeightTables::new(&padded_tensor(&[])),
			folded: None,
		};
		if log_rows == 0 {
			switchover.perform();
		}
		switchover
	}

	/// The two halves of the `chunk_index`-th aligned chunk of `2^chunk_vars` values of selector
	/// `selector`'s current partial evaluation: its highest variable clear, then set.
	///
	/// Before the switchover the halves are written to `scratch`; after it they are borrowed.
	///
	/// ## Preconditions
	///
	/// * at least one variable remains, and `chunk_vars` is below the remaining count
	/// * both scratch buffers hold `2^chunk_vars` values
	pub fn fill_halves<'s>(
		&'s self,
		selector: usize,
		chunk_vars: usize,
		chunk_index: usize,
		scratch: &'s mut [FieldBuffer<P>; 2],
	) -> [FieldSlice<'s, P>; 2] {
		if let Some(folded) = &self.folded {
			let folded = &folded[selector];
			assert!(chunk_vars < folded.log_len());
			let half_chunks = 1 << (folded.log_len() - 1 - chunk_vars);
			return [
				folded.chunk(chunk_vars, chunk_index),
				folded.chunk(chunk_vars, chunk_index + half_chunks),
			];
		}

		let r = self.challenges.len();
		assert!(chunk_vars < self.log_rows + self.log_cols - r);
		let [scratch_0, scratch_1] = scratch;
		assert_eq!(scratch_0.log_len(), chunk_vars);
		assert_eq!(scratch_1.log_len(), chunk_vars);

		let row = &self.rows[selector << self.log_cols..][..1 << self.log_cols];
		let half_log_rows = (self.log_rows - r - 1) as u32;
		let cols_mask = (1 << self.log_cols) - 1;
		let span_mask = u64::MAX >> (Word::BITS - (1 << r));
		let n_bytes = (1usize << r).div_ceil(8);

		// The two halves are the adjacent spans `2g'` and `2g' + 1` of one word, where `g'` is
		// the bit-reversed index within the half.
		let value = |index: usize, half: usize| {
			let g = reverse_bits(index >> self.log_cols, half_log_rows);
			let span = (row[index & cols_mask].0 >> ((2 * g + half) << r)) & span_mask;
			self.tables.fold_low_bytes(span, n_bytes)
		};

		let start = chunk_index << chunk_vars;
		let len = 1 << chunk_vars;
		for (k, (packed_0, packed_1)) in
			iter::zip(scratch_0.iter_packed_mut(), scratch_1.iter_packed_mut()).enumerate()
		{
			let offset = k * P::WIDTH;
			let value_or_zero = |i: usize, half| {
				if offset + i < len {
					value(start + offset + i, half)
				} else {
					F::ZERO
				}
			};
			*packed_0 = P::from_fn(|i| value_or_zero(i, 0));
			*packed_1 = P::from_fn(|i| value_or_zero(i, 1));
		}

		[scratch_0.as_view(), scratch_1.as_view()]
	}

	/// Binds the highest remaining variable of every selector to `challenge`.
	pub fn fold(&mut self, challenge: F) {
		if let Some(folded) = &mut self.folded {
			// One item is a whole selector, so the floor converts from elements to selectors.
			let min_selectors = min_len_for_bytes::<F>().div_ceil(folded[0].len());
			folded
				.par_iter_mut()
				.with_min_len(min_selectors)
				.for_each(|selector| fold_highest_var_inplace(selector, challenge));
			return;
		}

		self.challenges.push(challenge);
		if self.challenges.len() == self.log_rows {
			self.perform();
		} else {
			self.tables = BitWeightTables::new(&padded_tensor(&self.challenges));
		}
	}

	/// The 64 selectors' values at the bound point.
	///
	/// ## Preconditions
	///
	/// * every variable has been folded
	pub fn finish(self) -> [F; Word::BITS] {
		let folded = self.folded.expect("every variable has been folded");
		array::from_fn(|selector| {
			assert_eq!(folded[selector].log_len(), 0, "every variable has been folded");
			folded[selector].get(0)
		})
	}

	/// Folds every row's whole word against the tensor, one buffer of `C` elements per selector.
	fn perform(&mut self) {
		let folder = BitAxisFolder::new(&padded_tensor(&self.challenges));
		let folded = self
			.rows
			.par_chunks(1 << self.log_cols)
			// One item is a whole row, so the floor converts from words to rows.
			.with_min_len(min_len_for_bytes::<Word>().div_ceil(1 << self.log_cols))
			.map(|row| folder.fold(self.alloc, row))
			.collect();
		self.folded = Some(folded);
	}
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
		(0..=9usize).prop_flat_map(|n_vars| (Just(n_vars), 0..=1usize << n_vars, 0..=n_vars))
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
						let [chunk_0, chunk_1] =
							switchover.fill_halves(b, chunk_vars, chunk_index, &mut scratch);
						prop_assert_eq!(chunk_0, half_0.chunk(chunk_vars, chunk_index));
						prop_assert_eq!(chunk_1, half_1.chunk(chunk_vars, chunk_index));
					}
				}
			}
		}
	}
}
