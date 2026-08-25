// Copyright 2026 The Binius Developers

//! The weight a level's opened rows induce, and the two routes that build it.

use std::iter::zip;

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_iop::ligerito::{InducedBasis, LigeritoLevel};
use binius_math::{
	FieldBuffer, FieldVec, ReedSolomonCode,
	ntt::{AdditiveNTT, domain_context::GaoMateerOnTheFly},
};

/// The weight one level's opened rows induce on the message that level folded to.
///
/// The rows make `t` linear claims, batched by the powers of one coefficient:
///
/// ```text
///     <w, m> = sum_i alpha^i * encode(m)[q_i]
/// ```
///
/// Two routes produce the same `w`, and which is cheaper turns on the level's shape alone.
/// Holding the inputs together is what lets the choice be a method rather than a parameter.
pub(super) struct InducedWeight<'a, F, NTT> {
	/// The level whose rows were opened, whose shape picks the route.
	level: &'a LigeritoLevel,
	/// The transform the adjoint route runs its layers over.
	ntt: &'a NTT,
	/// The codeword positions the queries landed on, in the order they were drawn.
	indices: Vec<usize>,
	/// The row-batching coefficient whose powers scale the rows.
	alpha: F,
}

impl<'a, F, NTT> InducedWeight<'a, F, NTT>
where
	F: BinaryField,
	NTT: AdditiveNTT<Field = F> + Sync,
{
	/// Collects what a level's query round produced into one weight to be built.
	///
	/// ## Preconditions
	///
	/// * `ntt`'s domain covers the level's codeword domain.
	pub(super) fn new(level: &'a LigeritoLevel, ntt: &'a NTT, indices: &[Word], alpha: F) -> Self {
		Self {
			level,
			ntt,
			indices: indices
				.iter()
				.map(|index| index.as_u64() as usize)
				.collect(),
			alpha,
		}
	}

	/// Whether the adjoint route beats the row-by-row one at this shape.
	///
	/// Both routes produce `2^cols` entries, so dividing their costs by `2^cols` leaves
	///
	/// ```text
	///     row by row   t row entries
	///     adjoint      cols * 2^rate layer entries
	/// ```
	///
	/// where `t` is the number of opened rows.
	/// Neither cost depends on the level's lane count, which is why that field is not read.
	///
	/// A row entry and a layer entry are not the same price, so the comparison needs a constant.
	/// Timing both routes over 128-bit binary field elements puts a row entry at 2.9 ns.
	/// It puts a layer entry at 1.6 ns, steady to a few percent from `2^9` up to `2^24` entries.
	/// The ratio of the two, 1.9, is what the integers below encode.
	pub(super) const fn adjoint_is_cheaper(&self) -> bool {
		// The two costs above, each scaled by 10 so the measured 1.9 is an integer ratio.
		let row_cost = 19 * self.indices.len();
		let adjoint_cost = 10 * (self.level.log_msg_cols << self.level.log_inv_rate);
		adjoint_cost < row_cost
	}

	/// Builds the weight by whichever route is cheaper at this shape.
	pub(super) fn build<P, A>(&self, alloc: &A) -> FieldVec<P, A>
	where
		P: PackedField<Scalar = F>,
		A: Allocator,
	{
		if self.adjoint_is_cheaper() {
			self.by_adjoint(alloc)
		} else {
			self.by_rows(alloc)
		}
	}

	/// Builds the weight by expanding one opened row at a time.
	///
	/// A row is a tensor of `log_msg_cols` factors, so expanding it costs `2^log_msg_cols`
	/// products. Accumulating it into the running vector costs as many multiplications again.
	/// For `t` opened rows the whole build is therefore `2 * t * 2^log_msg_cols` multiplications.
	///
	/// This is the reference the adjoint route is tested against.
	pub(super) fn by_rows<P, A>(&self, alloc: &A) -> FieldVec<P, A>
	where
		P: PackedField<Scalar = F>,
		A: Allocator,
	{
		let domain_context = GaoMateerOnTheFly::generate(self.level.log_codeword_len());
		let dense =
			InducedBasis::new(&domain_context, self.level.log_msg_cols, &self.indices, self.alpha)
				.to_dense();
		FieldBuffer::from_values_in(alloc, &dense)
	}

	/// Builds the weight as one pass of the encoder's adjoint.
	///
	/// The weight is defined by what it does to a message:
	///
	/// ```text
	///     <w, m> = sum_i alpha^i * encode(m)[q_i] = <a, encode(m)>,   a[q_i] = alpha^i
	/// ```
	///
	/// The right-hand side is the encoder's adjoint identity read backwards.
	/// So `w` is that adjoint applied to the sparse weight `a`.
	///
	/// The cost is `log_msg_cols` butterfly layers over `2^(log_msg_cols + log_inv_rate)` entries.
	/// Nothing in it grows with the number of rows opened.
	pub(super) fn by_adjoint<P, A>(&self, alloc: &A) -> FieldVec<P, A>
	where
		P: PackedField<Scalar = F>,
		A: Allocator,
	{
		// A position can be drawn twice, so the powers landing on it accumulate rather than
		// overwrite.
		let mut weights = FieldBuffer::zeros_in(alloc, self.level.log_codeword_len());
		for (&index, power) in zip(&self.indices, self.alpha.powers()) {
			weights.set(index, weights.get(index) + power);
		}

		let code = ReedSolomonCode::<F>::new(self.level.log_msg_cols, self.level.log_inv_rate);
		code.encode_batch_transpose(self.ntt, weights.as_mut_view(), 0, alloc)
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, Ghash128b as B128, Random};
	use binius_math::{
		FieldBuffer,
		ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
	};
	use proptest::prelude::*;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;

	/// Both routes of the induced weight, over one level shape and one set of query indices.
	fn both_builds(
		log_msg_cols: usize,
		log_inv_rate: usize,
		indices: &[usize],
		alpha: B128,
	) -> (FieldBuffer<B128>, FieldBuffer<B128>) {
		// The lane count does not enter either build, so any value serves here.
		let level = LigeritoLevel {
			log_msg_cols,
			log_lanes: 1,
			log_inv_rate,
			n_queries: indices.len(),
		};
		let ntt = NeighborsLastSingleThread::new(GaoMateerOnTheFly::<B128>::generate(
			level.log_codeword_len(),
		));
		let words = indices
			.iter()
			.map(|&index| Word::from_u64(index as u64))
			.collect::<Vec<_>>();
		let weight = InducedWeight::new(&level, &ntt, &words, alpha);
		(weight.by_rows(&GlobalAllocator), weight.by_adjoint(&GlobalAllocator))
	}

	/// The two builds must agree entry for entry, not merely at one evaluation point.
	///
	/// The transposed build reaches the weight through the encoder's adjoint.
	/// The row build reaches it through the tensor expansion of each generator row.
	/// Nothing but a test connects the two derivations.
	#[test]
	fn the_transposed_build_matches_the_row_build_at_the_boundaries() {
		let alpha = B128::new(0x9e3779b97f4a7c15);

		// A message of one column, which leaves the transform no layer to run at all.
		let (rows, transposed) = both_builds(0, 3, &[0, 5, 5], alpha);
		assert_eq!(rows, transposed);

		// No rows opened, which induces the all-zero weight rather than an empty one.
		let (rows, transposed) = both_builds(4, 2, &[], alpha);
		assert_eq!(rows, transposed, "no rows");
		assert_eq!(rows, FieldBuffer::zeros(4));

		// Every rate the ladder search may pick, at one row and at a row drawn three times.
		// Sampling is with replacement.
		// So a repeated index must add its powers rather than overwrite them.
		for log_inv_rate in 1..=8 {
			let (rows, transposed) = both_builds(3, log_inv_rate, &[6], alpha);
			assert_eq!(rows, transposed, "one row at rate {log_inv_rate}");

			let (rows, transposed) = both_builds(3, log_inv_rate, &[2, 2, 2], alpha);
			assert_eq!(rows, transposed, "a repeated row at rate {log_inv_rate}");
		}
	}

	/// The selection rule, at every level shape the ladder search picks at 96-bit security.
	///
	/// The last column is the row build's measured time over the transposed build's.
	/// A value above one is therefore a shape where the transposed build won.
	/// The rule must agree with that comparison at every shape.
	#[test]
	fn the_selection_rule_follows_the_measured_crossover() {
		// (log_msg_cols, log_inv_rate, n_queries, measured speedup of the transposed build)
		let measured: &[(usize, usize, usize, f64)] = &[
			(9, 1, 232, 26.43),
			(9, 3, 116, 3.86),
			(9, 5, 101, 0.88),
			(9, 6, 99, 0.41),
			(10, 3, 116, 3.06),
			(10, 4, 106, 1.39),
			(10, 5, 101, 0.66),
			(10, 6, 99, 0.32),
			(10, 8, 97, 0.08),
			(11, 4, 106, 1.17),
			(11, 6, 99, 0.26),
			(12, 4, 106, 1.12),
			(12, 5, 101, 0.52),
			(13, 2, 142, 5.15),
			(13, 4, 106, 0.94),
			(13, 5, 101, 0.43),
			(13, 6, 99, 0.20),
			(14, 1, 232, 15.21),
			(14, 7, 98, 0.09),
			(15, 4, 106, 0.78),
			(15, 5, 101, 0.37),
			(16, 2, 142, 4.12),
			(16, 4, 106, 0.74),
			(17, 1, 232, 11.75),
			(17, 4, 106, 0.74),
			(17, 5, 101, 0.34),
			(18, 6, 99, 0.16),
			(19, 4, 106, 0.75),
			(19, 5, 101, 0.34),
			(20, 2, 142, 3.46),
			(21, 1, 232, 10.95),
			(21, 4, 106, 0.62),
			(22, 1, 232, 10.36),
			(23, 3, 116, 1.26),
			(24, 1, 232, 9.60),
			(24, 2, 142, 2.96),
		];
		for &(log_msg_cols, log_inv_rate, n_queries, speedup) in measured {
			let level = LigeritoLevel {
				log_msg_cols,
				log_lanes: 1,
				log_inv_rate,
				n_queries,
			};
			// The rule reads the level's shape and the row count, so any index and coefficient
			// serve; only how many there are matters.
			let ntt = NeighborsLastSingleThread::new(GaoMateerOnTheFly::<B128>::generate(
				level.log_codeword_len(),
			));
			let words = vec![Word::ZERO; n_queries];
			assert_eq!(
				InducedWeight::new(&level, &ntt, &words, B128::ONE).adjoint_is_cheaper(),
				speedup > 1.0,
				"cols={log_msg_cols} rate={log_inv_rate} t={n_queries} speedup={speedup}"
			);
		}
	}

	proptest! {
		/// The two builds must agree over the whole shape space, not the shapes a fixed test picks.
		///
		/// Indices are drawn with replacement, exactly as the channel draws them.
		/// A run can therefore land on the same row twice.
		#[test]
		fn the_transposed_build_matches_the_row_build(
			seed: u64,
			log_msg_cols in 0usize..7,
			log_inv_rate in 1usize..5,
			n_queries in 0usize..24,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let log_codeword_len = log_msg_cols + log_inv_rate;
			let indices = (0..n_queries)
				.map(|_| rng.random_range(0..1usize << log_codeword_len))
				.collect::<Vec<_>>();
			let alpha = B128::random(&mut rng);

			let (rows, transposed) = both_builds(log_msg_cols, log_inv_rate, &indices, alpha);
			prop_assert_eq!(rows, transposed);
		}
	}
}
