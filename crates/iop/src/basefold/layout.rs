// Copyright 2026 The Binius Developers

//! Placement of each round's oracles inside the one oracle the round is committed as.
//!
//! # The ZK pad
//!
//! The query phase opens the codeword `Enc(M_r)` of every round oracle at up to `q =
//! n_test_queries` positions, one per test query. The mask `ω_r` interleaved with a
//! zero-knowledge round hides the folding phase, but not these symbols. Such a round therefore
//! also carries a ZK pad: coordinates outside every constituent's extent, which the prover fills
//! with uniform elements. The pad hides the opened symbols when, for every set of at most `q`
//! positions, the symbols restricted to the pad are linearly independent functionals.
//!
//! Where the pad sits decides whether that holds. The encoder reads entry `i` of an `ℓ`-variate
//! table as the coefficient of the novel basis polynomial `X_{rev(i)}`, where `rev` reverses `ℓ`
//! bits. It evaluates over a subspace `S` that contains 0, at index 0. Every `X_c` with `c ≥ 2^k`
//! vanishes on `U_k = span(β_0, …, β_{k-1})`, so a symbol at a position in `U_k` reads only the
//! entries `rev(c)` with `c < 2^k`. Those sit at stride `2^{ℓ-k}`, starting with entry 0, which is
//! the first entry of the round's largest constituent. No pad outside the extents reaches them.
//!
//! A zero-knowledge round is therefore committed relabeled: entry `i` of its committed table is
//! `M_r[σ(i)]`, where [`zk_commit_index`] gives `σ`, which reverses and complements the bits of
//! `i`. Coefficient `c` of its codeword is then `M_r[2^ℓ - 1 - c]`. The pad is the top `q`
//! coordinates of `M_r`, so it fills the coefficients `0..q`, whose basis polynomials span the
//! polynomials of degree below `q`. Restricted to the pad, the symbols at `q` distinct positions
//! are evaluations of those polynomials at distinct points, and these are linearly independent by
//! interpolation.
//!
//! The opening runs on the committed table and does not see the relabeling, except where the
//! verifier evaluates a transparent: for a zero-knowledge round it does so at
//! [`zk_commit_point`] of the round's point.

use std::{cmp::Reverse, ops::Range};

use binius_field::FieldOps;
use binius_math::bit_reverse::reverse_bits;
use binius_utils::checked_arithmetics::log2_ceil_usize;

use crate::channel::{OracleSchedule, OracleSpec};

/// How one round's oracles sit inside the round oracle `M_r` they are committed as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundLayout {
	/// The base-2 logarithm of the round oracle's length.
	pub log_len: usize,
	/// Whether the round oracle is masked, which it is if any constituent is.
	pub is_zk: bool,
	/// Per constituent, in arrival order, the block it occupies.
	///
	/// Constituent `i` of `2^ℓ_i` entries occupies `M_r[b_i · 2^ℓ_i .. (b_i + 1) · 2^ℓ_i]`, and
	/// only its first `len_i` entries carry content.
	pub block_offsets: Vec<usize>,
	/// The length of the ZK pad, which is the top `pad_len` coordinates of `M_r`.
	///
	/// It is `n_test_queries` for a zero-knowledge round and zero for any other. See the
	/// [module documentation](self) for why the pad sits at the top.
	pub pad_len: usize,
}

impl RoundLayout {
	/// The spec of the round oracle itself.
	pub const fn spec(&self) -> OracleSpec {
		OracleSpec {
			len: 1 << self.log_len,
			is_zk: self.is_zk,
		}
	}

	/// The coordinates of `M_r` that make up the ZK pad.
	pub const fn pad(&self) -> Range<usize> {
		let len = 1 << self.log_len;
		len - self.pad_len..len
	}
}

/// Lays out every round of a schedule.
///
/// Constituents are placed in nonincreasing order of `ℓ_i`, each at the lowest multiple of `2^ℓ_i`
/// whose block avoids every extent already placed. An extent is the first `len_j` entries of a
/// placed block, so a constituent may nest in the slack above another's content:
///
/// ```text
/// [ private: 98332 of 2^17 | mask 2^9 | precommit 2^10 | unused | pad ]
///   0                        98816      99328            100352   2^17 - q   2^17
/// ```
///
/// A zero-knowledge round reserves its top `q = n_test_queries` coordinates as the ZK pad. The
/// round oracle is the least power of two holding every block with the pad above every extent.
/// With every `len_i` a full power of two, the blocks lie end to end, and a zero-knowledge round
/// whose blocks fill a power of two grows to the next one to make room for its pad.
pub fn layout_rounds(schedule: &OracleSchedule, n_test_queries: usize) -> Vec<RoundLayout> {
	schedule
		.rounds()
		.map(|specs| layout_round(specs, n_test_queries))
		.collect()
}

/// The coordinate of `M_r` that entry `index` of a zero-knowledge round's committed table holds.
///
/// The map reverses the low `log_len` bits of `index` and complements them. It is its own inverse,
/// so it also gives the entry that a coordinate is committed at. See the
/// [module documentation](self) for why a zero-knowledge round is committed relabeled.
pub const fn zk_commit_index(index: usize, log_len: usize) -> usize {
	reverse_bits(index, log_len as u32) ^ ((1 << log_len) - 1)
}

/// The point over `M_r` that corresponds to `point` over a zero-knowledge round's committed table.
///
/// For any table `T` over the round, let `T'` be `T` relabeled the way the round is committed,
/// `T'[i] = T[zk_commit_index(i)]`. The multilinear extension of `T'` at `point` equals that of
/// `T` at `zk_commit_point(point)`, which is `(1 - point[ℓ-1], …, 1 - point[0])`: reversing the
/// bits of an index reverses the coordinates, and complementing a bit complements its coordinate.
pub fn zk_commit_point<E: FieldOps>(point: &[E]) -> Vec<E> {
	point
		.iter()
		.rev()
		.map(|coordinate| E::one() - coordinate)
		.collect()
}

fn layout_round(specs: &[OracleSpec], n_test_queries: usize) -> RoundLayout {
	let mut order = (0..specs.len()).collect::<Vec<_>>();
	order.sort_by_key(|&i| Reverse(specs[i].len));

	// Every placed extent as a half-open scalar range, kept sorted and disjoint.
	let mut extents = Vec::<Range<usize>>::with_capacity(specs.len());
	let mut block_offsets = vec![0; specs.len()];
	let mut end = 1;
	for i in order {
		let log_block = specs[i].log_msg_len();

		// Scan the extents in order for the lowest block that fits before the next one, hopping
		// past each extent that it would overlap.
		// Because the extents are sorted, the offset only moves forward, and the break index is
		// also where the new extent is inserted.
		let mut offset = 0;
		let mut pos = extents.len();
		for (j, extent) in extents.iter().enumerate() {
			if (offset + 1) << log_block <= extent.start {
				pos = j;
				break;
			}
			offset = extent.end.div_ceil(1 << log_block);
		}

		let start = offset << log_block;
		let extent = start..start + specs[i].len;
		extents.insert(pos, extent);
		end = end.max((offset + 1) << log_block);
		block_offsets[i] = offset;
	}

	// The pad is the top `pad_len` coordinates, so the round must reach `pad_len` past the highest
	// extent, which is the last one because the extents are sorted and disjoint.
	let is_zk = specs.iter().any(|spec| spec.is_zk);
	let pad_len = if is_zk { n_test_queries } else { 0 };
	let content_end = extents.last().map_or(0, |extent| extent.end);
	end = end.max(content_end + pad_len);

	RoundLayout {
		log_len: log2_ceil_usize(end),
		is_zk,
		block_offsets,
		pad_len,
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use super::*;

	/// A query count small enough to keep the test shapes readable.
	const N_TEST_QUERIES: usize = 32;

	fn layout(specs: &[OracleSpec]) -> RoundLayout {
		let schedule = OracleSchedule::from_rounds([specs.iter().copied()]);
		let mut layouts = layout_rounds(&schedule, N_TEST_QUERIES);
		assert_eq!(layouts.len(), 1);
		layouts.pop().unwrap()
	}

	#[test]
	fn spartan_shape_fits_one_round_of_its_largest_oracle() {
		// [precommit 2^10, private 2^17 holding 98332, mask 2^9]: the small two nest in the
		// private oracle's slack and the pad fits above them, so the round stays at 2^17 rather
		// than growing to 2^18.
		let layout = layout(&[
			OracleSpec::new_zk(1 << 10),
			OracleSpec::new_zk(98332),
			OracleSpec::new_zk(1 << 9),
		]);
		assert_eq!(layout.log_len, 17);
		assert!(layout.is_zk);
		assert_eq!(layout.block_offsets, [97, 0, 193]);
		assert_eq!(layout.pad(), (1 << 17) - N_TEST_QUERIES..1 << 17);
	}

	#[test]
	fn a_tenant_nests_in_another_tenants_slack() {
		// The 2^4 oracle holds 9 entries and sits in the 2^5 one's slack at 16. The 2^2 oracle
		// finds no room in the 2^5 one's slack, and nests in the 2^4 one's at 28. It alone is
		// masked, which masks the whole round. Its extent ends at the round's top, so the pad
		// grows the round to 2^6.
		let layout = layout(&[
			OracleSpec::new(15),
			OracleSpec::new(9),
			OracleSpec::new_zk(1 << 2),
		]);
		assert_eq!(layout.log_len, 6);
		assert!(layout.is_zk);
		assert_eq!(layout.block_offsets, [0, 1, 7]);
		assert_eq!(layout.pad(), 32..64);
	}

	#[test]
	fn full_oracles_lie_end_to_end_and_grow_the_round() {
		// 2^2 + 2^4 + 2^2 = 24, so the round grows to 2^5 with the largest first.
		let layout = layout(&[
			OracleSpec::new(1 << 2),
			OracleSpec::new(1 << 4),
			OracleSpec::new(1 << 2),
		]);
		assert_eq!(layout.log_len, 5);
		assert!(!layout.is_zk);
		assert_eq!(layout.block_offsets, [4, 0, 5]);
	}

	#[test]
	fn oracles_sharing_a_log_length_are_placed_longest_first() {
		// Both oracles have log length 4, but the 16-entry one is longer, so it is placed first
		// despite coming second.
		let layout = layout(&[OracleSpec::new(9), OracleSpec::new(16)]);
		assert_eq!(layout.log_len, 5);
		assert_eq!(layout.block_offsets, [1, 0]);
	}

	#[test]
	fn a_lone_oracle_is_its_own_round() {
		let layout = layout(&[OracleSpec::new(5)]);
		assert_eq!(layout.spec(), OracleSpec::new(1 << 3));
		assert_eq!(layout.block_offsets, [0]);
	}

	#[test]
	fn a_non_zk_round_carries_no_pad() {
		let layout = layout(&[OracleSpec::new(1 << 10)]);
		assert_eq!(layout.log_len, 10);
		assert_eq!(layout.pad_len, 0);
		assert!(layout.pad().is_empty());
	}

	#[test]
	fn a_zk_round_with_slack_at_the_top_keeps_its_size() {
		let layout = layout(&[OracleSpec::new_zk(900)]);
		assert_eq!(layout.log_len, 10);
		assert_eq!(layout.pad(), 1024 - N_TEST_QUERIES..1024);
	}

	#[test]
	fn a_zk_round_without_slack_at_the_top_grows_for_its_pad() {
		// The content ends 16 short of the top, which is not room enough for 32 pad coordinates.
		let layout = layout(&[OracleSpec::new_zk(1024 - 16)]);
		assert_eq!(layout.log_len, 11);
		assert_eq!(layout.block_offsets, [0]);
		assert_eq!(layout.pad(), 2048 - N_TEST_QUERIES..2048);
	}

	#[test]
	fn the_pad_lies_above_every_extent() {
		let shapes: [&[OracleSpec]; 4] = [
			&[OracleSpec::new_zk(1 << 4)],
			&[OracleSpec::new_zk(1 << 5), OracleSpec::new(1 << 5)],
			&[
				OracleSpec::new_zk(40),
				OracleSpec::new(7),
				OracleSpec::new(1 << 3),
			],
			&[
				OracleSpec::new(100),
				OracleSpec::new_zk(1 << 6),
				OracleSpec::new(3),
			],
		];
		for specs in shapes {
			let layout = layout(specs);
			assert_eq!(layout.pad_len, N_TEST_QUERIES);
			for (spec, &offset) in iter::zip(specs, &layout.block_offsets) {
				let start = offset << spec.log_msg_len();
				assert!(start + spec.len <= layout.pad().start, "{specs:?}");
			}
		}
	}

	#[test]
	fn zk_commit_index_reverses_and_complements_the_bits() {
		assert_eq!(zk_commit_index(0b001, 3), 0b011);
		assert_eq!(zk_commit_index(0b110, 3), 0b100);
		assert_eq!(zk_commit_index(0, 0), 0);
		for log_len in 0..8 {
			for index in 0..1 << log_len {
				assert_eq!(zk_commit_index(zk_commit_index(index, log_len), log_len), index);
			}
		}
	}

	#[test]
	fn the_relabeling_puts_coordinate_c_from_the_top_on_coefficient_c() {
		// The encoder reads entry `i` of the committed table as the coefficient `rev(i)`.
		for log_len in 0..8 {
			for coordinate in 0..1 << log_len {
				let entry = zk_commit_index(coordinate, log_len);
				let coefficient = reverse_bits(entry, log_len as u32);
				assert_eq!(coefficient, (1 << log_len) - 1 - coordinate);
			}
		}
	}
}
