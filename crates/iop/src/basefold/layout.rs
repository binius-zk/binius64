// Copyright 2026 The Binius Developers

//! Placement of each round's oracles inside the one oracle the round is committed as.

use std::{cmp::Reverse, ops::Range};

use binius_ip::libra::mask_buffer_dimensions;
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
}

impl RoundLayout {
	/// The spec of the round oracle itself.
	pub const fn spec(&self) -> OracleSpec {
		OracleSpec {
			len: 1 << self.log_len,
			is_zk: self.is_zk,
		}
	}
}

/// Lays out every round of a schedule.
///
/// Constituents are placed in nonincreasing order of `ℓ_i`, each at the lowest multiple of `2^ℓ_i`
/// whose block avoids every extent already placed. An extent is the first `len_j` entries of a
/// placed block, so a constituent may nest in the slack above another's content:
///
/// ```text
/// [ private: 98332 of 2^17 | mask 2^9 | precommit 2^10 | unused ]
///   0                        98816      99328            100352   2^17
/// ```
///
/// The round oracle is the least power of two holding every block. With every `len_i` a full
/// power of two, the blocks lie end to end.
pub fn layout_rounds(schedule: &OracleSchedule) -> Vec<RoundLayout> {
	schedule.rounds().map(layout_round).collect()
}

/// The degree of each univariate of the opening's Libra mask, that of its summand.
pub const LIBRA_DEGREE: usize = 2;

/// The layout of the round the compiler commits the Libra oracle in, if the opening is masked.
///
/// The mask covers the batch sumcheck over `ℓ = max_r ℓ_r` of the schedule's `layouts`, and is
/// only needed when some round is zero-knowledge. Its coefficient table is a round of its own.
pub fn libra_layout(layouts: &[RoundLayout]) -> Option<RoundLayout> {
	if !layouts.iter().any(|layout| layout.is_zk) {
		return None;
	}
	let n_vars = layouts
		.iter()
		.map(|layout| layout.log_len)
		.max()
		.expect("a ZK layout exists");
	let (m_n, m_d) = mask_buffer_dimensions(n_vars, LIBRA_DEGREE, 0);
	Some(RoundLayout {
		log_len: m_n + m_d,
		is_zk: true,
		block_offsets: vec![0],
	})
}

fn layout_round(specs: &[OracleSpec]) -> RoundLayout {
	let mut order = (0..specs.len()).collect::<Vec<_>>();
	order.sort_by_key(|&i| Reverse(specs[i].log_msg_len()));

	// Every placed extent as a half-open scalar range, kept sorted and disjoint.
	let mut extents = Vec::<Range<usize>>::with_capacity(specs.len());
	let mut block_offsets = vec![0; specs.len()];
	let mut end = 1;
	for i in order {
		let log_block = specs[i].log_msg_len();

		// Scan the extents in order for the lowest block that fits before the next one, hopping
		// past each extent that it would overlap.
		let mut offset = 0;
		for extent in &extents {
			if (offset + 1) << log_block <= extent.start {
				break;
			}
			offset = offset.max(extent.end.div_ceil(1 << log_block));
		}

		let start = offset << log_block;
		let extent = start..start + specs[i].len;
		extents.insert(extents.partition_point(|e| e.start < start), extent);
		end = end.max((offset + 1) << log_block);
		block_offsets[i] = offset;
	}

	RoundLayout {
		log_len: log2_ceil_usize(end),
		is_zk: specs.iter().any(|spec| spec.is_zk),
		block_offsets,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn layout(specs: &[OracleSpec]) -> RoundLayout {
		let schedule = OracleSchedule::from_rounds([specs.iter().copied()]);
		let mut layouts = layout_rounds(&schedule);
		assert_eq!(layouts.len(), 1);
		layouts.pop().unwrap()
	}

	#[test]
	fn spartan_shape_fits_one_round_of_its_largest_oracle() {
		// [precommit 2^10, private 2^17 holding 98332, mask 2^9]: the small two nest in the
		// private oracle's slack, so the round stays at 2^17 rather than growing to 2^18.
		let layout = layout(&[
			OracleSpec::new_zk(1 << 10),
			OracleSpec::new_zk(98332),
			OracleSpec::new_zk(1 << 9),
		]);
		assert_eq!(layout.log_len, 17);
		assert!(layout.is_zk);
		assert_eq!(layout.block_offsets, [97, 0, 193]);
	}

	#[test]
	fn a_tenant_nests_in_another_tenants_slack() {
		// The 2^4 oracle holds 9 entries and sits in the 2^5 one's slack at 16. The 2^2 oracle
		// finds no room in the 2^5 one's slack, and nests in the 2^4 one's at 28. It alone is
		// masked, which masks the whole round.
		let layout = layout(&[
			OracleSpec::new(15),
			OracleSpec::new(9),
			OracleSpec::new_zk(1 << 2),
		]);
		assert_eq!(layout.log_len, 5);
		assert!(layout.is_zk);
		assert_eq!(layout.block_offsets, [0, 1, 7]);
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
	fn a_lone_oracle_is_its_own_round() {
		let layout = layout(&[OracleSpec::new_zk(5)]);
		assert_eq!(layout.spec(), OracleSpec::new_zk(1 << 3));
		assert_eq!(layout.block_offsets, [0]);
	}
}
