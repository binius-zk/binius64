// Copyright 2026 The Binius Developers

//! Placement of each round's oracles inside the one oracle the round is committed as.

use std::{cmp::Reverse, ops::Range};

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
	/// The ZK pad: coordinates outside every extent that the prover fills with uniform elements.
	/// Empty for a round that is not zero-knowledge.
	pub pad: Range<usize>,
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
/// [ private: 98332 of 2^17 | mask 2^9 | precommit 2^10 | pad | unused ]
///   0                        98816      99328            100352      2^17
/// ```
///
/// A zero-knowledge round also reserves a ZK pad of `pad_len` coordinates (binius.xyz#128, "ZK
/// pad"), placed like one more constituent of `len = pad_len` in the sort and the scan. The caller
/// passes `pad_len = q`, the codeword symbols the query phase opens. The round oracle is the least
/// power of two holding every block and the pad. With every `len_i` a full power of two,
/// the blocks lie end to end.
pub fn layout_rounds(schedule: &OracleSchedule, pad_len: usize) -> Vec<RoundLayout> {
	schedule
		.rounds()
		.map(|specs| layout_round(specs, pad_len))
		.collect()
}

fn layout_round(specs: &[OracleSpec], pad_len: usize) -> RoundLayout {
	let is_zk = specs.iter().any(|spec| spec.is_zk);

	// The constituents as `(log block size, len)`, the ZK pad last if there is one.
	let mut items = specs
		.iter()
		.map(|spec| (spec.log_msg_len(), spec.len))
		.collect::<Vec<_>>();
	if is_zk {
		items.push((log2_ceil_usize(pad_len), pad_len));
	}

	let mut order = (0..items.len()).collect::<Vec<_>>();
	order.sort_by_key(|&i| Reverse(items[i].0));

	// Every placed extent as a half-open scalar range, kept sorted and disjoint.
	let mut extents = Vec::<Range<usize>>::with_capacity(items.len());
	let mut block_offsets = vec![0; items.len()];
	let mut pad = 0..0;
	let mut end = 1;
	for i in order {
		let (log_block, len) = items[i];

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
		let extent = start..start + len;
		if is_zk && i == specs.len() {
			pad = extent.clone();
		}
		extents.insert(pos, extent);
		end = end.max((offset + 1) << log_block);
		block_offsets[i] = offset;
	}
	block_offsets.truncate(specs.len());

	RoundLayout {
		log_len: log2_ceil_usize(end),
		is_zk,
		block_offsets,
		pad,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The pad length of the tests, `q = n_test_queries · 2^η`.
	const PAD_LEN: usize = 232 * 4;

	fn layout(specs: &[OracleSpec]) -> RoundLayout {
		let schedule = OracleSchedule::from_rounds([specs.iter().copied()]);
		let mut layouts = layout_rounds(&schedule, PAD_LEN);
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
		// The precommit block is the last extent, ending at 98 · 2^10.
		assert_eq!(layout.pad, 100352..100352 + PAD_LEN);
		assert!(layout.pad.end <= 1 << 17);
	}

	#[test]
	fn a_tenant_nests_in_another_tenants_slack() {
		// The 2^4 oracle holds 9 entries and sits in the 2^5 one's slack at 16. The 2^2 oracle
		// finds no room in the 2^5 one's slack, and nests in the 2^4 one's at 28.
		let layout = layout(&[
			OracleSpec::new(15),
			OracleSpec::new(9),
			OracleSpec::new(1 << 2),
		]);
		assert_eq!(layout.log_len, 5);
		assert!(!layout.is_zk);
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
		let layout = layout(&[OracleSpec::new(5)]);
		assert_eq!(layout.spec(), OracleSpec::new(1 << 3));
		assert_eq!(layout.block_offsets, [0]);
	}

	#[test]
	fn a_non_zk_round_carries_no_pad() {
		let layout = layout(&[OracleSpec::new(1 << 10)]);
		assert_eq!(layout.log_len, 10);
		assert!(layout.pad.is_empty());
	}

	#[test]
	fn a_zk_round_with_no_slack_grows_by_one_for_its_pad() {
		let layout = layout(&[OracleSpec::new_zk(1 << 10)]);
		assert_eq!(layout.log_len, 11);
		assert_eq!(layout.pad, 1 << 10..(1 << 10) + PAD_LEN);
	}
}
