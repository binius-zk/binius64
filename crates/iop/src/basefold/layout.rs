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

/// Lays out every round of a schedule, then the Libra round if the opening is masked.
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
/// A zero-knowledge round reserves a ZK pad of `q + 2m + 2` coordinates right after its last
/// extent, `q = n_test_queries` and `m` the Libra round's variable count (binius.xyz#128, "ZK
/// pad"). The round oracle is the least power of two holding every block and the pad. With every
/// `len_i` a full power of two, the blocks lie end to end.
///
/// The Libra round, laid out by `libra_layout`, comes last. Its `m` depends on the largest round,
/// which depends on the pad, which depends on `m`, so the layout is iterated to a fixed point. `m`
/// only grows between iterations and grows logarithmically in the round sizes, so this ends.
pub fn layout_rounds(schedule: &OracleSchedule, n_test_queries: usize) -> Vec<RoundLayout> {
	let mut m = 0;
	loop {
		let pad_len = zk_pad_len(n_test_queries, m);
		let mut layouts = schedule
			.rounds()
			.map(|specs| layout_round(specs, pad_len))
			.collect::<Vec<_>>();
		let Some(libra) = libra_layout(&layouts, n_test_queries) else {
			return layouts;
		};
		if libra.log_len == m {
			layouts.push(libra);
			return layouts;
		}
		m = libra.log_len;
	}
}

/// The degree of each univariate of the opening's Libra mask, that of its summand.
pub const LIBRA_DEGREE: usize = 2;

/// The ZK pad length `q + 2m + 2`: one uniform coordinate per linear functional of a round oracle
/// the opening reveals, being `q` opened codeword symbols, the evaluations `α_r` and `α'_r`, and
/// the `2m` Phase A′ round coefficients.
const fn zk_pad_len(n_test_queries: usize, libra_log_len: usize) -> usize {
	n_test_queries + 2 * libra_log_len + 2
}

/// The layout of the round the compiler commits the Libra oracle in, if the opening is masked.
///
/// The mask covers the batch sumcheck over `ℓ = max_r ℓ_r` of the schedule's `layouts`, and is
/// only needed when some round is zero-knowledge. Its `2^{ν+δ}`-entry coefficient table is a round
/// of its own, followed by the ZK pad. The round's variable count is the least `m ≥ ν + δ` with
/// `2^m ≥ 2^{ν+δ} + q + 2m + 2`.
fn libra_layout(layouts: &[RoundLayout], n_test_queries: usize) -> Option<RoundLayout> {
	if !layouts.iter().any(|layout| layout.is_zk) {
		return None;
	}
	let n_vars = layouts
		.iter()
		.map(|layout| layout.log_len)
		.max()
		.expect("a ZK layout exists");
	let (m_n, m_d) = mask_buffer_dimensions(n_vars, LIBRA_DEGREE, 0);
	let table_len = 1 << (m_n + m_d);
	let log_len = (m_n + m_d..)
		.find(|&m| 1 << m >= table_len + zk_pad_len(n_test_queries, m))
		.expect("2^m outgrows any linear function of m");
	Some(RoundLayout {
		log_len,
		is_zk: true,
		block_offsets: vec![0],
		pad: table_len..table_len + zk_pad_len(n_test_queries, log_len),
	})
}

fn layout_round(specs: &[OracleSpec], zk_pad_len: usize) -> RoundLayout {
	let mut order = (0..specs.len()).collect::<Vec<_>>();
	order.sort_by_key(|&i| Reverse(specs[i].log_msg_len()));

	// Every placed extent, as a half-open scalar range.
	let mut extents = Vec::<(usize, usize)>::with_capacity(specs.len());
	let mut block_offsets = vec![0; specs.len()];
	let mut end = 1;
	for i in order {
		let log_block = specs[i].log_msg_len();
		let avoids_extents = |offset: usize| {
			let (start, end) = (offset << log_block, (offset + 1) << log_block);
			extents.iter().all(|&(lo, hi)| hi <= start || end <= lo)
		};

		// The lowest free block either starts at zero or right after some extent, rounded up to
		// the block size.
		let offset = std::iter::once(0)
			.chain(extents.iter().map(|&(_, hi)| hi.div_ceil(1 << log_block)))
			.filter(|&offset| avoids_extents(offset))
			.min()
			.expect("a block past every extent avoids them all");

		let start = offset << log_block;
		extents.push((start, start + specs[i].len));
		end = end.max((offset + 1) << log_block);
		block_offsets[i] = offset;
	}

	let is_zk = specs.iter().any(|spec| spec.is_zk);
	let pad = if is_zk {
		let start = extents.iter().map(|&(_, hi)| hi).max().unwrap_or(0);
		start..start + zk_pad_len
	} else {
		0..0
	};
	end = end.max(pad.end);

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

	/// The query count of the tests, a realistic one for 128-bit security.
	const N_TEST_QUERIES: usize = 232;

	fn layout(specs: &[OracleSpec]) -> RoundLayout {
		let schedule = OracleSchedule::from_rounds([specs.iter().copied()]);
		layout_rounds(&schedule, N_TEST_QUERIES).swap_remove(0)
	}

	#[test]
	fn spartan_shape_fits_one_round_of_its_largest_oracle() {
		// [precommit 2^10, private 2^17 holding 98332, mask 2^9]: the small two nest in the
		// private oracle's slack, and so does the pad after them, so the round stays at 2^17
		// rather than growing to 2^18.
		let layout = layout(&[
			OracleSpec::new_zk(1 << 10),
			OracleSpec::new_zk(98332),
			OracleSpec::new_zk(1 << 9),
		]);
		assert_eq!(layout.log_len, 17);
		assert!(layout.is_zk);
		assert_eq!(layout.block_offsets, [97, 0, 193]);
		// The precommit block is the last extent, ending at 98 · 2^10.
		assert_eq!(layout.pad.start, 100352);
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
		let layouts = layout_rounds(
			&OracleSchedule::from_rounds([[OracleSpec::new_zk(1 << 10)]]),
			N_TEST_QUERIES,
		);
		let [round, libra] = layouts.as_slice() else {
			panic!("expected the round and the Libra round");
		};
		assert_eq!(round.log_len, 11);
		let m = libra.log_len;
		assert_eq!(round.pad, 1 << 10..(1 << 10) + N_TEST_QUERIES + 2 * m + 2);
	}

	#[test]
	fn the_libra_round_is_the_least_holding_its_table_and_pad() {
		let schedule =
			OracleSchedule::from_rounds([[OracleSpec::new_zk(98332)], [OracleSpec::new(1 << 20)]]);
		let layouts = layout_rounds(&schedule, N_TEST_QUERIES);
		let [.., libra] = layouts.as_slice() else {
			unreachable!()
		};
		// `ℓ = 20`, so the table has `2^{⌈log2 20⌉ + 2} = 2^7` entries and the pad
		// `232 + 2m + 2`, which fits first at `m = 9`.
		assert_eq!(libra.log_len, 9);
		assert_eq!(libra.pad, 128..128 + N_TEST_QUERIES + 2 * 9 + 2);
		assert!(
			layouts
				.iter()
				.all(|layout| layout.pad.end <= 1 << layout.log_len)
		);
	}
}
