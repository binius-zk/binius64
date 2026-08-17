// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::word::Word;

/// The value vector's two committed segments, each at the width the protocol addresses it at.
///
/// A circuit declares fewer values than the reductions address: the public segment is padded to a
/// power of two and the hidden segment to at least that width. Both phases of the shift reduction
/// read the segments at those padded widths, so [`prove()`](super::prove::prove) fills them once
/// and hands the pair down rather than having each phase re-derive the split.
#[derive(Clone, Copy)]
pub struct SegmentWords<'a> {
	/// The constants and inout values, zero-filled to the public segment width.
	pub public: &'a [Word],
	/// The private values, zero-filled to the hidden segment width.
	pub hidden: &'a [Word],
}
