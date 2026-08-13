// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::word::Word;

/// Why a shift sequence's outer slot must hold the identity in the two-phase reduction.
///
/// The reduction folds one shift axis pair, so it reads the inner slot and ignores the outer one.
/// A term carrying both would be reduced against the wrong shifted word.
pub(crate) const DOUBLE_SHIFT_UNSUPPORTED: &str =
	"the two-phase shift reduction reads only the inner shift of a sequence";

/// The value vector's two committed segments, each at the width the protocol addresses it at.
///
/// A circuit declares fewer values than the reductions address: the public segment is padded to a
/// power of two and the hidden segment to at least that width. Both phases of the shift reduction
/// read the segments at those padded widths, so [`prove()`] fills them once and hands the pair down
/// rather than having each phase re-derive the split.
#[derive(Clone, Copy)]
pub struct SegmentWords<'a> {
	/// The constants and inout values, zero-filled to the public segment width.
	pub public: &'a [Word],
	/// The private values, zero-filled to the hidden segment width.
	pub hidden: &'a [Word],
}

mod key_collection;
// `monster`, `phase_1`, and `phase_2` are internal implementation, exposed (via `#[doc(hidden)]`
// `pub mod`) only so the `shift_reduction` benchmark can time individual phase functions (see
// `benches/shift_reduction.rs`). Not a stable API.
mod claims;
#[doc(hidden)]
pub mod monster;
#[doc(hidden)]
pub mod outer;
#[doc(hidden)]
pub mod phase_1;
#[doc(hidden)]
pub mod phase_2;
mod prove;
mod shift_ind;

pub use claims::{OperatorClaims, PreparedOperatorClaims};
pub use key_collection::{
	DenseShiftEncoding, KeyCollection, KeySegment, Operation, build_key_collection,
};
pub use prove::{OperatorData, PreparedOperatorData, prove};
pub use shift_ind::{Phase3Output, ShiftIndSumcheck};
