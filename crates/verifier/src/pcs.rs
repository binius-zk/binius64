// Copyright 2026 The Binius Developers

//! Which polynomial commitment scheme opens the committed trace.

/// The polynomial commitment scheme the trace oracle is committed and opened with.
///
/// Both schemes commit the same trace and discharge the same evaluation claim.
/// The reduction that produces that claim is identical either way.
/// So a proof differs only in the bytes the opening writes.
///
/// A proof is readable only by the scheme that wrote it.
/// The two lay the transcript out differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pcs {
	/// A sumcheck interleaved with one FRI over a codeword committed at a single rate.
	///
	/// The incumbent, and the default.
	#[default]
	BaseFold,
	/// A ladder of Reed-Solomon commitments whose rate falls at every level.
	///
	/// The first level encodes at the caller's rate.
	/// So it does the same encoding work the incumbent does.
	/// Every deeper level holds a shorter message, which is where a lower rate is affordable.
	Ligerito,
}
