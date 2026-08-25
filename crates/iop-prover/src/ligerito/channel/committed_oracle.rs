// Copyright 2026 The Binius Developers

//! Everything a Ligerito prover channel holds about one oracle it has committed.

use binius_compute::Allocator;
use binius_field::PackedField;
use binius_math::FieldVec;

use super::mask::Mask;
use crate::fri::BrakedownOracleProver;

/// One oracle a Ligerito prover channel has committed, with everything its opening will need.
///
/// The codeword exists from the moment the commitment goes out.
/// The mask exists alongside it for a zero-knowledge oracle, and never leaves the channel.
/// The message arrives later, when the caller hands the buffer back to be folded.
pub(super) struct CommittedOracle<P: PackedField, C, A: Allocator> {
	/// The level-0 codeword this oracle's query openings are written from.
	codeword: BrakedownOracleProver<P, C>,
	/// The mask committed beside the message, absent for an oracle committed in the clear.
	mask: Option<Mask<P, A>>,
	/// The committed message, absent until the caller finalizes the oracle.
	message: Option<FieldVec<P, A>>,
}

impl<P: PackedField, C, A: Allocator> CommittedOracle<P, C, A> {
	/// Records a committed codeword and, for a zero-knowledge oracle, the mask inside it.
	pub(super) const fn new(
		codeword: BrakedownOracleProver<P, C>,
		mask: Option<Mask<P, A>>,
	) -> Self {
		Self {
			codeword,
			mask,
			message: None,
		}
	}

	/// Takes the committed message, which the ladder folds down to the residual.
	///
	/// The codeword alone is not enough for that, which is why the buffer is handed back at all.
	///
	/// ## Preconditions
	///
	/// * The oracle was not already finalized.
	pub(super) fn finalize(&mut self, message: FieldVec<P, A>) {
		assert!(
			self.message.replace(message).is_none(),
			"precondition: an oracle is finalized at most once"
		);
	}

	/// Splits into the pieces the opening reads, leaving nothing behind.
	///
	/// ## Preconditions
	///
	/// * The oracle was finalized.
	pub(super) fn split(self) -> OracleParts<P, C, A> {
		let message = self
			.message
			.expect("precondition: the oracle was committed but never finalized");
		OracleParts {
			codeword: self.codeword,
			mask: self.mask,
			message,
		}
	}
}

/// What one committed oracle contributes to the opening, once its message has been handed back.
pub(super) struct OracleParts<P: PackedField, C, A: Allocator> {
	/// The level-0 codeword the query phase answers from.
	pub(super) codeword: BrakedownOracleProver<P, C>,
	/// The mask blended into the message, absent for an oracle committed in the clear.
	pub(super) mask: Option<Mask<P, A>>,
	/// The committed message the ladder folds down to the residual.
	pub(super) message: FieldVec<P, A>,
}
