// Copyright 2026 The Binius Developers

//! One committed message, as level 0 of a ladder sees it.

use binius_field::{BinaryField, field::FieldOps};
use binius_math::multilinear::hypercube::Hypercube;

use crate::{
	fri::batch::{BrakedownOracle, Error, ProxTestOracle},
	merkle_channel::MerkleIPVerifierChannel,
};

/// One committed message a ladder opens, as its level 0 sees it.
///
/// A ladder can open several messages together.
/// They share level 0's column count, so their codewords all have one length.
/// One set of query positions then addresses every one of them.
/// What differs between them is the lane count, and the coefficient the batch weighs each by.
///
/// # Soundness
///
/// The coefficient must be drawn only once every claim it combines is bound to the transcript.
/// It is taken rather than sampled here so that the ordering is visible where the batch is built.
#[derive(Debug, Clone)]
pub struct CommittedOracle<E, C> {
	/// The Merkle commitment to this message's level-0 interleaved codeword.
	commitment: C,
	/// log2 the number of interleaved lanes that codeword carries.
	log_lanes: usize,
	/// The coefficient this message's level-0 row claims enter the batch with.
	coefficient: E,
}

impl<E: FieldOps, C: Clone> CommittedOracle<E, C> {
	/// Names one committed message, its lane count, and the coefficient it is batched with.
	pub const fn new(commitment: C, log_lanes: usize, coefficient: E) -> Self {
		Self {
			commitment,
			log_lanes,
			coefficient,
		}
	}

	/// log2 the number of interleaved lanes this message's level-0 codeword carries.
	pub const fn log_lanes(&self) -> usize {
		self.log_lanes
	}

	/// Opens this message's rows at the shared positions, folded and scaled into the batch.
	///
	/// The batch folds as many variables as its longest message has lanes.
	/// A message with fewer lanes is that multilinear zero-extended over the missing ones.
	/// So the first rounds bind variables it is zero over.
	/// Sumcheck binds the highest variable first, so those are the *first* challenges.
	/// Binding a variable the multilinear vanishes above scales it rather than folding it.
	///
	/// ```text
	///     challenges = [ padding .. | this message's lanes .. ]
	///     scale      = coefficient * eq(0, padding)
	/// ```
	///
	/// ## Preconditions
	///
	/// * `challenges` holds at least [`Self::log_lanes`] entries.
	pub(super) fn open_scaled_rows<F, Channel>(
		&self,
		challenges: &[E],
		indices: &[Channel::Word],
		channel: &mut Channel,
	) -> Result<Vec<E>, Error>
	where
		F: BinaryField,
		E: FieldOps<Scalar = F>,
		Channel: MerkleIPVerifierChannel<F, Commitment = C, Elem = E>,
	{
		assert!(
			self.log_lanes <= challenges.len(),
			"precondition: {} lanes cannot be folded by {} challenges",
			self.log_lanes,
			challenges.len()
		);

		// The lane challenges are the trailing ones, and everything before them is padding.
		let (padding, lanes) = challenges.split_at(challenges.len() - self.log_lanes);
		let scale = self.coefficient.clone() * Hypercube::One.eq_ind_zero(padding);

		// Every message's codeword has the same length, so the shared positions address this one
		// directly and no lift is needed.
		let rows = BrakedownOracle::new(lanes.to_vec(), self.commitment.clone(), 0)
			.open_queries(indices, channel)?;

		Ok(rows.into_iter().map(|row| row * scale.clone()).collect())
	}
}
