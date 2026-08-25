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
/// A message committed for zero knowledge carries one lane more than it folds.
/// That lane holds a mask of the message's own length, drawn before any challenge exists.
/// The masking challenge blends the two into the message every later level is about:
///
/// ```text
///     pi' = (1 - gamma) * pi + gamma * omega
/// ```
///
/// # Soundness
///
/// The coefficient must be drawn only once every claim it combines is bound to the transcript.
/// It is taken rather than sampled here so that the ordering is visible where the batch is built.
/// The masking challenge is under the same obligation, and arrives the same way.
#[derive(Debug, Clone)]
pub struct CommittedOracle<E, C> {
	/// The Merkle commitment to this message's level-0 interleaved codeword.
	commitment: C,
	/// log2 the number of interleaved lanes that codeword carries, the mask lane included.
	log_lanes: usize,
	/// The coefficient this message's level-0 row claims enter the batch with.
	coefficient: E,
	/// The challenge the mask lane is blended in at, absent for a message committed in the clear.
	mask_challenge: Option<E>,
}

impl<E: FieldOps, C: Clone> CommittedOracle<E, C> {
	/// Names one committed message, its lane count, and the coefficient it is batched with.
	///
	/// The message is committed in the clear, so every one of its lanes is a lane the ladder folds.
	pub const fn new(commitment: C, log_lanes: usize, coefficient: E) -> Self {
		Self {
			commitment,
			log_lanes,
			coefficient,
			mask_challenge: None,
		}
	}

	/// The same, for a message whose highest interleaved lane holds a mask.
	///
	/// `log_lanes` counts every lane the codeword carries, so the mask lane is one of them.
	/// The ladder's own challenges fold one fewer lane than that.
	///
	/// ## Preconditions
	///
	/// * `log_lanes` is positive, since a masked codeword carries the mask lane at least.
	pub fn masked(commitment: C, log_lanes: usize, coefficient: E, mask_challenge: E) -> Self {
		assert!(log_lanes > 0, "precondition: a masked codeword interleaves at least a mask lane");
		Self {
			commitment,
			log_lanes,
			coefficient,
			mask_challenge: Some(mask_challenge),
		}
	}

	/// log2 the number of interleaved lanes this message's level-0 codeword carries.
	pub const fn log_lanes(&self) -> usize {
		self.log_lanes
	}

	/// log2 the number of lanes the ladder's own fold challenges bind.
	///
	/// A mask lane is folded by the masking challenge instead, so it is not one of these.
	pub const fn log_folded_lanes(&self) -> usize {
		self.log_lanes - self.mask_challenge.is_some() as usize
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
	/// A mask lane sits above every lane the ladder folds, so the masking challenge leads the fold:
	///
	/// ```text
	///     fold point = [ gamma | this message's lanes .. ]
	/// ```
	///
	/// The opened coset is then blended and folded in one multilinear evaluation.
	/// That is the encoding of the blended message at the queried position, by linearity of both.
	///
	/// ## Preconditions
	///
	/// * `challenges` holds at least [`CommittedOracle::log_folded_lanes`] entries.
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
		let log_folded_lanes = self.log_folded_lanes();
		assert!(
			log_folded_lanes <= challenges.len(),
			"precondition: {log_folded_lanes} lanes cannot be folded by {} challenges",
			challenges.len()
		);

		// The lane challenges are the trailing ones, and everything before them is padding.
		let (padding, lanes) = challenges.split_at(challenges.len() - log_folded_lanes);
		let scale = self.coefficient.clone() * Hypercube::One.eq_ind_zero(padding);

		// The coset's highest variable is the mask lane, so its challenge leads the fold point.
		let fold_point = self
			.mask_challenge
			.iter()
			.cloned()
			.chain(lanes.iter().cloned())
			.collect::<Vec<_>>();

		// Every message's codeword has the same length, so the shared positions address this one
		// directly and no lift is needed.
		let rows = BrakedownOracle::new(fold_point, self.commitment.clone(), 0)
			.open_queries(indices, channel)?;

		Ok(rows.into_iter().map(|row| row * scale.clone()).collect())
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Ghash128b as B128};

	use super::*;

	#[test]
	fn a_clear_message_folds_every_lane_it_carries() {
		// Fixture state: four interleaved lanes, none of them a mask.
		let oracle = CommittedOracle::<B128, ()>::new((), 2, B128::ONE);
		assert_eq!(oracle.log_lanes(), 2);
		assert_eq!(oracle.log_folded_lanes(), 2);
	}

	#[test]
	fn a_masked_message_keeps_one_lane_for_the_mask() {
		// Fixture state: eight interleaved lanes, the highest holding the mask.
		//
		//     committed lanes: 2^3
		//     mask lane      : 1, folded by gamma
		//     ladder folds   : 2^2
		let oracle = CommittedOracle::<B128, ()>::masked((), 3, B128::ONE, B128::ONE);
		assert_eq!(oracle.log_lanes(), 3);
		assert_eq!(oracle.log_folded_lanes(), 2);
	}

	#[test]
	#[should_panic(expected = "precondition: a masked codeword interleaves at least a mask lane")]
	fn a_masked_message_with_no_lane_at_all_is_refused() {
		// A codeword of one lane has nowhere to put a mask beside the message.
		CommittedOracle::<B128, ()>::masked((), 0, B128::ONE, B128::ONE);
	}
}
