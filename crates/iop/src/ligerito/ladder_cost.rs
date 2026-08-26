// Copyright 2026 The Binius Developers

use std::ops::Add;

/// The two prices a ladder shape pays, counted without running a prover.
///
/// One is what crosses the wire, and the other is what the prover spends to put it there.
/// A ladder that folds fewer lanes per level sends fewer bytes and encodes more matrices.
/// So the two move against each other, and a search over shapes has to weigh them.
///
/// Both entries are additive over the levels of a ladder.
/// That is what lets the search tabulate subproblems and still return the true optimum.
///
/// [`LadderObjective`](super::LadderObjective) is what collapses the pair into one number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LadderCost {
	/// Proof bytes the ladder writes, counted exactly rather than estimated.
	pub bytes: usize,
	/// Butterfly operations the prover's Reed-Solomon encoders run over the whole ladder.
	pub encode_butterflies: usize,
}

impl LadderCost {
	/// A price paid entirely in bytes.
	///
	/// The residual is the one part of a proof that costs nothing to encode.
	/// It is folded to, committed, and then sent in the clear.
	pub const fn from_bytes(bytes: usize) -> Self {
		Self {
			bytes,
			encode_butterflies: 0,
		}
	}
}

impl Add for LadderCost {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self {
			bytes: self.bytes + rhs.bytes,
			encode_butterflies: self
				.encode_butterflies
				.saturating_add(rhs.encode_butterflies),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn costs_add_axis_by_axis() {
		// Two levels priced separately cost what the pair costs together, on both axes.
		let first = LadderCost {
			bytes: 1_024,
			encode_butterflies: 4_096,
		};
		let second = LadderCost {
			bytes: 512,
			encode_butterflies: 1_024,
		};
		assert_eq!(
			first + second,
			LadderCost {
				bytes: 1_536,
				encode_butterflies: 5_120,
			}
		);

		// The empty ladder is the identity, which is what lets the search fold over levels.
		assert_eq!(first + LadderCost::default(), first);
	}

	#[test]
	fn the_residual_is_priced_in_bytes_alone() {
		// The cleartext residual is committed and sent, never encoded, so its encode axis is zero.
		let residual = LadderCost::from_bytes(8_192);
		assert_eq!(residual.bytes, 8_192);
		assert_eq!(residual.encode_butterflies, 0);
	}
}
