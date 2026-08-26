// Copyright 2026 The Binius Developers

use super::ladder_cost::LadderCost;

/// What a ladder search minimizes, as an exchange rate between the two prices of a ladder.
///
/// A ladder pays in proof bytes and in encoding work, and the two are not the same currency.
/// Naming an exchange rate is the only honest way to add them, so a caller names one.
/// The rate is read as: this many butterflies are worth spending to save one proof byte.
///
/// A large rate says bytes are scarce and encoding is cheap, which is the byte-optimal ladder.
/// A small rate says the opposite, and the search answers with a shallower rate ladder.
/// [`Self::BYTES_ONLY`] is the limit where encoding is free, and it is the default.
///
/// The score is linear in both axes, so it stays additive over the levels of a ladder.
/// That is what keeps the search's dynamic program exact.
/// A budget would not: minimizing bytes under a cap on encoding is not an additive objective.
/// A search under a cap has to carry the budget left in its state rather than in its score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LadderObjective {
	/// Proof bytes one encode butterfly is charged as.
	bytes_per_butterfly: f64,
}

impl LadderObjective {
	/// Minimizes proof bytes and ignores what the prover spends encoding them.
	///
	/// This is the objective a proof-size estimate alone describes.
	pub const BYTES_ONLY: Self = Self {
		bytes_per_butterfly: 0.0,
	};

	/// Butterflies one proof byte is worth, as this repository has already priced them.
	///
	/// The figure comes from the comparison that put a ladder beside a fold over a `2^22` witness:
	///
	/// - the ladder's opening costs 33 ms more, and its proof is about 96 KB smaller;
	/// - tracing that opening level by level puts the encoder at 1.5e6 butterflies per ms;
	/// - so 33 ms is some 50 million butterflies, which values a byte at about 500 of them.
	///
	/// Rounded down to 400, charging for encoding a little more dearly than that trade did.
	///
	/// A rate in butterflies per byte only says the same thing near the size it was measured at.
	/// [`Self::proportional`] is the scale-free way to ask for the same trade.
	pub const MEASURED_BUTTERFLIES_PER_BYTE: f64 = 400.0;

	/// The objective that charges for encoding at the rate the accepted comparison implies.
	pub const MEASURED: Self = Self {
		bytes_per_butterfly: 1.0 / Self::MEASURED_BUTTERFLIES_PER_BYTE,
	};

	/// An objective that will spend `butterflies_per_byte` of encoding to save one proof byte.
	///
	/// ## Preconditions
	///
	/// * `butterflies_per_byte` is positive and finite.
	pub fn new(butterflies_per_byte: f64) -> Self {
		assert!(
			butterflies_per_byte > 0.0 && butterflies_per_byte.is_finite(),
			"precondition: butterflies_per_byte must be positive and finite, got \
			 {butterflies_per_byte}"
		);
		Self {
			bytes_per_butterfly: 1.0 / butterflies_per_byte,
		}
	}

	/// An objective that trades the two axes in proportion to what they cost at one shape.
	///
	/// One part of the reference's bytes is worth `encode_per_bytes` parts of its encoding.
	/// So `1.0` takes a trade exactly when it saves a larger fraction than it costs.
	///
	/// The same number means the same trade at every message size, which an absolute rate cannot.
	/// A proof grows with the logarithm of the message, and its encoding grows with the message.
	/// So a rate balanced at one size charges for encoding as if it were everything at the next.
	///
	/// The byte-optimal ladder is the reference to reach for, and a byte-only search returns it.
	///
	/// ## Preconditions
	///
	/// * The reference encodes something, so its butterfly count is positive.
	/// * `encode_per_bytes` is positive and finite.
	pub fn proportional(reference: LadderCost, encode_per_bytes: f64) -> Self {
		assert!(
			reference.encode_butterflies > 0,
			"precondition: a reference shape must encode something"
		);
		assert!(
			encode_per_bytes > 0.0 && encode_per_bytes.is_finite(),
			"precondition: encode_per_bytes must be positive and finite, got {encode_per_bytes}"
		);
		Self {
			bytes_per_butterfly: reference.bytes as f64
				/ (encode_per_bytes * reference.encode_butterflies as f64),
		}
	}

	/// What one encode butterfly costs, in proof bytes.
	pub const fn bytes_per_butterfly(&self) -> f64 {
		self.bytes_per_butterfly
	}

	/// The scalar this objective ranks ladders by, in bytes.
	///
	/// Encoding is converted at the exchange rate and added to the bytes it stands beside.
	/// Ranking by this is ranking by proof size when nothing is charged for encoding.
	pub const fn bytes_equivalent(&self, cost: LadderCost) -> f64 {
		self.bytes_per_butterfly
			.mul_add(cost.encode_butterflies as f64, cost.bytes as f64)
	}
}

impl Default for LadderObjective {
	fn default() -> Self {
		Self::BYTES_ONLY
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn charging_nothing_for_encoding_ranks_ladders_by_their_bytes() {
		// Fixture state: two shapes, the second sending fewer bytes for far more encoding.
		let small_proof = LadderCost {
			bytes: 200_000,
			encode_butterflies: 400_000_000,
		};
		let cheap_encode = LadderCost {
			bytes: 240_000,
			encode_butterflies: 100_000_000,
		};

		// The default objective sees only the byte column, and its score is that column exactly.
		let bytes = LadderObjective::default();
		assert_eq!(bytes, LadderObjective::BYTES_ONLY);
		assert_eq!(bytes.bytes_equivalent(small_proof), 200_000.0);
		assert!(bytes.bytes_equivalent(small_proof) < bytes.bytes_equivalent(cheap_encode));

		// Priced at 400 butterflies per byte, the same pair reverses: the 300 million extra
		// butterflies are worth 750,000 bytes, and only 40,000 bytes were bought with them.
		let priced = LadderObjective::MEASURED;
		assert!(priced.bytes_equivalent(small_proof) > priced.bytes_equivalent(cheap_encode));
	}

	#[test]
	fn the_exchange_rate_is_the_reciprocal_of_the_byte_price() {
		// A caller states how much encoding a byte is worth, and the score charges the inverse.
		let objective = LadderObjective::new(500.0);
		assert_eq!(objective.bytes_per_butterfly(), 1.0 / 500.0);

		// So 500 butterflies cost exactly the one byte the rate names.
		let cost = LadderCost {
			bytes: 0,
			encode_butterflies: 500,
		};
		assert_eq!(objective.bytes_equivalent(cost), 1.0);

		// And the shipped rate is the one the measured trade implies.
		assert_eq!(
			LadderObjective::new(LadderObjective::MEASURED_BUTTERFLIES_PER_BYTE),
			LadderObjective::MEASURED
		);
	}

	#[test]
	fn the_score_is_additive_over_levels() {
		// Invariant: the search tabulates subproblems, so scoring a ladder level by level must
		// give what scoring the whole ladder gives. Linearity in both axes is what guarantees it.
		let objective = LadderObjective::new(64.0);
		let first = LadderCost {
			bytes: 3_000,
			encode_butterflies: 1_000_000,
		};
		let second = LadderCost {
			bytes: 1_500,
			encode_butterflies: 250_000,
		};

		let apart = objective.bytes_equivalent(first) + objective.bytes_equivalent(second);
		let together = objective.bytes_equivalent(first + second);
		assert!((apart - together).abs() < 1e-9, "{apart} {together}");
	}

	#[test]
	#[should_panic(expected = "precondition: butterflies_per_byte")]
	fn an_exchange_rate_of_zero_is_refused() {
		// Zero would mean no amount of encoding is worth one byte, which is a division by zero
		// rather than an objective. The bytes-only limit is a named constant instead.
		let _ = LadderObjective::new(0.0);
	}
}
