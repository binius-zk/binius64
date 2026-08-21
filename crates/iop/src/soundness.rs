// Copyright 2026 The Binius Developers

//! Soundness accounting for Reed-Solomon proximity tests.
//!
//! A proximity test bounds the chance that a word far from the code survives.
//! Two independent terms bound it, and the protocol's security is the worse of the two.
//!
//! ```text
//!     bits    = min(algebra, queries)
//!
//!     algebra = -log2(eps_ca)                  <- fixed by the code and the field
//!     queries = -n_queries * log2(1 - delta)   <- bought one query at a time
//! ```
//!
//! The right-hand term is the one everybody counts.
//! Sampling `n_queries` positions catches a `delta`-far word all but `(1 - delta)^n_queries` of the
//! time, so more queries always buy more bits.
//!
//! The left-hand term is the *correlated-agreement error*.
//! It says how often a random linear combination of two far words lands close to the code anyway.
//! It grows with the codeword length and shrinks with the field size, and **no number of queries
//! affects it**.
//! It is a ceiling, and `PROXIMITY_GAPS.md` at the repository root tabulates where that ceiling
//! falls for the shapes this repo commits at.
//!
//! # Which property a protocol needs
//!
//! Three properties form a chain, each stronger than the last.
//!
//! - *Proximity gap*: either almost none of the line is close to the code, or all of it is.
//! - *Correlated agreement* (CA): all the words agree with codewords on one shared set.
//! - *Mutual correlated agreement* (MCA): they agree on *every* set that witnesses closeness.
//!
//! FRI needs CA, because it only ever concludes that the folded word is close to the code.
//! BaseFold, WHIR and Ligerito need MCA, because they conclude things about the *message* the
//! nearby codeword encodes.
//! [`SoundnessRegime::correlated_agreement_bits`] returns the MCA bound, which upper-bounds the CA
//! one, so a caller that only needs CA is charged conservatively.
//!
//! # Scope
//!
//! Every bound here is quoted from a theorem, not fitted or extrapolated.
//! Three scope questions decide whether a quoted theorem covers this repository.
//!
//! *Which field?*
//! [BCHKS25] Cor. 1.4 and Theorem 4.6 are stated for `RS[F_q, D, k]` over any finite field.
//! Neither assumes a characteristic.
//! So both cover the binary field this repo draws challenges from.
//!
//! *Which evaluation domain?*
//! Both are stated for an arbitrary domain of definition `D`.
//! That matters, because this repo's `D` is an `F_2`-affine subspace, not a multiplicative
//! subgroup, and several results in the literature assume the latter.
//! In particular [KKH26]'s limitation is proven for smooth multiplicative domains, so it does not
//! transfer here as stated.
//!
//! *Which randomness?*
//! Both theorems bound a line `u_0 + z * u_1` or a power curve `sum_i z^i * u_i`.
//! The FRI fold draws one challenge per layer and is a line, so it is covered.
//! Oracle batching instead uses the tensor randomness of [DP24], via `eq_ind_partial_eval`.
//! A tensor is not a power curve, so the bounds here transfer to the batching step only through
//! [DP24]'s own analysis, and only in the unique-decoding regime.
//! Read the [`SoundnessRegime::Johnson`] numbers as a bound on the fold, not on the batch.
//!
//! # What this module does not count
//!
//! Two terms belong in a whole-protocol budget and are not here.
//! Both reference Ligerito implementations carry them, so the numbers here are a floor on the
//! work, not a complete accounting.
//!
//! *Out-of-domain binding.* Past the unique-decoding radius the prover is near a *list* of
//! codewords, not one.
//! Binding it to a single element costs a term of roughly `binom(L, 2) * (mu/|F|)^s` for `s`
//! out-of-domain samples on a `mu`-variate multilinear.
//! Without it the query term pays a factor `L` instead.
//! Only the Johnson regime needs this, and this repo cannot afford that regime anyway.
//!
//! *The fold row union.* These bounds are stated for one fold step.
//! A level that folds `2^l` interleaved lanes pays the bound once per lane-fold round, and the two
//! reference implementations disagree on how much: one charges a factor `l`, the other `2^(l-1)`.
//! [`crate::ligerito::LigeritoParams::correlated_agreement_bits`] charges the pessimistic one.
//!
//! Proof-of-work grinding is also absent, and it is what both references use to close the last
//! few bits.
//! Grinding is a transcript-level device rather than a property of the code, so it belongs to
//! whichever protocol adopts it.
//!
//! One warning is *stronger* in characteristic 2 than elsewhere.
//! [BCHKS25] Theorem 1.6 and Cor. 1.7 show proximity gaps past the Johnson radius fail, and they
//! prove it *for all fields of characteristic 2*, using an `F_2`-subspace domain.
//! Their counterexamples need `n` near `sqrt(q)`, far past any size this repo commits at.
//! But they do rule out a general theorem past Johnson in exactly this setting.
//! [BCHKS25] leaves open whether well-chosen characteristic-2 domains escape it.
//!
//! # References
//!
//! - [BCHKS25] Cor. 1.4 is the unique-decoding error, and is optimal on its range.
//! - [BCHKS25] Theorem 4.6 is mutual correlated agreement up to the Johnson bound.
//! - [DG25] §1.5 is the query floor no conjecture can undercut, over all characteristics.
//! - [ABF26] surveys all of the above.
//!
//! [BCHKS25]: <https://eprint.iacr.org/2025/2055>
//! [DG25]: <https://eprint.iacr.org/2025/2010>
//! [KKH26]: <https://eprint.iacr.org/2026/782>
//! [DP24]: <https://eprint.iacr.org/2024/504>
//! [ABF26]: <https://eprint.iacr.org/2026/680>

use crate::fri::calculate_n_test_queries;

/// Which proximity-testing regime a code's parameters are derived in.
///
/// The variants differ only in how far out the proximity parameter `delta` is pushed.
/// Pushing it out buys query-phase soundness and costs correlated-agreement soundness.
/// Neither variant rests on a conjecture; both are theorems.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SoundnessRegime {
	/// The unique-decoding radius, `delta = (1 - rho)/2`.
	///
	/// A word that far from the code has at most one codeword near it.
	/// The correlated-agreement error is then `(delta * n + 1)/|F|`, per [BCHKS25] Cor. 1.4.
	/// That bound is tight, so nothing better exists on this range.
	/// The resulting query count is exactly what FRI uses today.
	///
	/// [BCHKS25]: <https://eprint.iacr.org/2025/2055>
	#[default]
	UniqueDecoding,
	/// The unique-decoding radius held back by a constant, `delta = (1 - rho)/2 - loss`.
	///
	/// Giving up an arbitrarily small constant of distance buys an exceptional set that does not
	/// grow with the codeword: [BCHKS25] Theorem 1.3 gives
	///
	/// ```text
	///     a >= max( (d/g - 1) / (d - 2g) , 1 + g/loss )     d = 1 - rho,  g = d/2 - loss
	/// ```
	///
	/// with **no `n` in it**.
	/// [`Self::UniqueDecoding`]'s bound is tight but carries a factor `n`, so its ceiling falls one
	/// bit per doubling of the witness.
	/// This one does not move at all.
	/// The price is a slightly smaller `delta`, so slightly more queries.
	///
	/// Both reference Ligerito implementations use this variant rather than the lossless one for
	/// their conjecture-free profile.
	///
	/// [BCHKS25]: <https://eprint.iacr.org/2025/2055>
	UniqueDecodingWithLoss {
		/// The distance `loss > 0` given up below the unique-decoding radius.
		///
		/// Smaller values keep more distance, so fewer queries, but drive `a` up as `1/loss`.
		/// [`Self::optimal_unique_decoding`] picks the value that balances the two.
		loss: f64,
	},
	/// The Johnson list-decoding bound, `delta = 1 - sqrt(rho) - eta`.
	///
	/// Past the unique-decoding radius the list of nearby codewords is no longer a singleton.
	/// Soundness then needs *mutual* correlated agreement, which [BCHKS25] Theorem 4.6 proves holds
	/// up to the Johnson bound with error `O_rho(n / (eta^5 * |F|))`.
	///
	/// This variant is **not** conjecture-gated.
	/// It is gated on having enough field: the `eta^5` makes it expensive.
	/// Check [`SoundnessRegime::correlated_agreement_bits`] before choosing it.
	///
	/// [BCHKS25]: <https://eprint.iacr.org/2025/2055>
	Johnson {
		/// The slack `eta > 0` held back from the Johnson radius, in units of relative distance.
		///
		/// Smaller values push `delta` further out, so each query rules out more.
		/// They also drive the correlated-agreement error up as `eta^-5`.
		/// [`SoundnessRegime::optimal_johnson`] picks the value that balances the two.
		eta: f64,
	},
}

impl SoundnessRegime {
	/// The proximity parameter `delta` this regime tests at, for a code of the given rate.
	///
	/// # Panics
	/// Panics if `log_inv_rate` is zero, since at rate 1 there is nothing to test.
	/// Panics if a [`Self::Johnson`] slack is not in `(0, 1 - sqrt(rho))`.
	pub fn proximity_parameter(self, log_inv_rate: usize) -> f64 {
		assert!(log_inv_rate >= 1, "precondition: log_inv_rate must be at least 1");
		let rho = rate(log_inv_rate);
		match self {
			Self::UniqueDecoding => (1.0 - rho) / 2.0,
			Self::UniqueDecodingWithLoss { loss } => {
				assert!(loss > 0.0, "precondition: loss must be positive");
				let delta = (1.0 - rho) / 2.0 - loss;
				assert!(delta > 0.0, "precondition: loss must be less than (1 - rho)/2");
				delta
			}
			Self::Johnson { eta } => {
				assert!(eta > 0.0, "precondition: eta must be positive");
				let delta = 1.0 - rho.sqrt() - eta;
				assert!(delta > 0.0, "precondition: eta must be less than 1 - sqrt(rho)");
				delta
			}
		}
	}

	/// Bits of query-phase soundness one row query rules out, at the given rate.
	///
	/// This is `-log2(1 - delta)`, the chance a single sampled position misses the disagreement.
	///
	/// # Panics
	/// Those of [`Self::proximity_parameter`].
	pub fn bits_per_query(self, log_inv_rate: usize) -> f64 {
		-(1.0 - self.proximity_parameter(log_inv_rate)).log2()
	}

	/// Row queries needed for `security_bits` bits of *query-phase* soundness.
	///
	/// This is `ceil(security_bits / bits_per_query)`, and accounts for the query phase alone.
	/// It says nothing about whether the correlated-agreement term also clears the target.
	/// Use [`Self::plan_queries`] to get a count that clears both.
	///
	/// For [`Self::UniqueDecoding`] this delegates to [`calculate_n_test_queries`].
	/// A Ligerito level in that regime therefore costs exactly as many queries as an FRI round.
	///
	/// # Panics
	/// Those of [`Self::proximity_parameter`].
	pub fn n_queries(self, security_bits: usize, log_inv_rate: usize) -> usize {
		match self {
			Self::UniqueDecoding => {
				assert!(log_inv_rate >= 1, "precondition: log_inv_rate must be at least 1");
				calculate_n_test_queries(security_bits, log_inv_rate)
			}
			Self::UniqueDecodingWithLoss { .. } | Self::Johnson { .. } => {
				(security_bits as f64 / self.bits_per_query(log_inv_rate)).ceil() as usize
			}
		}
	}

	/// Bits of soundness the correlated-agreement term alone allows, whatever the query count.
	///
	/// `log_msg_len` and `log_inv_rate` fix the codeword length `n = 2^(log_msg_len +
	/// log_inv_rate)`, and `log_field_size` is `log2|F|` of the field challenges are drawn from.
	///
	/// In [`Self::UniqueDecoding`] the error is `(delta * n + 1)/|F|`, per [BCHKS25] Cor. 1.4.
	/// That corollary is stated for `delta` a hair inside the radius, short of it by
	/// `3/((1 - rho) * n)` of relative distance.
	/// Evaluating at the radius itself moves the bound by far less than a bit.
	///
	/// In [`Self::Johnson`] the error is [BCHKS25] Theorem 4.6 at `M = 1`, plus the Johnson list
	/// size over the field.
	/// The list term is what [ABF26] §6.4 adds alongside the MCA term when accounting a round.
	///
	/// # Panics
	/// Those of [`Self::proximity_parameter`].
	///
	/// [BCHKS25]: <https://eprint.iacr.org/2025/2055>
	/// [ABF26]: <https://eprint.iacr.org/2026/680>
	pub fn correlated_agreement_bits(
		self,
		log_msg_len: usize,
		log_inv_rate: usize,
		log_field_size: usize,
	) -> f64 {
		let rho = rate(log_inv_rate);
		let n = 2.0f64.powi((log_msg_len + log_inv_rate) as i32);
		let delta = self.proximity_parameter(log_inv_rate);

		let exceptional = match self {
			// Cor. 1.4: the exceptional set has size `delta * n + 1`, and that is tight.
			Self::UniqueDecoding => delta * n + 1.0,
			// Theorem 1.3, and the `n` is gone. `delta` here is already the held-back radius.
			Self::UniqueDecodingWithLoss { loss } => {
				let code_distance = 1.0 - rho;
				let interpolation = (code_distance / delta - 1.0) / (code_distance - 2.0 * delta);
				interpolation.max(1.0 + delta / loss)
			}
			Self::Johnson { eta } => {
				// Theorem 4.6 at M = 1. The `m^5` is why small `eta` is expensive.
				let m = (rho.sqrt() / eta).ceil().max(3.0) + 0.5;
				let leading = (2.0 * m.powi(5) + 3.0 * m * delta * rho) / (3.0 * rho.powf(1.5));
				// The Johnson list size `1/(2 * eta * sqrt(rho))` rides along in the same round.
				leading * n + m / rho.sqrt() + 1.0 / (2.0 * eta * rho.sqrt())
			}
		};

		log_field_size as f64 - exceptional.log2()
	}

	/// The security a configuration actually reaches: the worse of its two terms.
	///
	/// See the module documentation for what the two terms are, and why only one of them responds
	/// to `n_queries`.
	///
	/// # Panics
	/// Those of [`Self::proximity_parameter`].
	pub fn achieved_security_bits(
		self,
		log_msg_len: usize,
		log_inv_rate: usize,
		log_field_size: usize,
		n_queries: usize,
	) -> f64 {
		let algebra = self.correlated_agreement_bits(log_msg_len, log_inv_rate, log_field_size);
		let queries = n_queries as f64 * self.bits_per_query(log_inv_rate);
		algebra.min(queries)
	}

	/// The smallest query count reaching `security_bits`, or `None` if the algebra caps below it.
	///
	/// A `None` is not a parameter that needs more queries.
	/// It is a configuration that cannot reach the target at all, because
	/// [`Self::correlated_agreement_bits`] is already under it.
	/// Widen the field, grind, or lower the target.
	///
	/// # Panics
	/// Those of [`Self::proximity_parameter`].
	pub fn plan_queries(
		self,
		security_bits: usize,
		log_msg_len: usize,
		log_inv_rate: usize,
		log_field_size: usize,
	) -> Option<usize> {
		let algebra = self.correlated_agreement_bits(log_msg_len, log_inv_rate, log_field_size);
		if algebra < security_bits as f64 {
			return None;
		}
		Some(self.n_queries(security_bits, log_inv_rate))
	}

	/// The [`Self::Johnson`] slack minimizing queries at `security_bits`, with the count it
	/// reaches.
	///
	/// Sweeps `eta` over `(0, 1 - sqrt(rho))`.
	/// Small `eta` pushes `delta` out and cuts queries, but drives the `eta^-5` error term up.
	/// Large `eta` does the reverse, until `delta` falls back inside the unique-decoding radius and
	/// the regime stops paying for itself.
	///
	/// Returns `None` when no slack reaches the target, which over a field this size is the common
	/// case rather than the exception.
	///
	/// # Panics
	/// Panics if `log_inv_rate` is zero.
	pub fn optimal_johnson(
		security_bits: usize,
		log_msg_len: usize,
		log_inv_rate: usize,
		log_field_size: usize,
	) -> Option<(Self, usize)> {
		assert!(log_inv_rate >= 1, "precondition: log_inv_rate must be at least 1");
		let span = 1.0 - rate(log_inv_rate).sqrt();

		// A fixed grid keeps the answer reproducible across platforms, which a solver would not.
		const STEPS: usize = 4096;
		(1..STEPS)
			.map(|i| Self::Johnson {
				eta: span * i as f64 / STEPS as f64,
			})
			.filter_map(|regime| {
				let queries = regime.plan_queries(
					security_bits,
					log_msg_len,
					log_inv_rate,
					log_field_size,
				)?;
				Some((regime, queries))
			})
			.min_by_key(|&(_, queries)| queries)
	}

	/// The unique-decoding loss minimizing queries at `security_bits`, with the count it reaches.
	///
	/// Sweeps `loss` over `(0, (1 - rho)/2)`.
	/// Small `loss` keeps more distance, so fewer queries, but drives the error up as `1/loss`.
	/// Large `loss` does the reverse.
	///
	/// Unlike [`Self::optimal_johnson`] this reaches high targets over a 128-bit field, because
	/// [`Self::UniqueDecodingWithLoss`]'s error does not grow with the codeword.
	///
	/// Returns `None` when no loss reaches the target.
	///
	/// # Panics
	/// Panics if `log_inv_rate` is zero.
	pub fn optimal_unique_decoding(
		security_bits: usize,
		log_msg_len: usize,
		log_inv_rate: usize,
		log_field_size: usize,
	) -> Option<(Self, usize)> {
		assert!(log_inv_rate >= 1, "precondition: log_inv_rate must be at least 1");
		let span = (1.0 - rate(log_inv_rate)) / 2.0;

		// A fixed grid keeps the answer reproducible across platforms, which a solver would not.
		const STEPS: usize = 4096;
		(1..STEPS)
			.map(|i| Self::UniqueDecodingWithLoss {
				loss: span * i as f64 / STEPS as f64,
			})
			.chain(std::iter::once(Self::UniqueDecoding))
			.filter_map(|regime| {
				let queries = regime.plan_queries(
					security_bits,
					log_msg_len,
					log_inv_rate,
					log_field_size,
				)?;
				Some((regime, queries))
			})
			.min_by_key(|&(_, queries)| queries)
	}

	/// The query count no conjecture can undercut, from [DG25] §1.5.
	///
	/// Regime-independent: this is a floor every variant of [`Self`] must clear.
	///
	/// Once the radius-`z` Hamming balls around codewords fill the space, a random word is close to
	/// the code by accident and proximity testing cannot work at all.
	/// That happens within
	///
	/// ```text
	///     eta_min ~ log2(e / rho) * rho / log2(q)
	/// ```
	///
	/// of capacity, so any regime must keep `delta` below `1 - rho - eta_min`, which floors the
	/// query count at `security_bits / -log2(rho + eta_min)`.
	///
	/// [DG25]'s own advice is to stay above this line rather than at it.
	/// Both regimes here sit well clear of it; the function exists so that a future
	/// conjecture-gated regime cannot silently sink below a proven impossibility.
	///
	/// # Panics
	/// Panics if `log_inv_rate` is zero.
	///
	/// [DG25]: <https://eprint.iacr.org/2025/2010>
	pub fn random_word_query_floor(
		security_bits: usize,
		log_inv_rate: usize,
		log_field_size: usize,
	) -> usize {
		assert!(log_inv_rate >= 1, "precondition: log_inv_rate must be at least 1");
		let rho = rate(log_inv_rate);
		let eta_min = (std::f64::consts::E / rho).log2() * rho / log_field_size as f64;
		(security_bits as f64 / -(rho + eta_min).log2()).ceil() as usize
	}
}

/// The code rate `2^-log_inv_rate`.
fn rate(log_inv_rate: usize) -> f64 {
	2.0f64.powi(-(log_inv_rate as i32))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// binius64's own target, and the field its challenges are drawn from.
	const B128: usize = 128;

	#[test]
	fn unique_decoding_query_counts_agree_with_fri() {
		for security_bits in [96, 100, 128] {
			for log_inv_rate in 1..=8 {
				assert_eq!(
					SoundnessRegime::UniqueDecoding.n_queries(security_bits, log_inv_rate),
					calculate_n_test_queries(security_bits, log_inv_rate),
					"security_bits={security_bits} log_inv_rate={log_inv_rate}"
				);
			}
		}
	}

	#[test]
	fn unique_decoding_query_counts_at_100_bits() {
		// The table in `LIGERITO_PLAN.md`: rates 1/2 through 1/32 at 100-bit security.
		let counts = (1..=5)
			.map(|log_inv_rate| SoundnessRegime::UniqueDecoding.n_queries(100, log_inv_rate))
			.collect::<Vec<_>>();
		assert_eq!(counts, vec![241, 148, 121, 110, 105]);
	}

	#[test]
	fn johnson_needs_fewer_queries_than_unique_decoding() {
		let johnson = SoundnessRegime::Johnson { eta: 0.02 };
		for log_inv_rate in 1..=8 {
			let udr = SoundnessRegime::UniqueDecoding.n_queries(100, log_inv_rate);
			assert!(johnson.n_queries(100, log_inv_rate) < udr, "log_inv_rate={log_inv_rate}");
		}
	}

	#[test]
	fn the_correlated_agreement_ceiling_falls_one_bit_per_doubling() {
		// Unique decoding, rate 1/2, over B128. The exceptional set is linear in the codeword
		// length, so every extra message bit costs one bit of headroom, up to the additive one.
		let bits = |log_msg_len| {
			SoundnessRegime::UniqueDecoding.correlated_agreement_bits(log_msg_len, 1, B128)
		};
		for log_msg_len in 20..32 {
			let step = bits(log_msg_len) - bits(log_msg_len + 1);
			assert!((step - 1.0).abs() < 1e-4, "log_msg_len={log_msg_len} step={step}");
		}
	}

	#[test]
	fn binius64_reaches_its_own_target_but_not_by_much() {
		// The shipped configuration: 96 bits, rate 1/2, unique decoding, B128 challenges.
		for (log_msg_len, expected) in [(20, 109.0), (24, 105.0), (28, 101.0), (32, 97.0)] {
			let ceiling =
				SoundnessRegime::UniqueDecoding.correlated_agreement_bits(log_msg_len, 1, B128);
			assert!((ceiling - expected).abs() < 0.01, "log_msg_len={log_msg_len} {ceiling}");

			// The query count clears the target, and the algebra clears it too, so 96 holds.
			let queries = SoundnessRegime::UniqueDecoding.plan_queries(96, log_msg_len, 1, B128);
			assert_eq!(queries, Some(232));
			let achieved =
				SoundnessRegime::UniqueDecoding.achieved_security_bits(log_msg_len, 1, B128, 232);
			assert!(achieved >= 96.0, "log_msg_len={log_msg_len} achieved={achieved}");
		}
	}

	#[test]
	fn the_lossless_bound_decays_with_size_and_the_lossy_one_does_not() {
		// Cor. 1.4 is tight but carries a factor `n`, so its ceiling slides.
		// Theorem 1.3 gives up a constant of distance and buys an `n`-free exceptional set.
		let lossless = |log_msg_len| {
			SoundnessRegime::UniqueDecoding.correlated_agreement_bits(log_msg_len, 1, B128)
		};
		let lossy = |log_msg_len| {
			SoundnessRegime::UniqueDecodingWithLoss { loss: 1e-3 }.correlated_agreement_bits(
				log_msg_len,
				1,
				B128,
			)
		};
		for (log_msg_len, expected) in [(20, 109.0), (24, 105.0), (28, 101.0), (32, 97.0)] {
			assert!((lossless(log_msg_len) - expected).abs() < 0.01);
			// The lossy bound does not move at all across the same range.
			assert!((lossy(log_msg_len) - lossy(20)).abs() < 1e-9, "log_msg_len={log_msg_len}");
		}
	}

	#[test]
	fn a_constant_of_distance_buys_one_hundred_and_twenty_eight_bits_over_b128() {
		// The lossless bound cannot reach 128 bits at any size, and neither can Johnson.
		for log_msg_len in [20, 24, 28] {
			for log_inv_rate in 1..=8 {
				assert_eq!(
					SoundnessRegime::UniqueDecoding.plan_queries(
						128,
						log_msg_len,
						log_inv_rate,
						B128
					),
					None,
				);
				assert_eq!(
					SoundnessRegime::optimal_johnson(128, log_msg_len, log_inv_rate, B128),
					None
				);
			}
		}

		// Giving up a constant of distance does not reach 128 either, but it comes far closer and
		// it stops sliding: over B128 at rate 1/2 the ceiling is a flat 124.5 bits at every size.
		// A handful of proof-of-work bits closes that fixed gap, where no amount of grinding keeps
		// up with a ceiling that falls one bit per doubling.
		for log_msg_len in [20, 24, 28, 32] {
			assert_eq!(SoundnessRegime::optimal_unique_decoding(128, log_msg_len, 1, B128), None);

			let ceiling = (1..4096)
				.map(|i| SoundnessRegime::UniqueDecodingWithLoss {
					loss: 0.25 * f64::from(i) / 4096.0,
				})
				.map(|regime| regime.correlated_agreement_bits(log_msg_len, 1, B128))
				.fold(f64::NEG_INFINITY, f64::max);
			assert!((ceiling - 124.46).abs() < 0.01, "log_msg_len={log_msg_len} {ceiling}");
		}

		// 120 bits is reachable, at 292 queries, independent of the witness size. That is the
		// target and regime the reference Ligerito implementation ships for its own
		// conjecture-free profile, which is a useful outside check on this arithmetic.
		for log_msg_len in [20, 24, 28, 32] {
			let (regime, queries) =
				SoundnessRegime::optimal_unique_decoding(120, log_msg_len, 1, B128)
					.expect("120 bits is reachable with a constant loss");
			assert!(matches!(regime, SoundnessRegime::UniqueDecodingWithLoss { .. }));
			assert_eq!(queries, 292);
		}
	}

	#[test]
	fn the_johnson_regime_is_out_of_reach_over_b128_at_the_shipped_target() {
		// At 96 bits the Johnson regime needs more field than B128 has, at every size this repo
		// commits at. Only the smallest shape clears it, and it costs more queries than unique
		// decoding rather than fewer.
		for log_inv_rate in 1..=4 {
			for log_msg_len in [24, 28, 30] {
				assert_eq!(
					SoundnessRegime::optimal_johnson(96, log_msg_len, log_inv_rate, B128),
					None
				);
			}
		}
		let (_, johnson) = SoundnessRegime::optimal_johnson(96, 20, 1, B128)
			.expect("the smallest shape clears it");
		let udr = SoundnessRegime::UniqueDecoding.n_queries(96, 1);
		assert!(johnson > udr, "johnson={johnson} udr={udr}");
	}

	#[test]
	fn a_wider_field_puts_both_regimes_back_in_reach() {
		// A degree-2 extension of B128 lifts the ceiling past every target considered here.
		for log_msg_len in [24, 28, 30] {
			assert!(
				SoundnessRegime::UniqueDecoding
					.plan_queries(128, log_msg_len, 1, 256)
					.is_some()
			);
			assert!(SoundnessRegime::optimal_johnson(128, log_msg_len, 1, 256).is_some());
		}
	}

	#[test]
	fn the_johnson_optimum_reproduces_the_published_instantiation() {
		// [ABF26] section 6.4 instantiates a rate-1/2 interleaved Reed-Solomon code over a sextic
		// extension of the Koala Bear prime, so log2|F| is about 186, and reports that 128-bit
		// knowledge soundness needs 259 queries at eta near 2^-8.6.
		//
		// That instantiation is a prime field over a smooth domain, where this repo is binary over
		// an affine subspace. The theorems being evaluated hold for both, so reproducing their
		// number cross-checks this arithmetic rather than claiming their setting.
		//
		// This model is the more conservative of the two. [ABF26] restates the error with the
		// `m = ceil(sqrt(rho) / (2 * eta))` of [BCHKS25] Theorem 1.5, which bounds *correlated*
		// agreement. Theorem 4.6, which bounds the *mutual* variant this repo needs, carries
		// `m = ceil(sqrt(rho) / eta)` instead. So a few more queries is the expected direction.
		let (regime, queries) =
			SoundnessRegime::optimal_johnson(128, 20, 1, 186).expect("reachable over 2^186");
		assert!((259..=265).contains(&queries), "queries={queries}");
		let SoundnessRegime::Johnson { eta } = regime else {
			panic!("optimal_johnson returns a Johnson regime");
		};
		// The `m` above is twice [ABF26]'s, so the balance point sits about one bit higher.
		assert!((-8.6..=-7.0).contains(&eta.log2()), "log2(eta)={}", eta.log2());
	}

	#[test]
	fn every_shipped_configuration_clears_the_random_word_floor() {
		// [DG25]'s floor is a proven impossibility, so a regime that dips below it is unsound
		// whatever else is assumed.
		for log_inv_rate in 1..=8 {
			let floor = SoundnessRegime::random_word_query_floor(96, log_inv_rate, B128);
			let udr = SoundnessRegime::UniqueDecoding.n_queries(96, log_inv_rate);
			assert!(udr > floor, "log_inv_rate={log_inv_rate} udr={udr} floor={floor}");
		}
	}

	#[test]
	fn the_random_word_floor_matches_the_published_table() {
		// [DG25] Table 1, at 100 bits over a 128-bit field: 102.80, 50.98, 33.89, 25.38, 20.29.
		let floors = (1..=5)
			.map(|log_inv_rate| SoundnessRegime::random_word_query_floor(100, log_inv_rate, B128))
			.collect::<Vec<_>>();
		assert_eq!(floors, vec![103, 51, 34, 26, 21]);
	}

	#[test]
	#[should_panic(expected = "log_inv_rate must be at least 1")]
	fn rate_one_is_not_a_proximity_test() {
		SoundnessRegime::UniqueDecoding.bits_per_query(0);
	}

	#[test]
	#[should_panic(expected = "eta must be less than 1 - sqrt(rho)")]
	fn a_slack_past_the_johnson_radius_is_rejected() {
		SoundnessRegime::Johnson { eta: 0.5 }.bits_per_query(1);
	}
}
