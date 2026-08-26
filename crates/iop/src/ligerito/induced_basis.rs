// Copyright 2026 The Binius Developers

//! The weight vector a level's opened rows induce on its message.
//!
//! A Ligerito level opens `t` rows of a committed Reed-Solomon codeword.
//! Row `q` is one row of the generator matrix, so opening it asserts
//!
//! ```text
//!     <G_q, m> = y_q
//! ```
//!
//! for the level's message `m`.
//! That is `t` separate linear claims, and the protocol needs one.
//!
//! Batching them with the powers of a challenge `alpha` gives a single claim
//!
//! ```text
//!     <w, m> = sum_i alpha^i * y_i,      w[j] = sum_i alpha^i * G_{q_i}[j]
//! ```
//!
//! and `w` is what this module calls the *induced basis*.
//! It is the weight vector of a sumcheck over `m`, which is the shape `mlecheck` already handles.
//!
//! # The verifier never builds it
//!
//! `w` has `2^log_dim` entries, which at production sizes is millions of field elements.
//! A verifier only ever needs its multilinear extension at one point, and that has a closed form.
//!
//! Write `f_i[k]` for the `k`-th tensor factor of row `q_i`, so that
//! `G_{q_i}[j] = prod_{k : bit_k(j) = 1} f_i[k]`.
//! Expanding `eq` in characteristic 2, where `1 - x = 1 + x`:
//!
//! ```text
//!     MLE(w)(p) = sum_j w[j] * prod_{k : j_k = 1} p_k * prod_{k : j_k = 0} (1 + p_k)
//!               = sum_i alpha^i * prod_k ( (1 + p_k) + f_i[k] * p_k )
//!               = sum_i alpha^i * prod_k ( 1 + p_k * (1 + f_i[k]) )
//! ```
//!
//! The sum over `2^log_dim` terms collapsed into a product over `log_dim` of them.
//! [`InducedBasis::evaluate`] costs `O(t * log_dim)` multiplications and no divisions, against
//! `O(t * 2^log_dim)` to build the vector and read it.
//!
//! No division matters as much as the operation count.
//! A verifier compiled into a circuit cannot invert a witness-dependent value.
//! This path never asks it to.
//! Its other input `f_i[k]` is a subset sum of precomputed constants.
//! That is the gate shape the FRI fold already pays for its twiddles.
//!
//! [`InducedBasis::to_dense`] materializes `w` anyway, as the reference the closed form is tested
//! against, and for a prover that genuinely needs every entry.
//! It needs a packed field, so a verifier written against a channel cannot reach it at all.
//! [`InducedBasis::pair`] is how a terminal level pairs `w` with a message it holds in the clear.

use binius_field::{
	BinaryField, PackedField,
	field::FieldOps,
	util::{expand_subset_products, powers},
};
use binius_ip::channel::WordIPVerifierChannel;
use binius_math::{
	inner_product::inner_product_scalars,
	ntt::{DomainContext, subspace_polys::evals_at_domain_index},
};

/// The weight vector `t` opened rows induce on a level's message.
///
/// Holds the per-row tensor factors rather than the vector itself.
/// That is what lets [`Self::evaluate`] stay logarithmic in the message length.
#[derive(Debug, Clone)]
pub struct InducedBasis<F> {
	/// `factors[i][k]` is the `k`-th tensor factor of the generator row at query `i`.
	///
	/// Entry `j` of that row is the product of the factors its set bits select.
	/// So the row is `2^n_vars` values described by `n_vars` of them.
	factors: Vec<Vec<F>>,
	/// The coefficient each row is batched with, `alpha^0, .., alpha^{t-1}` as built.
	///
	/// Held here so that no caller can batch the two halves of the claim with different values.
	/// [`InducedBasis::fold_high`] scales each entry by what its row contributes to the fold, so a
	/// folded basis carries the powers of `alpha` multiplied through.
	batching: Vec<F>,
	/// The code's `log_dim`, and the number of variables the weight vector spans.
	///
	/// Stored rather than read off `factors`, which is empty when no rows are opened.
	n_vars: usize,
}

impl<F: FieldOps> InducedBasis<F> {
	/// Builds the basis induced by opening `indices` of a code of dimension `log_dim`.
	///
	/// `domain_context` must be the one the codeword was encoded over.
	/// Its `log_domain_size` is therefore `log_dim + log_inv_rate`, not `log_dim`.
	/// The generator row lives on the codeword domain.
	/// The message dimension only says how much of that row to keep.
	///
	/// Truncating to `log_dim` factors drops the dimensions encoding zero-pads.
	/// Reversing them applies the bit-reversal permutation `encode_batch` encodes under.
	/// Both are pinned by tests in [`binius_math::ntt::subspace_polys`].
	///
	/// ## Preconditions
	///
	/// * `log_dim` is at most `domain_context.log_domain_size()`.
	/// * every index is a codeword position, below `2^log_domain_size`.
	///
	/// The second is the channel's job, which masks what it samples to the domain.
	/// An index past the end is a wiring bug rather than a dishonest prover, so it panics.
	pub fn new<DC: DomainContext<Field = F>>(
		domain_context: &DC,
		log_dim: usize,
		indices: &[usize],
		alpha: F,
	) -> Self
	where
		F: BinaryField,
	{
		let log_domain_size = domain_context.log_domain_size();
		assert!(
			log_dim <= log_domain_size,
			"precondition: log_dim must be at most the domain's own dimension"
		);
		assert!(
			indices.iter().all(|&index| index < 1 << log_domain_size),
			"precondition: every index must be below 2^log_domain_size"
		);

		let factors = indices
			.iter()
			.map(|&index| {
				let mut evals = evals_at_domain_index(domain_context, index);
				evals.truncate(log_dim);
				evals.reverse();
				evals
			})
			.collect::<Vec<_>>();

		// Powers of the batching challenge, one per opened row.
		let batching = powers(alpha).take(indices.len()).collect();

		Self {
			factors,
			batching,
			n_vars: log_dim,
		}
	}

	/// Builds the basis from query indices the channel holds as opaque words.
	///
	/// This is the route a recursion circuit takes, where an index is a wire rather than a number.
	///
	/// [`Self::new`] reads factor `k` as `subspace(l - k).get(index >> k)`.
	/// A circuit cannot shift a word cheaply, so this shifts the *table* instead.
	/// Padding row `k` with `k` leading zeros lines bit `k + j` of the index up with entry `j`.
	/// The padding contributes nothing to the sum, so the two routes agree by construction.
	///
	/// Every factor is then one [`WordIPVerifierChannel::subset_sum`] over fixed constants.
	/// That is the gate shape the FRI fold already pays for its twiddles.
	///
	/// ## Preconditions
	///
	/// * `log_dim` is at most `domain_context.log_domain_size()`.
	pub fn from_query_words<FSub, DC, Channel>(
		domain_context: &DC,
		log_dim: usize,
		indices: &[Channel::Word],
		alpha: &F,
		channel: &mut Channel,
	) -> Self
	where
		FSub: BinaryField,
		F: From<FSub>,
		DC: DomainContext<Field = FSub>,
		Channel: WordIPVerifierChannel<FSub, Elem = F>,
	{
		let log_domain_size = domain_context.log_domain_size();
		assert!(
			log_dim <= log_domain_size,
			"precondition: log_dim must be at most the domain's own dimension"
		);

		// Row `k` of the table starts at `beta_k`, so bit `k + j` of an index selects entry `j`.
		let padded = (0..log_dim)
			.map(|k| {
				let subspace = domain_context.subspace(log_domain_size - k);
				std::iter::repeat_n(F::zero(), k)
					.chain(subspace.basis().iter().copied().map(F::from))
					.collect::<Vec<_>>()
			})
			.collect::<Vec<_>>();

		let factors = indices
			.iter()
			.map(|index| {
				let mut row = padded
					.iter()
					.map(|elems| channel.subset_sum(elems, index))
					.collect::<Vec<_>>();
				// Same reversal `Self::new` applies, for the same reason.
				row.reverse();
				row
			})
			.collect();

		Self {
			factors,
			batching: powers(alpha.clone()).take(indices.len()).collect(),
			n_vars: log_dim,
		}
	}

	/// The number of variables the weighted multilinear spans, which is the code's `log_dim`.
	pub const fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// The number of opened rows the basis batches.
	pub const fn n_rows(&self) -> usize {
		self.factors.len()
	}

	/// Evaluates the basis's multilinear extension at `point`, in `O(n_rows * n_vars)`.
	///
	/// This is the module documentation's closed form, and it is the verifier's only route.
	/// Opening no rows induces the zero weight vector, whose extension is zero everywhere.
	///
	/// ## Preconditions
	///
	/// * `point` has [`Self::n_vars`] coordinates.
	pub fn evaluate(&self, point: &[F]) -> F {
		assert_eq!(point.len(), self.n_vars, "precondition: point must have n_vars coordinates");

		let terms = self.factors.iter().map(|row| {
			// One factor per variable: `1 + p_k * (1 + f[k])`, a multiplication and two additions.
			std::iter::zip(row, point)
				.map(|(factor, coordinate)| {
					F::one() + coordinate.clone() * (F::one() + factor.clone())
				})
				.product::<F>()
		});
		inner_product_scalars(self.batching.iter().cloned(), terms)
	}

	/// Binds the top `challenges.len()` variables, giving the basis of the folded message.
	///
	/// A recursive level glues its induced basis into a running sumcheck, and every later level's
	/// rounds bind more of that sumcheck's variables.
	/// So a basis introduced at one level has to follow the message it weighs down the ladder.
	///
	/// Sumcheck binds the highest variable first, so `challenges[0]` binds variable `n_vars - 1`.
	/// A row is a tensor, and binding variable `k` of a tensor at `c` scales the whole row:
	///
	/// ```text
	///     (1 - c) * 1 + c * f[k]  =  1 + c * (1 + f[k])
	/// ```
	///
	/// which is the same per-variable factor [`Self::evaluate`] uses.
	/// The scale is therefore folded into the row's batching coefficient and the factor dropped,
	/// which costs `O(n_rows * challenges.len())` and no allocation per row beyond the truncation.
	///
	/// [`Self::enforced_sum`] is not meaningful on the result.
	/// The opened row values it batches belong to the level that introduced the basis, and this
	/// basis no longer weighs that level's message.
	///
	/// ## Preconditions
	///
	/// * `challenges.len()` is at most [`Self::n_vars`].
	pub fn fold_high(&self, challenges: &[F]) -> Self {
		assert!(
			challenges.len() <= self.n_vars,
			"precondition: cannot bind more variables than the basis has"
		);

		let n_vars = self.n_vars - challenges.len();
		let batching = std::iter::zip(&self.batching, &self.factors)
			.map(|(coefficient, row)| {
				// `challenges[0]` binds the highest variable, which is the last factor.
				let scale = std::iter::zip(row[n_vars..].iter().rev(), challenges)
					.map(|(factor, challenge)| {
						F::one() + challenge.clone() * (F::one() + factor.clone())
					})
					.product::<F>();
				coefficient.clone() * scale
			})
			.collect();
		let factors = self
			.factors
			.iter()
			.map(|row| row[..n_vars].to_vec())
			.collect();

		Self {
			factors,
			batching,
			n_vars,
		}
	}

	/// Pairs the basis with a message held in the clear, giving `<w, message>`.
	///
	/// This is the other side of the equation [`Self::enforced_sum`] gives, so a terminal level
	/// checks the two against each other.
	///
	/// Expanding the definition of `w`,
	///
	/// ```text
	///     <w, message> = sum_i alpha^i * <G_{q_i}, message>
	/// ```
	///
	/// and each row pairing folds `message` against that row's factors, one variable per step.
	/// Entry `j` of a row selects `f[k]` exactly when bit `k` of `j` is set, so folding the pair
	/// `(even, odd)` into `even + f[k] * odd` strips variable `k`.
	///
	/// Costs `O(n_rows * 2^n_vars)` and never materializes a row, so unlike [`Self::to_dense`] it
	/// is available wherever the basis itself is.
	/// A recursive level has no message in the clear to pair against and uses [`Self::evaluate`]
	/// inside a sumcheck instead.
	///
	/// ## Preconditions
	///
	/// * `message` has `2^n_vars` entries.
	pub fn pair(&self, message: &[F]) -> F {
		assert_eq!(
			message.len(),
			1 << self.n_vars,
			"precondition: message must have 2^n_vars entries"
		);

		let rows = self.factors.iter().map(|factors| {
			let mut folded = message.to_vec();
			for factor in factors {
				let half = folded.len() / 2;
				for j in 0..half {
					folded[j] = folded[2 * j].clone() + factor.clone() * folded[2 * j + 1].clone();
				}
				folded.truncate(half);
			}
			folded
				.pop()
				.expect("folding n_vars times leaves exactly one element")
		});
		inner_product_scalars(self.batching.iter().cloned(), rows)
	}

	/// The dense weight vector, `2^n_vars` entries.
	///
	/// This is the reference [`Self::evaluate`] is tested against, and the vector a prover folds.
	/// A verifier must not call it.
	/// The allocation alone defeats the point of the closed form.
	/// In a recursion circuit it is not expressible at all.
	pub fn to_dense(&self) -> Vec<F>
	where
		F: PackedField,
	{
		let mut dense = vec![F::zero(); 1 << self.n_vars];
		for (row, &coefficient) in std::iter::zip(&self.factors, &self.batching) {
			for (entry, weight) in std::iter::zip(&mut dense, expand_subset_products(row)) {
				*entry += coefficient * weight;
			}
		}
		dense
	}

	/// Batches the opened rows' own values into the claim [`Self::evaluate`] is checked against.
	///
	/// `row_values[i]` is row `q_i` folded by the level's challenges.
	/// So this returns `sum_i alpha^i * row_values[i]`.
	/// Pairing it with [`Self::evaluate`] turns the `t` row claims into one sumcheck claim.
	///
	/// ## Preconditions
	///
	/// * `row_values` has [`Self::n_rows`] entries.
	pub fn enforced_sum(&self, row_values: &[F]) -> F {
		assert_eq!(
			row_values.len(),
			self.n_rows(),
			"precondition: row_values must have one entry per opened row"
		);
		inner_product_scalars(self.batching.iter().cloned(), row_values.iter().cloned())
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{Field, Ghash128b as B128, Random};
	use binius_math::{
		FieldBuffer,
		multilinear::{MultilinearMut, evaluate::evaluate_inplace_scalars},
		ntt::{
			NeighborsLastSingleThread,
			domain_context::{GaoMateerOnTheFly, GaoMateerPreExpanded},
		},
		reed_solomon::ReedSolomonCode,
		test_utils::random_field_buffer,
	};
	use proptest::prelude::*;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;

	/// The codeword domain a `ReedSolomonCode` of this shape encodes over.
	fn domain(log_dim: usize, log_inv_rate: usize) -> GaoMateerPreExpanded<B128> {
		GaoMateerPreExpanded::generate(log_dim + log_inv_rate)
	}

	/// The identity the whole construction exists to provide.
	///
	/// Batching the opened rows by `alpha` must be the same claim as pairing the message with the
	/// induced weight vector.
	/// Were the two to disagree, prover and verifier would agree with each other and nothing else.
	#[test]
	fn the_induced_claim_is_the_batched_codeword_positions() {
		for log_dim in 1..7 {
			for log_inv_rate in 1..4 {
				let code = ReedSolomonCode::<B128>::new(log_dim, log_inv_rate);
				let ntt =
					NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(code.log_len()));
				let mut rng = StdRng::seed_from_u64(0);
				let message = random_field_buffer::<B128>(&mut rng, log_dim);
				let codeword = code.encode_batch(&ntt, message.as_view(), 0, &GlobalAllocator);

				// Open every position, which is the strongest version of the check.
				let indices = (0..1 << code.log_len()).collect::<Vec<_>>();
				let alpha = B128::random(&mut rng);
				let basis =
					InducedBasis::new(&domain(log_dim, log_inv_rate), log_dim, &indices, alpha);

				let opened = indices
					.iter()
					.map(|&index| codeword.as_ref()[index])
					.collect::<Vec<_>>();
				let paired =
					inner_product_scalars(basis.to_dense(), message.as_ref().iter().copied());
				assert_eq!(
					paired,
					basis.enforced_sum(&opened),
					"log_dim={log_dim} log_inv_rate={log_inv_rate}"
				);

				// `pair` is the route a verifier takes when it cannot build the dense vector, so it
				// has to reach the same claim.
				assert_eq!(
					basis.pair(message.as_ref()),
					paired,
					"log_dim={log_dim} log_inv_rate={log_inv_rate}"
				);
			}
		}
	}

	#[test]
	fn one_row_at_a_trivial_challenge_is_the_generator_row() {
		let (log_dim, log_inv_rate) = (4, 2);
		let code = ReedSolomonCode::<B128>::new(log_dim, log_inv_rate);
		let ntt = NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(code.log_len()));
		let mut rng = StdRng::seed_from_u64(1);
		let message = random_field_buffer::<B128>(&mut rng, log_dim);
		let codeword = code.encode_batch(&ntt, message.as_view(), 0, &GlobalAllocator);

		// A single row batched by alpha^0 = 1 is that row untouched, so pairing it with the
		// message must reproduce exactly that codeword position.
		for index in 0..1 << code.log_len() {
			let basis =
				InducedBasis::new(&domain(log_dim, log_inv_rate), log_dim, &[index], B128::ZERO);
			let paired = inner_product_scalars(basis.to_dense(), message.as_ref().iter().copied());
			assert_eq!(paired, codeword.as_ref()[index], "index={index}");
		}
	}

	#[test]
	fn opening_no_rows_induces_the_zero_vector() {
		let basis = InducedBasis::new(&domain(5, 1), 5, &[], B128::ONE);
		assert_eq!(basis.n_rows(), 0);
		assert_eq!(basis.n_vars(), 5);
		// Still a full-width vector of zeros, not an empty one.
		assert_eq!(basis.to_dense(), vec![B128::ZERO; 32]);
		assert_eq!(basis.evaluate(&[B128::ONE; 5]), B128::ZERO);
		assert_eq!(basis.enforced_sum(&[]), B128::ZERO);
	}

	#[test]
	#[should_panic(expected = "log_dim must be at most the domain's own dimension")]
	fn a_dimension_past_the_domain_is_rejected() {
		InducedBasis::new(&domain(3, 1), 5, &[0], B128::ONE);
	}

	#[test]
	#[should_panic(expected = "every index must be below 2^log_domain_size")]
	fn an_index_past_the_codeword_is_rejected() {
		InducedBasis::new(&domain(3, 1), 3, &[16], B128::ONE);
	}

	#[test]
	#[should_panic(expected = "point must have n_vars coordinates")]
	fn a_point_of_the_wrong_width_is_rejected() {
		InducedBasis::new(&domain(3, 1), 3, &[0], B128::ONE).evaluate(&[B128::ONE; 2]);
	}

	#[test]
	#[should_panic(expected = "row_values must have one entry per opened row")]
	fn a_row_count_mismatch_is_rejected() {
		InducedBasis::new(&domain(3, 1), 3, &[0, 1], B128::ONE).enforced_sum(&[B128::ONE]);
	}

	#[test]
	#[should_panic(expected = "cannot bind more variables than the basis has")]
	fn folding_past_the_last_variable_is_rejected() {
		InducedBasis::new(&domain(3, 1), 3, &[0], B128::ONE).fold_high(&[B128::ONE; 4]);
	}

	#[test]
	fn the_two_ends_of_a_fold_are_the_identity_and_a_scalar() {
		let basis = InducedBasis::new(&domain(4, 1), 4, &[0, 3, 9], B128::ONE);

		// Binding nothing leaves the weight vector alone.
		assert_eq!(basis.fold_high(&[]).to_dense(), basis.to_dense());

		// Binding every variable leaves a zero-variate weight, whose one entry is the whole
		// multilinear evaluated there.
		let point = [B128::ONE, B128::ZERO, B128::ONE, B128::ONE];
		let all = basis.fold_high(&[point[3], point[2], point[1], point[0]]);
		assert_eq!(all.n_vars(), 0);
		assert_eq!(all.to_dense(), vec![basis.evaluate(&point)]);
	}

	proptest! {
		/// The closed form is the verifier's only route.
		/// So it must agree with the dense vector everywhere, not at points a fixed test picks.
		#[test]
		fn the_succinct_evaluation_matches_the_dense_vector(
			seed: u64,
			log_dim in 1usize..7,
			log_inv_rate in 1usize..4,
			n_rows in 0usize..6,
		) {
			let mut rng = StdRng::seed_from_u64(seed);
			let log_len = log_dim + log_inv_rate;
			let indices = (0..n_rows)
				.map(|_| rng.random_range(0..1usize << log_len))
				.collect::<Vec<_>>();
			let alpha = B128::random(&mut rng);
			let basis = InducedBasis::new(&domain(log_dim, log_inv_rate), log_dim, &indices, alpha);

			let point = (0..log_dim).map(|_| B128::random(&mut rng)).collect::<Vec<_>>();
			let reference = evaluate_inplace_scalars(basis.to_dense(), &point);
			prop_assert_eq!(basis.evaluate(&point), reference);
		}

		/// A glued basis follows its message down the ladder, so folding it must be folding it.
		/// The reference is the dense weight vector put through the multilinear fold itself.
		#[test]
		fn folding_the_basis_folds_the_weight_vector(
			seed: u64,
			log_dim in 1usize..7,
			log_inv_rate in 1usize..4,
			n_rows in 0usize..6,
			n_folded in 0usize..7,
		) {
			let n_folded = n_folded.min(log_dim);
			let mut rng = StdRng::seed_from_u64(seed);
			let log_len = log_dim + log_inv_rate;
			let indices = (0..n_rows)
				.map(|_| rng.random_range(0..1usize << log_len))
				.collect::<Vec<_>>();
			let alpha = B128::random(&mut rng);
			let basis = InducedBasis::new(&domain(log_dim, log_inv_rate), log_dim, &indices, alpha);

			let challenges = (0..n_folded).map(|_| B128::random(&mut rng)).collect::<Vec<_>>();
			let mut reference = FieldBuffer::<B128>::from_values(&basis.to_dense());
			for challenge in &challenges {
				reference.fold_highest_var(*challenge);
			}

			let folded = basis.fold_high(&challenges);
			prop_assert_eq!(folded.n_vars(), log_dim - n_folded);
			prop_assert_eq!(folded.to_dense(), reference.as_ref().to_vec());
		}
	}
	/// The recursion-safe route must agree with the native one at every index.
	///
	/// A circuit cannot shift a word, so the channel route shifts the table instead.
	/// Were the two to disagree, a native verifier and an in-circuit one would accept different
	/// proofs.
	/// That is the failure mode the channel discipline exists to prevent.
	#[test]
	fn the_channel_route_agrees_with_the_native_one() {
		use binius_hash::{StdDigest, StdHashSuite};
		use binius_ip::channel::WordIPVerifierChannel;
		use binius_transcript::{VerifierTranscript, fiat_shamir::HasherChallenger};

		use crate::merkle_channel::VerifierMerkleTranscriptChannel;

		type StdChallenger = HasherChallenger<StdDigest>;

		for log_dim in 1..6 {
			for log_inv_rate in 1..4 {
				let log_len = log_dim + log_inv_rate;
				let dc = domain(log_dim, log_inv_rate);

				let mut transcript = VerifierTranscript::new(StdChallenger::default(), vec![]);
				let mut channel =
					VerifierMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::new(
						&mut transcript,
					);

				// Sample real query words, the way a verifier would.
				let words = (0..8)
					.map(|_| WordIPVerifierChannel::sample_bits(&mut channel, log_len))
					.collect::<Vec<_>>();
				let indices = words
					.iter()
					.map(|word| word.as_u64() as usize)
					.collect::<Vec<_>>();

				let alpha = B128::new(0x9e3779b97f4a7c15);
				let native = InducedBasis::new(&dc, log_dim, &indices, alpha);
				let via_channel =
					InducedBasis::from_query_words(&dc, log_dim, &words, &alpha, &mut channel);

				// Agreeing on the dense vector is the strongest form of the check: it pins every
				// factor, not just their product at one point.
				assert_eq!(
					native.to_dense(),
					via_channel.to_dense(),
					"log_dim={log_dim} log_inv_rate={log_inv_rate}"
				);
			}
		}
	}
}
