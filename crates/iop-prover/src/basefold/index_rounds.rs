// Copyright 2026 The Binius Developers

//! The oracle-index rounds of the batched BaseFold sumcheck, run on the per-round buffers.
//!
//! The batched sumcheck runs on `G(X, Z) = Σ_r G_r↑(X) · eq(r, Z)` and `T(X, Z) = Σ_r T_r↑(X) ·
//! eq(r, Z)`, and binds the oracle-index variables `Z` first. Materializing `G` and `T` costs
//! `2^(n_vars + log_n_oracles)` entries each. These rounds instead bind `Z` directly on the
//! per-round buffers: each round pairs oracle `2k` with oracle `2k + 1`, writes the round
//! polynomial from the pair's cross terms, and folds the pair at the challenge into one buffer.
//!
//! ## Lift and repeat
//!
//! A node holds a message `g` and a transparent `t` of `n` variables each, and stands for
//!
//! ```text
//! G↑(x) = g[(x mod 2^w) - (2^w - 2^n)]  if (x mod 2^w) >= 2^w - 2^n, else 0
//! T↑(x) = t[x - (2^n_vars - 2^n)]       if x >= 2^n_vars - 2^n,      else 0
//! ```
//!
//! over `x < 2^n_vars`, where `w >= n` is the node's `log_w`: the message sits in the top `2^n`
//! entries of each `2^w` block, and the block repeats; the transparent sits in the top `2^n`
//! entries of the whole space. Round oracles are committed reversed and the FRI folds with
//! complemented challenges, so the lift sits at the all-1 corner. For a round oracle
//! `w = n + log_lift`.
//!
//! The merge of a pair is `lo + α(lo + hi)` of the two lifted functions. The larger node (by `n`)
//! has the larger `w`, so the merge is again a node of the same form with `n` and `w` of the larger
//! one, and it is written in place over the larger node's buffers.

use std::iter;

use binius_alloc::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_ip_prover::sumcheck::round_evals::RoundEvals;
use binius_math::FieldBuffer;
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};

/// One oracle's (or merged group's) message and transparent, with the repeat period of the message.
pub struct IndexNode<P: PackedField, A: Allocator> {
	/// The message `g`, of `n` variables.
	pub message: FieldBuffer<P, A::Vec<P>>,
	/// The transparent `t`, of the same `n` variables.
	pub transparent: FieldBuffer<P, A::Vec<P>>,
	/// `w`, the log of the message's repeat period.
	pub log_w: usize,
}

impl<P: PackedField, A: Allocator> IndexNode<P, A> {
	const fn log_len(&self) -> usize {
		self.message.log_len()
	}
}

/// Which side of a pair is the larger node.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Big {
	Lo,
	Hi,
}

/// Binds the oracle-index variables of `nodes`, which are indexed by oracle with absent trailing
/// oracles `None`, and returns the single merged node.
///
/// `round` is called once per bound variable with that round's polynomial and returns the verifier
/// challenge. The sum claim is `claim`, and the claim after the last round is returned too.
pub fn bind_oracle_index_variables<P, A, F>(
	mut nodes: Vec<Option<IndexNode<P, A>>>,
	mut claim: F,
	mut round: impl FnMut(&RoundCoeffs<F>) -> F,
) -> (IndexNode<P, A>, F)
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert!(nodes.len().is_power_of_two());
	while nodes.len() > 1 {
		// Round polynomial: the sum over pairs of the pair's two sampled evaluations.
		let mut y_1 = F::ZERO;
		let mut y_inf = F::ZERO;
		for pair in nodes.chunks(2) {
			let (pair_1, pair_inf) = match (&pair[0], &pair[1]) {
				(Some(lo), Some(hi)) => cross_terms(lo, hi),
				(Some(lo), None) => (F::ZERO, inner_product(lo)),
				(None, None) => (F::ZERO, F::ZERO),
				(None, Some(_)) => unreachable!("absent oracles trail the present ones"),
			};
			y_1 += pair_1;
			y_inf += pair_inf;
		}

		let coeffs = RoundEvals::<F, 2>([y_1, y_inf]).interpolate(claim);
		let challenge = round(&coeffs);
		claim = coeffs.evaluate(&challenge);

		nodes = nodes
			.chunks_mut(2)
			.map(|pair| match (pair[0].take(), pair[1].take()) {
				(Some(lo), Some(hi)) => Some(fold_pair(lo, hi, challenge)),
				(Some(mut lo), None) => {
					// `lo + α(lo + 0)`
					let scale = P::broadcast(F::ONE + challenge);
					scale_in_place(&mut lo.message, scale);
					scale_in_place(&mut lo.transparent, scale);
					Some(lo)
				}
				(None, None) => None,
				(None, Some(_)) => unreachable!("absent oracles trail the present ones"),
			})
			.collect();
	}

	let node = nodes
		.pop()
		.flatten()
		.expect("the first oracle is always present");
	(node, claim)
}

fn scale_in_place<P: PackedField>(
	buffer: &mut FieldBuffer<P, impl std::ops::DerefMut<Target = [P]>>,
	scale: P,
) {
	buffer
		.as_mut()
		.par_iter_mut()
		.with_min_task(WorkPerItem::FieldMuls)
		.for_each(|word| *word *= scale);
}

/// `<g, t>` of one node.
fn inner_product<P: PackedField, A: Allocator>(node: &IndexNode<P, A>) -> P::Scalar {
	binius_math::inner_product::inner_product_par(&node.message, &node.transparent)
}

/// The geometry of merging a smaller node into a larger one.
struct Shape {
	n_big: usize,
	n_small: usize,
	/// `min(w_small, n_big)`: the repeat period of the small node's message, within `x < 2^n_big`.
	log_period: usize,
}

fn shape<P: PackedField, A: Allocator>(big: &IndexNode<P, A>, small: &IndexNode<P, A>) -> Shape {
	assert!(small.log_len() <= big.log_len());
	assert!(small.log_w <= big.log_w); // nested supports
	Shape {
		n_big: big.log_len(),
		n_small: small.log_len(),
		log_period: small.log_w.min(big.log_len()),
	}
}

const fn split<'a, P: PackedField, A: Allocator>(
	lo: &'a IndexNode<P, A>,
	hi: &'a IndexNode<P, A>,
) -> (Big, &'a IndexNode<P, A>, &'a IndexNode<P, A>) {
	if lo.log_len() >= hi.log_len() {
		(Big::Lo, lo, hi)
	} else {
		(Big::Hi, hi, lo)
	}
}

/// The small node's buffer as the words it occupies at the end of a block.
///
/// A node narrower than one packed word gets a single word, with the buffer in its top lanes and
/// zero below.
fn words_of<P: PackedField>(buffer: &FieldBuffer<P, impl std::ops::Deref<Target = [P]>>) -> Vec<P> {
	if buffer.log_len() >= P::LOG_WIDTH {
		buffer.as_ref()[..1 << (buffer.log_len() - P::LOG_WIDTH)].to_vec()
	} else {
		let n_zeros = P::WIDTH - buffer.len();
		vec![P::from_scalars(
			iter::repeat_n(P::Scalar::ZERO, n_zeros).chain(buffer.iter_scalars()),
		)]
	}
}

/// The sampled evaluations `(R(1), R(∞))` of the round polynomial of the pair `(lo, hi)`.
///
/// With the larger node `L` and the smaller `S`, the terms are `<L> = <gL, tL>`, `<S> = <gs, ts>`,
/// and the two cross terms `Σ_x gS↑(x) tL(x)` and `Σ_x gL(x) tS↑(x)`:
///
/// ```text
/// R(1) = <hi>        R(∞) = <L> + <S> + cross1 + cross2
/// ```
fn cross_terms<P, A, F>(lo: &IndexNode<P, A>, hi: &IndexNode<P, A>) -> (F, F)
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	let (big_side, big, small) = split(lo, hi);
	let shape = shape(big, small);

	let big_product = inner_product(big);
	let small_product = inner_product(small);

	let (cross_s_t_l, cross_l_t_s) = if shape.log_period >= P::LOG_WIDTH {
		let g_small = words_of(&small.message);
		let t_small = words_of(&small.transparent);
		let n_suffix = g_small.len();
		let period = 1usize << (shape.log_period - P::LOG_WIDTH);
		let t_big = &big.transparent.as_ref()[..1 << (shape.n_big - P::LOG_WIDTH)];
		let g_big = &big.message.as_ref()[(1 << (shape.n_big - P::LOG_WIDTH)) - n_suffix..];

		// `Σ_x gS↑(x) tL(x)`: the small message against the suffix of every period block of `tL`.
		let cross_s_t_l = t_big
			.par_chunks(period)
			.map(|block| {
				iter_sum::<P>(
					g_small
						.iter()
						.zip(&block[period - n_suffix..])
						.map(|(&a, &b)| a * b),
				)
			})
			.sum::<F>();
		// `Σ_x gL(x) tS↑(x)`: the suffix of `gL` against the small transparent.
		let cross_l_t_s = iter_sum::<P>(g_big.iter().zip(&t_small).map(|(&a, &b)| a * b));
		(cross_s_t_l, cross_l_t_s)
	} else {
		// A period narrower than a packed word, so only tiny buffers: go scalar by scalar.
		let mut cross_s_t_l = F::ZERO;
		let mut cross_l_t_s = F::ZERO;
		let period_mask = (1usize << shape.log_period) - 1;
		let message_start = (1usize << shape.log_period) - (1 << shape.n_small);
		let transparent_start = (1usize << shape.n_big) - (1 << shape.n_small);
		for x in 0..1usize << shape.n_big {
			let position = x & period_mask;
			if position >= message_start {
				cross_s_t_l += small.message.get(position - message_start) * big.transparent.get(x);
			}
			if x >= transparent_start {
				cross_l_t_s += big.message.get(x) * small.transparent.get(x - transparent_start);
			}
		}
		(cross_s_t_l, cross_l_t_s)
	};

	let y_1 = match big_side {
		Big::Lo => small_product, // `hi` is the small node
		Big::Hi => big_product,
	};
	(y_1, big_product + small_product + cross_s_t_l + cross_l_t_s)
}

fn iter_sum<P: PackedField>(words: impl Iterator<Item = P>) -> P::Scalar {
	words.fold(P::zero(), |acc, word| acc + word).iter().sum()
}

/// Folds `lo + α(lo + hi)` of the pair in place into the larger node's buffers.
fn fold_pair<P, A, F>(lo: IndexNode<P, A>, hi: IndexNode<P, A>, alpha: F) -> IndexNode<P, A>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	let (big_side, mut big, small) = if lo.log_len() >= hi.log_len() {
		(Big::Lo, lo, hi)
	} else {
		(Big::Hi, hi, lo)
	};
	let shape = shape(&big, &small);

	// The result is `c_big · big + c_small · small↑`.
	let (c_big, c_small) = match big_side {
		Big::Lo => (F::ONE + alpha, alpha),
		Big::Hi => (alpha, F::ONE + alpha),
	};

	if shape.log_period >= P::LOG_WIDTH {
		let period = 1usize << (shape.log_period - P::LOG_WIDTH);
		let g_small = words_of(&small.message);
		let t_small = words_of(&small.transparent);
		let n_suffix = g_small.len();
		let (c_big, c_small) = (P::broadcast(c_big), P::broadcast(c_small));

		// Each period block of the message holds the small message in its high words.
		big.message
			.as_mut()
			.par_chunks_mut(period)
			.for_each(|block| {
				let (rest, suffix) = block.split_at_mut(period - n_suffix);
				for word in rest {
					*word *= c_big;
				}
				for (word, &small_word) in suffix.iter_mut().zip(&g_small) {
					*word = c_big * *word + c_small * small_word;
				}
			});
		// The transparent holds it once, in the high words.
		let n_words = big.transparent.as_ref().len();
		let (rest, suffix) = big.transparent.as_mut().split_at_mut(n_words - n_suffix);
		for (word, &small_word) in suffix.iter_mut().zip(&t_small) {
			*word = c_big * *word + c_small * small_word;
		}
		rest.par_iter_mut()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|word| *word *= c_big);
	} else {
		let period_mask = (1usize << shape.log_period) - 1;
		let message_start = (1usize << shape.log_period) - (1 << shape.n_small);
		let transparent_start = (1usize << shape.n_big) - (1 << shape.n_small);
		for x in 0..1usize << shape.n_big {
			let position = x & period_mask;
			let small_g = if position >= message_start {
				small.message.get(position - message_start)
			} else {
				F::ZERO
			};
			let small_t = if x >= transparent_start {
				small.transparent.get(x - transparent_start)
			} else {
				F::ZERO
			};
			big.message
				.set(x, c_big * big.message.get(x) + c_small * small_g);
			big.transparent
				.set(x, c_big * big.transparent.get(x) + c_small * small_t);
		}
	}
	big
}

#[cfg(test)]
mod tests {
	use binius_alloc::GlobalAllocator;
	use binius_field::{Ghash128b, PackedGhash1x128b, PackedGhash4x128b, Random};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	const N_VARS: usize = 5;

	type F = Ghash128b;

	/// `G↑` of a message of `n` variables with repeat period `2^w`, over `x < 2^N_VARS`.
	fn lift_message(g: &[F], w: usize) -> Vec<F> {
		let start = (1usize << w) - g.len();
		(0..1usize << N_VARS)
			.map(|x| {
				let position = x & ((1 << w) - 1);
				if position >= start {
					g[position - start]
				} else {
					F::ZERO
				}
			})
			.collect()
	}

	/// `T↑` of a transparent of `n` variables, over `x < 2^N_VARS`.
	fn lift_transparent(t: &[F]) -> Vec<F> {
		let start = (1usize << N_VARS) - t.len();
		(0..1usize << N_VARS)
			.map(|x| if x >= start { t[x - start] } else { F::ZERO })
			.collect()
	}

	fn dot(a: &[F], b: &[F]) -> F {
		iter::zip(a, b).map(|(&a, &b)| a * b).sum()
	}

	/// `lo + α(lo + hi)` entrywise.
	fn fold(lo: &[F], hi: &[F], alpha: F) -> Vec<F> {
		iter::zip(lo, hi)
			.map(|(&lo, &hi)| lo + alpha * (lo + hi))
			.collect()
	}

	/// Runs the index rounds on nodes `(n, w)` and checks them against materialized `G(x, Z)` and
	/// `T(x, Z)` bound over `Z` low bit first.
	fn check<P: PackedField<Scalar = F>>(specs: &[(usize, usize)], seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let random = |rng: &mut StdRng, n: usize| -> Vec<F> {
			(0..1usize << n).map(|_| F::random(&mut *rng)).collect()
		};

		let mut nodes = Vec::new();
		// Materialized `G_r↑`, `T_r↑` per oracle index, with absent oracles zero.
		let mut g_ref = Vec::new();
		let mut t_ref = Vec::new();
		for &(n, w) in specs {
			let (g, t) = (random(&mut rng, n), random(&mut rng, n));
			g_ref.push(lift_message(&g, w));
			t_ref.push(lift_transparent(&t));
			nodes.push(Some(IndexNode::<P, GlobalAllocator> {
				message: FieldBuffer::from_values(&g),
				transparent: FieldBuffer::from_values(&t),
				log_w: w,
			}));
		}
		let n_oracles = specs.len().next_power_of_two();
		nodes.resize_with(n_oracles, || None);
		g_ref.resize(n_oracles, vec![F::ZERO; 1 << N_VARS]);
		t_ref.resize(n_oracles, vec![F::ZERO; 1 << N_VARS]);

		let claim = iter::zip(&g_ref, &t_ref).map(|(g, t)| dot(g, t)).sum::<F>();

		// Reference sumcheck over the oracle index, low bit first.
		let mut expected_rounds = Vec::new();
		let mut challenges = Vec::new();
		let mut expected_claim = claim;
		while g_ref.len() > 1 {
			let (mut r0, mut r1, mut r_inf) = (F::ZERO, F::ZERO, F::ZERO);
			for (g, t) in iter::zip(g_ref.chunks(2), t_ref.chunks(2)) {
				r0 += dot(&g[0], &t[0]);
				r1 += dot(&g[1], &t[1]);
				let sum =
					|a: &[F], b: &[F]| iter::zip(a, b).map(|(&a, &b)| a + b).collect::<Vec<_>>();
				r_inf += dot(&sum(&g[0], &g[1]), &sum(&t[0], &t[1]));
			}
			// R(X) = r0 + (r0 + r1 + r_inf) X + r_inf X^2 in characteristic 2.
			let coeffs = RoundCoeffs(vec![r0, r0 + r1 + r_inf, r_inf]);
			let alpha = F::random(&mut rng);
			expected_claim = coeffs.evaluate(&alpha);
			g_ref = g_ref
				.chunks(2)
				.map(|pair| fold(&pair[0], &pair[1], alpha))
				.collect();
			t_ref = t_ref
				.chunks(2)
				.map(|pair| fold(&pair[0], &pair[1], alpha))
				.collect();
			expected_rounds.push(coeffs);
			challenges.push(alpha);
		}

		let mut rounds = Vec::new();
		let mut challenges_iter = challenges.iter();
		let (node, final_claim) = bind_oracle_index_variables(nodes, claim, |coeffs| {
			rounds.push(coeffs.clone());
			*challenges_iter.next().expect("one challenge per round")
		});

		assert_eq!(rounds, expected_rounds);
		assert_eq!(final_claim, expected_claim);
		let g_node = node.message.iter_scalars().collect::<Vec<_>>();
		let t_node = node.transparent.iter_scalars().collect::<Vec<_>>();
		assert_eq!(lift_message(&g_node, node.log_w), g_ref[0]);
		assert_eq!(lift_transparent(&t_node), t_ref[0]);
		assert_eq!(dot(&g_ref[0], &t_ref[0]), final_claim);
	}

	/// The cases keep the nesting the merge relies on: a smaller node either has the larger node's
	/// period or the larger node's period is its own length.
	fn check_all<P: PackedField<Scalar = F>>() {
		let cases: &[&[(usize, usize)]] = &[
			// n_small == n_big
			&[(3, 3), (3, 3)],
			&[(3, 4), (3, 4)],
			// w_small < n_big, larger node on either side
			&[(2, 3), (4, 4)],
			&[(4, 4), (2, 3)],
			// w_small >= n_big
			&[(2, 5), (4, 5)],
			// a node with no variables
			&[(0, 0), (3, 3)],
			// a padded trailing oracle
			&[(4, 4), (3, 4), (1, 3)],
			&[(1, 1), (3, 3), (4, 4), (2, 3)],
		];
		for (seed, specs) in cases.iter().enumerate() {
			check::<P>(specs, seed as u64);
		}
	}

	#[test]
	fn bind_matches_materialized_reference_1x128b() {
		check_all::<PackedGhash1x128b>();
	}

	#[test]
	fn bind_matches_materialized_reference_4x128b() {
		check_all::<PackedGhash4x128b>();
	}
}
