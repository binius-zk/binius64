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
//! entries of the whole space. (Round oracles are committed reversed and the FRI folds with
//! complemented challenges, which moves the lift from the all-0 corner to the all-1 corner.) For a
//! round oracle `w = n + log_lift`.
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
