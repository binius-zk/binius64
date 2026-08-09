// Copyright 2025-2026 The Binius Developers

use std::iter;

use binius_compute::{Allocator, VecLike};
use binius_field::{Field, PackedField};
use binius_ip::{mlecheck, prodcheck::MultilinearEvalClaim, sumcheck::RoundCoeffs};
use binius_math::{
	FieldBuffer, FieldSlice, FieldVec,
	line::extrapolate_line_packed,
	multilinear::eq::{eq_ind_partial_eval, eq_one_var},
};
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};
use itertools::izip;

use crate::{
	channel::IPProverChannel,
	sumcheck::{
		ProveSingleOutput,
		bivariate_product_mle::{self, LayerProver},
		common::MleCheckProver,
		mle_store::ColId,
		prove_single_mlecheck,
	},
};

/// Witness-based prover for the product check protocol.
///
/// This prover reduces the claim that a multilinear polynomial evaluates to a product over a
/// Boolean hypercube to a single multilinear evaluation claim.
pub struct ProdcheckProver<'a, A: Allocator, P: PackedField> {
	/// Product layers from largest (original witness) to second-smallest.
	/// `layers[0]` is the original witness. The final products layer is returned
	/// separately from the constructor.
	layers: Vec<FieldVec<P, A>>,
	/// Allocator the product layers are drawn from.
	alloc: &'a A,
}

// A manual `Clone` impl (rather than `#[derive(Clone)]`) so the bound lands on the layer buffer
// `A::Vec<P>` rather than on `A` and `P`: a derive would require `A: Clone` yet still fail to clone
// the layers unless `A::Vec<P>: Clone`. Holds for the `Vec`-backed `GlobalAllocator`.
impl<A: Allocator, P: PackedField> Clone for ProdcheckProver<'_, A, P>
where
	A::Vec<P>: Clone,
{
	fn clone(&self) -> Self {
		Self {
			layers: self.layers.clone(),
			alloc: self.alloc,
		}
	}
}

impl<'a, A, F, P> ProdcheckProver<'a, A, P>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	/// Creates a new [`ProdcheckProver`].
	///
	/// Returns `(prover, products)` where `products` is the final layer containing the
	/// products over all `k` variables.
	///
	/// # Arguments
	/// * `k` - The number of variables over which the product is taken. Each reduction step reduces
	///   one variable by computing pairwise products.
	/// * `witness` - The witness polynomial
	///
	/// # Preconditions
	/// * `witness.log_len() >= k`
	pub fn new(k: usize, alloc: &'a A, witness: FieldVec<P, A>) -> (Self, FieldVec<P, A>) {
		assert!(witness.log_len() >= k); // precondition

		let mut layers = Vec::with_capacity(k + 1);
		layers.push(witness);

		for _ in 0..k {
			let prev_layer = layers.last().expect("layers is non-empty");
			let next_log_len = prev_layer.log_len() - 1;
			let (half_0, half_1) = prev_layer.split_half_ref();

			// Each layer is half the width of the one below it, down to a single word.
			// The last layers are too small to be worth splitting.
			let next_layer_evals: Vec<P> = (half_0.as_ref(), half_1.as_ref())
				.into_par_iter()
				.with_min_task(WorkPerItem::FieldMuls)
				.map(|(v0, v1)| *v0 * *v1)
				.collect();
			let mut next_data = alloc.alloc::<P>(next_layer_evals.len());
			next_data.extend_from_slice(&next_layer_evals);
			let next_layer = FieldBuffer::new(next_log_len, next_data);

			layers.push(next_layer);
		}

		let products = layers.pop().expect("layers has k+1 elements");
		(Self { layers, alloc }, products)
	}

	/// Returns the number of remaining layers to prove.
	pub const fn n_layers(&self) -> usize {
		self.layers.len()
	}

	/// Pops the narrowest remaining layer and returns it.
	///
	/// Reductions run from the root down, so this is the next layer to reduce.
	///
	/// # Preconditions
	/// * `self.n_layers() >= 1`
	pub fn pop_layer(&mut self) -> FieldVec<P, A> {
		self.layers
			.pop()
			.expect("precondition: layers is non-empty")
	}

	/// Recomputes the tree's root layer, the elementwise product of the narrowest remaining
	/// layer's two halves.
	///
	/// A batch reducing trees of unequal depths needs a shallow tree's root at each layer it
	/// spends waiting for the batch to come down to its depth, where the tree stands in as the
	/// one-padding of that root. Recomputing it costs one pass over the root's own length rather
	/// than a copy retained for the whole reduction.
	///
	/// # Preconditions
	/// * `self.n_layers() >= 1`
	pub fn root_layer(&self) -> FieldVec<P, A> {
		let layer = self
			.layers
			.last()
			.expect("precondition: layers is non-empty");
		let (half_0, half_1) = layer.split_half_ref();

		let products: Vec<P> = (half_0.as_ref(), half_1.as_ref())
			.into_par_iter()
			.with_min_task(WorkPerItem::FieldMuls)
			.map(|(v0, v1)| *v0 * *v1)
			.collect();
		let mut data = self.alloc.alloc::<P>(products.len());
		data.extend_from_slice(&products);
		FieldBuffer::new(layer.log_len() - 1, data)
	}

	/// Runs the product check protocol and returns the final evaluation claim.
	///
	/// This consumes the prover and runs sumcheck reductions from the smallest layer back to
	/// the largest.
	///
	/// # Arguments
	/// * `claim` - The initial multilinear evaluation claim
	/// * `channel` - The channel for sending prover messages and sampling challenges
	///
	/// # Preconditions
	/// * `claim.point.len() == witness.log_len() - k` (where k is the number of reduction layers)
	pub fn prove(
		mut self,
		mut claim: MultilinearEvalClaim<F>,
		channel: &mut impl IPProverChannel<F>,
	) -> MultilinearEvalClaim<F> {
		let alloc = self.alloc;

		while self.n_layers() > 0 {
			let layer = self.pop_layer();
			let MultilinearEvalClaim { eval, point } = claim;

			// The layer has one more variable than the claim point.
			// Its low and high halves are the two multilinears whose product this layer reduces.
			// Sharing the buffer between the halves avoids copying this (largest) layer.
			let (mle_prover, _children) =
				bivariate_product_mle::new_split_half(alloc, layer, point, eval);

			let ProveSingleOutput {
				multilinear_evals,
				challenges,
			} = prove_single_mlecheck(mle_prover, channel);

			let [eval_0, eval_1] = multilinear_evals
				.try_into()
				.expect("prover has two multilinears");

			channel.send_many(&[eval_0, eval_1]);

			let r = channel.sample();
			let next_eval = extrapolate_line_packed(eval_0, eval_1, r);

			let mut next_point = challenges;
			next_point.reverse();
			next_point.push(r);

			claim = MultilinearEvalClaim {
				eval: next_eval,
				point: next_point,
			};
		}

		claim
	}
}

/// Output of [`batch_prove`].
///
/// After the full `n_layers` reduction, `evals` holds each input prover's reduced evaluation at
/// `eval_point`. The batched claim the verifier checks is the eq(selector)-weighted combination
/// of these evaluations.
pub struct BatchProveOutput<F> {
	/// The reduced evaluation point (`selector ++ root ++ node`) shared by all input provers.
	pub eval_point: Vec<F>,
	/// Each input prover's reduced evaluation of its own witness, in input order.
	///
	/// A prover shallower than the batch is reduced over the one-padding of its witness, and its
	/// entry is the evaluation of the prover's *own* witness — at `eval_point` with the leading
	/// selector coordinates and that prover's own padding coordinates dropped. See
	/// [`batch_prove`].
	pub evals: Vec<F>,
}

/// Runs a batched product check protocol for multiple independent prodcheck provers.
///
/// This combines $N$ provers, each a product tree over a shared root domain, using multilinear
/// interpolation over $\kappa$ selector variables (where $N \le 2^\kappa$). The combined claim is
/// the multilinear extrapolation of the individual claimed products (padded with zeros to
/// $2^\kappa$) evaluated at the given point.
///
/// Prover $u$ has depth $k_u$, its `n_layers()`, and a root layer over the $\ell$ *root* variables
/// shared by every prover — the layers' passive dimension, which the reduction carries along and
/// never aggregates. `claimed_products` holds each root layer's evaluation at the shared
/// `root_point`. When the roots are scalars (each prover reduces over all of its variables),
/// `root_point` is empty.
///
/// # Arguments
/// * `provers` - Vec of $N$ prodcheck provers. The batch depth $n$ is the largest `n_layers()`
///   among them.
/// * `claimed_products` - Vec of $N$ claimed product values, one per prover. Each is the
///   corresponding prover's root layer evaluated at `root_point`.
/// * `selector_point` - Evaluation point for the selector variables. Length is $\kappa$.
/// * `root_point` - Shared evaluation point at which the claimed products are taken. Length is the
///   root-layer dimension $\ell$ (i.e. `witness.log_len() - n_layers`). Empty for scalar roots.
/// * `channel` - The channel for sending prover messages and sampling challenges.
///
/// # Preconditions
/// * `provers` must be non-empty, and each must have at least one layer.
/// * `2^selector_point.len() >= provers.len()`.
/// * `claimed_products.len() == provers.len()`.
/// * `root_point.len() == witness.log_len() - n_layers` for each prover.
///
/// The batched claim is checked by the ordinary `binius_ip::prodcheck::verify` recursion over $n$
/// layers (the eq(selector)-weighted combination of the returned evaluations), with the selector
/// coordinates forming the first $\kappa$ coordinates of the claim point.
///
/// # Returns
/// A [`BatchProveOutput`] with the reduced `eval_point` and each input prover's reduced eval.
///
/// # Trees of Unequal Depth
///
/// The provers need not share a depth. Each tree shallower than the deepest is proved as a product
/// check over the one-padding of its leaves — the same leaves with constant-1 positions filling the
/// extra depth, which leaves the tree's products unchanged. The transcript is then exactly that of
/// an equal-depth batch of depth $n$: the verifier runs the ordinary
/// `binius_ip::prodcheck::verify` over $n$ layers and never learns the individual depths.
///
/// The prover materializes neither the padded leaves nor their layers; each layer's messages are
/// corrected instead, in the three segments `PaddedLayerProver` passes through. The protocol, the
/// round polynomials, and the $O(n \cdot 2^\ell)$ padding overhead are specified in the *Batched
/// Product Checks of Unequal Depths* appendix of the Binius64 whitepaper.
///
/// The verifier's reduced claim is on the padded leaves, while [`BatchProveOutput::evals`] holds
/// claims on the trees' own leaves. Writing $\rho$ for the reduced point past the selector and
/// root coordinates, a tree padded by $\eta$ coordinates relates the two by
///
/// $$
/// L'(\tau, \rho) = 1 + \bigl( L(\tau, \rho_{\geq \eta}) - 1 \bigr) \cdot \textsf{eq}(0^\eta;
/// \rho_{< \eta}).
/// $$
///
/// # Mathematical Description
///
/// Let $f_u \in K[X_0, \ldots, X_{m-1}]$ be multilinear for all $u \in \{0, \ldots, N - 1\}$. The
/// $u$'th prover is a prodcheck prover for $f_u$. Let $p_u \in K$ be the claimed hypercube product
/// of $f_u$.
///
/// Let $y \in K^\kappa$ be the evaluation point. The prover is proving a claim that
///
/// $$
/// \sum_{u \in B_\kappa} \textsf{eq}(u; y) \prod_{j \in B_m} f_u(j) = \sum_{u \in B_\kappa}
/// \textsf{eq}(u; y) p_u, $$
///
/// reducing to an evaluation of the interpolated multilinear
///
/// $$
/// \hat{f}(Y_0, \ldots, Y_{\kappa-1}, X_0, \ldots, X_{m-1}) = \sum_{u \in B_\kappa}
/// \textsf{eq}(u; Y) f_u(X).
/// $$
pub fn batch_prove<'a, A: Allocator, F: Field, P: PackedField<Scalar = F>>(
	mut provers: Vec<ProdcheckProver<'a, A, P>>,
	claimed_products: Vec<F>,
	selector_point: Vec<F>,
	root_point: Vec<F>,
	channel: &mut impl IPProverChannel<F>,
) -> BatchProveOutput<F> {
	assert!(!provers.is_empty()); // precondition
	assert_eq!(claimed_products.len(), provers.len()); // precondition

	let k = selector_point.len();
	assert!(provers.len() <= (1 << k)); // precondition
	assert!(provers.iter().all(|prover| prover.n_layers() >= 1)); // precondition

	let n_layers = provers
		.iter()
		.map(ProdcheckProver::n_layers)
		.max()
		.expect("provers is non-empty");

	// How much depth each tree is padded by.
	let pad_lens = provers
		.iter()
		.map(|prover| n_layers - prover.n_layers())
		.collect::<Vec<_>>();

	// The point starts as `selector ++ root`; each layer appends one node coordinate at the end.
	let n_root = root_point.len();
	let mut eval_point = [selector_point, root_point].concat();
	let mut claims = claimed_products;

	for _ in 0..n_layers {
		let (next_claims, next_point) =
			prove_layer_rounds(&mut provers, &pad_lens, &claims, &eval_point, k, n_root, channel);
		claims = next_claims;
		eval_point = next_point;
	}
	debug_assert!(
		provers.iter().all(|prover| prover.n_layers() == 0),
		"every tree's layers are consumed by the batch depth"
	);

	BatchProveOutput {
		eval_point,
		evals: claims,
	}
}

/// The per-layer context every tree's [`PaddedLayerProver`] reads.
struct LayerContext<'alloc, 'layer, A, F: Field> {
	alloc: &'alloc A,
	/// The claim point's root segment.
	root_point: &'layer [F],
	/// The equality-indicator expansion of `root_point`, shared across trees.
	root_eq: &'layer FieldBuffer<F>,
}

/// One tree's prover for a single layer of a batched product check.
///
/// A tree shallower than the batch is reduced as if its leaves were one-padded: the same leaves
/// with constant-1 positions filling the extra depth. Its layer at the batch's current depth is
/// then the one-padding of one of its own layers — or, before the batch has come down to the tree's
/// depth at all, of its root layer. The padded layer's claim point splits into three segments,
///
/// ```text
///     [ root (l) | padding (mu) | aggregating (m) ]
/// ```
///
/// and MLE-check binds variables from the highest index down, so the tree passes through them in
/// reverse, one enum variant each. The padded layer is never materialized.
///
/// Each variant's round polynomial is stated before the padding correction
/// $R'(X) = w R(X) + (1 - w)$ that [`prove_layer_rounds`] applies, where $w$ is the equality weight
/// of the padding coordinates still unbound. Off the all-zeros padding slab both children of a
/// padded layer are one, which is where the residual $1 - w$ comes from.
enum PaddedLayerProver<'a, A: Allocator, F: Field, P: PackedField<Scalar = F>> {
	/// Aggregating rounds: an ordinary bivariate-product MLE-check over the tree's own layer.
	///
	/// An unpadded tree stays here for its root rounds too, since then the padding segment is
	/// empty and the correction is the identity.
	Aggregating {
		inner: LayerProver<'a, A, F, P>,
		/// Store ids of the layer's two children, read out at the end of the segment.
		children: [ColId; 2],
	},
	/// Padding rounds: every variable of the tree's own layer is bound, leaving its two children as
	/// fixed tables over the root variables.
	///
	/// Summing those tables against their equality weights collapses the round polynomial to the
	/// closed form $R(X) = 1 + (S - 2) E(X) + Q E(X)^2$ in the two moments below, where
	/// $E(X) = \textsf{eq}(0, X) \cdot$ `bound_eq` is the equality weight of the padding
	/// coordinates already bound together with this round's.
	Padding {
		children: [FieldVec<P, A>; 2],
		/// The equality-weighted moments $S = \langle g_0 + g_1 \rangle$ and $Q = \langle (g_0 -
		/// 1) (g_1 - 1) \rangle$ of the children over the root variables.
		moments: [F; 2],
		/// $\prod \textsf{eq}(0, r)$ over the padding challenges bound so far.
		bound_eq: F,
	},
	/// Root rounds: an MLE-check over the tree's own children whose composition carries the
	/// one-padding at the fully bound padding weight, so no further correction applies and the
	/// reduction still emits the children's *own* evaluations.
	Root {
		inner: LayerProver<'a, A, F, P>,
		/// $\prod \textsf{eq}(0, r)$ over every padding challenge.
		pad_eq: F,
	},
}

impl<'a, A: Allocator, F: Field, P: PackedField<Scalar = F>> PaddedLayerProver<'a, A, F, P> {
	/// Creates the prover for a layer of a tree the batch has come down to.
	///
	/// # Arguments
	///
	/// * `layer` - The tree's own layer, whose low and high halves on its highest variable are the
	///   two multilinears whose product this layer reduces.
	/// * `pad_len` - The length of the claim point's padding segment.
	/// * `node_point` - The claim point's node coordinates, the padding segment followed by the
	///   aggregating one.
	/// * `claim` - The tree's own claim on `layer`, at the root segment followed by the aggregating
	///   one.
	fn new(
		ctx: &LayerContext<'a, '_, A, F>,
		layer: FieldVec<P, A>,
		pad_len: usize,
		node_point: &[F],
		claim: F,
	) -> Self {
		// The unpadded reduction skips the padding segment: those variables are not the tree's own,
		// and they stay unbound while it reduces the aggregating ones.
		let inner_point = [ctx.root_point, &node_point[pad_len..]].concat();
		let (inner, children) =
			bivariate_product_mle::new_split_half(ctx.alloc, layer, inner_point, claim);

		Self::Aggregating { inner, children }.advance(ctx, pad_len, node_point.len())
	}

	/// Creates the prover for a layer of a tree the batch has not come down to yet.
	///
	/// Such a layer is the one-padding of the tree's root, whose two children are that root and the
	/// constant one. The moments then need no pass over the tables: $S$ is the running claim plus
	/// one, and $Q$ vanishes because the high child is identically one.
	fn unreached(
		ctx: &LayerContext<'a, '_, A, F>,
		root: FieldVec<P, A>,
		node_point: &[F],
		claim: F,
	) -> Self {
		// Every node variable of such a layer is a padding one.
		let ones = ones_in(ctx.alloc, root.log_len());
		Self::Padding {
			children: [root, ones],
			moments: [claim + F::ONE, F::ZERO],
			bound_eq: F::ONE,
		}
		.advance(ctx, node_point.len(), node_point.len())
	}

	/// Moves past a segment whose last variable is bound, if this round bound it.
	///
	/// The aggregating segment gives way to the padding one once the tree's own layer is fully
	/// reduced, which reads the layer's two children out of the store; the padding segment gives
	/// way to the root rounds once its last variable is bound, which is where the padded children's
	/// claim is formed.
	fn advance(
		self,
		ctx: &LayerContext<'a, '_, A, F>,
		pad_len: usize,
		node_vars_left: usize,
	) -> Self {
		let state = match self {
			// The inner reduction is down to the root variables, which the padding rounds precede.
			Self::Aggregating {
				mut inner,
				children,
			} if pad_len > 0 && inner.n_vars() == ctx.root_point.len() => {
				let children: [FieldVec<P, A>; 2] = children
					.map(|id| FieldBuffer::clone_from_slice(ctx.alloc, inner.store().column(id)));
				let moments =
					child_moments(children.each_ref().map(|child| child.to_ref()), ctx.root_eq);
				Self::Padding {
					children,
					moments,
					bound_eq: F::ONE,
				}
			}
			state => state,
		};

		match state {
			Self::Padding {
				children,
				moments,
				bound_eq,
			} if node_vars_left == 0 => Self::Root {
				// The padded children's claim at the root point is the closed form of the last
				// padding round evaluated at its challenge.
				inner: bivariate_product_mle::new_one_padded(
					ctx.alloc,
					children,
					bound_eq,
					ctx.root_point.to_vec(),
					padded_moment(moments, bound_eq),
				),
				pad_eq: bound_eq,
			},
			state => state,
		}
	}

	/// This round's polynomial for the tree, before the padding correction.
	fn execute(&mut self) -> RoundCoeffs<F> {
		match self {
			Self::Aggregating { inner, .. } | Self::Root { inner, .. } => {
				let mut round_coeffs = inner.execute();
				assert_eq!(round_coeffs.len(), 1, "the layer prover carries one claim");
				round_coeffs.pop().expect("the vector holds one element")
			}
			Self::Padding {
				moments, bound_eq, ..
			} => padding_round_poly(*moments, *bound_eq),
		}
	}

	/// Binds this round's variable to the verifier challenge.
	///
	/// `node_vars_left` is the number of the layer's node variables still unbound afterwards, which
	/// is zero exactly on the round that ends the padding segment.
	fn fold(
		self,
		challenge: F,
		ctx: &LayerContext<'a, '_, A, F>,
		pad_len: usize,
		node_vars_left: usize,
	) -> Self {
		let state = match self {
			Self::Aggregating {
				mut inner,
				children,
			} => {
				inner.fold(challenge);
				Self::Aggregating { inner, children }
			}
			// No multilinear is touched: binding a padding variable only sharpens the equality
			// weight the closed form carries.
			Self::Padding {
				children,
				moments,
				bound_eq,
			} => Self::Padding {
				children,
				moments,
				bound_eq: bound_eq * eq_one_var(F::ZERO, challenge),
			},
			Self::Root { mut inner, pad_eq } => {
				inner.fold(challenge);
				Self::Root { inner, pad_eq }
			}
		};
		state.advance(ctx, pad_len, node_vars_left)
	}

	/// The tree's own child evaluations and the padded layer's, once every variable is bound.
	fn finish(self) -> ([F; 2], [F; 2]) {
		let (inner, pad_eq) = match self {
			Self::Aggregating { inner, .. } => (inner, F::ONE),
			Self::Root { inner, pad_eq } => (inner, pad_eq),
			Self::Padding { .. } => panic!("finish requires every variable to be bound"),
		};
		let children: [F; 2] = inner
			.finish()
			.try_into()
			.expect("the layer prover reduces two multilinears");
		(children, children.map(|child| select(pad_eq, child)))
	}
}

/// The one-padding selector $\textsf{sel}(s, v) = 1 + (v - 1) s$.
///
/// It interpolates between the constant one at $s = 0$ and $v$ at $s = 1$, which is how a padded
/// leaf position holds a one while a real one holds the tree's own value.
fn select<F: Field>(s: F, v: F) -> F {
	F::ONE + (v - F::ONE) * s
}

/// A layer's two children pushed through the one-padding at selector value `s`, multiplied and
/// summed against the root point's equality weights: $1 + (S - 2) s + Q s^2$.
///
/// `moments` are the $S$ and $Q$ of [`PaddedLayerProver::Padding`].
fn padded_moment<F: Field>([sum, product]: [F; 2], s: F) -> F {
	F::ONE + (sum - F::ONE - F::ONE) * s + product * s * s
}

/// A padding round's polynomial: [`padded_moment`] composed with $E(X) = \textsf{eq}(0, X) \cdot$
/// `bound_eq`, in monomial coefficients.
///
/// $E$ weights the padding coordinates bound so far together with this round's, so the polynomial
/// at the round's challenge is [`padded_moment`] at the updated `bound_eq` — the claim the root
/// rounds open with once the last padding variable is bound.
fn padding_round_poly<F: Field>([sum, product]: [F; 2], bound_eq: F) -> RoundCoeffs<F> {
	// E in monomial coefficients.
	let [e_0, e_1] = [bound_eq, -bound_eq];
	let sum = sum - F::ONE - F::ONE;
	RoundCoeffs(vec![
		F::ONE + sum * e_0 + product * e_0 * e_0,
		sum * e_1 + product * (e_0 * e_1 + e_0 * e_1),
		product * e_1 * e_1,
	])
}

/// The equality-weighted moments $\langle g_0 + g_1 \rangle$ and $\langle (g_0 - 1)(g_1 - 1)
/// \rangle$ of a layer's two children over the root variables.
fn child_moments<F: Field, P: PackedField<Scalar = F>>(
	children: [FieldSlice<'_, P>; 2],
	root_eq: &FieldBuffer<F>,
) -> [F; 2] {
	let [child_0, child_1] = children;
	izip!(root_eq.iter_scalars(), child_0.iter_scalars(), child_1.iter_scalars()).fold(
		[F::ZERO; 2],
		|[sum, product], (weight, g_0, g_1)| {
			[
				sum + weight * (g_0 + g_1),
				product + weight * (g_0 - F::ONE) * (g_1 - F::ONE),
			]
		},
	)
}

/// A buffer of `2^log_len` ones, drawn from `alloc`.
fn ones_in<A: Allocator, F: Field, P: PackedField<Scalar = F>>(
	alloc: &A,
	log_len: usize,
) -> FieldVec<P, A> {
	let mut buffer = FieldBuffer::zeros_in(alloc, log_len);
	for index in 0..1 << log_len {
		buffer.set(index, F::ONE);
	}
	buffer
}

/// Runs one batched layer reduction.
///
/// The per-tree reductions run in lockstep, their round polynomials combined with the eq(selector)
/// weights, followed by the shared selector rounds over the per-tree child evaluation pairs. A tree
/// shallower than the batch is reduced over the one-padding of its layer; `pad_lens` gives each
/// tree's padding depth and [`PaddedLayerProver`] carries the correction. A tree the batch has not
/// come down to keeps all of its layers, so `provers` is borrowed rather than consumed.
///
/// # Returns
///
/// Each tree's claim on its own next layer, in input order, and the reduced evaluation point.
fn prove_layer_rounds<'a, A: Allocator, F: Field, P: PackedField<Scalar = F>>(
	provers: &mut [ProdcheckProver<'a, A, P>],
	pad_lens: &[usize],
	claims: &[F],
	eval_point: &[F],
	k: usize,
	n_root: usize,
	channel: &mut impl IPProverChannel<F>,
) -> (Vec<F>, Vec<F>) {
	let alloc = provers[0].alloc;
	// The point is `selector ++ root ++ node`, the node segment growing by one coordinate a layer.
	let (selector_coords, inner_coords) = eval_point.split_at(k);
	let (root_coords, node_coords) = inner_coords.split_at(n_root);
	let n_node = node_coords.len();

	// Compute eq weights for batching: eq(u, selector_coords) for all u in B_k.
	let eq_weights = eq_ind_partial_eval::<F>(selector_coords);
	let root_eq = eq_ind_partial_eval::<F>(root_coords);
	let ctx = LayerContext {
		alloc,
		root_point: root_coords,
		root_eq: &root_eq,
	};

	// Each tree's padding segment at this layer. It grows with the node segment until the batch
	// comes down to the tree's own depth, after which it is the tree's full padding depth.
	let segment_lens = pad_lens
		.iter()
		.map(|&pad_len| pad_len.min(n_node))
		.collect::<Vec<_>>();

	// Prefix products of eq(0, .) over the lowest node coordinates, which are the padding
	// coordinates of the trees shallower than the batch. Entry `c` is the equality weight of the
	// padding coordinates strictly below index `c`, so a tree reads its correction weight for the
	// round leaving `node_vars_left` node variables unbound at `node_vars_left.min(pad_len)`.
	let max_pad_len = segment_lens
		.iter()
		.copied()
		.max()
		.expect("provers is non-empty");
	let pad_eq_prefixes = iter::once(F::ONE)
		.chain(
			node_coords[..max_pad_len]
				.iter()
				.scan(F::ONE, |acc, &coord| {
					*acc *= eq_one_var(F::ZERO, coord);
					Some(*acc)
				}),
		)
		.collect::<Vec<_>>();

	let mut states = izip!(provers.iter_mut(), pad_lens, &segment_lens, claims)
		.map(|(prover, &pad_len, &segment_len, &claim)| {
			if n_node < pad_len {
				// The batch is still above this tree, so its own layers stay untouched.
				PaddedLayerProver::unreached(&ctx, prover.root_layer(), node_coords, claim)
			} else {
				let layer = prover.pop_layer();
				assert_eq!(
					layer.log_len(),
					n_root + n_node - segment_len + 1,
					"precondition: the witness has n_layers variables past the root dimension"
				);
				PaddedLayerProver::new(&ctx, layer, segment_len, node_coords, claim)
			}
		})
		.collect::<Vec<_>>();

	// Node and root rounds: the trees reduce independently.
	let mut challenges = Vec::with_capacity(eval_point.len());

	for round in 0..inner_coords.len() {
		// MLE-check binds variables from the highest index down.
		let node_vars_left = (inner_coords.len() - 1 - round).saturating_sub(n_root);

		// Execute each tree and accumulate the eq(selector)-weighted sum of the corrected round
		// polynomials.
		let mut coeffs = RoundCoeffs::default();
		let mut residual = F::ZERO;
		for (state, &segment_len, eq_weight) in
			izip!(&mut states, &segment_lens, eq_weights.iter_scalars())
		{
			let pad_eq = pad_eq_prefixes[node_vars_left.min(segment_len)];
			coeffs += &(state.execute() * (eq_weight * pad_eq));
			residual += eq_weight * (F::ONE - pad_eq);
		}
		coeffs.0[0] += residual;

		// Send truncated round proof to channel.
		channel.send_many(mlecheck::RoundProof::truncate(coeffs).coeffs());

		// Sample challenge and fold all trees.
		let challenge = channel.sample();
		challenges.push(challenge);

		states = izip!(states, &segment_lens)
			.map(|(state, &segment_len)| state.fold(challenge, &ctx, segment_len, node_vars_left))
			.collect();
	}

	// Finish the trees to get their own [eval_0, eval_1] pairs and their padded layers'.
	let (children, padded_children): (Vec<[F; 2]>, Vec<[F; 2]>) =
		states.into_iter().map(PaddedLayerProver::finish).unzip();
	let (mut vals_0, mut vals_1): (Vec<F>, Vec<F>) = padded_children
		.into_iter()
		.map(|[val_0, val_1]| (val_0, val_1))
		.unzip();

	// Pad vals_0 and vals_1 to 2^k with zeros for FieldBuffer::from_values.
	vals_0.resize(1 << k, F::ZERO);
	vals_1.resize(1 << k, F::ZERO);

	// Compute eval from buffers: sum_v eq(v, selector_coords) * vals_0[v] * vals_1[v].
	let eval = izip!(&vals_0, &vals_1, eq_weights.as_ref())
		.map(|(&v0, &v1, &eq_u)| v0 * v1 * eq_u)
		.sum();

	// Selector rounds: pack eval pairs straight into allocator buffers and use a single prover.
	let outer_prover = bivariate_product_mle::new(
		alloc,
		[
			FieldBuffer::<P, _>::from_values_in(alloc, &vals_0),
			FieldBuffer::<P, _>::from_values_in(alloc, &vals_1),
		],
		selector_coords.to_vec(),
		eval,
	);

	let ProveSingleOutput {
		multilinear_evals: outer_evals,
		challenges: outer_challenges,
	} = prove_single_mlecheck(outer_prover, channel);

	challenges.extend(outer_challenges);

	let [merged_eval_0, merged_eval_1]: [F; 2] =
		outer_evals.try_into().expect("prover has two multilinears");

	// Finalize layer: send evals, sample r, compute next claim.
	channel.send_many(&[merged_eval_0, merged_eval_1]);

	let r = channel.sample();

	let mut next_point = challenges;
	next_point.reverse();
	next_point.push(r);

	// Update each tree's claim for the next layer.
	let next_claims = izip!(pad_lens, children)
		.map(|(&pad_len, [child_0, child_1])| {
			if n_node < pad_len {
				// The variable this layer bound was one of the tree's padding variables, so its own
				// layers are untouched and the claim stays on its root, which the low child is.
				child_0
			} else {
				extrapolate_line_packed(child_0, child_1, r)
			}
		})
		.collect();

	(next_claims, next_point)
}

#[cfg(test)]
mod tests {
	use binius_field::PackedField;
	use binius_ip::prodcheck;
	use binius_math::{
		inner_product::inner_product,
		multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use binius_utils::checked_arithmetics::log2_ceil_usize;

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use binius_compute::GlobalAllocator;
	use rand::prelude::*;

	use super::*;

	fn test_prodcheck_prove_verify_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// 1. Create random witness with log_len = n + k
		let witness = random_field_buffer::<P>(&mut rng, n + k);

		// 2. Create prover (computes product layers)
		let (prover, products) = ProdcheckProver::new(k, &alloc, witness.clone());

		// 3. Generate random n-dimensional challenge point
		let eval_point = random_scalars::<P::Scalar>(&mut rng, n);

		// 4. Evaluate products layer at challenge point to create claim
		let products_eval = evaluate(&products, &eval_point);
		let claim = MultilinearEvalClaim {
			eval: products_eval,
			point: eval_point,
		};

		// 5. Run prover
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_output = prover.prove(claim.clone(), &mut prover_transcript);

		// 6. Run verifier
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output = prodcheck::verify(k, claim, &mut verifier_transcript).unwrap();

		// 7. Check outputs match
		assert_eq!(prover_output, verifier_output);

		// 8. Verify multilinear evaluation of original witness
		let expected_eval = evaluate(&witness, &verifier_output.point);
		assert_eq!(verifier_output.eval, expected_eval);
	}

	#[test]
	fn test_prodcheck_prove_verify() {
		test_prodcheck_prove_verify_helper::<Packed128b>(4, 3);
	}

	#[test]
	fn test_prodcheck_full_prove_verify() {
		test_prodcheck_prove_verify_helper::<Packed128b>(0, 4);
	}

	fn test_prodcheck_layer_computation_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// Create random witness with log_len = n + k
		let witness = random_field_buffer::<P>(&mut rng, n + k);

		// Create prover (computes product layers)
		let (_prover, products) = ProdcheckProver::new(k, &alloc, witness.clone());

		// For each index i in the products layer, verify it equals the product of witness values
		// at indices i + z * 2^n for z in 0..2^k (strided access, not contiguous)
		let stride = 1 << n;
		let num_terms = 1 << k;
		for i in 0..(1 << n) {
			let mut expected_product = P::Scalar::ONE;
			for z in 0..num_terms {
				expected_product *= witness.get(i + z * stride);
			}
			let actual = products.get(i);
			assert_eq!(actual, expected_product, "Product mismatch at index {i}");
		}
	}

	#[test]
	fn test_prodcheck_layer_computation() {
		test_prodcheck_layer_computation_helper::<Packed128b>(4, 3);
	}

	// ==================== batch_prove tests ====================

	/// One prover per entry of `depths`, each over the same `root_len`-variate root domain.
	#[allow(clippy::type_complexity)]
	fn batch_provers<'a, P: PackedField>(
		rng: &mut impl Rng,
		alloc: &'a GlobalAllocator,
		depths: &[usize],
		root_len: usize,
	) -> (Vec<FieldBuffer<P>>, Vec<ProdcheckProver<'a, GlobalAllocator, P>>, Vec<FieldBuffer<P>>) {
		itertools::multiunzip(depths.iter().map(|&depth| {
			let witness = random_field_buffer::<P>(&mut *rng, root_len + depth);
			let (prover, root) = ProdcheckProver::new(depth, alloc, witness.clone());
			assert_eq!(root.log_len(), root_len);
			(witness, prover, root)
		}))
	}

	/// The eq(selector)-weighted combination of per-tree claims, as the verifier forms it.
	fn combine_claims<P: PackedField>(
		claims: &[P::Scalar],
		selector_point: &[P::Scalar],
	) -> P::Scalar {
		let eq_weights = eq_ind_partial_eval::<P>(selector_point);
		inner_product(claims.iter().copied(), (0..claims.len()).map(|i| eq_weights.get(i)))
	}

	/// Lifts a claim on a tree's own leaves to the claim on their one-padding, which is what the
	/// depth-oblivious verifier reduces to.
	fn pad_claim<F: Field>(eval: F, pad_coords: &[F]) -> F {
		let pad_eq = pad_coords
			.iter()
			.map(|&coord| eq_one_var(F::ZERO, coord))
			.product::<F>();
		F::ONE + (eval - F::ONE) * pad_eq
	}

	/// Proves a batch of product trees of the given depths over a shared `root_len`-variate root
	/// domain, against the depth-oblivious verifier, then checks each tree's returned claim against
	/// that tree's own witness.
	fn test_batch_prove_helper<P: PackedField>(depths: &[usize], root_len: usize) {
		let mut rng = StdRng::seed_from_u64(11);
		let alloc = GlobalAllocator;

		let k = log2_ceil_usize(depths.len());
		let n_layers = *depths.iter().max().expect("depths is non-empty");

		let (witnesses, provers, roots) = batch_provers::<P>(&mut rng, &alloc, depths, root_len);

		// The verifier's input claim is the eq(selector)-weighted combination of the tree roots,
		// all evaluated at one shared root point.
		let root_point = random_scalars::<P::Scalar>(&mut rng, root_len);
		let claimed_products = roots
			.iter()
			.map(|root| evaluate(root, &root_point))
			.collect::<Vec<_>>();
		let selector_point = random_scalars::<P::Scalar>(&mut rng, k);
		let claim = MultilinearEvalClaim {
			eval: combine_claims::<P>(&claimed_products, &selector_point),
			point: [selector_point.clone(), root_point.clone()].concat(),
		};

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let BatchProveOutput { eval_point, evals } = batch_prove(
			provers,
			claimed_products,
			selector_point,
			root_point,
			&mut prover_transcript,
		);

		// The verifier's control flow depends only on the maximum depth.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output = prodcheck::verify(n_layers, claim, &mut verifier_transcript).unwrap();
		assert_eq!(verifier_output.point, eval_point);
		assert_eq!(eval_point.len(), k + root_len + n_layers);

		// Each tree's returned claim is on its own witness, at the root coordinates followed by the
		// node coordinates past that tree's padding.
		let (root_challenges, node_challenges) = eval_point[k..].split_at(root_len);
		for (i, (&depth, witness)) in iter::zip(depths, &witnesses).enumerate() {
			let point = [root_challenges, &node_challenges[n_layers - depth..]].concat();
			assert_eq!(evals[i], evaluate(witness, &point), "tree {i}");
		}

		// Padding those claims back up recovers the claim the verifier reduced to.
		let padded = iter::zip(depths, &evals)
			.map(|(&depth, &eval)| pad_claim(eval, &node_challenges[..n_layers - depth]))
			.collect::<Vec<_>>();
		assert_eq!(verifier_output.eval, combine_claims::<P>(&padded, &eval_point[..k]));
	}

	#[test]
	fn test_batch_prove_power_of_two_provers() {
		test_batch_prove_helper::<Packed128b>(&[3; 4], 0);
	}

	#[test]
	fn test_batch_prove_non_power_of_two_provers() {
		// 3 provers, so the selector dimension is padded out to 4.
		test_batch_prove_helper::<Packed128b>(&[4; 3], 0);
	}

	#[test]
	fn test_batch_prove_single_prover() {
		// 1 prover (edge case): the selector dimension is empty.
		test_batch_prove_helper::<Packed128b>(&[5], 0);
	}

	#[test]
	fn test_batch_prove_single_layer() {
		// Depth 1 throughout, the minimum: one layer and no aggregating rounds within it.
		test_batch_prove_helper::<Packed128b>(&[1; 4], 0);
	}

	#[test]
	fn test_batch_prove_with_root() {
		test_batch_prove_helper::<Packed128b>(&[4; 3], 2);
	}

	// ==================== unequal-depth tests ====================

	#[test]
	fn test_unequal_depths_mixed() {
		test_batch_prove_helper::<Packed128b>(&[2, 4, 5], 0);
	}

	#[test]
	fn test_unequal_depths_single_prover() {
		test_batch_prove_helper::<Packed128b>(&[3], 0);
	}

	#[test]
	fn test_unequal_depths_power_of_two_provers() {
		// The shallowest tree is padded by more than one layer, the deepest not at all.
		test_batch_prove_helper::<Packed128b>(&[1, 2, 5, 5], 0);
	}

	#[test]
	fn test_unequal_depths_maximal_padding() {
		// A single-layer tree beside a deep one: all but its last reduction is padding.
		test_batch_prove_helper::<Packed128b>(&[1, 6], 0);
	}

	#[test]
	fn test_unequal_depths_with_root() {
		test_batch_prove_helper::<Packed128b>(&[2, 4, 5], 2);
	}

	#[test]
	fn test_unequal_depths_maximal_padding_with_root() {
		test_batch_prove_helper::<Packed128b>(&[1, 6], 3);
	}

	#[test]
	fn test_unequal_depths_single_prover_with_root() {
		// One tree, so nothing is padded and the selector dimension is empty: the plain
		// root-carrying reduction.
		test_batch_prove_helper::<Packed128b>(&[4], 3);
	}
}
