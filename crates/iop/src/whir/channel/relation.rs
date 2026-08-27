// Copyright 2026 The Binius Developers

//! One relation queued against the committed oracle, and how a queue folds into one.

use binius_field::field::FieldOps;
use binius_math::univariate::evaluate_univariate;

use crate::channel::TransparentEvalFn;

/// A committed-oracle relation queued until the opening runs.
pub(super) struct QueuedRelation<Elem> {
	/// Evaluates the transparent multilinear at the point the relation sumcheck reduces to.
	pub(super) transparent: TransparentEvalFn<Elem>,
	/// The claimed inner product of the committed multilinear with the transparent one.
	pub(super) claim: Elem,
}

impl<Elem: FieldOps + 'static> QueuedRelation<Elem> {
	/// Folds every queued relation into one, against one transparent and one claim.
	///
	/// Relations `j = 0, 1, ...` are combined with the powers of `lambda`:
	///
	/// ```text
	///     T = sum_j lambda^j * t_j     the combined transparent
	///     S = sum_j lambda^j * s_j     the combined claim
	/// ```
	///
	/// An inner product is linear in the transparent.
	/// So `<pi, T> = S` holds exactly when every `<pi, t_j> = s_j` does.
	/// The exception has probability at most `(k - 1) / |F|` over `lambda`.
	///
	/// [NA25] section 4.3 batches with `k` independent coefficients rather than the powers of one.
	/// Powers cost one challenge instead of `k` and trade `1 / |F|` of error for `(k - 1) / |F|`,
	/// which is the same trade [BCIKS23] makes for Reed-Solomon proximity.
	///
	/// `lambda` is the caller's to draw, and it must be drawn only once every claim it combines is
	/// bound to the transcript, so that no claim can be chosen as a function of it.
	///
	/// ## Preconditions
	///
	/// * `relations` is non-empty.
	///
	/// [NA25]: <https://eprint.iacr.org/2025/1187>
	/// [BCIKS23]: <https://doi.org/10.1145/3614423>
	pub(super) fn batch(mut relations: Vec<Self>, lambda: Elem) -> Self {
		// A single relation folds nothing.
		// Evaluating one transparent directly is cheaper than a closure that wraps it.
		if relations.len() <= 1 {
			return relations
				.pop()
				.expect("precondition: the queue is non-empty");
		}

		// Split the queue so the combined claim can be a univariate evaluation at `lambda`.
		let (transparents, claims): (Vec<_>, Vec<_>) = relations
			.into_iter()
			.map(|relation| (relation.transparent, relation.claim))
			.unzip();
		let claim = evaluate_univariate(&claims, &lambda);

		Self {
			transparent: Box::new(move |point: &[Elem]| {
				// The combined transparent is only ever read at one point.
				// So it stays a closure rather than a materialized multilinear.
				let evals = transparents
					.iter()
					.map(|transparent| transparent(point))
					.collect::<Vec<_>>();
				evaluate_univariate(&evals, &lambda)
			}),
			claim,
		}
	}
}
