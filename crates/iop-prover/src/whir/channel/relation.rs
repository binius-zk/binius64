// Copyright 2026 The Binius Developers

//! One relation queued against the committed oracle, and how a queue folds into one.

use binius_compute::Allocator;
use binius_field::PackedField;
use binius_math::FieldVec;

/// A committed-oracle relation queued until the opening runs.
pub(super) struct QueuedRelation<P: PackedField, A: Allocator> {
	/// The transparent multilinear the message is opened against.
	/// It is backed by the caller's allocator.
	pub(super) transparent: FieldVec<P, A>,
	/// The claimed inner product of the committed multilinear with the transparent one.
	pub(super) claim: P::Scalar,
}

impl<P: PackedField, A: Allocator> QueuedRelation<P, A> {
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
	/// Powers cost one challenge instead of `k` and trade `1 / |F|` of error for `(k - 1) / |F|`.
	///
	/// `lambda` is the caller's to draw, and it must be drawn only once every claim it combines is
	/// bound to the transcript, so that no claim can be chosen as a function of it.
	///
	/// Mirrors the verifier-side batching, which folds the same claims at the same coefficient.
	///
	/// ## Preconditions
	///
	/// * `relations` is non-empty.
	///
	/// [NA25]: <https://eprint.iacr.org/2025/1187>
	pub(super) fn batch(relations: Vec<Self>, lambda: P::Scalar) -> Self {
		let mut relations = relations.into_iter();
		let mut batched = relations
			.next()
			.expect("precondition: the queue is non-empty");

		// Powers of the coefficient scale the remaining transparents into the first, in place.
		let mut coeff = lambda;
		for relation in relations {
			let scale = P::broadcast(coeff);
			for (accum, addend) in batched
				.transparent
				.as_mut()
				.iter_mut()
				.zip(relation.transparent.as_ref())
			{
				*accum += scale * *addend;
			}
			batched.claim += coeff * relation.claim;
			coeff *= lambda;
		}

		batched
	}
}
