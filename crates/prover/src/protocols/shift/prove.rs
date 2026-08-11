// Copyright 2025 Irreducible Inc.

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, util::powers};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace, FieldBuffer, inner_product::inner_product, multilinear::eq::eq_ind_partial_eval,
};

use super::{
	SegmentWords, claims::OperatorClaims, key_collection::KeyCollection, phase_1::prove_phase_1,
	phase_2::prove_phase_2,
};

/// One operation's operand evaluation claims, with the point they are claimed at.
///
/// An operation constrains a fixed number of operands at once, its arity:
///
/// ```text
/// ZERO 1   AND 3   IMUL 4   BMUL 6
/// ```
///
/// Each operand is one column of the witness, and each column carries one evaluation claim.
///
/// The arity is a type parameter, so no two operations share a type.
/// Two operations' claims therefore cannot be passed in each other's place.
///
/// This mirrors [`binius_verifier::protocols::shift::OperatorData`], which is already arity-typed.
///
/// Every operand of an operation is claimed at the same point.
/// So the point is stored once here rather than once per operand.
///
/// That point is oblong: a univariate coordinate over the bit axis, then the constraint index.
#[derive(Debug, Clone)]
pub struct OperatorData<F: Field, const ARITY: usize> {
	/// The claimed evaluation of each operand column, in the operation's operand order.
	pub evals: [F; ARITY],
	/// The univariate challenge folding the bit axis, shared by every operation.
	pub r_zhat_prime: F,
	/// The multilinear challenge over the constraint index.
	pub r_x_prime: Vec<F>,
}

impl<F: Field, const ARITY: usize> OperatorData<F, ARITY> {
	/// The claim of an operation the constraint system does not use.
	///
	/// Every operand evaluates to zero, at the empty constraint point.
	///
	/// That operation's constraint set is empty.
	/// So the shift finds no key naming it, and the claim contributes nothing to the batch.
	///
	/// # Arguments
	///
	/// - `r_zhat_prime`: the univariate challenge, shared by every operation.
	pub const fn zero_claim(r_zhat_prime: F) -> Self {
		Self {
			evals: [F::ZERO; ARITY],
			r_zhat_prime,
			r_x_prime: Vec::new(),
		}
	}
}

/// One operation's claims, with the expansions both proving phases need precomputed.
///
/// Every shift key of the operation reads the same two expansions, so each is built once here:
///
/// - the constraint point, expanded into its equality-indicator tensor;
/// - the batching coefficient, expanded into its powers, one per operand.
///
/// The arity is erased here, unlike in [`OperatorData`].
/// Both phases pick an operation at run time, from a shift key, so all four must share a type.
///
/// The individual operand claims do not survive that erasure, because nothing downstream reads
/// them one at a time — only their batched combination, which is a single scalar.
#[derive(Debug, Clone)]
pub struct PreparedOperatorData<F: Field> {
	/// The operand claims collapsed into one value by the batching coefficient.
	///
	/// Operand `i` is weighted by the `i`-th power, and the powers start at the first:
	///
	/// ```text
	/// batched_eval = sum_i evals[i] * lambda^(i + 1)
	/// ```
	///
	/// So this already carries a random factor unique to its operation.
	/// Two operations' batched values can therefore be summed with no further scaling.
	pub batched_eval: F,
	/// The univariate challenge folding the bit axis, shared by every operation.
	pub r_zhat_prime: F,
	/// The equality-indicator tensor of the constraint point, one weight per constraint.
	pub r_x_prime_tensor: FieldBuffer<F>,
	/// The batching coefficient's powers, one per operand, starting at the first power.
	///
	/// A shift key names the operand it acts on, so these stay indexable at run time.
	pub lambda_powers: Vec<F>,
}

impl<F: Field> PreparedOperatorData<F> {
	/// Expands one operation's claims against the batching coefficient drawn for it.
	///
	/// # Arguments
	///
	/// - `operator_data`: the operand claims, and the point they are claimed at.
	/// - `lambda`: the batching coefficient for this operation.
	pub fn new<const ARITY: usize>(operator_data: OperatorData<F, ARITY>, lambda: F) -> Self {
		let OperatorData {
			evals,
			r_zhat_prime,
			r_x_prime,
		} = operator_data;
		let r_x_prime_tensor = eq_ind_partial_eval::<F>(&r_x_prime);
		let lambda_powers: Vec<F> = powers(lambda).skip(1).take(ARITY).collect();
		Self {
			batched_eval: inner_product(evals, lambda_powers.iter().copied()),
			r_zhat_prime,
			r_x_prime_tensor,
			lambda_powers,
		}
	}
}

/// Proves the shift protocol reduction, collapsing every operation's claims into one.
///
/// The result is a single multilinear evaluation claim on the witness.
///
/// Two sumcheck phases run in sequence:
///
/// 1. phase 1 proves the batched claims over the shift variants and operand positions;
/// 2. phase 2 reduces those to a witness evaluation, against the monster multilinear.
///
/// # Parameters
/// - `key_collection`: the prover's key collection for the constraint system.
/// - `public_words`: the constants followed by the inout values, as the circuit declares them.
/// - `hidden_words`: the private values, as the circuit declares them.
/// - `claims`: the operand evaluation claim of each operation.
/// - `domain_subspace`: the univariate evaluation domain.
/// - `channel`: the prover channel driving the interactive protocol.
/// - `alloc`: the allocator backing the reduction's intermediate buffers.
///
/// # Returns
/// The `SumcheckOutput` with the final challenges and the witness evaluation.
pub fn prove<F, P, Channel, A>(
	key_collection: &KeyCollection,
	public_words: &[Word],
	hidden_words: &[Word],
	claims: OperatorClaims<F>,
	domain_subspace: &BinarySubspace<F>,
	channel: &mut Channel,
	alloc: &A,
) -> SumcheckOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	// The segments are passed as the circuit declares them, at whatever length that is. Phase 1
	// zips each word with its key range, so a segment shorter than its key ranges stops at its
	// last value — the words past it carry no keys — and phase 2's `fold_words` zero-pads each
	// fold up to `log2_ceil(len)` variables. Neither needs a padded segment.
	let words = SegmentWords {
		public: public_words,
		hidden: hidden_words,
	};

	// One batching coefficient per operation, expanded along with its constraint point.
	// SOUNDNESS: `prepare` draws in the order the verifier draws in; do not reorder it.
	let prepared = {
		let _scope = tracing::debug_span!("Expand tensor queries").entered();
		claims.prepare(|| channel.sample())
	};

	// Phase 1 outputs challenges `r_j || r_s`, and `gamma` as its evaluation (see paper).
	let phase_1_output = prove_phase_1::<_, P, _, _>(
		key_collection,
		words,
		&prepared,
		domain_subspace,
		channel,
		alloc,
	);

	// Phase 2 outputs challenges `r_y`, and the witness evaluation at the oblong point given by
	// the univariate variable `r_j` and the multilinear variable `r_y`.
	prove_phase_2::<_, P, _, _>(
		key_collection,
		words,
		&prepared,
		domain_subspace,
		phase_1_output,
		channel,
		alloc,
	)
}
