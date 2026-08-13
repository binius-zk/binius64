// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, util::powers};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace, FieldBuffer, inner_product::inner_product,
	multilinear::eq::eq_ind_partial_eval, univariate::lagrange_evals,
};

use super::{
	SegmentWords, claims::OperatorClaims, key_collection::KeyCollection, phase_1::prove_phase_1,
	phase_2::prove_phase_2, shift_ind::ShiftIndSumcheck,
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
/// One sumcheck runs, in five prover phases. A shifted value index names two shifts applied in
/// sequence, and the reduction peels them from the output end inward:
///
/// 1. phase 1 binds the outer shift slot, then the inner one, then the bit position within a word;
/// 2. phase 2 binds the bit index of the intermediate word, where the two shift indicators meet;
/// 3. phase 3 binds the output bit index the oblong weights attach to;
/// 4. phase 4 reduces what is left to a witness evaluation, against the monster multilinear.
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

	// The oblong weights the reduction's first factor carries. Phase 1 pushes them through every
	// shift to build its h multilinear and phase 3 runs its rounds over them directly, so they are
	// computed once here. All four operations share `r_zhat_prime`, so it is drawn from the BitAnd
	// claim.
	let oblong_weights = lagrange_evals(domain_subspace, prepared.bitand.r_zhat_prime);

	// Phases 1 and 2 bind the shift variant, the shift amount and the bit position, outputting
	// challenges `r_j || r_s || r_v` and the claim `gamma` (see paper).
	let phase_1_output = prove_phase_1::<_, P, _, _>(
		key_collection,
		words,
		&prepared,
		oblong_weights.as_ref(),
		channel,
		alloc,
	);

	// Phases 3 and 4 bind the two bit indices the shift indicators chain through, working back up
	// the chain: first the intermediate word's, where the inner indicator meets the outer one, then
	// the output bit the oblong weights attach to. Both are the same rounds over a weight vector
	// and an indicator, differing only in those two arguments.
	//
	// Phase 3 runs against the weights the outer rounds left behind, carrying phase 1's `g`
	// evaluation as a constant.
	let inner = ShiftIndSumcheck::<P, _>::new(
		alloc,
		&phase_1_output.psi,
		&phase_1_output.r_j,
		&phase_1_output.r_s_inner,
		&phase_1_output.r_v_inner,
		phase_1_output.g_eval,
	);
	debug_assert_eq!(inner.beta(), phase_1_output.gamma);
	let inner_output = inner.prove(channel, alloc);

	// Phase 4 runs against the oblong weights themselves. The constant it carries collects what is
	// already fixed: the inner indicator's evaluation and `g`'s. Its own weights evaluate to the
	// Lagrange factor the verifier recomputes, and `psi(r_k)` is what the two runs agree on, so no
	// division is needed to pass between them.
	let outer = ShiftIndSumcheck::<P, _>::new(
		alloc,
		oblong_weights.as_ref(),
		&inner_output.point,
		&phase_1_output.r_s_outer,
		&phase_1_output.r_v_outer,
		inner_output.ind_eval * phase_1_output.g_eval,
	);
	debug_assert_eq!(outer.beta(), inner_output.eval);
	let outer_output = outer.prove(channel, alloc);

	// Phase 5 outputs challenges `r_y`, and the witness evaluation at the oblong point given by
	// the univariate variable `r_j` and the multilinear variable `r_y`. Its monster multilinear is
	// scaled by the three bit-index factors the two runs reduced.
	prove_phase_2::<_, P, _, _>(
		key_collection,
		words,
		&prepared,
		phase_1_output,
		outer_output.weights_eval * outer_output.ind_eval * inner_output.ind_eval,
		outer_output.eval,
		channel,
		alloc,
	)
}
