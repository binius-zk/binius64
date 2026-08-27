// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, util::powers};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace, FieldBuffer, inner_product::inner_product,
	multilinear::eq::eq_ind_partial_eval, univariate::EvaluationDomain,
};

use super::{
	SegmentWords,
	claims::OperatorClaims,
	key_collection::KeyCollection,
	phase_1::prove_phase_1,
	phase_2::{ShiftOutput, prove_phase_2},
	shift_ind::{ShiftChallengePoint, ShiftIndSumcheck},
};

/// One operation's operand evaluation claims, with the point they are claimed at.
///
/// An operation constrains a fixed number of operands at once, its arity:
///
/// ```text
/// ZERO 1   AND 3   IMUL 4   BMUL 6
/// ```
///
/// The arity is a type parameter, so two operations' claims cannot be passed in each other's
/// place.
///
/// Every operand is claimed at the same point: an oblong pair of a univariate bit-axis
/// coordinate and a multilinear constraint-index coordinate.
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
	/// Every operand evaluates to zero, at the empty constraint point, so this claim
	/// contributes nothing to the batch.
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

/// One operation's claims, with the expansions every proving phase needs precomputed.
///
/// Every shift key of the operation reads the same two expansions, built once here:
///
/// - the constraint point, expanded into its equality-indicator tensor;
/// - the batching coefficient, expanded into its powers, one per operand.
///
/// The arity is erased, since every phase picks an operation at run time and all four have to
/// share one type.
/// Only the batched combination survives that erasure, as a single scalar.
#[derive(Debug, Clone)]
pub struct PreparedOperatorData<F: Field> {
	/// The operand claims collapsed into one value by the batching coefficient, starting at
	/// the first power:
	///
	/// ```text
	/// batched_eval = sum_i evals[i] * lambda^(i + 1)
	/// ```
	///
	/// Two operations' batched values can be summed directly, with no further scaling.
	pub batched_eval: F,
	/// The univariate challenge folding the bit axis, shared by every operation.
	pub r_zhat_prime: F,
	/// The equality-indicator tensor of the constraint point, one weight per constraint.
	pub r_x_prime_tensor: FieldBuffer<F>,
	/// The batching coefficient's powers, one per operand, starting at the first power.
	///
	/// Indexable at run time by the operand a shift key names.
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
/// It is reached in five prover phases.
/// A shifted value index names two shifts applied in sequence.
/// The reduction peels them off from the output end inward:
///
/// 1. bind the outer shift slot, then the inner one, then the bit position within a word;
/// 2. bind the bit index of the intermediate word, where the two shift indicators meet;
/// 3. bind the output bit index the reduction's first factor attaches to;
/// 4. reduce what is left to a witness evaluation, against the constraint-matrix multilinear.
///
/// # Arguments
///
/// - `key_collection`: the prover's key collection for the constraint system.
/// - `public_words`: the constants followed by the inout values, as the circuit declares them.
/// - `hidden_words`: the private values, as the circuit declares them.
/// - `claims`: the operand evaluation claim of each operation.
/// - `domain_subspace`: the univariate evaluation domain.
/// - `channel`: the prover channel the interactive rounds run over.
/// - `alloc`: the allocator the intermediate buffers are drawn from.
///
/// # Returns
///
/// The final challenges with the witness evaluation.
/// Also the wiring multilinear's evaluation, for the caller to send.
pub fn prove<F, P, Channel, A>(
	key_collection: &KeyCollection,
	public_words: &[Word],
	hidden_words: &[Word],
	claims: OperatorClaims<F>,
	domain_subspace: &BinarySubspace<F>,
	channel: &mut Channel,
	alloc: &A,
) -> ShiftOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	// The segments are passed as the circuit declares them, at whatever length that is.
	// Neither phase needs them padded.
	let words = SegmentWords {
		public: public_words,
		hidden: hidden_words,
	};

	// One batching coefficient per operation, expanded along with its constraint point.
	// SOUNDNESS: this must draw in the same order the verifier draws in.
	let prepared = {
		let _scope = tracing::debug_span!("Expand tensor queries").entered();
		claims.prepare(|| channel.sample())
	};

	// The weights the reduction's first factor carries, one per bit position.
	// Phase 1 and phase 3 both need them, so they are computed once here.
	// All four operations share `r_zhat_prime`, so it is drawn from the BitAnd claim.
	let oblong_weights = domain_subspace.lagrange_evals_buffer(prepared.bitand.r_zhat_prime);

	// Phase 1: bind the shift variant, the shift amount, and the bit position.
	let phase_1_output = prove_phase_1::<_, P, _, _>(
		key_collection,
		words,
		&prepared,
		oblong_weights.as_ref(),
		channel,
		alloc,
	);

	// Phases 2 and 3 bind the two bit indices the shift indicators chain through.
	// Phase 2 takes the intermediate word's, phase 3 the reduction's first-factor output bit.
	//
	// Phase 2 runs against phase 1's leftover weights, carrying its evaluation as a constant.
	let inner = ShiftIndSumcheck::<P, _>::new(
		alloc,
		&phase_1_output.psi,
		&ShiftChallengePoint::new(&phase_1_output.r_j, &phase_1_output.inner),
		phase_1_output.g_eval,
	);
	debug_assert_eq!(inner.beta(), phase_1_output.gamma);
	let inner_output = inner.prove(channel, alloc);

	// Phase 3 runs against the reduction's first-factor weights, carrying what phase 2 fixed.
	// Its own weights evaluate to a factor the verifier recomputes independently.
	// So no division is needed between phases.
	let outer = ShiftIndSumcheck::<P, _>::new(
		alloc,
		oblong_weights.as_ref(),
		&ShiftChallengePoint::new(&inner_output.point, &phase_1_output.outer),
		inner_output.ind_eval * phase_1_output.g_eval,
	);
	debug_assert_eq!(outer.beta(), inner_output.eval);
	let outer_output = outer.prove(channel, alloc);

	// Phase 4 reduces to the final challenges and witness evaluation.
	// It runs against the constraint-matrix multilinear, scaled by the three factors above.
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
