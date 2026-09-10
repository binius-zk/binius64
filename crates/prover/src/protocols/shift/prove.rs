// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{BinarySubspace, univariate::EvaluationDomain};

use super::{
	SegmentWords,
	claims::OperatorClaims,
	key_collection::KeyCollection,
	phase_1::prove_phase_1,
	phase_2::{ShiftOutput, prove_phase_2},
	shift_ind::{ShiftChallengePoint, ShiftIndSumcheck},
};

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
/// - `claims`: the operand evaluation claims, all at prefixes of one constraint point.
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

	// One batching coefficient per operation, folded into its prefix of the constraint point's
	// expansion.
	// SOUNDNESS: this must draw in the same order the verifier draws in.
	let prepared = {
		let _scope = tracing::debug_span!("Expand tensor queries").entered();
		claims.prepare(channel)
	};

	// The weights the reduction's first factor carries, one per bit position.
	// Phase 1 and phase 3 both need them, so they are computed once here.
	let oblong_weights = domain_subspace.lagrange_evals_buffer(prepared.r_zhat_prime);

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
