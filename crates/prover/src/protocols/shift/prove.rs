// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{cmp::max, marker::PhantomData};

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace, FieldBuffer, inner_product::inner_product,
	multilinear::eq::eq_ind_partial_eval, univariate::lagrange_evals,
};
use tracing::instrument;

use super::{
	SegmentWords,
	claims::{OperatorClaims, PreparedOperatorClaims},
	key_collection::KeyCollection,
	phase_1::{Phase1Output, SparseShiftRows},
	phase_2::{ShiftOutput, run_sumcheck, zero_extend},
	shift_ind::{ShiftChallengePoint, ShiftIndSumcheck},
};
use crate::fold_word::fold_words;

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
		let lambda_powers: Vec<F> = lambda.powers().skip(1).take(ARITY).collect();
		Self {
			batched_eval: inner_product(evals, lambda_powers.iter().copied()),
			r_zhat_prime,
			r_x_prime_tensor,
			lambda_powers,
		}
	}
}

/// Drives the shift protocol reduction, owning the prover channel and the allocator its
/// intermediate buffers are drawn from.
///
/// Every phase reads and writes the same channel and draws from the same allocator, carried
/// here once rather than threaded through each phase's own argument list.
pub struct ShiftProver<'a, 'alloc, A: Allocator, P, Channel> {
	/// Ties the reduction's packed-field type to this struct, though it is never stored.
	_p_marker: PhantomData<P>,
	/// The channel the reduction's interactive rounds run over.
	channel: &'a mut Channel,
	/// The allocator the reduction's intermediate buffers are drawn from.
	alloc: &'alloc A,
}

impl<'a, 'alloc, A: Allocator, P, Channel> ShiftProver<'a, 'alloc, A, P, Channel> {
	/// Builds a prover over the given channel and allocator.
	///
	/// The packed field is not inferred from either argument, so callers usually need a
	/// turbofish.
	pub const fn new(channel: &'a mut Channel, alloc: &'alloc A) -> Self {
		Self {
			_p_marker: PhantomData,
			channel,
			alloc,
		}
	}
}

impl<'alloc, A, F, P, Channel> ShiftProver<'_, 'alloc, A, P, Channel>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
	A: Allocator,
{
	/// Proves the shift protocol reduction, collapsing every operation's claims into one.
	///
	/// The result is a single multilinear evaluation claim on the witness, reached in five
	/// prover phases.
	/// A shifted value index names two shifts applied in sequence, and the reduction peels
	/// them off from the output end inward:
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
	///
	/// # Returns
	///
	/// The final challenges with the witness evaluation, and the wiring multilinear's
	/// evaluation for the caller to send.
	pub fn prove(
		&mut self,
		key_collection: &KeyCollection,
		public_words: &[Word],
		hidden_words: &[Word],
		claims: OperatorClaims<F>,
		domain_subspace: &BinarySubspace<F>,
	) -> ShiftOutput<F> {
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
			claims.prepare(|| self.channel.sample())
		};

		// The weights the reduction's first factor carries, one per bit position.
		// Phase 1 and phase 3 both need them, so they are computed once here, drawn from the
		// BitAnd claim.
		let oblong_weights = lagrange_evals(domain_subspace, prepared.bitand.r_zhat_prime);

		// Phase 1: bind the shift variant, the shift amount, and the bit position.
		let phase_1_output = self.phase1(key_collection, words, &prepared, oblong_weights.as_ref());

		// Phases 2 and 3 bind the two bit indices the shift indicators chain through: first
		// the intermediate word's, then the reduction's first-factor output bit.
		//
		// Phase 2 runs against phase 1's leftover weights, carrying phase 1's evaluation as a
		// constant.
		let inner = ShiftIndSumcheck::<P, _>::new(
			self.alloc,
			&phase_1_output.psi,
			&ShiftChallengePoint::new(&phase_1_output.r_j, &phase_1_output.inner),
			phase_1_output.g_eval,
		);
		debug_assert_eq!(inner.beta(), phase_1_output.gamma);
		let inner_output = inner.prove(self.channel, self.alloc);

		// Phase 3 runs against the reduction's first-factor weights, carrying what phase 2
		// fixed.
		// Its own weights evaluate to a factor the verifier recomputes independently, so no
		// division is needed between phases.
		let outer = ShiftIndSumcheck::<P, _>::new(
			self.alloc,
			oblong_weights.as_ref(),
			&ShiftChallengePoint::new(&inner_output.point, &phase_1_output.outer),
			inner_output.ind_eval * phase_1_output.g_eval,
		);
		debug_assert_eq!(outer.beta(), inner_output.eval);
		let outer_output = outer.prove(self.channel, self.alloc);

		// Phase 4 reduces to the final challenges and witness evaluation, against the
		// constraint-matrix multilinear scaled by the three bit-index factors above.
		self.phase2(
			key_collection,
			words,
			&prepared,
			phase_1_output,
			outer_output.weights_eval * outer_output.ind_eval * inner_output.ind_eval,
			outer_output.eval,
		)
	}

	/// Proves the first phase of the shift reduction.
	///
	/// Builds the witness-and-batching multilinear for both segments, concatenates their
	/// rows, and runs one sumcheck over their product with a weight table that is never
	/// formed as a table.
	///
	/// # Arguments
	///
	/// - `oblong_weights`: the weights of the reduction's first factor, one per bit position,
	///   pushed through both shift slots to build the weight table this phase's sumcheck runs
	///   against.
	#[instrument(skip_all, name = "prover_phase_1")]
	fn phase1(
		&mut self,
		key_collection: &KeyCollection,
		words: SegmentWords<'_>,
		prepared: &PreparedOperatorClaims<F>,
		oblong_weights: &[F],
	) -> Phase1Output<F> {
		// Accumulate the witness-and-batching rows of the public and hidden segments
		// separately.
		// The public words are the prefix of the value vector, and each segment's key ranges
		// are relative to its own segment.
		let public = key_collection
			.public
			.build_g::<_, P>(words.public, prepared);
		let hidden = key_collection
			.hidden
			.build_g::<_, P>(words.hidden, prepared);
		let g = SparseShiftRows::from_segments([
			(&public, &key_collection.public.dense_shift_enc),
			(&hidden, &key_collection.hidden.dense_shift_enc),
		]);

		g.run_phase_1_sumcheck(oblong_weights, prepared.batched_eval(), self.channel, self.alloc)
	}

	/// Proves the second phase of the shift protocol reduction.
	///
	/// Folds the value-vector words by the bit-position challenge, builds the
	/// constraint-matrix multilinear's two segments, and runs a sumcheck between them with a
	/// sparse first round over the segment selector.
	///
	/// # Arguments
	///
	/// - `key_collection`: the prover's key collection for the constraint system.
	/// - `words`: the value-vector words.
	/// - `prepared`: the prepared claim of each operation, indexed by the operation a key names.
	/// - `phase_1_output`: the challenges and evaluation the first phase produced.
	/// - `shift_ind_eval`: the scalar weighting every shift key, the product of the two indicator
	///   evaluations the earlier bit-index phases reduced to.
	/// - `epsilon`: the claim this phase's rounds prove.
	///
	/// # Returns
	///
	/// The combined challenges with the witness evaluation, and the wiring multilinear's
	/// evaluation.
	#[instrument(skip_all, name = "prove_phase_2")]
	fn phase2(
		&mut self,
		key_collection: &KeyCollection,
		words: SegmentWords<'_>,
		prepared: &PreparedOperatorClaims<F>,
		phase_1_output: Phase1Output<F>,
		shift_ind_eval: F,
		epsilon: F,
	) -> ShiftOutput<F> {
		let Phase1Output {
			r_j,
			inner,
			outer,
			psi: _,
			gamma: _,
			g_eval: _,
		} = phase_1_output;

		let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);

		// Fold each segment separately.
		// The combined witness is never materialized: each fold is zero-padded to enough
		// variables to cover its own segment's length.
		let public_folded = fold_words::<_, P, _>(self.alloc, words.public, r_j_tensor.as_ref());
		let hidden_folded = fold_words::<_, P, _>(self.alloc, words.hidden, r_j_tensor.as_ref());

		let (public_monster, hidden_monster) = key_collection.build_monster_segments(
			self.alloc,
			prepared,
			shift_ind_eval,
			&inner,
			&outer,
		);

		// Both halves of the sumcheck share one word-index space, spanning the wider of the
		// two segments.
		// The hidden segment is normally wider, but a system with more public words than
		// private values inverts that, so the hidden half is zero-extended to match.
		let log_segment_words = max(public_folded.log_len(), hidden_folded.log_len());
		let hidden_folded = zero_extend(self.alloc, hidden_folded, log_segment_words);
		let hidden_monster = zero_extend(self.alloc, hidden_monster, log_segment_words);

		run_sumcheck(
			&public_folded,
			hidden_folded,
			&public_monster,
			hidden_monster,
			shift_ind_eval,
			words.public,
			r_j,
			epsilon,
			self.channel,
			self.alloc,
		)
	}
}
