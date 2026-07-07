// Copyright 2025 Irreducible Inc.

//! Batched shift reduction prover for the M4 data-parallel proof system.
//!
//! M4 proves `K = 2^k` instances of one circuit at once, stacked instance-major.
//! The batched BitAnd/IntMul reductions leave operand claims at a local constraint point.
//! Their evaluation point also carries an instance part `r_kappa`.
//!
//! The instance index enters an operand claim only through the committed witness bit.
//! The shift structure is shared by every instance, so the instance factor folds onto the witness.
//! Every witness-derived buffer of the single-instance reduction then becomes an eq-weighted sum:
//!
//! ```text
//!   g_batch    = sum_kappa eq(r_kappa, kappa) * g(instance_kappa)
//!   fold_batch = sum_kappa eq(r_kappa, kappa) * fold(instance_kappa)
//! ```
//!
//! The g parts and the phase-2 folds are linear in the witness, so these combinations are exact.
//! The shift kernels and the monster are instance-independent, so they are reused verbatim.
//!
//! So this is the single-instance reduction run on the instance-folded witness.
//! Its transcript matches the single-instance prover, so `verify` and `check_eval` accept it as is.
//! Their witness evaluation is the committed batch witness partially evaluated at `r_kappa`.

use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace, FieldBuffer,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars},
};
use binius_verifier::config::LOG_WORD_SIZE_BITS;

use super::{
	key_collection::KeyCollection,
	monster::{build_h_parts, build_monster_segments},
	phase_1::{build_g_parts, run_phase_1_sumcheck},
	phase_2::run_sumcheck,
	prove::{OperatorData, PreparedOperatorData},
};
use crate::fold_word::fold_words;

/// Proves the batched shift reduction over `K = 2^k` instances of one circuit.
///
/// It runs the single-instance two-phase reduction on the instance-folded witness.
/// The transcript matches the single-instance prover, so the single-instance verifier accepts it.
///
/// # Arguments
///
/// - `key_collection`: the per-instance shift structure, shared by every instance.
/// - `instances`: one committed-word slice per instance, in instance order.
/// - `r_kappa`: the instance challenge of length `k`, the instance coordinates of the claim point.
/// - `bitand_data`: the BitAnd operand claim at the local constraint point.
/// - `intmul_data`: the IntMul operand claim at the local constraint point.
/// - `domain_subspace`: the Lagrange basis subspace over the word bits.
/// - `channel`: the prover channel.
///
/// # Returns
///
/// The challenges `[r_j, r_y]` and the witness evaluation.
/// That evaluation is the committed batch witness partially evaluated at `r_kappa`.
///
/// # Panics
///
/// Panics if the instance count is not `2^{r_kappa.len()}`.
pub fn prove_batch<F, P, Channel>(
	key_collection: &KeyCollection,
	instances: &[&[Word]],
	r_kappa: &[F],
	bitand_data: OperatorData<F>,
	intmul_data: OperatorData<F>,
	domain_subspace: &BinarySubspace<F>,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	// The batch is a clean hypercube of instances: exactly 2^k of them.
	assert_eq!(instances.len(), 1usize << r_kappa.len(), "instance count must be 2^r_kappa.len()");

	// Sample the operand-batching coefficients, matching the single-instance prover's order.
	let bitand_lambda = channel.sample();
	let intmul_lambda = channel.sample();
	let bitand_prep = PreparedOperatorData::new(bitand_data, bitand_lambda);
	let intmul_prep = PreparedOperatorData::new(intmul_data, intmul_lambda);

	// eq(r_kappa, .) expanded over the 2^k instances: one weight per instance.
	let eps = eq_ind_partial_eval_scalars(r_kappa);

	// Phase 1: g parts are the eq(r_kappa, .)-weighted sum of the per-instance g parts.
	let mut g_parts =
		build_g_parts::<F, P>(instances[0], key_collection, &bitand_prep, &intmul_prep);
	for part in &mut g_parts {
		scale(part, eps[0]);
	}
	for (kappa, &eps_k) in eps.iter().enumerate().skip(1) {
		let g_kappa =
			build_g_parts::<F, P>(instances[kappa], key_collection, &bitand_prep, &intmul_prep);
		for (acc, src) in g_parts.iter_mut().zip(&g_kappa) {
			add_scaled(acc, src, eps_k);
		}
	}
	let h_parts = build_h_parts::<F, P>(domain_subspace, bitand_prep.r_zhat_prime);

	let SumcheckOutput {
		challenges: mut r_jr_s,
		eval: gamma,
	} = run_phase_1_sumcheck(g_parts, h_parts, channel);

	// Split the phase-1 challenges: r_j is the low bit-index half, r_s the high shift-amount half.
	let r_s = r_jr_s.split_off(LOG_WORD_SIZE_BITS);
	let r_j = r_jr_s;
	let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);

	// Phase 2: fold each instance's segments at r_j.
	// Combine the per-instance folds over instances with the eq weights.
	let n_public = key_collection.public.n_words();
	let (public_0, hidden_0) = instances[0].split_at(n_public);
	let mut public_folded = fold_words::<F, P>(public_0, r_j_tensor.as_ref());
	let mut hidden_folded = fold_words::<F, P>(hidden_0, r_j_tensor.as_ref());
	scale(&mut public_folded, eps[0]);
	scale(&mut hidden_folded, eps[0]);
	for (kappa, &eps_k) in eps.iter().enumerate().skip(1) {
		let (public_words, hidden_words) = instances[kappa].split_at(n_public);
		add_scaled(
			&mut public_folded,
			&fold_words::<F, P>(public_words, r_j_tensor.as_ref()),
			eps_k,
		);
		add_scaled(
			&mut hidden_folded,
			&fold_words::<F, P>(hidden_words, r_j_tensor.as_ref()),
			eps_k,
		);
	}

	// The monster is instance-independent: the single-instance segments verbatim.
	let (public_monster, hidden_monster) = build_monster_segments::<F, P>(
		key_collection,
		&bitand_prep,
		&intmul_prep,
		domain_subspace,
		&r_j,
		&r_s,
	);

	// The public reconstruction words: the shared public segment.
	// This initial batch targets instances whose public inputs agree.
	let public_words = &instances[0][..n_public];

	run_sumcheck(
		public_folded,
		hidden_folded,
		public_monster,
		hidden_monster,
		public_words,
		r_j,
		gamma,
		channel,
	)
}

/// Scales every packed element of a buffer by a scalar, in place.
fn scale<P: PackedField>(buf: &mut FieldBuffer<P>, scalar: P::Scalar) {
	let broadcast = P::broadcast(scalar);
	for elem in buf.as_mut() {
		*elem *= broadcast;
	}
}

/// Adds `scalar * src` into `acc`, element-wise.
/// The two buffers have equal length.
fn add_scaled<P: PackedField>(acc: &mut FieldBuffer<P>, src: &FieldBuffer<P>, scalar: P::Scalar) {
	let broadcast = P::broadcast(scalar);
	for (acc_elem, &src_elem) in acc.as_mut().iter_mut().zip(src.as_ref()) {
		*acc_elem += broadcast * src_elem;
	}
}
