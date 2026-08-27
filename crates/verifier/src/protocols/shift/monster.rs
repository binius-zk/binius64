// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_core::constraint_system::{Operand, ValueIndex};
use binius_field::{BinaryField, FieldOps, WideMul, util::powers};
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};

use super::SHIFT_COUNT;

/// The evaluation of one operation's monster multilinear polynomial.
///
/// The monster multilinear encodes all `ARITY`-operand constraints of a single operation (BitAnd,
/// IntMul or BinMul) into one polynomial:
///
/// $$
/// \sum_{\text{m_idx} \in \text{enumerate(operands)}}
///     \lambda^{\text{m_idx}+1}
///     \sum_{\text{op}} h_{\text{op}}(r_j, r_s) \cdot M_{\text{m}, \text{op}}(r_x', r_y, r_s)
/// $$
///
/// where `m_idx` indexes the operand position (0 to `ARITY - 1`), `op` ranges over the shift
/// variants, `h_op` is the shift selector polynomial, and `M_{m,op}` is the multilinear extension
/// of the operand values.
///
/// Both evaluations read one borrowed weight table per axis, bundled as [`WiringWeights`]. The
/// tables the four operations share are then built once by the caller and lent to all four, rather
/// than copied into a per-operation buffer. [`Self::call`] evaluates generically over any `E`;
/// [`Self::call_native`] takes the `WideMul`-accelerated base-field path.
pub struct OperationEvalFn<'a, C, const ARITY: usize> {
	/// The operation's constraints, each exposing its `ARITY` operands as an array in storage
	/// order.
	constraints: &'a [C],
}

impl<'a, C, const ARITY: usize> OperationEvalFn<'a, C, ARITY> {
	/// Wraps an operation's constraints for monster-multilinear evaluation.
	pub const fn new(constraints: &'a [C]) -> Self {
		Self { constraints }
	}
}

/// One nonzero of the wiring tensor.
///
/// The tensor has five axes.
///
/// ```text
///     constraint index  x  operand position  x  inner slot  x  outer slot  x  value address
/// ```
///
/// A constraint names a handful of positions and nothing else, which is what makes it sparse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WiringEntry {
	/// Which constraint of the operation holds the entry.
	pub constraint: usize,
	/// Which of that constraint's operands names it.
	pub operand: usize,
	/// The inner shift slot's spelling, as a variant-and-amount index.
	pub inner_shift: usize,
	/// The outer shift slot's spelling, as a variant-and-amount index.
	pub outer_shift: usize,
	/// The value the entry reads.
	pub value: ValueIndex,
}

/// One weight table per axis of the wiring tensor.
///
/// Each table is indexed the way its axis is, so contracting is one lookup per axis and a product.
pub struct WiringWeights<'a, E> {
	/// One entry per constraint index, covering the padded constraint count.
	pub constraint: &'a [E],
	/// One entry per inner slot spelling paired with an operand position, the operand innermost.
	pub inner_operand: &'a [E],
	/// One entry per outer slot spelling.
	pub outer: &'a [E; SHIFT_COUNT],
	/// One entry per value address, in three runs: constants, then inout, then private.
	pub value: [&'a [E]; 3],
}

// Shared slices copy freely whatever `E` is. Deriving these would demand `E: Copy`, which the
// generic evaluation path does not have.
impl<E> Clone for WiringWeights<'_, E> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<E> Copy for WiringWeights<'_, E> {}

impl<C: AsRef<[Operand; ARITY]>, const ARITY: usize> OperationEvalFn<'_, C, ARITY> {
	/// Every nonzero of the wiring tensor, in constraint order.
	///
	/// The tensor is what a deferred wiring claim is about.
	///
	/// Enumerating it is what lets that claim be folded.
	/// It is also what lets the extension be evaluated away from the run that raised the claim.
	pub fn entries(&self) -> impl Iterator<Item = WiringEntry> + '_ {
		self.constraints
			.iter()
			.enumerate()
			.flat_map(|(constraint, terms)| {
				terms
					.as_ref()
					.iter()
					.enumerate()
					.flat_map(move |(operand, operand_terms)| {
						operand_terms.iter().map(move |svi| WiringEntry {
							constraint,
							operand,
							inner_shift: svi.inner().index(),
							outer_shift: svi.outer().index(),
							value: svi.value_index,
						})
					})
			})
	}

	/// Contracts the wiring tensor against one weight table per axis.
	///
	/// This is the tensor's defining sum, one nonzero at a time.
	///
	/// ```text
	///     sum over nonzeros of
	///         constraint[i] * inner_operand[inner * ARITY + a] * outer[o] * value[seg][idx]
	/// ```
	///
	/// Contract with equality indicators and the result is the tensor's multilinear extension.
	/// The point is the one those indicators were built from.
	///
	/// That extension is what an accumulated claim is about.
	/// Evaluating it is what settling such a claim comes down to.
	///
	/// The batched evaluation computes the same sum.
	/// It groups terms by constraint, so a constraint's weight multiplies in once per constraint.
	/// That grouping is why it is the path a circuit runs.
	///
	/// This one trades those saved multiplies for a flat walk, which is what a fold needs.
	pub fn contract<E: FieldOps>(&self, weights: WiringWeights<'_, E>) -> E {
		let mut acc = E::zero();
		for entry in self.entries() {
			// Two axes share one table: the inner slot's spelling with the operand innermost.
			let inner_operand = entry.inner_shift * ARITY + entry.operand;
			acc += weights.constraint[entry.constraint].clone()
				* &weights.inner_operand[inner_operand]
				* &weights.outer[entry.outer_shift]
				* &weights.value[entry.value.segment() as usize][entry.value.index() as usize];
		}
		acc
	}

	/// Evaluates the monster multilinear against one weight table per axis.
	///
	/// This is [`contract`](Self::contract)'s sum, grouped by constraint: a constraint's weight
	/// multiplies in once for the whole constraint rather than once per term. Grouping is exact,
	/// so the two agree; it is the cheaper path, and the one a circuit runs.
	///
	/// The constraint table covers the padded constraint count, so the walk stops at the last real
	/// constraint; padding rows carry no operand terms and contribute nothing.
	pub fn call<E: FieldOps>(&self, weights: WiringWeights<'_, E>) -> E {
		let mut eval = E::zero();
		for (constraint, constraint_weight) in iter::zip(self.constraints, weights.constraint) {
			let mut constraint_eval = E::zero();
			for (operand_id, operand) in constraint.as_ref().iter().enumerate() {
				for svi in operand {
					// Two axes share one table: the inner slot's spelling with the operand
					// innermost.
					let inner_operand = svi.inner().index() * ARITY + operand_id;
					constraint_eval += weights.inner_operand[inner_operand].clone()
						* &weights.outer[svi.outer().index()]
						* &weights.value[svi.value_index.segment() as usize]
							[svi.value_index.index() as usize];
				}
			}
			eval += constraint_eval * constraint_weight;
		}

		eval
	}

	/// Native fast path over the base field `F`.
	///
	/// Produces the identical result, but defers the `GF(2^128)` reductions: the per-constraint
	/// contributions accumulate into a single *unreduced* wide element, reduced exactly once at the
	/// end (reduction is `F`-linear, so this equals reducing each per-constraint product and
	/// summing). The generic [`call`](Self::call) can't do this because `E: FieldOps` does not
	/// imply `WideMul`.
	pub fn call_native<F: BinaryField>(&self, weights: WiringWeights<'_, F>) -> F
	where
		C: Sync,
	{
		// One unreduced wide product per constraint. The constraints partition cleanly across
		// rayon: each produces a single wide element and they are summed, so there is no large
		// per-task accumulator. The single final reduction is `F`-linear. The constraint table
		// covers the padded constraint count, so the zip stops at the last real constraint; the
		// padding rows have no operand terms and contribute nothing.
		//
		// A constraint names only a handful of terms.
		// So a minimum task size keeps each task above rayon's own handoff cost.
		let eval = self
			.constraints
			.par_iter()
			.zip(weights.constraint.par_iter())
			.with_min_task(WorkPerItem::FieldMuls)
			.map(|(constraint, &constraint_weight)| {
				let mut constraint_eval = F::ZERO;
				for (operand_id, operand) in constraint.as_ref().iter().enumerate() {
					for svi in operand {
						let inner_operand = svi.inner().index() * ARITY + operand_id;
						constraint_eval += weights.inner_operand[inner_operand]
							* weights.outer[svi.outer().index()]
							* weights.value[svi.value_index.segment() as usize]
								[svi.value_index.index() as usize];
					}
				}
				F::wide_mul(constraint_eval, constraint_weight)
			})
			.sum::<<F as WideMul>::Output>();
		F::reduce(eval)
	}
}

/// Folds the operand batching coefficients (λ powers) into the inner slot's shift scalars,
/// producing a table indexed by `(variant, amount, operand_id)` whose entry is
/// `inner[variant * Word::BITS + amount] · λ^{operand_id + 1}`.
///
/// The fan-out stays on this one table.
/// A term's outer-slot weight multiplies in where the term is read.
/// So the table is `SHIFT_COUNT * arity` entries rather than `SHIFT_COUNT^2 * arity`.
pub fn operand_shift_scalar_table<E: FieldOps>(
	shift_scalars: &[E; SHIFT_COUNT],
	lambda: &E,
	arity: usize,
) -> Vec<E> {
	let lambda_powers = powers(lambda.clone())
		.skip(1)
		.take(arity)
		.collect::<Vec<_>>();
	let mut table = Vec::with_capacity(shift_scalars.len() * arity);
	for shift_scalar in shift_scalars {
		for lambda_power in &lambda_powers {
			table.push(shift_scalar.clone() * lambda_power);
		}
	}
	table
}

#[cfg(test)]
mod tests {
	use binius_core::{
		ShiftVariant,
		constraint_system::{AndConstraint, Shift, ShiftedValueIndex, ValueIndex},
		word::Word,
	};
	use binius_field::{Field, Ghash128b, Random};
	use binius_math::{multilinear::eq::eq_ind_partial_eval_scalars, test_utils::random_scalars};
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use rand::prelude::*;

	use super::{super::SHIFT_VARIANT_COUNT, *};

	/// Builds `n_constraints` random arity-3 constraints (like `AndConstraint`), constraint-major:
	/// one array of operands per constraint.
	///
	/// Every term names a private word, so the constant and inout runs of the word-index tensor
	/// stay empty. Terms are drawn across all three classes — unshifted, singly shifted and doubly
	/// shifted — so both slots' tables are read at more than one index.
	///
	/// The evaluation is a sum over whatever terms it is handed, so the sequences here need not be
	/// canonical or non-collapsible the way a real constraint system's do.
	fn random_and_constraints(
		rng: &mut StdRng,
		n_constraints: usize,
		n_words: usize,
	) -> Vec<AndConstraint> {
		let shift_variants = [
			ShiftVariant::Sll,
			ShiftVariant::Slr,
			ShiftVariant::Sar,
			ShiftVariant::Rotr,
			ShiftVariant::Sll32,
			ShiftVariant::Srl32,
			ShiftVariant::Sra32,
			ShiftVariant::Rotr32,
		];
		let random_shift = |rng: &mut StdRng| Shift {
			variant: shift_variants[rng.random_range(0..SHIFT_VARIANT_COUNT)],
			amount: rng.random_range(0..Word::BITS) as u8,
		};
		(0..n_constraints)
			.map(|_| {
				AndConstraint(std::array::from_fn(|_| {
					(0..rng.random_range(0..=3))
						.map(|_| {
							let value_index =
								ValueIndex::private(rng.random_range(0..n_words) as u32);
							let inner = random_shift(rng);
							// A third of the terms carry a second shift, so the outer table is read
							// away from the identity as well as at it.
							if rng.random_range(0..3) == 0 {
								ShiftedValueIndex::new(value_index, [inner, random_shift(rng)])
							} else {
								ShiftedValueIndex::single(value_index, inner)
							}
						})
						.collect()
				}))
			})
			.collect()
	}

	/// The weight tables a run's own reduction reads, one per axis of the wiring tensor.
	fn run_weights<'a, F: BinaryField>(
		r_x_prime_tensor: &'a [F],
		inner_operand: &'a [F],
		outer: &'a [F; SHIFT_COUNT],
		value: [&'a [F]; 3],
	) -> WiringWeights<'a, F> {
		WiringWeights {
			constraint: r_x_prime_tensor,
			inner_operand,
			outer,
			value,
		}
	}

	#[test]
	fn evaluate_monster_scales_by_the_outer_slot_weight() {
		// Invariant: the outer slot's weight reaches every term, and reaches it as a factor.
		//
		// The evaluation is linear in the outer table, and each term reads exactly one of its
		// entries, so scaling the whole table scales the evaluation by the same factor. That holds
		// however the fixture's terms are spread across the table, which is what lets the fixture
		// carry doubly shifted terms rather than reading index 0 alone.
		type F = Ghash128b;
		let mut rng = StdRng::seed_from_u64(7);

		let n_words = 40usize;
		let constraints = random_and_constraints(&mut rng, 32, n_words);
		let r_x_prime = random_scalars::<F>(&mut rng, 5);
		let lambda = F::random(&mut rng);
		let inner: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
		let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
		let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

		let r_x_prime_tensor = eq_ind_partial_eval_scalars(&r_x_prime);
		let inner_operand =
			operand_shift_scalar_table(&inner, &lambda, constraints[0].as_ref().len());

		let eval_fn = OperationEvalFn::new(&constraints);
		let eval_with_outer = |outer: &[F; SHIFT_COUNT]| {
			eval_fn.call_native(run_weights(&r_x_prime_tensor, &inner_operand, outer, r_y_tensor))
		};

		let outer: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
		let baseline = eval_with_outer(&outer);
		// A non-degenerate fixture, or the scaling below proves nothing.
		assert_ne!(baseline, F::ZERO);

		// Scaling every entry scales the evaluation by the same factor.
		let scale = F::random(&mut rng);
		assert_eq!(eval_with_outer(&outer.map(|weight| weight * scale)), baseline * scale);

		// Zeroing the table kills the evaluation, so no term slipped past the outer factor.
		assert_eq!(eval_with_outer(&[F::ZERO; SHIFT_COUNT]), F::ZERO);

		// The identity-selecting table is what a term with no outer shift is weighed by, and it
		// reproduces what a reduction over one shift slot would give the same terms.
		let mut identity_selecting = [F::ZERO; SHIFT_COUNT];
		identity_selecting[0] = F::ONE;
		let singly_shifted_only = constraints
			.iter()
			.map(|constraint| {
				AndConstraint(std::array::from_fn(|operand| {
					constraint.0[operand]
						.iter()
						.filter(|svi| !svi.is_doubly_shifted())
						.copied()
						.collect()
				}))
			})
			.collect::<Vec<_>>();
		assert_eq!(
			eval_with_outer(&identity_selecting),
			OperationEvalFn::new(&singly_shifted_only).call_native(run_weights(
				&r_x_prime_tensor,
				&inner_operand,
				&identity_selecting,
				r_y_tensor
			))
		);
	}

	#[test]
	fn contracting_the_tensor_matches_the_batched_evaluation() {
		// Invariant: the flat walk and the constraint-grouped evaluation are one sum, reassociated.
		//
		// The grouped path multiplies a constraint's weight in once for the whole constraint.
		// The flat path multiplies it in per term.
		// Reassociating is exact, so a disagreement means the two read different nonzeros.
		//
		// This pins the enumeration's index conventions against the path already in use.
		type F = Ghash128b;
		let mut rng = StdRng::seed_from_u64(11);

		let n_words = 40usize;
		// A power-of-two count, and one whose weight tensor runs past the last constraint.
		for n_constraints in [64usize, 37] {
			let constraints = random_and_constraints(&mut rng, n_constraints, n_words);
			let arity = constraints[0].as_ref().len();

			let r_x_prime = random_scalars::<F>(&mut rng, log2_ceil_usize(n_constraints));
			let lambda = F::random(&mut rng);
			let inner: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
			let outer: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
			let hidden = random_scalars::<F>(&mut rng, n_words);
			let value = [&[][..], &[][..], &hidden[..]];

			let eval_fn = OperationEvalFn::new(&constraints);

			// Both paths read the same weights, one table per axis.
			let r_x_prime_tensor = eq_ind_partial_eval_scalars(&r_x_prime);
			let inner_operand = operand_shift_scalar_table(&inner, &lambda, arity);
			let weights = run_weights(&r_x_prime_tensor, &inner_operand, &outer, value);

			let batched = eval_fn.call::<F>(weights);
			let contracted = eval_fn.contract(weights);

			// A vacuous agreement on zero would prove nothing about which nonzeros were read.
			assert_ne!(batched, F::ZERO, "the fixture must evaluate to something");
			assert_eq!(contracted, batched, "n_constraints = {n_constraints}");
		}
	}

	#[test]
	fn the_extension_at_a_boolean_point_counts_the_nonzeros_there() {
		// Invariant: indicators selecting one index per axis return the entry at that position.
		//
		// That is the defining property of a multilinear extension.
		// It is also what the accumulated claim at the root of a tree is about.
		//
		// Entries live in characteristic two, so an entry is the nonzero count there, mod two.
		//
		//     indicators pick (c, a, inner, outer, v)  ->  contract == count(c,a,inner,outer,v) % 2
		type F = Ghash128b;
		let mut rng = StdRng::seed_from_u64(13);

		let n_words = 12usize;
		let n_constraints = 16usize;
		let constraints = random_and_constraints(&mut rng, n_constraints, n_words);
		let arity = constraints[0].as_ref().len();
		let eval_fn = OperationEvalFn::new(&constraints);

		// Every occupied position, plus one deliberately empty one to show the test can fail.
		let occupied = eval_fn.entries().collect::<Vec<_>>();
		assert!(!occupied.is_empty(), "the fixture must produce nonzeros");
		let empty = WiringEntry {
			constraint: n_constraints - 1,
			operand: arity - 1,
			inner_shift: SHIFT_COUNT - 1,
			outer_shift: SHIFT_COUNT - 1,
			value: ValueIndex::private(n_words as u32 - 1),
		};

		for target in occupied.iter().copied().take(8).chain([empty]) {
			// One indicator per axis: one at the target's index, zero everywhere else.
			let mut constraint = vec![F::ZERO; n_constraints];
			constraint[target.constraint] = F::ONE;
			let mut inner_operand = vec![F::ZERO; SHIFT_COUNT * arity];
			inner_operand[target.inner_shift * arity + target.operand] = F::ONE;
			let mut outer = [F::ZERO; SHIFT_COUNT];
			outer[target.outer_shift] = F::ONE;
			let mut hidden = vec![F::ZERO; n_words];
			hidden[target.value.index() as usize] = F::ONE;

			let got = eval_fn.contract(run_weights(
				&constraint,
				&inner_operand,
				&outer,
				[&[][..], &[][..], &hidden[..]],
			));

			// Count the nonzeros sitting exactly at the target position.
			let count = eval_fn.entries().filter(|e| *e == target).count();
			let want = if count % 2 == 1 { F::ONE } else { F::ZERO };
			assert_eq!(got, want, "at {target:?}, {count} nonzeros");
		}
	}

	#[test]
	fn a_repeated_term_cancels_and_the_slots_keep_their_order() {
		// Invariant: entries are characteristic-two multiplicities.
		// And the inner slot is the sequence's first, the outer its second.
		//
		// Fixture state: one constraint, whose first operand names one shifted value twice.
		// Its second operand names a single term with two distinct slots.
		//
		//     operand 0:  [ (v0, [s_a, s_b]) , (v0, [s_a, s_b]) ]  -> two entries at one position
		//     operand 1:  [ (v1, [s_a, s_b]) ]                     -> one entry
		//
		// The pair must contribute nothing, since a position's weight is added to itself.
		type F = Ghash128b;

		let s_a = Shift {
			variant: ShiftVariant::Slr,
			amount: 7,
		};
		let s_b = Shift {
			variant: ShiftVariant::Rotr,
			amount: 3,
		};
		let v0 = ValueIndex::private(0);
		let v1 = ValueIndex::private(1);
		let repeated = ShiftedValueIndex::new(v0, [s_a, s_b]);
		let constraints = vec![AndConstraint([
			vec![repeated, repeated],
			vec![ShiftedValueIndex::new(v1, [s_a, s_b])],
			vec![],
		])];
		let eval_fn = OperationEvalFn::new(&constraints);

		// The enumeration reports the repeat as two entries: cancellation is the field's doing.
		let entries = eval_fn.entries().collect::<Vec<_>>();
		assert_eq!(entries.len(), 3);
		// The first slot of the sequence is the inner one, the second the outer.
		assert_eq!(entries[0].inner_shift, s_a.index());
		assert_eq!(entries[0].outer_shift, s_b.index());
		assert_eq!(entries[0].operand, 0);
		assert_eq!(entries[2].operand, 1);
		assert_eq!(entries[0], entries[1], "the repeat sits at one position");

		// Weights of one everywhere: the sum is then the nonzero count mod two, which is one.
		let ones_inner = vec![F::ONE; SHIFT_COUNT * 3];
		let ones_outer = [F::ONE; SHIFT_COUNT];
		let ones_value = [F::ONE; 2];
		let got = eval_fn.contract(run_weights(
			&[F::ONE],
			&ones_inner,
			&ones_outer,
			[&[][..], &[][..], &ones_value[..]],
		));
		assert_eq!(got, F::ONE, "two entries cancel and the third survives");
	}

	/// The native `WideMul` variant must produce exactly the same result as the generic
	/// evaluation (deferred reduction is `F`-linear). Covers a power-of-two constraint count and a
	/// non-power-of-two one, whose `r_x'` tensor runs past the last constraint.
	#[test]
	fn evaluate_monster_native_matches_generic() {
		type F = Ghash128b;
		let mut rng = StdRng::seed_from_u64(3);

		let n_words = 40usize;
		for n_constraints in [64usize, 37] {
			let constraints = random_and_constraints(&mut rng, n_constraints, n_words);

			let r_x_prime = random_scalars::<F>(&mut rng, log2_ceil_usize(n_constraints));
			let lambda = F::random(&mut rng);
			// Both slots draw random weights, so a path that dropped the outer factor, or read
			// it from the inner table, would disagree with the other.
			let inner: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
			let outer: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
			let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
			let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

			let r_x_prime_tensor = eq_ind_partial_eval_scalars(&r_x_prime);
			let inner_operand =
				operand_shift_scalar_table(&inner, &lambda, constraints[0].as_ref().len());
			let weights = run_weights(&r_x_prime_tensor, &inner_operand, &outer, r_y_tensor);

			let eval_fn = OperationEvalFn::new(&constraints);
			let generic = eval_fn.call::<F>(weights);
			let native = eval_fn.call_native(weights);
			assert_eq!(generic, native, "n_constraints = {n_constraints}");
		}
	}

	/// Appending all-zero padding constraints must not change the evaluation: a padding constraint
	/// has no operand terms, so it contributes nothing. This is what lets the constraint system
	/// keep its true count while the reductions run over the padded one.
	///
	/// [`OperationEvalFn::call`] and [`OperationEvalFn::call_native`] walk the constraints over
	/// independent zips, so both are checked.
	#[test]
	fn evaluate_monster_ignores_zero_padding_constraints() {
		type F = Ghash128b;
		let mut rng = StdRng::seed_from_u64(5);

		let n_words = 40usize;
		let n_constraints = 21usize;
		let constraints = random_and_constraints(&mut rng, n_constraints, n_words);
		let padded = constraints
			.iter()
			.cloned()
			.chain(iter::repeat_n(
				AndConstraint::default(),
				n_constraints.next_power_of_two() - n_constraints,
			))
			.collect::<Vec<_>>();

		let r_x_prime = random_scalars::<F>(&mut rng, log2_ceil_usize(n_constraints));
		let lambda = F::random(&mut rng);
		let inner: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
		let outer: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
		let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
		let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

		let r_x_prime_tensor = eq_ind_partial_eval_scalars(&r_x_prime);
		let inner_operand =
			operand_shift_scalar_table(&inner, &lambda, constraints[0].as_ref().len());
		let weights = run_weights(&r_x_prime_tensor, &inner_operand, &outer, r_y_tensor);

		assert_eq!(
			OperationEvalFn::new(&constraints).call::<F>(weights),
			OperationEvalFn::new(&padded).call::<F>(weights)
		);
		assert_eq!(
			OperationEvalFn::new(&constraints).call_native(weights),
			OperationEvalFn::new(&padded).call_native(weights)
		);
	}
}
