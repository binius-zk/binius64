// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_core::constraint_system::Operand;
use binius_field::{
	BinaryField, FieldOps, WideMul,
	util::{FieldFn, powers},
};
use binius_math::multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars};
use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	rayon::{
		prelude::*,
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};

use super::SHIFT_COUNT;

/// Why a term's outer shift slot must hold the identity in the two-phase reduction.
///
/// A shift key names one shift, so the reduction reads the inner slot and ignores the outer one.
/// A term carrying both would verify against the wrong shifted word.
const DOUBLE_SHIFT_UNSUPPORTED: &str =
	"the two-phase shift reduction reads only the inner shift of a term";

/// A [`FieldFn`] evaluating one operation's monster multilinear polynomial.
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
/// The `FieldFn` input is the flat slice built by [`encode_operation_input`]: the constraint
/// challenge `r_x'`, then the batching coefficient `lambda`, then the shared shift scalars, then
/// the word-index tensor `r_y`. [`FieldFn::call`] evaluates generically over any `E`;
/// [`FieldFn::call_native`] takes the `WideMul`-accelerated base-field path.
pub struct OperationEvalFn<'a, C, const ARITY: usize> {
	/// The operation's constraints, each exposing its `ARITY` operands as an array in storage
	/// order.
	constraints: &'a [C],
	/// The number of constants the constraints may name.
	n_constants: usize,
	/// The number of inout values the constraints may name.
	n_inout: usize,
	/// The number of private values the constraints may name.
	n_hidden: usize,
}

impl<'a, C, const ARITY: usize> OperationEvalFn<'a, C, ARITY> {
	/// Wraps an operation's constraints for monster-multilinear evaluation.
	///
	/// The three counts are the segment lengths of the system holding the constraints. They are
	/// what the word-index tensor is cut along when the input is split back apart, so
	/// [`encode_operation_input`] must be given runs of matching lengths.
	pub const fn new(
		constraints: &'a [C],
		n_constants: usize,
		n_inout: usize,
		n_hidden: usize,
	) -> Self {
		Self {
			constraints,
			n_constants,
			n_inout,
			n_hidden,
		}
	}

	/// Splits the flat [`FieldFn`] input into its sections.
	///
	/// The `r_x'` section has `ceil(log2(constraints.len()))` entries — the reductions run over the
	/// constraint count rounded up to a power of two — so the split needs no state beyond the
	/// constraints.
	///
	/// Two shift-scalar tables follow, one per slot of a term's shift sequence.
	/// Each is [`SHIFT_COUNT`] entries wide; see [`ShiftScalars`] for why the weight splits that
	/// way.
	///
	/// The word-index tensor arrives as one run per value segment, in
	/// [`ValueSegment`](binius_core::constraint_system::ValueSegment) order, and
	/// comes back as an array indexed by that segment. An operand term is then read at
	/// `r_y_tensor[segment][index]`, which is the term's own `(segment, index)` pair — no address
	/// arithmetic in between. The runs hold only the words a constraint can name, so the padding
	/// between sections never reaches the input.
	fn split_input<'i, E>(
		&self,
		input: &'i [E],
	) -> (&'i [E], &'i E, ShiftScalars<'i, E>, [&'i [E]; 3]) {
		let n_vars = log2_ceil_usize(self.constraints.len());
		let (r_x_prime, rest) = input.split_at(n_vars);
		let (lambda, rest) = rest.split_first().expect("input encodes lambda");
		let (inner, rest) = rest.split_at(SHIFT_COUNT);
		let (outer, rest) = rest.split_at(SHIFT_COUNT);
		let shift_scalars = ShiftScalars {
			inner: inner
				.try_into()
				.expect("input encodes the inner shift scalars"),
			outer: outer
				.try_into()
				.expect("input encodes the outer shift scalars"),
		};

		let (constants, rest) = rest.split_at(self.n_constants);
		let (inout, rest) = rest.split_at(self.n_inout);
		let (hidden, _) = rest.split_at(self.n_hidden);

		(r_x_prime, lambda, shift_scalars, [constants, inout, hidden])
	}
}

/// The shift-sequence weight tables, one per slot of the sequence.
///
/// A term's sequence weight factorizes across its two slots:
///
/// ```text
/// eq(r_v1, v_1) * eq(r_s1, s_1)  *  eq(r_v2, v_2) * eq(r_s2, s_2)
/// \_______ inner table _______/     \_______ outer table ______/
/// ```
///
/// One table per slot holds the weights at `2 * SHIFT_COUNT` = 1,024 entries.
/// Keying a single table on the whole sequence would need `SHIFT_COUNT^2` = 262,144.
/// Fanning that out over the operand batching coefficients reaches roughly 1.5M multiplications at
/// BMUL's arity of six.
///
/// The cost of the split is one extra multiply per term.
pub struct ShiftScalars<'a, E> {
	/// The weight of each spelling the inner shift slot can take, with the operand batching
	/// coefficients yet to be fanned in.
	pub inner: &'a [E; SHIFT_COUNT],
	/// The weight of each spelling the outer shift slot can take.
	pub outer: &'a [E; SHIFT_COUNT],
}

// Two shared slices copy freely whatever `E` is. Deriving these would demand `E: Copy`, which the
// generic evaluation path does not have.
impl<E> Clone for ShiftScalars<'_, E> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<E> Copy for ShiftScalars<'_, E> {}

impl<F, C, const ARITY: usize> FieldFn<F> for OperationEvalFn<'_, C, ARITY>
where
	F: BinaryField,
	C: AsRef<[Operand; ARITY]> + Sync,
{
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, input: &[E]) -> E {
		let (r_x_prime, lambda, shift_scalars, r_y_tensor) = self.split_input(input);

		let r_x_prime_tensor = eq_ind_partial_eval_scalars(r_x_prime);
		// The batching coefficients fan into the inner table only, holding it to
		// `SHIFT_COUNT * arity`; the outer weight multiplies in per term.
		let operand_shift_scalars =
			operand_shift_scalar_table(shift_scalars.inner, lambda.clone(), ARITY);

		// One contribution per constraint.
		// Each term is weighted by its two slots' shift scalars and its word-index tensor entry.
		// The running sum then scales by the constraint-index tensor entry.
		//
		// The tensor covers the padded constraint count, so the zip stops at the last real
		// constraint; padding rows carry no operand terms and contribute nothing.
		let mut eval = E::zero();
		for (constraint, r_x_prime_entry) in iter::zip(self.constraints, &r_x_prime_tensor) {
			let mut constraint_eval = E::zero();
			for (operand_id, operand) in constraint.as_ref().iter().enumerate() {
				for svi in operand {
					assert!(!svi.is_doubly_shifted(), "{DOUBLE_SHIFT_UNSUPPORTED}");
					let inner = svi.inner().index() * ARITY + operand_id;
					let outer = svi.outer().index();
					constraint_eval += operand_shift_scalars[inner].clone()
						* &shift_scalars.outer[outer]
						* &r_y_tensor[svi.value_index.segment() as usize]
							[svi.value_index.index() as usize];
				}
			}
			eval += constraint_eval * r_x_prime_entry;
		}

		eval
	}

	/// Native fast path over the base field `F`.
	///
	/// Produces the identical result, but defers the `GF(2^128)` reductions: the per-constraint
	/// contributions accumulate into a single *unreduced* wide element, reduced exactly once at the
	/// end (reduction is `F`-linear, so this equals reducing each per-constraint product and
	/// summing). The generic [`call`](FieldFn::call) can't do this because `E: FieldOps` does not
	/// imply `WideMul`.
	fn call_native(&self, input: &[F]) -> F {
		let (r_x_prime, lambda, shift_scalars, r_y_tensor) = self.split_input(input);

		// The packed expansion threads the tensor's multiplications.
		// It applies over the base field, which is its own single-element packing.
		let r_x_prime_tensor = eq_ind_partial_eval::<F>(r_x_prime);
		let operand_shift_scalars = operand_shift_scalar_table(shift_scalars.inner, *lambda, ARITY);

		// One unreduced wide product per constraint. The constraints partition cleanly across
		// rayon: each produces a single wide element and they are summed, so there is no large
		// per-task accumulator. The single final reduction is `F`-linear. The tensor covers the
		// padded constraint count, so the zip stops at the last real constraint; the padding rows
		// have no operand terms and contribute nothing.
		//
		// A constraint names only a handful of terms.
		// So a minimum task size keeps each task above rayon's own handoff cost.
		let eval = self
			.constraints
			.par_iter()
			.zip(r_x_prime_tensor.as_ref().par_iter())
			.with_min_task(WorkPerItem::FieldMuls)
			.map(|(constraint, &r_x_prime_entry)| {
				let mut constraint_eval = F::ZERO;
				for (operand_id, operand) in constraint.as_ref().iter().enumerate() {
					for svi in operand {
						assert!(!svi.is_doubly_shifted(), "{DOUBLE_SHIFT_UNSUPPORTED}");
						let inner = svi.inner().index() * ARITY + operand_id;
						let outer = svi.outer().index();
						constraint_eval += operand_shift_scalars[inner]
							* shift_scalars.outer[outer]
							* r_y_tensor[svi.value_index.segment() as usize]
								[svi.value_index.index() as usize];
					}
				}
				F::wide_mul(constraint_eval, r_x_prime_entry)
			})
			.sum::<<F as WideMul>::Output>();
		F::reduce(eval)
	}
}

/// Builds the flat [`FieldFn`] input consumed by [`OperationEvalFn`].
///
/// Concatenates `r_x_prime ++ [lambda] ++ inner_shift_scalars ++ outer_shift_scalars ++
/// r_y_tensor`. `OperationEvalFn::split_input` is the inverse; it recovers the `r_x'` length from
/// the constraint count, so only `lambda` and the two fixed-length shift-scalar tables need a known
/// position.
pub fn encode_operation_input<E: Clone>(
	r_x_prime: &[E],
	lambda: E,
	shift_scalars: ShiftScalars<'_, E>,
	r_y_tensor: [&[E]; 3],
) -> Vec<E> {
	let n_words = r_y_tensor
		.iter()
		.map(|segment| segment.len())
		.sum::<usize>();
	let mut input = Vec::with_capacity(r_x_prime.len() + 1 + 2 * SHIFT_COUNT + n_words);
	input.extend_from_slice(r_x_prime);
	input.push(lambda);
	// The inner table leads the outer one, which is the order `split_input` cuts them back apart.
	input.extend_from_slice(shift_scalars.inner);
	input.extend_from_slice(shift_scalars.outer);
	// One run per value segment, in `ValueSegment` order, which is how `split_input` cuts them
	// back apart.
	for segment in r_y_tensor {
		input.extend_from_slice(segment);
	}
	input
}

/// Folds the operand batching coefficients (λ powers) into the inner slot's shift scalars,
/// producing a table indexed by `(variant, amount, operand_id)` whose entry is
/// `inner[variant * Word::BITS + amount] · λ^{operand_id + 1}`.
///
/// The fan-out stays on this one table.
/// A term's outer-slot weight multiplies in where the term is read.
/// So the table is `SHIFT_COUNT * arity` entries rather than `SHIFT_COUNT^2 * arity`.
fn operand_shift_scalar_table<E: FieldOps>(
	shift_scalars: &[E; SHIFT_COUNT],
	lambda: E,
	arity: usize,
) -> Vec<E> {
	let lambda_powers = powers(lambda).skip(1).take(arity).collect::<Vec<_>>();
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
	use binius_field::{BinaryField128bGhash, Field, Random};
	use binius_math::test_utils::random_scalars;
	use rand::prelude::*;

	use super::{super::SHIFT_VARIANT_COUNT, *};

	/// Builds `n_constraints` random arity-3 constraints (like `AndConstraint`), constraint-major:
	/// one array of operands per constraint.
	///
	/// Every term names a private word, so the constant and inout runs of the word-index tensor
	/// stay empty.
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
		(0..n_constraints)
			.map(|_| {
				AndConstraint(std::array::from_fn(|_| {
					(0..rng.random_range(0..=3))
						.map(|_| {
							// The reduction reads the inner slot, so the fixture leaves the outer
							// one at the identity.
							ShiftedValueIndex::single(
								ValueIndex::private(rng.random_range(0..n_words) as u32),
								Shift {
									variant: shift_variants
										[rng.random_range(0..SHIFT_VARIANT_COUNT)],
									amount: rng.random_range(0..Word::BITS) as u8,
								},
							)
						})
						.collect()
				}))
			})
			.collect()
	}

	#[test]
	fn evaluate_monster_scales_by_the_outer_slot_weight() {
		// Invariant: the outer slot's weight reaches the evaluation, and reaches it as a factor.
		//
		// Every term the fixture builds carries an identity outer shift, so index 0 is the only
		// entry read. Scaling it must scale the whole evaluation by the same factor.
		type F = BinaryField128bGhash;
		let mut rng = StdRng::seed_from_u64(7);

		let n_words = 40usize;
		let constraints = random_and_constraints(&mut rng, 32, n_words);
		let r_x_prime = random_scalars::<F>(&mut rng, 5);
		let lambda = F::random(&mut rng);
		let inner: [F; SHIFT_COUNT] = std::array::from_fn(|_| F::random(&mut rng));
		let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
		let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

		let eval_fn = OperationEvalFn::new(&constraints, 0, 0, n_words);
		let eval_with_outer = |outer: &[F; SHIFT_COUNT]| {
			let shift_scalars = ShiftScalars {
				inner: &inner,
				outer,
			};
			let input = encode_operation_input(&r_x_prime, lambda, shift_scalars, r_y_tensor);
			eval_fn.call_native(&input)
		};

		// The identity-selecting table, which is what the two-phase reduction supplies.
		let mut identity_selecting = [F::ZERO; SHIFT_COUNT];
		identity_selecting[0] = F::ONE;
		let baseline = eval_with_outer(&identity_selecting);
		// A non-degenerate fixture, or the scaling below proves nothing.
		assert_ne!(baseline, F::ZERO);

		// Scaling the entry every term reads scales the evaluation by the same factor.
		let scale = F::random(&mut rng);
		let mut scaled = [F::ZERO; SHIFT_COUNT];
		scaled[0] = scale;
		assert_eq!(eval_with_outer(&scaled), baseline * scale);

		// Zeroing it kills the evaluation, so no term slipped past the outer factor.
		assert_eq!(eval_with_outer(&[F::ZERO; SHIFT_COUNT]), F::ZERO);

		// The weight of a shift the fixture never names must not enter. Only index 0 is read, so
		// filling every other entry changes nothing.
		let mut noise = [F::random(&mut rng); SHIFT_COUNT];
		noise[0] = F::ONE;
		assert_eq!(eval_with_outer(&noise), baseline);
	}

	/// The native `WideMul` variant must produce exactly the same result as the generic
	/// evaluation (deferred reduction is `F`-linear). Covers a power-of-two constraint count and a
	/// non-power-of-two one, whose `r_x'` tensor runs past the last constraint.
	#[test]
	fn evaluate_monster_native_matches_generic() {
		type F = BinaryField128bGhash;
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
			let shift_scalars = ShiftScalars {
				inner: &inner,
				outer: &outer,
			};
			let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
			let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

			let eval_fn = OperationEvalFn::new(&constraints, 0, 0, n_words);
			let input = encode_operation_input(&r_x_prime, lambda, shift_scalars, r_y_tensor);
			let generic = eval_fn.call::<F>(&input);
			let native = eval_fn.call_native(&input);
			assert_eq!(generic, native, "n_constraints = {n_constraints}");
		}
	}

	/// Appending all-zero padding constraints must not change the evaluation: a padding constraint
	/// has no operand terms, so it contributes nothing. This is what lets the constraint system
	/// keep its true count while the reductions run over the padded one.
	///
	/// [`FieldFn::call`] and [`FieldFn::call_native`] walk the constraints over independent zips,
	/// so both are checked.
	#[test]
	fn evaluate_monster_ignores_zero_padding_constraints() {
		type F = BinaryField128bGhash;
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
		let shift_scalars = ShiftScalars {
			inner: &inner,
			outer: &outer,
		};
		let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
		let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

		let input = encode_operation_input(&r_x_prime, lambda, shift_scalars, r_y_tensor);
		assert_eq!(
			OperationEvalFn::new(&constraints, 0, 0, n_words).call::<F>(&input),
			OperationEvalFn::new(&padded, 0, 0, n_words).call::<F>(&input)
		);
		assert_eq!(
			OperationEvalFn::new(&constraints, 0, 0, n_words).call_native(&input),
			OperationEvalFn::new(&padded, 0, 0, n_words).call_native(&input)
		);
	}
}
