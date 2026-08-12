// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_core::{constraint_system::Operand, word::Word};
use binius_field::{
	BinaryField, FieldOps, WideMul,
	util::{FieldFn, powers},
};
use binius_math::multilinear::eq::eq_ind_partial_eval_scalars;
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};

use super::SHIFT_VARIANT_COUNT;

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

	/// Splits the flat [`FieldFn`] input into `(r_x_prime, lambda, shift_scalars, r_y_tensor)`.
	///
	/// The `r_x'` section has `ceil(log2(constraints.len()))` entries — the reductions run over the
	/// constraint count rounded up to a power of two — so the split needs no state beyond the
	/// constraints.
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
	) -> (&'i [E], &'i E, &'i [E; SHIFT_VARIANT_COUNT * Word::BITS], [&'i [E]; 3]) {
		let n_vars = log2_ceil_usize(self.constraints.len());
		let (r_x_prime, rest) = input.split_at(n_vars);
		let (lambda, rest) = rest.split_first().expect("input encodes lambda");
		let (shift_scalars, rest) = rest.split_at(SHIFT_VARIANT_COUNT * Word::BITS);
		let shift_scalars = shift_scalars
			.try_into()
			.expect("input encodes the shift scalars");

		let (constants, rest) = rest.split_at(self.n_constants);
		let (inout, rest) = rest.split_at(self.n_inout);
		let (hidden, _) = rest.split_at(self.n_hidden);

		(r_x_prime, lambda, shift_scalars, [constants, inout, hidden])
	}
}

impl<F, C, const ARITY: usize> FieldFn<F> for OperationEvalFn<'_, C, ARITY>
where
	F: BinaryField,
	C: AsRef<[Operand; ARITY]> + Sync,
{
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, input: &[E]) -> E {
		let (r_x_prime, lambda, shift_scalars, r_y_tensor) = self.split_input(input);

		let r_x_prime_tensor = eq_ind_partial_eval_scalars(r_x_prime);
		let operand_shift_scalars =
			operand_shift_scalar_table(shift_scalars, lambda.clone(), ARITY);

		// Accumulate one contribution per constraint. Within a constraint, each shifted-value term
		// over all operands is weighted by its operand shift scalar and the word-index tensor
		// entry; the running sum is then scaled by the constraint-index tensor entry. The tensor
		// covers the padded constraint count, so the zip stops at the last real constraint; the
		// padding rows have no operand terms and contribute nothing.
		let mut eval = E::zero();
		for (constraint, r_x_prime_entry) in iter::zip(self.constraints, &r_x_prime_tensor) {
			let mut constraint_eval = E::zero();
			for (operand_id, operand) in constraint.as_ref().iter().enumerate() {
				for svi in operand {
					assert!(!svi.is_doubly_shifted(), "{DOUBLE_SHIFT_UNSUPPORTED}");
					let variant = svi.inner().variant as usize;
					let index =
						(variant * Word::BITS + svi.inner().amount as usize) * ARITY + operand_id;
					constraint_eval += operand_shift_scalars[index].clone()
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

		let r_x_prime_tensor = eq_ind_partial_eval_scalars(r_x_prime);
		let operand_shift_scalars = operand_shift_scalar_table(shift_scalars, *lambda, ARITY);

		// One unreduced wide product per constraint. The constraints partition cleanly across
		// rayon: each produces a single wide element and they are summed, so there is no large
		// per-task accumulator. The single final reduction is `F`-linear. The tensor covers the
		// padded constraint count, so the zip stops at the last real constraint; the padding rows
		// have no operand terms and contribute nothing.
		let eval = self
			.constraints
			.par_iter()
			.zip(r_x_prime_tensor.par_iter())
			.map(|(constraint, &r_x_prime_entry)| {
				let mut constraint_eval = F::ZERO;
				for (operand_id, operand) in constraint.as_ref().iter().enumerate() {
					for svi in operand {
						assert!(!svi.is_doubly_shifted(), "{DOUBLE_SHIFT_UNSUPPORTED}");
						let variant = svi.inner().variant as usize;
						let index = (variant * Word::BITS + svi.inner().amount as usize) * ARITY
							+ operand_id;
						constraint_eval += operand_shift_scalars[index]
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
/// Concatenates `r_x_prime ++ [lambda] ++ shift_scalars ++ r_y_tensor`.
/// `OperationEvalFn::split_input` is the inverse; it recovers the `r_x'` length from the constraint
/// count, so only `lambda` and the fixed-length shift scalars need a known position.
pub fn encode_operation_input<E: Clone>(
	r_x_prime: &[E],
	lambda: E,
	shift_scalars: &[E; SHIFT_VARIANT_COUNT * Word::BITS],
	r_y_tensor: [&[E]; 3],
) -> Vec<E> {
	let n_words = r_y_tensor
		.iter()
		.map(|segment| segment.len())
		.sum::<usize>();
	let mut input = Vec::with_capacity(r_x_prime.len() + 1 + shift_scalars.len() + n_words);
	input.extend_from_slice(r_x_prime);
	input.push(lambda);
	input.extend_from_slice(shift_scalars);
	// One run per value segment, in `ValueSegment` order, which is how `split_input` cuts them
	// back apart.
	for segment in r_y_tensor {
		input.extend_from_slice(segment);
	}
	input
}

/// Folds the operand batching coefficients (λ powers) into the shared shift scalars, producing a
/// table indexed by `(variant, amount, operand_id)` whose entry is
/// `shift_scalars[variant * Word::BITS + amount] · λ^{operand_id + 1}` — the scalar that
/// multiplies each shifted-value term.
fn operand_shift_scalar_table<E: FieldOps>(
	shift_scalars: &[E; SHIFT_VARIANT_COUNT * Word::BITS],
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
	};
	use binius_field::{BinaryField128bGhash, Random};
	use binius_math::test_utils::random_scalars;
	use rand::prelude::*;

	use super::*;

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
			let shift_scalars: [F; SHIFT_VARIANT_COUNT * Word::BITS] =
				std::array::from_fn(|_| F::random(&mut rng));
			let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
			let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

			let eval_fn = OperationEvalFn::new(&constraints, 0, 0, n_words);
			let input = encode_operation_input(&r_x_prime, lambda, &shift_scalars, r_y_tensor);
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
		let shift_scalars: [F; SHIFT_VARIANT_COUNT * Word::BITS] =
			std::array::from_fn(|_| F::random(&mut rng));
		let hidden_tensor = random_scalars::<F>(&mut rng, n_words);
		let r_y_tensor = [&[][..], &[][..], &hidden_tensor[..]];

		let input = encode_operation_input(&r_x_prime, lambda, &shift_scalars, r_y_tensor);
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
