// Copyright 2025-2026 The Binius Developers

//! The materialized layers of a fractional-addition circuit, and the driver that proves them.

use binius_compute::Allocator;
use binius_field::{Field, PackedField};
use binius_ip::fracaddcheck::FracAddEvalClaim;
use binius_math::{FieldBuffer, FieldVec, multilinear::hypercube::Hypercube};
use binius_utils::{
	buffer::VecLike,
	rayon::{
		iter::{IntoParallelIterator, ParallelIterator},
		task_size::{IndexedParallelIteratorExt, WorkPerItem},
	},
};

use super::{LayerProver, fraction::Fraction};
use crate::{
	channel::IPProverChannel,
	sumcheck::{batch::batch_prove_mle, frac_add_mle},
};

/// The materialized layers of a fractional addition circuit.
///
/// Each layer holds the numerator and denominator of one fractional term per node.
/// A layer is half the width of the one below it, each node adding its two children:
/// $$\frac{a_0}{b_0} + \frac{a_1}{b_1} = \frac{a_0b_1 + a_1b_0}{b_0b_1}$$
pub struct FracAddCircuit<'a, A: Allocator, P: PackedField> {
	layers: Vec<Fraction<FieldVec<P, A>>>,
	/// Allocator the layer buffers are drawn from.
	pub(crate) alloc: &'a A,
}

impl<A: Allocator, P: PackedField> Clone for FracAddCircuit<'_, A, P>
where
	A::Vec<P>: Clone,
{
	fn clone(&self) -> Self {
		Self {
			layers: self.layers.clone(),
			alloc: self.alloc,
		}
	}
}

impl<'a, A, F, P> FracAddCircuit<'a, A, P>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	/// Materializes every layer of the circuit from `witness`.
	///
	/// Returns the circuit beside `sums`, its root layer.
	/// `sums` holds the fractional addition over all `k` reduced variables.
	///
	/// # Arguments
	/// * `k` - How many variables to reduce, one sibling fractional addition per step.
	/// * `witness` - The witness numerator/denominator layers
	///
	/// # Preconditions
	/// * `witness.num.log_len() >= k`
	pub fn build(
		k: usize,
		alloc: &'a A,
		witness: Fraction<FieldVec<P, A>>,
	) -> (Self, Fraction<FieldVec<P, A>>) {
		let Fraction {
			num: witness_num,
			den: witness_den,
		} = witness;
		assert_eq!(
			witness_num.log_len(),
			witness_den.log_len(),
			"numerator and denominator witnesses must have equal length"
		);
		assert!(witness_num.log_len() >= k);

		let mut layers = Vec::with_capacity(k + 1);
		layers.push(Fraction::new(witness_num, witness_den));

		for _ in 0..k {
			let prev_layer = layers.last().expect("layers is non-empty");

			let Fraction { num, den } = prev_layer;
			let num_log_len = num.log_len() - 1;
			let den_log_len = den.log_len() - 1;
			let (num_0, num_1) = num.split_half();
			let (den_0, den_1) = den.split_half();

			// One packed word of the next layer from the sibling halves, written straight into
			// the pooled buffers:
			//     a_0/b_0 + a_1/b_1 = (a_0*b_1 + a_1*b_0) / (b_0*b_1)
			// Workers each take a contiguous run of words.
			// One word is three multiplies and an add, a few nanoseconds of work.
			// A run must therefore be long enough to pay back handing it off.
			let out_len = num_0.as_ref().len();
			let mut num_data = alloc.alloc::<P>(out_len);
			let mut den_data = alloc.alloc::<P>(out_len);
			(
				num_data.spare_capacity_mut(),
				den_data.spare_capacity_mut(),
				num_0.as_ref(),
				den_0.as_ref(),
				num_1.as_ref(),
				den_1.as_ref(),
			)
				.into_par_iter()
				.with_min_task(WorkPerItem::FieldMuls)
				.for_each(|(num_out, den_out, &num_0, &den_0, &num_1, &den_1)| {
					num_out.write(num_0 * den_1 + num_1 * den_0);
					den_out.write(den_0 * den_1);
				});
			// Invariant: every zip input holds at least `out_len` words.
			//
			// A parallel zip yields as many items as its shortest input holds.
			// A shorter input would leave trailing slots uninitialized.
			//
			//     spare capacity:  >= out_len   allocated for at least that many
			//     sibling halves:  == out_len   halves of two equal-length buffers
			debug_assert!(
				num_data.capacity() - num_data.len() >= out_len
					&& den_data.capacity() - den_data.len() >= out_len,
				"allocated buffers must hold every claimed slot"
			);
			debug_assert!(
				[den_0.as_ref(), num_1.as_ref(), den_1.as_ref()]
					.iter()
					.all(|half| half.len() == out_len),
				"the four sibling halves must hold exactly one word per claimed slot"
			);
			// Safety: both length claims cover only initialized slots.
			// - The assertions above bound every zip input below by `out_len`.
			// - So the loop ran `out_len` items.
			// - Each item wrote one numerator slot and one denominator slot.
			unsafe {
				num_data.set_len(out_len);
				den_data.set_len(out_len);
			}
			let next_layer = Fraction::new(
				FieldBuffer::new(num_log_len, num_data),
				FieldBuffer::new(den_log_len, den_data),
			);

			layers.push(next_layer);
		}

		let sums = layers.pop().expect("layers has k+1 elements");
		(Self { layers, alloc }, sums)
	}

	/// Returns the number of remaining layers to prove.
	pub const fn n_layers(&self) -> usize {
		self.layers.len()
	}

	/// Pops the widest remaining layer as the MLE-check prover that reduces it.
	///
	/// The returned prover owns the popped buffers and borrows only the allocator.
	/// So it outlives this borrow, and the circuit stays in place while a caller drives it.
	///
	/// # Preconditions
	/// * `self.n_layers() >= 1`
	pub fn pop_layer(&mut self, claim: FracAddEvalClaim<F>) -> LayerProver<'a, A, F, P> {
		let Fraction { num, den } = self
			.layers
			.pop()
			.expect("precondition: self.n_layers() >= 1");

		// The MLE-check reduces four multilinears: the low and high halves of the numerator buffer
		// and of the denominator buffer. The store takes ownership of the two popped buffers and
		// shares each between its halves, so the prover is self-contained with no up-front copy of
		// the popped layer.
		frac_add_mle::new_split_half(
			self.alloc,
			num,
			den,
			claim.point,
			[claim.num_eval, claim.den_eval],
		)
	}

	/// Runs the fractional addition check protocol and returns the final evaluation claims.
	///
	/// This consumes the circuit, reducing from the smallest layer back to the largest.
	///
	/// # Arguments
	/// * `claim` - The numerator and denominator claims at their shared evaluation point.
	/// * `channel` - The channel for sending prover messages and sampling challenges.
	///
	/// # Preconditions
	/// * `claim.point.len() == witness.log_len() - k`, for `k` the number of reduction layers.
	pub fn prove(
		self,
		claim: FracAddEvalClaim<F>,
		channel: &mut impl IPProverChannel<F>,
	) -> FracAddEvalClaim<F> {
		// Proving the full circuit runs every layer, so delegate and drop the leftover circuit.
		let n_layers = self.n_layers();
		let (remaining, claim) = self.prove_layers(n_layers, claim, channel);
		debug_assert_eq!(remaining.n_layers(), 0, "proving every layer leaves none unproved");
		claim
	}

	/// Runs the first `n_layers` fractional-addition layers from a claim, returning the remainder.
	///
	/// Each layer adds one variable via a sumcheck and a line-fold.
	/// So starting from a claim over `d` variables, the returned claim is over `d + n_layers`.
	///
	/// This is the layer loop of [`Self::prove`], which runs every layer.
	/// The returned circuit still holds the layers that were not proved.
	///
	/// # Arguments
	/// * `n_layers` - The number of layers to prove, at most [`Self::n_layers`].
	/// * `claim` - The numerator and denominator claims at their shared evaluation point.
	/// * `channel` - The channel for sending prover messages and sampling challenges.
	///
	/// # Returns
	/// * the circuit, holding whatever layers were not proved,
	/// * the reduced numerator/denominator claims after `n_layers` layers.
	///
	/// # Preconditions
	/// * `n_layers <= self.n_layers()`.
	fn prove_layers(
		mut self,
		n_layers: usize,
		claim: FracAddEvalClaim<F>,
		channel: &mut impl IPProverChannel<F>,
	) -> (Self, FracAddEvalClaim<F>) {
		let mut claim = claim;

		for _ in 0..n_layers {
			let sumcheck_prover = self.pop_layer(claim);

			// The driver draws the batching coefficient and Horner-folds the layer's two claims,
			// which is the polynomial the verifier's `batch_verify_mle` reconstructs.
			let output = batch_prove_mle(vec![sumcheck_prover], channel);
			output.send_evals(channel);

			let mut multilinear_evals = output.multilinear_evals;
			let evals = multilinear_evals.pop().expect("batch contains one prover");

			let [num_0, num_1, den_0, den_1] = evals
				.try_into()
				.expect("prover evaluates four multilinears");

			// Fold the highest variable to combine the two halves into the next layer's claim.
			let r = channel.sample();

			let next_num = Hypercube::One.fold_var(num_0, num_1, &r);
			let next_den = Hypercube::One.fold_var(den_0, den_1, &r);

			// Sumcheck binds variables high-to-low; reverse to low-to-high for the claim point.
			let mut next_point = output.challenges;
			next_point.reverse();
			next_point.push(r);

			claim = FracAddEvalClaim {
				num_eval: next_num,
				den_eval: next_den,
				point: next_point,
			};
		}

		(self, claim)
	}
}

#[cfg(test)]
mod tests {
	use std::iter;

	use binius_compute::GlobalAllocator;
	use binius_ip::fracaddcheck;
	use binius_math::{
		multilinear::Multilinear,
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	type StdChallenger = HasherChallenger<sha2::Sha256>;

	fn test_frac_add_check_prove_verify_helper<P: PackedField>(n: usize, k: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// 1. Create random witness with log_len = n + k
		let witness_num = random_field_buffer::<P>(&mut rng, n + k);
		let witness_den = random_field_buffer::<P>(&mut rng, n + k);

		// 2. Create prover (computes fractional-add layers)
		let (prover, sums) = FracAddCircuit::build(
			k,
			&alloc,
			Fraction::new(witness_num.clone(), witness_den.clone()),
		);

		// 3. Generate random n-dimensional challenge point
		let eval_point = random_scalars::<P::Scalar>(&mut rng, n);

		// 4. Evaluate sums at challenge point to create claims
		let sum_num_eval = sums.num.evaluate(&eval_point);
		let sum_den_eval = sums.den.evaluate(&eval_point);
		// The prover and the verifier take the same claim type, so one claim serves both.
		let claim = FracAddEvalClaim {
			num_eval: sum_num_eval,
			den_eval: sum_den_eval,
			point: eval_point,
		};

		// 5. Run prover
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let prover_output = prover.prove(claim.clone(), &mut prover_transcript);

		// 6. Run verifier
		let mut verifier_transcript = prover_transcript.into_verifier();
		let verifier_output = fracaddcheck::verify(k, claim, &mut verifier_transcript).unwrap();

		// 7. Check outputs match
		assert_eq!(prover_output, verifier_output);

		// 8. Verify multilinear evaluation of original witness
		let expected_num = witness_num.evaluate(&verifier_output.point);
		let expected_den = witness_den.evaluate(&verifier_output.point);
		assert_eq!(verifier_output.num_eval, expected_num);
		assert_eq!(verifier_output.den_eval, expected_den);
	}

	#[test]
	fn test_frac_add_check_prove_verify() {
		test_frac_add_check_prove_verify_helper::<Packed128b>(4, 3);
	}

	#[test]
	fn test_frac_add_check_full_prove_verify() {
		test_frac_add_check_prove_verify_helper::<Packed128b>(0, 4);
	}

	fn check_all_layers<P: PackedField>(n: usize, k: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let alloc = GlobalAllocator;

		// Create random witness with log_len = n + k
		let witness_num = random_field_buffer::<P>(&mut rng, n + k);
		let witness_den = random_field_buffer::<P>(&mut rng, n + k);

		// Create prover (computes fractional-add layers)
		let (prover, sums) = FracAddCircuit::build(
			k,
			&alloc,
			Fraction::new(witness_num.clone(), witness_den.clone()),
		);

		// `build` pops the root off as `sums`, so the circuit is `layers` followed by it.
		for (j, layer) in prover.layers.iter().chain(iter::once(&sums)).enumerate() {
			// Entry i of layer j is the fractional sum of the 2^j witness values strided by that
			// layer's own width (strided access, not contiguous).
			let width = 1 << (n + k - j);
			let num_terms = 1 << j;
			for i in 0..width {
				let mut expected_num = witness_num.get(i);
				let mut expected_den = witness_den.get(i);
				for z in 1..num_terms {
					let idx = i + z * width;
					let num_z = witness_num.get(idx);
					let den_z = witness_den.get(idx);
					expected_num = expected_num * den_z + num_z * expected_den;
					expected_den *= den_z;
				}
				let actual_num = layer.num.get(i);
				let actual_den = layer.den.get(i);
				assert_eq!(actual_num, expected_num, "layer {j} numerator mismatch at index {i}");
				assert_eq!(actual_den, expected_den, "layer {j} denominator mismatch at index {i}");
			}
		}
	}

	proptest! {
		// Invariant: every layer of the circuit is the fractional-addition fold of the witness.
		//
		// Pinning each layer to that fold pins the sibling recurrence the layers are built from.
		// Only an end-to-end proof failure notices if `build` folds the wrong pairs.
		#[test]
		fn frac_add_check_layers_fold_the_witness(
			seed in any::<u64>(),
			n in 0usize..=4,
			k in 0usize..=4,
		) {
			check_all_layers::<Packed128b>(n, k, seed);
		}
	}
}
