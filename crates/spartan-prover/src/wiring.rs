// Copyright 2025 Irreducible Inc.

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, FieldSlice, multilinear::eq::eq_ind_partial_eval};
use binius_spartan_frontend::constraint_system::{
	MulConstraint, Operand, WitnessIndex, WitnessSegment,
};
use binius_utils::{checked_arithmetics::checked_log_2, rayon::prelude::*};

/// Transpose of the private wiring sparse matrix.
///
/// Only indexes private witness wires. The private segment's size determines
/// the buffer dimensions.
#[derive(Debug)]
pub struct WiringTranspose {
	flat_keys: Vec<Key>,
	keys_start_by_private_index: Vec<u32>,
	log_private_size: usize,
}

#[derive(Debug, Clone)]
pub struct Key {
	pub operand_idx: u8,
	pub constraint_idx: u32,
}

impl WiringTranspose {
	pub fn transpose(private_size: usize, mul_constraints: &[MulConstraint<WitnessIndex>]) -> Self {
		let mut keys_by_private_idx = vec![Vec::new(); private_size];

		let mut n_total_keys = 0;
		for (i, MulConstraint { a, b, c }) in mul_constraints.iter().enumerate() {
			for (operand_idx, operand) in [a, b, c].into_iter().enumerate() {
				for &witness_idx in operand.wires() {
					if witness_idx.segment == WitnessSegment::Private {
						keys_by_private_idx[witness_idx.index as usize].push(Key {
							operand_idx: operand_idx as u8,
							constraint_idx: i as u32,
						});
						n_total_keys += 1;
					}
				}
			}
		}

		// Flatten the sparse matrix representation.
		let mut flat_keys = Vec::with_capacity(n_total_keys);
		let mut keys_start = Vec::with_capacity(private_size);
		for keys in keys_by_private_idx {
			let start = flat_keys.len() as u32;
			flat_keys.extend(keys);
			keys_start.push(start);
		}

		let log_private_size = checked_log_2(private_size);

		Self {
			flat_keys,
			keys_start_by_private_index: keys_start,
			log_private_size,
		}
	}

	/// Returns the log2 of the private segment size.
	pub fn log_private_size(&self) -> usize {
		self.log_private_size
	}

	/// Returns the private segment size.
	pub fn private_size(&self) -> usize {
		1 << self.log_private_size
	}

	/// Returns the keys for a specific private witness index.
	pub fn keys_for_private(&self, private_idx: usize) -> &[Key] {
		let start = self.keys_start_by_private_index[private_idx] as usize;
		let end = self
			.keys_start_by_private_index
			.get(private_idx + 1)
			.map(|&x| x as usize)
			.unwrap_or(self.flat_keys.len());
		&self.flat_keys[start..end]
	}
}

/// Folds the private wiring matrix along the constraint axis by partially evaluating at r_x.
///
/// Also batches the three operands (a, b, c) using powers of lambda.
/// Returns a multilinear polynomial over private witness indices where each coefficient is the
/// weighted sum of constraint contributions.
pub fn fold_constraints<F: Field, P: PackedField<Scalar = F>>(
	transposed: &WiringTranspose,
	lambda: F,
	r_x: &[F],
) -> FieldBuffer<P> {
	// Compute eq indicator tensor for constraint evaluation points
	let r_x_tensor = eq_ind_partial_eval::<F>(r_x);

	// Batching powers for the three operands
	let lambda_powers = [F::ONE, lambda, lambda.square()];

	let private_size = transposed.private_size();
	let log_private_size = transposed.log_private_size();
	let len = 1 << log_private_size.saturating_sub(P::LOG_WIDTH);

	// Process in parallel over chunks of P::WIDTH private indices
	let result = (0..len)
		.into_par_iter()
		.map(|packed_idx| {
			let base_idx = packed_idx << P::LOG_WIDTH;

			P::from_fn(|scalar_idx| {
				let private_idx = base_idx + scalar_idx;
				if private_idx >= private_size {
					return F::ZERO;
				}

				let mut acc = F::ZERO;
				for key in transposed.keys_for_private(private_idx) {
					let r_x_weight = r_x_tensor[key.constraint_idx as usize];
					let lambda_weight = lambda_powers[key.operand_idx as usize];
					acc += r_x_weight * lambda_weight;
				}
				acc
			})
		})
		.collect::<Vec<_>>();

	FieldBuffer::new(log_private_size, result.into_boxed_slice())
}

/// Witness data for multiplication constraint checking.
///
/// Contains the evaluated operands a, b, and c for all multiplication constraints,
/// packed into field buffers for efficient processing.
pub struct MulCheckWitness<P: PackedField> {
	pub a: FieldBuffer<P>,
	pub b: FieldBuffer<P>,
	pub c: FieldBuffer<P>,
}

/// Evaluates an operand by XORing witness values at the specified indices.
fn eval_operand<F: Field, P: PackedField<Scalar = F>>(
	public: &[F],
	private_packed: &FieldSlice<P>,
	operand: &Operand<WitnessIndex>,
) -> F {
	operand
		.wires()
		.iter()
		.map(|idx| match idx.segment {
			WitnessSegment::Public => public[idx.index as usize],
			WitnessSegment::Private => private_packed.get(idx.index as usize),
		})
		.sum()
}

/// Builds the witness for multiplication constraint checking.
///
/// Extracts and packs the a, b, and c operand values for each multiplication constraint.
/// This is analogous to `build_bitand_witness` in binius-prover but works with B128
/// field elements instead of word-level operations.
#[tracing::instrument(skip_all, level = "debug")]
pub fn build_mulcheck_witness<F: Field, P: PackedField<Scalar = F>>(
	mul_constraints: &[MulConstraint<WitnessIndex>],
	public: &[F],
	private_packed: FieldSlice<P>,
) -> MulCheckWitness<P> {
	fn get_a(c: &MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex> {
		&c.a
	}
	fn get_b(c: &MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex> {
		&c.b
	}
	fn get_c(c: &MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex> {
		&c.c
	}

	let n_constraints = mul_constraints.len();
	assert!(n_constraints > 0, "mul_constraints must not be empty");

	let log_n_constraints = checked_log_2(n_constraints);

	let len = 1 << log_n_constraints.saturating_sub(P::LOG_WIDTH);
	let mut a = Vec::<P>::with_capacity(len);
	let mut b = Vec::<P>::with_capacity(len);
	let mut c = Vec::<P>::with_capacity(len);

	(a.spare_capacity_mut(), b.spare_capacity_mut(), c.spare_capacity_mut())
		.into_par_iter()
		.enumerate()
		.for_each(|(i, (a_i, b_i, c_i))| {
			let offset = i << P::LOG_WIDTH;

			for (dst, get_operand) in [
				(a_i, get_a as fn(&MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex>),
				(b_i, get_b as fn(&MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex>),
				(c_i, get_c as fn(&MulConstraint<WitnessIndex>) -> &Operand<WitnessIndex>),
			] {
				let val = P::from_fn(|j| {
					let constraint_idx = offset + j;
					if constraint_idx < n_constraints {
						eval_operand(
							public,
							&private_packed,
							get_operand(&mul_constraints[constraint_idx]),
						)
					} else {
						F::ZERO
					}
				});
				dst.write(val);
			}
		});

	// Safety: all entries in a, b, c are initialized in the parallel loop above.
	unsafe {
		a.set_len(len);
		b.set_len(len);
		c.set_len(len);
	}

	MulCheckWitness {
		a: FieldBuffer::new(log_n_constraints, a.into_boxed_slice()),
		b: FieldBuffer::new(log_n_constraints, b.into_boxed_slice()),
		c: FieldBuffer::new(log_n_constraints, c.into_boxed_slice()),
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128bGhash as B128, Field, Random};
	use binius_math::{
		multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
		test_utils::{Packed128b, random_scalars},
		univariate::evaluate_univariate,
	};
	use binius_spartan_frontend::constraint_system::{MulConstraint, Operand, WitnessIndex};
	use binius_spartan_verifier::wiring::evaluate_private_wiring_mle;
	use rand::{Rng, SeedableRng, rngs::StdRng};
	use smallvec::SmallVec;

	use super::*;

	/// Generate random MulConstraints for testing.
	/// Each operand has 0-4 random wires.
	fn generate_random_constraints(
		rng: &mut StdRng,
		n_constraints: usize,
		public_size: usize,
		private_size: usize,
	) -> Vec<MulConstraint<WitnessIndex>> {
		(0..n_constraints)
			.map(|_| {
				let a = generate_random_operand(rng, public_size, private_size);
				let b = generate_random_operand(rng, public_size, private_size);
				let c = generate_random_operand(rng, public_size, private_size);
				MulConstraint { a, b, c }
			})
			.collect()
	}

	fn generate_random_operand(
		rng: &mut StdRng,
		public_size: usize,
		private_size: usize,
	) -> Operand<WitnessIndex> {
		let total = public_size + private_size;
		let n_wires = rng.random_range(0..=4);
		let wires: SmallVec<[WitnessIndex; 4]> = (0..n_wires)
			.map(|_| {
				let flat = rng.random_range(0..total);
				if flat < public_size {
					WitnessIndex::public(flat as u32)
				} else {
					WitnessIndex::private((flat - public_size) as u32)
				}
			})
			.collect();
		Operand::new(wires)
	}

	/// Evaluate the private wiring MLE using the transposed representation.
	fn evaluate_private_wiring_mle_transposed<F: Field>(
		transposed: &WiringTranspose,
		lambda: F,
		r_x_tensor: &[F],
		r_y_tensor: &[F],
	) -> F {
		let mut acc = [F::ZERO; 3];

		for private_idx in 0..transposed.private_size() {
			let r_y_weight = r_y_tensor[private_idx];
			for key in transposed.keys_for_private(private_idx) {
				let r_x_weight = r_x_tensor[key.constraint_idx as usize];
				acc[key.operand_idx as usize] += r_x_weight * r_y_weight;
			}
		}

		evaluate_univariate(&acc, lambda)
	}

	#[test]
	fn test_wiring_transpose_equivalence() {
		let mut rng = StdRng::seed_from_u64(0);

		let log_n_constraints = 4;
		let log_public = 3;
		let log_private = 5;

		let n_constraints = 1 << log_n_constraints;
		let public_size = 1 << log_public;
		let private_size = 1 << log_private;

		let constraints =
			generate_random_constraints(&mut rng, n_constraints, public_size, private_size);

		// Sample random evaluation points
		let r_x = random_scalars::<B128>(&mut rng, log_n_constraints);
		let r_y = random_scalars::<B128>(&mut rng, log_private);
		let lambda = B128::random(&mut rng);

		// Compute expected result using the verifier's reference implementation
		let expected = evaluate_private_wiring_mle(&constraints, lambda, &r_x, &r_y);

		// Compute result using the transposed representation
		let transposed = WiringTranspose::transpose(private_size, &constraints);
		let r_x_tensor = eq_ind_partial_eval::<B128>(&r_x);
		let r_y_tensor = eq_ind_partial_eval::<B128>(&r_y);
		let actual = evaluate_private_wiring_mle_transposed(
			&transposed,
			lambda,
			r_x_tensor.as_ref(),
			r_y_tensor.as_ref(),
		);

		assert_eq!(actual, expected, "Transposed evaluation does not match original evaluation");
	}

	#[test]
	fn test_fold_constraints_equivalence() {
		let mut rng = StdRng::seed_from_u64(0);

		let log_n_constraints = 4;
		let log_public = 3;
		let log_private = 5;

		let n_constraints = 1 << log_n_constraints;
		let public_size = 1 << log_public;
		let private_size = 1 << log_private;

		let constraints =
			generate_random_constraints(&mut rng, n_constraints, public_size, private_size);

		// Sample random evaluation points
		let r_x = random_scalars::<B128>(&mut rng, log_n_constraints);
		let r_y = random_scalars::<B128>(&mut rng, log_private);
		let lambda = B128::random(&mut rng);

		// Method 1: Compute expected result using evaluate_private_wiring_mle
		let expected = evaluate_private_wiring_mle(&constraints, lambda, &r_x, &r_y);

		// Method 2: Use fold_constraints then evaluate at r_y
		let transposed = WiringTranspose::transpose(private_size, &constraints);
		let folded = fold_constraints::<_, Packed128b>(&transposed, lambda, &r_x);
		let actual = evaluate(&folded, &r_y);

		assert_eq!(
			actual, expected,
			"fold_constraints + evaluate does not match evaluate_private_wiring_mle"
		);
	}

	#[test]
	fn test_wiring_prove_verify() {
		use binius_hash::StdDigest;
		use binius_iop::{
			channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
			naive_channel::NaiveVerifierChannel,
		};
		use binius_iop_prover::{channel::IOPProverChannel, naive_channel::NaiveProverChannel};
		use binius_ip::channel::IPVerifierChannel;
		use binius_ip_prover::channel::IPProverChannel;
		use binius_math::{inner_product::inner_product_buffers, test_utils::random_field_buffer};
		use binius_spartan_verifier::wiring::evaluate_wiring_mle_public;
		use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

		type StdChallenger = HasherChallenger<StdDigest>;

		let mut rng = StdRng::seed_from_u64(0);

		// Parameters
		let log_n_constraints = 4;
		let log_public = 3;
		let log_private = 5;

		let n_constraints = 1 << log_n_constraints;
		let public_size = 1 << log_public;
		let private_size = 1 << log_private;

		// Generate random constraints
		let constraints =
			generate_random_constraints(&mut rng, n_constraints, public_size, private_size);

		// Create random public and private witness buffers
		let public = random_scalars::<B128>(&mut rng, public_size);
		let private_buf = random_field_buffer::<Packed128b>(&mut rng, log_private);

		// Compute mulcheck witness
		let mulcheck_witness = build_mulcheck_witness(&constraints, &public, private_buf.to_ref());

		// Sample r_x (sumcheck evaluation point for constraint axis)
		let r_x = random_scalars::<B128>(&mut rng, log_n_constraints);

		// Compute mulcheck evaluations at r_x
		let r_x_tensor = eq_ind_partial_eval::<Packed128b>(&r_x);
		let mulcheck_evals = [
			inner_product_buffers(&mulcheck_witness.a, &r_x_tensor),
			inner_product_buffers(&mulcheck_witness.b, &r_x_tensor),
			inner_product_buffers(&mulcheck_witness.c, &r_x_tensor),
		];

		// Create transposed wiring
		let wiring_transpose = WiringTranspose::transpose(private_size, &constraints);

		let oracle_specs = vec![OracleSpec {
			log_msg_len: log_private,
		}];

		// === PROVER SIDE ===
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel = NaiveProverChannel::<B128, Packed128b, _>::new(
			&mut prover_transcript,
			oracle_specs.clone(),
		);

		// Send private witness oracle
		let witness_oracle = prover_channel.send_oracle(private_buf.to_ref());

		// Sample lambda
		let lambda: B128 = prover_channel.sample();

		// Compute the batched sum and public contribution
		let batched_sum = evaluate_univariate(&mulcheck_evals, lambda);
		let public_eval =
			evaluate_wiring_mle_public(&constraints, log_public, &public, lambda, &r_x);
		let trace_claim = batched_sum - public_eval;

		// Fold constraints to get the private wiring polynomial
		let wiring_poly = fold_constraints::<_, Packed128b>(&wiring_transpose, lambda, &r_x);

		// Finish the IOP with the oracle relation
		prover_channel.prove_oracle_relations([(witness_oracle, wiring_poly.clone(), trace_claim)]);

		// === VERIFIER SIDE ===
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<B128, _>::new(&mut verifier_transcript, &oracle_specs);

		// Receive witness oracle
		let witness_oracle = verifier_channel
			.recv_oracle()
			.expect("recv_oracle should succeed");

		// Sample the same lambda as prover
		let verifier_lambda: B128 = verifier_channel.sample();

		// Compute the same claim on the verifier side
		let verifier_batched_sum = evaluate_univariate(&mulcheck_evals, verifier_lambda);
		let verifier_public_eval =
			evaluate_wiring_mle_public(&constraints, log_public, &public, verifier_lambda, &r_x);
		let verifier_trace_claim = verifier_batched_sum - verifier_public_eval;

		// Verify that prover and verifier computed the same trace_claim
		assert_eq!(trace_claim, verifier_trace_claim, "Prover and verifier trace_claim mismatch");

		// Build the transparent closure using the prover's wiring_poly for evaluation.
		let transparent = Box::new(move |point: &[_]| evaluate(&wiring_poly, point));

		// Finish verification.
		verifier_channel
			.verify_oracle_relations([OracleLinearRelation {
				oracle: witness_oracle,
				transparent,
				claim: verifier_trace_claim,
			}])
			.expect("verify_oracle_relations should succeed (inner product verified)");
	}
}
