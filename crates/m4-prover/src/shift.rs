// Copyright 2025 Irreducible Inc.

//! Prover glue for the M4 batched shift reduction (BitAnd only).
//!
//! The batched BitAnd reduction leaves a claim over the row index `row = kappa * n_and + x`.
//! The row splits into the local constraint index and the instance index:
//!
//! ```text
//!   eval_point = [ x .......... | kappa ...... ]
//!                  low c_and       high k coords
//! ```
//!
//! This splits the point, folds the instance dimension into the witness, and returns `r_kappa`.

use binius_field::PackedField;
use binius_ip_prover::channel::IPProverChannel;
use binius_prover::protocols::shift::{KeyCollection, OperatorData, prove_batch};
use binius_verifier::{config::B128, protocols::bitand::AndCheckOutput};

use crate::ValueTable;

/// Output of the M4 shift reduction proof.
///
/// The committed batch witness is evaluated at `(r_j, r_y, r_kappa)`.
/// The challenge point `[r_j, r_y]` comes from the shift reduction; `r_kappa` from the split.
#[derive(Debug)]
pub struct ShiftReductionOutput {
	/// The shift reduction challenge point `[r_j, r_y]` (bit index, then word index).
	pub challenges: Vec<B128>,
	/// The instance challenge, the high coordinates of the committed-witness evaluation point.
	pub r_kappa: Vec<B128>,
	/// The claimed evaluation of the committed batch witness at `(r_j, r_y, r_kappa)`.
	pub witness_eval: B128,
}

impl ShiftReductionOutput {
	/// Proves the M4 batched shift reduction for the BitAnd operands.
	///
	/// # Arguments
	///
	/// - `table`: the populated batch witness, one committed-word block per instance.
	/// - `key_collection`: the per-instance shift structure, shared by every instance.
	/// - `bitand`: the batched BitAnd reduction output, its point carrying the instance index high.
	/// - `channel`: the prover channel.
	pub fn prove<P, Channel>(
		table: &ValueTable,
		key_collection: &KeyCollection,
		bitand: AndCheckOutput<B128>,
		channel: &mut Channel,
	) -> Self
	where
		P: PackedField<Scalar = B128>,
		Channel: IPProverChannel<B128>,
	{
		let AndCheckOutput {
			a_eval,
			b_eval,
			c_eval,
			z_challenge,
			eval_point,
		} = bitand;

		// Split the row-index point: low = local constraint index, high k coords = instance index.
		let k = table.log_instances();
		let c_and = eval_point.len() - k;
		let (r_x, r_kappa) = eval_point.split_at(c_and);

		let bitand_data = OperatorData {
			evals: vec![a_eval, b_eval, c_eval],
			r_zhat_prime: z_challenge,
			r_x_prime: r_x.to_vec(),
		};

		// One committed-word slice per instance, in instance order.
		let instances = (0..table.n_instances())
			.map(|i| table.instance(i))
			.collect::<Vec<_>>();

		let reduced =
			prove_batch::<B128, P, _>(key_collection, &instances, r_kappa, bitand_data, channel);

		Self {
			challenges: reduced.challenges,
			r_kappa: r_kappa.to_vec(),
			witness_eval: reduced.eval,
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;
	use binius_field::PackedBinaryGhash1x128b;
	use binius_frontend::{Circuit, CircuitBuilder, Wire};
	use binius_math::{
		inner_product::inner_product_buffers,
		multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars},
	};
	use binius_prover::{
		fold_word::fold_words,
		protocols::shift::{OperatorData, build_key_collection, prove_batch},
	};
	use binius_transcript::ProverTranscript;
	use binius_utils::checked_arithmetics::checked_log_2;
	use binius_verifier::config::{B128, LOG_WORD_SIZE_BITS, StdChallenger};
	use proptest::prelude::*;

	use super::*;
	use crate::BatchAndCheckWitness;

	// A width-1 packed field keeps one scalar per element, so the SIMD sumcheck rounds stay simple.
	type P = PackedBinaryGhash1x128b;
	type F = B128;

	// A circuit asserting `z == (x & y) ^ w`, over four public words.
	//
	//     inputs : x, y, w, z   (all inout)
	//     gate   : and = x & y
	//     assert : and ^ w == z
	struct AndCircuit {
		circuit: Circuit,
		x: Wire,
		y: Wire,
		w: Wire,
		z: Wire,
	}

	fn and_circuit() -> AndCircuit {
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let y = builder.add_inout();
		let w = builder.add_inout();
		let z = builder.add_inout();
		let and = builder.band(x, y);
		let lhs = builder.bxor(and, w);
		builder.assert_eq("z_eq_x_and_y_xor_w", lhs, z);
		AndCircuit {
			circuit: builder.build(),
			x,
			y,
			w,
			z,
		}
	}

	// Populate one instance per `(x, y, w)` tuple; the instance count is the tuple count.
	// The output is derived as `z = (x & y) ^ w`, so every tuple satisfies the circuit.
	fn populate_table(c: &AndCircuit, inputs: &[(u64, u64, u64)]) -> ValueTable {
		let log_instances = checked_log_2(inputs.len());
		ValueTable::populate(&c.circuit, log_instances, |i, filler| {
			let (x, y, w) = inputs[i];
			filler[c.x] = Word(x);
			filler[c.y] = Word(y);
			filler[c.w] = Word(w);
			filler[c.z] = Word((x & y) ^ w);
		})
		.unwrap()
	}

	// The honest evaluation of the committed batch witness at (r_j, r_y, r_kappa).
	//
	// The batch witness bit function is folded along the batch by eq(r_kappa, .):
	//
	//     W_tilde(r_j, r_y) = sum_kappa eq(r_kappa, kappa) * eval_kappa(r_j, r_y)
	//
	// where eval_kappa is instance kappa's own bit-witness, folded at r_j and evaluated at r_y.
	fn honest_witness_eval(table: &ValueTable, r_j: &[F], r_y: &[F], r_kappa: &[F]) -> F {
		let r_j_tensor = eq_ind_partial_eval::<F>(r_j);
		let r_y_tensor = eq_ind_partial_eval::<F>(r_y);
		let eps = eq_ind_partial_eval_scalars(r_kappa);

		(0..table.n_instances())
			.map(|kappa| {
				// Fold instance kappa's words at r_j, then evaluate at r_y.
				let mut folded = fold_words::<F, F>(table.instance(kappa), r_j_tensor.as_ref());
				// Match the word-index variable count the reduction padded up to.
				if folded.log_len() < r_y.len() {
					folded.zero_extend(r_y.len());
				}
				eps[kappa] * inner_product_buffers(&folded, &r_y_tensor)
			})
			.sum()
	}

	// A circuit whose AND operands are *shifted* committed words, over three public words.
	//
	//     inputs : x, y, z0, z1   (all inout)
	//     assert : shl(x, 5)  & rotr(y, 13) == z0     (Sll amount 5, Rotr amount 13)
	//     assert : shr(x, 7)  & sar(y, 3)   == z1     (Slr amount 7, Sar amount 3)
	//
	// The shifts fold into the operands, so the reduction exercises four shift variants at
	// non-zero amounts, not just the amount-0 identity.
	struct ShiftedAndCircuit {
		circuit: Circuit,
		x: Wire,
		y: Wire,
		z0: Wire,
		z1: Wire,
	}

	fn shifted_and_circuit() -> ShiftedAndCircuit {
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let y = builder.add_inout();
		let z0 = builder.add_inout();
		let z1 = builder.add_inout();
		let and0 = builder.band(builder.shl(x, 5), builder.rotr(y, 13));
		builder.assert_eq("z0_eq_shl_x_and_rotr_y", and0, z0);
		let and1 = builder.band(builder.shr(x, 7), builder.sar(y, 3));
		builder.assert_eq("z1_eq_shr_x_and_sar_y", and1, z1);
		ShiftedAndCircuit {
			circuit: builder.build(),
			x,
			y,
			z0,
			z1,
		}
	}

	// Populate the shifted circuit; the outputs are derived so every instance is satisfying.
	fn populate_shifted_table(c: &ShiftedAndCircuit, inputs: &[(u64, u64)]) -> ValueTable {
		let log_instances = checked_log_2(inputs.len());
		ValueTable::populate(&c.circuit, log_instances, |i, filler| {
			let (x, y) = inputs[i];
			filler[c.x] = Word(x);
			filler[c.y] = Word(y);
			// z0 = shl(x, 5) & rotr(y, 13); z1 = shr(x, 7) & sar(y, 3).
			filler[c.z0] = Word((x << 5) & y.rotate_right(13));
			filler[c.z1] = Word((x >> 7) & (((y as i64) >> 3) as u64));
		})
		.unwrap()
	}

	// The prepared per-instance constraint system: constraints padded to a power of two.
	fn prepared_cs(circuit: &Circuit) -> binius_core::constraint_system::ConstraintSystem {
		let mut cs = circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		cs
	}

	// Runs the full chain: batched BitAnd reduction, then shift reduction, then verify.
	// Returns the honest witness eval and both sides' claimed evals for the caller to compare.
	fn run_chain(
		table: &ValueTable,
		cs: &binius_core::constraint_system::ConstraintSystem,
	) -> (F, F, F) {
		let key_collection = build_key_collection(cs);

		// Prover: batched BitAnd reduction feeds the shift reduction on one transcript.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let and_witness = BatchAndCheckWitness::build(table, &cs.and_constraints);
		let log_total = checked_log_2(and_witness.a().len());
		let and_output = and_witness.prove::<P, _>(&mut prover_transcript);
		let prove_out = ShiftReductionOutput::prove::<P, _>(
			table,
			&key_collection,
			and_output,
			&mut prover_transcript,
		);

		// Honest evaluation of the committed batch witness at the reduction's point.
		let r_j = &prove_out.challenges[..LOG_WORD_SIZE_BITS];
		let r_y = &prove_out.challenges[LOG_WORD_SIZE_BITS..];
		let honest = honest_witness_eval(table, r_j, r_y, &prove_out.r_kappa);

		// Verifier: replay the same transcript.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let and_output =
			binius_m4_verifier::verify_bitand_reduction(log_total, &mut verifier_transcript)
				.unwrap();
		let verify_out = binius_m4_verifier::ShiftReductionOutput::verify(
			cs,
			table.log_instances(),
			and_output,
			&mut verifier_transcript,
		)
		.unwrap();
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");

		// Both sides must agree on the challenge point and the instance challenge.
		assert_eq!(prove_out.challenges, verify_out.challenges);
		assert_eq!(prove_out.r_kappa, verify_out.r_kappa);

		(honest, prove_out.witness_eval, verify_out.witness_eval)
	}

	#[test]
	fn round_trip_reduces_to_the_honest_batch_evaluation() {
		// Fixture state: K = 4 satisfying instances of `z = (x & y) ^ w`.
		let c = and_circuit();
		let table = populate_table(&c, &[(1, 3, 7), (5, 6, 0), (9, 12, 0xFF), (0xF0, 0x0F, 1)]);
		let cs = prepared_cs(&c.circuit);

		// The reduction round-trips, and the claimed eval is the honest committed-witness eval.
		let (honest, prove_eval, verify_eval) = run_chain(&table, &cs);
		assert_eq!(prove_eval, verify_eval);
		assert_eq!(prove_eval, honest);
	}

	#[test]
	fn shifted_operands_round_trip() {
		// Fixture state: K = 4 satisfying instances whose AND operands are shifted words.
		let c = shifted_and_circuit();
		let table = populate_shifted_table(
			&c,
			&[
				(0x0123456789abcdef, 0xfedcba9876543210),
				(1, 2),
				(0xdead, 0xbeef),
				(u64::MAX, 7),
			],
		);
		let cs = prepared_cs(&c.circuit);

		// Invariant: the reduction must actually exercise non-zero shift amounts.
		// Otherwise the shift-variant paths in the g-build and the monster stay untested.
		let exercises_shift = cs.and_constraints.iter().any(|con| {
			[&con.a, &con.b, &con.c]
				.iter()
				.any(|operand| operand.iter().any(|term| term.amount != 0))
		});
		assert!(exercises_shift, "test circuit must feed shifted operands to the reduction");

		// The prover's flat monster and the verifier's monster evaluation must agree across the
		// exercised variants; a disagreement trips the verifier's closing check.
		let (honest, prove_eval, verify_eval) = run_chain(&table, &cs);
		assert_eq!(prove_eval, verify_eval);
		assert_eq!(prove_eval, honest);
	}

	#[test]
	fn single_instance_batch_round_trips() {
		// Fixture state: log_instances = 0 -> exactly one instance (K = 1), r_kappa is empty.
		//
		// This degenerate batch is the single-instance shift argument on that one instance.
		let c = and_circuit();
		let table = populate_table(&c, &[(0xABCD, 0x0F0F, 0x55)]);
		let cs = prepared_cs(&c.circuit);
		let (honest, prove_eval, verify_eval) = run_chain(&table, &cs);
		assert_eq!(prove_eval, verify_eval);
		assert_eq!(prove_eval, honest);
	}

	#[test]
	#[should_panic(expected = "per-instance committed-word count")]
	fn prove_batch_rejects_short_instance_slice() {
		let c = and_circuit();
		let cs = prepared_cs(&c.circuit);
		let key_collection = build_key_collection(&cs);

		// Fixture state: K = 2 instances, both satisfying.
		let table = populate_table(&c, &[(1, 3, 7), (5, 6, 0)]);
		let full = (0..table.n_instances())
			.map(|i| table.instance(i))
			.collect::<Vec<_>>();

		// Mutation: truncate instance 1 to one word short of the per-instance committed count.
		//
		//     instance 0: [ w_0 .. w_{n-1} ]      (full)
		//     instance 1: [ w_0 .. w_{n-2} ]      (one short) -> fold would read past its end
		let n_words = key_collection.n_words();
		let short = &full[1][..n_words - 1];
		let instances = [full[0], short];

		// A minimal operand claim; the length guard fires before any of it is used.
		let bitand_data = OperatorData {
			evals: vec![B128::default(); 3],
			r_zhat_prime: B128::default(),
			r_x_prime: Vec::new(),
		};
		let r_kappa = [B128::default()];

		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let _ = prove_batch::<B128, P, _>(
			&key_collection,
			&instances,
			&r_kappa,
			bitand_data,
			&mut transcript,
		);
	}

	#[test]
	fn tampered_witness_eval_is_rejected() {
		let c = and_circuit();
		let inputs = [(1, 3, 7), (5, 6, 0), (9, 12, 0xFF), (0xF0, 0x0F, 1)];
		let table = populate_table(&c, &inputs);
		let cs = prepared_cs(&c.circuit);
		let key_collection = build_key_collection(&cs);

		// Produce a faithful proof.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let and_witness = BatchAndCheckWitness::build(&table, &cs.and_constraints);
		let log_total = checked_log_2(and_witness.a().len());
		let and_output = and_witness.prove::<P, _>(&mut prover_transcript);
		let _ = ShiftReductionOutput::prove::<P, _>(
			&table,
			&key_collection,
			and_output,
			&mut prover_transcript,
		);
		let mut proof = prover_transcript.finalize();

		// Mutation: flip a bit in the tail, where the shift reduction wrote the witness evaluation.
		//
		//     The final monster identity `eval == witness_eval * monster_eval` no longer holds.
		let last = proof.len() - 1;
		proof[last] ^= 1;

		let mut verifier_transcript =
			binius_transcript::VerifierTranscript::new(StdChallenger::default(), proof);
		let and_output =
			binius_m4_verifier::verify_bitand_reduction(log_total, &mut verifier_transcript)
				.unwrap();
		let result = binius_m4_verifier::ShiftReductionOutput::verify(
			&cs,
			table.log_instances(),
			and_output,
			&mut verifier_transcript,
		);
		assert!(result.is_err(), "a tampered proof must not verify");
	}

	proptest! {
		// Invariant: for any satisfying batch, the shift reduction round-trips and its claimed eval
		// is the honest evaluation of the committed batch witness at (r_j, r_y, r_kappa).
		#[test]
		fn round_trip_matches_honest_eval(
			inputs in prop::collection::vec((any::<u64>(), any::<u64>()), 8),
		) {
			// The shifted circuit feeds four shift variants at non-zero amounts through the batch.
			let c = shifted_and_circuit();
			let table = populate_shifted_table(&c, &inputs);
			let cs = prepared_cs(&c.circuit);
			let (honest, prove_eval, verify_eval) = run_chain(&table, &cs);
			prop_assert_eq!(prove_eval, verify_eval);
			prop_assert_eq!(prove_eval, honest);
		}
	}
}
