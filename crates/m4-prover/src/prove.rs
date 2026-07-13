// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::PackedField;
use binius_hash::StdHashSuite;
use binius_iop_prover::{basefold_compiler::BaseFoldProverCompiler, channel::IOPProverChannel};
use binius_m4_verifier::Verifier;
use binius_math::ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded};
use binius_prover::{
	protocols::shift::{KeyCollection, build_key_collection},
	ring_switch::{self, RingSwitchOutput},
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_verifier::config::B128;

use crate::{
	ValueTable,
	reduction::{ReductionProverOutput, prove_reduction},
};

/// The multithreaded additive NTT used to encode the committed codeword.
type ProverNtt = NeighborsLastMultiThread<GenericPreExpanded<B128>>;

/// Proves the data-parallel M4 statement for a batch of `2^log_instances` circuit instances.
///
/// One-time setup builds the shift keys and the BaseFold prover, reusing the verifier's parameters.
/// A later proving call commits a witness table and proves it satisfies every AND constraint.
pub struct Prover<P>
where
	P: PackedField<Scalar = B128>,
{
	/// The prepared single-instance constraint system shared by every instance.
	cs: ConstraintSystem,
	/// The shift keys for the constraint system, built once and reused across proofs.
	key_collection: KeyCollection,
	/// The precomputed BaseFold prover, holding the NTT and the FRI parameters.
	basefold_compiler: BaseFoldProverCompiler<P, ProverNtt>,
}

impl<P> Prover<P>
where
	P: PackedField<Scalar = B128>,
{
	/// Builds the prover from a verifier, inheriting its constraint system and FRI parameters.
	///
	/// The prover encodes the codeword with the multithreaded NTT, spread across the cores.
	/// Reusing the verifier's compiler keeps both sides on one set of FRI parameters.
	pub fn setup(verifier: &Verifier) -> Self {
		// Reuse the verifier's evaluation domain so both sides agree on the code.
		let domain_context =
			GenericPreExpanded::generate_from_subspace(verifier.iop_compiler().max_subspace());

		// Spread the NTT across the available cores.
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		// Inherit the verifier's oracle specs and FRI parameters verbatim.
		let basefold_compiler =
			BaseFoldProverCompiler::from_verifier_compiler(verifier.iop_compiler(), ntt);

		// Build the shift keys once from the shared constraint system.
		let key_collection = build_key_collection(verifier.constraint_system());

		Self {
			cs: verifier.constraint_system().clone(),
			key_collection,
			basefold_compiler,
		}
	}

	/// Proves that every instance in the batch satisfies the constraint system.
	///
	/// The flow composes the commitment, the reduction, and the opening on one transcript:
	/// - Pack the table into one B128 multilinear and commit it as the trace oracle.
	/// - Run the AND-check and shift reduction to a claim about the instance-folded witness.
	/// - Ring-switch that claim onto the committed trace and open it.
	///
	/// The trace commits before the reduction draws its challenges.
	/// So Fiat-Shamir binds every challenge to the committed data.
	///
	/// The reduction ends with a claim about the witness folded over instances at `r_rho`.
	/// The trace's bit index is `[bit | instance | wire]`.
	/// Evaluating its instance coordinates at `r_rho` performs that fold.
	/// So the ring-switch opens the trace at `r_j || r_rho || r_y`, matching the reduced claim.
	///
	/// The trace oracle is not ZK, so the channel masks nothing and needs no randomness.
	///
	/// # Panics
	///
	/// Panics if the constraint system has any MUL constraints.
	pub fn prove<Challenger_>(
		&self,
		table: &ValueTable,
		transcript: &mut ProverTranscript<Challenger_>,
	) where
		Challenger_: Challenger,
	{
		let mut channel = self
			.basefold_compiler
			.create_channel_without_zk_from_transcript::<StdHashSuite, Challenger_, _>(transcript);

		// Pack the 2-D table into one multilinear and commit it as the trace oracle.
		let packed = table.pack::<P>();
		let trace_oracle = channel.send_oracle(packed.to_ref());

		// Reduce the AND constraints and shift to one folded-witness claim.
		// Every challenge is drawn now, after the commitment.
		let ReductionProverOutput {
			r_rho,
			witness_claim,
		} = prove_reduction::<P, _>(&self.cs, &self.key_collection, table, &mut channel);

		// Split the shift's final point `r_j || r_y || r_segment` into its three parts.
		// The bit index `r_j` is the low coordinates addressing a bit within a 64-bit word.
		// The segment selector `r_segment` is the last coordinate, choosing public or hidden words.
		// The hidden-only trace drops it.
		// The word index `r_y` is everything in between.
		let challenges = &witness_claim.challenges;
		let r_j = &challenges[..Word::LOG_BITS];
		let r_y = &challenges[Word::LOG_BITS..challenges.len() - 1];

		// Ring-switch the reduced claim onto the committed trace.
		// The point is `r_j || r_rho || r_y`.
		// Its instance coordinates fold the trace at `r_rho`.
		let trace_point = [r_j, r_rho.as_slice(), r_y].concat();
		let RingSwitchOutput {
			rs_eq_ind,
			sumcheck_claim,
		} = ring_switch::prove(&packed, &trace_point, &mut channel);

		// Queue the trace opening against the ring-switch's transparent multilinear.
		// The final call runs the single combined FRI opening and writes it to the transcript.
		channel.prove_oracle_relations([(trace_oracle, packed, rs_eq_ind, sumcheck_claim)]);
		channel.finish();
	}
}

#[cfg(test)]
mod tests {
	use std::array;

	use assert_matches::assert_matches;
	use binius_circuits::hash_based_sig::{
		winternitz_ots::{NONCE_WIRES_COUNT, WinternitzSpec},
		witness_utils::ValidatorSignatureData,
		xmss::{XmssSignature, circuit_xmss},
	};
	use binius_field::PackedBinaryGhash1x128b;
	use binius_frontend::{CircuitBuilder, Wire};
	use binius_iop::{
		basefold::{Error as BaseFoldError, VerificationError as BaseFoldVerificationError},
		channel::Error as IOPChannelError,
		fri::VerificationError as FriVerificationError,
		merkle_tree::VerificationError as MerkleVerificationError,
	};
	use binius_transcript::VerifierTranscript;
	use binius_verifier::{Error, config::StdChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::{
		BatchWitnessFiller,
		test_utils::{N_INPUT_WORDS, crc64_circuit, populate_crc64_witness},
	};

	// Packs little-endian bytes into 64-bit word wires, zero-filling any trailing wires.
	fn pack_bytes_le(w: &mut BatchWitnessFiller<'_, '_>, wires: &[Wire], bytes: &[u8]) {
		// Each wire takes the next 8 bytes, little-endian; a short final chunk is zero-extended.
		for (&wire, chunk) in wires.iter().zip(bytes.chunks(8)) {
			let mut word = [0u8; 8];
			word[..chunk.len()].copy_from_slice(chunk);
			w[wire] = Word(u64::from_le_bytes(word));
		}
		// Any wire past the packed bytes is zeroed.
		for &wire in &wires[bytes.len().div_ceil(8)..] {
			w[wire] = Word::ZERO;
		}
	}

	type P = PackedBinaryGhash1x128b;

	// Builds a batch of `2^log_instances` CRC-64 instances with random input words.
	fn setup_batch(log_instances: usize, seed: u64) -> (ConstraintSystem, ValueTable) {
		let c = crc64_circuit();
		let n_instances = 1usize << log_instances;
		let mut rng = StdRng::seed_from_u64(seed);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		(cs, table)
	}

	// The prover and verifier run the whole protocol on one transcript.
	// A faithful proof over 64 instances verifies and leaves no trailing data.
	#[test]
	fn protocol_round_trips() {
		let log_instances = 6;
		let (cs, table) = setup_batch(log_instances, 0);

		// Setup once: the verifier fixes the shape and FRI parameters.
		// The prover inherits them.
		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		// Prover: commit, reduce, and open on a fresh transcript.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);

		// Verifier: replay the same transcript end to end.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// Invariant: a batch of full XMSS signature verifications proves and verifies through M4.
	//
	// XMSS verification recomputes a Merkle root from a signature and asserts it equals the
	// committed root. The signature is a Winternitz one-time signature: one hash chain per
	// codeword coordinate, whose chain ends hash into a Merkle leaf, plus an authentication path
	// from that leaf to the root. Every hash is BLAKE3, so the circuit is MUL-free, as M4 requires.
	//
	// Fixture: spec_1 (72 Winternitz chains of length 4), a height-2 tree (4 epochs), 2^3 = 8
	// instances.
	#[test]
	fn xmss_batch_round_trips() {
		let spec = WinternitzSpec::spec_1();
		let tree_height = 2;
		let log_instances = 3;

		// Allocate every circuit input as a witness wire, so the circuit has no inout wires.
		// M4 forbids inout, so the public data (parameter, message, root) becomes witness too.
		// The verification still asserts the recomputed root equals the committed root.
		// So no gate is pruned even though nothing is a public output.
		let builder = CircuitBuilder::new();
		// The per-signer domain parameter is a byte string; each wire holds 8 of its bytes.
		let param: Vec<Wire> = (0..spec.domain_param_len.div_ceil(8))
			.map(|_| builder.add_witness())
			.collect();
		// The message digest is 32 bytes across four 64-bit wires.
		let message: Vec<Wire> = (0..4).map(|_| builder.add_witness()).collect();
		// The committed Merkle root is 32 bytes across four wires.
		let root_hash: [Wire; 4] = array::from_fn(|_| builder.add_witness());
		// The nonce feeds the message hash and fills four wires exactly.
		let nonce: Vec<Wire> = (0..NONCE_WIRES_COUNT)
			.map(|_| builder.add_witness())
			.collect();
		// The epoch is the signing leaf index within the tree.
		let epoch = builder.add_witness();
		// One chain value per Winternitz coordinate (the signature), each a four-wire digest.
		let signature_hashes: Vec<[Wire; 4]> = (0..spec.dimension())
			.map(|_| array::from_fn(|_| builder.add_witness()))
			.collect();
		// One chain end per coordinate (the one-time public key), each a four-wire digest.
		let public_key_hashes: Vec<[Wire; 4]> = (0..spec.dimension())
			.map(|_| array::from_fn(|_| builder.add_witness()))
			.collect();
		// One authentication-path node per tree level.
		let auth_path: Vec<[Wire; 4]> = (0..tree_height)
			.map(|_| array::from_fn(|_| builder.add_witness()))
			.collect();
		// Assemble the signature and emit the verification constraints.
		let signature = XmssSignature {
			nonce: nonce.clone(),
			epoch,
			signature_hashes: signature_hashes.clone(),
			public_key_hashes: public_key_hashes.clone(),
			auth_path: auth_path.clone(),
		};
		circuit_xmss(&builder, &spec, &param, &message, &signature, &root_hash);
		let circuit = builder.build();

		// Generate one valid signature.
		// This runs the nonce grind and builds the Merkle tree, so it is one-time setup here.
		let mut rng = StdRng::seed_from_u64(0);
		// A random per-signer domain parameter.
		let mut param_bytes = vec![0u8; spec.domain_param_len];
		rng.fill_bytes(&mut param_bytes);
		// A random 32-byte message.
		let mut message_bytes = [0u8; 32];
		rng.fill_bytes(&mut message_bytes);
		// A signing epoch inside the tree: any leaf index below 2^tree_height = 4.
		let sig_epoch = rng.next_u32() % (1u32 << tree_height);
		// Sign: derive the chain values, chain ends, Merkle leaf, and authentication path.
		let data = ValidatorSignatureData::generate(
			&mut rng,
			&param_bytes,
			&message_bytes,
			sig_epoch,
			&spec,
			tree_height,
		);

		// Fill all 8 instances with that one signature.
		// Proving is data-independent, so identical instances measure the same work as distinct
		// ones, and one signature suffices.
		let table = ValueTable::populate(&circuit, log_instances, |_, w| {
			// The public inputs, folded into the witness.
			pack_bytes_le(w, &param, &param_bytes);
			pack_bytes_le(w, &message, &message_bytes);
			pack_bytes_le(w, &root_hash, &data.root);
			// The signature's nonce and epoch.
			pack_bytes_le(w, &nonce, &data.nonce);
			w[epoch] = Word::from_u64(sig_epoch as u64);
			// One chain value per coordinate.
			for (dst, src) in signature_hashes.iter().zip(&data.signature_hashes) {
				pack_bytes_le(w, dst, src);
			}
			// One chain end per coordinate.
			for (dst, src) in public_key_hashes.iter().zip(&data.public_key_hashes) {
				pack_bytes_le(w, dst, src);
			}
			// One authentication-path node per tree level.
			for (dst, src) in auth_path.iter().zip(&data.auth_path) {
				pack_bytes_le(w, dst, src);
			}
		})
		.unwrap();

		// Prepare the shared constraint system, which fixes the public-segment layout.
		let mut cs = circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();

		// Setup once: the verifier fixes the shape and FRI parameters; the prover inherits them.
		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		// Prove: commit the batch witness, reduce, and open, on a fresh transcript.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);

		// Verify: replay the transcript end to end; it must accept and leave no trailing data.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify(&mut verifier_transcript)
			.expect("a faithful XMSS proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// Tampering with the trace opening breaks the final FRI check.
	#[test]
	fn tampered_opening_is_rejected() {
		let log_instances = 6;
		let (cs, table) = setup_batch(log_instances, 1);

		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		// Produce a faithful proof, then collect its bytes.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Flip one bit in the last byte, which lands in a FRI query's Merkle opening.
		// The opening no longer matches the committed root, so BaseFold verification rejects it.
		let last = proof.len() - 1;
		proof[last] ^= 1;

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let err = verifier.verify(&mut verifier_transcript).unwrap_err();
		assert_matches!(
			err,
			Error::IOPChannel(IOPChannelError::BaseFold(BaseFoldError::Verification(
				BaseFoldVerificationError::FRI(FriVerificationError::MerkleError(
					MerkleVerificationError::InvalidProof
				))
			)))
		);
	}
}
