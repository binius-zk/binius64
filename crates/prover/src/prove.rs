// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

use binius_core::{
	constraint_system::{AndConstraint, ConstraintSystem, MulConstraint, ValueVec},
	word::Word,
};
use binius_field::{
	AESTowerField8b as B8, BinaryField, Divisible, ExtensionField, Field, PackedField,
};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop_prover::{basefold_compiler::BaseFoldProverCompiler, channel::IOPProverChannel};
use binius_ip::sumcheck::SumcheckOutput;
use binius_math::{
	BinarySubspace, FieldBuffer, FieldSlice,
	inner_product::inner_product,
	multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
	ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
	univariate::lagrange_evals,
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::{SerializeBytes, checked_arithmetics::checked_log_2, rayon::prelude::*};
use binius_verifier::{
	IOPVerifier, Verifier,
	config::{
		B1, B128, LOG_WORD_SIZE_BITS, LOG_WORDS_PER_ELEM, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES,
	},
	protocols::{bitand::AndCheckOutput, intmul::IntMulOutput},
};
use digest::Output;

use super::error::Error;
use crate::{
	and_reduction::prover::OblongZerocheckProver,
	protocols::{
		intmul::{prove::IntMulProver, witness::Witness as IntMulWitness},
		shift::{
			KeyCollection, OperatorData, build_key_collection, prove as prove_shift_reduction,
		},
	},
	ring_switch,
};

/// Type alias for the prover NTT parameterized by field.
type ProverNTT<F> = NeighborsLastMultiThread<GenericPreExpanded<F>>;

/// IOP prover for a particular constraint system.
///
/// This struct encapsulates the constraint system and pre-computed keys,
/// providing the core proving logic independent of the specific IOP compilation strategy.
/// Most users should use [`Prover`] instead, which wraps this with a BaseFold compiler.
#[derive(Debug)]
pub struct IOPProver {
	constraint_system: ConstraintSystem,
	log_public_words: usize,
	log_witness_elems: usize,
	key_collection: KeyCollection,
}

impl IOPProver {
	/// Constructs an IOP prover from an IOP verifier and pre-computed keys.
	pub fn new(iop_verifier: IOPVerifier, key_collection: KeyCollection) -> Self {
		let log_public_words = iop_verifier.log_public_words();
		let log_witness_elems = iop_verifier.log_witness_elems();
		let constraint_system = iop_verifier.into_constraint_system();
		Self {
			constraint_system,
			log_public_words,
			log_witness_elems,
			key_collection,
		}
	}

	/// Returns the constraint system.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		&self.constraint_system
	}

	/// Returns a reference to the KeyCollection.
	///
	/// This can be used to serialize the KeyCollection for later use.
	pub const fn key_collection(&self) -> &KeyCollection {
		&self.key_collection
	}

	/// Proves using an IOP channel interface.
	///
	/// This is the core proving logic, independent of the specific IOP compilation strategy.
	/// For most users, [`Prover::prove`] is the simpler interface.
	pub fn prove<P, Channel>(&self, witness: ValueVec, channel: &mut Channel) -> Result<(), Error>
	where
		P: PackedField<Scalar = B128>,
		Channel: IOPProverChannel<P>,
	{
		let cs = &self.constraint_system;

		// [phase] Setup - initialization and constraint system setup
		let setup_guard = tracing::debug_span!("Prepare Witness").entered();
		let witness_packed = pack_witness::<P>(self.log_witness_elems, witness.combined_witness())?;
		drop(setup_guard);

		// Observe the public input as B128 elements (includes it in Fiat-Shamir).
		let n_public_elems = 1 << (self.log_public_words - LOG_WORDS_PER_ELEM);
		let public_elems = witness_packed
			.iter_scalars()
			.take(n_public_elems)
			.collect::<Vec<_>>();
		channel.observe_many(&public_elems);

		// [phase] Witness Commit - witness generation and commitment
		let witness_commit_guard = tracing::info_span!("Commit Witness").entered();

		// Commit witness via channel
		let trace_oracle = channel.send_oracle(witness_packed.to_ref());

		drop(witness_commit_guard);

		// [phase] IntMul Reduction - multiplication constraint reduction
		//
		// Skipped entirely (no transcript messages) when the constraint system has no MUL
		// constraints. The verifier applies the identical guard, so the transcript stays in sync;
		// the zero `OperatorData` synthesized below then contributes nothing to the shift
		// reduction.
		let intmul_output = if cs.n_mul_constraints() > 0 {
			let intmul_guard = tracing::info_span!(
				"[phase] IntMul Reduction",
				phase = "intmul_reduction",
				perfetto_category = "phase",
				n_constraints = cs.mul_constraints.len()
			)
			.entered();
			let mul_witness = build_intmul_witness(&cs.mul_constraints, &witness);
			let intmul_output = prove_intmul_reduction::<_, P, _>(mul_witness, &mut *channel)?;
			drop(intmul_guard);
			Some(intmul_output)
		} else {
			None
		};

		// [phase] BitAnd Reduction - AND constraint reduction
		let bitand_guard = tracing::info_span!(
			"[phase] BitAnd Reduction",
			phase = "bitand_reduction",
			perfetto_category = "phase",
			n_constraints = cs.and_constraints.len()
		)
		.entered();
		let bitand_claim = {
			let bitand_witness = build_bitand_witness(&cs.and_constraints, &witness);
			let AndCheckOutput {
				a_eval,
				b_eval,
				c_eval,
				z_challenge,
				eval_point,
			} = prove_bitand_reduction::<B128, P, _>(bitand_witness, &mut *channel);
			OperatorData {
				evals: vec![a_eval, b_eval, c_eval],
				r_zhat_prime: z_challenge,
				r_x_prime: eval_point,
			}
		};
		drop(bitand_guard);

		// Build `OperatorData` for IntMul using the same `r_zhat_prime`
		// challenge as in BitAnd. Sharing this univariate challenge
		// improves ShiftReduction perf. When IntMul was skipped, synthesize a zero claim (four
		// zero evals at an empty point): the shift reduction iterates the (empty) MUL constraints,
		// so this claim contributes zero to its batched evaluation.
		let intmul_claim = match intmul_output {
			Some(IntMulOutput {
				eval_point,
				a_evals,
				b_evals,
				c_lo_evals,
				c_hi_evals,
			}) => {
				let r_zhat_prime = bitand_claim.r_zhat_prime;
				let subspace = BinarySubspace::<B8>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();
				let l_tilde = lagrange_evals(&subspace, r_zhat_prime);
				let make_final_claim = |evals| inner_product(evals, l_tilde.iter_scalars());
				OperatorData {
					evals: vec![
						make_final_claim(a_evals),
						make_final_claim(b_evals),
						make_final_claim(c_lo_evals),
						make_final_claim(c_hi_evals),
					],
					r_zhat_prime,
					r_x_prime: eval_point,
				}
			}
			None => OperatorData {
				evals: vec![B128::ZERO; 4],
				r_zhat_prime: bitand_claim.r_zhat_prime,
				r_x_prime: Vec::new(),
			},
		};

		// [phase] Shift Reduction - shift operations
		let shift_guard = tracing::info_span!(
			"[phase] Shift Reduction",
			phase = "shift_reduction",
			perfetto_category = "phase"
		)
		.entered();
		let SumcheckOutput {
			challenges: eval_point,
			eval: _,
		} = prove_shift_reduction::<_, P, _>(
			&self.key_collection,
			witness.combined_witness(),
			bitand_claim,
			intmul_claim,
			&mut *channel,
		);
		drop(shift_guard);

		// [phase] Ring-Switching + PCS Opening
		let pcs_guard = tracing::info_span!(
			"[phase] PCS Opening",
			phase = "pcs_opening",
			perfetto_category = "phase"
		)
		.entered();

		// Ring-switching reduction
		let ring_switch::RingSwitchOutput {
			rs_eq_ind,
			sumcheck_claim,
		} = ring_switch::prove(&witness_packed, &eval_point, &mut *channel);

		// Public input check batched with ring-switch
		let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;

		let log_public_elems = self.log_public_words - LOG_WORDS_PER_ELEM;
		let pubcheck_point = &eval_point[log_packing..][..log_public_elems];
		let pubcheck_claim = {
			let public_elems_buf = FieldSlice::from_slice(log_public_elems, &public_elems);
			evaluate(&public_elems_buf, pubcheck_point)
		};

		let batch_coeff: B128 = channel.sample();
		let batched_claim = sumcheck_claim + batch_coeff * pubcheck_claim;

		// Batch the pubcheck transparent with the ring-switch transparent
		let batched_transparent =
			compute_batched_transparent(rs_eq_ind, pubcheck_point, batch_coeff);

		// Prove oracle relations via channel (runs BaseFold internally). The intmul pushforward
		// relation, when the IntMul reduction ran, was already queued inside phase 5.
		channel.prove_oracle_relations([(
			trace_oracle,
			witness_packed,
			batched_transparent,
			batched_claim,
		)]);

		drop(pcs_guard);

		Ok(())
	}
}

/// Struct for proving instances of a particular constraint system.
///
/// The [`Self::setup`] constructor pre-processes reusable structures for proving instances of the
/// given constraint system. Then [`Self::prove`] is called one or more times with individual
/// instances.
pub struct Prover<P, H>
where
	P: PackedField<Scalar = B128>,
	H: HashSuite,
{
	iop_prover: IOPProver,
	basefold_compiler: BaseFoldProverCompiler<P, ProverNTT<B128>>,
	/// The prover creates its Merkle transcript channels with the hash suite `H`.
	_hash_marker: PhantomData<H>,
}

impl<P, H> Prover<P, H>
where
	P: PackedField<Scalar = B128>,
	H: HashSuite,
	Output<H::LeafHash>: SerializeBytes,
{
	/// Constructs a prover corresponding to a constraint system verifier.
	///
	/// See [`Prover`] struct documentation for details.
	pub fn setup(verifier: Verifier<H>) -> Result<Self, Error> {
		let key_collection = build_key_collection(verifier.constraint_system());
		Self::setup_with_key_collection(verifier, key_collection)
	}

	/// Constructs a prover with a pre-built KeyCollection.
	///
	/// This allows loading a previously serialized KeyCollection to avoid
	/// the expensive key building phase during setup.
	pub fn setup_with_key_collection(
		verifier: Verifier<H>,
		key_collection: KeyCollection,
	) -> Result<Self, Error> {
		// Get max subspace from verifier's IOP compiler (reuses FRI params)
		let subspace = verifier.iop_compiler().max_subspace();
		let domain_context = GenericPreExpanded::generate_from_subspace(subspace);
		// FIXME TODO For mobile phones, the number of shares should potentially be more than the
		// number of threads, because the threads/cores have different performance (but in the NTT
		// each share has the same amount of work)
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		// Create prover compiler from verifier compiler (reuses FRI params and oracle specs)
		let basefold_compiler =
			BaseFoldProverCompiler::from_verifier_compiler(verifier.iop_compiler(), ntt);

		let iop_prover = IOPProver::new(verifier.into_iop_verifier(), key_collection);

		Ok(Prover {
			iop_prover,
			basefold_compiler,
			_hash_marker: PhantomData,
		})
	}

	/// Returns a reference to the IOP prover.
	pub const fn iop_prover(&self) -> &IOPProver {
		&self.iop_prover
	}

	/// Returns a reference to the KeyCollection.
	///
	/// This can be used to serialize the KeyCollection for later use.
	pub const fn key_collection(&self) -> &KeyCollection {
		self.iop_prover.key_collection()
	}

	pub fn prove<Challenger_: Challenger>(
		&self,
		witness: ValueVec,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		let cs = self.iop_prover.constraint_system();

		let _prove_guard = tracing::info_span!(
			"Prove",
			n_witness_words = cs.value_vec_layout.committed_total_len,
			n_bitand = cs.and_constraints.len(),
			n_intmul = cs.mul_constraints.len(),
		)
		.entered();

		// Create channel, delegate to IOPProver::prove, then finish it. The unified channel takes
		// an rng to mask ZK oracles, but a plain `Prover` produces a transparent proof whose only
		// oracle is non-ZK, so no masks are drawn and the rng is never consumed.
		let mut channel = self
			.basefold_compiler
			.create_channel_without_zk_from_transcript::<H, Challenger_, _>(transcript);
		self.iop_prover.prove::<P, _>(witness, &mut channel)?;
		channel.finish();
		Ok(())
	}
}

/// Batches the pubcheck transparent polynomial with the ring-switch equality indicator.
///
/// Computes `rs_eq_ind + batch_coeff * eq(pubcheck_point || 0, ·)`, adding the scaled
/// pubcheck equality indicator to the first `2^log_public_elems` entries of `rs_eq_ind`.
fn compute_batched_transparent<P: PackedField<Scalar = B128>>(
	mut rs_eq_ind: FieldBuffer<P>,
	pubcheck_point: &[B128],
	batch_coeff: B128,
) -> FieldBuffer<P> {
	let log_public_elems = pubcheck_point.len();
	let pubcheck_eq = eq_ind_partial_eval::<P>(pubcheck_point);
	let mut chunk = rs_eq_ind.chunk_mut(log_public_elems, 0);
	let mut chunk_data = chunk.get();
	let batch = P::broadcast(batch_coeff);
	for (dst, src) in std::iter::zip(chunk_data.as_mut(), pubcheck_eq.as_ref()) {
		*dst += *src * batch;
	}
	drop(chunk);
	rs_eq_ind
}

/// Packs committed witness words into the field buffer committed as the trace oracle.
///
/// Two 64-bit words are packed little-endian into one 128-bit field element.
/// The element sequence is zero-padded up to `2^log_witness_elems`.
///
/// # Arguments
///
/// - `log_witness_elems`: base-2 logarithm of the committed field-element count.
/// - `witness`: the committed witness words, in value-vector order.
///
/// # Returns
///
/// The packed multilinear over `log_witness_elems` variables, ready to commit.
///
/// # Errors
///
/// Returns an error when the words do not fit in `2^log_witness_elems` field elements.
pub fn pack_witness<P: PackedField<Scalar = B128>>(
	log_witness_elems: usize,
	witness: &[Word],
) -> Result<FieldBuffer<P>, Error> {
	// The number of field elements that constitute the packed witness.
	let n_witness_elems = witness.len().div_ceil(1 << LOG_WORDS_PER_ELEM);
	if n_witness_elems > 1 << log_witness_elems {
		return Err(Error::ArgumentError {
			arg: "witness".to_string(),
			msg: "witness element count is incompatible with the constraint system".to_string(),
		});
	}

	let len = 1 << log_witness_elems.saturating_sub(P::LOG_WIDTH);
	let mut padded_witness_elems = Vec::<P>::with_capacity(len);

	// Pack word pairs into B128 elements (2 words per field element), then group into P.
	// Zero-pad up to the power-of-two witness polynomial length after the real words.
	let (pairs, word_remaining) = witness.as_chunks::<2>();
	let aligned_len = pairs.len() / P::WIDTH * P::WIDTH;
	let (pairs_aligned, word_pair_remaining) = pairs.split_at(aligned_len);
	pairs_aligned
		.par_chunks(P::WIDTH)
		.map(|word_pairs| {
			P::from_scalars(
				word_pairs
					.iter()
					.map(|[w0, w1]| B128::new(((w1.0 as u128) << 64) | (w0.0 as u128))),
			)
		})
		.collect_into_vec(&mut padded_witness_elems);

	// The trailing partial group: any leftover word pairs (fewer than `P::WIDTH` of them) together
	// with a final unpaired word are packed into a single `P` element. This keeps the zero padding
	// strictly after the last real word, rather than splitting the unpaired word into a separate
	// element and leaving a zero in the middle of the witness (BINIUS-173).
	if !word_pair_remaining.is_empty() || !word_remaining.is_empty() {
		let word_pairs = word_pair_remaining
			.iter()
			.copied()
			.chain(word_remaining.iter().map(|&word| [word, Word::ZERO]));
		padded_witness_elems.push(P::from_scalars(
			word_pairs.map(|[w0, w1]| B128::new(((w1.0 as u128) << 64) | (w0.0 as u128))),
		));
	}

	padded_witness_elems.resize(len, P::default());

	Ok(FieldBuffer::new(log_witness_elems, padded_witness_elems.into_boxed_slice()))
}

fn prove_bitand_reduction<F, PChallenge, Channel>(
	witness: AndCheckWitness,
	channel: &mut Channel,
) -> AndCheckOutput<F>
where
	F: BinaryField + From<B8>,
	PChallenge: PackedField<Scalar = F>,
	Channel: binius_ip_prover::channel::IPProverChannel<F>,
{
	let prover_message_domain = BinarySubspace::<B8>::with_dim(LOG_WORD_SIZE_BITS + 1);
	let AndCheckWitness { a, b, c } = witness;

	let log_constraint_count = checked_log_2(a.len());

	let n_extra_zerocheck_challenges =
		log_constraint_count.saturating_sub(PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len());
	let big_field_zerocheck_challenges = channel.sample_many(n_extra_zerocheck_challenges);

	let prover = OblongZerocheckProver::<_, PChallenge>::new(
		log_constraint_count,
		a,
		b,
		c,
		big_field_zerocheck_challenges,
		prover_message_domain.isomorphic(),
	);

	prover.prove_with_channel(channel)
}

fn prove_intmul_reduction<F, P, Channel>(
	witness: MulCheckWitness,
	channel: &mut Channel,
) -> Result<IntMulOutput<F>, Error>
where
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
	Channel: IOPProverChannel<P>,
{
	let MulCheckWitness { a, b, lo, hi } = witness;

	let mut mulcheck_prover = IntMulProver::new(0, channel);

	let intmul_witness = IntMulWitness::<P>::new(&a, &b, &lo, &hi)?;

	Ok(mulcheck_prover.prove(intmul_witness))
}

struct AndCheckWitness {
	a: Vec<Word>,
	b: Vec<Word>,
	c: Vec<Word>,
}

struct MulCheckWitness {
	a: Vec<Word>,
	b: Vec<Word>,
	lo: Vec<Word>,
	hi: Vec<Word>,
}

#[tracing::instrument(skip_all, "Build BitAnd witness", level = "debug")]
fn build_bitand_witness(and_constraints: &[AndConstraint], witness: &ValueVec) -> AndCheckWitness {
	let n_constraints = and_constraints.len();

	let mut a = Vec::with_capacity(n_constraints);
	let mut b = Vec::with_capacity(n_constraints);
	let mut c = Vec::with_capacity(n_constraints);

	(and_constraints, a.spare_capacity_mut(), b.spare_capacity_mut(), c.spare_capacity_mut())
		.into_par_iter()
		.for_each(|(constraint, a_i, b_i, c_i)| {
			a_i.write(witness.eval_operand(&constraint.a));
			b_i.write(witness.eval_operand(&constraint.b));
			c_i.write(witness.eval_operand(&constraint.c));
		});

	// Safety: all entries in a, b, c are initialized in the parallel loop above.
	unsafe {
		a.set_len(n_constraints);
		b.set_len(n_constraints);
		c.set_len(n_constraints);
	}

	AndCheckWitness { a, b, c }
}

#[tracing::instrument(skip_all, "Build IntMul witness", level = "debug")]
fn build_intmul_witness(mul_constraints: &[MulConstraint], witness: &ValueVec) -> MulCheckWitness {
	let n_constraints = mul_constraints.len();

	let mut a = Vec::with_capacity(n_constraints);
	let mut b = Vec::with_capacity(n_constraints);
	let mut lo = Vec::with_capacity(n_constraints);
	let mut hi = Vec::with_capacity(n_constraints);

	(
		mul_constraints,
		a.spare_capacity_mut(),
		b.spare_capacity_mut(),
		lo.spare_capacity_mut(),
		hi.spare_capacity_mut(),
	)
		.into_par_iter()
		.for_each(|(constraint, a_i, b_i, lo_i, hi_i)| {
			a_i.write(witness.eval_operand(&constraint.a));
			b_i.write(witness.eval_operand(&constraint.b));
			lo_i.write(witness.eval_operand(&constraint.lo));
			hi_i.write(witness.eval_operand(&constraint.hi));
		});

	// Safety: all entries in a, b, lo, hi are initialized in the parallel loop above.
	unsafe {
		a.set_len(n_constraints);
		b.set_len(n_constraints);
		lo.set_len(n_constraints);
		hi.set_len(n_constraints);
	}

	MulCheckWitness { a, b, lo, hi }
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, PackedBinaryGhash2x128b};

	use super::{B128, Word, pack_witness};

	/// The packing `pack_witness` is specified to produce: consecutive little-endian B128 elements
	/// (low word in bits 0..64, high word in bits 64..128), a final unpaired word in the low half,
	/// then zero padding up to `n_elems`.
	fn expected_scalars(words: &[Word], n_elems: usize) -> Vec<B128> {
		let mut scalars = vec![B128::ZERO; n_elems];
		for (elem, pair) in scalars.iter_mut().zip(words.chunks(2)) {
			let lo = pair[0].0 as u128;
			let hi = pair.get(1).map_or(0, |w| w.0 as u128);
			*elem = B128::new((hi << 64) | lo);
		}
		scalars
	}

	/// Regression test for BINIUS-173: with `P::WIDTH = 2`, a witness of 7 words has 3 word-pairs
	/// (not a multiple of the packing width) plus a trailing unpaired word. The buggy code
	/// zero-padded the final partial `P` chunk and then pushed the unpaired word as a *separate*
	/// element, shifting the last real scalar by one position.
	#[test]
	fn test_pack_witness_unaligned_pair_count_with_remainder() {
		type P = PackedBinaryGhash2x128b;
		assert_eq!(P::WIDTH, 2, "this test is meaningful only when the packing width is 2");

		let words: Vec<Word> = (1..=7u64).map(Word).collect();
		let log_witness_elems = 3; // 8 field elements: 4 real, 4 zero-padding.

		let packed = pack_witness::<P>(log_witness_elems, &words).unwrap();
		let got: Vec<B128> = packed.iter_scalars().collect();

		assert_eq!(got, expected_scalars(&words, 1 << log_witness_elems));
	}

	/// Covers every residue of the word count around the `2 * P::WIDTH` boundary (aligned and
	/// unaligned, with and without a trailing word) plus a few larger sizes.
	#[test]
	fn test_pack_witness_various_lengths() {
		type P = PackedBinaryGhash2x128b;

		for n_words in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17] {
			let words: Vec<Word> = (0..n_words as u64).map(|i| Word(i + 100)).collect();
			let n_elems = n_words.div_ceil(2);
			// Round up to a power of two, and to at least one full packed element.
			let log_witness_elems = n_elems.max(P::WIDTH).next_power_of_two().ilog2() as usize;

			let packed = pack_witness::<P>(log_witness_elems, &words).unwrap();
			let got: Vec<B128> = packed.iter_scalars().collect();

			assert_eq!(
				got,
				expected_scalars(&words, 1 << log_witness_elems),
				"n_words = {n_words}"
			);
		}
	}
}
