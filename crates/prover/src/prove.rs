// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::marker::PhantomData;

use binius_compute::{Allocator, BufferPool, VecLike};
use binius_core::{
	constraint_system::{
		AndConstraint, BmulConstraint, ConstraintSystem, ImulConstraint, ValueVec,
	},
	word::Word,
};
use binius_field::{AESTowerField8b as B8, BinaryField, Divisible, Field, PackedField};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop_prover::{basefold::compiler::BaseFoldProverCompiler, channel::IOPProverChannel};
use binius_ip::sumcheck::SumcheckOutput;
use binius_math::{
	BinarySubspace, FieldBuffer,
	inner_product::inner_product,
	ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
	univariate::lagrange_evals,
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::{SerializeBytes, checked_arithmetics::checked_log_2, rayon::prelude::*};
use binius_verifier::{
	IOPVerifier, Verifier,
	config::{B128, LOG_WORDS_PER_ELEM, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES},
	protocols::{binmul::BinMulOutput, bitand::AndCheckOutput, intmul::IntMulOutput},
};
use digest::Output;

use super::error::Error;
use crate::{
	and_reduction::prover::OblongZerocheckProver,
	protocols::{
		binmul::{BinMulWitness, prove as prove_binmul},
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
	pub fn prove<A, P, Channel>(
		&self,
		witness: ValueVec,
		channel: &mut Channel,
		alloc: &A,
	) -> Result<(), Error>
	where
		A: Allocator,
		P: PackedField<Scalar = B128>,
		Channel: IOPProverChannel<P>,
	{
		let cs = &self.constraint_system;

		// [phase] Setup - initialization and constraint system setup
		//
		// Only the non-public words are committed as the trace oracle; the public segment is a
		// verifier-known polynomial.
		let setup_guard = tracing::debug_span!("Prepare witness").entered();
		let witness_packed = pack_witness::<P>(self.log_witness_elems, witness.non_public())?;
		drop(setup_guard);

		// Observe the public input as B128 elements (includes it in Fiat-Shamir).
		let public_packed =
			pack_witness::<P>(self.log_public_words - LOG_WORDS_PER_ELEM, witness.public())?;
		let public_elems = public_packed.iter_scalars().collect::<Vec<_>>();
		channel.observe_many(&public_elems);

		// [phase] Witness Commit - witness generation and commitment
		let witness_commit_guard = tracing::info_span!("Commit witness").entered();

		// Commit witness via channel
		let trace_oracle = channel.send_oracle(witness_packed.to_ref());

		drop(witness_commit_guard);

		// [phase] IntMul Reduction - multiplication constraint reduction
		//
		// Skipped entirely (no transcript messages) when the constraint system has no IMUL
		// constraints. The verifier applies the identical guard, so the transcript stays in sync;
		// the zero `OperatorData` synthesized below then contributes nothing to the shift
		// reduction.
		let intmul_output = if cs.n_imul_constraints() > 0 {
			let intmul_guard = tracing::info_span!(
				"[phase] IntMul check",
				n_constraints = cs.imul_constraints.len()
			)
			.entered();
			let mul_witness = tracing::debug_span!("Assemble columns")
				.in_scope(|| build_intmul_witness(&cs.imul_constraints, &witness, alloc));

			let intmul_output =
				prove_intmul_reduction::<_, _, P, _>(mul_witness, &mut *channel, alloc)?;
			drop(intmul_guard);
			Some(intmul_output)
		} else {
			None
		};

		// [phase] BinMul Reduction - GHASH-field multiplication constraint reduction
		//
		// Runs immediately after the IntMul reduction and before BitAnd, matching the verifier so
		// the transcript stays in sync. Skipped entirely (no transcript messages) when there are
		// no BMUL constraints; the zero `OperatorData` synthesized below then contributes nothing
		// to the shift reduction.
		let binmul_output = if cs.n_bmul_constraints() > 0 {
			let binmul_guard = tracing::info_span!(
				"[phase] BinMul check",
				n_constraints = cs.bmul_constraints.len()
			)
			.entered();
			let binmul_witness = tracing::debug_span!("Assemble columns")
				.in_scope(|| build_binmul_witness(&cs.bmul_constraints, &witness, alloc));

			let binmul_output =
				prove_binmul_reduction::<_, _, P, _>(binmul_witness, &mut *channel, alloc);
			drop(binmul_guard);
			Some(binmul_output)
		} else {
			None
		};

		// [phase] BitAnd Reduction - AND constraint reduction
		let bitand_guard =
			tracing::info_span!("[phase] BitAnd check", n_constraints = cs.and_constraints.len())
				.entered();
		let bitand_claim = {
			let bitand_witness = tracing::debug_span!("Assemble columns")
				.in_scope(|| build_bitand_witness(&cs.and_constraints, &witness, alloc));

			let AndCheckOutput {
				a_eval,
				b_eval,
				c_eval,
				z_challenge,
				eval_point,
			} = prove_bitand_reduction::<_, B128, P, _>(bitand_witness, &mut *channel, alloc);
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
		// zero evals at an empty point): the shift reduction iterates the (empty) IMUL constraints,
		// so this claim contributes zero to its batched evaluation.
		//
		// Build the oblong domain subspace once and pass it into the shift reduction, mirroring
		// the verifier side (`shift::check_eval` takes the domain subspace). It is reused for the
		// IntMul claim collapse below.
		let subspace = BinarySubspace::<B8>::with_dim(Word::LOG_BITS).isomorphic();
		let intmul_claim = match intmul_output {
			Some(IntMulOutput {
				eval_point,
				a_evals,
				b_evals,
				c_lo_evals,
				c_hi_evals,
			}) => {
				let r_zhat_prime = bitand_claim.r_zhat_prime;
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

		// Build `OperatorData` for BinMul using the same shared `r_zhat_prime` challenge,
		// collapsing each of the six per-bit operand columns identically to IntMul. When BinMul
		// was skipped, synthesize a zero claim (six zero evals at an empty point): the shift
		// reduction iterates the (empty) BMUL constraints, so this claim contributes zero to its
		// batched evaluation.
		let binmul_claim = match binmul_output {
			Some(BinMulOutput {
				eval_point,
				a_lo_evals,
				a_hi_evals,
				b_lo_evals,
				b_hi_evals,
				c_lo_evals,
				c_hi_evals,
			}) => {
				let r_zhat_prime = bitand_claim.r_zhat_prime;
				let l_tilde = lagrange_evals(&subspace, r_zhat_prime);
				let make_final_claim = |evals| inner_product(evals, l_tilde.iter_scalars());
				OperatorData {
					evals: vec![
						make_final_claim(a_lo_evals),
						make_final_claim(a_hi_evals),
						make_final_claim(b_lo_evals),
						make_final_claim(b_hi_evals),
						make_final_claim(c_lo_evals),
						make_final_claim(c_hi_evals),
					],
					r_zhat_prime,
					r_x_prime: eval_point,
				}
			}
			None => OperatorData {
				evals: vec![B128::ZERO; 6],
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
		} = prove_shift_reduction::<_, P, _, _>(
			&self.key_collection,
			witness.combined_witness(),
			bitand_claim,
			intmul_claim,
			binmul_claim,
			&subspace,
			&mut *channel,
			alloc,
		);
		drop(shift_guard);

		// [phase] Ring-Switching + PCS Opening
		let pcs_guard = tracing::info_span!(
			"[phase] PCS Opening",
			phase = "pcs_opening",
			perfetto_category = "phase"
		)
		.entered();

		// Ring-switching reduction of the witness claim. The top challenge is the witness's
		// segment selector, which the verifier consumes when reconstructing the full witness
		// evaluation.
		let witness_point = &eval_point[..eval_point.len() - 1];
		let ring_switch::RingSwitchOutput {
			rs_eq_ind,
			sumcheck_claim,
		} = ring_switch::prove(&witness_packed, witness_point, &mut *channel);

		// Prove oracle relations via channel (runs BaseFold internally). The intmul pushforward
		// relation, when the IntMul reduction ran, was already queued inside phase 5.
		channel.prove_oracle_relations([(trace_oracle, witness_packed, rs_eq_ind, sumcheck_claim)]);

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
	/// The pool that recycles this prover's working buffers. It lives for the prover's lifetime,
	/// so blocks freed by one `prove` call are reused by the next.
	pool: BufferPool,
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
			pool: BufferPool::new(),
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
			n_hidden_words = cs.value_vec_layout.n_hidden_words,
			n_bitand = cs.and_constraints.len(),
			n_intmul = cs.imul_constraints.len(),
		)
		.entered();

		// Create channel, delegate to IOPProver::prove, then finish it. The unified channel takes
		// an rng to mask ZK oracles, but a plain `Prover` produces a transparent proof whose only
		// oracle is non-ZK, so no masks are drawn and the rng is never consumed.
		let mut channel = self
			.basefold_compiler
			.create_channel_without_zk_from_transcript::<H, Challenger_, _>(transcript);
		// Working buffers for this proof are drawn from the prover's pool, recycling blocks freed
		// by earlier proofs. The pool is passed as an `&BufferPool` allocator.
		let alloc = &self.pool;
		self.iop_prover
			.prove::<_, P, _>(witness, &mut channel, &alloc)?;
		channel.finish(&alloc);
		Ok(())
	}
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

	Ok(FieldBuffer::new(log_witness_elems, padded_witness_elems))
}

fn prove_bitand_reduction<A, F, PChallenge, Channel>(
	witness: AndCheckWitness<A>,
	channel: &mut Channel,
	alloc: &A,
) -> AndCheckOutput<F>
where
	A: Allocator,
	F: BinaryField + From<B8>,
	PChallenge: PackedField<Scalar = F>,
	Channel: binius_ip_prover::channel::IPProverChannel<F>,
{
	let prover_message_domain = BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1);
	let AndCheckWitness { a, b } = witness;

	let log_constraint_count = checked_log_2(a.len());

	let n_extra_zerocheck_challenges =
		log_constraint_count.saturating_sub(PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len());
	let big_field_zerocheck_challenges = channel.sample_many(n_extra_zerocheck_challenges);

	let prover = OblongZerocheckProver::<_, PChallenge, _>::new(
		log_constraint_count,
		a,
		b,
		big_field_zerocheck_challenges,
		prover_message_domain.isomorphic(),
	);

	prover.prove_with_channel(channel, alloc)
}

fn prove_intmul_reduction<A, F, P, Channel>(
	witness: MulCheckWitness<A>,
	channel: &mut Channel,
	alloc: &A,
) -> Result<IntMulOutput<F>, Error>
where
	A: Allocator,
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
	Channel: IOPProverChannel<P>,
{
	let MulCheckWitness { a, b, lo, hi } = witness;

	let mut mulcheck_prover = IntMulProver::new(0, channel, alloc);

	let intmul_witness = tracing::debug_span!("Build IntMul witness")
		.in_scope(|| IntMulWitness::<_, P>::new(alloc, &a, &b, &lo, &hi))?;

	Ok(mulcheck_prover.prove(intmul_witness))
}

fn prove_binmul_reduction<A, F, P, Channel>(
	witness: BinMulCheckWitness<A>,
	channel: &mut Channel,
	alloc: &A,
) -> BinMulOutput<F>
where
	A: Allocator,
	F: BinaryField + From<u128>,
	P: PackedField<Scalar = F>,
	Channel: binius_ip_prover::channel::IPProverChannel<F>,
{
	// The word slices are borrowed by `BinMulWitness`, so keep the owned `Vec`s alive across the
	// call.
	let BinMulCheckWitness {
		a_lo,
		a_hi,
		b_lo,
		b_hi,
		c_lo,
		c_hi,
	} = witness;

	let binmul_witness = BinMulWitness {
		a_lo: &a_lo,
		a_hi: &a_hi,
		b_lo: &b_lo,
		b_hi: &b_hi,
		c_lo: &c_lo,
		c_hi: &c_hi,
	};

	prove_binmul::<_, F, P, _>(&binmul_witness, channel, alloc)
}

/// The two materialized BitAnd operand columns.
///
/// The C operand column is not built.
/// On a satisfying witness `C = A & B` holds word-by-word.
/// So the AND reduction derives C from these two columns on the fly.
struct AndCheckWitness<A: Allocator> {
	a: A::Vec<Word>,
	b: A::Vec<Word>,
}

struct MulCheckWitness<A: Allocator> {
	a: A::Vec<Word>,
	b: A::Vec<Word>,
	lo: A::Vec<Word>,
	hi: A::Vec<Word>,
}

struct BinMulCheckWitness<A: Allocator> {
	a_lo: A::Vec<Word>,
	a_hi: A::Vec<Word>,
	b_lo: A::Vec<Word>,
	b_hi: A::Vec<Word>,
	c_lo: A::Vec<Word>,
	c_hi: A::Vec<Word>,
}

fn build_bitand_witness<A: Allocator>(
	and_constraints: &[AndConstraint],
	witness: &ValueVec,
	alloc: &A,
) -> AndCheckWitness<A> {
	let n_constraints = and_constraints.len();

	let mut a = alloc.alloc::<Word>(n_constraints);
	let mut b = alloc.alloc::<Word>(n_constraints);

	(and_constraints, a.spare_capacity_mut(), b.spare_capacity_mut())
		.into_par_iter()
		.for_each(|(constraint, a_i, b_i)| {
			a_i.write(witness.eval_operand(&constraint.a));
			b_i.write(witness.eval_operand(&constraint.b));
		});

	// Safety: all entries in a and b are initialized in the parallel loop above.
	unsafe {
		a.set_len(n_constraints);
		b.set_len(n_constraints);
	}

	AndCheckWitness { a, b }
}

fn build_intmul_witness<A: Allocator>(
	imul_constraints: &[ImulConstraint],
	witness: &ValueVec,
	alloc: &A,
) -> MulCheckWitness<A> {
	let n_constraints = imul_constraints.len();

	let mut a = alloc.alloc::<Word>(n_constraints);
	let mut b = alloc.alloc::<Word>(n_constraints);
	let mut lo = alloc.alloc::<Word>(n_constraints);
	let mut hi = alloc.alloc::<Word>(n_constraints);

	(
		imul_constraints,
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

fn build_binmul_witness<A: Allocator>(
	bmul_constraints: &[BmulConstraint],
	witness: &ValueVec,
	alloc: &A,
) -> BinMulCheckWitness<A> {
	let n_constraints = bmul_constraints.len();

	let mut a_lo = alloc.alloc::<Word>(n_constraints);
	let mut a_hi = alloc.alloc::<Word>(n_constraints);
	let mut b_lo = alloc.alloc::<Word>(n_constraints);
	let mut b_hi = alloc.alloc::<Word>(n_constraints);
	let mut c_lo = alloc.alloc::<Word>(n_constraints);
	let mut c_hi = alloc.alloc::<Word>(n_constraints);

	(
		bmul_constraints,
		a_lo.spare_capacity_mut(),
		a_hi.spare_capacity_mut(),
		b_lo.spare_capacity_mut(),
		b_hi.spare_capacity_mut(),
		c_lo.spare_capacity_mut(),
		c_hi.spare_capacity_mut(),
	)
		.into_par_iter()
		.for_each(|(constraint, a_lo_i, a_hi_i, b_lo_i, b_hi_i, c_lo_i, c_hi_i)| {
			a_lo_i.write(witness.eval_operand(&constraint.a_lo));
			a_hi_i.write(witness.eval_operand(&constraint.a_hi));
			b_lo_i.write(witness.eval_operand(&constraint.b_lo));
			b_hi_i.write(witness.eval_operand(&constraint.b_hi));
			c_lo_i.write(witness.eval_operand(&constraint.c_lo));
			c_hi_i.write(witness.eval_operand(&constraint.c_hi));
		});

	// Safety: all entries in the six columns are initialized in the parallel loop above.
	unsafe {
		a_lo.set_len(n_constraints);
		a_hi.set_len(n_constraints);
		b_lo.set_len(n_constraints);
		b_hi.set_len(n_constraints);
		c_lo.set_len(n_constraints);
		c_hi.set_len(n_constraints);
	}

	BinMulCheckWitness {
		a_lo,
		a_hi,
		b_lo,
		b_hi,
		c_lo,
		c_hi,
	}
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
