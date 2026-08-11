// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{marker::PhantomData, mem::MaybeUninit};

use binius_compute::{Allocator, BufferPool, VecLike};
use binius_core::{
	constraint_system::{ConstraintSystem, InoutSegment, Operand, ValueVec},
	word::Word,
};
use binius_field::{AESTowerField8b as B8, Field, PackedField};
use binius_hash::binary_merkle_tree::HashSuite;
use binius_iop_prover::{basefold::compiler::BaseFoldProverCompiler, channel::IOPProverChannel};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::WordIPProverChannel;
use binius_math::{
	BinarySubspace, FieldBuffer, FieldVec,
	inner_product::inner_product,
	ntt::{NeighborsLastMultiThread, domain_context::GaoMateerPreExpanded},
	univariate::lagrange_evals,
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::{
	SerializeBytes,
	rayon::{prelude::*, task_size::IndexedParallelIteratorExt},
};
use binius_verifier::{
	IOPVerifier, Verifier,
	config::{B128, LOG_WORDS_PER_ELEM},
	protocols::{binmul::BinMulOutput, bitand::AndCheckOutput, intmul::IntMulOutput, zero},
};
use digest::Output;

use super::error::Error;
use crate::{
	and_reduction,
	protocols::{
		binmul, intmul,
		shift::{
			KeyCollection, OperatorClaims, OperatorData, build_key_collection,
			prove as prove_shift_reduction,
		},
	},
	ring_switch,
};

/// Type alias for the prover NTT parameterized by field.
type ProverNTT<F> = NeighborsLastMultiThread<GaoMateerPreExpanded<F>>;

/// IOP prover for a particular constraint system.
///
/// This struct encapsulates the constraint system and pre-computed keys,
/// providing the core proving logic independent of the specific IOP compilation strategy.
/// Most users should use [`Prover`] instead, which wraps this with a BaseFold compiler.
#[derive(Debug)]
pub struct IOPProver {
	constraint_system: ConstraintSystem,
	log_witness_elems: usize,
	key_collection: KeyCollection,
}

impl IOPProver {
	/// Constructs an IOP prover from an IOP verifier and pre-computed keys.
	pub fn new(iop_verifier: IOPVerifier, key_collection: KeyCollection) -> Self {
		let log_witness_elems = iop_verifier.log_witness_elems();
		let constraint_system = iop_verifier.into_constraint_system();
		Self {
			constraint_system,
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
		witness: &ValueVec,
		channel: &mut Channel,
		alloc: &A,
	) -> Result<(), Error>
	where
		A: Allocator,
		P: PackedField<Scalar = B128>,
		Channel: IOPProverChannel<P, A> + WordIPProverChannel<B128, Word = Word>,
	{
		let cs = &self.constraint_system;

		// [phase] Setup - initialization and constraint system setup
		//
		// Only the non-public words are committed as the trace oracle; the public segment is a
		// verifier-known polynomial.
		let setup_guard = tracing::debug_span!("Prepare witness").entered();
		let witness_packed =
			pack_witness::<P, _>(alloc, self.log_witness_elems, witness.non_public())?;
		drop(setup_guard);

		// Observe the inout words, which includes them in Fiat-Shamir. The constants are fixed by
		// the constraint system, so only the per-instance values are observed.
		channel.observe_words(witness.inout());

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
			let mul_columns = tracing::debug_span!("Assemble columns")
				.in_scope(|| build_operation_columns(&cs.imul_constraints, witness, alloc));

			let [a, b, lo, hi] = &mul_columns;
			let intmul_output = intmul::prove::<_, _, P, _>([a, b, lo, hi], &mut *channel, alloc)?;
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
			let binmul_columns = tracing::debug_span!("Assemble columns")
				.in_scope(|| build_operation_columns(&cs.bmul_constraints, witness, alloc));

			let [a_lo, a_hi, b_lo, b_hi, c_lo, c_hi] = &binmul_columns;
			let binmul_output = binmul::prove::<_, _, P, _>(
				[a_lo, a_hi, b_lo, b_hi, c_lo, c_hi],
				&mut *channel,
				alloc,
			);
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
			// Only the `A` and `B` columns are built; the reduction derives `C = A & B`.
			let bitand_columns = tracing::debug_span!("Assemble columns")
				.in_scope(|| build_operation_columns(&cs.and_constraints, witness, alloc));

			let AndCheckOutput {
				a_eval,
				b_eval,
				c_eval,
				z_challenge,
				eval_point,
			} = and_reduction::prove::<_, B128, P, _, _>(bitand_columns, &mut *channel, alloc);
			OperatorData {
				evals: [a_eval, b_eval, c_eval],
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
					evals: [
						make_final_claim(a_evals),
						make_final_claim(b_evals),
						make_final_claim(c_lo_evals),
						make_final_claim(c_hi_evals),
					],
					r_zhat_prime,
					r_x_prime: eval_point,
				}
			}
			None => OperatorData::zero_claim(bitand_claim.r_zhat_prime),
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
					evals: [
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
			None => OperatorData::zero_claim(bitand_claim.r_zhat_prime),
		};

		// [phase] Zero Reduction - linear constraint reduction
		//
		// The reduction's claim, at the point the BitAnd sumcheck just output. See
		// `IOPVerifier::verify` for why it carries no message.
		let log_n_zero = cs.log_zero_constraints().unwrap_or(0);
		let zero_claim = OperatorData {
			evals: [B128::ZERO],
			r_zhat_prime: bitand_claim.r_zhat_prime,
			r_x_prime: zero::reduction_point(&bitand_claim.r_x_prime, log_n_zero, || {
				channel.sample()
			}),
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
			witness.public(),
			witness.non_public(),
			OperatorClaims {
				zero: zero_claim,
				bitand: bitand_claim,
				intmul: intmul_claim,
				binmul: binmul_claim,
			},
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
		} = ring_switch::prove(alloc, witness_packed.to_ref(), witness_point, &mut *channel);

		// Prove oracle relations via channel (runs BaseFold internally). The intmul pushforward
		// relation, when the IntMul reduction ran, was already queued inside phase 5.
		channel.prove_oracle_relations([(trace_oracle, witness_packed, rs_eq_ind, sumcheck_claim)]);

		drop(pcs_guard);

		Ok(())
	}
}

/// Warns once per process if the CPU supports carryless multiply but this build does not use it.
///
/// The GHASH field arithmetic is selected at compile time, so a default-target x86_64 build runs
/// the software multiply even on CPUs with PCLMULQDQ, silently costing an order of magnitude in
/// prover throughput (see issue #1800). Building with `-C target-cpu=native` or
/// `-C target-feature=+pclmulqdq` selects the hardware path.
#[cfg(target_arch = "x86_64")]
fn warn_on_software_field_arithmetic() {
	use std::{arch::is_x86_feature_detected, sync::Once};

	static ONCE: Once = Once::new();
	ONCE.call_once(|| {
		if !cfg!(target_feature = "pclmulqdq") && is_x86_feature_detected!("pclmulqdq") {
			tracing::warn!(
				"this CPU supports carryless multiply (PCLMULQDQ), but the build does not \
				 enable it, so field arithmetic will run in software; rebuild with \
				 `-C target-cpu=native` or `-C target-feature=+pclmulqdq`"
			);
		}
	});
}

#[cfg(not(target_arch = "x86_64"))]
const fn warn_on_software_field_arithmetic() {}

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
		let key_collection =
			build_key_collection(verifier.constraint_system(), InoutSegment::Public);
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
		warn_on_software_field_arithmetic();

		// Rebuild the verifier's evaluation domain, which its compiler fixed as the Gao-Mateer
		// basis of that dimension.
		let domain_context =
			GaoMateerPreExpanded::generate(verifier.iop_compiler().max_subspace().dim());
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
		witness: &ValueVec,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		let cs = self.iop_prover.constraint_system();

		let _prove_guard = tracing::info_span!(
			"Prove",
			n_hidden_words = cs.n_hidden_words(InoutSegment::Public),
			n_bitand = cs.and_constraints.len(),
			n_intmul = cs.imul_constraints.len(),
		)
		.entered();

		// Create channel, delegate to IOPProver::prove, then finish it. The unified channel takes
		// an rng to mask ZK oracles, but a plain `Prover` produces a transparent proof whose only
		// oracle is non-ZK, so no masks are drawn and the rng is never consumed.
		let mut channel = self
			.basefold_compiler
			.create_channel_without_zk_from_transcript::<H, Challenger_, _, _>(transcript);
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
/// - `alloc`: the allocator the packed buffer is drawn from.
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
pub fn pack_witness<P: PackedField<Scalar = B128>, A: Allocator>(
	alloc: &A,
	log_witness_elems: usize,
	witness: &[Word],
) -> Result<FieldVec<P, A>, Error> {
	// The number of field elements that constitute the packed witness.
	let n_witness_elems = witness.len().div_ceil(1 << LOG_WORDS_PER_ELEM);
	if n_witness_elems > 1 << log_witness_elems {
		return Err(Error::ArgumentError {
			arg: "witness".to_string(),
			msg: "witness element count is incompatible with the constraint system".to_string(),
		});
	}

	let len = 1 << log_witness_elems.saturating_sub(P::LOG_WIDTH);
	let mut padded_witness_elems = alloc.alloc::<P>(len);

	// Pack word pairs into B128 elements (2 words per field element), then group into P.
	// Zero-pad up to the power-of-two witness polynomial length after the real words.
	let (pairs, word_remaining) = witness.as_chunks::<2>();
	let aligned_len = pairs.len() / P::WIDTH * P::WIDTH;
	let (pairs_aligned, word_pair_remaining) = pairs.split_at(aligned_len);
	// `collect_into_vec` needs a `&mut Vec`, which the generic buffer is not, so the aligned groups
	// are written straight into the buffer's spare capacity instead.
	let n_aligned_elems = aligned_len / P::WIDTH;
	(
		pairs_aligned.par_chunks(P::WIDTH),
		padded_witness_elems.spare_capacity_mut()[..n_aligned_elems].par_iter_mut(),
	)
		.into_par_iter()
		.with_min_task_bytes::<P>()
		.for_each(|(word_pairs, out)| {
			out.write(P::from_scalars(
				word_pairs
					.iter()
					.map(|[w0, w1]| B128::new(((w1.0 as u128) << 64) | (w0.0 as u128))),
			));
		});
	// Safety: `aligned_len` is an exact multiple of `P::WIDTH`, so the chunk iterator yields
	// exactly `n_aligned_elems` items — the same length as the spare-capacity slice it is zipped
	// with. That equal length is what makes this sound: rayon's zip silently truncates to the
	// shorter side, so a mismatch would leave slots uninitialized. With the lengths equal, the loop
	// above writes each of the first `n_aligned_elems` slots exactly once.
	unsafe { padded_witness_elems.set_len(n_aligned_elems) };

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

/// Evaluates the leading `N_COLS` operands of every constraint against the witness, one
/// materialized column per operand.
///
/// Column `i` holds operand `i` of every constraint, in the constraint type's storage order — the
/// order the shift reduction batches operands in. Each column has exactly one row per constraint,
/// in the same order, and nothing beyond them: every reduction rounds the constraint axis up to
/// `constraints.len().next_power_of_two()` itself and reads the rows past a column's end as zero,
/// which satisfies every constraint type.
///
/// An empty constraint slice still yields one zero row. [`ConstraintSystem::log_and_constraints`]
/// reports `None` for an empty AND set and the verifier reads that as *zero* constraint variables —
/// one all-zero row, not zero rows — and the BitAnd check has no skip branch. The two
/// multiplication checks run only on a non-empty constraint set, so only AND reaches this.
///
/// [`ConstraintSystem::log_and_constraints`]: binius_core::constraint_system::ConstraintSystem::log_and_constraints
///
/// `N_COLS` may be smaller than `ARITY`, in which case the trailing operands are not evaluated. The
/// BitAnd check uses that to skip its `C` column: on a satisfying witness `C = A & B` holds
/// word-by-word, so the reduction derives it from the `A` and `B` columns instead.
fn build_operation_columns<C, A, const ARITY: usize, const N_COLS: usize>(
	constraints: &[C],
	witness: &ValueVec,
	alloc: &A,
) -> [A::Vec<Word>; N_COLS]
where
	C: AsRef<[Operand; ARITY]> + Sync,
	A: Allocator,
{
	const {
		assert!(N_COLS <= ARITY, "N_COLS must not exceed the constraint arity");
	}

	let n_constraints = constraints.len();
	// One row per constraint, and one standing in for an empty set (see above).
	let n_rows = n_constraints.max(1);
	(0..N_COLS)
		.into_par_iter()
		.map(|op_idx| {
			let mut column = alloc.alloc::<Word>(n_rows);
			// The allocator may hand back more capacity than requested, so bound the spare slice to
			// the row count.
			let rows = &mut column.spare_capacity_mut()[..n_rows];
			// No constraint writes the empty set's row, so zero it here.
			if n_constraints == 0 {
				rows.fill(MaybeUninit::new(Word::ZERO));
			}
			(constraints, &mut *rows)
				.into_par_iter()
				.for_each(|(constraint, out)| {
					out.write(witness.eval_operand(&constraint.as_ref()[op_idx]));
				});
			// Safety: the parallel loop writes each of the `n_rows` entries exactly once, since the
			// zip is over equal-length sides — except when the constraint set is empty and the one
			// row standing in for it was zeroed above.
			unsafe { column.set_len(n_rows) };
			column
		})
		.collect::<Vec<_>>()
		.try_into()
		.unwrap_or_else(|_| unreachable!("source iterator has N_COLS elements"))
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_core::constraint_system::AndConstraint;
	use binius_field::{Field, PackedBinaryGhash2x128b};
	use binius_frontend::CircuitBuilder;

	use super::{B128, ValueVec, Word, build_operation_columns, pack_witness};

	/// A circuit of `n_gates` independent AND gates, and a witness satisfying it.
	///
	/// One gate yields one AND constraint, so the constraint count is `n_gates` exactly.
	fn and_gate_witness(n_gates: usize) -> (Vec<AndConstraint>, ValueVec) {
		let builder = CircuitBuilder::new();
		let wires: Vec<_> = (0..n_gates)
			.map(|_| {
				let x = builder.add_witness();
				let y = builder.add_witness();
				builder.force_commit(builder.band(x, y));
				(x, y)
			})
			.collect();
		let circuit = builder.build();

		let mut w = circuit.new_witness_filler();
		for (i, &(x, y)) in wires.iter().enumerate() {
			// Both operands are non-zero on every gate, so a zero row can only be padding.
			w[x] = Word(0x0123_4567_89AB_CDEF | (i as u64) << 32 | 1);
			w[y] = Word(0xFEDC_BA98_7654_3210 | (i as u64) | 1);
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		let cs = circuit.constraint_system().clone();
		cs.validate().unwrap();
		assert_eq!(cs.n_and_constraints(), n_gates);
		(cs.and_constraints, w.into_value_vec())
	}

	/// The columns stop at the last constraint rather than rounding up to a power of two: the
	/// reductions round the constraint axis up themselves and read the rows past a column's end as
	/// zero.
	#[test]
	fn build_operation_columns_stops_at_the_last_constraint() {
		let (constraints, witness) = and_gate_witness(3);
		let columns = build_operation_columns::<AndConstraint, _, 3, 2>(
			&constraints,
			&witness,
			&GlobalAllocator,
		);

		// Three rows, not the four the reduction runs over. The fixture makes every operand
		// non-zero, so a surviving padding row would show up as a zero tail.
		for column in &columns {
			assert_eq!(column.len(), 3);
			assert!(column.iter().all(|&word| word != Word::ZERO));
		}
	}

	/// An empty constraint set still yields one all-zero row: the verifier reads
	/// `log_and_constraints() == None` as zero constraint variables, which is one row, and the
	/// BitAnd check has no skip branch.
	#[test]
	fn build_operation_columns_gives_an_empty_set_one_zero_row() {
		let (_, witness) = and_gate_witness(1);
		let columns =
			build_operation_columns::<AndConstraint, _, 3, 2>(&[], &witness, &GlobalAllocator);

		for column in &columns {
			assert_eq!(column.len(), 1);
			assert_eq!(column[0], Word::ZERO);
		}
	}

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

		let packed = pack_witness::<P, _>(&GlobalAllocator, log_witness_elems, &words).unwrap();
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

			let packed = pack_witness::<P, _>(&GlobalAllocator, log_witness_elems, &words).unwrap();
			let got: Vec<B128> = packed.iter_scalars().collect();

			assert_eq!(
				got,
				expected_scalars(&words, 1 << log_witness_elems),
				"n_words = {n_words}"
			);
		}
	}
}
