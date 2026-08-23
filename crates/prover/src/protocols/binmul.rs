// Copyright 2026 The Binius Developers

use std::iter;

use binius_compute::{Allocator, VecLike};
use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{ProveSingleOutput, prove_single_mlecheck, quadratic_mlecheck_prover},
};
use binius_math::{FieldVec, field_buffer::FieldBuffer};
use binius_utils::checked_arithmetics::log2_ceil_usize;
pub use binius_verifier::protocols::binmul::BinMulOutput;

use crate::fold_word::WordFolder;

/// Build a packed GHASH-field multilinear table from a `(lo, hi)` pair of word columns.
///
/// The scalar at hypercube index `i` is the field element carried by the pair `(lo[i], hi[i])`:
/// `lo[i]` supplies the low 64 bits and `hi[i]` the high 64 bits of the 128-bit value.
///
/// The columns need not have a power-of-two length. The table spans the whole `2^n_vars` hypercube
/// regardless, and an index at or past the columns' end reads as the zero field element. The
/// padding therefore lives in this buffer, which is allocated at the hypercube's size either way,
/// rather than in a `Vec<Word>` the caller has to materialize.
///
/// # Preconditions
///
/// * The two columns have equal length.
fn build_table<A, F, P>(alloc: &A, lo: &[Word], hi: &[Word]) -> FieldVec<P, A>
where
	A: Allocator,
	F: BinaryField + From<u128>,
	P: PackedField<Scalar = F>,
{
	assert_eq!(lo.len(), hi.len());

	let n_vars = log2_ceil_usize(lo.len());
	let packed_len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);
	let elem =
		|(&lo, &hi): (&Word, &Word)| F::from((lo.as_u64() as u128) | ((hi.as_u64() as u128) << 64));

	let mut values = alloc.alloc::<P>(packed_len);

	// The packed elements the columns fill whole. `chunks_exact` stops at the last of them, so
	// every lane this loop packs is a real row.
	let (lo_chunks, hi_chunks) = (lo.chunks_exact(P::WIDTH), hi.chunks_exact(P::WIDTH));
	let (lo_tail, hi_tail) = (lo_chunks.remainder(), hi_chunks.remainder());
	values.extend(
		iter::zip(lo_chunks, hi_chunks)
			.map(|(lo_chunk, hi_chunk)| P::from_scalars(iter::zip(lo_chunk, hi_chunk).map(elem))),
	);

	// The columns' trailing words share one packed element with the start of the padding. The lanes
	// `from_scalars` is not given default to zero, which is exactly that padding.
	if !lo_tail.is_empty() {
		values.push(P::from_scalars(iter::zip(lo_tail, hi_tail).map(elem)));
	}

	// The rest of the hypercube is zero padding. A zero row satisfies `0 * 0 = 0`, so it
	// contributes nothing to the zerocheck.
	values.resize(packed_len, P::zero());

	FieldBuffer::new(n_vars, values)
}

/// Prove the binary-field multiplication check (BinMul) reduction.
///
/// Proves $\widetilde{A}(x) \cdot \widetilde{B}(x) = \widetilde{C}(x)$ for every $x$ on the boolean
/// hypercube $\mathbb{B}_\ell$ over the GHASH field, where each element is carried by a `(lo, hi)`
/// pair of 64-bit words. See [`binius_verifier::protocols::binmul::verify`] for the protocol
/// description and output shape.
///
/// The six `columns` are the `(lo, hi)` word pairs of the two multiplicands and the product, in the
/// order `[a_lo, a_hi, b_lo, b_hi, c_lo, c_hi]`, all of equal length. That length need not be a
/// power of two: the hypercube is $\mathbb{B}_\ell$ for $\ell = \lceil \log_2 n \rceil$, and a row
/// at or past the columns' end reads as the zero field element. A zero row satisfies
/// $0 \cdot 0 = 0$, so it contributes nothing to the zerocheck. The GHASH-field element for row $x$
/// is
/// $\langle\langle z_{\textsf{lo}}, z_{\textsf{hi}} \rangle\rangle = \sum_{i=0}^{63}
/// z_{\textsf{lo},x,i} \cdot X^i + \sum_{i=0}^{63} z_{\textsf{hi},x,i} \cdot X^{64+i}$ for each of
/// $z \in \{a, b, c\}$.
pub fn prove<A, F, P, Channel>(
	columns: [&[Word]; 6],
	channel: &mut Channel,
	alloc: &A,
) -> BinMulOutput<F>
where
	A: Allocator,
	F: BinaryField + From<u128>,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	let [a_lo, a_hi, b_lo, b_hi, c_lo, c_hi] = columns;

	let n_vars = log2_ceil_usize(a_lo.len());
	for column in [a_hi, b_lo, b_hi, c_lo, c_hi] {
		assert_eq!(column.len(), a_lo.len());
	}

	// Build the packed GHASH-field multilinear tables A, B, C from the (lo, hi) word pairs.
	let a = build_table::<A, F, P>(alloc, a_lo, a_hi);
	let b = build_table::<A, F, P>(alloc, b_lo, b_hi);
	let c = build_table::<A, F, P>(alloc, c_lo, c_hi);

	// Sample the zerocheck challenge r_z.
	let r_z = channel.sample_many(n_vars);

	// Product zerocheck: 0 = sum_x eq(r_z, x) * (A(x) * B(x) - C(x)). The composition A * B - C has
	// degree 2; the eq factor is folded internally by the MLE-check.
	let prover = quadratic_mlecheck_prover(
		alloc,
		[a, b, c],
		|[a, b, c]| a * b - c,
		|[a, b, _c]| a * b,
		r_z,
		F::ZERO,
	);

	let ProveSingleOutput {
		multilinear_evals: _,
		challenges: mut eval_point,
	} = prove_single_mlecheck(prover, channel);
	// `prove_single_mlecheck` folds high-to-low, so reverse to obtain the shared output point r_x.
	eval_point.reverse();

	// Send the raw per-bit output evaluations of each word column at r_x. One folder is built for
	// the shared point and reused across all six columns.
	let folder = WordFolder::<F>::new(&eval_point);
	let a_lo_evals = folder.fold(a_lo);
	let a_hi_evals = folder.fold(a_hi);
	let b_lo_evals = folder.fold(b_lo);
	let b_hi_evals = folder.fold(b_hi);
	let c_lo_evals = folder.fold(c_lo);
	let c_hi_evals = folder.fold(c_hi);

	channel.send_many(&a_lo_evals);
	channel.send_many(&a_hi_evals);
	channel.send_many(&b_lo_evals);
	channel.send_many(&b_hi_evals);
	channel.send_many(&c_lo_evals);
	channel.send_many(&c_hi_evals);

	BinMulOutput {
		eval_point,
		a_lo_evals,
		a_hi_evals,
		b_lo_evals,
		b_hi_evals,
		c_lo_evals,
		c_hi_evals,
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_core::word::Word;
	use binius_field::{Ghash128b, PackedBinaryGhash2x128b, Random};
	use binius_iop::channel::{OracleSpec, naive::NaiveVerifierChannel};
	use binius_iop_prover::channel::naive::NaiveProverChannel;
	use binius_math::{inner_product::inner_product_buffers, multilinear::eq::eq_ind_partial_eval};
	use binius_transcript::{ProverTranscript, VerifierTranscript};
	use binius_verifier::{
		config::StdChallenger,
		protocols::binmul::{BinMulOutput, verify},
	};
	use itertools::izip;
	use rand::prelude::*;

	use super::{log2_ceil_usize, prove};

	type F = Ghash128b;
	type P = PackedBinaryGhash2x128b;

	/// The six word columns `(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi)` of a BinMul witness.
	type WordColumns = (Vec<Word>, Vec<Word>, Vec<Word>, Vec<Word>, Vec<Word>, Vec<Word>);

	/// Evaluate the multilinear extension of a per-bit word column at a point, independently of the
	/// prover. The point's `Word::LOG_BITS`-coordinate prefix selects the bit within a word; the
	/// suffix selects the word (constraint) index.
	fn evaluate_witness(words: &[Word], eval_point: &[F]) -> F {
		let (prefix, suffix) = eval_point.split_at(Word::LOG_BITS);
		let prefix_tensor = eq_ind_partial_eval::<F>(prefix);
		let suffix_tensor = eq_ind_partial_eval::<F>(suffix);

		let partially_folded_witness = crate::fold_word::fold_words::<_, F, _>(
			&GlobalAllocator,
			words,
			prefix_tensor.as_ref(),
		);

		inner_product_buffers(&partially_folded_witness, &suffix_tensor)
	}

	/// Split a GHASH-field element into its `(lo, hi)` 64-bit word pair.
	fn to_word_pair(elem: F) -> (Word, Word) {
		let value = u128::from(elem);
		(Word::from_u64(value as u64), Word::from_u64((value >> 64) as u64))
	}

	/// Build a valid BinMul witness with `n` random constraints `a * b = c`, where `c` is computed
	/// with GHASH-field multiplication directly (an independent oracle).
	fn random_witness(rng: &mut impl Rng, n: usize) -> WordColumns {
		let mut a_lo = Vec::with_capacity(n);
		let mut a_hi = Vec::with_capacity(n);
		let mut b_lo = Vec::with_capacity(n);
		let mut b_hi = Vec::with_capacity(n);
		let mut c_lo = Vec::with_capacity(n);
		let mut c_hi = Vec::with_capacity(n);

		for _ in 0..n {
			let a = F::random(&mut *rng);
			let b = F::random(&mut *rng);
			let c = a * b;

			let (a_lo_i, a_hi_i) = to_word_pair(a);
			let (b_lo_i, b_hi_i) = to_word_pair(b);
			let (c_lo_i, c_hi_i) = to_word_pair(c);

			a_lo.push(a_lo_i);
			a_hi.push(a_hi_i);
			b_lo.push(b_lo_i);
			b_hi.push(b_hi_i);
			c_lo.push(c_lo_i);
			c_hi.push(c_hi_i);
		}

		(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi)
	}

	#[test]
	fn prove_and_verify() {
		let mut rng = StdRng::seed_from_u64(0);

		const LOG_N: usize = 5;
		let (a_lo, a_hi, b_lo, b_hi, c_lo, c_hi) = random_witness(&mut rng, 1 << LOG_N);

		// BinMul commits no oracles.
		let oracle_specs: [OracleSpec; 0] = [];

		// Run prover.
		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, oracle_specs.to_vec());
		let prove_output = prove::<_, F, P, _>(
			[&a_lo, &a_hi, &b_lo, &b_hi, &c_lo, &c_hi],
			&mut prover_channel,
			&GlobalAllocator,
		);
		prover_channel.finish();

		let BinMulOutput {
			eval_point,
			a_lo_evals,
			a_hi_evals,
			b_lo_evals,
			b_hi_evals,
			c_lo_evals,
			c_hi_evals,
		} = prove_output.clone();

		// Independently check each column's per-bit evals against the witness MLE. We batch the bit
		// columns with a `z_challenge` and compare at a single point
		// `consistency_check_eval_point`.
		let z_challenge: Vec<F> = (0..Word::LOG_BITS).map(|_| F::random(&mut rng)).collect();
		let z_tensor = eq_ind_partial_eval::<F>(&z_challenge);
		let consistency_check_eval_point = [z_challenge, eval_point].concat();
		let get_consistency_check_eval =
			|evals: [F; Word::BITS]| izip!(evals, z_tensor.as_ref()).map(|(x, y)| x * y).sum();

		let test_cases = [
			(&a_lo, a_lo_evals),
			(&a_hi, a_hi_evals),
			(&b_lo, b_lo_evals),
			(&b_hi, b_hi_evals),
			(&c_lo, c_lo_evals),
			(&c_hi, c_hi_evals),
		];
		for (words, evals) in test_cases {
			let expected_eval = evaluate_witness(words, &consistency_check_eval_point);
			let given_eval = get_consistency_check_eval(evals);
			assert_eq!(expected_eval, given_eval);
		}

		// Run verifier.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &oracle_specs);
		let verify_output = verify(LOG_N, &mut verifier_channel).unwrap();
		verifier_channel.finish();

		assert_eq!(prove_output, verify_output);
	}

	// A witness whose constraint count is not a power of two proves exactly as the same witness
	// zero-padded up to one: byte-identical transcript, identical output claims, and the proof
	// still verifies against the padded variable count the verifier derives from the constraint
	// system.
	//
	// This is the guarantee that lets the M4 prover stop materializing the padding rows
	// (BINIUS-388) without moving anything on the wire.
	#[test]
	fn unpadded_columns_prove_identically_to_padded_ones() {
		let mut rng = StdRng::seed_from_u64(2);

		// Constraint counts crossing the fold's 64-word chunk and the packed width: a single row, a
		// sub-chunk count, a count straddling the chunk boundary, and a multi-chunk count.
		for n in [1, 3, 65, 100] {
			let columns = random_witness(&mut rng, n);
			let n_vars = log2_ceil_usize(n);

			// The same witness with explicit zero rows up to the hypercube's size. A zero row
			// satisfies `0 * 0 = 0`, so the padded witness is equally valid.
			let padded = {
				let pad = |words: &Vec<Word>| {
					let mut padded = words.clone();
					padded.resize(1 << n_vars, Word::ZERO);
					padded
				};
				let (a_lo, a_hi, b_lo, b_hi, c_lo, c_hi) = &columns;
				(pad(a_lo), pad(a_hi), pad(b_lo), pad(b_hi), pad(c_lo), pad(c_hi))
			};

			// One proof per shape, each on a fresh transcript over the same challenger seed.
			let run = |columns: &WordColumns| {
				let (a_lo, a_hi, b_lo, b_hi, c_lo, c_hi) = columns;
				let oracle_specs: [OracleSpec; 0] = [];
				let mut transcript = ProverTranscript::<StdChallenger>::default();
				let mut channel =
					NaiveProverChannel::<F, _>::new(&mut transcript, oracle_specs.to_vec());
				let output = prove::<_, F, P, _>(
					[a_lo, a_hi, b_lo, b_hi, c_lo, c_hi],
					&mut channel,
					&GlobalAllocator,
				);
				channel.finish();
				(output, transcript.finalize())
			};
			let (unpadded_output, unpadded_proof) = run(&columns);
			let (padded_output, padded_proof) = run(&padded);

			assert_eq!(unpadded_output, padded_output, "claims differ at n = {n}");
			assert_eq!(unpadded_proof, padded_proof, "transcript differs at n = {n}");

			let oracle_specs: [OracleSpec; 0] = [];
			let mut verifier_transcript =
				VerifierTranscript::new(StdChallenger::default(), unpadded_proof);
			let mut verifier_channel =
				NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &oracle_specs);
			let verify_output = verify(n_vars, &mut verifier_channel).unwrap();
			verifier_channel.finish();

			assert_eq!(unpadded_output, verify_output, "verifier disagrees at n = {n}");
		}
	}

	#[test]
	fn verify_rejects_tampered_c() {
		let mut rng = StdRng::seed_from_u64(1);

		const LOG_N: usize = 5;
		let (a_lo, a_hi, b_lo, b_hi, mut c_lo, c_hi) = random_witness(&mut rng, 1 << LOG_N);

		// Corrupt one c_lo word so the constraint no longer holds.
		c_lo[3] = Word::from_u64(c_lo[3].as_u64() ^ 1);

		let oracle_specs: [OracleSpec; 0] = [];

		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, oracle_specs.to_vec());
		let _ = prove::<_, F, P, _>(
			[&a_lo, &a_hi, &b_lo, &b_hi, &c_lo, &c_hi],
			&mut prover_channel,
			&GlobalAllocator,
		);
		prover_channel.finish();

		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &oracle_specs);
		let result = verify(LOG_N, &mut verifier_channel);
		assert!(result.is_err(), "verifier must reject a tampered witness");
	}
}
