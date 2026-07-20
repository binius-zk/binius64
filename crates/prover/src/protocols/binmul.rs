// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_field::{BinaryField, PackedField};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{ProveSingleOutput, prove_single_mlecheck, quadratic_mlecheck_prover},
};
use binius_math::field_buffer::FieldBuffer;
use binius_utils::checked_arithmetics::strict_log_2;
pub use binius_verifier::protocols::binmul::BinMulOutput;

use crate::fold_word::WordFolder;

/// A witness for the BinMul reduction.
///
/// Each of the six word columns has length $2^\ell$, where $\ell$ is the log2 of the number of
/// constraints. The GHASH-field element for constraint $x$ is $\langle\langle z_{\textsf{lo}},
/// z_{\textsf{hi}} \rangle\rangle = \sum_{i=0}^{63} z_{\textsf{lo},x,i} \cdot X^i +
/// \sum_{i=0}^{63} z_{\textsf{hi},x,i} \cdot X^{64+i}$ for each of $z \in \{a, b, c\}$.
pub struct BinMulWitness<'a> {
	pub a_lo: &'a [Word],
	pub a_hi: &'a [Word],
	pub b_lo: &'a [Word],
	pub b_hi: &'a [Word],
	pub c_lo: &'a [Word],
	pub c_hi: &'a [Word],
}

/// Build a packed GHASH-field multilinear table from a `(lo, hi)` pair of word columns.
///
/// The scalar at hypercube index `i` is the field element carried by the pair `(lo[i], hi[i])`:
/// `lo[i]` supplies the low 64 bits and `hi[i]` the high 64 bits of the 128-bit value.
fn build_table<F, P>(lo: &[Word], hi: &[Word]) -> FieldBuffer<P>
where
	F: BinaryField + From<u128>,
	P: PackedField<Scalar = F>,
{
	let n_vars = strict_log_2(lo.len())
		.expect("precondition: the number of constraints must be a power of two");
	let p_width = P::WIDTH.min(1 << n_vars);
	let values = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
		.map(|i| {
			P::from_scalars((0..p_width).map(|j| {
				let index = i << P::LOG_WIDTH | j;
				F::from((lo[index].as_u64() as u128) | ((hi[index].as_u64() as u128) << 64))
			}))
		})
		.collect::<Vec<_>>();

	FieldBuffer::new(n_vars, values)
}

/// Prove the binary-field multiplication check (BinMul) reduction.
///
/// Proves $\widetilde{A}(x) \cdot \widetilde{B}(x) = \widetilde{C}(x)$ for every $x$ on the boolean
/// hypercube $\mathbb{B}_\ell$ over the GHASH field, where each element is carried by a `(lo, hi)`
/// pair of 64-bit words. See [`binius_verifier::protocols::binmul::verify`] for the protocol
/// description and output shape.
pub fn prove<F, P, Channel>(witness: &BinMulWitness, channel: &mut Channel) -> BinMulOutput<F>
where
	F: BinaryField + From<u128>,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	let n_vars = strict_log_2(witness.a_lo.len())
		.expect("precondition: the number of constraints must be a power of two");
	for column in [
		witness.a_hi,
		witness.b_lo,
		witness.b_hi,
		witness.c_lo,
		witness.c_hi,
	] {
		assert_eq!(column.len(), witness.a_lo.len());
	}

	// Build the packed GHASH-field multilinear tables A, B, C from the (lo, hi) word pairs.
	let a = build_table::<F, P>(witness.a_lo, witness.a_hi);
	let b = build_table::<F, P>(witness.b_lo, witness.b_hi);
	let c = build_table::<F, P>(witness.c_lo, witness.c_hi);

	// Sample the zerocheck challenge r_z.
	let r_z = channel.sample_many(n_vars);

	// Product zerocheck: 0 = sum_x eq(r_z, x) * (A(x) * B(x) - C(x)). The composition A * B - C has
	// degree 2; the eq factor is folded internally by the MLE-check.
	let prover = quadratic_mlecheck_prover(
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
	let a_lo_evals = folder.fold(witness.a_lo).to_vec();
	let a_hi_evals = folder.fold(witness.a_hi).to_vec();
	let b_lo_evals = folder.fold(witness.b_lo).to_vec();
	let b_hi_evals = folder.fold(witness.b_hi).to_vec();
	let c_lo_evals = folder.fold(witness.c_lo).to_vec();
	let c_hi_evals = folder.fold(witness.c_hi).to_vec();

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
	use binius_core::word::Word;
	use binius_field::{BinaryField128bGhash, PackedBinaryGhash2x128b, Random};
	use binius_iop::channel::{OracleSpec, naive::NaiveVerifierChannel};
	use binius_iop_prover::channel::naive::NaiveProverChannel;
	use binius_math::{inner_product::inner_product_buffers, multilinear::eq::eq_ind_partial_eval};
	use binius_transcript::ProverTranscript;
	use binius_verifier::{
		config::StdChallenger,
		protocols::binmul::{BinMulOutput, verify},
	};
	use itertools::izip;
	use rand::prelude::*;

	use super::{BinMulWitness, prove};

	type F = BinaryField128bGhash;
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

		let partially_folded_witness =
			crate::fold_word::fold_words::<_, F>(words, prefix_tensor.as_ref());

		inner_product_buffers(&partially_folded_witness, &suffix_tensor)
	}

	/// Split a GHASH-field element into its `(lo, hi)` 64-bit word pair.
	fn to_word_pair(elem: F) -> (Word, Word) {
		let value = u128::from(elem);
		(Word::from_u64(value as u64), Word::from_u64((value >> 64) as u64))
	}

	/// Build a valid BinMul witness with `1 << log_n` random constraints `a * b = c`, where `c` is
	/// computed with GHASH-field multiplication directly (an independent oracle).
	fn random_witness(rng: &mut impl Rng, log_n: usize) -> WordColumns {
		let n = 1 << log_n;
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
		let (a_lo, a_hi, b_lo, b_hi, c_lo, c_hi) = random_witness(&mut rng, LOG_N);

		let witness = BinMulWitness {
			a_lo: &a_lo,
			a_hi: &a_hi,
			b_lo: &b_lo,
			b_hi: &b_hi,
			c_lo: &c_lo,
			c_hi: &c_hi,
		};

		// BinMul commits no oracles.
		let oracle_specs: [OracleSpec; 0] = [];

		// Run prover.
		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, oracle_specs.to_vec());
		let prove_output = prove::<F, P, _>(&witness, &mut prover_channel);
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
			|evals: Vec<F>| izip!(evals, z_tensor.as_ref()).map(|(x, y)| x * y).sum();

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

	#[test]
	fn verify_rejects_tampered_c() {
		let mut rng = StdRng::seed_from_u64(1);

		const LOG_N: usize = 5;
		let (a_lo, a_hi, b_lo, b_hi, mut c_lo, c_hi) = random_witness(&mut rng, LOG_N);

		// Corrupt one c_lo word so the constraint no longer holds.
		c_lo[3] = Word::from_u64(c_lo[3].as_u64() ^ 1);

		let witness = BinMulWitness {
			a_lo: &a_lo,
			a_hi: &a_hi,
			b_lo: &b_lo,
			b_hi: &b_hi,
			c_lo: &c_lo,
			c_hi: &c_hi,
		};

		let oracle_specs: [OracleSpec; 0] = [];

		let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
		let mut prover_channel =
			NaiveProverChannel::<F, _>::new(&mut prover_transcript, oracle_specs.to_vec());
		let _ = prove::<F, P, _>(&witness, &mut prover_channel);
		prover_channel.finish();

		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			NaiveVerifierChannel::<F, _>::new(&mut verifier_transcript, &oracle_specs);
		let result = verify(LOG_N, &mut verifier_channel);
		assert!(result.is_err(), "verifier must reject a tampered witness");
	}
}
