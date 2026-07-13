//! Vendored fractional-addition prover (W1 port) — value- and transcript-identical to
//! the pinned rev's `binius_ip_prover::fracaddcheck::FracAddCheckProver`, restructured
//! for prover speed:
//!
//! * each layer is stored as its four OWNED halves (num_lo, num_hi, den_lo, den_hi), so
//!   `layer_prover` hands them to `frac_add_mle::new` with ZERO copies (upstream clones
//!   all four half-slices per layer — ~8 GiB of allocation+copy at 2^27);
//! * the layered tree is built with pre-allocated buffers under `par_chunks_mut`
//!   (upstream's rayon tuple-unzip `collect::<(Vec, Vec)>()` dominates its build time);
//! * the leaf layer is accepted directly as four halves, so the Phase-C leaf builder
//!   never materializes (then re-splits) the full union-domain buffers.
//!
//! Per-node formulas, layer order, and every transcript operation are IDENTICAL to
//! upstream (`prove` mirrors fracaddcheck.rs:145-187 verbatim), so proof bytes are
//! unchanged. Fractional addition rule per sibling pair (upstream :75):
//!   next_num[j] = num_lo[j] * den_hi[j] + num_hi[j] * den_lo[j]
//!   next_den[j] = den_lo[j] * den_hi[j]

use binius_field::PackedField;
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_ip_prover::{
	channel::IPProverChannel,
	fracaddcheck::Error as FracError,
	sumcheck::frac_add_mle,
	sumcheck::batch::batch_prove_mle_and_write_evals,
};
use binius_math::{FieldBuffer, line::extrapolate_line_packed};
use binius_utils::rayon::prelude::*;
use binius_verifier::config::B128;

/// One tree layer, stored as the four owned halves the layer sumcheck consumes.
pub struct FracLayer<P: PackedField> {
	pub num_lo: FieldBuffer<P>,
	pub num_hi: FieldBuffer<P>,
	pub den_lo: FieldBuffer<P>,
	pub den_hi: FieldBuffer<P>,
}

impl<P: PackedField<Scalar = B128>> FracLayer<P> {
	/// log2 length of the FULL layer (halves are one var shorter).
	pub fn log_len(&self) -> usize {
		self.num_lo.log_len() + 1
	}

	/// Computes the next (halved) layer. The next layer's halves are the contiguous
	/// first/second halves of the combined reduction output.
	fn reduce(&self) -> Self {
		let half_log = self.num_lo.log_len(); // input pair count = 2^half_log
		assert!(half_log >= 1, "reduce needs at least 2 sibling pairs");
		let q_log = half_log - 1;

		if half_log <= P::LOG_WIDTH + 4 {
			// Small tail layers: scalar path (identical values; padding handled by
			// from_values).
			let q = 1usize << q_log;
			let mut num = Vec::with_capacity(2 * q);
			let mut den = Vec::with_capacity(2 * q);
			for j in 0..2 * q {
				let (a0, b0, a1, b1) = (
					self.num_lo.get(j),
					self.den_lo.get(j),
					self.num_hi.get(j),
					self.den_hi.get(j),
				);
				num.push(a0 * b1 + a1 * b0);
				den.push(b0 * b1);
			}
			return Self {
				num_lo: FieldBuffer::from_values(&num[..q]),
				num_hi: FieldBuffer::from_values(&num[q..]),
				den_lo: FieldBuffer::from_values(&den[..q]),
				den_hi: FieldBuffer::from_values(&den[q..]),
			};
		}

		let q_packed = 1usize << (q_log - P::LOG_WIDTH);
		const CHUNK: usize = 1 << 13;
		// Region 0 (input j < Q) fills the lo outputs; region 1 fills the hi outputs.
		// Pre-allocated outputs, explicit chunked writes (no rayon collect machinery).
		let (nl, nh, dl, dh) = (
			self.num_lo.as_ref(),
			self.num_hi.as_ref(),
			self.den_lo.as_ref(),
			self.den_hi.as_ref(),
		);
		let mut halves: Vec<(Vec<P>, Vec<P>)> = (0..2usize)
			.map(|region| {
				let base = region * q_packed;
				let mut num_out = vec![P::zero(); q_packed];
				let mut den_out = vec![P::zero(); q_packed];
				num_out
					.par_chunks_mut(CHUNK)
					.zip(den_out.par_chunks_mut(CHUNK))
					.enumerate()
					.for_each(|(ci, (nc, dc))| {
						let off = base + ci * CHUNK;
						for j in 0..nc.len() {
							let a0 = nl[off + j];
							let b0 = dl[off + j];
							let a1 = nh[off + j];
							let b1 = dh[off + j];
							nc[j] = a0 * b1 + a1 * b0;
							dc[j] = b0 * b1;
						}
					});
				(num_out, den_out)
			})
			.collect();
		let (num_hi, den_hi) = halves.pop().expect("two regions");
		let (num_lo, den_lo) = halves.pop().expect("two regions");
		Self {
			num_lo: FieldBuffer::new(q_log, num_lo.into_boxed_slice()),
			num_hi: FieldBuffer::new(q_log, num_hi.into_boxed_slice()),
			den_lo: FieldBuffer::new(q_log, den_lo.into_boxed_slice()),
			den_hi: FieldBuffer::new(q_log, den_hi.into_boxed_slice()),
		}
	}

	/// Final reduction of a 2-element layer to the root pair (num_root, den_root).
	fn root(&self) -> (B128, B128) {
		assert_eq!(self.log_len(), 1);
		let (a0, b0, a1, b1) = (
			self.num_lo.get(0),
			self.den_lo.get(0),
			self.num_hi.get(0),
			self.den_hi.get(0),
		);
		(a0 * b1 + a1 * b0, b0 * b1)
	}
}

/// Drop-in replacement for the pinned rev's `FracAddCheckProver` (same protocol bytes).
pub struct FastFracAddProver<P: PackedField> {
	layers: Vec<FracLayer<P>>,
}

impl<P: PackedField<Scalar = B128>> FastFracAddProver<P> {
	/// Builds the layered tree from the leaf layer (given as four halves) down to the
	/// root. Returns `(prover, (num_root, den_root))` — the roots equal upstream's
	/// `sums` layer values for `k = leaf.log_len()`.
	pub fn new(leaf: FracLayer<P>) -> (Self, (B128, B128)) {
		let k = leaf.log_len();
		let mut layers = Vec::with_capacity(k);
		layers.push(leaf);
		while layers.last().expect("non-empty").log_len() > 1 {
			let next = layers.last().expect("non-empty").reduce();
			layers.push(next);
		}
		let root = layers.last().expect("non-empty").root();
		(Self { layers }, root)
	}

	/// Runs the protocol: identical control flow and transcript operations to upstream
	/// `FracAddCheckProver::prove` (fracaddcheck.rs:145-187), with zero-copy layer
	/// hand-off.
	pub fn prove(
		mut self,
		claim: (MultilinearEvalClaim<B128>, MultilinearEvalClaim<B128>),
		channel: &mut impl IPProverChannel<B128>,
	) -> Result<(MultilinearEvalClaim<B128>, MultilinearEvalClaim<B128>), FracError> {
		let mut claim = claim;
		while let Some(layer) = self.layers.pop() {
			let (num_claim, den_claim) = claim;
			assert_eq!(
				num_claim.point, den_claim.point,
				"fractional claims must share the evaluation point"
			);
			let sumcheck_prover = frac_add_mle::new(
				[layer.num_lo, layer.num_hi, layer.den_lo, layer.den_hi],
				num_claim.point.clone(),
				[num_claim.eval, den_claim.eval],
			)?;

			let output = batch_prove_mle_and_write_evals(vec![sumcheck_prover], channel)?;

			let mut multilinear_evals = output.multilinear_evals;
			let evals = multilinear_evals.pop().expect("batch contains one prover");
			let [num_0, num_1, den_0, den_1] = evals
				.try_into()
				.expect("prover evaluates four multilinears");

			let r = channel.sample();
			let next_num = extrapolate_line_packed(num_0, num_1, r);
			let next_den = extrapolate_line_packed(den_0, den_1, r);

			let mut next_point = output.challenges;
			next_point.push(r);
			claim = (
				MultilinearEvalClaim {
					eval: next_num,
					point: next_point.clone(),
				},
				MultilinearEvalClaim {
					eval: next_den,
					point: next_point,
				},
			);
		}
		Ok(claim)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, Random};
	use binius_ip::fracaddcheck::{self, FracAddEvalClaim};
	use binius_ip_prover::fracaddcheck::FracAddCheckProver;
	use binius_transcript::ProverTranscript;
	use binius_verifier::config::StdChallenger;
	use rand::prelude::*;

	use super::*;

	/// The vendored prover must produce byte-identical transcripts to upstream and
	/// pass the upstream verifier, across sizes that exercise both reduce paths.
	#[test]
	fn test_fast_fracadd_matches_upstream() {
		for k in [3usize, 7, 12] {
			let mut rng = StdRng::seed_from_u64(31 + k as u64);
			let n = 1usize << k;
			let num_vals: Vec<B128> = (0..n).map(|_| B128::random(&mut rng)).collect();
			let den_vals: Vec<B128> =
				(0..n).map(|_| B128::random(&mut rng) + B128::ONE).collect();

			// Upstream run.
			let up_bytes = {
				let num = FieldBuffer::<B128>::from_values(&num_vals);
				let den = FieldBuffer::<B128>::from_values(&den_vals);
				let (prover, sums) = FracAddCheckProver::<B128>::new(k, (num, den));
				let root_claim = (
					MultilinearEvalClaim {
						eval: sums.0.get(0),
						point: Vec::new(),
					},
					MultilinearEvalClaim {
						eval: sums.1.get(0),
						point: Vec::new(),
					},
				);
				let mut pt = ProverTranscript::new(StdChallenger::default());
				let out = prover.prove(root_claim, &mut pt).unwrap();
				(pt.finalize(), out, sums.0.get(0), sums.1.get(0))
			};

			// Vendored run.
			let fast_bytes = {
				let half = n / 2;
				let leaf = FracLayer {
					num_lo: FieldBuffer::<B128>::from_values(&num_vals[..half]),
					num_hi: FieldBuffer::<B128>::from_values(&num_vals[half..]),
					den_lo: FieldBuffer::<B128>::from_values(&den_vals[..half]),
					den_hi: FieldBuffer::<B128>::from_values(&den_vals[half..]),
				};
				let (prover, (num_root, den_root)) = FastFracAddProver::new(leaf);
				assert_eq!(num_root, up_bytes.2, "root numerator");
				assert_eq!(den_root, up_bytes.3, "root denominator");
				let root_claim = (
					MultilinearEvalClaim {
						eval: num_root,
						point: Vec::new(),
					},
					MultilinearEvalClaim {
						eval: den_root,
						point: Vec::new(),
					},
				);
				let mut pt = ProverTranscript::new(StdChallenger::default());
				let out = prover.prove(root_claim, &mut pt).unwrap();
				(pt.finalize(), out)
			};

			assert_eq!(up_bytes.0, fast_bytes.0, "transcript bytes k={k}");
			assert_eq!(up_bytes.1.0.point, fast_bytes.1.0.point);
			assert_eq!(up_bytes.1.0.eval, fast_bytes.1.0.eval);
			assert_eq!(up_bytes.1.1.eval, fast_bytes.1.1.eval);

			// And the upstream verifier accepts the vendored transcript.
			let mut pt = ProverTranscript::new(StdChallenger::default());
			{
				let half = n / 2;
				let leaf = FracLayer {
					num_lo: FieldBuffer::<B128>::from_values(&num_vals[..half]),
					num_hi: FieldBuffer::<B128>::from_values(&num_vals[half..]),
					den_lo: FieldBuffer::<B128>::from_values(&den_vals[..half]),
					den_hi: FieldBuffer::<B128>::from_values(&den_vals[half..]),
				};
				let (prover, (num_root, den_root)) = FastFracAddProver::new(leaf);
				let root_claim = (
					MultilinearEvalClaim {
						eval: num_root,
						point: Vec::new(),
					},
					MultilinearEvalClaim {
						eval: den_root,
						point: Vec::new(),
					},
				);
				prover.prove(root_claim, &mut pt).unwrap();
			}
			let mut vt = pt.into_verifier();
			fracaddcheck::verify::<B128, _>(
				k,
				FracAddEvalClaim {
					num_eval: up_bytes.2,
					den_eval: up_bytes.3,
					point: Vec::new(),
				},
				&mut vt,
			)
			.expect("upstream verifier must accept the vendored transcript");
		}
	}
}
