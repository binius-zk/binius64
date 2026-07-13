//! Packed-field parallel builders for the discharge PROVER hot paths (W1 port).
//!
//! All routines follow the pinned rev's own prover idioms (crates/ip-prover/src/
//! sumcheck/bivariate_product.rs, crates/iop-prover/src/fri internals): rayon
//! parallelism over packed slices, packed-lane accumulators summed once at the end,
//! and `PackedField::from_scalars` lane fills (the `binary_fold_high` pattern).
//! (The pinned rev c799aa10 predates upstream HEAD's `WideMul` unreduced-accumulation
//! API, so plain packed multiplications are used — same as the rev's own provers.)
//!
//! Everything here computes EXACTLY the same field values as the scalar originals
//! (char-2 addition is XOR: associative + commutative) — transcript bytes unchanged.
//!
//! The canonical prover packing is [`PB`] = `OptimalPackedB128` (4x128 on AVX-512,
//! 2x128 on AVX2, 1x128 on NEON/aarch64 — on this machine the win is rayon; on x86
//! the same code additionally vectorizes).

use binius_field::{Field, PackedField};
use binius_math::FieldBuffer;
use binius_utils::rayon::prelude::*;
use binius_verifier::config::B128;

use crate::table::{Term, TermTable};
use crate::vk::{tags, vk_entry};

/// The prover-side packed field for all discharge buffers.
///
/// MEASURED at the pinned rev (c799aa10) on aarch64/M2: `OptimalPackedB128`
/// (= `PackedBinaryGhash1x128b`) multiplies ~7x SLOWER than the scalar `B128`
/// (1.8 ns vs 12.5 ns per mult; `commit_interleaved` 3.5x slower end-to-end) — the
/// packed wrapper's strategy-dispatch does not inline at this rev, while the scalar
/// `Mul` inlines the identical NEON clmul kernel. `B128` IS a width-1 `PackedField`,
/// so every generic packed code path (rayon round loops, NTT encode, BaseFold,
/// fracaddcheck) is instantiated with it unchanged. On x86_64 the true SIMD packings
/// (2x/4x128) are used.
#[cfg(target_arch = "aarch64")]
pub type PB = B128;
#[cfg(not(target_arch = "aarch64"))]
pub type PB = binius_field::arch::OptimalPackedB128;

/// Parallel construction of a `FieldBuffer<P>` of length `2^log_len` from a scalar
/// generator. `generator(i)` must be pure and cheap; indices `i >= 2^log_len` are never
/// requested (log_len >= P::LOG_WIDTH is required by all callers here).
pub fn build_buffer_par<P, G>(log_len: usize, generator: G) -> FieldBuffer<P>
where
	P: PackedField<Scalar = B128>,
	G: Fn(usize) -> B128 + Sync,
{
	assert!(log_len >= P::LOG_WIDTH, "packed builder needs log_len >= LOG_WIDTH");
	let packed_len = 1usize << (log_len - P::LOG_WIDTH);
	let mut data: Vec<P> = Vec::with_capacity(packed_len);
	(0..packed_len)
		.into_par_iter()
		.map(|i| P::from_scalars((0..P::WIDTH).map(|j| generator((i << P::LOG_WIDTH) | j))))
		.collect_into_vec(&mut data);
	FieldBuffer::new(log_len, data.into_boxed_slice())
}

/// Parallel packed-slice copy (plain memcpy chunks under rayon).
pub fn par_copy<P: PackedField>(dst: &mut [P], src: &[P]) {
	assert_eq!(dst.len(), src.len());
	const CHUNK: usize = 1 << 16;
	dst.par_chunks_mut(CHUNK)
		.zip(src.par_chunks(CHUNK))
		.for_each(|(d, s)| d.copy_from_slice(s));
}

/// Phase-A virtual column gather: `col[t] = tensor[sel(terms[t])]` for `t < N`,
/// padded with `tensor[sel(dummy)] = tensor[0]` above (spec 1.3 fixed dummy tuple).
/// Parallel replacement for the scalar per-claim gather loops.
pub fn gather_column<P, S>(
	terms: &[Term],
	tensor: &[B128],
	log_len: usize,
	sel: S,
) -> FieldBuffer<P>
where
	P: PackedField<Scalar = B128>,
	S: Fn(&Term) -> usize + Sync,
{
	let pad = tensor[0];
	build_buffer_par(log_len, |t| match terms.get(t) {
		Some(term) => tensor[sel(term)],
		None => pad,
	})
}

/// Packed parallel `assemble_m_d`: identical layout to [`crate::table::assemble_m_d`]
/// (blocks [D_x | D_y | D_g padded | 0], selector-high).
pub fn assemble_m_d_packed<P: PackedField<Scalar = B128>>(
	dims: &crate::table::ShapeDims,
	h: &crate::table::Histograms,
) -> FieldBuffer<P> {
	let block = 1usize << dims.n_a;
	let mask = block - 1;
	build_buffer_par(dims.n_d, |w| {
		let blk = w >> dims.n_a;
		let t = w & mask;
		match blk {
			0 => h.d_x.get(t).copied().unwrap_or(B128::ZERO),
			1 => h.d_y.get(t).copied().unwrap_or(B128::ZERO),
			2 => h.d_g.get(t).copied().unwrap_or(B128::ZERO),
			_ => B128::ZERO,
		}
	})
}

/// Packed parallel Phase-C leaf columns over the `(n_l + 2)`-var union domain,
/// produced DIRECTLY as the four fracadd-layer halves (top-variable split; see
/// `fracadd::FracLayer`) so the tree prover starts with zero copies.
/// Value-identical to the scalar original: num = [eq(t, rho_ext) | eq | eq | M_D];
/// den = [tau + X | tau + Y | tau + U | tau + emb]; halves split blocks (0,1)|(2,3).
///
/// `eq_rho` is the packed eq tensor of rho (2^n_t scalars, n_t <= n_l); `m_d` the packed
/// M_D buffer (2^n_l scalars).
pub fn build_phase_c_leaf_halves<P: PackedField<Scalar = B128>>(
	table: &TermTable,
	tau: B128,
	eq_rho: &FieldBuffer<P>,
	m_d: &FieldBuffer<P>,
) -> anyhow::Result<crate::fracadd::FracLayer<P>> {
	let dims = &table.dims;
	let n_l = dims.n_d;
	anyhow::ensure!(
		tau.val() >= (1u128 << (dims.n_a + 2)),
		"tau hit the committed-value range (a pole): honest abort, resample the batch"
	);
	anyhow::ensure!(n_l >= P::LOG_WIDTH && dims.n_t >= P::LOG_WIDTH, "domain too small for packing");
	let kappa = tags(dims.n_a);
	let n = table.terms.len();

	// Numerator halves: lo = [eq | eq] (blocks 0, 1); hi = [eq | M_D] (blocks 2, 3).
	// Single-write packed builds (rho_ext's zero tail is the zero fill); the eq tensor
	// occupies the low 2^n_t slots of each block.
	let eq_len = 1usize << dims.n_t;
	let mask = (1usize << n_l) - 1;
	let eq_at = |t: usize| {
		if t < eq_len {
			eq_rho.get(t)
		} else {
			B128::ZERO
		}
	};
	let num_lo = build_buffer_par::<P, _>(n_l + 1, |w| eq_at(w & mask));
	let num_hi = build_buffer_par::<P, _>(n_l + 1, |w| {
		if w >> n_l == 0 {
			eq_at(w)
		} else {
			m_d.get(w & mask)
		}
	});

	// Denominator halves: lane-generated (vk_entry is a few bit ops; emb is the index).
	let terms = &table.terms;
	let den_for = |blk_base: usize| {
		build_buffer_par(n_l + 1, move |w| {
			let blk = blk_base + (w >> n_l);
			let t = w & mask;
			match blk {
				0 | 1 | 2 => {
					if t < n {
						let term = &terms[t];
						let addr = match blk {
							0 => term.x as u64,
							1 => term.y as u64,
							_ => term.u as u64,
						};
						tau + vk_entry(blk, addr, dims.n_a)
					} else {
						tau + kappa[blk]
					}
				}
				_ => tau + B128::new(t as u128),
			}
		})
	};
	let den_lo = den_for(0);
	let den_hi = den_for(2);

	Ok(crate::fracadd::FracLayer {
		num_lo,
		num_hi,
		den_lo,
		den_hi,
	})
}

/// Packed parallel `build_m_vk`: identical values/layout to [`crate::vk::build_m_vk`].
pub fn build_m_vk_packed<P: PackedField<Scalar = B128>>(table: &TermTable) -> FieldBuffer<P> {
	let dims = &table.dims;
	let n_l = dims.n_d;
	let kappa = tags(dims.n_a);
	let n = table.terms.len();
	let terms = &table.terms;
	let mask = (1usize << n_l) - 1;
	build_buffer_par(n_l + 2, |w| {
		let blk = w >> n_l;
		let t = w & mask;
		match blk {
			0 | 1 | 2 => {
				if t < n {
					let term = &terms[t];
					let addr = match blk {
						0 => term.x as u64,
						1 => term.y as u64,
						_ => term.u as u64,
					};
					vk_entry(blk, addr, dims.n_a)
				} else {
					kappa[blk] // kappa[0] == 0 for the X block
				}
			}
			_ => B128::ZERO,
		}
	})
}

/// Packed-lane accumulator for the three VK corner values + the real-row eq mass.
struct CornerAcc<P: PackedField> {
	vx: P,
	vy: P,
	vu: P,
	se: P,
}

impl<P: PackedField> Default for CornerAcc<P> {
	fn default() -> Self {
		Self {
			vx: P::zero(),
			vy: P::zero(),
			vu: P::zero(),
			se: P::zero(),
		}
	}
}

impl<P: PackedField> CornerAcc<P> {
	fn merge(self, rhs: Self) -> Self {
		Self {
			vx: self.vx + rhs.vx,
			vy: self.vy + rhs.vy,
			vu: self.vu + rhs.vu,
			se: self.se + rhs.se,
		}
	}
}

/// Packed parallel `vk_corner_values`: the three VK column MLE evaluations at pi_lo,
/// value-identical to the scalar original (fused single pass, no column
/// materialization). `eq_pi` is the packed eq tensor of pi_lo (2^n_l scalars).
pub fn vk_corner_values_packed<P: PackedField<Scalar = B128>>(
	table: &TermTable,
	eq_pi: &FieldBuffer<P>,
) -> [B128; 3] {
	let dims = &table.dims;
	let kappa = tags(dims.n_a);
	let n = table.terms.len();
	let terms = &table.terms;
	let n_full = n >> P::LOG_WIDTH; // full packed elements covering real rows

	let acc = eq_pi.as_ref()[..n_full]
		.par_iter()
		.enumerate()
		.fold(CornerAcc::<P>::default, |mut acc, (i, &e)| {
			let base = i << P::LOG_WIDTH;
			let tx = P::from_scalars(
				(0..P::WIDTH).map(|j| vk_entry(0, terms[base + j].x as u64, dims.n_a)),
			);
			let ty = P::from_scalars(
				(0..P::WIDTH).map(|j| vk_entry(1, terms[base + j].y as u64, dims.n_a)),
			);
			let tu = P::from_scalars(
				(0..P::WIDTH).map(|j| vk_entry(2, terms[base + j].u as u64, dims.n_a)),
			);
			acc.vx += e * tx;
			acc.vy += e * ty;
			acc.vu += e * tu;
			acc.se += e;
			acc
		})
		.reduce(CornerAcc::<P>::default, CornerAcc::merge);

	let mut v = [
		acc.vx.iter().sum::<B128>(),
		acc.vy.iter().sum::<B128>(),
		acc.vu.iter().sum::<B128>(),
	];
	let mut sum_real = acc.se.iter().sum::<B128>();

	// Boundary rows [n_full * WIDTH, n): scalar tail (at most WIDTH - 1 rows).
	for t in (n_full << P::LOG_WIDTH)..n {
		let e = eq_pi.get(t);
		v[0] += e * vk_entry(0, terms[t].x as u64, dims.n_a);
		v[1] += e * vk_entry(1, terms[t].y as u64, dims.n_a);
		v[2] += e * vk_entry(2, terms[t].u as u64, dims.n_a);
		sum_real += e;
	}

	// Pad rows: X contributes 0, Y/U contribute kappa * pad_mass (see scalar original).
	let pad_mass = B128::ONE + sum_real;
	v[1] += kappa[1] * pad_mass;
	v[2] += kappa[2] * pad_mass;
	v
}

/// Parallel dense axpy: `w_eq[i] += scale * tensor[i]` (Phase-B W_eq accumulation for
/// the dense Phase-C point). Value-identical to the scalar loop.
pub fn axpy_dense_par(w_eq: &mut [B128], tensor: &[B128], scale: B128) {
	assert!(tensor.len() <= w_eq.len());
	w_eq[..tensor.len()]
		.par_iter_mut()
		.zip(tensor.par_iter())
		.for_each(|(slot, t_val)| *slot += scale * *t_val);
}

#[cfg(test)]
mod tests {
	use binius_field::Random;
	use binius_math::multilinear::eq::{eq_ind_partial_eval, eq_ind_partial_eval_scalars};
	use rand::prelude::*;

	use super::*;
	use crate::{
		synth::synth_cs,
		table::{assemble_m_d, build_histograms, extract_table},
		vk::build_m_vk,
	};

	/// Packed builders must be value-identical to the scalar originals on a synthetic CS.
	#[test]
	fn test_packed_builders_match_scalar() {
		let mut cs = synth_cs(0);
		cs.validate_and_prepare().expect("prepare");
		let table = extract_table(&cs).expect("table");
		let dims = &table.dims;
		let mut rng = StdRng::seed_from_u64(11);
		let rho: Vec<B128> = (0..dims.n_t).map(|_| B128::random(&mut rng)).collect();
		let hist = build_histograms(&table, &rho);
		let m_d_scalar: FieldBuffer<B128> = assemble_m_d(dims, &hist);
		let m_d_packed: FieldBuffer<PB> = assemble_m_d_packed(dims, &hist);
		for i in 0..1usize << dims.n_d {
			assert_eq!(m_d_scalar.get(i), m_d_packed.get(i), "m_d slot {i}");
		}

		// M_VK
		let m_vk_scalar = build_m_vk(&table);
		let m_vk_packed: FieldBuffer<PB> = build_m_vk_packed(&table);
		for i in 0..1usize << (dims.n_d + 2) {
			assert_eq!(m_vk_scalar.get(i), m_vk_packed.get(i), "m_vk slot {i}");
		}

		// Phase-C leaf columns (as fracadd halves; blocks (0,1) in lo, (2,3) in hi)
		let tau = B128::new(1u128 << 90) + B128::random(&mut rng);
		let eq_rho_scalars = eq_ind_partial_eval_scalars(&rho);
		let eq_rho_packed: FieldBuffer<PB> = eq_ind_partial_eval(&rho);
		let leaf =
			build_phase_c_leaf_halves(&table, tau, &eq_rho_packed, &m_d_packed).expect("leaf");
		// scalar original (copied semantics)
		let n_l = dims.n_d;
		let block = 1usize << n_l;
		for w in 0..block << 2 {
			let blk = w >> n_l;
			let t = w & (block - 1);
			let (num_val, den_val) = if w < 2 * block {
				(leaf.num_lo.get(w), leaf.den_lo.get(w))
			} else {
				(leaf.num_hi.get(w - 2 * block), leaf.den_hi.get(w - 2 * block))
			};
			let num_expect = match blk {
				0 | 1 | 2 => {
					if t < eq_rho_scalars.len() {
						eq_rho_scalars[t]
					} else {
						B128::ZERO
					}
				}
				_ => m_d_scalar.get(t),
			};
			assert_eq!(num_val, num_expect, "num slot {w}");
			let den_expect = match blk {
				0 | 1 | 2 => {
					if t < table.terms.len() {
						let term = &table.terms[t];
						let addr = [term.x as u64, term.y as u64, term.u as u64][blk];
						tau + vk_entry(blk, addr, dims.n_a)
					} else {
						tau + tags(dims.n_a)[blk]
					}
				}
				_ => tau + B128::new(t as u128),
			};
			assert_eq!(den_val, den_expect, "den slot {w}");
		}

		// Corner values
		let pi_lo: Vec<B128> = (0..n_l).map(|_| B128::random(&mut rng)).collect();
		let eq_pi_scalars = eq_ind_partial_eval_scalars(&pi_lo);
		let eq_pi_packed: FieldBuffer<PB> = eq_ind_partial_eval(&pi_lo);
		let scalar_corners = {
			// inline copy of the scalar original from step2.rs
			let kappa = tags(dims.n_a);
			let mut v = [B128::ZERO; 3];
			let mut sum_real = B128::ZERO;
			for (t, term) in table.terms.iter().enumerate() {
				let e = eq_pi_scalars[t];
				v[0] += e * vk_entry(0, term.x as u64, dims.n_a);
				v[1] += e * vk_entry(1, term.y as u64, dims.n_a);
				v[2] += e * vk_entry(2, term.u as u64, dims.n_a);
				sum_real += e;
			}
			let pad_mass = B128::ONE + sum_real;
			v[1] += kappa[1] * pad_mass;
			v[2] += kappa[2] * pad_mass;
			v
		};
		assert_eq!(vk_corner_values_packed(&table, &eq_pi_packed), scalar_corners);

		// Column gather
		let x_tensor = eq_ind_partial_eval_scalars(
			&(0..dims.n_x).map(|_| B128::random(&mut rng)).collect::<Vec<_>>(),
		);
		let col: FieldBuffer<PB> =
			gather_column(&table.terms, &x_tensor, dims.n_t, |t| t.x as usize);
		for t in 0..1usize << dims.n_t {
			let expect = match table.terms.get(t) {
				Some(term) => x_tensor[term.x as usize],
				None => x_tensor[0],
			};
			assert_eq!(col.get(t), expect, "gather slot {t}");
		}
	}
}
