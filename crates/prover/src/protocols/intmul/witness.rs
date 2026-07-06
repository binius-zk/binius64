// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, mem::MaybeUninit, ops::Deref};

use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField};
use binius_ip_prover::prodcheck::ProdcheckProver;
use binius_math::field_buffer::FieldBuffer;
use binius_utils::{
	checked_arithmetics::{checked_log_2, strict_log_2},
	rayon::prelude::*,
	strided_array::StridedArray2DViewMut,
};
use binius_verifier::{
	config::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS},
	protocols::intmul::common::{LIMB_BITS, LOG_N_LIMBS, N_LIMBS},
};
use getset::Getters;
use itertools::iterate;

use super::error::Error;

/// An integer multiplication protocol witness. Created from integer slices, consumed during
/// proving.
///
/// The statement being proven is `a * b = c`, where `c` is represented as a pair `(c_lo, c_hi)`:
/// `WORD_SIZE_BITS`-wide multiplicands with a double-wide product.
///
/// For each of `a`, `c_lo`, `c_hi` the exponent words split into `N_LIMBS` limbs, and the
/// constant-base exponentiation factors as a product of `N_LIMBS` limb columns — column `l` reads
/// the row `limb_l(e)` of the power table of the base $G^{2^{wl}}$ (where $w$ is the limb bit
/// width). The columns are concatenated into one `(n_vars + LOG_N_LIMBS)`-variate buffer with the
/// limb index in the high bits, and a [`ProdcheckProver`] is constructed over each:
///  1) `a` and `c_lo` exponentiate the multiplicative group generator $G$
///  2) `c_hi` exponentiates $G^{2^{2^m}}$
///  3) `b` selects a variable base (the root of the `a` tree) per bit, over `WORD_SIZE_BITS`
///     per-bit leaves as before
///
/// The shared power table $T\colon i \mapsto G^i$ over $2^w$ rows is retained for the Phase 5
/// logup* lookup.
///
/// Protocol proves that ${(G^a)}^b = G^{c\\_lo} \times (G^{2^{2^m}})^{c\\_hi}$, which is equivalent
/// to $a \times b = c$ modulo $2^{2^{m+1}} - 1$. The special case of `0 * 0 = 1` is handled
/// separately.
#[derive(Clone, Getters)]
#[getset(get = "pub")]
pub struct Witness<'a, P: PackedField> {
	/// The exponents for `a` (needed for the phase 5 lookup indices and parity zerocheck on
	/// `a_0`).
	#[getset(skip)]
	pub a_exponents: &'a [Word],
	/// Prodcheck prover for the `a` exponentiation tree (leaf layer retained).
	pub a_prodcheck: ProdcheckProver<P>,
	/// The root of the `a` tree (product of all leaves element-wise); the `b` variable base.
	pub a_root: FieldBuffer<P>,
	/// The exponents for `b` (needed for phase 5).
	#[getset(skip)]
	pub b_exponents: &'a [Word],
	/// Concatenated b leaves for prodcheck: [L_0, L_1, ..., L_{2^k-1}].
	/// Has log_len = n_vars + LOG_WORD_SIZE_BITS.
	pub b_leaves: FieldBuffer<P>,
	/// The prover for the prodcheck reduction on b_leaves.
	pub b_prodcheck: ProdcheckProver<P>,
	/// The root of the b tree (product of all leaves element-wise).
	pub b_root: FieldBuffer<P>,
	/// The exponents for `c_lo` (needed for the phase 5 lookup indices, parity zerocheck on
	/// `c_lo_0`, and the raw per-bit output evaluations).
	#[getset(skip)]
	pub c_lo_exponents: &'a [Word],
	/// Prodcheck prover for the `c_lo` exponentiation tree (leaf layer retained).
	pub c_lo_prodcheck: ProdcheckProver<P>,
	/// The root of the `c_lo` tree.
	pub c_lo_root: FieldBuffer<P>,
	/// The exponents for `c_hi` (needed for the phase 5 lookup indices and raw per-bit output
	/// evaluations).
	#[getset(skip)]
	pub c_hi_exponents: &'a [Word],
	/// Prodcheck prover for the `c_hi` exponentiation tree (leaf layer retained).
	pub c_hi_prodcheck: ProdcheckProver<P>,
	/// The root of the `c_hi` tree.
	pub c_hi_root: FieldBuffer<P>,
	/// The 2·N_LIMBS twisted power tables: `tables[s][i] = (G^{2^{ws}})^i`. Limb column `(t, l)`
	/// is a gather from table `s(t, l)`; `tables[0]` is the shared table read by the Phase 5
	/// logup* lookup.
	pub tables: Vec<FieldBuffer<P>>,
}

impl<'a, F, P> Witness<'a, P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	/// Constructs a new integer multiplication witness from the statement.
	pub fn new(
		a: &'a [Word],
		b: &'a [Word],
		c_lo: &'a [Word],
		c_hi: &'a [Word],
	) -> Result<Self, Error> {
		assert!(2 * WORD_SIZE_BITS <= F::N_BITS);

		// Statement should be of pow-2 length.
		let Some(n_vars) = strict_log_2(a.len()) else {
			return Err(Error::ExponentsPowerOfTwoLengthRequired);
		};

		// All statement slices should be of same length.
		if [a, b, c_lo, c_hi]
			.iter()
			.any(|exponents| exponents.len() != 1 << n_vars)
		{
			return Err(Error::ExponentLengthMismatch);
		}

		// The 2·N_LIMBS twisted power tables: tables[s][i] = (G^{2^{ws}})^i. The `a` and `c_lo`
		// limb columns read tables 0..N_LIMBS; the `c_hi` limb columns (base
		// G^{2^{WORD_SIZE_BITS}}) read tables N_LIMBS..2·N_LIMBS. Table 0 is the shared logup*
		// table.
		let g = F::MULTIPLICATIVE_GENERATOR;
		let bases = iterate(g, |g| g.square())
			.step_by(LIMB_BITS)
			.take(2 * N_LIMBS)
			.collect::<Vec<_>>();
		let tables = bases
			.into_par_iter()
			.map(|base| power_table::<F, P>(LIMB_BITS, base))
			.collect::<Vec<_>>();

		// Build the per-limb leaf layers and their prodcheck provers. Each prover's products
		// layer is the corresponding tree root.
		let a_leaves = limb_leaves(&tables[..N_LIMBS], a);
		let (a_prodcheck, a_root) = ProdcheckProver::new(LOG_N_LIMBS, a_leaves);

		let c_lo_leaves = limb_leaves(&tables[..N_LIMBS], c_lo);
		let (c_lo_prodcheck, c_lo_root) = ProdcheckProver::new(LOG_N_LIMBS, c_lo_leaves);

		let c_hi_leaves = limb_leaves(&tables[N_LIMBS..], c_hi);
		let (c_hi_prodcheck, c_hi_root) = ProdcheckProver::new(LOG_N_LIMBS, c_hi_leaves);

		// Compute b_leaves as concatenated leaves for prodcheck; the variable base is the `a` root.
		let b_leaves = compute_b_leaves(&a_root, b);

		// Create the prodcheck prover; its products layer becomes b_root
		let (b_prodcheck, b_root) = ProdcheckProver::new(LOG_WORD_SIZE_BITS, b_leaves.clone());

		Ok(Self {
			a_exponents: a,
			a_prodcheck,
			a_root,
			b_exponents: b,
			b_leaves,
			b_prodcheck,
			b_root,
			c_lo_exponents: c_lo,
			c_lo_prodcheck,
			c_lo_root,
			c_hi_exponents: c_hi,
			c_hi_prodcheck,
			c_hi_root,
			tables,
		})
	}
}

/// Build the power table of `base` with `2^log_size` rows: row `i` holds `base^i`.
pub fn power_table<F, P>(log_size: usize, base: F) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	let mut scalars = Vec::with_capacity(1 << log_size);
	let mut row = F::ONE;
	for _ in 0..1usize << log_size {
		scalars.push(row);
		row *= base;
	}
	FieldBuffer::<P>::from_values(&scalars)
}

/// Extract limb `l` of an exponent word as a table row index.
pub(super) const fn limb_index(word: Word, limb: usize) -> usize {
	((word.0 >> (limb * LIMB_BITS)) & ((1 << LIMB_BITS) - 1)) as usize
}

/// Build the concatenated per-limb leaf columns for a constant-base GKR exponentiation tree.
///
/// Column `l` has entries `tables[l][limb_l(e)]` — the limb-`l` exponentiation of the
/// corresponding word. The columns are concatenated into one `(n_vars + LOG_N_LIMBS)`-variate
/// buffer with the limb index in the high bits, so the prodcheck's node reductions pair column `z`
/// with column `z + N_LIMBS/2`.
fn limb_leaves<F, P>(tables: &[FieldBuffer<P>], exponents: &[Word]) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert_eq!(tables.len(), N_LIMBS);

	let n_vars = checked_log_2(exponents.len());
	let scalars = (0..N_LIMBS)
		.flat_map(|limb| {
			let table = &tables[limb];
			exponents
				.iter()
				.map(move |&word| table.get(limb_index(word, limb)))
		})
		.collect::<Vec<_>>();

	debug_assert_eq!(scalars.len(), 1 << (n_vars + LOG_N_LIMBS));
	FieldBuffer::<P>::from_values(&scalars)
}

/// Compute concatenated b_leaves for prodcheck.
///
/// Each leaf `L_z` contains: if bit z of `exponents[i]` is set then `bases[i]^{2^z}` else 1
/// The leaves are concatenated: `[L_0, L_1, ..., L_{2^k-1}]`
#[doc(hidden)] // exposed for benchmarking (`benches/intmul.rs`), not a stable API
pub fn compute_b_leaves<F, P>(bases: &FieldBuffer<P>, exponents: &[Word]) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	let n_vars = bases.log_len();

	if P::LOG_WIDTH <= n_vars {
		// Parallel optimized path
		return compute_b_leaves_parallel(bases, exponents);
	}

	// Fallback: bases is too small to parallelize (n_vars < P::LOG_WIDTH)
	let mut out = FieldBuffer::zeros(n_vars + LOG_WORD_SIZE_BITS);
	let n_elems = 1 << n_vars;

	for (i, (mut base, &exp)) in iter::zip(bases.iter_scalars(), exponents).enumerate() {
		for z in 0..WORD_SIZE_BITS {
			let bit = exp.extract_bit(z);

			out.set(z * n_elems + i, if bit { base } else { F::ONE });

			base = base.square();
		}
	}

	out
}

/// Parallel implementation of compute_b_leaves for when bases is large enough to parallelize.
fn compute_b_leaves_parallel<F, P>(bases: &FieldBuffer<P>, exponents: &[Word]) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	let n_vars = bases.log_len();
	let n_packed = bases.as_ref().len();
	let height = WORD_SIZE_BITS;
	let total = n_packed * height;

	let mut out_vec: Vec<P> = Vec::with_capacity(total);

	{
		let spare: &mut [MaybeUninit<P>] = out_vec.spare_capacity_mut();

		let mut strided = StridedArray2DViewMut::without_stride(spare, height, n_packed)
			.expect("dimensions match capacity");

		(strided.par_iter_cols(), bases.as_ref(), exponents.par_chunks(P::WIDTH))
			.into_par_iter()
			.for_each(|(mut col, packed_base, exp_chunk)| {
				// Keep base as packed element for efficient squaring
				let mut packed_base = *packed_base;

				for z in 0..height {
					// Decompose to scalars, apply bit selection, recompose
					// TODO: Optimize with bit-masking for selection
					let scalars = packed_base.iter().zip(exp_chunk).map(|(base, &exp)| {
						let bit = exp.extract_bit(z);
						if bit { base } else { F::ONE }
					});

					col[z].write(P::from_scalars(scalars));

					// Square packed base for next iteration
					packed_base = packed_base.square();
				}
			});
	}

	// SAFETY: All elements initialized in the parallel loop above
	unsafe { out_vec.set_len(total) };

	FieldBuffer::new(n_vars + LOG_WORD_SIZE_BITS, out_vec.into_boxed_slice())
}

/// Compute the per-vertex bivariate product of two equally sized field buffers.
pub fn buffer_bivariate_product<P: PackedField, Data: Deref<Target = [P]>>(
	a: &FieldBuffer<P, Data>,
	b: &FieldBuffer<P, Data>,
) -> FieldBuffer<P> {
	assert_eq!(a.len(), b.len());
	let product = (a.as_ref(), b.as_ref())
		.into_par_iter()
		.map(|(&a, &b)| a * b)
		.collect::<Box<[P]>>();
	FieldBuffer::new(a.log_len(), product)
}

/// Constructs a field buffer with values selected from `elements` based on the bit values
/// of `exponents`.
pub fn two_valued_field_buffer<F, P>(
	bit_offset: usize,
	exponents: &[Word],
	elements: [F; 2],
) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	let n_vars = checked_log_2(exponents.len());
	let p_width = P::WIDTH.min(1 << n_vars);
	let values = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
		.map(|i| {
			let scalars = (0..p_width).map(|j| {
				let index = i << P::LOG_WIDTH | j;
				// Select `elements[1]` if bit `bit_offset` of `exponents[index]` is set, else
				// `elements[0]`.
				unsafe {
					// Safety:
					// - `index` is guaranteed to be in-bounds
					// - `elements` has two values
					let is_set = exponents.get_unchecked(index).extract_bit(bit_offset);
					*elements.get_unchecked(is_set as usize)
				}
			});
			P::from_scalars(scalars)
		})
		.collect::<Box<[_]>>();

	FieldBuffer::new(n_vars, values)
}

#[cfg(test)]
mod tests {
	use binius_math::test_utils::Packed128b;

	use super::*;

	type P = Packed128b;

	fn check_consistency<P: PackedField>(witness: &Witness<'_, P>) {
		// The variable-base `b`-exponent tree root must equal the full product `c` root
		// (`c_lo_root * c_hi_root`); this equality is what lets the prover reuse `b_root` in place
		// of a separately stored `c_root`.
		let c_root = buffer_bivariate_product(witness.c_lo_root(), witness.c_hi_root());
		assert_eq!(witness.b_root(), &c_root);
	}

	#[test]
	fn test_forwards() {
		let a = [Word::from_u64(2)];
		let b = [Word::from_u64(3)];
		let c_lo = [Word::from_u64(6)]; // 2*3 = 6
		let c_hi = [Word::from_u64(0)]; // no high bits

		let witness = Witness::<P>::new(&a, &b, &c_lo, &c_hi).unwrap();
		check_consistency(&witness);
	}

	#[test]
	fn test_forwards_larger() {
		let a = [Word::from_u64(1 << 32)];
		let b = [Word::from_u64(1 << 33)];
		let c_lo = [Word::from_u64(0)];
		let c_hi = [Word::from_u64(2)]; // 2^32 * 2^33 = 2^65, which is 2 in the high 64 bits

		let witness = Witness::<P>::new(&a, &b, &c_lo, &c_hi).unwrap();
		check_consistency(&witness);
	}

	#[test]
	fn test_forwards_multiple_random() {
		use rand::prelude::*;

		let mut rng = StdRng::seed_from_u64(0);

		const VECTOR_SIZE: usize = 8;
		let mut a = Vec::with_capacity(VECTOR_SIZE);
		let mut b = Vec::with_capacity(VECTOR_SIZE);
		let mut c_lo = Vec::with_capacity(VECTOR_SIZE);
		let mut c_hi = Vec::with_capacity(VECTOR_SIZE);

		for _ in 0..VECTOR_SIZE {
			let a_i = rng.random_range(1..u64::MAX);
			let b_i = rng.random_range(1..u64::MAX);

			let full_result = (a_i as u128) * (b_i as u128);
			let c_lo_i = full_result as u64;
			let c_hi_i = (full_result >> 64) as u64;

			a.push(Word::from_u64(a_i));
			b.push(Word::from_u64(b_i));
			c_lo.push(Word::from_u64(c_lo_i));
			c_hi.push(Word::from_u64(c_hi_i));
		}

		let witness = Witness::<P>::new(&a, &b, &c_lo, &c_hi).unwrap();
		check_consistency(&witness);
	}
}
