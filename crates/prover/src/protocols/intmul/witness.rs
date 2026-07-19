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
use binius_verifier::protocols::intmul::common::{LIMB_BITS, LOG_N_LIMBS, N_LIMBS};
use getset::Getters;
use itertools::iterate;

use super::error::Error;

/// An integer multiplication protocol witness. Created from integer slices, consumed during
/// proving.
///
/// The statement being proven is `a * b = c`, where `c` is represented as a pair `(c_lo, c_hi)`:
/// `Word::BITS`-wide multiplicands with a double-wide product.
///
/// For each of `a`, `c_lo`, `c_hi` the exponent words split into `N_LIMBS` limbs, and the
/// constant-base exponentiation factors as a product of `N_LIMBS` limb columns — column `l` reads
/// the row `limb_l(e)` of the power table of the base $G^{2^{wl}}$ (where $w$ is the limb bit
/// width). The columns are concatenated into one `(n_vars + LOG_N_LIMBS)`-variate buffer with the
/// limb index in the high bits, and a [`ProdcheckProver`] is constructed over each:
///  1) `a` and `c_lo` exponentiate the multiplicative group generator $G$
///  2) `c_hi` exponentiates $G^{2^{2^m}}$
///  3) `b` selects a variable base (the root of the `a` tree) per bit, over `Word::BITS` per-bit
///     leaves as before
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
	/// Has log_len = n_vars + Word::LOG_BITS.
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
		assert!(2 * Word::BITS <= F::N_BITS);

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
		// G^{2^{Word::BITS}}) read tables N_LIMBS..2·N_LIMBS. Table 0 is the shared logup*
		// table.
		let power_table_scope = tracing::debug_span!("Build power tables").entered();
		let g = F::MULTIPLICATIVE_GENERATOR;
		let bases = iterate(g, |g| g.square())
			.step_by(LIMB_BITS)
			.take(2 * N_LIMBS)
			.collect::<Vec<_>>();
		// Allocate the table buffers up front on this thread, so the parallel region only fills
		// them — no allocator traffic inside the rayon closures.
		let packed_len = 1 << LIMB_BITS.saturating_sub(P::LOG_WIDTH);
		let buffers = iter::repeat_with(|| Vec::<P>::with_capacity(packed_len))
			.take(bases.len())
			.collect::<Vec<_>>();
		let tables = (bases, buffers)
			.into_par_iter()
			.map(|(base, buffer)| power_table_into(LIMB_BITS, base, buffer))
			.collect::<Vec<_>>();
		drop(power_table_scope);

		// Build the per-limb leaf layers and their prodcheck provers. Each prover's products
		// layer is the corresponding tree root.
		let fixed_base_tree_scope =
			tracing::debug_span!("Compute fixed-base prodcheck layers").entered();
		let a_leaves = limb_leaves(&tables[..N_LIMBS], a);
		let (a_prodcheck, a_root) = ProdcheckProver::new(LOG_N_LIMBS, a_leaves);

		let c_lo_leaves = limb_leaves(&tables[..N_LIMBS], c_lo);
		let (c_lo_prodcheck, c_lo_root) = ProdcheckProver::new(LOG_N_LIMBS, c_lo_leaves);

		let c_hi_leaves = limb_leaves(&tables[N_LIMBS..], c_hi);
		let (c_hi_prodcheck, c_hi_root) = ProdcheckProver::new(LOG_N_LIMBS, c_hi_leaves);
		drop(fixed_base_tree_scope);

		// Compute b_leaves as concatenated leaves for prodcheck; the variable base is the `a` root.
		let variable_base_tree_scope =
			tracing::debug_span!("Compute variable-base prodcheck layers").entered();
		let b_leaves = compute_b_leaves(&a_root, b);
		let (b_prodcheck, b_root) = ProdcheckProver::new(Word::LOG_BITS, b_leaves.clone());
		drop(variable_base_tree_scope);

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

/// Number of packed columns filled as one independent group before chaining to the next block.
///
/// The power chain `base^i` is inherently serial; striding the fill into blocks of
/// `2^LOG_STRIDE` independent packed multiplies exposes instruction-level parallelism so the
/// multiplier pipeline stays busy. `4` keeps a block (16 packed elements) comfortably in registers.
const LOG_STRIDE: usize = 4;

/// The first `count` powers of `base` (`base^0 .. base^(count-1)`) plus the next power
/// `base^count`.
fn sequential_powers<F: Field>(count: usize, base: F) -> (Vec<F>, F) {
	let mut scalars = Vec::with_capacity(count);
	let mut row = F::ONE;
	for _ in 0..count {
		scalars.push(row);
		row *= base;
	}
	(scalars, row)
}

/// Build the power table of `base` with `2^log_size` rows: row `i` holds `base^i`.
pub fn power_table<F, P>(log_size: usize, base: F) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	let buffer = Vec::with_capacity(1 << log_size.saturating_sub(P::LOG_WIDTH));
	power_table_into(log_size, base, buffer)
}

/// Fill `buffer` with the power table of `base` and wrap it as a [`FieldBuffer`] of `2^log_size`
/// rows: row `i` holds `base^i`.
///
/// The caller supplies the backing `Vec`, so its allocation can be hoisted out of a hot or parallel
/// region; `buffer` is cleared and refilled here without reallocating.
///
/// # Preconditions
///
/// * `buffer.capacity()` must equal `1 << log_size.saturating_sub(P::LOG_WIDTH)`.
fn power_table_into<F, P>(log_size: usize, base: F, mut buffer: Vec<P>) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	let packed_len = 1 << log_size.saturating_sub(P::LOG_WIDTH);
	assert_eq!(
		buffer.capacity(),
		packed_len,
		"precondition: buffer capacity must match the packed table length"
	);
	buffer.clear();

	// The first block spans `2^log_block` scalars = `2^LOG_STRIDE` packed elements.
	let log_block = P::LOG_WIDTH + LOG_STRIDE;

	// Small tables (at most one block) don't benefit from striding; build them sequentially. This
	// also covers `log_size < P::LOG_WIDTH` (a single partially-filled packed element).
	if log_size <= log_block {
		let (scalars, _) = sequential_powers(1 << log_size, base);
		buffer.extend(
			scalars
				.chunks(P::WIDTH)
				.map(|chunk| P::from_scalars(chunk.iter().copied())),
		);
		return FieldBuffer::new(log_size, buffer.into_boxed_slice());
	}

	// Build the first block's `2^log_block` sequential powers, packed into `2^LOG_STRIDE` elements;
	// `incr = base^(2^log_block)` is the per-block increment.
	let (block_scalars, incr_scalar) = sequential_powers(1 << log_block, base);
	let incr = P::broadcast(incr_scalar);
	buffer.extend(
		block_scalars
			.chunks(P::WIDTH)
			.map(|chunk| P::from_scalars(chunk.iter().copied())),
	);

	// Fill each remaining packed element from the one a block earlier; the `block_len` multiplies
	// within a block are independent, so only the block-to-block step carries a dependency.
	let block_len = 1 << LOG_STRIDE; // packed elements per block
	for i in block_len..packed_len {
		let next = buffer[i - block_len] * incr;
		buffer.push(next);
	}

	FieldBuffer::new(log_size, buffer.into_boxed_slice())
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
	let mut out = FieldBuffer::zeros(n_vars + Word::LOG_BITS);
	let n_elems = 1 << n_vars;

	for (i, (mut base, &exp)) in iter::zip(bases.iter_scalars(), exponents).enumerate() {
		for z in 0..Word::BITS {
			// Branchless select of `base` when bit `z` is set, else `F::ONE`: on the selected lane
			// `mask` is all-ones so `select` keeps `base - 1` and the `+ 1` restores `base`; on the
			// unselected lane `select` yields `0` and the `+ 1` gives `F::ONE`.
			let mask = F::make_mask(iter::once(exp.extract_bit(z)));
			out.set(z * n_elems + i, F::ONE + (base - F::ONE).select(&mask));

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
	let height = Word::BITS;
	let total = n_packed * height;

	let mut out_vec: Vec<P> = Vec::with_capacity(total);

	{
		let spare: &mut [MaybeUninit<P>] = out_vec.spare_capacity_mut();

		let mut strided = StridedArray2DViewMut::without_stride(spare, height, n_packed)
			.expect("dimensions match capacity");

		let ones = P::broadcast(F::ONE);
		(strided.par_iter_cols(), bases.as_ref(), exponents.par_chunks(P::WIDTH))
			.into_par_iter()
			.for_each(|(mut col, packed_base, exp_chunk)| {
				// Keep base as packed element for efficient squaring
				let mut packed_base = *packed_base;

				for z in 0..height {
					// Branchless select, per lane, of `base` when bit `z` of the lane's exponent is
					// set, else `F::ONE`. On selected lanes `mask` is all-ones so `select` keeps
					// `base - 1` and the `+ 1` restores `base`; on unselected lanes `select` yields
					// `0` and the `+ 1` gives `F::ONE`.
					let mask = P::make_mask(exp_chunk.iter().map(|&exp| exp.extract_bit(z)));
					col[z].write(ones + (packed_base - ones).select(&mask));

					// Square packed base for next iteration
					packed_base = packed_base.square();
				}
			});
	}

	// SAFETY: All elements initialized in the parallel loop above
	unsafe { out_vec.set_len(total) };

	FieldBuffer::new(n_vars + Word::LOG_BITS, out_vec.into_boxed_slice())
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
	/// Directly checks `compute_b_leaves` against its specification — leaf `z` holds
	/// `bases[i]^(2^z)` where bit `z` of `exponents[i]` is set, else `F::ONE` — over both the
	/// parallel path (`n_vars >= P::LOG_WIDTH`) and the scalar fallback (`n_vars < P::LOG_WIDTH`).
	#[test]
	fn compute_b_leaves_matches_spec() {
		use binius_field::{BinaryField128bGhash, Random, arithmetic_traits::Square};
		use rand::prelude::*;

		type F = BinaryField128bGhash;

		let mut rng = StdRng::seed_from_u64(1);
		// `Packed128b` has `LOG_WIDTH == 2`: `n_vars = 0` exercises the scalar fallback and
		// `n_vars = 4` the parallel path.
		for n_vars in [0usize, 4] {
			let n_elems = 1 << n_vars;
			let base_scalars = (0..n_elems)
				.map(|_| F::random(&mut rng))
				.collect::<Vec<_>>();
			let bases = FieldBuffer::<P>::from_values(&base_scalars);
			let exponents = (0..n_elems)
				.map(|_| Word::from_u64(rng.random()))
				.collect::<Vec<_>>();

			let leaves = compute_b_leaves::<F, P>(&bases, &exponents);

			for (i, &base0) in base_scalars.iter().enumerate() {
				let mut base = base0;
				for z in 0..Word::BITS {
					let expected = if exponents[i].extract_bit(z) {
						base
					} else {
						F::ONE
					};
					assert_eq!(
						leaves.get(z * n_elems + i),
						expected,
						"mismatch at n_vars={n_vars}, i={i}, z={z}"
					);
					base = base.square();
				}
			}
		}
	}

	/// Checks `power_table` against a sequential reference (`row i == base^i`) across both the
	/// small sequential fallback (`log_size <= P::LOG_WIDTH + LOG_STRIDE`) and the strided packed
	/// path.
	#[test]
	fn power_table_matches_sequential() {
		use binius_field::{BinaryField128bGhash, Random};
		use rand::prelude::*;

		type F = BinaryField128bGhash;

		let mut rng = StdRng::seed_from_u64(2);
		let base = F::random(&mut rng);
		// `Packed128b` has `LOG_WIDTH == 2`, so `log_block == 6`: sizes up to 6 take the sequential
		// fallback; 7 and 10 (multiple blocks) exercise the strided path.
		for log_size in [0usize, 3, 6, 7, 10] {
			let table = power_table::<F, P>(log_size, base);
			let mut expected = F::ONE;
			for i in 0..1usize << log_size {
				assert_eq!(table.get(i), expected, "mismatch at log_size={log_size}, i={i}");
				expected *= base;
			}
		}
	}
}
