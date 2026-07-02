// Copyright 2025 Irreducible Inc.

use std::{iter, marker::PhantomData, mem::MaybeUninit, ops::Deref};

use binius_field::{BinaryField, Field, PackedField};
use binius_ip_prover::prodcheck::ProdcheckProver;
use binius_math::field_buffer::FieldBuffer;
use binius_utils::{
	bitwise::{BitSelector, Bitwise},
	checked_arithmetics::{checked_log_2, strict_log_2},
	random_access_sequence::RandomAccessSequence,
	rayon::prelude::*,
	strided_array::StridedArray2DViewMut,
};
use getset::Getters;
use itertools::iterate;

use super::error::Error;

/// An integer multiplication protocol witness. Created from integer slices, consumed during
/// proving.
///
/// The statement being proven is `a * b = c`, where `c` is represented as a pair `(c_lo, c_hi)`.
/// All four values are of the same bit width that is passed to the prover via `log_bits` parameter
/// (also denoted $m$). In Binius64, `log_bits = 6` for 64-bit multiplicands and 128-bit product.
///
/// For each of `a`, `c_lo`, `c_hi`, `b` we build the `2^log_bits` "selected" leaf multilinears of
/// the bivariate-product GKR tree (the widest layer), concatenated into one `(n_vars + log_bits)`
/// variate buffer with the node index in the high bits. We then construct a [`ProdcheckProver`]
/// over each, retaining the leaf layer for the final (Phase 5) GKR step:
///  1) `a` and `c_lo` select a multiplicative group generator $G$
///  2) `c_hi` selects $G^{2^{2^m}}$
///  3) `b` selects variable base which is equal to the root of the `a` tree
///
/// Protocol proves that ${(G^a)}^b = G^{c\\_lo} \times (G^{2^{2^m}})^{c\\_hi}$, which is equivalent
/// to $a \times b = c$ modulo $2^{2^{m+1}} - 1$. The special case of `0 * 0 = 1` is handled
/// separately.
#[derive(Clone, Getters)]
#[getset(get = "pub")]
pub struct Witness<P: PackedField, B: Bitwise, S: AsRef<[B]> + Sync> {
	/// The log of the bit width ($m$): the tree leaf layer has `2^log_bits` selected multilinears.
	pub log_bits: usize,
	/// The exponents for `a` (needed for the phase 5 parity zerocheck on `a_0`).
	pub a_exponents: S,
	/// Prodcheck prover for the `a` exponentiation tree (leaf layer retained).
	pub a_prodcheck: ProdcheckProver<P>,
	/// The root of the `a` tree (product of all leaves element-wise); the `b` variable base.
	pub a_root: FieldBuffer<P>,
	/// The exponents for `b` (needed for phase 5).
	pub b_exponents: S,
	/// Concatenated b leaves for prodcheck: [L_0, L_1, ..., L_{2^k-1}].
	/// Has log_len = n_vars + log_bits.
	pub b_leaves: FieldBuffer<P>,
	/// The prover for the prodcheck reduction on b_leaves.
	pub b_prodcheck: ProdcheckProver<P>,
	/// The root of the b tree (product of all leaves element-wise).
	pub b_root: FieldBuffer<P>,
	/// The exponents for `c_lo` (needed for the phase 5 parity zerocheck on `c_lo_0`).
	pub c_lo_exponents: S,
	/// Prodcheck prover for the `c_lo` exponentiation tree (leaf layer retained).
	pub c_lo_prodcheck: ProdcheckProver<P>,
	/// The root of the `c_lo` tree.
	pub c_lo_root: FieldBuffer<P>,
	/// Prodcheck prover for the `c_hi` exponentiation tree (leaf layer retained).
	pub c_hi_prodcheck: ProdcheckProver<P>,
	/// The root of the `c_hi` tree.
	pub c_hi_root: FieldBuffer<P>,
	/// The root of a `log_bits + 1` deep tree of the full product `c` (`c_lo_root * c_hi_root`).
	pub c_root: FieldBuffer<P>,
	#[getset(skip)]
	pub _b_marker: PhantomData<B>,
}

impl<F, P, B, S> Witness<P, B, S>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	B: Bitwise,
	S: AsRef<[B]> + Sync,
{
	/// Constructs a new integer multiplication witness from the statement.
	///
	/// For statement of size $2^\ell$ using $2^m$-wide integers, the upper bound on the
	/// witness size is $8^{\ell+m}$ large field elements.
	pub fn new(log_bits: usize, a: S, b: S, c_lo: S, c_hi: S) -> Result<Self, Error> {
		// Statement should be of pow-2 length.
		let Some(n_vars) = strict_log_2(a.as_ref().len()) else {
			return Err(Error::ExponentsPowerOfTwoLengthRequired);
		};

		// All statement slices should be of same length.
		if [&a, &b, &c_lo, &c_hi]
			.iter()
			.any(|exponents| exponents.as_ref().len() != 1 << n_vars)
		{
			return Err(Error::ExponentLengthMismatch);
		}

		let g = F::MULTIPLICATIVE_GENERATOR;
		let g_c_hi = iterate(g, |g| g.square())
			.nth(1 << log_bits)
			.expect("infinite iterator");

		// Build the constant-base leaf layers and their prodcheck provers. Each prover's products
		// layer is the corresponding tree root.
		let a_leaves = constant_base_leaves(log_bits, g, &a);
		let (a_prodcheck, a_root) = ProdcheckProver::new(log_bits, a_leaves);

		let c_lo_leaves = constant_base_leaves(log_bits, g, &c_lo);
		let (c_lo_prodcheck, c_lo_root) = ProdcheckProver::new(log_bits, c_lo_leaves);

		let c_hi_leaves = constant_base_leaves(log_bits, g_c_hi, &c_hi);
		let (c_hi_prodcheck, c_hi_root) = ProdcheckProver::new(log_bits, c_hi_leaves);

		// Compute b_leaves as concatenated leaves for prodcheck; the variable base is the `a` root.
		let b_leaves = compute_b_leaves(log_bits, a_root.clone(), &b);

		// Create the prodcheck prover; its products layer becomes b_root
		let (b_prodcheck, b_root) = ProdcheckProver::new(log_bits, b_leaves.clone());

		// The root of a `log_bits + 1` deep tree of the full product `c`.
		let c_root = buffer_bivariate_product(&c_lo_root, &c_hi_root);

		Ok(Self {
			log_bits,
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
			c_hi_prodcheck,
			c_hi_root,
			c_root,
			_b_marker: PhantomData,
		})
	}
}

/// Build the concatenated constant-base leaf layer for a GKR exponentiation tree.
///
/// The widest layer of the tree contains `2^log_bits` selected multilinears: the leaf for bit `b`
/// has value `base^{2^b}` where the `b`-th bit of the corresponding exponent is set, and `1`
/// otherwise.
///
/// The leaves are concatenated into one `(n_vars + log_bits)`-variate buffer with the node index in
/// the high bits, in natural bit order: node position `p` carries the leaf for bit `p`. This
/// matches the `b`-tree layout ([`compute_b_leaves`]). The prodcheck reduces on the highest node
/// bit, so its first reduction (and the verifier's final GKR layer) pairs leaf `z` with leaf
/// `z + 2^{log_bits-1}` — a strided pairing of bits `z` and `z + 2^{log_bits-1}`.
fn constant_base_leaves<F, P, B, S>(log_bits: usize, base: F, exponents: &S) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
	B: Bitwise,
	S: AsRef<[B]> + Sync,
{
	let bases = iterate(base, |g| g.square())
		.take(1 << log_bits)
		.collect::<Vec<_>>();

	let n_vars = checked_log_2(exponents.as_ref().len());
	let scalars = (0..1 << log_bits)
		.flat_map(|bit| {
			let leaf = two_valued_field_buffer::<F, P, S, B>(bit, exponents, [F::ONE, bases[bit]]);
			leaf.iter_scalars().collect::<Vec<_>>()
		})
		.collect::<Vec<_>>();

	debug_assert_eq!(scalars.len(), 1 << (n_vars + log_bits));
	FieldBuffer::<P>::from_values(&scalars)
}

/// Compute concatenated b_leaves for prodcheck.
///
/// Each leaf `L_z` contains: if bit z of `exponents[i]` is set then `bases[i]^{2^z}` else 1
/// The leaves are concatenated: `[L_0, L_1, ..., L_{2^k-1}]`
fn compute_b_leaves<F, P, B, S>(
	log_bits: usize,
	bases: FieldBuffer<P>,
	exponents: &S,
) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
	B: Bitwise,
	S: AsRef<[B]> + Sync,
{
	let n_vars = bases.log_len();

	if P::LOG_WIDTH <= n_vars {
		// Parallel optimized path
		return compute_b_leaves_parallel(log_bits, &bases, exponents);
	}

	// Fallback: bases is too small to parallelize (n_vars < P::LOG_WIDTH)
	let mut out = FieldBuffer::zeros(n_vars + log_bits);
	let n_elems = 1 << n_vars;

	let one_bit = B::from(1u8);
	for (i, (mut base, &exp)) in iter::zip(bases.iter_scalars(), exponents.as_ref()).enumerate() {
		for z in 0..1 << log_bits {
			let bit = (exp >> z) & one_bit == one_bit;

			out.set(z * n_elems + i, if bit { base } else { F::ONE });

			base = base.square();
		}
	}

	out
}

/// Parallel implementation of compute_b_leaves for when bases is large enough to parallelize.
fn compute_b_leaves_parallel<F, P, B, S>(
	log_bits: usize,
	bases: &FieldBuffer<P>,
	exponents: &S,
) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
	B: Bitwise,
	S: AsRef<[B]> + Sync,
{
	let n_vars = bases.log_len();
	let n_packed = bases.as_ref().len();
	let height = 1 << log_bits;
	let total = n_packed * height;

	let mut out_vec: Vec<P> = Vec::with_capacity(total);

	{
		let spare: &mut [MaybeUninit<P>] = out_vec.spare_capacity_mut();

		let mut strided = StridedArray2DViewMut::without_stride(spare, height, n_packed)
			.expect("dimensions match capacity");

		let one_bit = B::from(1u8);

		(strided.par_iter_cols(), bases.as_ref(), exponents.as_ref().par_chunks(P::WIDTH))
			.into_par_iter()
			.for_each(|(mut col, packed_base, exp_chunk)| {
				// Keep base as packed element for efficient squaring
				let mut packed_base = *packed_base;

				for z in 0..height {
					// Decompose to scalars, apply bit selection, recompose
					// TODO: Optimize with bit-masking for selection
					let scalars = packed_base.iter().zip(exp_chunk).map(|(base, &exp)| {
						let bit = (exp >> z) & one_bit == one_bit;
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

	FieldBuffer::new(n_vars + log_bits, out_vec.into_boxed_slice())
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
pub fn two_valued_field_buffer<F, P, S, B>(
	bit_offset: usize,
	exponents: &S,
	elements: [F; 2],
) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
	S: AsRef<[B]> + Sync,
	B: Bitwise,
{
	let n_vars = checked_log_2(exponents.as_ref().len());
	let p_width = P::WIDTH.min(1 << n_vars);
	let bits = BitSelector::new(bit_offset, &exponents);
	let values = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
		.map(|i| {
			let scalars = (0..p_width).map(|j| {
				// The following code is equivalent to
				// ```
				// if bits.get(i << P::LOG_WIDTH | j) {
				// 	elements[1]
				// } else {
				// 	elements[0]
				// }
				// ```
				unsafe {
					// Safety:
					// - `i << P::LOG_WIDTH | j` is guaranteed to be in-bounds
					// - elements has two values
					*elements.get_unchecked(bits.get_unchecked(i << P::LOG_WIDTH | j) as usize)
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

	const LOG_BITS: usize = 6;

	fn check_consistency<P: PackedField, B: Bitwise, S: AsRef<[B]> + Sync>(
		witness: &Witness<P, B, S>,
	) {
		let b_root = witness.b_root();
		let c_root = witness.c_root();
		assert_eq!(b_root, c_root);
	}

	#[test]
	fn test_forwards() {
		let a: u64 = 2;
		let b: u64 = 3;
		let c_lo: u64 = 6; // 2*3 = 6
		let c_hi: u64 = 0; // no high bits

		let witness = Witness::<P, _, [u64; 1]>::new(LOG_BITS, [a], [b], [c_lo], [c_hi]).unwrap();
		check_consistency(&witness);
	}

	#[test]
	fn test_forwards_larger() {
		let a: u64 = 1 << 32;
		let b: u64 = 1 << 33;
		let c_lo: u64 = 0;
		let c_hi: u64 = 2; // 2^32 * 2^33 = 2^65, which is 2 in the high 64 bits

		let witness = Witness::<P, _, [u64; 1]>::new(LOG_BITS, [a], [b], [c_lo], [c_hi]).unwrap();
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

			a.push(a_i);
			b.push(b_i);
			c_lo.push(c_lo_i);
			c_hi.push(c_hi_i);
		}

		let witness = Witness::<P, _, _>::new(LOG_BITS, a, b, c_lo, c_hi).unwrap();
		check_consistency(&witness);
	}
}
