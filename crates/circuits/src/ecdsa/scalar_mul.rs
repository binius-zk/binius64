// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
use std::iter;

use binius_core::{consts::WORD_SIZE_BITS, word::Word};
use binius_frontend::{CircuitBuilder, Wire};

use crate::{
	bignum::{BigUint, assert_eq, select as select_biguint},
	multiplexer::multi_wire_multiplex,
	secp256k1::{
		N_LIMBS, Secp256k1, Secp256k1Affine, coord_lambda, coord_zero,
		select as select_secp256k1_affine,
	},
};

/// Compute scalar multiplication `point * scalar` using the naive double-and-add algorithm.
///
/// This implementation does not use the secp256k1 endomorphism optimization.
///
/// # Parameters
/// - `b`: The circuit builder
/// - `curve`: The secp256k1 curve instance
/// - `bits`: Number of bits to process in the scalar
/// - `scalar`: The scalar to multiply by (as a BigUint, must have enough limbs for bits)
/// - `point`: The point to multiply (in affine coordinates)
///
/// # Returns
/// The result of `point * scalar` in affine coordinates
pub fn scalar_mul_naive(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	bits: usize,
	scalar: &BigUint,
	point: Secp256k1Affine,
) -> Secp256k1Affine {
	// Ensure scalar has enough limbs for the requested bits
	let required_limbs = bits.div_ceil(WORD_SIZE_BITS);
	assert!(
		scalar.limbs.len() >= required_limbs,
		"scalar must have at least {} limbs for {} bits, but has {}",
		required_limbs,
		bits,
		scalar.limbs.len()
	);

	let mut acc = Secp256k1Affine::point_at_infinity(b);

	for bit_index in (0..bits).rev() {
		let limb = bit_index / WORD_SIZE_BITS;
		let bit = bit_index % WORD_SIZE_BITS;

		if bit_index != bits - 1 {
			acc = curve.double(b, &acc);
		}

		let scalar_bit = b.shl(scalar.limbs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);

		// Add the selected point to the accumulator
		let acc_plus_point = curve.add_incomplete(b, &acc, &point);

		// Select whether to add point to accumulator based on scalar bit
		acc = select_secp256k1_affine(b, scalar_bit, &acc_plus_point, &acc);
	}

	acc
}

/// Compute scalar multiplication `point * scalar` using the secp256k1 endomorphism optimization.
///
/// This implementation uses the curve's endomorphism to split the scalar into two ~128-bit
/// components, reducing the number of doublings from 256 to 128.
///
/// # Parameters
/// - `b`: The circuit builder
/// - `curve`: The secp256k1 curve instance
/// - `scalar`: The scalar to multiply by (as a BigUint with N_LIMBS)
/// - `point`: The point to multiply (in affine coordinates)
///
/// # Returns
///
/// The result of `point * scalar` in affine coordinates
pub fn scalar_mul(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	scalar: &BigUint,
	point: Secp256k1Affine,
) -> Secp256k1Affine {
	assert_eq!(scalar.limbs.len(), N_LIMBS);

	// Nondeterministically split the scalar, constrain the split
	let (k1_neg, k2_neg, k1_abs, k2_abs) = b.secp256k1_endomorphism_split_hint(&scalar.limbs);

	check_endomorphism_split(b, curve, k1_neg, k2_neg, k1_abs, k2_abs, scalar);

	// Compute the endomorphism of the point
	let point_endo = curve.endomorphism(b, &point);

	// The split returns "signed scalars" (which is required to fit them into 128 bits).
	// Negate the base if needed to only care about positive exponents.
	let p1 = curve.negate_if(b, k1_neg, &point);
	let p2 = curve.negate_if(b, k2_neg, &point_endo);

	// Compute the 4-element lookup table: {0, P1, P2, P1+P2}
	let lookup = vec![
		Secp256k1Affine::point_at_infinity(b),
		p1.clone(),
		p2.clone(),
		curve.add_incomplete(b, &p1, &p2),
	];

	let mut acc = Secp256k1Affine::point_at_infinity(b);

	for bit_index in (0..128).rev() {
		let limb = bit_index / WORD_SIZE_BITS;
		let bit = bit_index % WORD_SIZE_BITS;

		if bit_index != 127 {
			acc = curve.double(b, &acc);
		}

		// Extract the current bit from each scalar component
		let k1_bit = b.shl(k1_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
		let k2_bit = b.shl(k2_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);

		// Perform 2-bit lookup using nested selection
		// This selects one of the 4 lookup table entries based on the two bits
		let mut level = lookup.clone();
		for sel_bit in [k1_bit, k2_bit] {
			let next_level = level
				.chunks(2)
				.map(|pair| {
					assert_eq!(pair.len(), 2);
					select_secp256k1_affine(b, sel_bit, &pair[1], &pair[0])
				})
				.collect();
			level = next_level;
		}

		assert_eq!(level.len(), 1);
		acc = curve.add_incomplete(b, &acc, &level[0]);
	}

	acc
}

/// A common trick to save doublings when computing multiexponentiations of the form
/// `G*g_mult + PK*pk_mult` - instead of doing two scalar multiplications separately and
/// adding their results, we share the doubling step of double-and-add.
///
/// For secp256k1, we can go one step further: the curve has an endomorphism `λ (x, y) = (βx, y)`
/// where `λ³=1 (mod n)` and `β³=1 (mod p)` (`n` being the scalar field modulus and `p` coordinate
/// field one). For a 256-bit scalar `k` it is possible to split it into `k1` and `k2` such that
/// `k1 + λ k2 = k (mod n)` and both `k1` and `k2` are no farther than `2^128` from zero.
///
/// Using the above fact, we can "split" both the G and PK 256-bit multiplier scalars into a total
/// of four 128-bit subscalars. Instead of 4-wide lookup in `shamirs_trick_naive`, we do a 16-wide
/// lookup for all subset sums of `{G, G_endo, PK, PK_endo}`, where `*_endo` points are obtained via
/// endomorphism. This halves the total number of doublings and additions at a cost of a larger
/// precomputation, but the eventual savings are still in the order of 2x.
///
/// Returns `G*g_mult + PK*pk_mult`.
pub fn shamirs_trick_endomorphism(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	g_mult: &BigUint,
	pk_mult: &BigUint,
	pk: Secp256k1Affine,
) -> Secp256k1Affine {
	assert_eq!(g_mult.limbs.len(), N_LIMBS);
	assert_eq!(pk_mult.limbs.len(), N_LIMBS);

	// Nondeterministically split both scalars, constrain the splits
	let (g1_mult_neg, g2_mult_neg, g1_mult_abs, g2_mult_abs) =
		b.secp256k1_endomorphism_split_hint(&g_mult.limbs);

	check_endomorphism_split(b, curve, g1_mult_neg, g2_mult_neg, g1_mult_abs, g2_mult_abs, g_mult);

	let (pk1_mult_neg, pk2_mult_neg, pk1_mult_abs, pk2_mult_abs) =
		b.secp256k1_endomorphism_split_hint(&pk_mult.limbs);

	check_endomorphism_split(
		b,
		curve,
		pk1_mult_neg,
		pk2_mult_neg,
		pk1_mult_abs,
		pk2_mult_abs,
		pk_mult,
	);

	// Compute the endomorphisms
	let g = Secp256k1Affine::generator(b);
	let g_endo = curve.endomorphism(b, &g);
	let pk_endo = curve.endomorphism(b, &pk);

	// The split returns "signed scalars" (which is required to fit them into 128 bits).
	// Negate the base if needed to only care about positive exponents.
	let g1 = curve.negate_if(b, g1_mult_neg, &g);
	let g2 = curve.negate_if(b, g2_mult_neg, &g_endo);

	let pk1 = curve.negate_if(b, pk1_mult_neg, &pk);
	let pk2 = curve.negate_if(b, pk2_mult_neg, &pk_endo);

	// Compute subset sums of {G, G_endo, PK, PK_endo} using a total of 11 additions
	let mut lookup = Vec::with_capacity(16);
	lookup.push(Secp256k1Affine::point_at_infinity(b));
	for (i, pt) in [g1, g2, pk1, pk2].into_iter().enumerate() {
		lookup.push(pt.clone());
		for j in 1..1 << i {
			lookup.push(curve.add_incomplete(b, &lookup[j], &pt));
		}
	}

	let mut acc = Secp256k1Affine::point_at_infinity(b);

	for bit_index in (0..128).rev() {
		let limb = bit_index / WORD_SIZE_BITS;
		let bit = bit_index % WORD_SIZE_BITS;

		if bit_index != 127 {
			acc = curve.double(b, &acc);
		}

		// This is essentially an inlined multi wire multiplexer, but due to the fact
		// it uses affine point conditional selections and separate wires instead of masks
		// it's simpler to inline it there.
		// TODO: replace it with a multiplexer once the abstraction is mature enough
		let g1_mult_bit = b.shl(g1_mult_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
		let g2_mult_bit = b.shl(g2_mult_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
		let pk1_mult_bit = b.shl(pk1_mult_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
		let pk2_mult_bit = b.shl(pk2_mult_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);

		let mut level = lookup.clone();
		for sel_bit in [g1_mult_bit, g2_mult_bit, pk1_mult_bit, pk2_mult_bit] {
			let next_level = level
				.chunks(2)
				.map(|pair| {
					assert_eq!(pair.len(), 2);
					select_secp256k1_affine(b, sel_bit, &pair[1], &pair[0])
				})
				.collect();

			level = next_level;
		}

		assert_eq!(level.len(), 1);
		acc = curve.add_incomplete(b, &acc, &level[0]);
	}

	acc
}

/// Compute a multi-scalar multiplication `Σ_i scalars[i] · points[i]` over secp256k1 using
/// Shamir's trick combined with the curve endomorphism.
///
/// `n = points.len()` must equal `scalars.len()` and is statically known to the circuit. Each
/// scalar is a 256-bit value (`N_LIMBS` limbs). Using the endomorphism `λ`, every
/// `(point, scalar)` pair is split into two ~128-bit signed subscalars and two base points (`P`
/// and `endo(P)`), each conditionally negated so that only positive subscalars remain. This
/// yields `2n` base points, a `2^(2n)`-entry table of all their subset sums, and a main loop of
/// just 128 conditional additions (versus 256 without the endomorphism — see [`msm_naive`]). It
/// is the `n`-point generalization of [`shamirs_trick_endomorphism`] (which is the `n = 2` special
/// case with the generator as one of the points).
///
/// The endomorphism halves the number of doublings at the cost of squaring the table size, so it
/// wins for small `n` but is overtaken by [`msm_naive`] once the `2^(2n)` precomputation dominates.
///
/// # Completeness gap
///
/// Point additions use [`Secp256k1::add_incomplete`], which asserts false when its inputs are
/// equal (it handles the point at infinity but not doubling). As with [`scalar_mul`] and
/// [`shamirs_trick_endomorphism`], the probability of the accumulator or a table entry hitting
/// such a collision for independent inputs is vanishingly low.
///
/// # Panics
///
/// Panics if `scalars.len() != points.len()`, if `n == 0`, or if any scalar does not have exactly
/// `N_LIMBS` limbs.
pub fn msm(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	scalars: &[BigUint],
	points: &[Secp256k1Affine],
) -> Secp256k1Affine {
	let n = points.len();
	assert_eq!(scalars.len(), n, "scalars and points must have the same length");
	assert!(n >= 1, "MSM requires at least one point");

	// Split every scalar via the endomorphism, collecting the 2n positive base points together
	// with their 128-bit subscalar magnitudes (each is `[Wire; 2]`, little-endian).
	let mut base_points = Vec::with_capacity(2 * n);
	let mut subscalars = Vec::with_capacity(2 * n);
	for (scalar, point) in iter::zip(scalars, points) {
		assert_eq!(scalar.limbs.len(), N_LIMBS);

		let (k1_neg, k2_neg, k1_abs, k2_abs) = b.secp256k1_endomorphism_split_hint(&scalar.limbs);
		check_endomorphism_split(b, curve, k1_neg, k2_neg, k1_abs, k2_abs, scalar);

		let point_endo = curve.endomorphism(b, point);
		base_points.push(curve.negate_if(b, k1_neg, point));
		subscalars.push(k1_abs);
		base_points.push(curve.negate_if(b, k2_neg, &point_endo));
		subscalars.push(k2_abs);
	}

	let subscalar_refs = subscalars
		.iter()
		.map(<[Wire; 2]>::as_slice)
		.collect::<Vec<_>>();
	// Each subscalar magnitude fits in 128 bits, so 128 conditional additions suffice.
	shamir_accumulate(b, curve, &base_points, &subscalar_refs, 128)
}

/// Compute a multi-scalar multiplication `Σ_i scalars[i] · points[i]` over secp256k1 using
/// Shamir's trick *without* the curve endomorphism.
///
/// `n = points.len()` must equal `scalars.len()` and is statically known to the circuit. Each
/// scalar is a 256-bit value (`N_LIMBS` limbs). The `n` points are used directly as base points,
/// giving a `2^n`-entry subset-sum table and a main loop of 256 conditional additions.
///
/// Compared to [`msm`], this trades twice as many doublings/additions for a quadratically smaller
/// table. It is therefore the cheaper strategy once `n` is large enough that the endomorphism's
/// `2^(2n)` precomputation dominates.
///
/// The completeness-gap caveat of [`msm`] applies here too.
///
/// # Panics
///
/// Panics if `scalars.len() != points.len()`, if `n == 0`, or if any scalar does not have exactly
/// `N_LIMBS` limbs.
pub fn msm_naive(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	scalars: &[BigUint],
	points: &[Secp256k1Affine],
) -> Secp256k1Affine {
	let n = points.len();
	assert_eq!(scalars.len(), n, "scalars and points must have the same length");
	assert!(n >= 1, "MSM requires at least one point");
	for scalar in scalars {
		assert_eq!(scalar.limbs.len(), N_LIMBS);
	}

	let subscalar_refs = scalars
		.iter()
		.map(|s| s.limbs.as_slice())
		.collect::<Vec<_>>();
	// Full 256-bit scalars, so 256 conditional additions are required.
	shamir_accumulate(b, curve, points, &subscalar_refs, 256)
}

/// Shared core of [`msm`] and [`msm_naive`]: double-and-add over `n_iters` bits, selecting at each
/// step which subset sum of `base_points` to add via Shamir's trick.
///
/// Precomputes a `2^m`-entry table of all subset sums of the `m` base points
/// (`table[mask] = Σ_{i : bit i of mask set} base_points[i]`), then for each bit position from
/// `n_iters - 1` down to 0 doubles the accumulator and adds the subset sum selected by that bit of
/// each subscalar. `subscalars[i]` supplies base point `i`'s exponent bit (it must have enough
/// limbs to cover `n_iters` bits).
///
/// The per-iteration lookup uses [`multi_wire_multiplex`], indexing the flattened table by a
/// selector word whose bit `i` is base point `i`'s current exponent bit — matching the table
/// layout.
fn shamir_accumulate(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	base_points: &[Secp256k1Affine],
	subscalars: &[&[Wire]],
	n_iters: usize,
) -> Secp256k1Affine {
	let m = base_points.len();
	assert_eq!(subscalars.len(), m, "one subscalar per base point");
	// The m exponent bits per iteration are packed into a single-word multiplexer selector.
	assert!(
		m < WORD_SIZE_BITS,
		"Shamir's trick supports at most {} base points",
		WORD_SIZE_BITS - 1
	);

	// Precompute the 2^m table of all subset sums of the base points.
	let mut table = Vec::with_capacity(1 << m);
	table.push(Secp256k1Affine::point_at_infinity(b));
	for (i, pt) in base_points.iter().enumerate() {
		table.push(pt.clone());
		for j in 1..1 << i {
			table.push(curve.add_incomplete(b, &table[j], pt));
		}
	}

	// Flatten each table entry into its constituent wires so the lookup can use the multi-wire
	// multiplexer, which selects across `2^m` groups using `m` bits of the selector word.
	let table_flat: Vec<Vec<Wire>> = table.iter().map(point_to_wires).collect();
	let table_refs: Vec<&[Wire]> = table_flat.iter().map(Vec::as_slice).collect();

	let one = b.add_constant_64(1);
	let mut acc = Secp256k1Affine::point_at_infinity(b);

	for bit_index in (0..n_iters).rev() {
		let limb = bit_index / WORD_SIZE_BITS;
		let bit = bit_index % WORD_SIZE_BITS;

		if bit_index != n_iters - 1 {
			acc = curve.double(b, &acc);
		}

		// Build the multiplexer selector. `multi_wire_multiplex` uses selector bit `i` as bit `i`
		// of the table index, so we place base point `i`'s current exponent bit at position `i`,
		// matching the subset-sum table layout (`table[mask]` includes base `i` iff bit `i` set).
		let mut sel = b.add_constant(Word::ZERO);
		for (i, subscalar) in subscalars.iter().enumerate() {
			let bit_val = b.band(b.shr(subscalar[limb], bit as u32), one);
			sel = b.bor(sel, b.shl(bit_val, i as u32));
		}

		let selected = point_from_wires(&multi_wire_multiplex(b, &table_refs, sel));
		acc = curve.add_incomplete(b, &acc, &selected);
	}

	acc
}

// Flatten an affine point into its constituent wires: x limbs, then y limbs, then the
// point-at-infinity flag. Inverse of `point_from_wires`.
fn point_to_wires(p: &Secp256k1Affine) -> Vec<Wire> {
	assert_eq!(p.x.limbs.len(), N_LIMBS);
	assert_eq!(p.y.limbs.len(), N_LIMBS);

	let mut wires = Vec::with_capacity(2 * N_LIMBS + 1);
	wires.extend_from_slice(&p.x.limbs);
	wires.extend_from_slice(&p.y.limbs);
	wires.push(p.is_point_at_infinity);
	wires
}

// Reconstruct an affine point from the flat wire layout produced by `point_to_wires`.
fn point_from_wires(wires: &[Wire]) -> Secp256k1Affine {
	assert_eq!(wires.len(), 2 * N_LIMBS + 1);
	Secp256k1Affine {
		x: BigUint {
			limbs: wires[..N_LIMBS].to_vec(),
		},
		y: BigUint {
			limbs: wires[N_LIMBS..2 * N_LIMBS].to_vec(),
		},
		is_point_at_infinity: wires[2 * N_LIMBS],
	}
}

// Constrain the return value of `CircuitBuilder::secp256k1_endomorphism_split_hint`.
// Verifies that `k1 + λ k2 = k (mod n)` where `n` is scalar field modulus.
fn check_endomorphism_split(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	k1_neg: Wire,
	k2_neg: Wire,
	k1_abs: [Wire; 2],
	k2_abs: [Wire; 2],
	k: &BigUint,
) {
	assert_eq!(k.limbs.len(), N_LIMBS);

	let k1_abs = BigUint {
		limbs: k1_abs.to_vec(),
	}
	.zero_extend(b, N_LIMBS);
	let k2_abs = BigUint {
		limbs: k2_abs.to_vec(),
	}
	.zero_extend(b, N_LIMBS);

	let f_scalar = curve.f_scalar();
	let k1 = select_biguint(b, k1_neg, &f_scalar.sub(b, &coord_zero(b), &k1_abs), &k1_abs);
	let k2 = select_biguint(b, k2_neg, &f_scalar.sub(b, &coord_zero(b), &k2_abs), &k2_abs);

	assert_eq(
		b,
		"endomorphism split k1 + λk2 = k (mod n)",
		k,
		&f_scalar.add(b, &k1, &f_scalar.mul(b, &k2, &coord_lambda(b))),
	);
}

/// A common trick to save doublings when computing multiexponentiations of the form
/// `G*g_mult + PK*pk_mult` - instead of doing two scalar multiplications separately and
/// adding their results, we share the doubling step of double-and-add.
///
/// This implementation relies on group axioms only. It is currently unused for secp256k1
/// but may prove useful for other curves.
#[allow(unused)]
pub fn shamirs_trick_naive(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	bits: usize,
	g_mult: &BigUint,
	pk_mult: &BigUint,
	pk: Secp256k1Affine,
) -> Secp256k1Affine {
	let g = Secp256k1Affine::generator(b);
	let g_pk = curve.add(b, &g, &pk);

	let mut acc = Secp256k1Affine::point_at_infinity(b);

	for bit_index in (0..bits).rev() {
		let limb = bit_index / WORD_SIZE_BITS;
		let bit = bit_index % WORD_SIZE_BITS;

		if bit_index != bits - 1 {
			acc = curve.double(b, &acc);
		}

		let g_mult_bit = b.shl(g_mult.limbs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
		let pk_mult_bit = b.shl(pk_mult.limbs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);

		// A 3-to-1 mux
		let x =
			select_biguint(b, pk_mult_bit, &select_biguint(b, g_mult_bit, &g_pk.x, &pk.x), &g.x);
		let y =
			select_biguint(b, pk_mult_bit, &select_biguint(b, g_mult_bit, &g_pk.y, &pk.y), &g.y);

		// Point at infinity flag is a single wire, allowing us to save a BigUint select.
		let is_point_at_infinity = b.band(b.bnot(g_mult_bit), b.bnot(pk_mult_bit));

		// Addition implementation is incomplete (it handles pai, but not doubling). When
		// the mask is zero, pai-to-pai support is needed. The probability of accumulator
		// assuming value G, PK, or G+PK at any point in the computation is vanishingly low.
		// We assert false in this case, resulting in a completeness gap.
		acc = curve.add_incomplete(
			b,
			&acc,
			&Secp256k1Affine {
				x,
				y,
				is_point_at_infinity,
			},
		);
	}

	acc
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;
	use binius_frontend::CircuitBuilder;
	use k256::{
		ProjectivePoint, Scalar, U256,
		elliptic_curve::{ops::MulByGenerator, scalar::FromUintUnchecked, sec1::ToEncodedPoint},
	};
	use rand::prelude::*;

	use super::*;
	use crate::{
		bignum::{BigUint, assert_eq},
		secp256k1::{Secp256k1, Secp256k1Affine},
	};

	#[test]
	fn test_scalar_mul_naive() {
		let builder = CircuitBuilder::new();
		let curve = Secp256k1::new(&builder);

		// Test with scalar = 69
		let scalar_value = 69u64;

		// Use k256 to compute the expected result
		let k256_scalar = Scalar::from(scalar_value);
		let k256_point = ProjectivePoint::mul_by_generator(&k256_scalar).to_affine();

		// Extract coordinates from k256 result
		let point_bytes = k256_point.to_encoded_point(false).to_bytes();
		// The uncompressed format is: 0x04 || x || y (65 bytes total)
		// We need to extract x and y coordinates (32 bytes each)
		let x_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[1..33]);
		let y_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[33..65]);

		// Create our scalar as BigUint
		let scalar = BigUint::new_constant(&builder, &num_bigint::BigUint::from(scalar_value));

		// Create expected coordinates as BigUint
		let expected_x = BigUint::new_constant(&builder, &x_coord);
		let expected_y = BigUint::new_constant(&builder, &y_coord);

		// Get the generator point
		let generator = Secp256k1Affine::generator(&builder);

		// Perform scalar multiplication with our implementation
		let result = scalar_mul_naive(&builder, &curve, 7, &scalar, generator);

		// Check that the result matches the expected point
		assert_eq(&builder, "result_x", &result.x, &expected_x);
		assert_eq(&builder, "result_y", &result.y, &expected_y);

		// Build and verify the circuit
		let cs = builder.build();
		let mut w = cs.new_witness_filler();
		assert!(cs.populate_wire_witness(&mut w).is_ok());

		// Also verify the point is not at infinity
		assert_eq!(w[result.is_point_at_infinity], Word::ZERO);
	}

	#[test]
	fn test_scalar_mul_with_endomorphism() {
		let builder = CircuitBuilder::new();
		let curve = Secp256k1::new(&builder);

		// Generate a random 256-bit scalar
		let mut rng = StdRng::seed_from_u64(0);
		let mut scalar_bytes = [0u8; 32];
		rng.fill(&mut scalar_bytes);

		// Create the scalar in both k256 and our format
		let k256_uint = U256::from_be_slice(&scalar_bytes);
		let k256_scalar = Scalar::from_uint_unchecked(k256_uint);
		let scalar_bigint = num_bigint::BigUint::from_bytes_be(&scalar_bytes);
		let scalar = BigUint::new_constant(&builder, &scalar_bigint).zero_extend(&builder, N_LIMBS);

		// Use k256 to compute the expected result with the generator point
		let k256_point = ProjectivePoint::mul_by_generator(&k256_scalar).to_affine();

		// Extract coordinates from k256 result
		let point_bytes = k256_point.to_encoded_point(false).to_bytes();
		let x_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[1..33]);
		let y_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[33..65]);

		// Create expected coordinates as BigUint
		let expected_x = BigUint::new_constant(&builder, &x_coord);
		let expected_y = BigUint::new_constant(&builder, &y_coord);

		// Get the generator point
		let generator = Secp256k1Affine::generator(&builder);

		// Perform scalar multiplication with our endomorphism implementation
		let result = scalar_mul(&builder, &curve, &scalar, generator);

		// Check that the result matches the expected point
		assert_eq(&builder, "result_x", &result.x, &expected_x);
		assert_eq(&builder, "result_y", &result.y, &expected_y);

		// Build and verify the circuit
		let cs = builder.build();
		let mut w = cs.new_witness_filler();
		assert!(cs.populate_wire_witness(&mut w).is_ok());

		// Verify the point is not at infinity
		assert_eq!(w[result.is_point_at_infinity], Word::ZERO);
	}

	type MsmFn = fn(&CircuitBuilder, &Secp256k1, &[BigUint], &[Secp256k1Affine]) -> Secp256k1Affine;

	// Build an MSM circuit (using `msm_fn`) over `n` random (scalar, point) pairs derived from
	// `seed`, compare the result against a k256-computed reference, and verify the witness
	// populates successfully.
	fn check_msm(msm_fn: MsmFn, n: usize, seed: u64) {
		let builder = CircuitBuilder::new();
		let curve = Secp256k1::new(&builder);
		let mut rng = StdRng::seed_from_u64(seed);

		let mut scalars = Vec::with_capacity(n);
		let mut points = Vec::with_capacity(n);
		let mut expected = ProjectivePoint::IDENTITY;

		for _ in 0..n {
			// Random 256-bit multiplier scalar.
			let mut scalar_bytes = [0u8; 32];
			rng.fill(&mut scalar_bytes);
			let k256_scalar = Scalar::from_uint_unchecked(U256::from_be_slice(&scalar_bytes));

			// Random curve point, generated as `g^r` so it is guaranteed on-curve.
			let mut point_seed = [0u8; 32];
			rng.fill(&mut point_seed);
			let r = Scalar::from_uint_unchecked(U256::from_be_slice(&point_seed));
			let point = ProjectivePoint::mul_by_generator(&r);

			expected += point * k256_scalar;

			let scalar_bigint = num_bigint::BigUint::from_bytes_be(&scalar_bytes);
			scalars.push(
				BigUint::new_constant(&builder, &scalar_bigint).zero_extend(&builder, N_LIMBS),
			);

			let point_bytes = point.to_affine().to_encoded_point(false).to_bytes();
			let x_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[1..33]);
			let y_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[33..65]);
			points.push(Secp256k1Affine {
				x: BigUint::new_constant(&builder, &x_coord).zero_extend(&builder, N_LIMBS),
				y: BigUint::new_constant(&builder, &y_coord).zero_extend(&builder, N_LIMBS),
				is_point_at_infinity: builder.add_constant(Word::ZERO),
			});
		}

		let result = msm_fn(&builder, &curve, &scalars, &points);

		let expected_bytes = expected.to_affine().to_encoded_point(false).to_bytes();
		let expected_x = BigUint::new_constant(
			&builder,
			&num_bigint::BigUint::from_bytes_be(&expected_bytes[1..33]),
		);
		let expected_y = BigUint::new_constant(
			&builder,
			&num_bigint::BigUint::from_bytes_be(&expected_bytes[33..65]),
		);

		assert_eq(&builder, "msm_x", &result.x, &expected_x);
		assert_eq(&builder, "msm_y", &result.y, &expected_y);

		let cs = builder.build();
		let mut w = cs.new_witness_filler();
		assert!(cs.populate_wire_witness(&mut w).is_ok());
		assert_eq!(w[result.is_point_at_infinity], Word::ZERO);
	}

	#[test]
	fn test_msm_single_point() {
		check_msm(msm, 1, 0);
	}

	#[test]
	fn test_msm_two_points() {
		check_msm(msm, 2, 1);
	}

	#[test]
	fn test_msm_three_points() {
		check_msm(msm, 3, 2);
	}

	#[test]
	fn test_msm_naive_single_point() {
		check_msm(msm_naive, 1, 0);
	}

	#[test]
	fn test_msm_naive_two_points() {
		check_msm(msm_naive, 2, 1);
	}

	#[test]
	fn test_msm_naive_three_points() {
		check_msm(msm_naive, 3, 2);
	}
}
