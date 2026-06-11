// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
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

/// Compute a multi-scalar multiplication `Σ_i scalars[i] · points[i]` over secp256k1 using the
/// fixed-window (Straus) algorithm.
///
/// `n = points.len()` must equal `scalars.len()`; both `n` and the window size `window` (in bits)
/// are statically known to the circuit. Each scalar is a 256-bit value (`N_LIMBS` limbs).
///
/// For each point `P_i` a table of its `2^window` small multiples `{0·P_i, …, (2^window − 1)·P_i}`
/// is precomputed. The exponents are then consumed `window` bits at a time from the most
/// significant window down: at each of the `ceil(256 / window)` steps every point contributes the
/// multiple selected by its current window (via [`multi_wire_multiplex`]), and the accumulator is
/// doubled `window` times *between* steps (skipped on the first window). Unlike Shamir's trick, the
/// total table size is `n · 2^window` — linear in `n` — so it scales to larger `n`.
///
/// `window = 4` is typically optimal. A window of 1 degenerates to a per-point double-and-add
/// sharing the doublings. See [`msm_strauss_endo`] for the GLV-endomorphism variant.
///
/// # Completeness gap
///
/// Point additions use [`Secp256k1::add_incomplete`], which asserts false when its inputs are equal
/// (it handles the point at infinity but not doubling). As with the other routines here, the
/// probability of the accumulator or a table entry hitting such a collision for independent inputs
/// is vanishingly low.
///
/// # Panics
///
/// Panics if `scalars.len() != points.len()`, if `n == 0`, if `window` is not in
/// `1..WORD_SIZE_BITS`, or if any scalar does not have exactly `N_LIMBS` limbs.
pub fn msm_strauss(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	window: usize,
	scalars: &[BigUint],
	points: &[Secp256k1Affine],
) -> Secp256k1Affine {
	let n = points.len();
	assert_eq!(scalars.len(), n, "scalars and points must have the same length");
	assert!(n >= 1, "MSM requires at least one point");
	assert!(0 < window && window < WORD_SIZE_BITS, "window must be in 1..WORD_SIZE_BITS");
	for scalar in scalars {
		assert_eq!(scalar.limbs.len(), N_LIMBS);
	}

	// Use each point directly as a base point and its full 256-bit scalar as the exponent.
	let tables = points
		.iter()
		.map(|point| build_strauss_table(b, curve, point, window))
		.collect::<Vec<_>>();
	let subscalars = scalars
		.iter()
		.map(|s| s.limbs.as_slice())
		.collect::<Vec<_>>();
	strauss_accumulate(b, curve, &tables, &subscalars, window, N_LIMBS * WORD_SIZE_BITS)
}

/// Compute a multi-scalar multiplication `Σ_i scalars[i] · points[i]` over secp256k1 using the
/// fixed-window (Straus) algorithm combined with the GLV endomorphism.
///
/// Every `(scalar, point)` pair is endomorphism-split into two ~128-bit signed subscalars and two
/// base points (`P` and `endo(P)`, conditionally negated to positive subscalars), so the `n`-point
/// 256-bit MSM becomes a `2n`-point 128-bit one. Relative to [`msm_strauss`] this halves the number
/// of doublings (128 vs 256) and the windows per point, while doubling the number of small-multiple
/// tables and adding one endomorphism-split constraint per scalar.
///
/// The `φ(P)` table is *not* recomputed with point additions: since `φ` is a homomorphism,
/// `φ(P)`'s small multiples are `φ` applied to `P`'s small multiples (`x · φ(P) = φ(x · P)`), i.e.
/// one field multiplication of each entry's x-coordinate by `β` rather than another
/// `2^window`-entry point-addition chain.
///
/// The window-lookup count and addition count are unchanged versus plain [`msm_strauss`]; the win
/// is the halved doubling chain. `window = 4` is typically optimal. The completeness-gap caveat of
/// [`msm_strauss`] applies here too.
///
/// # Panics
///
/// Panics if `scalars.len() != points.len()`, if `n == 0`, if `window` is not in
/// `1..WORD_SIZE_BITS`, or if any scalar does not have exactly `N_LIMBS` limbs.
pub fn msm_strauss_endo(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	window: usize,
	scalars: &[BigUint],
	points: &[Secp256k1Affine],
) -> Secp256k1Affine {
	let n = points.len();
	assert_eq!(scalars.len(), n, "scalars and points must have the same length");
	assert!(n >= 1, "MSM requires at least one point");
	assert!(0 < window && window < WORD_SIZE_BITS, "window must be in 1..WORD_SIZE_BITS");

	// Split each scalar via the endomorphism into two base points (`±P`, `±φ(P)`) with 128-bit
	// subscalars. Build `±P`'s table with point additions, then derive `±φ(P)`'s table by applying
	// `φ` entrywise rather than recomputing it.
	let mut tables = Vec::with_capacity(2 * n);
	let mut subscalars = Vec::with_capacity(2 * n);
	for (scalar, point) in scalars.iter().zip(points) {
		assert_eq!(scalar.limbs.len(), N_LIMBS);

		let (k1_neg, k2_neg, k1_abs, k2_abs) = b.secp256k1_endomorphism_split_hint(&scalar.limbs);
		check_endomorphism_split(b, curve, k1_neg, k2_neg, k1_abs, k2_abs, scalar);

		// Table for the (possibly negated) base point `±P`, built with point additions.
		let base = curve.negate_if(b, k1_neg, point);
		let table_p = build_strauss_table(b, curve, &base, window);

		// Table for `±φ(P)`: `φ(x · P) = x · φ(P)`, so apply `φ` to each entry. `φ` commutes with
		// negation, so the only correction is the relative sign of the two subscalars.
		let rel_neg = b.bxor(k1_neg, k2_neg);
		let table_phi = table_p
			.iter()
			.map(|q| {
				let phi = curve.endomorphism(b, q);
				curve.negate_if(b, rel_neg, &phi)
			})
			.collect::<Vec<_>>();

		tables.push(table_p);
		subscalars.push(k1_abs);
		tables.push(table_phi);
		subscalars.push(k2_abs);
	}

	let subscalar_refs = subscalars
		.iter()
		.map(<[Wire; 2]>::as_slice)
		.collect::<Vec<_>>();
	// Each endomorphism subscalar magnitude fits in 128 bits.
	strauss_accumulate(b, curve, &tables, &subscalar_refs, window, 128)
}

/// Build the table of small multiples `{0·P, 1·P, …, (2^window − 1)·P}` of a base point.
///
/// `table[2]` is a doubling (`add_incomplete` rejects equal inputs); every larger multiple is the
/// previous one plus `P`.
fn build_strauss_table(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	point: &Secp256k1Affine,
	window: usize,
) -> Vec<Secp256k1Affine> {
	let mut table = Vec::with_capacity(1 << window);
	table.push(Secp256k1Affine::point_at_infinity(b));
	table.push(point.clone());
	for x in 2..1 << window {
		let multiple = if x == 2 {
			curve.double(b, point)
		} else {
			curve.add_incomplete(b, &table[x - 1], point)
		};
		table.push(multiple);
	}
	table
}

/// Shared core of [`msm_strauss`] and [`msm_strauss_endo`]: fixed-window double-and-add over the
/// `m = tables.len()` base points whose `2^window`-entry small-multiple `tables` are precomputed
/// and whose exponents (`subscalars[i]`, bounded by `exponent_bits`) are consumed `window` bits at
/// a time.
///
/// At each of the `ceil(exponent_bits / window)` steps every base point contributes the multiple
/// selected by its current window (via [`multi_wire_multiplex`]), and the accumulator is doubled
/// `window` times between steps (skipped on the first, most-significant window).
fn strauss_accumulate(
	b: &CircuitBuilder,
	curve: &Secp256k1,
	tables: &[Vec<Secp256k1Affine>],
	subscalars: &[&[Wire]],
	window: usize,
	exponent_bits: usize,
) -> Secp256k1Affine {
	assert_eq!(subscalars.len(), tables.len(), "one subscalar per base point");

	// Flatten each table entry into its constituent wires for the multi-wire multiplexer.
	let tables_flat: Vec<Vec<Vec<Wire>>> = tables
		.iter()
		.map(|table| table.iter().map(point_to_wires).collect())
		.collect();
	let table_refs: Vec<Vec<&[Wire]>> = tables_flat
		.iter()
		.map(|table| table.iter().map(Vec::as_slice).collect())
		.collect();

	let one = b.add_constant_64(1);
	let n_windows = exponent_bits.div_ceil(window);
	let mut acc = Secp256k1Affine::point_at_infinity(b);

	for w_idx in (0..n_windows).rev() {
		// Double `window` times between windows (skipped on the first, most-significant window so
		// the result is not over-multiplied by `2^window`).
		if w_idx != n_windows - 1 {
			for _ in 0..window {
				acc = curve.double(b, &acc);
			}
		}

		let base_bit = w_idx * window;
		for (point_idx, subscalar) in subscalars.iter().enumerate() {
			// Pack this base point's `window`-bit exponent chunk into a selector word, bit by bit
			// so that windows crossing 64-bit limb boundaries (and bit positions past
			// `exponent_bits`) are handled uniformly. `multi_wire_multiplex` reads selector bit
			// `j` as bit `j` of the table index, matching `table[x] = x · P`.
			let mut sel = b.add_constant(Word::ZERO);
			for j in 0..window {
				let bit_index = base_bit + j;
				if bit_index >= exponent_bits {
					continue; // past the top of the exponent — contributes a zero bit
				}
				let limb = bit_index / WORD_SIZE_BITS;
				let bit = bit_index % WORD_SIZE_BITS;
				let bit_val = b.band(b.shr(subscalar[limb], bit as u32), one);
				sel = b.bor(sel, b.shl(bit_val, j as u32));
			}

			let selected = point_from_wires(&multi_wire_multiplex(b, &table_refs[point_idx], sel));
			acc = curve.add_incomplete(b, &acc, &selected);
		}
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

	type StraussFn =
		fn(&CircuitBuilder, &Secp256k1, usize, &[BigUint], &[Secp256k1Affine]) -> Secp256k1Affine;

	// Build a Straus MSM circuit (using `msm_fn` with the given `window`) over `n` random
	// (scalar, point) pairs derived from `seed`, compare against a k256 reference, and verify the
	// witness populates.
	fn check_msm_strauss(msm_fn: StraussFn, window: usize, n: usize, seed: u64) {
		let builder = CircuitBuilder::new();
		let curve = Secp256k1::new(&builder);
		let mut rng = StdRng::seed_from_u64(seed);

		let mut scalars = Vec::with_capacity(n);
		let mut points = Vec::with_capacity(n);
		let mut expected = ProjectivePoint::IDENTITY;

		for _ in 0..n {
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

		let result = msm_fn(&builder, &curve, window, &scalars, &points);

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

	// Window of 2 divides both 64 and 256, so windows never cross a limb boundary.
	#[test]
	fn test_msm_strauss_window2() {
		check_msm_strauss(msm_strauss, 2, 1, 0);
		check_msm_strauss(msm_strauss, 2, 2, 1);
		check_msm_strauss(msm_strauss, 2, 3, 2);
	}

	// Window of 3 does not divide 64, so windows cross limb boundaries and the top window reads
	// past bit 255 — exercising both the cross-limb extraction and the out-of-range bit guard.
	#[test]
	fn test_msm_strauss_window3() {
		check_msm_strauss(msm_strauss, 3, 1, 3);
		check_msm_strauss(msm_strauss, 3, 2, 4);
		check_msm_strauss(msm_strauss, 3, 3, 5);
	}

	// A window of 1 degenerates to plain double-and-add with shared doublings.
	#[test]
	fn test_msm_strauss_window1() {
		check_msm_strauss(msm_strauss, 1, 2, 6);
	}

	// GLV variant: the 128-bit subscalars mean window 3 (43 windows → 129 bits) still exercises the
	// cross-limb extraction and out-of-range guard, now at the 128-bit boundary.
	#[test]
	fn test_msm_strauss_endo_window2() {
		check_msm_strauss(msm_strauss_endo, 2, 1, 7);
		check_msm_strauss(msm_strauss_endo, 2, 2, 8);
		check_msm_strauss(msm_strauss_endo, 2, 3, 9);
	}

	#[test]
	fn test_msm_strauss_endo_window3() {
		check_msm_strauss(msm_strauss_endo, 3, 1, 10);
		check_msm_strauss(msm_strauss_endo, 3, 2, 11);
		check_msm_strauss(msm_strauss_endo, 3, 3, 12);
	}
}
