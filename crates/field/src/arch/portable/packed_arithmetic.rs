// Copyright 2024-2025 Irreducible Inc.

use crate::{
	PackedExtension, PackedField, TowerField,
	arch::PackedStrategy,
	arithmetic_traits::{
		MulAlpha, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedPackedTransformationFactory,
		TaggedSquare,
	},
	binary_field::{BinaryField, TowerExtensionField},
	linear_transformation::{FieldLinearTransformation, Transformation},
	packed::PackedBinaryField,
	underlier::{UnderlierType, UnderlierWithBitOps, WithUnderlier},
};

pub trait UnderlierWithBitConstants: UnderlierWithBitOps
where
	Self: 'static,
{
	const INTERLEAVE_EVEN_MASK: &'static [Self];
	const INTERLEAVE_ODD_MASK: &'static [Self];

	/// Interleave with the given bit size
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		// There are 2^7 = 128 bits in a u128
		assert!(log_block_len < Self::INTERLEAVE_EVEN_MASK.len());

		let block_len = 1 << log_block_len;

		// See Hacker's Delight, Section 7-3.
		// https://dl.acm.org/doi/10.5555/2462741
		let t = ((self >> block_len) ^ other) & Self::INTERLEAVE_EVEN_MASK[log_block_len];
		let c = self ^ t << block_len;
		let d = other ^ t;

		(c, d)
	}

	/// Transpose with the given bit size
	fn transpose(mut self, mut other: Self, log_block_len: usize) -> (Self, Self) {
		// There are 2^7 = 128 bits in a u128
		assert!(log_block_len < Self::INTERLEAVE_EVEN_MASK.len());

		for log_block_len in (log_block_len..Self::LOG_BITS).rev() {
			(self, other) = self.interleave(other, log_block_len);
		}

		(self, other)
	}
}

/// Abstraction for a packed tower field of height greater than 0.
///
/// Helper trait
pub(crate) trait PackedTowerField: PackedField + WithUnderlier {
	/// A scalar of a lower height
	type DirectSubfield: TowerField;
	/// Packed type with the same underlier with a lower height
	type PackedDirectSubfield: PackedField<Scalar = Self::DirectSubfield>
		+ WithUnderlier<Underlier = Self::Underlier>;

	/// Reinterpret value as a packed field over a lower height
	fn as_packed_subfield(self) -> Self::PackedDirectSubfield;

	/// Reinterpret packed subfield as a current type
	fn from_packed_subfield(value: Self::PackedDirectSubfield) -> Self;
}

impl<F, P, U: UnderlierType> PackedTowerField for P
where
	F: TowerExtensionField,
	P: PackedField<Scalar = F> + PackedExtension<F::DirectSubfield> + WithUnderlier<Underlier = U>,
	P::PackedSubfield: WithUnderlier<Underlier = U>,
{
	type DirectSubfield = F::DirectSubfield;
	type PackedDirectSubfield = <Self as PackedExtension<F::DirectSubfield>>::PackedSubfield;

	#[inline]
	fn as_packed_subfield(self) -> Self::PackedDirectSubfield {
		P::cast_base(self)
	}

	#[inline]
	fn from_packed_subfield(value: Self::PackedDirectSubfield) -> Self {
		P::cast_ext(value)
	}
}

/// Compile-time known constants needed for packed multiply implementation.
pub(crate) trait TowerConstants<U> {
	/// Alpha values in odd positions, zeroes in even.
	const ALPHAS_ODD: U;
}

macro_rules! impl_tower_constants {
	($tower_field:path, $underlier:ty, $value:tt) => {
		impl $crate::arch::portable::packed_arithmetic::TowerConstants<$underlier>
			for $tower_field
		{
			const ALPHAS_ODD: $underlier = $value;
		}
	};
}

pub(crate) use impl_tower_constants;

impl<PT> TaggedMul<PackedStrategy> for PT
where
	PT: PackedTowerField,
	PT::Underlier: UnderlierWithBitConstants,
	PT::DirectSubfield: TowerConstants<PT::Underlier>,
{
	/// Optimized packed field multiplication algorithm
	#[inline]
	fn mul(self, b: Self) -> Self {
		let a = self;

		// a and b can be interpreted as packed subfield elements:
		// a = <a_lo_0, a_hi_0, a_lo_1, a_hi_1, ...>
		// b = <b_lo_0, b_hi_0, b_lo_1, b_hi_1, ...>//
		// ab is the product of a * b as packed subfield elements
		// ab = <a_lo_0 * b_lo_0, a_hi_0 * b_hi_0, a_lo_1 * b_lo_1, a_hi_1 * b_hi_1, ...>
		let repacked_a = a.as_packed_subfield();
		let repacked_b = b.as_packed_subfield();
		let z0_even_z2_odd = repacked_a * repacked_b;

		// lo = <a_lo_0, b_lo_0, a_lo_1, b_lo_1, ...>
		// hi = <a_hi_0, b_hi_0, a_hi_1, b_hi_1, ...>
		let (lo, hi) =
			interleave::<PT::Underlier, PT::DirectSubfield>(a.to_underlier(), b.to_underlier());

		// <a_lo_0 + a_hi_0, b_lo_0 + b_hi_0, a_lo_1 + a_hi_1, b_lo_1 + b_hi_1, ...>
		let lo_plus_hi_a_even_b_odd = lo ^ hi;

		let odd_mask = <PT::Underlier as UnderlierWithBitConstants>::INTERLEAVE_ODD_MASK
			[PT::DirectSubfield::TOWER_LEVEL];

		let alphas = PT::DirectSubfield::ALPHAS_ODD;

		// <α, z2_0, α, z2_1, ...>
		let alpha_even_z2_odd = alphas ^ (z0_even_z2_odd.to_underlier() & odd_mask);

		// a_lo_plus_hi_even_z2_odd    = <a_lo_0 + a_hi_0, z2_0, a_lo_1 + a_hi_1, z2_1, ...>
		// b_lo_plus_hi_even_alpha_odd = <b_lo_0 + b_hi_0,    α, a_lo_1 + a_hi_1,   αz, ...>
		let (a_lo_plus_hi_even_alpha_odd, b_lo_plus_hi_even_z2_odd) =
			interleave::<PT::Underlier, PT::DirectSubfield>(
				lo_plus_hi_a_even_b_odd,
				alpha_even_z2_odd,
			);

		// <z1_0 + z0_0 + z2_0, z2a_0, z1_1 + z0_1 + z2_1, z2a_1, ...>
		let z1_plus_z0_plus_z2_even_z2a_odd =
			PT::PackedDirectSubfield::from_underlier(a_lo_plus_hi_even_alpha_odd)
				* PT::PackedDirectSubfield::from_underlier(b_lo_plus_hi_even_z2_odd);

		// <0, z1_0 + z2a_0 + z0_0 + z2_0, 0, z1_1 + z2a_1 + z0_1 + z2_1, ...>
		let zero_even_z1_plus_z2a_plus_z0_plus_z2_odd = (z1_plus_z0_plus_z2_even_z2a_odd
			.to_underlier()
			^ (z1_plus_z0_plus_z2_even_z2a_odd.to_underlier() << PT::DirectSubfield::N_BITS))
			& odd_mask;

		// <z0_0 + z2_0, z0_0 + z2_0, z0_1 + z2_1, z0_1 + z2_1, ...>
		let z0_plus_z2_dup =
			xor_adjacent::<PT::Underlier, PT::DirectSubfield>(z0_even_z2_odd.to_underlier());

		// <z0_0 + z2_0, z1_0 + z2a_0, z0_1 + z2_1, z1_1 + z2a_1, ...>
		Self::from_underlier(z0_plus_z2_dup ^ zero_even_z1_plus_z2a_plus_z0_plus_z2_odd)
	}
}

/// Generate the mask with alphas in the odd packed element positions and zeros in even
macro_rules! alphas {
	($underlier:ty, $tower_level:literal) => {{
		let mut alphas: $underlier = if $tower_level == 0 {
			1
		} else {
			1 << (1 << ($tower_level - 1))
		};

		let log_width = <$underlier as $crate::underlier::UnderlierType>::LOG_BITS - $tower_level;
		let mut i = 1;
		while i < log_width {
			alphas |= alphas << (1 << ($tower_level + i));
			i += 1;
		}

		alphas
	}};
}

pub(crate) use alphas;

/// Generate the mask with ones in the odd packed element positions and zeros in even
macro_rules! interleave_mask_even {
	($underlier:ty, $tower_level:literal) => {{
		let scalar_bits = 1 << $tower_level;

		let mut mask: $underlier = (1 << scalar_bits) - 1;
		let log_width = <$underlier>::LOG_BITS - $tower_level;
		let mut i = 1;
		while i < log_width {
			mask |= mask << (scalar_bits << i);
			i += 1;
		}

		mask
	}};
}

pub(crate) use interleave_mask_even;

/// Generate the mask with ones in the even packed element positions and zeros in odd
macro_rules! interleave_mask_odd {
	($underlier:ty, $tower_level:literal) => {
		interleave_mask_even!($underlier, $tower_level) << (1 << $tower_level)
	};
}

pub(crate) use interleave_mask_odd;

/// View the inputs as vectors of packed binary tower elements and transpose as 2x2 square matrices.
/// Given vectors <a_0, a_1, a_2, a_3, ...> and <b_0, b_1, b_2, b_3, ...>, returns a tuple with
/// <a0, b0, a2, b2, ...> and <a1, b1, a3, b3>.
fn interleave<U: UnderlierWithBitConstants, F: TowerField>(a: U, b: U) -> (U, U) {
	let mask = U::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];

	let block_len = F::N_BITS;
	let t = ((a >> block_len) ^ b) & mask;
	let c = a ^ (t << block_len);
	let d = b ^ t;

	(c, d)
}

/// View the input as a vector of packed binary tower elements and add the adjacent ones.
/// Given a vector <a_0, a_1, a_2, a_3, ...>, returns <a0 + a1, a0 + a1, a2 + a3, a2 + a3, ...>.
fn xor_adjacent<U: UnderlierWithBitConstants, F: TowerField>(a: U) -> U {
	let mask = U::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];

	let block_len = F::N_BITS;
	let t = ((a >> block_len) ^ a) & mask;

	t ^ (t << block_len)
}

impl<PT> TaggedMulAlpha<PackedStrategy> for PT
where
	PT: PackedTowerField,
	PT::PackedDirectSubfield: MulAlpha,
	PT::Underlier: UnderlierWithBitConstants,
{
	#[inline]
	fn mul_alpha(self) -> Self {
		let block_len = PT::DirectSubfield::N_BITS;
		let even_mask = <PT::Underlier as UnderlierWithBitConstants>::INTERLEAVE_EVEN_MASK
			[PT::DirectSubfield::TOWER_LEVEL];
		let odd_mask = <PT::Underlier as UnderlierWithBitConstants>::INTERLEAVE_ODD_MASK
			[PT::DirectSubfield::TOWER_LEVEL];

		let a = self.to_underlier();
		let a0 = a & even_mask;
		let a1 = a & odd_mask;
		let z1 = PT::PackedDirectSubfield::from_underlier(a1)
			.mul_alpha()
			.to_underlier();

		Self::from_underlier((a1 >> block_len) | ((a0 << block_len) ^ z1))
	}
}

impl<PT> TaggedSquare<PackedStrategy> for PT
where
	PT: PackedTowerField,
	PT::PackedDirectSubfield: MulAlpha,
	PT::Underlier: UnderlierWithBitConstants,
{
	#[inline]
	fn square(self) -> Self {
		let block_len = PT::DirectSubfield::N_BITS;
		let even_mask = <PT::Underlier as UnderlierWithBitConstants>::INTERLEAVE_EVEN_MASK
			[PT::DirectSubfield::TOWER_LEVEL];
		let odd_mask = <PT::Underlier as UnderlierWithBitConstants>::INTERLEAVE_ODD_MASK
			[PT::DirectSubfield::TOWER_LEVEL];

		let z_02 = PackedField::square(self.as_packed_subfield());
		let z_2a = z_02.mul_alpha().to_underlier() & odd_mask;

		let z_0_xor_z_2 = (z_02.to_underlier() ^ (z_02.to_underlier() >> block_len)) & even_mask;

		Self::from_underlier(z_0_xor_z_2 | z_2a)
	}
}

impl<PT> TaggedInvertOrZero<PackedStrategy> for PT
where
	PT: PackedTowerField,
	PT::PackedDirectSubfield: MulAlpha,
	PT::Underlier: UnderlierWithBitConstants,
{
	#[inline]
	fn invert_or_zero(self) -> Self {
		let block_len = PT::DirectSubfield::N_BITS;
		let even_mask = <PT::Underlier as UnderlierWithBitConstants>::INTERLEAVE_EVEN_MASK
			[PT::DirectSubfield::TOWER_LEVEL];
		let odd_mask = <PT::Underlier as UnderlierWithBitConstants>::INTERLEAVE_ODD_MASK
			[PT::DirectSubfield::TOWER_LEVEL];

		// has meaningful values in even positions
		let a_1_even = PT::PackedDirectSubfield::from_underlier(self.to_underlier() >> block_len);
		let intermediate = self.as_packed_subfield() + a_1_even.mul_alpha();
		let delta = self.as_packed_subfield() * intermediate
			+ <PT::PackedDirectSubfield as PackedField>::square(a_1_even);
		let delta_inv = PackedField::invert_or_zero(delta);

		// set values from even positions to odd as well
		let mut delta_inv_delta_inv = delta_inv.to_underlier() & even_mask;
		delta_inv_delta_inv |= delta_inv_delta_inv << block_len;

		let intermediate_a1 =
			(self.to_underlier() & odd_mask) | (intermediate.to_underlier() & even_mask);
		let result = PT::PackedDirectSubfield::from_underlier(delta_inv_delta_inv)
			* PT::PackedDirectSubfield::from_underlier(intermediate_a1);
		Self::from_underlier(result.to_underlier())
	}
}

/// Packed transformation implementation.
/// Stores bases in a form of:
/// [
///     [<base vec 1> ... <base vec 1>],
///     ...
///     [<base vec N> ... <base vec N>]
/// ]
/// Transformation complexity is `N*log(N)` where `N` is `OP::Scalar::DEGREE`.
pub struct PackedTransformation<OP> {
	bases: Vec<OP>,
}

impl<OP> PackedTransformation<OP>
where
	OP: PackedBinaryField,
{
	pub fn new<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: &FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self {
		Self {
			bases: transformation
				.bases()
				.iter()
				.map(|base| OP::broadcast(*base))
				.collect(),
		}
	}
}

/// Broadcast lowest field for each element, e.g. `[<0001><0000>] -> [<1111><0000>]`
fn broadcast_lowest_bit<U: UnderlierWithBitOps>(mut data: U, log_packed_bits: usize) -> U {
	for i in 0..log_packed_bits {
		data |= data << (1 << i)
	}

	data
}

impl<U, IP, OP, IF, OF> Transformation<IP, OP> for PackedTransformation<OP>
where
	IP: PackedField<Scalar = IF> + WithUnderlier<Underlier = U>,
	OP: PackedField<Scalar = OF> + WithUnderlier<Underlier = U>,
	IF: BinaryField,
	OF: BinaryField,
	U: UnderlierWithBitOps,
{
	fn transform(&self, input: &IP) -> OP {
		let mut result = OP::zero();
		let ones = OP::one().to_underlier();
		let mut input = input.to_underlier();

		for base in &self.bases {
			let base_component = input & ones;
			// contains ones at positions which correspond to non-zero components
			let mask = broadcast_lowest_bit(base_component, OF::LOG_DEGREE);
			result += OP::from_underlier(mask & base.to_underlier());
			input = input >> 1;
		}

		result
	}
}

impl<IP, OP> TaggedPackedTransformationFactory<PackedStrategy, OP> for IP
where
	IP: PackedBinaryField + WithUnderlier<Underlier: UnderlierWithBitOps>,
	OP: PackedBinaryField + WithUnderlier<Underlier = IP::Underlier>,
{
	type PackedTransformation<Data: AsRef<[OP::Scalar]> + Sync> = PackedTransformation<OP>;

	fn make_packed_transformation<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self::PackedTransformation<Data> {
		PackedTransformation::new(&transformation)
	}
}

#[cfg(test)]
mod tests {
	use std::fmt::Debug;

	use rand::prelude::*;

	use super::*;
	use crate::{
		BinaryField1b,
		test_utils::{define_invert_tests, define_multiply_tests, define_square_tests},
	};

	const NUM_TESTS: u64 = 100;

	fn check_interleave<F: TowerField>(a: u128, b: u128, c: u128, d: u128) {
		assert_eq!(interleave::<u128, F>(a, b), (c, d));
		assert_eq!(interleave::<u128, F>(c, d), (a, b));
	}

	#[test]
	fn test_interleave() {
		check_interleave::<BinaryField1b>(
			0x0000000000000000FFFFFFFFFFFFFFFF,
			0xFFFFFFFFFFFFFFFF0000000000000000,
			0xAAAAAAAAAAAAAAAA5555555555555555,
			0xAAAAAAAAAAAAAAAA5555555555555555,
		);
	}

	fn test_packed_multiply_alpha<P>()
	where
		P: PackedField + MulAlpha + Debug,
		P::Scalar: MulAlpha,
	{
		let mut rng = StdRng::seed_from_u64(0);

		for _ in 0..NUM_TESTS {
			let a = P::random(&mut rng);

			let result = a.mul_alpha();
			for i in 0..P::WIDTH {
				assert_eq!(result.get(i), MulAlpha::mul_alpha(a.get(i)));
			}
		}
	}

	define_multiply_tests!(TaggedMul<PackedStrategy>::mul, TaggedMul<PackedStrategy>);

	define_square_tests!(TaggedSquare<PackedStrategy>::square, TaggedSquare<PackedStrategy>);

	define_invert_tests!(
		TaggedInvertOrZero<PackedStrategy>::invert_or_zero,
		TaggedInvertOrZero<PackedStrategy>
	);
}
