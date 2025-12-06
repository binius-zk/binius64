// Copyright 2024-2025 Irreducible Inc.

use crate::{
	PackedExtension, PackedField, TowerField,
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

#[cfg(test)]
mod tests {

	use super::*;
	use crate::BinaryField1b;

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
}
