// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use rand::RngCore;

use crate::{BinaryField, BinaryField1b, ExtensionField, packed::PackedBinaryField};

/// Generic transformation trait that is used both for scalars and packed fields
pub trait Transformation<Input, Output>: Sync {
	fn transform(&self, data: &Input) -> Output;
}

/// An $\mathbb{F}_2$-linear transformation on binary fields.
///
/// Stores transposed transformation matrix as a collection of field elements. `Data` is a generic
/// parameter because we want to be able both to have const instances that reference static arrays
/// and owning vector elements.
#[derive(Debug, Clone)]
pub struct FieldLinearTransformation<OF: BinaryField, Data: AsRef<[OF]> + Sync = &'static [OF]> {
	bases: Data,
	_pd: PhantomData<OF>,
}

impl<OF: BinaryField> FieldLinearTransformation<OF, &'static [OF]> {
	pub const fn new_const(bases: &'static [OF]) -> Self {
		assert!(bases.len() == OF::DEGREE);

		Self {
			bases,
			_pd: PhantomData,
		}
	}
}

impl<OF: BinaryField, Data: AsRef<[OF]> + Sync> FieldLinearTransformation<OF, Data> {
	pub fn new(bases: Data) -> Self {
		debug_assert_eq!(bases.as_ref().len(), OF::DEGREE);

		Self {
			bases,
			_pd: PhantomData,
		}
	}

	pub fn bases(&self) -> &[OF] {
		self.bases.as_ref()
	}
}

impl<IF: BinaryField, OF: BinaryField, Data: AsRef<[OF]> + Sync> Transformation<IF, OF>
	for FieldLinearTransformation<OF, Data>
{
	fn transform(&self, data: &IF) -> OF {
		assert_eq!(IF::DEGREE, OF::DEGREE);

		ExtensionField::<BinaryField1b>::iter_bases(data)
			.zip(self.bases.as_ref().iter())
			.fold(OF::ZERO, |acc, (scalar, &basis_elem)| acc + basis_elem * scalar)
	}
}

impl<OF: BinaryField> FieldLinearTransformation<OF, Vec<OF>> {
	pub fn random(mut rng: impl RngCore) -> Self {
		Self {
			bases: (0..OF::DEGREE).map(|_| OF::random(&mut rng)).collect(),
			_pd: PhantomData,
		}
	}
}

/// This crates represents a type that creates a packed transformation from `Self` to a packed
/// field based on the scalar field transformation.
#[allow(private_bounds)]
pub trait PackedTransformationFactory<OP>: PackedBinaryField
where
	OP: PackedBinaryField,
{
	type PackedTransformation<Data: AsRef<[OP::Scalar]> + Sync>: Transformation<Self, OP>;

	fn make_packed_transformation<Data: AsRef<[OP::Scalar]> + Sync>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self::PackedTransformation<Data>;
}

pub struct IDTransformation;

impl<OP: PackedBinaryField> Transformation<OP, OP> for IDTransformation {
	fn transform(&self, data: &OP) -> OP {
		*data
	}
}
