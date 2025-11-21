// Copyright 2023-2025 Irreducible Inc.

/// Error thrown when a field operation fails.
#[derive(Clone, thiserror::Error, Debug)]
pub enum Error {
	#[error("the argument does not match the field extension degree")]
	ExtensionDegreeMismatch,
	#[error(
		"the amount of scalars in the first array is not the same as the amount of scalars in the second"
	)]
	MismatchedLengths,
	#[error("the argument has too large a field extension degree")]
	ExtensionDegreeTooHigh,
	#[error("index {index} is out of range 0..{max}")]
	IndexOutOfRange { index: usize, max: usize },
	/// Thrown when trying to initialize a binary field element with a value bigger than what fits
	/// in the binary field.
	#[error("value is not in the field")]
	NotInField,
}
