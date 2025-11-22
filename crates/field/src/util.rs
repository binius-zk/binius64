// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use crate::Field;

/// Iterate the powers of a given value, beginning with 1 (the 0'th power).
pub fn powers<F: Field>(val: F) -> impl Iterator<Item = F> {
	iter::successors(Some(F::ONE), move |&power| Some(power * val))
}
