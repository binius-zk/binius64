// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 Authors

//! These interfaces are taken from
//! [p3_symmetric](https://github.com/Plonky3/Plonky3/blob/main/symmetric/src/compression.rs) in
//! [Plonky3].
//!
//! Plonky3 is dual-licensed under MIT OR Apache 2.0. We use it under Apache 2.0.
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

/// An `N`-to-1 compression function used to build the inner nodes of a hash tree.
///
/// It folds `N` values into a single value of the same type, so it can be applied level by level:
/// the children of an inner node are compressed to produce that node.
pub trait CompressionFunction<T, const N: usize>: Clone {
	/// Maps the `N` inputs down to a single output of the same type.
	///
	/// In a hash tree this folds the `N` child node values into their parent node value.
	fn compress(&self, input: [T; N]) -> T;
}
