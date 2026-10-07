// Copyright 2024-2025 Irreducible Inc.

#![warn(rustdoc::missing_crate_level_docs)]

//! Utility modules used in Binius.

pub mod checked_arithmetics;
pub mod iter;
#[cfg(feature = "platform-diagnostics")]
pub mod platform_diagnostics;
pub mod rand;
pub mod rayon;
pub mod serialization;
pub mod strided_array;

pub use bytes;
pub use serialization::{
	DeserializeBytes, FixedSizeSerializeBytes, SerializationError, SerializeBytes,
};
