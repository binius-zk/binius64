// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Utilities for hash circuit examples

use anyhow::{Result, bail, ensure};
use binius_core::word::Word;
use clap::Args;
use rand::prelude::*;

/// Default message size for hash circuit examples (1 KiB)
///
/// This value is chosen as a reasonable default that:
/// - Is large enough to demonstrate performance characteristics
/// - Small enough for quick testing and development
/// - Aligns with common block sizes in many systems
pub const DEFAULT_HASH_MESSAGE_BYTES: usize = 1024;

/// Standard seed for reproducible random message generation
pub const DEFAULT_RANDOM_SEED: u64 = 42;

/// Circuit-shape parameters shared by every hasher example.
///
/// These are compile-time [`Params`](crate::ExampleCircuit::Params): they select which gadget the
/// circuit is built from, independently of the message that is later hashed.
///
/// * `--message-len` builds the fixed-length gadget for exactly that many bytes.
/// * `--max-message-len` builds the variable-length gadget with that capacity.
/// * The two are mutually exclusive; if neither is given, the fixed-length gadget is built with
///   [`DEFAULT_HASH_MESSAGE_BYTES`].
#[derive(Args, Debug, Clone)]
pub struct HasherParams {
	/// Fixed message length in bytes; builds the more efficient fixed-length gadget. Mutually
	/// exclusive with `--max-message-len`.
	#[arg(long, conflicts_with = "max_message_len")]
	pub message_len: Option<usize>,

	/// Maximum message length in bytes; builds the variable-length gadget with this capacity.
	/// Mutually exclusive with `--message-len`.
	#[arg(long)]
	pub max_message_len: Option<usize>,
}

/// Instance (witness) parameters shared by every hasher example.
///
/// These are proof-time [`Instance`](crate::ExampleCircuit::Instance) arguments: they choose the
/// concrete message that is hashed by a circuit whose shape is already fixed by [`HasherParams`].
///
/// * `--message <STR>` hashes a UTF-8 string.
/// * `--random-message` hashes reproducible random bytes; `--random-message-len <LEN>` sets the
///   length (defaulting to the circuit's message length / maximum). It requires `--random-message`.
/// * `--message` and `--random-message` are mutually exclusive; random is the default when neither
///   is given.
#[derive(Args, Debug, Clone)]
pub struct HasherInstance {
	/// Hash reproducible random bytes. This is the default when `--message` is not given.
	#[arg(long, conflicts_with = "message")]
	pub random_message: bool,

	/// Length in bytes of the random message. Requires `--random-message`; defaults to the
	/// circuit's fixed length (or maximum, for a variable-length circuit).
	#[arg(long, requires = "random_message")]
	pub random_message_len: Option<usize>,

	/// Hash this UTF-8 string. Mutually exclusive with `--random-message`.
	#[arg(long)]
	pub message: Option<String>,
}

/// The gadget shape selected by [`HasherParams`].
#[derive(Debug, Clone, Copy)]
pub enum HasherMode {
	/// Fixed-length gadget: the message length is a compile-time constant.
	Fixed { len_bytes: usize },
	/// Variable-length gadget: the message length is a runtime witness bounded by `max_len_bytes`.
	Variable { max_len_bytes: usize },
}

impl HasherMode {
	/// The default random-message length for this mode: the fixed length, or the maximum length.
	pub const fn capacity_bytes(&self) -> usize {
		match self {
			HasherMode::Fixed { len_bytes } => *len_bytes,
			HasherMode::Variable { max_len_bytes } => *max_len_bytes,
		}
	}
}

/// Resolve [`HasherParams`] into the gadget shape the circuit should be built from.
///
/// `supports_variable` is `false` for hashers that only have a fixed-length gadget (Blake2s,
/// Blake2b, Blake3); passing `--max-message-len` to those is an error.
pub fn resolve_hasher_mode(
	params: &HasherParams,
	hasher: &str,
	supports_variable: bool,
) -> Result<HasherMode> {
	match (params.message_len, params.max_message_len) {
		// `conflicts_with` makes clap reject this before we get here; guard anyway.
		(Some(_), Some(_)) => {
			bail!("--message-len and --max-message-len are mutually exclusive")
		}
		(Some(len_bytes), None) => Ok(HasherMode::Fixed { len_bytes }),
		(None, Some(max_len_bytes)) => {
			ensure!(
				supports_variable,
				"--max-message-len is not supported for {hasher}: only a fixed-length gadget exists"
			);
			Ok(HasherMode::Variable { max_len_bytes })
		}
		(None, None) => Ok(HasherMode::Fixed {
			len_bytes: DEFAULT_HASH_MESSAGE_BYTES,
		}),
	}
}

/// Resolve a [`HasherInstance`] into the concrete message bytes to hash, validated against `mode`.
///
/// A `--message` string is used verbatim; otherwise reproducible random bytes are generated with
/// length `--random-message-len` (defaulting to `mode`'s capacity). The resulting length must be
/// compatible with the circuit: exactly the fixed length, or at most the variable-length maximum.
pub fn resolve_hasher_message(mode: &HasherMode, instance: &HasherInstance) -> Result<Vec<u8>> {
	let message = if let Some(s) = &instance.message {
		s.clone().into_bytes()
	} else {
		let len = instance.random_message_len.unwrap_or(mode.capacity_bytes());
		let mut rng = StdRng::seed_from_u64(DEFAULT_RANDOM_SEED);
		let mut bytes = vec![0u8; len];
		rng.fill_bytes(&mut bytes);
		bytes
	};

	ensure!(!message.is_empty(), "Message length must be positive");
	match mode {
		HasherMode::Fixed { len_bytes } => ensure!(
			message.len() == *len_bytes,
			"message length ({}) must equal the fixed circuit length ({})",
			message.len(),
			len_bytes
		),
		HasherMode::Variable { max_len_bytes } => ensure!(
			message.len() <= *max_len_bytes,
			"message length ({}) exceeds the maximum ({})",
			message.len(),
			max_len_bytes
		),
	}

	Ok(message)
}

/// One-line summary of the circuit shape selected by [`HasherParams`], for stat output.
pub fn hasher_param_summary(params: &HasherParams) -> Option<String> {
	Some(match (params.message_len, params.max_message_len) {
		(_, Some(max)) => format!("var{max}b"),
		(Some(len), None) => format!("{len}b"),
		(None, None) => format!("{DEFAULT_HASH_MESSAGE_BYTES}b"),
	})
}

/// Pack message bytes into 32-bit words, one word per wire with the high 32 bits zero.
///
/// The final word is zero-padded if the message length is not a multiple of 4. `big_endian`
/// selects the byte order within each word.
pub fn pack_bytes_u32words(message: &[u8], big_endian: bool) -> Vec<Word> {
	let n_words = message.len().div_ceil(4);
	(0..n_words)
		.map(|i| {
			let mut buf = [0u8; 4];
			let start = i * 4;
			let end = (start + 4).min(message.len());
			buf[..end - start].copy_from_slice(&message[start..end]);
			let word = if big_endian {
				u32::from_be_bytes(buf)
			} else {
				u32::from_le_bytes(buf)
			};
			Word(word as u64)
		})
		.collect()
}

/// Pack message bytes into 64-bit words, one word per wire.
///
/// The final word is zero-padded if the message length is not a multiple of 8. `big_endian`
/// selects the byte order within each word.
pub fn pack_bytes_u64words(message: &[u8], big_endian: bool) -> Vec<Word> {
	let n_words = message.len().div_ceil(8);
	(0..n_words)
		.map(|i| {
			let mut buf = [0u8; 8];
			let start = i * 8;
			let end = (start + 8).min(message.len());
			buf[..end - start].copy_from_slice(&message[start..end]);
			let word = if big_endian {
				u64::from_be_bytes(buf)
			} else {
				u64::from_le_bytes(buf)
			};
			Word(word)
		})
		.collect()
}
