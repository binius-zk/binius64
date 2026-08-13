// Copyright 2026 The Binius Developers
//! XMSS signature verification.
//!
//! The scheme is the one leanVM-b's `xmss` crate implements, with BLAKE3 in place of BLAKE2s as
//! the hash underlying the tweakable hash. Every parameter, every hashed byte string and every
//! tweak is otherwise the same, so a signature is defined by exactly the same construction:
//!
//! - a Winternitz one-time signature over [`V`] chains of length [`CHAIN_LENGTH`], with a
//!   target-sum encoding and no checksum chains,
//! - whose chain ends hash into a Merkle leaf,
//! - which an authentication path links to the committed root.
//!
//! Both hashes compress 64 bytes at a time, so the swap costs nothing: a chain step is one
//! compression, a Merkle node one, the message encoding two and the WOTS public key eleven, and a
//! native verification is a constant 144 of them.
//!
//! ```text
//! 2 (encoding) + 99 (chains, fixed by the target sum) + 11 (leaf) + 32 (path) = 144
//! ```
//!
//! In circuit the chain work is not the 99 steps a verifier walks but every one of the
//! `V * (CHAIN_LENGTH - 1) = 294`: each step's tweak is a circuit constant, so a chain has to
//! evaluate all of its steps and take the hashed value only past its digit. The target sum buys
//! encoding validity here rather than verifier work, and a verification is 339 compressions.
//!
//! CREDIT: <https://github.com/leanEthereum/leanVM-b> (XMSS construction).

pub mod hashing;
pub mod wots;

/// Digest length in bytes: n = 128 bits.
pub const DIGEST_LEN: usize = 16;

/// A digest as 64-bit little-endian wires.
pub const DIGEST_WIRES: usize = DIGEST_LEN / 8;

/// Public parameter length in bytes. The parameter separates users.
pub const PUBLIC_PARAM_LEN: usize = 16;

/// Wires holding a public parameter.
pub const PUBLIC_PARAM_WIRES: usize = PUBLIC_PARAM_LEN / 8;

/// Signature randomness length in bytes, ground until the encoding is valid.
pub const RANDOMNESS_LEN: usize = 24;

/// Wires holding the randomness.
pub const RANDOMNESS_WIRES: usize = RANDOMNESS_LEN / 8;

/// The message to sign: a 256-bit message hash.
pub const MESSAGE_LEN: usize = 32;

/// Wires holding a message.
pub const MESSAGE_WIRES: usize = MESSAGE_LEN / 8;

/// Number of Winternitz hash chains.
pub const V: usize = 42;

/// Bits per encoding digit.
pub const W: usize = 3;

/// Chain length: a digit selects one of `2^W` positions.
pub const CHAIN_LENGTH: usize = 1 << W;

/// Chain hashes the verifier walks, summed over all chains: `sum(CHAIN_LENGTH - 1 - e_i)`.
///
/// Constant because the encoding sum is fixed to [`TARGET_SUM`].
pub const NUM_CHAIN_HASHES: usize = 99;

/// A WOTS encoding `(e_0, .., e_{V-1})` is valid exactly when every `e_i < CHAIN_LENGTH`, the
/// digits sum to this, and the two leftover digest bits are zero.
///
/// The signer grinds the randomness until the encoding is valid, which is what replaces the
/// checksum chains. 195 sits above the mean of 147 so the verifier walks fewer chain steps.
pub const TARGET_SUM: usize = V * (CHAIN_LENGTH - 1) - NUM_CHAIN_HASHES;

/// Merkle tree height: a key is valid for up to `2^LOG_LIFETIME` epochs.
pub const LOG_LIFETIME: usize = 32;

/// A 128-bit digest.
pub type Digest = [u8; DIGEST_LEN];

/// A per-signer public parameter.
pub type PublicParam = [u8; PUBLIC_PARAM_LEN];

/// Signature randomness.
pub type Randomness = [u8; RANDOMNESS_LEN];

/// The message to sign.
pub type Message = [u8; MESSAGE_LEN];

// The encoding uses V*W = 126 of the digest's 128 bits; the 2 leftover top bits are ground to
// zero, so the digest decomposes exactly into the digits.
const _: () = assert!(V * W + 2 == DIGEST_LEN * 8);

// The target sum is what fixes the verifier's chain work.
const _: () = assert!(TARGET_SUM == 195);
