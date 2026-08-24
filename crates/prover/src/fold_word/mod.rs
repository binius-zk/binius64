// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Contracting a word list against weights, along either of its two axes.
//!
//! A `&[Word]` list is a matrix over GF(2): row `i` is word `i`, and column `b` is bit position
//! `b` across every word.
//!
//! ```text
//!     words[0]  b63 b62 ... b1 b0
//!     words[1]  b63 b62 ... b1 b0
//!       ...
//!     words[n]  b63 b62 ... b1 b0
//! ```
//!
//! Every fold here contracts one axis of that matrix against one weight per position of it.
//!
//! | contracts | weights | leaves |
//! |---|---|---|
//! | the bit axis, the columns | one per bit position | one element per word |
//! | the word axis, the rows | an equality tensor over the rows | one element per bit position |
//!
//! Both use the [Method of Four Russians]: group eight weights, precompute all 256 of their subset
//! sums, then let one byte of the matrix index eight positions' whole contribution at once.
//!
//! The two directions differ in one step. Folding the bit axis reads a byte of the word directly.
//! Folding the word axis first transposes a group of eight rows, so that one byte carries eight
//! rows' bits at a single column.
//!
//! Each axis is folded through a folder that owns its lookup tables. Building the folder is what
//! costs, so a caller folding several lists against one point builds it once and reuses it.
//!
//! [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>

mod bit_axis;
mod lookup;
mod output;
mod word_axis;

use binius_core::word::Word;
pub use bit_axis::BitAxisFolder;
pub use word_axis::WordAxisFolder;

use crate::bit_matrix::WEIGHTS_PER_TABLE;

/// Number of words folded together within a single chunk.
///
/// One row-fold table covers one byte of a word, so a chunk spans every row those tables reach:
/// eight tables of eight rows each. That this equals the bit width of a word is arithmetic, not
/// definition.
const CHUNK_SIZE: usize = Word::BYTES * WEIGHTS_PER_TABLE;
/// Base-2 logarithm of the number of words folded together within a single chunk.
const LOG_CHUNK_SIZE: usize = CHUNK_SIZE.ilog2() as usize;
/// One 64-bit word with its bit axis expanded into full field elements.
///
/// Each bit position becomes one element, so the word is carried in oblong form.
pub type FoldedWord<F> = [F; Word::BITS];
