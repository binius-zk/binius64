// Copyright 2026 The Binius Developers

//! Minimum task sizes for parallel loops.
//!
//! Handing a slice of work to another worker costs about a microsecond.
//! A task shorter than that loses more to the handoff than it gains.
//!
//! ```text
//!     min items per task = budget per task / estimated cost per item
//! ```
//!
//! - Loops bound by memory traffic charge one item by the bytes it moves.
//! - Loops bound by arithmetic charge one item by a coarse work class.
//!
//! Charging by bytes is what keeps a floor correct across packing widths.
//! Doubling the scalars in a packed element halves the items a task needs.
//!
//! # Environment overrides
//!
//! - `BINIUS_TASK_TARGET_NS` sets the time budget per task, in nanoseconds.
//! - `BINIUS_MIN_TASK_BYTES` sets the byte budget per task of a memory-bound loop.
//!
//! Setting both to `1` floors every loop at one item, which disables the floors for an A/B run.
//!
//! # Examples
//!
//! ```
//! use binius_utils::rayon::prelude::*;
//! use binius_utils::rayon::task_size::{IndexedParallelIteratorExt, WorkPerItem};
//!
//! let data = vec![1u64; 1 << 10];
//!
//! // Memory-bound: one task moves at least the byte budget.
//! let sum: u64 = data.par_iter().with_min_task_bytes::<u64>().sum();
//!
//! // Arithmetic-bound: one task runs for at least the time budget.
//! let max = data.par_iter().with_min_task(WorkPerItem::FieldMuls).max();
//!
//! assert_eq!((sum, max), (1 << 10, Some(&1)));
//! ```

use std::sync::OnceLock;

use super::prelude::*;

/// Time budget for one task, in nanoseconds.
///
/// Handing work to another worker costs roughly one microsecond.
/// A budget of 100 microseconds holds that overhead near one percent.
/// Raising it further would stop mid-size loops from splitting at all.
const DEFAULT_TASK_TARGET_NS: u64 = 100_000;

/// Byte budget for one task of a memory-bound loop.
///
/// One mebibyte streams in roughly the time budget above.
/// That assumes tens of gigabytes per second of bandwidth per core.
const DEFAULT_MIN_TASK_BYTES: usize = 1 << 20;

/// Reads a budget from the environment, falling back to a default.
///
/// # Arguments
///
/// * `name` - environment variable holding the override
/// * `default` - value used when the variable is absent or malformed
fn env_or<T: Copy + std::str::FromStr>(name: &str, default: T) -> T {
	std::env::var(name)
		.ok()
		// A budget is a tuning knob, so a typo falls back instead of taking the process down.
		.and_then(|v| v.parse().ok())
		.unwrap_or(default)
}

/// Time budget for one task, in nanoseconds.
///
/// The environment is read once, then the value is cached for the process.
fn task_target_ns() -> u64 {
	static V: OnceLock<u64> = OnceLock::new();

	// Caching matters because the budget is read inside loops that run every round.
	*V.get_or_init(|| env_or("BINIUS_TASK_TARGET_NS", DEFAULT_TASK_TARGET_NS))
}

/// Byte budget for one task of a memory-bound loop.
///
/// The environment is read once, then the value is cached for the process.
fn min_task_bytes() -> usize {
	static V: OnceLock<usize> = OnceLock::new();

	*V.get_or_init(|| env_or("BINIUS_MIN_TASK_BYTES", DEFAULT_MIN_TASK_BYTES))
}

/// Estimated cost of processing one item of a loop bound by arithmetic.
///
/// A floor only has to land within an order of magnitude, so the classes are coarse.
/// Guessing low leaves about a microsecond of handoff overhead per surplus task.
/// Guessing high delays the split until a somewhat larger input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkPerItem {
	/// A handful of packed field multiplies.
	///
	/// One output word of a fractional addition is three multiplies and an add.
	FieldMuls,
	/// One hash compression, such as a Merkle inner node or a single message block.
	HashCompression,
	/// One scalar field inversion, roughly a hundred packed multiplies.
	Inversion,
}

impl WorkPerItem {
	/// Estimated time to process one item, in nanoseconds.
	///
	/// Measured sequentially on an Apple M1 Pro, over buffers far larger than cache:
	///
	/// ```text
	///     1 to 3 packed binary-field multiplies per word :   1.4 to 3.9 ns
	///     one two-to-one SHA-256 compression            :  30 ns
	///     one scalar inversion                          : 140 ns
	/// ```
	///
	/// Landing within a factor of two on other hardware is good enough.
	const fn ns_per_item(self) -> u64 {
		match self {
			Self::FieldMuls => 4,
			Self::HashCompression => 30,
			Self::Inversion => 150,
		}
	}
}

/// Items per task that together move at least the given byte budget.
///
/// Split from the public entry point so tests can pin the arithmetic directly.
fn min_len_for_bytes_with(min_task_bytes: usize, item_bytes: usize) -> usize {
	// A zero-sized item is charged one byte, since dividing by its size would trap.
	let per_item = item_bytes.max(1);

	// An item wider than the whole budget still yields one item per task.
	(min_task_bytes / per_item).max(1)
}

/// Items per task that together run for at least the given time budget.
///
/// Split from the public entry point so tests can pin the arithmetic directly.
fn min_len_for_work_with(task_target_ns: u64, ns_per_item: u64) -> usize {
	// An item slower than the whole budget still yields one item per task.
	(task_target_ns / ns_per_item).max(1) as usize
}

/// Minimum items per task for a loop whose cost is the bytes it moves.
///
/// The type parameter is the element the loop streams, usually the packed field type.
/// A loop zipping several streams names an array type to count all of them.
/// A three-element array charges one item for three words.
///
/// # Returns
///
/// The number of items one task must take to move the byte budget.
pub fn min_len_for_bytes<T>() -> usize {
	min_len_for_bytes_with(min_task_bytes(), size_of::<T>())
}

/// Minimum items per task for a loop bound by arithmetic.
///
/// # Arguments
///
/// * `work` - cost class of one item of the loop
///
/// # Returns
///
/// The number of items one task must take to fill the time budget.
pub fn min_len_for_work(work: WorkPerItem) -> usize {
	min_len_for_work_with(task_target_ns(), work.ns_per_item())
}

/// Elements per chunk so one chunk spans the byte budget.
///
/// Use this directly as the chunk size of a chunked parallel loop.
/// The chunk size already is the floor, so no further floor is needed.
pub fn task_chunk_len<T>() -> usize {
	min_len_for_bytes::<T>()
}

/// Task-size adapters for parallel iterators.
///
/// Each adapter sets the minimum items per task from a cost model.
/// A call site states what one item costs instead of hardcoding a count.
pub trait IndexedParallelIteratorExt: IndexedParallelIterator {
	/// Floors the split so one task moves at least the byte budget.
	///
	/// The type parameter counts the bytes one item moves.
	/// Use this for loops bound by memory traffic: copies, transposes, permutations.
	#[inline]
	fn with_min_task_bytes<T>(self) -> impl IndexedParallelIterator<Item = Self::Item>
	where
		Self: Sized,
	{
		self.with_min_len(min_len_for_bytes::<T>())
	}

	/// Floors the split so one task runs for at least the time budget.
	///
	/// Use this for loops bound by arithmetic, classified by the cost of one item.
	#[inline]
	fn with_min_task(self, work: WorkPerItem) -> impl IndexedParallelIterator<Item = Self::Item>
	where
		Self: Sized,
	{
		self.with_min_len(min_len_for_work(work))
	}
}

impl<I: IndexedParallelIterator> IndexedParallelIteratorExt for I {}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn byte_floor_scales_inversely_with_item_size() {
		// Invariant: a task moves a fixed number of bytes, whatever the item width.
		// So the item count must fall by exactly the factor the item widens.
		//
		//     budget 1 MiB / 16 B per item = 65536 items
		//     budget 1 MiB / 64 B per item = 16384 items
		//     → 65536 = 4 * 16384
		let narrow = min_len_for_bytes_with(1 << 20, 16);
		let wide = min_len_for_bytes_with(1 << 20, 64);
		assert_eq!(narrow, 4 * wide);

		// The same ratio holds through the public entry point, which measures the type.
		assert_eq!(min_len_for_bytes::<[u8; 16]>(), 4 * min_len_for_bytes::<[u8; 64]>());

		// An array type charges one item for every stream a zipped loop touches.
		// Four streams of one word cost the same as one stream of four words.
		assert_eq!(min_len_for_bytes::<u64>(), 4 * min_len_for_bytes::<[u64; 4]>());
	}

	#[test]
	fn byte_floor_boundary_items() {
		// A zero-sized item is charged one byte, so the division cannot trap.
		// The whole budget then maps to one item per byte.
		assert_eq!(min_len_for_bytes_with(1 << 20, 0), 1 << 20);
		assert_eq!(min_len_for_bytes::<()>(), min_task_bytes());

		// An item wider than the entire budget cannot be subdivided further.
		//
		//     budget 16 B / 64 B per item = 0 → floored to 1
		assert_eq!(min_len_for_bytes_with(16, 64), 1);
	}

	#[test]
	fn work_floor_scales_inversely_with_item_cost() {
		// Invariant: a task runs for the time budget, whatever one item costs.
		//
		//     budget 100000 ns / 8 ns per item = 12500 items
		assert_eq!(min_len_for_work_with(100_000, 8), 12_500);

		// An item slower than the entire budget cannot be subdivided further.
		//
		//     budget 100 ns / 200 ns per item = 0 → floored to 1
		assert_eq!(min_len_for_work_with(100, 200), 1);
	}

	#[test]
	fn work_classes_are_ordered_by_cost() {
		// Filling one budget takes fewer items as each item grows more expensive.
		// This pins the ordering of the classes, not their absolute estimates.
		//
		//     multiplies (4 ns) < compression (30 ns) < inversion (150 ns)
		//     → item counts run the other way
		assert!(
			min_len_for_work(WorkPerItem::FieldMuls)
				> min_len_for_work(WorkPerItem::HashCompression)
		);
		assert!(
			min_len_for_work(WorkPerItem::HashCompression)
				> min_len_for_work(WorkPerItem::Inversion)
		);
	}

	#[test]
	fn chunk_len_matches_byte_floor() {
		// A chunked loop sizes its chunk exactly as an item-wise loop sizes its floor.
		// Both must span one budget, so the two entry points cannot drift apart.
		assert_eq!(task_chunk_len::<u64>(), min_len_for_bytes::<u64>());
	}

	#[test]
	fn adapters_preserve_iteration() {
		// The adapters constrain only how work is divided, never what it covers.
		//
		// Fixture state: 1000 items, floored well above 1000 by either model.
		//     → the loop runs as a single task, and every item is still visited once.
		let data: Vec<u64> = (0..1000).collect();
		let expected = 1000 * 999 / 2;

		// Memory-bound floor: 1 MiB budget over 8-byte items.
		let sum: u64 = data.par_iter().with_min_task_bytes::<u64>().sum();
		assert_eq!(sum, expected);

		// Arithmetic-bound floor: 100 microsecond budget over 30 ns items.
		let sum: u64 = data
			.par_iter()
			.with_min_task(WorkPerItem::HashCompression)
			.sum();
		assert_eq!(sum, expected);
	}
}
