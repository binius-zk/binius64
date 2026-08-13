// Copyright 2026 The Binius Developers

//! That a pooled Merkle tree prover stops going to the global allocator for its nodes.
//!
//! Wall-clock is the wrong instrument for this change: the saving is one large allocation per
//! committed tree, which is a few percent of a commitment dominated by hashing, and well inside
//! the run-to-run spread of a loaded machine. The count of large global allocations is the same
//! fact measured exactly, and it does not depend on how busy the box is.
//!
//! This is an integration test rather than a unit test because it installs a
//! `#[global_allocator]`, which is a whole-binary choice.

use std::{
	alloc::{GlobalAlloc, Layout, System},
	sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use binius_compute::BufferPool;
use binius_field::BinaryField128bGhash as B128;
use binius_hash::StdHashSuite;
use binius_iop_prover::merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver};
use binius_math::test_utils::random_scalars;
use rand::{SeedableRng, rngs::StdRng};

/// Allocations at or above this size are counted.
///
/// The tree below is about 1 MiB of nodes, so its buffer is far above the threshold, while the
/// small incidental allocations a commitment makes are far below it.
const LARGE: usize = 256 * 1024;

static LARGE_ALLOCS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

/// Counts large allocations while armed, and otherwise just forwards to the system allocator.
struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if layout.size() >= LARGE && COUNTING.load(Ordering::Relaxed) {
			LARGE_ALLOCS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.alloc(layout) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `f` with counting armed, and returns how many large allocations it made.
fn count_large_allocs(f: impl FnOnce()) -> usize {
	LARGE_ALLOCS.store(0, Ordering::Relaxed);
	COUNTING.store(true, Ordering::Relaxed);
	f();
	COUNTING.store(false, Ordering::Relaxed);
	LARGE_ALLOCS.load(Ordering::Relaxed)
}

/// Base-2 logarithm of the number of leaves; 2^15 nodes of 32 bytes is about 1 MiB of tree.
const LOG_LEAVES: usize = 14;
const COMMITS: usize = 4;

#[test]
fn a_pooled_prover_stops_allocating_tree_nodes_globally() {
	// Invariant: a prover holding a pool reuses one block of node memory across the trees it
	// commits, where a global prover asks the OS for a fresh block every time.
	//
	// Fixture state: the same data committed four times through each prover, in leaves of 2.
	let mut rng = StdRng::seed_from_u64(0);
	let data = random_scalars::<B128>(&mut rng, 1 << (LOG_LEAVES + 1));

	// Warm both paths first, so neither count includes one-off setup — rayon spinning up its
	// thread pool, say — that has nothing to do with the allocator under test.
	let global = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();
	let pool = BufferPool::new();
	let pooled = BinaryMerkleTreeProver::<B128, StdHashSuite, _>::with_allocator(&pool);
	drop(global.commit(&data, 2));
	drop(pooled.commit(&data, 2));

	let global_allocs = count_large_allocs(|| {
		for _ in 0..COMMITS {
			drop(global.commit(&data, 2));
		}
	});
	let pooled_allocs = count_large_allocs(|| {
		for _ in 0..COMMITS {
			drop(pooled.commit(&data, 2));
		}
	});

	// The global prover cannot do better than one large block per tree.
	assert!(
		global_allocs >= COMMITS,
		"expected at least one large allocation per tree, got {global_allocs} for {COMMITS} trees"
	);
	// The pool was warmed above, so every tree here is served from a recycled block.
	assert!(
		pooled_allocs < global_allocs,
		"pooling must remove large allocations: {pooled_allocs} pooled against {global_allocs} global"
	);

	println!(
		"large allocations over {COMMITS} commitments: {global_allocs} global, {pooled_allocs} pooled"
	);
}
