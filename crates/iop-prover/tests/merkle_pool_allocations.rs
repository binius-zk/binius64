// Copyright 2026 The Binius Developers

//! That a pooled prover stops going to the global allocator for its large buffers.
//!
//! Covers the two buffers a BaseFold prover allocates per committed oracle: the Merkle tree's
//! nodes, and the Reed-Solomon codeword.
//!
//! Wall-clock is the wrong instrument for either: the saving is a handful of large allocations
//! against work dominated by hashing and NTTs, well inside the run-to-run spread of a loaded
//! machine. The count of large global allocations is the same fact measured exactly, and it does
//! not depend on how busy the box is.
//!
//! This is an integration test rather than a unit test because it installs a
//! `#[global_allocator]`, which is a whole-binary choice.

use std::{
	alloc::{GlobalAlloc, Layout, System},
	sync::{
		Mutex,
		atomic::{AtomicBool, AtomicUsize, Ordering},
	},
};

use binius_compute::{BufferPool, GlobalAllocator};
use binius_field::{BinaryField128bGhash as B128, PackedBinaryGhash1x128b};
use binius_hash::StdHashSuite;
use binius_iop::{channel::OracleSpec, fri::FRIParams};
use binius_iop_prover::{
	fri,
	merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver},
};
use binius_math::{
	ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
	test_utils::{random_field_buffer, random_scalars},
};
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

/// Serializes the tests in this binary.
///
/// The counter is process-global and counts allocations from every thread, so two tests measuring
/// at once would each see the other's buffers. Cargo runs tests in parallel by default, so each
/// test holds this for its whole body — not just its counting window, since the work outside the
/// window allocates too.
static MEASURING: Mutex<()> = Mutex::new(());

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
	let _measuring = MEASURING
		.lock()
		.expect("no test in this binary panics while holding this");
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

/// Base-2 logarithm of the encoded message dimension; the codeword is 2^(LOG_DIM+2) words.
const LOG_DIM: usize = 16;

#[test]
fn a_pooled_encode_stops_allocating_the_codeword_globally() {
	// Invariant: with the allocator threaded through, the Reed-Solomon codeword, the mask and the
	// concatenation temporary all come from the pool, so a repeat encode asks the OS for nothing.
	//
	// Fixture state: the same message encoded four times through each allocator.
	let _measuring = MEASURING
		.lock()
		.expect("no test in this binary panics while holding this");
	let mut rng = StdRng::seed_from_u64(0);

	let merkle_prover = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();
	let domain_context = GaoMateerOnTheFly::generate(LOG_DIM + 1 + 1);
	let ntt = NeighborsLastSingleThread::new(domain_context);
	let (params, _) =
		FRIParams::optimal_for_batch(merkle_prover.scheme(), &[OracleSpec::new_zk(LOG_DIM)], 1, 32);
	let message = random_field_buffer::<PackedBinaryGhash1x128b>(&mut rng, LOG_DIM);

	// Warm both paths, so neither count includes one-off setup unrelated to the allocator.
	let pool = BufferPool::new();
	drop(fri::encode_masked(&params, 0, &ntt, message.to_ref(), &mut rng, &GlobalAllocator));
	drop(fri::encode_masked(&params, 0, &ntt, message.to_ref(), &mut rng, &&pool));

	let global_allocs = count_large_allocs(|| {
		for _ in 0..COMMITS {
			drop(fri::encode_masked(
				&params,
				0,
				&ntt,
				message.to_ref(),
				&mut rng,
				&GlobalAllocator,
			));
		}
	});
	let pooled_allocs = count_large_allocs(|| {
		for _ in 0..COMMITS {
			drop(fri::encode_masked(&params, 0, &ntt, message.to_ref(), &mut rng, &&pool));
		}
	});

	assert!(
		global_allocs >= COMMITS,
		"expected at least one large allocation per encode, got {global_allocs} for {COMMITS}"
	);
	assert!(
		pooled_allocs < global_allocs,
		"pooling must remove large allocations: {pooled_allocs} pooled against {global_allocs} global"
	);

	println!(
		"large allocations over {COMMITS} encodes: {global_allocs} global, {pooled_allocs} pooled"
	);
}
