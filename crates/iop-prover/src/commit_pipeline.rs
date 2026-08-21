// Copyright 2026 The Binius Developers

//! Encode-and-commit a Reed-Solomon codeword, overlapping Merkle leaf-hashing against the NTT.
//!
//! The sequential path runs the leaf-hash strictly after the whole NTT finishes.
//!
//! [`ReedSolomonCode::encode_batch`] splits its NTT into disjoint chunks, one per thread.
//! Each chunk finishes independently of its siblings, well before the whole codeword is done.
//! [`encode_and_commit_pipelined`] hashes a chunk's leaves the instant that chunk finishes.
//! That runs concurrently with the NTT still transforming the remaining chunks.

use binius_compute::Allocator;
use binius_field::{BinaryField, PackedField};
use binius_hash::binary_merkle_tree::{BinaryMerkleTree, HashSuite};
use binius_math::{
	FieldBuffer, FieldSlice,
	ntt::{DomainContext, NeighborsLastMultiThread},
	reed_solomon::ReedSolomonCode,
};
use digest::{Output, OutputSizeUser, array::ArraySize};

/// The codeword and Merkle tree [`encode_and_commit_pipelined`] returns.
pub type PipelinedCommit<P, H, A> = (
	FieldBuffer<P, <A as Allocator>::Vec<P>>,
	BinaryMerkleTree<Output<<H as HashSuite>::LeafHash>, A>,
);

/// Encodes `message` with `rs_code` and commits the resulting codeword.
///
/// Hashes each independent NTT chunk's leaves the instant that chunk finishes transforming.
/// That's instead of waiting for the whole codeword before hashing any of it.
///
/// The returned codeword and Merkle tree are byte-identical to the sequential path:
/// [`ReedSolomonCode::encode_batch`], then
/// [`BinaryMerkleTree::from_leaves`](binius_hash::binary_merkle_tree::BinaryMerkleTree::from_leaves).
/// Only the scheduling of leaf-hashing relative to the NTT changes.
/// No hashed or transformed value differs.
///
/// ## Preconditions
///
/// * Same as [`ReedSolomonCode::encode_batch`].
/// * `log_leaf_len` must be at most the codeword's log length.
/// * Every independent NTT chunk (see
///   [`NeighborsLastMultiThread::forward_transform_with_callback`]) must span at least one whole
///   leaf.
/// * That holds unless `log_leaf_len` is unusually large relative to `ntt.log_num_shares`, in which
///   case an assertion inside the leaf writer catches it.
pub fn encode_and_commit_pipelined<F, P, DC, H, N, A>(
	rs_code: &ReedSolomonCode<F>,
	ntt: &NeighborsLastMultiThread<DC>,
	message: FieldSlice<P>,
	log_batch_size: usize,
	log_leaf_len: usize,
	alloc: &A,
) -> PipelinedCommit<P, H, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	DC: DomainContext<Field = F> + Sync,
	H: HashSuite<LeafHash: OutputSizeUser<OutputSize = N>>,
	N: ArraySize,
	A: Allocator,
{
	let leaf_scalars = 1usize << log_leaf_len;
	let log_output_len = rs_code.log_dim() + log_batch_size + rs_code.log_inv_rate();
	assert!(log_leaf_len <= log_output_len, "precondition: log_leaf_len <= codeword log length");
	let log_n_leaves = log_output_len - log_leaf_len;

	// `from_leaves_pipelined`'s `populate` closure runs exactly once.
	// So this is written exactly once before being read back below.
	let mut codeword_slot: Option<FieldBuffer<P, A::Vec<P>>> = None;

	let tree = BinaryMerkleTree::from_leaves_pipelined::<F, H>(
		log_n_leaves,
		leaf_scalars,
		alloc,
		|writer| {
			let codeword = rs_code.encode_batch_with_callback(
				ntt,
				message,
				log_batch_size,
				alloc,
				|block, chunk: &[P]| {
					// A leaf is `leaf_scalars` scalars.
					// Every chunk this NTT split produces is the same size.
					// So the chunk index alone gives its leaf offset.
					let chunk_log_len = chunk.len().ilog2() as usize + P::LOG_WIDTH;
					let n_leaves_per_chunk = (1usize << chunk_log_len) / leaf_scalars;
					let leaf_start = block * n_leaves_per_chunk;

					// Zero-copy: reads scalars straight out of the packed chunk.
					// So a finished chunk's leaves hash without first flattening into a buffer.
					let chunk_view = FieldSlice::from_slice(chunk_log_len, chunk);
					writer.write_range(leaf_start, chunk_view.par_chunk_scalars(log_leaf_len));
				},
			);
			codeword_slot = Some(codeword);
		},
	);

	let codeword =
		codeword_slot.expect("`populate` above always runs exactly once and always sets this slot");
	(codeword, tree)
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_field::{
		BinaryField128bGhash as B128, PackedBinaryGhash4x128b, arch::OptimalPackedB128,
	};
	use binius_hash::sha256::Sha256HashSuite;
	use binius_math::{ntt::domain_context::GaoMateerOnTheFly, test_utils::random_field_buffer};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	/// Pins [`encode_and_commit_pipelined`] against the sequential encode-then-commit path.
	/// Same codeword, same committed root.
	fn check<P: PackedField<Scalar = B128>>(
		log_dim: usize,
		log_inv_rate: usize,
		log_num_shares: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);
		let message = random_field_buffer::<P>(&mut rng, log_dim);

		let rs_code = ReedSolomonCode::<B128>::new(log_dim, log_inv_rate);
		let domain_context = GaoMateerOnTheFly::<B128>::generate(rs_code.log_len());
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		let log_leaf_len = 4.min(rs_code.log_len());

		let (pipelined_codeword, pipelined_tree) =
			encode_and_commit_pipelined::<B128, P, _, Sha256HashSuite, _, _>(
				&rs_code,
				&ntt,
				message.to_ref(),
				0,
				log_leaf_len,
				&GlobalAllocator,
			);

		let sequential_codeword = rs_code.encode_batch(&ntt, message.to_ref(), 0, &GlobalAllocator);
		let sequential_tree = BinaryMerkleTree::<_, GlobalAllocator>::new::<B128, Sha256HashSuite>(
			&sequential_codeword.iter_scalars().collect::<Vec<_>>(),
			1 << log_leaf_len,
			&GlobalAllocator,
		)
		.unwrap();

		assert_eq!(pipelined_codeword.as_ref(), sequential_codeword.as_ref());
		assert_eq!(pipelined_tree.root(), sequential_tree.root());
		for depth in 0..=pipelined_tree.log_len {
			assert_eq!(
				pipelined_tree.layer(depth).unwrap(),
				sequential_tree.layer(depth).unwrap(),
				"layer {depth} differs"
			);
		}
	}

	#[test]
	fn test_matches_sequential_encode_then_commit() {
		// Every case below spans 1 to 8 independent NTT chunks (log_num_shares 0..=3).
		// Each chunk spans several leaves, so the leaf-range split exercises more than one.
		for log_num_shares in 0..=3 {
			check::<OptimalPackedB128>(10, 2, log_num_shares);
		}
		// A wider packing width, to exercise the scalar-unpacking path with `P::WIDTH > 1`.
		check::<PackedBinaryGhash4x128b>(10, 2, 2);
	}
}
