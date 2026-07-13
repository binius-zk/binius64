//! STEP-2 VK: the once-per-CS-shape committed table M_VK = [X | Y | U | 0] and its
//! metadata blob VKM (spec section 2 "Committed objects" + P0.1 STEP-2 list).
//!
//! [REV S3] tag construction: ι maps address bit k to basis element β_k = B128(1 << k)
//! for k < n_a; the block tags are the LINEAR EXTENSION to the next two basis elements,
//! κ_c = ι'(c0, c1) = c0·β_{n_a} + c1·β_{n_a+1}:
//!   κ_x = 0, κ_y = β_{n_a}, κ_u = β_{n_a+1}, κ_pad = β_{n_a} + β_{n_a+1}.
//! The four pole families are then cosets of V = span(β_0..β_{n_a-1}) that are pairwise
//! disjoint BY CONSTRUCTION (κ_i ⊕ κ_j has a bit ≥ n_a, hence ∉ V); VKGen asserts this.
//! A bonus of the aligned basis: emb(a, c) = ι(a) + ι'(c) = B128(index as u128) — the
//! plain integer embedding of the (n_a + 2)-bit M_D index.
//!
//! VKGen is a PURE function of the canonical prepared-CS serialization; vk_digest ↔
//! cs_digest correspondence is trust root T1, discharged by the deterministic
//! regeneration audit (spec 8.12) — see tests (vkgen twice ⇒ byte-identical digest).

use std::time::Instant;

use anyhow::{Context, ensure};
use binius_field::Field;
use binius_hash::StdHashSuite;
use binius_iop::{
	fri::{FRIParams, PartialOracleSpec, calculate_n_test_queries},
	merkle_tree::{BinaryMerkleTreeScheme, MerkleTreeScheme},
};
use binius_iop_prover::{
	fri::{CommitOutput, commit_interleaved},
	merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver},
};
use binius_math::{
	BinarySubspace, FieldBuffer,
	ntt::{
		NeighborsLastMultiThread,
		domain_context::{GenericOnTheFly, GenericPreExpanded},
	},
};
use binius_utils::{DeserializeBytes, SerializeBytes};
use binius_verifier::config::B128;

use crate::table::{ShapeDims, TermTable, cube_y};

/// Spec-layout version frozen into the VKM (P0.1 "canonical row-order version").
/// v3: merged batched opening (W2) — ONE combined FRIParams over [M_VK, M_D].
pub const LAYOUT_VERSION: u64 = 3;
/// Canonical row order: constraint-major, slot-major a→b→c, operand-position; dummy
/// rows appended on [N, N_pad); zero-weight pad rows on [N_pad, 2^n_l).
pub const ROW_ORDER_VERSION: u64 = 1;
/// Merkle hash-suite id: 1 = SHA-256 (`binius_hash::StdHashSuite`).
pub const HASH_SUITE_ID: u64 = 1;
/// Security target for the discharge PCS: same budget class as the leaf proofs
/// (binius_verifier::verify::SECURITY_BITS = 96, log_inv_rate = 1).
pub const SECURITY_BITS: usize = 96;
/// log2 inverse Reed-Solomon rate for both discharge oracles.
pub const LOG_INV_RATE: usize = 1;

pub type Scheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;
pub type Digest = <Scheme as MerkleTreeScheme<B128>>::Digest;
pub type DischargeMerkleProver = BinaryMerkleTreeProver<B128, StdHashSuite>;
pub type DischargeNtt = NeighborsLastMultiThread<GenericPreExpanded<B128>>;

/// Pinned batched FRI shape for the merged opening (W2): ONE `FRIParams` over the two
/// oracles [M_VK (n_d + 2 vars), M_D (n_d vars)] with a UNIFORM interleave batch.
/// All fields FS-observed via `DischargeVkm::to_elems`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchFriShape {
	/// Uniform per-oracle log2 interleaved batch size (Merkle leaf coset width).
	pub log_batch_size: usize,
	/// The RESULTING combined fold arities (recorded, re-derived + asserted at build).
	pub fold_arities: Vec<usize>,
}

/// The VK metadata blob (P0.1, STEP-2 list): everything the verifier needs — it never
/// touches the ConstraintSystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DischargeVkm {
	pub layout_version: u64,
	pub row_order_version: u64,
	pub hash_suite_id: u64,
	/// P0.3 lane flag: this discharge admits AND-only shapes exclusively.
	pub lane_andonly: bool,
	/// SHA-256 of the canonical prepared-CS serialization (P0.2 anchor).
	pub cs_digest: [u8; 32],
	/// Serialized Merkle root of the M_VK commitment (transcript-free commit).
	pub vk_digest: Vec<u8>,
	/// Full shape dims (N, N_pad, parity, n_x, n_y, n_a, n_d, n_t, arity, ...).
	pub dims: ShapeDims,
	/// Tag basis indices (n_a, n_a + 1) — spec "[REV S3]".
	pub tag_bits: (usize, usize),
	pub log_inv_rate: usize,
	pub n_test_queries: usize,
	/// Batched FRI shape of the merged [M_VK, M_D] opening.
	pub fri_batch: BatchFriShape,
}

impl DischargeVkm {
	/// Canonical FS encoding, observed BEFORE mu (P0.1). Includes every field.
	pub fn to_elems(&self) -> Vec<B128> {
		let mut out = Vec::with_capacity(40);
		let push_u = |v: u64, out: &mut Vec<B128>| out.push(B128::new(v as u128));
		push_u(self.layout_version, &mut out);
		push_u(self.row_order_version, &mut out);
		push_u(self.hash_suite_id, &mut out);
		push_u(self.lane_andonly as u64, &mut out);
		out.push(B128::new(u128::from_le_bytes(
			self.cs_digest[..16].try_into().expect("16 bytes"),
		)));
		out.push(B128::new(u128::from_le_bytes(
			self.cs_digest[16..].try_into().expect("16 bytes"),
		)));
		// vk_digest: 32 bytes -> 2 elems (P0.1 hard requirement: FRI queries only bind
		// what the transcript absorbed, fri/verify.rs:200).
		assert_eq!(self.vk_digest.len(), 32, "sha256 digest");
		out.push(B128::new(u128::from_le_bytes(
			self.vk_digest[..16].try_into().expect("16 bytes"),
		)));
		out.push(B128::new(u128::from_le_bytes(
			self.vk_digest[16..].try_into().expect("16 bytes"),
		)));
		let d = &self.dims;
		for v in [
			d.n_x, d.n_x_mul, d.n_y, d.lp, d.n_pub, d.combined_len, d.n_a, d.n_d, d.n_terms,
			d.n_pad, d.n_t, d.parity as usize, d.arity, self.tag_bits.0, self.tag_bits.1,
			self.log_inv_rate, self.n_test_queries,
		] {
			push_u(v as u64, &mut out);
		}
		push_u(self.fri_batch.log_batch_size as u64, &mut out);
		push_u(self.fri_batch.fold_arities.len() as u64, &mut out);
		for &a in &self.fri_batch.fold_arities {
			push_u(a as u64, &mut out);
		}
		out
	}

	/// The typed Merkle digest for `basefold::verify`.
	pub fn vk_digest_typed(&self) -> anyhow::Result<Digest> {
		Digest::deserialize(&self.vk_digest[..]).context("vk_digest deserialize")
	}
}

pub fn serialize_digest(digest: &Digest) -> Vec<u8> {
	let mut buf = Vec::with_capacity(32);
	digest.serialize(&mut buf).expect("Vec<u8> write is infallible");
	buf
}

/// β_k as a field element: the aligned F2-basis used by ι/ι' (see module docs).
pub fn beta(k: usize) -> B128 {
	assert!(k < 128);
	B128::new(1u128 << k)
}

/// The four block tags κ_c = ι'(c0, c1) in block order [X, Y, U, pad].
pub fn tags(n_a: usize) -> [B128; 4] {
	[
		B128::ZERO,
		beta(n_a),
		beta(n_a + 1),
		beta(n_a) + beta(n_a + 1),
	]
}

/// The committed VK column entry for block `blk` at row t: ι(addr) + κ_blk.
/// Rows t >= N use the dummy tuple (x=0, y=0, u=0), i.e. plain κ_blk.
#[inline]
pub fn vk_entry(blk: usize, addr: u64, n_a: usize) -> B128 {
	debug_assert!((addr as u128) < (1u128 << n_a));
	match blk {
		0 => B128::new(addr as u128),
		1 => B128::new(addr as u128 | (1u128 << n_a)),
		2 => B128::new(addr as u128 | (1u128 << (n_a + 1))),
		_ => unreachable!("block 3 is the zero block"),
	}
}

/// Deterministic batched FRI shape for the merged [M_VK, M_D] opening (W2). Pure
/// function of the dims: uniform interleave batch 6, oracle order [M_VK, M_D],
/// combined arities chosen by `FRIParams::optimal_for_batch`.
pub fn fri_shapes(dims: &ShapeDims) -> (usize, BatchFriShape) {
	let scheme = Scheme::new();
	let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
	// Batch width 6: 64-element Merkle leaf cosets. Bounds the M_VK tree at
	// 2^{n_d + 2 - 6 + 1} digests and keeps per-query coset reads at 1 KiB.
	let log_batch_size = 6usize.min(dims.n_d.saturating_sub(1));
	let params = build_batch_params(dims, log_batch_size, n_test_queries, &scheme);
	(
		n_test_queries,
		BatchFriShape {
			log_batch_size,
			fold_arities: params.fold_arities().to_vec(),
		},
	)
}

fn build_batch_params(
	dims: &ShapeDims,
	log_batch_size: usize,
	n_test_queries: usize,
	scheme: &Scheme,
) -> FRIParams<B128> {
	let log_domain = dims.n_d + 2 - log_batch_size + LOG_INV_RATE;
	let subspace: BinarySubspace<B128> = BinarySubspace::with_dim(log_domain);
	let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
	let (params, _est) = FRIParams::optimal_for_batch(
		&domain_context,
		scheme,
		&[
			PartialOracleSpec {
				log_msg_len: dims.n_d + 2,
				log_batch_size: Some(log_batch_size),
			},
			PartialOracleSpec {
				log_msg_len: dims.n_d,
				log_batch_size: Some(log_batch_size),
			},
		],
		LOG_INV_RATE,
		n_test_queries,
	);
	params
}

/// PCS parameter set shared by prover and verifier, built deterministically from VKM
/// fields: ONE batched `FRIParams` over the oracle order [M_VK (index 0), M_D (1)].
pub struct Pcs {
	pub params: FRIParams<B128>,
	pub scheme: Scheme,
	/// Subspace dimension the prover-side NTT must span (max codeword length).
	pub log_domain: usize,
}

pub fn build_pcs(vkm: &DischargeVkm) -> anyhow::Result<Pcs> {
	ensure!(vkm.layout_version == LAYOUT_VERSION, "layout version mismatch");
	ensure!(vkm.row_order_version == ROW_ORDER_VERSION, "row order version mismatch");
	ensure!(vkm.hash_suite_id == HASH_SUITE_ID, "hash suite mismatch");
	ensure!(vkm.log_inv_rate == LOG_INV_RATE, "log_inv_rate mismatch");
	ensure!(vkm.lane_andonly, "P0.3: only AND-only shapes are admissible");
	ensure!(
		vkm.tag_bits == (vkm.dims.n_a, vkm.dims.n_a + 1),
		"tag basis indices must be (n_a, n_a+1)"
	);
	let scheme = Scheme::new();
	let log_domain = vkm.dims.n_d + 2 - vkm.fri_batch.log_batch_size + vkm.log_inv_rate;
	let params = build_batch_params(
		&vkm.dims,
		vkm.fri_batch.log_batch_size,
		vkm.n_test_queries,
		&scheme,
	);
	ensure!(
		params.fold_arities() == vkm.fri_batch.fold_arities,
		"VKM fold_arities mismatch: derived {:?} vs pinned {:?}",
		params.fold_arities(),
		vkm.fri_batch.fold_arities
	);
	Ok(Pcs {
		params,
		scheme,
		log_domain,
	})
}

/// Prover-side NTT over the shared domain (same construction as the main prover:
/// NeighborsLastMultiThread over GenericPreExpanded).
pub fn build_ntt(log_domain: usize) -> DischargeNtt {
	let subspace: BinarySubspace<B128> = BinarySubspace::with_dim(log_domain);
	let domain_context = GenericPreExpanded::generate_from_subspace(&subspace);
	let log_num_shares = (binius_utils::rayon::current_num_threads().ilog2() as usize)
		.min(log_domain.saturating_sub(6));
	NeighborsLastMultiThread::new(domain_context, log_num_shares)
}

/// Builds the M_VK message buffer per spec 1.1: index = t + 2^{n_d}·(c0 + 2c1), blocks
/// [X | Y | U | 0]. Direct indexed writes into the single flat buffer (no per-block
/// temporaries). Rows in [N, 2^{n_d}) hold the dummy tuple's entries.
pub fn build_m_vk(table: &TermTable) -> FieldBuffer<B128> {
	let dims = &table.dims;
	let n_l = dims.n_d;
	let block = 1usize << n_l;
	let mut vals = vec![B128::ZERO; block << 2];
	let kappa = tags(dims.n_a);
	// Block X (kappa_x = 0): real rows; dummy/pad rows are ι(0)+0 = 0, already zeroed.
	for (t, term) in table.terms.iter().enumerate() {
		vals[t] = vk_entry(0, term.x as u64, dims.n_a);
	}
	// Block Y (SEGMENTED cube index address).
	for (t, term) in table.terms.iter().enumerate() {
		vals[block + t] = vk_entry(1, cube_y(dims, term.y) as u64, dims.n_a);
	}
	vals[block + table.terms.len()..2 * block].fill(kappa[1]);
	// Block U.
	for (t, term) in table.terms.iter().enumerate() {
		vals[2 * block + t] = vk_entry(2, term.u as u64, dims.n_a);
	}
	vals[2 * block + table.terms.len()..3 * block].fill(kappa[2]);
	// Block 3 stays zero.
	FieldBuffer::new(n_l + 2, vals.into_boxed_slice())
}

/// Output of a VK commit: digest + the retained codeword/tree (for opening).
pub type VkCommit<P> =
	CommitOutput<P, Digest, <DischargeMerkleProver as MerkleTreeProver<B128>>::Committed>;

/// Commits M_VK transcript-free via `fri::commit_interleaved` (iop-prover fri/commit.rs)
/// as ORACLE 0 of the batched params. Generic over the prover packing (W1: instantiated
/// at `packed::PB`); the digest is a function of the scalar message values only.
pub fn commit_m_vk<P: binius_field::PackedField<Scalar = B128>>(
	pcs: &Pcs,
	ntt: &DischargeNtt,
	merkle_prover: &DischargeMerkleProver,
	m_vk: &FieldBuffer<P>,
) -> VkCommit<P> {
	commit_interleaved(&pcs.params, 0, ntt, merkle_prover, m_vk.to_ref())
}

/// VKGEN (once per CS shape): deterministic. Builds M_VK, asserts the [REV S3] coset
/// condition, commits, and returns the VKM. The commitment data is DROPPED (each
/// prover process re-commits deterministically and asserts digest equality, spec §4).
pub fn vkgen(table: &TermTable) -> anyhow::Result<(DischargeVkm, f64)> {
	let t0 = Instant::now();
	let dims = table.dims.clone();
	// P0.3 is enforced structurally: TermTable can only be extracted from an ANDONLY CS
	// (shape_dims rejects otherwise); record the lane flag.
	// [REV S3] coset-disjointness assert: κ_i ⊕ κ_j ∉ V = span(β_0..β_{n_a-1}).
	let kappa = tags(dims.n_a);
	for i in 0..4 {
		for j in 0..4 {
			if i != j {
				let diff = kappa[i] + kappa[j];
				// FWD-PORT: `B128::val()` now returns the `M128` underlier (not `u128`); test the
				// high bits against the zero underlier.
				ensure!(
					(diff.val() >> dims.n_a) != B128::ZERO.val(),
					"tag coset condition violated: kappa_{i} ^ kappa_{j} lies in V"
				);
			}
		}
	}
	ensure!(dims.n_a + 2 <= 128, "address width exceeds field");

	let (n_test_queries, fri_batch) = fri_shapes(&dims);
	let mut vkm = DischargeVkm {
		layout_version: LAYOUT_VERSION,
		row_order_version: ROW_ORDER_VERSION,
		hash_suite_id: HASH_SUITE_ID,
		lane_andonly: true,
		cs_digest: table.cs_digest,
		vk_digest: Vec::new(),
		tag_bits: (dims.n_a, dims.n_a + 1),
		dims,
		log_inv_rate: LOG_INV_RATE,
		n_test_queries,
		fri_batch,
	};
	let pcs = build_pcs(&vkm)?;
	let ntt = build_ntt(pcs.log_domain);
	let merkle_prover = DischargeMerkleProver::new();
	let m_vk = crate::packed::build_m_vk_packed::<crate::packed::PB>(table);
	let commit = commit_m_vk(&pcs, &ntt, &merkle_prover, &m_vk);
	vkm.vk_digest = serialize_digest(&commit.commitment);
	Ok((vkm, t0.elapsed().as_secs_f64()))
}
