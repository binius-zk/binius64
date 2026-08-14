// Copyright 2026 The Binius Developers

//! In-circuit verification of a binary Merkle commitment over 128-bit field elements.
//!
//! A recursive circuit re-checks the openings a native prover produced.
//! So these gadgets reproduce the committing scheme's hash rules exactly.
//!
//! - A leaf is hashed with a full padded SHA-256 over its elements, sixteen bytes each.
//! - An inner node is one raw block compression over the 64 bytes of its two child digests.
//! - That compression starts from a domain-separating state, not SHA-256's own.
//!
//! # Element convention
//!
//! An element rides on two wires, low half first.
//! The low wire carries its serialized bytes 0 to 7 and the high wire its bytes 8 to 15.
//! Each half is read little-endian, so the pair is the element's 128-bit value split in two.
//!
//! # Digest convention
//!
//! A digest rides on four wires, in the form a block compression consumes.
//! Wire `j` carries the two big-endian message words covering digest bytes `8j` to `8j + 8`:
//!
//! ```text
//!     bits  0..32 :  big-endian u32 of digest bytes 8j     .. 8j + 4
//!     bits 32..64 :  big-endian u32 of digest bytes 8j + 4 .. 8j + 8
//! ```
//!
//! Two properties make this the cheap choice, and both explain the unusual byte order.
//!
//! - A node's 64-byte message block is then pure rewiring off its children's wires.
//! - A leaf digest lands in it for free, since SHA-256 emits its state big-endian.
//!
//! Only an inner node's output needs a conversion.
//! The native compression serializes its state *little*-endian, so each output word reverses.
//!
//! Two helpers here turn a digest or an element into the wire values a witness must carry.
//!
//! # Cost
//!
//! Counts below are AND and BMUL constraints as the frontend compiles them.
//! Every one is pinned by this module's tests.
//!
//! One single-lane block compression is 726 AND constraints.
//! A gate costs one constraint whatever its width, so two compressions can share one core.
//! Running them as the two 32-bit lanes of that core brings each to about 380.
//! Every place that hashes independent values takes this route:
//!
//! - folding a layer of digests up to the root above it
//! - hashing the leaves of a whole committed vector
//! - climbing two openings' authentication paths in step
//!
//! A climb is sequential, so an opening cannot share cores with itself.
//! Two openings can, since the queries are independent, which is what the last of those does.
//! An odd query count leaves one opening climbing on its own.
//!
//! | what is measured                        | AND           | BMUL          |
//! | --------------------------------------- | ------------- | ------------- |
//! | a one-element leaf digest               | 697           | 0             |
//! | a sixteen-element leaf digest           | 2279          | 0             |
//! | two one-element leaves sharing a core   | 707 per pair  | 0             |
//! | one inner node on its own core          | 742           | 0             |
//! | one inner node sharing a core           | 380 per node  | 0             |
//! | one node of a verified layer's fold     | 380 per node  | 0             |
//! | one climbed level of a lone opening     | 738           | 8             |
//! | one climbed level of a paired opening   | 377 per query | 8 per query   |
//! | picking the layer entry of an opening   | 0             | 4 * (2^L - 1) |
//!
//! Write `D` for the tree depth and `L` for the depth of the layer an opening climbs to.
//! One opening of a pair therefore costs
//!
//! ```text
//!     AND  = half a shared leaf digest + 377 * (D - L)
//!     BMUL = 8 * (D - L) + 4 * (2^L - 1)
//! ```
//!
//! and the layer it lands on costs `380 * (2^L - 1)` AND once, shared by every query.
//!
//! ## Choosing the layer depth
//!
//! Raising `L` by one level changes the bill per query, out of `n_q` queries, by
//!
//! ```text
//!     AND  :  -377 + 380 * 2^L / n_q
//!     BMUL :  +4 * 2^L - 8
//! ```
//!
//! The two columns behave nothing alike.
//!
//! - The AND column breaks even at `2^L = 0.99 * n_q`, which is the native rule itself.
//! - That native rule sets the layer depth to `ceil(log2(n_q))`.
//! - The BMUL column gains nothing past `L = 1`, since the entry selector grows with the layer.
//! - The climb it shortens gives back only eight selects a level, which never covers that.
//!
//! AND and BMUL fill separate constraint columns, so there is no one budget to minimise.
//! At `D = 20`, `n_q = 100` and one-element leaves, charging a BMUL like an AND:
//!
//! ```text
//!     L = 6 :  5876 AND + 364 BMUL
//!     L = 7 :  5740 AND + 612 BMUL
//!     L = 8 :  5846 AND + 1116 BMUL
//! ```
//!
//! AND still leads BMUL several times over, so the AND column decides.
//!
//! - On AND alone the cheapest choice is `L = 7`.
//! - Charging BMUL at the same rate moves it to `L = 6`.
//! - Both sit a level below where lone climbs would put them, since a level now costs half.
//! - Neither is a constant, since both track `log2(n_q)`.
//! - A different query count means re-running the arithmetic above.
//! - Which column is scarce depends on what the rest of the circuit put in each one.

use std::array;

use binius_circuits::{
	bytes::swap_bytes_32,
	multiplexer::multi_wire_multiplex,
	sha256::{State, sha256_compress, sha256_compress_2x, sha256_fixed},
	util::clear_high_bits,
};
use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

#[cfg(test)]
mod tests;

/// Wires carrying one SHA-256 digest.
pub const DIGEST_WORDS: usize = 4;

/// Wires carrying one field element.
pub const ELEMENT_WORDS: usize = 2;

/// Bytes one field element serializes to.
const ELEMENT_BYTES: usize = 16;

/// 32-bit words in one SHA-256 digest, which is also half a message block.
const DIGEST_HALF_WORDS: usize = 8;

/// Initial state a pair of child digests is folded from.
///
/// The scheme replaces SHA-256's own initial state with `SHA-256("BINIUS SHA-256 COMPRESS")`,
/// read as eight little-endian words.
/// Domain-separating the two-to-one compression this way keeps a node digest from ever
/// coinciding with a plain SHA-256 hash of the same 64 bytes.
pub const COMPRESSION_IV: [u32; 8] = [
	0x16684ff5, 0x53a717d2, 0x1d4154c8, 0x574f1b56, 0xa37e524e, 0xf12dfd41, 0x6303f932, 0x3754018c,
];

/// A SHA-256 digest in circuit form, in the byte order the [module docs](self) describe.
pub type Digest = [Wire; DIGEST_WORDS];

/// One field element in circuit form, as its low 64 bits then its high 64 bits.
pub type Element = [Wire; ELEMENT_WORDS];

/// The wire values a [`Digest`] carries for a given 32-byte digest.
pub fn digest_words(bytes: &[u8; 32]) -> [u64; DIGEST_WORDS] {
	// A message word is four digest bytes read big-endian, the order a compression consumes.
	let word = |offset: usize| {
		u32::from_be_bytes(
			bytes[offset..offset + 4]
				.try_into()
				.expect("the slice spans exactly four bytes"),
		) as u64
	};
	// Two consecutive message words share a wire, the earlier one in the low half.
	array::from_fn(|j| word(8 * j) | (word(8 * j + 4) << 32))
}

/// Writes a 32-byte digest into the wires that carry it.
pub fn populate_digest(w: &mut WitnessFiller, digest: &Digest, bytes: &[u8; 32]) {
	// One wire per eight digest bytes, in the packed order the digest convention fixes.
	for (wire, value) in digest.iter().zip(digest_words(bytes)) {
		w[*wire] = Word(value);
	}
}

/// The wire values an [`Element`] carries for a given field element.
///
/// The argument is the element's sixteen-byte little-endian serialization read as one integer.
pub const fn element_words(value: u128) -> [u64; ELEMENT_WORDS] {
	// Serialized bytes 0 to 7 in the low wire, bytes 8 to 15 in the high wire.
	[value as u64, (value >> 64) as u64]
}

/// Writes a field element into the wires that carry it.
pub fn populate_element(w: &mut WitnessFiller, element: &Element, value: u128) {
	// Two wires per element, low serialized half first.
	for (wire, word) in element.iter().zip(element_words(value)) {
		w[*wire] = Word(word);
	}
}

/// Verifies that a slice of digests is one Merkle layer beneath a claimed root.
///
/// The digests are folded pairwise until one is left, which must equal the claimed root.
/// The fold takes one round per level, so a layer only passes at the depth its width claims.
///
/// # Panics
///
/// Panics unless the number of layer digests is a non-zero power of two.
pub fn verify_layer(builder: &CircuitBuilder, root: Digest, layer_digests: &[Digest]) {
	assert!(
		layer_digests.len().is_power_of_two(),
		"precondition: the number of layer digests must be a non-zero power of two"
	);

	// Naming the gates makes a failing assertion point back at this gadget.
	let builder = builder.subcircuit("verify_layer");
	// Rebuild every node above the layer, up to the single digest at the top.
	let folded = fold_to_root(&builder, layer_digests);
	// Equality with the claimed root is the whole check.
	builder.assert_eq_v("root", folded, root);
}

/// Verifies a whole committed vector against a claimed root, by rebuilding the tree over it.
///
/// The values arrive in leaf order, with a fixed number of them under each leaf.
///
/// # Panics
///
/// Panics unless all of the following hold.
///
/// - The number of values under one leaf is non-zero.
/// - It divides the total number of values.
/// - The resulting leaf count is a power of two.
pub fn verify_vector(builder: &CircuitBuilder, root: Digest, data: &[Element], batch_size: usize) {
	assert_ne!(batch_size, 0, "precondition: batch_size must be non-zero");
	assert!(
		data.len().is_multiple_of(batch_size),
		"precondition: data length must be a multiple of batch_size"
	);
	let n_leaves = data.len() / batch_size;
	assert!(
		n_leaves.is_power_of_two(),
		"precondition: data.len() / batch_size must be a non-zero power of two"
	);

	let builder = builder.subcircuit("verify_vector");
	// The values under one leaf, a contiguous run of the committed data.
	let leaf = |i: usize| &data[i * batch_size..(i + 1) * batch_size];

	// Leaves are independent, so hash them two at a time in the two lanes of one compression chain.
	let mut digests = Vec::with_capacity(n_leaves);
	let mut i = 0;
	while i + 1 < n_leaves {
		let b = builder.subcircuit(format!("leaf[{i}..{}]", i + 2));
		digests.extend(leaf_digest_2x(&b, [leaf(i), leaf(i + 1)]));
		i += 2;
	}
	// The leaf count is a power of two, so only a single-leaf tree leaves one unpaired.
	if i < n_leaves {
		digests.push(leaf_digest(&builder.subcircuit(format!("leaf[{i}]")), leaf(i)));
	}

	// Folding the leaf digests rebuilds every inner node, so the top digest covers all the data.
	let folded = fold_to_root(&builder, &digests);
	// It must be the root that was committed to.
	builder.assert_eq_v("root", folded, root);
}

/// Verifies one Merkle opening against an already-verified layer of digests.
///
/// The gadget hashes the opened leaf, then climbs one level per authentication-path sibling:
///
/// ```text
///     level k:  (running digest, sibling k)  ->  running digest at level k + 1
/// ```
///
/// Bit `k` of the index says which side the running digest sits on at level `k`.
///
/// - Clear puts it on the left, so that level's sibling is the right child.
/// - Set puts it on the right, so that level's sibling is the left child.
///
/// The digest the climb arrives at must equal the layer entry the remaining index bits address.
/// That is what ties the opened leaf to the root the layer was verified against.
///
/// [`verify_opening_2x`] climbs two openings at once, for about half the AND cost a level here.
///
/// # Why the index is a wire
///
/// A caller may derive the query index in-circuit, so it cannot be a build-time constant.
/// Reading it from a wire has one consequence worth stating.
///
/// - Only the low index bits spanning the tree depth are read.
/// - So the opening is verified at the index reduced modulo the number of leaves.
/// - A caller whose index may fall out of range must constrain it separately.
///
/// # Panics
///
/// Panics unless all of the following hold.
///
/// - The layer depth is at most the tree depth.
/// - The tree depth is below the wire width.
/// - The layer holds one digest per node at its own depth.
/// - The authentication path holds one sibling per level between the two depths.
pub fn verify_opening(
	builder: &CircuitBuilder,
	index: Wire,
	values: &[Element],
	layer_depth: usize,
	tree_depth: usize,
	layer_digests: &[Digest],
	branch: &[Digest],
) {
	assert!(layer_depth <= tree_depth, "precondition: layer_depth must be at most tree_depth");
	assert!(tree_depth < Word::BITS, "precondition: tree_depth must be below the wire width");
	assert_eq!(
		layer_digests.len(),
		1 << layer_depth,
		"precondition: layer_digests must have 2^layer_depth entries"
	);
	assert_eq!(
		branch.len(),
		tree_depth - layer_depth,
		"precondition: branch must have tree_depth - layer_depth siblings"
	);

	let builder = builder.subcircuit("verify_opening");

	// Bottom of the authentication path: the digest of the leaf the opening claims.
	let mut digest = leaf_digest(&builder.subcircuit("leaf"), values);

	for (level, sibling) in branch.iter().enumerate() {
		let b = builder.subcircuit(format!("climb[{level}]"));

		// A select gate reads bit 63 alone, so lift this level's index bit up to it.
		let on_right = b.shl(index, (Word::BITS - 1 - level) as u32);
		// Order the pair by that bit, putting the running digest on the side it names.
		let left = array::from_fn(|j| b.select(on_right, sibling[j], digest[j]));
		let right = array::from_fn(|j| b.select(on_right, digest[j], sibling[j]));

		// One node hash carries the running digest up a level.
		digest = compress_node(&b, left, right);
	}

	// The climb consumed one index bit per level, so the bits above them address the layer.
	let layer_index = builder.shr(index, (tree_depth - layer_depth) as u32);
	// The multiplexer reads its candidates as slices, one per layer entry.
	let entries = layer_digests.iter().map(|d| &d[..]).collect::<Vec<_>>();
	let selected = multi_wire_multiplex(&builder, &entries, layer_index);
	// Matching the entry there binds the leaf to the layer, hence to the root above it.
	builder.assert_eq_v(
		"layer_digest",
		digest,
		selected
			.try_into()
			.expect("the multiplexer returns one wire per digest wire"),
	);
}

/// Verifies two Merkle openings at once, climbing them in the two lanes of shared cores.
///
/// A climb is sequential, so an opening cannot share cores with itself.
/// Two openings can, since the queries are independent at every level:
///
/// ```text
///     level k:  (running digest of query q, sibling k of query q)  ->  its digest at level k + 1
/// ```
///
/// Both queries open the same commitment, so their climbs span the same levels.
/// Only the hashing is shared: each query orders its own pair by its own index bit, as
/// [`verify_opening`] does, and matches its own layer entry at the top.
///
/// # Panics
///
/// Panics unless all of the following hold.
///
/// - The layer depth is at most the tree depth.
/// - The tree depth is below the wire width.
/// - The layer holds one digest per node at its own depth.
/// - Both leaves hold the same number of elements.
/// - Both authentication paths hold one sibling per level between the two depths.
pub fn verify_opening_2x(
	builder: &CircuitBuilder,
	indices: [Wire; 2],
	values: [&[Element]; 2],
	layer_depth: usize,
	tree_depth: usize,
	layer_digests: &[Digest],
	branches: [&[Digest]; 2],
) {
	assert!(layer_depth <= tree_depth, "precondition: layer_depth must be at most tree_depth");
	assert!(tree_depth < Word::BITS, "precondition: tree_depth must be below the wire width");
	assert_eq!(
		layer_digests.len(),
		1 << layer_depth,
		"precondition: layer_digests must have 2^layer_depth entries"
	);
	for branch in branches {
		assert_eq!(
			branch.len(),
			tree_depth - layer_depth,
			"precondition: each branch must have tree_depth - layer_depth siblings"
		);
	}

	let builder = builder.subcircuit("verify_opening_2x");

	// Both leaves hold the same number of elements, so the two lanes hash them in lockstep.
	let mut digests = leaf_digest_2x(&builder.subcircuit("leaf"), values);

	for level in 0..tree_depth - layer_depth {
		let b = builder.subcircuit(format!("climb[{level}]"));

		// Ordering a pair reads one query's own index bit, so it stays per query.
		let nodes = array::from_fn(|q| {
			// A select gate reads bit 63 alone, so lift this level's index bit up to it.
			let on_right = b.shl(indices[q], (Word::BITS - 1 - level) as u32);
			let (digest, sibling) = (digests[q], branches[q][level]);
			// Order the pair by that bit, putting the running digest on the side it names.
			let left = array::from_fn(|j| b.select(on_right, sibling[j], digest[j]));
			let right = array::from_fn(|j| b.select(on_right, digest[j], sibling[j]));
			(left, right)
		});

		// The two nodes are independent, so one two-lane compression carries both up a level.
		digests = compress_node_2x(&b, nodes);
	}

	// The multiplexer reads its candidates as slices, one per layer entry.
	let entries = layer_digests.iter().map(|d| &d[..]).collect::<Vec<_>>();
	for (q, digest) in digests.into_iter().enumerate() {
		// The climb consumed one index bit per level, so the bits above them address the layer.
		let layer_index = builder.shr(indices[q], (tree_depth - layer_depth) as u32);
		let selected = multi_wire_multiplex(&builder, &entries, layer_index);
		// Each query is bound to the entry its own index addresses, hence to the root above it.
		builder.assert_eq_v(
			format!("layer_digest[{q}]"),
			digest,
			selected
				.try_into()
				.expect("the multiplexer returns one wire per digest wire"),
		);
	}
}

/// The digest of one leaf: SHA-256 over its elements, sixteen little-endian bytes each.
///
/// # Panics
///
/// Panics if the leaf holds no elements.
pub fn leaf_digest(builder: &CircuitBuilder, values: &[Element]) -> Digest {
	assert!(!values.is_empty(), "precondition: a leaf must hold at least one element");

	// The leaf's serialization, as the big-endian message words a compression consumes.
	let words = leaf_message_words(builder, values);
	// The byte length is known while the circuit is built, so the padding is all constants.
	let state = sha256_fixed(builder, &words, values.len() * ELEMENT_BYTES);

	// SHA-256 serializes its state big-endian, which is already the message-word view.
	pack_words(builder, &state)
}

/// One inner Merkle node: the scheme's compression over its two child digests concatenated.
pub fn compress_node(builder: &CircuitBuilder, left: Digest, right: Digest) -> Digest {
	// Left child then right child is 64 bytes, which is exactly one message block.
	let block = node_block(builder, left, right);
	// A raw compression from the domain-separating state, with no padding and no length field.
	let out = sha256_compress(builder, compression_iv(builder), block);

	// A node digest is the final state serialized little-endian, one word at a time.
	// So its big-endian message-word view is each state word byte-reversed.
	pack_words(builder, &out.0).map(|w| swap_bytes_32(builder, w))
}

/// Two inner Merkle nodes at once, as the two 32-bit lanes of a single compression.
///
/// Nodes at the same level are independent, and a gate costs one constraint whatever its width.
/// So a pair of nodes costs about what one node costs.
fn compress_node_2x(builder: &CircuitBuilder, nodes: [(Digest, Digest); 2]) -> [Digest; 2] {
	// One message block per node, each still a single-lane value in its low 32 bits.
	let block_0 = node_block(builder, nodes[0].0, nodes[0].1);
	let block_1 = node_block(builder, nodes[1].0, nodes[1].1);

	// Lane 0 in the low half of every wire, lane 1 in the high half.
	// The two halves are disjoint, so XOR is what merges them.
	let merged = array::from_fn(|k| builder.bxor(block_0[k], builder.shl(block_1[k], 32)));
	// Both lanes fold from the same state, so its words are replicated into both halves.
	let iv = replicate_lanes(builder, compression_iv(builder));
	let out = sha256_compress_2x(builder, iv, merged);

	// Split the lanes apart, then byte-reverse each word as the single-lane path does.
	[Lane::Low, Lane::High]
		.map(|lane| pack_lane(builder, &out.0, lane).map(|w| swap_bytes_32(builder, w)))
}

/// Folds a power-of-two layer of digests down to the single digest above them all.
///
/// Each round pairs neighbours and replaces them with their parent, halving the layer:
///
/// ```text
///     [d_0, d_1, ..., d_{n-1}]  ->  [node(d_0, d_1), ..., node(d_{n-2}, d_{n-1})]
/// ```
///
/// # Panics
///
/// Panics unless the number of digests is a non-zero power of two.
fn fold_to_root(builder: &CircuitBuilder, digests: &[Digest]) -> Digest {
	assert!(
		digests.len().is_power_of_two(),
		"precondition: the number of digests must be a non-zero power of two"
	);

	let mut layer = digests.to_vec();
	let mut level = 0;
	// Each round climbs one level, so this runs the base-two logarithm of the layer width.
	while layer.len() > 1 {
		let b = builder.subcircuit(format!("fold[{level}]"));
		let mut parents = Vec::with_capacity(layer.len() / 2);

		// Four digests give two independent parents, which share one two-lane compression.
		let mut quads = layer.chunks_exact(4);
		for quad in &mut quads {
			parents.extend(compress_node_2x(&b, [(quad[0], quad[1]), (quad[2], quad[3])]));
		}
		// A layer of exactly two digests has one parent, with no partner to share a core with.
		if let [left, right] = *quads.remainder() {
			parents.push(compress_node(&b, left, right));
		}

		// The parents become the next round's layer, half as wide.
		layer = parents;
		level += 1;
	}

	// One digest is left, the root above the layer that came in.
	layer[0]
}

/// The digests of two equal-size leaves, as the two 32-bit lanes of one compression chain.
///
/// Both leaves serialize to the same byte length.
/// So their padded messages hold the same number of blocks and the two lanes advance in step.
///
/// # Why no hint is needed
///
/// Blocks within one leaf chain, so splitting those across lanes would need a hint to decouple.
/// Nothing crosses between these two lanes, since the leaves are separate messages.
///
/// # Panics
///
/// Panics unless both leaves are non-empty and hold the same number of elements.
fn leaf_digest_2x(builder: &CircuitBuilder, values: [&[Element]; 2]) -> [Digest; 2] {
	assert!(!values[0].is_empty(), "precondition: a leaf must hold at least one element");
	assert_eq!(
		values[0].len(),
		values[1].len(),
		"precondition: both leaves must hold the same number of elements"
	);

	// Serialize and pad each leaf on its own: equal lengths in, equal block counts out.
	let padded = values.map(|v| padded_leaf_words(builder, &leaf_message_words(builder, v)));
	// Both lanes open from SHA-256's initial state, replicated into both halves.
	let mut state = replicate_lanes(builder, State::iv(builder));
	// Sixteen message words to a block, and the two lanes consume their blocks in lockstep.
	for (i, (block_0, block_1)) in padded[0]
		.chunks_exact(16)
		.zip(padded[1].chunks_exact(16))
		.enumerate()
	{
		let b = builder.subcircuit(format!("compress[{i}]"));
		// Lane 0 low, lane 1 high, merged by XOR since the halves are disjoint.
		let merged = array::from_fn(|k| b.bxor(block_0[k], b.shl(block_1[k], 32)));
		state = sha256_compress_2x(&b, state, merged);
	}

	// SHA-256 serializes its state big-endian, so no byte reversal is due here.
	[Lane::Low, Lane::High].map(|lane| pack_lane(builder, &state.0, lane))
}

/// Which 32-bit half of a two-lane wire a word is read from.
#[derive(Clone, Copy)]
enum Lane {
	/// Bits `0..32`.
	Low,
	/// Bits `32..64`.
	High,
}

/// Packs eight single-lane 32-bit words, pairwise, into the four wires of a [`Digest`].
///
/// Word `2j` lands in the low half of wire `j` and word `2j + 1` in the high half.
///
/// It is a precondition that every word's high half is already empty.
fn pack_words(builder: &CircuitBuilder, words: &[Wire; DIGEST_HALF_WORDS]) -> Digest {
	// The high half is empty by precondition, so the upper word shifts in without a collision.
	array::from_fn(|j| builder.bxor(words[2 * j], builder.shl(words[2 * j + 1], 32)))
}

/// Packs one lane of eight two-lane words, pairwise, into the four wires of a [`Digest`].
fn pack_lane(builder: &CircuitBuilder, words: &[Wire; DIGEST_HALF_WORDS], lane: Lane) -> Digest {
	// Two words per output wire, each carrying the other lane's word in its opposite half.
	array::from_fn(|j| {
		let (lo, hi) = (words[2 * j], words[2 * j + 1]);
		match lane {
			// The low word must be isolated, since the word above it is XORed on top.
			// The upper word needs no masking: shifting it up drops the lane below it.
			Lane::Low => builder.bxor(clear_high_bits(builder, lo, 32), builder.shl(hi, 32)),
			// Shifting down isolates the low word; shifting back up drops what sat beneath it.
			Lane::High => builder.bxor(builder.shr(lo, 32), builder.shl(builder.shr(hi, 32), 32)),
		}
	})
}

/// The eight big-endian message words a [`Digest`] contributes to a block, in block order.
fn digest_message_words(builder: &CircuitBuilder, digest: Digest) -> [Wire; DIGEST_HALF_WORDS] {
	array::from_fn(|k| {
		// Two message words share a wire, so the word index halves to a wire index.
		let packed = digest[k / 2];
		if k.is_multiple_of(2) {
			// Even words sit in the low half, which needs isolating from the word above it.
			clear_high_bits(builder, packed, 32)
		} else {
			// Odd words sit in the high half, so shifting down both isolates and places them.
			builder.shr(packed, 32)
		}
	})
}

/// The 64-byte message block of an inner node: the left child's digest, then the right child's.
fn node_block(builder: &CircuitBuilder, left: Digest, right: Digest) -> [Wire; 16] {
	// Eight message words per digest, so unpacking the children covers the whole block.
	let left = digest_message_words(builder, left);
	let right = digest_message_words(builder, right);
	// The left child fills the block's first half and the right child its second.
	array::from_fn(|k| {
		if k < DIGEST_HALF_WORDS {
			left[k]
		} else {
			right[k - DIGEST_HALF_WORDS]
		}
	})
}

/// The domain-separating initial state as a compression input, one 32-bit word per wire.
fn compression_iv(builder: &CircuitBuilder) -> State {
	// The state is fixed by the scheme, so it enters as constants and costs no constraints.
	State::new(array::from_fn(|i| builder.add_constant(Word(COMPRESSION_IV[i] as u64))))
}

/// Replicates each state word into both 32-bit lanes of its wire.
///
/// A two-lane compression starts both lanes from the same state.
fn replicate_lanes(builder: &CircuitBuilder, state: State) -> State {
	// The incoming high half is empty, so a copy shifted up sits beside the original.
	State::new(state.0.map(|w| builder.bxor(w, builder.shl(w, 32))))
}

/// The big-endian message words of one leaf's serialized elements, in byte order.
fn leaf_message_words(builder: &CircuitBuilder, values: &[Element]) -> Vec<Wire> {
	// Sixteen bytes an element and four bytes a message word, so four words an element.
	let mut words = Vec::with_capacity(values.len() * 4);
	for &[lo, hi] in values {
		// The low serialized half comes first in the message, then the high half.
		for half in [lo, hi] {
			// The element serializes little-endian, so each group of four bytes reverses into its
			// big-endian message word.
			let swapped = swap_bytes_32(builder, half);
			// The earlier four bytes now sit in the low half, the later four above them.
			words.push(clear_high_bits(builder, swapped, 32));
			words.push(builder.shr(swapped, 32));
		}
	}
	words
}

/// Appends SHA-256's padding to a leaf's message words, filling out whole blocks.
///
/// A leaf is a whole number of sixteen-byte elements, so its length is a multiple of four bytes.
/// The delimiter byte therefore always opens a fresh word.
/// Every appended word is a build-time constant, so the padding costs no constraints.
///
/// # Panics
///
/// Panics if the message bit length does not fit in the low half of SHA-256's length field.
fn padded_leaf_words(builder: &CircuitBuilder, message: &[Wire]) -> Vec<Wire> {
	// Four bytes to a message word, and SHA-256's length field counts bits.
	let len_bytes = message.len() * 4;
	let bit_len = (len_bytes as u64) * 8;
	assert!(bit_len <= u32::MAX as u64, "precondition: the message bit length must fit in 32 bits");

	// The padding needs the delimiter byte plus the eight-byte length field.
	// Rounding that up to whole 64-byte blocks gives sixteen words a block.
	let n_words = (len_bytes + 9).div_ceil(64) * 16;

	let mut words = Vec::with_capacity(n_words);
	words.extend_from_slice(message);
	// The delimiter is one set bit, at the top of the word just past the message.
	words.push(builder.add_constant(Word(0x8000_0000)));
	// Zero-fill up to the length field, whose high word stays zero at these lengths.
	words.resize(n_words - 1, builder.add_constant(Word::ZERO));
	// The last word closes the final block with the message's bit length.
	words.push(builder.add_constant(Word(bit_len)));
	words
}
