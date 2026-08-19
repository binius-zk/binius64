// Copyright 2026 The Binius Developers

//! The channel that turns a verifier run into a circuit.

use std::{array, rc::Rc};

use binius_circuits::{bytes::swap_bytes_32, multiplexer::multi_wire_multiplex};
use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, Field, FieldOps};
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_hash::StdHashSuite;
use binius_iop::{
	merkle_channel::{self, MerkleIPVerifierChannel},
	merkle_tree::{BinaryMerkleTreeScheme, MerkleTreeScheme},
};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel, select_word, subset_sum_word};

use crate::{
	challenger::Sha256Challenger,
	merkle::{self, Digest, ELEMENT_WORDS, Element},
	shared::Shared,
	symbolic::{SymbolicElem, SymbolicWord},
};

/// A Merkle commitment received by the builder channel.
///
/// The root is wires, since the prover supplies it in the proof. The shape is fixed by the
/// protocol, so it stays concrete.
#[derive(Clone)]
pub struct Commitment {
	/// The commitment root.
	pub root: Digest,
	/// Field elements in each leaf.
	pub leaf_size: usize,
	/// Base-2 logarithm of the number of leaves.
	pub depth: usize,
}

/// A compiled circuit and what a witness for it needs.
pub struct Recorded {
	/// The compiled circuit.
	pub circuit: Circuit,
	/// The wires the witness must supply, in the order the verifier reached them.
	pub inputs: Vec<crate::shared::Input>,
}

/// A channel that records a verifier run as a Binius64 circuit.
///
/// Drive a verifier with this in place of a transcript channel and the result is a circuit rather
/// than a verdict. Wrap it in a `BaseFoldVerifierChannel` for the oracle layer, exactly as the
/// transcript channel is wrapped.
///
/// The builder is shared through an [`Rc`] because `IOPVerifierChannel` requires `SymbolicElem:
/// 'static`, so an element cannot borrow the builder it was built on. The frontend's
/// `CircuitBuilder` takes `&self` throughout, so no interior-mutability wrapper is needed around
/// it.
///
/// See the crate docs for what this does and does not constrain.
pub struct Binius64BuilderChannel {
	shared: Rc<Shared>,
	/// The Fiat-Shamir state, over wires.
	///
	/// Every observed byte the native challenger absorbs is absorbed here too, in the same order.
	/// A missed absorb desyncs it, and every challenge after that point is wrong.
	challenger: Sha256Challenger,
	/// How many assertions have been recorded, so each gets a distinct name.
	n_assertions: usize,
	/// Consulted only for the layer depth an opening decommits to, so no tree is ever built.
	scheme: BinaryMerkleTreeScheme<B128, StdHashSuite>,
	/// Merkle verifications emitted so far, used to name subcircuits.
	n_merkle_checks: usize,
	/// Words bound to public inputs so far, so each binding gets a distinct name.
	n_public: usize,
}

impl Binius64BuilderChannel {
	/// Creates a channel over a fresh builder.
	pub fn new() -> Self {
		let shared = Rc::new(Shared::new());
		// The challenger opens on its protocol seed, which is constant and so costs nothing.
		let challenger = Sha256Challenger::new(shared.builder());
		Self {
			shared,
			challenger,
			n_assertions: 0,
			scheme: BinaryMerkleTreeScheme::new(),
			n_merkle_checks: 0,
			n_public: 0,
		}
	}

	/// Binds words to fresh public inputs, returning the inout wire allocated for each.
	///
	/// Each word keeps the witness wire the replay fills, and gains a public wire equal to it.
	/// An outer proof can then read the value rather than trust whoever filled it.
	///
	/// Which words to bind is the caller's choice, so part of a statement can stay unexposed.
	pub fn bind_public(&mut self, words: Vec<SymbolicWord>) -> Vec<Wire> {
		// Numbered across calls, so a failing binding names which word it was.
		let first = self.n_public;
		self.n_public += words.len();

		// One named subcircuit, so a broken binding is traceable to this gadget.
		let builder = self.shared.builder().subcircuit("bind_public");
		words
			.into_iter()
			.enumerate()
			.map(|(i, word)| {
				let public = builder.add_inout();
				builder.assert_eq(format!("{}", first + i), word.to_wire(&builder), public);
				public
			})
			.collect()
	}

	/// Consumes the channel and compiles what it recorded.
	///
	/// Every [`SymbolicElem`] and [`SymbolicWord`] derived from this channel must be dropped first:
	/// they hold weak handles to the builder, and using one afterwards panics.
	pub fn build(self) -> Recorded {
		// The challenger keeps its own clone of the builder alive to absorb or sample from.
		// Dropping it here is what leaves exactly one handle to the shared state below.
		let Self {
			shared, challenger, ..
		} = self;
		drop(challenger);

		let shared = Rc::try_unwrap(shared).unwrap_or_else(|_| {
			panic!("SymbolicElem and SymbolicWord values hold only weak handles")
		});
		let inputs = shared.inputs();
		Recorded {
			inputs,
			circuit: shared.into_builder().build(),
		}
	}

	/// Allocates the `(lo, hi)` pair one field element occupies, as circuit inputs.
	fn input_element(&mut self, kind: &'static str) -> Element {
		array::from_fn(|_| self.shared.input_wire(kind))
	}

	/// Allocates the wires one digest occupies, as circuit inputs.
	fn input_digest(&mut self, kind: &'static str) -> Digest {
		array::from_fn(|_| self.shared.input_wire(kind))
	}

	/// Allocates the input wires one query's opening occupies: its leaf, then its branch.
	///
	/// The tape carries an opening in this order, so a replay fills the wires in it too.
	fn input_opening(
		&mut self,
		leaf_size: usize,
		branch_len: usize,
	) -> (Vec<Element>, Vec<Digest>) {
		let leaf = (0..leaf_size)
			.map(|_| self.input_element("opening"))
			.collect();
		let branch = (0..branch_len)
			.map(|_| self.input_digest("merkle_branch"))
			.collect();
		(leaf, branch)
	}

	/// Lifts a wire pair to the element type the protocol sees.
	fn elem(&self, [lo, hi]: Element) -> SymbolicElem {
		SymbolicElem::wires(&self.shared, lo, hi)
	}

	/// A subcircuit named for the next Merkle verification.
	fn merkle_subcircuit(&mut self, what: &str) -> CircuitBuilder {
		// A distinct name per check keeps a failing assertion traceable to the check that broke.
		let name = format!("{what}[{}]", self.n_merkle_checks);
		self.n_merkle_checks += 1;
		self.shared.builder().subcircuit(name)
	}
}

impl Default for Binius64BuilderChannel {
	fn default() -> Self {
		Self::new()
	}
}

impl IPVerifierChannel<B128> for Binius64BuilderChannel {
	type Elem = SymbolicElem;

	fn recv_one(&mut self) -> Result<SymbolicElem, binius_ip::channel::Error> {
		// A received element is read through the transcript's *message* reader, which observes it.
		// Its two halves are already the little-endian words the challenger absorbs.
		let element = self.input_element("recv_one");
		self.challenger.observe_words(&element);
		Ok(self.elem(element))
	}

	fn sample(&mut self) -> SymbolicElem {
		// Sixteen bytes of sampler output, packed little-endian into an element's two halves.
		let (lo, hi) = self.challenger.sample_b128();
		self.elem([lo, hi])
	}

	fn observe_one(&mut self, _val: B128) -> SymbolicElem {
		// Absorbed as wires the replay fills, so the circuit is not tied to one value.
		let element = self.input_element("observe_one");
		self.challenger.observe_words(&element);
		self.elem(element)
	}

	fn assert_zero(&mut self, val: SymbolicElem) -> Result<(), binius_ip::channel::Error> {
		match val {
			// A build-time constant is decided here; a non-zero one is unsatisfiable.
			SymbolicElem::Constant(c) => {
				if c == B128::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			SymbolicElem::Wires { lo, hi, .. } => {
				// Number the assertions so an unsatisfied circuit names which check failed, and
				// which half of the element it failed in.
				let n = self.n_assertions;
				self.n_assertions += 1;
				self.shared
					.builder()
					.assert_zero(format!("assert_zero[{n}].lo"), lo);
				self.shared
					.builder()
					.assert_zero(format!("assert_zero[{n}].hi"), hi);
				Ok(())
			}
		}
	}
}

impl WordIPVerifierChannel<B128> for Binius64BuilderChannel {
	type Word = SymbolicWord;

	fn observe_words(&mut self, words: &[Word]) -> Vec<SymbolicWord> {
		// The statement enters as wires the replay fills, not as constants.
		// Absorbing them binds every challenge below to this statement.
		let wires = words
			.iter()
			.map(|_| self.shared.input_wire("observe_words"))
			.collect::<Vec<_>>();
		self.challenger.observe_words(&wires);
		wires
			.into_iter()
			.map(|wire| SymbolicWord::wire(&self.shared, wire))
			.collect()
	}

	fn subset_sum(&mut self, elems: &[SymbolicElem], word: &SymbolicWord) -> SymbolicElem {
		assert!(elems.len() <= Word::BITS); // precondition

		// A word fixed while the circuit is built decides the selection there too, so it costs no
		// gates. `fold_coset` reaches this on every round of the terminal fold, where the coset
		// index is a constant.
		if let SymbolicWord::Constant(word) = word {
			return subset_sum_word(elems, *word);
		}

		// Each element is kept or dropped by its own bit, and the survivors are summed. The sum is
		// XOR, which the constraint system absorbs, so the cost is the keep-or-drop.
		let builder = self.shared.builder();
		let (zero_lo, zero_hi) = SymbolicElem::zero().to_wires(builder);
		(0..elems.len())
			.map(|bit| {
				// Move the bit into the most significant position, where `select` reads it.
				let sel = (word.clone() << (Word::BITS - 1 - bit) as u32).to_wire(builder);
				let (lo, hi) = elems[bit].to_wires(builder);
				SymbolicElem::wires(
					&self.shared,
					builder.select(sel, lo, zero_lo),
					builder.select(sel, hi, zero_hi),
				)
			})
			.sum()
	}

	fn select(&mut self, elems: &[SymbolicElem], word: &SymbolicWord) -> SymbolicElem {
		assert!(!elems.is_empty() && elems.len().is_power_of_two()); // precondition

		// As in `subset_sum`, a constant index picks its element while the circuit is built.
		if let SymbolicWord::Constant(word) = word {
			return select_word(elems, *word);
		}

		// One multiplexer over the `(lo, hi)` pairs, which is the same select-gate tree per wire
		// position with the index bits read inside.
		let builder = self.shared.builder();
		let pairs = elems
			.iter()
			.map(|elem| {
				let (lo, hi) = elem.to_wires(builder);
				[lo, hi]
			})
			.collect::<Vec<_>>();
		let groups = pairs.iter().map(|pair| pair.as_slice()).collect::<Vec<_>>();
		let sel = word.to_wire(builder);
		let selected = multi_wire_multiplex(builder, &groups, sel);
		SymbolicElem::wires(&self.shared, selected[0], selected[1])
	}

	fn sample_bits(&mut self, bits: usize) -> SymbolicWord {
		// The gadget masks the draw with a real gate, so the result provably lies below 2^bits.
		// The FRI code takes that bound as given rather than asserting it.
		// So deriving the index and masking it are one change, not two.
		SymbolicWord::wire(&self.shared, self.challenger.sample_bits(bits))
	}

	fn pack_words(&mut self, words: &[SymbolicWord]) -> Vec<SymbolicElem> {
		// A `SymbolicElem` *is* the low and high wire of a 128-bit element, and a word fills half
		// of it, so packing is pairing the wires up. It costs no gates, and a trailing odd word
		// takes the low half against a zero high half.
		let builder = self.shared.builder();
		words
			.chunks(ELEMENT_WORDS)
			.map(|chunk| {
				let lo = chunk[0].to_wire(builder);
				let hi = chunk
					.get(1)
					.map_or_else(|| builder.add_constant_64(0), |word| word.to_wire(builder));
				SymbolicElem::wires(&self.shared, lo, hi)
			})
			.collect()
	}
}

impl MerkleIPVerifierChannel<B128> for Binius64BuilderChannel {
	type Commitment = Commitment;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Commitment, merkle_channel::Error> {
		let root = self.input_digest("merkle_root");

		// A root is read through the message reader, so the challenger absorbs it.
		//
		// The wires hold the digest packed big-endian, which the hashing gadgets read.
		// The challenger absorbs the tape's little-endian words, so each wire is reversed back.
		let stream = {
			let builder = self.shared.builder();
			root.map(|word| swap_bytes_32(builder, word))
		};
		self.challenger.observe_words(&stream);

		Ok(Commitment {
			root,
			leaf_size,
			depth,
		})
	}

	/// Opens the commitment at every query index, returning the elements the opened leaves hold.
	///
	/// The opened values stay circuit inputs, since they are proof data.
	/// They stop being *free*: each leaf is hashed and climbed to a layer the root fixes.
	///
	/// Queries are climbed two at a time, sharing the hash cores their levels compress in.
	/// The advice is still read one query's leaf and branch at a time, as the tape carries it.
	///
	/// The index is a wire, so no range check is possible while the circuit is built.
	/// An opening is verified at the index reduced modulo the leaf count.
	/// Masking the sampled index so that reduction is a no-op is BINIUS-470.
	fn recv_openings(
		&mut self,
		commitment: &Commitment,
		indices: &[SymbolicWord],
	) -> Result<Vec<SymbolicElem>, merkle_channel::Error> {
		let tree_depth = commitment.depth;
		// The same rule the native verifier applies, so both sides stop climbing at the same level.
		let layer_depth = self.scheme.optimal_verify_layer(indices.len(), tree_depth);

		// One internal layer, folded to the root once and then shared by every climb below.
		let layer = (0..1 << layer_depth)
			.map(|_| self.input_digest("merkle_layer"))
			.collect::<Vec<_>>();
		let builder = self.merkle_subcircuit("layer");
		merkle::verify_layer(&builder, commitment.root, &layer);

		// Every query then climbs from its own leaf up to that layer.
		// Two climbs are independent, so they share hash cores, which halves what a level costs.
		let mut values = Vec::with_capacity(indices.len() * commitment.leaf_size);
		let mut pairs = indices.chunks_exact(2);
		for pair in &mut pairs {
			// The tape carries a leaf and a branch per query, whatever shares a core.
			let openings = [(); 2]
				.map(|()| self.input_opening(commitment.leaf_size, tree_depth - layer_depth));

			// Both leaves hashed, both branches climbed, then each index picks its layer entry.
			let builder = self.merkle_subcircuit("opening");
			let indices = array::from_fn(|q| pair[q].to_wire(&builder));
			merkle::verify_opening_2x(
				&builder,
				indices,
				[&openings[0].0, &openings[1].0],
				layer_depth,
				tree_depth,
				&layer,
				[&openings[0].1, &openings[1].1],
			);
			for (leaf, _) in openings {
				values.extend(leaf.into_iter().map(|element| self.elem(element)));
			}
		}
		// An odd query count leaves the last climb with no partner to share cores with.
		if let [index] = pairs.remainder() {
			let (leaf, branch) = self.input_opening(commitment.leaf_size, tree_depth - layer_depth);

			let builder = self.merkle_subcircuit("opening");
			let index = index.to_wire(&builder);
			merkle::verify_opening(
				&builder,
				index,
				&leaf,
				layer_depth,
				tree_depth,
				&layer,
				&branch,
			);
			values.extend(leaf.into_iter().map(|element| self.elem(element)));
		}
		Ok(values)
	}

	/// Receives the whole committed vector, checked by rebuilding its tree.
	///
	/// The data arrives in the clear, so there is no path to climb: the tree is rebuilt over it.
	fn recv_committed_vector(
		&mut self,
		commitment: &Commitment,
	) -> Result<Vec<SymbolicElem>, merkle_channel::Error> {
		// One leaf's worth of elements per leaf, across every leaf of the tree.
		let len = commitment.leaf_size << commitment.depth;
		let data = (0..len)
			.map(|_| self.input_element("committed_vector"))
			.collect::<Vec<_>>();

		let builder = self.merkle_subcircuit("vector");
		merkle::verify_vector(&builder, commitment.root, &data, commitment.leaf_size);

		Ok(data.into_iter().map(|element| self.elem(element)).collect())
	}
}

#[cfg(test)]
mod tests {
	use binius_transcript::{
		Buf,
		fiat_shamir::{Challenger, HasherChallenger, sample_bits_reader},
	};
	use sha2::Sha256;

	use super::*;
	use crate::merkle::element_words;

	/// Every query index the channel hands out is the native one, and fits the width it asked for.
	///
	/// Four sampler bytes are read whatever the width, so a narrow one almost always overflows.
	/// An unmasked draw would then differ from the native value and fail the pin below.
	///
	/// The bound matters on its own.
	/// The FRI code takes it as given, so a wide index could point a query anywhere satisfiable.
	#[test]
	fn a_sampled_index_is_the_native_one_and_fits_its_width() {
		// Widths where the mask drops bits, one that needs no mask, and one that clamps to 32.
		let widths = [1usize, 3, 7, 8, 11, 16, 31, 32, 33];

		let mut native = HasherChallenger::<Sha256>::default();
		let mut channel = Binius64BuilderChannel::new();

		let expected = widths
			.iter()
			.map(|&bits| sample_bits_reader(native.sampler(), bits))
			.collect::<Vec<_>>();
		let sampled = widths
			.iter()
			.map(|&bits| WordIPVerifierChannel::<B128>::sample_bits(&mut channel, bits))
			.collect::<Vec<_>>();

		// The reference values are in range, so pinning to them is what carries the bound.
		for (&bits, &want) in widths.iter().zip(&expected) {
			let width = bits.min(u32::BITS as usize);
			assert!(
				u64::from(want) < 1u64 << width,
				"the native index must fit the width it was drawn at"
			);
		}

		let pins = {
			let builder = channel.shared.builder();
			sampled
				.iter()
				.zip(&expected)
				.enumerate()
				.map(|(i, (word, &want))| {
					let claimed = builder.add_inout();
					builder.assert_eq(format!("index[{i}]"), word.to_wire(builder), claimed);
					(claimed, want)
				})
				.collect::<Vec<_>>()
		};

		// Every symbolic value must be dropped before the circuit is built.
		drop(sampled);
		let recorded = channel.build();
		assert!(
			recorded.inputs.is_empty(),
			"a derived index records nothing for a replay to fill: {:?}",
			recorded.inputs
		);

		let mut w = recorded.circuit.new_witness_filler();
		for (claimed, want) in pins {
			w[claimed] = Word(u64::from(want));
		}
		recorded.circuit.populate_wire_witness(&mut w).unwrap();
	}

	/// Draws a challenge, two query indices, then a second challenge, on both channels.
	///
	/// A bit sample reads from the same buffer a challenge reads from.
	/// A channel that skipped the gadget for one would hand the next challenge spent bytes.
	///
	/// The index itself is still unconstrained, which is why only the challenges are pinned.
	#[test]
	fn a_challenge_after_a_bit_sample_matches_the_native_challenger() {
		let mut native = HasherChallenger::<Sha256>::default();
		let mut channel = Binius64BuilderChannel::new();

		// The widths are arbitrary, but 11 and 5 are not multiples of eight, so the mask is live.
		// A 128-bit challenge deserializes from sixteen little-endian sampler bytes.
		let draw = |native: &mut HasherChallenger<Sha256>| {
			let mut bytes = [0u8; 16];
			native.sampler().copy_to_slice(&mut bytes);
			u128::from_le_bytes(bytes)
		};
		let first = draw(&mut native);
		sample_bits_reader(native.sampler(), 11);
		sample_bits_reader(native.sampler(), 5);
		let second = draw(&mut native);

		let one = IPVerifierChannel::<B128>::sample(&mut channel);
		WordIPVerifierChannel::<B128>::sample_bits(&mut channel, 11);
		WordIPVerifierChannel::<B128>::sample_bits(&mut channel, 5);
		let two = IPVerifierChannel::<B128>::sample(&mut channel);

		// Each challenge is pinned to a public wire carrying the native value.
		// A divergence then fails witness population rather than a Rust comparison.
		let pins = {
			let builder = channel.shared.builder();
			[(one, first), (two, second)]
				.iter()
				.enumerate()
				.map(|(i, (elem, want))| {
					let (lo, hi) = elem.to_wires(builder);
					let claimed = [builder.add_inout(), builder.add_inout()];
					builder.assert_eq_v(format!("challenge[{i}]"), [lo, hi], claimed);
					(claimed, *want)
				})
				.collect::<Vec<_>>()
		};

		// Every symbolic value must be dropped before the circuit is built.
		let recorded = channel.build();
		let mut w = recorded.circuit.new_witness_filler();
		for (claimed, want) in pins {
			for (wire, word) in claimed.iter().zip(element_words(want)) {
				w[*wire] = Word(word);
			}
		}
		// The discarded index draws are the only recorded inputs, and nothing reads their values.
		for input in &recorded.inputs {
			w[input.wire] = Word::ZERO;
		}
		recorded.circuit.populate_wire_witness(&mut w).unwrap();
	}
}
