// Copyright 2026 The Binius Developers

//! The channel that turns a verifier run into a circuit.

use std::{array, rc::Rc};

use binius_circuits::multiplexer::multi_wire_multiplex;
use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, Field, FieldOps, util::FieldFn};
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_hash::StdHashSuite;
use binius_iop::{
	merkle_channel::{self, MerkleIPVerifierChannel},
	merkle_tree::{BinaryMerkleTreeScheme, MerkleTreeScheme},
};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel, select_word, subset_sum_word};

use crate::{
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
		Self {
			shared: Rc::new(Shared::new()),
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
		let shared = Rc::try_unwrap(self.shared).unwrap_or_else(|_| {
			panic!("SymbolicElem and SymbolicWord values hold only weak handles")
		});
		Recorded {
			inputs: shared.inputs(),
			circuit: shared.builder().build(),
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

	/// Allocates one field element as circuit inputs, in the form the protocol reads.
	fn input_elem(&mut self, kind: &'static str) -> SymbolicElem {
		let element = self.input_element(kind);
		self.elem(element)
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
		Ok(self.input_elem("recv_one"))
	}

	fn sample(&mut self) -> SymbolicElem {
		// UNCONSTRAINED: a challenge is the Fiat-Shamir state's output, which needs the
		// challenger's hashing in-circuit. Until then it is an input the replay supplies.
		self.input_elem("sample")
	}

	fn observe_one(&mut self, _val: B128) -> SymbolicElem {
		// UNCONSTRAINED: nothing absorbs the value.
		self.input_elem("observe_one")
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

	fn compute_public_value(
		&mut self,
		inputs: &[SymbolicElem],
		f: impl FieldFn<B128>,
	) -> SymbolicElem {
		// Evaluated symbolically rather than hinted, so the circuit constrains it. For the
		// constraint system's monster multilinear that is proportional to the inner circuit's
		// size, which is what an accumulation or Spark argument would remove.
		f.call::<SymbolicElem>(inputs)
	}
}

impl WordIPVerifierChannel<B128> for Binius64BuilderChannel {
	type Word = SymbolicWord;

	fn observe_words(&mut self, words: &[Word]) -> Vec<SymbolicWord> {
		// The statement enters the circuit here, one input wire per word, so everything downstream
		// reads it symbolically instead of baking it in as constants. That is what makes the
		// recorded circuit verify a *statement* rather than one fixed instance of it.
		//
		// UNCONSTRAINED: nothing feeds these into a Fiat-Shamir state yet, so the circuit is not
		// yet bound to the statement it claims to verify.
		words
			.iter()
			.map(|_| SymbolicWord::wire(&self.shared, self.shared.input_wire("observe_words")))
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

	fn sample_bits(&mut self, _bits: usize) -> SymbolicWord {
		// UNCONSTRAINED, twice over: the index should come from the Fiat-Shamir state, and it
		// should be masked to `bits` bits, which the FRI code relies on rather than asserting.
		SymbolicWord::wire(&self.shared, self.shared.input_wire("sample_bits"))
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
		let mut values = Vec::with_capacity(indices.len() * commitment.leaf_size);
		for index in indices {
			let leaf = (0..commitment.leaf_size)
				.map(|_| self.input_element("opening"))
				.collect::<Vec<_>>();
			let branch = (0..tree_depth - layer_depth)
				.map(|_| self.input_digest("merkle_branch"))
				.collect::<Vec<_>>();

			// Hashes the leaf, climbs the branch, then matches the layer entry the index addresses.
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
