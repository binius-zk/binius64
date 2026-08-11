// Copyright 2026 The Binius Developers

//! The channel that turns a verifier run into a circuit.

use std::rc::Rc;

use binius_circuits::multiplexer::multi_wire_multiplex;
use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, Field, FieldOps, util::FieldFn};
use binius_frontend::{Circuit, Wire};
use binius_iop::merkle_channel::{self, MerkleIPVerifierChannel};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel, select_word, subset_sum_word};

use crate::{
	shared::Shared,
	symbolic::{SymbolicElem, SymbolicWord},
};

/// Number of 64-bit words a SHA-256 digest occupies.
pub(crate) const DIGEST_WORDS: usize = 4;

/// A Merkle commitment received by the builder channel.
///
/// The root is wires, since the prover supplies it in the proof. The shape is fixed by the
/// protocol, so it stays concrete.
#[derive(Clone)]
pub struct Commitment {
	/// The commitment root.
	pub root: [Wire; DIGEST_WORDS],
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
	/// The wires holding the inner statement, which are this circuit's public input.
	pub inout: Vec<Wire>,
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
	inout: Vec<Wire>,
	/// How many assertions have been recorded, so each gets a distinct name.
	n_assertions: usize,
}

impl Binius64BuilderChannel {
	/// Creates a channel over a fresh builder, for an inner statement of `n_inout` words.
	pub fn new(n_inout: usize) -> Self {
		let shared = Rc::new(Shared::new());
		let inout = (0..n_inout).map(|_| shared.builder().add_inout()).collect();
		Self {
			shared,
			inout,
			n_assertions: 0,
		}
	}

	/// The inner statement, as this circuit's own public input.
	///
	/// Pass this to `IOPVerifier::verify`. The circuit it records then verifies any instance of
	/// the inner system, since nothing about the statement is fixed while building.
	pub fn statement(&self) -> Vec<SymbolicWord> {
		self.inout
			.iter()
			.map(|&wire| SymbolicWord::wire(&self.shared, wire))
			.collect()
	}

	/// Consumes the channel and compiles what it recorded.
	///
	/// Every [`SymbolicElem`] and [`SymbolicWord`] derived from this channel must be dropped first:
	/// they hold weak handles to the builder, and using one afterwards panics.
	pub fn build(self) -> Recorded {
		let Self { shared, inout, .. } = self;
		let shared = Rc::try_unwrap(shared).unwrap_or_else(|_| {
			panic!("SymbolicElem and SymbolicWord values hold only weak handles")
		});
		Recorded {
			inputs: shared.inputs(),
			circuit: shared.builder().build(),
			inout,
		}
	}

	/// Allocates the `(lo, hi)` pair one field element occupies, as circuit inputs.
	fn input_elem(&mut self, kind: &'static str) -> SymbolicElem {
		let lo = self.shared.input_wire(kind);
		let hi = self.shared.input_wire(kind);
		SymbolicElem::wires(&self.shared, lo, hi)
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

	fn observe_words(&mut self, _words: &[SymbolicWord]) {
		// UNCONSTRAINED: the statement must reach the Fiat-Shamir state for the circuit to be
		// bound to what it claims to verify.
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
}

impl MerkleIPVerifierChannel<B128> for Binius64BuilderChannel {
	type Commitment = Commitment;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Commitment, merkle_channel::Error> {
		let root = std::array::from_fn(|_| self.shared.input_wire("merkle_root"));
		Ok(Commitment {
			root,
			leaf_size,
			depth,
		})
	}

	fn recv_openings(
		&mut self,
		commitment: &Commitment,
		indices: &[SymbolicWord],
	) -> Result<Vec<SymbolicElem>, merkle_channel::Error> {
		// UNCONSTRAINED: nothing binds the opened values to the root. Hashing each leaf and
		// climbing the path is the gadget this skeleton leaves out.
		Ok((0..indices.len() * commitment.leaf_size)
			.map(|_| self.input_elem("opening"))
			.collect())
	}

	fn recv_committed_vector(
		&mut self,
		commitment: &Commitment,
	) -> Result<Vec<SymbolicElem>, merkle_channel::Error> {
		// UNCONSTRAINED: nothing rebuilds the tree over the vector.
		Ok((0..commitment.leaf_size << commitment.depth)
			.map(|_| self.input_elem("committed_vector"))
			.collect())
	}
}
