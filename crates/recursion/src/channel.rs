// Copyright 2026 The Binius Developers

//! The channel that turns a verifier run into a circuit.

use std::rc::Rc;

use binius_circuits::multiplexer::multi_wire_multiplex;
use binius_field::{BinaryField128bGhash as B128, Field, FieldOps, util::FieldFn};
use binius_frontend::{Circuit, Wire};
use binius_iop::merkle_channel::{self, MerkleIPVerifierChannel};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};

use crate::{Elem, Word, shared::Shared};

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
	pub inputs: Vec<Wire>,
	/// The wires holding the inner statement.
	pub inout: Vec<Wire>,
}

/// A channel that records a verifier run as a Binius64 circuit.
///
/// Drive a verifier with this in place of a transcript channel and the result is a circuit rather
/// than a verdict. Wrap it in a `BaseFoldVerifierChannel` for the oracle layer, exactly as the
/// transcript channel is wrapped.
///
/// The builder is shared through an [`Rc`] because `IOPVerifierChannel` requires `Elem: 'static`,
/// so an element cannot borrow the builder it was built on. The frontend's `CircuitBuilder` takes
/// `&self` throughout, so no interior-mutability wrapper is needed around it.
///
/// See the crate docs for what this does and does not constrain.
pub struct Binius64BuilderChannel {
	shared: Rc<Shared>,
	inout: Vec<Wire>,
}

impl Binius64BuilderChannel {
	/// Creates a channel over a fresh builder, with an inner statement of `n_inout` words.
	pub fn new(n_inout: usize) -> Self {
		let shared = Rc::new(Shared::new());
		let inout = (0..n_inout).map(|_| shared.builder().add_inout()).collect();
		Self { shared, inout }
	}

	/// The inner statement, as this circuit's public input.
	pub fn statement(&self) -> Vec<Word> {
		self.inout
			.iter()
			.map(|&wire| Word::wire(&self.shared, wire))
			.collect()
	}

	/// Consumes the channel and compiles what it recorded.
	///
	/// Every [`Elem`] and [`Word`] derived from this channel must be dropped first: they hold weak
	/// handles to the builder, and using one afterwards panics.
	pub fn build(self) -> Recorded {
		let Self { shared, inout } = self;
		let shared = Rc::try_unwrap(shared)
			.unwrap_or_else(|_| panic!("Elem and Word values hold only weak handles"));
		Recorded {
			inputs: shared.inputs(),
			circuit: shared.builder().build(),
			inout,
		}
	}

	/// Allocates the `(lo, hi)` pair one field element occupies, as circuit inputs.
	fn input_elem(&mut self) -> Elem {
		let lo = self.shared.input_wire();
		let hi = self.shared.input_wire();
		Elem::wires(&self.shared, lo, hi)
	}
}

impl IPVerifierChannel<B128> for Binius64BuilderChannel {
	type Elem = Elem;

	fn recv_one(&mut self) -> Result<Elem, binius_ip::channel::Error> {
		Ok(self.input_elem())
	}

	fn sample(&mut self) -> Elem {
		// UNCONSTRAINED: a challenge is the Fiat-Shamir state's output, which needs the
		// challenger's hashing in-circuit. Until then it is an input the replay supplies.
		self.input_elem()
	}

	fn observe_one(&mut self, _val: B128) -> Elem {
		// UNCONSTRAINED: nothing absorbs the value.
		self.input_elem()
	}

	fn assert_zero(&mut self, val: Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			// A build-time constant is decided here; a non-zero one is unsatisfiable.
			Elem::Constant(c) => {
				if c == B128::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			Elem::Wires { lo, hi, .. } => {
				self.shared.builder().assert_zero("channel", lo);
				self.shared.builder().assert_zero("channel", hi);
				Ok(())
			}
		}
	}

	fn compute_public_value(&mut self, inputs: &[Elem], f: impl FieldFn<B128>) -> Elem {
		// Evaluated symbolically rather than hinted, so the circuit constrains it. For the
		// constraint system's monster multilinear that is proportional to the inner circuit's
		// size, which is what an accumulation or Spark argument would remove.
		f.call::<Elem>(inputs)
	}
}

impl WordIPVerifierChannel<B128> for Binius64BuilderChannel {
	type Word = Word;

	fn observe_words(&mut self, _words: &[Word]) {
		// UNCONSTRAINED: the statement must reach the Fiat-Shamir state for the circuit to be
		// bound to what it claims to verify.
	}

	fn subset_sum(&mut self, elems: &[Elem], word: &Word) -> Elem {
		assert!(elems.len() <= binius_core::word::Word::BITS); // precondition

		// Each element is kept or dropped by its own bit, and the survivors are summed. The sum is
		// XOR, which the constraint system absorbs, so the cost is the keep-or-drop.
		let builder = self.shared.builder();
		let (zero_lo, zero_hi) = Elem::zero().to_wires(builder);
		(0..elems.len())
			.map(|bit| {
				let sel = word.bit_at_msb(builder, bit);
				let (lo, hi) = elems[bit].to_wires(builder);
				Elem::wires(
					&self.shared,
					builder.select(sel, lo, zero_lo),
					builder.select(sel, hi, zero_hi),
				)
			})
			.sum()
	}

	fn select(&mut self, elems: &[Elem], word: &Word) -> Elem {
		assert!(!elems.is_empty() && elems.len().is_power_of_two()); // precondition

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
		Elem::wires(&self.shared, selected[0], selected[1])
	}

	fn sample_bits(&mut self, _bits: usize) -> Word {
		// UNCONSTRAINED, twice over: the index should come from the Fiat-Shamir state, and it
		// should be masked to `bits` bits, which the FRI code relies on rather than asserting.
		Word::wire(&self.shared, self.shared.input_wire())
	}
}

impl MerkleIPVerifierChannel<B128> for Binius64BuilderChannel {
	type Commitment = Commitment;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Commitment, merkle_channel::Error> {
		let root = std::array::from_fn(|_| self.shared.input_wire());
		Ok(Commitment {
			root,
			leaf_size,
			depth,
		})
	}

	fn recv_openings(
		&mut self,
		commitment: &Commitment,
		indices: &[Word],
	) -> Result<Vec<Elem>, merkle_channel::Error> {
		// UNCONSTRAINED: nothing binds the opened values to the root. Hashing each leaf and
		// climbing the path is the gadget this skeleton leaves out.
		Ok((0..indices.len() * commitment.leaf_size)
			.map(|_| self.input_elem())
			.collect())
	}

	fn recv_committed_vector(
		&mut self,
		commitment: &Commitment,
	) -> Result<Vec<Elem>, merkle_channel::Error> {
		// UNCONSTRAINED: nothing rebuilds the tree over the vector.
		Ok((0..commitment.leaf_size << commitment.depth)
			.map(|_| self.input_elem())
			.collect())
	}
}
