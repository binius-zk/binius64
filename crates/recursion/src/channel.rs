// Copyright 2026 The Binius Developers

//! The channel that turns a verifier run into a circuit.

use std::rc::Rc;

use binius_circuits::multiplexer::multi_wire_multiplex;
use binius_field::{BinaryField128bGhash as B128, Field, FieldOps, util::FieldFn};
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_iop::{
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	merkle_channel::{self, MerkleIPVerifierChannel},
};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};

use crate::{Elem, Word};

/// Number of 64-bit words a SHA-256 digest occupies.
const DIGEST_WORDS: usize = 4;

/// A Merkle commitment received by the builder channel.
///
/// The root is wires rather than bytes, since the prover supplies it in the proof. The shape is
/// fixed by the protocol, so it stays concrete.
#[derive(Clone)]
pub struct Commitment {
	/// The commitment root, as `DIGEST_WORDS` wires.
	pub root: [Wire; DIGEST_WORDS],
	/// Field elements in each leaf.
	pub leaf_size: usize,
	/// Base-2 logarithm of the number of leaves.
	pub depth: usize,
}

/// A channel that records a verifier run as a Binius64 circuit.
///
/// Drive a verifier written against the channel traits with this in place of a transcript channel
/// and the result is a circuit rather than a verdict. See the crate docs for what is and is not
/// constrained yet.
///
/// The builder is shared through an [`Rc`] because [`IOPVerifierChannel`] requires
/// `Elem: 'static`, so an element cannot borrow the builder it was built on. The frontend's
/// `CircuitBuilder` takes `&self` throughout, so no interior-mutability wrapper is needed on top.
pub struct Binius64BuilderChannel {
	builder: Rc<CircuitBuilder>,
	/// Wires holding the proof, in the order the verifier reads them.
	transcript: Vec<Wire>,
	/// Wires holding the inner statement, which is the recursive circuit's own public input.
	inout: Vec<Wire>,
	oracle_specs: Vec<OracleSpec>,
	next_oracle_index: usize,
}

impl Binius64BuilderChannel {
	/// Creates a channel over a fresh builder, expecting the given oracles and an inner statement
	/// of `n_inout` words.
	///
	/// The statement's wires are allocated up front, so the caller can hand them to the verifier
	/// and read them back with [`Self::inout`].
	pub fn new(oracle_specs: Vec<OracleSpec>, n_inout: usize) -> Self {
		let builder = Rc::new(CircuitBuilder::new());
		let inout = (0..n_inout).map(|_| builder.add_inout()).collect();
		Self {
			builder,
			transcript: Vec::new(),
			inout,
			oracle_specs,
			next_oracle_index: 0,
		}
	}

	/// The inner statement, as this circuit's public input.
	pub fn statement(&self) -> Vec<Word> {
		self.inout
			.iter()
			.map(|&wire| Word::wire(&self.builder, wire))
			.collect()
	}

	/// The wires the inner statement occupies, in order.
	pub fn inout(&self) -> &[Wire] {
		&self.inout
	}

	/// The wires the proof is written into, in read order.
	pub fn transcript(&self) -> &[Wire] {
		&self.transcript
	}

	/// Consumes the channel and compiles the recorded circuit.
	///
	/// Every [`Elem`] and [`Word`] derived from this channel must be dropped first: they hold weak
	/// handles to the builder, and using one afterwards panics.
	pub fn build(self) -> Circuit {
		Rc::try_unwrap(self.builder)
			.unwrap_or_else(|_| panic!("Elem and Word values hold only weak handles"))
			.build()
	}

	/// Allocates a witness wire and records it as the next word of the proof.
	fn recv_word(&mut self) -> Wire {
		let wire = self.builder.add_witness();
		self.transcript.push(wire);
		wire
	}

	/// Allocates the `(lo, hi)` pair one received field element occupies.
	fn recv_elem(&mut self) -> Elem {
		let lo = self.recv_word();
		let hi = self.recv_word();
		Elem::wires(&self.builder, lo, hi)
	}
}

impl IPVerifierChannel<B128> for Binius64BuilderChannel {
	type Elem = Elem;

	fn recv_one(&mut self) -> Result<Elem, binius_ip::channel::Error> {
		Ok(self.recv_elem())
	}

	fn sample(&mut self) -> Elem {
		// UNCONSTRAINED: the challenge must be the Fiat-Shamir state's output, which means running
		// the challenger's SHA-256 over the observed transcript in-circuit. Until that lands the
		// challenge is a free witness, so a prover could choose it.
		let lo = self.builder.add_witness();
		let hi = self.builder.add_witness();
		Elem::wires(&self.builder, lo, hi)
	}

	fn observe_one(&mut self, _val: B128) -> Elem {
		// The statement reaches the channel through `observe_words`; this arm exists for protocols
		// that observe field elements directly, which the Binius64 verifier does not.
		todo!("observe a field element into the in-circuit Fiat-Shamir state")
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
				self.builder.assert_zero("channel", lo);
				self.builder.assert_zero("channel", hi);
				Ok(())
			}
		}
	}

	fn compute_public_value(&mut self, inputs: &[Elem], f: impl FieldFn<B128>) -> Elem {
		// Evaluated symbolically rather than hinted, so the circuit constrains it. The cost is the
		// caller's: for the constraint system's monster multilinear this is proportional to the
		// inner circuit's size, which is what an accumulation or Spark argument has to remove.
		f.call::<Elem>(inputs)
	}
}

impl WordIPVerifierChannel<B128> for Binius64BuilderChannel {
	type Word = Word;

	fn observe_words(&mut self, _words: &[Word]) {
		// UNCONSTRAINED: these words are the statement, and they must enter the Fiat-Shamir state
		// for the circuit to be bound to what it claims to verify.
		todo!("feed words into the in-circuit Fiat-Shamir state")
	}

	fn subset_sum(&mut self, elems: &[Elem], word: &Word) -> Elem {
		assert!(elems.len() <= binius_core::word::Word::BITS); // precondition

		// Each element is kept or dropped by its own bit, and the survivors are summed. The sum is
		// XOR, which the constraint system absorbs, so the cost is the keep-or-drop: one select
		// gate per word of each element.
		let (zero_lo, zero_hi) = Elem::zero().to_wires(&self.builder);
		(0..elems.len())
			.map(|bit| {
				let sel = word.bit_at_msb(&self.builder, bit);
				let (lo, hi) = elems[bit].to_wires(&self.builder);
				Elem::wires(
					&self.builder,
					self.builder.select(sel, lo, zero_lo),
					self.builder.select(sel, hi, zero_hi),
				)
			})
			.sum()
	}

	fn select(&mut self, elems: &[Elem], word: &Word) -> Elem {
		assert!(!elems.is_empty() && elems.len().is_power_of_two()); // precondition

		// One multiplexer over the `(lo, hi)` pairs. `multi_wire_multiplex` builds the same
		// select-gate tree per wire position and reads the index bits itself.
		let pairs = elems
			.iter()
			.map(|elem| {
				let (lo, hi) = elem.to_wires(&self.builder);
				[lo, hi]
			})
			.collect::<Vec<_>>();
		let groups = pairs.iter().map(|pair| pair.as_slice()).collect::<Vec<_>>();
		let sel = word.to_wire(&self.builder);
		let selected = multi_wire_multiplex(&self.builder, &groups, sel);
		Elem::wires(&self.builder, selected[0], selected[1])
	}

	fn sample_bits(&mut self, _bits: usize) -> Word {
		// UNCONSTRAINED, twice over: the index must come from the Fiat-Shamir state, and it must
		// be masked to `bits` bits, which the FRI code now relies on rather than asserting.
		Word::wire(&self.builder, self.builder.add_witness())
	}
}

impl MerkleIPVerifierChannel<B128> for Binius64BuilderChannel {
	type Commitment = Commitment;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Commitment, merkle_channel::Error> {
		let root = std::array::from_fn(|_| self.recv_word());
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
		// The opened values are read here so the wire order matches the proof, but nothing yet
		// binds them to the commitment.
		let values = (0..indices.len() * commitment.leaf_size)
			.map(|_| self.recv_elem())
			.collect();
		// UNCONSTRAINED: hash each leaf, climb the path ordering pairs by the index bits, and check
		// the result against the layer digests and the root.
		let _ = commitment.root;
		Ok(values)
	}

	fn recv_committed_vector(
		&mut self,
		commitment: &Commitment,
	) -> Result<Vec<Elem>, merkle_channel::Error> {
		let values = (0..commitment.leaf_size << commitment.depth)
			.map(|_| self.recv_elem())
			.collect();
		// UNCONSTRAINED: rebuild the tree over the whole vector and check it against the root.
		Ok(values)
	}
}

impl IOPVerifierChannel<B128> for Binius64BuilderChannel {
	type Oracle = usize;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn recv_oracle(
		&mut self,
		_log_msg_len: usize,
		_is_witness_dependent: bool,
	) -> Result<usize, binius_iop::channel::Error> {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining oracle specs"
		);
		let index = self.next_oracle_index;
		self.next_oracle_index += 1;
		Ok(index)
	}

	fn verify_oracle_relations(
		&mut self,
		_oracle_relations: impl IntoIterator<Item = OracleLinearRelation<usize, Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		// The opening runs through `BaseFoldVerifierChannel`, which drives this channel's Merkle
		// methods. Wiring that up is what turns the pieces above into a verifier.
		todo!("hand the queued relations to the BaseFold opening")
	}
}
