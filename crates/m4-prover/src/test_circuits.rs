// Copyright 2026 The Binius Developers
//! Single-instance circuits over hash primitives and integer multiplication.
//!
//! The proving benchmarks and the proving timing tests both prove batches of these, so the circuits
//! live here rather than in either. Each is generic over the number `N` of independent primitives
//! one instance computes: a larger `N` fills the value vector more densely, so a smaller share of
//! the committed trace is padding.

use std::array;

use binius_circuits::{
	blake3::blake3_compress_2x,
	keccak::permutation::keccak_f1600,
	sha256::{State, sha256_compress_2x},
};
use binius_core::word::Word;
use binius_frontend::{BatchWitnessFiller, Circuit, CircuitBuilder, Wire};
use rand::prelude::*;

/// The number of 64-bit lanes in a Keccak-f1600 state.
const KECCAK_STATE_LANES: usize = 25;

/// A single-instance circuit over some primitive, together with a filler for its witness inputs.
pub trait TestCircuit {
	/// The number of primitives one instance computes.
	///
	/// This is a throughput count only: a batch of `2^n` instances proves
	/// `2^n * PRIMITIVES_PER_INSTANCE` primitives.
	const PRIMITIVES_PER_INSTANCE: u64;

	/// Builds the circuit.
	fn new() -> Self;

	/// The built single-instance circuit, to be replicated across a batch.
	fn circuit(&self) -> &Circuit;

	/// Assigns instance `instance`'s witness inputs, setting every one of them.
	fn fill(&self, instance: usize, w: &mut BatchWitnessFiller<'_, '_>);
}

/// Packs two independent 32-bit lane values into one 64-bit word.
const fn pack_lanes(lane0: u32, lane1: u32) -> Word {
	Word((lane0 as u64) | ((lane1 as u64) << 32))
}

/// The public wires of one two-lane BLAKE3 compression.
///
/// Each wire packs two independent compressions: lane 0 in the low 32 bits, lane 1 in the high 32.
struct Blake3Wires {
	/// The 8-word input chaining value, two lanes per word.
	cv: [Wire; 8],
	/// The 16-word message block, two lanes per word.
	block: [Wire; 16],
	/// The low 32 bits of the 64-bit block counter, two lanes.
	counter_lo: Wire,
	/// The high 32 bits of the 64-bit block counter, two lanes.
	counter_hi: Wire,
	/// The block length in bytes, two lanes.
	block_len: Wire,
	/// The domain-separation flags, two lanes.
	flags: Wire,
}

/// A circuit of `N` independent two-lane BLAKE3 compressions.
///
/// Each two-lane compression is itself two independent compressions, so one instance computes
/// `2 * N` of them. Reusing the two-lane core matches the efficient representation the signature
/// circuits use.
pub struct Blake3Circuit<const N: usize> {
	circuit: Circuit,
	wires: [Blake3Wires; N],
}

impl<const N: usize> TestCircuit for Blake3Circuit<N> {
	const PRIMITIVES_PER_INSTANCE: u64 = 2 * N as u64;

	fn new() -> Self {
		let builder = CircuitBuilder::new();

		// Every compression's inputs are public, filled per instance.
		let wires: [Blake3Wires; N] = array::from_fn(|_| Blake3Wires {
			cv: array::from_fn(|_| builder.add_inout()),
			block: array::from_fn(|_| builder.add_inout()),
			counter_lo: builder.add_inout(),
			counter_hi: builder.add_inout(),
			block_len: builder.add_inout(),
			flags: builder.add_inout(),
		});

		for &Blake3Wires {
			cv,
			block,
			counter_lo,
			counter_hi,
			block_len,
			flags,
		} in &wires
		{
			// Promote each output chaining-value word to a public output.
			// That promotion keeps the compression alive under dead-code elimination.
			let out =
				blake3_compress_2x(&builder, cv, block, counter_lo, counter_hi, block_len, flags);
			for wire in out {
				builder.mark_inout(wire);
			}
		}

		Self {
			circuit: builder.build(),
			wires,
		}
	}

	fn circuit(&self) -> &Circuit {
		&self.circuit
	}

	/// Assigns one instance's two-lane inputs from an instance-seeded RNG.
	///
	/// The compression derives its output from these inputs, so any assignment is valid. Seeding
	/// per instance keeps the data non-degenerate and reproducible.
	fn fill(&self, instance: usize, w: &mut BatchWitnessFiller<'_, '_>) {
		let mut rng = StdRng::seed_from_u64(instance as u64);

		for wires in &self.wires {
			// A 32-bit value per chaining-value word, per lane.
			for wire in wires.cv {
				w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
			}
			// A 32-bit value per message word, per lane.
			for wire in wires.block {
				w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
			}
			// The 64-bit counter, split into low and high halves, per lane.
			w[wires.counter_lo] = pack_lanes(rng.next_u32(), rng.next_u32());
			w[wires.counter_hi] = pack_lanes(rng.next_u32(), rng.next_u32());
			// A byte length in 0..=64, per lane.
			w[wires.block_len] = pack_lanes(rng.next_u32() % 65, rng.next_u32() % 65);
			// Arbitrary domain-separation flags, per lane.
			w[wires.flags] = pack_lanes(rng.next_u32(), rng.next_u32());
		}
	}
}

/// The public wires of one two-lane SHA-256 compression.
///
/// Each wire packs two independent compressions: lane 0 in the low 32 bits, lane 1 in the high 32.
struct Sha256Wires {
	/// The 8-word input chaining value, two lanes per word.
	state: [Wire; 8],
	/// The 16-word message block, two lanes per word.
	block: [Wire; 16],
}

/// A circuit of `N` independent two-lane SHA-256 compressions.
///
/// Each two-lane compression is itself two independent compressions, so one instance computes
/// `2 * N` of them.
pub struct Sha256Circuit<const N: usize> {
	circuit: Circuit,
	wires: [Sha256Wires; N],
}

impl<const N: usize> TestCircuit for Sha256Circuit<N> {
	const PRIMITIVES_PER_INSTANCE: u64 = 2 * N as u64;

	fn new() -> Self {
		let builder = CircuitBuilder::new();

		// Every compression's inputs are public, filled per instance.
		let wires: [Sha256Wires; N] = array::from_fn(|_| Sha256Wires {
			state: array::from_fn(|_| builder.add_inout()),
			block: array::from_fn(|_| builder.add_inout()),
		});

		for &Sha256Wires { state, block } in &wires {
			// Promote each output state word to a public output.
			// That promotion keeps the compression alive under dead-code elimination.
			let out = sha256_compress_2x(&builder, State::new(state), block);
			for wire in out.0 {
				builder.mark_inout(wire);
			}
		}

		Self {
			circuit: builder.build(),
			wires,
		}
	}

	fn circuit(&self) -> &Circuit {
		&self.circuit
	}

	/// Assigns one instance's two-lane inputs from an instance-seeded RNG.
	///
	/// The compression derives its output from these inputs, so any assignment is valid. Seeding
	/// per instance keeps the data non-degenerate and reproducible.
	fn fill(&self, instance: usize, w: &mut BatchWitnessFiller<'_, '_>) {
		let mut rng = StdRng::seed_from_u64(instance as u64);

		for wires in &self.wires {
			// A 32-bit value per chaining-value word, per lane.
			for wire in wires.state {
				w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
			}
			// A 32-bit value per message word, per lane.
			for wire in wires.block {
				w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
			}
		}
	}
}

/// A circuit of `N` independent Keccak-f1600 permutations.
///
/// The permutations share no wires, so one instance computes `N` of them.
pub struct KeccakCircuit<const N: usize> {
	circuit: Circuit,
	/// The input state lanes of each permutation.
	states: [[Wire; KECCAK_STATE_LANES]; N],
}

impl<const N: usize> TestCircuit for KeccakCircuit<N> {
	const PRIMITIVES_PER_INSTANCE: u64 = N as u64;

	fn new() -> Self {
		let builder = CircuitBuilder::new();

		// Each permutation gets its own 25-lane public input state.
		let states: [[Wire; KECCAK_STATE_LANES]; N] =
			array::from_fn(|_| array::from_fn(|_| builder.add_inout()));

		for input in states {
			// Permute a copy of the input in place.
			// After the call, `state` holds the output lanes.
			let mut state = input;
			keccak_f1600(&builder, &mut state);

			// Promote the permuted lanes to public outputs.
			// That promotion keeps the permutation alive under dead-code elimination.
			for wire in state {
				builder.mark_inout(wire);
			}
		}

		Self {
			circuit: builder.build(),
			states,
		}
	}

	fn circuit(&self) -> &Circuit {
		&self.circuit
	}

	/// Assigns every input state lane a random 64-bit word from an instance-seeded RNG.
	///
	/// Keccak's circuit shape and proving cost are data-independent, so these only need to be
	/// non-degenerate and reproducible.
	fn fill(&self, instance: usize, w: &mut BatchWitnessFiller<'_, '_>) {
		let mut rng = StdRng::seed_from_u64(instance as u64);

		for &wire in self.states.iter().flatten() {
			w[wire] = Word(rng.next_u64());
		}
	}
}

/// A circuit of one 64×64→128-bit integer multiplication.
///
/// Unlike the hash primitives, which are purely bitwise and carry-adder based, this circuit has
/// IMUL constraints, so its M4 proof commits the extra IntMul logup pushforward oracle.
pub struct ImulCircuit {
	circuit: Circuit,
	/// The two 64-bit factors.
	factors: [Wire; 2],
}

impl TestCircuit for ImulCircuit {
	const PRIMITIVES_PER_INSTANCE: u64 = 1;

	fn new() -> Self {
		let builder = CircuitBuilder::new();

		// Two 64-bit public factors.
		let factors = [builder.add_inout(), builder.add_inout()];

		// Promoting the product halves keeps the multiplication alive under dead-code elimination.
		let (hi, lo) = builder.imul(factors[0], factors[1]);
		builder.mark_inout(hi);
		builder.mark_inout(lo);

		Self {
			circuit: builder.build(),
			factors,
		}
	}

	fn circuit(&self) -> &Circuit {
		&self.circuit
	}

	/// Assigns the two factors random 64-bit words from an instance-seeded RNG.
	///
	/// The product is derived from these inputs, so any assignment is valid.
	fn fill(&self, instance: usize, w: &mut BatchWitnessFiller<'_, '_>) {
		let mut rng = StdRng::seed_from_u64(instance as u64);

		for &wire in &self.factors {
			w[wire] = Word(rng.next_u64());
		}
	}
}
