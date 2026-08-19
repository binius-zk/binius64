// Copyright 2025 Irreducible Inc.
use std::{array, iter};

use binius_core::word::Word;
use binius_frontend::{ChipGadget, CircuitBuilder, Hint, Wire};

// ι round constants
pub const RC: [u64; 24] = [
	0x0000_0000_0000_0001,
	0x0000_0000_0000_8082,
	0x8000_0000_0000_808A,
	0x8000_0000_8000_8000,
	0x0000_0000_0000_808B,
	0x0000_0000_8000_0001,
	0x8000_0000_8000_8081,
	0x8000_0000_0000_8009,
	0x0000_0000_0000_008A,
	0x0000_0000_0000_0088,
	0x0000_0000_8000_8009,
	0x0000_0000_8000_000A,
	0x0000_0000_8000_808B,
	0x8000_0000_0000_008B,
	0x8000_0000_0000_8089,
	0x8000_0000_0000_8003,
	0x8000_0000_0000_8002,
	0x8000_0000_0000_0080,
	0x0000_0000_0000_800A,
	0x8000_0000_8000_000A,
	0x8000_0000_8000_8081,
	0x8000_0000_0000_8080,
	0x0000_0000_8000_0001,
	0x8000_0000_8000_8008,
];

// ρ rotation offsets r[x,y] in lane order (i = x + 5*y)
#[rustfmt::skip]
pub const R: [u32; 25] =  [
	 0,  1, 62, 28, 27,
	36, 44,  6, 55, 20,
	 3, 10, 43, 25, 39,
	41, 45, 15, 21,  8,
	18,  2, 61, 56, 14,
];

#[inline(always)]
pub const fn idx(x: usize, y: usize) -> usize {
	x + 5 * y
}

/// Perform the Keccak f\[1600\] permutation in place on a 25-lane state.
///
/// # Arguments
///
/// * `b` - The circuit builder to use.
/// * `state` - The 25-word state to permute.
///
/// # Chips
///
/// This is a [`ChipGadget`]. A circuit that calls
/// [`register_chip`](CircuitBuilder::register_chip) with [`KeccakF1600`] before building turns
/// every permutation under it into a chip call, including the ones
/// [`keccak256_varlen`](super::keccak256_varlen) and
/// [`fixed_length::keccak256`](super::fixed_length::keccak256) reach.
pub fn keccak_f1600(b: &CircuitBuilder, state: &mut [Wire; 25]) {
	let outputs = b.build_gadget(KeccakF1600, &[], state);
	state.copy_from_slice(&outputs);
}

/// [`keccak_f1600`] as a gadget, so that a circuit can make it a chip.
///
/// Its interface is the 25 input lanes and the 25 output lanes, both in lane order `x + 5 * y`.
/// Every lane is a full 64-bit word; there is no lane packing.
pub struct KeccakF1600;

impl Hint for KeccakF1600 {
	const NAME: &'static str = "binius.keccak_f1600";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(25, 25)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let mut state: [u64; 25] = array::from_fn(|i| inputs[i].as_u64());
		ref_keccak_f1600(&mut state);

		for (slot, lane) in iter::zip(outputs, state) {
			*slot = Word(lane);
		}
	}
}

impl ChipGadget for KeccakF1600 {
	fn build(&self, builder: &CircuitBuilder, _dimensions: &[usize], inputs: &[Wire]) -> Vec<Wire> {
		let mut state: [Wire; 25] = array::from_fn(|i| inputs[i]);
		f1600_gates(builder, &mut state);
		state.to_vec()
	}
}

/// [`keccak_f1600`] in gates, whatever the building circuit does with the gadget.
fn f1600_gates(b: &CircuitBuilder, state: &mut [Wire; 25]) {
	for round in 0..24 {
		keccak_permutation_round(b, state, round);
	}
}

pub fn keccak_permutation_round(b: &CircuitBuilder, state: &mut [Wire; 25], round: usize) {
	theta(b, state);
	rho_pi(b, state);
	chi_iota(b, state, round);
}

fn theta(b: &CircuitBuilder, state: &mut [Wire; 25]) {
	let c0 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(0, y)]));
	let c1 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(1, y)]));
	let c2 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(2, y)]));
	let c3 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(3, y)]));
	let c4 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(4, y)]));

	// D[x] = C[x-1] ^ rotl1(C[x+1])
	let d0 = b.bxor(c4, b.rotl(c1, 1));
	let d1 = b.bxor(c0, b.rotl(c2, 1));
	let d2 = b.bxor(c1, b.rotl(c3, 1));
	let d3 = b.bxor(c2, b.rotl(c4, 1));
	let d4 = b.bxor(c3, b.rotl(c0, 1));

	// A'[x,y] = A[x,y] ^ D[x]
	for y in 0..5 {
		state[idx(0, y)] = b.bxor(state[idx(0, y)], d0);
		state[idx(1, y)] = b.bxor(state[idx(1, y)], d1);
		state[idx(2, y)] = b.bxor(state[idx(2, y)], d2);
		state[idx(3, y)] = b.bxor(state[idx(3, y)], d3);
		state[idx(4, y)] = b.bxor(state[idx(4, y)], d4);
	}
}

fn chi_iota(b: &CircuitBuilder, state: &mut [Wire; 25], round: usize) {
	let rc = b.add_constant(Word(RC[round]));

	for y in 0..5 {
		let a0 = state[idx(0, y)];
		let a1 = state[idx(1, y)];
		let a2 = state[idx(2, y)];
		let a3 = state[idx(3, y)];
		let a4 = state[idx(4, y)];

		// For the state element at (0, 0), the output state has the round constant accumulated in.
		// For circuit efficiency, we do this as part of the fused AND-XOR.
		let a0_iota = if y == 0 { b.bxor(a0, rc) } else { a0 };

		state[idx(0, y)] = b.fax(b.bnot(a1), a2, a0_iota);
		state[idx(1, y)] = b.fax(b.bnot(a2), a3, a1);
		state[idx(2, y)] = b.fax(b.bnot(a3), a4, a2);
		state[idx(3, y)] = b.fax(b.bnot(a4), a0, a3);
		state[idx(4, y)] = b.fax(b.bnot(a0), a1, a4);
	}
}

fn rho_pi(b: &CircuitBuilder, state: &mut [Wire; 25]) {
	let mut temp = [state[0]; 25];
	for y in 0..5 {
		for x in 0..5 {
			// no need to rotate if rotating by 0
			if R[idx(x, y)] == 0 {
				continue;
			}
			temp[idx(y, (2 * x + 3 * y) % 5)] = b.rotl(state[idx(x, y)], R[idx(x, y)]);
		}
	}
	*state = temp;
}

// --- Pure-Rust reference implementation of Keccak-f[1600] ---------------------------
//
// Shared by [`KeccakF1600`] (prover-side witness generation) and the tests. One function per
// circuit step, so that each step can be checked against its gates alone.

fn ref_theta(state: &mut [u64; 25]) {
	let mut c = [0u64; 5];
	for x in 0..5 {
		c[x] = state[idx(x, 0)]
			^ state[idx(x, 1)]
			^ state[idx(x, 2)]
			^ state[idx(x, 3)]
			^ state[idx(x, 4)];
	}
	let d = [
		c[4] ^ c[1].rotate_left(1),
		c[0] ^ c[2].rotate_left(1),
		c[1] ^ c[3].rotate_left(1),
		c[2] ^ c[4].rotate_left(1),
		c[3] ^ c[0].rotate_left(1),
	];

	for y in 0..5 {
		for x in 0..5 {
			state[idx(x, y)] ^= d[x];
		}
	}
}

fn ref_rho_pi(state: &mut [u64; 25]) {
	let mut temp = [state[0]; 25];
	for y in 0..5 {
		for x in 0..5 {
			temp[idx(y, (2 * x + 3 * y) % 5)] = state[idx(x, y)].rotate_left(R[idx(x, y)]);
		}
	}
	*state = temp;
}

fn ref_chi(state: &mut [u64; 25]) {
	for y in 0..5 {
		let a0 = state[idx(0, y)];
		let a1 = state[idx(1, y)];
		let a2 = state[idx(2, y)];
		let a3 = state[idx(3, y)];
		let a4 = state[idx(4, y)];
		state[idx(0, y)] = a0 ^ ((!a1) & a2);
		state[idx(1, y)] = a1 ^ ((!a2) & a3);
		state[idx(2, y)] = a2 ^ ((!a3) & a4);
		state[idx(3, y)] = a3 ^ ((!a4) & a0);
		state[idx(4, y)] = a4 ^ ((!a0) & a1);
	}
}

const fn ref_iota(state: &mut [u64; 25], round: usize) {
	state[0] ^= RC[round];
}

fn ref_keccak_permutation_round(state: &mut [u64; 25], round: usize) {
	ref_theta(state);
	ref_rho_pi(state);
	ref_chi(state);
	ref_iota(state, round);
}

/// Pure-Rust Keccak-f\[1600\] permutation of a 25-lane state, matching the in-circuit
/// permutation exactly.
///
/// Used for prover-side witness generation and as the test reference.
pub fn ref_keccak_f1600(state: &mut [u64; 25]) {
	for round in 0..24 {
		ref_keccak_permutation_round(state, round);
	}
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;
	use binius_frontend::CircuitBuilder;
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	fn validate_circuit_component(
		circuit_fn: impl FnOnce(&CircuitBuilder, &mut [Wire; 25]),
		reference_fn: impl FnOnce(&mut [u64; 25]),
		input_state: [u64; 25],
	) {
		let builder = CircuitBuilder::new();

		let input_wires: [Wire; 25] = std::array::from_fn(|_| builder.add_inout());

		let mut state_wires = input_wires;
		circuit_fn(&builder, &mut state_wires);
		// The test reads these directly, so pin them or pooling could reclaim their slots first.
		for &wire in &state_wires {
			builder.force_commit(wire);
		}
		let circuit = builder.build();

		let mut expected_output = input_state;
		reference_fn(&mut expected_output);

		let mut w = circuit.new_witness_filler();
		for i in 0..25 {
			w[input_wires[i]] = Word(input_state[i]);
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		for i in 0..25 {
			assert_eq!(
				w[state_wires[i]],
				Word(expected_output[i]),
				"Output mismatch at index {}: circuit={:?}, expected={:?}",
				i,
				w[state_wires[i]],
				Word(expected_output[i])
			);
		}

		let cs = circuit.constraint_system();
		cs.verify(&w.into_value_vec()).unwrap();
	}

	#[test]
	fn test_keccak_f1600() {
		let mut rng = StdRng::seed_from_u64(0);
		let input_state = rng.random::<[u64; 25]>();

		validate_circuit_component(keccak_f1600, ref_keccak_f1600, input_state);
	}

	#[test]
	fn test_keccak_permutation_round() {
		let mut rng = StdRng::seed_from_u64(0);
		let input_state = rng.random::<[u64; 25]>();

		validate_circuit_component(
			|b, state| keccak_permutation_round(b, state, 0),
			|state| ref_keccak_permutation_round(state, 0),
			input_state,
		);
	}

	#[test]
	fn test_theta() {
		let mut rng = StdRng::seed_from_u64(0);
		let input_state = rng.random::<[u64; 25]>();

		validate_circuit_component(theta, ref_theta, input_state);
	}

	#[test]
	fn test_rho_pi() {
		let mut rng = StdRng::seed_from_u64(0);
		let input_state = rng.random::<[u64; 25]>();

		validate_circuit_component(rho_pi, ref_rho_pi, input_state);
	}

	#[test]
	fn test_chi_iota() {
		let mut rng = StdRng::seed_from_u64(0);
		let input_state = rng.random::<[u64; 25]>();

		// The circuit fuses χ and ι into one pass, so it is checked against the two reference
		// steps run back to back. Round 6 has a round constant with bits spread across all four
		// 16-bit halves, so a misplaced constant shows up in the output.
		const ROUND: usize = 6;

		validate_circuit_component(
			|b, state| chi_iota(b, state, ROUND),
			|state| {
				ref_chi(state);
				ref_iota(state, ROUND);
			},
			input_state,
		);
	}

	/// Runs [`keccak_f1600`] in gates over its flat lane interface.
	fn run_f1600_words(inputs: [u64; 25]) -> [u64; 25] {
		let builder = CircuitBuilder::new();
		let wires: [Wire; 25] = std::array::from_fn(|_| builder.add_witness());
		let mut out = wires;
		f1600_gates(&builder, &mut out);
		for wire in out {
			builder.mark_inout(wire);
		}

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		for (wire, word) in std::iter::zip(wires, inputs) {
			w[wire] = Word(word);
		}
		circuit.populate_wire_witness(&mut w).unwrap();

		std::array::from_fn(|i| w[out[i]].as_u64())
	}

	proptest! {
		// The chip path takes the outputs from `KeccakF1600::execute` and the chip recomputes them
		// from its gates, so the two have to agree on every word a circuit can reach them with.
		// Every lane is a full 64-bit value, so random words cover the whole interface.
		#[test]
		fn f1600_hint_matches_its_gates(words in prop::collection::vec(any::<u64>(), 25)) {
			let inputs: [u64; 25] = std::array::from_fn(|i| words[i]);

			let mut hinted = [Word::ZERO; 25];
			KeccakF1600.execute(&[], &inputs.map(Word), &mut hinted);

			prop_assert_eq!(hinted.map(|word| word.as_u64()), run_f1600_words(inputs));
		}
	}
}
