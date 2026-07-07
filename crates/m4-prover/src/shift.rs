// Copyright 2026 The Binius Developers

//! The batched shift-reduction prover for the data-parallel Binius64 M4 proof system.

#[cfg(test)]
mod tests {
	use binius_core::{verify::verify_constraints, word::Word};
	use binius_frontend::{Circuit, CircuitBuilder, Wire};
	use rand::prelude::*;

	use crate::ValueTable;

	/// CRC-64/GO-ISO parameters.
	///
	/// The generator polynomial is `x^64 + x^4 + x^3 + x + 1`, normal form `0x1b`. Both input and
	/// output are reflected, so the polynomial enters the register in its bit-reversed form.
	const POLY_REFLECTED: u64 = 0xd800000000000000;
	/// The register is preset to all ones before absorbing the message.
	const INIT: u64 = 0xffffffffffffffff;
	/// The final register is XORed with all ones before being returned.
	const XOR_OUT: u64 = 0xffffffffffffffff;

	/// The number of 64-bit input words the CRC circuit consumes.
	const N_INPUT_WORDS: usize = 4;

	/// Computes CRC-64/GO-ISO over `words`, absorbing bits least-significant-first.
	///
	/// Each input word contributes its 64 bits in order from bit 0 up to bit 63, and the words are
	/// absorbed in index order. This is the reflected bitwise algorithm: for every message bit, the
	/// register's low bit is combined with the message bit, the register is shifted right by one,
	/// and the polynomial is conditionally mixed in.
	///
	/// The `Circuit` counterpart mirrors this loop gate for gate, so the two agree bit for bit.
	fn crc64_iso_reference(words: &[u64; N_INPUT_WORDS]) -> u64 {
		let mut crc = INIT;
		for &word in words {
			for i in 0..64 {
				let bit = (word >> i) & 1;
				let mix = (crc ^ bit) & 1;
				crc >>= 1;
				if mix != 0 {
					crc ^= POLY_REFLECTED;
				}
			}
		}
		crc ^ XOR_OUT
	}

	/// A circuit computing CRC-64/GO-ISO over four private witness words.
	///
	/// The four inputs are ordinary witness wires, not public inout wires, so the whole computation
	/// lives in the private witness. The output wire is force-committed: without an assertion or a
	/// public output reading it, dead-code elimination would otherwise prune the entire CRC.
	struct Crc64Circuit {
		circuit: Circuit,
		input: [Wire; N_INPUT_WORDS],
		output: Wire,
	}

	/// Builds the CRC-64/GO-ISO circuit, mirroring [`crc64_iso_reference`] gate for gate.
	fn crc64_circuit() -> Crc64Circuit {
		let builder = CircuitBuilder::new();

		// The four message words are private witnesses supplied by the prover.
		let input = std::array::from_fn(|_| builder.add_witness());

		// The register starts at the all-ones preset and the polynomial is a constant.
		let mut crc = builder.add_constant_64(INIT);
		let poly = builder.add_constant_64(POLY_REFLECTED);

		for word in input {
			for i in 0..64 {
				// Isolate message bit `i` into the low bit; the higher bits are junk we discard.
				let bit = if i == 0 { word } else { builder.shr(word, i) };

				// The low bit that decides whether the polynomial is mixed in this step.
				let mixed = builder.bxor(crc, bit);

				// Broadcast that low bit across the whole word: all ones iff it is set, else zero.
				// Shifting it up to bit 63 then arithmetic-shifting back fills every bit from it.
				let to_msb = builder.shl(mixed, 63);
				let mask = builder.sar(to_msb, 63);
				let poly_term = builder.band(mask, poly);

				// Advance the register: shift right by one, then conditionally mix the polynomial.
				let shifted = builder.shr(crc, 1);
				crc = builder.bxor(shifted, poly_term);
			}
		}

		// Apply the final output XOR to produce the CRC value.
		let output = builder.bxor(crc, builder.add_constant_64(XOR_OUT));

		// Pin the output so the constraint compiler keeps the CRC computation alive.
		builder.force_commit(output);

		Crc64Circuit {
			circuit: builder.build(),
			input,
			output,
		}
	}

	#[test]
	fn circuit_matches_reference() {
		let c = crc64_circuit();

		// A handful of fixed inputs, checked against the standalone reference implementation.
		let cases: [[u64; N_INPUT_WORDS]; 3] = [
			[0, 0, 0, 0],
			[1, 2, 3, 4],
			[
				0x0123456789abcdef,
				0xfedcba9876543210,
				0xdeadbeefcafebabe,
				0x00ff00ff00ff00ff,
			],
		];

		for words in cases {
			let mut filler = c.circuit.new_witness_filler();
			for (wire, &w) in c.input.iter().zip(&words) {
				filler[*wire] = Word(w);
			}
			c.circuit.populate_wire_witness(&mut filler).unwrap();

			assert_eq!(filler[c.output], Word(crc64_iso_reference(&words)));
		}
	}

	#[test]
	fn populate_batch_of_random_instances() {
		let c = crc64_circuit();

		// A batch of 2^10 instances, each with an independent random message.
		let log_instances = 10;
		let n_instances = 1usize << log_instances;

		// Sample every instance's inputs up front so the fill closure is a pure lookup and the
		// reference check below sees the same words.
		let mut rng = StdRng::seed_from_u64(0);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| std::array::from_fn(|_| rng.random()))
			.collect();

		let table = ValueTable::populate(&c.circuit, log_instances, |i, filler| {
			for (wire, &w) in c.input.iter().zip(&inputs[i]) {
				filler[*wire] = Word(w);
			}
		})
		.unwrap();

		// Shape: 2^10 instances, one committed witness per instance.
		let stride = c
			.circuit
			.constraint_system()
			.value_vec_layout
			.combined_len();
		assert_eq!(table.log_instances(), log_instances);
		assert_eq!(table.n_instances(), n_instances);
		assert_eq!(table.instance_stride(), stride);

		// Spot-check a few instances: each reconstructs to a valid single-instance witness whose
		// output word is the reference CRC of its inputs.
		let output_index = c.circuit.witness_index(c.output);
		for i in [0, 1, 42, n_instances - 1] {
			let vv = table.instance_value_vec(i);
			verify_constraints(c.circuit.constraint_system(), &vv)
				.unwrap_or_else(|e| panic!("instance {i} failed verification: {e}"));
			assert_eq!(vv[output_index], Word(crc64_iso_reference(&inputs[i])));
		}
	}
}
