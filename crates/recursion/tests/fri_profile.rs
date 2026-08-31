// Copyright 2026 The Binius Developers

//! What the inner proof's rate costs the circuit that verifies it.
//!
//! A recursive circuit replays a verifier, and most of that replay is per query: one Merkle path
//! climbed and hashed, one coset picked out of a decommitted layer.
//!
//! So the circuit's size tracks the query count, and the rate is what sets the query count.
//!
//! ```text
//!     n_q = ceil(security_bits / -log2((1 + rate) / 2))
//! ```
//!
//! At 96 bits that is 232 queries at rate 1/2 and 106 at rate 1/16.
//!
//! Lowering the rate pulls two ways at once, which is why this measures rather than argues.
//!
//! ```text
//!     fewer queries    ->  fewer paths to climb
//!     longer codeword  ->  each path one level deeper, and eventually one more oracle to open
//! ```
//!
//! Every point below is priced at one security level, with the query count derived from it.
//! So a lower rate buys fewer queries rather than less soundness.
//!
//! Only the rate moves here.
//! The fold arities under it are still chosen to minimize proof size.
//! So these numbers are a floor on what a schedule costed in constraints would find.

use binius_core::{constraint_system::ValueVec, word::Word};
use binius_field::arch::OptimalPackedB128;
use binius_frontend::{Circuit, CircuitBuilder, CircuitStat, Wire};
use binius_hash::StdHashSuite;
use binius_prover::Prover;
use binius_recursion::{Discharge, RecursiveCircuit};
use binius_transcript::ProverTranscript;
use binius_verifier::{
	SECURITY_BITS, Verifier, config::StdChallenger, fri::calculate_n_test_queries,
};

/// How many rates one sweep covers.
const N_RATES: usize = 4;

/// The rates swept, as `log2(1 / rate)`.
///
/// One is what every proof in the tree is built at today.
/// Four is past the point where the query count stops paying for the longer codeword.
const LOG_INV_RATES: [usize; N_RATES] = [1, 2, 3, 4];

// CRC-64/GO-ISO, reflected.
const POLY_REFLECTED: u64 = 0xd800000000000000;
const INIT: u64 = 0xffffffffffffffff;
const XOR_OUT: u64 = 0xffffffffffffffff;

/// A CRC-64 circuit over a message of the given length, with its message wires.
fn crc64_circuit(n_words: usize) -> (Circuit, Vec<Wire>) {
	let builder = CircuitBuilder::new();
	let input = (0..n_words)
		.map(|_| builder.add_inout())
		.collect::<Vec<_>>();

	// One round per message bit: mix the bit into the low end, then divide by the polynomial.
	let mut crc = builder.add_constant_64(INIT);
	let poly = builder.add_constant_64(POLY_REFLECTED);
	for &word in &input {
		for i in 0..64 {
			let bit = if i == 0 { word } else { builder.shr(word, i) };
			let mixed = builder.bxor(crc, bit);
			// Broadcast the low bit across the word: all ones iff it is set.
			let to_msb = builder.shl(mixed, 63);
			let mask = builder.sar(to_msb, 63);
			let poly_term = builder.band(mask, poly);
			let shifted = builder.shr(crc, 1);
			crc = builder.bxor(shifted, poly_term);
		}
	}
	let xor_out = builder.add_constant_64(XOR_OUT);
	let output = builder.bxor(crc, xor_out);
	builder.mark_inout(output);

	(builder.build(), input)
}

/// Fills the witness for a message of counting words.
fn crc64_witness(circuit: &Circuit, input: &[Wire]) -> ValueVec {
	let mut filler = circuit.new_witness_filler();
	for (i, &wire) in input.iter().enumerate() {
		filler[wire] = Word::from_u64(i as u64 + 1);
	}
	circuit.populate_wire_witness(&mut filler).unwrap();
	filler.into_value_vec()
}

/// One sweep point: the inner proof's shape, and what verifying it costs in constraints.
struct Priced {
	/// Queries the rate asks for, which is the multiplier on the whole per-query replay.
	n_queries: usize,
	/// Oracles a query opens, each one a Merkle path of its own.
	n_oracles: usize,
	/// Bits a query index spans, so the depth the deepest path climbs from.
	index_bits: usize,
	/// Bytes of a real inner transcript.
	proof_bytes: usize,
	/// AND constraints, which is where the circuit's hashing lands.
	and: usize,
	/// BMUL constraints, one per field multiplication and per selection.
	bmul: usize,
	/// ZERO constraints, which is where its assertions land.
	zero: usize,
}

impl Priced {
	/// Constraints in all three columns, which is the row count the prover pays for.
	const fn rows(&self) -> usize {
		self.and + self.bmul + self.zero
	}
}

/// Proves one CRC-64 statement at the given rate, and prices the circuit that verifies it.
fn price(n_words: usize, log_inv_rate: usize) -> Priced {
	let (circuit, input) = crc64_circuit(n_words);
	let witness = crc64_witness(&circuit, &input);

	let verifier =
		Verifier::<StdHashSuite>::setup(circuit.constraint_system().clone(), log_inv_rate).unwrap();
	let prover = Prover::<OptimalPackedB128, StdHashSuite>::setup(verifier.clone()).unwrap();
	let mut transcript = ProverTranscript::new(StdChallenger::default());
	prover.prove(&witness, &mut transcript).unwrap();
	let proof_bytes = transcript.finalize().len();

	// The claim is deferred, since that is the shape a node in a tower runs.
	// Discharging it in-circuit would add a term tracking the inner size and drown the rate.
	let fri = verifier
		.fri_params()
		.expect("the sweep runs the FRI scheme");
	let (n_oracles, index_bits) = (fri.n_oracles(), fri.index_bits());
	let recursive = RecursiveCircuit::build_with(verifier, Discharge::Deferred).unwrap();
	let stat = CircuitStat::collect(recursive.circuit());

	Priced {
		n_queries: calculate_n_test_queries(SECURITY_BITS, log_inv_rate),
		n_oracles,
		index_bits,
		proof_bytes,
		and: stat.n_and_constraints,
		bmul: stat.n_bmul_constraints,
		zero: stat.n_zero_constraints,
	}
}

/// Sweeps every rate at each given inner size, printing the surface and returning it.
fn cost_surface(sizes: &[usize]) -> Vec<[Priced; N_RATES]> {
	println!(
		"\n{:>6} {:>5} {:>8} {:>8} {:>6} {:>9} {:>10} {:>9} {:>10} {:>7}",
		"words", "rate", "queries", "oracles", "index", "bytes", "AND", "BMUL", "rows", "vs 1/2"
	);
	sizes
		.iter()
		.map(|&n_words| {
			let row = LOG_INV_RATES.map(|log_inv_rate| price(n_words, log_inv_rate));
			// The native rate leads every row, so it is what the last column is read against.
			let native = row[0].rows() as f64;
			for (log_inv_rate, priced) in LOG_INV_RATES.iter().zip(&row) {
				println!(
					"{:>6} {:>5} {:>8} {:>8} {:>6} {:>9} {:>10} {:>9} {:>10} {:>6.1}%",
					n_words,
					format!("1/{}", 1 << log_inv_rate),
					priced.n_queries,
					priced.n_oracles,
					priced.index_bits,
					priced.proof_bytes,
					priced.and,
					priced.bmul,
					priced.rows(),
					100.0 * (priced.rows() as f64 / native - 1.0),
				);
			}
			row
		})
		.collect()
}

/// The three facts the surface establishes, at whatever sizes it covers.
fn check(surface: &[[Priced; N_RATES]]) {
	for row in surface {
		// The query count is what a lower rate buys, and it is arithmetic rather than measurement.
		for pair in row.windows(2) {
			assert!(
				pair[1].n_queries < pair[0].n_queries,
				"a lower rate must ask fewer queries: {} then {}",
				pair[0].n_queries,
				pair[1].n_queries
			);
		}

		// The cheapest rate is an interior one, so neither end of the sweep is the answer.
		//
		//     rate 1/2   many queries, each path short
		//     ...        the two costs cross
		//     rate 1/16  few queries, each path long and one more oracle deep
		let best = row
			.iter()
			.enumerate()
			.min_by_key(|(_, priced)| priced.rows())
			.map(|(index, _)| index)
			.expect("the sweep covers at least one rate");
		assert!(
			best > 0 && best + 1 < N_RATES,
			"the cheapest rate must be an interior one, not index {best}"
		);

		// Past that rate the proof grows as well, so nothing is being traded for the rows.
		assert!(
			row[best + 1].proof_bytes > row[best].proof_bytes,
			"a rate past the cheapest must cost bytes: {} against {}",
			row[best + 1].proof_bytes,
			row[best].proof_bytes
		);

		// The saving lands in the multiplication column.
		// That is what the per-query selection out of a decommitted layer costs.
		//
		// Hashing barely moves: fewer paths to climb offsets each one being deeper.
		let rows_left = row[best].rows() as f64 / row[0].rows() as f64;
		let bmul_left = row[best].bmul as f64 / row[0].bmul as f64;
		assert!(
			bmul_left < rows_left,
			"the multiplication column must fall furthest: {bmul_left:.3} against {rows_left:.3}"
		);
	}
}

#[test]
fn the_cost_surface_over_inner_rates() {
	// Invariant: the rate an inner proof is built at is a lever on the circuit verifying it.
	//
	// Fixture state: two CRC-64 messages, each proved at four rates.
	//
	//     4 words   ->  2^8 index bits at the native rate
	//     16 words  ->  2^9
	//
	// Every point defers the wiring claim, so nothing here tracks the inner constraint count
	// except through the commitment the replay opens.
	check(&cost_surface(&[4, 16]));
}

#[test]
#[ignore = "four proofs at 2^13 and 2^15 index bits, about forty seconds"]
fn the_cost_surface_at_node_scale() {
	// Invariant: the lever strengthens as the inner trace grows.
	//
	// Fixture state: the same sweep, two decades up.
	//
	// This is the end a node is sized against, and it is where the byte column turns over: the
	// cheapest rate is smaller than the native one on both axes at once.
	check(&cost_surface(&[64, 256]));
}
