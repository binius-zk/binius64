// Copyright 2026 The Binius Developers

//! Timestamp buffers for Spark's offline memory-checking argument.
//!
//! Spark reads a memory $\text{mem}$ of $m$ cells at a sequence of $n$ addresses, and proves that
//! the claimed read values are the memory's actual contents. The memory is read-only, so each
//! operation is modelled as a read of a cell followed by a write-back of the same value under a
//! fresh timestamp. Correctness of the reads reduces to an equality of multisets of
//! $(\text{value}, \text{timestamp})$ tuples,
//!
//! $$
//! \text{Init} \cup \text{WS} = \text{RS} \cup \text{Audit},
//! $$
//!
//! where $\text{Init}$ and $\text{Audit}$ hold one tuple per cell — at its initial and final
//! timestamp — and $\text{RS}$ and $\text{WS}$ hold one tuple per operation, at the timestamp read
//! from the cell and the timestamp written back to it. Addresses are dropped from the tuples
//! because in Spark each cell holds $\widetilde{\text{eq}}(a, r)$ for a random $r$, which is
//! injective in the address $a$, so the value already determines it.
//!
//! ## Timestamps are powers of the multiplicative generator
//!
//! Offline memory checking conventionally counts timestamps with the integers $0, 1, 2, \ldots$,
//! which collapse in a field of characteristic 2. Timestamp $t$ is instead represented by the field
//! element $g^t$, where $g$ is the multiplicative generator
//! ([`Field::MULTIPLICATIVE_GENERATOR`]); incrementing a counter becomes a multiplication by $g$.
//! The powers are distinct as long as the order of $g$ exceeds the number of operations, which is
//! no practical constraint over the 128-bit fields Spark runs in.
//!
//! ## Counters are per cell
//!
//! Each cell counts its own accesses rather than sampling a global operation counter, so the
//! timestamp written back by an operation is the timestamp it read, advanced by one tick:
//! $\text{write\\_ts}_k = \text{read\\_ts}_k \cdot g$. Write timestamps therefore never need to be
//! materialized or committed — the verifier derives them from the read timestamps (Spartan
//! optimization #3, sound here because the memory is read-only).

use binius_field::{Field, util::powers};

/// The timestamp buffers of one memory, derived from its access pattern.
///
/// The access pattern is public and fixed at setup, so these buffers are precomputed once by
/// shared prover and verifier code. See the [module documentation](self) for the timestamp
/// convention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryTimestamps<F> {
	/// The timestamp read by each operation: the one last written to the cell it addresses.
	pub read_ts: Vec<F>,
	/// The timestamp last written to each cell, over the whole access sequence.
	pub final_ts: Vec<F>,
}

impl<F: Field> MemoryTimestamps<F> {
	/// The timestamp every cell holds before the first operation, $g^0 = 1$.
	pub const INIT_TS: F = F::ONE;

	/// Computes the timestamp buffers for the access pattern `addrs` over a memory of `mem_size`
	/// cells.
	///
	/// ```
	/// use binius_field::{Field, arch::OptimalB128 as B128};
	/// use binius_ip::spark::timestamps::MemoryTimestamps;
	///
	/// // Cell 1 is accessed twice, cell 0 once and cell 2 never.
	/// let timestamps = MemoryTimestamps::<B128>::new(&[1, 0, 1], 3);
	///
	/// let g = B128::MULTIPLICATIVE_GENERATOR;
	/// assert_eq!(timestamps.read_ts, [B128::ONE, B128::ONE, g]);
	/// assert_eq!(timestamps.final_ts, [g, g * g, B128::ONE]);
	/// ```
	///
	/// # Panics
	///
	/// Panics if any address is at least `mem_size`.
	pub fn new(addrs: &[usize], mem_size: usize) -> Self {
		// A single cell may be accessed by every operation, so the timestamps range over
		// g^0, ..., g^n. Tabulating the powers keeps each timestamp a lookup.
		let g_powers = powers(F::MULTIPLICATIVE_GENERATOR)
			.take(addrs.len() + 1)
			.collect::<Vec<_>>();

		let mut access_counts = vec![0usize; mem_size];
		let read_ts = addrs
			.iter()
			.map(|&addr| {
				let access_count = &mut access_counts[addr];
				let read_ts = g_powers[*access_count];
				*access_count += 1;
				read_ts
			})
			.collect();
		let final_ts = access_counts
			.iter()
			.map(|&access_count| g_powers[access_count])
			.collect();

		Self { read_ts, final_ts }
	}

	/// The timestamp each operation writes back, one tick past the timestamp it read.
	pub fn write_ts(&self) -> impl Iterator<Item = F> + '_ {
		self.read_ts
			.iter()
			.map(|&read_ts| read_ts * F::MULTIPLICATIVE_GENERATOR)
	}
}

#[cfg(test)]
mod tests {
	use std::iter::{repeat_with, zip};

	use binius_field::{Random, arch::OptimalB128 as B128};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;

	/// The multiset hash $H_\gamma(S) = \prod_{(v, t) \in S} (v \gamma_1 + t - \gamma)$.
	fn multiset_hash(
		tuples: impl IntoIterator<Item = (B128, B128)>,
		gamma: B128,
		gamma_1: B128,
	) -> B128 {
		tuples
			.into_iter()
			.map(|(value, ts)| value * gamma_1 + ts - gamma)
			.product()
	}

	/// Asserts that the buffers for `addrs` satisfy `H(Init) * H(WS) = H(RS) * H(Audit)`.
	fn assert_multiset_identity(addrs: &[usize], mem_size: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		// Distinct cell values stand in for the injective eq̃(a, r) that Spark reads.
		let mem = repeat_with(|| B128::random(&mut rng))
			.take(mem_size)
			.collect::<Vec<_>>();
		let gamma = B128::random(&mut rng);
		let gamma_1 = B128::random(&mut rng);

		let timestamps = MemoryTimestamps::<B128>::new(addrs, mem_size);

		let init = mem
			.iter()
			.map(|&value| (value, MemoryTimestamps::<B128>::INIT_TS));
		let audit = zip(mem.iter().copied(), timestamps.final_ts.iter().copied());
		let read_set = zip(addrs, &timestamps.read_ts).map(|(&addr, &ts)| (mem[addr], ts));
		let write_set = zip(addrs, timestamps.write_ts()).map(|(&addr, ts)| (mem[addr], ts));

		assert_eq!(
			multiset_hash(init, gamma, gamma_1) * multiset_hash(write_set, gamma, gamma_1),
			multiset_hash(read_set, gamma, gamma_1) * multiset_hash(audit, gamma, gamma_1)
		);
	}

	proptest! {
		/// Covers empty and single-operation sequences, repeated addresses, never-accessed cells
		/// and single-cell memories.
		#[test]
		fn multiset_identity_holds(
			(mem_size, addrs) in (1usize..16).prop_flat_map(|mem_size| {
				(Just(mem_size), prop::collection::vec(0..mem_size, 0..64))
			})
		) {
			assert_multiset_identity(&addrs, mem_size);
		}
	}

	#[test]
	fn timestamps_count_accesses_per_cell() {
		let g_powers = powers(B128::MULTIPLICATIVE_GENERATOR)
			.take(5)
			.collect::<Vec<_>>();

		let timestamps = MemoryTimestamps::<B128>::new(&[2, 0, 2, 2, 1], 4);

		// Each read observes the number of preceding accesses to the same cell.
		assert_eq!(
			timestamps.read_ts,
			[
				g_powers[0],
				g_powers[0],
				g_powers[1],
				g_powers[2],
				g_powers[0]
			]
		);
		// Each cell ends at its total access count; cell 3 is never accessed and stays initial.
		assert_eq!(timestamps.final_ts, [g_powers[1], g_powers[1], g_powers[3], g_powers[0]]);
		assert_eq!(timestamps.final_ts[3], MemoryTimestamps::<B128>::INIT_TS);
		assert_eq!(
			timestamps.write_ts().collect::<Vec<_>>(),
			[
				g_powers[1],
				g_powers[1],
				g_powers[2],
				g_powers[3],
				g_powers[1]
			]
		);
	}

	#[test]
	#[should_panic]
	fn address_out_of_range_panics() {
		MemoryTimestamps::<B128>::new(&[4], 4);
	}
}
