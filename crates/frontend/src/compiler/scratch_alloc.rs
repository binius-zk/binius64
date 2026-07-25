// Copyright 2026 The Binius Developers
//! Slot assignment for the scratch segment of the value vector.
//!
//! A value no constraint references is never committed.
//! It only has to survive from the gate that writes it to the gate that last reads it.
//! Gate fusion produces many: it inlines an operation and leaves its result uncommitted.
//!
//! Those lifetimes are short, so one slot can serve several values in turn.
//!
//! ```text
//!   gate:      0     1     2     3
//!   value a:   w-----r
//!   value b:         w-----r
//!   value c:               w-----r
//!
//!   a slot each:    a  b  c       -> 3 slots
//!   shared slots:   a  b  a       -> 2 slots, since a dies before c is written
//! ```
//!
//! The batched witness buffer holds one column per instance.
//! So every slot dropped here is saved once per instance.
//!
//! # Why this cannot change a proof
//!
//! A value in this segment appears in no constraint operand, by the definition of the segment.
//! Only the recorded length of the segment moves.

use cranelift_entity::{EntitySet, SecondaryMap};

use crate::compiler::gate_graph::{GateGraph, Wire};

/// How the scratch segment lays out its values.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScratchPolicy {
	/// One permanent slot per value, numbered in the order the values were created.
	PerWire,
	/// Slots are shared, each reclaimed once the gate that last reads its value has run.
	Pooled,
}

/// The scratch-segment slot holding each value in the segment.
pub struct ScratchAlloc {
	/// Slot index within the segment, set only for values that belong to the segment.
	slot: SecondaryMap<Wire, Option<u32>>,
	/// Length of the segment under the chosen policy.
	n_slots: usize,
	/// Length the segment would have if its slots were shared.
	///
	/// This is the largest number of segment values alive at the same time.
	/// It is recorded under either policy, so the unused headroom stays visible.
	peak_live: usize,
}

impl ScratchAlloc {
	/// Assigns a slot to every value in the scratch segment.
	///
	/// # Arguments
	///
	/// * `graph` - the gate graph, visited in the order the evaluation form runs it.
	/// * `scratch` - the values no constraint operand references.
	/// * `policy` - whether slots are shared or permanent.
	pub fn new(graph: &GateGraph, scratch: &EntitySet<Wire>, policy: ScratchPolicy) -> Self {
		// When each value dies, which is what tells the sweep when a slot comes free.
		let deaths = order_deaths(graph, scratch);

		// The shared assignment is computed either way.
		// Its length is the headroom figure reported under both policies.
		let (pooled_slot, peak_live) = pool_slots(graph, scratch, &deaths);

		match policy {
			// Sharing: the segment is exactly as long as the largest set of simultaneous lifetimes.
			ScratchPolicy::Pooled => Self {
				slot: pooled_slot,
				n_slots: peak_live,
				peak_live,
			},
			// No sharing: the shared assignment is discarded, only its length is kept.
			ScratchPolicy::PerWire => {
				let (slot, n_slots) = slot_per_wire(graph, scratch);
				Self {
					slot,
					n_slots,
					peak_live,
				}
			}
		}
	}

	/// Returns the slot within the scratch segment that holds the given value.
	///
	/// # Panics
	///
	/// Panics if the value is not part of the scratch segment.
	pub fn slot(&self, wire: Wire) -> u32 {
		self.slot[wire].expect("a slot exists only for a value in the scratch segment")
	}

	/// Returns the length of the scratch segment.
	pub const fn n_slots(&self) -> usize {
		self.n_slots
	}

	/// Returns the largest number of scratch values alive at the same time.
	pub const fn peak_live(&self) -> usize {
		self.peak_live
	}
}

/// Pairs each scratch value with the last gate that mentions it, ordered by that gate.
///
/// # Algorithm
///
/// A gate stores every value it touches in one slice.
/// That covers constants, inputs, outputs, auxiliaries and its own temporaries.
/// Gates are emitted in graph order and each one compiles to a contiguous run of instructions.
/// So the last gate mentioning a value is the last gate that can read it.
fn order_deaths(graph: &GateGraph, scratch: &EntitySet<Wire>) -> Vec<(u32, Wire)> {
	// Phase 1: overwrite the recorded gate on every mention, leaving the last one.
	let mut last_gate: SecondaryMap<Wire, Option<u32>> = SecondaryMap::new();
	for (index, (_, data)) in graph.gates.iter().enumerate() {
		for &wire in &data.wires {
			// Constants, inputs and committed outputs keep fixed slots, so they are skipped.
			if scratch.contains(wire) {
				last_gate[wire] = Some(index as u32);
			}
		}
	}

	// Phase 2: invert into a list the sweep can walk with a single cursor.
	//
	// A value no gate mentions has no death to record.
	// A well-formed graph has none, since some gate must write it.
	// Skipping it keeps the sweep total rather than relying on that.
	let mut deaths: Vec<(u32, Wire)> = scratch
		.iter()
		.filter_map(|wire| last_gate[wire].map(|gate| (gate, wire)))
		.collect();

	// Sorting makes the sweep independent of the iteration order of the set.
	// That keeps the compiled layout reproducible.
	deaths.sort_unstable();
	deaths
}

/// Hands out shared slots by sweeping the gates in the order they run.
///
/// # Algorithm
///
/// A value takes a slot at the first gate that mentions it, which is the gate that writes it.
/// The slot returns to a free list once the gate that last reads the value has finished.
///
/// ```text
///   gate 0:  take slot 0 for a
///   gate 1:  take slot 1 for b, then a dies -> slot 0 goes back
///   gate 2:  take slot 0 for c, then b dies -> slot 1 goes back
/// ```
///
/// # Returns
///
/// The assignment, and the largest number of values alive at the same time.
fn pool_slots(
	graph: &GateGraph,
	scratch: &EntitySet<Wire>,
	deaths: &[(u32, Wire)],
) -> (SecondaryMap<Wire, Option<u32>>, usize) {
	let mut slot: SecondaryMap<Wire, Option<u32>> = SecondaryMap::new();
	// Slots given back and not yet reused.
	let mut free: Vec<u32> = Vec::new();
	// Slots handed out at least once, which is the segment length being built.
	let mut n_slots = 0u32;
	// Values alive right now, and the high-water mark over the whole sweep.
	let mut live = 0usize;
	let mut peak_live = 0usize;
	// Cursor into the death list, which advances monotonically because the list is gate-ordered.
	let mut next_death = 0usize;

	for (index, (_, data)) in graph.gates.iter().enumerate() {
		// Phase 1: give a slot to every value this gate is the first to mention.
		for &wire in &data.wires {
			if scratch.contains(wire) && slot[wire].is_none() {
				// Reuse a returned slot when one is available, otherwise extend the segment.
				slot[wire] = Some(free.pop().unwrap_or_else(|| {
					let fresh = n_slots;
					n_slots += 1;
					fresh
				}));
				live += 1;
			}
		}

		// The mark is sampled after allocating and before reclaiming.
		// So a value written here and one dying here count as overlapping.
		peak_live = peak_live.max(live);

		// Phase 2: reclaim the slots of values this gate was the last to read.
		//
		// Invariant: a slot is reclaimed only once a whole gate has finished.
		// So a gate can never be handed a slot still holding an input it has yet to read.
		while let Some(&(death_gate, wire)) = deaths.get(next_death)
			&& death_gate == index as u32
		{
			free.push(slot[wire].expect("a dying value took a slot at the gate that wrote it"));
			live -= 1;
			next_death += 1;
		}
	}

	// Left-to-right assignment over lifetimes on a line is optimal.
	// It uses exactly as many slots as the largest overlapping set.
	debug_assert_eq!(n_slots as usize, peak_live);
	// Every value in the segment is written by the gate that created it.
	// So the sweep reaches all of them and the assignment is total.
	debug_assert!(scratch.iter().all(|wire| slot[wire].is_some()));

	(slot, peak_live)
}

/// Gives each scratch value its own slot, numbered in the order the values were created.
///
/// This reproduces the layout of a build that shares nothing.
fn slot_per_wire(
	graph: &GateGraph,
	scratch: &EntitySet<Wire>,
) -> (SecondaryMap<Wire, Option<u32>>, usize) {
	let mut slot: SecondaryMap<Wire, Option<u32>> = SecondaryMap::new();
	let mut n_slots = 0u32;

	// Creation order is the order the value allocator walks the graph.
	// Numbering in that order reproduces the indices of an unshared build.
	for wire in graph.wires.keys() {
		if scratch.contains(wire) {
			slot[wire] = Some(n_slots);
			n_slots += 1;
		}
	}

	(slot, n_slots as usize)
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;

	use super::*;
	use crate::compiler::{gate::opcode::Opcode, gate_graph::GateGraph};

	/// Builds a chain of gates, each reading only the value the previous one produced.
	///
	/// ```text
	///   input --> gate 0 --> t0 --> gate 1 --> t1 --> gate 2 --> t2 ...
	/// ```
	///
	/// Every temporary dies at the next gate.
	/// So at most two are alive at once, however long the chain grows.
	///
	/// # Returns
	///
	/// The graph, and the set of temporaries that form its scratch segment.
	fn chain(length: usize) -> (GateGraph, EntitySet<Wire>) {
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();
		let mut scratch = EntitySet::new();

		// A public input starts the chain, so the first gate has something committed to read.
		let mut cur = graph.add_inout();
		// A constant second operand keeps every link identical apart from its input.
		let mask = graph.add_constant(Word(0xff));

		for _ in 0..length {
			// Each link consumes the previous result and produces a fresh temporary.
			let next = graph.add_internal();
			graph.emit_gate(root, Opcode::Band, vec![cur, mask], vec![next]);
			scratch.insert(next);
			cur = next;
		}

		(graph, scratch)
	}

	#[test]
	fn pooling_collapses_a_chain_to_its_peak() {
		// Invariant: the segment shrinks to the largest set of simultaneous lifetimes.
		// It does not shrink to the number of values.
		//
		// Fixture state: a chain of n links, each temporary read once by the following link.
		//
		//   gate:  0     1     2     3
		//   t0:    w-----r
		//   t1:          w-----r
		//   t2:                w-----r
		//
		// Two lifetimes overlap at any gate, so two slots suffice at every length past one.
		for length in [1usize, 2, 8, 64] {
			let per_wire = {
				let (graph, scratch) = chain(length);
				ScratchAlloc::new(&graph, &scratch, ScratchPolicy::PerWire)
			};
			// Without sharing the segment grows with the chain.
			assert_eq!(per_wire.n_slots(), length);

			let pooled = {
				let (graph, scratch) = chain(length);
				ScratchAlloc::new(&graph, &scratch, ScratchPolicy::Pooled)
			};
			// A one-link chain peaks at one live value; every longer chain peaks at two.
			assert_eq!(pooled.n_slots(), length.min(2));

			// The peak describes the graph, so both policies must report the same figure.
			assert_eq!(per_wire.peak_live(), pooled.peak_live());
			// Under sharing the segment length is that peak.
			assert_eq!(pooled.peak_live(), pooled.n_slots());
		}
	}

	#[test]
	fn a_slot_is_never_shared_by_two_live_values() {
		// Invariant: two values whose lifetimes overlap must never land on the same slot.
		// This is the property the whole segment rests on.
		// It is checked directly rather than inferred from the resulting length.
		//
		// Fixture state: a 32-link chain, so slots are reused many times over.
		let length = 32;
		let (graph, scratch) = chain(length);
		let pooled = ScratchAlloc::new(&graph, &scratch, ScratchPolicy::Pooled);

		// Recover each lifetime independently of the assignment being tested.
		// A lifetime spans the first gate mentioning a value to the last.
		let mut first = SecondaryMap::<Wire, Option<u32>>::new();
		let mut last = SecondaryMap::<Wire, Option<u32>>::new();
		for (index, (_, data)) in graph.gates.iter().enumerate() {
			for &wire in &data.wires {
				if scratch.contains(wire) {
					let index = index as u32;
					// The first mention is the write, so it is recorded once and kept.
					first[wire].get_or_insert(index);
					// The last mention is the final read, so it is overwritten each time.
					last[wire] = Some(index);
				}
			}
		}

		// Compare every pair that shares a slot and require their spans to be disjoint.
		let wires: Vec<Wire> = scratch.iter().collect();
		for (i, &a) in wires.iter().enumerate() {
			for &b in &wires[i + 1..] {
				if pooled.slot(a) != pooled.slot(b) {
					continue;
				}
				let (a_first, a_last) = (first[a].unwrap(), last[a].unwrap());
				let (b_first, b_last) = (first[b].unwrap(), last[b].unwrap());
				// Disjoint means one span ends strictly before the other begins.
				assert!(
					a_last < b_first || b_last < a_first,
					"{a:?} [{a_first},{a_last}] and {b:?} [{b_first},{b_last}] share slot {}",
					pooled.slot(a)
				);
			}
		}
	}

	#[test]
	fn simultaneously_live_values_each_take_a_slot() {
		// Invariant: sharing helps only where lifetimes are disjoint.
		// Where they all overlap, no slot can be saved.
		//
		// Fixture state: sixteen results, all consumed by one final gate.
		//
		//   gate:    0  1  2 ... 15    16
		//   t0:      w--------------------r
		//   t1:         w-----------------r
		//   ...                           r
		//
		// Every lifetime reaches the last gate, so all sixteen overlap and none can be shared.
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();
		let mut scratch = EntitySet::new();

		let x = graph.add_inout();
		let mask = graph.add_constant(Word(0xff));
		let leaves: Vec<Wire> = (0..16)
			.map(|_| {
				let out = graph.add_internal();
				graph.emit_gate(root, Opcode::Band, vec![x, mask], vec![out]);
				scratch.insert(out);
				out
			})
			.collect();

		// One multi-input gate reads all sixteen, keeping every one alive to the end.
		let sink = graph.add_internal();
		graph.emit_gate_generic(
			root,
			Opcode::BxorMulti,
			leaves.iter().copied(),
			[sink],
			&[leaves.len()],
			&[],
		);

		let pooled = ScratchAlloc::new(&graph, &scratch, ScratchPolicy::Pooled);
		assert_eq!(pooled.n_slots(), 16);
	}

	#[test]
	fn an_empty_segment_needs_no_slots() {
		// Invariant: a circuit whose values are all committed has no segment to lay out.
		//
		// Fixture state: a four-link chain whose temporaries are all treated as committed.
		// They count as committed by being left out of the segment set.
		let (graph, _) = chain(4);
		let empty = EntitySet::new();

		let pooled = ScratchAlloc::new(&graph, &empty, ScratchPolicy::Pooled);
		assert_eq!(pooled.n_slots(), 0);
		assert_eq!(pooled.peak_live(), 0);

		// The unshared policy has to agree, since there is nothing to number.
		let per_wire = ScratchAlloc::new(&graph, &empty, ScratchPolicy::PerWire);
		assert_eq!(per_wire.n_slots(), 0);
	}
}
