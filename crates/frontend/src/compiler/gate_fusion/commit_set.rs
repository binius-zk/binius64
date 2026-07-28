// Copyright 2025 Irreducible Inc.
use petgraph::{
	Direction,
	visit::{DfsPostOrder, EdgeRef},
};

use super::{LeGraph, Stat};
use crate::compiler::constraint_builder::{Shift, ShiftKind};

pub const MAX_DEPTH: usize = 6;

/// Which shift kinds appeared on a path, and the largest distance any of them used.
///
/// One question is asked of a path, once per graph edge.
/// Can a further shift compose with every shift already on it?
///
/// Two shifts compose only when one is the identity, or when both share a kind.
/// So only two facts about the path decide the answer:
///
/// - which kinds are present, since a second kind already rules out composing;
/// - the largest distance among them, since that is what pushes a sum past the width.
///
/// Both fit in an integer, so the context is [`Copy`] and allocates nothing.
#[derive(Copy, Clone, Default)]
struct ShiftSummary {
	/// One bit per shift kind seen, excluding the identity.
	///
	/// A kind's bit is its discriminant.
	/// There are eight kinds, so a byte holds them all.
	kinds: u8,
	/// Largest distance among the kinds seen.
	///
	/// Zero when only the identity was seen, which imposes no distance.
	max_amount: u32,
}

impl ShiftSummary {
	/// The summary of a path carrying one shift.
	fn of(shift: Shift) -> Self {
		Self::default().with(shift)
	}

	/// The summary of this path extended by one more shift.
	fn with(self, shift: Shift) -> Self {
		match shift.kind_and_amount() {
			// The identity composes with anything, so it constrains nothing.
			None => self,
			Some((kind, amount)) => Self {
				kinds: self.kinds | bit_of(kind),
				max_amount: self.max_amount.max(amount),
			},
		}
	}

	/// The summary covering every path in `summaries`.
	fn union(summaries: impl Iterator<Item = Self>) -> Self {
		summaries.fold(Self::default(), |acc, s| Self {
			kinds: acc.kinds | s.kinds,
			max_amount: acc.max_amount.max(s.max_amount),
		})
	}

	/// Whether `shift` composes with every shift on this path.
	const fn composable(self, shift: Shift) -> bool {
		let Some((kind, amount)) = shift.kind_and_amount() else {
			// The identity composes with anything already seen.
			return true;
		};

		// Only the identity was seen, which imposes no kind and no distance.
		if self.kinds == 0 {
			return true;
		}

		// A second kind on the path can never compose with this one.
		if self.kinds != bit_of(kind) {
			return false;
		}

		// A cyclic kind loses nothing, so distances always compose.
		// Otherwise the largest distance seen plus this one has to stay inside the width.
		kind.is_cyclic() || self.max_amount + amount < kind.width()
	}
}

/// The bit standing for one kind in [`ShiftSummary::kinds`].
const fn bit_of(kind: ShiftKind) -> u8 {
	1 << kind as u8
}

#[derive(Copy, Clone)]
struct CommitSetCx {
	/// The shifts used on the path to reach this node.
	shifts: ShiftSummary,
	/// Number of nodes we should visit from the current node to get back to one of the roots (or
	/// committed linear expression)
	///
	/// This is used as a proxy to estimate the impact of inlining.
	depth: usize,
}

impl CommitSetCx {
	/// Create a new context for an edge with depth 0.
	fn new(seed_shift: Shift) -> Self {
		Self {
			shifts: ShiftSummary::of(seed_shift),
			depth: 0,
		}
	}

	/// Returns if every shift is composable with the given one.
	const fn composable(&self, shift: Shift) -> bool {
		self.shifts.composable(shift)
	}

	/// Merge multiple contexts into a single one.
	fn join<'a>(iter: impl Iterator<Item = &'a CommitSetCx>) -> Self {
		let mut depth = 0;
		let mut summaries = Vec::new();
		for cx in iter {
			depth = depth.max(cx.depth);
			summaries.push(cx.shifts);
		}
		Self {
			shifts: ShiftSummary::union(summaries.into_iter()),
			depth,
		}
	}

	/// Create a new context by adding a new shift and incrementing depth.
	fn add(&self, out_shift: Shift) -> CommitSetCx {
		Self {
			shifts: self.shifts.with(out_shift),
			depth: self.depth + 1,
		}
	}
}

/// Traverse the linear expression graph and decide which linear expressions to commit.
///
/// There are two cases where we might commit a linear expression:
///
/// 1. When inlining a linear expression is not possible because it does not fit into a single AND
///    constraint. For example, an expression that uses a shift right operator cannot be inlined
///    into a user that uses shift left operator.
///
/// 2. Inlining is prone to term explosion. To prevent that we avoid inlining expressions that lie
///    past a certain depth.
///
/// Note that this is all-or-nothing decision: if at least one user cannot inline an expression
/// then no users should inline it.
pub fn run_decide_commit_set(leg: &mut LeGraph, stat: &mut Stat) {
	// Context carried for each graph edge during the commit-set decision.
	//
	// Edge identifiers are dense integers from zero up to the edge count.
	// A slot in a vector therefore addresses each edge directly, without hashing.
	//
	// Invariant: no edge is added or removed during this pass.
	// So an edge identifier stays a valid index for the whole traversal.
	let mut per_edge: Vec<Option<CommitSetCx>> = Vec::new();
	per_edge.resize_with(leg.pg.edge_count(), || None);

	// Iterate the graph in the postorder. That is, we iterate the producers before their consumers.
	// IOW, when visiting a node all of its children have been already visited.
	//
	// Remember that Linear Expression Graph (legraph) is a directed graph where edges point towards
	// the consumers. We propagate information along the edges from the consumers up to the
	// producers.
	//
	// We seed iteration from the "sources" of graph. A source is a node with no incoming edges and
	// those are the opaque wires in our legraph. However, this is a postorder iteration and that
	// means that we start processing at the "sinks", ie. the first node to be popped out from
	// `next` is a sink. A sink is a node that does not have any outgoing edges. In legraph
	// sinks are our roots, ie. non-linear constraints.
	//
	// The information is captured by the `CommitSetCx` which represent the relevant data for the
	// inlining process.
	//
	// With all of that, what we are doing is examining every linear expression node and see if
	// every user's shifts compose with the current node shifts which are stored in the incoming
	// edges and additionally the node does not lie too deep in the graph for any of the users.
	let mut postorder = DfsPostOrder::empty(&leg.pg);
	for source in &leg.opaque {
		postorder.move_to(*source);
		while let Some(node) = postorder.next(&leg.pg) {
			if leg.is_root(node) {
				// Special handling for the root nodes.
				//
				// Just create a new context for each root node with the seed shift.
				for in_edge in leg.pg.edges_directed(node, Direction::Incoming) {
					let shift = in_edge.weight().shift;
					per_edge[in_edge.id().index()] = Some(CommitSetCx::new(shift));
				}
				continue;
			}
			if leg.is_opaque(node) {
				// Special handling for opaque nodes, or lack of there of.
				continue;
			}

			// Must be a linear definition then.
			//
			// Check whether the incoming edges are composing with every outcoming edges.
			let lin_def_wire = leg.lin_dst(node);
			let incoming = leg.pg.edges_directed(node, Direction::Incoming);
			let outcoming = leg.pg.edges_directed(node, Direction::Outgoing);

			let mut composable = true;
			let mut depth = 0;

			'out: for out_edge in outcoming.clone() {
				let out_edge_cx = per_edge[out_edge.id().index()]
					.as_ref()
					.expect("consumer edge context is set before the producer is visited");
				depth = out_edge_cx.depth.max(depth);
				for in_edge in incoming.clone() {
					let in_shift = in_edge.weight().shift;
					if !out_edge_cx.composable(in_shift) {
						composable = false;
						break 'out;
					}
				}
			}

			if depth > MAX_DEPTH || !composable {
				// Decision: commit.
				//
				// Every incoming edge context is going to be a brand new one seeded with the
				// current shift.
				for in_edge in incoming {
					let in_shift = in_edge.weight().shift;
					per_edge[in_edge.id().index()] = Some(CommitSetCx::new(in_shift));
				}

				// Insert into the committed set verifying that this wire was not inserted before.
				assert!(leg.lin_committed.insert(lin_def_wire));

				stat.note_committed();
				if depth > MAX_DEPTH {
					stat.note_committed_linear_depth();
				}
			} else {
				// Decision: inline.
				//
				// This node will beget a new context by joining outcoming contexts. Then every
				// incoming edge will get combined with the outcoming shift type.
				//
				// TODO: note that we've already visited every child, so we could free up memory
				// required for their context.
				let join_cx = CommitSetCx::join(outcoming.map(|edge| {
					per_edge[edge.id().index()]
						.as_ref()
						.expect("consumer edge context is set before the producer is visited")
				}));
				for in_edge in incoming {
					let in_shift = in_edge.weight().shift;
					per_edge[in_edge.id().index()] = Some(join_cx.add(in_shift));
				}
			}

			stat.note_visited();
		}
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;

	/// The predicate the summary replaces, evaluated over the shifts themselves.
	///
	/// Kept as the reference the summary is checked against.
	fn composable_reference(path: &[Shift], shift: Shift) -> bool {
		path.iter().all(|s| Shift::compose(*s, shift).is_some())
	}

	/// Every shift kind, at a distance inside its own width.
	fn any_shift() -> impl Strategy<Value = Shift> {
		prop_oneof![
			Just(Shift::None),
			(0u32..64).prop_map(Shift::Sll),
			(0u32..32).prop_map(Shift::Sll32),
			(0u32..64).prop_map(Shift::Srl),
			(0u32..32).prop_map(Shift::Srl32),
			(0u32..64).prop_map(Shift::Sar),
			(0u32..32).prop_map(Shift::Sra32),
			(0u32..64).prop_map(Shift::Rotr),
			(0u32..32).prop_map(Shift::Rotr32),
		]
	}

	proptest! {
		#[test]
		fn summary_answers_as_the_shift_list_does(
			path in prop::collection::vec(any_shift(), 1..12),
			query in any_shift(),
		) {
			// Invariant: the summary decides composability exactly as walking the shifts would.
			let summary = ShiftSummary::union(path.iter().copied().map(ShiftSummary::of));
			prop_assert_eq!(summary.composable(query), composable_reference(&path, query));
		}

		#[test]
		fn extending_a_path_matches_appending_to_the_list(
			path in prop::collection::vec(any_shift(), 1..12),
			extra in any_shift(),
			query in any_shift(),
		) {
			// Invariant: extending a summary is the same as appending to the list it stands for.
			let summary = ShiftSummary::union(path.iter().copied().map(ShiftSummary::of));

			let mut extended_path = path;
			extended_path.push(extra);

			prop_assert_eq!(
				summary.with(extra).composable(query),
				composable_reference(&extended_path, query)
			);
		}

		#[test]
		fn joining_paths_matches_concatenating_the_lists(
			left in prop::collection::vec(any_shift(), 1..8),
			right in prop::collection::vec(any_shift(), 1..8),
			query in any_shift(),
		) {
			// Invariant: joining two summaries is the same as concatenating the two lists.
			let left_summary = ShiftSummary::union(left.iter().copied().map(ShiftSummary::of));
			let right_summary = ShiftSummary::union(right.iter().copied().map(ShiftSummary::of));
			let joined = ShiftSummary::union([left_summary, right_summary].into_iter());

			let concatenated: Vec<Shift> = left.iter().chain(&right).copied().collect();

			prop_assert_eq!(
				joined.composable(query),
				composable_reference(&concatenated, query)
			);
		}
	}
}
