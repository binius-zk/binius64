// Copyright 2025 Irreducible Inc.
use std::array;

use binius_core::constraint_system::{Shift, ShiftVariant};
use petgraph::{
	Direction,
	visit::{DfsPostOrder, EdgeRef},
};

use super::{LeGraph, Stat};

/// Longest inline chain a definition of two or more terms may sit at the top of.
///
/// Inlining substitutes a definition's whole cone into every consumer.
/// Chaining that without a bound multiplies operand sizes.
/// Past this depth a definition is committed instead, which truncates the cone.
///
/// A definition of a single term is exempt, since it substitutes one term for one term.
///
/// # Why this value
///
/// The bound is empirical, not derived.
/// It trades committed words against operand size, and both costs are prover-side.
pub const MAX_DEPTH: usize = 6;

/// The largest shift budget any caller of this pass asks for.
///
/// A shifted term carries a sequence of shifts, and the pass is told how many slots of that
/// sequence it may fill. Sizing a summary by the largest budget keeps it [`Copy`] and inline.
const MAX_SHIFT_SLOTS: usize = 2;

/// The slot count standing for "more slots than any budget allows".
///
/// A path past this point is refused whatever its runs hold, so the runs past the recorded ones
/// need not be kept.
const OVER_BUDGET: u8 = MAX_SHIFT_SLOTS as u8 + 1;

/// One run of shifts on a path that collapse into a single slot.
///
/// Two shifts collapse under two conditions.
///
/// - One of them is the identity, which collapses with anything.
/// - Both share a variant, and their distances together stay inside the word width.
///
/// So only two facts about a run decide whether a further shift joins it.
///
/// - Which variants are present, since a second variant already rules out collapsing.
/// - The distance the run accumulates, since that is what a further distance adds to.
#[derive(Copy, Clone, Default)]
struct Run {
	/// One bit per shift variant in this run, excluding the identity.
	///
	/// A variant's bit is its discriminant.
	/// There are eight variants, so a byte holds them all.
	variants: u8,
	/// Total distance covered by the farthest-shifting path this run spans.
	///
	/// Zero when only the identity was seen, which imposes no distance.
	///
	/// Only the comparison against a variant's width matters, and no width exceeds 64, so a
	/// saturating byte answers that comparison exactly as a wider counter would.
	total_amount: u8,
}

impl Run {
	/// The run holding one shift.
	const fn of(shift: Shift) -> Self {
		Self {
			variants: bit_of(shift.variant),
			total_amount: shift.amount,
		}
	}

	/// This run with `shift` folded in, or nothing when the shift needs a slot of its own.
	///
	/// Collapsing two shifts of one variant adds their distances.
	/// So a run accumulates the distances of its links:
	/// ```text
	///     sll(3) then sll(5) then sll(4)  ->  sll(12)
	/// ```
	const fn absorb(self, shift: Shift) -> Option<Self> {
		// A run of another variant never collapses with this shift. A run of two variants — which
		// a join over several paths can produce — collapses with nothing at all.
		if self.variants != bit_of(shift.variant) {
			return None;
		}

		// A cyclic variant loses no bits, so its distances wrap and always collapse.
		// Any other variant drops the bits it shifts out.
		// So the distance the run already covers plus this one has to stay inside the width.
		// Saturating keeps an absurdly long run from wrapping back to a small total.
		let total_amount = self.total_amount.saturating_add(shift.amount);
		if !shift.variant.is_cyclic() && total_amount as usize >= shift.variant.max_amount() {
			return None;
		}

		Some(Self {
			variants: self.variants,
			total_amount,
		})
	}

	/// The run covering both of these.
	fn union(self, other: Self) -> Self {
		// A further shift has to collapse with every path at once.
		// So the path that shifts farthest is the one that binds:
		//     path A accumulates sll(10)
		//     path B accumulates sll(50)   <- binding
		//     -> a further sll(13) fits A but leaves the width on B, so it is rejected
		Self {
			variants: self.variants | other.variants,
			total_amount: self.total_amount.max(other.total_amount),
		}
	}
}

/// The shifts on a path, grouped into the slots a term needs to spell them.
///
/// One question is asked of a path, once per graph edge.
/// Can a further shift join it without needing more slots than the budget allows?
///
/// Inlining collapses each run of the path into one shift, so a further shift meets the innermost
/// run rather than the individual links. Grouping greedily from the outside in is what inlining
/// itself does, so the two agree on how many slots a path costs.
///
/// The summary is a conservative approximation of [`Shift::compose`]: it may report more slots than
/// a chain really needs, never fewer. `patch::process_term` rests on that direction alone.
#[derive(Copy, Clone, Default)]
struct ShiftSummary {
	/// The runs, outermost first.
	///
	/// Only the entries below `len` are meaningful, and only the first [`MAX_SHIFT_SLOTS`] runs
	/// are recorded at all.
	slots: [Run; MAX_SHIFT_SLOTS],
	/// How many slots this path needs, capped at [`OVER_BUDGET`].
	len: u8,
}

impl ShiftSummary {
	/// The summary of a path carrying one shift.
	fn of(shift: Shift) -> Self {
		Self::default().with(shift)
	}

	/// The summary of this path extended by one more shift, applied inside everything on it.
	///
	/// A shift of no distance is the identity whatever variant spells it, so it constrains
	/// nothing and drops out here.
	fn with(self, shift: Shift) -> Self {
		if shift.is_identity() {
			return self;
		}

		// The pass walks a path from its consumer towards its producer, so the shift arriving
		// here is applied inside every shift already summarized. It therefore meets the innermost
		// run first: joining that run costs no slot, and anything else opens one.
		let mut next = self;
		if let Some(innermost) = next.innermost_mut()
			&& let Some(absorbed) = innermost.absorb(shift)
		{
			*innermost = absorbed;
			return next;
		}

		// Past the recorded runs only the count matters: no budget exceeds `MAX_SHIFT_SLOTS`, so a
		// path that has opened that many runs is refused whatever the next one holds.
		if let Some(slot) = next.slots.get_mut(usize::from(next.len)) {
			*slot = Run::of(shift);
		}
		next.len = next.len.saturating_add(1).min(OVER_BUDGET);
		next
	}

	/// The innermost run, when the path has one and it is still recorded.
	fn innermost_mut(&mut self) -> Option<&mut Run> {
		self.slots.get_mut(usize::from(self.len).checked_sub(1)?)
	}

	/// The summary covering every path in `summaries`.
	fn union(summaries: impl Iterator<Item = Self>) -> Self {
		// Runs line up from the outside in, since that is the end every path shares. A path with
		// fewer runs than the widest is charged the widest one's count, which can only refuse
		// inlining the summary would otherwise have allowed.
		summaries.fold(Self::default(), |acc, s| Self {
			slots: array::from_fn(|slot| acc.slots[slot].union(s.slots[slot])),
			len: acc.len.max(s.len),
		})
	}

	/// Whether `shift` joins this path within a budget of `slots` shifts per term.
	fn composable(self, shift: Shift, slots: usize) -> bool {
		usize::from(self.with(shift).len) <= slots
	}
}

/// The bit standing for one variant in [`Run::variants`].
const fn bit_of(variant: ShiftVariant) -> u8 {
	1 << variant as u8
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

	/// Returns if the given shift joins this path within a budget of `slots` shifts per term.
	fn composable(&self, shift: Shift, slots: usize) -> bool {
		self.shifts.composable(shift, slots)
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
/// A definition of a single term is exempt from the second case.
/// Substituting it swaps one term for one term, so it cannot make any operand grow.
/// Depth is still counted through such a definition.
/// So a definition of two or more terms further up still reaches the cap and commits there.
///
/// Note that this is all-or-nothing decision: if at least one user cannot inline an expression
/// then no users should inline it.
///
/// # Arguments
///
/// - `shift_slots`: how many shifts a term of the lowered constraint system may carry. A path whose
///   shifts do not fit in that many slots falls under the first case above.
pub fn run_decide_commit_set(leg: &mut LeGraph, stat: &mut Stat, shift_slots: usize) {
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
					if !out_edge_cx.composable(in_shift, shift_slots) {
						composable = false;
						break 'out;
					}
				}
			}

			// One incoming edge is one term on the right-hand side.
			// Substituting such a definition rewrites a consumer's term in place:
			//
			//     definition:  y = sll(x, 4)
			//     consumer:    ... ^ sll(y, 3) ^ ...
			//     substituted: ... ^ sll(x, 7) ^ ...
			//
			// The operand keeps its size, so no operand can grow out of a chain of these.
			// The depth cap guards against operands growing, so it has nothing to guard here.
			let single_term = incoming.clone().count() == 1;
			let too_deep = depth > MAX_DEPTH && !single_term;

			if too_deep || !composable {
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
				if too_deep {
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
	use binius_core::constraint_system::Composition;
	use proptest::prelude::*;

	use super::*;

	/// The single shift a chain collapses to, or nothing when it does not collapse.
	///
	/// This is what inlining a whole path leaves behind:
	/// ```text
	///     [sll(3), sll(5), sll(4)]  ->  sll(12)
	///     [sll(3), srl(5)]          ->  nothing, two kinds never merge
	///     [sll(40), sll(30)]        ->  nothing, 70 leaves the 64-bit width
	/// ```
	fn collapse(chain: &[Shift]) -> Option<Shift> {
		// Composing shifts of one kind adds their distances, and addition commutes.
		// So folding left over the chain gives the same answer as any other order.
		chain
			.iter()
			.try_fold(Shift::IDENTITY, |acc, s| match Shift::compose(acc, *s) {
				Composition::Single(shift) => Some(shift),
				// A chain that needs two slots, or that clears the word, is not one the pass may
				// inline into a single shifted term.
				Composition::Zero | Composition::Pair => None,
			})
	}

	/// Whether one more shift composes with a chain once that chain has collapsed.
	///
	/// This is the predicate the summary stands in for at a budget of one slot.
	/// It is the reference the summary is checked against.
	fn composable_reference(chain: &[Shift], shift: Shift) -> bool {
		collapse(chain).is_some_and(|collapsed| {
			matches!(Shift::compose(collapsed, shift), Composition::Single(_))
		})
	}

	/// How many slots a chain really needs, grouping it greedily from the outside in.
	///
	/// The chain arrives outermost first, the order the pass walks a path in, so each link is
	/// applied inside the run built so far. This is what inlining does term by term, so it is the
	/// count the summary stands in for at any budget.
	///
	/// Returns nothing when the chain clears the word: that term is dropped rather than spelled,
	/// so it needs no slots at all.
	fn greedy_slots(chain: &[Shift]) -> Option<usize> {
		let mut closed = 0;
		// The run being built, as the single shift its links have collapsed to so far.
		let mut open: Option<Shift> = None;
		for &shift in chain {
			// An identity link is legal anywhere and fills no slot.
			if shift.is_identity() {
				continue;
			}
			open = match open {
				None => Some(shift),
				Some(run) => match Shift::compose(shift, run) {
					// A run that has collapsed back to the identity spells nothing.
					Composition::Single(collapsed) if collapsed.is_identity() => None,
					Composition::Single(collapsed) => Some(collapsed),
					Composition::Pair => {
						closed += 1;
						Some(shift)
					}
					Composition::Zero => return None,
				},
			};
		}
		Some(closed + usize::from(open.is_some()))
	}

	/// A chain of arbitrary shifts, as a path that mixes variants carries.
	fn any_chain() -> impl Strategy<Value = Vec<Shift>> {
		prop::collection::vec(any_shift(), 1..8)
	}

	/// The summary of one chain, accumulated link by link the way the pass accumulates it.
	fn summary_of(chain: &[Shift]) -> ShiftSummary {
		chain
			.iter()
			.fold(ShiftSummary::default(), |summary, shift| summary.with(*shift))
	}

	/// Every shift kind, at a distance inside its own width.
	fn any_shift() -> impl Strategy<Value = Shift> {
		prop_oneof![
			Just(Shift::IDENTITY),
			(0u32..64).prop_map(|n| Shift::sll(n as usize)),
			(0u32..32).prop_map(|n| Shift::sll32(n as usize)),
			(0u32..64).prop_map(|n| Shift::srl(n as usize)),
			(0u32..32).prop_map(|n| Shift::srl32(n as usize)),
			(0u32..64).prop_map(|n| Shift::sar(n as usize)),
			(0u32..32).prop_map(|n| Shift::sra32(n as usize)),
			(0u32..64).prop_map(|n| Shift::rotr(n as usize)),
			(0u32..32).prop_map(|n| Shift::rotr32(n as usize)),
		]
	}

	/// The eight shift constructors, in discriminant order.
	const KINDS: [fn(usize) -> Shift; 8] = [
		Shift::sll,
		Shift::srl,
		Shift::sar,
		Shift::rotr,
		Shift::sll32,
		Shift::srl32,
		Shift::sra32,
		Shift::rotr32,
	];

	/// A chain of one kind, mixed with identity links, as an inlined path carries.
	///
	/// A chain of two kinds never collapses, so it is out of scope here.
	///
	/// Distances run up to 8 over up to 11 links, which puts the total between 0 and 88.
	/// That straddles both widths, 32 and 64, instead of blowing past them every time.
	/// So the collapsing and the non-collapsing side of the boundary are both reachable.
	fn same_kind_chain() -> impl Strategy<Value = Vec<Shift>> {
		(0usize..KINDS.len(), prop::collection::vec(prop::option::of(0u32..9), 1..12)).prop_map(
			|(kind, amounts)| {
				// An absent distance stands for an identity link.
				// Identity links are legal anywhere and must not sway the answer.
				amounts
					.into_iter()
					.map(|amount| match amount {
						Some(amount) => KINDS[kind](amount as usize),
						None => Shift::IDENTITY,
					})
					.collect()
			},
		)
	}

	proptest! {
		#[test]
		fn summary_answers_as_the_collapsed_chain_does(
			chain in same_kind_chain(),
			query in any_shift(),
		) {
			// Only chains that collapse are in scope.
			// The pass commits the rest rather than inlining them, so no summary is ever built.
			prop_assume!(collapse(&chain).is_some());

			// Invariant: whatever the summary permits, the collapsed chain really composes.
			//
			//     chain:   [sll(3), sll(5), sll(4)]   summary: variant sll, total 12
			//     collapse:              sll(12)
			//     query sll(51) -> 12 + 51 = 63, inside 64  -> permitted, and it composes
			//     query sll(52) -> 12 + 52 = 64, at the width -> refused
			//
			// The converse does not hold, and need not: the summary tracks a total distance
			// rather than the composition itself, so it refuses chains core collapses anyway —
			// `sar(40)` then `sar(40)` saturates to `sar(63)`, but the summary sees 80 and says
			// no. That costs inlining, never correctness, and `patch::process_term` panics only
			// on the direction asserted here.
			prop_assert!(
				!summary_of(&chain).composable(query, 1) || composable_reference(&chain, query)
			);
		}

		#[test]
		fn extending_a_chain_matches_appending_a_link(
			chain in same_kind_chain(),
			query in any_shift(),
		) {
			prop_assume!(collapse(&chain).is_some());

			// Split the last link off, so the head is a shorter chain and the link is new to it.
			let (extra, head) = chain.split_last().expect("the chain has at least one link");

			// Invariant: extending a summary by one link answers as the longer chain does.
			// This is the step the pass takes at every graph edge it walks.
			prop_assert!(
				!summary_of(head).with(*extra).composable(query, 1)
					|| composable_reference(&chain, query)
			);
		}

		#[test]
		fn joining_chains_answers_for_both(
			left in same_kind_chain(),
			right in same_kind_chain(),
			query in any_shift(),
		) {
			prop_assume!(collapse(&left).is_some());
			prop_assume!(collapse(&right).is_some());

			// A definition with two consumers reaches its roots by two paths.
			// Its summary spans both, since inlining it has to satisfy both at once.
			let joined = ShiftSummary::union([summary_of(&left), summary_of(&right)].into_iter());

			// Invariant: a shift joins the spanning summary only when it composes with each path.
			prop_assert!(
				!joined.composable(query, 1)
					|| composable_reference(&left, query) && composable_reference(&right, query)
			);
		}

		/// The budget-one properties above pin the rule the pass runs today. These two pin it at
		/// every budget the summary can be asked about, against a reference that groups with
		/// `Shift::compose` itself rather than with the summary's variant-and-distance stand-in.
		#[test]
		fn a_chain_reported_within_budget_really_fits_it(
			chain in any_chain(),
			slots in 1usize..=MAX_SHIFT_SLOTS,
		) {
			// Invariant: whatever the summary says fits, the chain really spells in that many
			// shifts. The converse does not hold and need not: the summary approximates
			// `Shift::compose` with a variant-and-distance test, so it opens runs that composing
			// would have collapsed — `slr(3)` inside `sar(5)` is one shift to core and two to the
			// summary. That costs inlining, never correctness.
			prop_assert!(
				usize::from(summary_of(&chain).len) > slots
					|| greedy_slots(&chain).unwrap_or(0) <= slots
			);
		}

		#[test]
		fn a_permitted_shift_keeps_the_chain_within_budget(
			chain in any_chain(),
			query in any_shift(),
			slots in 1usize..=MAX_SHIFT_SLOTS,
		) {
			// The query lands inside everything on the path, which is the innermost end — the end
			// the summary accumulates at.
			let extended = [chain.as_slice(), &[query]].concat();

			// Invariant: the shift the summary admits leaves a chain the budget really covers.
			// This is the direction `patch::process_term` rests on at any budget.
			prop_assert!(
				!summary_of(&chain).composable(query, slots)
					|| greedy_slots(&extended).unwrap_or(0) <= slots
			);
		}
	}
}
