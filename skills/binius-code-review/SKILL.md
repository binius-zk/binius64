---
name: binius-code-review
description: Review a binius64 diff/PR to jimpo's taste — a mechanical style pass against CONTRIBUTING.md plus a judgment pass over accumulated design principles (simplicity, field-crate ease-of-use over ease-of-extension, no backwards compatibility, avoid premature generalization, …). Also the canonical place to record new taste distilled from jimpo's reviews. Use when asked to review a change, or proactively before opening a PR.
---

# Binius code review (jimpo's taste)

A review has two layers, graded differently:

1. **Mechanical style — hard rules.** Enforced by `binius64/CONTRIBUTING.md` (and fmt/clippy).
   Pass/fail; cite the rule. Don't restate them here — read CONTRIBUTING.md.
2. **Taste — design principles.** The *Principles* below. They read as mandates ("Prefer X over Y")
   to stay pithy and memorable, but they are **tradeoff guidelines, not rules**: the **Why** and
   **Reconsider when** are the substance. Surface where a change prefers the wrong side as a tradeoff
   to discuss — don't auto-reject.

## Procedure

1. **Get the diff.** `git diff <base>...HEAD` (or the PR). Know what changed and which crates.
2. **Mechanical pass.** Walk CONTRIBUTING.md and flag violations with `file:line`. The usual
   suspects: copyright header on new/modified files; present-tense docs & commit messages;
   crate-level `//!` docs; doc examples on public API; naming (`F`/`P`, descriptive CamelCase
   generics, namespacing — `sumcheck::prove` not `sumcheck_prove`); functional style (iterator
   combinators / `iter::zip` over mutable loops); no `unwrap` in library code (`expect` with a
   justification); turbofish over local type annotations; prover/verifier separation; precondition
   contracts & error-handling boundaries. Run fmt (`cargo +nightly-2026-01-01 fmt --check`) and
   crate-scoped clippy (see the sandbox `CLAUDE.md` — `--all-features` is broken here).
3. **Taste pass.** For each Principle, ask "does this change prefer the wrong side?" If so, raise it:
   state the principle, why it leans that way, and the **Reconsider when** signals so the author can
   overrule with eyes open. On a feature-gated / arch-specific path, also apply
   [[cross-arch-trait-obligation]].
4. **Report.** Separate **must-fix style** from **taste observations**. Lead with a one-line verdict.
   Frame taste items as "this prefers X at the cost of Y — worth it here?"
5. **Capture.** If jimpo states a new preference or tradeoff call (in a review thread or chat),
   distill it into a Principle (format below). This step is what makes the skill accumulate.

## Design Goals

### Simplicity

Simplicity is a foundational design goal. This codebase develops security critical code. Keeping the
code simple improves understandability, security, and often performance.

Simplicity pertains to how easy it is to understand something and how fast it is to communicate that
knowledge. Simple code is loosely coupled. To understand a module, if it is loosely coupled, one
needs to understand the public interfaces of the dependencies and very little about their
implementation.

## Principles

Each principle reads **Prefer** [_strongly_ / _weakly_ / —] *X* **over** *Y*, then **Why** (the
durable value) and **Reconsider when** (the signals that flip it). Omit the adverb for a
default-strength lean.

### Field crate ease-of-use over ease-of-extension

- **Prefer** keeping `binius-field` easy for library consumers to *use* (callers, downstream
  signatures) **over** making `binius-field` easy to extend.
- **Why:** the crate is consumed across the entire codebase but extended rarely, by experts. A bound
  threaded through public APIs is noise that *every* caller and downstream signature pays; a harder
  impl is paid once, by someone who already understands the crate. Push complexity *down into* the
  crate to keep its surface clean. Concretely: make an always-definable capability a **parent trait**
  of `PackedField`/`Field` (every impl supplies it, usually trivially) instead of a `+ Bound`
  propagated through call sites.
- **Reconsider when:** no sensible trivial/default impl exists for every type, or making it a parent
  trait forces a genuine coherence mess (overlapping blanket impls) whose cost to implementers
  outweighs the caller win — then a narrower, opt-in bound may be right.

### No backwards compatibility

- **Prefer** _strongly_ building clean and targeted interfaces **over** maintaining backwards
  compatible interfaces.
- **Why:** the codebase is not in production and backwards compatibility is not a requirement. If
  making a breaking interface change, try to isolate that to a commit.
- **Reconsider when:** the breakage becomes a whole-codebase change without good justification.

### Avoid premature generalization

- **Prefer** _weakly_ simple logic **over** perfectly DRY code.
- **Why:** this codebase places more importance on readability and understandability than
  extensibility. If some structure or logic is repeated with small modifications in a few places,
  that is better than one combined, generalized, confusing instance.
- **Reconsider when:** the generalization is just as simple, and decouples logic in a way that
  improves understandability.

### Generous bounds over caller contortion

- **Prefer** adding a trait or associated-type bound that *every* impl trivially satisfies (e.g.
  `Clone`, `Sum`) **over** keeping the bound minimal at the cost of gymnastics at the call sites.
- **Why:** the bound costs nothing when all implementers already meet it; the call sites are where
  the ergonomics and the bugs live. Don't make many callers contort (`array::from_fn`, `mem::take`)
  to keep a bound theoretically minimal. (The ease-of-use value, applied to bounds.)
- **Reconsider when:** some impl can't satisfy it, or the narrowness encodes a real safety/coherence
  constraint worth protecting.

### Don't trade loose coupling for method-call convenience

- **Prefer** keeping a function where its module boundary keeps responsibilities separate (e.g.
  algorithm/phase code apart from plain data types) **over** making it a method on a type it happens
  to take as an argument.
- **Why:** *Simplicity* above already commits to loose coupling as the thing that keeps a module
  understandable from its interface alone. Converting a free function to a method is only a win when
  the type is genuinely the sole owner of that responsibility; when the function has several
  arguments of comparable centrality, or the move drags a data-structure module into importing the
  algorithm/compute/rayon machinery it previously didn't need, the "convenience" of `x.method()`
  syntax is bought with real coupling the module didn't have before.
- **Reconsider when:** the type really is the one thing the function is about, and moving it doesn't
  add imports that cross the algorithm/data-structure boundary.

### Cohesive commits and PRs

- **Prefer** multiple small, logically cohesive commits and PRs **over** one large monolithic change.
- **Why:** Smaller, focused PRs (targeting ~200–500 lines changed, though some are naturally smaller
  or larger) are easier to review, easier to reason about in isolation, and easier to revert. A
  stack of well-scoped PRs communicates design incrementally — each step is legible on its own. jj
  makes stacked PR workflows tractable; leverage it.
- **Reconsider when:** the changes are genuinely inseparable — each part only makes sense in the
  context of the whole (e.g. a mechanical rename across the entire codebase where splitting creates
  a half-renamed intermediate state that doesn't compile).

## Maintaining this skill

- When jimpo makes a taste call, write a Principle for the **value behind it**, not the bare action —
  a rule ("add `Sum` to bounds") misfires next time; the principle ("a bound all impls meet, to spare
  callers") transfers. Give it a pithy, memorable name and the **Prefer X over Y** shape.
- Prefer **editing/merging** an existing Principle (sharpen the lean, adjust the strength adverb)
  over adding a near-duplicate.
- Keep hard rules out — those belong in `CONTRIBUTING.md`. If a soft lean hardens into something
  mechanically checkable, propose moving it there instead.
