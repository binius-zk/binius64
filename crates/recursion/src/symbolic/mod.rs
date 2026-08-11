// Copyright 2026 The Binius Developers

//! The value types a circuit-building channel carries.
//!
//! A verifier written against the channel traits handles two kinds of value it cannot name
//! concretely: field elements and 64-bit words. [`SymbolicElem`] and [`SymbolicWord`] are those,
//! carried as wires in a circuit under construction rather than as values.
//!
//! Both follow the same shape. Each is either a constant, folded while the circuit is built and
//! costing nothing, or wires plus a [`Weak`](std::rc::Weak) handle to the builder. The handle is
//! what lets arithmetic be plain operators — `a * b`, `word >> n` — rather than channel methods,
//! which is what the trait bounds ask for.

mod elem;
mod word;

pub use elem::SymbolicElem;
pub use word::SymbolicWord;
