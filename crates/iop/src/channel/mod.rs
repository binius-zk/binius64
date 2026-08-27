// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) verifiers.

pub mod grinding;
pub mod merge;
pub mod naive;
pub mod oracle_setup;
pub mod size_tracking;

use std::iter;

use binius_field::Field;
use binius_ip::channel::IPVerifierChannel;
use binius_utils::checked_arithmetics::log2_ceil_usize;

use crate::{basefold, ligerito};

/// Error type for IOP verifier channel operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("proof is empty")]
	ProofEmpty,
	#[error("BaseFold verification failed: {0}")]
	BaseFold(#[from] basefold::Error),
	#[error("Ligerito verification failed: {0}")]
	Ligerito(#[from] ligerito::Error),
	#[error("IP channel error: {0}")]
	IPChannel(#[from] binius_ip::channel::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] binius_ip::sumcheck::Error),
	#[error("Merkle channel error: {0}")]
	Merkle(#[from] crate::merkle_channel::Error),
}

/// Specification for an oracle to be committed in the IOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleSpec {
	/// Log2 of the message length (number of field elements).
	pub log_msg_len: usize,
	/// Whether the oracle is committed with zero-knowledge (hiding) masking.
	///
	/// ZK oracles interleave the message with a fresh mask and are folded by a shared masking
	/// challenge γ in the batched BaseFold opening; non-ZK oracles are committed without a mask.
	pub is_zk: bool,
}

impl OracleSpec {
	/// A non-ZK (unmasked) oracle of the given message length.
	pub const fn new(log_msg_len: usize) -> Self {
		Self {
			log_msg_len,
			is_zk: false,
		}
	}

	/// A ZK (masked, hiding) oracle of the given message length.
	pub const fn new_zk(log_msg_len: usize) -> Self {
		Self {
			log_msg_len,
			is_zk: true,
		}
	}
}

/// The length of the one oracle a round's oracles are committed as.
///
/// The oracles lie end to end, so this is the smallest power of two covering their total.
fn merged_log_msg_len(log_msg_lens: impl IntoIterator<Item = usize>) -> usize {
	let total_len: usize = log_msg_lens.into_iter().map(|n| 1usize << n).sum();
	log2_ceil_usize(total_len)
}

/// Every oracle an IOP commits, grouped into the rounds they are committed in.
///
/// A round is the run of oracles sent between two challenge samples.
///
/// A challenge can only be derived once the commitments before it are absorbed.
///
/// So a round closes the moment its challenge is drawn, and takes no further members.
///
/// ```text
/// recv, recv, sample, recv, sample, recv, recv, recv
/// \__________/        \__/          \______________/
///    round 0         round 1           round 2
/// ```
///
/// A flat spec list cannot say where those boundaries fall.
///
/// A caller that commits a whole round as one oracle needs them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OracleSchedule {
	/// Every oracle, in arrival order, across all rounds.
	specs: Vec<OracleSpec>,

	/// The exclusive end of each closed round, as an index into `specs`.
	///
	/// Round `r` spans `specs[ends[r - 1]..ends[r]]`, and round `0` starts at `0`.
	///
	/// Anything past the last entry is in the round still open.
	ends: Vec<usize>,
}

impl OracleSchedule {
	/// Creates an empty schedule.
	pub const fn new() -> Self {
		Self {
			specs: Vec::new(),
			ends: Vec::new(),
		}
	}

	/// Appends one oracle to the round currently open.
	pub fn push(&mut self, spec: OracleSpec) {
		self.specs.push(spec);
	}

	/// Closes the round currently open, so later oracles start a new one.
	///
	/// Does nothing when no oracle has arrived since the last close.
	///
	/// Calling it wherever a challenge could be drawn is therefore always safe.
	pub fn end_round(&mut self) {
		let open_start = self.ends.last().copied().unwrap_or(0);
		if self.specs.len() > open_start {
			self.ends.push(self.specs.len());
		}
	}

	/// Every oracle in the schedule, in arrival order, with round boundaries dropped.
	pub fn specs(&self) -> &[OracleSpec] {
		&self.specs
	}

	/// Consumes the schedule and returns every oracle, with round boundaries dropped.
	pub fn into_specs(self) -> Vec<OracleSpec> {
		self.specs
	}

	/// The number of closed rounds.
	pub const fn n_rounds(&self) -> usize {
		self.ends.len()
	}

	/// The oracles of each closed round, in commit order.
	pub fn rounds(&self) -> impl Iterator<Item = &[OracleSpec]> {
		let starts = iter::once(0).chain(self.ends.iter().copied());
		iter::zip(starts, self.ends.iter().copied()).map(|(start, end)| &self.specs[start..end])
	}

	/// One spec per round: the oracle that round's oracles are committed as.
	///
	/// This is the coarser list an underlying channel is configured with.
	///
	/// A round is masked as a whole, so its oracle is zero-knowledge if any member is.
	pub fn merged_specs(&self) -> Vec<OracleSpec> {
		self.rounds()
			.map(|round| OracleSpec {
				log_msg_len: merged_log_msg_len(round.iter().map(|spec| spec.log_msg_len)),
				is_zk: round.iter().any(|spec| spec.is_zk),
			})
			.collect()
	}
}

/// A boxed closure that evaluates a transparent MLE at a given point.
///
/// The closure receives the challenge point sampled during the opening and returns the evaluation
/// of the transparent polynomial's MLE there. It is `'static` and owns every value it reads,
/// sharing large data via `Rc`/`Arc`, so a channel that defers the opening can store it and
/// evaluate it later.
pub type TransparentEvalFn<Elem> = Box<dyn Fn(&[Elem]) -> Elem + 'static>;

/// Channel for IOP verifiers that extends the IP verifier channel with oracle operations.
///
/// In an IOP, the verifier can:
/// 1. Receive field elements from the prover via `recv_*` methods (inherited)
/// 2. Sample random challenges via `sample` (inherited)
/// 3. Receive oracle commitments from the prover
/// 4. Query oracles at specific positions and verify opening proofs
///
/// # Contract
///
/// The caller must call `recv_oracle()` exactly `remaining_oracle_specs().len()` times before
/// calling `verify_oracle_relation()`. The oracles must be received in order and match their
/// specifications.
pub trait IOPVerifierChannel<F: Field>: IPVerifierChannel<F, Elem: 'static> {
	type Oracle: Clone;

	/// Returns the specifications for the remaining oracles to be received.
	///
	/// This slice shrinks as oracles are received via `recv_oracle()`.
	fn remaining_oracle_specs(&self) -> &[OracleSpec];

	/// Receives an oracle commitment from the prover.
	///
	/// The caller describes the oracle being received: `log_msg_len` is the log2 of the message
	/// length, and `is_witness_dependent` is whether the oracle's contents depend on the witness.
	/// These let a channel record the oracle's [`OracleSpec`] rather than requiring the specs to be
	/// supplied up front. The resulting oracle is zero-knowledge iff the channel is configured for
	/// ZK *and* the oracle is witness-dependent — a non-witness-dependent oracle (e.g. a
	/// pre-indexed commitment to the wiring matrix for succinctness, a planned feature) is never
	/// masked.
	fn recv_oracle(
		&mut self,
		log_msg_len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error>;

	/// Queues one oracle linear relation to be opened.
	///
	/// Implementations may either verify the relation immediately, or queue it and defer the
	/// actual opening (masking + sumcheck + FRI) to `finish()`. Either way, the relation asserts
	/// that `<oracle_poly, transparent> = claim`. An oracle may carry any number of relations.
	///
	/// # Preconditions
	///
	/// * `oracle` must be a valid handle returned by `recv_oracle()`.
	fn verify_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: TransparentEvalFn<Self::Elem>,
		claim: Self::Elem,
	) -> Result<(), Error>;
}
