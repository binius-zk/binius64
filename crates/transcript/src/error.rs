// Copyright 2024-2025 Irreducible Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Transcript is not empty, {remaining} bytes")]
	TranscriptNotEmpty { remaining: usize },
	#[error("Not enough bytes in the buffer")]
	NotEnoughBytes,
	#[error("Serialization error: {0}")]
	Serialization(#[from] binius_utils::SerializationError),
	/// A proof of work in the transcript does not meet the difficulty the verifier asked for.
	#[error("proof of work is short: {bits} zero bits required, transcript sampled {sampled}")]
	InsufficientWork {
		/// Number of zero bits the verifier required.
		bits: usize,
		/// Bits the transcript actually sampled after observing the prover's nonce.
		sampled: u32,
	},
}
