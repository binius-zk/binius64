// Copyright 2026 The Binius Developers

//! Signature of knowledge support.
//!
//! A Binius64 proof is a non-interactive proof of knowledge: an interactive protocol made
//! non-interactive with the Fiat-Shamir transform. Such a proof can be converted into a
//! *signature of knowledge* (SoK) over a message `m` by binding `m` into the Fiat-Shamir
//! transcript before any other data is observed. The verifier then needs the message in order
//! to recompute the same transcript and check the proof; a proof produced for one message does
//! not verify against any other message.
//!
//! Rather than observing the message bytes directly, we observe the *hash* of the message,
//! computed with the same hash function the Merkle commitment scheme uses for its leaves
//! ([`HashSuite::LeafHash`]). This keeps the amount of data fed into the challenger fixed-size
//! and independent of the message length, and insulates the construction from future changes to
//! the transcript API.
//!
//! The message (and its hash) are *observed* into the transcript, not written to the proof tape:
//! they affect the Fiat-Shamir challenges but are not part of the serialized proof. The verifier
//! must supply the message out of band, exactly as a signature verifier supplies the signed
//! message.
//!
//! Signing is only exposed on the zero-knowledge prover and verifier (`ZKProver::prove_sig` and
//! [`ZKVerifier::verify_sig`](crate::zk_config::ZKVerifier::verify_sig)): a signature of knowledge
//! should not reveal the witness it is signed under, and only the ZK configuration hides the
//! witness. The [`observe_message`] helper below is generic over the hash suite and is shared by
//! both.
//!
//! # Security analysis
//!
//! - **Binding / non-malleability.** The message hash is observed *first*, before the public input
//!   and before the prover's witness commitment. Every subsequent Fiat-Shamir challenge therefore
//!   depends on the message. An adversary who is given a valid proof for message `m` cannot maul it
//!   into a valid proof for a different message `m'`: changing the message changes the first
//!   challenger absorption and hence every sampled challenge, so the rest of the (replayed) proof
//!   no longer satisfies the verifier's checks. This is the standard "strong Fiat-Shamir"
//!   requirement that the full statement — here extended with the message — be bound into the
//!   transcript.
//!
//! - **Reduction to collision resistance.** Because we bind `H(m)` rather than `m`, the binding is
//!   only as strong as the collision resistance of `H = HashSuite::LeafHash`. If an adversary finds
//!   `m != m'` with `H(m) = H(m')`, a signature on `m` also verifies for `m'`. The leaf hashes used
//!   by the shipped suites (SHA-256) are collision resistant, so this is not a practical concern;
//!   it is the same assumption the Merkle commitment already relies on.
//!
//! - **Knowledge soundness is preserved.** Observing extra public data at the start of the
//!   transcript does not weaken the soundness of the underlying argument: it is equivalent to
//!   running the original protocol on a statement augmented with a public message-hash that does
//!   not participate in any constraint. A forger who cannot satisfy the constraint system still
//!   cannot, for any message.
//!
//! - **Zero knowledge is preserved.** The message hash is public input to the Fiat-Shamir transform
//!   and never touches the witness, so binding it leaks nothing about the witness in the ZK
//!   configuration.
//!
//! - **Prover/verifier must agree on the mode.** Whether a message is bound, and what it is, is not
//!   encoded in the proof. A verifier that checks a SoK proof without the message (or with the
//!   wrong message) samples different challenges and rejects. Callers must therefore agree out of
//!   band on whether a proof is a plain proof of knowledge or a signature of knowledge. Note that
//!   an empty message (`Some(&[])`, which binds `H("")`) is distinct from no message at all
//!   (`None`, which binds nothing).

use binius_hash::binary_merkle_tree::HashSuite;
use binius_transcript::{BufMut, TranscriptWriter};
use digest::Digest;

/// Observes the hash of a signed message into a transcript, turning a proof of knowledge into a
/// signature of knowledge.
///
/// The message is hashed with `H::LeafHash` (the Merkle scheme's leaf hash) and the resulting
/// digest is observed into the transcript via the provided observing writer. Both the prover and
/// the verifier must call this with the same message *before any other transcript interaction*.
///
/// The `writer` must be an *observing* writer (obtained from `transcript.observe()`), so the
/// digest is mixed into the Fiat-Shamir state without being written to the proof tape.
pub fn observe_message<H, B>(writer: &mut TranscriptWriter<B>, message: &[u8])
where
	H: HashSuite,
	B: BufMut,
{
	let digest = <H::LeafHash as Digest>::digest(message);
	writer.write_bytes(digest.as_ref());
}
