// Copyright 2026 The Binius Developers

//! Ligerito compiler for IOP verifiers.

use std::{borrow::BorrowMut, marker::PhantomData};

use binius_field::BinaryField;
use binius_hash::binary_merkle_tree::HashSuite;
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, FixedSizeSerializeBytes};
use digest::Output;

use super::{LigeritoParams, channel::LigeritoVerifierChannel};
use crate::{
	channel::OracleSpec,
	merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
	merkle_tree::MerkleTreeScheme,
	soundness::SoundnessRegime,
};

/// A compiler that creates Ligerito verifier channels from a precomputed ladder.
///
/// The ladder is chosen once, for the one oracle the channel opens.
/// Every channel the compiler creates reuses it.
///
/// Ligerito commits no mask, so the oracle it opens is never zero-knowledge.
#[derive(Debug, Clone)]
pub struct LigeritoVerifierCompiler<F> {
	/// The one oracle every channel this compiler makes will open.
	oracle_specs: Vec<OracleSpec>,
	/// The ladder each of those openings runs down.
	params: LigeritoParams,
	/// Pins the field the created channels open over, which the ladder itself does not name.
	/// Ties the field to the compiler without storing a value of it.
	_marker: PhantomData<F>,
}

impl<F> LigeritoVerifierCompiler<F>
where
	F: BinaryField,
{
	/// Creates a compiler from an explicit ladder.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` holds exactly one spec.
	/// * That spec is not zero-knowledge.
	/// * Its message length is the ladder's message length.
	pub fn new(oracle_specs: Vec<OracleSpec>, params: LigeritoParams) -> Self {
		assert_eq!(
			oracle_specs.len(),
			1,
			"precondition: a Ligerito compiler serves exactly one oracle, got {}",
			oracle_specs.len()
		);
		assert!(
			!oracle_specs[0].is_zk,
			"precondition: Ligerito commits no mask, so a zero-knowledge oracle cannot be opened"
		);
		assert_eq!(
			oracle_specs[0].log_msg_len,
			params.log_msg_len(),
			"precondition: the oracle's message length must be the ladder's message length"
		);

		Self {
			oracle_specs,
			params,
			_marker: PhantomData,
		}
	}

	/// Creates a compiler whose ladder is the proof-size-minimizing one for the oracle.
	///
	/// Level 0's rate is pinned by the caller, because level 0's encoding dominates prover time.
	/// The deeper levels are small, so the search is free to drop their rate as far as it likes.
	///
	/// `None` means no ladder over this message reaches the security target.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` holds exactly one spec.
	/// * That spec is not zero-knowledge.
	/// * `l0_log_inv_rate` is a usable inverse rate and `security_bits` is positive.
	pub fn optimal<MerkleScheme>(
		merkle_scheme: &MerkleScheme,
		oracle_specs: Vec<OracleSpec>,
		l0_log_inv_rate: usize,
		regime: SoundnessRegime,
		security_bits: usize,
	) -> Option<Self>
	where
		MerkleScheme: MerkleTreeScheme<F>,
	{
		assert_eq!(
			oracle_specs.len(),
			1,
			"precondition: a Ligerito compiler serves exactly one oracle, got {}",
			oracle_specs.len()
		);

		// The ladder is searched over the one message it commits, so its size is the search input.
		let (params, _proof_size) = LigeritoParams::optimal_ladder::<F, _>(
			merkle_scheme,
			oracle_specs[0].log_msg_len,
			l0_log_inv_rate,
			regime,
			security_bits,
		)?;

		Some(Self::new(oracle_specs, params))
	}

	/// Returns a reference to the oracle specifications.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs
	}

	/// Returns a reference to the precomputed ladder.
	pub const fn params(&self) -> &LigeritoParams {
		&self.params
	}

	/// The dimension of the largest evaluation domain the ladder needs.
	///
	/// A prover builds its additive transform from this.
	/// The basis is not communicated because the Reed-Solomon code fixes it.
	/// It is the Gao-Mateer basis of this dimension.
	pub fn max_log_domain_size(&self) -> usize {
		self.params.max_log_codeword_len()
	}

	/// Creates a verifier channel over the given Merkle channel.
	///
	/// The returned channel drives all prover interaction through the one it is given.
	/// So the caller decides how commitments are received and verified.
	pub fn create_channel<Channel>(
		&self,
		channel: Channel,
	) -> LigeritoVerifierChannel<'_, F, Channel>
	where
		Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
	{
		LigeritoVerifierChannel::new(channel, &self.oracle_specs, &self.params)
	}

	/// Creates a verifier channel over a transcript, for the common case.
	///
	/// The transcript may be owned or mutably borrowed.
	/// It is wrapped in a Merkle transcript channel for the given hash suite.
	/// That channel is then handed to the general constructor.
	pub fn create_channel_from_transcript<H, Challenger_, T>(
		&self,
		transcript: T,
	) -> LigeritoVerifierChannel<'_, F, VerifierMerkleTranscriptChannel<T, Challenger_, F, H>>
	where
		F: FixedSizeSerializeBytes,
		H: HashSuite,
		Challenger_: Challenger,
		T: BorrowMut<VerifierTranscript<Challenger_>>,
		Output<H::LeafHash>: DeserializeBytes,
	{
		self.create_channel(VerifierMerkleTranscriptChannel::new(transcript))
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b as B128;
	use binius_hash::StdHashSuite;

	use super::*;
	use crate::{ligerito::LigeritoLevel, merkle_tree::BinaryMerkleTreeScheme};

	/// A two-level ladder over a 2^8 message: 2^6 columns and 4 lanes, then 2^5 columns and 2.
	fn params() -> LigeritoParams {
		LigeritoParams::new(
			vec![
				LigeritoLevel {
					log_msg_cols: 6,
					log_lanes: 2,
					log_inv_rate: 1,
					n_queries: 5,
				},
				LigeritoLevel {
					log_msg_cols: 5,
					log_lanes: 1,
					log_inv_rate: 2,
					n_queries: 5,
				},
			],
			SoundnessRegime::UniqueDecoding,
			32,
		)
	}

	#[test]
	fn an_explicit_ladder_is_kept_as_given() {
		let compiler = LigeritoVerifierCompiler::<B128>::new(vec![OracleSpec::new(8)], params());

		// Fixture state: the ladder is 2^6 columns then 2^5, at inverse rates 2 and 4.
		//
		//     level 0: 2^(6 + 1) = 2^7 codeword positions
		//     level 1: 2^(5 + 2) = 2^7 codeword positions
		//
		// One transform covers both, and 2^7 is the domain it must reach.
		assert_eq!(compiler.max_log_domain_size(), 7);
		assert_eq!(compiler.params().log_msg_len(), 8);
		assert_eq!(compiler.oracle_specs(), &[OracleSpec::new(8)]);
	}

	#[test]
	fn the_searched_ladder_commits_the_oracle_it_was_given() {
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		let compiler = LigeritoVerifierCompiler::<B128>::optimal(
			&scheme,
			vec![OracleSpec::new(20)],
			1,
			SoundnessRegime::UniqueDecoding,
			96,
		)
		.expect("a 2^20 message at rate 1/2 admits a ladder at 96 bits");

		// The search is free in every dimension except the one the oracle pins.
		// The ladder must commit exactly the message the channel will hand it.
		assert_eq!(compiler.params().log_msg_len(), 20);
		// Level 0's rate is the caller's, since level 0's encoding dominates prover time.
		assert_eq!(compiler.params().levels()[0].log_inv_rate, 1);
	}

	#[test]
	fn no_ladder_reaches_the_target_over_a_message_this_small() {
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		// A 2^4 message at rate 1/2 has 2^5 codeword positions at level 0.
		// That is far fewer than a 96-bit target needs, so no ladder over it is coherent.
		let compiler = LigeritoVerifierCompiler::<B128>::optimal(
			&scheme,
			vec![OracleSpec::new(4)],
			1,
			SoundnessRegime::UniqueDecoding,
			96,
		);
		assert!(compiler.is_none());
	}

	#[test]
	#[should_panic(expected = "precondition: a Ligerito compiler serves exactly one oracle")]
	fn two_oracles_are_refused() {
		// Batching several committed oracles into one ladder is not implemented.
		// So the compiler refuses the shape rather than silently opening only the first.
		LigeritoVerifierCompiler::<B128>::new(
			vec![OracleSpec::new(8), OracleSpec::new(8)],
			params(),
		);
	}

	#[test]
	#[should_panic(expected = "precondition: Ligerito commits no mask")]
	fn a_zero_knowledge_oracle_is_refused() {
		// A masked oracle interleaves its message with a mask, which the ladder has no place for.
		// Its residual would reach the verifier in the clear regardless.
		LigeritoVerifierCompiler::<B128>::new(vec![OracleSpec::new_zk(8)], params());
	}

	#[test]
	#[should_panic(expected = "precondition: the oracle's message length must be the ladder's")]
	fn a_ladder_over_a_different_message_length_is_refused() {
		// The ladder commits 2^8 elements and the oracle claims 2^9.
		// So level 0's codeword would not encode the buffer the channel is handed.
		LigeritoVerifierCompiler::<B128>::new(vec![OracleSpec::new(9)], params());
	}
}
