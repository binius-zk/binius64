// Copyright 2026 The Binius Developers

//! Ligerito compiler for IOP verifiers.

use std::{borrow::BorrowMut, marker::PhantomData};

use binius_field::BinaryField;
use binius_hash::binary_merkle_tree::HashSuite;
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, FixedSizeSerializeBytes};
use digest::Output;

use super::{LevelZeroBatch, LigeritoParams, channel::LigeritoVerifierChannel};
use crate::{
	channel::OracleSpec,
	merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
	merkle_tree::MerkleTreeScheme,
	soundness::{Grinding, SoundnessRegime},
};

/// A compiler that creates Ligerito verifier channels from a precomputed ladder.
///
/// The ladder is chosen once, for the longest oracle the channel opens.
/// Every channel the compiler creates reuses it, and every oracle shares its column count.
///
/// An oracle asking for zero knowledge is committed with a mask interleaved beside its message.
/// That widens level 0's row union, which the ladder has to be able to afford.
#[derive(Debug, Clone)]
pub struct LigeritoVerifierCompiler<F> {
	/// The oracles every channel this compiler makes will open, in the order they arrive.
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
	/// * `oracle_specs` is non-empty.
	/// * The longest message is the ladder's message length.
	/// * The batch these oracles form reaches the target, unless the ladder misses it alone.
	pub fn new(oracle_specs: Vec<OracleSpec>, params: LigeritoParams) -> Self {
		assert!(
			!oracle_specs.is_empty(),
			"precondition: a Ligerito compiler serves at least one oracle"
		);
		assert_eq!(
			oracle_specs
				.iter()
				.map(|spec| spec.log_msg_len)
				.max()
				.expect("oracle_specs is non-empty"),
			params.log_msg_len(),
			"precondition: the longest oracle's message length must be the ladder's message length"
		);
		// Batching widens level 0's row union, which lowers a ceiling no query count can raise.
		// So does a mask, which is one more interleaved lane at level 0 and nothing else.
		// A ladder sized for one clear message can therefore miss a target it otherwise clears,
		// and the miss is silent: the unbatched figure keeps reporting the number it does not pay.
		// Whether the ladder clears its target on its own is `LigeritoParams`'s business, so this
		// refuses only the shapes that cause the shortfall.
		let target = params.security_bits() as f64;
		let batch = LevelZeroBatch::from_specs(&oracle_specs);
		let batched = params.batched_achieved_security_bits(F::N_BITS, batch);
		assert!(
			batched >= target || params.achieved_security_bits(F::N_BITS) < target,
			"precondition: {} oracles, masked {}, reach {batched:.2} bits against a target of \
			 {}",
			batch.n_oracles(),
			batch.is_masked(),
			params.security_bits()
		);

		Self {
			oracle_specs,
			params,
			_marker: PhantomData,
		}
	}

	/// Creates a compiler whose ladder is the proof-size-minimizing one for the longest oracle.
	///
	/// Level 0's rate is pinned by the caller, because level 0's encoding dominates prover time.
	/// The deeper levels are small, so the search is free to drop their rate as far as it likes.
	///
	/// The search input is the longest message, since every oracle shares level 0's column count.
	/// A shorter one then simply carries fewer interleaved lanes.
	///
	/// `None` means no ladder over that message reaches the security target.
	///
	/// `grinding` is what the ladder will pay per level.
	/// The search prices it rather than assuming it away.
	/// Pass [`Grinding::NONE`] for a transcript with no proof of work in it.
	///
	/// The search prices one clear message, so a batch or a mask is checked against it afterwards.
	///
	/// ## Preconditions
	///
	/// * `oracle_specs` is non-empty.
	/// * `l0_log_inv_rate` is a usable inverse rate and `security_bits` is positive.
	/// * The batch these oracles form reaches the target, unless the ladder misses it alone.
	pub fn optimal<MerkleScheme>(
		merkle_scheme: &MerkleScheme,
		oracle_specs: Vec<OracleSpec>,
		l0_log_inv_rate: usize,
		regime: SoundnessRegime,
		security_bits: usize,
		grinding: Grinding,
	) -> Option<Self>
	where
		MerkleScheme: MerkleTreeScheme<F>,
	{
		assert!(
			!oracle_specs.is_empty(),
			"precondition: a Ligerito compiler serves at least one oracle"
		);

		// Level 0's shape is shared, so the longest message is what the ladder is searched over.
		let (params, _proof_size) = LigeritoParams::optimal_ladder::<F, _>(
			merkle_scheme,
			oracle_specs
				.iter()
				.map(|spec| spec.log_msg_len)
				.max()
				.expect("oracle_specs is non-empty"),
			l0_log_inv_rate,
			regime,
			security_bits,
			grinding,
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
			Grinding::NONE,
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
			Grinding::NONE,
		);
		assert!(compiler.is_none());
	}

	#[test]
	fn several_oracles_share_the_ladder_of_the_longest() {
		// Fixture state: level 0 is 2^6 columns by 2^2 lanes, so the ladder commits 2^8 elements.
		//
		//     2^8 elements -> 2^2 lanes   the oracle the ladder was sized for
		//     2^7 elements -> 2^1 lanes   one lane fewer over the same codeword
		//     2^5 elements -> 2^0 lanes   a single lane, zero-padded out to 2^6 columns
		let specs = vec![OracleSpec::new(8), OracleSpec::new(7), OracleSpec::new(5)];
		let compiler = LigeritoVerifierCompiler::<B128>::new(specs.clone(), params());

		assert_eq!(compiler.oracle_specs(), &specs);
		for (spec, log_lanes) in std::iter::zip(&specs, [2, 1, 0]) {
			let shape = compiler.params().level_zero_shape(spec);
			assert_eq!(shape.log_lanes, log_lanes);
			// One codeword length, so one set of query positions serves every oracle.
			assert_eq!(shape.log_codeword_len(), 7);
		}
	}

	#[test]
	#[should_panic(expected = "precondition: a Ligerito compiler serves at least one oracle")]
	fn no_oracle_at_all_is_refused() {
		// A ladder with nothing to open is a mis-specified channel rather than a trivial one.
		LigeritoVerifierCompiler::<B128>::new(Vec::new(), params());
	}

	#[test]
	fn a_zero_knowledge_oracle_commits_one_extra_lane() {
		// Fixture state: level 0 is 2^6 columns by 2^2 lanes over a 2^8 message.
		//
		//     clear  : 2^2 lanes, one per quarter of the message
		//     masked : 2^3 lanes, the extra one holding a mask of the message's own length
		//
		// The codeword length is untouched, so one query set still serves clear and masked alike.
		let compiler = LigeritoVerifierCompiler::<B128>::new(vec![OracleSpec::new_zk(8)], params());
		let shape = compiler
			.params()
			.level_zero_shape(&compiler.oracle_specs()[0]);
		assert_eq!(shape.log_lanes, 3);
		assert_eq!(shape.log_codeword_len(), 7);
	}

	#[test]
	#[should_panic(expected = "precondition: the longest oracle's message length must be the")]
	fn a_ladder_over_a_different_message_length_is_refused() {
		// The ladder commits 2^8 elements and the longest oracle claims 2^9.
		// So level 0's codeword would not encode the buffer the channel is handed.
		LigeritoVerifierCompiler::<B128>::new(vec![OracleSpec::new(9)], params());
	}

	#[test]
	#[should_panic(expected = "precondition: the longest oracle's message length must be the")]
	fn a_ladder_wider_than_every_oracle_is_refused() {
		// The ladder commits 2^8 elements and no oracle reaches that.
		// Level 0 would then fold lanes that none of them has, wasting a round on nothing.
		LigeritoVerifierCompiler::<B128>::new(
			vec![OracleSpec::new(7), OracleSpec::new(6)],
			params(),
		);
	}

	/// The ladder a 120-bit target admits over a 2^24 message, where the ceiling binds.
	///
	/// At the shipped 96-bit target the query term binds instead.
	/// Neither batching nor masking shows there, so a rejection test needs this shape.
	fn ceiling_bound_params() -> LigeritoParams {
		let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		let (regime, _) = SoundnessRegime::optimal_unique_decoding(120, 24, 1, 128)
			.expect("120 bits is reachable with a constant loss");
		let (params, _) =
			LigeritoParams::optimal_ladder::<B128, _>(&scheme, 24, 1, regime, 120, Grinding::NONE)
				.expect("a 120-bit ladder exists for one message");

		// One clear oracle clears 120, so any shortfall below is caused by the shape under test.
		assert!(params.achieved_security_bits(128) >= 120.0);
		params
	}

	/// A batch that would miss the target the ladder was sized for must be refused.
	///
	/// Each doubling of the oracle count takes one bit off the ceiling.
	/// A ladder sized for one message misses its target once several share it.
	#[test]
	#[should_panic(expected = "oracles, masked false, reach")]
	fn a_batch_that_would_miss_the_target_is_refused() {
		let params = ceiling_bound_params();
		let spec = OracleSpec::new(params.log_msg_len());
		LigeritoVerifierCompiler::<B128>::new(vec![spec, spec], params);
	}

	/// A mask that would miss the same target must be refused by the same guard.
	///
	/// A mask is one more interleaved lane at level 0.
	/// So it widens the row union exactly as a second oracle does.
	/// Letting it through would report a number that union does not deliver.
	#[test]
	#[should_panic(expected = "oracles, masked true, reach")]
	fn a_mask_that_would_miss_the_target_is_refused() {
		let params = ceiling_bound_params();
		let spec = OracleSpec::new_zk(params.log_msg_len());
		LigeritoVerifierCompiler::<B128>::new(vec![spec], params);
	}

	/// The same mask is accepted once a challenge grind pays for the lane it adds.
	///
	/// The grind raises the ceiling one bit per bit ground, and the mask costs exactly one.
	/// So the two cancel and the masked oracle reaches the target the clear one reached.
	#[test]
	fn a_ground_ladder_affords_the_mask() {
		let params = ceiling_bound_params().with_grinding(Grinding::new(1, 0));
		let spec = OracleSpec::new_zk(params.log_msg_len());
		LigeritoVerifierCompiler::<B128>::new(vec![spec], params);
	}
}
