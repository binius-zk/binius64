// Copyright 2026 The Binius Developers

//! BaseFold compiler for IOP verifiers.

use std::borrow::BorrowMut;

use binius_field::BinaryField;
use binius_hash::HashSuite;
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{DeserializeBytes, FixedSizeSerializeBytes};
use digest::Output;

use crate::{
	basefold::{
		channel::BaseFoldVerifierChannel,
		layout::{RoundLayout, layout_rounds},
		libra::libra_layout,
	},
	channel::OracleSchedule,
	fri::{AritySelectionStrategy, FRIParams},
	merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
	merkle_tree::BinaryMerkleTreeScheme,
};

/// A compiler that creates BaseFold ZK verifier channels with precomputed parameters.
///
/// This compiler commits each round of the schedule as one oracle, laid out by [`layout_rounds`],
/// and builds a single combined FRI over the round oracles. ZK round oracles configure FRI
/// parameters for zero-knowledge mode (`log_msg_len + 1` as the message length and
/// `log_batch_size = 1`); non-ZK ones take a flexible batch size with no mask.
#[derive(Clone)]
pub struct BaseFoldVerifierCompiler<F>
where
	F: BinaryField,
{
	schedule: OracleSchedule,
	layouts: Vec<RoundLayout>,
	fri_params: FRIParams<F>,
}

impl<F> BaseFoldVerifierCompiler<F>
where
	F: BinaryField,
{
	/// Creates a new compiler with precomputed combined FRI parameters.
	///
	/// The `merkle_scheme` is consulted only for proof-size estimation while choosing the FRI
	/// parameters; it is not stored. Each round oracle's batch size is derived from its ZK flag: a
	/// ZK oracle fixes `log_batch_size = 1` (message ‖ equal-length mask), a non-ZK oracle takes a
	/// flexible batch size. Requires at least one oracle.
	pub fn new<H, Strategy>(
		merkle_scheme: &BinaryMerkleTreeScheme<F, H>,
		schedule: OracleSchedule,
		log_inv_rate: usize,
		n_test_queries: usize,
		_arity_strategy: &Strategy,
	) -> Self
	where
		H: HashSuite,
		Strategy: AritySelectionStrategy,
	{
		assert!(
			!schedule.specs().is_empty(),
			"BaseFoldVerifierCompiler requires at least one oracle spec"
		);
		// A masked opening commits its Libra oracle in a round of its own, after the schedule's.
		let mut layouts = layout_rounds(&schedule);
		layouts.extend(libra_layout(&layouts));
		let round_specs = layouts.iter().map(RoundLayout::spec).collect::<Vec<_>>();

		// The single combined FRI parameters over all oracles. `optimal_for_batch` chooses the fold
		// arities to minimize proof size, so `_arity_strategy` is not consulted here. It derives
		// each oracle's batch size from its ZK flag: ZK oracles fix `log_batch_size = 1` (message
		// ‖ mask), non-ZK oracles take a flexible batch size.
		let (fri_params, _) =
			FRIParams::optimal_for_batch(merkle_scheme, &round_specs, log_inv_rate, n_test_queries);

		Self {
			schedule,
			layouts,
			fri_params,
		}
	}

	/// Returns the schedule of oracles the channels commit.
	pub const fn oracle_schedule(&self) -> &OracleSchedule {
		&self.schedule
	}

	/// Returns the layout of each round oracle, the Libra oracle's last if the opening is masked.
	pub fn layouts(&self) -> &[RoundLayout] {
		&self.layouts
	}

	/// Returns a reference to the precomputed combined FRI parameters.
	pub const fn fri_params(&self) -> &FRIParams<F> {
		&self.fri_params
	}

	/// The dimension of the largest evaluation domain the combined FRI parameters need.
	///
	/// A prover builds its NTT domain context from this. The basis is not communicated because
	/// [`ReedSolomonCode`](binius_math::reed_solomon::ReedSolomonCode) fixes it: the Gao-Mateer
	/// basis of this dimension.
	pub fn max_log_domain_size(&self) -> usize {
		self.fri_params.rs_code().log_len()
	}

	/// Creates a ZK verifier channel over the given Merkle channel.
	///
	/// The returned channel drives all prover interaction through `channel`, opening oracles with
	/// this compiler's schedule, round layouts and combined FRI parameters. The caller constructs
	/// the Merkle channel, so it decides how commitments are received and verified.
	pub fn create_channel<Channel>(
		&self,
		channel: Channel,
	) -> BaseFoldVerifierChannel<'_, F, Channel>
	where
		Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
	{
		BaseFoldVerifierChannel::new(channel, &self.schedule, &self.layouts, &self.fri_params)
	}

	/// Creates a ZK verifier channel over a transcript, for the common case.
	///
	/// The transcript may be owned or mutably borrowed.
	/// It is wrapped in a [`VerifierMerkleTranscriptChannel`] for the given hash suite.
	/// That channel is then passed to [`Self::create_channel`].
	pub fn create_channel_from_transcript<H, Challenger_, T>(
		&self,
		transcript: T,
	) -> BaseFoldVerifierChannel<'_, F, VerifierMerkleTranscriptChannel<T, Challenger_, F, H>>
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
	use binius_field::Ghash128b;
	use binius_hash::StdHashSuite;

	use super::*;
	use crate::{channel::OracleSpec, fri::MinProofSizeStrategy};

	fn compile(specs: [OracleSpec; 2]) -> BaseFoldVerifierCompiler<Ghash128b> {
		BaseFoldVerifierCompiler::new(
			&BinaryMerkleTreeScheme::<Ghash128b, StdHashSuite>::new(),
			OracleSchedule::from_rounds(specs.map(|spec| [spec])),
			1,
			32,
			&MinProofSizeStrategy,
		)
	}

	#[test]
	fn a_masked_opening_commits_the_libra_oracle_last() {
		// `ℓ = 9`, so the table has `2^{⌈log2 9⌉ + 2} = 2^6` entries.
		let compiler = compile([OracleSpec::new(1 << 9), OracleSpec::new_zk(1 << 6)]);
		let layouts = compiler.layouts();
		assert_eq!(layouts.len(), 3);
		assert_eq!(layouts[2].spec(), OracleSpec::new_zk(1 << 6));
		assert_eq!(compiler.fri_params().input_oracles().len(), 3);
	}

	#[test]
	fn an_unmasked_opening_commits_no_libra_oracle() {
		let compiler = compile([OracleSpec::new(1 << 9), OracleSpec::new(1 << 6)]);
		assert_eq!(compiler.layouts().len(), 2);
		assert_eq!(compiler.fri_params().input_oracles().len(), 2);
	}
}
