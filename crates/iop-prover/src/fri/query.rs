// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField};
use binius_math::FieldBuffer;
use tracing::instrument;

use crate::merkle_channel::MerkleIPProverChannel;

/// The prover counterpart of a `ProxTestOracle` (verifier side), producing the per-oracle query
/// openings.
pub trait ProxTestOracleProver<F: Field> {
	/// The Merkle commitment handle for the committed oracle.
	type Commitment;

	/// Sends the per-oracle batched query openings: the oracle's optimal Merkle layer once,
	/// followed by each queried coset's values and Merkle opening proof.
	fn open_queries<Channel>(&self, indices: &[usize], channel: &mut Channel)
	where
		Channel: MerkleIPProverChannel<F, Commitment = Self::Commitment>;
}

/// The [`ProxTestOracleProver`] for a [Brakedown]-style interleaved code proximity check.
///
/// [Brakedown]: <https://dl.acm.org/doi/10.1007/978-3-031-38545-2_7>
pub struct BrakedownOracleProver<P, C>
where
	P: PackedField,
{
	codeword: FieldBuffer<P>,
	commitment: C,
	/// log2 the lift factor (oracle padding). The committed codeword is virtually duplicated
	/// `2^log_lift` times to reach the common first-round length; a query at global index `k`
	/// opens the committed codeword at `k >> log_lift`. Zero when no lifting is needed.
	log_lift: usize,
}

impl<P, C> BrakedownOracleProver<P, C>
where
	P: PackedField,
{
	/// Constructs a new oracle prover wrapping a committed interleaved codeword.
	///
	/// `log_lift` is the oracle-padding lift factor (the committed codeword is virtually duplicated
	/// `2^log_lift` times to reach the common first-round length); pass `0` when no lifting is
	/// needed.
	pub const fn new(codeword: FieldBuffer<P>, commitment: C, log_lift: usize) -> Self {
		Self {
			codeword,
			commitment,
			log_lift,
		}
	}
}

impl<F, P, C> ProxTestOracleProver<F> for BrakedownOracleProver<P, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	type Commitment = C;

	fn open_queries<Channel>(&self, indices: &[usize], channel: &mut Channel)
	where
		Channel: MerkleIPProverChannel<F, Commitment = Self::Commitment>,
	{
		// When the oracle is lifted (`log_lift > 0`), each global query index is translated to the
		// committed codeword by dropping its low `log_lift` bits (the duplicated copies), mirroring
		// the verifier.
		let lifted_indices = indices
			.iter()
			.map(|&index| index >> self.log_lift)
			.collect::<Vec<_>>();
		channel.send_openings(&self.commitment, self.codeword.to_ref(), &lifted_indices);
	}
}

/// A [`ProxTestOracleProver`] bundling several separately committed [`BrakedownOracleProver`]s.
///
/// The bundled oracles all wrap interleaved codewords of the same length that are batched into a
/// single folded codeword during the first FRI fold. Their query openings are written sequentially,
/// one oracle's full decommitment after another, so the verifier reads each committed oracle's
/// advice in turn.
pub struct BatchBrakedownOracleProver<P, C>
where
	P: PackedField,
{
	oracles: Vec<BrakedownOracleProver<P, C>>,
}

impl<P, C> BatchBrakedownOracleProver<P, C>
where
	P: PackedField,
{
	/// Constructs a batch oracle prover from the per-commitment oracle provers.
	pub const fn new(oracles: Vec<BrakedownOracleProver<P, C>>) -> Self {
		Self { oracles }
	}
}

impl<F, P, C> ProxTestOracleProver<F> for BatchBrakedownOracleProver<P, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	type Commitment = C;

	fn open_queries<Channel>(&self, indices: &[usize], channel: &mut Channel)
	where
		Channel: MerkleIPProverChannel<F, Commitment = Self::Commitment>,
	{
		for oracle in &self.oracles {
			oracle.open_queries(indices, channel);
		}
	}
}

/// The [`ProxTestOracleProver`] for a FRI-style code proximity check.
pub struct FRIOracleProver<F, C>
where
	F: Field,
{
	codeword: FieldBuffer<F>,
	commitment: C,
	/// The base-2 log of the size of each coset opened from the committed oracle: the arity of
	/// the fold that consumes this codeword, which is also the commitment's leaf size.
	coset_log_size: usize,
}

impl<F, C> FRIOracleProver<F, C>
where
	F: Field,
{
	/// Constructs a new oracle prover wrapping a committed fold-round codeword.
	///
	/// `coset_log_size` is the base-2 log of the coset size of the fold that consumes this
	/// codeword, which must equal the commitment's leaf size.
	pub const fn new(codeword: FieldBuffer<F>, commitment: C, coset_log_size: usize) -> Self {
		Self {
			codeword,
			commitment,
			coset_log_size,
		}
	}

	/// The base-2 log of the size of each coset opened from the committed oracle.
	const fn coset_log_size(&self) -> usize {
		self.coset_log_size
	}
}

impl<F, C> ProxTestOracleProver<F> for FRIOracleProver<F, C>
where
	F: Field,
{
	type Commitment = C;

	fn open_queries<Channel>(&self, indices: &[usize], channel: &mut Channel)
	where
		Channel: MerkleIPProverChannel<F, Commitment = Self::Commitment>,
	{
		channel.send_openings(&self.commitment, self.codeword.to_ref(), indices);
	}
}

/// A prover for the FRI query phase.
///
/// This is a composition of [`ProxTestOracleProver`]s mirroring the verifier's `FRIQueryVerifier`:
/// a [`BatchBrakedownOracleProver`] for the codeword's interleaved reduction, then one
/// [`FRIOracleProver`] per fold arity for the subsequent reductions.
pub struct FRIQueryProver<F, P, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	codeword_oracle: BatchBrakedownOracleProver<P, C>,
	fri_oracles: Vec<FRIOracleProver<F, C>>,
}

impl<F, P, C> FRIQueryProver<F, P, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	/// Constructs a query prover from the per-oracle provers built during the fold phase.
	///
	/// `codeword_oracle` is the [`BatchBrakedownOracleProver`] for the originally committed
	/// interleaved codeword(s), and `fri_oracles` holds one [`FRIOracleProver`] per fold arity. The
	/// terminal codeword is sent in full and therefore is not represented here.
	pub const fn new(
		codeword_oracle: BatchBrakedownOracleProver<P, C>,
		fri_oracles: Vec<FRIOracleProver<F, C>>,
	) -> Self {
		Self {
			codeword_oracle,
			fri_oracles,
		}
	}

	/// Number of oracles sent during the fold rounds.
	pub const fn n_oracles(&self) -> usize {
		1 + self.fri_oracles.len()
	}

	/// Proves the FRI challenge queries, batched per oracle.
	///
	/// For each committed oracle (the codeword first, then each fold-round oracle excluding the
	/// terminal codeword) this sends the oracle's optimal Merkle layer once, followed by the coset
	/// opening for each query index. This per-oracle batched layout matches the verifier, which
	/// receives each oracle's layer and then all of its query openings together.
	///
	/// ## Arguments
	///
	/// * `indices` - the sampled query indices into the original codeword domain
	#[instrument(skip_all, name = "fri::FRIQueryProver::prove_queries", level = "debug")]
	pub fn prove_queries<Channel>(&self, indices: &[usize], channel: &mut Channel)
	where
		Channel: MerkleIPProverChannel<F, Commitment = C>,
	{
		self.codeword_oracle.open_queries(indices, channel);

		// Each subsequent oracle indexes the previous virtual oracle, so shift the query indices
		// right by the round's arity before opening it.
		let mut indices = indices.to_vec();
		for fri_oracle in &self.fri_oracles {
			for index in &mut indices {
				*index >>= fri_oracle.coset_log_size();
			}
			fri_oracle.open_queries(&indices, channel);
		}
	}
}
