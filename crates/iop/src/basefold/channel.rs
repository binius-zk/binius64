// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP verifier channel.

use binius_core::word::Word;
use binius_field::{BinaryField, FieldOps, util::powers};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_math::{
	line::extrapolate_line,
	multilinear::eq::{eq_ind, eq_ind_partial_eval_scalars, eq_ind_zero},
	univariate::evaluate_univariate,
};
use itertools::izip;

use crate::{
	basefold::{self, layout::RoundLayout},
	channel::{Error, IOPVerifierChannel, OracleSchedule, OracleSpec, TransparentEvalFn},
	fri::FRIParams,
	merkle_channel::MerkleIPVerifierChannel,
};

/// Oracle handle returned by [`BaseFoldVerifierChannel::recv_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldOracle {
	/// The round whose oracle this one is committed in.
	round: usize,
	/// The oracle's block within its round oracle, in units of its own length.
	block_offset: usize,
	/// The oracle's own `log_msg_len`.
	log_len: usize,
}

/// One round oracle's operand in a queued relation.
struct Operand<Elem> {
	/// The round whose oracle this is an operand on.
	round: usize,
	/// Evaluates the transparent multilinear, over the round oracle's variables, at the point the
	/// opening reduces to.
	transparent: TransparentEvalFn<Elem>,
}

/// A relation queued by [`IOPVerifierChannel::verify_multi_oracle_relation`]: its claim and the
/// operands whose inner products with their oracles sum to it.
struct Relation<Elem> {
	claim: Elem,
	operands: Vec<Operand<Elem>>,
}

/// A verifier channel that uses ZK BaseFold for all oracle commitments and openings.
///
/// Each round of the schedule is committed as one round oracle, laid out by
/// [`layout_rounds`](basefold::layout::layout_rounds), whose commitment arrives with the round's
/// last oracle. The opening runs over the round oracles.
///
/// # Type Parameters
///
/// - `'a`: Lifetime for borrowed references
/// - `F`: The binary field type
/// - `Channel`: The Merkle channel carrying all prover interaction
pub struct BaseFoldVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	/// The Merkle channel carrying all prover interaction: field elements, challenges,
	/// commitments, and openings.
	channel: Channel,
	schedule: &'a OracleSchedule,
	layouts: &'a [RoundLayout],
	fri_params: &'a FRIParams<F>,
	/// One commitment per round committed so far, so its length is also the open round's index.
	oracle_commitments: Vec<Channel::Commitment>,
	/// How many oracles have been received so far.
	n_received: usize,
	/// How many of the open round's oracles have been received so far.
	n_in_round: usize,
	/// The relations queued by [`IOPVerifierChannel::verify_multi_oracle_relation`], opened
	/// together in [`Self::finish`].
	relations: Vec<Relation<Channel::Elem>>,
}

impl<'a, F, Channel> BaseFoldVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	/// Creates a new BaseFold ZK verifier channel over a Merkle channel from precomputed FRI
	/// parameters.
	///
	/// The FRI parameters should already account for ZK (log_batch_size = 1, doubled message
	/// length).
	pub const fn new(
		channel: Channel,
		schedule: &'a OracleSchedule,
		layouts: &'a [RoundLayout],
		fri_params: &'a FRIParams<F>,
	) -> Self {
		Self {
			channel,
			schedule,
			layouts,
			fri_params,
			oracle_commitments: Vec::new(),
			n_received: 0,
			n_in_round: 0,
			relations: Vec::new(),
		}
	}

	/// Receives the commitment to the next round oracle, the one at index
	/// `oracle_commitments.len()`.
	fn recv_round_commitment(&mut self) -> Result<(), Error> {
		let index = self.oracle_commitments.len();
		let layout = &self.layouts[index];
		// Receive the commitment with its Merkle tree shape, matching the prover-side commit: the
		// oracle's codeword has dimension `log_dim - log_lift` and one interleaved coset of
		// `2^log_batch_size` scalars per leaf.
		let fri_oracle = &self.fri_params.input_oracles()[index];
		let depth = (self.fri_params.rs_code().log_dim() - fri_oracle.log_lift)
			+ self.fri_params.rs_code().log_inv_rate();

		// The committed message length implied by this shape is `log_batch_size + depth -
		// log_inv_rate`; it must cover the round oracle plus, for a ZK one, the equal-length
		// interleaved mask.
		assert_eq!(
			fri_oracle.log_batch_size() + depth - self.fri_params.rs_code().log_inv_rate(),
			layout.log_len + usize::from(layout.is_zk),
			"invariant: the FRI commitment shape must be consistent with the round oracle's length"
		);

		let commitment = self
			.channel
			.recv_merkle_commitment(1 << fri_oracle.log_batch_size(), depth)?;

		self.oracle_commitments.push(commitment);

		Ok(())
	}

	/// Consumes the channel and verifies the single combined opening over **all** committed
	/// oracles.
	///
	/// All oracle relations queued by
	/// [`verify_multi_oracle_relation`](IOPVerifierChannel::verify_multi_oracle_relation) across
	/// every call are processed here in one batch: batching into one relation, then one BaseFold
	/// sumcheck opening over every committed oracle (in oracle-index order). Because the whole
	/// opening is deferred to this point, every oracle is committed, so the precomputed combined
	/// `FRIParams` (`optimal_for_batch` over all oracle specs) serves the opening.
	///
	/// Returns the Merkle channel, so a caller can still reach what it accumulated.
	pub fn finish(self) -> Result<Channel, Error> {
		let n_remaining = self.schedule.specs().len() - self.n_received;
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining",);

		let Self {
			mut channel,
			schedule: _,
			layouts,
			fri_params,
			oracle_commitments,
			n_received: _,
			n_in_round: _,
			relations,
		} = self;

		if !relations.is_empty() {
			verify_batch_zk_basefold(
				&mut channel,
				layouts,
				fri_params,
				&oracle_commitments,
				relations,
			)?;
		}

		Ok(channel)
	}
}

/// Verifies the combined ZK BaseFold opening over all committed round oracles.
///
/// This drives `channel` — the Merkle channel taken from the destructured
/// [`BaseFoldVerifierChannel`] — through its [`MerkleIPVerifierChannel`] interface, in the steps of
/// binius.xyz#128, "Batched ZK BaseFold opening":
///
/// 1. Batch the queued relations into `T_r = Σ_j α^j t_{j,r}` and `s = Σ_j α^j s_j`.
/// 2. If some round is masked, receive the mask claim `s' = Σ_r <ω_r, T_r>`.
/// 3. Sample the mask fold γ and take the target `lerp(γ, s, s')`. An unmasked round has `ω_r =
///    M_r`, so folding it is a no-op.
/// 4. Run one BaseFold sumcheck on `<G, T>` with `G(X, Z) = Σ_r M^γ_r↑(X) · eq(r, Z)` and `T(X, Z)
///    = Σ_r T_r↑(X) · eq(r, Z)`. The final point `(ρ, ζ)` has `T(ρ, ζ)` evaluated here from the
///    operand closures.
///
/// With no masked round there is no mask claim and no γ, and the target is `s`.
///
/// `T_r↑` is `T_r` placed at the origin of the opening's variables: its value at `ρ` is that of
/// `T_r` at the first `ℓ_r` coordinates, times `eq(0, ρ_{≥ℓ_r})`. The sum of `M_r↑ · T_r↑` over
/// `X` is then `<M_r, T_r>` however `M_r↑` is lifted and repeated by the FRI.
fn verify_batch_zk_basefold<F, Channel>(
	channel: &mut Channel,
	layouts: &[RoundLayout],
	fri_params: &FRIParams<F>,
	oracle_commitments: &[Channel::Commitment],
	relations: Vec<Relation<Channel::Elem>>,
) -> Result<(), Error>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	assert_eq!(layouts.len(), oracle_commitments.len());

	// `ℓ = max_r ℓ_r`, the variable count of the opening.
	let n_vars = layouts
		.iter()
		.map(|layout| layout.log_len)
		.max()
		.expect("at least one oracle");

	// Batch the relations at one sampled μ: the combined claim is `s = Σ_j μ^j · s_j`.
	let mu = channel.sample();
	let claims = relations
		.iter()
		.map(|relation| relation.claim.clone())
		.collect::<Vec<_>>();
	let claim = evaluate_univariate(&claims, &mu);

	// A masked opening receives the mask claim and folds with γ.
	let (target, gamma) = if layouts.iter().any(|layout| layout.is_zk) {
		let s_prime = channel.recv_one()?;
		let gamma = channel.sample();
		(extrapolate_line(claim, s_prime, gamma.clone()), Some(gamma))
	} else {
		(claim, None)
	};

	// `T_r(ρ) = Σ_j μ^j Σ_{operands of relation j on round r} t(ρ_{<ℓ_r})`.
	let multiplicand_eval = move |rho: &[Channel::Elem], zeta: &[Channel::Elem]| {
		let mut round_evals = vec![Channel::Elem::zero(); layouts.len()];
		for (mu_j, relation) in izip!(powers(mu), &relations) {
			for operand in &relation.operands {
				let n_r = layouts[operand.round].log_len;
				round_evals[operand.round] += mu_j.clone() * (operand.transparent)(&rho[..n_r]);
			}
		}

		let eq_tensor = eq_ind_partial_eval_scalars(zeta);
		izip!(round_evals, layouts, eq_tensor)
			.map(|(eval, layout, eq_r)| eq_r * eval * eq_ind_zero(&rho[layout.log_len..]))
			.sum::<Channel::Elem>()
	};

	basefold::verify_sumcheck_basefold(
		fri_params,
		oracle_commitments,
		target,
		n_vars,
		gamma,
		multiplicand_eval,
		channel,
	)?;

	Ok(())
}

impl<F, Channel> IPVerifierChannel<F> for BaseFoldVerifierChannel<'_, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	type Elem = Channel::Elem;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		self.channel.recv_one()
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<Self::Elem>, binius_ip::channel::Error> {
		self.channel.recv_many(n)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[Self::Elem; N], binius_ip::channel::Error> {
		self.channel.recv_array()
	}

	fn recv_public_claim(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		self.channel.recv_public_claim()
	}

	fn sample(&mut self) -> Self::Elem {
		self.channel.sample()
	}

	fn observe_one(&mut self, val: F) -> Self::Elem {
		self.channel.observe_one(val)
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<Self::Elem> {
		self.channel.observe_many(vals)
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		self.channel.assert_zero(val)
	}
}

impl<F, Channel> WordIPVerifierChannel<F> for BaseFoldVerifierChannel<'_, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	type Word = Channel::Word;

	fn observe_words(&mut self, words: &[Word]) -> Vec<Self::Word> {
		self.channel.observe_words(words)
	}

	fn subset_sum(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem {
		self.channel.subset_sum(elems, word)
	}

	fn select(&mut self, elems: &[Self::Elem], word: &Self::Word) -> Self::Elem {
		self.channel.select(elems, word)
	}

	fn sample_bits(&mut self, bits: usize) -> Self::Word {
		self.channel.sample_bits(bits)
	}

	fn pack_words(&mut self, words: &[Self::Word]) -> Vec<Self::Elem> {
		self.channel.pack_words(words)
	}
}

impl<'a, F, Channel> IOPVerifierChannel<F> for BaseFoldVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	type Oracle = BaseFoldOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.schedule.specs()[self.n_received..]
	}

	fn recv_oracle(
		&mut self,
		len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error> {
		let spec = *self
			.remaining_oracle_specs()
			.first()
			.expect("recv_oracle called but no remaining oracle specs");
		assert_eq!(len, spec.len, "oracle content length must match its spec");
		// A spec is zero-knowledge iff the protocol is and the oracle is witness-dependent. The
		// protocol-level flag is not known here, so only one direction can be checked.
		assert!(
			!spec.is_zk || is_witness_dependent,
			"a zero-knowledge oracle spec must be received as witness-dependent"
		);

		let index = self.oracle_commitments.len();
		let layout = &self.layouts[index];
		let oracle = BaseFoldOracle {
			round: index,
			block_offset: layout.block_offsets[self.n_in_round],
			log_len: spec.log_msg_len(),
		};
		self.n_received += 1;
		self.n_in_round += 1;

		// The round's last oracle brings the commitment to the whole round oracle.
		if self.n_in_round < layout.block_offsets.len() {
			return Ok(oracle);
		}
		self.n_in_round = 0;

		self.recv_round_commitment()?;

		Ok(oracle)
	}

	fn verify_multi_oracle_relation(
		&mut self,
		operands: impl IntoIterator<Item = (Self::Oracle, TransparentEvalFn<Self::Elem>)>,
		claim: Self::Elem,
	) -> Result<(), Error> {
		// Queue the relation; the actual opening (sumchecks + combined
		// FRI) happens once, over all committed round oracles, in [`Self::finish`].
		let n_committed = self.oracle_commitments.len();
		let mut queued = Vec::new();
		for (oracle, transparent) in operands {
			let BaseFoldOracle {
				round,
				block_offset,
				log_len,
			} = oracle;
			assert!(round < n_committed, "oracle's round {round} is not committed yet");

			// On the round oracle, the operand is `t(v) · eq(b, u)`: the oracle's own
			// transparent over its low variables, and zero outside its block `b` over the high.
			let block_bits = (0..self.layouts[round].log_len - log_len)
				.map(|bit| {
					if (block_offset >> bit) & 1 == 1 {
						Channel::Elem::one()
					} else {
						Channel::Elem::zero()
					}
				})
				.collect::<Vec<_>>();
			let transparent = Box::new(move |point: &[Channel::Elem]| {
				let (low, high) = point.split_at(log_len);
				eq_ind(high, &block_bits) * transparent(low)
			});

			queued.push(Operand { round, transparent });
		}
		self.relations.push(Relation {
			claim,
			operands: queued,
		});
		Ok(())
	}
}
