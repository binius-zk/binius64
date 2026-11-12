// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP verifier channel.

use std::rc::Rc;

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
struct QueuedOperand<Elem> {
	/// Index of the relation, into [`BaseFoldVerifierChannel::claims`].
	relation: usize,
	/// Evaluates the transparent multilinear, over the round oracle's variables, at the point the
	/// opening reduces to.
	transparent: TransparentEvalFn<Elem>,
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
	/// Operands of the relations queued by [`IOPVerifierChannel::verify_multi_oracle_relation`],
	/// indexed by round and opened together in [`Self::finish`]. One entry per committed round.
	queue: Vec<Vec<QueuedOperand<Channel::Elem>>>,
	/// The claim of each queued relation, in queue order.
	claims: Vec<Channel::Elem>,
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
			queue: Vec::new(),
			claims: Vec::new(),
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
			queue,
			claims,
		} = self;

		if !claims.is_empty() {
			verify_batch_zk_basefold(
				&mut channel,
				layouts,
				fri_params,
				&oracle_commitments,
				queue,
				&claims,
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
	operands: Vec<Vec<QueuedOperand<Channel::Elem>>>,
	claims: &[Channel::Elem],
) -> Result<(), Error>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	assert_eq!(layouts.len(), oracle_commitments.len());
	assert_eq!(layouts.len(), operands.len());

	// `ℓ = max_r ℓ_r`, the variable count of the opening.
	let n_vars = layouts
		.iter()
		.map(|layout| layout.log_len)
		.max()
		.expect("at least one oracle");

	let (transparents, claim) = batch_relations(channel, operands, claims);

	// A masked opening receives the mask claim and folds with γ.
	let (target, gamma) = if layouts.iter().any(|layout| layout.is_zk) {
		let s_prime = channel.recv_one()?;
		let gamma = channel.sample();
		(extrapolate_line(claim, s_prime, gamma.clone()), Some(gamma))
	} else {
		(claim, None)
	};

	let multiplicand_eval = |rho: &[Channel::Elem], zeta: &[Channel::Elem]| {
		let eq_tensor = eq_ind_partial_eval_scalars(zeta);
		izip!(&transparents, layouts, eq_tensor)
			.map(|(transparent, layout, eq_r)| {
				let n_r = layout.log_len;
				eq_r * transparent(&rho[..n_r]) * eq_ind_zero(&rho[n_r..])
			})
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

/// Batches the queued relations into the single relation `Σ_i <π_i, T_i> = S`.
///
/// Relation `j`, `Σ_i <π_i, t_ji> = s_j`, is weighted by `μ^j` for one sampled challenge μ:
///
/// ```text
/// T_i = Σ_j μ^j · t_ji     the combined transparent of oracle i
/// S   = Σ_j μ^j · s_j      the combined claim
/// ```
///
/// The inner product is linear in the transparent, so the combined relation holds exactly when
/// every relation does, except with probability at most `(k - 1) / |F|` over μ for `k` relations.
/// μ is drawn after every claim it combines is already bound to the transcript, so no claim can be
/// chosen as a function of it. An oracle in no relation gets `T_i = 0`.
///
/// Returns the evaluator of each `T_i`, in oracle-index order, and `S`.
///
/// Mirrors the prover-side batching in `binius_iop_prover::basefold::channel`.
fn batch_relations<F, Channel>(
	channel: &mut Channel,
	operands: Vec<Vec<QueuedOperand<Channel::Elem>>>,
	claims: &[Channel::Elem],
) -> (Vec<TransparentEvalFn<Channel::Elem>>, Channel::Elem)
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem: From<F> + 'static>,
{
	let mu = channel.sample();
	let claim = evaluate_univariate(claims, &mu);
	let mu_powers = Rc::new(powers(mu).take(claims.len()).collect::<Vec<_>>());

	let transparents = operands
		.into_iter()
		.map(|operands| {
			let mu_powers = Rc::clone(&mu_powers);
			Box::new(move |point: &[Channel::Elem]| {
				operands
					.iter()
					.map(|operand| {
						mu_powers[operand.relation].clone() * (operand.transparent)(point)
					})
					.sum()
			}) as TransparentEvalFn<_>
		})
		.collect();

	(transparents, claim)
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
		self.queue.push(Vec::new());

		Ok(oracle)
	}

	fn verify_multi_oracle_relation(
		&mut self,
		operands: impl IntoIterator<Item = (Self::Oracle, TransparentEvalFn<Self::Elem>)>,
		claim: Self::Elem,
	) -> Result<(), Error> {
		// Queue each operand under its round; the actual opening (sumchecks + combined
		// FRI) happens once, over all committed round oracles, in [`Self::finish`].
		let relation = self.claims.len();
		let n_committed = self.queue.len();
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

			self.queue[round].push(QueuedOperand {
				relation,
				transparent,
			});
		}
		self.claims.push(claim);
		Ok(())
	}
}
