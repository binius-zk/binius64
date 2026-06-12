// Copyright 2026 The Binius Developers

//! ZK-wrapped verifier channel that delegates to a BaseFold ZK channel and an outer IOP verifier.
//!
//! [`ZKWrappedVerifierChannel`] wraps a [`BaseFoldZKVerifierChannel`] and an [`IOPVerifier`].
//! Inner-channel values flow through the wrapper as [`WrappedWire`] elements, tracked as
//! `Constant` / `InOut` / `Private` to distinguish what the verifier does and does not know
//! concretely, and every value entering the channel is recorded in an interaction vec (mirroring
//! the prover side). [`finish()`] replays the public-side gates of the recorded gate sequence
//! against the interaction to derive the outer public-input segment, then runs the outer verifier
//! against the inner channel (BINIUS-43).
//!
//! [`finish()`]: ZKWrappedVerifierChannel::finish

use std::{array, cell::RefCell, marker::PhantomData, rc::Rc};

use binius_field::{BinaryField, Field};
use binius_iop::{
	basefold_zk_channel::{BaseFoldZKOracle, BaseFoldZKVerifierChannel},
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	merkle_tree::MerkleTreeScheme,
};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{circuit_builder::CircuitBuilder, gate::GateSequence};
use binius_transcript::fiat_shamir::Challenger;
use binius_utils::DeserializeBytes;

use crate::{
	Error, IOPVerifier,
	wrapper::circuit_elem::{CircuitElem, CircuitWire},
};

/// [`CircuitWire`] backend used by [`ZKWrappedVerifierChannel`].
///
/// Mirrors the verifier-side [`BuilderWire`](super::builder_channel::BuilderWire):
///
/// - `Constant(F)` — known at compile time.
/// - `InOut(F)` — F is known to the wrapper (received, sampled, or observed on the inner channel,
///   or derived from such values by public-only arithmetic).
/// - `Private` — F is unknown to the wrapper (analog of `BuilderWire::Private`; e.g. the precommit
///   OTP key, never sent through the inner channel — it lives only in the outer prover's
///   commitment, accessed by the outer verifier circuit).
///
/// The wrapper performs no constraint building and no witness generation; the outer public-input
/// segment is derived in [`ZKWrappedVerifierChannel::finish`] by replaying the recorded gate
/// sequence, so the wires only track values.
#[derive(Debug, Clone)]
pub enum WrappedWire<F: Field> {
	Constant(F),
	InOut(F),
	Private,
}

impl<F: Field> CircuitWire<F> for WrappedWire<F> {
	type Builder = NoopBuilder<F>;

	fn combine<const IN: usize, const OUT: usize>(
		_builder: &mut Self::Builder,
		wires: [&Self; IN],
		f_op: impl Fn([F; IN]) -> [F; OUT],
		_builder_op: impl Fn(&mut Self::Builder, [(); IN]) -> [(); OUT],
	) -> [Self; OUT] {
		let inner_values = array_util::try_map(wires, |wire| match wire {
			Self::Constant(val) | Self::InOut(val) => Some(*val),
			Self::Private => None,
		});
		if let Some(inner_values) = inner_values {
			let ret_values = f_op(inner_values);
			let all_constant = wires.iter().all(|w| matches!(w, Self::Constant(_)));
			if all_constant {
				ret_values.map(Self::Constant)
			} else {
				// Mix of Constant + InOut, no Private: the result is still value-known.
				ret_values.map(Self::InOut)
			}
		} else {
			array::from_fn(|_| Self::Private)
		}
	}

	fn combine_varlen(
		_builder: &mut Self::Builder,
		wires: &[&Self],
		n_out: usize,
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		_builder_op: impl FnOnce(&mut Self::Builder, &[()]) -> Vec<()>,
	) -> Vec<Self> {
		let inner_values = wires
			.iter()
			.map(|wire| match wire {
				Self::Constant(val) | Self::InOut(val) => Some(*val),
				Self::Private => None,
			})
			.collect::<Option<Vec<_>>>();
		if let Some(inner_values) = inner_values {
			let ret_values = f_op(&inner_values);
			debug_assert_eq!(ret_values.len(), n_out);
			let all_constant = wires.iter().all(|w| matches!(w, Self::Constant(_)));
			if all_constant {
				ret_values.into_iter().map(Self::Constant).collect()
			} else {
				ret_values.into_iter().map(Self::InOut).collect()
			}
		} else {
			(0..n_out).map(|_| Self::Private).collect()
		}
	}
}

/// Stateless [`CircuitBuilder`] backing [`WrappedWire`].
///
/// The wrapper neither builds constraints nor generates a witness, but [`CircuitElem`] anchors
/// every wire to a shared builder, so this type exists purely to satisfy that plumbing. Its
/// methods are never invoked: [`WrappedWire::combine`] computes every value at the F level and
/// never calls `builder_op`.
#[derive(Debug, Default)]
pub struct NoopBuilder<F>(PhantomData<F>);

impl<F: Field> CircuitBuilder for NoopBuilder<F> {
	type Wire = ();
	type Field = F;

	fn assert_zero(&mut self, _wire: Self::Wire) {}

	fn constant(&mut self, _val: Self::Field) -> Self::Wire {}

	fn add(&mut self, _lhs: Self::Wire, _rhs: Self::Wire) -> Self::Wire {}

	fn mul(&mut self, _lhs: Self::Wire, _rhs: Self::Wire) -> Self::Wire {}

	fn hint<
		H: Fn([Self::Field; IN]) -> [Self::Field; OUT] + 'static,
		const IN: usize,
		const OUT: usize,
	>(
		&mut self,
		_inputs: [Self::Wire; IN],
		_f: H,
	) -> [Self::Wire; OUT] {
		[(); OUT]
	}
}

/// A verifier channel that wraps a [`BaseFoldZKVerifierChannel`] and an [`IOPVerifier`].
///
/// `Self::Elem = CircuitElem<F, WrappedWire<F>>`. F values from the inner channel become
/// [`WrappedWire::InOut`] elements and are appended to the interaction vec in channel-call order
/// — the same order the symbolic build records its `Input` gates, since both run the same inner
/// verifier. [`Self::finish`] replays the public-side gates against the interaction to derive the
/// outer public-input segment.
///
/// `transparent` closures supplied via [`OracleLinearRelation`] must depend only on
/// `Constant` and `InOut` inputs (sampled challenges), never on `Private` ones —
/// `verify_oracle_relations` panics otherwise.
pub struct ZKWrappedVerifierChannel<'a, F, MTScheme, Challenger_>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F>,
	Challenger_: Challenger,
{
	inner_channel: BaseFoldZKVerifierChannel<'a, F, MTScheme, Challenger_>,
	outer_verifier: &'a IOPVerifier<F>,
	precommit_oracle: BaseFoldZKOracle,
	/// Anchor for the [`CircuitElem`] wires handed to the inner verifier; see [`NoopBuilder`].
	builder: Rc<RefCell<NoopBuilder<F>>>,
	/// Every value entering the channel (recv/sample/observe/decrypted claims), in call order —
	/// the verifier-side counterpart of `ZKWrappedProverChannel`'s interaction vec. Supplies the
	/// `Input` gate values when [`Self::finish`] replays the recorded gate sequence.
	interaction: Vec<F>,
	/// Number of outer oracles still to be received on `inner_channel` after inner verification
	/// completes (i.e. the outer verifier's non-precommit oracles — private and mask).
	n_outer_suffix_oracles: usize,
}

impl<'a, F, MTScheme, Challenger_> ZKWrappedVerifierChannel<'a, F, MTScheme, Challenger_>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	/// Creates a new ZK-wrapped verifier channel.
	///
	/// The outer verifier's oracle specs are expected to straddle the inner channel specs:
	/// the outer precommit spec is at position 0 (committed before any inner interaction), and
	/// the remaining outer specs (private, mask) form a suffix that will be received after the
	/// inner verification completes. `new` receives the outer precommit oracle from the inner
	/// channel and stores the handle for use in [`Self::finish`].
	///
	/// # Panics
	///
	/// Panics if the channel's oracle specs do not match the expected layout
	/// `[outer_precommit, inner..., outer_private, outer_mask]`.
	pub fn new(
		mut inner_channel: BaseFoldZKVerifierChannel<'a, F, MTScheme, Challenger_>,
		outer_verifier: &'a IOPVerifier<F>,
	) -> Result<Self, Error> {
		let outer_oracle_specs = outer_verifier.oracle_specs();
		let channel_oracle_specs = inner_channel.remaining_oracle_specs();

		let n_outer = outer_oracle_specs.len();
		let n_total = channel_oracle_specs.len();
		assert!(
			n_outer >= 1 && n_outer <= n_total,
			"outer oracle specs ({n_outer}) exceed channel oracle specs ({n_total}) or are empty"
		);
		assert_eq!(
			channel_oracle_specs[0], outer_oracle_specs[0],
			"outer precommit oracle spec must be the first spec on the channel"
		);
		let suffix_len = n_outer - 1;
		assert_eq!(
			&channel_oracle_specs[n_total - suffix_len..],
			&outer_oracle_specs[1..],
			"outer private/mask oracle specs must be the final suffix of channel specs"
		);

		let precommit_oracle = inner_channel.recv_oracle()?;

		Ok(Self {
			inner_channel,
			outer_verifier,
			precommit_oracle,
			builder: Rc::new(RefCell::new(NoopBuilder::default())),
			interaction: Vec::new(),
			n_outer_suffix_oracles: suffix_len,
		})
	}

	fn inout_elem(&mut self, val: F) -> CircuitElem<F, WrappedWire<F>> {
		self.interaction.push(val);
		CircuitElem::wire(&self.builder, WrappedWire::InOut(val))
	}

	/// Consumes the channel and runs the outer verifier.
	///
	/// Replays the public-side gates of `gate_seq` (recorded by the symbolic build that produced
	/// the outer constraint system) against the recorded interaction, deriving the outer
	/// public-input segment, then runs [`IOPVerifier::verify`] against the inner channel.
	pub fn finish(self, gate_seq: GateSequence<F>) -> Result<(), Error> {
		let outer_cs = self.outer_verifier.constraint_system();
		let public_size = 1 << outer_cs.log_public();

		let mut public = outer_cs.constants().to_vec();
		let n_constants = public.len();
		public.resize(public_size, F::ZERO);
		gate_seq.replay_public(n_constants, &mut public, self.interaction.into_iter());

		let mut inner_channel = self.inner_channel;
		self.outer_verifier
			.verify(self.precommit_oracle, public, &mut inner_channel)?;
		// Both the inner and outer proofs queued their oracle relations onto `inner_channel`; run
		// the single combined opening over all committed oracles now.
		inner_channel.finish()?;
		Ok(())
	}
}

impl<F, MTScheme, Challenger_> IPVerifierChannel<F>
	for ZKWrappedVerifierChannel<'_, F, MTScheme, Challenger_>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	type Elem = CircuitElem<F, WrappedWire<F>>;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		// The received value is the OTP-encrypted element; record it for the replay. The
		// decrypted plaintext (`inout - key` in the symbolic build) is unknown to the wrapper —
		// the result is Private.
		let val = self.inner_channel.recv_one()?;
		self.interaction.push(val);
		Ok(CircuitElem::wire(&self.builder, WrappedWire::Private))
	}

	fn sample(&mut self) -> Self::Elem {
		let val = self.inner_channel.sample();
		self.inout_elem(val)
	}

	fn observe_one(&mut self, val: F) -> Self::Elem {
		let elem = self.inner_channel.observe_one(val);
		self.inout_elem(elem)
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			CircuitElem::Constant(c)
			| CircuitElem::Wire {
				wire: WrappedWire::Constant(c) | WrappedWire::InOut(c),
				..
			} => {
				if c == F::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			CircuitElem::Wire {
				wire: WrappedWire::Private,
				..
			} => {
				// No-op: the corresponding constraint exists in the outer CS and is checked by
				// the outer verifier.
				Ok(())
			}
		}
	}

	fn compute_public_value(
		&mut self,
		inputs: &[Self::Elem],
		f: impl FnOnce(&[F]) -> F,
	) -> Self::Elem {
		let input_refs = inputs.iter().collect::<Vec<_>>();
		let outs = CircuitElem::combine_varlen(
			&input_refs,
			1,
			move |inputs| vec![f(inputs)],
			|_, _| {
				// Self::Elem::combine_varlen will only call the builder_op closure if any inputs
				// are non-public.
				panic!("compute_public_value: input is not public")
			},
		);
		outs.into_iter()
			.next()
			.expect("combine_varlen returns Vec with len = n_out; n_out = 1")
	}
}

impl<'a, F, MTScheme, Challenger_> IOPVerifierChannel<'a, F>
	for ZKWrappedVerifierChannel<'a, F, MTScheme, Challenger_>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	type Oracle = BaseFoldZKOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		let all = self.inner_channel.remaining_oracle_specs();
		let n_remaining_inner = all.len() - self.n_outer_suffix_oracles;
		&all[..n_remaining_inner]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, binius_iop::channel::Error> {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining inner oracle specs"
		);
		self.inner_channel.recv_oracle()
	}

	fn verify_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'a, Self::Oracle, Self::Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		let oracle_relations = oracle_relations
			.into_iter()
			.map(
				|OracleLinearRelation {
				     oracle,
				     transparent,
				     claim: _,
				 }| {
					// For each oracle opening, the prover sends the decrypted evaluation. Record
					// it in the interaction (it's an InOut value the gate replay derives for the
					// outer verifier) and rebuild the relation for the inner channel.
					let decrypted_claim = self.inner_channel.recv_one()?;
					self.interaction.push(decrypted_claim);

					// Wrap the sumcheck challenge coordinates for the transparent closure (which
					// expects `CircuitElem`s). The closure can do further arithmetic; results are
					// required to be value-known (Constant or InOut), never Private.
					//
					// HACK: the coordinates are sampled challenges, so they are wrapped as
					// `Constant`s rather than channel-anchored InOut wires. This is sound only
					// because the symbolic outer circuit (`IronSpartanBuilderChannel`) never
					// invokes the transparent closure — it attests only `claim ==
					// decrypted_claim`, with the transparent evaluation performed out of circuit.
					// This F->CircuitElem->F bridge should eventually be replaced by an F-level
					// transparent evaluator.
					let eval_fn = move |vals: &[F]| {
						let wrapped_vals = vals
							.iter()
							.map(|val| CircuitElem::Constant(*val))
							.collect::<Vec<_>>();

						match transparent(&wrapped_vals) {
							CircuitElem::Constant(val)
							| CircuitElem::Wire {
								wire: WrappedWire::Constant(val) | WrappedWire::InOut(val),
								..
							} => val,
							CircuitElem::Wire {
								wire: WrappedWire::Private,
								..
							} => {
								panic!(
									"precondition: the transparent polynomial evaluation must depend only on known values (constants or sampled challenges)"
								);
							}
						}
					};
					Ok(OracleLinearRelation {
						oracle,
						claim: decrypted_claim,
						transparent: Box::new(eval_fn),
					})
				},
			)
			.collect::<Result<Vec<_>, binius_iop::channel::Error>>()?;
		self.inner_channel.verify_oracle_relations(oracle_relations)
	}
}
