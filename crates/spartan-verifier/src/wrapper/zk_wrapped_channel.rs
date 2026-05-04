// Copyright 2026 The Binius Developers

//! ZK-wrapped verifier channel that delegates to a BaseFold ZK channel and an outer IOP verifier.
//!
//! [`ZKWrappedVerifierChannel`] wraps a [`BaseFoldZKVerifierChannel`] and an [`IOPVerifier`],
//! recording all channel values as outer public inputs. In [`finish()`], it prepends constants,
//! pads to the required public size, and runs the outer verifier against the inner channel.
//!
//! [`finish()`]: ZKWrappedVerifierChannel::finish

use std::{cell::RefCell, rc::Rc};

use binius_field::BinaryField;
use binius_iop::{
	basefold_zk_channel::{BaseFoldZKOracle, BaseFoldZKVerifierChannel},
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	merkle_tree::MerkleTreeScheme,
};
use binius_ip::channel::IPVerifierChannel;
use binius_transcript::fiat_shamir::Challenger;
use binius_utils::DeserializeBytes;

use crate::{
	Error, IOPVerifier,
	wrapper::circuit_elem::{CircuitElem, NoopBuilder, WrappedWire},
};

/// A verifier channel that wraps a [`BaseFoldZKVerifierChannel`] and an [`IOPVerifier`].
///
/// `Self::Elem` is `CircuitElem<F, WrappedWire<F>>`. All concrete F values that the inner channel
/// produces (received, sampled, observed) are recorded as outer public inputs; the
/// [`CircuitElem`]s returned to the outer verifier carry only the [`WrappedWire`] tag — `Encrypted`
/// for inner-channel values, `Decrypted` for transparent evaluations of sampled challenges, and
/// `Constant` for compile-time constants. [`finish()`](Self::finish) prepends the outer constraint
/// system's constants to the recorded public values, pads, and runs [`IOPVerifier::verify`]
/// against the inner channel directly.
///
/// `transparent` closures supplied via [`OracleLinearRelation`](binius_iop::channel::OracleLinearRelation)
/// must depend only on `Constant` and `Decrypted` inputs (sampled challenges), never on
/// `Encrypted` ones — `verify_oracle_relations` panics otherwise.
pub struct ZKWrappedVerifierChannel<'a, F, MTScheme, Challenger_>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F>,
	Challenger_: Challenger,
{
	inner_channel: BaseFoldZKVerifierChannel<'a, F, MTScheme, Challenger_>,
	outer_verifier: &'a IOPVerifier<F>,
	precommit_oracle: BaseFoldZKOracle,
	public_values: Vec<F>,
	/// Phantom anchor for the `Weak<RefCell<NoopBuilder>>` references stored inside
	/// [`CircuitElem::Wire`] values returned from this channel. The inner builder is never used
	/// to record constraints — the wrapped channel doesn't build a circuit, it just records
	/// public inputs.
	inout_builder: Rc<RefCell<NoopBuilder<F>>>,
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

		let outer_public_size = 1 << outer_verifier.constraint_system().log_public();
		Ok(Self {
			inner_channel,
			outer_verifier,
			precommit_oracle,
			public_values: Vec::with_capacity(outer_public_size),
			inout_builder: Rc::new(RefCell::new(NoopBuilder::default())),
			n_outer_suffix_oracles: suffix_len,
		})
	}

	/// Consumes the channel and runs the outer verifier.
	///
	/// Prepends the outer constraint system's constants to the recorded public values, pads to
	/// the required public size, and runs [`IOPVerifier::verify`] against the inner channel.
	pub fn finish(mut self) -> Result<(), Error> {
		let outer_cs = self.outer_verifier.constraint_system();
		let public_size = 1 << outer_cs.log_public();

		let mut public = outer_cs.constants().to_vec();
		public.append(&mut self.public_values);
		public.resize(public_size, F::ZERO);

		// IOPVerifier::verify takes Vec<Channel::Elem>, not &[F].
		self.outer_verifier
			.verify(self.precommit_oracle, public, &mut self.inner_channel)?;
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
		let val = self.inner_channel.recv_one()?;
		self.public_values.push(val);
		Ok(CircuitElem::wire(&self.inout_builder, WrappedWire::Encrypted))
	}

	fn sample(&mut self) -> Self::Elem {
		let val = self.inner_channel.sample();
		self.public_values.push(val);
		CircuitElem::wire(&self.inout_builder, WrappedWire::Encrypted)
	}

	fn observe_one(&mut self, val: F) -> Self::Elem {
		let elem = self.inner_channel.observe_one(val);
		self.public_values.push(elem);
		CircuitElem::wire(&self.inout_builder, WrappedWire::Encrypted)
	}

	fn assert_zero(&mut self, _val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		// No-op: inner assertions are checked by the outer verifier.
		Ok(())
	}
}

impl<F, MTScheme, Challenger_> IOPVerifierChannel<F>
	for ZKWrappedVerifierChannel<'_, F, MTScheme, Challenger_>
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

	fn verify_oracle_relations<'b>(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'b, Self::Oracle, Self::Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		let oracle_relations = oracle_relations
			.into_iter()
			.map(
				|OracleLinearRelation {
				     oracle,
				     transparent,
				     claim: _,
				 }| {
					// For each oracle opening, the prover sends the decrypted evaluation. The outer
					// verifier checks in the circuit equality of this value with the expected
					// expression over encrypted values.
					let decrypted_claim = self.inner_channel.recv_one()?;
					self.public_values.push(decrypted_claim);

					let builder = self.inout_builder.clone();

					// Create a transparent evaluation function for F values that internally uses
					// the symbolic evaluation function provided.
					let eval_fn = move |vals: &[F]| {
						let wrapped_vals = vals
							.iter()
							.map(|val| CircuitElem::wire(&builder, WrappedWire::Decrypted(*val)))
							.collect::<Vec<_>>();

						match transparent(&wrapped_vals) {
							CircuitElem::Constant(val)
							| CircuitElem::Wire {
								wire: WrappedWire::Constant(val) | WrappedWire::Decrypted(val),
								..
							} => val,
							CircuitElem::Wire {
								wire: WrappedWire::Encrypted,
								..
							} => {
								panic!(
									"precondition: the transparent polynomial evaluation must depend only on decrypted values (ie. sampled challenges)"
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
