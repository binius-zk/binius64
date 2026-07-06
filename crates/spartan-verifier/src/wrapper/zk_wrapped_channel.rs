// Copyright 2026 The Binius Developers

//! ZK-wrapped verifier channel that delegates to a BaseFold ZK channel and an outer IOP verifier.
//!
//! [`ZKWrappedVerifierChannel`] wraps a [`BaseFoldVerifierChannel`] and an [`IOPVerifier`].
//! Inner-channel values flow through the wrapper as `CircuitElem`s backed by an
//! [`InstanceGenerator`], which reconstructs the outer constraint system's public-input vector
//! `[constants | inout | derived]` exactly as the prover's witness generator does — public-derived
//! intermediate values are recomputed natively rather than tracked by the wrapper. [`finish()`]
//! hands that public vector to the outer verifier and runs it against the inner channel.
//!
//! [`finish()`]: ZKWrappedVerifierChannel::finish

use std::{cell::RefCell, rc::Rc, sync::Arc};

use binius_field::{BinaryField, util::FieldFn};
use binius_iop::{
	basefold_channel::{BaseFoldOracle, BaseFoldVerifierChannel},
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	merkle_channel::MerkleIPVerifierChannel,
};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::{
	circuit_builder::{CircuitBuilder, InstanceGenerator, WireAllocator},
	constraint_system::{WireKind, WitnessLayout},
};

use crate::{Error, IOPVerifier, wrapper::circuit_elem::CircuitElem};

/// A verifier channel that wraps a [`BaseFoldVerifierChannel`] and an [`IOPVerifier`].
///
/// `Self::Elem = CircuitElem<F, InstanceGenerator>`. F values received or sampled from the inner
/// channel are written into the [`InstanceGenerator`]'s public segment as inout wires (in the same
/// order the symbolic
/// [`IronSpartanBuilderChannel`](super::builder_channel::IronSpartanBuilderChannel) allocates
/// them); arithmetic over public values produces derived public values, and mixing with a precommit
/// key (as `recv_one` arranges via `inout - key`) yields a value-less private result.
///
/// `transparent` closures supplied via [`OracleLinearRelation`] must depend only on public inputs
/// (constants and sampled challenges), never on private ones — `verify_oracle_relations` panics
/// otherwise.
pub struct ZKWrappedVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem = F>,
{
	inner_channel: BaseFoldVerifierChannel<'a, F, Channel>,
	outer_verifier: &'a IOPVerifier<F>,
	precommit_oracle: BaseFoldOracle,
	/// Reconstructs the outer public-input vector as the channel replays the inner verifier;
	/// `build()` yields the `[constants | inout | derived]` segment for the outer verify.
	instance_gen: Rc<RefCell<InstanceGenerator<F>>>,
	/// Allocators for the InOut and Precommit segments. They live here, not on the
	/// [`InstanceGenerator`], because allocating wires in interaction order is the channel's job;
	/// the generator just writes a value to a given wire. Allocation order must match the symbolic
	/// [`IronSpartanBuilderChannel`](super::builder_channel::IronSpartanBuilderChannel) so the
	/// wire ids align with the outer layout.
	inout_alloc: WireAllocator,
	precommit_alloc: WireAllocator,
	/// Number of outer oracles still to be received on `inner_channel` after inner verification
	/// completes (i.e. the outer verifier's non-precommit oracles — private and mask).
	n_outer_suffix_oracles: usize,
}

impl<'a, F, Channel> ZKWrappedVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem = F>,
{
	/// Creates a new ZK-wrapped verifier channel.
	///
	/// The outer verifier's oracle specs are expected to straddle the inner channel specs:
	/// the outer precommit spec is at position 0 (committed before any inner interaction), and
	/// the remaining outer specs (private, mask) form a suffix that will be received after the
	/// inner verification completes. `new` receives the outer precommit oracle from the inner
	/// channel and stores the handle for use in [`Self::finish`].
	///
	/// `outer_layout` is the witness layout of the outer constraint system (the same layout the
	/// prover used); it backs the [`InstanceGenerator`] that reconstructs the public-input vector.
	/// It is a shared `Arc`, not a borrow, so the generator is `'static`.
	/// Its `CircuitElem`s live in the transparent closures queued onto `inner_channel`.
	/// The `Arc` shares the layout the config owns rather than cloning it.
	///
	/// # Panics
	///
	/// Panics if the channel's oracle specs do not match the expected layout
	/// `[outer_precommit, inner..., outer_private, outer_mask]`.
	pub fn new(
		mut inner_channel: BaseFoldVerifierChannel<'a, F, Channel>,
		outer_verifier: &'a IOPVerifier<F>,
		outer_layout: Arc<WitnessLayout<F>>,
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

		let precommit_oracle =
			inner_channel.recv_oracle(outer_oracle_specs[0].log_msg_len, true)?;

		Ok(Self {
			inner_channel,
			outer_verifier,
			precommit_oracle,
			instance_gen: Rc::new(RefCell::new(InstanceGenerator::new(outer_layout))),
			inout_alloc: WireAllocator::new(WireKind::InOut),
			precommit_alloc: WireAllocator::new(WireKind::Precommit),
			n_outer_suffix_oracles: suffix_len,
		})
	}

	/// Allocates the next inout wire, writing `value` into the public segment, and wraps it as an
	/// element. Allocation order must match the symbolic
	/// [`IronSpartanBuilderChannel`](super::builder_channel::IronSpartanBuilderChannel).
	fn alloc_inout_elem(&mut self, value: F) -> CircuitElem<F, InstanceGenerator<F>> {
		let wire = self.inout_alloc.alloc();
		let public_wire = self.instance_gen.borrow_mut().write_inout(wire, value);
		CircuitElem::wire(&self.instance_gen, public_wire)
	}

	/// Allocates the next precommit wire (value-less to the verifier) as an element.
	fn alloc_precommit_elem(&mut self) -> CircuitElem<F, InstanceGenerator<F>> {
		let wire = self.precommit_alloc.alloc();
		let public_wire = self.instance_gen.borrow_mut().placeholder_precommit(wire);
		CircuitElem::wire(&self.instance_gen, public_wire)
	}

	/// Consumes the channel and runs the outer verifier.
	///
	/// Reads the outer public-input vector `[constants | inout | derived]` from the
	/// [`InstanceGenerator`] and runs [`IOPVerifier::verify`] against the inner channel.
	pub fn finish(self) -> Result<(), Error> {
		// The instance generator produced every public value (inout + alive-derived) as the inner
		// verifier ran, so the public segment is already final here — read it by borrow. We must
		// NOT consume the generator: the oracle relations queued onto `inner_channel` carry
		// transparent closures that still hold (`Weak`) references to it, and opening them in
		// `inner_channel.finish()` evaluates those closures. They only allocate dead derived wires
		// (ids past the layout's count, so `layout.get` returns `None` and nothing is written), so
		// the public vector read here stays correct.
		let public = self.instance_gen.borrow().public().to_vec();

		let mut inner_channel = self.inner_channel;
		self.outer_verifier
			.verify(self.precommit_oracle, public, &mut inner_channel)?;
		// Both the inner and outer proofs queued their oracle relations onto `inner_channel`; run
		// the single combined opening over all committed oracles now. `instance_gen` stays alive in
		// `self` for the duration, so the transparent closures' `Weak` upgrades succeed.
		inner_channel.finish()?;
		Ok(())
	}
}

impl<'a, F, Channel> IPVerifierChannel<F> for ZKWrappedVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem = F>,
{
	type Elem = CircuitElem<F, InstanceGenerator<F>>;

	fn recv_one(&mut self) -> Result<Self::Elem, binius_ip::channel::Error> {
		// Mirror `IronSpartanBuilderChannel::recv_one`'s shape: `inout - key`. The inout carries
		// the encrypted F received from the inner channel (written into the public segment); the
		// key is a precommit wire whose value the verifier does not know (`PublicWire(None)`).
		// The subtraction yields a private result, matching the symbolic phase's private result
		// wire.
		let val = self.inner_channel.recv_one()?;
		let inout = self.alloc_inout_elem(val);
		let key = self.alloc_precommit_elem();
		Ok(inout - key)
	}

	fn sample(&mut self) -> Self::Elem {
		let val = self.inner_channel.sample();
		self.alloc_inout_elem(val)
	}

	fn observe_one(&mut self, val: F) -> Self::Elem {
		let elem = self.inner_channel.observe_one(val);
		self.alloc_inout_elem(elem)
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			// A compile-time constant is checked here; a non-zero one is an unsatisfiable
			// assertion.
			CircuitElem::Constant(c) => {
				if c == F::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			// No-op for wires: the corresponding constraint was recorded symbolically and is
			// checked by the outer verifier over the reconstructed public segment.
			CircuitElem::Wire { .. } => Ok(()),
		}
	}

	fn compute_public_value(&mut self, inputs: &[Self::Elem], f: impl FieldFn<F>) -> Self::Elem {
		// The function's result enters as a single derived public wire (matching the symbolic
		// builder's `hint_varsize`), whose value the verifier computes natively from the
		// public-derived inputs. See `IronSpartanBuilderChannel::compute_public_value`.
		let out_wire = {
			let mut instance_gen = self.instance_gen.borrow_mut();
			let input_wires: Vec<_> = inputs
				.iter()
				.map(|elem| elem.to_wire(&mut instance_gen))
				.collect();
			instance_gen.hint_varsize(&input_wires, 1, move |vals| vec![f.call::<F>(vals)])[0]
		};
		CircuitElem::wire(&self.instance_gen, out_wire)
	}
}

impl<'a, F, Channel> IOPVerifierChannel<F> for ZKWrappedVerifierChannel<'a, F, Channel>
where
	F: BinaryField,
	Channel: MerkleIPVerifierChannel<F, Elem = F>,
{
	type Oracle = BaseFoldOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		let all = self.inner_channel.remaining_oracle_specs();
		let n_remaining_inner = all.len() - self.n_outer_suffix_oracles;
		&all[..n_remaining_inner]
	}

	fn recv_oracle(
		&mut self,
		log_msg_len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, binius_iop::channel::Error> {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining inner oracle specs"
		);
		self.inner_channel
			.recv_oracle(log_msg_len, is_witness_dependent)
	}

	fn verify_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<Self::Oracle, Self::Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		let mut inner_relations = Vec::new();
		for OracleLinearRelation {
			oracle,
			transparent,
			claim,
		} in oracle_relations
		{
			// For each oracle opening, the prover sends the decrypted evaluation. Allocate it as an
			// inout wire (written into the public segment) and attest `claim == decrypted_claim`,
			// exactly as the symbolic `IronSpartanBuilderChannel` does. The assertion is a no-op on
			// values here, but evaluating `claim - decrypted_claim` keeps the instance generator's
			// wire allocation aligned with the symbolic constraint system. Rebuild the relation
			// with the decrypted value for the inner channel.
			let decrypted_value = self.inner_channel.recv_one()?;
			let decrypted_claim = self.alloc_inout_elem(decrypted_value);
			self.assert_zero(claim - decrypted_claim)?;

			// Wrap the sumcheck challenge coordinates for the transparent closure (which expects
			// `CircuitElem`s). The closure can do further arithmetic; results are required to be
			// value-known (public), never private.
			//
			// HACK: the coordinates are sampled challenges, so they are wrapped as `Constant`s
			// rather than builder-backed wires. This frees the closure from holding a reference to
			// the instance generator, and is sound only because the symbolic outer circuit
			// (`IronSpartanBuilderChannel`) never invokes the transparent closure — it attests only
			// `claim == decrypted_claim`, with the transparent evaluation performed out of circuit.
			// This F->CircuitElem->F bridge should eventually be replaced by an F-level transparent
			// evaluator.
			let eval_fn = move |vals: &[F]| {
				let wrapped_vals = vals
					.iter()
					.map(|val| CircuitElem::Constant(*val))
					.collect::<Vec<_>>();

				match transparent(&wrapped_vals) {
					CircuitElem::Constant(val) => val,
					CircuitElem::Wire { wire, .. } => wire.value().expect(
						"precondition: the transparent polynomial evaluation must depend only on known values (constants or sampled challenges)",
					),
				}
			};
			inner_relations.push(OracleLinearRelation {
				oracle,
				claim: decrypted_value,
				transparent: Box::new(eval_fn),
			});
		}
		self.inner_channel.verify_oracle_relations(inner_relations)
	}
}
