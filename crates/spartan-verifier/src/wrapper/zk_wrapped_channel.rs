// Copyright 2026 The Binius Developers

//! ZK-wrapped verifier channel that delegates to a BaseFold ZK channel and an outer IOP verifier.
//!
//! [`ZKWrappedVerifierChannel`] wraps a [`BaseFoldZKVerifierChannel`] and an [`IOPVerifier`].
//! Inner-channel values flow through the wrapper as lazy [`WrappedWire::InOut`] elements; an InOut
//! is pushed into the outer constraint system's public-input segment only if the inner verifier
//! forces materialization (the same way [`BuilderWire::InOut`] only allocates an InOut wire when
//! the symbolic phase forces it). [`finish()`] prepends the outer constants to the materialized
//! public values, pads to the required public size, and runs the outer verifier against the inner
//! channel.
//!
//! [`BuilderWire::InOut`]: super::builder_channel::BuilderWire
//! [`finish()`]: ZKWrappedVerifierChannel::finish

use std::{
	array,
	cell::{Cell, RefCell},
	rc::Rc,
};

use binius_field::{BinaryField, Field};
use binius_iop::{
	basefold_zk_channel::{BaseFoldZKOracle, BaseFoldZKVerifierChannel},
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	merkle_tree::MerkleTreeScheme,
};
use binius_ip::channel::IPVerifierChannel;
use binius_spartan_frontend::circuit_builder::CircuitBuilder;
use binius_transcript::fiat_shamir::Challenger;
use binius_utils::DeserializeBytes;

use crate::{
	Error, IOPVerifier,
	wrapper::circuit_elem::{CircuitElem, CircuitWire},
};

/// [`CircuitWire`] backend used by [`ZKWrappedVerifierChannel`].
///
/// Mirrors the verifier-side [`BuilderWire`](super::builder_channel::BuilderWire) and the
/// prover-side [`WitnessGenWire`](binius_spartan_prover::wrapper::replay_channel::WitnessGenWire):
///
/// - `Constant(F)` — known at compile time.
/// - `InOut { value, is_materialized }` — F is known to the wrapper. Will be pushed to the outer
///   constraint system's public-input segment if/when the wire is materialized (see
///   [`Self::materialize`]). `is_materialized` is shared via [`Rc`] so all clones of one logical
///   wire push exactly once.
/// - `Private` — F is unknown to the wrapper (analog of [`BuilderWire::Private`]; e.g. the
///   precommit OTP key, never sent through the inner channel — it lives only in the outer
///   prover's commitment, accessed by the outer verifier circuit).
#[derive(Debug, Clone)]
pub enum WrappedWire<F: Field> {
	Constant(F),
	InOut {
		value: F,
		is_materialized: Rc<Cell<bool>>,
	},
	Private,
}

impl<F: Field> WrappedWire<F> {
	fn lazy_inout(value: F) -> Self {
		Self::InOut {
			value,
			is_materialized: Rc::new(Cell::new(false)),
		}
	}

	/// If `wire` is an unmaterialized [`Self::InOut`], push its value into the
	/// [`InOutSegmentBuilder`]'s public-values vec and mark it materialized. Otherwise no-op.
	fn materialize(builder: &mut InOutSegmentBuilder<F>, wire: &Self) {
		if let Self::InOut {
			value,
			is_materialized,
		} = wire && !is_materialized.get()
		{
			builder.next_inout(*value);
			is_materialized.set(true);
		}
	}
}

impl<F: Field> CircuitWire<F> for WrappedWire<F> {
	type Builder = InOutSegmentBuilder<F>;

	fn combine<const IN: usize, const OUT: usize>(
		builder: &mut Self::Builder,
		wires: [&Self; IN],
		f_op: impl Fn([F; IN]) -> [F; OUT],
		_builder_op: impl Fn(&mut Self::Builder, [(); IN]) -> [(); OUT],
	) -> [Self; OUT] {
		let inner_values = array_util::try_map(wires, |wire| match wire {
			Self::Constant(val) => Some(*val),
			Self::InOut { value, .. } => Some(*value),
			Self::Private => None,
		});
		if let Some(inner_values) = inner_values {
			let ret_values = f_op(inner_values);
			let all_constant = wires.iter().all(|w| matches!(w, Self::Constant(_)));
			if all_constant {
				ret_values.map(Self::Constant)
			} else {
				// Mix of Constant + InOut, no Private: fresh lazy InOut.
				ret_values.map(Self::lazy_inout)
			}
		} else {
			// Some Private: materialize all InOut inputs (push their F to public_values),
			// result is Private.
			for w in &wires {
				Self::materialize(builder, w);
			}
			array::from_fn(|_| Self::Private)
		}
	}

	fn combine_varlen(
		builder: &mut Self::Builder,
		wires: &[&Self],
		n_out: usize,
		f_op: impl FnOnce(&[F]) -> Vec<F>,
		_builder_op: impl FnOnce(&mut Self::Builder, &[()]) -> Vec<()>,
	) -> Vec<Self> {
		let inner_values = wires
			.iter()
			.map(|wire| match wire {
				Self::Constant(val) => Some(*val),
				Self::InOut { value, .. } => Some(*value),
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
				ret_values.into_iter().map(Self::lazy_inout).collect()
			}
		} else {
			for w in wires {
				Self::materialize(builder, w);
			}
			(0..n_out).map(|_| Self::Private).collect()
		}
	}
}

/// [`CircuitBuilder`] backend used by [`ZKWrappedVerifierChannel`]. Has `Wire = ()` because the
/// wrapper doesn't build a constraint system itself; its sole job is to accumulate the public
/// values that `WrappedWire::InOut` materialization pushes (in materialization order, which
/// matches the InOut allocation order in the outer constraint system built symbolically by
/// [`IronSpartanBuilderChannel`](super::builder_channel::IronSpartanBuilderChannel)).
///
/// The `CircuitBuilder` methods (`assert_zero`, `add`, `mul`, `hint`, …) all return `()` and have
/// no side effects: [`WrappedWire::combine`] never invokes `builder_op` because every value flows
/// through `f_op` at the F level.
#[derive(Debug, Default)]
pub struct InOutSegmentBuilder<F> {
	public_values: Vec<F>,
}

impl<F: Field> InOutSegmentBuilder<F> {
	pub fn with_capacity(cap: usize) -> Self {
		Self {
			public_values: Vec::with_capacity(cap),
		}
	}

	/// Records a freshly-materialized InOut value. Call order must match the order
	/// [`BuilderWire::InOut`](super::builder_channel::BuilderWire) materializes in the symbolic
	/// phase, so the resulting vec aligns 1:1 with the outer CS's InOut segment.
	pub fn next_inout(&mut self, value: F) {
		self.public_values.push(value);
	}

	pub fn into_public_values(self) -> Vec<F> {
		self.public_values
	}
}

impl<F: Field> CircuitBuilder for InOutSegmentBuilder<F> {
	type Wire = ();
	type Field = F;

	fn assert_zero(&mut self, _wire: Self::Wire) {}

	fn constant(&mut self, _val: Self::Field) -> Self::Wire {}

	fn add(&mut self, _lhs: Self::Wire, _rhs: Self::Wire) -> Self::Wire {}

	fn mul(&mut self, _lhs: Self::Wire, _rhs: Self::Wire) -> Self::Wire {}

	fn hint<H: Fn([Self::Field; IN]) -> [Self::Field; OUT], const IN: usize, const OUT: usize>(
		&mut self,
		_inputs: [Self::Wire; IN],
		_f: H,
	) -> [Self::Wire; OUT] {
		[(); OUT]
	}
}

/// A verifier channel that wraps a [`BaseFoldZKVerifierChannel`] and an [`IOPVerifier`].
///
/// `Self::Elem = CircuitElem<F, WrappedWire<F>>`. F values from the inner channel become lazy
/// [`WrappedWire::InOut`] elements; the wrapper pushes them into `inout_builder`'s public-values
/// vec only when arithmetic forces materialization (i.e. mixing with a [`WrappedWire::Private`],
/// which is what `recv_one` arranges via its `inout - key` shape — same trigger as in
/// [`IronSpartanBuilderChannel::recv_one`](super::builder_channel::IronSpartanBuilderChannel)).
/// `sample` and `observe_one` return purely lazy InOuts that materialize only if the inner
/// verifier later combines them with a Private.
///
/// `transparent` closures supplied via
/// [`OracleLinearRelation`](binius_iop::channel::OracleLinearRelation) must depend only on
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
	/// Owns the materialized public-values vec and exposes `next_inout(F)` for
	/// [`WrappedWire::materialize`] to push into.
	inout_builder: Rc<RefCell<InOutSegmentBuilder<F>>>,
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
			inout_builder: Rc::new(RefCell::new(InOutSegmentBuilder::with_capacity(
				outer_public_size,
			))),
			n_outer_suffix_oracles: suffix_len,
		})
	}

	/// Consumes the channel and runs the outer verifier.
	///
	/// Prepends the outer constraint system's constants to the materialized public values, pads to
	/// the required public size, and runs [`IOPVerifier::verify`] against the inner channel.
	pub fn finish(self) -> Result<(), Error> {
		let outer_cs = self.outer_verifier.constraint_system();
		let public_size = 1 << outer_cs.log_public();

		let inout_builder = Rc::try_unwrap(self.inout_builder)
			.expect("CircuitElem values should only hold Weak references")
			.into_inner();

		let mut public = outer_cs.constants().to_vec();
		public.extend(inout_builder.into_public_values());
		public.resize(public_size, F::ZERO);

		let mut inner_channel = self.inner_channel;
		self.outer_verifier
			.verify(self.precommit_oracle, public, &mut inner_channel)?;
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
		// Mirror `IronSpartanBuilderChannel::recv_one`'s shape: `inout - key`. The InOut carries
		// the encrypted F received from the inner channel; the key is `Private` (the OTP key is
		// not known to the wrapper). The subtraction triggers materialization of the InOut
		// (pushing the encrypted F to `public_values`), and the result is `Private` — matching
		// the symbolic phase's Private result wire.
		let val = self.inner_channel.recv_one()?;
		let inout = CircuitElem::wire(&self.inout_builder, WrappedWire::lazy_inout(val));
		let key = CircuitElem::wire(&self.inout_builder, WrappedWire::Private);
		Ok(inout - key)
	}

	fn sample(&mut self) -> Self::Elem {
		let val = self.inner_channel.sample();
		CircuitElem::wire(&self.inout_builder, WrappedWire::lazy_inout(val))
	}

	fn observe_one(&mut self, val: F) -> Self::Elem {
		let elem = self.inner_channel.observe_one(val);
		CircuitElem::wire(&self.inout_builder, WrappedWire::lazy_inout(elem))
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), binius_ip::channel::Error> {
		match val {
			CircuitElem::Constant(c)
			| CircuitElem::Wire {
				wire: WrappedWire::Constant(c),
				..
			} => {
				if c == F::ZERO {
					Ok(())
				} else {
					Err(binius_ip::channel::Error::InvalidAssert)
				}
			}
			CircuitElem::Wire {
				wire: WrappedWire::InOut { value, .. },
				..
			} => {
				if value == F::ZERO {
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
					// For each oracle opening, the prover sends the decrypted evaluation. Push it
					// to public_values directly (it's a known InOut value the outer verifier
					// reads) and rebuild the relation for the inner channel.
					let decrypted_claim = self.inner_channel.recv_one()?;
					self.inout_builder
						.borrow_mut()
						.next_inout(decrypted_claim);

					let builder = self.inout_builder.clone();

					// Wrap F values as lazy InOut for the transparent closure (which expects
					// CircuitElems). The closure can do further arithmetic; results are required
					// to be value-known (Constant or InOut), never Private.
					let eval_fn = move |vals: &[F]| {
						let wrapped_vals = vals
							.iter()
							.map(|val| CircuitElem::wire(&builder, WrappedWire::lazy_inout(*val)))
							.collect::<Vec<_>>();

						match transparent(&wrapped_vals) {
							CircuitElem::Constant(val)
							| CircuitElem::Wire {
								wire: WrappedWire::Constant(val),
								..
							}
							| CircuitElem::Wire {
								wire: WrappedWire::InOut { value: val, .. },
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
