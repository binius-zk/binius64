// Copyright 2026 The Binius Developers

//! An [`IOPVerifierChannel`] that records the sequence of oracle specifications an IOP uses,
//! without performing any actual verification.
//!
//! Running an IOP verifier against an `OracleSetupChannel` (a dry run: all `recv_*` methods return
//! dummy values and `assert_zero` is a no-op, like [`SizeTrackingChannel`]) discovers the
//! [`OracleSpec`] sequence directly from the `recv_oracle` calls, so the specs need not be
//! hardcoded and kept in sync with the verification logic by hand.
//!
//! [`SizeTrackingChannel`]: crate::size_tracking_channel::SizeTrackingChannel

use std::{
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_field::{
	BinaryField128bGhash, ExtensionField, Field, FieldOps,
	arithmetic_traits::{InvertOrZero, Square},
	util::FieldFn,
};
use binius_ip::channel::IPVerifierChannel;

use crate::channel::{Error, IOPVerifierChannel, OracleLinearRelation, OracleSpec};

/// A zero-sized dummy field element for [`OracleSetupChannel`].
///
/// The setup channel performs no real verification, so the field values flowing through it are
/// never inspected. `DummyElem` is a stand-in whose arithmetic is all no-ops; it lets the setup
/// channel be a single non-generic type and avoids doing (pointless) real field arithmetic during
/// the structural dry run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DummyElem;

macro_rules! dummy_binop {
	($trait:ident, $method:ident) => {
		impl $trait for DummyElem {
			type Output = Self;
			fn $method(self, _rhs: Self) -> Self {
				self
			}
		}
		impl $trait<&DummyElem> for DummyElem {
			type Output = Self;
			fn $method(self, _rhs: &Self) -> Self {
				self
			}
		}
	};
}
dummy_binop!(Add, add);
dummy_binop!(Sub, sub);
dummy_binop!(Mul, mul);

macro_rules! dummy_assign {
	($trait:ident, $method:ident) => {
		impl $trait for DummyElem {
			fn $method(&mut self, _rhs: Self) {}
		}
		impl $trait<&DummyElem> for DummyElem {
			fn $method(&mut self, _rhs: &Self) {}
		}
	};
}
dummy_assign!(AddAssign, add_assign);
dummy_assign!(SubAssign, sub_assign);
dummy_assign!(MulAssign, mul_assign);

impl Neg for DummyElem {
	type Output = Self;
	fn neg(self) -> Self {
		self
	}
}

impl Sum for DummyElem {
	fn sum<I: Iterator<Item = Self>>(_iter: I) -> Self {
		Self
	}
}
impl<'a> Sum<&'a DummyElem> for DummyElem {
	fn sum<I: Iterator<Item = &'a Self>>(_iter: I) -> Self {
		Self
	}
}
impl Product for DummyElem {
	fn product<I: Iterator<Item = Self>>(_iter: I) -> Self {
		Self
	}
}
impl<'a> Product<&'a DummyElem> for DummyElem {
	fn product<I: Iterator<Item = &'a Self>>(_iter: I) -> Self {
		Self
	}
}

impl Square for DummyElem {
	fn square(self) -> Self {
		self
	}
}
impl InvertOrZero for DummyElem {
	fn invert_or_zero(self) -> Self {
		self
	}
}

impl From<BinaryField128bGhash> for DummyElem {
	fn from(_value: BinaryField128bGhash) -> Self {
		Self
	}
}

impl FieldOps for DummyElem {
	// The scalar is arbitrary; nothing reads it. `BinaryField128bGhash` satisfies the
	// `FieldOps<Scalar = B128>` bound that `binius_verifier`'s IOP verify requires.
	type Scalar = BinaryField128bGhash;

	fn zero() -> Self {
		Self
	}

	fn one() -> Self {
		Self
	}

	fn square_transpose<FSub: Field>(_elems: &mut [Self])
	where
		BinaryField128bGhash: ExtensionField<FSub>,
	{
	}
}

/// An [`IOPVerifierChannel`] that records the [`OracleSpec`] of each received oracle.
///
/// This performs no verification: `recv_*` methods return dummy values, and sampling, observation,
/// and `assert_zero` are no-ops. Drive an IOP verifier with this channel and then read the
/// recorded specs via [`into_oracle_specs`](Self::into_oracle_specs).
///
/// The channel is configured with a single `is_zk` flag (the protocol-level zero-knowledge
/// choice). Each `recv_oracle(log_msg_len, is_witness_dependent)` records
/// `OracleSpec { log_msg_len, is_zk: self.is_zk && is_witness_dependent }`.
#[derive(Debug, Default, Clone)]
pub struct OracleSetupChannel {
	is_zk: bool,
	oracle_specs: Vec<OracleSpec>,
}

impl OracleSetupChannel {
	/// Creates a new setup channel with the given protocol-level zero-knowledge flag.
	pub const fn new(is_zk: bool) -> Self {
		Self {
			is_zk,
			oracle_specs: Vec::new(),
		}
	}

	/// Returns the oracle specs recorded so far.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs
	}

	/// Consumes the channel and returns the recorded oracle specs, in the order received.
	pub fn into_oracle_specs(self) -> Vec<OracleSpec> {
		self.oracle_specs
	}
}

impl<F: Field> IPVerifierChannel<F> for OracleSetupChannel {
	type Elem = DummyElem;

	fn recv_one(&mut self) -> Result<DummyElem, binius_ip::channel::Error> {
		Ok(DummyElem)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<DummyElem>, binius_ip::channel::Error> {
		Ok(vec![DummyElem; n])
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[DummyElem; N], binius_ip::channel::Error> {
		Ok([DummyElem; N])
	}

	fn sample(&mut self) -> DummyElem {
		DummyElem
	}

	fn observe_one(&mut self, _val: F) -> DummyElem {
		DummyElem
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<DummyElem> {
		vec![DummyElem; vals.len()]
	}

	fn assert_zero(&mut self, _val: DummyElem) -> Result<(), binius_ip::channel::Error> {
		Ok(())
	}

	fn compute_public_value(&mut self, _inputs: &[DummyElem], _f: impl FieldFn<F>) -> DummyElem {
		// The setup channel performs no real computation; skipping `f` is permitted (see the
		// `IPVerifierChannel::compute_public_value` contract).
		DummyElem
	}
}

impl<'r, F: Field> IOPVerifierChannel<'r, F> for OracleSetupChannel {
	type Oracle = ();

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		// A setup channel has no pre-supplied specs; it records them as they are received.
		&[]
	}

	fn recv_oracle(
		&mut self,
		log_msg_len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error> {
		self.oracle_specs.push(OracleSpec {
			log_msg_len,
			is_zk: self.is_zk && is_witness_dependent,
		});
		Ok(())
	}

	fn verify_oracle_relations(
		&mut self,
		_oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'r, Self::Oracle, Self::Elem>>,
	) -> Result<(), Error> {
		Ok(())
	}
}
