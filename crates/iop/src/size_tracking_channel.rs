// Copyright 2026 The Binius Developers

//! A lightweight [`IOPVerifierChannel`] implementation that counts proof bytes without
//! performing any actual verification.
//!
//! This is useful for estimating proof sizes without running the full protocol.

use binius_field::Field;
use binius_ip::channel::IPVerifierChannel;

use crate::channel::{Error, IOPVerifierChannel, OracleLinearRelation, OracleSpec};

/// Default size in bytes for a single field element.
const DEFAULT_ELEMENT_SIZE: usize = 16;

/// Default size in bytes for a single oracle commitment.
const DEFAULT_ORACLE_SIZE: usize = 32;

/// An [`IOPVerifierChannel`] that tracks proof size without doing verification.
///
/// All `recv_*` methods return dummy zero values and accumulate the expected byte count
/// into the `&mut usize` provided at construction. Sampling and observation methods are
/// no-ops.
///
/// After [`IOPVerifierChannel::finish`] consumes the channel, the caller can read the
/// accumulated proof size directly from the `usize` it passed in.
pub struct SizeTrackingChannel<'a> {
	element_size: usize,
	oracle_size: usize,
	oracle_specs: Vec<OracleSpec>,
	next_oracle_index: usize,
	proof_size: &'a mut usize,
}

impl<'a> SizeTrackingChannel<'a> {
	/// Creates a new size-tracking channel with default element (16) and oracle (32) sizes.
	pub fn new(oracle_specs: Vec<OracleSpec>, proof_size: &'a mut usize) -> Self {
		Self::with_sizes(oracle_specs, DEFAULT_ELEMENT_SIZE, DEFAULT_ORACLE_SIZE, proof_size)
	}

	/// Creates a new size-tracking channel with custom element and oracle sizes.
	pub fn with_sizes(
		oracle_specs: Vec<OracleSpec>,
		element_size: usize,
		oracle_size: usize,
		proof_size: &'a mut usize,
	) -> Self {
		Self {
			element_size,
			oracle_size,
			oracle_specs,
			next_oracle_index: 0,
			proof_size,
		}
	}
}

impl<F: Field> IPVerifierChannel<F> for SizeTrackingChannel<'_> {
	type Elem = F;

	fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
		*self.proof_size += self.element_size;
		Ok(F::ZERO)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
		*self.proof_size += n * self.element_size;
		Ok(vec![F::ZERO; n])
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
		*self.proof_size += N * self.element_size;
		Ok([F::ZERO; N])
	}

	fn sample(&mut self) -> F {
		F::ZERO
	}

	fn observe_one(&mut self, _val: F) -> F {
		F::ZERO
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
		vec![F::ZERO; vals.len()]
	}

	fn assert_zero(&mut self, _val: F) -> Result<(), binius_ip::channel::Error> {
		Ok(())
	}
}

impl<F: Field> IOPVerifierChannel<F> for SizeTrackingChannel<'_> {
	type Oracle = ();

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, Error> {
		*self.proof_size += self.oracle_size;
		self.next_oracle_index += 1;
		Ok(())
	}

	fn verify_oracle_relations(
		&mut self,
		_oracle_relations: &[OracleLinearRelation<'_, Self::Oracle, Self::Elem>],
	) -> Result<(), Error> {
		Ok(())
	}
}
