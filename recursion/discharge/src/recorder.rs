//! RecorderChannel — a thin wrapper around the plain BaseFold verifier channel that
//! delegates ALL `IPVerifierChannel` + `IOPVerifierChannel` methods verbatim and
//! intercepts `compute_public_value` to record (inputs = claim point c_l,
//! output = claimed v_l) into a sink.
//!
//! The leaf IOP verifier contains exactly ONE `compute_public_value` call site
//! (crates/verifier/src/protocols/shift/verify.rs:254), so one leaf verification
//! records exactly one claim. Forwarding of the `*_many` variants is explicit rather
//! than relying on trait defaults, per the recon notes (concrete channels override
//! them; FS bytes are identical either way).

use std::{cell::RefCell, rc::Rc};

use binius_field::Field;
use binius_hash::StdHashSuite;
use binius_ip::channel::{Error as IPError, IPVerifierChannel};
use binius_iop::channel::{
	Error as IOPError, IOPVerifierChannel, OracleLinearRelation, OracleSpec,
};
use binius_transcript::VerifierTranscript;
use binius_verifier::{Verifier, config::StdChallenger};

use crate::table::Claim;

/// One recorded (claim point, claimed value) pair.
#[derive(Debug, Clone)]
pub struct ClaimRecord<F> {
	pub inputs: Vec<F>,
	pub output: F,
}

pub type ClaimSink<F> = Rc<RefCell<Vec<ClaimRecord<F>>>>;

/// Wrapper channel recording all `compute_public_value` invocations.
pub struct RecorderChannel<'c, F: Field, C> {
	inner: &'c mut C,
	sink: ClaimSink<F>,
}

impl<'c, F: Field, C> RecorderChannel<'c, F, C> {
	pub fn new(inner: &'c mut C, sink: ClaimSink<F>) -> Self {
		Self { inner, sink }
	}
}

impl<'c, F, C> IPVerifierChannel<F> for RecorderChannel<'c, F, C>
where
	F: Field,
	C: IPVerifierChannel<F, Elem = F>,
{
	type Elem = F;

	fn recv_one(&mut self) -> Result<F, IPError> {
		self.inner.recv_one()
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, IPError> {
		self.inner.recv_many(n)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], IPError> {
		self.inner.recv_array::<N>()
	}

	fn sample(&mut self) -> F {
		self.inner.sample()
	}

	fn sample_many(&mut self, n: usize) -> Vec<F> {
		self.inner.sample_many(n)
	}

	fn sample_array<const N: usize>(&mut self) -> [F; N] {
		self.inner.sample_array::<N>()
	}

	fn observe_one(&mut self, val: F) -> F {
		self.inner.observe_one(val)
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
		self.inner.observe_many(vals)
	}

	fn assert_zero(&mut self, val: F) -> Result<(), IPError> {
		self.inner.assert_zero(val)
	}

	fn compute_public_value(&mut self, inputs: &[F], f: impl FnOnce(&[F]) -> F) -> F {
		let sink = Rc::clone(&self.sink);
		self.inner.compute_public_value(inputs, move |vals| {
			let out = f(vals);
			sink.borrow_mut().push(ClaimRecord {
				inputs: vals.to_vec(),
				output: out,
			});
			out
		})
	}
}

impl<'c, 'r, F, C> IOPVerifierChannel<'r, F> for RecorderChannel<'c, F, C>
where
	F: Field,
	C: IOPVerifierChannel<'r, F, Elem = F>,
{
	type Oracle = C::Oracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		self.inner.remaining_oracle_specs()
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, IOPError> {
		self.inner.recv_oracle()
	}

	fn verify_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'r, Self::Oracle, F>>,
	) -> Result<(), IOPError> {
		self.inner.verify_oracle_relations(oracle_relations)
	}
}

/// Runs the plain (non-ZK) leaf verification with a recording channel and returns the
/// single captured monster claim. Mirrors `Verifier::verify` (verify.rs:372-390): same
/// channel type from the same compiler, wrapped by the recorder.
pub fn verify_and_capture(
	verifier: &Verifier<StdHashSuite>,
	public: &[binius_core::word::Word],
	proof_bytes: Vec<u8>,
	expected_arity: usize,
) -> anyhow::Result<Claim> {
	use crate::B128;
	let sink: ClaimSink<B128> = Rc::new(RefCell::new(Vec::new()));
	let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof_bytes);
	{
		let mut channel = verifier.iop_compiler().create_channel(&mut transcript);
		let mut recorder = RecorderChannel::new(&mut channel, Rc::clone(&sink));
		verifier
			.iop_verifier()
			.verify(public, &mut recorder)
			.map_err(|e| anyhow::anyhow!("leaf verify failed: {e}"))?;
	}
	transcript
		.finalize()
		.map_err(|e| anyhow::anyhow!("transcript finalize: {e}"))?;

	let records = Rc::try_unwrap(sink)
		.map_err(|_| anyhow::anyhow!("sink still shared"))?
		.into_inner();
	anyhow::ensure!(
		records.len() == 1,
		"expected exactly one monster claim per leaf verification, got {}",
		records.len()
	);
	let rec = records.into_iter().next().expect("len checked");
	anyhow::ensure!(
		rec.inputs.len() == expected_arity,
		"captured claim arity {} != shape's expected arity {}",
		rec.inputs.len(),
		expected_arity
	);
	Ok(Claim {
		point: rec.inputs,
		value: rec.output,
	})
}
