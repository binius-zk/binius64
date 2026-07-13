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

use binius_field::{Field, field::FieldOps, util::FieldFn};
use binius_hash::StdHashSuite;
use binius_ip::channel::{Error as IPError, IPVerifierChannel};
use binius_iop::channel::{
	Error as IOPError, IOPVerifierChannel, OracleLinearRelation, OracleSpec,
};
use binius_transcript::VerifierTranscript;
use binius_verifier::{Verifier, config::StdChallenger};

use crate::table::Claim;

/// A [`FieldFn`] wrapper that delegates to an inner [`FieldFn`] and, on the NATIVE evaluation
/// (`call_native` — the only path the concrete verifier channel takes, ip/src/channel.rs:156),
/// records `(inputs, output)` into the claim sink. The generic `call::<E>` path (circuit-element
/// evaluation, `E != F`) merely delegates — there is nothing to record into an `F`-typed sink and
/// the real-capture pipeline is native.
///
/// This replaces the old closure interceptor: upstream #1554-era `compute_public_value` now takes
/// an `impl FieldFn<F>` (field-generic) rather than an `impl FnOnce(&[F]) -> F`.
struct RecordingFn<F: Field, G: FieldFn<F>> {
	inner: G,
	sink: ClaimSink<F>,
}

impl<F: Field, G: FieldFn<F>> FieldFn<F> for RecordingFn<F, G> {
	fn call<E: FieldOps<Scalar = F> + From<F>>(&self, inputs: &[E]) -> E {
		self.inner.call(inputs)
	}

	fn call_native(&self, inputs: &[F]) -> F {
		let out = self.inner.call_native(inputs);
		self.sink.borrow_mut().push(ClaimRecord {
			inputs: inputs.to_vec(),
			output: out,
		});
		out
	}
}

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

	fn compute_public_value(&mut self, inputs: &[F], f: impl FieldFn<F>) -> F {
		let sink = Rc::clone(&self.sink);
		self.inner
			.compute_public_value(inputs, RecordingFn { inner: f, sink })
	}
}

impl<'c, F, C> IOPVerifierChannel<F> for RecorderChannel<'c, F, C>
where
	F: Field,
	C: IOPVerifierChannel<F, Elem = F>,
{
	type Oracle = C::Oracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		self.inner.remaining_oracle_specs()
	}

	fn recv_oracle(
		&mut self,
		log_msg_len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, IOPError> {
		self.inner.recv_oracle(log_msg_len, is_witness_dependent)
	}

	fn verify_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<Self::Oracle, F>>,
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
	// FWD-PORT (#1693 Merkle-IP-channel refactor): the IOP compiler no longer takes a raw
	// transcript; it wraps a `MerkleIPVerifierChannel`. `create_channel_from_transcript` builds
	// the (non-hiding) `VerifierMerkleTranscriptChannel` for us.
	let mut channel = verifier
		.iop_compiler()
		.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(&mut transcript);
	{
		let mut recorder = RecorderChannel::new(&mut channel, Rc::clone(&sink));
		verifier
			.iop_verifier()
			.verify(public, &mut recorder)
			.map_err(|e| anyhow::anyhow!("leaf verify failed: {e}"))?;
	}
	// FWD-PORT (#1500/#1586 batched BaseFold): the channel DEFERS the combined FRI opening to
	// `finish()`; it must run to consume the opening bytes (else the transcript is non-empty).
	channel
		.finish()
		.map_err(|e| anyhow::anyhow!("basefold channel finish: {e}"))?;
	transcript
		.finalize()
		.map_err(|e| anyhow::anyhow!("transcript finalize: {e}"))?;

	let records = Rc::try_unwrap(sink)
		.map_err(|_| anyhow::anyhow!("sink still shared"))?
		.into_inner();
	// TWO-SITE (upstream #1554/#1585): `check_eval` now calls `compute_public_value` TWICE —
	//   1. MonsterEvalFn: the O(N) deferred monster claim we DISCHARGE (arity == expected_arity =
	//      16 + n_x + lw).
	//   2. PublicWordsEvalFn: an O(public) MLE of the verifier's OWN public words at
	//      (r_j ++ r_y_low), arity 6 + log_public_words. This is not prover-forgeable and is
	//      already in the K·O(small) budget, so the final verifier / guest computes it natively;
	//      we do NOT discharge it.
	// The two arities never collide (6+lp < 16+n_x+lw since lp <= lw), so select the monster by
	// its expected arity.
	let arities: Vec<usize> = records.iter().map(|r| r.inputs.len()).collect();
	anyhow::ensure!(
		records.len() == 2,
		"expected exactly two compute_public_value records per leaf verification \
		 (monster + public-words), got {} (arities {:?})",
		records.len(),
		arities,
	);
	let mut matching: Vec<ClaimRecord<crate::B128>> = records
		.into_iter()
		.filter(|r| r.inputs.len() == expected_arity)
		.collect();
	anyhow::ensure!(
		matching.len() == 1,
		"expected exactly one record with the monster arity {} among the two sites (arities {:?})",
		expected_arity,
		arities,
	);
	let rec = matching.pop().expect("len checked");
	Ok(Claim {
		point: rec.inputs,
		value: rec.output,
	})
}
