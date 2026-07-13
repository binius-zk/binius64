//! SubstitutingChannel — the spec §5.1 capture/substitution seam.
//!
//! A generic interposer over ANY verifier channel (`Elem` fully generic, so it works
//! identically over the three wrapper backends: `IronSpartanBuilderChannel` (symbolic),
//! `ZKWrappedVerifierChannel` (verify time) and `ReplayChannel` (prover witness fill)).
//! Every `IPVerifierChannel` / `IOPVerifierChannel` method forwards verbatim; ONLY
//! `compute_public_value` is intercepted, and even there the interposer delegates to
//! the inner channel's own `compute_public_value` with a WRAPPED closure. The inner
//! channel therefore performs its native wire construction / lazy-InOut materialization
//! byte-for-byte identically to the un-interposed Phase-1b path — the outer CS shape and
//! the public-vector materialization order are untouched by construction (the upstream
//! contract, crates/ip/src/channel.rs:100-111, explicitly permits the impl not to invoke
//! the caller's closure; the symbolic builder never invokes it, so the builder needs no
//! interposer at all and the outer CS is the exact Phase-1b circuit).
//!
//! The wrapped closure is the S1 single binding site: it produces ONE field value v
//! (either by invoking the original monster closure — prover replay / Phase-1b-style
//! verify — or by consuming the next artifact-supplied value, skipping the O(N) monster
//! evaluation entirely), pushes the `(inputs, v)` ClaimRecord to the sink, and returns
//! that same v to become the one Elem that `check_eval` multiplies
//! (shift/verify.rs:315-316). The recorded value and the consumed wire value are the
//! same `F` produced by the same closure invocation — there is no second recv/observe
//! of v anywhere (the substitution performs ZERO transcript operations; v becomes
//! FS-bound when the discharge statement — built by the VERIFIER from its own sink —
//! is observed at the head of the discharge segment, before mu is sampled; P0.1/P0.4).

use std::{cell::RefCell, rc::Rc};

use binius_field::Field;
use binius_ip::channel::{Error as IPError, IPVerifierChannel};
use binius_iop::channel::{Error as IOPError, IOPVerifierChannel, OracleLinearRelation, OracleSpec};

/// One recorded (claim point, claimed/computed value) pair, in capture order.
#[derive(Debug, Clone)]
pub struct ClaimRecord<F> {
	pub inputs: Vec<F>,
	pub output: F,
}

pub type ClaimSink<F> = Rc<RefCell<Vec<ClaimRecord<F>>>>;

/// Where the monster value comes from at the interception site.
#[derive(Clone)]
pub enum ValueSource<F> {
	/// Invoke the original closure (native O(N) monster evaluation). Used at prover
	/// replay (witness fill) and for the Phase-1b baseline verify mode.
	Compute,
	/// Consume the next prover-supplied value IN ORDER; the original closure is
	/// deliberately NOT invoked (no O(N) work). Used at verify time.
	Substitute(Rc<RefCell<std::vec::IntoIter<F>>>),
	/// TEST ONLY (adversarial prover): invoke the original closure, then add `delta`
	/// to the value of claim index `at` (claim indices in sink order).
	#[doc(hidden)]
	ComputeTampered { delta: F, at: usize },
}

impl<F: Field> ValueSource<F> {
	pub fn substitute(values: Vec<F>) -> Self {
		Self::Substitute(Rc::new(RefCell::new(values.into_iter())))
	}

	/// Remaining un-consumed substitution values (coverage assert helper).
	pub fn remaining(&self) -> Option<usize> {
		match self {
			Self::Substitute(feed) => Some(feed.borrow().len()),
			_ => None,
		}
	}
}

/// The interposer. Wraps `inner` for the duration of one (or more) leaf verifications.
pub struct SubstitutingChannel<'c, F: Field, C> {
	inner: &'c mut C,
	source: ValueSource<F>,
	sink: ClaimSink<F>,
}

impl<'c, F: Field, C> SubstitutingChannel<'c, F, C> {
	pub fn new(inner: &'c mut C, source: ValueSource<F>, sink: ClaimSink<F>) -> Self {
		Self { inner, source, sink }
	}
}

impl<'c, F, C> IPVerifierChannel<F> for SubstitutingChannel<'c, F, C>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	type Elem = C::Elem;

	fn recv_one(&mut self) -> Result<Self::Elem, IPError> {
		self.inner.recv_one()
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<Self::Elem>, IPError> {
		self.inner.recv_many(n)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[Self::Elem; N], IPError> {
		self.inner.recv_array::<N>()
	}

	fn sample(&mut self) -> Self::Elem {
		self.inner.sample()
	}

	fn sample_many(&mut self, n: usize) -> Vec<Self::Elem> {
		self.inner.sample_many(n)
	}

	fn sample_array<const N: usize>(&mut self) -> [Self::Elem; N] {
		self.inner.sample_array::<N>()
	}

	fn observe_one(&mut self, val: F) -> Self::Elem {
		self.inner.observe_one(val)
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<Self::Elem> {
		self.inner.observe_many(vals)
	}

	fn assert_zero(&mut self, val: Self::Elem) -> Result<(), IPError> {
		self.inner.assert_zero(val)
	}

	fn compute_public_value(
		&mut self,
		inputs: &[Self::Elem],
		f: impl FnOnce(&[F]) -> F,
	) -> Self::Elem {
		let sink = Rc::clone(&self.sink);
		let source = self.source.clone();
		// Delegate to the inner channel so its own Elem/wire machinery runs unchanged.
		// The inner impl invokes our closure exactly once at value-known backends
		// (wrapped verify / replay) and not at all at the symbolic builder.
		self.inner.compute_public_value(inputs, move |vals| {
			let out = match &source {
				ValueSource::Compute => f(vals),
				ValueSource::Substitute(feed) => {
					// f is deliberately dropped un-invoked: NO O(N) monster work.
					feed.borrow_mut().next().expect(
						"substitution feed exhausted: more compute_public_value sites \
						 than supplied monster values (leaf shape has exactly one site)",
					)
				}
				ValueSource::ComputeTampered { delta, at } => {
					let honest = f(vals);
					if sink.borrow().len() == *at { honest + *delta } else { honest }
				}
			};
			sink.borrow_mut().push(ClaimRecord {
				inputs: vals.to_vec(),
				output: out,
			});
			out
		})
	}
}

impl<'c, 'r, F, C> IOPVerifierChannel<'r, F> for SubstitutingChannel<'c, F, C>
where
	F: Field,
	C: IOPVerifierChannel<'r, F>,
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
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'r, Self::Oracle, Self::Elem>>,
	) -> Result<(), IOPError> {
		self.inner.verify_oracle_relations(oracle_relations)
	}
}
