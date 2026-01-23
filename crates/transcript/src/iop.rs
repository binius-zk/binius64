trait IOPProverTranscript<P> {
	type Oracle;

	fn write_oracle(&mut self, buffer: FieldSlice<P>) -> Self::Oracle;

	fn write_elem(&mut self, val: P::Scalar);

	fn write_elems(&mut self, val: &[F]);

	// Or use CanSample trait
	fn sample(&mut self) -> P::Scalar;

	fn finish(self, oracle_relations: &[(Self::Oracle, FieldBuffer<P>, P::Scalar)]);
}

trait IOPVerifierTranscript<F> {
	type Oracle;

	fn read_oracle(&mut self, buffer: FieldSlice<P>) -> Result<Self::Oracle, Error>;

	fn read_elem(&mut self) -> Result<F, Error>;

	fn read_elems(&mut self, n: usize) -> Result<Vec<F>, Error>;

	// Or use CanSample trait
	fn sample(&mut self) -> P::Scalar;

	fn finish(
		self,
		oracle_relations: &[(Self::Oracle, P::Scalar)],
	) -> Result<OpeningReduction<F>, Error>;
}

pub struct OpeningReduction<F> {
	claims: Vec<F>,
	eval_point: Vec<F>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("proof is empty")]
	ProofEmpty,
}
