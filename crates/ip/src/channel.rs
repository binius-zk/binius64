// Copyright 2026 The Binius Developers

use std::iter::repeat_with;

pub trait IPVerifierChannel<F> {
	fn recv_one(&mut self) -> Result<F, Error>;

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, Error> {
		repeat_with(|| self.recv_one()).take(n).collect()
	}

	fn sample(&mut self) -> F;
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("proof is empty")]
	ProofEmpty,
}
