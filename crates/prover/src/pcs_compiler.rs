// Copyright 2026 The Binius Developers

//! The prover-side compiler for whichever commitment scheme the verifier selected.

use binius_field::{BinaryField, PackedField};
use binius_iop_prover::{
	basefold::compiler::BaseFoldProverCompiler, whir::compiler::WHIRProverCompiler,
};
use binius_math::ntt::AdditiveNTT;
use binius_verifier::PcsVerifierCompiler;

/// A compiler that creates prover channels for the scheme the verifier chose.
///
/// The mirror of the verifier's compiler, carrying the additive transform as well.
/// It is only ever built from a verifier's compiler.
/// So the two sides cannot disagree on the scheme or on its parameters.
#[derive(Debug)]
pub enum PcsProverCompiler<P, NTT>
where
	P: PackedField<Scalar: BinaryField>,
	NTT: AdditiveNTT<Field = P::Scalar> + Sync,
{
	/// A sumcheck interleaved with one FRI over a codeword committed at a single rate.
	BaseFold(BaseFoldProverCompiler<P, NTT>),
	/// A ladder of Reed-Solomon commitments whose rate falls at every level.
	WHIR(WHIRProverCompiler<P, NTT>),
}

impl<F, P, NTT> PcsProverCompiler<P, NTT>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
{
	/// Mirrors a verifier's compiler, reusing its scheme, parameters and oracle specifications.
	///
	/// The transform is built by the caller.
	/// Its size comes from the verifier's own answer for how large a domain the parameters reach.
	pub fn from_verifier_compiler(verifier_compiler: &PcsVerifierCompiler<F>, ntt: NTT) -> Self {
		match verifier_compiler {
			PcsVerifierCompiler::BaseFold(compiler) => {
				Self::BaseFold(BaseFoldProverCompiler::from_verifier_compiler(compiler, ntt))
			}
			PcsVerifierCompiler::WHIR(compiler) => {
				Self::WHIR(WHIRProverCompiler::from_verifier_compiler(compiler, ntt))
			}
		}
	}
}
