// Copyright 2026 The Binius Developers

//! The verifier-side compiler for whichever commitment scheme was selected.

use binius_field::BinaryField;
use binius_hash::HashSuite;
use binius_iop::{
	basefold::compiler::BaseFoldVerifierCompiler,
	channel::OracleSpec,
	ligerito::compiler::LigeritoVerifierCompiler,
	soundness::{Grinding, SoundnessRegime},
};

use crate::{
	error::Error,
	fri::{ConstantArityStrategy, calculate_n_test_queries},
	merkle_tree::BinaryMerkleTreeScheme,
	pcs::Pcs,
};

/// A compiler that creates verifier channels for the selected commitment scheme.
///
/// One variant per scheme, each holding the parameters that scheme chose for the trace oracle.
/// The variant is fixed at setup, so a single verifier never mixes the two.
#[derive(Clone)]
pub enum PcsVerifierCompiler<F>
where
	F: BinaryField,
{
	/// A sumcheck interleaved with one FRI over a codeword committed at a single rate.
	BaseFold(BaseFoldVerifierCompiler<F>),
	/// A ladder of Reed-Solomon commitments whose rate falls at every level.
	Ligerito(LigeritoVerifierCompiler<F>),
}

impl<F> PcsVerifierCompiler<F>
where
	F: BinaryField,
{
	/// Chooses parameters for the selected scheme over the given oracles.
	///
	/// The two schemes read `log_inv_rate` the same way, which is what makes them comparable.
	/// A FRI codeword is committed at that rate, and so is the first level of a ladder.
	/// So both do the same encoding work over the message the oracle commits.
	///
	/// # Errors
	///
	/// Returns an error when no rate ladder over this oracle reaches the security target.
	/// One cause is too few codeword positions to hold the queries a level opens.
	/// The other is a correlated-agreement ceiling already under the target.
	///
	/// # Panics
	///
	/// Panics if `oracle_specs` is empty, or if the ladder is asked for more than one oracle.
	pub fn new<H>(
		scheme: Pcs,
		merkle_scheme: &BinaryMerkleTreeScheme<F, H>,
		oracle_specs: Vec<OracleSpec>,
		log_inv_rate: usize,
		security_bits: usize,
	) -> Result<Self, Error>
	where
		H: HashSuite,
	{
		assert!(
			!oracle_specs.is_empty(),
			"precondition: a commitment scheme needs at least one oracle to commit"
		);

		match scheme {
			Pcs::BaseFold => {
				let n_test_queries = calculate_n_test_queries(security_bits, log_inv_rate);
				// The fold arities come from the parameter search itself, so the strategy handed
				// over is a formality that keeps the constructor's shape.
				let log_code_len = oracle_specs[0].log_msg_len + log_inv_rate;
				let arity =
					ConstantArityStrategy::with_optimal_arity::<F, _>(merkle_scheme, log_code_len)
						.arity;
				Ok(Self::BaseFold(BaseFoldVerifierCompiler::new(
					merkle_scheme,
					oracle_specs,
					log_inv_rate,
					n_test_queries,
					&ConstantArityStrategy::new(arity),
				)))
			}
			Pcs::Ligerito => {
				// The query counts come out of the same unique-decoding radius FRI uses, so the
				// two schemes are priced against one another rather than against two regimes.
				let log_msg_len = oracle_specs[0].log_msg_len;
				// No proof of work: the two schemes are compared on the protocol alone, and FRI
				// grinds nothing either.
				LigeritoVerifierCompiler::optimal(
					merkle_scheme,
					oracle_specs,
					log_inv_rate,
					SoundnessRegime::UniqueDecoding,
					security_bits,
					Grinding::NONE,
				)
				.map(Self::Ligerito)
				.ok_or(Error::NoLigeritoLadder {
					log_msg_len,
					log_inv_rate,
					security_bits,
				})
			}
		}
	}

	/// Which scheme this compiler builds channels for.
	pub const fn scheme(&self) -> Pcs {
		match self {
			Self::BaseFold(_) => Pcs::BaseFold,
			Self::Ligerito(_) => Pcs::Ligerito,
		}
	}

	/// The oracle specifications both schemes agree the prover commits.
	pub fn oracle_specs(&self) -> &[OracleSpec] {
		match self {
			Self::BaseFold(compiler) => compiler.oracle_specs(),
			Self::Ligerito(compiler) => compiler.oracle_specs(),
		}
	}

	/// The dimension of the largest evaluation domain the chosen parameters need.
	///
	/// A prover builds its additive transform from this.
	pub fn max_log_domain_size(&self) -> usize {
		match self {
			Self::BaseFold(compiler) => compiler.max_log_domain_size(),
			Self::Ligerito(compiler) => compiler.max_log_domain_size(),
		}
	}

	/// The FRI compiler, when that is the scheme in use.
	///
	/// The recursion circuit is written against this one scheme, so it asks for it by name.
	pub const fn as_basefold(&self) -> Option<&BaseFoldVerifierCompiler<F>> {
		match self {
			Self::BaseFold(compiler) => Some(compiler),
			Self::Ligerito(_) => None,
		}
	}

	/// The ladder compiler, when that is the scheme in use.
	pub const fn as_ligerito(&self) -> Option<&LigeritoVerifierCompiler<F>> {
		match self {
			Self::BaseFold(_) => None,
			Self::Ligerito(compiler) => Some(compiler),
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Ghash128b as B128;
	use binius_hash::StdHashSuite;

	use super::*;

	fn scheme() -> BinaryMerkleTreeScheme<B128, StdHashSuite> {
		BinaryMerkleTreeScheme::new()
	}

	/// Both schemes must accept the same oracle, or the switch would not be a switch.
	///
	/// Fixture state: one 2^20 oracle at inverse rate 2, the shipped 96-bit target.
	#[test]
	fn both_schemes_compile_the_same_oracle() {
		let specs = vec![OracleSpec::new(20)];

		let basefold =
			PcsVerifierCompiler::new(Pcs::BaseFold, &scheme(), specs.clone(), 1, 96).unwrap();
		let ligerito = PcsVerifierCompiler::new(Pcs::Ligerito, &scheme(), specs.clone(), 1, 96)
			.expect("a 2^20 message at rate 1/2 admits a ladder at 96 bits");

		assert_eq!(basefold.scheme(), Pcs::BaseFold);
		assert_eq!(ligerito.scheme(), Pcs::Ligerito);
		assert_eq!(basefold.oracle_specs(), specs.as_slice());
		assert_eq!(ligerito.oracle_specs(), specs.as_slice());

		// Level 0 encodes at the rate the caller asked for, which is the whole basis of the
		// comparison: the same message goes through the same encode on both sides.
		//
		//     FRI     : 2^20 message at rate 1/2 -> 2^21 codeword positions
		//     ladder  : level 0 the same, deeper levels shorter and at lower rates
		let params = ligerito.as_ligerito().unwrap().params();
		let level_0 = &params.levels()[0];
		assert_eq!(params.log_msg_len(), 20);
		assert_eq!(level_0.log_inv_rate, 1);
		// Interleaved lanes are what the ladder splits the message across, so the codeword's
		// positions are the per-lane length times the lane count.
		assert_eq!(level_0.log_codeword_len() + level_0.log_lanes, 21);
	}

	/// Each variant answers for its own scheme and denies the other.
	#[test]
	fn a_compiler_names_only_the_scheme_it_holds() {
		let specs = vec![OracleSpec::new(20)];
		let basefold = PcsVerifierCompiler::new(Pcs::BaseFold, &scheme(), specs.clone(), 1, 96)
			.expect("FRI parameters exist at every size");
		let ligerito = PcsVerifierCompiler::new(Pcs::Ligerito, &scheme(), specs, 1, 96)
			.expect("a 2^20 message at rate 1/2 admits a ladder at 96 bits");

		assert!(basefold.as_basefold().is_some());
		assert!(basefold.as_ligerito().is_none());
		assert!(ligerito.as_ligerito().is_some());
		assert!(ligerito.as_basefold().is_none());
	}

	/// A message too short to hold the query phase has no ladder, and that is reported.
	///
	/// Fixture state: a 2^4 message at rate 1/2 gives level 0 only 2^5 codeword positions.
	/// A 96-bit target at that rate needs 241 of them opened, which 32 positions cannot serve.
	#[test]
	fn a_message_with_no_ladder_is_an_error_rather_than_a_panic() {
		let Err(err) =
			PcsVerifierCompiler::new(Pcs::Ligerito, &scheme(), vec![OracleSpec::new(4)], 1, 96)
		else {
			panic!("no ladder over a 2^4 message reaches 96 bits");
		};

		match err {
			Error::NoLigeritoLadder {
				log_msg_len,
				log_inv_rate,
				security_bits,
			} => {
				assert_eq!(log_msg_len, 4);
				assert_eq!(log_inv_rate, 1);
				assert_eq!(security_bits, 96);
			}
			other => panic!("wrong error variant: {other:?}"),
		}
	}
}
