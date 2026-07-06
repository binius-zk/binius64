// Copyright 2026 The Binius Developers

/// The value a sumcheck-style prover carries between two consecutive protocol phases.
///
/// A prover alternates between two phases, once per variable:
/// - producing the round polynomial(s) for the variable being bound,
/// - reducing them with the verifier challenge to the claim(s) for the next variable.
///
/// A prover holds exactly one of the two at any time.
/// Tracking which one lets the mandated call order be validated in a single place.
#[derive(Debug, Clone)]
pub enum RoundState<Coeffs, Claim> {
	/// Round polynomial(s) already produced for this variable, awaiting the reduction step.
	Coeffs(Coeffs),
	/// Claim(s) awaiting the next round polynomial(s): the initial claim, or a reduction result.
	Claim(Claim),
}

impl<Coeffs, Claim> RoundState<Coeffs, Claim> {
	/// Borrows the carried claim, needed to start producing this round's polynomial(s).
	///
	/// # Panics
	///
	/// Panics when the prover still holds an unreduced round polynomial.
	/// That means execute was called before the previous round's fold.
	pub fn claim(&self) -> &Claim {
		match self {
			// A carried claim is the expected input to the round-polynomial phase.
			Self::Claim(claim) => claim,
			// Holding coefficients means the reduction step is still owed.
			Self::Coeffs(_) => panic!("execute called out of order; expected fold"),
		}
	}

	/// Borrows this round's polynomial(s), needed to start the reduction step.
	///
	/// # Panics
	///
	/// Panics when the prover instead holds a claim.
	/// That means fold was called before this round's execute produced a polynomial.
	pub fn coeffs(&self) -> &Coeffs {
		match self {
			// The round polynomial is the expected input to the reduction phase.
			Self::Coeffs(coeffs) => coeffs,
			// Holding a claim means the round polynomial is still owed.
			Self::Claim(_) => panic!("fold called out of order; expected execute"),
		}
	}
}
