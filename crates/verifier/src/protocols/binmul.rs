// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_field::{BinaryField, BinaryField1b, ExtensionField, field::FieldOps};
use binius_ip::{channel::IPVerifierChannel, mlecheck, sumcheck::SumcheckOutput};
use binius_math::inner_product::inner_product_scalars;

use crate::Error;

/// Output of the BinMul reduction.
///
/// The reduction proves the constraint $\widetilde{A}(x) \cdot \widetilde{B}(x) =
/// \widetilde{C}(x)$ for every $x$ on the boolean hypercube $\mathbb{B}_\ell$, where each
/// $\mathbb{F}_{2^{128}}$ element is carried by a `(lo, hi)` pair of 64-bit words. It reduces the
/// constraint to per-bit evaluation claims on the six word columns at a common evaluation point
/// `eval_point` ($r_x \in K^\ell$).
///
/// Each `*_evals` vector holds `Word::BITS` per-bit evaluation claims $\widetilde{z}(r_x, i)$ for
/// $i \in \{0, \ldots, 63\}$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinMulOutput<F> {
	pub eval_point: Vec<F>,
	pub a_lo_evals: Vec<F>,
	pub a_hi_evals: Vec<F>,
	pub b_lo_evals: Vec<F>,
	pub b_hi_evals: Vec<F>,
	pub c_lo_evals: Vec<F>,
	pub c_hi_evals: Vec<F>,
}

/// Verify the binary-field multiplication check (BinMul) reduction.
///
/// The BinMul reduction proves the constraint $\widetilde{A}(x) \cdot \widetilde{B}(x) =
/// \widetilde{C}(x)$ for every $x$ on the boolean hypercube $\mathbb{B}_\ell$ (where $\ell =$
/// `n_vars`), over the GHASH field $\mathbb{F}_{2^{128}}$. Each field element is carried by a
/// `(lo, hi)` pair of 64-bit words via $\langle\langle z_{\textsf{lo}}, z_{\textsf{hi}}
/// \rangle\rangle = \sum_{i=0}^{63} z_{\textsf{lo},i} \cdot X^i + \sum_{i=0}^{63}
/// z_{\textsf{hi},i} \cdot X^{64+i}$, so $\widetilde{A}, \widetilde{B}, \widetilde{C}$ are the
/// $\mathbb{F}_{2^{128}}$-valued multilinears with those packed hypercube values.
///
/// ## Protocol
///
/// 1. The verifier samples $r_z \in K^\ell$.
/// 2. The parties run an $\ell$-round MLE-check on $0 = \sum_x \textsf{eq}(r_z, x) \cdot
///    (\widetilde{A}(x) \cdot \widetilde{B}(x) - \widetilde{C}(x))$, yielding a challenge point
///    $r_x$ and a final target `eval`. The composition $A \cdot B - C$ has degree 2; the eq factor
///    is folded internally by the MLE-check.
/// 3. The prover sends the six `Word::BITS`-wide per-bit evaluation columns at $r_x$; the verifier
///    recombines $\alpha_A = \sum_i \textsf{basis}(i) \cdot \widetilde{a}_{\textsf{lo}}(r_x, i) +
///    \textsf{basis}(64+i) \cdot \widetilde{a}_{\textsf{hi}}(r_x, i)$ (and $\alpha_B, \alpha_C$
///    likewise), and checks $\textsf{eval} = \alpha_A \cdot \alpha_B - \alpha_C$.
///
/// The reduction outputs the six per-bit evaluation columns at the shared point $r_x$.
pub fn verify<F, C>(n_vars: usize, channel: &mut C) -> Result<BinMulOutput<C::Elem>, Error>
where
	F: BinaryField,
	C: IPVerifierChannel<F>,
	C::Elem: From<F>,
{
	// Sample the zerocheck challenge r_z.
	let r_z = channel.sample_many(n_vars);

	// Run the degree-2 product MLE-check; the claimed value is zero. The eq factor is folded
	// internally by `mlecheck::verify`.
	let SumcheckOutput {
		eval,
		challenges: mut eval_point,
	} = mlecheck::verify(&r_z, 2, C::Elem::zero(), channel)?;
	eval_point.reverse();

	// The prover sends the six per-bit evaluation columns at r_x.
	let a_lo_evals = channel.recv_many(Word::BITS)?;
	let a_hi_evals = channel.recv_many(Word::BITS)?;
	let b_lo_evals = channel.recv_many(Word::BITS)?;
	let b_hi_evals = channel.recv_many(Word::BITS)?;
	let c_lo_evals = channel.recv_many(Word::BITS)?;
	let c_hi_evals = channel.recv_many(Word::BITS)?;

	// Precompute the basis-element vectors used to recombine a (lo, hi) per-bit column pair into
	// the packed field element evaluation alpha = sum_i basis(i) * lo[i] + basis(64 + i) * hi[i].
	let basis_lo: Vec<C::Elem> = (0..Word::BITS)
		.map(|i| C::Elem::from(<F as ExtensionField<BinaryField1b>>::basis(i)))
		.collect();
	let basis_hi: Vec<C::Elem> = (0..Word::BITS)
		.map(|i| C::Elem::from(<F as ExtensionField<BinaryField1b>>::basis(Word::BITS + i)))
		.collect();
	let recombine = |lo: &[C::Elem], hi: &[C::Elem]| -> C::Elem {
		inner_product_scalars(lo.iter().cloned(), basis_lo.iter().cloned())
			+ inner_product_scalars(hi.iter().cloned(), basis_hi.iter().cloned())
	};
	let alpha_a = recombine(&a_lo_evals, &a_hi_evals);
	let alpha_b = recombine(&b_lo_evals, &b_hi_evals);
	let alpha_c = recombine(&c_lo_evals, &c_hi_evals);

	// The MLE-check already folded the eq factor, so no explicit eq(r_z, r_x) term appears here.
	channel.assert_zero(alpha_a * &alpha_b - &alpha_c - &eval)?;

	Ok(BinMulOutput {
		eval_point,
		a_lo_evals,
		a_hi_evals,
		b_lo_evals,
		b_hi_evals,
		c_lo_evals,
		c_hi_evals,
	})
}
