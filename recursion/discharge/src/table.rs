//! Table extraction (spec section 1.1 layout), claim parsing, transparents, and the
//! native histogram/M_D rebuild used by the prover (Phase B input) and the verifier
//! (STEP-1 final check).
//!
//! Term semantics mirror crates/verifier/src/protocols/shift/monster.rs:165-219
//! (`evaluate_matrices`) EXACTLY: one term per `ShiftedValueIndex` occurrence in slot
//! m (0=a, 1=b, 2=c) of AND-constraint x. u_t is the 11-bit meta index:
//! s = shift amount (bits 0-5), op = shift variant id (bits 6-8), m = slot (bits 9-10).

use anyhow::{Context, bail, ensure};
use binius_core::constraint_system::{ConstraintSystem, ShiftVariant};
use binius_core::word::Word;
use binius_field::{Field, arithmetic_traits::InvertOrZero};
use binius_math::{
	BinarySubspace, FieldBuffer,
	multilinear::{
		eq::{
			eq_ind, eq_ind_partial_eval_scalars, eq_ind_zero, eq_one_var,
			scaled_eq_ind_partial_eval_scalars,
		},
		evaluate::evaluate,
	},
	univariate::lagrange_evals_scalars,
};
use binius_utils::SerializeBytes;
use binius_verifier::{
	config::B128,
	protocols::shift::{SHIFT_VARIANT_COUNT, evaluate_h_op},
};

// FWD-PORT (#1611-era config cleanup): `binius_verifier::config::LOG_WORD_SIZE_BITS` was removed;
// the canonical constant is now `binius_core::word::Word::LOG_BITS` (= 6). See NOTE finding.
const LOG_WORD_SIZE_BITS: usize = Word::LOG_BITS;
use sha2::Digest;

type B8 = binius_field::AESTowerField8b;

/// Number of meta-index variables: 6 (shift amount) + 3 (shift op) + 2 (operand slot).
pub const N_U: usize = 11;

/// One flattened AND-operand term (spec tuple tau_t = (x_t, y_t, u_t)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Term {
	/// AND-constraint index.
	pub x: u32,
	/// Witness word index (`ShiftedValueIndex::value_index`).
	pub y: u32,
	/// 11-bit meta index: s | op << 6 | m << 9.
	pub u: u16,
}

/// Shape dimensions of a prepared AND-only constraint system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeDims {
	/// log2 of the prepared AND-constraint count (= |r_x'_and|).
	pub n_x: usize,
	/// log2 of the prepared MUL-constraint count (= |r_x'_mul|); 0 for AND-only CS.
	pub n_x_mul: usize,
	/// SEGMENTED Y-block address width = `log_witness_words() + 1` (= the leaf's second
	/// sumcheck `log_word_count`). Since upstream #1554/#1583/#1724/#1585 the value-vec MLE
	/// is two zero-padded segments (public low half, hidden high half) selected by a top
	/// word-index variable `r_segment`. The Y histogram `D_seg` lives on the (lw+1)-var
	/// SEGMENTED CUBE: public word `y < n_pub` -> cube index `y`; hidden word `y >= n_pub`
	/// -> cube index `(y - n_pub) + 2^lw`. `r_y` in the claim point has length `lw = n_y-1`
	/// and is followed by the extra `r_segment` coordinate.
	pub n_y: usize,
	/// `log_public_words()` — the public segment occupies value indices `[0, n_pub)` and its
	/// eq tensor uses only these low `lp` word-index coordinates.
	pub lp: usize,
	/// `n_public_words()` (= `2^lp`, a power of two) — the public/hidden split boundary.
	pub n_pub: usize,
	/// `combined_len()` = `n_pub + n_hidden_words` — the number of value-vector words the
	/// constraint operands can reference (need NOT be a power of two). It is the length of the
	/// segmented `r_y_tensor` (indexed by flat value index), NOT a sumcheck domain width.
	pub combined_len: usize,
	/// Address width of M_D blocks: max(n_x, n_y, N_U, n_t - 2).
	///
	/// The `n_t - 2` term pads the M_D address space so that `n_d == max(n_t, n_d_natural)`,
	/// i.e. M_D's index domain and the term domain share one "low width" n_l := n_d. This is
	/// the STEP-2 union-domain alignment (spec section 1.1: for the nominal shape n_t = 25 the
	/// M_D oracle is 25-var and M_VK is 27-var); STEP 1 is layout-agnostic and uses the same
	/// padded layout so both steps share one M_D convention.
	pub n_a: usize,
	/// M_D variable count: n_a + 2 (two selector coordinates, HIGH per spec 1.1).
	pub n_d: usize,
	/// Number of real terms N.
	pub n_terms: usize,
	/// N padded to the next power of two (>= 1).
	pub n_pad: usize,
	/// log2(n_pad) — the Phase-A sumcheck variable count.
	pub n_t: usize,
	/// (n_pad - n_terms) mod 2 — the spec section 1.3 parity flag.
	pub parity: bool,
	/// Expected monster claim-point arity: 3 + n_x + n_x_mul + 6 + 6 + n_y.
	pub arity: usize,
}

/// The flattened term table of one CS shape.
#[derive(Debug, Clone)]
pub struct TermTable {
	pub dims: ShapeDims,
	pub terms: Vec<Term>,
	pub cs_digest: [u8; 32],
}

/// P0.3 ANDONLY admission predicate: every MUL constraint must equal the empty-operand
/// default produced by `validate_and_prepare` padding.
pub fn andonly(cs: &ConstraintSystem) -> bool {
	cs.mul_constraints
		.iter()
		.all(|mc| mc.a.is_empty() && mc.b.is_empty() && mc.hi.is_empty() && mc.lo.is_empty())
}

/// cs_digest := SHA-256 of the canonical versioned `SerializeBytes` encoding (P0.1/P0.2).
pub fn cs_digest_bytes(cs: &ConstraintSystem) -> [u8; 32] {
	let mut buf: Vec<u8> = Vec::new();
	cs.serialize(&mut buf).expect("ConstraintSystem::serialize is infallible for Vec<u8>");
	sha2::Sha256::digest(&buf).into()
}

fn strict_log2(v: usize, what: &str) -> anyhow::Result<usize> {
	ensure!(v.is_power_of_two(), "{what} = {v} is not a power of two (CS not prepared?)");
	Ok(v.ilog2() as usize)
}

/// Computes shape dims for a PREPARED (validate_and_prepare'd) constraint system.
pub fn shape_dims(cs: &ConstraintSystem) -> anyhow::Result<ShapeDims> {
	ensure!(andonly(cs), "P0.3 ANDONLY violated: CS has non-empty MUL constraints");
	let n_x = strict_log2(cs.and_constraints.len(), "and_constraints.len()")?;
	// FWD-PORT: the prepared AND-only CS now carries ZERO mul constraints (upstream dropped the
	// MIN_MUL_CONSTRAINTS=1 padding), so the intmul reduction emits an EMPTY r_x_mul. Match the
	// captured claim: n_x_mul = 0 when there are no mul constraints.
	let n_x_mul = if cs.mul_constraints.is_empty() {
		0
	} else {
		strict_log2(cs.mul_constraints.len(), "mul_constraints.len()")?
	};
	// RE-DERIVED (upstream #1554/#1583/#1724/#1585 segmented value vector): the value-vec MLE
	// is now two zero-padded segments (public low half, hidden high half) selected by a top
	// word-index variable. The captured claim carries `r_y` of length `log_witness_words()`
	// PLUS a trailing `r_segment`; the Y histogram lives on the `(lw+1)`-var segmented cube.
	// `n_y` is therefore the SEGMENTED Y-address width `lw + 1` (= the leaf `log_word_count`),
	// and is what the M_D (1,0) block / the y-point / the arity all key off. `combined_len`
	// (which need no longer be a power of two) is NOT used as a domain width anywhere.
	let layout = &cs.value_vec_layout;
	let lp = layout.log_public_words();
	let n_pub = layout.n_public_words();
	let lw = layout.log_witness_words();
	let combined_len = layout.combined_len();
	ensure!(n_pub.is_power_of_two(), "public segment length must be a power of two");
	ensure!(lw >= lp, "hidden segment must be at least as long as the public segment");
	let n_y = lw + 1;
	let n_terms: usize = cs
		.and_constraints
		.iter()
		.map(|c| c.a.len() + c.b.len() + c.c.len())
		.sum();
	ensure!(n_terms > 0, "empty term table");
	let n_pad = n_terms.next_power_of_two();
	let n_t = n_pad.ilog2() as usize;
	let n_a = n_x.max(n_y).max(N_U).max(n_t.saturating_sub(2));
	Ok(ShapeDims {
		n_x,
		n_x_mul,
		n_y,
		lp,
		n_pub,
		combined_len,
		n_a,
		n_d: n_a + 2,
		n_terms,
		n_pad,
		n_t,
		parity: (n_pad - n_terms) % 2 == 1,
		arity: 3 + n_x + n_x_mul + 2 * LOG_WORD_SIZE_BITS + n_y,
	})
}

/// Flattens the AND constraints into the canonical term list (constraint-major,
/// slot-major a->b->c, operand-position order). Mirrors `evaluate_matrices`
/// iteration order (monster.rs:165-219). Padded (empty) constraints contribute
/// zero terms, exactly as they contribute zero mults in the monster.
pub fn extract_table(cs: &ConstraintSystem) -> anyhow::Result<TermTable> {
	let dims = shape_dims(cs)?;
	let mut terms = Vec::with_capacity(dims.n_terms);
	for (x, con) in cs.and_constraints.iter().enumerate() {
		for (m, operand) in [&con.a, &con.b, &con.c].into_iter().enumerate() {
			for svi in operand {
				let shift_id = match svi.shift_variant {
					ShiftVariant::Sll => 0u16,
					ShiftVariant::Slr => 1,
					ShiftVariant::Sar => 2,
					ShiftVariant::Rotr => 3,
					ShiftVariant::Sll32 => 4,
					ShiftVariant::Srl32 => 5,
					ShiftVariant::Sra32 => 6,
					ShiftVariant::Rotr32 => 7,
				};
				ensure!(svi.amount < 64, "shift amount out of range");
				terms.push(Term {
					x: x as u32,
					y: svi.value_index.0,
					u: (svi.amount as u16) | (shift_id << 6) | ((m as u16) << 9),
				});
			}
		}
	}
	ensure!(terms.len() == dims.n_terms, "term count mismatch");
	Ok(TermTable {
		dims,
		terms,
		cs_digest: cs_digest_bytes(cs),
	})
}

/// Builds the SEGMENTED word-index tensor `R`, indexed by flat value index `y ∈ [0, combined_len)`,
/// EXACTLY as the leaf verifier's `MonsterEvalFn` (crates/verifier/src/protocols/shift/verify.rs:
/// 435-445): the public words occupy the low `n_pub = 2^lp` entries, the hidden words the next
/// `n_hidden = combined_len - n_pub` entries.
///
/// * public (`y < n_pub`): `R[y] = (1+r_seg)·eq_ind_zero(r_y[lp:])·eq(y, r_y[0:lp])`
/// * hidden (`y >= n_pub`): `R[y] = r_seg·eq(y - n_pub, r_y)`
///
/// `r_y.len() == lw = dims.n_y - 1`. This is the crux of the re-derivation: `R` is the monster's
/// `r_y_tensor`, so `native_term_sum` (which multiplies `R[t.y]`) reproduces `monster_eval`
/// bit-for-bit.
pub fn segmented_y_tensor(dims: &ShapeDims, r_y: &[B128], r_segment: B128) -> Vec<B128> {
	let lp = dims.lp;
	debug_assert_eq!(r_y.len(), dims.n_y - 1, "r_y must have length lw = n_y - 1");
	let public_scale = eq_one_var(r_segment, B128::ZERO) * eq_ind_zero(&r_y[lp..]);
	let public_tensor = scaled_eq_ind_partial_eval_scalars(&r_y[..lp], public_scale);
	let hidden_tensor = scaled_eq_ind_partial_eval_scalars(r_y, r_segment);
	let mut r_y_tensor = Vec::with_capacity(dims.combined_len);
	r_y_tensor.extend_from_slice(&public_tensor);
	r_y_tensor.extend_from_slice(&hidden_tensor[..dims.combined_len - dims.n_pub]);
	r_y_tensor
}

/// The segmented-CUBE index of a flat value index `y` (the address into the `(lw+1)`-var Y block
/// `D_seg` / the M_VK Y column): public words keep their index in the low half, hidden words are
/// re-indexed into the low positions of the high half.
///
/// * `y < n_pub`  -> `y`                    (segment 0)
/// * `y >= n_pub` -> `(y - n_pub) + 2^lw`   (segment 1)
///
/// This is exactly the compaction under which `R[y] = eq(cube_y(y), r_y ++ r_segment)`.
#[inline]
pub fn cube_y(dims: &ShapeDims, y: u32) -> u32 {
	let n_pub = dims.n_pub as u32;
	if y < n_pub {
		y
	} else {
		(y - n_pub) + (1u32 << (dims.n_y - 1))
	}
}

/// A captured deferred monster claim: the exact `compute_public_value` inputs (claim
/// point c_l, concatenation order of shift/verify.rs:244-252) and output (v_l).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
	pub point: Vec<B128>,
	pub value: B128,
}

/// Claim point parsed into named components.
#[derive(Debug, Clone)]
pub struct ParsedClaim {
	pub r_zhat: B128,
	pub lambda_and: B128,
	pub lambda_int: B128,
	pub r_x: Vec<B128>,
	pub r_x_mul: Vec<B128>,
	pub r_j: Vec<B128>,
	pub r_s: Vec<B128>,
	/// The word-index challenge, length `lw = log_witness_words() = n_y - 1` (low cube coords).
	pub r_y: Vec<B128>,
	/// The segment-selector challenge (the top word-index coordinate, appended after `r_y` in
	/// the captured claim point — shift/verify.rs:304). Weights the public (1+r_seg) vs hidden
	/// (r_seg) sub-tensors of the segmented value vector.
	pub r_segment: B128,
}

/// Parses a claim point against the shape dims; enforces the arity assert (P0.4).
pub fn parse_claim(dims: &ShapeDims, claim: &Claim) -> anyhow::Result<ParsedClaim> {
	ensure!(
		claim.point.len() == dims.arity,
		"claim arity mismatch: got {}, shape expects {}",
		claim.point.len(),
		dims.arity
	);
	let p = &claim.point;
	let mut off = 0usize;
	let mut take = |n: usize| {
		let s = p[off..off + n].to_vec();
		off += n;
		s
	};
	let head = take(3);
	let r_x = take(dims.n_x);
	let r_x_mul = take(dims.n_x_mul);
	let r_j = take(LOG_WORD_SIZE_BITS);
	let r_s = take(LOG_WORD_SIZE_BITS);
	// r_y has length lw = n_y - 1; r_segment is the final coordinate (the segment selector).
	let r_y = take(dims.n_y - 1);
	let r_segment = take(1)[0];
	ensure!(off == p.len(), "claim parse length bug");
	let parsed = ParsedClaim {
		r_zhat: head[0],
		lambda_and: head[1],
		lambda_int: head[2],
		r_x,
		r_x_mul,
		r_j,
		r_s,
		r_y,
		r_segment,
	};
	// (G) decomposition requires lambda_and not in {0, 1} (spec 1.2; FS-random, abort
	// probability <= 2/2^128 per claim).
	if parsed.lambda_and == B128::ZERO || parsed.lambda_and == B128::ONE {
		bail!("lambda_and in {{0,1}}: abort per spec section 1.2 (G)");
	}
	Ok(parsed)
}

/// The evaluation domain subspace used by the leaf verifier for l_tilde
/// (mirrors IOPVerifier::verify, crates/verifier/src/verify.rs:125-127).
fn domain_subspace() -> BinarySubspace<B128> {
	let subfield_subspace: BinarySubspace<B128> = BinarySubspace::<B8>::default().isomorphic();
	let extended = subfield_subspace.reduce_dim(LOG_WORD_SIZE_BITS + 1);
	extended.reduce_dim(LOG_WORD_SIZE_BITS)
}

/// h_op evaluations for a claim: `evaluate_h_op(l_tilde, r_j, r_s)` with
/// l_tilde = lagrange_evals_scalars(domain_subspace, r_zhat') — exactly the leaf
/// verifier's transparent (verify.rs:179 + shift/verify.rs:270-271).
pub fn h_ops_for_claim(parsed: &ParsedClaim) -> [B128; SHIFT_VARIANT_COUNT] {
	let subspace = domain_subspace();
	let l_tilde = lagrange_evals_scalars(&subspace, parsed.r_zhat);
	evaluate_h_op(&l_tilde, &parsed.r_j, &parsed.r_s)
}

/// The 2^11-entry meta-weight table: g_table[u] = lambda^{m+1} * h_op[op] * eq6(s, r_s).
/// This is the tensor-product multilinear G^{(l)} of spec 1.2 restricted to the cube.
pub fn g_table(
	lambda_and: B128,
	h_ops: &[B128; SHIFT_VARIANT_COUNT],
	r_s: &[B128],
) -> Vec<B128> {
	assert_eq!(r_s.len(), LOG_WORD_SIZE_BITS);
	let rs_tensor = eq_ind_partial_eval_scalars(r_s); // 64 entries, low-first
	let lam_pows = {
		// lambda^{m+1} for m = 0..4 (m = 3 is the MLE-consistent value on the unused slot)
		let mut v = [B128::ONE; 4];
		let mut acc = lambda_and;
		for slot in v.iter_mut() {
			*slot = acc;
			acc *= lambda_and;
		}
		v
	};
	let mut out = Vec::with_capacity(1 << N_U);
	for u in 0..(1usize << N_U) {
		let s = u & 63;
		let op = (u >> 6) & 7;
		let m = u >> 9;
		out.push(lam_pows[m] * h_ops[op] * rs_tensor[s]);
	}
	out
}

/// Per-claim transparent data reused across phases.
///
/// The verifier-side transparents (parsed point, h_ops, gammas, m-coords, w_d) cost
/// O(arity) mults; the eq TENSORS (2^n_x / 2^n_y / 2^11 entries) are PROVER data —
/// [`ClaimTransparents::new_light`] leaves them empty so the STEP-2 verifier carries
/// no O(2^n_x) term (spec: verifier = polylog + K*O(|point|)).
pub struct ClaimTransparents {
	pub parsed: ParsedClaim,
	pub h_ops: [B128; SHIFT_VARIANT_COUNT],
	/// eq tensor of r_x' (2^n_x entries; EMPTY in light contexts).
	pub x_tensor: Vec<B128>,
	/// eq tensor of r_y (2^n_y entries; EMPTY in light contexts).
	pub y_tensor: Vec<B128>,
	/// 2^11 meta weights (EMPTY in light contexts).
	pub g_tab: Vec<B128>,
	/// w_d: the single dummy-row weight (spec 1.3).
	pub dummy_weight: B128,
}

impl ClaimTransparents {
	pub fn new(dims: &ShapeDims, claim: &Claim) -> anyhow::Result<Self> {
		let mut ctx = Self::new_light(dims, claim)?;
		ctx.x_tensor = eq_ind_partial_eval_scalars(&ctx.parsed.r_x);
		// RE-DERIVED: the Y virtual column is the SEGMENTED tensor R (indexed by flat value
		// index), NOT the old flat eq(r_y). This makes native_term_sum == the segmented
		// monster_eval bit-for-bit.
		ctx.y_tensor = segmented_y_tensor(dims, &ctx.parsed.r_y, ctx.parsed.r_segment);
		ctx.g_tab = g_table(ctx.parsed.lambda_and, &ctx.h_ops, &ctx.parsed.r_s);
		debug_assert_eq!(
			ctx.dummy_weight,
			ctx.x_tensor[0] * ctx.y_tensor[0] * ctx.g_tab[0],
			"light w_d must equal the tensor-derived w_d"
		);
		Ok(ctx)
	}

	/// Verifier-side context: no eq tensors. w_d computed directly per spec 1.3:
	/// w_d = lambda_and * h_0 * prod(1+r_s_i) * prod(1+r_x'_i) * prod(1+r_y_i)
	/// (eq(0, r) = 1 + r in char 2).
	pub fn new_light(dims: &ShapeDims, claim: &Claim) -> anyhow::Result<Self> {
		let parsed = parse_claim(dims, claim)?;
		let h_ops = h_ops_for_claim(&parsed);
		let prod_one_plus = |v: &[B128]| {
			v.iter()
				.fold(B128::ONE, |acc, &r| acc * (B128::ONE + r))
		};
		// RE-DERIVED: the dummy row has y=0 (public segment), so its Y weight is
		// R[0] = (1+r_seg)·Π_{k<lw}(1+r_y[k]) — the extra (1+r_segment) factor is new.
		let dummy_weight = parsed.lambda_and
			* h_ops[0]
			* prod_one_plus(&parsed.r_s)
			* prod_one_plus(&parsed.r_x)
			* (B128::ONE + parsed.r_segment)
			* prod_one_plus(&parsed.r_y);
		Ok(Self {
			parsed,
			h_ops,
			x_tensor: Vec::new(),
			y_tensor: Vec::new(),
			g_tab: Vec::new(),
			dummy_weight,
		})
	}

	/// gamma_{l,o} = h_o * lambda * (lambda+1)^3 (spec 1.2 (G)).
	pub fn gammas(&self) -> [B128; SHIFT_VARIANT_COUNT] {
		let lam = self.parsed.lambda_and;
		let lp1 = lam + B128::ONE;
		let scale = lam * lp1 * lp1 * lp1;
		std::array::from_fn(|o| self.h_ops[o] * scale)
	}

	/// (p0, p1): the m-coordinates of the g-points z_{l,o}.
	/// p0 = (lambda+1)^{-1} + 1, p1 = ((lambda+1)^2)^{-1} + 1.
	pub fn m_coords(&self) -> (B128, B128) {
		let lp1 = self.parsed.lambda_and + B128::ONE;
		let p0 = lp1.invert_or_zero() + B128::ONE;
		let lp1_sq = lp1 * lp1;
		let p1 = lp1_sq.invert_or_zero() + B128::ONE;
		(p0, p1)
	}
}

/// Natively evaluates the term-list sum at the claim point over the REAL rows only.
/// Must equal the captured v_l bit-for-bit (the make-or-break table-extraction gate).
pub fn native_term_sum(table: &TermTable, tr: &ClaimTransparents) -> B128 {
	let mut acc = B128::ZERO;
	for t in &table.terms {
		acc += tr.x_tensor[t.x as usize] * tr.y_tensor[t.y as usize] * tr.g_tab[t.u as usize];
	}
	acc
}

/// The rho-weighted histograms D_x, D_seg, D_g (spec identity (D), re-derived), over ALL n_pad
/// rows (pads scatter to address 0 in each block).
///
/// `d_seg` is the SEGMENTED Y histogram over the `(lw+1)`-var cube (width `2^n_y`): the reduced
/// Y-column eval is `b_ℓ = D̃_seg([r_y, r_segment])` (a single M_D claim), replacing the old flat
/// `D̃_y(r_y)`.
pub struct Histograms {
	pub d_x: Vec<B128>,
	pub d_seg: Vec<B128>,
	pub d_g: Vec<B128>,
}

/// One table pass: expand eq(., rho) tensor (2^n_t), scatter-add into the three
/// histograms. rho must be low-coordinate-first. The Y address is the SEGMENTED cube index
/// `cube_y(term.y)`, so `D_seg` is the histogram whose MLE at `[r_y, r_segment]` reproduces the
/// segmented monster's reduced Y-column eval.
pub fn build_histograms(table: &TermTable, rho: &[B128]) -> Histograms {
	let dims = &table.dims;
	assert_eq!(rho.len(), dims.n_t);
	let eq_rho = eq_ind_partial_eval_scalars(rho);
	let mut d_x = vec![B128::ZERO; 1 << dims.n_x];
	let mut d_seg = vec![B128::ZERO; 1 << dims.n_y];
	let mut d_g = vec![B128::ZERO; 1 << N_U];
	for (t, term) in table.terms.iter().enumerate() {
		let w = eq_rho[t];
		d_x[term.x as usize] += w;
		d_seg[cube_y(dims, term.y) as usize] += w;
		d_g[term.u as usize] += w;
	}
	// Pad rows: fixed dummy tuple (x=0, y=0, u=0); y=0 maps to cube index 0 (public low).
	for &w in &eq_rho[dims.n_terms..] {
		d_x[0] += w;
		d_seg[0] += w;
		d_g[0] += w;
	}
	Histograms { d_x, d_seg, d_g }
}

/// Assembles the M_D buffer (2^{n_d} entries) per spec 1.1:
/// index = a + 2^{n_a} * (c0 + 2*c1); blocks c=(0,0) -> D_x, (1,0) -> D_seg (the SEGMENTED
/// Y histogram, 2^{n_y} = 2^{lw+1} entries), (0,1) -> D_g (zero-padded), (1,1) -> 0.
pub fn assemble_m_d(dims: &ShapeDims, h: &Histograms) -> FieldBuffer<B128> {
	let block = 1usize << dims.n_a;
	let mut vals = vec![B128::ZERO; 1 << dims.n_d];
	vals[..h.d_x.len()].copy_from_slice(&h.d_x);
	vals[block..block + h.d_seg.len()].copy_from_slice(&h.d_seg);
	vals[2 * block..2 * block + h.d_g.len()].copy_from_slice(&h.d_g);
	FieldBuffer::from_values(&vals)
}

/// The 10 M_D claim points for one claim, in canonical batch order:
/// [x-point, y-point, g-point_0, ..., g-point_7] (spec Phase A (iii) / Phase B).
/// All points low-coordinate-first, selectors at the top two coordinates.
pub fn m_d_points(dims: &ShapeDims, tr: &ClaimTransparents) -> Vec<Vec<B128>> {
	let mut points = Vec::with_capacity(10);
	let zero = B128::ZERO;
	let one = B128::ONE;
	// x-claim: [r_x' | 0^(n_a-n_x) | c=(0,0)]
	let mut px = tr.parsed.r_x.clone();
	px.resize(dims.n_a, zero);
	px.push(zero);
	px.push(zero);
	points.push(px);
	// y-claim (SEGMENTED): address = [r_y (lw) | r_segment | 0^(n_a - n_y) | c=(1,0)]. The eval
	// is b_ℓ = D̃_seg([r_y, r_segment]) — the single re-derived Y claim.
	let mut py = tr.parsed.r_y.clone();
	py.push(tr.parsed.r_segment);
	py.resize(dims.n_a, zero);
	py.push(one);
	py.push(zero);
	points.push(py);
	// g-claims: [r_s(6) | bits(o)(3) | p0,p1 | 0^(n_a-11) | c=(0,1)]
	let (p0, p1) = tr.m_coords();
	for o in 0..SHIFT_VARIANT_COUNT {
		let mut pg = tr.parsed.r_s.clone();
		for k in 0..3 {
			pg.push(if (o >> k) & 1 == 1 { one } else { zero });
		}
		pg.push(p0);
		pg.push(p1);
		pg.resize(dims.n_a, zero);
		pg.push(zero);
		pg.push(one);
		points.push(pg);
	}
	points
}

/// Evaluate the D_g MLE (11 vars) at an 11-coordinate prefix of a g-point.
pub fn evaluate_d_g(d_g: &[B128], point11: &[B128]) -> B128 {
	assert_eq!(d_g.len(), 1 << N_U);
	assert_eq!(point11.len(), N_U);
	let buf = FieldBuffer::<B128>::from_values(d_g);
	evaluate(&buf, point11)
}

/// eq_ind convenience for full points.
pub fn eq_point(a: &[B128], b: &[B128]) -> B128 {
	eq_ind(a, b)
}

/// Native monster reference: recompute the leaf verifier's monster_eval transparent
/// for a claim from the CS (used for the timing baseline "K x native monster ms").
/// This is exactly the closure body of shift/verify.rs:254-307 for an AND-only CS.
pub fn native_monster_eval(cs: &ConstraintSystem, parsed: &ParsedClaim) -> B128 {
	use binius_verifier::protocols::shift::evaluate_monster_multilinear_for_operation;
	use itertools::Itertools as _;
	// RE-DERIVED: build the SEGMENTED r_y_tensor exactly as MonsterEvalFn (verify.rs:435-445).
	let layout = &cs.value_vec_layout;
	let lp = layout.log_public_words();
	let n_pub = layout.n_public_words();
	let combined_len = layout.combined_len();
	let public_scale =
		eq_one_var(parsed.r_segment, B128::ZERO) * eq_ind_zero(&parsed.r_y[lp..]);
	let public_tensor = scaled_eq_ind_partial_eval_scalars(&parsed.r_y[..lp], public_scale);
	let hidden_tensor = scaled_eq_ind_partial_eval_scalars(&parsed.r_y, parsed.r_segment);
	let mut r_y_tensor = Vec::with_capacity(combined_len);
	r_y_tensor.extend_from_slice(&public_tensor);
	r_y_tensor.extend_from_slice(&hidden_tensor[..combined_len - n_pub]);
	let subspace = domain_subspace();
	let l_tilde = lagrange_evals_scalars(&subspace, parsed.r_zhat);
	let h_op_evals = evaluate_h_op(&l_tilde, &parsed.r_j, &parsed.r_s);
	// FWD-PORT (#1728 "Rewrite evaluate_monster_multilinear_for_operation"): the function now takes
	// a single pre-tensored `shift_scalars: &[E; SHIFT_VARIANT_COUNT * Word::BITS]` (indexed by
	// `variant * Word::BITS + amount`) in place of the old `(r_s, r_y_tensor, h_op_evals)` triple,
	// and returns `E` directly (no Result / no `.expect`). shift_scalars[i] = h_op[i/64]·eq(r_s)[i%64]
	// mirrors crates/verifier/src/protocols/shift/verify.rs:452-456.
	let eq_r_s = eq_ind_partial_eval_scalars(&parsed.r_s);
	let shift_scalars: [B128; SHIFT_VARIANT_COUNT * Word::BITS] =
		std::array::from_fn(|i| h_op_evals[i / Word::BITS] * eq_r_s[i % Word::BITS]);
	let bitand_part = {
		let (a, b, c) = cs
			.and_constraints
			.iter()
			.map(|con| (&con.a, &con.b, &con.c))
			.multiunzip();
		evaluate_monster_multilinear_for_operation::<B128, B128>(
			&[a, b, c],
			&parsed.r_x,
			parsed.lambda_and,
			&shift_scalars,
			&r_y_tensor,
		)
	};
	// FWD-PORT (empty-mul guard): match upstream MonsterEvalFn (verify.rs:469-485) — the prepared
	// AND-only CS carries ZERO mul constraints, and `evaluate_monster_multilinear_for_operation`
	// asserts `operands.len() == 1 << r_x_prime.len()` (monster.rs:130), which fails for empty
	// operand vecs. Skip the intmul contribution entirely when there are no mul constraints.
	let intmul_part = if cs.mul_constraints.is_empty() {
		B128::ZERO
	} else {
		let (a, b, lo, hi) = cs
			.mul_constraints
			.iter()
			.map(|con| (&con.a, &con.b, &con.lo, &con.hi))
			.multiunzip();
		evaluate_monster_multilinear_for_operation::<B128, B128>(
			&[a, b, lo, hi],
			&parsed.r_x_mul,
			parsed.lambda_int,
			&shift_scalars,
			&r_y_tensor,
		)
	};
	bitand_part + intmul_part
}

/// Loads a claim's context or fails with a uniform error prefix.
pub fn claim_context(dims: &ShapeDims, claim: &Claim) -> anyhow::Result<ClaimTransparents> {
	ClaimTransparents::new(dims, claim).context("claim intake")
}
