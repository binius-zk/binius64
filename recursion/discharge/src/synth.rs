//! Synthetic AND-only constraint systems and standalone monster claims — the
//! Binius64-general, application-free reproduction path for the discharge headline
//! numbers.
//!
//! Two families live here:
//!
//! 1. A TINY odd-parity shape ([`synth_cs`] / [`synth_witness`]): 5 real AND
//!    constraints, N = 15, N_pad = 16, so `N_pad - N = 1` is ODD and the spec §1.3
//!    `w_d` parity correction is load-bearing. Two variants share every shape dimension
//!    but have DIFFERENT term tables (different shift meta), giving a same-dims foreign
//!    shape for the table-swap negatives. These are proven for real by
//!    [`crate::leaf::LeafPipeline`] and the deferred claim is captured through the
//!    recorder, so the tiny path exercises the FULL real capture pipeline.
//!
//! 2. A SIZED shape ([`synth_cs_sized`] / [`prepared_synth_table`]) plus a STANDALONE
//!    claim synthesizer ([`synth_claim`]). The discharge protocol's inputs are K
//!    `(claim point c_l, value v_l = monster_eval(c_l))` pairs; whether they come from a
//!    real leaf proof (captured through the recorder) or are generated directly from the
//!    term table is irrelevant to the discharge machinery — this is exactly the spec
//!    §P0.4 "standalone path" where "the (c, v) pairs ARE the statement". Synthesizing
//!    claims natively (`v_l := native_term_sum(table, c_l)`) lets us drive the discharge
//!    at a LARGE term count without a large real proof, which is what reproduces the
//!    N-independence of the STEP-2 verify endgame.

use anyhow::ensure;
use binius_core::{
	constraint_system::{
		AndConstraint, ConstraintSystem, ShiftVariant, ShiftedValueIndex, ValueIndex, ValueVec,
		ValueVecLayout,
	},
	verify::verify_constraints,
	word::Word,
};
use binius_field::{Field, Random};
use binius_verifier::config::B128;
use rand::{SeedableRng, rngs::StdRng};

use crate::table::{Claim, ClaimTransparents, ShapeDims, TermTable, extract_table, native_term_sum};

/// Deterministic input words (indices 8..13).
const INPUTS: [u64; 5] = [
	0xdeadbeef_cafe1234,
	0x0f0f0f0f_33cc55aa,
	0xffff0000_ffff0000,
	0x12345678_9abcdef0,
	0xa5a5a5a5_5a5a5a5a,
];

/// Builds one synthetic AND-only CS. `variant` selects the shift meta (0 or 1): both
/// variants have identical shape dims but different term tables.
///
/// Term count N = 15 (5 constraints x 3 single-operand terms) and N_pad = 16, so
/// N_pad - N = 1 is ODD: parity = true (the spec 1.3 w_d correction is load-bearing).
pub fn synth_cs(variant: u8) -> ConstraintSystem {
	let constants = vec![Word(1), Word(42), Word(0xDEADBEEF)];
	// Segmented layout (upstream #1554/#1724): public segment = offset_witness = 8 words
	// (n_pub=8, lp=3); hidden segment = 24 words (lw=log2_ceil(24)=5); combined_len = 32.
	let layout = ValueVecLayout {
		n_const: 3,
		n_inout: 2,
		n_witness: 24,
		n_internal: 0,
		offset_inout: 4,
		offset_witness: 8,
		n_hidden_words: 24,
		n_scratch: 0,
	};
	let mut and_constraints = Vec::new();
	for i in 0..5u32 {
		let (va, sa, vb, sb) = shift_meta(variant, i);
		let a = ShiftedValueIndex {
			value_index: ValueIndex(8 + i),
			shift_variant: va,
			amount: sa,
		};
		let b = ShiftedValueIndex {
			value_index: ValueIndex(8 + ((i + 1) % 5)),
			shift_variant: vb,
			amount: sb,
		};
		let c = ShiftedValueIndex::plain(ValueIndex(16 + i));
		and_constraints.push(AndConstraint {
			a: vec![a],
			b: vec![b],
			c: vec![c],
		});
	}
	ConstraintSystem::new(constants, layout, and_constraints, Vec::new())
}

fn shift_meta(variant: u8, i: u32) -> (ShiftVariant, u8, ShiftVariant, u8) {
	match variant {
		0 => (ShiftVariant::Sll, ((3 * i as usize + 1) % 64) as u8, ShiftVariant::Slr, ((5 * i as usize + 2) % 64) as u8),
		_ => (ShiftVariant::Rotr, ((7 * i as usize + 3) % 64) as u8, ShiftVariant::Sar, ((11 * i as usize + 5) % 64) as u8),
	}
}

/// Builds a satisfying witness for `synth_cs(variant)` with the given public instance
/// word (varying it yields genuinely distinct proofs/claims of the same CS shape).
pub fn synth_witness(cs: &ConstraintSystem, instance: u64) -> anyhow::Result<ValueVec> {
	let mut vv = cs.new_value_vec();
	for (i, c) in cs.constants.iter().enumerate() {
		vv[ValueIndex(i as u32)] = *c;
	}
	vv[ValueIndex(4)] = Word(instance);
	vv[ValueIndex(5)] = Word(instance.wrapping_mul(0x9e3779b97f4a7c15));
	for (i, w) in INPUTS.iter().enumerate() {
		vv[ValueIndex(8 + i as u32)] = Word(*w);
	}
	// Solve the c-words: c = (A & B) for each constraint (single-term a/b operands).
	for (idx, con) in cs.and_constraints.iter().enumerate() {
		if con.a.is_empty() {
			continue; // padding constraint
		}
		let a = vv.eval_operand(&con.a);
		let b = vv.eval_operand(&con.b);
		let c_index = con.c[0].value_index;
		ensure!(
			con.c.len() == 1 && con.c[0].amount == 0,
			"constraint {idx}: expected plain single-term c"
		);
		vv[c_index] = Word(a.as_u64() & b.as_u64());
	}
	verify_constraints(cs, &vv).map_err(|e| anyhow::anyhow!("synth witness unsatisfied: {e}"))?;
	Ok(vv)
}

// ---------------------------------------------------------------------------
// Sized synthetic shape + standalone claim synthesis (large-N scaling path).
// ---------------------------------------------------------------------------

/// Builds a RAW (un-prepared) synthetic AND-only CS with `1 << log2_constraints`
/// single-operand AND constraints over a `1 << n_y_log`-word value vector. Each
/// constraint contributes three terms (one `a`, one `b`, one `c` occurrence), so the
/// prepared term table has exactly `N = 3 << log2_constraints` terms. `variant` selects
/// the shift meta so two calls give same-dims / different-table shapes.
///
/// The caller must run [`ConstraintSystem::validate_and_prepare`] before extraction (see
/// [`prepared_synth_table`]); the constraint count is already a power of two so
/// preparation only appends the empty MUL padding (keeping the shape AND-only).
pub fn synth_cs_sized(log2_constraints: usize, n_y_log: usize, variant: u8) -> ConstraintSystem {
	assert!(n_y_log >= 4, "value vector too small to index operands");
	let combined_total_len = 1usize << n_y_log;
	let offset_witness = 8usize;
	let witness_span = combined_total_len - offset_witness;
	// combined_len = offset_witness + n_hidden_words = 2^n_y_log (all operands land in the
	// hidden segment; the public segment holds only constants/inout).
	let layout = ValueVecLayout {
		n_const: 3,
		n_inout: 2,
		n_witness: witness_span,
		n_internal: 0,
		offset_inout: 4,
		offset_witness,
		n_hidden_words: witness_span,
		n_scratch: 0,
	};
	let constants = vec![Word(1), Word(42), Word(0xDEADBEEF)];
	let c = 1usize << log2_constraints;
	// A small per-variant multiplier set so variants differ in shift meta everywhere.
	let (ka, kb, kc): (u64, u64, u64) = match variant {
		0 => (1, 2, 3),
		_ => (5, 7, 11),
	};
	let idx = |seed: u64| offset_witness + (seed as usize % witness_span);
	// Canonical non-plain shift amount in [1, 31]: amount 0 is legal only for the SLL
	// "plain" form, and upstream `validate_and_prepare` rejects amount >= 32 for the
	// 32-bit-lane variants (Sll32/Srl32/Sra32/Rotr32) this table mixes in. Capping at 31 is
	// valid for all eight variants and the discharge is agnostic to the exact amount.
	let amt = |seed: u64| (1 + (seed as usize % 31)) as u8;
	let mut and_constraints = Vec::with_capacity(c);
	for i in 0..c as u64 {
		let a = ShiftedValueIndex {
			value_index: ValueIndex(idx(ka.wrapping_mul(i).wrapping_add(1)) as u32),
			shift_variant: SHIFT_VARIANTS[(ka.wrapping_mul(i) as usize) % SHIFT_VARIANTS.len()],
			amount: amt(3 * i + 1),
		};
		let b = ShiftedValueIndex {
			value_index: ValueIndex(idx(kb.wrapping_mul(i).wrapping_add(2)) as u32),
			shift_variant: SHIFT_VARIANTS[(kb.wrapping_mul(i) as usize) % SHIFT_VARIANTS.len()],
			amount: amt(5 * i + 2),
		};
		let c_op = ShiftedValueIndex::plain(ValueIndex(idx(kc.wrapping_mul(i).wrapping_add(3)) as u32));
		and_constraints.push(AndConstraint {
			a: vec![a],
			b: vec![b],
			c: vec![c_op],
		});
	}
	ConstraintSystem::new(constants, layout, and_constraints, Vec::new())
}

const SHIFT_VARIANTS: [ShiftVariant; 8] = [
	ShiftVariant::Sll,
	ShiftVariant::Slr,
	ShiftVariant::Sar,
	ShiftVariant::Rotr,
	ShiftVariant::Sll32,
	ShiftVariant::Srl32,
	ShiftVariant::Sra32,
	ShiftVariant::Rotr32,
];

/// Prepares a sized synthetic CS and extracts its term table. Returns both the prepared
/// CS (for the STEP-1 native-check verifier / the CS-driven STEP-2 prover) and the term
/// table (for VKGen and standalone claim synthesis).
pub fn prepared_synth_table(
	log2_constraints: usize,
	n_y_log: usize,
	variant: u8,
) -> anyhow::Result<(ConstraintSystem, TermTable)> {
	let mut cs = synth_cs_sized(log2_constraints, n_y_log, variant);
	cs.validate_and_prepare()
		.map_err(|e| anyhow::anyhow!("validate_and_prepare: {e}"))?;
	let table = extract_table(&cs)?;
	Ok((cs, table))
}

/// Synthesizes ONE standalone monster claim for `table`: a random claim point of the
/// shape's arity (little-endian scalar order per `parse_claim`, with `lambda_and` forced
/// out of {0, 1}) together with its TRUE deferred value `v = native_term_sum(table, c)`.
///
/// The resulting `(c, v)` is exactly what leaf `check_eval` would defer, minus the leaf
/// proof — the discharge cannot tell the difference (spec §P0.4 standalone path). K
/// distinct seeds give K distinct claims of one shape.
pub fn synth_claim(dims: &ShapeDims, table: &TermTable, seed: u64) -> anyhow::Result<Claim> {
	let mut rng = StdRng::seed_from_u64(seed);
	let mut point: Vec<B128> = (0..dims.arity).map(|_| B128::random(&mut rng)).collect();
	// Index 1 is lambda_and; the (G) decomposition aborts on {0, 1}, so resample it out.
	loop {
		let lam = B128::random(&mut rng);
		if lam != B128::ZERO && lam != B128::ONE {
			point[1] = lam;
			break;
		}
	}
	let claim0 = Claim {
		point,
		value: B128::ZERO,
	};
	let tr = ClaimTransparents::new(dims, &claim0)?;
	let value = native_term_sum(table, &tr);
	Ok(Claim {
		point: claim0.point,
		value,
	})
}

/// K distinct standalone claims of one sized shape (seeds `seed_base + 0..K`).
pub fn synth_claims(dims: &ShapeDims, table: &TermTable, seed_base: u64, k: usize) -> anyhow::Result<Vec<Claim>> {
	(0..k as u64).map(|i| synth_claim(dims, table, seed_base + i)).collect()
}
