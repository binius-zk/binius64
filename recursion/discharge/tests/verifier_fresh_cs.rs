//! INDEPENDENT VERIFIER — fresh-CS cross-validation of the segmented Y-block.
//!
//! This test is written by an independent verifier (NOT the re-derivation author) to guard
//! against a hand-tuned segmentation that compiles and passes the author's fixed-shape
//! `synth_cs` but silently gets the public/hidden weighting wrong.
//!
//! It differs from `rederivation_step1.rs` in TWO load-bearing ways:
//!   1. DIFFERENT LAYOUT DIMS: lp=2, n_pub=4, lw=4, combined_len=16 (vs synth_cs's
//!      lp=3, n_pub=8, lw=5, combined_len=32). A segmentation that hard-codes synth's
//!      dimensions would break here.
//!   2. PUBLIC-REFERENCING OPERANDS: several a/b operands reference inout words (indices
//!      2,3 < n_pub), so the term table contains BOTH public (y<n_pub) and hidden
//!      (y>=n_pub) terms. `synth_cs` places EVERY operand at index >= 8 = hidden, so its
//!      cross-validate never tests the public `(1+r_segment)` sub-tensor against the real
//!      monster — it only enters via the self-consistent `w_d`. Here the public weighting
//!      is validated against the REAL upstream `monster_eval`.

use binius_core::constraint_system::{
	AndConstraint, ConstraintSystem, ShiftVariant, ShiftedValueIndex, ValueIndex, ValueVec,
	ValueVecLayout,
};
use binius_core::verify::verify_constraints;
use binius_core::word::Word;
use binius_field::Field;
use binius_math::multilinear::eq::{
	eq_ind_zero, scaled_eq_ind_partial_eval_scalars,
};
use binius_recursion_discharge::{
	discharge::{DischargeStatement, cross_validate_claim, discharge_prove, discharge_verify},
	leaf::LeafPipeline,
	recorder::verify_and_capture,
	table::{
		Claim, ClaimTransparents, extract_table, native_monster_eval, native_term_sum, parse_claim,
	},
};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::{B128, StdChallenger};

/// A fresh AND-only CS with a layout distinct from `synth_cs` and operands that reference
/// BOTH public (inout indices 2,3) and hidden (witness indices 4..15) words.
fn fresh_cs() -> ConstraintSystem {
	// Public segment: 2 const (0,1) + 2 inout (2,3) padded to offset_witness=4 -> n_pub=4, lp=2.
	// Hidden segment: 12 words (indices 4..15) -> lw = log2_ceil(12) = 4, combined_len = 16.
	let layout = ValueVecLayout {
		n_const: 2,
		n_inout: 2,
		n_witness: 12,
		n_internal: 0,
		offset_inout: 2,
		offset_witness: 4,
		n_hidden_words: 12,
		n_scratch: 0,
	};
	let constants = vec![Word(1), Word(42)];
	// (a_idx, a_variant, a_amt, b_idx, b_variant, b_amt, c_idx). Inputs are inout {2,3} (PUBLIC)
	// and witness {11,12,13,14,15} (HIDDEN); outputs are witness {4..10} (HIDDEN). No a/b ref is
	// an output word, so the witness solver order (inputs then outputs) is trivial.
	let specs: [(u32, ShiftVariant, u8, u32, ShiftVariant, u8, u32); 7] = [
		(2, ShiftVariant::Sll, 3, 11, ShiftVariant::Slr, 5, 4), // a PUBLIC (inout 2)
		(12, ShiftVariant::Sar, 7, 3, ShiftVariant::Rotr, 9, 5), // b PUBLIC (inout 3)
		(2, ShiftVariant::Rotr, 11, 13, ShiftVariant::Sll, 13, 6), // a PUBLIC (inout 2)
		(14, ShiftVariant::Slr, 15, 15, ShiftVariant::Sar, 17, 7), // both hidden
		(11, ShiftVariant::Sll, 19, 12, ShiftVariant::Rotr, 21, 8), // both hidden
		(3, ShiftVariant::Sar, 23, 13, ShiftVariant::Slr, 25, 9), // a PUBLIC (inout 3)
		(14, ShiftVariant::Rotr, 27, 15, ShiftVariant::Sll, 29, 10), // both hidden
	];
	let and_constraints = specs
		.iter()
		.map(|&(ai, av, aa, bi, bv, ba, ci)| AndConstraint {
			a: vec![ShiftedValueIndex { value_index: ValueIndex(ai), shift_variant: av, amount: aa }],
			b: vec![ShiftedValueIndex { value_index: ValueIndex(bi), shift_variant: bv, amount: ba }],
			c: vec![ShiftedValueIndex::plain(ValueIndex(ci))],
		})
		.collect();
	ConstraintSystem::new(constants, layout, and_constraints, Vec::new())
}

/// Builds a satisfying witness for the PREPARED `cs` (c = a & b for each real constraint).
fn fresh_witness(cs: &ConstraintSystem, instance: u64) -> anyhow::Result<ValueVec> {
	let mut vv = cs.new_value_vec();
	for (i, c) in cs.constants.iter().enumerate() {
		vv[ValueIndex(i as u32)] = *c;
	}
	// inout (public inputs)
	vv[ValueIndex(2)] = Word(instance);
	vv[ValueIndex(3)] = Word(instance.wrapping_mul(0x9e3779b97f4a7c15) ^ 0xa5a5a5a5);
	// hidden input words 11..15
	for (k, idx) in (11u32..=15).enumerate() {
		vv[ValueIndex(idx)] = Word(0x0102030405060708u64.wrapping_mul((k as u64) + 1) ^ instance);
	}
	// outputs c = a & b
	for con in cs.and_constraints.iter() {
		if con.a.is_empty() {
			continue; // padding constraint appended by validate_and_prepare
		}
		let a = vv.eval_operand(&con.a);
		let b = vv.eval_operand(&con.b);
		let c_index = con.c[0].value_index;
		vv[c_index] = Word(a.as_u64() & b.as_u64());
	}
	verify_constraints(cs, &vv).map_err(|e| anyhow::anyhow!("fresh witness unsatisfied: {e}"))?;
	Ok(vv)
}

#[test]
fn fresh_cs_cross_validates_against_real_monster() {
	let pipeline = LeafPipeline::setup(fresh_cs()).expect("leaf setup");
	let cs = pipeline.verifier.constraint_system().clone();
	let table = extract_table(&cs).expect("table");
	let d = &table.dims;
	eprintln!(
		"[fresh] N={} N_pad=2^{} parity={} n_x={} n_y(lw+1)={} lp={} n_pub={} combined_len={} arity={}",
		d.n_terms, d.n_t, d.parity, d.n_x, d.n_y, d.lp, d.n_pub, d.combined_len, d.arity
	);

	// This shape must genuinely differ from synth_cs (lp=3/n_pub=8/lw=5/combined=32).
	assert_eq!(d.lp, 2, "distinct public-log");
	assert_eq!(d.n_pub, 4, "distinct public size");
	assert_eq!(d.n_y - 1, 4, "distinct lw");
	assert_eq!(d.combined_len, 16, "distinct combined_len");

	// The term table MUST contain both a public (y<n_pub) and a hidden (y>=n_pub) term, else
	// the public sub-tensor is never exercised against the real monster.
	let n_pub = d.n_pub as u32;
	let n_public_terms = table.terms.iter().filter(|t| t.y < n_pub).count();
	let n_hidden_terms = table.terms.iter().filter(|t| t.y >= n_pub).count();
	assert!(n_public_terms >= 1, "need >=1 public-referencing term (got {n_public_terms})");
	assert!(n_hidden_terms >= 1, "need >=1 hidden-referencing term (got {n_hidden_terms})");
	eprintln!("[fresh] public terms={n_public_terms} hidden terms={n_hidden_terms}");

	// Capture two REAL deferred monster claims from the upstream Verifier::verify.
	for seed in [7000u64, 7001] {
		let wit = fresh_witness(&cs, seed).expect("witness");
		let public = LeafPipeline::public(&wit);
		let proof = pipeline.prove(&wit).expect("prove");
		let claim = verify_and_capture(&pipeline.verifier, &public, proof, d.arity).expect("capture");

		// Sanity: the claim point ends with r_segment; r_y has length lw.
		let parsed = parse_claim(d, &claim).expect("parse");
		assert_eq!(parsed.r_y.len(), d.n_y - 1);

		// (1) GATE: native segmented term-sum reproduces the REAL captured monster value.
		//     With public terms present, this validates BOTH the public (1+r_seg) and hidden
		//     (r_seg) sub-tensors against reality.
		cross_validate_claim(&table, &claim)
			.unwrap_or_else(|e| panic!("[fresh] cross-validate FAILED (re-derivation wrong): {e}"));

		// (2) Independent check via upstream's own evaluate_monster_multilinear_for_operation.
		assert_eq!(
			native_monster_eval(&cs, &parsed),
			claim.value,
			"native_monster_eval must equal the captured monster value"
		);

		// (3) DIFFERENTIAL control: a WRONG public weighting (r_segment instead of the correct
		//     (1+r_segment)) must DIVERGE from the real monster. Because public terms exist, this
		//     proves the exact (1+r_segment) factor is load-bearing (not a self-consistent artifact).
		let mut tr = ClaimTransparents::new(d, &claim).expect("transparents");
		tr.y_tensor = wrong_public_tensor(d, &parsed.r_y, parsed.r_segment);
		let wrong_sum = native_term_sum(&table, &tr);
		assert_ne!(
			wrong_sum, claim.value,
			"WRONG public weighting matched the monster — public sub-tensor not discriminated"
		);
		eprintln!("[fresh] seed {seed}: cross-validate OK, native_monster_eval OK, wrong-public DIVERGES");
	}
	eprintln!("[fresh] PASS: fresh-shape segmented Y-block reproduces the real monster bit-for-bit");
}

/// STEP-1 discharge E2E + negatives on the FRESH (un-tuned) shape: proves the segmented
/// histogram path (cube_y -> D_seg, segmented y-point, w_d with (1+r_segment), native M_D
/// rebuild) is self-consistent on a shape the author never hand-tuned, and that soundness
/// checks REJECT tampering for the right reason.
#[test]
fn fresh_cs_step1_discharge_and_negatives() {
	let pipeline = LeafPipeline::setup(fresh_cs()).expect("leaf setup");
	let cs = pipeline.verifier.constraint_system().clone();
	let table = extract_table(&cs).expect("table");
	assert!(table.dims.parity, "fresh shape is odd-parity (w_d correction load-bearing)");

	let mut claims: Vec<Claim> = Vec::new();
	for seed in [8000u64, 8001, 8002] {
		let wit = fresh_witness(&cs, seed).expect("witness");
		let public = LeafPipeline::public(&wit);
		let proof = pipeline.prove(&wit).expect("prove");
		claims.push(verify_and_capture(&pipeline.verifier, &public, proof, table.dims.arity).expect("capture"));
	}

	// Positive: STEP-1 discharge verifies K=3 real captured claims of the fresh shape.
	let stmt = DischargeStatement::new(&table, claims.clone()).expect("stmt");
	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove(&cs, &stmt, &mut pt).expect("prove");
	let bytes = pt.finalize();
	let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes.clone());
	discharge_verify(&cs, &stmt, &mut vt).expect("verify");
	vt.finalize().expect("finalize");
	eprintln!("[fresh] STEP-1 E2E on 3 fresh captured claims OK ({} bytes)", bytes.len());

	// Negative A (tampered r_y coordinate, not r_segment): a flipped word-index challenge must
	// break cross-validate — the segmented Y-block binds ALL of r_y, not just the segment bit.
	// Layout: head(3) | r_x(3) | r_j(6) | r_s(6) | r_y(4)=[18..22] | r_segment[22]. Index 20 is r_y[2].
	let mut bad_point = claims[0].clone();
	bad_point.point[20] += B128::ONE;
	assert!(
		cross_validate_claim(&table, &bad_point).is_err(),
		"a tampered claim-point coordinate MUST fail cross-validate"
	);

	// Negative B (tampered value -> Phase A / native M_D check): the discharge verifier must
	// reject a statement whose claimed monster value is wrong, because the CS-rebuilt M_D binds
	// the reduced columns to the TRUE value.
	let mut bad_claims = claims;
	bad_claims[1].value += B128::ONE;
	let bad_stmt = DischargeStatement::new(&table, bad_claims).expect("stmt");
	let mut pt2 = ProverTranscript::new(StdChallenger::default());
	discharge_prove(&cs, &bad_stmt, &mut pt2).expect("prover proves the given (wrong) sum");
	let bad_bytes = pt2.finalize();
	let mut vt2 = VerifierTranscript::new(StdChallenger::default(), bad_bytes);
	assert!(
		discharge_verify(&cs, &bad_stmt, &mut vt2).is_err(),
		"a tampered monster value MUST be rejected by the STEP-1 discharge verifier"
	);
	eprintln!("[fresh] negatives (tampered-point, tampered-value) REJECT correctly");
}

/// Same as `table::segmented_y_tensor` but with the public segment scaled by `r_segment`
/// instead of the correct `(1 + r_segment) = eq_one_var(r_segment, 0)`.
fn wrong_public_tensor(
	dims: &binius_recursion_discharge::table::ShapeDims,
	r_y: &[B128],
	r_segment: B128,
) -> Vec<B128> {
	let lp = dims.lp;
	let wrong_scale = r_segment * eq_ind_zero(&r_y[lp..]);
	let public_tensor = scaled_eq_ind_partial_eval_scalars(&r_y[..lp], wrong_scale);
	let hidden_tensor = scaled_eq_ind_partial_eval_scalars(r_y, r_segment);
	let mut out = Vec::with_capacity(dims.combined_len);
	out.extend_from_slice(&public_tensor);
	out.extend_from_slice(&hidden_tensor[..dims.combined_len - dims.n_pub]);
	out
}
