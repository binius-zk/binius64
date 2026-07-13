#![cfg(feature = "step2")]
//! Independent-verifier STEP-2 adversarial tests (written by the reviewing session,
//! NOT the builder): a fully COPIED STEP-2 prover (no crate tamper knobs) that follows
//! the honest protocol and lies at ONE chosen message. Assert on the ERROR KIND.
//!
//! Knobs:
//! - NoTamper       -> control: the copied prover must verify. The copy computes the
//!                     VK corner values by DIRECT block-MLE evaluation (vs the crate's
//!                     fused pad-mass pass) and W_eq the NAIVE way (full tensors), so
//!                     the control also cross-checks those two crate optimizations.
//! - CornerForgery  -> v10 += eq_blk[2](pi_hi), v01 += eq_blk[1](pi_hi): the den
//!                     transparent is UNCHANGED (char-2 cancellation:
//!                     eq1*e10 + eq2*e01 = eq1*eq2 + eq2*eq1 = 0), num transparent
//!                     does not involve corners, Phase B and the M_D opening are
//!                     untouched. The ONLY thing that can catch it is the corner-trick
//!                     M_VK opening at the post-hoc random rho_c. Tests that the
//!                     opening claim is derived from the recv'd corners AND that
//!                     sumcheck_fri_consistency is live on the VK opening.
//! - MPiLie         -> m_pi += 1 at the corner message: the fracaddcheck tree is
//!                     honest, so the verifier's Phase-C NUMERATOR transparent must
//!                     reject (num_expected uses the lied m_pi, leaf num_eval is true).

use binius_core::constraint_system::ConstraintSystem;
use binius_field::Field;
use binius_hash::StdHashSuite;
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_ip_prover::{
	channel::IPProverChannel,
	fracaddcheck::FracAddCheckProver,
	sumcheck::{
		batch::batch_prove_and_write_evals, bivariate_product::BivariateProductSumcheckProver,
		prove_single,
	},
};
use binius_iop_prover::{
	fri::encode_interleaved,
	merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel},
};
use binius_math::{
	FieldBuffer,
	multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
	univariate::evaluate_univariate,
};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::Challenger};
use binius_verifier::{
	config::{B128, StdChallenger},
	protocols::shift::SHIFT_VARIANT_COUNT,
};
use binius_recursion_discharge::{
	cubic::CubicProductSumcheckProver,
	discharge::DischargeStatement,
	leaf::LeafPipeline,
	merged::OpenClaim,
	recorder::verify_and_capture,
	step2::discharge_verify_step2,
	synth::{synth_cs, synth_witness},
	table::{ClaimTransparents, TermTable, assemble_m_d, build_histograms, cube_y, eq_point, evaluate_d_g, extract_table, m_d_points, N_U},
	vk::{DischargeVkm, build_m_vk, build_ntt, build_pcs, encode_m_vk, tags, vk_entry, vkgen},
};

const K: usize = 2;

#[derive(Clone, Copy, PartialEq)]
enum Tamper {
	NoTamper,
	CornerForgery,
	MPiLie,
}

/// Copied STEP-2 prover with tamper knobs at the Phase-C corner message.
fn copied_prove_step2<C: Challenger>(
	cs: &ConstraintSystem,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	tamper: Tamper,
	transcript: &mut ProverTranscript<C>,
) -> anyhow::Result<()> {
	let table: TermTable = extract_table(cs)?;
	anyhow::ensure!(table.cs_digest == vkm.cs_digest && table.dims == vkm.dims, "shape");
	let dims = table.dims.clone();
	let n_l = dims.n_d;
	let block = 1usize << n_l;
	let k = stmt.claims.len();

	let ctxs: Vec<ClaimTransparents> = stmt
		.claims
		.iter()
		.map(|c| ClaimTransparents::new(&dims, c).expect("claim intake"))
		.collect();
	let sums: Vec<B128> = stmt
		.claims
		.iter()
		.zip(&ctxs)
		.map(|(c, tr)| {
			let mut s = c.value;
			if stmt.parity {
				s += tr.dummy_weight;
			}
			s
		})
		.collect();

	let pcs = build_pcs(vkm)?;
	let ntt = build_ntt(pcs.log_domain);

	// Phase 0: VKM + statement absorption (must match the crate's encodings).
	IPProverChannel::<B128>::observe_many(transcript, &vkm.to_elems());
	let stmt_elems = {
		let mut elems: Vec<B128> = Vec::new();
		let lo = u128::from_le_bytes(stmt.cs_digest[..16].try_into().unwrap());
		let hi = u128::from_le_bytes(stmt.cs_digest[16..].try_into().unwrap());
		elems.push(B128::new(lo));
		elems.push(B128::new(hi));
		elems.push(B128::new(stmt.n_terms as u128));
		elems.push(B128::new(stmt.n_pad as u128));
		elems.push(B128::new(stmt.parity as u128));
		elems.push(B128::new(stmt.claims.len() as u128));
		for c in &stmt.claims {
			elems.extend_from_slice(&c.point);
			elems.push(c.value);
		}
		elems
	};
	IPProverChannel::<B128>::observe_many(transcript, &stmt_elems);

	// Phase A.
	let mut provers = Vec::with_capacity(k);
	for (tr, sum) in ctxs.iter().zip(&sums) {
		let mut e_x = Vec::with_capacity(dims.n_pad);
		let mut e_y = Vec::with_capacity(dims.n_pad);
		let mut e_g = Vec::with_capacity(dims.n_pad);
		for term in &table.terms {
			e_x.push(tr.x_tensor[term.x as usize]);
			e_y.push(tr.y_tensor[term.y as usize]);
			e_g.push(tr.g_tab[term.u as usize]);
		}
		e_x.resize(dims.n_pad, tr.x_tensor[0]);
		e_y.resize(dims.n_pad, tr.y_tensor[0]);
		e_g.resize(dims.n_pad, tr.g_tab[0]);
		provers.push(CubicProductSumcheckProver::new(
			[
				FieldBuffer::<B128>::from_values(&e_x),
				FieldBuffer::<B128>::from_values(&e_y),
				FieldBuffer::<B128>::from_values(&e_g),
			],
			*sum,
		));
	}
	let output = batch_prove_and_write_evals(provers, transcript);
	let rho = output.challenges.clone(); // low-first on the prover side

	// Histograms + 8K d values.
	let hist = build_histograms(&table, &rho);
	let mut d_vals = Vec::with_capacity(k * SHIFT_VARIANT_COUNT);
	for tr in &ctxs {
		let points = m_d_points(&dims, tr);
		for point in &points[2..] {
			d_vals.push(evaluate_d_g(&hist.d_g, &point[..N_U]));
		}
	}
	IPProverChannel::<B128>::send_many(transcript, &d_vals);

	// Commit M_D (honest; oracle 1 of the batched params) over a Merkle channel, which
	// writes digest_D as an observed message before tau.
	let m_d = assemble_m_d(&dims, &hist);
	let m_d_codeword = encode_interleaved(&pcs.params, 1, &ntt, m_d.to_ref());
	let d_commitment = {
		let mut mchan =
			ProverMerkleTranscriptChannel::<_, C, B128, StdHashSuite>::new(&mut *transcript);
		mchan.send_merkle_commitment(m_d_codeword.to_ref(), pcs.leaf_size(1))
	};

	// Phase C: leaf columns built the straightforward way (independent of the crate's
	// direct-write builder).
	let tau: B128 = IPProverChannel::<B128>::sample(transcript);
	anyhow::ensure!(tau.val() >= (1u128 << (dims.n_a + 2)).into(), "tau pole guard");
	let kappa = tags(dims.n_a);
	let eq_rho = eq_ind_partial_eval_scalars(&rho);
	let mut num = vec![B128::ZERO; block << 2];
	num[..eq_rho.len()].copy_from_slice(&eq_rho);
	num.copy_within(..block, block);
	num.copy_within(..block, 2 * block);
	num[3 * block..].copy_from_slice(m_d.as_ref());
	let mut den = vec![B128::ZERO; block << 2];
	for t in 0..block {
		if t < table.terms.len() {
			let term = &table.terms[t];
			den[t] = tau + vk_entry(0, term.x as u64, dims.n_a);
			den[block + t] = tau + vk_entry(1, cube_y(&dims, term.y) as u64, dims.n_a);
			den[2 * block + t] = tau + vk_entry(2, term.u as u64, dims.n_a);
		} else {
			den[t] = tau; // kappa_x = 0
			den[block + t] = tau + kappa[1];
			den[2 * block + t] = tau + kappa[2];
		}
		den[3 * block + t] = tau + B128::new(t as u128);
	}
	let (frac_prover, roots) = FracAddCheckProver::<B128>::new(
		n_l + 2,
		(
			FieldBuffer::<B128>::from_values(&num),
			FieldBuffer::<B128>::from_values(&den),
		),
	);
	let num_root = roots.0.get(0);
	let den_root = roots.1.get(0);
	anyhow::ensure!(num_root == B128::ZERO, "copied prover: num root must be 0 (honest data)");
	anyhow::ensure!(den_root != B128::ZERO, "copied prover: den root zero");
	IPProverChannel::<B128>::send_one(transcript, den_root);
	let root_claim = (
		MultilinearEvalClaim {
			eval: B128::ZERO,
			point: Vec::new(),
		},
		MultilinearEvalClaim {
			eval: den_root,
			point: Vec::new(),
		},
	);
	let (num_leaf_claim, _den_leaf_claim) = frac_prover.prove(root_claim, transcript);
	let pi = num_leaf_claim.point;
	anyhow::ensure!(pi.len() == n_l + 2, "leaf point arity");
	let (pi_lo, pi_hi) = pi.split_at(n_l);

	// Corner values by DIRECT block-MLE evaluation of the committed M_VK message
	// (independent cross-check of the crate's fused pad-mass pass).
	let m_vk = build_m_vk(&table);
	let v_honest: [B128; 3] = {
		let vals = m_vk.as_ref();
		let x = evaluate(&FieldBuffer::<B128>::from_values(&vals[..block]), pi_lo);
		let y = evaluate(&FieldBuffer::<B128>::from_values(&vals[block..2 * block]), pi_lo);
		let u = evaluate(&FieldBuffer::<B128>::from_values(&vals[2 * block..3 * block]), pi_lo);
		[x, y, u]
	};
	let m_pi = evaluate(&m_d, pi_lo);

	// ---- TAMPER POINT: the corner message. ----
	let zero = B128::ZERO;
	let one = B128::ONE;
	let eq_blk = [
		eq_point(&[zero, zero], pi_hi),
		eq_point(&[one, zero], pi_hi),
		eq_point(&[zero, one], pi_hi),
		eq_point(&[one, one], pi_hi),
	];
	let (v_sent, m_pi_sent) = match tamper {
		Tamper::NoTamper => (v_honest, m_pi),
		Tamper::CornerForgery => {
			// e10 = eq_blk[2], e01 = eq_blk[1]: den transparent unchanged in char 2.
			(
				[v_honest[0], v_honest[1] + eq_blk[2], v_honest[2] + eq_blk[1]],
				m_pi,
			)
		}
		Tamper::MPiLie => (v_honest, m_pi + one),
	};
	IPProverChannel::<B128>::send_many(
		transcript,
		&[v_sent[0], v_sent[1], v_sent[2], m_pi_sent],
	);

	// Phase B (honest data; round bytes are claim-independent).
	let phi: B128 = IPProverChannel::<B128>::sample(transcript);
	let mut points = Vec::with_capacity(10 * k + 1);
	let mut evals = Vec::with_capacity(10 * k + 1);
	for (l, tr) in ctxs.iter().enumerate() {
		let claim_points = m_d_points(&dims, tr);
		let finish = &output.multilinear_evals[l];
		evals.push(finish[0]);
		evals.push(finish[1]);
		evals.extend_from_slice(&d_vals[l * SHIFT_VARIANT_COUNT..(l + 1) * SHIFT_VARIANT_COUNT]);
		points.extend(claim_points);
	}
	points.push(pi_lo.to_vec());
	evals.push(m_pi); // honest value: prover-side sum anchor (bytes independent of it)
	let combined = evaluate_univariate(&evals, phi);

	// NAIVE W_eq (full tensor per point) — independent of the crate's prefix builder.
	let mut w_eq = vec![B128::ZERO; 1 << dims.n_d];
	let mut phi_pow = B128::ONE;
	for p in &points {
		let tensor = eq_ind_partial_eval_scalars(p);
		for (slot, t) in w_eq.iter_mut().zip(&tensor) {
			*slot += phi_pow * *t;
		}
		phi_pow *= phi;
	}
	let prover_b = BivariateProductSumcheckProver::new(
		[FieldBuffer::<B128>::from_values(&w_eq), m_d.clone()],
		combined,
	);
	let out_b = prove_single(prover_b, transcript);
	let mut sigma = out_b.challenges.clone();
	sigma.reverse();
	let m = out_b.multilinear_evals[1];
	IPProverChannel::<B128>::send_one(transcript, m);

	// Merged [M_VK, M_D] opening (honest claims — the opening's own claim_vk is derived
	// from the HONEST corners v_honest, while the verifier derives ITS claim_vk from the
	// SENT corners v_sent, so a corner forgery diverges at the merged-opening reduction).
	let rho_c: Vec<B128> = IPProverChannel::<B128>::sample_many(transcript, 2);
	let vk_codeword = encode_m_vk(&pcs, &ntt, &m_vk);
	let claim_vk = eq_point(&[zero, zero], &rho_c) * v_honest[0]
		+ eq_point(&[one, zero], &rho_c) * v_honest[1]
		+ eq_point(&[zero, one], &rho_c) * v_honest[2];
	let mut vk_point = pi_lo.to_vec();
	vk_point.extend_from_slice(&rho_c);
	{
		let mut mchan =
			ProverMerkleTranscriptChannel::<_, C, B128, StdHashSuite>::new(&mut *transcript);
		let vk_commitment = mchan.send_merkle_commitment(vk_codeword.to_ref(), pcs.leaf_size(0));
		binius_recursion_discharge::merged::prove_merged_openings(
			&pcs.params,
			&ntt,
			&mut mchan,
			[
				OpenClaim {
					message: m_vk,
					point: &vk_point,
					eval: claim_vk,
				},
				OpenClaim {
					message: m_d,
					point: &sigma,
					eval: m,
				},
			],
			vec![(vk_codeword, vk_commitment), (m_d_codeword, d_commitment)],
		)
		.map_err(|e| anyhow::anyhow!("merged open: {e}"))?;
	}
	Ok(())
}

#[test]
fn step2_adaptive_attacks() {
	// Build the synth context (mirrors tests/step2_small.rs).
	let pipeline = LeafPipeline::setup(synth_cs(0)).expect("pipeline");
	let cs = pipeline.verifier.constraint_system().clone();
	let table = extract_table(&cs).expect("table");
	let mut claims = Vec::with_capacity(K);
	for i in 0..K {
		let wit = synth_witness(&cs, 2000 + i as u64).expect("witness");
		let public = LeafPipeline::public(&wit);
		let proof = pipeline.prove(&wit).expect("prove");
		claims.push(
			verify_and_capture(&pipeline.verifier, &public, proof, table.dims.arity)
				.expect("capture"),
		);
	}
	let (vkm, _) = vkgen(&table).expect("vkgen");
	let stmt = DischargeStatement::new(&table, claims).expect("statement");

	let run = |tamper: Tamper| -> anyhow::Result<()> {
		let mut pt = ProverTranscript::new(StdChallenger::default());
		copied_prove_step2(&cs, &vkm, &stmt, tamper, &mut pt)?;
		let bytes = pt.finalize();
		let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes);
		discharge_verify_step2(&vkm, &stmt, &mut vt)?;
		vt.finalize().map_err(|e| anyhow::anyhow!("finalize: {e}"))?;
		Ok(())
	};

	// Control: copied prover (direct-MLE corners, naive W_eq, straightforward Phase-C
	// leaf build) must verify against the crate's verifier.
	run(Tamper::NoTamper).expect("control: copied honest STEP-2 prover must verify");
	eprintln!("[step2-adversarial] control (NoTamper, copied prover): VERIFIED");

	// Corner forgery: invisible to the den transparent BY CONSTRUCTION; must be caught
	// inside the merged-opening layer (the verifier derives the VK claim from the
	// forged corners; the reduction/FRI cannot both be satisfied).
	let err = run(Tamper::CornerForgery).expect_err("corner forgery MUST be rejected");
	let msg = format!("{err:#}");
	eprintln!("[step2-adversarial] CornerForgery rejected: {msg}");
	assert!(
		msg.contains("merged") || msg.contains("reduction"),
		"CornerForgery must be caught by the merged-opening layer, got: {msg}"
	);
	assert!(
		!msg.contains("phase C"),
		"CornerForgery must NOT die at the phase C transparents (it is den-invariant): {msg}"
	);

	// m_pi lie: the tree is honest, so the Phase-C numerator transparent must fire.
	let err = run(Tamper::MPiLie).expect_err("m_pi lie MUST be rejected");
	let msg = format!("{err:#}");
	eprintln!("[step2-adversarial] MPiLie rejected: {msg}");
	assert!(
		msg.contains("phase C numerator transparent"),
		"MPiLie must be caught by the Phase-C numerator transparent, got: {msg}"
	);
}
