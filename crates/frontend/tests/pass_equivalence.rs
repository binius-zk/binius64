// Copyright 2026 The Binius Developers

//! Compiling a circuit under different optimization settings must not change what it computes.

use binius_core::{ValueVec, Word};
use binius_frontend::{Circuit, CircuitBuilder, CircuitStat, Options, Wire};
use proptest::prelude::*;

/// The optimization flags, in the order [`options`] assigns them.
const FLAGS: [&str; 7] = [
	"gate fusion",
	"constant propagation",
	"common subexpression elimination",
	"dead code elimination",
	"algebraic folding",
	"scratch pooling",
	"zero propagation",
];

/// Turns a flag vector into the settings it names.
fn options(flags: [bool; 7]) -> Options {
	let mut opts = Options::default();
	let fields = [
		&mut opts.enable_gate_fusion,
		&mut opts.enable_constant_propagation,
		&mut opts.enable_common_subexpression_elimination,
		&mut opts.enable_dead_code_elimination,
		&mut opts.enable_algebraic_folding,
		&mut opts.enable_scratch_pooling,
		&mut opts.enable_zero_propagation,
	];
	for (field, on) in fields.into_iter().zip(flags) {
		*field = on;
	}
	opts
}

/// The settings under test, with the unoptimized reference first.
fn configurations(extra: [bool; 7]) -> Vec<(String, Options)> {
	let mut configs = vec![
		("all off".to_owned(), options([false; 7])),
		("all on".to_owned(), options([true; 7])),
		("default".to_owned(), Options::default()),
		(format!("random {extra:?}"), options(extra)),
	];
	for (i, name) in FLAGS.iter().enumerate() {
		let mut flags = [true; 7];
		flags[i] = false;
		configs.push((format!("{name} off"), options(flags)));
	}
	configs
}

/// A shift or rotate kind.
#[derive(Clone, Copy, Debug)]
enum Shift {
	Rotl,
	Rotr,
	Shl,
	Shr,
	Sar,
	Rotl32,
	Rotr32,
	Srl32,
	Sll32,
	Sra32,
}

impl Shift {
	fn emit(self, b: &CircuitBuilder, x: Wire, n: u32) -> Wire {
		match self {
			Self::Rotl => b.rotl(x, n % 64),
			Self::Rotr => b.rotr(x, n % 64),
			Self::Shl => b.shl(x, n % 64),
			Self::Shr => b.shr(x, n % 64),
			Self::Sar => b.sar(x, n % 64),
			Self::Rotl32 => b.rotl32(x, n % 32),
			Self::Rotr32 => b.rotr32(x, n % 32),
			Self::Srl32 => b.srl32(x, n % 32),
			Self::Sll32 => b.sll32(x, n % 32),
			Self::Sra32 => b.sra32(x, n % 32),
		}
	}
}

/// One builder call, naming its operands by position in the wire pool.
#[derive(Clone, Debug)]
enum Op {
	Band(usize, usize),
	Bxor(usize, usize),
	BxorMulti(Vec<usize>),
	Bor(usize, usize),
	Bnot(usize),
	Fax(usize, usize, usize),
	Select(usize, usize, usize),
	Iadd(usize, usize),
	IaddCinCout(usize, usize, usize),
	IsubBinBout(usize, usize, usize),
	Iadd32(usize, usize),
	Iadd32CinCout(usize, usize, usize),
	Imul(usize, usize),
	Bmul(usize, usize, usize, usize),
	Smul(usize, usize),
	IcmpEq(usize, usize),
	IcmpUlt(usize, usize),
	Shifted(Shift, usize, u32),
}

/// Emits one operation and appends its results to the pool.
fn emit(b: &CircuitBuilder, pool: &mut Vec<Wire>, op: &Op) {
	let n = pool.len();
	let w = |i: &usize| pool[i % n];
	let results = match op {
		Op::Band(x, y) => vec![b.band(w(x), w(y))],
		Op::Bxor(x, y) => vec![b.bxor(w(x), w(y))],
		Op::BxorMulti(xs) => vec![b.bxor_multi(&xs.iter().map(w).collect::<Vec<_>>())],
		Op::Bor(x, y) => vec![b.bor(w(x), w(y))],
		Op::Bnot(x) => vec![b.bnot(w(x))],
		Op::Fax(x, y, z) => vec![b.fax(w(x), w(y), w(z))],
		Op::Select(c, t, f) => vec![b.select(w(c), w(t), w(f))],
		Op::Iadd(x, y) => {
			let (sum, cout) = b.iadd(w(x), w(y));
			vec![sum, cout]
		}
		Op::IaddCinCout(x, y, cin) => {
			let (sum, cout) = b.iadd_cin_cout(w(x), w(y), w(cin));
			vec![sum, cout]
		}
		Op::IsubBinBout(x, y, bin) => {
			let (diff, bout) = b.isub_bin_bout(w(x), w(y), w(bin));
			vec![diff, bout]
		}
		Op::Iadd32(x, y) => vec![b.iadd_32(w(x), w(y))],
		Op::Iadd32CinCout(x, y, cin) => {
			let (sum, cout) = b.iadd32_cin_cout(w(x), w(y), w(cin));
			vec![sum, cout]
		}
		Op::Imul(x, y) => {
			let (hi, lo) = b.imul(w(x), w(y));
			vec![hi, lo]
		}
		Op::Bmul(a_lo, a_hi, b_lo, b_hi) => {
			let (c_lo, c_hi) = b.bmul(w(a_lo), w(a_hi), w(b_lo), w(b_hi));
			vec![c_lo, c_hi]
		}
		Op::Smul(x, y) => {
			let (hi, lo) = b.smul(w(x), w(y));
			vec![hi, lo]
		}
		Op::IcmpEq(x, y) => vec![b.icmp_eq(w(x), w(y))],
		Op::IcmpUlt(x, y) => vec![b.icmp_ult(w(x), w(y))],
		Op::Shifted(kind, x, amount) => vec![kind.emit(b, w(x), *amount)],
	};
	pool.extend(results);
}

/// A circuit shape, replayable under any settings.
#[derive(Clone, Debug)]
struct Spec {
	n_witness: usize,
	n_inout: usize,
	constants: Vec<u64>,
	ops: Vec<Op>,
	promotions: Vec<usize>,
	assertions: Vec<(usize, usize)>,
	inputs: Vec<u64>,
}

/// One compilation of a [`Spec`].
struct Built {
	circuit: Circuit,
	/// The wires a filler must assign, in an order the settings do not change.
	inputs: Vec<Wire>,
	/// The wires the settings are compared on.
	observed: Vec<Wire>,
	/// A committed word whose defining constraint no pass can drop.
	anchor: Wire,
}

fn build(spec: &Spec, opts: Options) -> Built {
	let b = CircuitBuilder::with_opts(opts);

	// Zero propagation and algebraic folding fire on these two words alone, so both are seeded.
	let zero = b.add_constant(Word::ZERO);
	let all_one = b.add_constant(Word::ALL_ONE);

	let mut inputs = Vec::new();
	let mut pool = vec![zero, all_one];
	for _ in 0..spec.n_witness {
		let wire = b.add_witness();
		inputs.push(wire);
		pool.push(wire);
	}
	for _ in 0..spec.n_inout {
		let wire = b.add_inout();
		inputs.push(wire);
		pool.push(wire);
	}
	for &c in &spec.constants {
		pool.push(b.add_constant_64(c));
	}

	for op in &spec.ops {
		emit(&b, &mut pool, op);
	}

	// `fax(x, all-1, 0)` is `x` and never folds at build time, so every promotion is a gate output.
	let observed = spec
		.promotions
		.iter()
		.map(|i| {
			let copy = b.fax(pool[i % pool.len()], all_one, zero);
			b.mark_inout(copy);
			copy
		})
		.collect();

	for (x, y) in &spec.assertions {
		let lhs = pool[x % pool.len()];
		let rhs = pool[y % pool.len()];
		b.assert_eq("copy", lhs, b.fax(lhs, all_one, zero));
		b.assert_eq_cond("never_checked", lhs, rhs, zero);
	}

	// Two fresh inputs, so no pass can fold, dedup or strand the AND constraint defining `anchor`.
	let m0 = b.add_witness();
	let m1 = b.add_witness();
	inputs.push(m0);
	inputs.push(m1);
	let anchor = b.band(m0, m1);
	b.mark_inout(anchor);

	Built {
		circuit: b.build(),
		inputs,
		observed,
		anchor,
	}
}

/// Assigns the inputs from `values`, cycled, and evaluates the circuit.
fn populate(built: &Built, values: &[u64]) -> Result<ValueVec, String> {
	let mut filler = built.circuit.new_witness_filler();
	for (k, &wire) in built.inputs.iter().enumerate() {
		filler[wire] = Word(values[k % values.len()]);
	}
	built
		.circuit
		.populate_wire_witness(&mut filler)
		.map_err(|err| err.to_string())?;
	Ok(filler.into_value_vec())
}

/// The words on the observed wires.
fn observed(built: &Built, values: &ValueVec) -> Vec<Word> {
	// Settings shift the value-vector layout, so a word is located by wire and never by position.
	built
		.observed
		.iter()
		.chain([&built.anchor])
		.map(|&wire| values[built.circuit.witness_index(wire)])
		.collect()
}

fn index() -> impl Strategy<Value = usize> {
	0usize..32
}

fn shift() -> impl Strategy<Value = Shift> {
	prop_oneof![
		Just(Shift::Rotl),
		Just(Shift::Rotr),
		Just(Shift::Shl),
		Just(Shift::Shr),
		Just(Shift::Sar),
		Just(Shift::Rotl32),
		Just(Shift::Rotr32),
		Just(Shift::Srl32),
		Just(Shift::Sll32),
		Just(Shift::Sra32),
	]
}

fn op() -> impl Strategy<Value = Op> {
	prop_oneof![
		(index(), index()).prop_map(|(x, y)| Op::Band(x, y)),
		(index(), index()).prop_map(|(x, y)| Op::Bxor(x, y)),
		prop::collection::vec(index(), 1..5).prop_map(Op::BxorMulti),
		(index(), index()).prop_map(|(x, y)| Op::Bor(x, y)),
		index().prop_map(Op::Bnot),
		(index(), index(), index()).prop_map(|(x, y, z)| Op::Fax(x, y, z)),
		(index(), index(), index()).prop_map(|(c, t, f)| Op::Select(c, t, f)),
		(index(), index()).prop_map(|(x, y)| Op::Iadd(x, y)),
		(index(), index(), index()).prop_map(|(x, y, c)| Op::IaddCinCout(x, y, c)),
		(index(), index(), index()).prop_map(|(x, y, c)| Op::IsubBinBout(x, y, c)),
		(index(), index()).prop_map(|(x, y)| Op::Iadd32(x, y)),
		(index(), index(), index()).prop_map(|(x, y, c)| Op::Iadd32CinCout(x, y, c)),
		(index(), index()).prop_map(|(x, y)| Op::Imul(x, y)),
		(index(), index(), index(), index()).prop_map(|(a, b, c, d)| Op::Bmul(a, b, c, d)),
		(index(), index()).prop_map(|(x, y)| Op::Smul(x, y)),
		(index(), index()).prop_map(|(x, y)| Op::IcmpEq(x, y)),
		(index(), index()).prop_map(|(x, y)| Op::IcmpUlt(x, y)),
		(shift(), index(), 0u32..64).prop_map(|(k, x, n)| Op::Shifted(k, x, n)),
	]
}

fn spec() -> impl Strategy<Value = Spec> {
	(
		1usize..4,
		1usize..3,
		prop::collection::vec(any::<u64>(), 0..3),
		prop::collection::vec(op(), 1..20),
		prop::collection::vec(index(), 1..5),
		prop::collection::vec((index(), index()), 0..3),
		prop::collection::vec(any::<u64>(), 1..8),
	)
		.prop_map(|(n_witness, n_inout, constants, ops, promotions, assertions, inputs)| Spec {
			n_witness,
			n_inout,
			constants,
			ops,
			promotions,
			assertions,
			inputs,
		})
}

fn flags() -> impl Strategy<Value = [bool; 7]> {
	any::<[bool; 7]>()
}

/// An integration test has no crate root to persist a regression seed under.
fn config() -> ProptestConfig {
	ProptestConfig {
		failure_persistence: None,
		..ProptestConfig::default()
	}
}

proptest! {
	#![proptest_config(config())]

	#[test]
	fn settings_preserve_the_witness(spec in spec(), extra in flags()) {
		let configs = configurations(extra);
		let reference = build(&spec, configs[0].1);
		let reference_values = populate(&reference, &spec.inputs)
			.map_err(|err| TestCaseError::fail(format!("reference not satisfied: {err}")))?;
		let expected = observed(&reference, &reference_values);
		prop_assert!(
			reference.circuit.constraint_system().verify(&reference_values).is_ok(),
			"reference rejected its own witness",
		);

		for (name, opts) in &configs[1..] {
			let built = build(&spec, *opts);
			let values = populate(&built, &spec.inputs)
				.map_err(|err| TestCaseError::fail(format!("{name}: {err}")))?;
			prop_assert_eq!(observed(&built, &values), expected.clone(), "{}", name);
			prop_assert!(
				built.circuit.constraint_system().verify(&values).is_ok(),
				"{} rejected its own witness",
				name,
			);
		}
	}

	#[test]
	fn a_corrupted_committed_word_is_rejected(
		spec in spec(),
		extra in flags(),
		mask in 1u64..,
	) {
		for (name, opts) in configurations(extra) {
			let built = build(&spec, opts);
			let mut values = populate(&built, &spec.inputs)
				.map_err(|err| TestCaseError::fail(format!("{name}: {err}")))?;

			let index = built.circuit.witness_index(built.anchor);
			values[index] = values[index] ^ Word(mask);
			prop_assert!(
				built.circuit.constraint_system().verify(&values).is_err(),
				"{} accepted a corrupted anchor",
				name,
			);
		}
	}
}

#[test]
fn every_flag_changes_a_generated_circuit() {
	// Pool: 0 zero, 1 all-ones, 2-3 witness, 4 inout, 5 constant, then one entry per result.
	// Each operation is a shape one pass fires on, and the last result is read by nothing.
	let spec = Spec {
		n_witness: 2,
		n_inout: 1,
		constants: vec![7],
		ops: vec![
			Op::Bxor(2, 3),
			Op::Band(6, 4),
			Op::Iadd32(2, 0),
			Op::Band(3, 3),
			Op::Band(2, 4),
			Op::Band(2, 4),
			Op::IcmpUlt(5, 5),
			Op::Band(2, 3),
		],
		promotions: vec![7, 8, 9, 10, 11, 12],
		assertions: vec![(2, 3)],
		inputs: vec![0x0123_4567_89ab_cdef, 0, 0xffff_ffff],
	};

	let baseline = CircuitStat::collect(&build(&spec, options([true; 7])).circuit);
	for (i, name) in FLAGS.iter().enumerate() {
		let mut flags = [true; 7];
		flags[i] = false;
		let stat = CircuitStat::collect(&build(&spec, options(flags)).circuit);
		assert_ne!(
			(
				stat.n_zero_constraints,
				stat.n_and_constraints,
				stat.n_imul_constraints,
				stat.n_bmul_constraints,
				stat.n_scratch,
			),
			(
				baseline.n_zero_constraints,
				baseline.n_and_constraints,
				baseline.n_imul_constraints,
				baseline.n_bmul_constraints,
				baseline.n_scratch,
			),
			"turning off {name} left the built circuit unchanged"
		);
	}
}
