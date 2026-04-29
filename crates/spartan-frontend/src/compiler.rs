// Copyright 2025 Irreducible Inc.

use binius_field::Field;

use crate::{
	circuit_builder::ConstraintBuilder,
	constraint_system::{ConstraintSystem, WitnessLayout},
	wire_elimination::{CostModel, run_wire_elimination},
};

pub fn compile<F: Field>(builder: ConstraintBuilder<F>) -> (ConstraintSystem<F>, WitnessLayout<F>) {
	let ir = builder.build();
	let ir = run_wire_elimination(CostModel::default(), ir);
	ir.finalize()
}
